"""
Batch-assembly correctness gates.

Every failure this file catches is SILENT in production: wrong positions, a
wrong last-page length, or a slot_mapping that disagrees with the block table
all produce fluent output, no exception, and no metric movement. So the tests
assert exact integer contents, not shapes.

Pure CPU. No GPU, no model, no weights — which is most of the point of keeping
the addressing arithmetic in plain Python ints.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from engine.attention_backend import BatchMeta

from serving.engine_iface.batch import ScheduledSeq, build_batch_meta, build_token_tensor
from serving.memory.allocator import BlockAllocator
from serving.memory.block_table import SequenceBlocks, build_csr

PAGE = 16
DEV = torch.device("cpu")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_alloc(num_blocks: int = 256, block_size: int = PAGE) -> BlockAllocator:
    return BlockAllocator(num_blocks=num_blocks, block_size=block_size)


def grow(seq: SequenceBlocks, n: int, first_id: int = 0) -> ScheduledSeq:
    """Append `n` tokens to `seq` and wrap it as this step's contribution."""
    seq.append(n)
    return ScheduledSeq(blocks=seq, new_token_ids=list(range(first_id, first_id + n)))


def ints(t: torch.Tensor) -> list[int]:
    return [int(x) for x in t]


def expected_slot(seq: SequenceBlocks, pos: int, page_size: int = PAGE) -> int:
    """The addressing contract, written out longhand and independently here."""
    return seq.block_ids[pos // page_size] * page_size + (pos % page_size)


# ---------------------------------------------------------------------------
# 1. single decode step
# ---------------------------------------------------------------------------


def test_single_decode_every_field():
    alloc = make_alloc()
    s = SequenceBlocks(alloc, seq_id=0)
    s.append(20)  # history
    sched = grow(s, 1, first_id=777)  # kv_len 21 after append

    meta = build_batch_meta([sched], DEV, PAGE)
    toks = build_token_tensor([sched], DEV)

    assert ints(toks) == [777]
    assert meta.n_seqs == 1
    assert meta.n_tokens == 1
    assert ints(meta.query_lens) == [1]
    assert ints(meta.cu_query_lens) == [0, 1]
    assert ints(meta.kv_lens) == [21]
    assert ints(meta.positions) == [20]  # absolute: the 21st token is at index 20
    assert ints(meta.last_token_ix) == [0]
    assert ints(meta.batch_indices) == [0]
    assert ints(meta.kv_indptr) == [0, 2]  # 21 tokens -> 2 pages
    assert ints(meta.kv_indices) == s.block_ids
    assert ints(meta.kv_last_page_len) == [5]  # 21 - 16
    assert ints(meta.slot_mapping) == [expected_slot(s, 20)]
    assert meta.page_size == PAGE
    assert meta.is_prefill is False


# ---------------------------------------------------------------------------
# 2. multi-sequence pure decode
# ---------------------------------------------------------------------------


def test_multi_sequence_pure_decode():
    alloc = make_alloc()
    scheds = []
    for i in range(5):
        s = SequenceBlocks(alloc, seq_id=i)
        s.append(10 + i)
        scheds.append(grow(s, 1, first_id=100 + i))

    meta = build_batch_meta(scheds, DEV, PAGE)

    assert ints(meta.query_lens) == [1] * 5
    assert ints(meta.cu_query_lens) == [0, 1, 2, 3, 4, 5]
    assert ints(meta.last_token_ix) == [0, 1, 2, 3, 4]
    assert ints(meta.batch_indices) == [0, 1, 2, 3, 4]
    assert ints(meta.kv_lens) == [11, 12, 13, 14, 15]
    assert ints(meta.positions) == [10, 11, 12, 13, 14]
    assert meta.is_prefill is False
    assert ints(build_token_tensor(scheds, DEV)) == [100, 101, 102, 103, 104]


# ---------------------------------------------------------------------------
# 3. mixed prefill + decode
# ---------------------------------------------------------------------------


def test_mixed_prefill_and_decode():
    alloc = make_alloc()
    a = SequenceBlocks(alloc, seq_id=0)  # fresh prefill of 8
    b = SequenceBlocks(alloc, seq_id=1)
    c = SequenceBlocks(alloc, seq_id=2)
    b.append(30)
    c.append(5)

    scheds = [grow(a, 8), grow(b, 1), grow(c, 1)]
    meta = build_batch_meta(scheds, DEV, PAGE)

    assert ints(meta.query_lens) == [8, 1, 1]
    assert ints(meta.cu_query_lens) == [0, 8, 9, 10]
    assert ints(meta.last_token_ix) == [7, 8, 9]
    assert ints(meta.batch_indices) == [0] * 8 + [1, 2]
    assert ints(meta.kv_lens) == [8, 31, 6]
    assert ints(meta.positions) == list(range(8)) + [30, 5]
    assert meta.is_prefill is True
    assert meta.n_tokens == 10


# ---------------------------------------------------------------------------
# 4. POSITIONS ARE ABSOLUTE
# ---------------------------------------------------------------------------


def test_positions_are_absolute_not_chunk_relative():
    """
    THE chunked-prefill bug, stated unmistakably.

    A sequence holding 32 tokens of history that contributes 8 more has kv_len 40
    and MUST carry positions 32..39. Emitting 0..7 makes RoPE rotate every key in
    the chunk as if it were the start of the prompt. Nothing crashes; the model
    just attends to a geometry that never existed.
    """
    alloc = make_alloc()
    s = SequenceBlocks(alloc, seq_id=0)
    s.append(32)  # history
    sched = grow(s, 8)  # kv_len == 40

    meta = build_batch_meta([sched], DEV, PAGE)

    assert int(meta.kv_lens[0]) == 40
    assert ints(meta.positions) == [32, 33, 34, 35, 36, 37, 38, 39]
    assert ints(meta.positions) != list(range(8)), (
        "positions restarted at 0 — chunked prefill is broken"
    )
    assert int(meta.positions[0]) == 40 - 8
    assert int(meta.positions[-1]) == 39


# ---------------------------------------------------------------------------
# 5. chunked prefill across two steps
# ---------------------------------------------------------------------------


def test_chunked_prefill_positions_continue_across_steps():
    alloc = make_alloc()
    s = SequenceBlocks(alloc, seq_id=0)
    prompt = list(range(1000, 1100))  # 100-token prompt
    chunk = 24

    seen_positions: list[int] = []
    seen_kv_lens: list[int] = []
    offset = 0
    while offset < len(prompt):
        take = prompt[offset : offset + chunk]
        s.append(len(take))
        sched = ScheduledSeq(
            blocks=s, new_token_ids=take, wants_logits=offset + len(take) >= len(prompt)
        )
        meta = build_batch_meta([sched], DEV, PAGE)
        seen_positions.extend(ints(meta.positions))
        seen_kv_lens.append(int(meta.kv_lens[0]))
        # each chunk starts exactly where the previous one ended
        assert int(meta.positions[0]) == offset
        assert int(meta.positions[-1]) == offset + len(take) - 1
        offset += len(take)

    assert seen_positions == list(range(100)), "positions must tile the prompt exactly once"
    assert seen_kv_lens == [24, 48, 72, 96, 100]
    assert s.num_tokens == 100


def test_second_chunk_does_not_restart_at_zero():
    alloc = make_alloc()
    s = SequenceBlocks(alloc, seq_id=0)

    s.append(16)
    first = build_batch_meta([ScheduledSeq(s, list(range(16)))], DEV, PAGE)
    assert ints(first.positions) == list(range(0, 16))

    s.append(16)
    second = build_batch_meta([ScheduledSeq(s, list(range(16, 32)))], DEV, PAGE)
    assert ints(second.positions) == list(range(16, 32))
    assert int(second.positions[0]) != 0
    assert int(second.kv_lens[0]) == 32


# ---------------------------------------------------------------------------
# 6. slot_mapping vs block tables, on a FRAGMENTED pool
# ---------------------------------------------------------------------------


def _fragmented_seqs(alloc: BlockAllocator, n_seqs: int, rounds: int) -> list[SequenceBlocks]:
    """
    Interleave growth so each sequence's pages are non-contiguous.

    Round-robin appends of one full page per sequence give seq0 pages 0, n, 2n...
    Any implementation that assumes `slot == base + pos` passes on a contiguous
    pool and fails here, which is the entire reason this helper exists.
    """
    seqs = [SequenceBlocks(alloc, seq_id=i) for i in range(n_seqs)]
    for _ in range(rounds):
        for s in seqs:
            s.append(alloc.block_size)
    return seqs


def test_slot_mapping_matches_block_tables_when_fragmented():
    alloc = make_alloc()
    seqs = _fragmented_seqs(alloc, n_seqs=3, rounds=3)

    # Sanity: the pool really is fragmented for each sequence.
    for s in seqs:
        assert len(s.block_ids) == 3
        assert s.block_ids != list(range(s.block_ids[0], s.block_ids[0] + 3))

    scheds = [grow(s, 5) for s in seqs]  # 5 more tokens each, straddling a page edge
    meta = build_batch_meta(scheds, DEV, PAGE)

    got = ints(meta.slot_mapping)
    want: list[int] = []
    for s in seqs:
        for pos in range(s.num_tokens - 5, s.num_tokens):
            want.append(expected_slot(s, pos))
    assert got == want

    # And restate it per-token against positions/batch_indices, which is how a
    # backend actually consumes the pair.
    for tok in range(meta.n_tokens):
        seq = seqs[int(meta.batch_indices[tok])]
        pos = int(meta.positions[tok])
        assert int(meta.slot_mapping[tok]) == seq.block_ids[pos // PAGE] * PAGE + (pos % PAGE)


def test_slot_mapping_straddles_page_boundary():
    """A chunk that crosses a page edge must jump to a distant physical page."""
    alloc = make_alloc()
    seqs = _fragmented_seqs(alloc, n_seqs=2, rounds=1)
    s = seqs[0]
    s.append(14)  # 16 -> 30, so the next 4 tokens sit at 30,31 | 32,33
    seqs[1].append(PAGE)  # steal the next free page so s's third page is not adjacent
    sched = grow(s, 4)
    assert s.block_ids[2] != s.block_ids[1] + 1
    meta = build_batch_meta([sched], DEV, PAGE)

    slots = ints(meta.slot_mapping)
    assert ints(meta.positions) == [30, 31, 32, 33]
    assert slots[1] + 1 != slots[2], "page boundary produced contiguous slots — block table ignored"
    assert slots[:2] == [s.block_ids[1] * PAGE + 14, s.block_ids[1] * PAGE + 15]
    assert slots[2:] == [s.block_ids[2] * PAGE + 0, s.block_ids[2] * PAGE + 1]


# ---------------------------------------------------------------------------
# 7. kv_last_page_len sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1, 15, 16, 17, 31, 32, 33])
def test_kv_last_page_len_sweep(length):
    alloc = make_alloc()
    s = SequenceBlocks(alloc, seq_id=0)
    sched = grow(s, length)
    meta = build_batch_meta([sched], DEV, PAGE)

    lp = int(meta.kv_last_page_len[0])
    assert 1 <= lp <= PAGE
    if length % PAGE == 0:
        assert lp == PAGE, "an exact multiple of page_size must report page_size, NOT 0"
    else:
        assert lp == length % PAGE
    # pages allocated cover exactly the claimed kv_len
    assert int(meta.kv_indptr[1]) == (length + PAGE - 1) // PAGE


def test_kv_last_page_len_sweep_in_one_batch():
    alloc = make_alloc()
    lengths = [1, 15, 16, 17, 31, 32, 33]
    scheds = []
    for i, n in enumerate(lengths):
        s = SequenceBlocks(alloc, seq_id=i)
        scheds.append(grow(s, n))
    meta = build_batch_meta(scheds, DEV, PAGE)

    assert ints(meta.kv_last_page_len) == [1, 15, 16, 1, 15, 16, 1]
    assert int(meta.kv_last_page_len.min()) >= 1
    assert int(meta.kv_last_page_len.max()) <= PAGE


# ---------------------------------------------------------------------------
# 8. validate() is called, and it actually rejects
# ---------------------------------------------------------------------------


def _one_seq_meta(n_hist: int = 20, n_new: int = 4) -> BatchMeta:
    alloc = make_alloc()
    s = SequenceBlocks(alloc, seq_id=0)
    s.append(n_hist)
    return build_batch_meta([grow(s, n_new)], DEV, PAGE)


def test_validate_is_actually_called(monkeypatch):
    calls = []
    original = BatchMeta.validate

    def spy(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(BatchMeta, "validate", spy)
    _one_seq_meta()
    assert len(calls) == 1, "build_batch_meta must call meta.validate() by default"

    calls.clear()
    alloc = make_alloc()
    s = SequenceBlocks(alloc, seq_id=0)
    build_batch_meta([grow(s, 4)], DEV, PAGE, validate=False)
    assert calls == [], "validate=False must skip the check"


def test_validate_rejects_zeroed_last_page_len():
    """The off-by-one: a full page reported as 0 instead of page_size."""
    alloc = make_alloc()
    s = SequenceBlocks(alloc, seq_id=0)
    meta = build_batch_meta([grow(s, 32)], DEV, PAGE)
    assert int(meta.kv_last_page_len[0]) == PAGE

    corrupt = dataclasses.replace(
        meta, kv_last_page_len=torch.tensor([0], dtype=torch.int32, device=DEV)
    )
    with pytest.raises(AssertionError, match="kv_last_page_len"):
        corrupt.validate()


def test_validate_rejects_bad_cu_query_lens():
    meta = _one_seq_meta()
    corrupt = dataclasses.replace(
        meta, cu_query_lens=torch.tensor([0, 99], dtype=torch.int32, device=DEV)
    )
    with pytest.raises(AssertionError):
        corrupt.validate()


def test_validate_rejects_page_count_mismatch():
    """kv_lens inflated without allocating pages — the preemption-bug shape."""
    meta = _one_seq_meta()
    corrupt = dataclasses.replace(
        meta, kv_lens=torch.tensor([999], dtype=torch.int32, device=DEV)
    )
    with pytest.raises(AssertionError, match="pages allocated"):
        corrupt.validate()


def test_validate_rejects_short_batch_indices():
    meta = _one_seq_meta()
    corrupt = dataclasses.replace(
        meta, batch_indices=torch.tensor([0], dtype=torch.int32, device=DEV)
    )
    with pytest.raises(AssertionError, match="batch_indices"):
        corrupt.validate()


def test_assembly_rejects_ungrown_block_table():
    """query_len > kv_len: blocks.append() was not called before assembly."""
    alloc = make_alloc()
    s = SequenceBlocks(alloc, seq_id=0)
    s.append(2)
    with pytest.raises(ValueError, match="blocks.append"):
        build_batch_meta([ScheduledSeq(s, [1, 2, 3, 4, 5])], DEV, PAGE)


def test_assembly_rejects_page_size_mismatch():
    alloc = make_alloc(block_size=8)
    s = SequenceBlocks(alloc, seq_id=0)
    with pytest.raises(ValueError, match="page_size"):
        build_batch_meta([grow(s, 4)], DEV, page_size=16)


def test_assembly_rejects_freed_sequence():
    alloc = make_alloc()
    s = SequenceBlocks(alloc, seq_id=0)
    s.append(4)
    sched = ScheduledSeq(s, [1, 2, 3, 4])
    s.free()
    with pytest.raises(ValueError, match="freed"):
        build_batch_meta([sched], DEV, PAGE)


# ---------------------------------------------------------------------------
# 9. round trip against build_csr
# ---------------------------------------------------------------------------


def test_csr_slices_name_exactly_each_sequences_blocks():
    alloc = make_alloc()
    seqs = _fragmented_seqs(alloc, n_seqs=4, rounds=2)
    scheds = [grow(s, 3) for s in seqs]
    meta = build_batch_meta(scheds, DEV, PAGE)

    indptr = ints(meta.kv_indptr)
    indices = ints(meta.kv_indices)

    assert indptr[0] == 0
    assert indptr[-1] == len(indices)
    for i, s in enumerate(seqs):
        assert indices[indptr[i] : indptr[i + 1]] == s.block_ids

    # and identical to build_csr called directly
    csr_indptr, csr_indices, csr_last = build_csr([sc.blocks for sc in scheds])
    assert indptr == csr_indptr
    assert indices == csr_indices
    assert ints(meta.kv_last_page_len) == csr_last


# ---------------------------------------------------------------------------
# 10. edge cases
# ---------------------------------------------------------------------------


def test_empty_batch_raises():
    """Documented: an empty batch is a scheduler bug, not a valid zero-work step."""
    with pytest.raises(ValueError, match="empty sequence list"):
        build_batch_meta([], DEV, PAGE)
    assert ints(build_token_tensor([], DEV)) == []  # token packing is trivially empty


def test_zero_token_sequence_raises():
    alloc = make_alloc()
    s = SequenceBlocks(alloc, seq_id=0)
    s.append(4)
    with pytest.raises(ValueError, match="contributes 0 tokens"):
        build_batch_meta([ScheduledSeq(s, [])], DEV, PAGE)


def test_single_token_sequence_from_scratch():
    """A one-token prompt: kv_len 1, position 0, one page, last_page_len 1."""
    alloc = make_alloc()
    s = SequenceBlocks(alloc, seq_id=0)
    meta = build_batch_meta([grow(s, 1, first_id=42)], DEV, PAGE)

    assert ints(meta.query_lens) == [1]
    assert ints(meta.kv_lens) == [1]
    assert ints(meta.positions) == [0]
    assert ints(meta.cu_query_lens) == [0, 1]
    assert ints(meta.last_token_ix) == [0]
    assert ints(meta.kv_indptr) == [0, 1]
    assert ints(meta.kv_last_page_len) == [1]
    assert ints(meta.slot_mapping) == [s.block_ids[0] * PAGE]
    assert meta.is_prefill is False


def test_page_size_one_degenerate():
    """page_size == 1: every token is its own page; last_page_len always 1."""
    alloc = make_alloc(num_blocks=64, block_size=1)
    s = SequenceBlocks(alloc, seq_id=0)
    sched = grow(s, 5)
    meta = build_batch_meta([sched], DEV, page_size=1)

    assert ints(meta.kv_last_page_len) == [1]
    assert int(meta.kv_indptr[1]) == 5
    assert ints(meta.slot_mapping) == s.block_ids


def test_large_mixed_batch_validates():
    """A wide, ragged batch: the invariants must hold at scale, not just n=1."""
    alloc = make_alloc(num_blocks=1024)
    scheds = []
    for i in range(24):
        s = SequenceBlocks(alloc, seq_id=i)
        s.append(i * 3)  # heterogeneous, includes 0 history
        scheds.append(grow(s, 1 if i % 3 else 7))

    meta = build_batch_meta(scheds, DEV, PAGE)  # validate=True by default
    meta.validate()
    assert meta.n_seqs == 24
    assert meta.n_tokens == sum(sc.query_len for sc in scheds)
    assert ints(meta.cu_query_lens)[-1] == meta.n_tokens
    # last_token_ix is always the last index of each sequence's half-open range
    cu = ints(meta.cu_query_lens)
    assert ints(meta.last_token_ix) == [cu[i + 1] - 1 for i in range(24)]


# ---------------------------------------------------------------------------
# 11. device and dtype
# ---------------------------------------------------------------------------


def test_dtypes_and_device_match_the_contract():
    alloc = make_alloc()
    seqs = [SequenceBlocks(alloc, seq_id=i) for i in range(3)]
    scheds = [grow(s, 4 + i) for i, s in enumerate(seqs)]
    meta = build_batch_meta(scheds, DEV, PAGE)
    toks = build_token_tensor(scheds, DEV)

    int32_fields = [
        "query_lens",
        "cu_query_lens",
        "kv_lens",
        "last_token_ix",
        "kv_indptr",
        "kv_indices",
        "kv_last_page_len",
        "batch_indices",
        "slot_mapping",
    ]
    for name in int32_fields:
        t = getattr(meta, name)
        assert t.dtype == torch.int32, f"{name} must be int32, got {t.dtype}"
        assert t.device.type == DEV.type, f"{name} on {t.device}, expected {DEV}"

    assert meta.positions.dtype == torch.int64, "positions indexes the RoPE tables"
    assert meta.positions.device.type == DEV.type
    assert toks.dtype == torch.int64, "token ids index the embedding table"
    assert toks.device.type == DEV.type

    # shapes, restated against the contract
    n, t = meta.n_seqs, meta.n_tokens
    assert meta.query_lens.shape == (n,)
    assert meta.cu_query_lens.shape == (n + 1,)
    assert meta.kv_indptr.shape == (n + 1,)
    assert meta.positions.shape == (t,)
    assert meta.batch_indices.shape == (t,)
    assert meta.slot_mapping.shape == (t,)


def test_dtype_index_override_is_honoured():
    """positions stays int64 even when the index dtype is widened."""
    alloc = make_alloc()
    s = SequenceBlocks(alloc, seq_id=0)
    meta = build_batch_meta([grow(s, 4)], DEV, PAGE, dtype_index=torch.int64)
    assert meta.query_lens.dtype == torch.int64
    assert meta.kv_indices.dtype == torch.int64
    assert meta.positions.dtype == torch.int64
