"""
Tests for PagedTorchBackend.

THE ARGUMENT THESE TESTS MAKE
-----------------------------
PagedTorchBackend is the differential oracle FlashInfer will be checked against
(R9). An oracle that is itself wrong is worse than no oracle, so the burden of
proof here is higher than for ordinary code: every test below is aimed at a
failure mode that produces *fluent, plausible, wrong* output and raises nothing.

  * test_equivalence_vs_dense  — the core claim: paging changes addressing, not
    arithmetic. Compared against a naive implementation written in a
    deliberately different style (Python loops, float64) so a shared
    misconception cannot cancel out.
  * test_block_straddle        — R8. Off-by-one on the partial last page.
  * test_non_contiguous_pages  — a gather that accidentally assumes contiguity
    passes every contiguous-only test.
  * test_gqa_head_mapping      — repeat vs repeat_interleave. Both run.
  * test_multi_seq_isolation   — cross-sequence KV contamination.
  * test_causality             — attending to the future.

Everything runs on CPU with dtype=float32: no GPU, no model, no weights. That
is most of why these invariants are cheap to verify at all.
"""

from __future__ import annotations

import pytest
import torch
from engine.attention_backend import BatchMeta

from serving.backends.paged_torch import PagedTorchBackend
from serving.memory.allocator import BlockAllocator
from serving.memory.block_table import SequenceBlocks, build_csr

DEV = "cpu"
DT = torch.float32
TOL = dict(atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_backend(
    num_layers=1,
    num_blocks=32,
    block_size=16,
    n_kv_heads=2,
    n_heads=4,
    head_dim=8,
) -> PagedTorchBackend:
    return PagedTorchBackend(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        n_heads=n_heads,
        head_dim=head_dim,
        device=DEV,
        dtype=DT,
    )


def make_meta(seqs: list[SequenceBlocks], query_lens: list[int], block_size: int) -> BatchMeta:
    """
    Assemble BatchMeta exactly as the scheduler will: CSR from the block tables,
    slot_mapping from the per-sequence slot arithmetic. Called AFTER append().
    """
    kv_indptr, kv_indices, last_page = build_csr(seqs)

    slots: list[int] = []
    positions: list[int] = []
    batch_indices: list[int] = []
    for i, (s, ql) in enumerate(zip(seqs, query_lens, strict=False)):
        slots.extend(s.slots_for_new_tokens(ql))
        positions.extend(range(s.num_tokens - ql, s.num_tokens))
        batch_indices.extend([i] * ql)

    cu = [0]
    for ql in query_lens:
        cu.append(cu[-1] + ql)

    def i32(x):
        return torch.tensor(x, dtype=torch.int32, device=DEV)

    meta = BatchMeta(
        query_lens=i32(query_lens),
        cu_query_lens=i32(cu),
        kv_lens=i32([s.num_tokens for s in seqs]),
        positions=torch.tensor(positions, dtype=torch.int64, device=DEV),
        last_token_ix=i32([cu[i + 1] - 1 for i in range(len(seqs))]),
        kv_indptr=i32(kv_indptr),
        kv_indices=i32(kv_indices),
        kv_last_page_len=i32(last_page),
        batch_indices=i32(batch_indices),
        slot_mapping=i32(slots),
        page_size=block_size,
        is_prefill=any(q > 1 for q in query_lens),
    )
    meta.validate()  # never trust hand-built metadata
    return meta


def naive_attention(
    q: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    scale: float,
    n_kv_heads: int,
) -> torch.Tensor:
    """
    Textbook causal GQA attention on DENSE, UNPAGED K/V.

    Written on purpose in a style that shares nothing with the backend: explicit
    Python loops over heads and queries, masking by simply not summing the
    disallowed keys, arithmetic in float64. If both implementations were written
    the same way, a shared misconception would cancel and the comparison would
    prove nothing.

    q       : (q_len, n_heads, head_dim)
    k/v_full: (kv_len, n_kv_heads, head_dim) — the whole history
    """
    q_len, n_heads, head_dim = q.shape
    kv_len = k_full.shape[0]
    groups = n_heads // n_kv_heads
    out = torch.zeros_like(q)

    for h in range(n_heads):
        kvh = h // groups  # THE GQA mapping, spelled out
        for j in range(q_len):
            # Query j sits at true position (kv_len - q_len + j) and may attend
            # to keys [0, that position] inclusive.
            limit = kv_len - q_len + j
            scores = torch.tensor(
                [
                    float(torch.dot(q[j, h].double(), k_full[t, kvh].double())) * scale
                    for t in range(limit + 1)
                ],
                dtype=torch.float64,
            )
            probs = torch.softmax(scores, dim=0)
            acc = torch.zeros(head_dim, dtype=torch.float64)
            for t in range(limit + 1):
                acc += probs[t] * v_full[t, kvh].double()
            out[j, h] = acc.to(out.dtype)

    return out


def write_history(
    backend: PagedTorchBackend,
    alloc: BlockAllocator,
    lengths: list[int],
    layer: int = 0,
    seed: int = 0,
) -> tuple[list[SequenceBlocks], list[torch.Tensor], list[torch.Tensor]]:
    """
    Allocate `lengths` sequences, generate random K/V for each, and write it all
    through the backend in ONE prefill-shaped step. Returns the block tables and
    the dense ground-truth K/V per sequence.
    """
    g = torch.Generator().manual_seed(seed)
    seqs, ks, vs = [], [], []
    for i, n in enumerate(lengths):
        s = SequenceBlocks(alloc, seq_id=i)
        s.append(n)
        seqs.append(s)
        ks.append(torch.randn(n, backend.n_kv_heads, backend.head_dim, generator=g, dtype=DT))
        vs.append(torch.randn(n, backend.n_kv_heads, backend.head_dim, generator=g, dtype=DT))

    meta = make_meta(seqs, lengths, alloc.block_size)
    backend.append_kv(layer, torch.cat(ks), torch.cat(vs), meta)
    return seqs, ks, vs


# --------------------------------------------------------------------------
# 1. the write path
# --------------------------------------------------------------------------


def test_append_kv_lands_in_the_named_slots():
    """K/V land in exactly the slots slot_mapping named — read the pool directly."""
    bk = make_backend()
    alloc = BlockAllocator(num_blocks=32, block_size=16)

    s = SequenceBlocks(alloc, seq_id=0)
    s.append(20)  # two blocks: 16 + 4
    meta = make_meta([s], [20], alloc.block_size)

    k = torch.randn(20, bk.n_kv_heads, bk.head_dim, dtype=DT)
    v = torch.randn(20, bk.n_kv_heads, bk.head_dim, dtype=DT)
    bk.append_kv(0, k, v, meta)

    k_flat = bk.k_pool[0].view(-1, bk.n_kv_heads, bk.head_dim)
    v_flat = bk.v_pool[0].view(-1, bk.n_kv_heads, bk.head_dim)
    for token, slot in enumerate(meta.slot_mapping.tolist()):
        assert torch.equal(k_flat[slot], k[token]), f"K token {token} missing from slot {slot}"
        assert torch.equal(v_flat[slot], v[token]), f"V token {token} missing from slot {slot}"

    # And nowhere else: exactly 20 slots differ from zero in each pool.
    written = set(meta.slot_mapping.tolist())
    nonzero = {int(i) for i in (k_flat.abs().sum(dim=(1, 2)) != 0).nonzero().flatten()}
    assert nonzero == written, "append_kv touched slots it was not given"


def test_append_kv_is_idempotent():
    """A retried step must not double-append. An indexed write makes this free."""
    bk = make_backend()
    alloc = BlockAllocator(num_blocks=32, block_size=16)
    s = SequenceBlocks(alloc)
    s.append(9)
    meta = make_meta([s], [9], alloc.block_size)

    k = torch.randn(9, bk.n_kv_heads, bk.head_dim, dtype=DT)
    v = torch.randn(9, bk.n_kv_heads, bk.head_dim, dtype=DT)
    bk.append_kv(0, k, v, meta)
    first = bk.k_pool[0].clone()
    bk.append_kv(0, k, v, meta)
    assert torch.equal(bk.k_pool[0], first)


def test_layers_are_independent():
    """Writing layer 0 must not disturb layer 1. A shared pool is a silent bug."""
    bk = make_backend(num_layers=2)
    alloc = BlockAllocator(num_blocks=32, block_size=16)
    s = SequenceBlocks(alloc)
    s.append(5)
    meta = make_meta([s], [5], alloc.block_size)

    k = torch.randn(5, bk.n_kv_heads, bk.head_dim, dtype=DT)
    bk.append_kv(0, k, k, meta)
    assert bk.k_pool[1].abs().sum() == 0


# --------------------------------------------------------------------------
# 2. THE KEY EQUIVALENCE TEST
# --------------------------------------------------------------------------


def test_equivalence_vs_dense_reference():
    """
    Paged attention == dense attention on the same K/V with no paging at all.

    This is the whole correctness argument. Everything else in this file is a
    targeted probe of one way this could be wrong; this is the general claim.
    """
    bk = make_backend(n_kv_heads=2, n_heads=4, head_dim=8)
    alloc = BlockAllocator(num_blocks=32, block_size=16)
    lengths = [5, 17, 32]

    seqs, ks, vs = write_history(bk, alloc, lengths, seed=1)
    meta = make_meta(seqs, lengths, alloc.block_size)

    scale = 1.0 / (bk.head_dim**0.5)
    q = torch.randn(sum(lengths), bk.n_heads, bk.head_dim, dtype=DT)
    got = bk.attend(q, 0, scale, meta)

    cu = meta.cu_query_lens.tolist()
    for i, _n in enumerate(lengths):
        want = naive_attention(q[cu[i] : cu[i + 1]], ks[i], vs[i], scale, bk.n_kv_heads)
        torch.testing.assert_close(got[cu[i] : cu[i + 1]], want, **TOL)


def test_equivalence_vs_dense_reference_decode():
    """Same claim for a pure decode step — the no-mask path, which is different code."""
    bk = make_backend()
    alloc = BlockAllocator(num_blocks=32, block_size=16)
    lengths = [7, 16, 33]

    seqs, ks, vs = write_history(bk, alloc, lengths, seed=2)

    # Now one more token each: the decode step.
    new_k, new_v = [], []
    for i, s in enumerate(seqs):
        s.append(1)
        nk = torch.randn(1, bk.n_kv_heads, bk.head_dim, dtype=DT)
        nv = torch.randn(1, bk.n_kv_heads, bk.head_dim, dtype=DT)
        new_k.append(nk)
        new_v.append(nv)
        ks[i] = torch.cat([ks[i], nk])
        vs[i] = torch.cat([vs[i], nv])

    meta = make_meta(seqs, [1, 1, 1], alloc.block_size)
    assert not meta.is_prefill
    bk.append_kv(0, torch.cat(new_k), torch.cat(new_v), meta)

    scale = 1.0 / (bk.head_dim**0.5)
    q = torch.randn(3, bk.n_heads, bk.head_dim, dtype=DT)
    got = bk.attend(q, 0, scale, meta)

    for i in range(3):
        want = naive_attention(q[i : i + 1], ks[i], vs[i], scale, bk.n_kv_heads)
        torch.testing.assert_close(got[i : i + 1], want, **TOL)


# --------------------------------------------------------------------------
# 3. block straddle (R8)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 15, 16, 17, 31, 32, 33])
def test_block_straddle(n):
    """
    Lengths either side of every block boundary, block_size=16.

    n=16 and n=32 are the ones that matter: an exact multiple must report
    last_page_len == block_size, NOT 0. Get it wrong and attention silently
    drops or invents a whole page. Nothing raises; the text stays fluent.

    The pool is pre-poisoned so that the unwritten tail of the final page holds
    huge values: if the gather read past kv_len, the softmax would be dominated
    by garbage and the comparison would fail loudly instead of quietly passing
    on lucky zeros.
    """
    bk = make_backend()
    alloc = BlockAllocator(num_blocks=32, block_size=16)
    for pool in (bk.k_pool[0], bk.v_pool[0]):
        pool.fill_(1e3)

    seqs, ks, vs = write_history(bk, alloc, [n], seed=n)
    assert seqs[0].last_page_len() == (n % 16 or 16)

    meta = make_meta(seqs, [n], alloc.block_size)
    scale = 1.0 / (bk.head_dim**0.5)
    q = torch.randn(n, bk.n_heads, bk.head_dim, dtype=DT)

    got = bk.attend(q, 0, scale, meta)
    want = naive_attention(q, ks[0], vs[0], scale, bk.n_kv_heads)
    torch.testing.assert_close(got, want, **TOL)


# --------------------------------------------------------------------------
# 4. non-contiguous physical pages
# --------------------------------------------------------------------------


def test_non_contiguous_pages_match_contiguous():
    """
    Fragment the allocator so a sequence's pages are scattered and out of order,
    then assert the output is bit-identical to the contiguous case.

    A gather that accidentally assumes contiguity — `pool[first : first + n]`
    instead of `pool[page_ids]` — passes every test that allocates from a fresh
    pool, because a fresh pool always hands out consecutive ids. This is the
    test that fails.
    """
    lengths = [40]  # 3 pages
    scale = 1.0 / (8**0.5)

    # -- contiguous: fresh allocator, pages 0,1,2 --------------------------
    bk_a = make_backend()
    alloc_a = BlockAllocator(num_blocks=32, block_size=16)
    seqs_a, ks, vs = write_history(bk_a, alloc_a, lengths, seed=7)
    assert seqs_a[0].block_ids == [0, 1, 2]
    meta_a = make_meta(seqs_a, lengths, 16)

    # -- fragmented: force scattered, non-monotonic page ids ---------------
    bk_b = make_backend()
    alloc_b = BlockAllocator(num_blocks=32, block_size=16)
    filler = alloc_b.allocate(32)  # take everything
    assert filler == list(range(32))
    # FIFO free list, so freeing in this order hands them back in this order.
    alloc_b.free([21, 4, 13])
    alloc_b.check_invariants()

    s_b = SequenceBlocks(alloc_b, seq_id=0)
    s_b.append(40)
    assert s_b.block_ids == [21, 4, 13], "expected scattered, non-monotonic pages"

    meta_b = make_meta([s_b], lengths, 16)
    bk_b.append_kv(0, ks[0], vs[0], meta_b)

    q = torch.randn(40, bk_a.n_heads, bk_a.head_dim, dtype=DT)
    out_a = bk_a.attend(q, 0, scale, meta_a)
    out_b = bk_b.attend(q, 0, scale, meta_b)

    assert torch.equal(out_a, out_b), "physical page order changed the answer"
    # And both are still right in absolute terms.
    torch.testing.assert_close(
        out_b, naive_attention(q, ks[0], vs[0], scale, bk_a.n_kv_heads), **TOL
    )


def test_page_order_is_logical_not_sorted():
    """
    Pages must be read in the order the block table lists them, not sorted.

    Descending page ids make the two orders maximally different: if the backend
    sorted (or the gather implicitly assumed monotonic ids), history would be
    reversed page-wise and the answer would change.
    """
    bk = make_backend()
    alloc = BlockAllocator(num_blocks=8, block_size=16)
    alloc.allocate(8)
    alloc.free([7, 3, 0])  # sequence will get pages [7, 3, 0] — strictly descending

    s = SequenceBlocks(alloc)
    s.append(35)
    assert s.block_ids == [7, 3, 0]

    meta = make_meta([s], [35], 16)
    k = torch.randn(35, bk.n_kv_heads, bk.head_dim, dtype=DT)
    v = torch.randn(35, bk.n_kv_heads, bk.head_dim, dtype=DT)
    bk.append_kv(0, k, v, meta)

    scale = 1.0 / (bk.head_dim**0.5)
    q = torch.randn(35, bk.n_heads, bk.head_dim, dtype=DT)
    torch.testing.assert_close(
        bk.attend(q, 0, scale, meta), naive_attention(q, k, v, scale, bk.n_kv_heads), **TOL
    )


# --------------------------------------------------------------------------
# 5. GQA head mapping
# --------------------------------------------------------------------------


def test_gqa_value_mapping_is_floor_divide():
    """
    Query head h reads KV head h // (n_heads // n_kv_heads).

    Made directly observable by using kv_len == 1: with a single key the softmax
    is exactly 1.0, so the output IS the value vector, and each KV head is given
    a distinct constant. `repeat` instead of `repeat_interleave` would produce
    h % n_kv_heads — it runs, it is plausible, and this assert catches it.
    """
    n_kv_heads, n_heads, head_dim = 2, 6, 8
    groups = n_heads // n_kv_heads
    bk = make_backend(n_kv_heads=n_kv_heads, n_heads=n_heads, head_dim=head_dim)
    alloc = BlockAllocator(num_blocks=8, block_size=16)

    s = SequenceBlocks(alloc)
    s.append(1)
    meta = make_meta([s], [1], 16)

    k = torch.zeros(1, n_kv_heads, head_dim, dtype=DT)
    v = torch.zeros(1, n_kv_heads, head_dim, dtype=DT)
    for j in range(n_kv_heads):
        v[0, j] = float(j + 1)  # distinct per KV head
    bk.append_kv(0, k, v, meta)

    q = torch.randn(1, n_heads, head_dim, dtype=DT)
    out = bk.attend(q, 0, 1.0, meta)

    for h in range(n_heads):
        expected = float(h // groups + 1)
        assert torch.allclose(out[0, h], torch.full((head_dim,), expected, dtype=DT)), (
            f"query head {h} read KV head with value {out[0, h, 0]}, expected "
            f"KV head {h // groups} (value {expected})"
        )


def test_gqa_key_mapping_is_floor_divide():
    """
    The same mapping on the KEY side, which the value test above cannot see.

    Each KV head j gets two orthogonal keys: key0 along basis vector j, key1
    along basis vector n_kv_heads + j. Query head h points along basis vector
    h // groups, so under the CORRECT mapping it aligns with its own KV head's
    key0 and the softmax collapses onto it (value +1). Under h % n_kv_heads the
    query is orthogonal to both of that head's keys, the softmax goes uniform,
    and the output lands near 0.
    """
    n_kv_heads, n_heads, head_dim = 2, 4, 8
    groups = n_heads // n_kv_heads
    bk = make_backend(n_kv_heads=n_kv_heads, n_heads=n_heads, head_dim=head_dim)
    alloc = BlockAllocator(num_blocks=8, block_size=16)

    s = SequenceBlocks(alloc)
    s.append(2)
    meta = make_meta([s], [2], 16)

    k = torch.zeros(2, n_kv_heads, head_dim, dtype=DT)
    v = torch.zeros(2, n_kv_heads, head_dim, dtype=DT)
    for j in range(n_kv_heads):
        k[0, j, j] = 50.0
        k[1, j, n_kv_heads + j] = 50.0
        v[0, j] = 1.0
        v[1, j] = -1.0
    bk.append_kv(0, k, v, meta)

    # Query both positions; look at the LAST query, which may see both keys.
    q = torch.zeros(2, n_heads, head_dim, dtype=DT)
    for h in range(n_heads):
        q[:, h, h // groups] = 1.0

    out = bk.attend(q, 0, 1.0, meta)
    for h in range(n_heads):
        assert out[1, h, 0] > 0.99, (
            f"query head {h} did not lock onto KV head {h // groups}'s first key "
            f"(got {out[1, h, 0]:.3f}, expected ~1.0)"
        )


# --------------------------------------------------------------------------
# 6. multi-sequence isolation
# --------------------------------------------------------------------------


def test_sequences_do_not_see_each_others_kv():
    """
    Each sequence gets a constant, distinguishable V. With one key per sequence
    the output is that constant exactly, so any leakage across sequences shows
    up as a value that belongs to a neighbour.
    """
    bk = make_backend(n_kv_heads=2, n_heads=4, head_dim=8)
    alloc = BlockAllocator(num_blocks=32, block_size=16)
    lengths = [3, 1, 20]

    seqs = []
    ks, vs = [], []
    for i, n in enumerate(lengths):
        s = SequenceBlocks(alloc, seq_id=i)
        s.append(n)
        seqs.append(s)
        # Zero keys => uniform softmax; constant values => output == that constant.
        ks.append(torch.zeros(n, bk.n_kv_heads, bk.head_dim, dtype=DT))
        vs.append(torch.full((n, bk.n_kv_heads, bk.head_dim), float(i + 1), dtype=DT))

    meta = make_meta(seqs, lengths, 16)
    bk.append_kv(0, torch.cat(ks), torch.cat(vs), meta)

    q = torch.randn(sum(lengths), bk.n_heads, bk.head_dim, dtype=DT)
    out = bk.attend(q, 0, 1.0, meta)

    cu = meta.cu_query_lens.tolist()
    for i in range(len(lengths)):
        seg = out[cu[i] : cu[i + 1]]
        assert torch.allclose(seg, torch.full_like(seg, float(i + 1)), atol=1e-5), (
            f"sequence {i} saw KV that is not its own: values {seg.unique().tolist()}"
        )


# --------------------------------------------------------------------------
# 7. mixed prefill + decode
# --------------------------------------------------------------------------


def test_mixed_prefill_and_decode_matches_running_each_alone():
    """
    query_lens = [4, 1, 1]: one chunked-prefill sequence batched with two
    decodes — the thing continuous batching exists to do.

    Batching must be an implementation detail, so the batched result has to
    equal the results of running each sequence in its own batch of one. A
    per-sequence causal offset computed from batch-wide quantities instead of
    per-sequence ones fails exactly here and nowhere else.
    """
    bk = make_backend()
    alloc = BlockAllocator(num_blocks=32, block_size=16)
    scale = 1.0 / (bk.head_dim**0.5)

    # History: seq0 has 6 tokens then prefills 4 more; seq1/seq2 decode one each.
    hist = [6, 18, 16]
    seqs, ks, vs = write_history(bk, alloc, hist, seed=11)

    query_lens = [4, 1, 1]
    new_k, new_v = [], []
    for i, (s, ql) in enumerate(zip(seqs, query_lens, strict=False)):
        s.append(ql)
        nk = torch.randn(ql, bk.n_kv_heads, bk.head_dim, dtype=DT)
        nv = torch.randn(ql, bk.n_kv_heads, bk.head_dim, dtype=DT)
        new_k.append(nk)
        new_v.append(nv)
        ks[i] = torch.cat([ks[i], nk])
        vs[i] = torch.cat([vs[i], nv])

    meta = make_meta(seqs, query_lens, 16)
    bk.append_kv(0, torch.cat(new_k), torch.cat(new_v), meta)

    q = torch.randn(sum(query_lens), bk.n_heads, bk.head_dim, dtype=DT)
    batched = bk.attend(q, 0, scale, meta)

    cu = meta.cu_query_lens.tolist()
    for i, ql in enumerate(query_lens):
        # Same backend, same pool, but a batch containing only sequence i.
        solo_meta = make_meta([seqs[i]], [ql], 16)
        solo = bk.attend(q[cu[i] : cu[i + 1]], 0, scale, solo_meta)
        assert torch.equal(batched[cu[i] : cu[i + 1]], solo), (
            f"sequence {i} got a different answer when batched"
        )
        # ...and both agree with the dense reference.
        torch.testing.assert_close(
            solo, naive_attention(q[cu[i] : cu[i + 1]], ks[i], vs[i], scale, bk.n_kv_heads), **TOL
        )


# --------------------------------------------------------------------------
# 8. causality
# --------------------------------------------------------------------------


def test_prefill_query_cannot_attend_to_future_keys():
    """
    A prefill query at position p must not attend to keys > p.

    Tested by overwriting the LAST key/value in the sequence with an extreme
    vector after a baseline run. Queries 0..n-2 must be bit-identical to the
    baseline; only the final query, which is legitimately allowed to see that
    key, may change. If causality were broken the extreme value would swamp
    every earlier query's softmax and the equality would fail.
    """
    bk = make_backend()
    alloc = BlockAllocator(num_blocks=32, block_size=16)
    scale = 1.0 / (bk.head_dim**0.5)
    n = 6

    seqs, ks, vs = write_history(bk, alloc, [n], seed=21)
    meta = make_meta(seqs, [n], 16)
    q = torch.randn(n, bk.n_heads, bk.head_dim, dtype=DT)
    baseline = bk.attend(q, 0, scale, meta)

    # Poison the final position's KV in place, via its physical slot.
    slot = seqs[0].slot_for(n - 1)
    bk.k_pool[0].view(-1, bk.n_kv_heads, bk.head_dim)[slot] = 100.0
    bk.v_pool[0].view(-1, bk.n_kv_heads, bk.head_dim)[slot] = -999.0

    poisoned = bk.attend(q, 0, scale, meta)

    assert torch.equal(baseline[: n - 1], poisoned[: n - 1]), (
        "an earlier query's output changed when a LATER key changed — the mask "
        "is not causal"
    )
    assert not torch.allclose(baseline[n - 1], poisoned[n - 1]), (
        "the last query ignored a key it is allowed to see — the mask is too strict"
    )


def test_decode_sees_the_token_just_appended():
    """
    The mirror-image failure: too *much* masking. A decoding token must attend
    to itself, which is why append_kv is ordered before attend
    (components_gpu.py:162-163). With kv_len == 1 the only key is the new one,
    so the output must be exactly that new value.
    """
    bk = make_backend()
    alloc = BlockAllocator(num_blocks=8, block_size=16)
    s = SequenceBlocks(alloc)
    s.append(1)
    meta = make_meta([s], [1], 16)

    k = torch.zeros(1, bk.n_kv_heads, bk.head_dim, dtype=DT)
    v = torch.full((1, bk.n_kv_heads, bk.head_dim), 7.0, dtype=DT)
    bk.append_kv(0, k, v, meta)

    out = bk.attend(torch.randn(1, bk.n_heads, bk.head_dim, dtype=DT), 0, 1.0, meta)
    assert torch.allclose(out, torch.full_like(out, 7.0))


# --------------------------------------------------------------------------
# accounting + guardrails
# --------------------------------------------------------------------------


def test_pool_accounting():
    """pool_bytes must make sizing computable, not guessable (ARCHITECTURE §3.1)."""
    bk = PagedTorchBackend(
        num_layers=16,
        num_blocks=100,
        block_size=16,
        n_kv_heads=8,
        n_heads=32,
        head_dim=64,
        device=DEV,
        dtype=torch.float16,
    )
    # 2 (K,V) x 16 layers x 16 tokens x 8 kv_heads x 64 head_dim x 2 bytes = 512 KiB/block
    assert bk.block_bytes() == 512 * 1024
    assert bk.pool_bytes() == 100 * 512 * 1024
    assert bk.pool_bytes() == PagedTorchBackend.estimate_pool_bytes(
        16, 100, 16, 8, 64, torch.float16
    )
    assert bk.tokens_capacity() == 1600
    assert bk.zeros_like_pool().shape == (100, 16, 8, 64)


def test_rejects_indivisible_head_counts():
    with pytest.raises(ValueError, match="divisible"):
        make_backend(n_kv_heads=3, n_heads=4)


def test_rejects_bad_layer_index():
    bk = make_backend(num_layers=2)
    alloc = BlockAllocator(num_blocks=8, block_size=16)
    s = SequenceBlocks(alloc)
    s.append(1)
    meta = make_meta([s], [1], 16)
    with pytest.raises(ValueError, match="layer_idx"):
        bk.attend(torch.zeros(1, bk.n_heads, bk.head_dim, dtype=DT), 2, 1.0, meta)


def test_rejects_inconsistent_last_page_len():
    """
    The silent off-by-one, caught loudly. kv_last_page_len == 0 for a length
    that is an exact multiple of block_size would drop a whole page of keys.
    """
    bk = make_backend()
    alloc = BlockAllocator(num_blocks=8, block_size=16)
    s = SequenceBlocks(alloc)
    s.append(16)
    meta = make_meta([s], [16], 16)
    assert int(meta.kv_last_page_len[0]) == 16  # NOT 0

    bad = torch.tensor([9], dtype=torch.int32)
    with pytest.raises(ValueError, match="kv_last_page_len"):
        bk._gather(0, meta.kv_indices.long(), 16, int(bad[0]))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fp16_on_gpu_matches_cpu_fp32_closely():
    """
    The production configuration: fp16 pool on CUDA. Tolerance is fp16's, not
    fp32's — the point is that the paging logic is unchanged, not that fp16 is
    exact.
    """
    lengths = [5, 17]
    scale = 1.0 / (8**0.5)

    cpu = make_backend()
    alloc = BlockAllocator(num_blocks=32, block_size=16)
    seqs, ks, vs = write_history(cpu, alloc, lengths, seed=3)
    meta = make_meta(seqs, lengths, 16)
    q = torch.randn(sum(lengths), cpu.n_heads, cpu.head_dim, dtype=DT)
    want = cpu.attend(q, 0, scale, meta)

    gpu = PagedTorchBackend(1, 32, 16, 2, 4, 8, device="cuda:0", dtype=torch.float16)
    gpu.append_kv(0, torch.cat(ks).half().cuda(), torch.cat(vs).half().cuda(), meta)
    got = gpu.attend(q.half().cuda(), 0, scale, meta)

    torch.testing.assert_close(got.float().cpu(), want, atol=2e-2, rtol=2e-2)
