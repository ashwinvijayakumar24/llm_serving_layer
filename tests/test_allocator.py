"""
Paged KV allocator and per-sequence block table tests. Pure CPU, no GPU, no
torch, no weights — runs in CI on every push.

That is the point of the design: the allocator hands out integers, so the two
risks that have no other symptom can be pinned down exhaustively for free.

    R7  a live block handed to a second owner. Attention reads another
        sequence's KV. Output stays fluent, no metric moves, no error.
    R8  slot_mapping / last_page_len off-by-one at block boundaries. Writes KV
        to the wrong physical slot, or drops a whole page of keys. Also silent,
        and invisible at batch 1 with short prompts that never cross a boundary.

Every test below names the failure mode it catches and why that failure would
otherwise go unnoticed.

    pytest tests/test_allocator.py -v
"""

import random

import pytest

from serving.memory.allocator import AllocationError, BlockAllocator
from serving.memory.block_table import SequenceBlocks, build_csr

BS = 16  # block size used throughout, matching the engine's page_size


def make_alloc(num_blocks=32, block_size=BS, watermark_blocks=0):
    return BlockAllocator(num_blocks, block_size, watermark_blocks)


# ==========================================================================
# BlockAllocator — construction
# ==========================================================================


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"num_blocks": 0}, "num_blocks"),
        ({"num_blocks": -1}, "num_blocks"),
        ({"block_size": 0}, "block_size"),
        ({"block_size": -8}, "block_size"),
        ({"watermark_blocks": -1}, "watermark_blocks"),
        ({"watermark_blocks": 32}, "watermark_blocks"),   # == num_blocks
        ({"watermark_blocks": 99}, "watermark_blocks"),
    ],
)
def test_construction_validation(kwargs, needle):
    """
    A degenerate pool is not a small pool — it is an allocator that can never
    hand out a block, so every admission decision below it silently becomes
    "reject" and the server looks merely slow rather than misconfigured. A
    watermark >= num_blocks is the same thing with a subtler cause: admission
    can never succeed even with an empty pool.
    """
    base = dict(num_blocks=32, block_size=BS, watermark_blocks=0)
    base.update(kwargs)
    with pytest.raises(ValueError, match=needle):
        BlockAllocator(**base)


def test_watermark_at_upper_bound_is_exclusive():
    """num_blocks - 1 is legal, num_blocks is not. Pin the boundary explicitly."""
    BlockAllocator(8, BS, watermark_blocks=7)          # must not raise
    with pytest.raises(ValueError):
        BlockAllocator(8, BS, watermark_blocks=8)


def test_fresh_allocator_is_fully_free():
    a = make_alloc(num_blocks=32)
    assert a.num_free == 32
    assert a.num_used == 0
    assert a.utilization == 0.0
    assert a.tokens_capacity() == 32 * BS
    a.check_invariants()


# ==========================================================================
# BlockAllocator — allocate / free round trip
# ==========================================================================


def test_allocate_free_round_trip():
    """
    The baseline accounting. If num_free and num_used ever disagree with the
    refcount table, every later invariant check is measuring the wrong thing.
    """
    a = make_alloc(num_blocks=10)
    blocks = a.allocate(4)
    assert len(blocks) == 4
    assert len(set(blocks)) == 4, "allocate handed out the same block twice"
    assert a.num_free == 6
    assert a.num_used == 4
    assert all(a.refcount(b) == 1 for b in blocks)
    a.check_invariants()

    a.free(blocks)
    assert a.num_free == 10
    assert a.num_used == 0
    assert all(a.refcount(b) == 0 for b in blocks)
    a.check_invariants()


def test_allocate_zero_is_a_noop():
    """Admission may compute a need of 0 blocks for a sequence that still fits."""
    a = make_alloc(num_blocks=4)
    assert a.allocate(0) == []
    assert a.num_free == 4
    a.check_invariants()


def test_allocate_negative_raises():
    a = make_alloc()
    with pytest.raises(ValueError, match="non-negative"):
        a.allocate(-1)
    with pytest.raises(ValueError, match="non-negative"):
        a.can_allocate(-1)


def test_allocated_ids_are_in_range():
    """
    Block ids index the physical KV pool directly. An out-of-range id is an
    out-of-bounds GPU write, which on CUDA may corrupt an unrelated tensor
    rather than fault.
    """
    a = make_alloc(num_blocks=6)
    for b in a.allocate(6):
        assert 0 <= b < 6


# ==========================================================================
# BlockAllocator — all-or-nothing
# ==========================================================================


def test_over_allocation_raises_and_allocates_nothing():
    """
    ALL OR NOTHING (R7-adjacent). A partial allocation leaves the caller holding
    blocks for a sequence it cannot run; the natural next move is to abandon
    them without freeing, which leaks capacity that only shows up much later as
    a server that mysteriously admits fewer requests than it used to.
    """
    a = make_alloc(num_blocks=8)
    a.allocate(5)
    free_before = a.num_free
    used_before = a.num_used
    total_before = a.total_allocated

    with pytest.raises(AllocationError, match="Cannot allocate"):
        a.allocate(4)                                  # only 3 free

    assert a.num_free == free_before, "over-allocation consumed blocks before raising"
    assert a.num_used == used_before
    assert a.total_allocated == total_before, "stats counted a failed allocation"
    a.check_invariants()


def test_exact_capacity_allocation_succeeds():
    """The boundary next to the failure above: exactly num_free must be legal."""
    a = make_alloc(num_blocks=8)
    assert len(a.allocate(8)) == 8
    assert a.num_free == 0
    with pytest.raises(AllocationError):
        a.allocate(1)
    a.check_invariants()


# ==========================================================================
# BlockAllocator — watermark
# ==========================================================================


def test_can_allocate_respects_watermark():
    """
    can_allocate is the ADMISSION gate. Admitting until zero blocks remain
    guarantees preemption on the very next step, because every running sequence
    needs one more block as it grows. That failure does not look like a bug — it
    looks like thrashing throughput.
    """
    a = make_alloc(num_blocks=10, watermark_blocks=3)
    assert a.can_allocate(7) is True                    # leaves exactly 3
    assert a.can_allocate(8) is False                   # would leave 2
    assert a.can_allocate(0) is True


def test_allocate_ignores_watermark():
    """
    The other direction, and the one that is easy to get wrong by "helpfully"
    routing allocate() through can_allocate(). An already-admitted sequence must
    be able to take its next block; blocking it on headroom it is itself the
    reason for stalls a sequence that the scheduler already promised to run.
    """
    a = make_alloc(num_blocks=10, watermark_blocks=3)
    assert a.can_allocate(9) is False
    blocks = a.allocate(9)                              # must succeed anyway
    assert len(blocks) == 9
    assert a.num_free == 1
    a.check_invariants()


def test_watermark_and_allocate_disagree_only_in_the_headroom_band():
    """
    Sweep the whole range so the two predicates are pinned against each other
    rather than at one convenient point.
    """
    a = make_alloc(num_blocks=8, watermark_blocks=2)
    for n in range(0, 9):
        expected_admit = (8 - n) >= 2
        assert a.can_allocate(n) is expected_admit, n
    # allocate's own limit is free count, watermark-free.
    a.allocate(8)
    assert a.num_free == 0


def test_zero_watermark_makes_the_predicates_agree():
    a = make_alloc(num_blocks=5, watermark_blocks=0)
    for n in range(6):
        assert a.can_allocate(n) is True
    assert a.can_allocate(6) is False


# ==========================================================================
# BlockAllocator — reference counting (R7)
# ==========================================================================


def test_incref_delays_return_to_free_list():
    """
    R7. A prefix block shared by two sequences must survive the first sequence
    retiring. If the first free() returned it to the pool, the second sequence's
    attention would read whatever the next allocation writes there — fluent
    output, wrong content, no error anywhere.
    """
    a = make_alloc(num_blocks=8)
    (b,) = a.allocate(1)
    a.incref([b])
    assert a.refcount(b) == 2

    a.free([b])
    assert a.refcount(b) == 1
    assert a.num_free == 7, "block returned to the pool while still referenced"
    a.check_invariants()

    a.free([b])
    assert a.refcount(b) == 0
    assert a.num_free == 8
    a.check_invariants()


def test_block_shared_k_ways_needs_k_frees():
    """
    Generalises the above. An off-by-one in the refcount decrement path shows up
    only at K > 2, which a two-holder test cannot distinguish from "free always
    releases".
    """
    K = 5
    a = make_alloc(num_blocks=8)
    (b,) = a.allocate(1)
    for _ in range(K - 1):
        a.incref([b])
    assert a.refcount(b) == K

    for i in range(K - 1):
        a.free([b])
        assert a.num_free == 7, f"released after {i + 1} of {K} frees"
        a.check_invariants()

    a.free([b])
    assert a.num_free == 8
    assert a.total_freed == 1, "total_freed counts block releases, not decrements"


def test_double_free_raises():
    """
    R7, stated directly. A tolerated double free puts a live block back on the
    free list, where the next allocation hands it to a second owner. Nothing
    downstream can detect that: both sequences keep generating plausible text.
    Loud here beats fluent-but-wrong later.
    """
    a = make_alloc(num_blocks=4)
    blocks = a.allocate(2)
    a.free(blocks)
    with pytest.raises(AllocationError, match="Double free"):
        a.free(blocks)


def test_double_free_of_never_allocated_block_raises():
    """A free of a block nobody owns is the same corruption with a different cause."""
    a = make_alloc(num_blocks=4)
    with pytest.raises(AllocationError, match="Double free"):
        a.free([3])


def test_incref_on_free_block_raises():
    """
    A refcount-0 block is on the free list and its contents are garbage.
    increfing it would resurrect a block whose KV is stale — the reader would
    attend over whatever the previous owner left behind, which decodes to
    fluent nonsense rather than an error.
    """
    a = make_alloc(num_blocks=4)
    (b,) = a.allocate(1)
    a.free([b])
    with pytest.raises(AllocationError, match="refcount is 0"):
        a.incref([b])


@pytest.mark.parametrize("bad_id", [-1, 4, 100])
def test_block_id_range_validation(bad_id):
    """
    Out-of-range ids must be rejected at the boundary. Python's negative
    indexing makes -1 especially dangerous: it would silently address the LAST
    block instead of erroring.
    """
    a = make_alloc(num_blocks=4)
    with pytest.raises(ValueError, match="out of range"):
        a.refcount(bad_id)
    with pytest.raises(ValueError, match="out of range"):
        a.free([bad_id])
    with pytest.raises(ValueError, match="out of range"):
        a.incref([bad_id])


# ==========================================================================
# BlockAllocator — check_invariants catches corruption
# ==========================================================================


def test_check_invariants_passes_on_healthy_state():
    a = make_alloc(num_blocks=8)
    blocks = a.allocate(3)
    a.check_invariants()
    a.incref(blocks[:1])
    a.check_invariants()
    a.free(blocks)
    a.free(blocks[:1])
    a.check_invariants()


def test_check_invariants_catches_live_block_on_free_list():
    """
    THE R7 detector. A referenced block sitting on the free list is about to be
    handed to a second owner. No other check in the system can see this: the
    refcount table alone looks fine, and the free list alone looks fine.
    """
    a = make_alloc(num_blocks=4)
    (b,) = a.allocate(1)
    a._free.append(b)                                   # corrupt: live AND free
    with pytest.raises(AssertionError, match="second owner"):
        a.check_invariants()


def test_check_invariants_catches_leaked_block():
    """
    The inverse corruption: refcount 0 but absent from the free list. Nothing
    fails — capacity just quietly shrinks, so the server admits fewer requests
    over time and the symptom looks like load, not a bug.
    """
    a = make_alloc(num_blocks=4)
    (b,) = a.allocate(1)
    a._refcount[b] = 0                                  # dropped without freeing
    with pytest.raises(AssertionError, match="leaked"):
        a.check_invariants()


def test_check_invariants_catches_duplicate_free_list_entries():
    """
    A duplicated free-list entry means one physical block gets handed to two
    sequences on two different allocations — R7 again, arriving through the
    free list rather than the refcounts.
    """
    a = make_alloc(num_blocks=4)
    a._free.append(2)                                   # 2 is already free
    with pytest.raises(AssertionError, match="duplicate"):
        a.check_invariants()


def test_check_invariants_catches_negative_refcount():
    """A negative refcount means a decrement escaped the double-free guard."""
    a = make_alloc(num_blocks=4)
    a._refcount[1] = -1
    with pytest.raises(AssertionError):
        a.check_invariants()


# ==========================================================================
# BlockAllocator — FIFO reuse order
# ==========================================================================


def test_free_list_is_fifo_not_lifo():
    """
    A LIFO free list returns the just-freed block on the very next allocation,
    so a use-after-free reads data that still happens to look right and the bug
    stays invisible for as long as the workload is quiet. FIFO maximises the
    delay before reuse, turning the same bug into obvious garbage.

    This is a deliberate design property, not an implementation detail, so it is
    tested rather than left to chance.
    """
    a = make_alloc(num_blocks=8)
    first = a.allocate(4)                               # takes the 4 oldest
    a.free([first[0]])                                  # freed most recently
    (nxt,) = a.allocate(1)
    assert nxt != first[0], "free list is LIFO — use-after-free would be hidden"
    assert nxt not in first, "reused a still-held block"


def test_fifo_order_is_exact():
    """Pin the ordering itself, so a change to the data structure is caught."""
    a = make_alloc(num_blocks=4)
    assert a.allocate(4) == [0, 1, 2, 3]
    a.free([2])
    a.free([0])
    assert a.allocate(2) == [2, 0], "blocks must come back in free order"


# ==========================================================================
# BlockAllocator — the leak test (Phase 1 DoD, R7)
# ==========================================================================


def test_no_leak_after_many_allocate_free_cycles():
    """
    THE LEAK TEST named in the phase plan's DoD and in R7's detection column.

    After N sequential requests the free list must return to its initial count
    EXACTLY. A leak of one block per request is undetectable at small N — the
    server simply admits fewer requests as it runs, which is indistinguishable
    from rising load until capacity is gone.
    """
    a = make_alloc(num_blocks=32)
    initial_free = a.num_free

    for i in range(200):
        n = (i % 7) + 1
        blocks = a.allocate(n)
        if i % 3 == 0:                                  # exercise the shared path too
            a.incref(blocks)
            a.free(blocks)
        a.free(blocks)
        a.check_invariants()

    assert a.num_free == initial_free, f"leaked {initial_free - a.num_free} blocks"
    assert a.num_used == 0
    a.check_invariants()


def test_no_leak_through_sequence_lifecycle():
    """
    Same leak check one level up: SequenceBlocks owns references, so a sequence
    retired without free() leaks its whole block table at once.
    """
    a = make_alloc(num_blocks=32, block_size=BS)
    initial_free = a.num_free
    for i in range(100):
        s = SequenceBlocks(a, seq_id=i)
        s.append((i % 60) + 1)
        s.append(i % 5)
        s.free()
        a.check_invariants()
    assert a.num_free == initial_free


# ==========================================================================
# BlockAllocator — stats
# ==========================================================================


def test_stats_track_allocation_history():
    """
    Diagnostics are the only way to notice the allocator, not the GPU, is what
    is limiting throughput. total_* are cumulative history; num_* are current
    state, and conflating them makes utilization meaningless.
    """
    a = make_alloc(num_blocks=10)
    b1 = a.allocate(4)
    assert a.total_allocated == 4
    assert a.peak_used == 4
    assert a.utilization == pytest.approx(0.4)

    b2 = a.allocate(3)
    assert a.total_allocated == 7
    assert a.peak_used == 7
    assert a.utilization == pytest.approx(0.7)

    a.free(b1)
    assert a.total_freed == 4
    assert a.num_used == 3
    assert a.peak_used == 7, "peak_used must be a high-water mark, not current usage"
    assert a.utilization == pytest.approx(0.3)

    a.free(b2)
    assert a.total_freed == 7
    assert a.total_allocated == a.total_freed, "every allocated block was released"


def test_tokens_capacity_is_the_memory_claim():
    """
    tokens_capacity is the number Phase 1's memory claim is about. A wrong
    block_size here misstates capacity by a constant factor and every derived
    number inherits it.
    """
    assert make_alloc(num_blocks=100, block_size=16).tokens_capacity() == 1600
    assert make_alloc(num_blocks=7, block_size=32).tokens_capacity() == 224


def test_repr_does_not_raise():
    """repr lands in error messages and logs; a raising repr masks the real failure."""
    a = make_alloc(num_blocks=4)
    a.allocate(2)
    assert "BlockAllocator" in repr(a)


# ==========================================================================
# SequenceBlocks — growth
# ==========================================================================


def test_append_allocates_only_when_a_boundary_is_crossed():
    """
    Allocating a block per token would exhaust the pool ~block_size times too
    fast; allocating too late writes KV past the end of the last block, which is
    an out-of-bounds write with no exception (R8).
    """
    a = make_alloc(num_blocks=8, block_size=4)
    s = SequenceBlocks(a, seq_id=1)

    s.append(1)
    assert s.num_blocks == 1 and s.num_tokens == 1
    for extra in (1, 1, 1):                             # fills the block to 4
        s.append(extra)
    assert s.num_blocks == 1 and s.num_tokens == 4
    assert a.num_used == 1

    s.append(1)                                         # crosses into block 2
    assert s.num_blocks == 2 and s.num_tokens == 5
    assert a.num_used == 2, "crossing a boundary must allocate exactly one block"


def test_append_multi_token_allocates_exactly_enough():
    """A prefill appends many tokens at once; the block count must match ceil()."""
    a = make_alloc(num_blocks=32, block_size=BS)
    s = SequenceBlocks(a)
    s.append(33)
    assert s.num_blocks == 3                            # ceil(33/16)
    assert s.capacity == 48
    assert a.num_used == 3


def test_append_zero_and_negative():
    a = make_alloc(num_blocks=8, block_size=4)
    s = SequenceBlocks(a)
    s.append(0)
    assert s.num_tokens == 0 and s.num_blocks == 0
    with pytest.raises(ValueError, match="non-negative"):
        s.append(-1)


def test_append_beyond_pool_raises_and_leaves_state_intact():
    """
    If this raises, the scheduler failed to preempt in time. The sequence must
    remain usable and consistent so the scheduler can preempt and retry rather
    than inheriting a half-grown block table.
    """
    a = make_alloc(num_blocks=2, block_size=4)
    s = SequenceBlocks(a)
    s.append(8)                                         # takes both blocks
    with pytest.raises(AllocationError):
        s.append(1)
    assert s.num_tokens == 8, "token count advanced despite a failed allocation"
    assert s.num_blocks == 2
    a.check_invariants()


@pytest.mark.parametrize(
    "existing,new,expected",
    [
        (0, 0, 0),
        (0, 1, 1),
        (0, 16, 1),                                     # exact fit, still one block
        (0, 17, 2),
        (16, 0, 0),                                     # full block, nothing new
        (16, 1, 1),                                     # first token of the next block
        (15, 1, 0),                                     # fits in the existing block
        (15, 2, 1),
        (32, 16, 1),
        (32, 17, 2),
    ],
)
def test_blocks_needed_for_at_block_boundaries(existing, new, expected):
    """
    R8's arithmetic, one step earlier. Admission uses this to decline a request
    it cannot house; an off-by-one here either rejects requests that fit
    (throughput loss, looks like load) or admits one it cannot house (an
    AllocationError mid-forward-pass).

    The cases at exact multiples of block_size are the ones a naive
    `total // block_size` gets wrong.
    """
    a = make_alloc(num_blocks=64, block_size=BS)
    s = SequenceBlocks(a)
    if existing:
        s.append(existing)
    assert s.blocks_needed_for(new) == expected


def test_blocks_needed_for_negative_raises():
    a = make_alloc(num_blocks=8)
    s = SequenceBlocks(a)
    with pytest.raises(ValueError, match="non-negative"):
        s.blocks_needed_for(-1)


# ==========================================================================
# SequenceBlocks — slot arithmetic (R8)
# ==========================================================================


def scatter_allocator(block_size=4, num_blocks=16):
    """
    Return an allocator whose free list is NON-CONTIGUOUS.

    Allocate everything one block at a time, then release every other block.
    The FIFO free list then holds [1, 3, 5, ...], so the next sequence's block
    table is genuinely scattered. This matters: a slot_for() implementation that
    assumes physical contiguity passes every test built on a fresh pool, because
    a fresh pool hands out 0,1,2,3... in order.
    """
    a = BlockAllocator(num_blocks, block_size)
    singles = [a.allocate(1)[0] for _ in range(num_blocks)]
    a.free([b for i, b in enumerate(singles) if i % 2 == 0])
    return a


def test_slot_for_with_non_contiguous_blocks():
    """
    R8 in its real form. slot = block_ids[p // bs] * bs + (p % bs). With
    contiguous ids the indirection is the identity, so a naive implementation
    that ignores block_ids entirely still passes. Only scattered ids
    distinguish them — and the wrong version writes KV into another sequence's
    block, which produces fluent, wrong text and no error.
    """
    bs = 4
    a = scatter_allocator(block_size=bs)
    s = SequenceBlocks(a, seq_id=7)
    s.append(4 * bs)                                    # 4 blocks

    ids = s.block_ids
    assert ids != list(range(ids[0], ids[0] + len(ids))), (
        "test setup failed: blocks came out contiguous, so this test proves nothing"
    )

    for p in range(s.num_tokens):
        expected = ids[p // bs] * bs + (p % bs)
        assert s.slot_for(p) == expected, f"position {p}"


def test_slot_for_straddles_every_block_boundary():
    """
    Walk the last position of one block and the first of the next, at every
    boundary. R8's detection column calls for lengths chosen to straddle
    boundaries explicitly rather than left to chance.
    """
    bs = 4
    a = scatter_allocator(block_size=bs)
    s = SequenceBlocks(a)
    s.append(4 * bs)
    ids = s.block_ids

    for k in range(1, len(ids)):
        last_of_prev = k * bs - 1
        first_of_next = k * bs
        assert s.slot_for(last_of_prev) == ids[k - 1] * bs + (bs - 1)
        assert s.slot_for(first_of_next) == ids[k] * bs
        assert s.slot_for(first_of_next) != s.slot_for(last_of_prev) + 1 or (
            ids[k] == ids[k - 1] + 1
        ), "consecutive slots across a boundary imply contiguity that does not hold"


def test_slots_are_globally_unique_across_sequences():
    """
    Two sequences must never map a position to the same physical slot. That is
    R7 observed from the addressing side: identical slots mean one sequence's
    KV write lands on the other's cache.
    """
    bs = 4
    a = BlockAllocator(16, bs)
    seqs = [SequenceBlocks(a, seq_id=i) for i in range(4)]
    for i, s in enumerate(seqs):
        s.append(bs * (i + 1))

    all_slots = [sl for s in seqs for sl in s.slots_for_range(0, s.num_tokens)]
    assert len(all_slots) == len(set(all_slots)), "two sequences share a physical slot"


@pytest.mark.parametrize("position", [-1, 10, 11, 100])
def test_slot_for_out_of_range_raises(position):
    """
    Positions must be bounded by num_tokens, not by capacity. A position in the
    allocated-but-unwritten tail of the last block addresses uninitialised KV,
    and Python's negative indexing would silently address the last block.
    """
    a = make_alloc(num_blocks=8, block_size=4)
    s = SequenceBlocks(a)
    s.append(10)                                        # capacity 12, tokens 10
    with pytest.raises(IndexError):
        s.slot_for(position)


def test_slots_for_new_tokens_returns_the_tail():
    """
    slot_mapping is built AFTER append(), from the tail of the sequence. Taking
    the head instead overwrites the prompt's KV with the new tokens' KV —
    silently, and only for sequences long enough for the two ranges to differ.
    """
    bs = 4
    a = scatter_allocator(block_size=bs)
    s = SequenceBlocks(a)
    s.append(10)                                        # prefill
    s.append(3)                                         # three new tokens

    new = s.slots_for_new_tokens(3)
    assert new == [s.slot_for(10), s.slot_for(11), s.slot_for(12)]
    assert len(new) == 3
    assert new == s.slots_for_range(10, 3)


def test_slots_for_new_tokens_single_decode_step():
    """The common case: one token per step, and it must be the LAST position."""
    a = scatter_allocator(block_size=4)
    s = SequenceBlocks(a)
    s.append(7)
    s.append(1)
    assert s.slots_for_new_tokens(1) == [s.slot_for(7)]


def test_slots_for_new_tokens_spanning_a_boundary():
    """New tokens that cross into a freshly allocated block must use its id."""
    bs = 4
    a = scatter_allocator(block_size=bs)
    s = SequenceBlocks(a)
    s.append(3)
    s.append(3)                                         # positions 3,4,5 — crosses at 4
    ids = s.block_ids
    assert s.slots_for_new_tokens(3) == [
        ids[0] * bs + 3,
        ids[1] * bs + 0,
        ids[1] * bs + 1,
    ]


# ==========================================================================
# SequenceBlocks — last_page_len (R8 / R9, the silent off-by-one)
# ==========================================================================


@pytest.mark.parametrize("n_tokens", list(range(1, 41)))
def test_last_page_len_sweep(n_tokens):
    """
    THE OFF-BY-ONE, swept.

    FlashInfer requires 1 <= kv_last_page_len <= page_size, and an exact
    multiple of page_size reports page_size, NOT 0
    (flashinfer-python 0.6.16; RISK_REGISTER R9). Report 0 and attention
    silently drops a whole page of keys; report page_size + 1 and it reads a
    page of uninitialised memory. Output stays fluent either way — there is no
    exception, no NaN, and no metric that moves.

    num_tokens=16 and 32 with block_size=16 are the smallest failing cases, so
    the sweep covers 1..40 and asserts them by name below.
    """
    a = make_alloc(num_blocks=8, block_size=BS)
    s = SequenceBlocks(a)
    s.append(n_tokens)

    lpl = s.last_page_len()
    assert 1 <= lpl <= BS, f"num_tokens={n_tokens} gave last_page_len={lpl}, outside [1,16]"

    expected = n_tokens - (s.num_blocks - 1) * BS
    assert lpl == expected, f"num_tokens={n_tokens}"
    # Tokens accounted for by the full pages plus the last page must be exact.
    assert (s.num_blocks - 1) * BS + lpl == n_tokens


@pytest.mark.parametrize("n_tokens", [16, 32, 48, 64])
def test_last_page_len_at_exact_multiples_is_block_size_not_zero(n_tokens):
    """
    Called out separately from the sweep so a failure is unmistakable in the
    report. An exact multiple of block_size MUST report block_size. Zero is the
    natural output of `num_tokens % block_size` and is exactly wrong.
    """
    a = make_alloc(num_blocks=8, block_size=BS)
    s = SequenceBlocks(a)
    s.append(n_tokens)
    assert s.last_page_len() == BS, (
        f"num_tokens={n_tokens} is an exact multiple of {BS}: last_page_len must be "
        f"{BS}, not 0 — 0 makes attention drop a whole page of keys, silently"
    )


def test_last_page_len_after_incremental_appends_matches_bulk():
    """
    Decode reaches length N one token at a time; prefill reaches it in one call.
    Both must report the same last page, or the same sequence attends
    differently depending on how it got there.
    """
    for n in range(1, 35):
        a1 = make_alloc(num_blocks=8, block_size=BS)
        bulk = SequenceBlocks(a1)
        bulk.append(n)

        a2 = make_alloc(num_blocks=8, block_size=BS)
        incr = SequenceBlocks(a2)
        for _ in range(n):
            incr.append(1)

        assert incr.last_page_len() == bulk.last_page_len(), n
        assert incr.num_blocks == bulk.num_blocks, n


def test_last_page_len_on_empty_sequence_raises():
    """
    A zero-token sequence has no last page. Returning 0 would be indistinguishable
    from the exact-multiple case above, so the two silent errors would alias.
    A sequence with no tokens should never reach a batch.
    """
    a = make_alloc(num_blocks=4)
    s = SequenceBlocks(a)
    with pytest.raises(ValueError, match="empty sequence"):
        s.last_page_len()


# ==========================================================================
# SequenceBlocks — lifecycle
# ==========================================================================


def test_free_returns_exactly_the_blocks_it_held():
    """
    Freeing too few leaks capacity; freeing too many is R7. The allocator's free
    count is the ledger both are checked against.
    """
    a = make_alloc(num_blocks=16, block_size=4)
    before = a.num_free
    s = SequenceBlocks(a)
    s.append(9)                                         # 3 blocks
    held = list(s.block_ids)
    assert a.num_free == before - 3

    s.free()
    assert a.num_free == before
    assert all(a.refcount(b) == 0 for b in held)
    a.check_invariants()


def test_free_is_idempotent():
    """
    A scheduler may retire a sequence on a path that also runs during
    cancellation. Making the second call an error trades a leak for a crash; a
    second call that actually re-freed the blocks would be R7.
    """
    a = make_alloc(num_blocks=8, block_size=4)
    s = SequenceBlocks(a)
    s.append(5)
    s.free()
    free_after_first = a.num_free
    s.free()                                            # must not raise
    s.free()
    assert a.num_free == free_after_first, "second free released blocks it no longer owned"
    assert s.is_freed
    a.check_invariants()


@pytest.mark.parametrize(
    "op",
    [
        lambda s: s.append(1),
        lambda s: s.slot_for(0),
        lambda s: s.slots_for_new_tokens(1),
        lambda s: s.last_page_len(),
    ],
)
def test_using_a_freed_sequence_raises(op):
    """
    A freed sequence's blocks may already belong to someone else. Using it would
    read or write another sequence's KV — R7 arriving through a stale handle
    rather than a stale refcount. The handle must fail loudly.
    """
    a = make_alloc(num_blocks=8, block_size=4)
    s = SequenceBlocks(a)
    s.append(5)
    s.free()
    with pytest.raises(RuntimeError, match="was freed"):
        op(s)


def test_freed_sequence_repr_does_not_raise():
    """repr is used in the very error messages that report this state."""
    a = make_alloc(num_blocks=4)
    s = SequenceBlocks(a)
    s.append(3)
    assert "3 tokens" in repr(s)
    s.free()
    assert "freed" in repr(s)


def test_len_matches_num_tokens():
    a = make_alloc(num_blocks=8, block_size=4)
    s = SequenceBlocks(a)
    s.append(7)
    assert len(s) == 7 == s.num_tokens


# ==========================================================================
# build_csr
# ==========================================================================


def make_batch(lengths, block_size=BS, num_blocks=64, scatter=True):
    """Build a batch of sequences, optionally over a fragmented free list."""
    if scatter:
        a = scatter_allocator(block_size=block_size, num_blocks=num_blocks)
    else:
        a = BlockAllocator(num_blocks, block_size)
    seqs = []
    for i, n in enumerate(lengths):
        s = SequenceBlocks(a, seq_id=i)
        s.append(n)
        seqs.append(s)
    return a, seqs


def test_build_csr_shapes_and_contents():
    """
    The CSR triple is what the attention backend dereferences. kv_indptr with a
    wrong length silently shifts which pages belong to which sequence — every
    sequence then attends over its neighbour's KV, fluently.
    """
    lengths = [1, 16, 17, 40, 33]
    a, seqs = make_batch(lengths)
    kv_indptr, kv_indices, last_page = build_csr(seqs)

    assert len(kv_indptr) == len(seqs) + 1, "kv_indptr must have n_seqs + 1 entries"
    assert len(last_page) == len(seqs)
    assert len(kv_indices) == sum(s.num_blocks for s in seqs)
    assert last_page == [s.last_page_len() for s in seqs]
    a.check_invariants()


def test_build_csr_indptr_endpoints_and_page_counts():
    """
    kv_indptr[0] == 0 and kv_indptr[-1] == len(kv_indices) are the two
    conditions that make the slicing well-defined. Per-sequence page counts must
    equal ceil(num_tokens / block_size) — the boundary cases (exact multiples)
    are where a `//` instead of a ceil silently drops the final page.
    """
    lengths = [1, 15, 16, 17, 31, 32, 33]
    _, seqs = make_batch(lengths)
    kv_indptr, kv_indices, _ = build_csr(seqs)

    assert kv_indptr[0] == 0
    assert kv_indptr[-1] == len(kv_indices)
    assert kv_indptr == sorted(kv_indptr), "kv_indptr must be non-decreasing"

    for i, n in enumerate(lengths):
        pages = kv_indptr[i + 1] - kv_indptr[i]
        assert pages == -(-n // BS), f"seq {i} with {n} tokens"


def test_build_csr_round_trip_recovers_each_block_table():
    """
    The slice named by kv_indptr must be exactly that sequence's block_ids, IN
    ORDER. Order matters as much as membership: a permuted page list makes
    attention read the sequence's own KV in the wrong order, which is still
    fluent and still wrong.
    """
    lengths = [40, 5, 17, 64, 1]
    _, seqs = make_batch(lengths)
    kv_indptr, kv_indices, last_page = build_csr(seqs)

    for i, s in enumerate(seqs):
        pages = kv_indices[kv_indptr[i]:kv_indptr[i + 1]]
        assert pages == s.block_ids, f"seq {i}: CSR slice does not match its block table"
        assert last_page[i] == s.last_page_len()
        # And the addressing derived from the CSR slice must match slot_for().
        for p in range(s.num_tokens):
            assert pages[p // BS] * BS + (p % BS) == s.slot_for(p)


def test_build_csr_pages_are_disjoint_across_sequences():
    """A page appearing in two sequences' slices is R7 visible in the batch metadata."""
    _, seqs = make_batch([33, 17, 48, 9])
    _, kv_indices, _ = build_csr(seqs)
    assert len(kv_indices) == len(set(kv_indices)), "a physical page is claimed twice"


def test_build_csr_empty_batch():
    """An empty batch is a valid step — the scheduler may have nothing to run."""
    kv_indptr, kv_indices, last_page = build_csr([])
    assert kv_indptr == [0]
    assert kv_indices == []
    assert last_page == []


def test_build_csr_rejects_empty_sequence():
    """
    A zero-token sequence has no last page, and build_csr must surface that
    rather than emitting a 0 the backend would read as "drop this page".
    """
    a = make_alloc(num_blocks=8)
    s = SequenceBlocks(a)
    with pytest.raises(ValueError, match="empty sequence"):
        build_csr([s])


# ==========================================================================
# Randomised property test
# ==========================================================================


def test_randomised_operations_preserve_invariants():
    """
    Property test over random allocate / append / free schedules.

    The hand-written tests above each pin one failure mode; this one looks for
    the interleavings nobody thought to write down, since refcount and free-list
    corruption (R7) is a function of ORDER, not of any single operation. It
    asserts check_invariants() after every step and that the pool returns to
    full capacity once every sequence is freed — a leak of one block in one rare
    ordering is otherwise invisible.

    The seed is FIXED. A nondeterministic property test that fails once and
    passes on rerun teaches nothing; the operation log is printed on failure so
    any failure is reproducible by hand.
    """
    rng = random.Random(20260731)
    NUM_BLOCKS, BLOCK_SIZE = 24, 4
    a = BlockAllocator(NUM_BLOCKS, BLOCK_SIZE, watermark_blocks=2)
    live: list[SequenceBlocks] = []
    log: list[str] = []
    next_id = 0

    def fail(msg):
        raise AssertionError(msg + "\n\nOperation log:\n  " + "\n  ".join(log))

    try:
        for step in range(500):
            choice = rng.random()

            if choice < 0.30 or not live:
                s = SequenceBlocks(a, seq_id=next_id)
                n = rng.randint(1, 20)
                need = s.blocks_needed_for(n)
                if not a.can_allocate(need):
                    log.append(f"{step}: admit seq{next_id} n={n} DECLINED (need={need})")
                else:
                    s.append(n)
                    live.append(s)
                    log.append(f"{step}: admit seq{next_id} n={n} blocks={s.block_ids}")
                    next_id += 1

            elif choice < 0.75:
                s = rng.choice(live)
                n = rng.randint(1, 5)
                need = s.blocks_needed_for(n)
                if need > a.num_free:
                    log.append(f"{step}: grow seq{s.seq_id} by {n} SKIPPED (need={need})")
                else:
                    s.append(n)
                    log.append(f"{step}: grow seq{s.seq_id} by {n} -> {s.num_tokens} tok")
                    # Addressing must stay coherent as the table grows.
                    tail = s.slots_for_new_tokens(n)
                    if len(set(tail)) != len(tail):
                        fail(f"seq{s.seq_id} mapped two new tokens to the same slot")
                    if not 1 <= s.last_page_len() <= BLOCK_SIZE:
                        fail(f"seq{s.seq_id} last_page_len={s.last_page_len()} out of range")

            else:
                s = live.pop(rng.randrange(len(live)))
                log.append(f"{step}: free seq{s.seq_id} blocks={s.block_ids}")
                s.free()

            # No two live sequences may claim the same physical page.
            owned = [b for t in live for b in t.block_ids]
            if len(owned) != len(set(owned)):
                fail("two live sequences hold the same physical block (R7)")

            a.check_invariants()

        for s in live:
            log.append(f"final: free seq{s.seq_id} blocks={s.block_ids}")
            s.free()
        live.clear()

        a.check_invariants()
        if a.num_free != NUM_BLOCKS:
            fail(f"leaked {NUM_BLOCKS - a.num_free} blocks: num_free={a.num_free}")
        if a.total_allocated != a.total_freed:
            fail(f"total_allocated={a.total_allocated} != total_freed={a.total_freed}")

    except AssertionError:
        raise
    except Exception as exc:                            # noqa: BLE001 — reproducibility
        raise AssertionError(
            f"{type(exc).__name__}: {exc}\n\nOperation log:\n  " + "\n  ".join(log)
        ) from exc
