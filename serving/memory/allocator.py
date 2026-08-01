"""
Physical KV block allocator.

WHAT PROBLEM THIS SOLVES
------------------------
The engine allocates a fresh KV cache per request, sized for the worst case:
`KVCacheGPU(n_layers, max_seq=2048, ...)` built inside generate() and dropped on
return (engine/scheduler.py:16,26-27). A 30-token request therefore reserves
2048 slots x 16 layers, and nothing is ever reused, shared, or reclaimed.

Paging replaces that with a fixed pool of small fixed-size blocks handed out on
demand. A sequence holds a *list* of physical block ids — its block table — and
those blocks need not be adjacent. Memory then scales with tokens actually
generated rather than with the longest sequence anyone might send.

WHY REFERENCE COUNTS ARE HERE AND NOT IN THE RADIX CACHE
--------------------------------------------------------
It would be tempting to keep refcounting with the prefix cache (Phase 4), since
sharing is what the cache does. That would be wrong. The allocator owns physical
blocks, so the allocator must own the answer to "may this block be reused yet".
If two components can both decide a block is free, they will eventually disagree,
and the failure mode is silent: a live sequence's attention reads another
sequence's KV, output stays fluent, no metric moves (R7).

So: one owner, refcount 0 means free, and freeing a block whose refcount is
already 0 is a bug that raises rather than being tolerated.

WHY A WATERMARK
---------------
Admitting requests until zero blocks remain guarantees preemption on the very
next step, because every running sequence needs at least one more block as it
grows. The watermark reserves headroom so admission stops *before* the running
set becomes unsteppable. Preemption then handles genuine memory pressure rather
than routine growth.

Concretely: `can_allocate()` respects the watermark and is what admission calls;
`allocate()` does not, and is what an already-admitted sequence calls to grow.
A sequence that has been admitted must be able to take its next block.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

__all__ = ["BlockAllocator", "AllocationError"]


class AllocationError(RuntimeError):
    """Raised when a request cannot be satisfied. Never returns a partial result."""


class BlockAllocator:
    """
    Fixed pool of physical KV blocks with reference counting.

    Block ids are indices into the physical KV pool, `[0, num_blocks)`. This class
    knows nothing about tensors, layers, or sequences — it hands out integers.
    That separation is deliberate: it makes the allocator exhaustively testable on
    CPU with no GPU, no model, and no weights, which is most of why the hardest
    invariants here are cheap to verify.

    Not thread-safe, by design. One scheduler owns one allocator on one event
    loop; adding a lock would imply a concurrency model this system does not have
    (see docs/ARCHITECTURE.md §7).
    """

    def __init__(self, num_blocks: int, block_size: int = 16, watermark_blocks: int = 0):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if not 0 <= watermark_blocks < num_blocks:
            raise ValueError(
                f"watermark_blocks must lie in [0, {num_blocks}), got {watermark_blocks}"
            )

        self._num_blocks = num_blocks
        self._block_size = block_size
        self._watermark = watermark_blocks

        # FIFO rather than LIFO. A LIFO free list hands back the just-freed block
        # immediately, which makes use-after-free bugs *invisible* — the stale
        # reader sees data that happens to still look right. FIFO maximises the
        # delay before reuse, so such a bug surfaces as obvious garbage instead.
        self._free: deque[int] = deque(range(num_blocks))

        # refcount[i] == 0  <=>  block i is in _free. Asserted by check_invariants().
        self._refcount: list[int] = [0] * num_blocks

        # Diagnostics. Cheap, and the only way to notice the allocator is the
        # thing limiting throughput rather than the GPU.
        self.total_allocated = 0
        self.total_freed = 0
        self.peak_used = 0

    # -- introspection ------------------------------------------------------

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def watermark_blocks(self) -> int:
        return self._watermark

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self._num_blocks - len(self._free)

    @property
    def utilization(self) -> float:
        return self.num_used / self._num_blocks

    def refcount(self, block_id: int) -> int:
        self._check_id(block_id)
        return self._refcount[block_id]

    def tokens_capacity(self) -> int:
        """Total tokens of KV the pool can hold. The number S1 is a claim about."""
        return self._num_blocks * self._block_size

    # -- allocation ---------------------------------------------------------

    def can_allocate(self, n_blocks: int) -> bool:
        """
        May ADMISSION hand out n_blocks? Respects the watermark.

        Distinct from `allocate()` on purpose. Admission must leave headroom for
        the running set to take its next step; an already-running sequence must
        not be blocked by headroom it is itself the reason for.
        """
        if n_blocks < 0:
            raise ValueError(f"n_blocks must be non-negative, got {n_blocks}")
        return len(self._free) - n_blocks >= self._watermark

    def allocate(self, n_blocks: int) -> list[int]:
        """
        Take n_blocks from the free list, each with refcount 1.

        ALL OR NOTHING. A partial allocation would leave the caller holding blocks
        for a sequence it cannot actually run, and unwinding that correctly at
        every call site is exactly the kind of thing that gets skipped once.
        """
        if n_blocks < 0:
            raise ValueError(f"n_blocks must be non-negative, got {n_blocks}")
        if n_blocks == 0:
            return []
        if n_blocks > len(self._free):
            raise AllocationError(
                f"Cannot allocate {n_blocks} blocks: only {len(self._free)} free "
                f"of {self._num_blocks} (watermark {self._watermark}). "
                "The scheduler should have preempted before reaching this point."
            )

        out = [self._free.popleft() for _ in range(n_blocks)]
        for b in out:
            # A block on the free list must have refcount 0. If it does not, the
            # free list and the refcounts have diverged, and every subsequent
            # allocation is suspect.
            assert self._refcount[b] == 0, (
                f"Block {b} was on the free list with refcount {self._refcount[b]}. "
                "Allocator state is corrupt."
            )
            self._refcount[b] = 1

        self.total_allocated += n_blocks
        self.peak_used = max(self.peak_used, self.num_used)
        return out

    # -- reference counting -------------------------------------------------

    def incref(self, block_ids: Iterable[int]) -> None:
        """
        Claim an additional reference. Used when a sequence reuses cached prefix
        blocks (Phase 4) — the blocks stay live until every holder releases them.
        """
        for b in block_ids:
            self._check_id(b)
            if self._refcount[b] == 0:
                raise AllocationError(
                    f"Cannot incref block {b}: refcount is 0, so it is free and its "
                    "contents are not guaranteed. Allocate it instead."
                )
            self._refcount[b] += 1

    def free(self, block_ids: Sequence[int]) -> None:
        """
        Release one reference per id. A block returns to the free list only when
        its refcount reaches 0.

        Double-freeing RAISES. It is tempting to make this tolerant — the caller
        "obviously" meant well — but a double free means some other holder's
        block is about to be handed to a different sequence, and that corruption
        is silent (R7). Loud here beats fluent-but-wrong later.
        """
        for b in block_ids:
            self._check_id(b)
            if self._refcount[b] == 0:
                raise AllocationError(
                    f"Double free of block {b}: refcount is already 0. "
                    "Some other sequence's KV is about to be handed out."
                )
            self._refcount[b] -= 1
            if self._refcount[b] == 0:
                self._free.append(b)
                self.total_freed += 1

    # -- integrity ----------------------------------------------------------

    def check_invariants(self) -> None:
        """
        Full structural check. O(num_blocks) — cheap enough to run after every
        scheduler step in debug mode and in every test.

        Catches exactly the failure that has no other symptom: a block that is
        both referenced and on the free list, which produces cross-sequence KV
        contamination with fluent output and no error.
        """
        free_set = set(self._free)

        if len(free_set) != len(self._free):
            dupes = [b for b in self._free if list(self._free).count(b) > 1]
            raise AssertionError(f"Free list contains duplicates: {sorted(set(dupes))}")

        for b in range(self._num_blocks):
            rc, is_free = self._refcount[b], b in free_set
            if rc == 0 and not is_free:
                raise AssertionError(
                    f"Block {b} has refcount 0 but is not on the free list — leaked"
                )
            if rc > 0 and is_free:
                raise AssertionError(
                    f"Block {b} has refcount {rc} but IS on the free list — "
                    "it is about to be handed to a second owner"
                )
            if rc < 0:
                raise AssertionError(f"Block {b} has negative refcount {rc}")

    def reset(self) -> None:
        """Return every block to the pool. Tests only — never on a live scheduler."""
        self._free = deque(range(self._num_blocks))
        self._refcount = [0] * self._num_blocks

    def _check_id(self, block_id: int) -> None:
        if not 0 <= block_id < self._num_blocks:
            raise ValueError(f"Block id {block_id} out of range [0, {self._num_blocks})")

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(num_blocks={self._num_blocks}, "
            f"block_size={self._block_size}, "
            f"free={self.num_free}, used={self.num_used}, watermark={self._watermark})"
        )
