"""
Per-sequence block table — the logical-to-physical mapping for one sequence.

A sequence's KV lives in scattered physical blocks. Its block table records
which, in order:

    logical position p  ->  physical slot
                            block_ids[p // block_size] * block_size + (p % block_size)

That indirection is the whole idea of paged attention. Everything above it — the
scheduler, the radix cache, the router — manipulates block tables; only the
attention backend ever dereferences one.

THE INVARIANT THAT BREAKS THINGS SILENTLY
-----------------------------------------
`last_page_len` must lie in [1, block_size], and a sequence whose length is an
exact multiple of block_size reports block_size, NOT 0.

FlashInfer documents this explicitly (flashinfer-python 0.6.16), and it is the
kind of off-by-one that never raises: report 0 and attention silently drops a
whole page of keys; report block_size + 1 and it reads a page of uninitialised
memory. Output stays fluent either way. `num_tokens=32, block_size=16` is the
smallest case that exercises it, which is why the tests sweep exact multiples.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serving.memory.allocator import BlockAllocator

__all__ = ["SequenceBlocks"]


class SequenceBlocks:
    """
    Block table for one sequence, plus the arithmetic that maps positions to
    physical slots.

    Owns references, not blocks. Every block in `block_ids` carries a reference
    held by this object, released on `free()`. A sequence that is dropped without
    `free()` leaks its blocks — which the allocator's leak test is designed to
    catch, since a leak is otherwise only visible as gradually falling capacity.
    """

    def __init__(self, allocator: BlockAllocator, seq_id: int = 0):
        self._alloc = allocator
        self.seq_id = seq_id
        self.block_size = allocator.block_size
        self.block_ids: list[int] = []
        self.num_tokens = 0
        self._freed = False

    # -- capacity -----------------------------------------------------------

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    @property
    def capacity(self) -> int:
        """Tokens this sequence can hold before another block is needed."""
        return len(self.block_ids) * self.block_size

    def blocks_needed_for(self, n_new_tokens: int) -> int:
        """
        How many NEW blocks appending n_new_tokens would require.

        Admission calls this before committing, so it can decline a request it
        cannot house rather than discovering that mid-forward-pass.
        """
        if n_new_tokens < 0:
            raise ValueError(f"n_new_tokens must be non-negative, got {n_new_tokens}")
        total = self.num_tokens + n_new_tokens
        return max(0, self._blocks_for(total) - len(self.block_ids))

    def append(self, n_new_tokens: int) -> None:
        """
        Grow the sequence by n_new_tokens, allocating blocks as needed.

        Uses `allocate()`, not `can_allocate()`: this sequence is already running,
        and the watermark exists to stop *admission*, not to starve a sequence
        that was admitted. If this raises AllocationError the scheduler failed to
        preempt in time — a real bug, and one worth surfacing loudly.
        """
        self._assert_live()
        if n_new_tokens < 0:
            raise ValueError(f"n_new_tokens must be non-negative, got {n_new_tokens}")
        if n_new_tokens == 0:
            return
        need = self.blocks_needed_for(n_new_tokens)
        if need:
            self.block_ids.extend(self._alloc.allocate(need))
        self.num_tokens += n_new_tokens

    # -- addressing ---------------------------------------------------------

    def slot_for(self, position: int) -> int:
        """Physical slot holding logical `position`. The core indirection."""
        self._assert_live()
        if not 0 <= position < self.num_tokens:
            raise IndexError(
                f"position {position} outside sequence of length {self.num_tokens}"
            )
        return self.block_ids[position // self.block_size] * self.block_size + (
            position % self.block_size
        )

    def slots_for_range(self, start: int, count: int) -> list[int]:
        """Slots for `count` consecutive positions from `start`."""
        return [self.slot_for(p) for p in range(start, start + count)]

    def slots_for_new_tokens(self, n_new_tokens: int) -> list[int]:
        """
        Slots for the most recently appended n_new_tokens.

        Called AFTER append(), when building BatchMeta.slot_mapping — the write
        addressing for this step's K/V.
        """
        return self.slots_for_range(self.num_tokens - n_new_tokens, n_new_tokens)

    def last_page_len(self) -> int:
        """
        Occupancy of the final block, in [1, block_size].

        An exact multiple of block_size returns block_size, NOT 0. See the module
        docstring — this is the silent off-by-one.
        """
        self._assert_live()
        if self.num_tokens == 0:
            raise ValueError(
                "last_page_len is undefined for an empty sequence: there is no last "
                "page. A sequence with no tokens should not appear in a batch."
            )
        rem = self.num_tokens % self.block_size
        return rem if rem else self.block_size

    # -- lifecycle ----------------------------------------------------------

    def free(self) -> None:
        """
        Release every block reference. Idempotent — a scheduler may retire a
        sequence on a path that also runs during cancellation, and making the
        second call an error would trade a leak for a crash.
        """
        if self._freed:
            return
        if self.block_ids:
            self._alloc.free(self.block_ids)
        self.block_ids = []
        self.num_tokens = 0
        self._freed = True

    @property
    def is_freed(self) -> bool:
        return self._freed

    # -- internals ----------------------------------------------------------

    def _blocks_for(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def _assert_live(self) -> None:
        if self._freed:
            raise RuntimeError(
                f"SequenceBlocks(seq_id={self.seq_id}) was freed; its blocks may now "
                "belong to another sequence. Using it would read someone else's KV."
            )

    def __len__(self) -> int:
        return self.num_tokens

    def __repr__(self) -> str:
        state = "freed" if self._freed else f"{self.num_tokens} tokens"
        return (
            f"SequenceBlocks(seq_id={self.seq_id}, {state}, "
            f"blocks={self.block_ids}, last_page_len="
            f"{self.last_page_len() if self.num_tokens and not self._freed else '-'})"
        )


def build_csr(seqs: Sequence[SequenceBlocks]) -> tuple[list[int], list[int], list[int]]:
    """
    Flatten block tables into the CSR triple the attention backends consume.

        sequence i owns pages  kv_indices[kv_indptr[i] : kv_indptr[i+1]]

    Returns (kv_indptr, kv_indices, kv_last_page_len).

    CSR rather than a padded (n_seqs, max_blocks) matrix because that is what
    FlashInfer expects (verified against flashinfer-python 0.6.16 —
    `append_paged_kv_cache(..., kv_indices, kv_indptr, kv_last_page_len)`), and
    because it needs no padding and no sentinel value. The padded form is vLLM's
    convention.

    This is the ONLY place that has to know the backend's addressing convention;
    the allocator and block tables above it stay in plain Python lists.
    """
    kv_indptr = [0]
    kv_indices: list[int] = []
    last_page: list[int] = []
    for s in seqs:
        kv_indices.extend(s.block_ids)
        kv_indptr.append(len(kv_indices))
        last_page.append(s.last_page_len())
    return kv_indptr, kv_indices, last_page
