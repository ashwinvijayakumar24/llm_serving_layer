"""
Batch assembly — scheduler state -> `BatchMeta`.

This module is the ONLY place in the serving layer that knows the attention
backend's addressing conventions. Above it, the scheduler manipulates
`SequenceBlocks` objects holding plain `list[int]` page ids; below it, a backend
dereferences CSR page tables and flat slot indices. Everything the engine's
`forward_varlen()` needs about "where does this token live and what may it see"
is decided here, once per step.

That concentration is deliberate (docs/ARCHITECTURE.md §2.3.1): when FlashInfer's
real API turned out to want a CSR triple rather than a padded block-table matrix,
the cost was a change *in this file* and nowhere else. The price of that
concentration is that this file must be exhaustively correct, because every
mistake it can make is SILENT — wrong attention, fluent output, no exception, no
metric movement.

THE THREE THINGS THAT GO WRONG HERE
-----------------------------------
1. **Positions are absolute, not chunk-relative.** A sequence whose KV length is
   40 after appending 8 tokens contributes positions 32..39. Emitting 0..7
   instead is the chunked-prefill bug the engine's `BatchMeta` docstring calls
   out: harmless while prefill runs exactly once per request, catastrophic and
   invisible the moment a prompt is split into chunks. RoPE indexes `cos[
   positions]`, so wrong positions produce plausible text with a wrong attention
   geometry.

2. **`kv_last_page_len` lives in [1, page_size], never 0.** A sequence whose
   length is an exact multiple of `page_size` reports `page_size`. Reporting 0
   silently drops a whole page of keys. `SequenceBlocks.last_page_len()` already
   gets this right; `build_csr` just carries it, and `validate()` re-checks it.

3. **`slot_mapping` must agree with the block table.** For logical position `p`,
   `slot == block_ids[p // page_size] * page_size + (p % page_size)`. Physical
   pages are NOT contiguous — the allocator hands out whatever is free — so any
   code that "optimises" this into `base + p` is wrong on every fragmented pool
   and right on every trivially-allocated test. `slots_for_new_tokens()` is the
   single source of truth; this module never recomputes the arithmetic itself.

ORDERING CONTRACT
-----------------
`ScheduledSeq.blocks` must ALREADY have been grown (`blocks.append(n)`) to
include this step's tokens before assembly runs. Assembly is a pure read of
scheduler state — it allocates nothing, mutates nothing, and can therefore be
retried. `kv_lens[i]` is `blocks.num_tokens`, i.e. the length AFTER the append,
exactly as the `BatchMeta` field doc specifies.

CPU-ONLY BY CONSTRUCTION
------------------------
Every index is computed in Python `int`s and handed to `torch.tensor(...,
device=device)` once per field. This is a handful of small H2D copies per step
against a multi-millisecond forward pass, and it keeps the arithmetic testable
with no GPU. If it ever shows up in a profile the fix is pinned-memory staging
buffers, not moving the arithmetic onto the device.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from engine.attention_backend import BatchMeta

from serving.memory.block_table import SequenceBlocks, build_csr

__all__ = ["ScheduledSeq", "build_batch_meta", "build_token_tensor"]


# `positions` is int64 because it indexes the RoPE tables directly
# (`cos[positions]`), and torch requires int64 for index tensors. Every other
# metadata tensor is int32 by the BatchMeta contract — they are consumed by
# kernels, and FlashInfer's CSR arguments are int32.
POSITIONS_DTYPE = torch.int64


@dataclass
class ScheduledSeq:
    """
    One sequence's contribution to one forward pass.

    blocks
        The sequence's block table, ALREADY grown to include this step's tokens.
        `blocks.num_tokens` is therefore the post-append KV length, which is what
        `BatchMeta.kv_lens` wants. Assembly never calls `append()` itself: the
        scheduler must decide admission and preemption before it knows a batch is
        runnable, and an assembler that allocates could fail halfway through
        building a batch it has no way to unwind.

    new_token_ids
        The token ids this sequence contributes THIS step, in order. Length is
        the sequence's `query_len`: 1 for a decode, many for a prefill chunk.
        These are packed into the flat token axis by `build_token_tensor` in the
        same sequence order used for every other field.

    wants_logits
        Whether this step needs next-token logits for this sequence. False for a
        non-final prefill chunk, where there is nothing to sample yet. Carried
        because `last_token_ix` is always emitted (the field is not optional and
        the gather is cheap), so the flag is the only place that distinction can
        live; a sampler reads it to decide which rows of the logits matrix are
        real. It does not affect any tensor this module builds.
    """

    blocks: SequenceBlocks
    new_token_ids: list[int] = field(default_factory=list)
    wants_logits: bool = True

    # -- derived, and the two definitions everything else is built from --------

    @property
    def query_len(self) -> int:
        """New tokens contributed this step."""
        return len(self.new_token_ids)

    @property
    def kv_len(self) -> int:
        """Total KV length AFTER this step's append."""
        return self.blocks.num_tokens

    @property
    def start_position(self) -> int:
        """
        Absolute position of this step's FIRST token.

        `kv_len - query_len`. This one subtraction is the entire chunked-prefill
        fix: the sequence's history is `[0, kv_len - query_len)` and this step
        appends on top of it, so the first new token sits at index
        `kv_len - query_len`, not at 0.
        """
        return self.kv_len - self.query_len

    def positions(self) -> list[int]:
        """ABSOLUTE RoPE positions of this step's tokens: `K-q .. K-1`."""
        return list(range(self.start_position, self.kv_len))


def _check_seq(i: int, s: ScheduledSeq) -> None:
    """
    Reject inputs whose downstream failure would be silent.

    These fire before any tensor is built, so the error names the offending
    sequence rather than surfacing as a shape mismatch fifteen frames deeper.
    """
    if s.blocks.is_freed:
        raise ValueError(
            f"seq {i} (seq_id={s.blocks.seq_id}) has been freed; its blocks may already "
            "belong to another sequence. Scheduling it would read someone else's KV."
        )
    q = s.query_len
    k = s.kv_len
    if q <= 0:
        raise ValueError(
            f"seq {i} (seq_id={s.blocks.seq_id}) contributes {q} tokens. A sequence that "
            "contributes nothing must not appear in the batch: it would occupy a "
            "cu_query_lens slot with an empty token range and make last_token_ix point "
            "at another sequence's token."
        )
    if q > k:
        raise ValueError(
            f"seq {i} (seq_id={s.blocks.seq_id}) contributes {q} tokens but its block "
            f"table holds only {k}. blocks.append() must run BEFORE assembly — kv_lens "
            "is the length AFTER the append."
        )
    if k > s.blocks.capacity:
        raise ValueError(
            f"seq {i} (seq_id={s.blocks.seq_id}) claims kv_len={k} but its "
            f"{s.blocks.num_blocks} blocks hold only {s.blocks.capacity} tokens."
        )


def build_token_tensor(seqs: list[ScheduledSeq], device) -> torch.Tensor:
    """
    Pack every sequence's new token ids along ONE axis, in sequence order.

    Returns `(tokens,) int64` — int64 because this tensor is used as an index
    into the embedding table (`w[...][ids]`), and torch requires int64 there.

    The packing order is the contract: sequence `i` owns
    `[cu_query_lens[i], cu_query_lens[i+1])` of this tensor and of every other
    per-token field (`positions`, `batch_indices`, `slot_mapping`). Build them
    from the same `seqs` list, in the same order, or the batch is quietly wrong.
    """
    flat: list[int] = []
    for i, s in enumerate(seqs):
        _check_seq(i, s)
        flat.extend(s.new_token_ids)
    return torch.tensor(flat, dtype=torch.int64, device=device)


def build_batch_meta(
    seqs: list[ScheduledSeq],
    device,
    page_size: int,
    dtype_index: torch.dtype = torch.int32,
    validate: bool = True,
) -> BatchMeta:
    """
    Turn scheduler state into the `BatchMeta` the engine's `forward_varlen()`
    consumes.

    Parameters
    ----------
    seqs
        Sequences in batch order. This order defines the token axis, the CSR row
        order, and the row order of the returned logits. Must be non-empty (see
        below).
    device
        Where the metadata tensors are built. Must be the model's device: the
        engine indexes `cos[positions]` with no implicit transfer.
    page_size
        Tokens per physical page. Must equal every sequence's `blocks.block_size`
        — a mismatch would make `slot_mapping` and the CSR disagree about page
        geometry, which is exactly the silent class of bug this layer exists to
        prevent, so it is checked rather than assumed.
    dtype_index
        Dtype for the index/metadata tensors. int32 per the contract; exposed
        only so a backend that genuinely needs int64 CSR can ask for it.
        `positions` is ALWAYS int64 regardless — it indexes the RoPE tables.
    validate
        Run `meta.validate()` before returning. Default ON. Every invariant it
        checks corresponds to a failure with no symptom, and the cost is a few
        dozen scalar reads against a forward pass measured in milliseconds. Turn
        it off only with a profile that says it matters.

    Raises
    ------
    ValueError
        On an empty batch, on a sequence contributing zero tokens, on a sequence
        whose block table has not been grown yet, or on a page_size mismatch.

        EMPTY BATCH IS AN ERROR, NOT AN EMPTY BATCH. `BatchMeta` with
        `n_seqs == 0` would pass most of `validate()` (the per-sequence loop runs
        zero times) but `kv_last_page_len.min()` on an empty tensor raises deep
        inside torch, and a zero-token forward pass is meaningless work. The
        scheduler already knows whether it has anything to run; making it say so
        here is cheaper than making every backend defend against a null step.
    """
    if not seqs:
        raise ValueError(
            "build_batch_meta() called with an empty sequence list. A step with no "
            "sequences is a scheduler bug, not a valid batch — skip the forward pass."
        )
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    query_lens: list[int] = []
    kv_lens: list[int] = []
    positions: list[int] = []
    batch_indices: list[int] = []
    slot_mapping: list[int] = []
    cu_query_lens: list[int] = [0]  # exclusive prefix sum; always starts at 0
    last_token_ix: list[int] = []

    for i, s in enumerate(seqs):
        _check_seq(i, s)
        if s.blocks.block_size != page_size:
            raise ValueError(
                f"seq {i} (seq_id={s.blocks.seq_id}) has block_size="
                f"{s.blocks.block_size} but the batch declares page_size={page_size}. "
                "slot_mapping and the CSR page table would describe different geometries."
            )

        q = s.query_len
        k = s.kv_len

        query_lens.append(q)
        kv_lens.append(k)

        # ABSOLUTE positions: K-q .. K-1. See the module docstring, point 1.
        # A chunk starting at absolute position 32 carries 32.., never 0...
        positions.extend(s.positions())

        # Which sequence owns each token. Derivable from cu_query_lens, but
        # FlashInfer's append API consumes it directly, so materialise it once
        # per step rather than per layer.
        batch_indices.extend([i] * q)

        # Physical slot per new token. Delegated to SequenceBlocks so the
        # page-indirection arithmetic exists in exactly one place and works on a
        # fragmented pool where consecutive positions land on distant pages.
        slot_mapping.extend(s.blocks.slots_for_new_tokens(q))

        # Exclusive prefix sum: sequence i owns tokens [cu[i], cu[i+1]).
        cu_query_lens.append(cu_query_lens[-1] + q)

        # The final token of this sequence on the packed axis — where logits are
        # wanted. In a ragged batch "the last token" is a gather, not a slice.
        # It is the last index of this sequence's half-open range, hence -1.
        last_token_ix.append(cu_query_lens[-1] - 1)

    # CSR page table. build_csr concatenates each sequence's block_ids and takes
    # the prefix sum, and carries last_page_len (which is in [1, page_size] by
    # SequenceBlocks' own invariant — an exact multiple reports page_size, not 0).
    kv_indptr, kv_indices, kv_last_page_len = build_csr([s.blocks for s in seqs])

    def _idx(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype_index, device=device)

    meta = BatchMeta(
        query_lens=_idx(query_lens),
        cu_query_lens=_idx(cu_query_lens),
        kv_lens=_idx(kv_lens),
        positions=torch.tensor(positions, dtype=POSITIONS_DTYPE, device=device),
        last_token_ix=_idx(last_token_ix),
        kv_indptr=_idx(kv_indptr),
        kv_indices=_idx(kv_indices),
        kv_last_page_len=_idx(kv_last_page_len),
        batch_indices=_idx(batch_indices),
        slot_mapping=_idx(slot_mapping),
        page_size=page_size,
        # A hint, never a correctness input: True when ANY sequence contributes
        # more than one token, so a mixed prefill+decode batch is prefill-shaped.
        is_prefill=any(q > 1 for q in query_lens),
    )

    if validate:
        meta.validate()
    return meta
