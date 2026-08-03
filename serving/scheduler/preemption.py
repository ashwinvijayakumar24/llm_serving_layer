"""
Preemption policies — what to do when the KV pool cannot fund the next step.

WHY PREEMPTION EXISTS AT ALL
----------------------------
Admission is watermark-gated (`BlockAllocator.can_allocate`), so in the common
case the running set is always steppable. But the watermark is a *static*
reservation and sequence growth is dynamic: a batch of 32 sequences that each
happen to cross a page boundary on the same step wants 32 new blocks at once.
Sizing the watermark for that worst case would mean admitting almost nothing.

So the design is: admit optimistically against a modest watermark, and keep a
mechanism that can *always* make room. That mechanism is preemption. Without it,
the honest behaviour above the knee is an `AllocationError` mid-forward-pass —
i.e. the server falls over instead of degrading (the phase plan §6, property 4).

THE TWO POLICIES (ARCHITECTURE §5.2)
------------------------------------
**RECOMPUTE** — free the victim's blocks; requeue it at the FRONT of the waiting
queue with `prompt + tokens-generated-so-far` as its new prompt. Frees 100% of
its blocks immediately, needs no host memory and no PCIe traffic, and pays
O(current length) of prefill compute *later*.

**SWAP** — copy the victim's KV blocks GPU->pinned host, free the GPU blocks,
and copy back into freshly allocated blocks on resume. Costs 2x PCIe transfer of
the victim's KV and consumes pinned host memory; recomputes nothing.

Which wins is not universal. It is the ratio of recompute cost to transfer cost,
which depends on model size, sequence length and PCIe bandwidth. For Llama 3.2
1B (32 KB of KV per token, 16 layers, tiny prefill) recompute is expected to win
at nearly all lengths — which is exactly why both are implemented and measured
rather than asserted.

WHY BIT-IDENTITY IS THE ONLY ACCEPTABLE GATE (R3)
-------------------------------------------------
Both policies are *supposed* to be invisible: the token stream a client receives
must not depend on whether its request was preempted. A bug here does not raise,
does not NaN, and does not move a metric — it produces fluent, wrong text, under
load, rarely. The only detector is exact token equality against an unpreempted
run, per request, for each policy independently (`tests/test_preemption_gpu.py`).

The two policies fail in different directions, which is why neither may be
tested only through the other:

  * RECOMPUTE's failure is an off-by-one in *what* is re-prefilled. Re-prefill
    `prompt + output_ids` (all of it): the sequence's KV then covers exactly the
    tokens it covered before, and the logits produced by that final prefill
    chunk are the logits for the *next* token — the same one the unpreempted run
    would have sampled. Re-prefilling `output_ids[:-1]` instead duplicates a
    token; re-prefilling and then *also* emitting the sampled token as if it
    were new duplicates it too. Both produce fluent output.
  * SWAP's failure is an incomplete or misordered copy — one layer missed, K
    copied but not V, or blocks restored in a different order than the block
    table records. Attention then reads a plausible-looking mixture of two
    sequences' history.

STARVATION GUARD: A PREFERENCE, NOT AN EXCLUSION (R40)
------------------------------------------------------
`select_victim` deprioritises sequences already preempted K times. It does NOT
make them ineligible. The exclusion version deadlocks: when every running
sequence has reached K, victim selection returns nothing while the batch is
non-empty, and the scheduler can neither step nor make room — throughput goes to
zero and every request hangs.

The invariant, and it is the one worth stating plainly:

    victim selection must NEVER return "no victim" while the batch is non-empty.

Falling back to "preempt the newest anyway" is forward progress for someone
instead of deadlock for everyone. But a fallback firing means the watermark let
the system reach a state it should not have been able to reach, so it is counted
and surfaced as an ADMISSION-CONTROL ALARM rather than silently absorbed.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

import torch

if TYPE_CHECKING:
    from serving.memory.block_table import SequenceBlocks

__all__ = [
    "PreemptionPolicy",
    "PreemptionStats",
    "SwapHandle",
    "KVSwapSpace",
    "SwapSpaceFull",
    "select_victim",
]


class PreemptionPolicy(StrEnum):
    """
    The policy flag from ARCHITECTURE §5.2. A `StrEnum` so a config can carry
    the literal string `"swap"` and still compare equal to the enum member —
    useful when the policy arrives from a CLI flag or a benchmark matrix.
    """

    RECOMPUTE = "recompute"
    SWAP = "swap"


class Preemptible(Protocol):
    """
    The only thing victim selection needs to know about a request.

    Structural rather than an import of `Request`: `select_victim` is a pure
    ordering function over two integers, and keeping it that way means it can be
    tested with plain objects and cannot accidentally acquire a dependency on
    scheduler state.
    """

    arrival_seq: int
    preemption_count: int

    def is_terminal(self) -> bool: ...


# ---------------------------------------------------------------------------
# victim selection
# ---------------------------------------------------------------------------


def select_victim(
    running: Sequence[Preemptible],
    starvation_k: int,
    stats: PreemptionStats | None = None,
) -> Any | None:
    """
    Choose which sequence loses its memory. LIFO, with the starvation guard as a
    preference ordering.

    WHY LIFO AND NOT FIFO
    ---------------------
    Preempting the *newest* request preserves the progress of older ones and
    bounds worst-case latency for requests already deep into generation: a
    sequence that has generated 400 tokens has 400 tokens of work at risk, a
    sequence admitted this step has almost none. FIFO preemption repeatedly
    punishes the oldest request — it is the one holding the most state, so it is
    the most expensive to evict and the one whose latency is already worst.
    That is both unfair and unbounded: under sustained pressure the oldest
    request can be evicted every time it is readmitted and never finish.

    LIFO also matches the shape of the pressure. Memory pressure is usually
    caused by the most recent admissions, so evicting them undoes the decision
    that created the problem.

    THE GUARD
    ---------
    1. Prefer victims with `preemption_count < starvation_k`, newest first.
    2. If every running sequence is at or above K, preempt the newest anyway.
    3. Count that fallback. It is an admission-control alarm (R40).

    Returns None ONLY when there is no non-terminal running sequence at all.
    Any other None would be the deadlock this function exists to avoid.
    """
    if starvation_k < 0:
        raise ValueError(f"starvation_k must be non-negative, got {starvation_k}")

    # Terminal requests are about to be retired and their blocks freed anyway;
    # preempting one would do work to free memory that is already leaving.
    candidates = [r for r in running if not r.is_terminal()]
    if not candidates:
        return None

    # Newest first. The enumerate index is a tie-break so the order is total and
    # deterministic even if two requests share an arrival_seq (they should not —
    # the scheduler stamps a monotonic counter — but a selection function whose
    # answer depends on sort stability is a bug waiting for a different Python).
    newest_first = [
        r
        for _, r in sorted(
            enumerate(candidates),
            key=lambda t: (t[1].arrival_seq, t[0]),
            reverse=True,
        )
    ]

    for req in newest_first:
        if req.preemption_count < starvation_k:
            return req

    # Every candidate has hit K. The exclusion version of this guard returns
    # None here and the scheduler deadlocks. Instead: preempt the newest, and
    # say loudly that admission control let this happen.
    if stats is not None:
        stats.starvation_fallbacks += 1
    return newest_first[0]


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------


@dataclass
class PreemptionStats:
    """
    Everything Phase 3 claims, counted.

    Deliberately includes the numbers that make the recompute-vs-swap comparison
    possible (`tokens_recomputed` vs `bytes_swapped_*`) and the one number that
    is a bug report rather than a measurement (`starvation_fallbacks`).
    """

    total: int = 0
    by_policy: dict[str, int] = field(
        default_factory=lambda: {p.value: 0 for p in PreemptionPolicy}
    )

    # RECOMPUTE cost: KV tokens thrown away and later recomputed as prefill.
    tokens_recomputed: int = 0

    # SWAP cost: PCIe traffic, both directions, and the wall time it took.
    bytes_swapped_out: int = 0
    bytes_swapped_in: int = 0
    swap_out_seconds: float = 0.0

    # Resume latency, two ways: wall-clock seconds for the resume operation
    # itself (the copy-back, which is what swap pays) and scheduler STEPS spent
    # preempted (the stall the client actually perceives, which is what both
    # policies pay and what recompute pays most of).
    resumes: int = 0
    resume_seconds_total: float = 0.0
    resume_seconds_max: float = 0.0
    resume_steps_total: int = 0
    resume_steps_max: int = 0

    # R40 alarm. Non-zero means the watermark is wrong, not that the guard works.
    starvation_fallbacks: int = 0

    # Host swap space ran out and SWAP degraded to RECOMPUTE. Not an error — the
    # fallback is correct — but it silently changes which policy is being
    # measured, which would corrupt a head-to-head benchmark if uncounted.
    swap_space_exhausted: int = 0

    def record(self, policy: PreemptionPolicy) -> None:
        self.total += 1
        self.by_policy[str(policy)] = self.by_policy.get(str(policy), 0) + 1

    def record_resume(self, seconds: float, steps: int) -> None:
        self.resumes += 1
        self.resume_seconds_total += seconds
        self.resume_seconds_max = max(self.resume_seconds_max, seconds)
        self.resume_steps_total += steps
        self.resume_steps_max = max(self.resume_steps_max, steps)

    @property
    def admission_control_alarm(self) -> bool:
        """
        True iff the starvation fallback has fired.

        Exposed as a named property rather than left as a counter to read,
        because the whole point of R40's mitigation is that this must be
        *surfaced* — a fallback absorbed silently is indistinguishable from a
        healthy system right up until requests start hanging.
        """
        return self.starvation_fallbacks > 0

    @property
    def mean_resume_seconds(self) -> float:
        return self.resume_seconds_total / self.resumes if self.resumes else 0.0

    @property
    def mean_resume_steps(self) -> float:
        return self.resume_steps_total / self.resumes if self.resumes else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "preemptions_total": self.total,
            "preemptions_by_policy": dict(self.by_policy),
            "tokens_recomputed": self.tokens_recomputed,
            "bytes_swapped_out": self.bytes_swapped_out,
            "bytes_swapped_in": self.bytes_swapped_in,
            "swap_out_seconds": round(self.swap_out_seconds, 6),
            "resumes": self.resumes,
            "resume_seconds_total": round(self.resume_seconds_total, 6),
            "resume_seconds_mean": round(self.mean_resume_seconds, 6),
            "resume_seconds_max": round(self.resume_seconds_max, 6),
            "resume_steps_mean": round(self.mean_resume_steps, 3),
            "resume_steps_max": self.resume_steps_max,
            "starvation_fallbacks": self.starvation_fallbacks,
            "admission_control_alarm": self.admission_control_alarm,
            "swap_space_exhausted": self.swap_space_exhausted,
        }


# ---------------------------------------------------------------------------
# swap space
# ---------------------------------------------------------------------------


class SwapSpaceFull(RuntimeError):
    """Host swap budget exhausted. Callers degrade to RECOMPUTE rather than fail."""


@dataclass
class SwapHandle:
    """
    One sequence's KV, parked on the host.

    `k`/`v` are `(num_layers, num_blocks, block_size, n_kv_heads, head_dim)` —
    the GPU pool's own layout with the layer axis stacked on the front, so
    restoring is an index-put per layer and nothing has to be transposed or
    reinterpreted. A layout conversion in the swap path would be a second place
    that has to agree with `PagedTorchBackend`'s pool geometry, and disagreement
    there is silent.

    The block IDS are deliberately NOT stored. The victim's physical blocks are
    freed the moment the copy completes and will be handed to other sequences;
    resuming into them would be a use-after-free. What is stored is the LOGICAL
    order — index `i` of the host tensor is the i-th block of the sequence — and
    resume allocates a fresh set and fills it in that same order.
    """

    request_id: str
    seq_id: int
    num_tokens: int
    num_blocks: int
    k: torch.Tensor
    v: torch.Tensor
    nbytes: int
    block_size: int
    n_kv_heads: int
    head_dim: int
    released: bool = False


class KVSwapSpace:
    """
    GPU <-> pinned-host movement for preempted sequences' KV.

    Reads the backend's `k_pool`/`v_pool` directly. That is a deliberate,
    read-only coupling: the pools ARE the physical memory, and routing the copy
    through an abstraction would mean the swap path and the attention path could
    disagree about layout — exactly the silent class of bug this project spends
    its test budget on. The backend is not modified.

    PINNED HOST MEMORY, AND WHY IT IS CONDITIONAL
    ---------------------------------------------
    Pinned (page-locked) host memory is what makes a D2H copy DMA-able and
    asynchronous-capable; pageable memory forces the driver to stage through an
    internal pinned buffer, roughly halving effective bandwidth. It is also a
    scarce, process-wide resource, hence `max_bytes`. On a CPU-only box
    `pin_memory=True` is meaningless (and raises without CUDA), so it is enabled
    only when the pool actually lives on a CUDA device. The CPU tests therefore
    exercise the same code path with the same tensors, just without the pinning.

    The copies are SYNCHRONOUS on purpose. An async D2H into pinned memory
    followed by an immediate `free()` of the source blocks would let the
    allocator hand those blocks to another sequence while the copy is still in
    flight — a race whose symptom is, once again, fluent wrong text. Overlapping
    the copy with compute is a real optimisation and it needs an event to gate
    the free on; that is not free to get right and is not what Phase 3 claims.
    """

    def __init__(self, backend, max_bytes: int | None = None):
        for attr in ("k_pool", "v_pool", "num_layers", "block_size", "n_kv_heads",
                     "head_dim"):
            if not hasattr(backend, attr):
                raise TypeError(
                    f"KVSwapSpace needs a paged backend exposing .{attr}; got "
                    f"{type(backend).__name__}. The SWAP policy moves physical KV, "
                    "so it cannot run against a backend that does not own a pool."
                )
        self.backend = backend
        self.max_bytes = max_bytes
        self.bytes_in_use = 0
        self.peak_bytes = 0
        self.last_swap_out_seconds = 0.0
        self._live: dict[int, SwapHandle] = {}

        pool = backend.k_pool[0]
        self.device = pool.device
        self.dtype = pool.dtype
        # Pinning is a CUDA concept. Asking for it without CUDA raises.
        self.pinned = self.device.type == "cuda" and torch.cuda.is_available()

    # -- accounting ---------------------------------------------------------

    def bytes_for(self, num_blocks: int) -> int:
        """Host bytes a `num_blocks`-long sequence would occupy. K and V, all layers."""
        pool = self.backend.k_pool[0]
        per_block = pool[0].numel() * pool.element_size()
        return 2 * self.backend.num_layers * num_blocks * per_block

    @property
    def num_swapped(self) -> int:
        return len(self._live)

    # -- movement -----------------------------------------------------------

    def swap_out(self, request_id: str, blocks: SequenceBlocks) -> SwapHandle:
        """
        Copy every block of `blocks` to the host. Does NOT free them — the caller
        frees, after this returns, so a failure here leaves the sequence intact
        and resumable rather than half-evicted.

        Raises `SwapSpaceFull` if the host budget cannot fund it.
        """
        if blocks.is_freed:
            raise RuntimeError(
                f"swap_out on freed SequenceBlocks(seq_id={blocks.seq_id}); its blocks "
                "may already belong to another sequence and the copy would capture "
                "someone else's KV."
            )

        nb = len(blocks.block_ids)
        want = self.bytes_for(nb)
        if self.max_bytes is not None and self.bytes_in_use + want > self.max_bytes:
            raise SwapSpaceFull(
                f"swapping {nb} blocks needs {want} B; {self.bytes_in_use} B of "
                f"{self.max_bytes} B host swap space already in use"
            )

        t0 = time.perf_counter()
        k_host, v_host = self._to_host(blocks.block_ids)
        elapsed = time.perf_counter() - t0

        handle = SwapHandle(
            request_id=request_id,
            seq_id=blocks.seq_id,
            num_tokens=blocks.num_tokens,
            num_blocks=nb,
            k=k_host,
            v=v_host,
            nbytes=want,
            block_size=self.backend.block_size,
            n_kv_heads=self.backend.n_kv_heads,
            head_dim=self.backend.head_dim,
        )
        self._live[id(handle)] = handle
        self.bytes_in_use += want
        self.peak_bytes = max(self.peak_bytes, self.bytes_in_use)
        self.last_swap_out_seconds = elapsed
        return handle

    def swap_in(self, handle: SwapHandle, blocks: SequenceBlocks) -> None:
        """
        Restore a handle into freshly allocated blocks.

        `blocks` must already have been grown to `handle.num_tokens` — the block
        COUNT must match exactly, because host index `i` is written to
        `blocks.block_ids[i]` and a mismatch would silently shift the sequence's
        history by a page. Checked, not assumed.
        """
        if handle.released:
            raise RuntimeError(f"swap_in on a released handle for {handle.request_id!r}")
        if blocks.num_tokens != handle.num_tokens:
            raise RuntimeError(
                f"resume mismatch for {handle.request_id!r}: block table holds "
                f"{blocks.num_tokens} tokens, handle holds {handle.num_tokens}"
            )
        if len(blocks.block_ids) != handle.num_blocks:
            raise RuntimeError(
                f"resume mismatch for {handle.request_id!r}: {len(blocks.block_ids)} "
                f"blocks allocated, {handle.num_blocks} swapped out. Logical block "
                "order would shift and attention would read a different history."
            )
        if handle.block_size != self.backend.block_size:
            raise RuntimeError(
                f"handle block_size={handle.block_size} != backend "
                f"{self.backend.block_size}"
            )

        self._from_host(handle, blocks.block_ids)

    def release(self, handle: SwapHandle) -> None:
        """Drop a handle's host memory. Idempotent — retirement paths overlap."""
        if handle.released:
            return
        handle.released = True
        self._live.pop(id(handle), None)
        self.bytes_in_use -= handle.nbytes
        # Drop the references so the (possibly pinned) host allocation can go.
        handle.k = torch.empty(0)
        handle.v = torch.empty(0)

    # -- the copies ---------------------------------------------------------

    def _index(self, block_ids: list[int]) -> torch.Tensor:
        return torch.as_tensor(block_ids, dtype=torch.long, device=self.device)

    def _to_host(self, block_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        nb = len(block_ids)
        pool = self.backend.k_pool[0]
        shape = (self.backend.num_layers, nb, *pool.shape[1:])
        k_host = torch.empty(shape, dtype=self.dtype, pin_memory=self.pinned)
        v_host = torch.empty(shape, dtype=self.dtype, pin_memory=self.pinned)
        if nb == 0:
            return k_host, v_host

        idx = self._index(block_ids)
        for layer in range(self.backend.num_layers):
            # EVERY layer. A loop that stops at layer 0 restores a sequence whose
            # first layer is right and whose rest is another sequence's history —
            # fluent output, no error. The GPU gate's multi-layer model is what
            # makes that detectable.
            k_host[layer].copy_(self.backend.k_pool[layer][idx])
            v_host[layer].copy_(self.backend.v_pool[layer][idx])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return k_host, v_host

    def _from_host(self, handle: SwapHandle, block_ids: list[int]) -> None:
        if handle.num_blocks == 0:
            return
        idx = self._index(block_ids)
        for layer in range(self.backend.num_layers):
            self.backend.k_pool[layer][idx] = handle.k[layer].to(
                self.device, non_blocking=False
            )
            self.backend.v_pool[layer][idx] = handle.v[layer].to(
                self.device, non_blocking=False
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
