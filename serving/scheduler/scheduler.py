"""
Iteration-level scheduler — continuous batching.

WHAT CONTINUOUS BATCHING ACTUALLY IS
------------------------------------
Static batching collects N requests, runs them together until *all* of them
finish, then starts the next batch. A sequence that produces 20 tokens sits in
the batch doing nothing while a sequence producing 400 tokens finishes. Its
slot is occupied and useless. With a long-tailed output-length distribution —
which real traffic has — most of the batch is idle most of the time.

Continuous batching (Orca, OSDI 2022) makes the batch composition a per-step
decision instead of a per-batch one. After every forward pass: retire whatever
finished, admit whatever fits. A finished sequence's slot is reused on the very
next step rather than at the end of the batch.

The unit of scheduling is one decode step, not one request. Everything else
here follows from that.

WHY THIS IS SEPARATE FROM THE ENGINE
------------------------------------
`engine/scheduler.py` is named "scheduler" but is a single-request loop
(`generate()`, :11-45): prefill, then decode until EOS, one sequence, no queue,
no admission, no memory pressure. That is the correct primitive for a reference
implementation and the wrong shape for a server. This file is what "scheduler"
means once more than one request exists.

PREEMPTION (PHASE 3)
--------------------
Admission is watermark-gated, but the watermark is a static reservation against
dynamic growth: it cannot be sized for "every running sequence crosses a page
boundary on the same step" without admitting almost nothing. So the scheduler
admits optimistically and keeps a mechanism that can always make room.

The trigger is a PRE-STEP check (`_preempt_if_needed`): compute the blocks this
step would need across the running batch, compare against free blocks, and if it
does not fit, DO NOT enter the forward pass — preempt first. Discovering the
shortfall inside `blocks.append()` mid-assembly would mean unwinding a
half-built batch, which is exactly the code path nobody gets right.

Policy (recompute vs swap), victim selection (LIFO) and the starvation guard
live in `preemption.py`; this file owns the trigger, the request-state
transitions, and resume. The correctness argument is written out at
`_preempt_recompute` — it is the one place an off-by-one produces fluent, wrong
text (R3).

The radix prefix cache is Phase 4 and has landed. `_reuse_cached_prefix` is the
seam it attaches at: an optional `RadixCache` is passed to the constructor, and
with none passed every path below behaves exactly as it did in Phase 3.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch

from serving.cache.radix import attach_prefix
from serving.engine_iface.batch import ScheduledSeq, build_batch_meta, build_token_tensor
from serving.memory.block_table import SequenceBlocks
from serving.scheduler.preemption import (
    KVSwapSpace,
    PreemptionPolicy,
    PreemptionStats,
    SwapHandle,
    SwapSpaceFull,
    select_victim,
)

if TYPE_CHECKING:
    from serving.cache.radix import RadixCache
    from serving.memory.allocator import BlockAllocator

__all__ = [
    "Scheduler",
    "SchedulerConfig",
    "Request",
    "RequestState",
    "StepStats",
    "PreemptionPolicy",
    "PreemptionStats",
]

EOS_IDS = {128001, 128008, 128009}


class RequestState(StrEnum):
    WAITING = "waiting"      # admitted to the queue, no blocks yet
    PREFILL = "prefill"      # blocks held, prompt not fully processed
    DECODE = "decode"        # prompt done, generating
    SWAPPED = "swapped"      # preempted; KV parked on the host, no GPU blocks
    FINISHED = "finished"    # EOS or max_tokens
    CANCELLED = "cancelled"  # client disconnected
    FAILED = "failed"


@dataclass
class Request:
    """
    One request's full lifetime inside the scheduler.

    `prompt_ids` is fixed; `output_ids` grows. `prefill_pos` tracks how much of
    the prompt has been processed, which is what makes chunked prefill
    expressible: a request can be partway through its prompt and still yield the
    step back to the scheduler.
    """

    request_id: str
    prompt_ids: list[int]
    max_tokens: int = 64
    arrival_time: float = 0.0
    ignore_eos: bool = False
    """
    Run to `max_tokens` regardless of EOS. BENCHMARK CONTROL, never a serving
    default.

    docs/BENCHMARK_METHODOLOGY.md §4 requires output length to be CONTROLLED
    rather than model-determined, or the workload is not reproducible across
    configurations — a scheduling change that alters which token is sampled also
    alters how much work each request represents, and the comparison stops being
    about scheduling.

    This is not hypothetical. Job 11599377 asked for 64 tokens and got a mean of
    12 (max 55): the model emitted EOS early, so the benchmark was mostly
    measuring prefill, and any decode-throughput reading from it described a
    workload nobody chose.
    """

    state: RequestState = RequestState.WAITING
    blocks: SequenceBlocks | None = None
    output_ids: list[int] = field(default_factory=list)
    prefill_pos: int = 0                       # prompt tokens processed so far
    cached_blocks: int = 0                     # reused from the prefix cache (Phase 4)
    error: str | None = None

    # -- preemption state (Phase 3) -----------------------------------------

    arrival_seq: int = 0
    """
    Monotonic admission-order stamp, assigned by `Scheduler.add_request`.

    Victim selection is LIFO, so it needs a total order on "newness".
    `arrival_time` is a float supplied by the caller and defaults to 0.0 for
    every request in most tests — ordering by it would make victim selection
    depend on sort stability, which is not an ordering at all.
    """

    preemption_count: int = 0
    """How many times this request has been preempted. Feeds the starvation guard."""

    preempted_at_step: int = 0
    """Step number of the most recent preemption; resume latency is measured from it."""

    swap_handle: SwapHandle | None = None
    """Host-resident KV while SWAPPED. None on every other path."""

    resume_tokens: list[int] | None = None
    """
    Under RECOMPUTE, the token sequence to re-prefill: `prompt_ids +
    output_ids`. None until the first recompute-preemption, at which point it
    REPLACES `prompt_ids` as the prefill source (see `prefill_ids`).

    `prompt_ids` and `output_ids` are never rewritten. The request's reported
    output must be exactly what a client would have received without
    preemption, and folding generated tokens into the prompt would make the
    two indistinguishable — which is how a preemption bug becomes unfindable.
    """

    # Set by the scheduler; the transport layer awaits on these.
    on_token: Callable[[int], None] | None = None
    on_finish: Callable[[Request], None] | None = None

    @property
    def prefill_ids(self) -> list[int]:
        """
        The tokens prefill walks. `prompt_ids` normally; `prompt + generated` for
        a request resumed under RECOMPUTE.
        """
        return self.resume_tokens if self.resume_tokens is not None else self.prompt_ids

    @property
    def prompt_len(self) -> int:
        return len(self.prefill_ids)

    @property
    def original_prompt_len(self) -> int:
        """Length of what the client actually sent. Unaffected by preemption."""
        return len(self.prompt_ids)

    @property
    def prefill_done(self) -> bool:
        return self.prefill_pos >= self.prompt_len

    @property
    def total_len(self) -> int:
        """Tokens of KV this request needs once fully processed so far."""
        return self.prefill_pos + len(self.output_ids)

    def is_terminal(self) -> bool:
        return self.state in (RequestState.FINISHED, RequestState.CANCELLED, RequestState.FAILED)


@dataclass
class SchedulerConfig:
    max_batch_size: int = 32
    max_prefill_tokens: int = 512
    """
    Cap on prompt tokens processed in ONE step, across the whole batch.

    Two jobs, and the second is the one people miss:

    1. Latency. A 2000-token prefill run as a single unit stalls every decoding
       sequence in the batch for its full duration — one unrelated request
       causes a visible ITL spike for every concurrent user.

    2. Concurrency. The scheduler runs as a cooperative asyncio task, so the
       event loop is only responsive if a STEP IS SHORT. This cap is what bounds
       step duration. The throughput feature and the concurrency model are the
       same mechanism.

    Phase 2 shipped this as a bare cap. Phase 4 turns it into a scheduling
    feature — see `max_batch_tokens`, `chunk_to_block_boundary` and
    `max_prefills_per_step` below.
    """
    max_waiting: int = 1024
    """Queue depth beyond which new requests are SHED rather than queued."""

    # -- chunked prefill as a SCHEDULING feature (Phase 4) -------------------

    max_batch_tokens: int | None = None
    """
    Cap on TOTAL tokens in one step — prefill chunks plus decodes together.

    `max_prefill_tokens` alone bounds only the prefill half, so a step could
    carry `max_prefill_tokens` prompt tokens on top of `max_batch_size` decodes
    and the "bounded step duration" argument (ARCHITECTURE.md §5.1) would only
    hold for one of the two terms. Since step duration is what keeps the
    cooperative event loop responsive, the bound has to cover the whole batch.

    Decodes are charged against this budget FIRST and are never refused: they
    are one token each and already hold memory, and dropping one to make room
    for a prefill chunk converts a throughput decision into an ITL spike for a
    user who did nothing wrong. Prefill takes what is left, capped additionally
    by `max_prefill_tokens`.

    `None` means `max_prefill_tokens + max_batch_size` — the smallest budget
    that can never starve the decode half.
    """

    chunk_to_block_boundary: bool = True
    """
    Round a non-final prefill chunk DOWN to a block boundary.

    A CHUNK BOUNDARY AND A CACHE-HIT BOUNDARY ARE DIFFERENT THINGS and must not
    be conflated. The cache-hit boundary is a property of the prompt's content:
    where this prompt stops agreeing with something already cached, always a
    whole number of blocks. The chunk boundary is a property of the step budget:
    how much prefill compute fits in one forward pass, and it moves with batch
    composition. Neither constrains the other, and `prefill_pos` is the only
    thing both of them write.

    Aligning chunks to blocks does not make them the same thing; it makes the
    INTERACTION cheap. Every intermediate `prefill_pos` then names a whole
    number of complete blocks, so a chunk boundary is a legal publication point
    for the cache and a long prompt becomes reusable while it is still being
    prefilled rather than only at the end. With a ragged chunk the trailing
    partial block simply is not published — correct, just less useful, which is
    exactly why this is a default and not an invariant.
    """

    max_prefills_per_step: int = 0
    """
    Cap on how many DISTINCT requests may contribute a prefill chunk to one
    step. 0 = unlimited.

    The token budget alone does not bound assembly cost or the tail of the
    step's critical path: sixteen requests contributing 32 tokens each cost the
    same tokens as one contributing 512 but produce a far more ragged batch. A
    low cap also keeps prefill work concentrated, which finishes individual
    prompts sooner and gets their blocks into the prefix cache earlier.
    """

    static_batching: bool = False
    """
    BASELINE B2, not a feature. Admit a batch, run it to completion, admit the
    next — the thing continuous batching is measured against.

    Implemented as a one-line change to admission (see `_admit`) precisely so the
    comparison is honest: the kernels, the paged memory manager, the HTTP stack,
    the tokenizer and the model are all held constant, and the ONLY difference
    between the two configurations is WHEN a request is allowed to join. Any
    goodput delta is therefore attributable to scheduling and to nothing else.

    A separate static-batching server would have been easier to write and
    worthless to compare against, because every other difference would confound
    the result (docs/BENCHMARK_METHODOLOGY.md §6, B2).

    The waste this exposes: under skewed output lengths a finished sequence's
    slot stays occupied until the LONGEST sequence in its batch finishes. That
    idle-slot time is what continuous batching reclaims, and it is why the
    comparison should be reported as slot occupancy as well as throughput.
    """

    # -- preemption (Phase 3) -----------------------------------------------

    enable_preemption: bool = True
    """
    Off turns block exhaustion back into a loud `AllocationError` from
    `SequenceBlocks.append`. Useful for exactly one thing: proving a test rig
    genuinely reaches exhaustion before asserting that preemption handled it.
    A forced-pressure test that never actually forces pressure is the failure
    mode this flag exists to rule out.
    """

    preemption_policy: PreemptionPolicy = PreemptionPolicy.RECOMPUTE
    """
    RECOMPUTE or SWAP (docs/ARCHITECTURE.md §5.2). Both are implemented; the
    head-to-head under forced pressure is the Phase 3 deliverable, so neither is
    "the real one" and both are gated independently — a bug in one must not be
    masked by the other being the default (R3).
    """

    starvation_k: int = 3
    """
    Preemption count past which a sequence is DEPRIORITISED as a victim.

    Deprioritised, not excluded. The exclusion version deadlocks when every
    running sequence reaches K (R40); see `preemption.select_victim`.
    """

    swap_space_bytes: int | None = None
    """
    Cap on pinned host memory held by swapped-out sequences. None = unbounded.
    On exhaustion, SWAP degrades to RECOMPUTE for that victim and the event is
    counted — an uncounted degradation would silently change which policy a
    head-to-head benchmark is actually measuring.
    """


@dataclass
class StepStats:
    """Per-step telemetry. Cheap, and the only way to see what the scheduler did."""

    step: int = 0
    n_running: int = 0
    n_waiting: int = 0
    n_prefill: int = 0
    n_decode: int = 0
    tokens_in_batch: int = 0
    admitted: int = 0
    retired: int = 0
    blocks_free: int = 0
    blocks_used: int = 0
    preempted: int = 0
    resumed: int = 0
    n_swapped: int = 0

    # -- prefix cache (Phase 4) ---------------------------------------------
    blocks_reused: int = 0
    """Blocks served from the prefix cache to requests admitted THIS step."""
    blocks_required: int = 0
    """Blocks those same requests would have needed with no cache. The
    denominator of the block-granularity hit rate (methodology §7)."""
    prefill_tokens_saved: int = 0
    """Prompt tokens that did not have to be prefilled. The TTFT claim, in the
    only unit that supports it."""


class Scheduler:
    """
    Owns the waiting queue, the running batch, and the memory that backs it.

    Single-threaded by design. One scheduler, one allocator, one CUDA stream,
    one event loop — the engine has no thread safety anywhere (no locks in
    `engine/server.py` or `engine/scheduler.py`), and adding one here would
    imply a concurrency model this system does not have.
    """

    def __init__(
        self,
        model,
        backend,
        allocator: BlockAllocator,
        config: SchedulerConfig | None = None,
        prefix_cache: RadixCache | None = None,
    ):
        self.model = model
        self.backend = backend
        self.allocator = allocator
        self.config = config or SchedulerConfig()

        self.prefix_cache = prefix_cache
        """
        Phase 4 radix prefix cache, or None.

        OPTIONAL, and the A/B switch for R6. `None` (or a cache constructed with
        `enabled=False`) leaves every path in this file behaving exactly as it
        did in Phase 3, which is what makes "bit-identical greedy output with
        cache on vs off" a comparison of one flag rather than of two systems.
        """

        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.swapped: list[Request] = []
        """
        Preempted under SWAP: no GPU blocks, KV on the host, waiting for room.

        A THIRD queue rather than a flag on `waiting`, because the two are
        readmitted by different mechanisms. A waiting request needs blocks for
        its first prefill chunk and then computes its KV; a swapped request needs
        blocks for its ENTIRE history at once and then copies its KV back. Sizing
        one admission check to cover both would make it wrong for both.
        """

        self._seq_ids = itertools.count()
        self._arrivals = itertools.count()
        self.step_count = 0
        self.last_stats = StepStats()

        self.preemption = PreemptionStats()
        self._swap: KVSwapSpace | None = None

        # Cache accounting for the step currently being assembled. Reset by
        # `step()`; written by `_reuse_cached_prefix` via `_admit`, which is also
        # called directly by tests and so cannot take a StepStats argument.
        self._admit_blocks_reused = 0
        self._admit_blocks_required = 0
        self._admit_tokens_saved = 0

    # -- admission ----------------------------------------------------------

    def add_request(self, req: Request) -> bool:
        """
        Enqueue a request. Returns False if it was SHED.

        Shedding is a real answer, not a failure. Under overload a system that
        queues without bound converts a throughput problem into an unbounded
        latency problem, and every request ends up violating its SLO instead of
        some requests succeeding. Better to reject early and say so.
        """
        if len(self.waiting) >= self.config.max_waiting:
            req.state = RequestState.FAILED
            req.error = (
                f"queue full ({self.config.max_waiting}); "
                "shed rather than queued unboundedly"
            )
            return False
        # Stamped here, not at admission: "last arrived" must mean what a client
        # would call it. A request that queues behind a long prefill and is
        # admitted later is still the older request.
        req.arrival_seq = next(self._arrivals)
        self.waiting.append(req)
        return True

    def _can_admit(self, req: Request) -> bool:
        """
        Would this request fit, leaving the watermark intact?

        Uses `can_allocate` (watermark-respecting), NOT `allocate`. Admission is
        exactly the decision the watermark exists to gate: it must leave enough
        headroom for every already-running sequence to take its next step.

        CACHE-AWARE SIZING (Phase 4). A request whose first blocks are already
        resident does not need them allocated, so sizing admission against the
        raw prompt length would refuse requests the cache makes nearly free —
        the pathological case being a queue of identical prompts, every one of
        which is cheap and every one of which looks expensive. The probe is
        `match()`, which increfs nothing: a probe that took references would
        have to unwind them on the common path where admission then declines,
        and refcount unwinding on an error path is exactly the code that gets
        skipped once and corrupts memory silently.

        The probe is advisory, and deliberately so. Between here and
        `_reuse_cached_prefix` the eviction below may drop some of the very
        blocks just matched (they have no users yet, so they are legal victims),
        in which case the real acquire matches fewer and needs more. That is a
        sizing miss, not a correctness hole — `_reuse_cached_prefix` re-checks
        against what it actually got.
        """
        bs = self.allocator.block_size
        cached_tokens = 0
        if self.prefix_cache is not None and self.prefix_cache.enabled:
            cached_tokens = self.prefix_cache.match(req.prefill_ids).n_tokens

        remaining = req.prompt_len - cached_tokens
        first_chunk = min(remaining, self.config.max_prefill_tokens)
        blocks_needed = (first_chunk + bs - 1) // bs

        # THE RUNNING SET'S NEXT STEP IS COUNTED EXPLICITLY, not left to the
        # watermark (Phase 3). Once preemption exists, an admission rule that
        # ignores the batch it is joining LIVELOCKS: admit until the pool is
        # full, trigger preemption, evict the newest — which is the request just
        # admitted — requeue it at the front, admit it again next step, forever.
        # No forward pass ever runs and no metric says why. The watermark is a
        # static reservation and cannot express "this specific batch is one step
        # from a page boundary"; `blocks_needed_for_step()` can, and it is exact.
        need = blocks_needed + self.blocks_needed_for_step()

        if self.allocator.can_allocate(need):
            return True
        # The pool may be full of blocks the CACHE is holding and nobody is
        # using. Those are reclaimable without preempting anyone, and reclaiming
        # them is strictly better than refusing admission.
        if self.prefix_cache is not None:
            return self.prefix_cache.reserve(need)
        return False

    def _admit(self) -> int:
        """Move requests from waiting to running while they fit. FIFO — no priority yet."""
        # STATIC BATCHING (B2): a batch runs to completion before the next is
        # admitted. One condition, one baseline.
        if self.config.static_batching and self.running:
            return 0

        # SWAPPED SEQUENCES OUTRANK NEW ARRIVALS. A swapped request has already
        # been paid for in prefill compute and is holding host memory; letting a
        # stream of new admissions consume the room it is waiting for turns
        # preemption into a one-way door. `_resume_swapped` already ran this
        # step and took what it could, so anything still parked did not fit —
        # and admitting on top of that would only make it fit less.
        if self.swapped:
            return 0

        admitted = 0
        while self.waiting and len(self.running) < self.config.max_batch_size:
            req = self.waiting[0]

            if req.state == RequestState.CANCELLED:
                self.waiting.pop(0)
                self._retire(req)
                continue

            if not self._can_admit(req):
                # Head-of-line blocking, and it is deliberate. Skipping ahead to a
                # smaller request would starve large ones indefinitely under load.
                # Fairness beats utilisation here; revisit with priorities later.
                break

            self.waiting.pop(0)
            req.blocks = SequenceBlocks(self.allocator, seq_id=next(self._seq_ids))
            self._reuse_cached_prefix(req)
            req.state = RequestState.PREFILL
            self.running.append(req)
            admitted += 1
        return admitted

    def _reuse_cached_prefix(self, req: Request) -> None:
        """
        THE PHASE 4 SEAM. Match this request against the radix cache, take
        references on whatever matched, and advance `prefill_pos` past it so the
        forward pass computes only the remainder.

        This is docs/ARCHITECTURE.md §9.1 steps 2–4, and the ordering is the
        whole safety argument:

          1. `acquire()` walks the trie and increfs the matched blocks. From
             that instant the blocks are pinned — eviction sees `users > 0` and
             refuses. Doing it in the other order (size first, take references
             later) leaves a window in which the blocks being sized against can
             be evicted.
          2. `attach_prefix()` installs them at the FRONT of the block table and
             sets `num_tokens` to exactly `n_matched_blocks * block_size`.
          3. `prefill_pos` moves to the same number. Prefill then starts at a
             block boundary, `positions` are absolute (`arange(64, 100)` in the
             §9.1 walkthrough), and attention covers reused and recomputed KV
             alike — the kernel cannot tell which was which.

        `prefill_ids`, not `prompt_ids`: a request resumed under RECOMPUTE
        re-prefills its prompt AND its generated tokens, and its own prefix
        blocks are usually still resident. That is §9.2 step 5 — recompute is
        cheaper than the naive analysis says precisely because this lookup runs
        as a published claim path too, with no special case.

        The match is always a whole number of blocks. A prompt sharing 70 tokens
        at `block_size=16` gets FOUR blocks (64 tokens), not four-and-a-bit: the
        straddling block is recomputed. That truncation lives in
        `RadixCache.match` and is the single thing R6 is about.
        """
        req.cached_blocks = 0
        cache = self.prefix_cache
        if cache is None or not cache.enabled or req.blocks is None:
            return

        res = cache.acquire(req.prefill_ids)
        self._admit_blocks_reused += len(res.block_ids)
        self._admit_blocks_required += res.blocks_required
        self._admit_tokens_saved += res.n_tokens
        if not res.block_ids:
            return

        attach_prefix(req.blocks, res.block_ids, res.n_tokens)
        req.prefill_pos = res.n_tokens
        req.cached_blocks = len(res.block_ids)

        # The sizing done in `_can_admit` may have been against a different
        # match (see its docstring). Re-reserve against what was actually
        # obtained, without the watermark: this request is now admitted, and the
        # watermark exists to stop admission, not to starve a sequence that has
        # already taken references.
        bs = self.allocator.block_size
        remaining = min(req.prompt_len - req.prefill_pos, self.config.max_prefill_tokens)
        cache.reserve((remaining + bs - 1) // bs, watermark=False)

    # -- the step -----------------------------------------------------------

    def _select_batch(self) -> tuple[list[Request], list[int]]:
        """
        Choose what runs this step, and how many tokens each contributes.

        Decodes are taken first and unconditionally: they are one token each,
        they are already holding memory, and letting a large prefill crowd them
        out is how ITL spikes happen. Prefill chunks then fill whatever remains
        of the token budget.

        CHUNKED PREFILL AS A SCHEDULING FEATURE (Phase 4)
        -------------------------------------------------
        Phase 2 shipped a bare per-step prefill cap, which bounded step duration
        and nothing else. Three things make it a policy rather than a limit:

        1. **One budget over the whole batch.** Decodes are charged first
           (`max_batch_tokens`), so the bound that keeps the cooperative event
           loop responsive covers the mixed batch that actually runs, not just
           its prefill half (ARCHITECTURE.md §5.1).
        2. **Chunks land on block boundaries.** See `chunk_to_block_boundary` —
           this is where the chunk boundary meets the cache-hit boundary, and
           the point is that they remain DIFFERENT THINGS. `prefill_pos` is the
           only state both write: the cache sets it once, at admission, to a
           block multiple; chunking advances it, also by block multiples until
           the final ragged chunk. Because both only ever leave it block-
           aligned mid-prompt, every intermediate publication into the trie
           covers whole blocks. Conflating them — treating "cached up to here"
           as "already chunked to here", or resetting `prefill_pos` per chunk —
           produces a prompt prefilled from the wrong offset, which is fluent
           and wrong.
        3. **A cap on concurrent prefills** (`max_prefills_per_step`), because
           the token budget alone does not bound batch raggedness.

        A request whose entire prompt came from the cache is impossible here:
        `RadixCache.match` never returns the whole prompt, so `remaining >= 1`
        always and no admitted request can sit in the batch contributing zero
        tokens.
        """
        cfg = self.config
        bs = self.allocator.block_size
        batch: list[Request] = []
        query_lens: list[int] = []

        total_budget = cfg.max_batch_tokens
        if total_budget is None:
            total_budget = cfg.max_prefill_tokens + cfg.max_batch_size

        for req in self.running:
            if req.is_terminal() or not req.prefill_done:
                continue
            batch.append(req)
            query_lens.append(1)

        # Decodes are never refused, so they may exhaust the budget outright;
        # prefill then simply waits a step. That asymmetry is the policy.
        budget = min(cfg.max_prefill_tokens, max(0, total_budget - len(batch)))
        n_prefills = 0

        for req in self.running:
            if req.is_terminal() or req.prefill_done or budget <= 0:
                continue
            if cfg.max_prefills_per_step and n_prefills >= cfg.max_prefills_per_step:
                break
            remaining = req.prompt_len - req.prefill_pos
            chunk = min(remaining, budget)

            if chunk < remaining and cfg.chunk_to_block_boundary:
                # Round DOWN to a block boundary — never up, which would exceed
                # the budget the bound depends on. Falls back to the ragged
                # chunk when rounding down would leave nothing to do, since
                # scheduling zero tokens is worse than a misaligned chunk.
                end = req.prefill_pos + chunk
                aligned = (end // bs) * bs
                if aligned > req.prefill_pos:
                    chunk = aligned - req.prefill_pos

            if chunk <= 0:
                continue
            batch.append(req)
            query_lens.append(chunk)
            budget -= chunk
            n_prefills += 1

        return batch, query_lens

    def step(self) -> StepStats:
        """
        One scheduler iteration: retire, resume, admit, PREEMPT, select, forward,
        sample, update.

        Ordering matters, and every position here is load-bearing:

        * **Retire first.** Retiring frees blocks that resume and admission can
          then use in the same step. The other order makes the pool look fuller
          than it is, so admission is needlessly conservative and throughput
          drops for no reason.
        * **Resume before admit.** A swapped sequence has already consumed
          prefill compute and is holding host memory; admitting a brand-new
          request ahead of it would make preemption a one-way door under
          sustained load.
        * **Preempt after admit, before the forward pass.** After, because
          admission is what creates the pressure and it must be measured against
          the batch that will actually run. Before, because the whole point of
          the trigger is that the forward pass is never entered with a batch the
          pool cannot fund — discovering the shortfall inside `blocks.append()`
          would mean unwinding a partially assembled batch.
        """
        self.step_count += 1
        stats = StepStats(step=self.step_count)

        self._admit_blocks_reused = 0
        self._admit_blocks_required = 0
        self._admit_tokens_saved = 0

        stats.retired = self._retire_finished()
        stats.resumed = self._resume_swapped()
        stats.admitted = self._admit()
        stats.preempted = self._preempt_if_needed()

        stats.blocks_reused = self._admit_blocks_reused
        stats.blocks_required = self._admit_blocks_required
        stats.prefill_tokens_saved = self._admit_tokens_saved

        batch, query_lens = self._select_batch()
        if not batch:
            stats.n_waiting = len(self.waiting)
            stats.n_swapped = len(self.swapped)
            stats.blocks_free = self.allocator.num_free
            self.last_stats = stats
            return stats

        scheduled: list[ScheduledSeq] = []
        for req, q in zip(batch, query_lens, strict=True):
            new_ids = self._tokens_for(req, q)
            # Grow the block table BEFORE assembly: build_batch_meta reads kv_len
            # and the page list off the table, so it must already describe this
            # step's tokens or the KV would land outside the pages the metadata
            # names.
            req.blocks.append(q)
            scheduled.append(ScheduledSeq(blocks=req.blocks, new_token_ids=new_ids))

        meta = build_batch_meta(scheduled, device=self.model.device,
                                page_size=self.allocator.block_size)
        tokens = build_token_tensor(scheduled, device=self.model.device)

        logits = self.model.forward_varlen(tokens, meta, self.backend)   # (n_seqs, vocab)
        next_ids = torch.argmax(logits, dim=-1).tolist()

        for req, q, tok in zip(batch, query_lens, next_ids, strict=True):
            self._apply(req, q, int(tok))

        stats.n_running = len(self.running)
        stats.n_waiting = len(self.waiting)
        stats.n_swapped = len(self.swapped)
        stats.n_prefill = sum(1 for q in query_lens if q > 1)
        stats.n_decode = sum(1 for q in query_lens if q == 1)
        stats.tokens_in_batch = sum(query_lens)
        stats.blocks_free = self.allocator.num_free
        stats.blocks_used = self.allocator.num_used
        self.last_stats = stats
        return stats

    def _tokens_for(self, req: Request, q: int) -> list[int]:
        """The token ids this request contributes to this step."""
        if not req.prefill_done:
            # `prefill_ids`, not `prompt_ids`: a request resumed under RECOMPUTE
            # re-prefills its prompt AND the tokens it already generated.
            return req.prefill_ids[req.prefill_pos:req.prefill_pos + q]
        assert q == 1, "a decoding request contributes exactly one token per step"
        # The token sampled last step. For a request that just finished prefill
        # this is its first generated token.
        return [req.output_ids[-1]]

    def _apply(self, req: Request, q: int, token_id: int) -> None:
        """
        Fold one step's result into a request.

        The subtlety: a request mid-prefill produces logits, but they are not a
        next token yet — only the chunk that COMPLETES the prompt yields one.
        Emitting a token from a partial chunk would generate text from half a
        prompt.
        """
        if not req.prefill_done:
            req.prefill_pos += q
            # Publish whatever whole blocks this chunk completed. Done at every
            # chunk boundary, not only at the end of the prompt: a long prompt
            # under chunked prefill is reusable by a concurrent request as soon
            # as its early blocks exist, and waiting for the last chunk would
            # throw that away for exactly the prompts big enough to matter.
            self._cache_insert(req)
            if not req.prefill_done:
                return                      # more prompt to go; no token yet
            req.output_ids.append(token_id)
            req.state = RequestState.DECODE
        else:
            req.output_ids.append(token_id)

        if req.on_token:
            req.on_token(token_id)

        hit_eos = token_id in EOS_IDS and not req.ignore_eos
        if hit_eos or len(req.output_ids) >= req.max_tokens:
            req.state = RequestState.FINISHED

    # -- preemption (Phase 3) -----------------------------------------------

    def blocks_needed_for_step(self) -> int:
        """
        Blocks the NEXT forward pass would have to allocate, across the running
        batch.

        Exact, not an estimate: it asks each sequence's own block table how many
        new blocks its own contribution would need, which is the same call
        `append()` will make. Per-sequence needs are independent (each grows its
        own page list), so the sum is the true requirement.

        Most steps this is 0 or 1 — a decoding sequence needs a new block only on
        the step that crosses a page boundary. Pressure is bursty precisely
        because those boundaries can align.
        """
        batch, query_lens = self._select_batch()
        return sum(
            req.blocks.blocks_needed_for(q)
            for req, q in zip(batch, query_lens, strict=True)
        )

    def _preempt_if_needed(self) -> int:
        """
        THE TRIGGER. Preempt until this step's batch fits in the free pool.

        Loop rather than a single preemption: one victim may not free enough,
        and re-deriving the requirement after each eviction is what keeps the
        check exact (removing a sequence changes both the demand and the prefill
        token budget, which changes which chunks other sequences contribute).

        Terminates. Each iteration removes one request from `self.running`, and
        the loop exits when `self.running` is empty — at which point nothing is
        holding blocks on behalf of the batch and the next step starts clean.
        """
        if not self.config.enable_preemption:
            return 0

        n = 0
        while self.running:
            need = self.blocks_needed_for_step()
            if need <= self.allocator.num_free:
                break

            if len(self.running) == 1:
                # Nothing else holds blocks (waiting and swapped sequences hold
                # none), so the entire pool minus this one sequence is already
                # free and it STILL cannot take one step. Preempting it would
                # free the pool and change nothing: it would be readmitted and
                # arrive back here. That is a livelock, and the honest answer is
                # that the sequence outgrew the KV pool. Fail it, loudly, with
                # the number that explains it.
                self._fail_oversized(self.running[0], need)
                break

            victim = select_victim(self.running, self.config.starvation_k, self.preemption)
            if victim is None:
                # Unreachable by construction: `self.running` is non-empty and
                # terminal requests were retired at the top of the step. If it
                # ever fires, the starvation guard has become an exclusion rule
                # again and the scheduler is one step from deadlock (R40).
                raise RuntimeError(
                    f"victim selection returned None with {len(self.running)} running "
                    f"requests needing {need} blocks against {self.allocator.num_free} "
                    "free. Preemption policy must never return 'no victim' while the "
                    "batch is non-empty."
                )
            self._preempt(victim)
            n += 1
        return n

    def _preempt(self, req: Request) -> None:
        """Evict one victim under the configured policy."""
        req.preemption_count += 1
        req.preempted_at_step = self.step_count

        if self.config.preemption_policy == PreemptionPolicy.SWAP:
            if self._preempt_swap(req):
                self.preemption.record(PreemptionPolicy.SWAP)
                return
            # Host swap space is full. Degrading to recompute is correct — the
            # request still makes progress — but it changes which policy is
            # being measured, so it is counted rather than absorbed.
            self.preemption.swap_space_exhausted += 1

        self._preempt_recompute(req)
        self.preemption.record(PreemptionPolicy.RECOMPUTE)

    def _preempt_recompute(self, req: Request) -> None:
        """
        RECOMPUTE: free everything, requeue at the FRONT with
        `prompt + generated-so-far` as the new prompt.

        THE OFF-BY-ONE THAT WOULD PRODUCE FLUENT WRONG TEXT
        ---------------------------------------------------
        Say the prompt is `P` (length L) and the request has generated
        `t1..tk`. At the moment of preemption its KV covers `L + k - 1`
        positions: the prefill wrote `P`, then each decode step fed `t_i` and
        wrote one more, and `t_k` was SAMPLED last step but has not been fed
        yet.

        Re-prefill `P + t1..tk` — all k, including the one never fed. That
        produces KV over `L + k` positions and logits at the last of them, which
        is exactly the state the unpreempted run reaches after its next decode
        step (feed `t_k`, KV becomes `L + k`, sample `t_{k+1}`). So the token
        sampled off that final prefill chunk IS `t_{k+1}`, and `_apply` appends
        it as the next output token. Same tokens, same positions, same history.

        The two adjacent wrong answers both produce fluent output:
          * re-prefill `P + t1..t_{k-1}` (stopping at what was fed) rebuilds the
            pre-preemption KV, and the next sampled token is `t_k` AGAIN —
            duplicated in the stream;
          * re-prefill `P + t1..tk` but ALSO re-emit the prefill's sampled token
            as if it were the whole point, without noticing `_apply` already
            appends it — same duplication from the other direction.

        `output_ids` is never truncated or rewritten, so the request's reported
        output stays byte-for-byte what the client would have received. That is
        what makes the bit-identity gate meaningful rather than tautological.

        FRONT of the queue, not the back: the victim already holds prefill
        compute the system paid for, and sending it to the back invites the
        cycle where the same request is evicted every time it is readmitted.
        """
        kv_tokens = req.blocks.num_tokens if req.blocks is not None else 0

        if req.blocks is not None:
            req.blocks.free()
            req.blocks = None

        if req.output_ids:
            req.resume_tokens = list(req.prompt_ids) + list(req.output_ids)
        # If nothing was generated yet the request was mid-prefill; its prefill
        # source is unchanged and it simply restarts from token 0.

        req.prefill_pos = 0
        req.cached_blocks = 0
        req.state = RequestState.WAITING
        self.running.remove(req)
        self.waiting.insert(0, req)

        # The cost this policy defers: KV that existed, was thrown away, and
        # will be recomputed as prefill. The number the head-to-head is about.
        self.preemption.tokens_recomputed += kv_tokens

    def _preempt_swap(self, req: Request) -> bool:
        """
        SWAP: copy the victim's KV to the host, then free its GPU blocks.

        Returns False (having changed nothing) if host swap space is exhausted,
        so the caller can degrade to RECOMPUTE.

        COPY THEN FREE, NEVER THE REVERSE. The free returns physical blocks to
        the allocator, which will hand them to another sequence on the very next
        allocation — the FIFO free list delays that, but delay is not safety.
        Reading them after the free would capture whatever the new owner wrote.
        """
        swap = self._swap_space()
        try:
            handle = swap.swap_out(req.request_id, req.blocks)
        except SwapSpaceFull:
            return False

        self.preemption.bytes_swapped_out += handle.nbytes
        self.preemption.swap_out_seconds += swap.last_swap_out_seconds

        req.blocks.free()
        req.blocks = None
        req.swap_handle = handle
        req.state = RequestState.SWAPPED
        self.running.remove(req)
        self.swapped.append(req)
        return True

    def _swap_space(self) -> KVSwapSpace:
        """Built lazily: a RECOMPUTE-only scheduler never touches host memory."""
        if self._swap is None:
            self._swap = KVSwapSpace(self.backend, max_bytes=self.config.swap_space_bytes)
        return self._swap

    def _resume_swapped(self) -> int:
        """
        Copy swapped sequences back in and return them to the running batch.

        Oldest-arrived first — the mirror of LIFO eviction. Taken together the
        two rules say the same thing: the longer a request has been in the
        system, the more its progress is protected.

        HEAD-OF-LINE ON PURPOSE. If the oldest swapped sequence does not fit,
        the loop stops rather than resuming a smaller one behind it. Skipping
        ahead would let a stream of short sequences keep a long one parked
        indefinitely — the swap-side version of the starvation the LIFO rule
        exists to prevent.

        The room test is deliberately stricter than plain `can_allocate`: a
        resumed sequence needs its whole history back at once AND must be
        steppable immediately, so resuming into space that the next step's
        growth already claims would just preempt it again. That thrash is not a
        deadlock, but it is unbounded wasted PCIe traffic.
        """
        if not self.swapped:
            return 0

        n = 0
        for req in sorted(self.swapped, key=lambda r: r.arrival_seq):
            if req.is_terminal():
                self._drop_swapped(req)
                self._retire(req)
                continue
            if len(self.running) >= self.config.max_batch_size:
                break

            handle = req.swap_handle
            assert handle is not None, "a SWAPPED request must hold a swap handle"

            # Its own blocks, plus one step of growth for everything that will
            # then be running (itself included).
            headroom = sum(
                r.blocks.blocks_needed_for(1) for r in self.running if r.blocks is not None
            )
            need = handle.num_blocks + headroom + 1

            if not self.allocator.can_allocate(need):
                if self.allocator.num_free == self.allocator.num_blocks:
                    # The pool is entirely free and this sequence still does not
                    # fit. No amount of waiting changes that, and silently
                    # parking it forever is the worst available outcome: the
                    # client hangs and the host memory is never reclaimed.
                    self._fail_unschedulable(req, need)
                    continue
                break

            t0 = time.perf_counter()
            blocks = SequenceBlocks(self.allocator, seq_id=handle.seq_id)
            blocks.append(handle.num_tokens)
            self._swap_space().swap_in(handle, blocks)
            elapsed = time.perf_counter() - t0

            self.preemption.bytes_swapped_in += handle.nbytes
            self.preemption.record_resume(elapsed, self.step_count - req.preempted_at_step)

            self._swap_space().release(handle)
            req.swap_handle = None
            req.blocks = blocks
            # Prefill position and output history are untouched by a swap, so
            # the request resumes at the exact token it was interrupted on —
            # including a victim that had not finished its prefill.
            req.state = RequestState.DECODE if req.prefill_done else RequestState.PREFILL

            self._drop_swapped(req)
            self.running.append(req)
            n += 1
        return n

    def _drop_swapped(self, req: Request) -> None:
        if req in self.swapped:
            self.swapped.remove(req)

    def _fail_oversized(self, req: Request, need: int) -> None:
        """
        The last running sequence cannot take one step even with the whole pool.

        This is the one case preemption genuinely cannot fix, and it must not be
        silent: the alternative is a request that is evicted and readmitted
        forever while throughput reads zero and no error is ever raised.
        """
        req.state = RequestState.FAILED
        req.error = (
            f"sequence outgrew the KV pool: needs {need} more block(s) for its next "
            f"step, holds {req.blocks.num_blocks if req.blocks else 0} of "
            f"{self.allocator.num_blocks}, and it is the only running request — "
            "preemption cannot free memory that nobody else is holding."
        )
        self._retire(req)
        self.running.remove(req)

    def _fail_unschedulable(self, req: Request, need: int) -> None:
        """A swapped sequence that cannot fit in an empty pool. Fail it loudly."""
        req.state = RequestState.FAILED
        req.error = (
            f"cannot resume: needs {need} blocks but the pool holds "
            f"{self.allocator.num_blocks} (watermark {self.allocator.watermark_blocks}). "
            "The sequence outgrew the KV pool; admission should have capped its length."
        )
        self._drop_swapped(req)
        self._retire(req)

    # -- retirement ---------------------------------------------------------

    def _retire_finished(self) -> int:
        still_running, n = [], 0
        for req in self.running:
            if req.is_terminal():
                self._retire(req)
                n += 1
            else:
                still_running.append(req)
        self.running = still_running

        # Swapped requests can be cancelled while parked. They hold no GPU
        # blocks but they DO hold host memory, and a queue nobody sweeps is a
        # leak that only shows up as pinned memory the process never returns.
        for req in list(self.swapped):
            if req.is_terminal():
                self._drop_swapped(req)
                self._retire(req)
                n += 1
        return n

    def _cache_insert(self, req: Request) -> None:
        """
        Publish this request's COMPLETED blocks into the prefix cache.

        The only hard part is saying precisely which tokens have KV resident,
        because publishing one token more than is actually computed caches a block whose
        tail slots were never written, and a later reader would attend over
        uninitialised memory with no symptom.

            during prefill   prefill_ids[:prefill_pos]
            during decode    prefill_ids + output_ids[:-1]

        The `[:-1]` is the subtle one. `output_ids[-1]` was sampled from the
        step that just ran; it becomes an INPUT on the next step, and only then
        is its KV written. Including it would over-claim by exactly one token,
        which at the wrong prompt length is one whole block.

        The result is then clamped to `blocks.num_tokens`, the block table's own
        account of what it holds, so the two can never disagree — and `insert`
        itself keeps only whole blocks, dropping any ragged tail.
        """
        cache = self.prefix_cache
        if cache is None or not cache.enabled:
            return
        if req.blocks is None or req.blocks.is_freed or not req.blocks.block_ids:
            return

        ids = list(req.prefill_ids[: req.prefill_pos])
        if req.prefill_done and req.output_ids:
            ids.extend(req.output_ids[:-1])
        n = min(len(ids), req.blocks.num_tokens)
        if n < cache.block_size:
            return
        cache.insert(ids[:n], req.blocks.block_ids)

    def _retire(self, req: Request) -> None:
        """
        Release a request's memory. MUST be idempotent and MUST run on every
        exit path — normal finish, EOS, cancellation, error.

        A leak here is invisible for one request and fatal over a benchmark:
        capacity silently falls until admission starts failing for no apparent
        reason. Cancellation is the dangerous path, because it is the one that
        does not go through the normal completion code.
        """
        # PUBLISH BEFORE FREEING, and the order is the entire point. `insert`
        # takes the cache's own reference on each block it keeps, so a block
        # that goes on to be cached never reaches refcount 0 and is never
        # returned to the free list. Freeing first and inserting after would
        # hand the trie block ids the allocator has already given away — R7 with
        # extra steps.
        #
        # This is also where multi-turn conversational reuse comes from: the
        # tokens published here are prompt PLUS generated, so turn n+1, whose
        # prompt contains turn n's full history, walks straight down the trie.
        self._cache_insert(req)
        if req.blocks is not None and not req.blocks.is_freed:
            req.blocks.free()
        # Host swap space is memory too, and it is the kind nobody notices
        # leaking: no allocator invariant covers it and it shows up only as a
        # process whose RSS never comes back down.
        if req.swap_handle is not None:
            if self._swap is not None:
                self._swap.release(req.swap_handle)
            req.swap_handle = None
        if req.on_finish:
            req.on_finish(req)
        self.finished.append(req)

    def cancel(self, request_id: str) -> bool:
        """
        Client disconnected. Mark for retirement at the next step boundary.

        Not freed inline: this may be called from the HTTP layer while a step is
        mid-flight, and freeing blocks out from under a running forward pass
        would corrupt the batch. Marking is safe; the scheduler collects it.
        """
        for req in self.running:
            if req.request_id == request_id and not req.is_terminal():
                req.state = RequestState.CANCELLED
                return True
        for req in self.waiting:
            if req.request_id == request_id and not req.is_terminal():
                req.state = RequestState.CANCELLED
                return True
        # Swapped requests are still live requests with a client attached. A
        # cancel that silently missed them would leave host memory pinned until
        # the sequence resumed and finished generating output nobody wants.
        for req in self.swapped:
            if req.request_id == request_id and not req.is_terminal():
                req.state = RequestState.CANCELLED
                return True
        return False

    # -- introspection ------------------------------------------------------

    @property
    def has_work(self) -> bool:
        return bool(self.running or self.waiting or self.swapped)

    def snapshot(self) -> dict[str, Any]:
        snap = {
            "step": self.step_count,
            "running": len(self.running),
            "waiting": len(self.waiting),
            "swapped": len(self.swapped),
            "finished": len(self.finished),
            "blocks_free": self.allocator.num_free,
            "blocks_used": self.allocator.num_used,
            "block_utilization": round(self.allocator.utilization, 4),
            "preemption_policy": str(self.config.preemption_policy),
        }
        snap.update(self.preemption.as_dict())
        if self.prefix_cache is not None:
            # Prefixed rather than merged flat: `evictions` means something
            # different here than it would for preemption, and R16 is the
            # standing lesson about two quantities sharing one metric name.
            snap.update(
                {f"cache_{k}": v for k, v in self.prefix_cache.snapshot().items()
                 if k != "definitions"}
            )
        if self._swap is not None:
            snap["swap_bytes_in_use"] = self._swap.bytes_in_use
            snap["swap_bytes_peak"] = self._swap.peak_bytes
        return snap

    def run_until_idle(self, max_steps: int = 100_000) -> int:
        """Drive to completion. Testing and offline batch use; the server awaits steps."""
        steps = 0
        while self.has_work and steps < max_steps:
            self.step()
            steps += 1
        return steps
