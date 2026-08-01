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

WHAT IS DELIBERATELY NOT HERE YET
---------------------------------
Preemption is Phase 3. This scheduler ADMITS conservatively — the watermark
(`BlockAllocator.can_allocate`) refuses admission while the running set might
not be steppable — so it should never reach block exhaustion. If it does, that
is a bug in admission, not an occasion to improvise an eviction policy, and it
raises loudly rather than silently degrading.

The radix prefix cache is Phase 4. `_reuse_cached_prefix` is the single seam
where it will attach; today it always reports a miss.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch

from serving.engine_iface.batch import ScheduledSeq, build_batch_meta, build_token_tensor
from serving.memory.block_table import SequenceBlocks

if TYPE_CHECKING:
    from serving.memory.allocator import BlockAllocator

__all__ = ["Scheduler", "SchedulerConfig", "Request", "RequestState", "StepStats"]

EOS_IDS = {128001, 128008, 128009}


class RequestState(StrEnum):
    WAITING = "waiting"      # admitted to the queue, no blocks yet
    PREFILL = "prefill"      # blocks held, prompt not fully processed
    DECODE = "decode"        # prompt done, generating
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

    # Set by the scheduler; the transport layer awaits on these.
    on_token: Callable[[int], None] | None = None
    on_finish: Callable[[Request], None] | None = None

    @property
    def prompt_len(self) -> int:
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

    Phase 4 turns this into full chunked prefill with cache interaction; here it
    is deliberately just a cap.
    """
    max_waiting: int = 1024
    """Queue depth beyond which new requests are SHED rather than queued."""

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
    ):
        self.model = model
        self.backend = backend
        self.allocator = allocator
        self.config = config or SchedulerConfig()

        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []

        self._seq_ids = itertools.count()
        self.step_count = 0
        self.last_stats = StepStats()

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
        self.waiting.append(req)
        return True

    def _can_admit(self, req: Request) -> bool:
        """
        Would this request fit, leaving the watermark intact?

        Uses `can_allocate` (watermark-respecting), NOT `allocate`. Admission is
        exactly the decision the watermark exists to gate: it must leave enough
        headroom for every already-running sequence to take its next step.
        """
        first_chunk = min(req.prompt_len, self.config.max_prefill_tokens)
        blocks_needed = (first_chunk + self.allocator.block_size - 1) // self.allocator.block_size
        return self.allocator.can_allocate(blocks_needed)

    def _admit(self) -> int:
        """Move requests from waiting to running while they fit. FIFO — no priority yet."""
        # STATIC BATCHING (B2): a batch runs to completion before the next is
        # admitted. One condition, one baseline.
        if self.config.static_batching and self.running:
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
        Phase 4 seam: a radix cache would match `req.prompt_ids` against cached
        blocks here, incref the matched blocks into `req.blocks`, and advance
        `req.prefill_pos` past them so prefill only computes the remainder.

        Today it is always a miss. Kept as a named method rather than a comment
        so the attachment point is unambiguous and testable.
        """
        req.cached_blocks = 0

    # -- the step -----------------------------------------------------------

    def _select_batch(self) -> tuple[list[Request], list[int]]:
        """
        Choose what runs this step, and how many tokens each contributes.

        Decodes are taken first and unconditionally: they are one token each,
        they are already holding memory, and letting a large prefill crowd them
        out is how ITL spikes happen. Prefill chunks then fill whatever remains
        of the token budget.
        """
        batch: list[Request] = []
        query_lens: list[int] = []
        budget = self.config.max_prefill_tokens

        for req in self.running:
            if req.is_terminal() or not req.prefill_done:
                continue
            batch.append(req)
            query_lens.append(1)

        for req in self.running:
            if req.is_terminal() or req.prefill_done or budget <= 0:
                continue
            remaining = req.prompt_len - req.prefill_pos
            chunk = min(remaining, budget)
            if chunk <= 0:
                continue
            batch.append(req)
            query_lens.append(chunk)
            budget -= chunk

        return batch, query_lens

    def step(self) -> StepStats:
        """
        One scheduler iteration: retire, admit, select, forward, sample, update.

        Ordering matters. Retiring first frees blocks that admission can then
        use in the same step — do it the other way round and the pool looks
        fuller than it is, so admission is needlessly conservative and
        throughput drops for no reason.
        """
        self.step_count += 1
        stats = StepStats(step=self.step_count)

        stats.retired = self._retire_finished()
        stats.admitted = self._admit()

        batch, query_lens = self._select_batch()
        if not batch:
            stats.n_waiting = len(self.waiting)
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
            return req.prompt_ids[req.prefill_pos:req.prefill_pos + q]
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
        return n

    def _retire(self, req: Request) -> None:
        """
        Release a request's memory. MUST be idempotent and MUST run on every
        exit path — normal finish, EOS, cancellation, error.

        A leak here is invisible for one request and fatal over a benchmark:
        capacity silently falls until admission starts failing for no apparent
        reason. Cancellation is the dangerous path, because it is the one that
        does not go through the normal completion code.
        """
        if req.blocks is not None and not req.blocks.is_freed:
            req.blocks.free()
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
        return False

    # -- introspection ------------------------------------------------------

    @property
    def has_work(self) -> bool:
        return bool(self.running or self.waiting)

    def snapshot(self) -> dict[str, Any]:
        return {
            "step": self.step_count,
            "running": len(self.running),
            "waiting": len(self.waiting),
            "finished": len(self.finished),
            "blocks_free": self.allocator.num_free,
            "blocks_used": self.allocator.num_used,
            "block_utilization": round(self.allocator.utilization, 4),
        }

    def run_until_idle(self, max_steps: int = 100_000) -> int:
        """Drive to completion. Testing and offline batch use; the server awaits steps."""
        steps = 0
        while self.has_work and steps < max_steps:
            self.step()
            steps += 1
        return steps
