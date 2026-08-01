"""
The replica's HTTP surface: OpenAI-compatible ingress, SSE streaming, cancellation.

WHAT THIS FILE EXISTS TO FIX
----------------------------
`vendor/llm_inference_engine/engine/server.py:69` iterates a BLOCKING SYNCHRONOUS
generator inside an `async def` event stream:

    async def _event_stream():
        for token_id in generate(_model, token_ids, sampler_fn, ...):
            yield f"data: ...\\n\\n"

`generate()` is an ordinary generator whose `next()` runs a full forward pass on
the GPU. There is no `await` anywhere between tokens, so from the moment the
first request enters that loop until it emits EOS, the event loop cannot run any
other callback: not the ASGI server's read of a second request, not another
stream's writer, not a disconnect notification. Request 2's TTFT therefore
CONTAINS request 1's entire decode. The server is concurrent in syntax and
serial in behaviour, and any p99-under-load number measured against it is a
statement about arrival order, not about the system.

That is not a performance bug that batching happens to also fix. It is the
reason a "concurrency" claim about that server cannot be made at all, and
undoing it is the entire point of this module.

THE CONCURRENCY MODEL (docs/ARCHITECTURE.md §7 — implemented here, not restated)
-------------------------------------------------------------------------------
ONE asyncio event loop carries HTTP ingress, SSE writing, and disconnect
detection. The scheduler runs on that same loop as a COOPERATIVE TASK
(`SchedulerLoop`), and every iteration of that task is one bounded forward pass
followed by an `await`.

**Why a bounded step is the whole mechanism.** A cooperative task only leaves
the loop responsive if it gives the loop back promptly. `Scheduler.step()` is
synchronous and un-interruptible once entered, so the loop's worst-case latency
to any other callback is exactly one step's duration. What bounds a step is
`SchedulerConfig.max_prefill_tokens`: decodes contribute one token each and
prefills are capped at that many prompt tokens per step across the whole batch
(`serving/scheduler/scheduler.py:_select_batch`). Raise that cap and every
concurrent request's ITL tail rises with it; there is no separate knob for
"responsiveness". The throughput feature and the concurrency model are the same
mechanism, which is why this file must not be read without reading that one.

**One CUDA stream, one thread.** The model is touched only from
`SchedulerLoop.run`'s call to `Scheduler.step()`, on the loop thread. The engine
has zero thread safety — no locks anywhere in `engine/server.py` or
`engine/scheduler.py` — so a thread
pool here would be a correctness bug wearing a performance costume. Batching IS
the parallelism; there is no intra-replica parallelism across requests.

**The handoff.** Each in-flight request owns an `asyncio.Queue`. The scheduler
writes to it through `Request.on_token` / `Request.on_finish`, which run inside
`step()` on this same loop, so `put_nowait` on an unbounded queue is both
non-blocking and safe. The SSE writer awaits that queue. No lock, no thread, no
`run_coroutine_threadsafe` — those become necessary only if the escape hatch in
ADR-007 is ever taken.

CANCELLATION IS A MEMORY PROBLEM, NOT A TIDINESS PROBLEM
--------------------------------------------------------
A disconnected client whose request keeps running holds its KV blocks until
`max_tokens`. The pool is finite and SHARED, so a disconnect-heavy workload
drains it and admission starts failing for requests that have nothing to do with
the disconnects — with no error, no exception, and no metric that obviously
points at the cause. Every exit path from the SSE generator therefore reaches
`scheduler.cancel(request_id)`, and both mechanisms are handled because which
one fires depends on the ASGI server:

  * `await request.is_disconnected()` polled between chunks — Starlette's read of
    an `http.disconnect` message.
  * `GeneratorExit` / `asyncio.CancelledError` raised at the `yield` when the
    server tears the response down — Starlette's `StreamingResponse` runs a
    disconnect listener that cancels the streaming task, so under uvicorn this
    is usually the one that actually fires.

Both funnel into one `finally:`. `Scheduler.cancel` only MARKS the request; the
blocks are freed at the next step boundary, because freeing them inline from the
HTTP layer could pull memory out from under a forward pass that is mid-flight.

TIMING (docs/RISK_REGISTER.md R2)
---------------------------------
`/metrics` reports `step_duration_host_ms`, measured with `time.perf_counter()`
around `Scheduler.step()`. That is honest ONLY because the step currently
contains a device->host copy: `torch.argmax(logits, dim=-1).tolist()` in
`Scheduler.step` forces a CUDA synchronisation, so the host clock sees execution
rather than kernel-launch queueing. If sampling ever moves on-device — which is
exactly what `forward_varlen` returning a device tensor invites — this number
silently becomes a measure of launch queueing, gets FASTER, and stays plausible.
It is named for its clock, excluded from the registered metric namespace, and
labelled in the payload as diagnostic-only for that reason. Client-side timings
(bench/loadgen.py) are unaffected: they measure network-observable behaviour,
which is what they should measure.

METRIC NAMES (docs/RISK_REGISTER.md R16)
----------------------------------------
Names that already exist in `serving/metrics/artifact.py`'s REGISTRY are reused
for exactly the quantity the REGISTRY says they mean, and their specs are echoed
into the payload. Everything else is server-local and deliberately named so it
cannot be mistaken for one of them. `queue_wait_ms` is registered but is NOT
emitted here: it is defined as *intended dispatch -> admission*, and the server
cannot observe intended dispatch — only the load generator knows it. Emitting
arrival-to-admission under that name would be the engine's `peak_mem_mb`
mistake with a different label.

`GET /metrics/prometheus` exposes the same state in the Prometheus text format
(`serving/metrics/prometheus.py`), under the same name policy. It is additive:
the JSON `/metrics` above is what the benchmark harness reads and is unchanged.

A CUDA ERROR IS FATAL TO THIS REPLICA (docs/RISK_REGISTER.md R10)
-----------------------------------------------------------------
`SchedulerLoop.run` checks for CUDA errors once per step
(`serving/metrics/cuda_guard.py`, frequency set by
`ServerConfig.cuda_check_every_n_steps`). An unchecked launch failure does not
raise: the forward pass returns a correctly-shaped tensor of garbage, argmax
picks a token, and the replica emits plausible text at full throughput with every
metric green. That is why the check exists and why finding one is terminal — the
context is poisoned and cannot be reset in-process under PyTorch's caching
allocator.

The error therefore lands in the same handler as any other failed step: the loop
marks itself unhealthy, fails EVERY in-flight stream explicitly with an error
chunk (so the fault appears in the client's accounting rather than as a timeout),
stops stepping, and `/health` returns 503 with `fatal.kind == "cuda"` so a router
quarantines the replica instead of retrying into it. The per-step
`torch.cuda.synchronize()` is also what keeps `step_duration_host_ms` a measure
of execution rather than launch queueing (R2), so it is not purely a cost.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import time
import uuid
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from serving.metrics import prometheus as prom
from serving.metrics.artifact import REGISTRY
from serving.metrics.cuda_guard import CudaFatalError, CudaGuard
from serving.scheduler.scheduler import EOS_IDS
from serving.scheduler.scheduler import Request as SchedRequest

__all__ = [
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatTokenizer",
    "HFChatTokenizer",
    "SchedulerLoop",
    "ServerMetrics",
    "ServerConfig",
    "create_app",
    "build_default_app",
]

MODEL_ID_DEFAULT = "llama-3.2-1b-instruct"

# Metric names taken from serving/metrics/artifact.py's REGISTRY. Listed
# explicitly so that adding one to the payload without registering it — or
# reusing a registered name for a different quantity — fails at import time
# rather than in a comparison six weeks later (R16).
REGISTERED_SERVER_METRICS = ("output_tok_s", "host_rss_mb", "gpu_mem_mb")
for _name in REGISTERED_SERVER_METRICS:
    REGISTRY.get(_name)


# ---------------------------------------------------------------------------
# Wire schema
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """
    The subset of the OpenAI chat-completions body this replica honours.

    `temperature` / `top_p` / `top_k` / `seed` are ACCEPTED AND IGNORED: the
    scheduler samples with `torch.argmax` (`Scheduler.step`), i.e. greedy, and
    every correctness gate in this project (batch invariance, preemption
    equality, backend differential) is a claim about greedy output. Accepting
    them keeps generic clients working; silently honouring only some of them
    would make those gates untestable. `/health` says so in `capabilities`
    rather than leaving it to be discovered.
    """

    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None
    max_tokens: int = Field(default=128, ge=1)
    ignore_eos: bool = Field(
        default=False,
        description=(
            "Generate exactly max_tokens, ignoring EOS. A BENCHMARK control, not a "
            "serving feature: methodology §4 requires output length to be controlled "
            "rather than model-determined, or a scheduling change silently changes "
            "how much work each request represents."
        ),
    )
    stream: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None


# ---------------------------------------------------------------------------
# Tokenizer seam
# ---------------------------------------------------------------------------


@runtime_checkable
class ChatTokenizer(Protocol):
    """
    The only thing this module needs from a tokenizer.

    Narrow on purpose: it is what makes the whole HTTP surface constructible
    with no GPU, no weights, and no `transformers` import, which is why the
    concurrency test below can be a CI test instead of a manual one.
    """

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        """Chat template applied, generation prompt appended. Prompt token ids."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Detokenize. Special tokens are expected to render as ''."""


class HFChatTokenizer:
    """Adapter over a HuggingFace `AutoTokenizer`, matching engine/server.py:55,70."""

    def __init__(self, tokenizer: Any):
        self._tok = tokenizer

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        """
        Chat messages -> token ids, with the return type CHECKED rather than assumed.

        `apply_chat_template` does not have a stable return type across
        transformers versions: depending on version and defaults it returns
        `list[int]`, a nested `list[list[int]]` for a batch, a `BatchEncoding`,
        or — on transformers 5.x — a plain STRING.

        The string case is why this method is not a one-liner. `list("<|begin_of…")`
        is a list of single CHARACTERS, which is a perfectly valid Python list
        that flows all the way down to `torch.tensor(...)` before dying with
        `ValueError: too many dimensions 'str'` from inside the scheduler step —
        an error naming neither the tokenizer nor the chat template. It cost a
        GPU job to localise (job 11599044: every request returned a well-formed,
        empty SSE stream and HTTP 200).

        `engine/server.py:55` does `list(apply_chat_template(...))` too, so the
        engine's reference server has the same latent bug on this transformers
        version. Not fixed there: that path is out of scope here, but it is worth
        knowing before trusting it.
        """
        out = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )

        # BatchEncoding / dict
        if hasattr(out, "get") and not isinstance(out, list | str):
            out = out.get("input_ids", out)
        # Tensors
        if hasattr(out, "tolist"):
            out = out.tolist()
        # Batched: [[ids]] -> [ids]
        if isinstance(out, list) and out and isinstance(out[0], list):
            out = out[0]

        if isinstance(out, str):
            raise TypeError(
                "apply_chat_template returned a string despite tokenize=True "
                f"(transformers version issue). Got: {out[:80]!r}. "
                "Tokenize explicitly instead of iterating the string — "
                "list(str) yields characters and fails much later, inside the "
                "scheduler, as 'too many dimensions str'."
            )
        ids = list(out)
        if not ids or not all(isinstance(i, int) for i in ids):
            bad = next((type(i).__name__ for i in ids if not isinstance(i, int)), "empty")
            raise TypeError(
                f"encode_chat must produce a non-empty list[int]; got {len(ids)} "
                f"items containing {bad}. Anything else reaches torch.tensor() and "
                "dies with an error that names neither the tokenizer nor this call."
            )
        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tok.decode(list(token_ids), skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Per-request handoff
# ---------------------------------------------------------------------------


@dataclass
class _Stream:
    """
    The scheduler -> SSE-writer channel for one request.

    UNBOUNDED, deliberately. A bounded queue would mean the scheduler blocks (or
    drops tokens) when one client reads slowly, and a scheduler that blocks on a
    slow socket has turned one slow client into a head-of-line stall for every
    other request in the batch. The bound that actually matters is `max_tokens`,
    and the guard against a client that never reads is the disconnect path.

    `on_token` / `on_finish` are called from inside `Scheduler.step()`, on this
    same event loop and this same thread, so `put_nowait` is correct and no
    thread-safe handoff is needed.
    """

    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    finished: bool = False

    def on_token(self, token_id: int) -> None:
        self.queue.put_nowait(("token", token_id))

    def on_finish(self, req: SchedRequest) -> None:
        self.finished = True
        self.queue.put_nowait(("finish", req))

    def fail(self, message: str) -> None:
        self.finished = True
        self.queue.put_nowait(("error", message))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class ServerMetrics:
    """Counters the harness reads. Cheap, monotonic, and never reset."""

    started_at: float = field(default_factory=time.perf_counter)
    requests_received: int = 0
    requests_shed_429: int = 0
    requests_streaming: int = 0
    requests_completed: int = 0
    requests_cancelled: int = 0
    requests_failed: int = 0
    output_tokens_total: int = 0
    prompt_tokens_total: int = 0
    empty_detokenizations_total: int = 0
    """
    Tokens whose detokenization was the empty string — special tokens under
    `skip_special_tokens=True`. Counted rather than emitted as a visible chunk
    (see `_sse_stream`), so suppressing them cannot hide a template that emits
    dozens of them.
    """

    histograms: dict[str, prom.Histogram] = field(default_factory=lambda: {
        "ttft_from_arrival_ms": prom.Histogram(prom.LATENCY_BUCKETS_MS),
        "itl_server_ms": prom.Histogram(prom.LATENCY_BUCKETS_MS),
        "e2e_from_arrival_ms": prom.Histogram(prom.LATENCY_BUCKETS_MS),
        "step_duration_host_ms": prom.Histogram(prom.STEP_BUCKETS_MS),
    })
    """
    Fixed-bucket histograms for the Prometheus endpoint. Bounded memory and no
    sample retention, because `observe` runs once per token on the request path.

    NAMED FOR WHAT THE SERVER CAN ACTUALLY SEE (R16). These are `*_from_arrival`
    / `*_server`, not `ttft_ms` / `itl_ms` / `e2e_ms`: the registered metrics are
    defined from the client's INTENDED DISPATCH, and the interval between
    intended dispatch and arrival here is exactly the coordinated-omission
    interval (R1) that this process cannot observe. Published latency numbers
    come from bench/loadgen.py; these are for dashboards and alerts.
    """

    def observe(self, name: str, value_ms: float) -> None:
        hist = self.histograms.get(name)
        if hist is not None:
            hist.observe(value_ms)

    @property
    def uptime_s(self) -> float:
        return max(1e-9, time.perf_counter() - self.started_at)


# ---------------------------------------------------------------------------
# The cooperative scheduler task
# ---------------------------------------------------------------------------


class SchedulerLoop:
    """
    Drives `Scheduler.step()` forever, on the HTTP event loop, yielding between
    iterations.

    Two awaits, and they answer two different questions:

      * IDLE — `await asyncio.sleep(idle_sleep_s)`. With no work there is nothing
        to step, and a bare `sleep(0)` here would spin a core at 100% doing
        nothing, which on a shared node is a benchmark hazard for every other
        job. A few milliseconds of added admission latency on the first request
        of an idle period is a fair price and is bounded by `idle_sleep_s`.

      * BUSY — `await asyncio.sleep(0)`. Yields to the loop exactly once per
        step, so every ready callback (a new POST, an SSE write, a disconnect)
        runs between forward passes, and then comes straight back. Sleeping a
        nonzero amount here would insert idle GPU time into every step and cap
        throughput for no reason; not yielding at all would reproduce the engine's
        bug with extra steps.

    So the loop's responsiveness is: one step's duration, and nothing else. That
    is the claim `SchedulerConfig.max_prefill_tokens` has to keep true.
    """

    def __init__(
        self,
        scheduler: Any,
        metrics: ServerMetrics,
        idle_sleep_s: float = 0.002,
        cuda_guard: CudaGuard | None = None,
    ):
        self.scheduler = scheduler
        self.metrics = metrics
        self.idle_sleep_s = idle_sleep_s
        # Default to a guard rather than to None: "no CUDA error checking
        # anywhere" is the inherited defect (R10), so the safe default has to be
        # ON. On a machine without CUDA every entry point is a no-op, which is
        # why the CPU test suite runs the identical code path.
        self.cuda_guard = cuda_guard if cuda_guard is not None else CudaGuard()

        self.steps = 0
        self.idle_polls = 0
        self.step_time_total_s = 0.0
        self.step_time_max_s = 0.0
        self.last_error: str | None = None
        self.fatal_kind: str | None = None
        self.healthy = True

        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._streams: dict[str, _Stream] = {}

    # -- lifecycle ----------------------------------------------------------

    def ensure_started(self) -> None:
        """
        Start the task if it is not already running on THIS loop.

        Idempotent, and called from the request path as well as from the ASGI
        lifespan. That is not belt-and-braces: `httpx.ASGITransport` does not run
        the lifespan protocol at all, so an app tested through it would otherwise
        have no scheduler and every request would hang. Making startup lazy is
        what keeps the production wiring and the test wiring the same object.
        """
        running = asyncio.get_running_loop()
        if self._task is not None and not self._task.done() and self._loop is running:
            return
        self._loop = running
        self._task = running.create_task(self.run(), name="scheduler-loop")

    async def aclose(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # -- stream registry ----------------------------------------------------

    def register(self, request_id: str, stream: _Stream) -> None:
        self._streams[request_id] = stream

    def unregister(self, request_id: str) -> None:
        self._streams.pop(request_id, None)

    # -- the loop -----------------------------------------------------------

    async def run(self) -> None:
        while True:
            if not self.scheduler.has_work:
                self.idle_polls += 1
                await asyncio.sleep(self.idle_sleep_s)
                continue

            t0 = time.perf_counter()
            try:
                self.scheduler.step()
                # THE DECLARED CUDA CHECK POINT (R10). Inside the try, so a
                # poisoned context takes exactly the same terminal path as a
                # thrown step — the difference is only what `fatal_kind` says.
                # Placed AFTER step() and inside the timed region because the
                # synchronise is also what makes the host clock below measure
                # execution rather than kernel-launch queueing (R2).
                self.cuda_guard.maybe_check("scheduler.step", self.steps)
            except asyncio.CancelledError:
                raise
            except CudaFatalError as exc:
                # A CUDA error is FATAL TO THE REPLICA and is never retried. The
                # context is poisoned; every subsequent forward pass returns
                # garbage of the correct shape, which the client cannot tell from
                # a correct answer. Continuing would convert a detected fault
                # back into the silent-wrong-output mode the check exists to
                # prevent, so this replica stops serving and says so.
                self.last_error = str(exc)
                self.fatal_kind = "cuda"
                self.healthy = False
                for stream in list(self._streams.values()):
                    stream.fail(self.last_error)
                self._streams.clear()
                return
            except Exception as exc:  # noqa: BLE001 — see below
                # A failed step is not recoverable by retrying: the batch that
                # failed is still the batch that will be assembled next step, so
                # a retry loop would spin on the same exception while every
                # client waits forever. Fail LOUDLY instead — every in-flight
                # stream is terminated with an error and /health starts returning
                # 503, so an orchestrator sees a dead replica rather than a
                # silently wedged one (docs/ARCHITECTURE.md §8.2).
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.fatal_kind = "step"
                self.healthy = False
                for stream in list(self._streams.values()):
                    stream.fail(self.last_error)
                self._streams.clear()
                return
            dt = time.perf_counter() - t0

            self.steps += 1
            self.step_time_total_s += dt
            self.step_time_max_s = max(self.step_time_max_s, dt)
            self.metrics.observe("step_duration_host_ms", dt * 1e3)

            # The single most important line in this file. See the class
            # docstring: one yield per bounded step is what makes the server
            # concurrent instead of merely asynchronous.
            await asyncio.sleep(0)

    def stats(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "healthy": self.healthy,
            "steps_total": self.steps,
            "idle_polls_total": self.idle_polls,
            "idle_sleep_s": self.idle_sleep_s,
            "last_error": self.last_error,
            "fatal_kind": self.fatal_kind,
            "cuda": self.cuda_guard.stats(),
            "step_duration_host_ms": {
                "mean": (self.step_time_total_s / self.steps * 1e3) if self.steps else None,
                "max": self.step_time_max_s * 1e3 if self.steps else None,
                "clock": "time.perf_counter() around Scheduler.step()",
                "sync_note": (
                    "HOST CLOCK, DIAGNOSTIC ONLY (R2). Valid only because "
                    "Scheduler.step() ends in torch.argmax(...).tolist(), a "
                    "device->host copy that forces a CUDA sync. If sampling moves "
                    "on-device this measures kernel-launch queueing, gets faster, "
                    "and stays plausible. Not a registered metric; must not back a "
                    "published number without a CUDA-event measurement."
                ),
            },
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ServerConfig:
    model_id: str = MODEL_ID_DEFAULT
    idle_sleep_s: float = 0.002
    disconnect_poll_s: float = 0.05
    """
    How often the SSE writer checks `request.is_disconnected()` while waiting for
    the next token. It is a POLL because ASGI exposes disconnect as a message on
    the receive channel, not as a callback. The cost of a longer interval is
    bounded and specific: up to this many extra seconds of a departed client's
    blocks being held. It is not a correctness bound — the `GeneratorExit` path
    covers a server that tears the response down promptly.
    """
    max_tokens_cap: int = 4096
    max_prompt_tokens: int = 8192
    cuda_check_every_n_steps: int = 1
    """
    How often `SchedulerLoop` checks for CUDA errors (R10). 1 = every step,
    n = every n-th step, 0 = never.

    1 by default. The cost is one `torch.cuda.synchronize()` per step, bounded by
    the step's own device time — work the host has to wait for anyway before the
    next step's sampled tokens are usable. What a larger value buys is host time;
    what it costs is that up to n-1 steps of possibly-garbage output are streamed
    to clients before the fault is noticed, and that
    `step_duration_host_ms` loses its synchronisation guarantee on the unchecked
    steps (R2). Both are declared costs, which is the point of the knob.
    """


# ---------------------------------------------------------------------------
# Chunk construction
# ---------------------------------------------------------------------------


def _chunk(completion_id: str, created: int, model_id: str,
           delta: dict[str, Any], finish_reason: str | None) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _finish_reason(req: SchedRequest) -> str:
    """
    OpenAI's vocabulary, mapped from scheduler state.

    `length` vs `stop` is checked in that order because a request that hits
    `max_tokens` on the same step as EOS is length-limited from the client's
    point of view — it did not get to finish.
    """
    if req.state == "cancelled":
        return "cancelled"
    if req.state == "failed":
        return "error"
    if len(req.output_ids) >= req.max_tokens:
        return "length"
    if req.output_ids and req.output_ids[-1] in EOS_IDS:
        return "stop"
    return "stop"


async def _is_disconnected(request: Request) -> bool:
    """
    Best-effort ASGI disconnect check.

    Wrapped because `Request.is_disconnected()` reads the receive channel, and
    Starlette's own `StreamingResponse` disconnect listener reads the SAME
    channel — under some servers one of the two consumes the message and the
    other raises or blocks. A failure to detect here is not fatal: the
    `GeneratorExit` path in `_sse_stream`'s `finally` still frees the blocks.
    Treating an exception as "not disconnected" keeps a healthy client streaming.
    """
    try:
        return await request.is_disconnected()
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# The app factory
# ---------------------------------------------------------------------------


def create_app(
    scheduler: Any,
    tokenizer: ChatTokenizer,
    config: ServerConfig | None = None,
    metrics: ServerMetrics | None = None,
) -> FastAPI:
    """
    Build the ASGI app around an ALREADY-CONSTRUCTED scheduler and tokenizer.

    Injection rather than construction is what makes this file testable: a fake
    scheduler and a fake tokenizer exercise every path in `tests/test_server.py`
    on CPU with no weights, and the real wiring lives in `build_default_app`
    where it can be nothing but wiring.

    `scheduler` must provide `add_request(Request) -> bool`, `cancel(str) -> bool`,
    `step()`, `has_work`, `snapshot()` and `allocator` — the surface of
    `serving/scheduler/scheduler.Scheduler`, structurally rather than nominally
    so a test double does not have to subclass it.
    """
    cfg = config or ServerConfig()
    m = metrics or ServerMetrics()
    guard = CudaGuard(every_n_steps=cfg.cuda_check_every_n_steps)
    loop = SchedulerLoop(scheduler, m, idle_sleep_s=cfg.idle_sleep_s, cuda_guard=guard)
    ids = itertools.count()

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        loop.ensure_started()
        try:
            yield
        finally:
            await loop.aclose()

    app = FastAPI(title="llm serving layer — replica", lifespan=lifespan)
    app.state.scheduler = scheduler
    app.state.scheduler_loop = loop
    app.state.tokenizer = tokenizer
    app.state.metrics = m
    app.state.config = cfg
    app.state.cuda_guard = guard

    # -- admission ----------------------------------------------------------

    def _admit(body: ChatCompletionRequest) -> tuple[SchedRequest, _Stream] | JSONResponse:
        prompt_ids = tokenizer.encode_chat(
            [{"role": msg.role, "content": msg.content} for msg in body.messages]
        )
        if not prompt_ids:
            return JSONResponse(
                status_code=400,
                content={"error": {"type": "invalid_request_error",
                                   "message": "messages tokenized to zero tokens"}},
            )
        if len(prompt_ids) > cfg.max_prompt_tokens:
            return JSONResponse(
                status_code=413,
                content={"error": {"type": "invalid_request_error",
                                   "message": f"prompt is {len(prompt_ids)} tokens; "
                                              f"cap is {cfg.max_prompt_tokens}"}},
            )

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}-{next(ids)}"
        stream = _Stream()
        req = SchedRequest(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_tokens=min(body.max_tokens, cfg.max_tokens_cap),
            ignore_eos=body.ignore_eos,
            arrival_time=time.perf_counter(),
            on_token=stream.on_token,
            on_finish=stream.on_finish,
        )

        m.requests_received += 1
        m.prompt_tokens_total += len(prompt_ids)

        if not scheduler.add_request(req):
            # LOAD SHEDDING, and it is an answer rather than a failure. Queueing
            # without bound under overload converts a throughput problem into an
            # unbounded latency problem in which EVERY request misses its SLO;
            # shedding keeps the served fraction servable. 429 (not 503) because
            # the condition is per-replica queue depth and a router should retry
            # elsewhere immediately.
            m.requests_shed_429 += 1
            return JSONResponse(
                status_code=429,
                headers={"retry-after": "1"},
                content={"error": {
                    "type": "queue_full",
                    "message": req.error or "scheduler queue is full; request was shed",
                    "scheduler": scheduler.snapshot(),
                }},
            )

        loop.register(request_id, stream)
        return req, stream

    # -- streaming ----------------------------------------------------------

    async def _sse_stream(
        raw_request: Request, req: SchedRequest, stream: _Stream,
    ) -> AsyncGenerator[str, None]:
        completion_id = req.request_id
        created = int(time.time())
        done = False
        # Server-side latency observation. Anchored on `req.arrival_time`, which
        # is when this process first saw the request — NOT the client's intended
        # dispatch. The two differ by exactly the interval coordinated omission
        # hides (R1), which is why these feed histograms named `*_from_arrival`
        # and never the registered ttft_ms / e2e_ms (R16).
        last_content_at: float | None = None
        try:
            # Role chunk first, OpenAI-style. It carries NO content, which is
            # exactly why bench/loadgen.py times TTFT to the first non-empty
            # `delta.content` and not to the first chunk (BENCHMARK_METHODOLOGY
            # §2): timing to this would report a TTFT of roughly zero for every
            # request and the number would be pure fiction.
            yield _chunk(completion_id, created, cfg.model_id, {"role": "assistant"}, None)

            while True:
                if await _is_disconnected(raw_request):
                    break
                try:
                    kind, payload = await asyncio.wait_for(
                        stream.queue.get(), timeout=cfg.disconnect_poll_s
                    )
                except TimeoutError:
                    continue

                if kind == "token":
                    token_id = int(payload)
                    if token_id in EOS_IDS:
                        # The stop token is a control signal, not output. The
                        # terminal chunk carries `finish_reason` instead.
                        continue
                    text = tokenizer.decode([token_id])
                    if not text:
                        # A special token that detokenizes to "". The engine
                        # emitted a chunk for it anyway (engine/server.py:70-78);
                        # doing so publishes an SSE event carrying no information
                        # whose only effect is to mislead any client that times to
                        # the first chunk. Counted, not emitted.
                        m.empty_detokenizations_total += 1
                        continue
                    m.output_tokens_total += 1
                    now = time.perf_counter()
                    if last_content_at is None:
                        m.observe("ttft_from_arrival_ms",
                                  (now - req.arrival_time) * 1e3)
                    else:
                        m.observe("itl_server_ms", (now - last_content_at) * 1e3)
                    last_content_at = now
                    yield _chunk(completion_id, created, cfg.model_id, {"content": text}, None)

                elif kind == "error":
                    m.requests_failed += 1
                    done = True
                    yield _chunk(completion_id, created, cfg.model_id, {}, "error")
                    yield "data: [DONE]\n\n"
                    return

                else:  # "finish"
                    finished: SchedRequest = payload
                    done = True
                    reason = _finish_reason(finished)
                    if reason == "cancelled":
                        m.requests_cancelled += 1
                    else:
                        m.requests_completed += 1
                        m.observe("e2e_from_arrival_ms",
                                  (time.perf_counter() - req.arrival_time) * 1e3)
                    yield _chunk(completion_id, created, cfg.model_id, {}, reason)
                    yield "data: [DONE]\n\n"
                    return
        finally:
            # THE LEAK GUARD. Reached on normal completion, on the polled
            # disconnect above, and on GeneratorExit / CancelledError when the
            # ASGI server tears this response down — which is the path that
            # actually fires under uvicorn, because Starlette's StreamingResponse
            # cancels the streaming task when it sees http.disconnect. Only one of
            # the three needs to work for the blocks to come back; handling all
            # three is why a disconnect-heavy workload does not drain the pool.
            loop.unregister(req.request_id)
            if not done:
                scheduler.cancel(req.request_id)
                m.requests_cancelled += 1

    # -- routes -------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, raw_request: Request):
        loop.ensure_started()
        admitted = _admit(body)
        if isinstance(admitted, JSONResponse):
            return admitted
        req, stream = admitted

        if body.stream:
            m.requests_streaming += 1
            return StreamingResponse(
                _sse_stream(raw_request, req, stream),
                media_type="text/event-stream",
                headers={
                    "cache-control": "no-cache",
                    # Told to every proxy in the path: buffering an SSE stream
                    # turns per-token latency into per-response latency and would
                    # silently invalidate every ITL number measured through it.
                    "x-accel-buffering": "no",
                    "connection": "keep-alive",
                },
            )

        return await _collect(raw_request, req, stream)

    async def _collect(raw_request: Request, req: SchedRequest, stream: _Stream):
        """
        Non-streaming: await the whole generation, then answer once.

        Still fully cooperative — the await is on the queue, so the loop runs
        every other request's work while this one accumulates. The disconnect
        poll is kept because a non-streaming client that goes away is exactly as
        capable of leaking blocks as a streaming one, and it has no chunk
        boundary at which anyone would otherwise notice.
        """
        token_ids: list[int] = []
        finished: SchedRequest | None = None
        try:
            while True:
                if await _is_disconnected(raw_request):
                    scheduler.cancel(req.request_id)
                    m.requests_cancelled += 1
                    return Response(status_code=499)
                try:
                    kind, payload = await asyncio.wait_for(
                        stream.queue.get(), timeout=cfg.disconnect_poll_s
                    )
                except TimeoutError:
                    continue
                if kind == "token":
                    if int(payload) not in EOS_IDS:
                        token_ids.append(int(payload))
                elif kind == "error":
                    m.requests_failed += 1
                    return JSONResponse(
                        status_code=500,
                        content={"error": {"type": "scheduler_error", "message": str(payload)}},
                    )
                else:
                    finished = payload
                    break
        except asyncio.CancelledError:
            scheduler.cancel(req.request_id)
            m.requests_cancelled += 1
            raise
        finally:
            loop.unregister(req.request_id)

        # Decoded as ONE sequence rather than token by token: BPE pieces
        # reassemble correctly across token boundaries this way, which per-token
        # decoding cannot guarantee. The streaming path pays that cost because it
        # has no choice; this one does not have to.
        text = tokenizer.decode(token_ids)
        m.output_tokens_total += len(token_ids)
        m.requests_completed += 1
        return {
            "id": req.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": cfg.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _finish_reason(finished) if finished else "stop",
            }],
            "usage": {
                "prompt_tokens": len(req.prompt_ids),
                "completion_tokens": len(token_ids),
                "total_tokens": len(req.prompt_ids) + len(token_ids),
            },
        }

    @app.get("/health")
    async def health():
        """
        Liveness plus the scheduler snapshot a router needs to make a decision.

        503 when the scheduler task is dead, because a replica whose loop has
        stopped still accepts TCP connections and still returns 200 to a naive
        health check while every request hangs forever. "Healthy" and "able to
        make progress" are different properties and only one of them is worth
        checking (docs/ARCHITECTURE.md §9.3, step 5).
        """
        loop.ensure_started()
        ok = loop.running and loop.healthy
        payload = {
            "status": "ok" if ok else "unhealthy",
            "uptime_s": m.uptime_s,
            "model": cfg.model_id,
            # THE ROUTER'S QUARANTINE SIGNAL (R10). A poisoned CUDA context is
            # not recoverable in-process, so this replica is finished: `fatal`
            # says so explicitly, in a machine-readable field, rather than
            # leaving a router to infer it from a 503 that might be transient.
            # `restart_required` is the distinction that matters — retrying into
            # this replica cannot succeed, and draining it cannot fix it.
            "fatal": None if ok and loop.fatal_kind is None else {
                "kind": loop.fatal_kind,
                "error": loop.last_error,
                "restart_required": loop.fatal_kind is not None,
                "quarantine": loop.fatal_kind is not None,
            },
            "scheduler": scheduler.snapshot(),
            "loop": loop.stats(),
            "capabilities": {
                "streaming": True,
                "sampling": "greedy only (Scheduler.step uses torch.argmax); "
                            "temperature/top_p/top_k/seed are accepted and ignored",
                "prefix_cache": False,
                "preemption": False,
            },
        }
        return JSONResponse(status_code=200 if ok else 503, content=payload)

    @app.get("/metrics")
    async def metrics_endpoint():
        """
        JSON, read by the benchmark harness. `/metrics/prometheus` renders the
        same state for scrapers; this one stays the source the artifact pipeline
        uses, because a scrape format is lossy about units and source in exactly
        the way the artifact schema refuses to be.
        """
        alloc = getattr(scheduler, "allocator", None)
        registered: dict[str, Any] = {
            "output_tok_s": m.output_tokens_total / m.uptime_s,
        }
        registered.update(_process_memory())
        return {
            # Names below come from serving/metrics/artifact.py's REGISTRY and
            # mean exactly what it says they mean (R16). Their specs travel with
            # them so a reader never has to guess the source of a number.
            "registered": registered,
            "metric_specs": {
                name: {
                    "quantity": REGISTRY.get(name).quantity,
                    "unit": REGISTRY.get(name).unit,
                    "source": REGISTRY.get(name).source,
                }
                for name in sorted(registered)
            },
            "not_emitted": {
                "queue_wait_ms": (
                    "Registered as 'intended dispatch -> admission'. The server "
                    "cannot observe intended dispatch — only the load generator "
                    "knows it (R1) — so emitting arrival-to-admission under this "
                    "name would put two quantities behind one name (R16)."
                ),
            },
            "server": {
                "requests_received": m.requests_received,
                "requests_streaming": m.requests_streaming,
                "requests_completed": m.requests_completed,
                "requests_cancelled": m.requests_cancelled,
                "requests_shed_429": m.requests_shed_429,
                "requests_failed": m.requests_failed,
                "output_tokens_total": m.output_tokens_total,
                "prompt_tokens_total": m.prompt_tokens_total,
                "empty_detokenizations_total": m.empty_detokenizations_total,
                "uptime_s": m.uptime_s,
            },
            "scheduler": scheduler.snapshot(),
            "allocator": {
                "num_blocks": alloc.num_blocks,
                "block_size": alloc.block_size,
                "watermark_blocks": alloc.watermark_blocks,
                "blocks_free": alloc.num_free,
                "blocks_used": alloc.num_used,
                "utilization": alloc.utilization,
                "total_allocated": alloc.total_allocated,
                "total_freed": alloc.total_freed,
                "peak_used": alloc.peak_used,
                "tokens_capacity": alloc.tokens_capacity(),
            } if alloc is not None else None,
            "loop": loop.stats(),
        }

    @app.get("/metrics/prometheus")
    async def metrics_prometheus():
        """
        The same state as `/metrics`, in Prometheus exposition format.

        ADDITIVE, not a replacement. The benchmark harness reads the JSON
        endpoint and the artifact schema is what backs a published number; this
        one is for scraping, dashboards, and alerts. Two renderings of one set of
        counters — there is no second source of truth.

        Content type is the text-format 0.0.4 media type; `version=0.0.4` is not
        decoration, it is how a scraper knows not to try to parse this as
        protobuf or OpenMetrics.
        """
        return Response(
            content=prom.render_server_metrics(
                m, scheduler, loop, guard=guard, process_memory=_process_memory()
            ),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


def _process_memory() -> dict[str, float]:
    """`host_rss_mb` and `gpu_mem_mb` — two numbers, two names, on purpose (R16)."""
    out: dict[str, float] = {}
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes. Getting this wrong by 1024x is
        # the kind of thing that survives review because the number still looks
        # like a memory figure.
        import sys as _sys

        out["host_rss_mb"] = rss / (1024 * 1024) if _sys.platform == "darwin" else rss / 1024
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        if torch.cuda.is_available():
            out["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------------------
# Production wiring
# ---------------------------------------------------------------------------


def build_default_app(
    weights_path: str | None = None,
    device: str = "cuda:0",
    block_size: int = 16,
    watermark_blocks: int | None = None,
    max_batch_size: int = 32,
    max_prefill_tokens: int = 512,
    max_waiting: int = 1024,
    total_vram_bytes: int | None = None,
    prefer_flashinfer: bool = True,
    config: ServerConfig | None = None,
) -> FastAPI:
    """
    Wire the real engine + backend + allocator + scheduler, then hand them to
    `create_app`.

    Deliberately nothing but wiring, and deliberately NOT imported at module
    scope: `torch`, `transformers`, the engine, and the weights are all required
    here and none of them are required to test the HTTP surface. Every import in
    this function is local for that reason.

    Note what is NOT here: no global mutable model, no `@app.on_event("startup")`
    that populates module-level `_model` / `_tokenizer` (engine/server.py:15-31).
    Those globals are what make that server impossible to instantiate twice, and
    impossible to test at all without weights on disk.
    """
    # `uvicorn --factory` calls this with NO arguments, so every default here has
    # to be correct unattended. Weights are large and gated and therefore live
    # outside the repo; LLM_WEIGHTS_PATH is how the Slurm job points at them
    # without symlinking into vendor/ (which dirties the submodule and makes the
    # run unpublishable — see results/p1/RESULTS.md §3).
    if weights_path is None:
        weights_path = os.environ.get("LLM_WEIGHTS_PATH", "weights")

    # Baseline B2 is selected by environment because the server is launched by a
    # Slurm script, not by Python. Same process, same wiring, one flag.
    _static = os.environ.get("SERVING_STATIC_BATCHING") == "1"

    import torch
    from engine.loader import load_config, load_weights_gpu
    from engine.model_gpu import LlamaModelGPU
    from transformers import AutoTokenizer

    from serving.backends.paged_torch import PagedTorchBackend
    from serving.memory.allocator import BlockAllocator
    from serving.memory.sizing import ModelKVShape, plan_kv_pool
    from serving.scheduler.scheduler import Scheduler, SchedulerConfig

    model_cfg = load_config(weights_path)
    weights = load_weights_gpu(weights_path, model_cfg, device=device)
    model = LlamaModelGPU(weights, model_cfg, device=device)
    tokenizer = HFChatTokenizer(AutoTokenizer.from_pretrained(weights_path))

    n_layers = model_cfg["num_hidden_layers"]
    n_kv_heads = model_cfg["num_key_value_heads"]
    n_heads = model_cfg["num_attention_heads"]
    head_dim = model_cfg["head_dim"]

    # Size the pool from the REAL device capacity when there is one. The nominal
    # constant in serving/memory/sizing.py is a planning figure; the driver
    # reports somewhat less, and sizing a pool from marketing capacity is how a
    # replica OOMs on its first full batch.
    if total_vram_bytes is None:
        total_vram_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    weight_bytes = sum(
        t.numel() * t.element_size()
        for t in {id(v): v for v in weights.values() if torch.is_tensor(v)}.values()
    )
    plan = plan_kv_pool(
        total_vram_bytes=total_vram_bytes,
        model_weight_bytes=weight_bytes,
        block_size=block_size,
        shape=ModelKVShape(
            name=model_cfg.get("_name_or_path", "model"),
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype_bytes=2,
        ),
    )

    backend = None
    if prefer_flashinfer:
        try:
            from serving.backends.flashinfer_backend import FlashInferBackend

            backend = FlashInferBackend(
                num_layers=n_layers, num_blocks=plan.num_blocks, block_size=block_size,
                n_kv_heads=n_kv_heads, n_heads=n_heads, head_dim=head_dim, device=device,
            )
        except Exception:  # noqa: BLE001 — the fallback IS the design (R18)
            backend = None
    if backend is None:
        backend = PagedTorchBackend(
            num_layers=n_layers, num_blocks=plan.num_blocks, block_size=block_size,
            n_kv_heads=n_kv_heads, n_heads=n_heads, head_dim=head_dim, device=device,
        )

    if watermark_blocks is None:
        # Enough headroom for every sequence in a full batch to take one more
        # block. That is the condition the watermark exists to guarantee
        # (docs/ARCHITECTURE.md §3.3): admission must stop BEFORE the running set
        # becomes unsteppable, not after.
        watermark_blocks = max_batch_size

    allocator = BlockAllocator(
        num_blocks=plan.num_blocks, block_size=block_size, watermark_blocks=watermark_blocks
    )
    scheduler = Scheduler(
        model, backend, allocator,
        SchedulerConfig(
            max_batch_size=max_batch_size,
            max_prefill_tokens=max_prefill_tokens,
            max_waiting=max_waiting,
            static_batching=_static,
        ),
    )
    if _static:
        # Loud, because a baseline silently measured as the system under test is
        # the worst possible benchmark error: both configurations would report
        # the same number and the comparison would look like a null result.
        print(
            "SERVING_STATIC_BATCHING=1 — running BASELINE B2 (static batching), "
            "not the system under test.",
            flush=True,
        )
    return create_app(scheduler, tokenizer, config=config)
