"""
The router process: OpenAI-compatible ingress, prefix-aware fan-out to N replicas.

THE ORGANIZING PRINCIPLE — THE ROUTER HOLDS HINTS, THE REPLICA HOLDS TRUTH
--------------------------------------------------------------------------
Every piece of state in this process is an approximation that may be stale: the
prefix->replica hint table, the in-flight counts, the health verdicts. If any of
them is wrong, a request lands on a suboptimal replica and runs slightly slower.
**It never produces a wrong answer, never corrupts cache state, and never
requires a distributed transaction.** That property is what makes this layer safe
to build without consensus, and it is the first thing to say when asked "how do
you keep the router's view of the cache consistent?" — you don't, and that's the
design (docs/ARCHITECTURE.md §1, §6).

Which is why this file contains no lock, no lease, no two-phase anything, and no
shared store. Replicas share nothing (§1); the router coordinates nothing. The
whole distributed-systems story is: one advisory table, and a failure path that
is careful about exactly one thing.

THE ONE PLACE THE HINT ARGUMENT DOES NOT APPLY
----------------------------------------------
A stale *hint* costs latency. A stale belief about *liveness* can cost a client's
response. So the failure path (§9.3, implemented in `health.py` and used here)
classifies each in-flight request by how much the client has already been told,
and refuses to transparently retry anything that has already streamed visible
tokens. Re-generating from the prompt would either duplicate output or produce a
different continuation stitched onto what the client already has — silently wrong
output, which is worse than a visible failure. Those requests get an explicit SSE
error event and are counted separately. See `_terminate_with_error`.

WHAT THIS PROCESS IS NOT
------------------------
It is not a load balancer with cache flavouring bolted on, and it is not
highly available. It is a SINGLE POINT OF FAILURE, deliberately
(docs/ARCHITECTURE.md §8.1, R23): making it HA needs either a shared consistent
view of replica state — and the entire point of §6 is not having one — or
multiple routers with divergent hint tables. The second is actually fine, since
hints are advisory and two routers disagreeing costs cache locality rather than
correctness, which is precisely what makes that fix cheap *if* it is ever wanted.
It is out of scope here and stated rather than hidden.

CPU ONLY. Nothing in this module imports torch, the engine, or a tokenizer. That
is what lets the whole router be developed and tested against mock replicas with
no GPU and no network, so PACE queue time never blocks router work (R21).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from serving.router.health import (
    HealthConfig,
    InFlightRecord,
    ReplicaPool,
    ReplicaStatus,
    ReplicaTarget,
)
from serving.router.policy import (
    LeastOutstanding,
    PrefixAware,
    PrefixKeyer,
    RouteRequest,
    RoutingPolicy,
    build_policy,
)

__all__ = [
    "RouterConfig",
    "RouterMetrics",
    "ReplicaClient",
    "ReplicaUnavailable",
    "HttpxReplicaClient",
    "create_router_app",
    "build_default_router",
]

MODEL_ID_DEFAULT = "llama-3.2-1b-instruct"


# ---------------------------------------------------------------------------
# Wire schema — the ingress is the replica's, unchanged
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class RouterChatRequest(BaseModel):
    """
    Deliberately the same body `serving/server/app.py` accepts, forwarded verbatim.

    The router does not interpret sampling parameters, does not rewrite prompts,
    and does not tokenize (see `PrefixKeyer`). Anything it added or dropped here
    would be a second place where a benchmark's request differs from the request
    the replica was measured on.
    """

    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None
    max_tokens: int = Field(default=128, ge=1)
    ignore_eos: bool = False
    stream: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None


# ---------------------------------------------------------------------------
# The replica seam
# ---------------------------------------------------------------------------


class ReplicaUnavailable(Exception):
    """
    A replica could not serve this request.

    `quarantine` separates "this replica is broken" from "this replica is full".
    A 429 from a replica is its admission controller doing its job — the request
    should go somewhere else, and taking a working replica out of service for
    correctly shedding would remove capacity at exactly the moment capacity is
    scarce. Transport errors, timeouts, and 5xx mean broken.
    """

    def __init__(self, message: str, *, status: int | None = None, quarantine: bool = True):
        super().__init__(message)
        self.status = status
        self.quarantine = quarantine


class ReplicaClient(Protocol):
    """
    Everything the router needs from a replica. Three methods, all injectable.

    This is the seam that makes `tests/test_router_app.py` a CPU test with no
    network: a mock replica is an object with these three methods and a dict of
    behaviours. Nothing above it knows whether there is an HTTP stack under it.
    """

    async def probe(self, target: ReplicaTarget, timeout_s: float) -> bool:
        """Active health check. True iff the replica reports it can make progress."""

    def stream_chat(
        self, target: ReplicaTarget, payload: dict[str, Any], timeout_s: float
    ) -> AsyncIterator[str]:
        """Yield SSE frames (`'data: {...}\\n\\n'`) as the replica produces them."""

    async def complete_chat(
        self, target: ReplicaTarget, payload: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        """Non-streaming completion. Raises `ReplicaUnavailable` on failure."""


class HttpxReplicaClient:
    """
    The production client. Streams straight through; never buffers a response.

    Buffering here would defeat the whole system: per-token latency would become
    per-response latency and every ITL measured through the router would be
    fiction (the same reason the replica sets `x-accel-buffering: no`).
    """

    def __init__(self, client: Any = None):
        self._client = client
        self._owned = client is None

    async def _ensure(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=None)
        return self._client

    async def aclose(self) -> None:
        if self._owned and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def probe(self, target: ReplicaTarget, timeout_s: float) -> bool:
        client = await self._ensure()
        try:
            resp = await client.get(target.health_url, timeout=timeout_s)
        except Exception:  # noqa: BLE001 — a probe outcome is data, not an exception path
            return False
        # The replica returns 503 when its scheduler task is dead while still
        # accepting TCP and still answering. That is the hung-but-connected case
        # the active probe exists for; a naive "did it respond" check would miss
        # exactly the failure it was added to catch.
        return resp.status_code == 200

    async def stream_chat(
        self, target: ReplicaTarget, payload: dict[str, Any], timeout_s: float
    ) -> AsyncIterator[str]:
        client = await self._ensure()
        try:
            async with client.stream(
                "POST", target.chat_url, json=payload, timeout=timeout_s
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise ReplicaUnavailable(
                        f"HTTP {resp.status_code}: {body[:200]!r}",
                        status=resp.status_code,
                        quarantine=resp.status_code != 429,
                    )
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line:
                        yield f"{line}\n\n"
        except ReplicaUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ReplicaUnavailable(f"{type(exc).__name__}: {exc}") from exc

    async def complete_chat(
        self, target: ReplicaTarget, payload: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        client = await self._ensure()
        try:
            resp = await client.post(target.chat_url, json=payload, timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001
            raise ReplicaUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code != 200:
            raise ReplicaUnavailable(
                f"HTTP {resp.status_code}", status=resp.status_code,
                quarantine=resp.status_code != 429,
            )
        return resp.json()


# ---------------------------------------------------------------------------
# Config and metrics
# ---------------------------------------------------------------------------


@dataclass
class RouterConfig:
    model_id: str = MODEL_ID_DEFAULT
    request_timeout_s: float = 120.0
    probe_enabled: bool = True
    keyer: PrefixKeyer = field(default_factory=PrefixKeyer)
    drain_poll_s: float = 0.01
    shed_retry_after_s: int = 1


@dataclass
class RouterMetrics:
    """
    Full request accounting, because S7 requires it (the phase plan §8).

    `failed_midstream` is the number that matters and the number a less honest
    router would fold into `failed`: requests whose replica died after visible
    tokens had reached the client, which were terminated explicitly rather than
    silently retried. A fault-injection run publishes it next to `retried`, and
    the two together are what "handled the failure" means concretely.
    """

    started_at: float = field(default_factory=time.perf_counter)
    received: int = 0
    routed: int = 0
    completed: int = 0
    retried: int = 0
    shed_429: int = 0
    unavailable_503: int = 0
    failed: int = 0
    failed_midstream: int = 0
    content_chunks_forwarded: int = 0

    @property
    def uptime_s(self) -> float:
        return max(1e-9, time.perf_counter() - self.started_at)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests_received": self.received,
            "requests_routed": self.routed,
            "requests_completed": self.completed,
            "requests_retried": self.retried,
            "requests_shed_429": self.shed_429,
            "requests_unavailable_503": self.unavailable_503,
            "requests_failed": self.failed,
            "requests_failed_midstream": self.failed_midstream,
            "content_chunks_forwarded": self.content_chunks_forwarded,
            "uptime_s": self.uptime_s,
        }


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _frame(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _content_length(frame: str) -> int:
    """
    Characters of visible content in an SSE frame, or 0.

    Visible content — not "a chunk" — because the first chunk carries
    `{"role": "assistant"}` and no content, and the whole retry classification
    turns on whether the CLIENT has seen anything (methodology §2 makes the same
    distinction for TTFT). Treating the role chunk as streamed output would
    forfeit a retry that is completely safe.
    """
    line = frame.strip()
    if not line.startswith("data:"):
        return 0
    payload = line[len("data:"):].strip()
    if payload == "[DONE]" or not payload:
        return 0
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return 0
    choices = obj.get("choices") or [{}]
    delta = (choices[0] or {}).get("delta") or {}
    return len(delta.get("content") or "")


# ---------------------------------------------------------------------------
# The app factory
# ---------------------------------------------------------------------------


def create_router_app(
    targets: Sequence[ReplicaTarget],
    policy: RoutingPolicy | None = None,
    client: ReplicaClient | None = None,
    config: RouterConfig | None = None,
    health: HealthConfig | None = None,
    pool: ReplicaPool | None = None,
) -> FastAPI:
    """
    Build the router around an injected policy, client, and pool.

    Injection everywhere for the same reason `serving/server/app.py` does it: the
    entire routing layer is then exercisable against mock replicas on CPU, which
    is R21's mitigation — multi-process GPU orchestration under Slurm is fiddly,
    the policy is pure logic, and queue time must never block this work.
    """
    cfg = config or RouterConfig()
    pol = policy or PrefixAware()
    cli = client or HttpxReplicaClient()
    hcfg = health or HealthConfig()
    metrics = RouterMetrics()

    replicas = pool or ReplicaPool(targets, hcfg)
    # THE WIRE THAT MAKES §9.3 STEP 2 REAL: quarantine purges the policy's hints
    # for that replica. Set here rather than inside ReplicaPool so health.py
    # never imports a policy and stays testable on its own.
    replicas.on_quarantine = pol.purge_replica

    app = FastAPI(title="llm serving layer — router", lifespan=_lifespan)
    app.state.config = cfg
    app.state.policy = pol
    app.state.pool = replicas
    app.state.client = cli
    app.state.metrics = metrics
    app.state.accepting = True
    app.state.inflight = 0
    app.state.prober = _Prober(replicas, cli, cfg)

    # -- selection ----------------------------------------------------------

    def _route_request(body: RouterChatRequest, request_id: str) -> RouteRequest:
        return RouteRequest(
            request_id=request_id,
            prefix_key=cfg.keyer.key_from_messages(body.messages),
            prompt_tokens=sum(len(m.content) for m in body.messages) // 4,
        )

    def _select(req: RouteRequest) -> str | None:
        now = time.monotonic()
        return pol.select(req, replicas.views(now), now)

    def _payload(body: RouterChatRequest) -> dict[str, Any]:
        out = body.model_dump()
        out["model"] = body.model or cfg.model_id
        return out

    # -- streaming with retry ----------------------------------------------

    async def _stream(
        body: RouterChatRequest, req: RouteRequest, first_replica: str,
    ) -> AsyncGenerator[str, None]:
        """
        Proxy one streaming request, re-routing it if a replica dies under it.

        The loop is §9.3 steps 1b -> 3 -> 4 in order: an error on the request
        path is the fast detection signal, the record's phase decides whether a
        retry is even permitted, and the delay before the retry is jittered.
        """
        payload = _payload(body)
        replica_id: str | None = first_replica
        record = InFlightRecord(
            request_id=req.request_id, replica_id=first_replica, started_at=time.monotonic()
        )
        attempt = 0

        while True:
            if replica_id is None:
                async for f in _terminate_with_error(
                    req, record, "no eligible replica remains", metrics
                ):
                    yield f
                return

            record.replica_id = replica_id
            record.attempts = attempt + 1
            replicas.acquire(replica_id)
            pol.on_assign(req, replica_id, time.monotonic())
            metrics.routed += 1
            failure: ReplicaUnavailable | None = None
            try:
                async for frame in cli.stream_chat(
                    replicas.target(replica_id), payload, cfg.request_timeout_s
                ):
                    record.note_response_started()
                    n = _content_length(frame)
                    if n:
                        # THE POINT OF NO RETURN. Once this frame is yielded the
                        # client has it, and no later retry can un-send it.
                        record.note_content()
                        metrics.content_chunks_forwarded += 1
                    yield frame
                replicas.on_request_success(replica_id)
                metrics.completed += 1
                return
            except ReplicaUnavailable as exc:
                failure = exc
            except (asyncio.CancelledError, GeneratorExit):
                # Client went away. Nothing to re-route; just give the slot back.
                raise
            finally:
                replicas.release(replica_id)
                pol.on_complete(req, replica_id, time.monotonic(), ok=failure is None)

            # -- failure path ---------------------------------------------
            if failure is not None and failure.quarantine:
                replicas.on_request_failure(replica_id, str(failure))

            if not replicas.classify_and_count(record):
                async for f in _terminate_with_error(req, record, str(failure), metrics):
                    yield f
                return

            metrics.retried += 1
            delay = replicas.retry_delay(attempt)
            if delay > 0:
                await asyncio.sleep(delay)
            attempt += 1
            replica_id = _select(req)

    async def _terminate_with_error(
        req: RouteRequest, record: InFlightRecord, message: str, m: RouterMetrics,
    ) -> AsyncGenerator[str, None]:
        """
        End the stream with an EXPLICIT error event (§9.3 step 3, third bullet).

        Two deliberate choices in five lines:

        * `event: error` plus an error payload, so any client is told what
          happened rather than left to infer it from a stream that just stops.
        * **No `data: [DONE]` and no `finish_reason`.** A terminal chunk here
          would make the response indistinguishable from a successful one at the
          protocol level, and `bench/loadgen.py` would score it `completed` —
          a mid-stream failure counted as a success is precisely the silent
          accounting error this whole path exists to avoid. Without them the
          harness records `incomplete`, which is the truth.
        """
        m.failed += 1
        if record.content_chunks:
            m.failed_midstream += 1
        yield "event: error\n"
        yield _frame({
            "id": req.request_id,
            "object": "error",
            "error": {
                "type": "replica_failure",
                "message": message,
                "replica_id": record.replica_id,
                "attempts": record.attempts,
                "phase": record.phase.value,
                "tokens_already_streamed": record.content_chunks,
                "retryable": record.retryable,
                "note": (
                    "Not retried: tokens had already reached the client, and there is "
                    "no KV replication, so a retry would duplicate or diverge from "
                    "output already delivered (ARCHITECTURE §9.3 step 3)."
                ) if not record.retryable else "Retries exhausted.",
            },
        })

    # -- routes -------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(body: RouterChatRequest):
        app.state.prober.ensure_started()
        metrics.received += 1

        if not app.state.accepting:
            # Router-level drain (§9.3 step 6): nothing new is accepted, and
            # in-flight requests are being allowed to finish.
            metrics.unavailable_503 += 1
            return JSONResponse(
                status_code=503,
                headers={"retry-after": str(cfg.shed_retry_after_s)},
                content={"error": {"type": "draining",
                                   "message": "router is draining; not accepting new requests"}},
            )

        if not replicas.has_capacity():
            # BACKPRESSURE / LOAD SHEDDING. Shedding is an answer, not a failure:
            # unbounded queueing turns a throughput problem into an unbounded
            # latency problem in which EVERY request misses its SLO, and holding
            # goodput flat above the knee is the entire argument for having this
            # (methodology §3). 429 when replicas exist but are full; 503 when
            # none is eligible at all — a router should be told which.
            any_alive = any(
                r.status is ReplicaStatus.HEALTHY for r in replicas.replicas.values()
            )
            replicas.note_shed()
            if any_alive:
                metrics.shed_429 += 1
                return JSONResponse(
                    status_code=429,
                    headers={"retry-after": str(cfg.shed_retry_after_s)},
                    content={"error": {"type": "router_backpressure",
                                       "message": "all eligible replicas are at capacity",
                                       "fleet": replicas.snapshot()}},
                )
            metrics.unavailable_503 += 1
            return JSONResponse(
                status_code=503,
                content={"error": {"type": "no_healthy_replica",
                                   "message": "every replica is quarantined or draining",
                                   "fleet": replicas.snapshot()}},
            )

        request_id = f"router-{uuid.uuid4().hex[:12]}"
        req = _route_request(body, request_id)
        chosen = _select(req)
        if chosen is None:
            replicas.note_shed()
            metrics.unavailable_503 += 1
            return JSONResponse(
                status_code=503,
                content={"error": {"type": "no_healthy_replica",
                                   "message": "routing policy found no eligible replica"}},
            )

        if body.stream:
            return StreamingResponse(
                _stream(body, req, chosen),
                media_type="text/event-stream",
                headers={
                    "cache-control": "no-cache",
                    "x-accel-buffering": "no",
                    "connection": "keep-alive",
                    "x-router-replica": chosen,
                },
            )
        return await _complete(body, req, chosen)

    async def _complete(body: RouterChatRequest, req: RouteRequest, replica_id: str):
        """
        Non-streaming. Retry is unconditionally safe here — nothing is streamed,
        so there is no partially-delivered response to contradict.
        """
        payload = _payload(body)
        record = InFlightRecord(request_id=req.request_id, replica_id=replica_id)
        attempt = 0
        last: str = "no attempt made"
        while True:
            record.replica_id = replica_id
            record.attempts = attempt + 1
            replicas.acquire(replica_id)
            pol.on_assign(req, replica_id, time.monotonic())
            metrics.routed += 1
            ok = False
            try:
                out = await cli.complete_chat(
                    replicas.target(replica_id), payload, cfg.request_timeout_s
                )
                ok = True
                replicas.on_request_success(replica_id)
                metrics.completed += 1
                return JSONResponse(status_code=200, content=out,
                                    headers={"x-router-replica": replica_id})
            except ReplicaUnavailable as exc:
                last = str(exc)
                if exc.quarantine:
                    replicas.on_request_failure(replica_id, last)
            finally:
                replicas.release(replica_id)
                pol.on_complete(req, replica_id, time.monotonic(), ok=ok)

            if not replicas.classify_and_count(record):
                metrics.failed += 1
                return JSONResponse(
                    status_code=503,
                    content={"error": {"type": "replica_failure", "message": last,
                                       "attempts": record.attempts}},
                )
            metrics.retried += 1
            await asyncio.sleep(replicas.retry_delay(attempt))
            attempt += 1
            nxt = _select(req)
            if nxt is None:
                metrics.failed += 1
                return JSONResponse(
                    status_code=503,
                    content={"error": {"type": "no_healthy_replica", "message": last}},
                )
            replica_id = nxt

    @app.get("/health")
    async def health_endpoint():
        app.state.prober.ensure_started()
        ok = app.state.accepting and any(v.eligible for v in replicas.views())
        return JSONResponse(
            status_code=200 if ok else 503,
            content={
                "status": "ok" if ok else "unavailable",
                "accepting": app.state.accepting,
                "policy": pol.stats(),
                "fleet": replicas.snapshot(),
                "spof": (
                    "This router is a single point of failure, deliberately "
                    "(ARCHITECTURE §8.1, R23). The hint-only design makes the fix "
                    "cheap — two routers with divergent hint tables cost cache "
                    "locality, not correctness — but it is out of scope."
                ),
            },
        )

    @app.get("/metrics")
    async def metrics_endpoint():
        return {
            "router": metrics.as_dict(),
            "policy": pol.stats(),
            "fleet": replicas.snapshot(),
            "note": (
                "Router-local counters. Latency is measured CLIENT-side by "
                "bench/loadgen.py from intended dispatch (R1); nothing here is a "
                "latency number and nothing here should become one."
            ),
        }

    # -- admin: fault injection and lifecycle -------------------------------

    @app.post("/admin/replicas/{replica_id}/quarantine")
    async def admin_quarantine(replica_id: str):
        if replica_id not in replicas:
            return JSONResponse(status_code=404, content={"error": "unknown replica"})
        replicas.quarantine(replica_id, "operator/fault-injection request")
        return {"replica_id": replica_id, "status": replicas.get(replica_id).status.value}

    @app.post("/admin/replicas/{replica_id}/drain")
    async def admin_drain_replica(replica_id: str):
        if replica_id not in replicas:
            return JSONResponse(status_code=404, content={"error": "unknown replica"})
        replicas.drain(replica_id, "operator request")
        return {"replica_id": replica_id, "status": replicas.get(replica_id).status.value,
                "inflight": replicas.get(replica_id).inflight}

    @app.post("/admin/replicas/{replica_id}/undrain")
    async def admin_undrain_replica(replica_id: str):
        if replica_id not in replicas:
            return JSONResponse(status_code=404, content={"error": "unknown replica"})
        replicas.undrain(replica_id)
        return {"replica_id": replica_id, "status": replicas.get(replica_id).status.value}

    @app.post("/admin/drain")
    async def admin_drain_router(timeout_s: float = 30.0):
        """
        Graceful drain of the whole router: stop accepting, let in-flight finish.

        This is what makes deploys and benchmark teardown clean (§9.3 step 6) —
        and it is also what keeps a measurement window honest, since a teardown
        that kills in-flight requests injects a burst of failures into the drain
        phase the harness is already excluding.
        """
        app.state.accepting = False
        replicas.drain_all("router drain")
        deadline = time.monotonic() + timeout_s
        while replicas.inflight_total > 0 and time.monotonic() < deadline:
            await asyncio.sleep(cfg.drain_poll_s)
        return {
            "draining": True,
            "inflight_remaining": replicas.inflight_total,
            "complete": replicas.inflight_total == 0,
            "fleet": replicas.snapshot(),
        }

    return app


# ---------------------------------------------------------------------------
# The active health prober
# ---------------------------------------------------------------------------


class _Prober:
    """
    Background probe task. THE SLOW SIGNAL, and the one that catches a hang.

    Lazily started from the request path as well as from the lifespan, for the
    same reason `SchedulerLoop.ensure_started` is: `httpx.ASGITransport` does not
    run the lifespan protocol, so an app tested through it would otherwise never
    probe at all and the tests would be testing a different object from the one
    that ships.
    """

    def __init__(self, pool: ReplicaPool, client: ReplicaClient, config: RouterConfig):
        self.pool = pool
        self.client = client
        self.config = config
        self.rounds = 0
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def ensure_started(self) -> None:
        if not self.config.probe_enabled:
            return
        running = asyncio.get_running_loop()
        if self._task is not None and not self._task.done() and self._loop is running:
            return
        self._loop = running
        self._task = running.create_task(self.run(), name="router-prober")

    async def aclose(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def probe_once(self) -> None:
        """
        One probe round. Concurrent across replicas: probing serially would make
        the detection interval a function of how many replicas are already hung,
        which is backwards.
        """
        pool = self.pool
        ids = list(pool.replicas)
        timeout = pool.config.probe_timeout_s
        results = await asyncio.gather(
            *(self.client.probe(pool.target(i), timeout) for i in ids),
            return_exceptions=True,
        )
        for rid, res in zip(ids, results, strict=True):
            ok = res is True
            pool.on_probe_result(rid, ok, "" if ok else f"probe: {res!r}")
        self.rounds += 1

    async def run(self) -> None:
        while True:
            with contextlib.suppress(Exception):
                await self.probe_once()
            await asyncio.sleep(self.pool.config.probe_interval_s)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.prober.ensure_started()
    try:
        yield
    finally:
        await app.state.prober.aclose()
        client = app.state.client
        if hasattr(client, "aclose"):
            await client.aclose()


# ---------------------------------------------------------------------------
# Production wiring
# ---------------------------------------------------------------------------


def build_default_router(
    urls: Sequence[str],
    policy_name: str = "prefix_aware",
    *,
    blend: float = 0.7,
    saturation_inflight: float | None = 8.0,
    hint_ttl_s: float = 60.0,
    block_size: int = 16,
    n_blocks: int = 4,
    model_id: str = MODEL_ID_DEFAULT,
    health: HealthConfig | None = None,
) -> FastAPI:
    """
    Wire N replica URLs into a router. Nothing but wiring, on purpose.

    `policy_name` is explicit and unknown names raise (`build_policy`) rather
    than falling back to a default: a run whose artifact says `prefix_aware`
    while B5 was actually serving is the worst-shaped error this project can
    make, and it would look like a null result rather than like a bug.
    """
    keyer = PrefixKeyer(block_size=block_size, n_blocks=n_blocks)
    if policy_name == "prefix_aware":
        policy: RoutingPolicy = build_policy(
            "prefix_aware", blend=blend, saturation_inflight=saturation_inflight,
            hint_ttl_s=hint_ttl_s, keyer=keyer,
        )
    else:
        policy = build_policy(policy_name)
    targets = [ReplicaTarget(replica_id=f"r{i}", url=u) for i, u in enumerate(urls)]
    return create_router_app(
        targets,
        policy=policy,
        config=RouterConfig(model_id=model_id, keyer=keyer),
        health=health,
    )


def default_policy() -> RoutingPolicy:
    """B5, the real baseline — the safe thing to fall back to, and never silent."""
    return LeastOutstanding()
