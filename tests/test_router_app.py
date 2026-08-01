"""
Router process tests: MOCK REPLICAS, CPU only, no GPU, no network, no weights.

A "replica" here is a `MockReplica` — a script of SSE frames and a switch that
turns it into a corpse. That is the whole point of R21's mitigation: the router
is pure logic over an injected client seam, so every one of
docs/ARCHITECTURE.md §9.3's six steps is testable in milliseconds on a laptop
and PACE queue time never blocks router work.

The load-bearing test in this file is
`test_partially_streamed_request_fails_explicitly_and_is_never_retried`.
Everything else protects a status code or a counter; that one protects the
single place where the router's hint-only design does NOT save us. A stale hint
costs latency; a transparent retry of a half-delivered stream costs the client a
response that duplicates or diverges from what it already received, with no
error anywhere. Silent, plausible, wrong — the worst shape a bug can have in
this project.

    python3 -m pytest tests/test_router_app.py -q
"""

from __future__ import annotations

import asyncio
import json
import random

import httpx
import pytest

from serving.router.app import (
    ReplicaUnavailable,
    RouterConfig,
    create_router_app,
)
from serving.router.health import (
    HealthConfig,
    InFlightPhase,
    InFlightRecord,
    ReplicaPool,
    ReplicaStatus,
    ReplicaTarget,
)
from serving.router.policy import (
    LeastOutstanding,
    PrefixAware,
    PrefixAwareConfig,
    PrefixKeyer,
)

# ---------------------------------------------------------------------------
# Mock replicas
# ---------------------------------------------------------------------------


class MockReplica:
    """
    A replica that is a script, not a model.

    `die_after_content` is the knob every §9.3 test turns:
      * `0`  -> dies before any visible token. Client has seen nothing -> retryable.
      * `>0` -> dies mid-stream, after that many content chunks have reached the
               client -> NOT retryable, and must fail explicitly.
      * `None` -> never dies.
    """

    def __init__(
        self,
        replica_id: str,
        *,
        tokens: int = 3,
        healthy: bool = True,
        die_after_content: int | None = None,
        status: int | None = None,
        chunk_delay: float = 0.0,
    ):
        self.replica_id = replica_id
        self.tokens = tokens
        self.healthy = healthy
        self.die_after_content = die_after_content
        self.status = status
        self.chunk_delay = chunk_delay
        self.requests = 0
        self.probes = 0


class MockFleet:
    """`ReplicaClient` over a dict of `MockReplica`. No sockets anywhere."""

    def __init__(self, replicas: dict[str, MockReplica]):
        self.replicas = replicas
        self.dispatch_log: list[str] = []

    async def probe(self, target: ReplicaTarget, timeout_s: float) -> bool:
        r = self.replicas[target.replica_id]
        r.probes += 1
        return r.healthy

    async def stream_chat(self, target, payload, timeout_s):
        r = self.replicas[target.replica_id]
        r.requests += 1
        self.dispatch_log.append(r.replica_id)
        if not r.healthy:
            raise ReplicaUnavailable(f"{r.replica_id} is down (mock transport error)")
        if r.status is not None:
            raise ReplicaUnavailable(f"HTTP {r.status}", status=r.status,
                                     quarantine=r.status != 429)
        # Role chunk: NO content. The client has still seen nothing.
        yield _sse({"choices": [{"index": 0, "delta": {"role": "assistant"},
                                 "finish_reason": None}]})
        for i in range(r.tokens):
            if r.die_after_content is not None and i >= r.die_after_content:
                raise ReplicaUnavailable(f"{r.replica_id} died mid-stream")
            if r.chunk_delay:
                await asyncio.sleep(r.chunk_delay)
            yield _sse({"choices": [{"index": 0, "delta": {"content": f"t{i}"},
                                     "finish_reason": None}]})
        yield _sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        yield "data: [DONE]\n\n"

    async def complete_chat(self, target, payload, timeout_s):
        r = self.replicas[target.replica_id]
        r.requests += 1
        self.dispatch_log.append(r.replica_id)
        if not r.healthy:
            raise ReplicaUnavailable(f"{r.replica_id} is down")
        return {"id": "x", "object": "chat.completion", "model": payload["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


class FakeClock:
    """An injectable monotonic clock. A 30s ramp must be testable in 0ms."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------


def make_router(
    n: int = 3,
    *,
    policy=None,
    health: HealthConfig | None = None,
    probe_enabled: bool = False,
    **replica_kwargs,
):
    mocks = {f"r{i}": MockReplica(f"r{i}", **replica_kwargs) for i in range(n)}
    fleet = MockFleet(mocks)
    targets = [ReplicaTarget(f"r{i}", f"http://mock/r{i}") for i in range(n)]
    pol = policy or PrefixAware(PrefixAwareConfig(blend=0.8, saturation_inflight=8))
    hcfg = health or HealthConfig(retry_base_s=0.0, probe_failures_to_quarantine=3)
    app = create_router_app(
        targets, policy=pol, client=fleet,
        config=RouterConfig(probe_enabled=probe_enabled, keyer=PrefixKeyer(block_size=4,
                                                                          n_blocks=2)),
        health=hcfg,
    )
    return app, fleet, mocks, pol


def body(prompt: str = "x" * 200, stream: bool = True) -> dict:
    return {"model": "mock", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8, "stream": stream}


def run(coro):
    return asyncio.run(coro)


async def post(app, path, payload=None):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://router") as c:
        return await c.post(path, json=payload)


async def get(app, path):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://router") as c:
        return await c.get(path)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_streaming_request_is_proxied_verbatim():
    app, fleet, mocks, _ = make_router()

    resp = run(post(app, "/v1/chat/completions", body()))
    assert resp.status_code == 200
    assert resp.headers["x-router-replica"] in mocks
    text = resp.text
    assert text.count('"content"') == 3
    assert text.endswith("data: [DONE]\n\n")
    assert app.state.metrics.completed == 1
    assert app.state.metrics.failed == 0


def test_non_streaming_request_is_proxied():
    app, _, _, _ = make_router()
    resp = run(post(app, "/v1/chat/completions", body(stream=False)))
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hi"
    assert resp.headers["x-router-replica"].startswith("r")


def test_same_prefix_goes_to_the_same_replica_across_requests():
    """
    End-to-end affinity: the hint is written on assignment and honoured on the
    next request with the same block-aligned prompt head, as long as load
    permits. This is the whole feature, observed at the HTTP boundary.
    """
    app, fleet, _, _ = make_router()

    async def scenario():
        picks = []
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://router") as c:
            for _ in range(5):
                r = await c.post("/v1/chat/completions", json=body("shared-system-prompt " * 20))
                picks.append(r.headers["x-router-replica"])
        return picks

    picks = run(scenario())
    assert len(set(picks)) == 1


# ---------------------------------------------------------------------------
# §9.3 — replica failure
# ---------------------------------------------------------------------------


def test_dead_replica_is_detected_quarantined_purged_and_the_request_is_rerouted():
    """
    §9.3 steps 1b, 2, 3 (NOT_STARTED), 4 — the whole eligible-retry path.

    The in-flight error is the FAST detector: no probe has run, and the router
    already knows. The hint purge is asserted directly, because a hint surviving
    its replica's death is not merely stale — it steers the next same-prefix
    request at a corpse, and after a restart at the one replica guaranteed to be
    cold.
    """
    pol = PrefixAware(PrefixAwareConfig(blend=0.9, saturation_inflight=8))
    app, fleet, mocks, _ = make_router(policy=pol)

    # Warm a hint for the prompt, then kill whichever replica owns it.
    first = run(post(app, "/v1/chat/completions", body("cache-me " * 40)))
    victim = first.headers["x-router-replica"]
    assert len(pol.hints) == 1
    mocks[victim].healthy = False

    resp = run(post(app, "/v1/chat/completions", body("cache-me " * 40)))

    assert resp.status_code == 200                       # the client never saw the failure
    assert resp.text.endswith("data: [DONE]\n\n")
    assert resp.text.count('"content"') == 3
    pool = app.state.pool
    assert pool.get(victim).status is ReplicaStatus.QUARANTINED
    assert victim not in pool.eligible_ids()
    assert all(victim not in v for v in pol.hints._entries.values())   # hints purged
    assert app.state.metrics.retried == 1
    assert app.state.metrics.failed == 0


def test_partially_streamed_request_fails_explicitly_and_is_never_retried():
    """
    §9.3 step 3, third bullet — THE honest part of the failure story.

    The replica dies after two visible tokens have already been written to the
    client's socket. There is no KV replication, so a retry re-generates from
    the prompt: the client would receive those tokens twice, or a different
    continuation stitched onto what it already has. Both are silently wrong
    output. So: no retry, an explicit error event, and — deliberately — NO
    `[DONE]` and NO `finish_reason`, because a terminal chunk would make this
    indistinguishable from success at the protocol level and `bench/loadgen.py`
    would score a mid-stream failure as `completed`.
    """
    app, fleet, mocks, _ = make_router(n=3, die_after_content=2, tokens=6)

    resp = run(post(app, "/v1/chat/completions", body()))
    text = resp.text

    assert text.count('"content"') == 2            # exactly what the client received
    assert "event: error" in text
    assert "replica_failure" in text
    assert '"retryable": false' in text
    assert "[DONE]" not in text
    assert '"finish_reason": "stop"' not in text

    m = app.state.metrics
    assert m.failed_midstream == 1
    assert m.retried == 0                          # NOT silently retried
    assert m.completed == 0
    assert app.state.pool.total_failed_midstream == 1
    # Exactly one replica was ever dispatched to: no second attempt happened.
    assert len(fleet.dispatch_log) == 1


def test_a_replica_that_sheds_429_is_retried_elsewhere_but_not_quarantined():
    """
    A 429 is the replica's admission controller doing its job, not a fault.
    Quarantining a correctly-shedding replica would remove capacity at exactly
    the moment capacity is scarce.
    """
    from serving.router.policy import RoundRobin

    app, fleet, mocks, _ = make_router(n=3, policy=RoundRobin())
    mocks["r0"].status = 429
    mocks["r1"].status = 429

    resp = run(post(app, "/v1/chat/completions", body()))
    assert resp.status_code == 200
    assert fleet.dispatch_log == ["r0", "r1", "r2"]      # shed, shed, served
    pool = app.state.pool
    assert all(r.status is ReplicaStatus.HEALTHY for r in pool.replicas.values())
    assert app.state.metrics.retried == 2


def test_active_probes_quarantine_only_after_N_consecutive_failures():
    """
    §9.3 step 1a — the SLOW signal, and the one that catches a hung-but-connected
    replica no in-flight error would ever report. Consecutive, not cumulative: a
    single dropped probe under load must not remove a healthy replica.
    """
    app, fleet, mocks, _ = make_router(n=3, health=HealthConfig(
        probe_failures_to_quarantine=3, probe_successes_to_recover=2, retry_base_s=0.0))
    pool = app.state.pool
    prober = app.state.prober
    mocks["r1"].healthy = False

    run(prober.probe_once())
    run(prober.probe_once())
    assert pool.get("r1").status is ReplicaStatus.HEALTHY      # 2 < 3, still in service
    run(prober.probe_once())
    assert pool.get("r1").status is ReplicaStatus.QUARANTINED

    mocks["r1"].healthy = True
    run(prober.probe_once())
    assert pool.get("r1").status is ReplicaStatus.QUARANTINED  # 1 < 2 successes
    run(prober.probe_once())
    assert pool.get("r1").status is ReplicaStatus.HEALTHY


def test_a_single_dropped_probe_does_not_break_the_consecutive_counter_silently():
    pool = ReplicaPool([ReplicaTarget("r0", "u")], HealthConfig(probe_failures_to_quarantine=3))
    pool.on_probe_result("r0", False)
    pool.on_probe_result("r0", False)
    pool.on_probe_result("r0", True)          # recovered mid-streak: counter resets
    pool.on_probe_result("r0", False)
    pool.on_probe_result("r0", False)
    assert pool.get("r0").status is ReplicaStatus.HEALTHY


# ---------------------------------------------------------------------------
# §9.3 step 5 — ramp-in
# ---------------------------------------------------------------------------


def test_recovered_replica_ramps_in_gradually_rather_than_at_full_weight():
    """
    §9.3 step 5, the non-obvious one.

    A restarted replica passes its health check in seconds and its KV cache is
    EMPTY. "Healthy" and "warm" are different properties and only one of them is
    health-checked, so a cache-aware router that treats it as equal sends it
    traffic it serves slowly. The ramp is asserted on three things: it starts
    well below full weight, it is monotonically non-decreasing, and it takes
    more than one step to arrive.
    """
    clock = FakeClock()
    cfg = HealthConfig(ramp_seconds=30.0, ramp_steps=6, ramp_initial_weight=0.1,
                       probe_successes_to_recover=1)
    pool = ReplicaPool([ReplicaTarget("r0", "u"), ReplicaTarget("r1", "u")],
                       cfg, time_fn=clock)

    assert pool.views()[0].weight == 1.0            # not ramping at startup
    pool.quarantine("r0", "killed")
    pool.on_probe_result("r0", True)
    assert pool.get("r0").status is ReplicaStatus.HEALTHY

    weights = []
    for _ in range(8):
        v = {x.replica_id: x for x in pool.views()}["r0"]
        weights.append(v.weight)
        clock.advance(5.0)

    assert weights[0] == pytest.approx(0.1)
    assert weights == sorted(weights)               # monotonically non-decreasing
    assert len(set(weights)) > 3                    # gradual, not a single jump
    assert weights[-1] == 1.0
    assert pool.views()[0].warm is True          # warm only once the ramp completes


def test_a_cold_replica_receives_less_traffic_than_a_warm_one():
    """The ramp has to change ROUTING, not just a number in a snapshot."""
    clock = FakeClock()
    cfg = HealthConfig(ramp_seconds=30.0, ramp_steps=6, ramp_initial_weight=0.1,
                       probe_successes_to_recover=1)
    pool = ReplicaPool([ReplicaTarget("r0", "u"), ReplicaTarget("r1", "u")],
                       cfg, time_fn=clock)
    pool.quarantine("r1", "killed")
    pool.on_probe_result("r1", True)                # healthy again, but cold

    pol = LeastOutstanding()
    picks = []
    for i in range(12):
        chosen = pol.select(
            type("R", (), {"request_id": str(i), "prefix_key": None, "prompt_tokens": 0})(),
            pool.views(), float(i),
        )
        picks.append(chosen)
        pool.acquire(chosen)

    assert picks.count("r1") < picks.count("r0")
    assert picks.count("r1") >= 1                   # ramping IN, not excluded


# ---------------------------------------------------------------------------
# §9.3 step 4 — jittered retry
# ---------------------------------------------------------------------------


def test_retry_delays_are_jittered_and_seeded():
    """
    Twelve requests orphaned by one replica's death all become retryable at the
    same instant and all compute the same 'least loaded' answer. Without jitter
    they arrive together and turn one replica's failure into a second replica's
    overload — a thundering herd the router inflicted on itself.

    Asserted as SPREAD, not as inequality of two samples: 'delay ± epsilon' would
    pass a naive check and still be a burst.
    """
    cfg = HealthConfig(retry_base_s=0.1, retry_max_s=2.0, retry_jitter=True)
    pool = ReplicaPool([ReplicaTarget("r0", "u")], cfg, rng=random.Random(1234))
    delays = [pool.retry_delay(0) for _ in range(200)]

    assert len(set(delays)) > 150                       # genuinely spread
    assert all(0.0 <= d <= 0.1 for d in delays)         # inside the attempt-0 ceiling
    assert 0.02 < (max(delays) - min(delays))           # not a narrow band
    assert 0.02 < _stdev(delays)

    # Reproducible under the same seed — every artifact records one (methodology §4).
    again = ReplicaPool([ReplicaTarget("r0", "u")], cfg, rng=random.Random(1234))
    assert [again.retry_delay(0) for _ in range(200)] == delays

    # And the exponential ceiling still grows with the attempt number.
    assert max(pool.retry_delay(3) for _ in range(200)) > 0.1


def test_retry_backoff_can_be_made_deterministic_for_a_controlled_experiment():
    cfg = HealthConfig(retry_base_s=0.05, retry_max_s=1.0, retry_jitter=False)
    pool = ReplicaPool([ReplicaTarget("r0", "u")], cfg)
    assert [pool.retry_delay(i) for i in range(4)] == [0.05, 0.1, 0.2, 0.4]


def _stdev(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


# ---------------------------------------------------------------------------
# §9.3 step 6 — drain
# ---------------------------------------------------------------------------


def test_draining_a_replica_stops_new_work_but_lets_in_flight_finish():
    pool = ReplicaPool([ReplicaTarget("r0", "u"), ReplicaTarget("r1", "u")])
    pool.acquire("r0")
    pool.acquire("r0")
    pool.drain("r0", "deploy")

    assert pool.get("r0").status is ReplicaStatus.DRAINING
    assert pool.eligible_ids() == ["r1"]              # nothing new
    assert pool.get("r0").inflight == 2               # in-flight untouched

    pool.release("r0")
    assert pool.get("r0").status is ReplicaStatus.DRAINING
    pool.release("r0")
    assert pool.get("r0").status is ReplicaStatus.DRAINED
    assert pool.fully_drained is False                # r1 is still serving


def test_draining_does_not_purge_hints_because_the_cache_is_still_warm():
    """
    The asymmetry between drain and quarantine, and it is not an oversight: a
    drained replica's cache is intact and correct. Purging its hints would
    throw away a warm cache on a replica that may be re-admitted a minute later
    when a deploy rolls back.
    """
    pol = PrefixAware()
    pool = ReplicaPool([ReplicaTarget("r0", "u"), ReplicaTarget("r1", "u")],
                       on_quarantine=pol.purge_replica)
    pol.hints.put("k", "r0", 0.0)

    pool.drain("r0")
    assert pol.hints.get("k", 1.0) == ["r0"]
    pool.quarantine("r0", "crash")
    assert pol.hints.get("k", 1.0) == []


def test_router_drain_completes_in_flight_and_accepts_nothing_new():
    """§9.3 step 6 end to end: what makes deploys and benchmark teardown clean."""
    app, fleet, mocks, _ = make_router(n=2, tokens=4, chunk_delay=0.02)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://router") as c:
            inflight = asyncio.create_task(c.post("/v1/chat/completions", json=body()))
            await asyncio.sleep(0.03)                 # let it start streaming
            drain = asyncio.create_task(c.post("/admin/drain"))
            await asyncio.sleep(0.005)
            rejected = await c.post("/v1/chat/completions", json=body())
            return await inflight, await drain, rejected

    served, drained, rejected = run(scenario())

    assert served.status_code == 200
    assert served.text.count('"content"') == 4        # in-flight request FINISHED
    assert rejected.status_code == 503                # and nothing new was accepted
    assert rejected.json()["error"]["type"] == "draining"
    assert drained.json()["complete"] is True
    assert drained.json()["inflight_remaining"] == 0


# ---------------------------------------------------------------------------
# Backpressure and load shedding
# ---------------------------------------------------------------------------


def test_backpressure_sheds_429_while_replicas_are_alive_but_full():
    """
    Shedding is an ANSWER, not a failure (methodology §3): unbounded queueing
    turns a throughput problem into an unbounded latency problem in which every
    request misses its SLO. 429 rather than 503 because the replicas are healthy
    and the caller should retry, not fail over.
    """
    app, _, _, _ = make_router(n=2, health=HealthConfig(max_inflight_total=0))
    resp = run(post(app, "/v1/chat/completions", body()))
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "1"
    assert app.state.metrics.shed_429 == 1
    assert app.state.pool.total_shed == 1


def test_503_when_every_replica_is_quarantined():
    app, _, mocks, _ = make_router(n=2)
    for rid in ("r0", "r1"):
        app.state.pool.quarantine(rid, "test")
    resp = run(post(app, "/v1/chat/completions", body()))
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "no_healthy_replica"
    assert app.state.metrics.unavailable_503 == 1


def test_per_replica_inflight_cap_makes_a_replica_ineligible_not_dead():
    pool = ReplicaPool([ReplicaTarget("r0", "u"), ReplicaTarget("r1", "u")],
                       HealthConfig(max_inflight_per_replica=1))
    pool.acquire("r0")
    assert pool.eligible_ids() == ["r1"]
    assert pool.get("r0").status is ReplicaStatus.HEALTHY     # full, not broken


# ---------------------------------------------------------------------------
# In-flight classification, as a unit
# ---------------------------------------------------------------------------


def test_in_flight_phases_classify_retry_eligibility():
    """§9.3 step 3's three-way split, isolated from any transport."""
    rec = InFlightRecord("q", "r0")
    assert rec.phase is InFlightPhase.NOT_STARTED and rec.retryable

    rec.note_response_started()
    assert rec.phase is InFlightPhase.STARTED_NO_TOKENS and rec.retryable

    rec.note_content(0)                       # a role chunk carries no content
    assert rec.phase is InFlightPhase.STARTED_NO_TOKENS and rec.retryable

    rec.note_content(1)
    assert rec.phase is InFlightPhase.PARTIALLY_STREAMED
    assert not rec.retryable                  # the point of no return


def test_retry_budget_is_finite_and_midstream_is_counted_separately():
    pool = ReplicaPool([ReplicaTarget("r0", "u")], HealthConfig(max_retries=2))
    ok = InFlightRecord("a", "r0", attempts=1)
    assert pool.classify_and_count(ok) is True
    exhausted = InFlightRecord("b", "r0", attempts=3)
    assert pool.classify_and_count(exhausted) is False
    assert pool.total_failed_midstream == 0    # exhausted != mid-stream

    streamed = InFlightRecord("c", "r0", phase=InFlightPhase.PARTIALLY_STREAMED, attempts=1)
    assert pool.classify_and_count(streamed) is False
    assert pool.total_failed_midstream == 1


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------


def test_health_reports_the_fleet_and_names_the_spof():
    app, _, _, _ = make_router(n=2)
    resp = run(get(app, "/health"))
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert len(payload["fleet"]["replicas"]) == 2
    # R23 / §8.1: stated, not hidden.
    assert "single point of failure" in payload["spof"]

    for rid in ("r0", "r1"):
        app.state.pool.quarantine(rid, "test")
    assert run(get(app, "/health")).status_code == 503


def test_metrics_expose_full_request_accounting():
    app, _, mocks, _ = make_router(n=2)
    run(post(app, "/v1/chat/completions", body()))
    payload = run(get(app, "/metrics")).json()

    router = payload["router"]
    for key in ("requests_received", "requests_routed", "requests_completed",
                "requests_retried", "requests_shed_429", "requests_failed",
                "requests_failed_midstream"):
        assert key in router
    assert payload["policy"]["policy"] == "prefix_aware"
    assert "failed_midstream_note" in payload["fleet"]
    # Nothing here is a latency number, and the payload says so (R1/R16).
    assert "loadgen" in payload["note"]


def test_admin_endpoints_drive_fault_injection():
    app, _, _, _ = make_router(n=2)
    assert run(post(app, "/admin/replicas/r0/quarantine")).json()["status"] == "quarantined"
    assert run(post(app, "/admin/replicas/nope/drain")).status_code == 404
    assert run(post(app, "/admin/replicas/r1/drain")).json()["status"] == "drained"
    assert run(post(app, "/admin/replicas/r1/undrain")).json()["status"] == "healthy"


def test_router_stream_is_parseable_by_the_real_loadgen_client():
    """
    Feed the ROUTER's actual SSE output to `bench.loadgen.stream_one` — the real
    parser, imported, not reimplemented. If the harness cannot read the router,
    every routing number this project publishes is zero.
    """
    import time as _time

    from bench.loadgen import LoadGenConfig, Outcome, Phase, RequestSpec, stream_one

    app, _, _, _ = make_router(n=2, tokens=5)
    cfg = LoadGenConfig(url="http://router/v1/chat/completions", request_timeout_s=30)
    spec = RequestSpec(request_id=0, intended_send_time=0.0, prompt="hello " * 60,
                       max_tokens=5, phase=Phase.STEADY)

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://router") as c:
            return await stream_one(c, spec, cfg, t0=_time.perf_counter())

    res = run(go())
    assert res.outcome == Outcome.COMPLETED, res.error
    assert res.output_tokens == 5
    assert res.finish_reason == "stop"


def test_a_midstream_failure_is_never_scored_as_completed_by_the_harness():
    """
    The accounting consequence of `_terminate_with_error`, asserted against the
    real scorer rather than against a comment.

    A terminal chunk plus `[DONE]` on the error path would make a mid-stream
    failure indistinguishable from a success at the protocol level, and the
    harness would count it toward goodput — a failure silently improving the
    headline metric. It must land in a non-completed outcome, which is the
    truth about what the client received.
    """
    import time as _time

    from bench.loadgen import LoadGenConfig, Outcome, Phase, RequestSpec, stream_one

    app, _, _, _ = make_router(n=2, tokens=6, die_after_content=2)
    cfg = LoadGenConfig(url="http://router/v1/chat/completions", request_timeout_s=30)
    spec = RequestSpec(request_id=0, intended_send_time=0.0, prompt="hello " * 60,
                       max_tokens=6, phase=Phase.STEADY)

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://router") as c:
            return await stream_one(c, spec, cfg, t0=_time.perf_counter())

    res = run(go())
    assert res.outcome != Outcome.COMPLETED
    assert res.outcome == Outcome.INCOMPLETE
    assert res.output_tokens == 2          # exactly what was delivered, counted honestly


def test_build_default_router_wires_urls_and_refuses_unknown_policies():
    from serving.router.app import build_default_router

    app = build_default_router(["http://a:8000", "http://b:8000"], "least_outstanding")
    assert [t.replica_id for t in
            [r.target for r in app.state.pool.replicas.values()]] == ["r0", "r1"]
    assert app.state.policy.name == "least_outstanding"

    blended = build_default_router(["http://a:8000"], "prefix_aware", blend=0.4)
    assert blended.state.policy.config.blend == 0.4

    with pytest.raises(KeyError):
        build_default_router(["http://a:8000"], "prefix-aware")


def test_events_are_recorded_for_every_transition():
    """S7 is 'fault injection with FULL request accounting' — this is the trail."""
    app, _, mocks, _ = make_router(n=2)
    pool = app.state.pool
    pool.quarantine("r0", "injected")
    pool.on_probe_result("r0", True)
    pool.on_probe_result("r0", True)
    kinds = [e.kind for e in pool.events]
    assert kinds == ["quarantined", "recovered"]
    assert pool.snapshot()["events"][0]["detail"] == "injected"
