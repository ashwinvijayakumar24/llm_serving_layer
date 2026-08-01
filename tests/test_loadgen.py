"""
Open-loop load generator tests. Pure CPU, no server, no GPU, runs in CI.

The harness is tested against an in-process fake SSE transport
(`httpx.AsyncBaseTransport`), so every timing path — arrival schedule, dispatch,
SSE parse, phase assignment, analysis — is exercised end to end with nothing but
asyncio and a scripted byte stream.

The load-bearing test in this file is
`test_coordinated_omission_latency_is_measured_from_intended_dispatch` and its
end-to-end companion. Everything else here protects a number; that one protects
whether ANY number in this project means anything (docs/RISK_REGISTER.md R1).

    python3 -m pytest tests/test_loadgen.py -q
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import time

import httpx
import pytest

from bench.loadgen import (
    ArrivalProcess,
    LoadGenConfig,
    LoadGenRun,
    Outcome,
    Phase,
    RequestResult,
    RequestSpec,
    analyze,
    arrival_schedule,
    build_schedule,
    meets_slo,
    percentile,
    run_load,
    stationarity,
    stream_one,
    validate_run,
)

# ---------------------------------------------------------------------------
# Fake SSE transport
# ---------------------------------------------------------------------------


def sse(content: str | None = None, finish: str | None = None) -> bytes:
    """One OpenAI-style SSE chunk. `content=None` means an EMPTY-content chunk."""
    delta = {} if content is None else {"content": content}
    obj = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(obj)}\n\n".encode()


DONE = b"data: [DONE]\n\n"


class ScriptedSSE(httpx.AsyncBaseTransport):
    """
    Replays a scripted list of (delay_seconds, bytes) per request.

    `script` may be a constant list or a callable taking the parsed request body,
    so a test can make the response depend on the request (e.g. max_tokens) or on
    how many requests are already in flight (a queueing model).
    """

    def __init__(self, script, status: int = 200) -> None:
        self._script = script
        self._status = status
        self.inflight = 0
        self.peak_inflight = 0
        self.send_times: list[float] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        body = json.loads(request.content)
        self.send_times.append(time.perf_counter())
        script = self._script(body) if callable(self._script) else list(self._script)
        transport = self

        async def gen():
            transport.inflight += 1
            transport.peak_inflight = max(transport.peak_inflight, transport.inflight)
            try:
                for delay, payload in script:
                    if delay:
                        await asyncio.sleep(delay)
                    yield payload
            finally:
                transport.inflight -= 1

        return httpx.Response(
            self._status,
            headers={"content-type": "text/event-stream"},
            content=gen(),
        )


def client_for(transport: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport)


def cfg_for(**kw) -> LoadGenConfig:
    base = dict(
        url="http://fake/v1/chat/completions",
        rate_rps=20.0,
        duration_s=0.4,
        warmup_s=0.2,
        drain_s=0.2,
        seed=7,
        prompt_mean_tokens=4,
        output_mean_tokens=4,
        inflight_sample_interval_s=0.01,
        max_dispatch_drift_ms=50.0,
    )
    base.update(kw)
    return LoadGenConfig(**base)


def spec_at(t: float, rid: int = 0, phase: str = Phase.STEADY) -> RequestSpec:
    return RequestSpec(
        request_id=rid, intended_send_time=t, prompt="p", max_tokens=8, phase=phase
    )


# ---------------------------------------------------------------------------
# 1. Arrival process
# ---------------------------------------------------------------------------


def test_poisson_mean_inter_arrival_matches_one_over_rate():
    """Mean gap -> 1/lambda, and the CV of an exponential is 1 (a Poisson process,
    not a jittered periodic one, which would have CV << 1)."""
    rate = 50.0
    times = arrival_schedule(rate, horizon_s=400.0, process=ArrivalProcess.POISSON, seed=1)
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert len(gaps) > 15_000
    mean = statistics.fmean(gaps)
    assert mean == pytest.approx(1.0 / rate, rel=0.05)
    cv = statistics.pstdev(gaps) / mean
    assert cv == pytest.approx(1.0, abs=0.08)


def test_poisson_schedule_is_seeded_and_reproducible():
    a = arrival_schedule(10.0, 50.0, ArrivalProcess.POISSON, seed=42)
    b = arrival_schedule(10.0, 50.0, ArrivalProcess.POISSON, seed=42)
    c = arrival_schedule(10.0, 50.0, ArrivalProcess.POISSON, seed=43)
    assert a == b
    assert a != c
    assert all(x < y for x, y in zip(a, a[1:], strict=False))  # monotone


def test_deterministic_schedule_is_exactly_periodic():
    times = arrival_schedule(8.0, 4.0, ArrivalProcess.DETERMINISTIC, seed=99)
    assert len(times) == 32
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert all(g == pytest.approx(0.125, abs=1e-12) for g in gaps)
    assert statistics.pstdev(gaps) == pytest.approx(0.0, abs=1e-12)


def test_poisson_and_deterministic_offer_the_same_mean_load():
    """The comparison that isolates BURSTINESS from LOAD: same rate, same count,
    different clustering. Any measured gap between them is a scheduler result."""
    det = arrival_schedule(20.0, 100.0, ArrivalProcess.DETERMINISTIC, seed=3)
    poi = arrival_schedule(20.0, 100.0, ArrivalProcess.POISSON, seed=3)
    assert len(poi) == pytest.approx(len(det), rel=0.05)
    poi_gaps = [b - a for a, b in zip(poi, poi[1:], strict=False)]
    assert max(poi_gaps) > 5 * 0.05        # genuine bursts and genuine idle gaps
    assert min(poi_gaps) < 0.05 / 5


def test_arrival_schedule_rejects_bad_parameters():
    with pytest.raises(ValueError):
        arrival_schedule(0.0, 10.0)
    with pytest.raises(ValueError):
        arrival_schedule(1.0, 0.0)
    with pytest.raises(ValueError):
        arrival_schedule(1.0, 10.0, process="round-robin")


# ---------------------------------------------------------------------------
# 2. COORDINATED OMISSION — R1. The most important tests in this file.
# ---------------------------------------------------------------------------


def test_coordinated_omission_latency_is_measured_from_intended_dispatch():
    """
    A request that was SUPPOSED to be sent at t=0 but is actually sent at t=0.30
    must report the 0.30s of lateness inside its latency.

    Construction: the run's clock origin is now, the fake server takes ~40ms to
    first token, and the request's intended dispatch time is 300ms in the PAST.
    That is exactly what a stalled generator produces under saturation.

    The wrong (coordinated-omission) answer is ~40ms — the service time alone,
    with the queueing the client itself experienced silently deleted. The right
    answer is ~340ms. This test asserts the reported TTFT is the second one and
    is nowhere near the first, so an implementation that timed from
    `actual_send_time` fails here rather than silently publishing a fast p99.
    """
    lateness = 0.30
    service = 0.04
    script = [(service, sse("hi")), (0.01, sse("there", finish="stop")), (0.0, DONE)]
    transport = ScriptedSSE(script)
    cfg = cfg_for()

    async def go():
        async with client_for(transport) as client:
            t0 = time.perf_counter() - lateness      # origin 300ms in the past
            spec = spec_at(t=0.0)                    # ... but it was due at t=0
            return await stream_one(client, spec, cfg, t0)

    res = asyncio.run(go())

    assert res.outcome == Outcome.COMPLETED
    assert res.ttft_ms == pytest.approx((lateness + service) * 1e3, rel=0.25)
    assert res.ttft_ms > 250.0, "lateness was dropped: this is coordinated omission"

    # The number a naive harness would have published, computed here explicitly so
    # the failure mode is visible rather than described.
    naive_ttft_ms = (res.token_times[0] - res.actual_send_time) * 1e3
    assert naive_ttft_ms < 100.0
    assert res.ttft_ms > naive_ttft_ms + 200.0

    # E2E carries it too, and the drift metric names the harness as the culprit.
    assert res.e2e_ms > res.ttft_ms
    assert res.e2e_ms > 250.0
    assert res.dispatch_drift_ms == pytest.approx(lateness * 1e3, rel=0.2)

    # ITL is a server-side gap and must NOT absorb the harness lateness — it is
    # measured between token arrivals, not from dispatch.
    assert res.itls_ms[0] == pytest.approx(10.0, abs=25.0)


def test_coordinated_omission_visible_end_to_end_when_harness_throttles():
    """
    The same guard, through the whole runner.

    `max_concurrency=1` against a slow server is a deliberately sabotaged harness
    — it is a CLOSED loop with one client. Requests queue up inside the load
    generator, get sent far too late, and a harness that timed from actual send
    would report a beautiful ~30ms TTFT for every one of them.

    Assertions: reported latency grows with position in the run (the queue is
    visible), drift is large, and the run is marked INVALID for both reasons.
    """
    script = [(0.03, sse("a")), (0.0, sse("b", finish="stop")), (0.0, DONE)]
    transport = ScriptedSSE(script)
    cfg = cfg_for(
        rate_rps=60.0, warmup_s=0.0, duration_s=0.5, drain_s=0.0,
        process=ArrivalProcess.DETERMINISTIC, max_concurrency=1,
    )

    async def go():
        async with client_for(transport) as client:
            return await run_load(cfg, client=client)

    run = asyncio.run(go())
    steady = [r for r in run.results if r.spec.phase == Phase.STEADY]
    assert len(steady) > 10

    ttfts = [r.ttft_ms for r in steady if r.ttft_ms is not None]
    assert ttfts[-1] > ttfts[0] + 100.0, "queueing inside the harness was invisible"
    assert max(r.dispatch_drift_ms for r in steady) > 100.0
    assert any(r.concurrency_capped for r in steady)

    validity = validate_run(run.results, cfg)
    assert not validity.valid
    joined = " ".join(validity.reasons)
    assert "COORDINATED OMISSION" in joined
    assert "CLOSED-loop" in joined


def test_dispatch_drift_within_threshold_leaves_the_run_valid():
    results = [
        RequestResult(spec=spec_at(0.1 * i, i), actual_send_time=0.1 * i + 0.001)
        for i in range(50)
    ]
    cfg = cfg_for(max_dispatch_drift_ms=50.0)
    v = validate_run(results, cfg)
    assert v.valid, v.reasons
    assert v.drift["n"] == 50
    assert v.drift["p99"] == pytest.approx(1.0, abs=0.5)


def test_dispatch_drift_above_threshold_marks_the_run_invalid():
    good = [
        RequestResult(spec=spec_at(0.1 * i, i), actual_send_time=0.1 * i + 0.001)
        for i in range(90)
    ]
    late = [
        RequestResult(spec=spec_at(0.1 * i, i), actual_send_time=0.1 * i + 0.5)
        for i in range(90, 100)
    ]
    cfg = cfg_for(max_dispatch_drift_ms=50.0)
    v = validate_run(good + late, cfg)
    assert not v.valid
    assert "COORDINATED OMISSION" in v.reasons[0]
    assert v.drift["threshold_ms"] == 50.0
    assert v.drift["max"] == pytest.approx(500.0, rel=0.01)


def test_invalidity_reaches_the_artifact():
    cfg = cfg_for(max_dispatch_drift_ms=1.0, duration_s=1.0, warmup_s=0.0, drain_s=0.0)
    results = [
        _synthetic_result(rid=i, intended=0.01 * i, actual=0.01 * i + 0.2, tokens=5)
        for i in range(20)
    ]
    run = LoadGenRun(
        cfg=cfg, schedule=[r.spec for r in results], results=results,
        inflight_samples=[(t / 20.0, 3.0) for t in range(20)],
        wall_seconds=1.0, started_utc="2026-07-31T00:00:00",
    )
    a = analyze(run)
    assert not a.validity.valid
    assert a.artifact.notes[0].startswith("RUN INVALID")
    assert a.artifact.realized_workload["run_validity"]["valid"] is False
    assert "dispatch_drift_ms" in a.artifact.samples


# ---------------------------------------------------------------------------
# 3. Open loop: dispatch does not wait on responses
# ---------------------------------------------------------------------------


def test_dispatch_continues_while_the_server_is_stalled():
    """
    A closed-loop harness cannot have more than `n_clients` requests outstanding.
    Here the server holds EVERY response for most of the run, so if dispatch were
    gated on responses only one request would ever be sent.
    """
    script = [(0.35, sse("a", finish="stop")), (0.0, DONE)]
    transport = ScriptedSSE(script)
    cfg = cfg_for(
        rate_rps=100.0, warmup_s=0.0, duration_s=0.3, drain_s=0.0,
        process=ArrivalProcess.DETERMINISTIC, max_concurrency=0,
    )

    async def go():
        async with client_for(transport) as client:
            return await run_load(cfg, client=client)

    run = asyncio.run(go())
    assert len(run.results) == 30
    assert transport.peak_inflight > 20, "concurrency was bounded — this is a closed loop"
    assert max(r.dispatch_drift_ms for r in run.results) < 60.0
    assert max(v for _, v in run.inflight_samples) > 20


# ---------------------------------------------------------------------------
# 4. Measurement window — R11
# ---------------------------------------------------------------------------


def test_warmup_and_drain_are_issued_but_excluded():
    script = [(0.001, sse("a")), (0.001, sse("b", finish="stop")), (0.0, DONE)]
    transport = ScriptedSSE(script)
    cfg = cfg_for(
        rate_rps=50.0, warmup_s=0.2, duration_s=0.3, drain_s=0.2,
        process=ArrivalProcess.DETERMINISTIC,
    )

    async def go():
        async with client_for(transport) as client:
            return await run_load(cfg, client=client)

    run = asyncio.run(go())
    phases = {p: sum(1 for r in run.results if r.spec.phase == p) for p in
              (Phase.WARMUP, Phase.STEADY, Phase.DRAIN)}
    assert phases[Phase.WARMUP] == 10
    assert phases[Phase.STEADY] == 15
    assert phases[Phase.DRAIN] == 10
    # Drain requests were actually SENT: the steady window is surrounded by load,
    # not measured against an emptying queue.
    assert len(transport.send_times) == 35

    a = analyze(run)
    art = a.artifact
    assert art.window["steady_start_s"] == 0.2
    assert art.window["steady_end_s"] == 0.5
    assert art.window["requests_steady"] == 15
    assert art.window["requests_warmup"] == 10
    assert art.window["requests_drain"] == 10
    # Only steady samples were pooled.
    assert len(art.samples["ttft_ms"]) == 15
    assert len(art.samples["e2e_ms"]) == 15


def test_build_schedule_assigns_phases_by_intended_time():
    cfg = cfg_for(rate_rps=10.0, warmup_s=1.0, duration_s=2.0, drain_s=1.0,
                  process=ArrivalProcess.DETERMINISTIC)
    specs = build_schedule(cfg)
    for s in specs:
        expected = (
            Phase.WARMUP if s.intended_send_time < 1.0
            else Phase.STEADY if s.intended_send_time < 3.0
            else Phase.DRAIN
        )
        assert s.phase == expected


def test_stationarity_holds_when_flat_and_fails_when_trending():
    flat = [(i * 0.1, 8.0 + (i % 2)) for i in range(60)]
    held = stationarity(flat, 0.0, 6.0, tolerance=0.25)
    assert held["held"] is True
    assert held["label"] == "steady state"

    trending = [(i * 0.1, 1.0 + i) for i in range(60)]
    trend = stationarity(trending, 0.0, 6.0, tolerance=0.25)
    assert trend["held"] is False
    assert trend["label"] == "unsaturated-window measurement"
    assert trend["ols_slope_per_s"] > 0
    assert trend["trend_material"] and trend["trend_significant"]

    # A queue that fluctuates without going anywhere must NOT be flagged: a
    # stationarity check that cries wolf is a check nobody acts on.
    rng = random.Random(11)
    noisy = [(i * 0.1, 5.0 + rng.gauss(0, 1.5)) for i in range(200)]
    assert stationarity(noisy, 0.0, 20.0, tolerance=0.25)["held"] is True

    # A trend small enough to be noise-sized in slope but real in magnitude is
    # still caught by the half-vs-half check, which has no significance escape.
    creeping = [(i * 0.1, 4.0 + 0.06 * i) for i in range(60)]
    assert stationarity(creeping, 0.0, 6.0, tolerance=0.25)["held"] is False

    sparse = stationarity([(0.0, 1.0), (0.1, 1.0)], 0.0, 6.0)
    assert sparse["held"] is False
    assert "unverifiable" in sparse["reason"]


def test_trending_window_is_labeled_not_silently_published():
    cfg = cfg_for(duration_s=1.0, warmup_s=0.0, drain_s=0.0)
    results = [_synthetic_result(i, 0.05 * i, 0.05 * i, tokens=4) for i in range(20)]
    run = LoadGenRun(
        cfg=cfg, schedule=[r.spec for r in results], results=results,
        inflight_samples=[(i * 0.05, float(i)) for i in range(20)],
        wall_seconds=1.0, started_utc="x",
    )
    a = analyze(run)
    assert a.stationarity["held"] is False
    assert a.artifact.realized_workload["window_label"] == "unsaturated-window measurement"
    assert not a.validity.valid


# ---------------------------------------------------------------------------
# 5. Metric definitions — §2
# ---------------------------------------------------------------------------


def test_ttft_ignores_empty_content_chunks():
    """
    The engine's server emits a chunk per token id and detokenizes with
    skip_special_tokens=True, so leading special tokens arrive as chunks whose
    content is "". Three of them here, 30ms of them, before any visible text.
    Timing to the first CHUNK would report ~10ms; the definition says ~40ms.
    """
    script = [
        (0.01, sse(None)),           # empty-content chunk: NOT a first token
        (0.01, sse(None)),
        (0.01, sse("")),             # explicit empty string, same thing
        (0.01, sse("Hello")),        # <- the actual first token, ~40ms in
        (0.01, sse(" world", finish="stop")),
        (0.0, DONE),
    ]
    cfg = cfg_for()

    async def go():
        async with client_for(ScriptedSSE(script)) as client:
            t0 = time.perf_counter()
            return await stream_one(client, spec_at(0.0), cfg, t0)

    res = asyncio.run(go())
    assert res.outcome == Outcome.COMPLETED
    assert res.empty_chunks == 3
    assert res.output_tokens == 2
    assert res.ttft_ms == pytest.approx(40.0, abs=20.0)
    assert res.ttft_ms > 25.0, "TTFT was timed to the first chunk, not first content"
    assert len(res.chunk_times) == 5   # every data event is still counted


def test_itl_series_matches_a_known_arrival_pattern():
    script = [
        (0.02, sse("a")),            # first token at ~20ms
        (0.05, sse("b")),            # ITL 1: 50ms
        (0.01, sse(None)),           # empty chunk between tokens: NOT an ITL boundary
        (0.02, sse("c")),            # ITL 2: 30ms (10 + 20), spanning the empty chunk
        (0.08, sse("d", finish="stop")),   # ITL 3: 80ms
        (0.0, DONE),
    ]
    cfg = cfg_for()

    async def go():
        async with client_for(ScriptedSSE(script)) as client:
            return await stream_one(client, spec_at(0.0), cfg, time.perf_counter())

    res = asyncio.run(go())
    itls = res.itls_ms
    assert len(itls) == 3
    assert itls[0] == pytest.approx(50.0, abs=25.0)
    assert itls[1] == pytest.approx(30.0, abs=25.0)
    assert itls[2] == pytest.approx(80.0, abs=25.0)
    assert res.tpot_ms == pytest.approx(sum(itls) / 3, rel=0.01)
    assert res.output_tokens == 4


def test_e2e_runs_to_the_final_chunk():
    script = [(0.02, sse("a")), (0.05, sse("b", finish="stop")), (0.0, DONE)]
    cfg = cfg_for()

    async def go():
        async with client_for(ScriptedSSE(script)) as client:
            return await stream_one(client, spec_at(0.0), cfg, time.perf_counter())

    res = asyncio.run(go())
    assert res.finish_reason == "stop"
    assert res.e2e_ms == pytest.approx(70.0, abs=30.0)
    assert res.e2e_ms > res.ttft_ms


def test_percentile_is_linear_interpolation_between_order_statistics():
    xs = [float(i) for i in range(1, 101)]
    assert percentile(xs, 50) == pytest.approx(50.5)
    assert percentile(xs, 99) == pytest.approx(99.01, abs=1e-6)
    assert percentile([5.0], 99) == 5.0
    with pytest.raises(ValueError):
        percentile([], 50)


# ---------------------------------------------------------------------------
# 6. Failure outcomes are results, not missing data
# ---------------------------------------------------------------------------


def test_http_errors_are_counted_not_dropped():
    transport = ScriptedSSE([(0.0, b"nope")], status=503)
    cfg = cfg_for()

    async def go():
        async with client_for(transport) as client:
            return await stream_one(client, spec_at(0.0), cfg, time.perf_counter())

    res = asyncio.run(go())
    assert res.outcome == Outcome.ERROR
    assert res.status_code == 503
    assert "503" in res.error
    ok, why = meets_slo(res, cfg)
    assert not ok and "outcome=error" in why


def test_timeouts_are_counted_not_dropped():
    class Timeouter(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ReadTimeout("too slow", request=request)

    cfg = cfg_for(request_timeout_s=0.05)

    async def go():
        async with client_for(Timeouter()) as client:
            return await stream_one(client, spec_at(0.0), cfg, time.perf_counter())

    res = asyncio.run(go())
    assert res.outcome == Outcome.TIMEOUT
    assert "ReadTimeout" in res.error


def test_truncated_stream_is_incomplete_and_malformed_payload_is_an_error():
    async def go(script):
        async with client_for(ScriptedSSE(script)) as client:
            return await stream_one(client, spec_at(0.0), cfg_for(), time.perf_counter())

    truncated = asyncio.run(go([(0.0, sse("a"))]))      # no finish_reason, no [DONE]
    assert truncated.outcome == Outcome.INCOMPLETE

    garbage = asyncio.run(go([(0.0, b"data: {not json}\n\n")]))
    assert garbage.outcome == Outcome.ERROR
    assert "unparseable" in garbage.error

    empty_only = asyncio.run(go([(0.0, sse(None, finish="stop")), (0.0, DONE)]))
    assert empty_only.outcome == Outcome.NO_CONTENT
    assert empty_only.ttft_ms is None


def test_failed_requests_appear_in_the_artifact_counts():
    cfg = cfg_for(rate_rps=40.0, warmup_s=0.0, duration_s=0.25, drain_s=0.0,
                  process=ArrivalProcess.DETERMINISTIC)
    transport = ScriptedSSE([(0.0, b"boom")], status=500)

    async def go():
        async with client_for(transport) as client:
            return await run_load(cfg, client=client)

    a = analyze(asyncio.run(go()))
    outcomes = a.artifact.realized_workload["outcomes_steady"]
    assert outcomes[Outcome.ERROR] == 10
    assert outcomes[Outcome.COMPLETED] == 0
    assert a.artifact.scalars["goodput_rps"] == 0.0
    assert a.artifact.scalars["slo_attainment"] == 0.0
    assert a.artifact.realized_workload["slo_miss_reasons_steady"] == {"outcome=error": 10}


# ---------------------------------------------------------------------------
# 7. Goodput under SLO — §3
# ---------------------------------------------------------------------------


def _synthetic_result(
    rid: int,
    intended: float,
    actual: float,
    tokens: int = 5,
    ttft_s: float = 0.05,
    itl_s: float = 0.01,
    outcome: str = Outcome.COMPLETED,
    phase: str = Phase.STEADY,
) -> RequestResult:
    """A RequestResult with hand-chosen timings, built the way the client would."""
    first = intended + ttft_s
    token_times = [first + i * itl_s for i in range(tokens)]
    return RequestResult(
        spec=RequestSpec(rid, intended, "p", 16, phase),
        actual_send_time=actual,
        outcome=outcome,
        token_times=token_times,
        chunk_times=list(token_times),
        stream_end_time=token_times[-1] if token_times else intended,
        finish_reason="stop" if outcome == Outcome.COMPLETED else None,
    )


def test_goodput_on_a_hand_constructed_set():
    """
    10 requests, one second of steady window, four distinct fates:
      4 meet both clauses
      2 breach TTFT
      2 breach p95 ITL (one slow gap each, at the tail of the series)
      2 errored
    Goodput must be 4 req/s and attainment 0.4 — not 4/8, not 6/10.
    """
    cfg = cfg_for(
        duration_s=1.0, warmup_s=0.0, drain_s=0.0,
        slo_ttft_ms=200.0, slo_itl_ms=50.0, slo_itl_percentile=95.0,
    )
    results = []
    for i in range(4):                       # meet
        results.append(_synthetic_result(i, 0.05 * i, 0.05 * i, 10, 0.05, 0.01))
    for i in range(4, 6):                    # TTFT breach
        results.append(_synthetic_result(i, 0.05 * i, 0.05 * i, 10, 0.35, 0.01))
    for i in range(6, 8):                    # ITL breach
        r = _synthetic_result(i, 0.05 * i, 0.05 * i, 10, 0.05, 0.01)
        r.token_times = r.token_times[:-1] + [r.token_times[-2] + 0.5]
        results.append(r)
    for i in range(8, 10):                   # errors
        results.append(_synthetic_result(i, 0.05 * i, 0.05 * i, 0, outcome=Outcome.ERROR))

    verdicts = [meets_slo(r, cfg) for r in results]
    assert [ok for ok, _ in verdicts] == [True] * 4 + [False] * 6
    assert "ttft" in verdicts[4][1]
    # The per-token clause is judged on TPOT, not p95 ITL: client ITL is bursty
    # and its p95 reports the burst gap rather than generation speed.
    assert "tpot" in verdicts[6][1]
    assert "outcome=error" in verdicts[8][1]

    run = LoadGenRun(
        cfg=cfg, schedule=[r.spec for r in results], results=results,
        inflight_samples=[(i * 0.05, 5.0) for i in range(20)],
        wall_seconds=1.0, started_utc="x",
    )
    art = analyze(run).artifact
    assert art.scalars["goodput_rps"] == pytest.approx(4.0)
    assert art.scalars["completed_rps"] == pytest.approx(8.0)
    assert art.scalars["slo_attainment"] == pytest.approx(0.4)
    assert art.scalars["offered_load_rps"] == cfg.rate_rps
    # Offered load, completion rate and goodput are three different numbers.
    assert art.scalars["offered_load_rps"] != art.scalars["goodput_rps"]


def test_single_token_response_is_judged_without_an_itl_clause():
    cfg = cfg_for(slo_ttft_ms=200.0, slo_itl_ms=1.0)
    r = _synthetic_result(0, 0.0, 0.0, tokens=1, ttft_s=0.05)
    ok, why = meets_slo(r, cfg)
    assert ok and "no per-token rate to judge" in why


def test_slo_thresholds_are_parameters_not_constants():
    r = _synthetic_result(0, 0.0, 0.0, tokens=5, ttft_s=0.10, itl_s=0.02)
    assert meets_slo(r, cfg_for(slo_ttft_ms=200.0, slo_itl_ms=50.0))[0]
    assert not meets_slo(r, cfg_for(slo_ttft_ms=50.0, slo_itl_ms=50.0))[0]
    assert not meets_slo(r, cfg_for(slo_ttft_ms=200.0, slo_itl_ms=10.0))[0]


# ---------------------------------------------------------------------------
# 8. The artifact — raw samples survive (§5, R15) and R14 evidence is present
# ---------------------------------------------------------------------------


def test_raw_samples_survive_into_the_artifact():
    cfg = cfg_for(duration_s=1.0, warmup_s=0.0, drain_s=0.0)
    results = [_synthetic_result(i, 0.02 * i, 0.02 * i, tokens=6) for i in range(25)]
    run = LoadGenRun(
        cfg=cfg, schedule=[r.spec for r in results], results=results,
        inflight_samples=[(i * 0.04, 4.0) for i in range(25)],
        wall_seconds=1.0, started_utc="x",
    )
    a = analyze(run)
    art = a.artifact

    assert len(art.samples["ttft_ms"]) == 25
    assert len(art.samples["e2e_ms"]) == 25
    assert len(art.samples["itl_ms"]) == 25 * 5     # per-TOKEN samples, not per-request
    assert len(art.samples["dispatch_drift_ms"]) == 25
    assert len(art.samples["output_tokens_per_request"]) == 25
    assert art.samples["inflight_requests"]
    assert art.validate() == []

    # Percentiles are DERIVED and recomputable from the samples, never a
    # substitute for them.
    for name in ("ttft_ms", "itl_ms", "e2e_ms"):
        d = art.realized_workload["derived"][name]
        assert d["n"] == len(art.samples[name])
        assert d["p99"] == pytest.approx(percentile(art.samples[name], 99))
        assert d["p99_resolvable"] is False        # 25 samples: says so out loud
        assert "R15" in d["p99_note"]

    blob = json.loads(json.dumps(art.to_dict()))
    assert len(blob["samples"]["itl_ms"]) == 125    # survives serialization
    assert blob["metric_specs"]["ttft_ms"]["unit"] == "ms"


def test_artifact_records_the_realized_workload_not_the_requested_one():
    cfg = cfg_for(rate_rps=40.0, warmup_s=0.0, duration_s=0.25, drain_s=0.0)
    script = [(0.002, sse("a")), (0.002, sse("b", finish="stop")), (0.0, DONE)]

    async def go():
        async with client_for(ScriptedSSE(script)) as client:
            return await run_load(cfg, client=client)

    art = analyze(asyncio.run(go())).artifact
    rw = art.realized_workload
    assert rw["requested"]["rate_rps"] == 40.0
    assert rw["realized_inter_arrival_ms"]["n"] >= 1
    assert rw["realized_output_tokens"]["mean"] == pytest.approx(2.0)
    assert rw["realized_max_tokens"]["n"] >= 1
    assert "stationarity" in rw
    assert rw["realized_rate_rps"] > 0


def test_artifact_writes_to_disk_with_window_and_provenance(tmp_path):
    cfg = cfg_for(duration_s=1.0, warmup_s=0.5, drain_s=0.5, seed=1234)
    results = [_synthetic_result(i, 0.5 + 0.02 * i, 0.5 + 0.02 * i) for i in range(20)]
    run = LoadGenRun(
        cfg=cfg, schedule=[r.spec for r in results], results=results,
        inflight_samples=[(0.5 + i * 0.04, 3.0) for i in range(25)],
        wall_seconds=2.0, started_utc="x",
    )
    path = analyze(run).artifact.write(tmp_path)
    blob = json.loads(path.read_text())
    assert blob["window"]["steady_start_s"] == 0.5
    assert blob["window"]["steady_end_s"] == 1.5
    assert blob["provenance"]["seed"] == 1234
    assert blob["config"]["latency_origin"].startswith("INTENDED")
    assert "publishable" in blob
