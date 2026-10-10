"""
Closed-loop runner for W1: exactly N requests in flight.

N workers share one seeded request list. Each worker takes the next request
off the list, streams it to completion, and immediately takes the next. So the
in-flight count is N from the first dispatch until the list runs out.

WHY A CLOSED LOOP HERE, WHEN bench/loadgen.py ARGUES AGAINST ONE
----------------------------------------------------------------
bench/loadgen.py's docstring is right that a closed loop cannot find a
saturation knee: it never builds a queue deeper than N. W1 does not ask that
question. It asks "how fast is each engine at a FIXED batch occupancy of N",
which is the comparison vLLM's and SGLang's own `--max-concurrency` benchmarks
make, and the only one where the engines' batch sizes are held equal by
construction. The open-loop question is W2's. Every W1 artifact is labelled
`workload.loop = "closed"` so the two are never confused.

COORDINATED OMISSION DOES NOT APPLY — AND WHAT LATENCY MEANS INSTEAD
--------------------------------------------------------------------
Coordinated omission is the error of timing from actual send when the
generator was late relative to a SCHEDULE. A closed loop has no schedule: a
request's "intended" send time is, by definition, the moment its worker became
free. So `intended_send_time` is set to that moment (just before the request
is built), and `bench.loadgen.stream_one` then times TTFT/E2E from it exactly
as in the open loop. The dispatch drift it records is pure harness overhead
(microseconds), and it is reported but does not gate validity.

What a closed-loop latency does NOT contain is queueing in front of the
server: there is never more than N outstanding. That is the definition of the
workload, not an omission.

VALIDITY
--------
The one property this mode promises is N in flight during the measured window.
`inflight_check` verifies it from the sampled in-flight series and the run is
marked invalid if the mean falls below 95% of N — e.g. if the request list was
too short for the drain to cover the last measured requests, or if requests
failed so fast that workers spent the window between requests.

Phases are by position in the list: the first `warmup` requests are issued
and discarded, the next `measured` are the window, and `drain` more keep N in
flight while the last measured requests finish.
"""

from __future__ import annotations

import asyncio
import dataclasses
import statistics
import time
from typing import Any

import httpx

from bench.loadgen import LoadGenConfig, LoadGenRun, Phase, RequestResult, RequestSpec, stream_one


async def run_closed_loop(
    cfg: LoadGenConfig,
    specs: list[RequestSpec],
    concurrency: int,
    client: Any = None,
    sample_interval_s: float | None = None,
    deadline_s: float | None = None,
) -> LoadGenRun:
    """
    Run `specs` with exactly `concurrency` workers. Returns a real
    `LoadGenRun` whose `schedule` holds the specs AS DISPATCHED (with the
    worker-free time as `intended_send_time`), results in list order.

    `deadline_s` bounds the point's wall-clock time. When it passes, workers are
    cancelled and in-flight requests are dropped; the returned run carries
    `deadline_hit = True` so the caller marks it invalid. This exists because a
    server that stops making progress (W4, job 13931404) would otherwise hold a
    closed-loop point until every request times out.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            timeout=cfg.request_timeout_s,
        )
    interval = sample_interval_s or cfg.inflight_sample_interval_s
    inflight = [0]
    inflight_samples: list[tuple[float, float]] = []
    dispatched: list[RequestSpec | None] = [None] * len(specs)
    results: list[RequestResult | None] = [None] * len(specs)
    next_idx = [0]
    stop = asyncio.Event()
    t0 = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    async def sampler() -> None:
        while not stop.is_set():
            inflight_samples.append((time.perf_counter() - t0, float(inflight[0])))
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                continue

    async def worker() -> None:
        # No await between one request finishing (stream_one's _finish
        # decrements `inflight`) and the next one starting (stream_one
        # increments it before its first await), so the count never dips
        # between requests: it is exactly N while the list lasts.
        while True:
            i = next_idx[0]
            if i >= len(specs):
                return
            next_idx[0] = i + 1
            spec = dataclasses.replace(specs[i], intended_send_time=time.perf_counter() - t0)
            dispatched[i] = spec
            results[i] = await stream_one(client, spec, cfg, t0, inflight)

    sampler_task = asyncio.create_task(sampler())
    deadline_hit = False
    try:
        workers = asyncio.gather(*(worker() for _ in range(concurrency)))
        if deadline_s:
            try:
                await asyncio.wait_for(workers, timeout=deadline_s)
            except TimeoutError:
                deadline_hit = True
        else:
            await workers
    finally:
        stop.set()
        await sampler_task
        if owns_client:
            await client.aclose()
    inflight_samples.append((time.perf_counter() - t0, float(inflight[0])))

    run = LoadGenRun(
        cfg=cfg,
        schedule=[
            s for s, r in zip(dispatched, results, strict=True) if s is not None and r is not None
        ],
        results=[r for r in results if r is not None],
        inflight_samples=inflight_samples,
        wall_seconds=time.perf_counter() - t0,
        started_utc=started_utc,
    )
    run.deadline_hit = deadline_hit  # type: ignore[attr-defined]
    run.dropped_in_flight = sum(  # type: ignore[attr-defined]
        1 for s, r in zip(dispatched, results, strict=True) if s is not None and r is None
    )
    return run


def steady_window(run: LoadGenRun) -> tuple[float, float] | None:
    """
    [first measured dispatch, last measured stream end). The rate denominator
    for a closed-loop run, since it has no configured duration.
    """
    steady = [r for r in run.results if r.spec.phase == Phase.STEADY]
    if not steady:
        return None
    start = min(r.spec.intended_send_time for r in steady)
    ends = [r.stream_end_time for r in steady if r.stream_end_time is not None]
    end = max(ends) if ends else start
    return (start, end)


def inflight_check(run: LoadGenRun, concurrency: int, min_fraction: float = 0.95) -> dict[str, Any]:
    """Did the measured window actually hold N in flight?"""
    win = steady_window(run)
    if win is None:
        return {"held": False, "reason": "no measured requests", "concurrency": concurrency}
    t_start, t_end = win
    vals = [v for t, v in run.inflight_samples if t_start <= t < t_end]
    if not vals:
        return {
            "held": False,
            "reason": "no in-flight samples inside the measured window",
            "concurrency": concurrency,
        }
    mean = statistics.fmean(vals)
    held = mean >= min_fraction * concurrency
    return {
        "held": held,
        "concurrency": concurrency,
        "mean_inflight": mean,
        "min_inflight": min(vals),
        "max_inflight": max(vals),
        "n_samples": len(vals),
        "min_fraction": min_fraction,
        "reason": None
        if held
        else (
            f"mean in-flight {mean:.2f} < {min_fraction:.0%} of N={concurrency} during the "
            "measured window: the run did not measure the concurrency it claims"
        ),
    }
