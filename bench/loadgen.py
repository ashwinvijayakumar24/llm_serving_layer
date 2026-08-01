"""
Open-loop load generator. The component every published number depends on.

WHY OPEN LOOP (docs/BENCHMARK_METHODOLOGY.md §1)
-----------------------------------------------
A closed-loop harness — N clients, each sending, waiting for the full response,
then sending again — cannot create a queue deeper than N by construction. Offered
load becomes a function of system speed: when the server slows down, the harness
slows down with it, arrivals stop, and the queue never grows. Such a system
appears to have no saturation point at all. Latency degrades smoothly and
forever, and the knee — the capacity number, the only interesting result here —
is structurally invisible.

So requests here are issued on a schedule COMPUTED IN ADVANCE from an arrival
process, and dispatched whether or not anything has come back. Nothing in the
dispatch loop awaits a response. Concurrency is unbounded by default; if a bound
is configured it is recorded in the artifact and a warning is emitted the moment
it is reached, because a bound that binds has silently turned this back into a
closed-loop harness.

WHY LATENCY IS MEASURED FROM INTENDED DISPATCH (R1, CRITICAL)
------------------------------------------------------------
Coordinated omission is the single most likely way this project publishes a
confidently wrong number. If the generator is late issuing a request — event loop
starved, connection pool exhausted, GIL contention, a bounded semaphore — then
the time the request spent NOT YET SENT is exactly the time the system was too
busy to serve it. Timing from the actual send time deletes that interval from
the distribution. The worse the saturation, the more it deletes. A saturated
system then reports an excellent p99, nothing errors, and the number is not
merely imprecise, it is backwards.

Therefore: every latency in this file is `observation - intended_send_time`,
where `intended_send_time` comes from the precomputed schedule and is fixed
before the run starts. `dispatch_drift_ms = actual_send - intended_send` is a
first-class metric, emitted into every artifact, and a run whose drift exceeds
the declared threshold is marked INVALID — not caveated.

Note what that means for reading the numbers: TTFT here includes harness delay.
That is deliberate. It is the conservative direction, and the drift series is
published alongside so a reader can see how much of the tail is the harness.

WHAT IS RAW AND WHAT IS DERIVED (§5, R15)
-----------------------------------------
Raw per-request and per-token samples go into the artifact. Percentiles are
computed at analysis time over the POOLED set, with sample counts attached, and
they live under `realized_workload["derived"]` — a view over the samples, not a
replacement for them. Averaging per-run percentiles is a different operation from
pooling, and once the samples are discarded the correct number is unrecoverable
(the engine's harness discards them: bench/harness.py:136-137,178).

USAGE
-----
    python3 bench/loadgen.py --url http://127.0.0.1:8000/v1/chat/completions \\
        --rate 8 --duration 60 --warmup 15 --drain 10 --seed 20260731 \\
        --slo-ttft-ms 500 --slo-itl-ms 50 --write results/p2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # runnable as `python3 bench/loadgen.py`
    sys.path.insert(0, str(REPO_ROOT))

from serving.metrics.artifact import (  # noqa: E402
    REGISTRY,
    Artifact,
    MetricSpec,
    Provenance,
)

__all__ = [
    "ArrivalProcess",
    "Outcome",
    "RequestSpec",
    "RequestResult",
    "LoadGenConfig",
    "LoadGenRun",
    "RunValidity",
    "arrival_schedule",
    "build_schedule",
    "percentile",
    "stationarity",
    "meets_slo",
    "validate_run",
    "stream_one",
    "run_load",
    "analyze",
    "render",
    "main",
]

# ---------------------------------------------------------------------------
# Metric definitions. Registered before use so a later harness cannot reuse one
# of these names for a different quantity (R16 — the engine's `peak_mem_mb`).
# ttft_ms / itl_ms / e2e_ms / goodput_rps / offered_load_rps / output_tok_s /
# dispatch_drift_ms are already registered in serving/metrics/artifact.py; they
# are used, not redefined.
# ---------------------------------------------------------------------------

for _spec in [
    MetricSpec(
        "tpot_ms", "time per output token", "ms", "client wall clock",
        "(last content chunk - first content chunk) / (output tokens - 1), one "
        "scalar per request. Reported ALONGSIDE the ITL percentiles, never "
        "instead of them — a mean hides the tail that the SLO is about.",
    ),
    MetricSpec(
        "inter_arrival_ms", "realized inter-arrival gap", "ms",
        "load generator arrival schedule",
        "Gaps in the PRECOMPUTED schedule, not observed sends. Published so a "
        "reader can confirm the arrival process was what it claimed to be — a "
        "seeding bug that collapses Poisson toward deterministic is a degenerate "
        "workload with no error message (R14).",
    ),
    MetricSpec(
        "inflight_requests", "client-observed in-flight requests", "count",
        "load generator sampling of dispatched-but-unfinished requests",
        "Sampled on a fixed interval. The stationarity evidence for R11: a "
        "steady-state window whose in-flight count trends is not steady state, "
        "and its p99 is a statement about the ramp, not the system.",
    ),
    MetricSpec(
        "output_tokens_per_request", "output content chunks", "count",
        "client SSE parse, non-empty delta.content only",
        "Counts chunks with NON-EMPTY content. Empty-content chunks (special "
        "tokens detokenized with skip_special_tokens) are counted separately and "
        "are not tokens for timing purposes (BENCHMARK_METHODOLOGY §2).",
    ),
    MetricSpec(
        "slo_attainment", "SLO attainment", "ratio",
        "client aggregate over steady-state window",
        "Steady-window requests meeting the full SLO / steady-window requests "
        "issued. Failures include errors and timeouts — they are SLO misses, not "
        "missing data.",
    ),
    MetricSpec(
        "completed_rps", "achieved completion rate", "requests/s",
        "client aggregate over steady-state window",
        "Completions per second REGARDLESS of SLO. Distinct from goodput_rps, and "
        "distinct from offered_load_rps which is a parameter. Reporting any of "
        "the three as another is the classic error.",
    ),
]:
    REGISTRY.register(_spec)


# ---------------------------------------------------------------------------
# Arrival process
# ---------------------------------------------------------------------------


class ArrivalProcess:
    POISSON = "poisson"
    DETERMINISTIC = "deterministic"
    ALL = (POISSON, DETERMINISTIC)


def arrival_schedule(
    rate_rps: float,
    horizon_s: float,
    process: str = ArrivalProcess.POISSON,
    seed: int = 0,
) -> list[float]:
    """
    Arrival times in seconds from run start, computed IN ADVANCE.

    `poisson` draws exponential inter-arrival times with mean 1/rate. It is the
    default because it produces genuine bursts, and burstiness is what stresses
    admission control and queueing — the thing being measured.

    `deterministic` places arrivals at exactly k/rate.

    RUNNING BOTH AT THE SAME MEAN RATE ISOLATES BURSTINESS FROM LOAD. Offered
    load is identical by construction; the only difference is the clustering of
    arrivals. Any gap between the two curves is a real result about the
    scheduler's queueing behaviour and not about how hard it was pushed. A
    project that only ever measures deterministic arrivals has smoothed away the
    phenomenon it claims to study.

    Seeded and reproducible: the same (rate, horizon, seed) yields the identical
    list, and the seed is recorded in the artifact's provenance.
    """
    if rate_rps <= 0:
        raise ValueError(f"rate_rps must be positive, got {rate_rps}")
    if horizon_s <= 0:
        raise ValueError(f"horizon_s must be positive, got {horizon_s}")

    if process == ArrivalProcess.DETERMINISTIC:
        n = int(math.floor(horizon_s * rate_rps))
        return [k / rate_rps for k in range(n)]

    if process == ArrivalProcess.POISSON:
        rng = random.Random(seed)
        out: list[float] = []
        t = 0.0
        while True:
            t += rng.expovariate(rate_rps)
            if t >= horizon_s:
                return out
            out.append(t)

    raise ValueError(f"unknown arrival process {process!r}; expected {ArrivalProcess.ALL}")


# ---------------------------------------------------------------------------
# Request specs and results
# ---------------------------------------------------------------------------


class Phase:
    WARMUP = "warmup"
    STEADY = "steady"
    DRAIN = "drain"


class Outcome:
    """
    Every request ends in exactly one of these and every one is counted.

    Errors and timeouts are RESULTS, not missing data. Dropping them is how a
    saturated system gets a flattering distribution: the requests that failed are
    precisely the ones that were slowest.
    """

    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    INCOMPLETE = "incomplete"     # stream ended without [DONE] / finish_reason
    NO_CONTENT = "no_content"     # stream completed but never emitted content
    ALL = (COMPLETED, ERROR, TIMEOUT, INCOMPLETE, NO_CONTENT)


@dataclass(frozen=True)
class RequestSpec:
    """
    One scheduled request. `intended_send_time` is FIXED BEFORE THE RUN STARTS
    and is the origin for every latency this request contributes (R1).
    """

    request_id: int
    intended_send_time: float      # seconds from run start
    prompt: str
    max_tokens: int
    phase: str = Phase.STEADY

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "intended_send_time": self.intended_send_time,
            "prompt_chars": len(self.prompt),
            "max_tokens": self.max_tokens,
            "phase": self.phase,
        }


@dataclass
class RequestResult:
    """
    Everything observed for one request. All times are seconds from run start on
    the same monotonic clock as `RequestSpec.intended_send_time`.

    The latency properties below subtract `intended_send_time`, never
    `actual_send_time`. That is the coordinated-omission guard in one line.
    """

    spec: RequestSpec
    actual_send_time: float
    outcome: str = Outcome.INCOMPLETE
    token_times: list[float] = field(default_factory=list)   # NON-EMPTY content only
    chunk_times: list[float] = field(default_factory=list)   # every SSE data event
    empty_chunks: int = 0
    stream_end_time: float | None = None
    finish_reason: str | None = None
    status_code: int | None = None
    error: str | None = None
    text_chars: int = 0
    concurrency_capped: bool = False

    # -- latencies, all from INTENDED dispatch --------------------------------

    @property
    def dispatch_drift_ms(self) -> float:
        """actual - intended. Harness health, not system performance (R1)."""
        return (self.actual_send_time - self.spec.intended_send_time) * 1e3

    @property
    def ttft_ms(self) -> float | None:
        """
        Intended dispatch -> first chunk with NON-EMPTY content.

        Empty-content chunks are skipped by construction: they never enter
        `token_times`. The engine's server emits one chunk per token id and
        detokenizes with skip_special_tokens=True (engine/server.py:70), so a
        leading special token produces a chunk whose content is "". Timing to the
        first CHUNK rather than the first visible CONTENT would understate TTFT
        by however many special tokens the template happens to start with.
        """
        if not self.token_times:
            return None
        return (self.token_times[0] - self.spec.intended_send_time) * 1e3

    @property
    def e2e_ms(self) -> float | None:
        """Intended dispatch -> final chunk."""
        end = self.chunk_times[-1] if self.chunk_times else self.stream_end_time
        if end is None:
            return None
        return (end - self.spec.intended_send_time) * 1e3

    @property
    def itls_ms(self) -> list[float]:
        """Gaps between consecutive non-empty content chunks. May be empty."""
        t = self.token_times
        return [(t[i] - t[i - 1]) * 1e3 for i in range(1, len(t))]

    @property
    def tpot_ms(self) -> float | None:
        """Mean ITL for this request. One scalar; does not replace the series."""
        t = self.token_times
        if len(t) < 2:
            return None
        return (t[-1] - t[0]) * 1e3 / (len(t) - 1)

    @property
    def output_tokens(self) -> int:
        return len(self.token_times)

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.spec.as_dict(),
            "actual_send_time": self.actual_send_time,
            "dispatch_drift_ms": self.dispatch_drift_ms,
            "outcome": self.outcome,
            "ttft_ms": self.ttft_ms,
            "e2e_ms": self.e2e_ms,
            "tpot_ms": self.tpot_ms,
            "output_tokens": self.output_tokens,
            "empty_chunks": self.empty_chunks,
            "finish_reason": self.finish_reason,
            "status_code": self.status_code,
            "error": self.error,
            "concurrency_capped": self.concurrency_capped,
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LoadGenConfig:
    url: str = "http://127.0.0.1:8000/v1/chat/completions"
    model: str = "llama-3.2-1b"
    rate_rps: float = 4.0
    duration_s: float = 60.0
    warmup_s: float = 15.0
    drain_s: float = 10.0
    process: str = ArrivalProcess.POISSON
    seed: int = 0

    # SLO. PARAMETERS, deliberately not constants: docs/BENCHMARK_METHODOLOGY §3
    # sets them from a calibration run against the unloaded batch-1 engine on the
    # target GPU, then freezes them. Hardcoding a round number here would be an
    # SLO chosen for its roundness.
    slo_ttft_ms: float = 1000.0
    slo_itl_ms: float = 100.0
    slo_itl_percentile: float = 95.0

    # R1 threshold. A run above this is INVALID and does not get published.
    # Applied to the p99 of drift; the max is recorded too but a single scheduler
    # hiccup should not discard an otherwise clean run.
    max_dispatch_drift_ms: float = 50.0
    drift_percentile: float = 99.0

    # 0 == unbounded, which is the design. A bound above the expected in-flight
    # count is permitted for socket-exhaustion safety; it is recorded, and if it
    # is ever reached the run is flagged because a binding bound is a closed loop.
    max_concurrency: int = 0

    request_timeout_s: float = 120.0
    inflight_sample_interval_s: float = 0.05
    stationarity_tolerance: float = 0.25

    # Workload. Skewed by default (§4): fixed lengths remove the phenomena the
    # benchmark exists to study.
    prompt_prefix: str = "You are a helpful assistant. Answer concisely."
    prompt_mean_tokens: int = 256
    prompt_sigma: float = 0.8
    output_mean_tokens: int = 128
    output_sigma: float = 1.0
    output_max_tokens: int = 2048
    ignore_eos: bool = True
    """
    Generate exactly max_tokens. DEFAULT TRUE FOR BENCHMARKS, and the default is
    the point: a reproducible workload cannot let the model decide how much work
    each request is. Set False only to measure a natural-length workload, and
    then say so in the artifact.
    """

    name: str = "loadgen_open_loop"
    extra_config: dict[str, Any] = field(default_factory=dict)

    @property
    def horizon_s(self) -> float:
        return self.warmup_s + self.duration_s + self.drain_s

    @property
    def steady_start_s(self) -> float:
        return self.warmup_s

    @property
    def steady_end_s(self) -> float:
        return self.warmup_s + self.duration_s

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra_config"}
        d.update(self.extra_config)
        d["loop_model"] = "OPEN — dispatch schedule is independent of responses"
        return d


# ---------------------------------------------------------------------------
# Workload construction
# ---------------------------------------------------------------------------


def _lognormal_int(rng: random.Random, mean: int, sigma: float, cap: int) -> int:
    """Lognormal with the requested MEAN (not median), clipped to [1, cap]."""
    if sigma <= 0:
        return max(1, min(cap, mean))
    mu = math.log(mean) - (sigma**2) / 2.0
    return max(1, min(cap, int(round(rng.lognormvariate(mu, sigma)))))


def build_schedule(cfg: LoadGenConfig) -> list[RequestSpec]:
    """
    The whole run's requests, decided before a single byte is sent.

    Phases (R11):
      warmup  [0, warmup)                 issued, DISCARDED at analysis
      steady  [warmup, warmup+duration)   the measurement window
      drain   [.., +drain)                issued, DISCARDED

    The drain requests are issued rather than simply stopping. If arrivals halted
    at the end of the steady window, the last steady requests would finish into a
    queue that is emptying, and their tail latency would be better than the
    system's — which is exactly the mechanism that publishes a p99 the system
    does not have. Load continues past the window so the window is surrounded by
    load on both sides.
    """
    rng = random.Random(cfg.seed)
    times = arrival_schedule(cfg.rate_rps, cfg.horizon_s, cfg.process, cfg.seed)

    specs: list[RequestSpec] = []
    for i, t in enumerate(times):
        if t < cfg.steady_start_s:
            phase = Phase.WARMUP
        elif t < cfg.steady_end_s:
            phase = Phase.STEADY
        else:
            phase = Phase.DRAIN
        n_prompt = _lognormal_int(rng, cfg.prompt_mean_tokens, cfg.prompt_sigma, 8192)
        n_out = _lognormal_int(rng, cfg.output_mean_tokens, cfg.output_sigma, cfg.output_max_tokens)
        # Synthetic prompt, and SAID SO in the artifact rather than implied to be
        # a dataset (§4; the engine shipped a `wikitext_sample.txt` that was not
        # WikiText and it was called the highest-severity issue in that repo).
        # `request_id` in the body makes every prompt unique, i.e. the zero-
        # sharing control. Prefix-sharing structures live in bench/workloads/.
        body = " ".join(f"w{(i * 7919 + k) % 50021}" for k in range(n_prompt))
        specs.append(
            RequestSpec(
                request_id=i,
                intended_send_time=t,
                prompt=f"{cfg.prompt_prefix}\n[req {i}] {body}",
                max_tokens=n_out,
                phase=phase,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def percentile(samples: list[float], q: float) -> float:
    """
    Linear interpolation between order statistics, STATED because tools disagree
    and the disagreement is visible in exactly the tail that gets published (§5).
    Same rule as the engine's bench/harness.py:50-51 and bench/capacity.py, so
    numbers from the three are comparable.
    """
    if not samples:
        raise ValueError("no samples")
    xs = sorted(samples)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * (q / 100.0)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[int(pos)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def describe(samples: list[float], name: str) -> dict[str, Any]:
    """
    Pooled percentiles WITH the sample count attached (R15).

    A p99 over 127 samples is interpolated between the top two order statistics
    and carries about one sample of resolution. Publishing it without n implies a
    precision that is not there, so n travels with every percentile in this dict
    and `p99_resolvable` says outright whether the number means anything.
    """
    n = len(samples)
    if n == 0:
        return {"metric": name, "n": 0, "note": "no samples"}
    return {
        "metric": name,
        "n": n,
        "mean": statistics.fmean(samples),
        "stdev": statistics.pstdev(samples) if n > 1 else 0.0,
        "min": min(samples),
        "p50": percentile(samples, 50),
        "p90": percentile(samples, 90),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "max": max(samples),
        "percentile_method": "linear interpolation between order statistics",
        "p99_resolvable": n >= 1000,
        "p99_note": None if n >= 1000 else (
            f"n={n}: p99 is interpolated near the top of the sample and carries "
            "roughly one sample of resolution (R15)."
        ),
    }


def stationarity(
    samples: list[tuple[float, float]],
    t_start: float,
    t_end: float,
    tolerance: float = 0.25,
) -> dict[str, Any]:
    """
    Was the in-flight count stationary across the measurement window? (R11)

    Steady state is VERIFIED, not assumed. Two independent checks, both reported:
    a half-vs-half mean comparison (robust, easy to read) and an OLS slope
    (sensitive to a slow trend that halves would average away).

    The slope must be BOTH materially large (> tolerance of the window mean) AND
    statistically distinguishable from noise (|t| > 2) before it counts as a
    trend. In-flight count is a fluctuating queue: on a short window its OLS
    slope wanders well past any fixed relative threshold by chance, and a check
    that flags those runs is a check nobody will believe by the third time it
    fires. The t-statistic assumes independent residuals, which an autocorrelated
    queue series violates — so it is a conservative screen against noise, not a
    hypothesis test, and it is reported rather than merely applied.

    A run that fails this is not necessarily wrong — above the knee, steady state
    does not exist by definition, because the queue grows without bound. Those
    runs are labeled `unsaturated-window measurement`, never `steady state`, and
    that label is emitted into the artifact rather than left to the reader.
    """
    win = [(t, v) for t, v in samples if t_start <= t < t_end]
    if len(win) < 4:
        return {
            "held": False,
            "reason": f"only {len(win)} in-flight samples inside the window; "
                      "stationarity is unverifiable, so it is not claimed",
            "n_samples": len(win),
        }

    ts = [t for t, _ in win]
    vs = [v for _, v in win]
    mid = len(win) // 2
    m1 = statistics.fmean(vs[:mid])
    m2 = statistics.fmean(vs[mid:])
    denom = max(1e-9, statistics.fmean(vs))
    half_rel = (m2 - m1) / denom

    tbar, vbar = statistics.fmean(ts), statistics.fmean(vs)
    sxx = sum((t - tbar) ** 2 for t in ts)
    slope = 0.0 if sxx == 0 else sum((t - tbar) * (v - vbar) for t, v in win) / sxx
    span = max(1e-9, ts[-1] - ts[0])
    slope_rel = (slope * span) / denom

    intercept = vbar - slope * tbar
    resid = [v - (intercept + slope * t) for t, v in win]
    dof = len(win) - 2
    se_slope = float("inf")
    if dof > 0 and sxx > 0:
        s2 = sum(r * r for r in resid) / dof
        se_slope = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
    t_stat = 0.0 if se_slope in (0.0, float("inf")) else slope / se_slope

    trend_material = abs(slope_rel) > tolerance
    trend_significant = abs(t_stat) > 2.0
    held = abs(half_rel) <= tolerance and not (trend_material and trend_significant)
    return {
        "held": held,
        "n_samples": len(win),
        "mean_inflight": vbar,
        "first_half_mean": m1,
        "second_half_mean": m2,
        "half_vs_half_relative_change": half_rel,
        "ols_slope_per_s": slope,
        "ols_relative_change_over_window": slope_rel,
        "ols_t_statistic": t_stat,
        "trend_material": trend_material,
        "trend_significant": trend_significant,
        "tolerance": tolerance,
        "label": "steady state" if held else "unsaturated-window measurement",
        "note": None if held else (
            "In-flight count TRENDS across the window: this is not steady state. "
            "Either the run is above the knee (where steady state cannot exist) "
            "or the window is too short. Percentiles from it describe a ramp."
        ),
    }


# ---------------------------------------------------------------------------
# SLO
# ---------------------------------------------------------------------------


def meets_slo(r: RequestResult, cfg: LoadGenConfig) -> tuple[bool, str]:
    """
    A request meets the SLO iff it COMPLETED, its TTFT (from intended dispatch)
    is under the TTFT budget, and its own p95 ITL is under the ITL budget
    (docs/BENCHMARK_METHODOLOGY §3).

    Goodput is the headline because it is the one metric gameable in neither
    direction: batch harder and throughput rises while SLO attainment falls;
    serve one request at a time and latency is perfect while goodput is nil.

    Requests with fewer than two content chunks have no ITL series. They are
    judged on TTFT and completion alone and the reason string says so, so a
    workload that degenerates into one-token replies cannot quietly pass the ITL
    clause by having no ITLs.

    THE PER-TOKEN CLAUSE IS JUDGED ON TPOT, NOT ON p95 ITL.
    -------------------------------------------------------
    Client-observed ITL is bimodal: SSE chunks arrive in bursts, so a request's
    ITL series is a mix of near-zero gaps and occasional large ones. Its p95
    therefore reports the burst GAP, not the per-token generation time, and it
    exceeds any threshold derived from real decode speed.

    Measured: unloaded TPOT p50 9.3 ms, raw ITL p50 0.08 ms with p90 ~30 ms
    (job 11602207; reproduced on CPU against a fake model doing a fixed 10 ms
    per token, so it is the server's emission pattern, not a client artefact).

    An earlier version anchored the THRESHOLD on TPOT but still EVALUATED
    against p95 ITL. Every request then failed the per-token clause while its
    TTFT sat comfortably inside budget, and goodput read ~0 at every offered
    load — a system that was in fact serving 1049 tok/s looked completely
    broken. Threshold and evaluation must use the same statistic.

    TPOT = (last_token - first_token) / (n_tokens - 1), averaged WITHIN the
    request, is immune to how tokens were bunched on the wire. The ITL
    distribution is still recorded in the artifact, because burstiness is real
    and a user perceives it — it is simply not what the pass/fail turns on.
    """
    if r.outcome != Outcome.COMPLETED:
        return False, f"outcome={r.outcome}"
    ttft = r.ttft_ms
    if ttft is None:
        return False, "no content chunk ever arrived"
    if ttft >= cfg.slo_ttft_ms:
        return False, f"ttft {ttft:.1f}ms >= {cfg.slo_ttft_ms}ms"
    tpot = r.tpot_ms
    if tpot is None:
        return True, "met (single-token response: no per-token rate to judge)"
    if tpot >= cfg.slo_itl_ms:
        return False, f"tpot {tpot:.1f}ms >= {cfg.slo_itl_ms}ms"
    return True, "met"


# ---------------------------------------------------------------------------
# Run validity — R1's enforcement point
# ---------------------------------------------------------------------------


@dataclass
class RunValidity:
    valid: bool
    reasons: list[str]
    drift: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "reasons": self.reasons, "dispatch_drift": self.drift}


def validate_run(
    results: list[RequestResult],
    cfg: LoadGenConfig,
    stationarity_result: dict[str, Any] | None = None,
) -> RunValidity:
    """
    Decide whether this run may back a number at all.

    Drift above threshold means the harness, not the system, was the bottleneck
    for some requests — and since latency is measured from intended dispatch, the
    harness's own stall is being attributed to the server. INVALID, discarded,
    not caveated (R1). The alternative — publishing with a footnote — is how a
    footnote ends up under a graph nobody reads.
    """
    reasons: list[str] = []
    drifts = [r.dispatch_drift_ms for r in results]
    d = describe(drifts, "dispatch_drift_ms") if drifts else {"n": 0}
    if drifts:
        p = percentile(drifts, cfg.drift_percentile)
        d["threshold_ms"] = cfg.max_dispatch_drift_ms
        d["threshold_percentile"] = cfg.drift_percentile
        if p >= cfg.max_dispatch_drift_ms:
            reasons.append(
                f"COORDINATED OMISSION RISK (R1): p{cfg.drift_percentile:g} dispatch "
                f"drift {p:.1f}ms >= threshold {cfg.max_dispatch_drift_ms}ms. The "
                "generator could not issue requests on schedule, so the offered load "
                "was not the configured load. Run is INVALID and must not be published."
            )
    else:
        reasons.append("No requests were dispatched.")

    capped = sum(1 for r in results if r.concurrency_capped)
    if capped:
        reasons.append(
            f"{capped} request(s) blocked on the concurrency cap "
            f"(max_concurrency={cfg.max_concurrency}). A bound that binds makes this "
            "a CLOSED-loop harness for those requests, and a closed loop cannot "
            "produce a queue deeper than its cap (§1). Raise the cap and re-run."
        )

    if stationarity_result is not None and not stationarity_result.get("held", False):
        reasons.append(
            "Steady state NOT verified: " + str(
                stationarity_result.get("note") or stationarity_result.get("reason")
            ) + " Numbers from this window are labeled "
            f"{stationarity_result.get('label', 'unsaturated-window measurement')!r}."
        )

    return RunValidity(valid=not reasons, reasons=reasons, drift=d)


# ---------------------------------------------------------------------------
# The HTTP/SSE client path
# ---------------------------------------------------------------------------


def _parse_sse_line(line: str) -> tuple[str, Any] | None:
    """
    ('done', None) | ('chunk', obj) | ('bad', raw) | None for lines to ignore.

    OpenAI-style SSE: `data: {...}` lines, terminated by `data: [DONE]`. Comment
    lines (`:`), event/id fields, and blank separators are ignored.
    """
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return ("done", None)
    try:
        return ("chunk", json.loads(payload))
    except json.JSONDecodeError:
        return ("bad", payload)


async def stream_one(
    client: Any,
    spec: RequestSpec,
    cfg: LoadGenConfig,
    t0: float,
    inflight: list[int] | None = None,
    sem: asyncio.Semaphore | None = None,
) -> RequestResult:
    """
    Issue one request and time its stream.

    `t0` is the run's monotonic origin; every recorded time is relative to it and
    therefore directly comparable with `spec.intended_send_time`. Nothing in here
    ever looks at the actual send time to compute a latency — `actual_send_time`
    is recorded for the drift metric alone.
    """
    capped = False
    if sem is not None:
        if sem.locked():
            capped = True   # we are about to WAIT: this request is being throttled
        await sem.acquire()

    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": spec.prompt}],
        "max_tokens": spec.max_tokens,
        "stream": True,
        "temperature": 0.0,
        # Output length must be CONTROLLED, not model-determined (methodology
        # §4). Without this the model emits EOS whenever it likes and each
        # request represents a different amount of work — job 11599377 asked for
        # 64 tokens and averaged 12, so it was measuring prefill and calling the
        # result decode throughput. Worse, a scheduling change that alters which
        # token is sampled also alters the workload, and the comparison stops
        # being about scheduling at all.
        "ignore_eos": cfg.ignore_eos,
    }

    if inflight is not None:
        inflight[0] += 1
    actual = time.perf_counter() - t0
    res = RequestResult(spec=spec, actual_send_time=actual, concurrency_capped=capped)
    saw_done = False
    try:
        async with client.stream(
            "POST", cfg.url, json=payload, timeout=cfg.request_timeout_s
        ) as resp:
            res.status_code = resp.status_code
            if resp.status_code != 200:
                body = await resp.aread()
                res.outcome = Outcome.ERROR
                res.error = f"HTTP {resp.status_code}: {body[:200]!r}"
                res.stream_end_time = time.perf_counter() - t0
                return res
            async for line in resp.aiter_lines():
                parsed = _parse_sse_line(line)
                if parsed is None:
                    continue
                kind, obj = parsed
                now = time.perf_counter() - t0
                if kind == "done":
                    saw_done = True
                    break
                if kind == "bad":
                    res.outcome = Outcome.ERROR
                    res.error = f"unparseable SSE payload: {obj[:200]!r}"
                    res.stream_end_time = now
                    return res
                res.chunk_times.append(now)
                choices = obj.get("choices") or [{}]
                choice = choices[0] if choices else {}
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    # THE definitional point (§2): only a non-empty content chunk
                    # is a token. Empty ones are counted, never timed.
                    res.token_times.append(now)
                    res.text_chars += len(content)
                else:
                    res.empty_chunks += 1
                if choice.get("finish_reason"):
                    res.finish_reason = choice["finish_reason"]
            res.stream_end_time = time.perf_counter() - t0
    except TimeoutError:
        res.outcome = Outcome.TIMEOUT
        res.error = f"asyncio timeout after {cfg.request_timeout_s}s"
        res.stream_end_time = time.perf_counter() - t0
        return _finish(res, inflight, sem)
    except Exception as exc:  # noqa: BLE001 — outcome is data, not an exception path
        name = type(exc).__name__
        res.outcome = Outcome.TIMEOUT if "Timeout" in name else Outcome.ERROR
        res.error = f"{name}: {exc}"
        res.stream_end_time = time.perf_counter() - t0
        return _finish(res, inflight, sem)
    except BaseException:
        # Cancellation still has to give the slot back, or a cancelled run leaks
        # in-flight count and the stationarity series becomes fiction.
        _finish(res, inflight, sem)
        raise

    if not (saw_done or res.finish_reason):
        res.outcome = Outcome.INCOMPLETE
        res.error = res.error or "stream ended without [DONE] or finish_reason"
    elif not res.token_times:
        res.outcome = Outcome.NO_CONTENT
        res.error = res.error or "stream completed but emitted no non-empty content"
    else:
        res.outcome = Outcome.COMPLETED
    return _finish(res, inflight, sem)


def _finish(
    res: RequestResult, inflight: list[int] | None, sem: asyncio.Semaphore | None
) -> RequestResult:
    if inflight is not None:
        inflight[0] -= 1
    if sem is not None:
        sem.release()
    return res


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


@dataclass
class LoadGenRun:
    cfg: LoadGenConfig
    schedule: list[RequestSpec]
    results: list[RequestResult]
    inflight_samples: list[tuple[float, float]]
    wall_seconds: float
    started_utc: str


async def run_load(cfg: LoadGenConfig, client: Any = None) -> LoadGenRun:
    """
    Dispatch the precomputed schedule. THE DISPATCH LOOP NEVER AWAITS A RESPONSE.

    That sentence is the whole design. `asyncio.create_task` fires the request and
    returns immediately; the loop's only await is a sleep until the next intended
    arrival. If the server stalls, tasks accumulate and in-flight count grows
    without bound — which is the queue this benchmark exists to measure and which
    a closed-loop harness cannot create.

    A caller may inject `client` (an httpx.AsyncClient, or any object with the
    same `.stream()` contract) so the whole path is testable against a mock
    transport with no server anywhere.
    """
    if cfg.process not in ArrivalProcess.ALL:
        raise ValueError(f"unknown arrival process {cfg.process!r}")

    schedule = build_schedule(cfg)
    owns_client = client is None
    if owns_client:
        import httpx

        # Connection limits far above any plausible in-flight count: a pool that
        # blocks is a closed loop wearing an open loop's clothes, and it would
        # show up as dispatch drift rather than as an error.
        client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            timeout=cfg.request_timeout_s,
        )

    sem = asyncio.Semaphore(cfg.max_concurrency) if cfg.max_concurrency > 0 else None
    inflight = [0]
    inflight_samples: list[tuple[float, float]] = []
    stop = asyncio.Event()

    t0 = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    async def sampler() -> None:
        while not stop.is_set():
            inflight_samples.append((time.perf_counter() - t0, float(inflight[0])))
            try:
                await asyncio.wait_for(stop.wait(), timeout=cfg.inflight_sample_interval_s)
            except TimeoutError:
                continue

    sampler_task = asyncio.create_task(sampler())
    tasks: list[asyncio.Task[RequestResult]] = []
    try:
        for spec in schedule:
            delay = spec.intended_send_time - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(
                asyncio.create_task(stream_one(client, spec, cfg, t0, inflight, sem))
            )
        results = list(await asyncio.gather(*tasks))
    finally:
        stop.set()
        await sampler_task
        if owns_client:
            await client.aclose()

    inflight_samples.append((time.perf_counter() - t0, float(inflight[0])))
    return LoadGenRun(
        cfg=cfg,
        schedule=schedule,
        results=results,
        inflight_samples=inflight_samples,
        wall_seconds=time.perf_counter() - t0,
        started_utc=started_utc,
    )


# ---------------------------------------------------------------------------
# Analysis -> Artifact
# ---------------------------------------------------------------------------


@dataclass
class LoadGenAnalysis:
    artifact: Artifact
    validity: RunValidity
    stationarity: dict[str, Any]
    summary: dict[str, Any]


def analyze(run: LoadGenRun, repo_root: str | Path = REPO_ROOT) -> LoadGenAnalysis:
    """
    Pool the steady-state samples, compute goodput, and emit an Artifact.

    Only steady-phase requests enter the metrics (R11). Warmup and drain requests
    are issued, counted, and then discarded here — their counts are recorded so a
    reader can see the window was surrounded by load rather than sitting at the
    edge of an emptying queue.

    RAW samples go into `artifact.samples`. Percentiles go into
    `realized_workload["derived"]` as a VIEW over them, with n attached to each.
    """
    cfg = run.cfg
    steady = [r for r in run.results if r.spec.phase == Phase.STEADY]
    window_s = max(1e-9, cfg.duration_s)

    stat = stationarity(
        run.inflight_samples, cfg.steady_start_s, cfg.steady_end_s, cfg.stationarity_tolerance
    )
    validity = validate_run(run.results, cfg, stat)

    ttfts = [r.ttft_ms for r in steady if r.ttft_ms is not None]
    e2es = [r.e2e_ms for r in steady if r.e2e_ms is not None]
    itls: list[float] = []
    for r in steady:
        itls.extend(r.itls_ms)
    tpots = [r.tpot_ms for r in steady if r.tpot_ms is not None]
    drifts_all = [r.dispatch_drift_ms for r in run.results]
    drifts_steady = [r.dispatch_drift_ms for r in steady]
    out_toks = [float(r.output_tokens) for r in steady]

    slo_flags = [meets_slo(r, cfg) for r in steady]
    n_met = sum(1 for ok, _ in slo_flags if ok)
    n_completed = sum(1 for r in steady if r.outcome == Outcome.COMPLETED)

    outcomes: dict[str, int] = {k: 0 for k in Outcome.ALL}
    for r in steady:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
    miss_reasons: dict[str, int] = {}
    for ok, why in slo_flags:
        if not ok:
            key = why.split(" ")[0]     # "outcome=error" / "ttft" / "p95"
            miss_reasons[key] = miss_reasons.get(key, 0) + 1

    inter_arrivals = [
        (b.intended_send_time - a.intended_send_time) * 1e3
        for a, b in zip(run.schedule, run.schedule[1:], strict=False)
        if a.phase == Phase.STEADY and b.phase == Phase.STEADY
    ]

    goodput_rps = n_met / window_s
    completed_rps = n_completed / window_s
    attainment = (n_met / len(steady)) if steady else 0.0
    output_tok_s = sum(out_toks) / window_s

    prov = Provenance.capture(repo_root=repo_root, seed=cfg.seed)
    art = Artifact(
        name=cfg.name,
        provenance=prov,
        config={
            **cfg.as_dict(),
            "slo_definition": (
                "completed AND ttft_ms < slo_ttft_ms AND "
                f"p{cfg.slo_itl_percentile:g}(itl within the request) < slo_itl_ms"
            ),
            "latency_origin": "INTENDED dispatch time from the arrival schedule (R1)",
            "ttft_definition": "first SSE chunk with NON-EMPTY delta.content",
            "prompt_source": "SYNTHETIC token noise, not a dataset — stated, not implied",
            "run_started_utc": run.started_utc,
        },
        window={
            "warmup_s": cfg.warmup_s,
            "steady_start_s": cfg.steady_start_s,
            "steady_end_s": cfg.steady_end_s,
            "drain_s": cfg.drain_s,
            "duration_s": cfg.duration_s,
            "wall_seconds": run.wall_seconds,
            "requests_warmup": float(sum(1 for r in run.results if r.spec.phase == Phase.WARMUP)),
            "requests_steady": float(len(steady)),
            "requests_drain": float(sum(1 for r in run.results if r.spec.phase == Phase.DRAIN)),
        },
        realized_workload={
            "requested": {
                "rate_rps": cfg.rate_rps,
                "process": cfg.process,
                "seed": cfg.seed,
                "prompt_mean_tokens": cfg.prompt_mean_tokens,
                "output_mean_tokens": cfg.output_mean_tokens,
            },
            # R14: what the generator ACTUALLY produced, so a collapsed
            # distribution announces itself instead of quietly measuring nothing.
            "realized_inter_arrival_ms": describe(inter_arrivals, "inter_arrival_ms"),
            "realized_max_tokens": describe(
                [float(s.max_tokens) for s in run.schedule if s.phase == Phase.STEADY],
                "requested_max_tokens",
            ),
            "realized_output_tokens": describe(out_toks, "output_tokens_per_request"),
            "realized_rate_rps": len(steady) / window_s,
            "outcomes_steady": outcomes,
            "slo_miss_reasons_steady": miss_reasons,
            "empty_content_chunks_steady": sum(r.empty_chunks for r in steady),
            "stationarity": stat,
            "run_validity": validity.as_dict(),
            "window_label": stat.get("label", "unsaturated-window measurement"),
            # DERIVED VIEW over the raw samples above. The samples are the record;
            # these are a convenience, recomputable and discardable (§5, R15).
            "derived": {
                "ttft_ms": describe(ttfts, "ttft_ms"),
                "itl_ms": describe(itls, "itl_ms"),
                "e2e_ms": describe(e2es, "e2e_ms"),
                "tpot_ms": describe(tpots, "tpot_ms"),
                "dispatch_drift_ms_steady": describe(drifts_steady, "dispatch_drift_ms"),
                "dispatch_drift_ms_all_phases": describe(drifts_all, "dispatch_drift_ms"),
            },
        },
        notes=[
            "OPEN LOOP: requests were dispatched on a precomputed schedule, never "
            "gated on a response. Concurrency was "
            + ("unbounded." if cfg.max_concurrency == 0 else
               f"bounded at {cfg.max_concurrency}."),
            "All latencies are measured from INTENDED dispatch time, so they "
            "INCLUDE any harness lateness. That is the conservative direction and "
            "the guard against coordinated omission (R1). The drift series is in "
            "this artifact so a reader can subtract it if they want to.",
            f"Steady-state window [{cfg.steady_start_s:g}s, {cfg.steady_end_s:g}s); "
            f"warmup and drain requests were issued and DISCARDED (R11).",
        ],
    )

    art.add_samples("ttft_ms", ttfts)
    art.add_samples("itl_ms", itls)
    art.add_samples("e2e_ms", e2es)
    art.add_samples("tpot_ms", tpots)
    art.add_samples("dispatch_drift_ms", drifts_all)
    art.add_samples("output_tokens_per_request", out_toks)
    art.add_samples("inter_arrival_ms", inter_arrivals)
    art.add_samples(
        "inflight_requests",
        [v for t, v in run.inflight_samples if cfg.steady_start_s <= t < cfg.steady_end_s],
    )

    art.set_scalar("goodput_rps", goodput_rps)
    art.set_scalar("offered_load_rps", cfg.rate_rps)
    art.set_scalar("completed_rps", completed_rps)
    art.set_scalar("slo_attainment", attainment)
    art.set_scalar("output_tok_s", output_tok_s)

    if not validity.valid:
        art.notes.insert(0, "RUN INVALID — must not back a published number: "
                         + " | ".join(validity.reasons))
    ok_pub, blockers = prov.is_publishable()
    if not ok_pub:
        art.notes.append("NOT PUBLISHABLE: " + " | ".join(blockers))

    summary = {
        "goodput_rps": goodput_rps,
        "offered_load_rps": cfg.rate_rps,
        "completed_rps": completed_rps,
        "slo_attainment": attainment,
        "output_tok_s": output_tok_s,
        "steady_requests": len(steady),
        "steady_completed": n_completed,
        "steady_met_slo": n_met,
        "outcomes": outcomes,
        "derived": art.realized_workload["derived"],
    }
    return LoadGenAnalysis(artifact=art, validity=validity, stationarity=stat, summary=summary)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(d: dict[str, Any]) -> str:
    if not d.get("n"):
        return "        (no samples)"
    return (
        f"        n={d['n']:<7d} p50 {d['p50']:>9.2f}  p95 {d['p95']:>9.2f}  "
        f"p99 {d['p99']:>9.2f}  max {d['max']:>9.2f}"
        + ("" if d.get("p99_resolvable") else "   [p99 under-resolved, R15]")
    )


def render(a: LoadGenAnalysis) -> str:
    art = a.artifact
    cfg_rate = art.scalars["offered_load_rps"]
    d = art.realized_workload["derived"]
    lines = [
        "Open-loop load generation result",
        "=" * 78,
        f"  window            [{art.window['steady_start_s']:g}s, "
        f"{art.window['steady_end_s']:g}s)  "
        f"warmup {art.window['warmup_s']:g}s / drain {art.window['drain_s']:g}s",
        f"  requests          warmup {art.window['requests_warmup']:.0f} · "
        f"steady {art.window['requests_steady']:.0f} · "
        f"drain {art.window['requests_drain']:.0f}",
        f"  label             {art.realized_workload['window_label']}",
        "",
        "Latency (ms), pooled over the steady window, FROM INTENDED DISPATCH",
        "-" * 78,
        "    TTFT", _fmt(d["ttft_ms"]),
        "    ITL", _fmt(d["itl_ms"]),
        "    E2E", _fmt(d["e2e_ms"]),
        "",
        "Harness health — dispatch drift (R1)",
        "-" * 78,
        "    steady", _fmt(d["dispatch_drift_ms_steady"]),
        "    all phases", _fmt(d["dispatch_drift_ms_all_phases"]),
        "",
        "Headline",
        "-" * 78,
        f"  offered load       {cfg_rate:>10.3f} req/s   (a PARAMETER)",
        f"  completed          {art.scalars['completed_rps']:>10.3f} req/s",
        f"  GOODPUT under SLO  {art.scalars['goodput_rps']:>10.3f} req/s",
        f"  SLO attainment     {art.scalars['slo_attainment'] * 100:>10.2f} %",
        f"  output throughput  {art.scalars['output_tok_s']:>10.2f} tok/s",
        "",
        f"  outcomes: {art.realized_workload['outcomes_steady']}",
        f"  SLO miss reasons: {art.realized_workload['slo_miss_reasons_steady']}",
        "",
        "Validity",
        "-" * 78,
    ]
    if a.validity.valid:
        lines.append("  VALID — dispatch drift within threshold, steady state verified.")
    else:
        lines.append("  *** RUN INVALID — this run may not back a published number ***")
        lines += [f"    - {r}" for r in a.validity.reasons]
    lines += ["", "Notes", "-" * 78] + [f"  - {n}" for n in art.notes]
    lines.append("")
    lines.append(
        "  A single goodput number is not a result. The publishable form is a "
        "goodput-vs-offered-load CURVE (§3): sweep --rate and plot these points."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Open-loop load generator with a coordinated-omission guard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url", default=LoadGenConfig.url, help="OpenAI-style streaming endpoint")
    p.add_argument("--model", default=LoadGenConfig.model)
    p.add_argument("--rate", type=float, default=4.0, help="offered load, req/s (a PARAMETER)")
    p.add_argument("--duration", type=float, default=60.0, help="steady-state window, s")
    p.add_argument("--warmup", type=float, default=15.0, help="discarded ramp-up, s")
    p.add_argument("--drain", type=float, default=10.0,
                   help="discarded tail; load CONTINUES here so the window is not "
                        "measured against an emptying queue")
    p.add_argument("--process", choices=list(ArrivalProcess.ALL), default=ArrivalProcess.POISSON)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slo-ttft-ms", type=float, default=1000.0)
    p.add_argument("--slo-itl-ms", type=float, default=100.0)
    p.add_argument("--slo-itl-percentile", type=float, default=95.0)
    p.add_argument("--max-drift-ms", type=float, default=50.0,
                   help="run is INVALID above this drift percentile (R1)")
    p.add_argument("--max-concurrency", type=int, default=0,
                   help="0 = unbounded, which is the design; a bound that binds is "
                        "a closed loop and flags the run")
    p.add_argument("--timeout", type=float, default=120.0, help="per-request timeout, s")
    p.add_argument("--prompt-tokens", type=int, default=256)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--name", default="loadgen_open_loop")
    p.add_argument("--write", metavar="DIR", default=None,
                   help="write the artifact JSON (raw samples included) to DIR")
    p.add_argument("--json", action="store_true", help="print the artifact JSON to stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = LoadGenConfig(
        url=args.url,
        model=args.model,
        rate_rps=args.rate,
        duration_s=args.duration,
        warmup_s=args.warmup,
        drain_s=args.drain,
        process=args.process,
        seed=args.seed,
        slo_ttft_ms=args.slo_ttft_ms,
        slo_itl_ms=args.slo_itl_ms,
        slo_itl_percentile=args.slo_itl_percentile,
        max_dispatch_drift_ms=args.max_drift_ms,
        max_concurrency=args.max_concurrency,
        request_timeout_s=args.timeout,
        prompt_mean_tokens=args.prompt_tokens,
        output_mean_tokens=args.output_tokens,
        name=args.name,
    )
    run = asyncio.run(run_load(cfg))
    analysis = analyze(run)
    print(render(analysis))
    if args.json:
        print(json.dumps(analysis.artifact.to_dict(), indent=2, sort_keys=True))
    if args.write:
        path = analysis.artifact.write(args.write)
        print(f"\nartifact -> {path}")
    # Non-zero exit on an invalid run: a CI sweep must not quietly collect a run
    # that cannot be published.
    return 0 if analysis.validity.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
