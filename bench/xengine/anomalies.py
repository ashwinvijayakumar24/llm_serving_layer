"""
Anomaly detectors. SPEC hard rule 2: anomalies are FLAGGED, not smoothed.

Every detector returns a list of `{"kind": str, "detail": str, ...}` dicts that
go straight into the artifact's `anomalies` list. A detector never alters the
samples or the metrics; the renderer prints the flag next to the cell.

BIMODALITY — WHY A KDE MODE COUNT AND NOT SARLE'S COEFFICIENT ALONE
-------------------------------------------------------------------
Sarle's bimodality coefficient BC = (skew^2 + 1) / (excess kurtosis + 3(n-1)^2
/ ((n-2)(n-3))) is cheap, but its 5/9 threshold is crossed by plainly unimodal
shapes this study produces all the time: an exponential sits at ~0.6, a
lognormal with sigma 1 at ~0.66, and a uniform at ~0.56. Latency distributions
are skewed by nature, so BC alone would flag nearly every heavy-tailed TTFT.

The test used instead counts the modes of a Gaussian kernel density estimate
of log(x), with the normal-reference bandwidth h = 1.06 * sd * n^-1/5.

  * Log scale, because latency is positive and right-skewed: a lognormal
    becomes a normal there, so skew alone can no longer masquerade as a second
    mode, while two genuinely separate populations stay separate.
  * The normal-reference bandwidth OVER-smooths multimodal data (a mixture
    inflates sd). That is the conservative direction: a second mode that
    survives it is real. (Silverman's 0.9*min(sd, IQR/1.34) variant was tried
    and rejected: its IQR term shrinks h to the main population's width and
    splits a wide minor population into several spurious bumps.)

A second mode is reported only if

  * the valley between it and its neighbour is at most `valley_ratio` (0.75)
    of the smaller peak's density — a shoulder on a tail is not a mode, and
  * the smaller side holds at least `min_mass` (5%) of the probability mass —
    a handful of outliers is a tail, not a population.

Calibrated on synthetic data (60 seeds, n=300): zero false positives on
normal, lognormal (sigma 1 and 1.5), exponential, uniform and a 2% heavy tail;
detected every time on a 90/10 well-separated mixture and on two equal
populations whose means differ by 4 standard deviations.

BC is still computed and reported next to the verdict as supporting evidence.
Run on TTFT, TPOT and E2E. NOT run on ITL: client-observed ITL is bimodal by
construction (SSE chunks arrive in bursts — results/p2/RESULTS.md), so the
flag would fire on every run of every engine and carry no information.

WARMUP EFFECT
-------------
Median of the first k steady-window requests (k = max(5, 10% of n), dispatch
order) vs the median of the rest. Flagged when the ratio exceeds 1.5x in either
direction AND the absolute gap exceeds 5 ms, so a 0.2 ms -> 0.4 ms change on a
trivially fast metric does not fire. A flag here means the warmup was too short
for this engine (JIT compilation, CUDA-graph capture, cache fill), and the
steady window still contains a ramp.

REP-TO-REP SPREAD
-----------------
Computed when all reps of a point exist: coefficient of variation (stdev/mean,
population stdev) of each headline metric across reps. Flagged above 0.10 —
the same threshold the renderer uses (DEFAULT_CV_THRESHOLD).

GPU CLOCK / TEMPERATURE DRIFT
-----------------------------
If `hardware.GpuSampler` ran, compare the SM clock in the first and last
quarter of the run (drop > 10% = throttling) and flag any temperature at or
above 85 C. Absent samples produce no flag and no claim.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

BIMODAL_METRICS = ("ttft_ms", "tpot_ms", "e2e_ms")
REP_SPREAD_METRICS = {
    "output_tok_s": ("metrics", "output_tok_s"),
    "goodput_rps": ("metrics", "goodput_rps"),
    "ttft_p50": ("metrics", "ttft_ms", "p50"),
    "tpot_p50": ("metrics", "tpot_ms", "p50"),
}
DEFAULT_CV_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Bimodality
# ---------------------------------------------------------------------------


def sarle_bc(xs: Sequence[float]) -> float | None:
    """Sarle's bimodality coefficient with bias-corrected skew/kurtosis. n >= 4."""
    n = len(xs)
    if n < 4:
        return None
    m = statistics.fmean(xs)
    s2 = sum((x - m) ** 2 for x in xs) / n
    if s2 <= 0:
        return None
    m3 = sum((x - m) ** 3 for x in xs) / n
    m4 = sum((x - m) ** 4 for x in xs) / n
    g1 = m3 / s2**1.5
    g2 = m4 / s2**2 - 3.0
    skew = g1 * math.sqrt(n * (n - 1)) / (n - 2)
    kurt = ((n + 1) * g2 + 6.0) * (n - 1) / ((n - 2) * (n - 3))
    denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (skew**2 + 1.0) / denom if denom > 0 else None


def _subsample(xs: Sequence[float], cap: int) -> list[float]:
    """Deterministic, order-statistic-preserving thinning for large n."""
    s = sorted(xs)
    if len(s) <= cap:
        return s
    step = (len(s) - 1) / (cap - 1)
    return [s[round(i * step)] for i in range(cap)]


def kde_modes(
    xs: Sequence[float],
    grid: int = 256,
    min_mass: float = 0.05,
    valley_ratio: float = 0.75,
    cap: int = 2000,
    log_scale: bool = True,
) -> dict[str, Any]:
    """
    Count significant modes of a normal-reference-bandwidth Gaussian KDE.
    With `log_scale` and all-positive data the KDE runs on log(x) and the
    reported locations are mapped back to x.
    """
    s = _subsample(xs, cap)
    n = len(s)
    if n < 2:
        return {"n_modes": len(s), "bandwidth": None, "splits": [], "log_scale": False}
    use_log = log_scale and s[0] > 0
    if use_log:
        s = [math.log(x) for x in s]
    back = math.exp if use_log else (lambda v: v)
    sd = statistics.pstdev(s)
    if sd <= 0:
        return {"n_modes": 1, "bandwidth": 0.0, "splits": [], "log_scale": use_log}
    h = 1.06 * sd * n ** (-0.2)
    lo, hi = s[0] - 3 * h, s[-1] + 3 * h
    pts = [lo + (hi - lo) * i / (grid - 1) for i in range(grid)]
    dens = [sum(math.exp(-0.5 * ((x - g) / h) ** 2) for x in s) for g in pts]
    total = sum(dens)
    peaks = [i for i in range(1, grid - 1) if dens[i] > dens[i - 1] and dens[i] >= dens[i + 1]]

    # Merge peaks whose separating valley is too shallow or whose side is too
    # light; what survives is the set of significant modes.
    changed = True
    while changed and len(peaks) > 1:
        changed = False
        for j in range(len(peaks) - 1):
            a, b = peaks[j], peaks[j + 1]
            v = min(range(a, b + 1), key=lambda i: dens[i])
            left = sum(dens[:v]) / total
            right = 1.0 - left
            shallow = dens[v] > valley_ratio * min(dens[a], dens[b])
            light = min(left, right) < min_mass
            if shallow or light:
                peaks.pop(j + 1 if dens[b] < dens[a] else j)
                changed = True
                break

    splits = []
    for a, b in zip(peaks, peaks[1:], strict=False):
        v = min(range(a, b + 1), key=lambda i: dens[i])
        splits.append(
            {
                "valley_at": back(pts[v]),
                "modes_at": [back(pts[a]), back(pts[b])],
                "valley_over_smaller_peak": dens[v] / min(dens[a], dens[b]),
                "mass_below_valley": sum(dens[:v]) / total,
            }
        )
    return {"n_modes": max(1, len(peaks)), "bandwidth": h, "splits": splits, "log_scale": use_log}


def detect_bimodal(
    xs: Sequence[float], metric: str, min_n: int = 30, **kw: Any
) -> list[dict[str, Any]]:
    if len(xs) < min_n:
        return []
    res = kde_modes(xs, **kw)
    if res["n_modes"] < 2:
        return []
    bc = sarle_bc(xs)
    sp = res["splits"][0]
    return [
        {
            "kind": f"bimodal_{metric.removesuffix('_ms')}",
            "detail": (
                f"{metric}: {res['n_modes']} KDE modes near "
                + ", ".join(f"{m:.1f}" for m in sp["modes_at"])
                + f"; valley at {sp['valley_at']:.1f} holds "
                f"{sp['valley_over_smaller_peak']:.2f}x the smaller peak; "
                f"{sp['mass_below_valley']:.0%} of mass below it; Sarle BC "
                + (f"{bc:.3f}" if bc is not None else "n/a")
                + f" (n={len(xs)})"
            ),
            "metric": metric,
            "n_modes": res["n_modes"],
            "sarle_bc": bc,
            "splits": res["splits"],
        }
    ]


# ---------------------------------------------------------------------------
# Sparse tail outliers
# ---------------------------------------------------------------------------
# A few isolated samples far above the median, too few to form a KDE mode,
# can still own the p99: in pilot job 13918362 two of 100 SGLang TTFTs at
# concurrency 1 were ~115 ms against a 12 ms median, mid-run (not warmup), and
# set p99 alone. This flags that case so a p99 is not read as typical behavior.
OUTLIER_FACTOR = 5.0  # sample > factor x median
OUTLIER_MAX_FRACTION = 0.05  # more than this is a mode/tail, not sparse outliers


def detect_tail_outliers(
    xs_in_order: Sequence[float],
    metric: str,
    factor: float = OUTLIER_FACTOR,
    max_fraction: float = OUTLIER_MAX_FRACTION,
    min_n: int = 30,
) -> list[dict[str, Any]]:
    xs = list(xs_in_order)
    if len(xs) < min_n:
        return []
    med = sorted(xs)[len(xs) // 2]
    if med <= 0:
        return []
    idx = [i for i, x in enumerate(xs) if x > factor * med]
    if not idx or len(idx) / len(xs) > max_fraction:
        return []
    return [
        {
            "kind": f"tail_outliers_{metric.removesuffix('_ms')}",
            "detail": (
                f"{metric}: {len(idx)}/{len(xs)} samples > {factor:g}x the median "
                f"({med:.1f}); max {max(xs):.1f} at positions {idx[:10]} in send "
                "order. p99 may be set by these alone."
            ),
            "metric": metric,
            "median": med,
            "positions": idx,
            "values": [xs[i] for i in idx],
        }
    ]


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def detect_warmup(
    xs_in_order: Sequence[float],
    metric: str,
    ratio: float = 1.5,
    abs_floor_ms: float = 5.0,
    min_rest: int = 10,
) -> list[dict[str, Any]]:
    n = len(xs_in_order)
    k = max(5, n // 10)
    if n - k < min_rest:
        return []
    first = statistics.median(xs_in_order[:k])
    rest = statistics.median(xs_in_order[k:])
    if first <= 0 or rest <= 0:
        return []
    r = first / rest
    if (r > ratio or r < 1.0 / ratio) and abs(first - rest) > abs_floor_ms:
        return [
            {
                "kind": f"warmup_{metric.removesuffix('_ms')}",
                "detail": (
                    f"{metric}: median of first {k} steady requests {first:.1f} vs "
                    f"{rest:.1f} for the remaining {n - k} ({r:.2f}x); the steady "
                    "window still contains a ramp"
                ),
                "metric": metric,
                "first_k": k,
                "first_median": first,
                "rest_median": rest,
                "ratio": r,
            }
        ]
    return []


# ---------------------------------------------------------------------------
# Rep-to-rep spread (aggregate time)
# ---------------------------------------------------------------------------


def _dig(d: dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def rep_spread(
    artifacts: Sequence[dict[str, Any]], threshold: float = DEFAULT_CV_THRESHOLD
) -> list[dict[str, Any]]:
    """CV across reps of one (workload, arm, point). Needs >= 2 reps."""
    out = []
    for name, path in REP_SPREAD_METRICS.items():
        vals = [
            v
            for v in (_dig(a, path) for a in artifacts)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        ]
        if len(vals) < 2:
            continue
        mean = statistics.fmean(vals)
        if mean == 0:
            continue
        cv = statistics.pstdev(vals) / abs(mean)
        if cv > threshold:
            out.append(
                {
                    "kind": f"rep_spread_{name}",
                    "detail": (
                        f"{name} CV {cv:.3f} > {threshold:.2f} across {len(vals)} reps "
                        f"(values {', '.join(f'{v:.4g}' for v in vals)})"
                    ),
                    "metric": name,
                    "cv": cv,
                    "values": vals,
                }
            )
    return out


# ---------------------------------------------------------------------------
# GPU clock / temperature
# ---------------------------------------------------------------------------


def detect_gpu_drift(
    samples: Sequence[dict[str, Any]],
    clock_drop: float = 0.10,
    temp_limit_c: float = 85.0,
) -> list[dict[str, Any]]:
    """`samples`: [{"t": s, "sm_clock_mhz": .., "temp_c": .., "power_w": ..}, ...]"""
    out: list[dict[str, Any]] = []
    clocks = [s["sm_clock_mhz"] for s in samples if isinstance(s.get("sm_clock_mhz"), (int, float))]
    if len(clocks) >= 8:
        q = len(clocks) // 4
        head, tail = statistics.fmean(clocks[:q]), statistics.fmean(clocks[-q:])
        if head > 0 and (head - tail) / head > clock_drop:
            out.append(
                {
                    "kind": "gpu_clock_drift",
                    "detail": (
                        f"SM clock fell from {head:.0f} to {tail:.0f} MHz "
                        f"({(head - tail) / head:.0%}) between first and last quarter"
                    ),
                    "head_mhz": head,
                    "tail_mhz": tail,
                }
            )
    temps = [s["temp_c"] for s in samples if isinstance(s.get("temp_c"), (int, float))]
    if temps and max(temps) >= temp_limit_c:
        out.append(
            {
                "kind": "gpu_temperature",
                "detail": f"GPU reached {max(temps):.0f} C (limit {temp_limit_c:.0f} C)",
                "max_temp_c": max(temps),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Per-run bundle
# ---------------------------------------------------------------------------


def run_anomalies(
    samples_in_order: dict[str, Sequence[float]],
    gpu_samples: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Everything that can be judged from ONE run. Rep spread is added later."""
    out: list[dict[str, Any]] = []
    for metric in BIMODAL_METRICS:
        xs = samples_in_order.get(metric) or []
        out += detect_bimodal(list(xs), metric)
    for metric in ("ttft_ms", "tpot_ms"):
        out += detect_warmup(list(samples_in_order.get(metric) or []), metric)
        out += detect_tail_outliers(list(samples_in_order.get(metric) or []), metric)
    if gpu_samples:
        out += detect_gpu_drift(gpu_samples)
    return out
