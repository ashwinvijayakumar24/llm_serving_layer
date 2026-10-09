"""Anomaly detectors on synthetic data (bench/xengine/anomalies.py)."""

from __future__ import annotations

import random

import pytest

from bench.xengine import anomalies as an


def _mix(seed: int, n_major: int, n_minor: int) -> list[float]:
    r = random.Random(seed)
    return [r.gauss(50, 5) for _ in range(n_major)] + [r.gauss(200, 15) for _ in range(n_minor)]


@pytest.mark.parametrize("seed", range(5))
def test_bimodal_mixture_is_flagged(seed):
    found = an.detect_bimodal(_mix(seed, 270, 30), "ttft_ms")
    assert len(found) == 1
    a = found[0]
    assert a["kind"] == "bimodal_ttft"
    assert a["n_modes"] == 2
    lo, hi = a["splits"][0]["modes_at"]
    assert 35 < lo < 65 and 150 < hi < 250
    assert a["sarle_bc"] > 5 / 9  # supporting evidence agrees here


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "draw",
    [
        lambda r: r.gauss(100, 10),
        lambda r: r.lognormvariate(4, 1.0),  # skewed: Sarle's BC alone would fire
        lambda r: r.expovariate(0.1),
        lambda r: r.uniform(10, 20),
    ],
    ids=["normal", "lognormal", "exponential", "uniform"],
)
def test_unimodal_shapes_are_not_flagged(seed, draw):
    r = random.Random(seed)
    xs = [draw(r) for _ in range(300)]
    assert an.detect_bimodal(xs, "ttft_ms") == []


def test_sarle_bc_alone_would_misfire_on_skew():
    """Why the KDE test exists: BC crosses 5/9 on a unimodal lognormal."""
    r = random.Random(1)
    xs = [r.lognormvariate(4, 1.0) for _ in range(500)]
    assert an.sarle_bc(xs) > 5 / 9
    assert an.kde_modes(xs)["n_modes"] == 1


def test_small_samples_and_constants_are_not_judged():
    assert an.detect_bimodal(_mix(0, 10, 10), "ttft_ms") == []  # n < 30
    assert an.detect_bimodal([5.0] * 100, "ttft_ms") == []
    assert an.sarle_bc([1.0, 2.0]) is None


def test_tiny_minor_population_is_a_tail_not_a_mode():
    """2% outliers far away: below min_mass, not reported as a mode."""
    assert an.detect_bimodal(_mix(3, 294, 6), "e2e_ms") == []


def test_warmup_effect_detected_and_absent():
    r = random.Random(0)
    slow_start = [r.gauss(300, 10) for _ in range(10)] + [r.gauss(100, 10) for _ in range(90)]
    found = an.detect_warmup(slow_start, "ttft_ms")
    assert found and found[0]["kind"] == "warmup_ttft"
    assert found[0]["ratio"] > 2.5

    flat = [r.gauss(100, 10) for _ in range(100)]
    assert an.detect_warmup(flat, "ttft_ms") == []
    # ratio large but absolute gap tiny -> not a warmup effect worth flagging
    tiny = [0.4] * 10 + [0.2] * 90
    assert an.detect_warmup(tiny, "tpot_ms") == []
    assert an.detect_warmup([1.0] * 12, "ttft_ms") == []  # too few to judge


def _art(tok_s: float, ttft: float = 50.0) -> dict:
    return {
        "metrics": {
            "output_tok_s": tok_s,
            "goodput_rps": 1.0,
            "ttft_ms": {"p50": ttft},
            "tpot_ms": {"p50": 9.0},
        }
    }


def test_rep_spread_cv_threshold():
    tight = [_art(1000), _art(1010), _art(990)]
    assert an.rep_spread(tight) == []
    loose = [_art(1000), _art(1300), _art(700)]
    found = an.rep_spread(loose)
    assert [a["kind"] for a in found] == ["rep_spread_output_tok_s"]
    assert found[0]["cv"] > 0.10
    assert an.rep_spread([_art(1000)]) == []  # one rep: no spread
    # None metrics are skipped, not treated as 0
    assert an.rep_spread([_art(1000, ttft=None), _art(1000, ttft=None)]) == []


def test_gpu_drift():
    steady = [{"sm_clock_mhz": 1980.0, "temp_c": 60.0} for _ in range(20)]
    assert an.detect_gpu_drift(steady) == []
    throttled = steady[:10] + [{"sm_clock_mhz": 1500.0, "temp_c": 88.0} for _ in range(10)]
    kinds = {a["kind"] for a in an.detect_gpu_drift(throttled)}
    assert kinds == {"gpu_clock_drift", "gpu_temperature"}
    assert an.detect_gpu_drift([]) == []


def test_run_anomalies_bundle_skips_itl():
    xs = _mix(1, 270, 30)
    found = an.run_anomalies({"ttft_ms": xs, "itl_ms": xs, "tpot_ms": [], "e2e_ms": []})
    kinds = [a["kind"] for a in found]
    assert "bimodal_ttft" in kinds
    assert not any("itl" in k for k in kinds)
    assert all(isinstance(a["detail"], str) for a in found)
