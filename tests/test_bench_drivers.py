"""
Tests for the Phase 3/4/5 benchmark drivers. PURE CPU: no GPU, no server, no
network. Every driver is structured so the logic that decides what may be
published is reachable without either.

WHAT IS ACTUALLY PROTECTED HERE
-------------------------------
Three properties, each of which has a specific failure this repo already lived
through or wrote down in advance:

  1. **Percentiles come from RAW SAMPLES.** `bench/run_p2.py` originally did
     `scalars.get('ttft_ms_p50', 0)`, silently got the default, and printed 0.0 in
     every row of a seven-rate sweep. `test_percentiles_are_read_from_samples_*`
     asserts the drivers read `artifact.samples` and that a plausible-looking
     scalar of the same name is never consulted.

  2. **A zero is not a measurement.** A preemption benchmark whose server never
     preempted must be INVALID, because a table of zeros reads as "preemption is
     free" rather than "preemption did not happen".

  3. **A missing number renders as `n/a`, never as 0.0.** NaN, None and an empty
     table all have to survive formatting visibly.

    python3 -m pytest tests/test_bench_drivers.py -q
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import time

import httpx
import pytest

from bench import driver_common as dc
from bench import run_p3, run_p4, run_p5
from bench.loadgen import LoadGenConfig, Outcome, Phase, analyze
from serving.metrics.artifact import REGISTRY, Artifact, Provenance

DRIVERS = ["bench.run_p3", "bench.run_p4", "bench.run_p5"]


def _artifact(name: str = "t", **samples) -> Artifact:
    art = Artifact(name=name, provenance=Provenance.capture(repo_root=".", seed=1))
    art.window = {"duration_s": 1.0}
    for metric, values in samples.items():
        art.add_samples(metric, list(values))
    return art


# ---------------------------------------------------------------------------
# 1. The drivers import, and their CLIs work as modules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", DRIVERS)
def test_driver_imports(module: str) -> None:
    m = importlib.import_module(module)
    assert callable(m.main)
    assert callable(m.build_parser)


@pytest.mark.parametrize("module", DRIVERS)
def test_help_exits_zero(module: str, capsys) -> None:
    """`python3 -m bench.run_pN --help` must work — module entry, not script."""
    m = importlib.import_module(module)
    with pytest.raises(SystemExit) as exc:
        m.build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower()


@pytest.mark.parametrize("module", DRIVERS)
def test_parser_has_seed_and_results_dir(module: str) -> None:
    """Provenance needs a seed; artifacts need somewhere to land."""
    m = importlib.import_module(module)
    dests = {a.dest for a in m.build_parser()._actions}
    assert {"seed", "results_dir"} <= dests


def test_p3_argparse_defaults_and_lengths() -> None:
    args = run_p3.build_parser().parse_args([])
    lengths = [int(x) for x in args.lengths.split(",")]
    assert len(lengths) >= 3, "the sequence-length sweep is the point of P3"
    assert args.swap_url is None and args.control_url is None


def test_p4_zero_structure_is_prepended_and_cannot_be_dropped() -> None:
    assert run_p4.resolve_structures("system,conversational")[0] == "zero"
    assert run_p4.resolve_structures("zero") == ["zero"]
    # All four generator structures survive the round trip.
    assert set(run_p4.resolve_structures("system,conversational,adversarial,zero")) == {
        "zero", "system", "conversational", "adversarial"
    }


def test_p4_rejects_unknown_structure() -> None:
    with pytest.raises(SystemExit):
        run_p4.resolve_structures("system,made_up")


def test_p5_requires_the_b5_baseline() -> None:
    """B5 is the real baseline; a run without it cannot support claim S5."""
    with pytest.raises(SystemExit) as exc:
        run_p5.parse_routers(["prefix_aware=http://a/v1/chat/completions"])
    assert "least_outstanding" in str(exc.value)

    with pytest.raises(SystemExit):
        run_p5.parse_routers(["least_outstanding=http://a/v1/chat/completions"])

    with pytest.raises(SystemExit):
        run_p5.parse_routers(["prefix_aware", "least_outstanding=http://b"])

    routers = run_p5.parse_routers([
        "prefix_aware=http://a/v1/chat/completions",
        "least_outstanding=http://b/v1/chat/completions",
        "round_robin=http://c/v1/chat/completions",
    ])
    assert set(routers) == {"prefix_aware", "least_outstanding", "round_robin"}


def test_p5_scenarios_cover_the_mandatory_losing_cases() -> None:
    """§10 commits to cases 1, 2 and 7 as workloads; 3 is the load axis."""
    refs = {s.methodology_ref for s in run_p5.SCENARIOS.values() if s.losing_case}
    assert {"§10.1", "§10.2", "§10.7"} <= refs
    assert "§10.3" in run_p5.ABOVE_KNEE_NOTE


# ---------------------------------------------------------------------------
# 2. Table formatting: empty, NaN, None
# ---------------------------------------------------------------------------


def test_fmt_num_renders_missing_as_na_not_zero() -> None:
    assert dc.fmt_num(None) == dc.NA
    assert dc.fmt_num(float("nan")) == dc.NA
    assert dc.fmt_num(float("inf")) == dc.NA
    assert dc.fmt_num(0.0) == "0.0"          # a real zero still prints as a zero
    assert dc.fmt_num(1.23456, 3) == "1.235"
    assert dc.fmt_num("recompute") == "recompute"


def test_render_table_handles_empty_rows() -> None:
    out = dc.render_table(["a", "b"], [])
    assert "no rows" in out
    assert out.strip() != ""


def test_render_table_handles_nan_and_none_cells() -> None:
    out = dc.render_table(
        ["length", "p99", "tax"],
        [[128, float("nan"), None], [256, 12.5, -3.25]],
        precisions=[0, 1, 2],
    )
    lines = out.splitlines()
    assert len(lines) == 4                     # header, rule, two rows
    assert lines[2].count(dc.NA) == 2
    assert "12.5" in lines[3] and "-3.25" in lines[3]
    assert "0.0" not in lines[2], "a missing measurement must never render as 0.0"


def test_render_table_ragged_row_does_not_crash() -> None:
    out = dc.render_table(["a", "b"], [[1, 2, 3]], precisions=[0])
    assert out.splitlines()[-1].strip().startswith("1")


# ---------------------------------------------------------------------------
# 3. Percentiles read SAMPLES, never scalars
# ---------------------------------------------------------------------------


def test_percentiles_are_read_from_samples_not_scalars() -> None:
    art = _artifact(ttft_ms=[10.0, 20.0, 30.0, 40.0])
    # A plausible-looking pre-computed scalar that must never be consulted. It is
    # deliberately WRONG: if any driver reads a scalar, the assertion below fails.
    art.set_scalar("goodput_rps", 999.0)
    assert dc.samples_percentile(art, "ttft_ms", 50) == pytest.approx(25.0)
    assert dc.samples_percentile(art, "ttft_ms", 99) == pytest.approx(39.7, abs=0.5)
    assert dc.sample_count(art, "ttft_ms") == 4


def test_missing_samples_yield_nan_never_a_default_zero() -> None:
    """The P2 bug in one assertion: `.get(name, 0)` would return 0.0 here."""
    art = _artifact()
    v = dc.samples_percentile(art, "ttft_ms", 50)
    assert math.isnan(v)
    assert dc.fmt_num(v) == dc.NA
    assert math.isnan(dc.scalar(art, "goodput_rps"))


def test_p3_armresult_percentiles_come_from_samples() -> None:
    art = _artifact(ttft_ms=[5.0, 15.0], e2e_ms=[100.0, 300.0])
    arm = run_p3.ArmResult(policy="recompute", length=512, artifact=art)
    assert arm.pct("ttft_ms", 50) == pytest.approx(10.0)
    assert arm.pct("e2e_ms", 50) == pytest.approx(200.0)
    assert math.isnan(arm.pct("itl_ms", 50))


def test_p5_cell_percentiles_come_from_samples() -> None:
    art = _artifact(ttft_ms=[1.0, 3.0])
    cell = run_p5.Cell(scenario="zero_sharing", load=4.0, policy="prefix_aware", artifact=art)
    assert cell.ttft(50) == pytest.approx(2.0)
    assert math.isnan(cell.goodput), "no scalar set -> NaN, not 0.0"


def test_p4_cell_ttft_delta_is_a_difference_of_sample_percentiles() -> None:
    on = _artifact("on", ttft_ms=[100.0, 100.0])
    off = _artifact("off", ttft_ms=[130.0, 130.0])
    cell = run_p4.CacheCell(structure="system", sharing_rate=0.5, on_art=on, off_art=off)
    assert cell.ttft_delta(50) == pytest.approx(-30.0)   # negative == cache helping
    cell_no_off = run_p4.CacheCell(structure="system", sharing_rate=0.5, on_art=on)
    assert math.isnan(cell_no_off.ttft_delta(50))


# ---------------------------------------------------------------------------
# 4. Validity — a zero-preemption P3 run is INVALID
# ---------------------------------------------------------------------------


def _delta(**kw) -> dict[str, float]:
    base = {
        "step": 1000.0, "preemptions_total": 0.0, "tokens_recomputed": 0.0,
        "bytes_swapped_out": 0.0, "bytes_swapped_in": 0.0, "resumes": 0.0,
        "starvation_fallbacks": 0.0, "swap_space_exhausted": 0.0,
    }
    base.update(kw)
    return base


def test_zero_preemption_run_is_invalid() -> None:
    v = run_p3.preemption_validity("recompute", _delta(preemptions_total=0.0))
    assert not v.valid
    assert any("ZERO PREEMPTIONS" in r for r in v.reasons)
    assert v.label == "INVALID"


def test_preemption_run_with_events_is_valid() -> None:
    v = run_p3.preemption_validity(
        "recompute", _delta(preemptions_total=17.0, tokens_recomputed=4096.0)
    )
    assert v.valid and v.reasons == []


def test_missing_preemption_counter_is_invalid_not_zero() -> None:
    v = run_p3.preemption_validity("recompute", _delta(preemptions_total=float("nan")))
    assert not v.valid
    assert any("did not report" in r for r in v.reasons)


def test_swap_arm_that_degraded_to_recompute_is_invalid() -> None:
    v = run_p3.preemption_validity(
        "swap", _delta(preemptions_total=9.0, bytes_swapped_out=1e6,
                       swap_space_exhausted=3.0)
    )
    assert not v.valid
    assert any("DEGRADED TO RECOMPUTE" in r for r in v.reasons)


def test_swap_arm_that_recomputed_tokens_is_a_mixture() -> None:
    v = run_p3.preemption_validity(
        "swap", _delta(preemptions_total=4.0, tokens_recomputed=512.0)
    )
    assert not v.valid
    assert any("mixture" in r for r in v.reasons)


def test_recompute_arm_that_swapped_bytes_is_a_mixture() -> None:
    v = run_p3.preemption_validity(
        "recompute", _delta(preemptions_total=4.0, bytes_swapped_out=1024.0)
    )
    assert not v.valid


def test_starvation_fallback_is_surfaced_as_an_admission_control_alarm() -> None:
    v = run_p3.preemption_validity(
        "recompute", _delta(preemptions_total=5.0, starvation_fallbacks=2.0)
    )
    assert not v.valid
    assert any("ADMISSION-CONTROL ALARM" in r for r in v.reasons)


def test_control_that_preempted_is_not_a_control() -> None:
    v = run_p3.control_validity(_delta(preemptions_total=3.0), 4096.0, 256.0)
    assert not v.valid
    assert any("CONTROL preempted" in r for r in v.reasons)


def test_control_pool_must_be_larger_than_the_pressure_pool() -> None:
    v = run_p3.control_validity(_delta(), 128.0, 256.0)
    assert not v.valid
    assert any("not\nlarger" in r.replace("  ", " ") or "KV pool" in r for r in v.reasons)


def test_clean_control_is_valid() -> None:
    assert run_p3.control_validity(_delta(), 4096.0, 256.0).valid


def test_latency_tax_needs_a_valid_control() -> None:
    arm = run_p3.ArmResult("recompute", 512, artifact=_artifact(ttft_ms=[200.0]))
    good = run_p3.ArmResult("control", 512, artifact=_artifact("c", ttft_ms=[50.0]))
    assert run_p3.latency_tax(arm, good, "ttft_ms", 50) == pytest.approx(150.0)
    assert math.isnan(run_p3.latency_tax(arm, None, "ttft_ms", 50))
    bad = run_p3.ArmResult("control", 512, artifact=_artifact("c", ttft_ms=[50.0]),
                           verdict=dc.Verdict.bad("preempted"))
    assert math.isnan(run_p3.latency_tax(arm, bad, "ttft_ms", 50))


def test_verdict_merge_is_sticky() -> None:
    v = dc.Verdict.ok().merge(dc.Verdict.bad("nope")).merge(dc.Verdict.ok())
    assert not v.valid and v.reasons == ["nope"]
    assert dc.Verdict.ok().merge(None).valid


# ---------------------------------------------------------------------------
# 5. P4 validity: the zero-sharing control, and degenerate workloads
# ---------------------------------------------------------------------------


def test_zero_sharing_control_with_hits_is_invalid() -> None:
    cell = run_p4.CacheCell(
        structure="zero", sharing_rate=0.0,
        delta={"cache_blocks_reused": 500.0, "cache_blocks_required": 1000.0},
    )
    v = run_p4.zero_sharing_validity(cell)
    assert not v.valid
    assert any("zero-sharing control measured" in r for r in v.reasons)


def test_zero_sharing_control_without_hits_is_valid() -> None:
    cell = run_p4.CacheCell(
        structure="zero", sharing_rate=0.0,
        delta={"cache_blocks_reused": 0.0, "cache_blocks_required": 1000.0},
    )
    assert run_p4.zero_sharing_validity(cell).valid


def test_zero_sharing_workload_really_has_no_sharing() -> None:
    """The generator's own control, checked through the driver's builder."""
    args = run_p4.build_parser().parse_args([])
    w = run_p4.build_workload("zero", 0.0, 48, args)
    assert w.realized["sharing"]["realized_block_sharing_rate"] == 0.0
    assert run_p4.workload_validity(w).valid


def test_degenerate_workload_is_invalid() -> None:
    args = run_p4.build_parser().parse_args(["--shared-prefix-tokens", "4",
                                             "--block-size", "16"])
    w = run_p4.build_workload("system", 1.0, 48, args)
    v = run_p4.workload_validity(w)
    assert not v.valid
    assert any("R14" in r for r in v.reasons)


def test_window_mean_recovers_a_windowed_mean_from_lifetime_gauges() -> None:
    before = {"cache_mean_partial_hit_depth": 2.0, "cache_lookups": 100}
    after = {"cache_mean_partial_hit_depth": 3.0, "cache_lookups": 200}
    # (3*200 - 2*100) / 100 == 4.0
    assert run_p4.window_mean(before, after, "cache_mean_partial_hit_depth",
                              "cache_lookups") == pytest.approx(4.0)


def test_window_mean_is_nan_when_nothing_was_looked_up() -> None:
    same = {"cache_mean_partial_hit_depth": 2.0, "cache_lookups": 100}
    assert math.isnan(run_p4.window_mean(same, dict(same),
                                         "cache_mean_partial_hit_depth", "cache_lookups"))
    assert math.isnan(run_p4.window_mean({}, {}, "x", "y"))


def test_hit_rate_is_computed_from_window_deltas() -> None:
    cell = run_p4.CacheCell(
        structure="system", sharing_rate=0.5,
        delta={"cache_blocks_reused": 300.0, "cache_blocks_required": 1200.0,
               "cache_requests_with_a_hit": 40.0, "cache_lookups": 50.0},
    )
    assert cell.block_hit_rate == pytest.approx(0.25)
    assert cell.request_hit_rate == pytest.approx(0.8)
    empty = run_p4.CacheCell(structure="system", sharing_rate=0.5)
    assert math.isnan(empty.block_hit_rate)


def test_overhead_statement_names_the_direction() -> None:
    on = _artifact("on", ttft_ms=[120.0])
    off = _artifact("off", ttft_ms=[100.0])
    cell = run_p4.CacheCell(structure="zero", sharing_rate=0.0, on_art=on, off_art=off,
                            delta={"cache_blocks_reused": 0.0,
                                   "cache_blocks_required": 10.0})
    text = " ".join(run_p4.overhead_statement([cell]))
    assert "COSTS" in text and "20.00 ms" in text
    assert "NOT RUN" in " ".join(run_p4.overhead_statement([]))


def test_overhead_statement_says_unmeasured_when_an_arm_is_missing() -> None:
    cell = run_p4.CacheCell(structure="zero", sharing_rate=0.0,
                            on_art=_artifact("on", ttft_ms=[120.0]))
    text = " ".join(run_p4.overhead_statement([cell]))
    assert "NOT MEASURED" in text and "not zero" in text


# ---------------------------------------------------------------------------
# 6. Curves: knee and crossover
# ---------------------------------------------------------------------------


def test_find_crossover_locates_the_sign_change() -> None:
    out = dc.find_crossover([1.0, 2.0, 3.0, 4.0], [3.0, 1.0, -1.0, -3.0])
    assert out["crossed"] is True
    assert out["crossover_x"] == pytest.approx(2.5)


def test_find_crossover_reports_never_ahead() -> None:
    out = dc.find_crossover([1.0, 2.0], [-1.0, -2.0])
    assert out["crossed"] is False
    assert "never ahead" in out["reason"]


def test_find_crossover_reports_no_crossing_in_range() -> None:
    out = dc.find_crossover([1.0, 2.0, 3.0], [5.0, 4.0, 3.0])
    assert out["crossed"] is False
    assert "NO CROSSOVER WAS FOUND" in out["reason"]


def test_find_crossover_needs_two_points() -> None:
    assert dc.find_crossover([1.0], [1.0])["crossed"] is False
    assert dc.find_crossover([], [])["crossover_x"] is None


def test_find_crossover_ignores_nan_points() -> None:
    out = dc.find_crossover([1.0, 2.0, 3.0], [2.0, float("nan"), -2.0])
    assert out["crossed"] is True
    assert out["crossover_x"] == pytest.approx(2.0)


def test_find_knee_reports_the_turn() -> None:
    loads = [1.0, 2.0, 4.0, 8.0, 16.0]
    goodput = [1.0, 2.0, 3.9, 4.0, 4.0]
    attain = [1.0, 1.0, 0.99, 0.5, 0.25]
    out = dc.find_knee(loads, goodput, attain)
    assert out["found"] is True
    assert out["knee_load_rps"] == 4.0


def test_find_knee_says_so_when_the_sweep_is_too_short() -> None:
    out = dc.find_knee([1.0, 2.0], [1.0, 2.0], [1.0, 1.0])
    assert out["found"] is False
    assert "KNEE WAS NOT FOUND" in out["reason"]


def test_find_knee_with_no_points() -> None:
    assert dc.find_knee([], [])["knee_load_rps"] is None


# ---------------------------------------------------------------------------
# 7. P5 analysis: losing cases, invalid cells, B4-only results
# ---------------------------------------------------------------------------


def _cell(scenario: str, load: float, policy: str, goodput: float, valid: bool = True):
    art = _artifact(f"{scenario}_{policy}_{load}", ttft_ms=[10.0, 20.0])
    art.set_scalar("goodput_rps", goodput)
    art.set_scalar("slo_attainment", 1.0)
    return run_p5.Cell(
        scenario=scenario, load=load, policy=policy, artifact=art,
        verdict=dc.Verdict.ok() if valid else dc.Verdict.bad("drift"),
    )


def test_advantage_series_excludes_invalid_cells() -> None:
    cells = [
        _cell("s", 1.0, "prefix_aware", 1.0), _cell("s", 1.0, "least_outstanding", 0.5),
        _cell("s", 2.0, "prefix_aware", 2.0, valid=False),
        _cell("s", 2.0, "least_outstanding", 1.0),
    ]
    xs, adv = run_p5.advantage_series(cells, "s", [1.0, 2.0], "prefix_aware",
                                      "least_outstanding")
    assert xs == [1.0]
    assert adv == [pytest.approx(0.5)]


def test_losing_case_prediction_held_is_stated_as_such() -> None:
    sc = run_p5.SCENARIOS["uniform_prefix"]
    v = run_p5.scenario_verdict(sc, [1.0, 2.0], [-0.1, -0.2], {"reason": "never ahead"})
    assert "PREDICTION HELD" in v and "§10.2" in v


def test_losing_case_that_won_is_flagged_not_celebrated() -> None:
    sc = run_p5.SCENARIOS["hot_prefix_skew"]
    v = run_p5.scenario_verdict(sc, [1.0, 2.0], [0.4, 0.2], {"reason": "no crossing"})
    assert "PREDICTION DID NOT HOLD" in v
    assert "investigating" in v


def test_unmeasured_scenario_is_not_reported_as_confirmed() -> None:
    sc = run_p5.SCENARIOS["zero_sharing"]
    v = run_p5.scenario_verdict(sc, [], [], {})
    assert "NOT MEASURED" in v and "not the same as" in v


def test_beats_b4_but_not_b5_is_published_in_those_words() -> None:
    cells = [
        _cell("s", 1.0, "prefix_aware", 1.0),
        _cell("s", 1.0, "least_outstanding", 1.5),
        _cell("s", 1.0, "round_robin", 0.5),
    ]
    msg = run_p5.b4_only_warning(cells, "s", [1.0])
    assert msg is not None
    assert "LOAD BALANCING DEMONSTRATED, PREFIX AWARENESS NOT" in msg


def test_b4_only_warning_silent_when_prefix_beats_b5() -> None:
    cells = [
        _cell("s", 1.0, "prefix_aware", 2.0),
        _cell("s", 1.0, "least_outstanding", 1.5),
        _cell("s", 1.0, "round_robin", 0.5),
    ]
    assert run_p5.b4_only_warning(cells, "s", [1.0]) is None


def test_p5_scenario_table_renders_with_a_missing_policy() -> None:
    cells = [_cell("s", 1.0, "least_outstanding", 1.0)]
    out = run_p5._scenario_table(cells, "s", [1.0], ["prefix_aware", "least_outstanding"])
    assert "least_outstanding" in out and "prefix_aware" not in out


# ---------------------------------------------------------------------------
# 8. Metric registry: new names are registered, existing ones are reused (R16)
# ---------------------------------------------------------------------------


def test_new_metrics_are_registered_with_unit_and_source() -> None:
    for name in ["preemption_latency_tax_ms", "tokens_recomputed", "bytes_swapped",
                 "resume_latency_ms", "partial_hit_depth_blocks", "cache_evictions",
                 "shared_blocks", "cache_overhead_ttft_ms", "goodput_delta_rps"]:
        spec = REGISTRY.get(name)
        assert spec.unit and spec.source


def test_reused_metric_names_keep_their_original_meaning() -> None:
    assert REGISTRY.get("cache_hit_rate").unit == "ratio"
    assert REGISTRY.get("preemption_rate").quantity == "preemption rate"
    assert REGISTRY.get("ttft_ms").source.startswith("client wall clock")


def test_registering_a_colliding_name_still_raises() -> None:
    from serving.metrics.artifact import MetricSpec

    with pytest.raises(ValueError):
        REGISTRY.register(MetricSpec("cache_evictions", "something else", "ms", "elsewhere"))


# ---------------------------------------------------------------------------
# 9. Schedule construction and open-loop dispatch, against a fake transport
# ---------------------------------------------------------------------------


def _cfg(**kw) -> LoadGenConfig:
    base = dict(
        url="http://fake/v1/chat/completions", rate_rps=20.0, duration_s=0.3,
        warmup_s=0.15, drain_s=0.15, seed=5, inflight_sample_interval_s=0.01,
    )
    base.update(kw)
    return LoadGenConfig(**base)


def test_specs_from_prompts_assigns_phases_and_intended_times() -> None:
    cfg = _cfg()
    prompts = [f"p{i}" for i in range(500)]
    specs = dc.specs_from_prompts(cfg, prompts, [4] * 500)
    assert specs, "the schedule must not be empty"
    assert [s.intended_send_time for s in specs] == sorted(s.intended_send_time for s in specs)
    phases = {s.phase for s in specs}
    assert Phase.STEADY in phases
    for s in specs:
        expected = (Phase.WARMUP if s.intended_send_time < cfg.steady_start_s
                    else Phase.STEADY if s.intended_send_time < cfg.steady_end_s
                    else Phase.DRAIN)
        assert s.phase == expected
    assert all(s.prompt.startswith("p") for s in specs)


def test_prompt_text_from_ids_preserves_shared_prefixes() -> None:
    a = dc.prompt_text_from_ids([1, 2, 3, 400])
    b = dc.prompt_text_from_ids([1, 2, 3, 999])
    common = a[:len(a) - len(a.split()[-1])]
    assert b.startswith(common)
    assert a != b


class _ScriptedSSE(httpx.AsyncBaseTransport):
    """Two content chunks then [DONE]; enough to produce TTFT, ITL and E2E."""

    def __init__(self) -> None:
        self.requests = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        json.loads(request.content)
        self.requests += 1

        async def gen():
            for text in ("a", "b"):
                await asyncio.sleep(0.005)
                yield (
                    b"data: " + json.dumps({
                        "choices": [{"index": 0, "delta": {"content": text},
                                     "finish_reason": None}]
                    }).encode() + b"\n\n"
                )
            yield (
                b"data: " + json.dumps({
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]
                }).encode() + b"\n\n"
            )
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=gen())


def test_open_loop_dispatch_produces_an_analyzable_run_with_no_server() -> None:
    """
    The P4/P5 dispatch path end to end against a scripted transport: the
    caller-supplied schedule flows through `stream_one` and out of `analyze()`
    with real samples, so the drivers' logic is exercisable without a GPU, a
    server, or a network.
    """
    cfg = _cfg()
    transport = _ScriptedSSE()

    async def go():
        client = httpx.AsyncClient(transport=transport)
        try:
            specs = dc.specs_from_prompts(cfg, [f"p{i}" for i in range(500)], [2] * 500)
            return await dc.open_loop_dispatch(cfg, specs, client=client)
        finally:
            await client.aclose()

    run = asyncio.run(go())
    assert transport.requests == len(run.results) > 0
    assert all(r.outcome == Outcome.COMPLETED for r in run.results)

    a = analyze(run)
    art = a.artifact
    assert art.samples["ttft_ms"], "TTFT samples must exist for the driver to read"
    # The percentile the drivers print comes from these samples, not from a scalar.
    assert dc.samples_percentile(art, "ttft_ms", 50) > 0
    assert dc.sample_count(art, "ttft_ms") == int(art.window["requests_steady"])


def test_open_loop_dispatch_does_not_await_responses() -> None:
    """
    Open loop means the dispatch loop never gates on a reply. A transport that
    holds every response open until the run is over would serialize a closed-loop
    harness; here all requests must be in flight simultaneously.
    """
    release = asyncio.Event()
    peak = {"n": 0, "cur": 0}

    class Holder(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await request.aread()

            async def gen():
                peak["cur"] += 1
                peak["n"] = max(peak["n"], peak["cur"])
                try:
                    await release.wait()
                    yield b"data: [DONE]\n\n"
                finally:
                    peak["cur"] -= 1

            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=gen())

    cfg = _cfg(rate_rps=200.0, duration_s=0.1, warmup_s=0.02, drain_s=0.02)

    async def go():
        client = httpx.AsyncClient(transport=Holder())
        specs = dc.specs_from_prompts(cfg, [f"p{i}" for i in range(200)], [1] * 200)

        async def releaser():
            await asyncio.sleep(0.3)
            release.set()

        task = asyncio.create_task(releaser())
        try:
            return await dc.open_loop_dispatch(cfg, specs, client=client)
        finally:
            await task
            await client.aclose()

    run = asyncio.run(go())
    assert peak["n"] > 1, "requests were serialized: this is a closed loop"
    assert len(run.results) > 1


# ---------------------------------------------------------------------------
# 10. Server-interaction helpers (pure parts only)
# ---------------------------------------------------------------------------


def test_base_url_of_strips_the_endpoint() -> None:
    assert dc.base_url_of("http://h:8000/v1/chat/completions") == "http://h:8000"
    assert dc.base_url_of("http://h:8000") == "http://h:8000"


def test_counter_delta_is_nan_for_missing_keys_not_zero() -> None:
    d = dc.counter_delta({"a": 1, "b": 2}, {"a": 5}, ["a", "b", "c"])
    assert d["a"] == 4.0
    assert math.isnan(d["b"]) and math.isnan(d["c"])


def test_counter_delta_ignores_booleans() -> None:
    d = dc.counter_delta({"enabled": False}, {"enabled": True}, ["enabled"])
    assert math.isnan(d["enabled"])


def test_write_summary_round_trips(tmp_path) -> None:
    path = dc.write_summary(tmp_path, "s.json", {"x": 1, "nan": float("nan")})
    assert path.exists()
    assert json.loads(path.read_text().replace("NaN", "null"))["x"] == 1


def test_banner_surfaces_publishability(capsys) -> None:
    prov = Provenance.capture(repo_root=".", seed=1)
    ok, blockers = dc.banner(prov, "TEST")
    out = capsys.readouterr().out
    assert "publishable" in out
    assert ok == (not blockers)
    if not ok:
        assert "NOT PUBLISHABLE" in out


def test_p3_prediction_is_printable_and_names_the_source() -> None:
    assert "ARCHITECTURE.md §5.2" in run_p3.PREDICTION
    assert "RECOMPUTE TO WIN AT NEARLY ALL" in run_p3.PREDICTION


def test_p3_prediction_verdict_reports_a_failed_prediction() -> None:
    lines = run_p3.prediction_verdict(
        [128, 512, 2048], [10.0, 20.0, 40.0], [30.0, 25.0, 20.0],
        {"reason": "crossing at 900"},
    )
    text = " ".join(lines)
    assert "DID NOT HOLD" in text and "crossing at 900" in text


def test_p3_prediction_verdict_reports_a_held_prediction() -> None:
    text = " ".join(run_p3.prediction_verdict(
        [128, 512], [10.0, 20.0], [30.0, 40.0], {"reason": "no crossing"}))
    assert "P1 HELD" in text


def test_p3_prediction_verdict_is_untested_without_two_lengths() -> None:
    text = " ".join(run_p3.prediction_verdict([128], [10.0], [20.0], {}))
    assert "UNTESTED" in text and "not evidence" in text


def test_timing_helpers_do_not_need_a_clock_source() -> None:
    """Sanity: nothing imported here starts a background loop or blocks."""
    t = time.perf_counter()
    dc.fmt_num(1.0)
    assert time.perf_counter() - t < 1.0


def test_argparse_namespace_is_all_the_drivers_need() -> None:
    """The workload builders take a plain Namespace: no server, no globals."""
    ns = run_p4.build_parser().parse_args([])
    assert isinstance(ns, argparse.Namespace)
    w = run_p4.build_workload("system", 0.5, 32, ns)
    assert len(w.requests) == 32
    sc = run_p5.SCENARIOS["zero_sharing"]
    ns5 = run_p5.build_parser().parse_args([
        "--router", "prefix_aware=http://a/v1/chat/completions",
        "--router", "least_outstanding=http://b/v1/chat/completions",
    ])
    w5 = run_p5.build_workload(sc, 16, ns5)
    assert len(w5.requests) == 16
    assert w5.realized["sharing"]["realized_block_sharing_rate"] == 0.0
