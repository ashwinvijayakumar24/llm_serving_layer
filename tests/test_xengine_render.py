"""Tests for bench/xengine/render.py — the no-fabrication renderer (SPEC hard rule 1)."""

from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path

import pytest

from bench.xengine import render as R


def make_artifact(
    arm="ours",
    workload="W1",
    point=None,
    rep=1,
    tok_s=100.0,
    ttft_p99=50.0,
    valid=True,
    reasons=None,
    gpu="NVIDIA H200",
    engine_version="0.1.0",
    anomalies=None,
    preemptions=None,
):
    point = {"concurrency": 16} if point is None else point
    engine_name = arm.split("-")[0]
    pct = {"p50": 1.0, "p95": 2.0, "p99": 3.0, "n": 10}
    return {
        "schema": "xengine-run/1",
        "arm": arm,
        "engine": {
            "name": engine_name,
            "version": engine_version,
            "launch_cmd": ["x"],
            "flags": {},
        },
        "workload": {
            "id": workload,
            "config_path": "c.yaml",
            "config_sha256": "a" * 64,
            "seed": 0,
            "point": point,
        },
        "rep": rep,
        "hardware": {
            "gpu_name": gpu,
            "gpu_count": 1,
            "driver": "550",
            "cuda": "12.9",
            "node": "n1",
            "slurm_job_id": "123",
        },
        "provenance": {
            "repo_sha": "deadbeef",
            "repo_dirty": False,
            "started_at": "t",
            "harness_version": "1",
        },
        "slo": {"ttft_ms": 434, "tpot_ms": 27.8, "source": "results/p2/RESULTS.md"},
        "validity": {
            "valid": valid,
            "reasons": reasons or [],
            "attention_backend": "flashinfer" if engine_name == "ours" else None,
        },
        "metrics": {
            "ttft_ms": {"p50": 10.0, "p95": 20.0, "p99": ttft_p99, "n": 10},
            "itl_ms": dict(pct),
            "tpot_ms": dict(pct),
            "e2e_ms": dict(pct),
            "output_tok_s": tok_s,
            "request_rps": 1.0,
            "goodput_rps": 1.0,
            "slo_attainment": 1.0,
            "completed": 10,
            "failed": 0,
        },
        "server_counters": {
            "preemptions": preemptions,
            "evictions": None,
            "prefix_hit_rate": None,
            "raw": {},
        },
        "anomalies": anomalies or [],
        "samples": {k: [1.0] * 10 for k in ("ttft_ms", "itl_ms", "tpot_ms")},
    }


def write(root: Path, art: dict, name: str | None = None) -> Path:
    pt = "_".join(f"{k}{v}" for k, v in art["workload"]["point"].items())
    p = root / art["workload"]["id"] / art["arm"] / (name or f"{pt}_rep{art['rep']}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(art))
    return p


SKELETON = """# Title

Hand-written intro that must survive.

<!-- BEGIN GENERATED: metadata -->
old
<!-- END GENERATED: metadata -->

Hand-written middle paragraph.

<!-- BEGIN GENERATED: w1_table -->
old
<!-- END GENERATED: w1_table -->

<!-- BEGIN GENERATED: inventory -->
old
<!-- END GENERATED: inventory -->

<!-- BEGIN GENERATED: something_else -->
keep me
<!-- END GENERATED: something_else -->

## Analysis

Hand-written analysis that must survive.
"""


def cell(grid, workload="W1", arm="ours", point=None):
    point = {"concurrency": 16} if point is None else point
    return grid.get((workload, arm, R.point_key(point)))


# --------------------------------------------------------------------------


def test_aggregation_mean_min_max_stdev(tmp_path):
    vals = [100.0, 110.0, 120.0]
    for i, v in enumerate(vals, 1):
        write(tmp_path, make_artifact(rep=i, tok_s=v))
    grid = R.aggregate(R.load_runs(tmp_path))
    c = cell(grid)
    assert c.n_reps == 3
    st = c.stat("output_tok_s")
    assert st.mean == pytest.approx(110.0)
    assert (st.min, st.max) == (100.0, 120.0)
    assert st.stdev == pytest.approx(statistics.stdev(vals))
    text = R.fmt_cell(c, "output_tok_s", cv_threshold=0.5)
    assert text.startswith("110 ± 10.0 (100–120)")
    assert "[" not in text  # no flags: 3 reps, low CV


def test_reps_grouped_per_arm_and_point(tmp_path):
    write(tmp_path, make_artifact(arm="ours", rep=1, tok_s=1.0))
    write(tmp_path, make_artifact(arm="vllm", rep=1, tok_s=2.0))
    write(tmp_path, make_artifact(arm="ours", point={"concurrency": 32}, rep=1, tok_s=3.0))
    grid = R.aggregate(R.load_runs(tmp_path))
    assert cell(grid, arm="ours").stat("output_tok_s").mean == 1.0
    assert cell(grid, arm="vllm").stat("output_tok_s").mean == 2.0
    assert cell(grid, point={"concurrency": 32}).stat("output_tok_s").mean == 3.0


def test_missing_cell_renders_todo(tmp_path):
    for i in (1, 2, 3):
        write(tmp_path, make_artifact(arm="ours", rep=i))
    grid = R.aggregate(R.load_runs(tmp_path))
    table = R.render_workload("W1", grid, 0.1)
    rows = [ln for ln in table.splitlines() if ln.startswith("| concurrency=16 |")]
    baseline_row = rows[0]
    cols = [c.strip() for c in baseline_row.strip("|").split("|")]
    assert cols[1].startswith("100")  # ours
    assert cols[2] == "TODO" and cols[3] == "TODO"  # vllm, sglang: no artifacts
    # A W1 point with no artifact at all still shows up, as TODO everywhere.
    c1 = [ln for ln in table.splitlines() if ln.startswith("| concurrency=1 |")][0]
    assert set(c.strip() for c in c1.strip("|").split("|")[1:]) == {"TODO"}


def test_invalid_runs_excluded_and_listed(tmp_path):
    write(tmp_path, make_artifact(rep=1, tok_s=100.0))
    write(tmp_path, make_artifact(rep=2, tok_s=100.0))
    write(tmp_path, make_artifact(rep=3, tok_s=100.0))
    write(tmp_path, make_artifact(rep=4, tok_s=9999.0, valid=False, reasons=["server crashed"]))
    # Our arm on a PagedTorch fallback is invalid even if the harness said valid.
    bad = make_artifact(rep=5, tok_s=8888.0)
    bad["validity"]["attention_backend"] = "paged_torch"
    write(tmp_path, bad)
    # Structurally broken artifacts and garbage files are listed, never aggregated.
    broken = make_artifact(rep=6, tok_s=7777.0)
    del broken["metrics"]["output_tok_s"]
    write(tmp_path, broken)
    (tmp_path / "W1" / "ours" / "garbage.json").write_text("{not json")

    runs = R.load_runs(tmp_path)
    grid = R.aggregate(runs)
    c = cell(grid)
    assert c.n_reps == 3
    assert c.stat("output_tok_s").mean == 100.0
    assert len(c.invalid_runs) == 2
    assert "2 invalid excluded" in R.fmt_cell(c, "output_tok_s", 0.1)

    inv = R.render_inventory(runs)
    assert "7 artifacts, 3 valid, 4 invalid" in inv
    assert "server crashed" in inv
    assert "paged_torch" in inv
    assert "metrics.output_tok_s" in inv
    assert "unreadable" in inv
    assert "garbage.json" in inv


def test_all_invalid_cell_is_todo_with_flag(tmp_path):
    write(tmp_path, make_artifact(rep=1, valid=False, reasons=["oom"]))
    grid = R.aggregate(R.load_runs(tmp_path))
    text = R.fmt_cell(cell(grid), "output_tok_s", 0.1)
    assert text.startswith("TODO")
    assert "1 invalid excluded" in text


def test_n_reps_below_three_flagged(tmp_path):
    write(tmp_path, make_artifact(rep=1))
    write(tmp_path, make_artifact(rep=2))
    grid = R.aggregate(R.load_runs(tmp_path))
    assert "n=2<3" in R.fmt_cell(cell(grid), "output_tok_s", 0.1)


def test_high_cv_flagged(tmp_path):
    for i, v in enumerate([50.0, 100.0, 150.0], 1):
        write(tmp_path, make_artifact(rep=i, tok_s=v))
    grid = R.aggregate(R.load_runs(tmp_path))
    assert "CV 50%" in R.fmt_cell(cell(grid), "output_tok_s", 0.1)
    assert "CV" not in R.fmt_cell(cell(grid), "output_tok_s", 0.9)


def test_anomalies_propagated(tmp_path):
    for i in (1, 2, 3):
        an = [{"kind": "bimodal_ttft", "detail": "two modes"}] if i == 2 else []
        write(tmp_path, make_artifact(rep=i, anomalies=an))
    runs = R.load_runs(tmp_path)
    grid = R.aggregate(runs)
    assert "anomaly: bimodal_ttft" in R.fmt_cell(cell(grid), "ttft_p99", 0.1)
    assert "two modes" in R.render_inventory(runs)


def test_null_counter_never_becomes_zero(tmp_path):
    for i in (1, 2, 3):
        write(tmp_path, make_artifact(arm="vllm", workload="W4", point={"rate": 4}, rep=i))
    grid = R.aggregate(R.load_runs(tmp_path))
    c = cell(grid, workload="W4", arm="vllm", point={"rate": 4})
    text = R.fmt_cell(c, "preemptions", 0.1)
    assert text.startswith("n/a (not exposed)")
    assert "0" not in text.split("**")[0]


def test_version_mismatch_flagged(tmp_path):
    write(tmp_path, make_artifact(arm="vllm", rep=1, engine_version="0.9.0"))
    write(tmp_path, make_artifact(arm="vllm", rep=2, engine_version="0.9.1"))
    write(tmp_path, make_artifact(arm="ours", rep=1, gpu="NVIDIA H100"))
    md = R.render_metadata(R.load_runs(tmp_path))
    assert "WARNING — METADATA MISMATCH" in md
    assert "engine vllm version differs" in md
    assert "GPU differs" in md


def test_consistent_metadata_has_no_warning(tmp_path):
    for i in (1, 2, 3):
        write(tmp_path, make_artifact(rep=i))
    md = R.render_metadata(R.load_runs(tmp_path))
    assert "WARNING" not in md
    assert "NVIDIA H200" in md


def test_rewrite_preserves_handwritten_text():
    new, missing = R.rewrite_generated(
        SKELETON, {"metadata": "META", "w1_table": "TABLE\nROW", "nope": "x"}
    )
    assert missing == ["nope"]
    assert "META" in new and "TABLE\nROW" in new
    for keep in (
        "Hand-written intro that must survive.",
        "Hand-written middle paragraph.",
        "Hand-written analysis that must survive.",
        "keep me",  # region the renderer does not own is untouched
    ):
        assert keep in new
    # inventory not in the regions dict -> left as-is
    assert "<!-- BEGIN GENERATED: inventory -->\nold\n" in new
    # Idempotent: rendering twice gives the same text.
    again, _ = R.rewrite_generated(new, {"metadata": "META", "w1_table": "TABLE\nROW"})
    assert again == new
    # Outside the regions the text is byte-identical.
    strip = R._REGION_RE.sub("", SKELETON)
    assert R._REGION_RE.sub("", new) == strip


def test_rewrite_handles_backslashes_in_content():
    new, _ = R.rewrite_generated(SKELETON, {"metadata": r"a \1 \g<0> b"})
    assert r"a \1 \g<0> b" in new


def test_zero_artifacts_all_todo(tmp_path):
    doc_src = Path(__file__).resolve().parents[1] / "docs" / "xengine" / "BENCHMARKS.md"
    doc = tmp_path / "BENCHMARKS.md"
    doc.write_text(doc_src.read_text())
    figs = tmp_path / "figures"
    missing = R.render(tmp_path / "no_results", doc, figs)
    assert missing == []
    out = doc.read_text()
    for m in R._REGION_RE.finditer(out):
        body = m.group("body")
        assert "TODO" in body, m.group("name")
        # no numbers rendered from nowhere: table cells are all TODO
        for line in body.splitlines():
            if line.startswith("| ") and not line.startswith("| point"):
                assert set(c.strip() for c in line.strip("|").split("|")[1:]) == {"TODO"}
    assert not any(figs.glob("*.png"))


def test_full_render_with_data_draws_figures(tmp_path):
    res = tmp_path / "results"
    for arm in ("ours", "vllm", "vllm-eager"):
        for c in (1, 2, 4):
            for rep in (1, 2, 3):
                write(res, make_artifact(arm=arm, point={"concurrency": c}, rep=rep))
    for arm in ("ours", "ours-noprefix", "sglang"):
        for rep in (1, 2, 3):
            write(res, make_artifact(arm=arm, workload="W3", point={"rate": 2}, rep=rep))
    for rep in (1, 2, 3):
        write(
            res, make_artifact(arm="ours", workload="W4", point={"rate": 2}, rep=rep, preemptions=5)
        )
    doc = tmp_path / "BENCHMARKS.md"
    doc.write_text(
        SKELETON
        + "\n<!-- BEGIN GENERATED: fig_w1_throughput -->\n"
        + "<!-- END GENERATED: fig_w1_throughput -->\n"
    )
    figs = tmp_path / "figures"
    R.render(res, doc, figs)
    for name in R.FIGURES.values():
        assert (figs / name).exists(), name
    out = doc.read_text()
    assert "![w1_throughput](figures/" in out
    assert "Hand-written analysis that must survive." in out


def test_stale_figure_removed_when_data_gone(tmp_path):
    figs = tmp_path / "figures"
    figs.mkdir()
    stale = figs / R.FIGURES["w1_throughput"]
    stale.write_bytes(b"old")
    R.render_figures({}, figs)
    assert not stale.exists()


def test_validate_artifact_rejects_wrong_schema():
    a = make_artifact()
    assert R.validate_artifact(a) == []
    b = copy.deepcopy(a)
    b["schema"] = "xengine-run/0"
    assert any("schema" in e for e in R.validate_artifact(b))
    c = copy.deepcopy(a)
    c["metrics"]["ttft_ms"]["p99"] = float("nan")
    assert any("ttft_ms.p99" in e for e in R.validate_artifact(c))
    assert R.validate_artifact([]) == ["artifact is not a JSON object"]


def _scaling_grid(tmp_path: Path, base_vals, c16_vals):
    for rep, v in enumerate(base_vals, 1):
        write(tmp_path, make_artifact(point={"concurrency": 1}, rep=rep, tok_s=v))
    for rep, v in enumerate(c16_vals, 1):
        write(tmp_path, make_artifact(point={"concurrency": 16}, rep=rep, tok_s=v))
    return R.aggregate(R.load_runs(tmp_path))


def test_w1_scaling_is_ratio_of_measured_means(tmp_path):
    grid = _scaling_grid(tmp_path, [100.0, 100.0, 100.0], [700.0, 800.0, 900.0])
    out = R.render_w1_scaling(grid, R.DEFAULT_CV_THRESHOLD)
    assert "DERIVED, not measured" in out
    row16 = next(line for line in out.splitlines() if line.startswith("| concurrency=16"))
    # ours = 800/100; vllm and sglang have no artifacts -> TODO, never a value
    assert row16.split("|")[2].strip().startswith("8.00x")
    assert row16.split("|")[3].strip() == "TODO"
    assert row16.split("|")[4].strip() == "TODO"


def test_w1_scaling_todo_without_batch1_base(tmp_path):
    for rep in (1, 2, 3):
        write(tmp_path, make_artifact(point={"concurrency": 16}, rep=rep, tok_s=800.0))
    grid = R.aggregate(R.load_runs(tmp_path))
    out = R.render_w1_scaling(grid, R.DEFAULT_CV_THRESHOLD)
    row16 = next(line for line in out.splitlines() if line.startswith("| concurrency=16"))
    assert row16.split("|")[2].strip() == "TODO"


def test_w1_scaling_carries_base_flags(tmp_path):
    grid = _scaling_grid(tmp_path, [100.0], [800.0, 800.0, 800.0])
    out = R.render_w1_scaling(grid, R.DEFAULT_CV_THRESHOLD)
    row16 = next(line for line in out.splitlines() if line.startswith("| concurrency=16"))
    assert "base: n=1<3" in row16


def test_tail_percentile_flagged_when_samples_few(tmp_path):
    for rep in (1, 2, 3):
        write(tmp_path, make_artifact(rep=rep))  # fixture: ttft n = 10
    grid = R.aggregate(R.load_runs(tmp_path))
    c = cell(grid)
    assert "p99 from n=10<100" in R.fmt_cell(c, "ttft_p99", 0.1)
    assert "p99 from n=" not in R.fmt_cell(c, "output_tok_s", 0.1)


def test_pool_held_by_cache_flag(tmp_path):
    for rep in (1, 2, 3):
        a = make_artifact(rep=rep)
        a["server_counters"]["raw"] = {
            "before": {
                "scheduler": {"running": 0, "waiting": 0, "blocks_free": 3, "blocks_used": 997}
            }
        }
        write(tmp_path, a)
    c = cell(R.aggregate(R.load_runs(tmp_path)))
    assert "pool_held_by_cache_at_start" in R.fmt_cell(c, "output_tok_s", 0.1)
