"""Render cross-engine benchmark artifacts into docs/xengine/BENCHMARKS.md.

Contract: docs/xengine/SPEC.md. Reads every ``results/xengine/**/*.json``
(schema ``xengine-run/1``), aggregates repetitions per workload x arm x point,
draws the figures into ``docs/xengine/figures/``, and rewrites ONLY the regions
of BENCHMARKS.md between ``<!-- BEGIN GENERATED: name -->`` and
``<!-- END GENERATED: name -->``. Hand-written text outside those markers is
never touched.

The one rule that matters (SPEC hard rule 1): every number this module prints is
an aggregate (mean / min / max / stdev / count) of values read from artifacts.
A cell with no valid artifact renders as ``TODO``. Nothing is interpolated,
extrapolated, estimated, or derived across cells.

Usage::

    python3 -m bench.xengine.render                       # defaults
    python3 -m bench.xengine.render --results results/xengine \\
        --doc docs/xengine/BENCHMARKS.md --figures docs/xengine/figures
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "xengine-run/1"
TODO = "TODO"
DEFAULT_CV_THRESHOLD = 0.10  # stdev / mean above this is flagged (SPEC hard rule 2)
MIN_REPS = 3  # SPEC hard rule 3

# --------------------------------------------------------------------------
# Arms and workloads from SPEC.md. Baselines and diagnostics are kept apart in
# every table because a diagnostic arm is never "the competitor" (hard rule 6).
# --------------------------------------------------------------------------
BASELINE_ARMS = ["ours", "vllm", "sglang"]
DIAGNOSTIC_ARMS = [
    "ours-noprefix",
    "vllm-eager",
    "vllm-nograph",
    "vllm-noasync",
    "vllm-noprefix",
    "vllm-matched",
    "sglang-noradix",
    "sglang-nooverlap",
    "sglang-eager",
    "sglang-lpm",
]
ALL_ARMS = BASELINE_ARMS + DIAGNOSTIC_ARMS
# W5 appendix arm (SPEC "Engines and arms").
INT8_ARM = "ours-int8"

WORKLOADS = ["W1", "W2", "W3", "W4"]


# Expected point grid per workload, read from the harness's workload YAMLs
# (bench/xengine/configs/workloads/W*.yaml `points:`), so a point that was never
# run shows as a TODO row rather than being silently absent.
def _expected_points() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    try:
        from bench.xengine.config import load_workload
    except Exception:  # renderer must still work without the harness deps
        return {"W1": [{"concurrency": c} for c in (1, 2, 4, 8, 16, 32, 64)]}
    for wl in WORKLOADS:
        try:
            out[wl] = load_workload(wl).points()
        except Exception:
            out[wl] = []
    return out


EXPECTED_POINTS: dict[str, list[dict[str, Any]]] = _expected_points()

# Prefix-cache on/off pairs for W3 (on-arm, off-arm).
PREFIX_PAIRS = [
    ("ours", "ours-noprefix"),
    ("vllm", "vllm-noprefix"),
    ("sglang", "sglang-noradix"),
]

# Metric paths aggregated per cell. A path is dotted into the artifact.
METRICS = {
    "output_tok_s": ("metrics", "output_tok_s"),
    "request_rps": ("metrics", "request_rps"),
    "goodput_rps": ("metrics", "goodput_rps"),
    "slo_attainment": ("metrics", "slo_attainment"),
    "ttft_p50": ("metrics", "ttft_ms", "p50"),
    "ttft_p99": ("metrics", "ttft_ms", "p99"),
    "tpot_p50": ("metrics", "tpot_ms", "p50"),
    "tpot_p99": ("metrics", "tpot_ms", "p99"),
    "itl_p99": ("metrics", "itl_ms", "p99"),
    "e2e_p99": ("metrics", "e2e_ms", "p99"),
    "failed": ("metrics", "failed"),
    # server_counters may be null ("engine does not expose it", never 0).
    "preemptions": ("server_counters", "preemptions"),
    "evictions": ("server_counters", "evictions"),
    "prefix_hit_rate": ("server_counters", "prefix_hit_rate"),
}
NULLABLE_METRICS = {"preemptions", "evictions", "prefix_hit_rate"}
# Tail percentiles and the distribution whose sample count backs them. With
# fewer than MIN_TAIL_SAMPLES samples, p99 is essentially the maximum, so the
# cell is flagged rather than presented as a stable tail.
TAIL_SOURCE = {
    "ttft_p99": "ttft_ms",
    "tpot_p99": "tpot_ms",
    "itl_p99": "itl_ms",
    "e2e_p99": "e2e_ms",
}
MIN_TAIL_SAMPLES = 100

METRIC_LABELS = {
    "output_tok_s": "output tok/s",
    "request_rps": "request rps",
    "goodput_rps": "goodput rps",
    "slo_attainment": "SLO attainment (fraction)",
    "ttft_p50": "TTFT p50 (ms)",
    "ttft_p99": "TTFT p99 (ms)",
    "tpot_p50": "TPOT p50 (ms)",
    "tpot_p99": "TPOT p99 (ms)",
    "itl_p99": "ITL p99 (ms)",
    "e2e_p99": "E2E p99 (ms)",
    "failed": "failed requests",
    "preemptions": "preemptions",
    "evictions": "evictions",
    "prefix_hit_rate": "prefix hit rate",
}

# Which metric tables each workload section shows.
WORKLOAD_TABLES = {
    "W1": ["output_tok_s", "ttft_p99", "tpot_p99"],
    "W2": ["goodput_rps", "slo_attainment", "ttft_p99", "tpot_p99"],
    "W3": ["ttft_p99", "goodput_rps", "prefix_hit_rate"],
    "W4": ["goodput_rps", "preemptions", "ttft_p99", "failed"],
}

# Figure file names (relative to the figures dir) and the region that links them.
FIGURES = {
    "w1_throughput": "w1_output_tok_s_vs_concurrency.png",
    "w1_ttft": "w1_ttft_p99_vs_concurrency.png",
    "w3_prefix": "w3_ttft_p99_prefix_on_off.png",
    "w4_goodput": "w4_goodput_and_preemptions.png",
}

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
# Minimal local check (schema tag, required keys, numeric types). When the
# harness is importable, load_runs also applies its stricter validate_artifact.
REQUIRED_TOP = [
    "schema",
    "arm",
    "engine",
    "workload",
    "rep",
    "hardware",
    "provenance",
    "slo",
    "validity",
    "metrics",
    "server_counters",
    "anomalies",
]
REQUIRED_PCTL = ["p50", "p95", "p99", "n"]
REQUIRED_SCALAR_METRICS = [
    "output_tok_s",
    "request_rps",
    "goodput_rps",
    "slo_attainment",
    "completed",
    "failed",
]
REQUIRED_DIST_METRICS = ["ttft_ms", "itl_ms", "tpot_ms", "e2e_ms"]


def _is_num(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def validate_artifact(d: Any) -> list[str]:
    """Return a list of problems; empty means structurally valid.

    Structural validity is separate from the run's own ``validity.valid`` flag:
    a well-formed artifact can still describe an invalid run.
    """
    if not isinstance(d, dict):
        return ["artifact is not a JSON object"]
    errs: list[str] = []
    if d.get("schema") != SCHEMA:
        errs.append(f"schema is {d.get('schema')!r}, expected {SCHEMA!r}")
    for k in REQUIRED_TOP:
        if k not in d:
            errs.append(f"missing key: {k}")
    if errs:
        return errs
    if not isinstance(d["arm"], str) or not d["arm"]:
        errs.append("arm must be a non-empty string")
    if not isinstance(d["rep"], int) or isinstance(d["rep"], bool):
        errs.append("rep must be an integer")
    wl = d["workload"]
    if not isinstance(wl, dict) or not isinstance(wl.get("id"), str):
        errs.append("workload.id missing")
    elif not isinstance(wl.get("point"), dict):
        errs.append("workload.point must be an object")
    val = d["validity"]
    if not isinstance(val, dict) or not isinstance(val.get("valid"), bool):
        errs.append("validity.valid must be a boolean")
    m = d["metrics"]
    if not isinstance(m, dict):
        return errs + ["metrics must be an object"]
    for k in REQUIRED_SCALAR_METRICS:
        if not _is_num(m.get(k)):
            errs.append(f"metrics.{k} missing or not a finite number")
    for k in REQUIRED_DIST_METRICS:
        dist = m.get(k)
        if not isinstance(dist, dict):
            errs.append(f"metrics.{k} missing")
            continue
        for p in REQUIRED_PCTL:
            if not _is_num(dist.get(p)):
                errs.append(f"metrics.{k}.{p} missing or not a finite number")
    sc = d["server_counters"]
    if not isinstance(sc, dict):
        errs.append("server_counters must be an object")
    else:
        for k in ("preemptions", "evictions", "prefix_hit_rate"):
            if k in sc and sc[k] is not None and not _is_num(sc[k]):
                errs.append(f"server_counters.{k} must be a number or null")
    if not isinstance(d["anomalies"], list):
        errs.append("anomalies must be a list")
    return errs


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass
class Run:
    """One artifact file, loaded and judged."""

    path: Path
    data: dict[str, Any] | None
    errors: list[str]  # structural problems (unparseable, schema, keys)
    reasons: list[str]  # why the run is invalid (structural + run-reported)

    @property
    def valid(self) -> bool:
        return not self.reasons

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    @property
    def arm(self) -> str:
        return str(self.get("arm", default="?"))

    @property
    def workload(self) -> str:
        return str(self.get("workload", "id", default="?"))

    @property
    def point(self) -> dict[str, Any]:
        p = self.get("workload", "point", default={})
        return p if isinstance(p, dict) else {}

    @property
    def rep(self) -> Any:
        return self.get("rep", default="?")


def _harness_validate(data: dict[str, Any]) -> list[str]:
    try:
        from bench.xengine.artifact import validate_artifact as strict
    except Exception:
        return []
    return [f"harness validator: {e}" for e in strict(data)]


def load_runs(root: Path) -> list[Run]:
    """Load every ``*.json`` under ``root``. Unreadable files become invalid runs."""
    runs: list[Run] = []
    if not root.exists():
        return runs
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            runs.append(Run(path, None, [f"unreadable: {e}"], [f"unreadable: {e}"]))
            continue
        errors = validate_artifact(data)
        reasons = list(errors)
        if not errors:
            # The harness's stricter validator (types, null-vs-number, sample
            # counts, the ours backend rule) when importable. Its findings make a
            # run invalid but keep its cell identity, so the cell shows
            # "k invalid excluded" instead of the run vanishing.
            reasons.extend(_harness_validate(data))
        if not errors:
            data_valid = data["validity"]["valid"]
            run_reasons = data["validity"].get("reasons") or []
            if not data_valid:
                reasons.extend(str(r) for r in run_reasons)
                if not run_reasons:
                    reasons.append("validity.valid is false (no reason given)")
            # Defence in depth for SPEC "Engines and arms": an `ours*` run on a
            # non-FlashInfer backend is invalid even if the harness missed it.
            backend = data["validity"].get("attention_backend")
            if data["arm"].startswith("ours") and backend != "flashinfer":
                reasons.append(f"attention_backend is {backend!r}, SPEC requires 'flashinfer'")
        runs.append(Run(path, data if isinstance(data, dict) else None, errors, reasons))
    return runs


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def point_key(point: dict[str, Any]) -> tuple:
    """Canonical, sortable identity of a workload point."""
    return tuple(sorted((str(k), point[k]) for k in point))


def point_label(point: dict[str, Any]) -> str:
    if not point:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in sorted(point.items()))


def point_x(point: dict[str, Any]) -> float | None:
    """The numeric x-coordinate of a single-key point, else None."""
    if len(point) != 1:
        return None
    (v,) = point.values()
    return float(v) if _is_num(v) else None


def _sort_points(points: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def k(p: dict[str, Any]) -> tuple:
        return tuple((name, (0, v) if _is_num(v) else (1, str(v))) for name, v in point_key(p))

    return sorted(points, key=k)


@dataclass
class Stat:
    mean: float
    min: float
    max: float
    stdev: float | None  # sample stdev; None when n < 2
    n: int
    n_null: int = 0  # reps where a nullable counter was null

    @property
    def cv(self) -> float | None:
        if self.stdev is None or self.mean == 0:
            return None
        return abs(self.stdev / self.mean)


def summarize(values: list[float], n_null: int = 0) -> Stat | None:
    if not values:
        return None
    sd = statistics.stdev(values) if len(values) >= 2 else None
    return Stat(statistics.fmean(values), min(values), max(values), sd, len(values), n_null)


@dataclass
class Cell:
    workload: str
    arm: str
    point: dict[str, Any]
    valid_runs: list[Run] = field(default_factory=list)
    invalid_runs: list[Run] = field(default_factory=list)

    @property
    def n_reps(self) -> int:
        return len(self.valid_runs)

    def stat(self, metric: str) -> Stat | None:
        path = METRICS[metric]
        vals: list[float] = []
        n_null = 0
        for r in self.valid_runs:
            v = r.get(*path)
            if v is None:
                n_null += 1
            elif _is_num(v):
                vals.append(float(v))
        return summarize(vals, n_null)

    def anomalies(self) -> list[str]:
        kinds: list[str] = []
        for r in self.valid_runs:
            for a in r.get("anomalies", default=[]) or []:
                kind = a.get("kind", "unknown") if isinstance(a, dict) else str(a)
                if kind not in kinds:
                    kinds.append(kind)
        return kinds

    def duplicate_reps(self) -> list[Any]:
        seen: dict[Any, int] = {}
        for r in self.valid_runs:
            seen[r.rep] = seen.get(r.rep, 0) + 1
        return sorted((k for k, c in seen.items() if c > 1), key=str)


Grid = dict[tuple[str, str, tuple], Cell]


def aggregate(runs: list[Run]) -> Grid:
    """Group runs by (workload, arm, point). Invalid runs are kept but not counted."""
    grid: Grid = {}
    for r in runs:
        if r.data is None or r.errors:
            continue  # structurally broken: inventory only, no cell identity to trust
        key = (r.workload, r.arm, point_key(r.point))
        cell = grid.setdefault(key, Cell(r.workload, r.arm, r.point))
        (cell.valid_runs if r.valid else cell.invalid_runs).append(r)
    return grid


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def fmt_num(x: float) -> str:
    a = abs(x)
    if a >= 100:
        return f"{x:.0f}"
    if a >= 10:
        return f"{x:.1f}"
    if a >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def cell_flags(cell: Cell | None, stat: Stat | None, cv_threshold: float) -> list[str]:
    if cell is None:
        return []
    flags: list[str] = []
    if 0 < cell.n_reps < MIN_REPS:
        flags.append(f"n={cell.n_reps}<{MIN_REPS}")
    if stat is not None and stat.cv is not None and stat.cv > cv_threshold:
        flags.append(f"CV {stat.cv * 100:.0f}%")
    if stat is not None and stat.n_null:
        flags.append(f"null in {stat.n_null}/{cell.n_reps} reps")
    if cell.invalid_runs:
        flags.append(f"{len(cell.invalid_runs)} invalid excluded")
    dup = cell.duplicate_reps()
    if dup:
        flags.append("duplicate rep " + ",".join(str(d) for d in dup))
    flags.extend(f"anomaly: {k}" for k in cell.anomalies())
    return flags


def tail_flag(cell: Cell | None, metric: str) -> str | None:
    """Flag a p99 cell whose smallest per-rep sample count is below MIN_TAIL_SAMPLES."""
    dist = TAIL_SOURCE.get(metric)
    if cell is None or dist is None or not cell.valid_runs:
        return None
    counts = [r.get("metrics", dist, "n") for r in cell.valid_runs]
    counts = [int(c) for c in counts if _is_num(c)]
    if counts and min(counts) < MIN_TAIL_SAMPLES:
        return f"p99 from n={min(counts)}<{MIN_TAIL_SAMPLES}"
    return None


def fmt_cell(cell: Cell | None, metric: str, cv_threshold: float) -> str:
    """``mean ± sd (min–max)`` plus flags, or TODO. Never invents a value."""
    stat = cell.stat(metric) if cell is not None and cell.n_reps else None
    flags = cell_flags(cell, stat, cv_threshold)
    tf = tail_flag(cell, metric)
    if tf:
        flags.append(tf)
    if stat is None:
        if cell is not None and cell.n_reps and metric in NULLABLE_METRICS:
            body = "n/a (not exposed)"
        else:
            body = TODO
    else:
        body = fmt_num(stat.mean)
        if stat.stdev is not None:
            body += f" ± {fmt_num(stat.stdev)}"
        if stat.n > 1:
            body += f" ({fmt_num(stat.min)}–{fmt_num(stat.max)})"
    if flags:
        body += " **[" + "; ".join(flags) + "]**"
    return body


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    def esc(s: str) -> str:
        return s.replace("|", "\\|")

    out = ["| " + " | ".join(esc(h) for h in header) + " |"]
    out.append("|" + "|".join("---" for _ in header) + "|")
    out.extend("| " + " | ".join(esc(c) for c in row) + " |" for row in rows)
    return "\n".join(out)


def points_for(workload: str, grid: Grid) -> list[dict[str, Any]]:
    """SPEC-fixed points plus any observed in artifacts (observed never replace fixed)."""
    pts: dict[tuple, dict[str, Any]] = {point_key(p): p for p in EXPECTED_POINTS.get(workload, [])}
    for (wl, _arm, pk), cell in grid.items():
        if wl == workload:
            pts.setdefault(pk, cell.point)
    return _sort_points(pts.values())


def extra_arms(workload: str, grid: Grid, known: list[str]) -> list[str]:
    return sorted({arm for (wl, arm, _) in grid if wl == workload and arm not in known})


def pivot_table(
    workload: str, metric: str, arms: list[str], grid: Grid, cv_threshold: float
) -> str:
    pts = points_for(workload, grid)
    header = ["point"] + arms
    rows: list[list[str]] = []
    if not pts:
        rows.append([TODO] + [TODO for _ in arms])
    for p in pts:
        pk = point_key(p)
        rows.append(
            [point_label(p)]
            + [fmt_cell(grid.get((workload, a, pk)), metric, cv_threshold) for a in arms]
        )
    return _md_table(header, rows)


CELL_LEGEND = (
    "Cells: `mean ± sample stdev (min–max)` across valid repetitions. "
    "`TODO` = no valid artifact. Bold brackets are flags: `n=k<3` too few reps, "
    "`CV x%` spread above the threshold, `p99 from n=k<100` tail backed by too few "
    "samples to be stable, `k invalid excluded` runs dropped from the "
    "aggregate (listed in the run inventory), `anomaly: kind` reported by the harness. "
    "`n/a (not exposed)` = the engine reported `null` for that counter."
)


def render_workload(workload: str, grid: Grid, cv_threshold: float) -> str:
    parts: list[str] = [CELL_LEGEND, ""]
    diag = DIAGNOSTIC_ARMS + extra_arms(workload, grid, ALL_ARMS + [INT8_ARM])
    for metric in WORKLOAD_TABLES[workload]:
        label = METRIC_LABELS[metric]
        parts.append(f"**{workload} — {label} — baseline arms**")
        parts.append("")
        parts.append(pivot_table(workload, metric, BASELINE_ARMS, grid, cv_threshold))
        parts.append("")
        parts.append(
            f"**{workload} — {label} — diagnostic arms** "
            "(attribution only; never a competitor baseline)"
        )
        parts.append("")
        parts.append(pivot_table(workload, metric, diag, grid, cv_threshold))
        parts.append("")
    return "\n".join(parts).rstrip()


def render_w5(grid: Grid, cv_threshold: float) -> str:
    parts = [
        f"`ours` (fp16) vs `{INT8_ARM}` (int8 weight-only, KV pool pinned to the "
        "fp16 pool), both on W1 and W2 (`make bench-w5`).",
        "",
        CELL_LEGEND,
        "",
    ]
    for workload, metric in [("W1", "output_tok_s"), ("W1", "ttft_p99"), ("W2", "goodput_rps")]:
        parts.append(f"**W5 on {workload} — {METRIC_LABELS[metric]}**")
        parts.append("")
        parts.append(pivot_table(workload, metric, ["ours", INT8_ARM], grid, cv_threshold))
        parts.append("")
    return "\n".join(parts).rstrip()


DERIVED_NOTE = (
    "**DERIVED, not measured.** Each cell is `mean(output tok/s at c) / "
    "mean(output tok/s at c=1)` for the same arm, both means aggregated from W1 "
    "artifacts in the tables above. It is a ratio of two measured cells and nothing "
    "else: no interpolation, no fitting. `TODO` = either source cell has no valid "
    "artifact. Flags from either source cell are carried over."
)


def render_w1_scaling(grid: Grid, cv_threshold: float) -> str:
    """Finding (e): each arm's W1 throughput divided by its own batch-1 throughput.

    Normalizing by the engine's own concurrency-1 cell cancels kernel quality and
    leaves scheduling: does each curve bend at the same relative load (ADR-013's
    retained scaling-shape comparison, ADR-025)?
    """
    pts = points_for("W1", grid)
    base_key = point_key({"concurrency": 1})
    header = ["point"] + BASELINE_ARMS
    rows: list[list[str]] = []
    for p in pts:
        pk = point_key(p)
        row = [point_label(p)]
        for arm in BASELINE_ARMS:
            base = grid.get(("W1", arm, base_key))
            cell = grid.get(("W1", arm, pk))
            bs = base.stat("output_tok_s") if base is not None and base.n_reps else None
            cs = cell.stat("output_tok_s") if cell is not None and cell.n_reps else None
            if bs is None or cs is None or bs.mean == 0:
                row.append(TODO)
                continue
            body = f"{cs.mean / bs.mean:.2f}x"
            flags = cell_flags(cell, cs, cv_threshold)
            if pk != base_key:
                flags += [f"base: {f}" for f in cell_flags(base, bs, cv_threshold)]
            if flags:
                body += " **[" + "; ".join(flags) + "]**"
            row.append(body)
        rows.append(row)
    return DERIVED_NOTE + "\n\n" + _md_table(header, rows)


def render_inventory(runs: list[Run]) -> str:
    if not runs:
        return f"{TODO}: no artifacts under `results/xengine/` yet."
    header = ["artifact", "arm", "workload", "point", "rep", "valid?", "GPU", "node", "job id"]
    header.append("reasons / anomalies")
    rows = []
    for r in runs:
        notes = list(r.reasons)
        for a in r.get("anomalies", default=[]) or []:
            if isinstance(a, dict):
                notes.append(f"anomaly {a.get('kind', '?')}: {a.get('detail', '')}".strip())
        rows.append(
            [
                f"`{r.path.as_posix()}`",
                r.arm,
                r.workload,
                point_label(r.point),
                str(r.rep),
                "yes" if r.valid else "**NO**",
                str(r.get("hardware", "gpu_name", default="?")),
                str(r.get("hardware", "node", default="?")),
                str(r.get("hardware", "slurm_job_id", default="?")),
                "; ".join(notes) or "—",
            ]
        )
    n_bad = sum(not r.valid for r in runs)
    summary = f"{len(runs)} artifacts, {len(runs) - n_bad} valid, {n_bad} invalid.\n\n"
    return summary + _md_table(header, rows)


# Fields whose values must agree across every artifact in one study. Hardware
# drift breaks SPEC hard rule 5 (same node, same allocation); SLO drift breaks the
# frozen-SLO rule; repo drift means arms ran different harness code.
METADATA_FIELDS = [
    ("GPU", ("hardware", "gpu_name")),
    ("GPU count", ("hardware", "gpu_count")),
    ("driver", ("hardware", "driver")),
    ("CUDA", ("hardware", "cuda")),
    ("node", ("hardware", "node")),
    ("Slurm job id", ("hardware", "slurm_job_id")),
    ("repo sha", ("provenance", "repo_sha")),
    ("repo dirty", ("provenance", "repo_dirty")),
    ("harness version", ("provenance", "harness_version")),
    ("SLO TTFT (ms)", ("slo", "ttft_ms")),
    ("SLO TPOT (ms)", ("slo", "tpot_ms")),
    ("SLO source", ("slo", "source")),
]


def collect_metadata(runs: list[Run]) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Distinct recorded values per field, and the list of mismatch warnings."""
    loaded = [r for r in runs if r.data is not None]
    rows: list[tuple[str, list[str]]] = []
    warnings: list[str] = []

    def distinct(vals: Iterable[Any]) -> list[str]:
        out: list[str] = []
        for v in vals:
            s = json.dumps(v) if not isinstance(v, str) else v
            if s not in out:
                out.append(s)
        return out

    for label, path in METADATA_FIELDS:
        vals = distinct(r.get(*path, default="(not recorded)") for r in loaded)
        rows.append((label, vals))
        if len(vals) > 1:
            warnings.append(f"{label} differs across artifacts: {', '.join(vals)}")
    if any(r.get("provenance", "repo_dirty") is True for r in loaded):
        warnings.append("at least one artifact was produced from a dirty repo (repo_dirty=true)")

    engines = sorted({str(r.get("engine", "name", default="?")) for r in loaded})
    for name in engines:
        vals = distinct(
            r.get("engine", "version", default="(not recorded)")
            for r in loaded
            if str(r.get("engine", "name", default="?")) == name
        )
        rows.append((f"engine `{name}` version", vals))
        if len(vals) > 1:
            warnings.append(f"engine {name} version differs across artifacts: {', '.join(vals)}")
    return rows, warnings


def render_metadata(runs: list[Run]) -> str:
    if not [r for r in runs if r.data is not None]:
        return (
            f"{TODO}: GPU, driver, CUDA and engine versions render from artifacts once they exist."
        )
    rows, warnings = collect_metadata(runs)
    parts: list[str] = []
    if warnings:
        parts.append("> **WARNING — METADATA MISMATCH ACROSS ARTIFACTS.** These runs did not all")
        parts.append("> share one environment, so cross-arm comparisons may be confounded:")
        parts.append(">")
        parts.extend(f"> - **{w}**" for w in warnings)
        parts.append("")
    parts.append(
        "Values as recorded in the artifacts, invalid runs included "
        "(every distinct value is listed)."
    )
    parts.append("")
    parts.append(_md_table(["field", "recorded value(s)"], [[k, ", ".join(v)] for k, v in rows]))
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _series(
    grid: Grid, workload: str, arm: str, metric: str
) -> tuple[list[float], list[float], list[float], list[float]]:
    """x, mean, lower err, upper err for valid cells with a numeric x. Nothing filled in."""
    xs, ys, lo, hi = [], [], [], []
    for p in points_for(workload, grid):
        cell = grid.get((workload, arm, point_key(p)))
        x = point_x(p)
        if cell is None or not cell.n_reps or x is None:
            continue
        st = cell.stat(metric)
        if st is None:
            continue
        xs.append(x)
        ys.append(st.mean)
        lo.append(st.mean - st.min)
        hi.append(st.max - st.mean)
    return xs, ys, lo, hi


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


ERRBAR_NOTE = "error bars: min–max across valid reps"


def _line_chart(
    grid: Grid, workload: str, metric: str, arms: list[str], out: Path, xlabel: str, title: str
) -> bool:
    series = {a: _series(grid, workload, a, metric) for a in arms}
    series = {a: s for a, s in series.items() if s[0]}
    if not series:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, (xs, ys, lo, hi) in series.items():
        style = "-" if arm in BASELINE_ARMS else "--"
        ax.errorbar(xs, ys, yerr=[lo, hi], label=arm, marker="o", linestyle=style, capsize=3)
    if workload == "W1":
        ax.set_xscale("log", base=2)
        ax.set_xticks([p["concurrency"] for p in EXPECTED_POINTS["W1"]])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"{title}\n({ERRBAR_NOTE}; dashed = diagnostic arm)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def _w3_chart(grid: Grid, out: Path) -> bool:
    pts = [p for p in points_for("W3", grid)]
    panels = []
    for p in pts:
        pk = point_key(p)
        bars = []
        for on, off in PREFIX_PAIRS:
            pair = []
            for arm in (on, off):
                cell = grid.get(("W3", arm, pk))
                st = cell.stat("ttft_p99") if cell is not None and cell.n_reps else None
                pair.append(st)
            bars.append(pair)
        if any(s is not None for pair in bars for s in pair):
            panels.append((p, bars))
    if not panels:
        return False
    plt = _plt()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5), squeeze=False)
    engines = [on for on, _ in PREFIX_PAIRS]
    width = 0.38
    for ax, (p, bars) in zip(axes[0], panels, strict=True):
        for j, (label, color) in enumerate([("prefix cache on", "C0"), ("prefix cache off", "C1")]):
            xs, ys, lo, hi = [], [], [], []
            for i, pair in enumerate(bars):
                st = pair[j]
                if st is None:
                    continue  # missing bar is left missing, not drawn as zero
                xs.append(i + (j - 0.5) * width)
                ys.append(st.mean)
                lo.append(st.mean - st.min)
                hi.append(st.max - st.mean)
            ax.bar(xs, ys, width, yerr=[lo, hi], capsize=3, label=label, color=color)
        ax.set_xticks(range(len(engines)))
        ax.set_xticklabels(engines)
        ax.set_title(point_label(p), fontsize=10)
        ax.set_ylabel(METRIC_LABELS["ttft_p99"])
        ax.grid(True, axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    title = f"W3 shared-prefix: TTFT p99, prefix cache on vs off\n({ERRBAR_NOTE})"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def _w4_chart(grid: Grid, out: Path) -> bool:
    arms = BASELINE_ARMS + ["vllm-matched"]
    good = {a: _series(grid, "W4", a, "goodput_rps") for a in arms}
    pre = {a: _series(grid, "W4", a, "preemptions") for a in arms}
    if not any(s[0] for s in good.values()) and not any(s[0] for s in pre.values()):
        return False
    keys = sorted({k for p in points_for("W4", grid) for k in p})
    xname = ", ".join(keys) or "W4 point"
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, data, metric in [(a1, good, "goodput_rps"), (a2, pre, "preemptions")]:
        for arm, (xs, ys, lo, hi) in data.items():
            if not xs:
                continue
            style = "-" if arm in BASELINE_ARMS else "--"
            ax.errorbar(xs, ys, yerr=[lo, hi], label=arm, marker="o", linestyle=style, capsize=3)
        ax.set_xlabel(f"offered load ({xname})")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    a1.set_title("goodput vs offered load", fontsize=10)
    a2.set_title("preemptions vs offered load (null counters omitted)", fontsize=10)
    fig.suptitle(f"W4 forced preemption, equal KV pool\n({ERRBAR_NOTE})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def render_figures(grid: Grid, fig_dir: Path) -> dict[str, bool]:
    """Draw every figure that has data. A figure without data is removed, not faked."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    w1_arms = ALL_ARMS + extra_arms("W1", grid, ALL_ARMS + [INT8_ARM])
    made = {
        "w1_throughput": _line_chart(
            grid,
            "W1",
            "output_tok_s",
            w1_arms,
            fig_dir / FIGURES["w1_throughput"],
            "concurrency (in-flight requests)",
            "W1 closed loop, prompt 512 / output 128: output tok/s",
        ),
        "w1_ttft": _line_chart(
            grid,
            "W1",
            "ttft_p99",
            w1_arms,
            fig_dir / FIGURES["w1_ttft"],
            "concurrency (in-flight requests)",
            "W1 closed loop, prompt 512 / output 128: TTFT p99",
        ),
        "w3_prefix": _w3_chart(grid, fig_dir / FIGURES["w3_prefix"]),
        "w4_goodput": _w4_chart(grid, fig_dir / FIGURES["w4_goodput"]),
    }
    for name, ok in made.items():
        if not ok:
            (fig_dir / FIGURES[name]).unlink(missing_ok=True)  # stale figure would mislead
    return made


def figure_region(name: str, made: bool, rel_dir: str) -> str:
    if not made:
        return f"{TODO}: figure `{FIGURES[name]}` renders once valid artifacts exist."
    return f"![{name}]({rel_dir}/{FIGURES[name]})"


# --------------------------------------------------------------------------
# Generated-region rewrite
# --------------------------------------------------------------------------
_REGION_RE = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[\w-]+) -->)(?P<body>.*?)(<!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)


def rewrite_generated(text: str, regions: dict[str, str]) -> tuple[str, list[str]]:
    """Replace the body of each marked region named in ``regions``.

    Returns the new text and a list of region names that were requested but not
    found in the document. Unknown regions in the document are left untouched.
    Text outside markers is copied byte-for-byte.
    """
    found: set[str] = set()

    def sub(m: re.Match) -> str:
        name = m.group("name")
        if name not in regions:
            return m.group(0)
        found.add(name)
        return f"{m.group(1)}\n{regions[name].rstrip()}\n{m.group(4)}"

    new = _REGION_RE.sub(sub, text)
    return new, sorted(set(regions) - found)


def build_regions(
    runs: list[Run], grid: Grid, made: dict[str, bool], fig_rel: str, cv_threshold: float
) -> dict[str, str]:
    regions = {
        "metadata": render_metadata(runs),
        "inventory": render_inventory(runs),
        "w5_table": render_w5(grid, cv_threshold),
        "w1_scaling": render_w1_scaling(grid, cv_threshold),
    }
    for wl in WORKLOADS:
        regions[f"{wl.lower()}_table"] = render_workload(wl, grid, cv_threshold)
    for name in FIGURES:
        regions[f"fig_{name}"] = figure_region(name, made.get(name, False), fig_rel)
    return regions


def render(
    results: Path,
    doc: Path,
    figures: Path,
    cv_threshold: float = DEFAULT_CV_THRESHOLD,
    draw: bool = True,
) -> list[str]:
    """Run the whole pipeline. Returns region names missing from the doc."""
    runs = load_runs(results)
    grid = aggregate(runs)
    made = render_figures(grid, figures) if draw else {}
    try:
        fig_rel = figures.resolve().relative_to(doc.resolve().parent).as_posix()
    except ValueError:
        fig_rel = figures.as_posix()
    regions = build_regions(runs, grid, made, fig_rel, cv_threshold)
    new, missing = rewrite_generated(doc.read_text(), regions)
    doc.write_text(new)
    return missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results/xengine"))
    ap.add_argument("--doc", type=Path, default=Path("docs/xengine/BENCHMARKS.md"))
    ap.add_argument("--figures", type=Path, default=Path("docs/xengine/figures"))
    ap.add_argument("--cv-threshold", type=float, default=DEFAULT_CV_THRESHOLD)
    ap.add_argument("--no-figures", action="store_true", help="skip matplotlib")
    args = ap.parse_args(argv)
    missing = render(args.results, args.doc, args.figures, args.cv_threshold, not args.no_figures)
    print(f"rendered {len(load_runs(args.results))} artifacts into {args.doc}")
    if missing:
        print(f"WARNING: regions missing from doc: {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
