"""
The `xengine-run/1` artifact: compute it, assemble it, validate it.

The schema is docs/xengine/SPEC.md "Result artifact". The top-level key set is
EXACTLY the SPEC's; nested objects carry the SPEC's keys and may add more
(e.g. `workload.loop`, `workload.request_stream_sha256`, `validity.checks`).

Metric semantics, shared by both loops so W1 and W2-W4 numbers mean the same
thing:

  ttft_ms / e2e_ms  per steady request, from (intended) dispatch — see
                    bench/loadgen.py and bench/xengine/closed_loop.py for what
                    "intended" means in each loop
  itl_ms            gaps between consecutive non-empty content chunks, POOLED
                    over steady requests
  tpot_ms           one value per steady request with >= 2 tokens
  output_tok_s      steady requests' output tokens / window
  request_rps       steady requests COMPLETED / window
  goodput_rps       steady requests meeting the SLO (bench/loadgen.py:meets_slo)
                    / window
  slo_attainment    met / steady requests issued (failures are misses)
  completed/failed  steady requests completed / not completed

The window is `duration_s` for an open-loop run (loadgen's definition) and
[first measured dispatch, last measured completion) for a closed-loop run.

Percentiles are linear interpolation between order statistics
(bench/loadgen.py:percentile). A metric with no samples has n = 0 and `null`
percentiles — never 0.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from bench.loadgen import LoadGenConfig, LoadGenRun, Outcome, Phase, meets_slo, percentile
from bench.xengine import SCHEMA

TOP_LEVEL_KEYS = (
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
    "samples",
)
DIST_METRICS = ("ttft_ms", "itl_ms", "tpot_ms", "e2e_ms")
SCALAR_METRICS = ("output_tok_s", "request_rps", "goodput_rps", "slo_attainment")
COUNT_METRICS = ("completed", "failed")
SAMPLE_KEYS = ("ttft_ms", "itl_ms", "tpot_ms")
COUNTER_KEYS = ("preemptions", "evictions", "prefix_hit_rate")
SAMPLE_DECIMALS = 3  # microsecond resolution; keeps ITL-heavy files small


def summarize(xs: Sequence[float]) -> dict[str, Any]:
    if not xs:
        return {"p50": None, "p95": None, "p99": None, "n": 0}
    xs = list(xs)
    return {
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "n": len(xs),
    }


def compute_metrics(
    run: LoadGenRun, cfg: LoadGenConfig, window_s: float
) -> tuple[dict[str, Any], dict[str, list[float]], dict[str, list[float]]]:
    """
    -> (metrics, samples for the artifact, per-request series in dispatch
    order for the anomaly detectors).
    """
    steady = [r for r in run.results if r.spec.phase == Phase.STEADY]
    steady.sort(key=lambda r: (r.spec.intended_send_time, r.spec.request_id))
    window_s = max(window_s, 1e-9)

    ttft = [r.ttft_ms for r in steady if r.ttft_ms is not None]
    tpot = [r.tpot_ms for r in steady if r.tpot_ms is not None]
    e2e = [r.e2e_ms for r in steady if r.e2e_ms is not None]
    itl: list[float] = []
    for r in steady:
        itl.extend(r.itls_ms)

    completed = sum(1 for r in steady if r.outcome == Outcome.COMPLETED)
    met = sum(1 for r in steady if meets_slo(r, cfg)[0])
    out_tokens = sum(r.output_tokens for r in steady)

    metrics = {
        "ttft_ms": summarize(ttft),
        "itl_ms": summarize(itl),
        "tpot_ms": summarize(tpot),
        "e2e_ms": summarize(e2e),
        "output_tok_s": out_tokens / window_s,
        "request_rps": completed / window_s,
        "goodput_rps": met / window_s,
        "slo_attainment": (met / len(steady)) if steady else 0.0,
        "completed": completed,
        "failed": len(steady) - completed,
        "window_s": window_s,
        "steady_requests": len(steady),
        "outcomes": _outcomes(steady),
    }

    def rnd(xs: list[float]) -> list[float]:
        return [round(x, SAMPLE_DECIMALS) for x in xs]

    samples = {"ttft_ms": rnd(ttft), "itl_ms": rnd(itl), "tpot_ms": rnd(tpot)}
    in_order = {"ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": e2e}
    return metrics, samples, in_order


def _outcomes(results: Sequence[Any]) -> dict[str, int]:
    out = {k: 0 for k in Outcome.ALL}
    for r in results:
        out[r.outcome] = out.get(r.outcome, 0) + 1
    return out


def build_artifact(
    *,
    arm: str,
    engine: dict[str, Any],
    workload: dict[str, Any],
    rep: int,
    hardware: dict[str, Any],
    provenance: dict[str, Any],
    slo: dict[str, Any],
    validity: dict[str, Any],
    metrics: dict[str, Any],
    server_counters: dict[str, Any],
    anomalies: list[dict[str, Any]],
    samples: dict[str, list[float]],
) -> dict[str, Any]:
    art = {
        "schema": SCHEMA,
        "arm": arm,
        "engine": engine,
        "workload": workload,
        "rep": rep,
        "hardware": hardware,
        "provenance": provenance,
        "slo": slo,
        "validity": validity,
        "metrics": metrics,
        "server_counters": server_counters,
        "anomalies": anomalies,
        "samples": samples,
    }
    errs = validate_artifact(art)
    if errs:
        raise ValueError("refusing to build an invalid artifact:\n  " + "\n  ".join(errs))
    return art


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _str_or_none(x: Any) -> bool:
    return x is None or isinstance(x, str)


def _need(d: Any, where: str, keys: Sequence[str], errs: list[str]) -> bool:
    if not isinstance(d, dict):
        errs.append(f"{where} must be an object")
        return False
    for k in keys:
        if k not in d:
            errs.append(f"{where}.{k} missing")
    return True


def validate_artifact(d: Any) -> list[str]:
    """
    Structural + semantic check against SPEC's `xengine-run/1`. Returns a list
    of problems; empty means valid. Stricter than bench/xengine/render.py's
    copy: it also checks types, null-vs-number rules, sample counts and the
    `ours` attention-backend rule.
    """
    from bench.xengine.engines import ARMS

    if not isinstance(d, dict):
        return ["artifact is not a JSON object"]
    errs: list[str] = []
    if d.get("schema") != SCHEMA:
        errs.append(f"schema is {d.get('schema')!r}, expected {SCHEMA!r}")
    keys = set(d)
    missing = [k for k in TOP_LEVEL_KEYS if k not in keys]
    extra = sorted(keys - set(TOP_LEVEL_KEYS))
    if missing:
        errs.append(f"missing top-level keys: {missing}")
    if extra:
        errs.append(f"unexpected top-level keys: {extra}")
    if missing:
        return errs

    arm = d["arm"]
    arm_spec = ARMS.get(arm) if isinstance(arm, str) else None
    if arm_spec is None:
        errs.append(f"arm {arm!r} is not in the SPEC arm table")

    eng = d["engine"]
    if _need(eng, "engine", ("name", "version", "launch_cmd", "flags"), errs):
        if arm_spec is not None and eng.get("name") != arm_spec.engine:
            errs.append(f"engine.name {eng.get('name')!r} does not match arm {arm!r}")
        if not _str_or_none(eng.get("version")):
            errs.append("engine.version must be a string or null")
        lc = eng.get("launch_cmd")
        if not isinstance(lc, list) or not all(isinstance(x, str) for x in lc):
            errs.append("engine.launch_cmd must be a list of strings")
        if not isinstance(eng.get("flags"), dict):
            errs.append("engine.flags must be an object")

    wl = d["workload"]
    if _need(wl, "workload", ("id", "config_path", "config_sha256", "seed", "point"), errs):
        if not isinstance(wl.get("id"), str) or not re.fullmatch(r"W\d+", wl.get("id") or ""):
            errs.append(f"workload.id {wl.get('id')!r} is not W<k>")
        if not isinstance(wl.get("config_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", wl.get("config_sha256") or ""
        ):
            errs.append("workload.config_sha256 must be a sha256 hex digest")
        if not _int(wl.get("seed")):
            errs.append("workload.seed must be an integer")
        pt = wl.get("point")
        if not isinstance(pt, dict) or len(pt) != 1 or not all(_num(v) for v in pt.values()):
            errs.append(f"workload.point must be a single-key numeric mapping, got {pt!r}")
        if "loop" in wl and wl["loop"] not in ("open", "closed"):
            errs.append(f"workload.loop must be open|closed, got {wl['loop']!r}")

    if not _int(d["rep"]) or d["rep"] < 1:
        errs.append("rep must be an integer >= 1")

    hw = d["hardware"]
    hw_keys = ("gpu_name", "gpu_count", "driver", "cuda", "node", "slurm_job_id")
    if _need(hw, "hardware", hw_keys, errs):
        for k in ("gpu_name", "driver", "cuda", "node", "slurm_job_id"):
            if not _str_or_none(hw.get(k)):
                errs.append(f"hardware.{k} must be a string or null")
        if hw.get("gpu_count") is not None and not _int(hw.get("gpu_count")):
            errs.append("hardware.gpu_count must be an integer or null")

    prov_keys = ("repo_sha", "repo_dirty", "started_at", "harness_version")
    if _need(d["provenance"], "provenance", prov_keys, errs):
        if not isinstance(d["provenance"].get("started_at"), str):
            errs.append("provenance.started_at must be a string")

    slo = d["slo"]
    if _need(slo, "slo", ("ttft_ms", "tpot_ms", "source"), errs):
        if not (_num(slo.get("ttft_ms")) and _num(slo.get("tpot_ms"))):
            errs.append("slo.ttft_ms / slo.tpot_ms must be numbers")
        if not isinstance(slo.get("source"), str) or not slo.get("source"):
            errs.append("slo.source must be a non-empty string")

    val = d["validity"]
    if _need(val, "validity", ("valid", "reasons", "attention_backend"), errs):
        reasons = val.get("reasons")
        if not isinstance(val.get("valid"), bool):
            errs.append("validity.valid must be a boolean")
        if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
            errs.append("validity.reasons must be a list of strings")
        elif val.get("valid") is True and reasons:
            errs.append("validity.valid is true but reasons are listed")
        elif val.get("valid") is False and not reasons:
            errs.append("validity.valid is false with no reason given")
        if not _str_or_none(val.get("attention_backend")):
            errs.append("validity.attention_backend must be a string or null")
        if (
            arm_spec is not None
            and arm_spec.engine == "ours"
            and val.get("valid") is True
            and val.get("attention_backend") != "flashinfer"
        ):
            errs.append(
                f"ours arm marked valid with attention_backend "
                f"{val.get('attention_backend')!r}; SPEC requires 'flashinfer'"
            )

    m = d["metrics"]
    if _need(m, "metrics", DIST_METRICS + SCALAR_METRICS + COUNT_METRICS, errs):
        for k in DIST_METRICS:
            dist = m.get(k)
            if not _need(dist, f"metrics.{k}", ("p50", "p95", "p99", "n"), errs):
                continue
            n = dist.get("n")
            if not _int(n) or n < 0:
                errs.append(f"metrics.{k}.n must be a non-negative integer")
                continue
            for p in ("p50", "p95", "p99"):
                v = dist.get(p)
                if n == 0 and v is not None:
                    errs.append(f"metrics.{k}.{p} must be null when n == 0")
                if n > 0 and not _num(v):
                    errs.append(f"metrics.{k}.{p} must be a finite number when n > 0")
        for k in SCALAR_METRICS:
            if not _num(m.get(k)) or m.get(k) < 0:
                errs.append(f"metrics.{k} must be a non-negative finite number")
        for k in COUNT_METRICS:
            if not _int(m.get(k)) or m.get(k) < 0:
                errs.append(f"metrics.{k} must be a non-negative integer")
        if _num(m.get("slo_attainment")) and m["slo_attainment"] > 1.0:
            errs.append("metrics.slo_attainment must be <= 1")

    sc = d["server_counters"]
    if _need(sc, "server_counters", COUNTER_KEYS + ("raw",), errs):
        for k in COUNTER_KEYS:
            if sc.get(k) is not None and not _num(sc.get(k)):
                errs.append(f"server_counters.{k} must be a number or null")
        if not isinstance(sc.get("raw"), dict):
            errs.append("server_counters.raw must be an object")

    an = d["anomalies"]
    if not isinstance(an, list):
        errs.append("anomalies must be a list")
    else:
        for i, a in enumerate(an):
            if (
                not isinstance(a, dict)
                or not isinstance(a.get("kind"), str)
                or not isinstance(a.get("detail"), str)
            ):
                errs.append(f"anomalies[{i}] must be an object with string kind and detail")

    s = d["samples"]
    if isinstance(s, dict):
        if set(s) != set(SAMPLE_KEYS):
            errs.append(f"samples keys must be exactly {list(SAMPLE_KEYS)}, got {sorted(s)}")
        for k in SAMPLE_KEYS:
            xs = s.get(k)
            if not isinstance(xs, list) or not all(_num(x) for x in xs):
                errs.append(f"samples.{k} must be a list of finite numbers")
            elif isinstance(m, dict) and isinstance(m.get(k), dict) and m[k].get("n") != len(xs):
                errs.append(
                    f"samples.{k} has {len(xs)} values but metrics.{k}.n is {m[k].get('n')}"
                )
    else:
        errs.append("samples must be an object")
    return errs


def artifact_relpath(workload_id: str, arm: str, point_label: str, rep: int) -> str:
    """SPEC path: <workload>/<arm>/<point>_rep<k>.json"""
    return f"{workload_id}/{arm}/{point_label}_rep{rep}.json"
