"""
Run one cell of the cross-engine matrix: one arm x one workload, every point,
N repetitions. Writes one `xengine-run/1` artifact per point x rep.

    python3 -m bench.xengine.run --arm vllm --workload W1 --reps 3 --out results/xengine
    python3 -m bench.xengine.run --arm ours --workload W3 --url http://127.0.0.1:8000

Without --url the harness launches the arm's server itself (interpreter from
$XENGINE_SERVER_PY, see bench/xengine/engines.py), waits for /health, runs,
and terminates it. With --url it measures a server someone else started; the
launch command is then recorded as [] and, for `ours`, the backend log line
is only available if --server-log points at that server's log.

Layout under --out (SPEC "Result artifact"):

    <workload>/<arm>/<point>_rep<k>.json     one artifact per point x rep
    <workload>/<arm>/server_<utc>.log        the server's stdout+stderr
    <workload>/<arm>/engine_version.txt      resolved engine version + launch

Side files are deliberately NOT .json: the renderer reads every *.json under
results/xengine as an artifact.

ORDER: reps outer, points inner. Slow drift over a long allocation then
spreads across the whole sweep instead of landing on the last point.

REPS ARE APPENDED: if <point>_rep1..3 exist, the next run writes rep4.. with
fresh derived seeds. Rep-to-rep spread is recomputed over every artifact of
the point and written back into each of them.

Exit codes: 0 all artifacts valid; 2 at least one invalid artifact written;
3 the cell cannot be run (blocked arm/workload, missing input); 4 the server
failed to start.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from bench.driver_common import open_loop_dispatch
from bench.loadgen import Outcome, stationarity, validate_run
from bench.xengine import HARNESS_VERSION, anomalies
from bench.xengine.artifact import artifact_relpath, build_artifact, compute_metrics
from bench.xengine.closed_loop import inflight_check, run_closed_loop, steady_window
from bench.xengine.config import (
    SLO,
    RequestStream,
    WorkloadSpec,
    build_stream,
    load_slo,
    load_workload,
    make_renderer,
    point_label,
)
from bench.xengine.engines import (
    ArmSpec,
    EngineAdapter,
    LaunchSpec,
    OursAdapter,
    ServerProcess,
    ServerStartError,
    adapter_for,
    base_url_for,
    get_arm,
    ours_kv_pool_tokens_from,
)
from bench.xengine.hardware import GpuSampler, capture_hardware, publishable
from serving.metrics.artifact import Provenance

REPO_ROOT = Path(__file__).resolve().parents[2]

CLOSED_LOOP_NOTE = (
    "CLOSED LOOP: N workers, each sends its next request when the previous one "
    "finishes, so exactly N are in flight. Latency is timed from the moment the "
    "worker became free; coordinated omission does not apply because there is no "
    "schedule to fall behind, and no queue deeper than N can form by construction. "
    "This mode cannot locate a saturation knee (that is W2's job)."
)
OPEN_LOOP_NOTE = (
    "OPEN LOOP: precomputed Poisson schedule (bench/loadgen.py discipline); latency "
    "is timed from INTENDED dispatch, so harness lateness is included, and a run "
    "whose dispatch drift exceeds the threshold is invalid."
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def existing_reps(arm_dir: Path, label: str) -> list[int]:
    reps = []
    for p in arm_dir.glob(f"{label}_rep*.json"):
        m = re.fullmatch(rf"{re.escape(label)}_rep(\d+)\.json", p.name)
        if m:
            reps.append(int(m.group(1)))
    return sorted(reps)


def _relpath(p: str | Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def healthy(adapter: EngineAdapter, base_url: str) -> bool:
    try:
        return httpx.get(adapter.health_url(base_url), timeout=10.0).status_code == 200
    except httpx.HTTPError:
        return False


def resolve_kv_pool_tokens(
    cli: int | None, spec: WorkloadSpec, arm: ArmSpec, out: Path
) -> tuple[int | None, str | None]:
    """CLI > workload YAML > (vllm-matched only) ours' recorded pool capacity."""
    if cli:
        return cli, "--kv-pool-tokens"
    if spec.kv_pool_tokens:
        return spec.kv_pool_tokens, f"{spec.id}.yaml kv_pool_tokens"
    if arm.matched:
        for p in sorted((out / spec.id / "ours").glob("*.json"), reverse=True):
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            cap = ours_kv_pool_tokens_from((d.get("server_counters") or {}).get("raw") or {})
            if cap:
                return cap, f"ours allocator.tokens_capacity from {_relpath(p)}"
    return None, None


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def _loadgen_reason_code(reason: str) -> str:
    if "COORDINATED OMISSION" in reason:
        return "dispatch_drift"
    if "Steady state NOT verified" in reason:
        return "not_steady_state"
    if "concurrency cap" in reason:
        return "concurrency_cap_bound"
    return "loadgen"


def resolve_attention_backend(
    arm: ArmSpec, preflight: dict[str, Any] | None, log_backend: str | None
) -> tuple[str | None, list[str]]:
    """
    -> (validity.attention_backend, reasons). For ours, BOTH pieces of evidence
    must agree on flashinfer when both exist; either one saying otherwise wins.
    """
    if arm.engine != "ours":
        return log_backend, []
    pf = (preflight or {}).get("backend")
    if pf is None and log_backend is None:
        return "unknown", [
            "attention_backend_unverified: no pre-flight verdict (no --model-path) and no "
            "backend line in the server log; SPEC requires FlashInfer to be established"
        ]
    if pf is not None and pf != "flashinfer":
        return pf, [f"attention_backend_not_flashinfer: pre-flight says {pf}"]
    if log_backend is not None and log_backend != "flashinfer":
        return log_backend, [
            f"attention_backend_not_flashinfer: running server built {log_backend} "
            f"(pre-flight said {pf or 'n/a'})"
        ]
    return "flashinfer", []


def ours_config_reasons(flags: dict[str, Any], log_cfg: dict[str, Any] | None) -> list[str]:
    """Cross-check launch env against the startup `config:` line (F-001)."""
    if log_cfg is None:
        return []
    out = []
    if "prefix_cache" in flags and bool(flags["prefix_cache"]) != log_cfg["prefix_cache"]:
        out.append(
            f"config_mismatch: launch env says prefix_cache={flags['prefix_cache']} but the "
            f"server printed prefix_cache={'on' if log_cfg['prefix_cache'] else 'OFF'}"
        )
    if flags.get("kv_blocks") and flags["kv_blocks"] != log_cfg["kv_blocks"]:
        out.append(
            f"config_mismatch: SERVING_KV_BLOCKS={flags['kv_blocks']} but the server "
            f"printed kv_blocks={log_cfg['kv_blocks']}"
        )
    if log_cfg.get("static_batching"):
        out.append("config_mismatch: server is running static batching (baseline B2)")
    return out


# ---------------------------------------------------------------------------
# One measured run
# ---------------------------------------------------------------------------


def measure(
    stream: RequestStream, client: httpx.AsyncClient | None = None
) -> tuple[Any, float, list[str], dict[str, Any]]:
    """
    Run one stream. -> (LoadGenRun, window_s, validity reasons, loop checks).
    """
    reasons: list[str] = []
    checks: dict[str, Any] = {}
    if stream.loop == "closed":
        assert stream.concurrency is not None
        run = asyncio.run(
            run_closed_loop(stream.lcfg, stream.specs, stream.concurrency, client=client)
        )
        win = steady_window(run)
        window_s = (win[1] - win[0]) if win else 0.0
        chk = inflight_check(run, stream.concurrency)
        checks["inflight"] = chk
        if not chk["held"]:
            reasons.append(f"inflight_not_held: {chk['reason']}")
    else:
        run = asyncio.run(open_loop_dispatch(stream.lcfg, stream.specs, client=client))
        window_s = stream.lcfg.duration_s
        stat = stationarity(
            run.inflight_samples,
            stream.lcfg.steady_start_s,
            stream.lcfg.steady_end_s,
            stream.lcfg.stationarity_tolerance,
        )
        v = validate_run(run.results, stream.lcfg, stat)
        checks["stationarity"] = stat
        checks["dispatch_drift"] = v.drift
        reasons += [f"{_loadgen_reason_code(r)}: {r}" for r in v.reasons]
    return run, window_s, reasons, checks


# ---------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------


def run_cell(args: argparse.Namespace) -> int:
    arm = get_arm(args.arm)
    if arm.blocked_reason:
        print(f"BLOCKED [{arm.id}]: {arm.blocked_reason}", file=sys.stderr)
        return 3
    spec = load_workload(args.workload)
    if spec.blocked:
        print(f"BLOCKED [{spec.id}]: {spec.blocked_reason}", file=sys.stderr)
        return 3
    if spec.arms and arm.id not in spec.arms:
        print(f"note: arm {arm.id} is not in {spec.id}'s arm list {spec.arms}", file=sys.stderr)

    slo: SLO = load_slo(args.slo) if args.slo else load_slo()
    out = Path(args.out)
    arm_dir = out / spec.id / arm.id
    arm_dir.mkdir(parents=True, exist_ok=True)

    adapter = adapter_for(
        arm.engine, **({"probe_backend": not args.plain_factory} if arm.engine == "ours" else {})
    )
    model_path = args.model_path or os.environ.get("LLM_WEIGHTS_PATH")
    kv_tokens, kv_source = resolve_kv_pool_tokens(args.kv_pool_tokens, spec, arm, out)
    renderer = make_renderer(args.renderer)

    points = spec.points()
    if args.points:
        wanted = {float(x) for x in args.points.split(",")}
        points = [p for p in points if float(next(iter(p.values()))) in wanted]
        if not points:
            print(f"no points of {spec.id} match --points {args.points}", file=sys.stderr)
            return 3

    # -- launch ---------------------------------------------------------------
    launch: LaunchSpec | None = None
    server: ServerProcess | None = None
    preflight: dict[str, Any] | None = None
    if not args.url and not model_path:
        print("need --model-path or $LLM_WEIGHTS_PATH to launch a server", file=sys.stderr)
        return 3
    if isinstance(adapter, OursAdapter) and model_path and not args.skip_preflight:
        preflight = adapter.preflight_attention_backend(model_path)
        print(f"[{arm.id}] attention-backend pre-flight: {preflight.get('backend')}")
    if args.url:
        base_url = args.url.split("/v1/", 1)[0].rstrip("/")
        server_log = Path(args.server_log) if args.server_log else None
    else:
        port = args.port or free_port()
        try:
            launch = adapter.launch_spec(arm, model_path, args.host, port, kv_tokens)
        except ValueError as exc:
            print(f"cannot launch {arm.id}: {exc}", file=sys.stderr)
            return 3
        base_url = base_url_for(args.host, port)
        server_log = arm_dir / f"server_{utc_stamp()}.log"
        if args.dry_run:
            print(json.dumps(launch.as_dict(), indent=2))
            return 0
        server = ServerProcess(launch, server_log, adapter.health_url(base_url)).start()
        try:
            server.wait_healthy(timeout_s=args.health_timeout)
        except ServerStartError as exc:
            print(f"FATAL [{arm.id}]: {exc}", file=sys.stderr)
            return 4
        print(f"[{arm.id}] healthy after {server.healthy_after_s:.1f}s at {base_url}")

    try:
        return _run_points(
            args,
            arm,
            spec,
            slo,
            adapter,
            base_url,
            launch,
            server,
            server_log,
            preflight,
            kv_tokens,
            kv_source,
            renderer,
            points,
            arm_dir,
        )
    finally:
        if server is not None:
            rc = server.stop()
            print(f"[{arm.id}] server stopped (rc={rc})")


def _run_points(
    args: argparse.Namespace,
    arm: ArmSpec,
    spec: WorkloadSpec,
    slo: SLO,
    adapter: EngineAdapter,
    base_url: str,
    launch: LaunchSpec | None,
    server: ServerProcess | None,
    server_log: Path | None,
    preflight: dict[str, Any] | None,
    kv_tokens: int | None,
    kv_source: str | None,
    renderer: Any,
    points: list[dict[str, Any]],
    arm_dir: Path,
) -> int:
    version = adapter.query_version(base_url)
    log_text = server_log.read_text(errors="replace") if server_log and server_log.exists() else ""
    log_backend = adapter.detect_attention_backend(log_text)
    attention_backend, backend_reasons = resolve_attention_backend(arm, preflight, log_backend)
    log_cfg = OursAdapter.config_from_log(log_text) if arm.engine == "ours" else None
    flags = dict(launch.flags) if launch else {"launched_by_harness": False}
    if kv_tokens and not launch:
        flags["kv_pool_tokens_requested"] = kv_tokens
    flags["kv_pool_tokens_source"] = kv_source
    # Defaults as the engine RESOLVED them at startup (e.g. vLLM's
    # max_num_seqs / max_num_batched_tokens), grepped from its log.
    flags["resolved"] = adapter.resolved_defaults(log_text)
    static_reasons = backend_reasons + (
        ours_config_reasons(flags, log_cfg) if arm.engine == "ours" else []
    )
    if renderer.name == "words":
        static_reasons.append(
            "prompt_render_words_fallback: no tokenizer available; prompts were rendered "
            "as 'w<id>' words, so prompt token counts do not match the workload config"
        )

    engine_block = {
        "name": arm.engine,
        "version": version,
        "launch_cmd": list(launch.cmd) if launch else [],
        "flags": flags,
        "model_name": adapter.model_name,
        "launch_env": dict(launch.env) if launch else {},
        "launched_by_harness": launch is not None,
        "base_url": base_url,
        "healthy_after_s": server.healthy_after_s if server else None,
        "server_log": _relpath(server_log) if server_log else None,
        "server_python": launch.cmd[0] if launch else None,
        "attention_backend_evidence": {
            "preflight": preflight,
            "server_log": log_backend,
            "startup_config_line": log_cfg,
        },
    }
    (arm_dir / "engine_version.txt").write_text(
        json.dumps(
            {
                "arm": arm.id,
                "engine": arm.engine,
                "version": version,
                "launch_cmd": engine_block["launch_cmd"],
                "flags": flags,
                "captured_at": utc_stamp(),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"[{arm.id}] engine version: {version}")

    hardware = capture_hardware()
    hw_ok, hw_blockers = publishable(hardware)
    any_invalid = False
    written: dict[str, list[Path]] = {}

    start_rep = {
        point_label(p): (max(existing_reps(arm_dir, point_label(p)), default=0) + 1) for p in points
    }
    for i in range(args.reps):
        for point in points:
            label = point_label(point)
            rep = start_rep[label] + i
            stream = build_stream(
                spec, point, rep, slo, adapter.chat_url(base_url), renderer, adapter.model_name
            )
            before = adapter.scrape_counters(base_url)
            sampler = (
                GpuSampler(args.gpu_sample_interval).start()
                if args.gpu_sample_interval > 0
                else None
            )
            print(
                f"[{arm.id}] {spec.id} {label} rep{rep}: {stream.n_requests} requests "
                f"({stream.loop} loop) ...",
                flush=True,
            )
            run, window_s, reasons, checks = measure(stream)
            gpu_samples = sampler.stop() if sampler else []
            after = adapter.scrape_counters(base_url)
            counters = adapter.server_counters(before, after)

            metrics, samples, in_order = compute_metrics(run, stream.lcfg, window_s)
            reasons = list(static_reasons) + reasons
            if metrics["completed"] == 0:
                reasons.append("no_completed_requests: no steady-window request completed")
            if not healthy(adapter, base_url):
                reasons.append("server_unhealthy_after_run: /health did not return 200")
            if server is not None and not server.alive():
                reasons.append("server_exited_during_run")

            found = anomalies.run_anomalies(in_order, gpu_samples)
            if spec.kv_pool_tokens and counters.get("preemptions") == 0:
                found.append(
                    {
                        "kind": "no_preemption_under_kv_pressure",
                        "detail": f"{spec.id} caps the KV pool at {kv_tokens} tokens "
                        "to force preemption, but the engine reported 0",
                    }
                )

            prov = Provenance.capture(repo_root=REPO_ROOT, seed=stream.seed)
            prov_ok, prov_blockers = prov.is_publishable()
            blockers = prov_blockers + hw_blockers
            art = build_artifact(
                arm=arm.id,
                engine=engine_block,
                workload={
                    "id": spec.id,
                    "config_path": _relpath(spec.config_path),
                    "config_sha256": spec.config_sha256,
                    "seed": stream.seed,
                    "point": point,
                    "loop": stream.loop,
                    "base_seed": spec.seed,
                    "concurrency": stream.concurrency,
                    "kv_pool_tokens": kv_tokens,
                    "request_stream_sha256": stream.sha256,
                    "workload_fingerprint": stream.workload_fingerprint,
                    "n_requests": stream.n_requests,
                    "prompt_render": stream.renderer,
                    "realized": {
                        k: stream.realized.get(k)
                        for k in (
                            "prompt_len",
                            "output_len",
                            "sharing",
                            "rendered_prompt_tokens_mean",
                            "rendered_prompt_tokens_min",
                            "rendered_prompt_tokens_max",
                            "degeneracy_warnings",
                        )
                    },
                    "server_usage": usage_summary(run, arm.engine),
                    "notes": [CLOSED_LOOP_NOTE if stream.loop == "closed" else OPEN_LOOP_NOTE],
                },
                rep=rep,
                hardware={**hardware, "gpu_samples": gpu_samples},
                provenance={
                    "repo_sha": prov.repo_sha,
                    "repo_dirty": prov.repo_dirty,
                    "started_at": run.started_utc + "Z",
                    "harness_version": HARNESS_VERSION,
                    "publishable": prov_ok and hw_ok,
                    "publish_blockers": blockers,
                    "allocation_id": prov.allocation_id,
                    "slurm_qos": prov.slurm_qos,
                    "python": prov.python_version,
                    "torch": prov.torch_version,
                    "flashinfer_version": prov.flashinfer_version,
                    "wall_seconds": run.wall_seconds,
                },
                slo=slo.as_dict(),
                validity={
                    "valid": not reasons,
                    "reasons": reasons,
                    "attention_backend": attention_backend,
                    "checks": checks,
                },
                metrics=metrics,
                server_counters=counters,
                anomalies=found,
                samples=samples,
            )
            path = arm_dir.parent.parent / artifact_relpath(spec.id, arm.id, label, rep)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(art, indent=1, default=str))
            written.setdefault(label, []).append(path)
            any_invalid |= bool(reasons)
            status = (
                "VALID"
                if not reasons
                else "INVALID: " + "; ".join(r.split(":")[0] for r in reasons)
            )
            print(
                f"[{arm.id}] -> {_relpath(path)}  {status}  "
                f"out {metrics['output_tok_s']:.1f} tok/s  goodput {metrics['goodput_rps']:.2f}"
                f"  ok {metrics['completed']}/{metrics['steady_requests']}"
                f"  ({sum(1 for r in run.results if r.outcome == Outcome.COMPLETED)} total)",
                flush=True,
            )

    for label in written:
        apply_rep_spread(sorted(arm_dir.glob(f"{label}_rep*.json")), args.cv_threshold)
    return 2 if any_invalid else 0


def usage_summary(run: Any, engine: str) -> dict[str, Any]:
    """
    Server-reported prompt/cached token counts for the steady window, from the
    final `usage` chunk (stream_options.include_usage). Ours does not send one
    (docs/xengine/ENGINE_FLAGS.md §3): recorded as null with a note, and the
    tokenizer-side count is in `realized.rendered_prompt_tokens_*`.
    """
    steady = [r for r in run.results if r.spec.phase == "steady"]
    usages = [r.usage for r in steady if isinstance(getattr(r, "usage", None), dict)]
    prompt = [u["prompt_tokens"] for u in usages if isinstance(u.get("prompt_tokens"), int)]
    cached = [
        (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        for u in usages
        if isinstance((u.get("prompt_tokens_details") or {}).get("cached_tokens"), int)
    ]
    if not prompt:
        return {
            "prompt_tokens": None,
            "cached_tokens_total": None,
            "n_with_usage": len(usages),
            "note": (
                "server sent no usage in the stream"
                + (" (ours ignores stream_options)" if engine == "ours" else "")
            ),
        }
    return {
        "prompt_tokens": {
            "mean": sum(prompt) / len(prompt),
            "min": min(prompt),
            "max": max(prompt),
            "total": sum(prompt),
        },
        "cached_tokens_total": sum(cached) if cached else None,
        "n_with_usage": len(usages),
        "note": None,
    }


def apply_rep_spread(paths: list[Path], threshold: float) -> list[dict[str, Any]]:
    """Recompute rep-to-rep spread over every artifact of a point; write it into each."""
    arts = []
    for p in paths:
        try:
            arts.append((p, json.loads(p.read_text())))
        except (OSError, ValueError):
            continue
    spread = anomalies.rep_spread([a for _, a in arts], threshold)
    for p, a in arts:
        kept = [
            x
            for x in a.get("anomalies", [])
            if not str(x.get("kind", "")).startswith("rep_spread_")
        ]
        a["anomalies"] = kept + spread
        p.write_text(json.dumps(a, indent=1, default=str))
    return spread


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross-engine benchmark: one arm x one workload (docs/xengine/SPEC.md).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--arm", required=True, help="arm id from the SPEC arm table")
    p.add_argument("--workload", required=True, help="W1..W5 or a path to a workload YAML")
    p.add_argument("--reps", type=int, default=3, help="repetitions per point (SPEC: >= 3)")
    p.add_argument("--out", default="results/xengine")
    p.add_argument(
        "--url",
        default=None,
        help="measure an already-running server at this base URL instead of launching one",
    )
    p.add_argument(
        "--server-log",
        default=None,
        help="with --url: that server's log, for backend/config detection",
    )
    p.add_argument("--model-path", default=None, help="checkpoint dir (default $LLM_WEIGHTS_PATH)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    p.add_argument(
        "--points", default=None, help="comma-separated subset of point values, e.g. 1,4,16"
    )
    p.add_argument(
        "--kv-pool-tokens",
        type=int,
        default=None,
        help="KV pool size in tokens (overrides the workload's; vllm-matched)",
    )
    p.add_argument(
        "--renderer", default="auto", help="prompt renderer: auto | words | <tokenizer path>"
    )
    p.add_argument("--slo", default=None, help="SLO YAML (default configs/slo.yaml)")
    p.add_argument("--health-timeout", type=float, default=900.0)
    p.add_argument(
        "--gpu-sample-interval",
        type=float,
        default=2.0,
        help="seconds between nvidia-smi samples; 0 disables",
    )
    p.add_argument("--cv-threshold", type=float, default=anomalies.DEFAULT_CV_THRESHOLD)
    p.add_argument(
        "--plain-factory",
        action="store_true",
        help="ours: launch serving.server.app:build_default_app directly "
        "(no backend log line; pre-flight only)",
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="ours: skip the FlashInfer pre-flight (the run is then invalid "
        "unless the server log names the backend)",
    )
    p.add_argument("--dry-run", action="store_true", help="print the resolved launch spec and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reps < 1:
        print("--reps must be >= 1", file=sys.stderr)
        return 3
    if args.reps < 3:
        print(
            f"warning: --reps {args.reps} < 3; SPEC hard rule 3 needs >= 3 per cell",
            file=sys.stderr,
        )
    return run_cell(args)


if __name__ == "__main__":
    raise SystemExit(main())
