"""
Engine adapters, the arm table, and server process lifecycle.

One `EngineAdapter` per engine (ours, vllm, sglang). An adapter knows how to:

  * build the launch command and environment for an arm,
  * name the model on the wire, find the health endpoint, ask the version,
  * scrape the engine's own metrics and normalize them into SPEC's
    `server_counters` {preemptions, evictions, prefix_hit_rate, raw},
  * find the active attention backend in the server's stdout log.

NORMALIZED COUNTERS ARE `None` WHEN NOT EXPOSED, NEVER 0
--------------------------------------------------------
"The engine does not report evictions" and "the engine evicted nothing" are
different statements, and only the second is a measurement. Every normalized
counter is a DELTA between a scrape just before the measured run and one just
after it; if either scrape lacks the metric, the value is `None`. The raw
scrapes travel in `raw` so a reader can audit the derivation.

The three engines do not measure the same thing under the same name, so each
normalized value carries its definition and unit in `raw`:

  prefix_hit_rate  ours: BLOCK granularity (delta cache_blocks_reused /
                   delta cache_blocks_required, JSON scheduler snapshot)
                   vLLM: TOKENS (delta prefix_cache_hits / delta queries)
                   SGLang: TOKENS (delta prefill_effective_tokens with
                   mode in {device,host,storage}_hit / delta over all modes)
  preemptions      ours: scheduler preemptions_total; vLLM: num_preemptions;
                   SGLang: num_retracted_requests (its name for preemption)
  evictions        ours: radix-cache block evictions (BLOCKS); SGLang:
                   evicted_tokens (TOKENS); vLLM: not exposed -> None

Flag and metric names are verified against the pinned sources
(docs/xengine/ENGINE_FLAGS.md); they live in one block below.

ATTENTION BACKEND (ours) — docs/xengine/FINDINGS_OURS.md F-002
---------------------------------------------------------------
Our server does not expose the active attention backend anywhere: not in
/health, not in /metrics, not in its stdout (the only startup line is
`config: kv_blocks=... preemption=... prefix_cache=... static_batching=...`).
FlashInfer -> PagedTorch fallback is silent by design (R18). Two independent
pieces of evidence are collected, and the run is valid only if neither says
anything other than FlashInfer:

  1. PRE-FLIGHT (always, before launch): in the server's interpreter
     ($XENGINE_SERVER_PY), on the same GPU, construct FlashInferBackend with
     the same arguments build_default_app uses. Verdict: "flashinfer" or
     "paged_torch(<exception>)". This is what `validity.attention_backend`
     records.
  2. LOG LINE (default launch only): the server is started through
     `bench.xengine.ours_factory:build_app`, which calls the same
     `build_default_app()` and prints `xengine: attention_backend=<class>`.
     `detect_attention_backend` greps the server log for it. This closes the
     gap the pre-flight cannot: a failure that only happens at the real,
     VRAM-sized pool. With the plain factory nothing distinguishing is logged
     and only the pre-flight is available.

If neither piece of evidence exists, the backend is "unknown" and the run is
invalid with reason `attention_backend_unverified`.

The startup `config:` line IS parsed and cross-checked against the launch
environment (prefix cache on/off, KV blocks), because /health.capabilities is
hardcoded and cannot be trusted for that (F-001).

vLLM and SGLang do log their backend choice; it is grepped and recorded for
information but does not affect validity (SPEC requires it only for ours).
"""

from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from bench.xengine.config import SERVED_MODEL_NAME
from bench.xengine.ours_factory import BACKEND_LOG_PREFIX

REPO_ROOT = Path(__file__).resolve().parents[2]

# ===========================================================================
# ENGINE FLAGS AND METRIC NAMES — verified against source at the pinned
# versions, vLLM 0.31.0 (db9527a4) and SGLang 0.5.21 (e00930c5). See
# docs/xengine/ENGINE_FLAGS.md (path:line citations) and SOURCE_NOTES.md.
#
# Every vLLM / SGLang CLI flag and Prometheus metric name the harness depends
# on is in this block and nowhere else. Still UNVERIFIED until a live scrape on
# the study node (ENGINE_FLAGS.md §4): the exact label sets in /metrics. A
# missing metric shows up as `None` plus the raw dump, never as a wrong number.
# ===========================================================================

# --- shared ----------------------------------------------------------------
# A capability limit, not a performance knob. Applied identically to vLLM and
# SGLang so the longest W2 request (4096 prompt + 4096 output + chat template)
# fits, and so vLLM does not refuse to start when W4 caps the KV pool below the
# model's native 131072-token context (vLLM requires max_model_len to fit in
# the pool).
MAX_MODEL_LEN = 16384
KV_BLOCK_SIZE = 16  # ours (build_default_app default) and vLLM --block-size

# Request-body extras (stream_options) live in bench/xengine/config.py:
# STREAM_OPTIONS, because they are part of the request stream's identity.

# --- vLLM 0.31.0 -----------------------------------------------------------
# Launched as `$XENGINE_SERVER_PY -m <module> --model <path>` rather than via a
# `vllm` binary on PATH, because each engine lives in its own env (Makefile).
# ENGINE_FLAGS.md documents `vllm serve <model>`; the module form is the
# OpenAI server entry point it dispatches to. UNVERIFIED that the module still
# accepts `--model` identically at 0.31.0 — check on first launch.
VLLM_MODULE = "vllm.entrypoints.openai.api_server"
VLLM_FLAG_MODEL = "--model"
VLLM_FLAG_DTYPE = "--dtype"  # arg_utils.py:937
VLLM_FLAG_HOST = "--host"
VLLM_FLAG_PORT = "--port"
VLLM_FLAG_SERVED_NAME = "--served-model-name"
VLLM_FLAG_SEED = "--seed"  # arg_utils.py:938
VLLM_FLAG_MAX_MODEL_LEN = "--max-model-len"
VLLM_FLAG_ENFORCE_EAGER = "--enforce-eager"  # arg_utils.py:960 (graphs AND compile)
VLLM_FLAG_COMPILATION_CONFIG = "--compilation-config"  # arg_utils.py:1818 (-cc)
VLLM_NO_CUDAGRAPH = '{"cudagraph_mode": "NONE"}'  # compilation.py:53-63
VLLM_FLAG_NO_ASYNC_SCHED = "--no-async-scheduling"  # arg_utils.py:1739
VLLM_FLAG_NO_PREFIX_CACHING = "--no-enable-prefix-caching"  # arg_utils.py:1339
VLLM_FLAG_MAX_NUM_SEQS = "--max-num-seqs"  # arg_utils.py:1682
VLLM_FLAG_NUM_GPU_BLOCKS = "--num-gpu-blocks-override"  # arg_utils.py:1336 (blocks)
VLLM_FLAG_BLOCK_SIZE = "--block-size"  # arg_utils.py:1324
VLLM_FLAG_PROMPT_TOKENS_DETAILS = "--enable-prompt-tokens-details"  # cli_args.py:145
VLLM_HEALTH_PATH = "/health"
VLLM_VERSION_PATH = "/version"  # -> {"version": "x.y.z"}
VLLM_METRICS_PATH = "/metrics"  # always mounted; do not pass --disable-log-stats
VLLM_METRIC_PREEMPTIONS = "vllm:num_preemptions_total"  # loggers.py:677
VLLM_METRIC_PREFIX_HITS = "vllm:prefix_cache_hits_total"  # loggers.py:611 (tokens)
VLLM_METRIC_PREFIX_QUERIES = "vllm:prefix_cache_queries_total"  # loggers.py:600 (tokens)
# Evictions: no counter in vLLM 0.31.0 (only opt-in histograms) -> None.
VLLM_BACKEND_LOG_RE = re.compile(r"Using ([A-Za-z0-9_ ]+?) backend", re.IGNORECASE)
# Resolved defaults vLLM prints at startup (non-default args dump, chunked
# prefill notice, KV cache size). Grepped into engine.flags.resolved.
VLLM_RESOLVED_RES = {
    "max_num_seqs": re.compile(r"max_num_seqs['\"]?\s*[=:]\s*(\d+)"),
    "max_num_batched_tokens": re.compile(r"max_num_batched_tokens['\"]?\s*[=:]\s*(\d+)"),
    "max_model_len": re.compile(r"max_model_len['\"]?\s*[=:]\s*(\d+)"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"),
    "num_gpu_blocks": re.compile(r"#\s*GPU blocks:\s*([\d,]+)"),
}

# --- SGLang 0.5.21 ---------------------------------------------------------
SGLANG_MODULE = "sglang.launch_server"
SGLANG_FLAG_MODEL = "--model-path"
SGLANG_FLAG_DTYPE = "--dtype"  # fields/model.py:152-168
SGLANG_FLAG_HOST = "--host"
SGLANG_FLAG_PORT = "--port"
SGLANG_FLAG_SERVED_NAME = "--served-model-name"
SGLANG_FLAG_SEED = "--random-seed"  # fields/device.py:47 (default: random)
SGLANG_FLAG_CONTEXT_LEN = "--context-length"
SGLANG_FLAG_ENABLE_METRICS = "--enable-metrics"  # observability.py:70; no counters without
SGLANG_FLAG_CACHE_REPORT = "--enable-cache-report"  # fields/serving.py:189
SGLANG_FLAG_DISABLE_RADIX = "--disable-radix-cache"  # fields/memory.py:64
SGLANG_FLAG_DISABLE_OVERLAP = "--disable-overlap-schedule"  # fields/schedule.py:187
SGLANG_FLAG_DISABLE_DECODE_GRAPH = "--disable-decode-cuda-graph"  # exec_.py:537-544
SGLANG_FLAG_DISABLE_PREFILL_GRAPH = "--disable-prefill-cuda-graph"  # exec_.py:537-544
SGLANG_FLAG_SCHEDULE_POLICY = "--schedule-policy"  # schedule.py:82-98 (default fcfs)
SGLANG_FLAG_MAX_TOTAL_TOKENS = "--max-total-tokens"  # schedule.py:39 (tokens; lowers only)
SGLANG_FLAG_MAX_RUNNING = "--max-running-requests"  # schedule.py:31
SGLANG_HEALTH_PATH = "/health"
SGLANG_VERSION_PATHS = ["/version", "/get_server_info"]  # look for "version"
SGLANG_METRICS_PATH = "/metrics"
SGLANG_METRIC_PREEMPTIONS = "sglang:num_retracted_requests_total"  # collector.py:474-479
SGLANG_METRIC_PREFILL_TOKENS = "sglang:prefill_effective_tokens_total"  # :913-921, by `mode`
SGLANG_PREFIX_HIT_MODES = ("device_hit", "host_hit", "storage_hit")
SGLANG_METRIC_EVICTIONS = "sglang:evicted_tokens_total"  # collector.py:2213-2220 (tokens)
# NOT used: sglang:cache_hit_rate is the last prefill batch only (:305-310).
SGLANG_BACKEND_LOG_RE = re.compile(r"attention_backend='?([A-Za-z0-9_]+)'?")
SGLANG_RESOLVED_RES = {
    "max_running_requests": re.compile(r"max_running_requests=(\d+)"),
    "chunked_prefill_size": re.compile(r"chunked_prefill_size=(-?\d+)"),
    "max_total_num_tokens": re.compile(r"max_total_num_tokens=(\d+)"),
    "max_prefill_tokens": re.compile(r"max_prefill_tokens=(\d+)"),
    "schedule_policy": re.compile(r"schedule_policy='?([a-z_-]+)'?"),
    "page_size": re.compile(r"page_size=(\d+)"),
}
# ===========================================================================
# end ENGINE FLAGS block
# ===========================================================================

# --- ours (read from this repo, not guessed) --------------------------------
OURS_FACTORY_PROBE = "bench.xengine.ours_factory:build_app"
OURS_FACTORY_PLAIN = "serving.server.app:build_default_app"
OURS_HEALTH_PATH = "/health"
OURS_METRICS_PATH = "/metrics"  # JSON; see serving/server/app.py
OURS_MAX_BATCH_SIZE = 32  # build_default_app default
OURS_BACKEND_NAMES = {"FlashInferBackend": "flashinfer", "PagedTorchBackend": "paged_torch"}
# The startup line build_default_app prints (serving/server/app.py:1263-1269).
OURS_CONFIG_LINE_RE = re.compile(
    r"config: kv_blocks=(\d+) preemption=(\S+) prefix_cache=(on|OFF) static_batching=(\S+)"
)


def server_python() -> str:
    """
    Interpreter that launches an arm's SERVER. The Makefile sets
    XENGINE_SERVER_PY per arm (vLLM, SGLang and ours each have their own env);
    the harness client itself always runs in ours.
    """
    return os.environ.get("XENGINE_SERVER_PY") or sys.executable


def grep_resolved(log_text: str, patterns: dict[str, re.Pattern[str]]) -> dict[str, Any]:
    """Last match of each pattern in the server log; ints where they parse."""
    out: dict[str, Any] = {}
    for key, rx in patterns.items():
        found = rx.findall(log_text)
        if found:
            v = found[-1].replace(",", "")
            out[key] = int(v) if re.fullmatch(r"-?\d+", v) else v
    return out


# ---------------------------------------------------------------------------
# Arms (docs/xengine/SPEC.md "Engines and arms")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    id: str
    engine: str  # "ours" | "vllm" | "sglang"
    diagnostic: bool
    description: str
    extra_args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    matched: bool = False  # vllm-matched: needs a KV pool to match
    blocked_reason: str | None = None  # set => the arm cannot be run at all


ARMS: dict[str, ArmSpec] = {
    a.id: a
    for a in [
        ArmSpec("ours", "ours", False, "this serving layer, defaults (FlashInfer required)"),
        ArmSpec(
            "ours-noprefix",
            "ours",
            True,
            "ours with the radix prefix cache off",
            env=(("SERVING_PREFIX_CACHE", "0"),),
        ),
        # W5 appendix. The fp16 side reuses the `ours` cells.
        ArmSpec(
            "ours-int8",
            "ours",
            True,
            "ours with int8 weight-only quantization (W5 appendix)",
            blocked_reason=(
                "build_default_app (serving/server/app.py) has no int8 / quantization "
                "knob, and serving/ may not be modified in this study (SPEC rule 4). "
                "Refusing rather than running fp16 under an int8 label."
            ),
        ),
        ArmSpec("vllm", "vllm", False, "vLLM defaults, fp16"),
        ArmSpec(
            "vllm-eager",
            "vllm",
            True,
            "CUDA graphs AND torch.compile off (diagnostic)",
            extra_args=(VLLM_FLAG_ENFORCE_EAGER,),
        ),
        ArmSpec(
            "vllm-nograph",
            "vllm",
            True,
            "CUDA graphs off, torch.compile kept (diagnostic)",
            extra_args=(VLLM_FLAG_COMPILATION_CONFIG, VLLM_NO_CUDAGRAPH),
        ),
        ArmSpec(
            "vllm-noasync",
            "vllm",
            True,
            "async (CPU/GPU-overlapped) scheduling off (diagnostic)",
            extra_args=(VLLM_FLAG_NO_ASYNC_SCHED,),
        ),
        ArmSpec(
            "vllm-noprefix",
            "vllm",
            True,
            "prefix caching off (diagnostic)",
            extra_args=(VLLM_FLAG_NO_PREFIX_CACHING,),
        ),
        ArmSpec(
            "vllm-matched",
            "vllm",
            True,
            "max-num-seqs and KV pool matched to ours (diagnostic)",
            extra_args=(VLLM_FLAG_MAX_NUM_SEQS, str(OURS_MAX_BATCH_SIZE)),
            matched=True,
        ),
        ArmSpec("sglang", "sglang", False, "SGLang defaults, fp16 (FCFS schedule policy)"),
        ArmSpec(
            "sglang-noradix",
            "sglang",
            True,
            "radix cache off (diagnostic)",
            extra_args=(SGLANG_FLAG_DISABLE_RADIX,),
        ),
        ArmSpec(
            "sglang-nooverlap",
            "sglang",
            True,
            "overlap scheduler off (diagnostic)",
            extra_args=(SGLANG_FLAG_DISABLE_OVERLAP,),
        ),
        ArmSpec(
            "sglang-eager",
            "sglang",
            True,
            "decode and prefill CUDA graphs off (diagnostic)",
            extra_args=(SGLANG_FLAG_DISABLE_DECODE_GRAPH, SGLANG_FLAG_DISABLE_PREFILL_GRAPH),
        ),
        ArmSpec(
            "sglang-lpm",
            "sglang",
            True,
            "longest-prefix-match schedule policy ON (diagnostic, W3)",
            extra_args=(SGLANG_FLAG_SCHEDULE_POLICY, "lpm"),
        ),
    ]
}


def get_arm(arm_id: str) -> ArmSpec:
    if arm_id not in ARMS:
        raise KeyError(f"unknown arm {arm_id!r}; expected one of {sorted(ARMS)}")
    return ARMS[arm_id]


# ---------------------------------------------------------------------------
# Prometheus text
# ---------------------------------------------------------------------------

_PROM_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+]?[0-9.eE+-]+|NaN|[+-]?Inf)(\s+\d+)?$"
)
_PROM_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> dict[str, float]:
    """
    Flatten Prometheus exposition text into `{key: value}`:

      name                  value summed over every label set (one model per
                            server, so usually the single series)
      name{label="value"}   value summed over series carrying that one label
                            pair — how SGLang's per-`mode` prefill counter is
                            split into hits and queries

    Histogram `_bucket` series and prometheus_client's `_created` timestamps are
    dropped; `_sum` and `_count` are kept.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2), m.group(3)
        if name.endswith("_bucket") or name.endswith("_created"):
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + v
        for k, lv in _PROM_LABEL.findall(labels or ""):
            if k in ("model_name", "engine"):
                continue
            key = f'{name}{{{k}="{lv}"}}'
            out[key] = out.get(key, 0.0) + v
    return out


def _delta(before: dict[str, Any], after: dict[str, Any], name: str | None) -> float | None:
    if name is None:
        return None
    b, a = before.get(name), after.get(name)
    if not isinstance(b, (int, float)) or not isinstance(a, (int, float)):
        return None
    if isinstance(a, bool) or isinstance(b, bool):
        return None
    return float(a) - float(b)


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return num / den


# ---------------------------------------------------------------------------
# Launch spec + process lifecycle
# ---------------------------------------------------------------------------


@dataclass
class LaunchSpec:
    cmd: list[str]
    env: dict[str, str]  # ADDITIONS to the inherited environment
    flags: dict[str, Any]  # resolved knobs, recorded in the artifact

    def as_dict(self) -> dict[str, Any]:
        return {"launch_cmd": list(self.cmd), "env": dict(self.env), "flags": dict(self.flags)}


class ServerStartError(RuntimeError):
    pass


@dataclass
class ServerProcess:
    """
    One engine server as a subprocess, stdout+stderr to `log_path`.

    Started in its own session so `stop()` can signal the whole process group:
    vLLM and SGLang fork worker processes, and terminating only the parent can
    leave a GPU-holding child alive that poisons the next arm's launch.
    """

    spec: LaunchSpec
    log_path: Path
    health_url: str
    proc: subprocess.Popen[bytes] | None = None
    started_at: float | None = None
    healthy_after_s: float | None = None
    _log_fh: Any = field(default=None, repr=False)

    def start(self) -> ServerProcess:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_path, "ab")  # closed in stop()
        self._log_fh.write(f"# xengine launch: {' '.join(self.spec.cmd)}\n".encode())
        self._log_fh.write(f"# xengine env additions: {self.spec.env}\n".encode())
        self._log_fh.flush()
        env = {**os.environ, **self.spec.env}
        self.proc = subprocess.Popen(
            self.spec.cmd,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(REPO_ROOT),
            start_new_session=True,
        )
        self.started_at = time.monotonic()
        return self

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, n: int = 40) -> str:
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])

    def wait_healthy(self, timeout_s: float = 600.0, poll_s: float = 1.0) -> float:
        """Poll the health URL until 200. Raises if the process dies or time runs out."""
        assert self.proc is not None and self.started_at is not None, "start() first"
        deadline = self.started_at + timeout_s
        with httpx.Client(timeout=5.0) as c:
            while time.monotonic() < deadline:
                if not self.alive():
                    raise ServerStartError(
                        f"server exited with code {self.proc.returncode} during startup; "
                        f"log {self.log_path}:\n{self.log_tail()}"
                    )
                try:
                    if c.get(self.health_url).status_code == 200:
                        self.healthy_after_s = time.monotonic() - self.started_at
                        return self.healthy_after_s
                except httpx.HTTPError:
                    pass
                time.sleep(poll_s)
        self.stop()
        raise ServerStartError(
            f"server not healthy at {self.health_url} after {timeout_s:.0f}s; "
            f"log {self.log_path}:\n{self.log_tail()}"
        )

    def stop(self, grace_s: float = 30.0) -> int | None:
        rc = None
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    self.proc.terminate()
                try:
                    self.proc.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        self.proc.kill()
                    self.proc.wait(timeout=10)
            rc = self.proc.returncode
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        return rc

    def __enter__(self) -> ServerProcess:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def _local_version(module: str, python: str | None = None) -> str | None:
    """`import <module>; print(__version__)` in the launch interpreter."""
    try:
        out = subprocess.run(
            [python or server_python(), "-c", f"import {module}; print({module}.__version__)"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


PREFLIGHT_TAG = "XENGINE_PREFLIGHT "
# Mirrors serving/server/app.py:1208-1223 (FlashInferBackend kwargs) with the
# model shape read from the checkpoint's config.json (what engine.loader reads).
PREFLIGHT_SCRIPT = r"""
import json, os, sys
tag = "XENGINE_PREFLIGHT "
device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
try:
    with open(os.path.join(os.environ["LLM_WEIGHTS_PATH"], "config.json")) as f:
        cfg = json.load(f)
    n_heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_heads
    from serving.backends.flashinfer_backend import FlashInferBackend
    FlashInferBackend(
        num_layers=cfg["num_hidden_layers"], num_blocks=16, block_size=16,
        n_kv_heads=cfg["num_key_value_heads"], n_heads=n_heads, head_dim=head_dim,
        device=device,
    )
    try:
        import flashinfer
        fv = getattr(flashinfer, "__version__", "unknown")
    except Exception:
        fv = None
    print(tag + json.dumps({"backend": "flashinfer", "flashinfer_version": fv}), flush=True)
except Exception as exc:
    msg = f"{type(exc).__name__}: {exc}".replace(chr(10), " ")[:300]
    print(tag + json.dumps({"backend": f"paged_torch({msg})"}), flush=True)
"""


def parse_preflight_output(stdout: str, returncode: int | None, stderr: str = "") -> dict[str, Any]:
    import json

    for line in reversed((stdout or "").splitlines()):
        if line.startswith(PREFLIGHT_TAG):
            try:
                d = dict(json.loads(line[len(PREFLIGHT_TAG) :]))
            except ValueError:
                break
            d["returncode"] = returncode
            return d
    tail = " ".join((stderr or "").strip().splitlines()[-3:])[:300]
    return {
        "backend": f"paged_torch(preflight produced no verdict, rc={returncode}: {tail})",
        "returncode": returncode,
    }


class EngineAdapter:
    """Base: shared plumbing. Subclasses fill the engine-specific parts."""

    name: str = ""
    health_path: str = "/health"
    metrics_path: str = "/metrics"
    model_name: str = SERVED_MODEL_NAME

    # -- launch ---------------------------------------------------------------

    def launch_spec(
        self,
        arm: ArmSpec,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        kv_pool_tokens: int | None = None,
    ) -> LaunchSpec:
        raise NotImplementedError

    # -- URLs -----------------------------------------------------------------

    def health_url(self, base_url: str) -> str:
        return base_url.rstrip("/") + self.health_path

    def chat_url(self, base_url: str) -> str:
        return base_url.rstrip("/") + "/v1/chat/completions"

    # -- version --------------------------------------------------------------

    def query_version(self, base_url: str, client: httpx.Client | None = None) -> str | None:
        raise NotImplementedError

    # -- counters -------------------------------------------------------------

    def scrape_counters(self, base_url: str, client: httpx.Client | None = None) -> dict[str, Any]:
        """One raw snapshot of the engine's metrics, as a flat-ish dict."""
        owns = client is None
        c = client or httpx.Client(timeout=10.0)
        try:
            resp = c.get(base_url.rstrip("/") + self.metrics_path)
            if resp.status_code != 200:
                return {"_error": f"HTTP {resp.status_code} from {self.metrics_path}"}
            return self.parse_metrics(resp.text)
        except httpx.HTTPError as exc:
            return {"_error": f"{type(exc).__name__}: {exc}"}
        finally:
            if owns:
                c.close()

    def parse_metrics(self, body: str) -> dict[str, Any]:
        return parse_prometheus(body)

    def server_counters(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        """SPEC `server_counters` from two snapshots. Missing -> None, never 0."""
        raise NotImplementedError

    # -- logs -----------------------------------------------------------------

    def detect_attention_backend(self, log_text: str) -> str | None:
        return None

    def resolved_defaults(self, log_text: str) -> dict[str, Any]:
        """Engine defaults as RESOLVED at startup, grepped from its log."""
        return {}


class OursAdapter(EngineAdapter):
    name = "ours"
    health_path = OURS_HEALTH_PATH
    metrics_path = OURS_METRICS_PATH

    def __init__(self, probe_backend: bool = True, python: str | None = None) -> None:
        self.probe_backend = probe_backend
        self.python = python or server_python()

    def launch_spec(self, arm, model_path, host="127.0.0.1", port=8000, kv_pool_tokens=None):
        factory = OURS_FACTORY_PROBE if self.probe_backend else OURS_FACTORY_PLAIN
        cmd = [
            self.python,
            "-m",
            "uvicorn",
            factory,
            "--factory",
            "--host",
            host,
            "--port",
            str(port),
            "--log-level",
            "warning",
        ]
        # Same environment scripts/p2_bench.sbatch sets up.
        env = self.launch_env(model_path)
        env.update(dict(arm.env))
        flags: dict[str, Any] = {
            "factory": factory,
            # From the launch environment, NOT /health.capabilities, which is
            # hardcoded False (docs/xengine/FINDINGS_OURS.md F-001).
            "prefix_cache": env.get(
                "SERVING_PREFIX_CACHE", os.environ.get("SERVING_PREFIX_CACHE", "1")
            )
            != "0",
            "preemption": (
                env.get("SERVING_PREEMPTION") or os.environ.get("SERVING_PREEMPTION") or "recompute"
            ).lower(),
            "block_size": KV_BLOCK_SIZE,
            "max_batch_size": OURS_MAX_BATCH_SIZE,
            "watermark_blocks": OURS_MAX_BATCH_SIZE,
            "dtype": "engine loader default (no dtype knob in build_default_app)",
            "kv_pool_tokens": None,
        }
        if kv_pool_tokens:
            blocks = kv_pool_tokens // KV_BLOCK_SIZE
            env["SERVING_KV_BLOCKS"] = str(blocks)
            flags["kv_pool_tokens"] = blocks * KV_BLOCK_SIZE
            flags["kv_blocks"] = blocks
            flags["kv_pool_note"] = (
                f"ours reserves a watermark of {OURS_MAX_BATCH_SIZE} blocks "
                f"({OURS_MAX_BATCH_SIZE * KV_BLOCK_SIZE} tokens) from admission"
            )
        return LaunchSpec(cmd=cmd, env=env, flags=flags)

    def query_version(self, base_url, client=None):
        """Ours is this repo: version = commit (+ engine submodule tag)."""
        from serving.metrics.artifact import Provenance

        prov = Provenance.capture(repo_root=REPO_ROOT)
        if prov.repo_sha is None:
            return None
        v = f"git:{prov.repo_sha[:12]}" + ("-dirty" if prov.repo_dirty else "")
        if prov.engine_tag or prov.engine_sha:
            v += f"+engine:{prov.engine_tag or (prov.engine_sha or '')[:12]}"
        return v

    def parse_metrics(self, body: str) -> dict[str, Any]:
        import json

        try:
            d = json.loads(body)
        except json.JSONDecodeError as exc:
            return {"_error": f"non-JSON /metrics: {exc}"}
        # Keep the parts that carry counters; drop specs/notes text.
        return {k: d.get(k) for k in ("server", "scheduler", "allocator") if k in d}

    def server_counters(self, before, after):
        sb = (before or {}).get("scheduler") or {}
        sa = (after or {}).get("scheduler") or {}
        pre = _delta(sb, sa, "preemptions_total" if "preemptions_total" in sa else None)
        ev = _delta(sb, sa, "cache_evictions" if "cache_evictions" in sa else None)
        reused = _delta(sb, sa, "cache_blocks_reused" if "cache_blocks_reused" in sa else None)
        required = _delta(
            sb, sa, "cache_blocks_required" if "cache_blocks_required" in sa else None
        )
        return {
            "preemptions": pre,
            "evictions": ev,
            "prefix_hit_rate": _ratio(reused, required),
            "raw": {
                "before": before,
                "after": after,
                "definitions": {
                    "preemptions": "delta scheduler.preemptions_total",
                    "evictions": "delta scheduler.cache_evictions (radix block evictions); "
                    "None when the prefix cache is off",
                    "prefix_hit_rate": "delta cache_blocks_reused / delta "
                    "cache_blocks_required (BLOCK granularity)",
                },
            },
        }

    def preflight_attention_backend(
        self, model_path: str, device: str = "cuda:0", timeout_s: float = 300.0
    ) -> dict[str, Any]:
        """
        F-002 pre-flight: in the SERVER's interpreter, on the same GPU, construct
        FlashInferBackend exactly as build_default_app does
        (serving/server/app.py:1208-1223) and report whether it succeeded.

        Returns {"backend": "flashinfer" | "paged_torch(<exception>)", ...}.
        Run BEFORE the server starts so the GPU is free. The pool here is tiny
        (16 blocks) rather than VRAM-sized, so an OOM-only failure at full size
        would not reproduce — that residual gap is what the probe factory's
        log line (detect_attention_backend) closes.
        """
        env = {**os.environ, **self.launch_env(model_path)}
        try:
            out = subprocess.run(
                [self.python, "-c", PREFLIGHT_SCRIPT, device],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
                cwd=str(REPO_ROOT),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"backend": f"paged_torch(preflight could not run: {exc})", "returncode": None}
        return parse_preflight_output(out.stdout, out.returncode, out.stderr)

    def launch_env(self, model_path: str) -> dict[str, str]:
        pythonpath = os.pathsep.join(
            p
            for p in [
                str(REPO_ROOT),
                str(REPO_ROOT / "vendor" / "llm_inference_engine"),
                os.environ.get("PYTHONPATH", ""),
            ]
            if p
        )
        return {"LLM_WEIGHTS_PATH": model_path, "HF_HUB_OFFLINE": "1", "PYTHONPATH": pythonpath}

    def resolved_defaults(self, log_text: str) -> dict[str, Any]:
        return self.config_from_log(log_text) or {}

    @staticmethod
    def config_from_log(log_text: str) -> dict[str, Any] | None:
        """Parse the startup `config:` line (serving/server/app.py:1263-1269)."""
        found = OURS_CONFIG_LINE_RE.findall(log_text)
        if not found:
            return None
        kv, pre, pc, static = found[-1]
        return {
            "kv_blocks": int(kv),
            "preemption": pre,
            "prefix_cache": pc == "on",
            "static_batching": static == "True",
        }

    def detect_attention_backend(self, log_text: str) -> str | None:
        for line in reversed(log_text.splitlines()):
            if BACKEND_LOG_PREFIX in line:
                cls = line.split(BACKEND_LOG_PREFIX, 1)[1].strip().split()[0]
                return OURS_BACKEND_NAMES.get(cls, cls)
        return None


class VLLMAdapter(EngineAdapter):
    name = "vllm"
    health_path = VLLM_HEALTH_PATH
    metrics_path = VLLM_METRICS_PATH

    def __init__(self, python: str | None = None) -> None:
        self.python = python or server_python()

    def launch_spec(self, arm, model_path, host="127.0.0.1", port=8000, kv_pool_tokens=None):
        cmd = [
            self.python, "-m", VLLM_MODULE,
            VLLM_FLAG_MODEL, model_path,
            VLLM_FLAG_DTYPE, "float16",
            VLLM_FLAG_HOST, host,
            VLLM_FLAG_PORT, str(port),
            VLLM_FLAG_SERVED_NAME, self.model_name,
            VLLM_FLAG_SEED, "0",
            VLLM_FLAG_MAX_MODEL_LEN, str(MAX_MODEL_LEN),
            VLLM_FLAG_PROMPT_TOKENS_DETAILS,
            *arm.extra_args,
        ]  # fmt: skip
        flags: dict[str, Any] = {
            "dtype": "float16",
            "seed": 0,
            "max_model_len": MAX_MODEL_LEN,
            "enforce_eager": VLLM_FLAG_ENFORCE_EAGER in arm.extra_args,
            "cuda_graphs": not (
                VLLM_FLAG_ENFORCE_EAGER in arm.extra_args or VLLM_NO_CUDAGRAPH in arm.extra_args
            ),
            "torch_compile": VLLM_FLAG_ENFORCE_EAGER not in arm.extra_args,
            "async_scheduling": VLLM_FLAG_NO_ASYNC_SCHED not in arm.extra_args,
            "prefix_caching": VLLM_FLAG_NO_PREFIX_CACHING not in arm.extra_args,
            "kv_pool_tokens": None,
        }
        if VLLM_FLAG_MAX_NUM_SEQS in arm.extra_args:
            flags["max_num_seqs"] = int(
                arm.extra_args[arm.extra_args.index(VLLM_FLAG_MAX_NUM_SEQS) + 1]
            )
        if arm.matched and not kv_pool_tokens:
            raise ValueError(
                "vllm-matched needs the KV pool size of ours in tokens: pass "
                "--kv-pool-tokens, run it on a workload with kv_pool_tokens, or run "
                "the `ours` arm first so its allocator capacity can be read back"
            )
        if kv_pool_tokens:
            blocks = kv_pool_tokens // KV_BLOCK_SIZE
            cmd += [VLLM_FLAG_NUM_GPU_BLOCKS, str(blocks), VLLM_FLAG_BLOCK_SIZE, str(KV_BLOCK_SIZE)]
            flags.update(
                kv_pool_tokens=blocks * KV_BLOCK_SIZE,
                num_gpu_blocks=blocks,
                block_size=KV_BLOCK_SIZE,
            )
        return LaunchSpec(cmd=cmd, env={}, flags=flags)

    def query_version(self, base_url, client=None):
        owns = client is None
        c = client or httpx.Client(timeout=10.0)
        try:
            r = c.get(base_url.rstrip("/") + VLLM_VERSION_PATH)
            if r.status_code == 200:
                v = r.json().get("version")
                if v:
                    return str(v)
        except (httpx.HTTPError, ValueError):
            pass
        finally:
            if owns:
                c.close()
        return _local_version("vllm", self.python)

    def server_counters(self, before, after):
        before, after = before or {}, after or {}
        hits = _delta(before, after, VLLM_METRIC_PREFIX_HITS)
        queries = _delta(before, after, VLLM_METRIC_PREFIX_QUERIES)
        return {
            "preemptions": _delta(before, after, VLLM_METRIC_PREEMPTIONS),
            "evictions": None,
            "prefix_hit_rate": _ratio(hits, queries),
            "raw": {
                "before": before,
                "after": after,
                "prefix_hits_delta": hits,
                "prefix_queries_delta": queries,
                "units": {"prefix_hit_rate": "tokens", "preemptions": "requests"},
                "definitions": {
                    "preemptions": f"delta {VLLM_METRIC_PREEMPTIONS}",
                    "evictions": "not exposed by vLLM 0.31.0 (no eviction counter)",
                    "prefix_hit_rate": (
                        f"delta {VLLM_METRIC_PREFIX_HITS} / delta {VLLM_METRIC_PREFIX_QUERIES} "
                        "(TOKEN granularity; preempted requests' re-lookups are not "
                        "exported, so under preemption this understates reuse)"
                    ),
                },
            },
        }

    def resolved_defaults(self, log_text: str) -> dict[str, Any]:
        return grep_resolved(log_text, VLLM_RESOLVED_RES)

    def detect_attention_backend(self, log_text: str) -> str | None:
        found = VLLM_BACKEND_LOG_RE.findall(log_text)
        return found[-1].strip().lower().replace(" ", "_") if found else None


class SGLangAdapter(EngineAdapter):
    name = "sglang"
    health_path = SGLANG_HEALTH_PATH
    metrics_path = SGLANG_METRICS_PATH

    def __init__(self, python: str | None = None) -> None:
        self.python = python or server_python()

    def launch_spec(self, arm, model_path, host="127.0.0.1", port=8000, kv_pool_tokens=None):
        cmd = [
            self.python, "-m", SGLANG_MODULE,
            SGLANG_FLAG_MODEL, model_path,
            SGLANG_FLAG_DTYPE, "float16",
            SGLANG_FLAG_HOST, host,
            SGLANG_FLAG_PORT, str(port),
            SGLANG_FLAG_SERVED_NAME, self.model_name,
            SGLANG_FLAG_SEED, "0",
            SGLANG_FLAG_CONTEXT_LEN, str(MAX_MODEL_LEN),
            SGLANG_FLAG_ENABLE_METRICS,
            SGLANG_FLAG_CACHE_REPORT,
            *arm.extra_args,
        ]  # fmt: skip
        ea = arm.extra_args
        flags: dict[str, Any] = {
            "dtype": "float16",
            "random_seed": 0,
            "context_length": MAX_MODEL_LEN,
            "radix_cache": SGLANG_FLAG_DISABLE_RADIX not in ea,
            "overlap_schedule": SGLANG_FLAG_DISABLE_OVERLAP not in ea,
            "decode_cuda_graph": SGLANG_FLAG_DISABLE_DECODE_GRAPH not in ea,
            "prefill_cuda_graph": SGLANG_FLAG_DISABLE_PREFILL_GRAPH not in ea,
            "schedule_policy": (
                ea[ea.index(SGLANG_FLAG_SCHEDULE_POLICY) + 1]
                if SGLANG_FLAG_SCHEDULE_POLICY in ea
                else "fcfs (default)"
            ),
            "kv_pool_tokens": None,
        }
        if kv_pool_tokens:
            # Same 16-token granularity as ours/vLLM so the three pools are equal.
            tokens = (kv_pool_tokens // KV_BLOCK_SIZE) * KV_BLOCK_SIZE
            cmd += [SGLANG_FLAG_MAX_TOTAL_TOKENS, str(tokens)]
            flags["kv_pool_tokens"] = tokens
            flags["kv_pool_note"] = (
                "--max-total-tokens can only LOWER the profiled pool; the actual size is "
                "in flags.resolved.max_total_num_tokens when the log reports it"
            )
        return LaunchSpec(cmd=cmd, env={}, flags=flags)

    def query_version(self, base_url, client=None):
        owns = client is None
        c = client or httpx.Client(timeout=10.0)
        try:
            for path in SGLANG_VERSION_PATHS:
                try:
                    r = c.get(base_url.rstrip("/") + path)
                    if r.status_code == 200:
                        v = r.json().get("version")
                        if v:
                            return str(v)
                except (httpx.HTTPError, ValueError):
                    continue
        finally:
            if owns:
                c.close()
        return _local_version("sglang", self.python)

    def server_counters(self, before, after):
        before, after = before or {}, after or {}
        base = SGLANG_METRIC_PREFILL_TOKENS
        queries = _delta(before, after, base)
        hits: float | None = None
        if queries is not None:
            parts = [
                _delta(before, after, f'{base}{{mode="{m}"}}') for m in SGLANG_PREFIX_HIT_MODES
            ]
            # A mode never seen has no series yet: that is 0 hits of that kind,
            # but only once the counter family itself is known to exist.
            hits = sum(p for p in parts if p is not None)
        return {
            "preemptions": _delta(before, after, SGLANG_METRIC_PREEMPTIONS),
            "evictions": _delta(before, after, SGLANG_METRIC_EVICTIONS),
            "prefix_hit_rate": _ratio(hits, queries),
            "raw": {
                "before": before,
                "after": after,
                "prefix_hits_delta": hits,
                "prefix_queries_delta": queries,
                "units": {
                    "prefix_hit_rate": "tokens",
                    "evictions": "tokens",
                    "preemptions": "requests (retractions)",
                },
                "definitions": {
                    "preemptions": f"delta {SGLANG_METRIC_PREEMPTIONS}",
                    "evictions": f"delta {SGLANG_METRIC_EVICTIONS} (TOKENS, not blocks)",
                    "prefix_hit_rate": (
                        f"delta {base} with mode in {list(SGLANG_PREFIX_HIT_MODES)} / delta "
                        f"{base} over all modes (TOKEN granularity). sglang:cache_hit_rate is "
                        "NOT used: it covers the last prefill batch only"
                    ),
                },
            },
        }

    def resolved_defaults(self, log_text: str) -> dict[str, Any]:
        return grep_resolved(log_text, SGLANG_RESOLVED_RES)

    def detect_attention_backend(self, log_text: str) -> str | None:
        found = [b for b in SGLANG_BACKEND_LOG_RE.findall(log_text) if b != "None"]
        return found[-1] if found else None


def adapter_for(engine: str, **kw: Any) -> EngineAdapter:
    if engine == "ours":
        return OursAdapter(**kw)
    if engine == "vllm":
        return VLLMAdapter(**kw)
    if engine == "sglang":
        return SGLangAdapter(**kw)
    raise KeyError(f"unknown engine {engine!r}")


def base_url_for(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def ours_kv_pool_tokens_from(raw_counters: dict[str, Any]) -> int | None:
    """Read ours' pool capacity back out of a server_counters.raw snapshot."""
    for side in ("after", "before"):
        alloc = ((raw_counters or {}).get(side) or {}).get("allocator") or {}
        cap = alloc.get("tokens_capacity")
        if isinstance(cap, (int, float)) and not isinstance(cap, bool) and cap > 0:
            return int(math.floor(cap))
    return None
