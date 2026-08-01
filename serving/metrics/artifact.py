"""
Benchmark artifact schema and provenance capture.

WHY THIS IS PHASE 0 WORK AND NOT "LATER"
----------------------------------------
Provenance cannot be retrofitted. Once a benchmark has run and the allocation is
released, there is no way to recover which node it ran on, which QOS it used, or
which engine commit was linked. A result without that context is not a weaker
result — it is an unusable one, because the comparison it participates in cannot
be validated.

Three risks from docs/RISK_REGISTER.md are addressed here, and all three are
SILENT — they produce a plausible number, not an error:

  R12  Cross-allocation A/B. Node contention and clock variation move absolute
       throughput ~25% for identical code: the engine measured ~79 tok/s on one
       node and ~60 on another (BENCHMARKS.md:17). A delta computed across two
       allocations measures node assignment, not the code change. Fixed by
       recording allocation identity and REFUSING to render such comparisons.

  R13  Benchmark preempted under the `embers` QOS. `embers` is free but
       preemptible with an 8h cap; a preempted run truncates its measurement
       window into something indistinguishable from a completed short run.
       Fixed by recording QOS and rejecting `embers` for published numbers.

  R16  Metric name collisions. The engine has a live instance: `peak_mem_mb`
       means host RSS in bench/harness.py:44-47 and CUDA max_memory_allocated in
       bench/baseline_hf.py:108 — two quantities, one name, silently compared
       (its own known gap #1, BENCHMARKS.md:247). Fixed by making every metric
       carry (quantity, unit, source) and refusing to register a name twice with
       different semantics.

WHY RAW SAMPLES, NOT SUMMARY STATISTICS
---------------------------------------
The engine's harness stores per-run p50/p99 and discards the underlying samples
(bench/harness.py:136-137,178). That is fine for a single-stream microbenchmark
and disqualifying for a serving benchmark: percentiles must be pooled across all
requests in the measurement window, and averaging per-run percentiles is not the
same operation. Once the raw samples are gone the correct number cannot be
recovered, so this schema stores samples and computes percentiles at analysis
time.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "MetricSpec",
    "MetricRegistry",
    "Provenance",
    "Artifact",
    "PUBLISHABLE_QOS",
]

# QOS values whose runs may back a published number.
#
# `embers` is deliberately absent. It is free and preemptible; a preemption
# mid-run silently truncates the measurement window (R13). Develop on embers,
# publish from inferno.
PUBLISHABLE_QOS = frozenset({"inferno"})


@dataclass(frozen=True)
class MetricSpec:
    """
    What a metric name means. Registered once, enforced everywhere.

    `source` is the part that prevents the engine's `peak_mem_mb` collision: two
    metrics may both be memory in MB and still be incomparable if one is host RSS
    and the other is CUDA allocator high-water mark.
    """

    name: str
    quantity: str        # what is being measured, e.g. "time to first token"
    unit: str            # "ms", "tokens/s", "MB", "ratio", "count"
    source: str          # where the number comes from, e.g. "client wall clock"
    description: str = ""

    def key(self) -> tuple[str, str, str]:
        return (self.quantity, self.unit, self.source)


class MetricRegistry:
    """
    Enforces that a metric name maps to exactly one (quantity, unit, source).

    Registering the same name with different semantics raises. That is the check
    the engine lacked when `peak_mem_mb` came to mean two different things.
    """

    def __init__(self) -> None:
        self._specs: dict[str, MetricSpec] = {}

    def register(self, spec: MetricSpec) -> MetricSpec:
        existing = self._specs.get(spec.name)
        if existing is not None and existing.key() != spec.key():
            raise ValueError(
                f"Metric name collision on {spec.name!r}:\n"
                f"  already registered as {existing.key()}\n"
                f"  now being registered as {spec.key()}\n"
                "Two quantities under one name is how incomparable numbers get "
                "compared. Pick a distinct name."
            )
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> MetricSpec:
        if name not in self._specs:
            raise KeyError(
                f"Unregistered metric {name!r}. Every published metric must "
                "declare its (quantity, unit, source) before use."
            )
        return self._specs[name]

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def all(self) -> dict[str, MetricSpec]:
        return dict(self._specs)


# The canonical registry. Definitions match docs/BENCHMARK_METHODOLOGY.md §2.
REGISTRY = MetricRegistry()

for _spec in [
    MetricSpec(
        "ttft_ms", "time to first token", "ms", "client wall clock from intended dispatch",
        "Intended dispatch -> first SSE chunk with NON-EMPTY content. Measured from "
        "INTENDED dispatch, not actual send, to avoid coordinated omission (R1). "
        "An empty-content chunk is not a first token.",
    ),
    MetricSpec(
        "itl_ms", "inter-token latency", "ms", "client wall clock",
        "Gap between consecutive non-empty content chunks. Per-request series; "
        "pooled across requests before computing percentiles.",
    ),
    MetricSpec(
        "e2e_ms", "end-to-end latency", "ms", "client wall clock from intended dispatch",
        "Intended dispatch -> final chunk.",
    ),
    MetricSpec(
        "queue_wait_ms", "queue wait", "ms", "server instrumentation",
        "Intended dispatch -> admission into a running batch.",
    ),
    MetricSpec(
        "output_tok_s", "output token throughput", "tokens/s", "server aggregate",
        "OUTPUT tokens only. Prefill tokens are reported separately; a combined "
        "total can be inflated arbitrarily by lengthening prompts.",
    ),
    MetricSpec(
        "goodput_rps", "goodput under SLO", "requests/s", "client aggregate over steady-state window",
        "Requests/sec that MET the SLO. The headline metric — raw throughput is "
        "gameable in both directions.",
    ),
    MetricSpec(
        "offered_load_rps", "offered load", "requests/s", "load generator configuration",
        "A PARAMETER, not an outcome. Reporting this as achieved throughput is a "
        "classic error.",
    ),
    MetricSpec(
        "cache_hit_rate", "prefix cache hit rate", "ratio", "server instrumentation, block granularity",
        "blocks_reused / blocks_required_at_prefill. Meaningless without the "
        "workload's prefix-sharing structure attached.",
    ),
    MetricSpec(
        "preemption_rate", "preemption rate", "ratio", "server instrumentation",
        "Preemption events / decode iterations. Recompute and swap broken out.",
    ),
    MetricSpec(
        "gpu_mem_mb", "GPU memory high-water mark", "MB", "torch.cuda.max_memory_allocated",
        "NAMED FOR ITS SOURCE ON PURPOSE. The engine's `peak_mem_mb` meant host "
        "RSS in one harness and this in another (BENCHMARKS.md:247).",
    ),
    MetricSpec(
        "host_rss_mb", "host resident set size", "MB", "resource.getrusage(RUSAGE_SELF)",
        "Distinct metric, distinct name. Not comparable to gpu_mem_mb.",
    ),
    MetricSpec(
        "dispatch_drift_ms", "load generator dispatch drift", "ms", "load generator",
        "actual_send - intended_send. A HARNESS HEALTH metric: drift above "
        "threshold invalidates the run (R1). Published alongside every goodput curve.",
    ),
]:
    REGISTRY.register(_spec)


def _run(cmd: list[str], cwd: str | None = None) -> str | None:
    """Best-effort shell capture. Missing provenance is recorded as None, never faked."""
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class Provenance:
    """
    Everything needed to decide whether a number is valid and whether two numbers
    may be compared.
    """

    timestamp_utc: str
    hostname: str
    platform: str

    # Slurm — the allocation identity. R12 depends on these.
    slurm_job_id: str | None = None
    slurm_node: str | None = None
    slurm_partition: str | None = None
    slurm_qos: str | None = None
    gpu_name: str | None = None
    gpu_count: int | None = None

    # Code identity
    repo_sha: str | None = None
    repo_dirty: bool | None = None
    engine_tag: str | None = None
    engine_sha: str | None = None

    # Environment
    python_version: str = ""
    torch_version: str | None = None
    cuda_version: str | None = None
    flashinfer_version: str | None = None

    seed: int | None = None

    @classmethod
    def capture(cls, repo_root: str | Path = ".", seed: int | None = None) -> "Provenance":
        """Collect provenance from the environment. Absent fields stay None."""
        repo_root = str(repo_root)

        torch_v = cuda_v = gpu_name = fi_v = None
        gpu_count = None
        try:
            import torch

            torch_v = torch.__version__
            cuda_v = torch.version.cuda
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
        except Exception:
            pass

        try:
            import flashinfer

            fi_v = getattr(flashinfer, "__version__", "unknown")
        except Exception:
            pass

        sha = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
        dirty = None
        status = _run(["git", "status", "--porcelain"], cwd=repo_root)
        if status is not None or sha is not None:
            dirty = bool(status)

        engine_dir = os.path.join(repo_root, "vendor", "llm_inference_engine")
        engine_sha = engine_tag = None
        if os.path.isdir(engine_dir):
            engine_sha = _run(["git", "rev-parse", "HEAD"], cwd=engine_dir)
            engine_tag = _run(["git", "describe", "--tags", "--exact-match"], cwd=engine_dir)

        return cls(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            hostname=socket.gethostname(),
            platform=platform.platform(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            slurm_node=os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
            slurm_partition=os.environ.get("SLURM_JOB_PARTITION"),
            slurm_qos=os.environ.get("SLURM_JOB_QOS"),
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            repo_sha=sha,
            repo_dirty=dirty,
            engine_tag=engine_tag,
            engine_sha=engine_sha,
            python_version=platform.python_version(),
            torch_version=torch_v,
            cuda_version=cuda_v,
            flashinfer_version=fi_v,
            seed=seed,
        )

    @property
    def allocation_id(self) -> str | None:
        """
        Identity of the Slurm allocation. Two artifacts sharing this ran
        back-to-back on the same node and MAY be compared (R12).
        """
        if self.slurm_job_id is None:
            return None
        return f"{self.slurm_job_id}@{self.slurm_node or 'unknown'}"

    def is_publishable(self) -> tuple[bool, list[str]]:
        """
        May a number from this run be published? Returns (ok, reasons_if_not).

        Deliberately strict. A rejected artifact is still written to disk — it is
        real data — but must not back a claim.
        """
        reasons = []
        if self.slurm_qos is not None and self.slurm_qos not in PUBLISHABLE_QOS:
            reasons.append(
                f"QOS is {self.slurm_qos!r}; only {sorted(PUBLISHABLE_QOS)} may back a "
                "published number. `embers` is preemptible and a preempted run "
                "truncates its measurement window (R13)."
            )
        if self.repo_dirty:
            reasons.append("Working tree is dirty — the measured code is not any commit.")
        if self.repo_sha is None:
            reasons.append("No repo SHA; the run is not reproducible.")
        if self.engine_tag is None and self.engine_sha is None:
            reasons.append("No engine version recorded; the dependency is unpinned.")
        if self.seed is None:
            reasons.append("No seed recorded; the workload is not reproducible.")
        return (not reasons, reasons)


@dataclass
class Artifact:
    """
    One benchmark run. Written to results/ and tracked in git.

    `samples` holds RAW per-request and per-token observations keyed by metric
    name. `realized_workload` holds what the workload generator actually produced
    — not what it was asked for. Divergence between the two is itself the alarm
    for a degenerate workload (R14).
    """

    name: str
    provenance: Provenance
    config: dict[str, Any] = field(default_factory=dict)
    samples: dict[str, list[float]] = field(default_factory=dict)
    realized_workload: dict[str, Any] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    window: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    schema_version: int = 1

    def add_samples(self, metric: str, values: list[float]) -> None:
        REGISTRY.get(metric)          # unregistered metric -> KeyError, loudly
        self.samples.setdefault(metric, []).extend(float(v) for v in values)

    def set_scalar(self, metric: str, value: float) -> None:
        REGISTRY.get(metric)
        self.scalars[metric] = float(value)

    def validate(self) -> list[str]:
        """Structural problems that make this artifact unusable. Empty list == fine."""
        problems = []
        if not self.name:
            problems.append("Artifact has no name.")
        for metric in list(self.samples) + list(self.scalars):
            if metric not in REGISTRY:
                problems.append(f"Metric {metric!r} is not registered.")
        if self.samples and not self.window:
            problems.append(
                "Samples present but no measurement window recorded. Without explicit "
                "boundaries a run may silently include ramp-up or drain, and the drain "
                "tail flatters p99 (R11)."
            )
        return problems

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        publishable, reasons = self.provenance.is_publishable()
        d["publishable"] = publishable
        d["publishable_blockers"] = reasons
        d["allocation_id"] = self.provenance.allocation_id
        d["metric_specs"] = {
            name: asdict(REGISTRY.get(name))
            for name in sorted(set(self.samples) | set(self.scalars))
        }
        return d

    def write(self, results_dir: str | Path = "results") -> Path:
        problems = self.validate()
        if problems:
            raise ValueError("Artifact is not well-formed:\n  " + "\n  ".join(problems))

        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.provenance.timestamp_utc.replace(":", "").replace("-", "")[:15]
        host = self.provenance.hostname.split(".")[0]
        path = results_dir / f"{stamp}_{host}_{self.name}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path


def assert_comparable(a: Artifact, b: Artifact) -> None:
    """
    Refuse to compare two artifacts from different allocations.

    R12 in executable form. The engine's own numbers moved ~79 -> ~60 tok/s
    across nodes for identical code, so a cross-allocation delta measures node
    assignment. This RAISES rather than warning: a warning in a notebook is a
    warning nobody reads.
    """
    ida, idb = a.provenance.allocation_id, b.provenance.allocation_id
    if ida is None or idb is None:
        raise ValueError(
            f"Cannot verify comparability: allocation id missing "
            f"({a.name}={ida}, {b.name}={idb}). Runs outside Slurm are not "
            "comparable A/B measurements."
        )
    if ida != idb:
        raise ValueError(
            f"Refusing to compare across allocations: {a.name} ran in {ida}, "
            f"{b.name} ran in {idb}.\n"
            "Node contention and clock variation move absolute throughput ~25% "
            "for identical code (BENCHMARKS.md:17). Re-run both back-to-back in "
            "one allocation, or report them as two separate absolute measurements."
        )
