"""
Prometheus text-format exporter. Written by hand, on purpose, and name-checked
against the artifact REGISTRY.

WHY NO `prometheus_client` DEPENDENCY
------------------------------------
The exposition format is a few hundred bytes of specification: one `# HELP` line
and one `# TYPE` line per metric family, then `name{labels} value` per series,
with histograms expanded into cumulative `_bucket`/`_sum`/`_count` series. Taking
a dependency to emit that would add a package to a benchmark process whose
provenance is published (`serving/metrics/artifact.py`), and would hand control
of metric NAMING to a library's conventions at exactly the point where naming is
the thing this file has to get right (R16). Writing it directly is ~200 lines and
keeps the name policy enforceable in-process.

THE NAME POLICY (docs/RISK_REGISTER.md R16) — READ BEFORE ADDING A METRIC
--------------------------------------------------------------------------
The engine's known gap #1: `peak_mem_mb` means host RSS in `bench/harness.py:44`
and `torch.cuda.max_memory_allocated()` in `bench/baseline_hf.py:108`. Two
quantities, one name, silently compared. `serving/metrics/artifact.py` fixes that
by making every registered name map to exactly one (quantity, unit, source).

A Prometheus endpoint is the easiest place in a serving system to reintroduce
that bug, because a scrape name looks like a free-form label and gets copied into
dashboards, alerts, and screenshots where the definition does not travel with it.
So this module enforces the policy structurally, and every family falls into
exactly one of two classes:

  REGISTRY-BACKED. The exported name is `llm_` + a REGISTRY name, and the family
    carries `registry_name`. `assert_registry_alignment` checks the spec exists
    and stitches (quantity, unit, source) into the HELP text, so the definition
    travels with the series into whatever dashboard it lands in. Currently:
    `llm_output_tok_s`, `llm_host_rss_mb`, `llm_gpu_mem_mb`.

  SERVER-LOCAL. The name, with the `llm_` prefix and any `_total` suffix
    stripped, MUST NOT collide with a REGISTRY name. `assert_registry_alignment`
    raises if it does. This is what makes an accidental collision impossible
    rather than merely discouraged.

WHAT IS DELIBERATELY NOT EXPORTED, AND WHY (see `NOT_EXPORTED`)
---------------------------------------------------------------
`serving/server/app.py`'s module docstring already establishes the rule and this
file follows it. The registry defines `ttft_ms`, `itl_ms`, `e2e_ms` and
`queue_wait_ms` from the CLIENT's intended dispatch time. **The server cannot
observe intended dispatch** — only the load generator, which computed the arrival
schedule in advance, knows it (R1). Arrival-at-the-HTTP-handler is a different
instant, and the gap between them is precisely the coordinated-omission interval
that makes a saturated system look fast.

So the server's latency histograms are exported as `llm_ttft_from_arrival_ms`,
`llm_itl_server_ms`, `llm_e2e_from_arrival_ms` — names that cannot be mistaken
for the registry's, with HELP text that says what they miss. They are a
diagnostic view of the server's own behaviour. **Published latency numbers come
from `bench/loadgen.py`, not from a scrape of this endpoint.** Exporting
server-side TTFT as `ttft_ms` would be the `peak_mem_mb` mistake with a nicer
dashboard.

Two registry ratios are exported as their COMPONENTS rather than as the ratio:
`cache_hit_rate` as block-granularity hit/miss counters, and `preemption_rate` as
a preemption counter over a step counter. That is the Prometheus-native form (a
counter ratio is computed at query time over a window, where a scraped instant
ratio would be a run-lifetime average that no alert can use), and the components
are defined to divide out to exactly the registry's definition.

CARDINALITY
-----------
No label anywhere carries a request id, a prompt, a model-instance id, or
anything else unbounded. The only labels used are `policy` on preemptions, whose
domain is the closed set of preemption policies. An unbounded label is not a
cosmetic problem: it multiplies series count without bound and takes the scrape
target down, which would make the observability system the outage.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from serving.metrics.artifact import REGISTRY

__all__ = [
    "Histogram",
    "Sample",
    "Family",
    "LATENCY_BUCKETS_MS",
    "STEP_BUCKETS_MS",
    "NOT_EXPORTED",
    "PREFIX",
    "assert_registry_alignment",
    "render",
    "render_server_metrics",
    "counter",
    "gauge",
    "histogram_family",
]

PREFIX = "llm_"

#: Prometheus name grammar. Enforced rather than assumed: a name with a hyphen or
#: a dot parses as something else entirely on the scrape side and the series
#: silently disappears instead of erroring.
_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

#: Latency buckets, in MILLISECONDS. Not the Prometheus-idiomatic seconds,
#: because every latency definition in docs/BENCHMARK_METHODOLOGY.md §2 and every
#: registered latency metric is in ms, and a unit change between the definition
#: and the export is the same class of error as a name change. The unit is in the
#: metric name (`_ms`) so a reader is never guessing.
#: Range covers unloaded batch-1 ITL (~12 ms on A100, BENCHMARKS.md:37) through
#: a saturated TTFT tail; the SLO thresholds this project declares live inside it.
LATENCY_BUCKETS_MS: tuple[float, ...] = (
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
)

#: Step duration is a much tighter distribution than request latency: it is one
#: bounded forward pass, capped by SchedulerConfig.max_prefill_tokens.
STEP_BUCKETS_MS: tuple[float, ...] = (
    0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
)

#: Registry names this endpoint refuses to emit, with the reason. Rendered as
#: comments into the exposition so the omission is visible at the scrape, not
#: only in this docstring — an absent metric otherwise reads as an oversight and
#: someone helpfully adds it back under the wrong definition.
NOT_EXPORTED: dict[str, str] = {
    "ttft_ms": (
        "defined from INTENDED DISPATCH, which only the load generator knows (R1); "
        "server-observable arrival-based TTFT is exported as llm_ttft_from_arrival_ms"
    ),
    "itl_ms": (
        "defined from the CLIENT wall clock; the server sees its own write times, "
        "exported as llm_itl_server_ms"
    ),
    "e2e_ms": (
        "defined from INTENDED DISPATCH; exported here as llm_e2e_from_arrival_ms"
    ),
    "queue_wait_ms": (
        "defined as intended dispatch -> admission; the server cannot observe "
        "intended dispatch, so nothing is emitted under this name (R16)"
    ),
    "goodput_rps": (
        "an SLO-conditioned CLIENT measurement over a declared steady-state window; "
        "not computable from a scrape"
    ),
    "offered_load_rps": "a load-generator PARAMETER, not a server observation",
    "cache_hit_rate": (
        "exported as its components, llm_cache_block_hits_total / "
        "(hits + llm_cache_block_misses_total), which divide out to the registered "
        "block-granularity definition"
    ),
    "preemption_rate": (
        "exported as its components, llm_preemptions_total / llm_scheduler_steps_total"
    ),
    "dispatch_drift_ms": "a load-generator harness-health metric; the server cannot see it",
}


# ---------------------------------------------------------------------------
# Exposition primitives
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """
    Format a value the way the exposition format requires.

    `+Inf` and `NaN` are spelled exactly like that — Python's `inf`/`nan` are not
    valid there and a parser rejects the whole scrape, not just the line.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    v = float(value)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def _escape_help(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _escape_label_value(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(labels: Sequence[tuple[str, str]]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape_label_value(v)}"' for k, v in labels)
    return "{" + inner + "}"


@dataclass(frozen=True)
class Sample:
    """One series line. `suffix` carries histogram's `_bucket` / `_sum` / `_count`."""

    value: float
    suffix: str = ""
    labels: tuple[tuple[str, str], ...] = ()

    def render(self, family_name: str) -> str:
        return f"{family_name}{self.suffix}{_render_labels(self.labels)} {_fmt(self.value)}"


@dataclass
class Family:
    """
    A metric family: one name, one type, one HELP, N series.

    `registry_name` is the R16 hook. When set, this family claims to be exactly
    the registered metric of that name and `assert_registry_alignment` verifies
    it; when unset, the same function verifies the name does NOT shadow a
    registered one.
    """

    name: str
    type: str
    help: str
    samples: list[Sample] = field(default_factory=list)
    registry_name: str | None = None

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"invalid Prometheus metric name: {self.name!r}")
        if self.type not in ("counter", "gauge", "histogram", "summary", "untyped"):
            raise ValueError(f"invalid metric type: {self.type!r}")
        if not self.help:
            raise ValueError(f"metric {self.name!r} has no HELP text")
        for s in self.samples:
            for k, _ in s.labels:
                if not _LABEL_RE.match(k):
                    raise ValueError(f"invalid label name {k!r} on {self.name!r}")

    def render(self) -> str:
        lines = [
            f"# HELP {self.name} {_escape_help(self.help)}",
            f"# TYPE {self.name} {self.type}",
        ]
        lines.extend(s.render(self.name) for s in self.samples)
        return "\n".join(lines)


class Histogram:
    """
    Fixed-bucket histogram. Bounded memory, no sample retention.

    Deliberately NOT a reservoir or a sample list: this object is updated on the
    request path (once per token for ITL), so it must be O(buckets) with no
    allocation. Raw samples for PUBLISHED percentiles are the load generator's
    job — `serving/metrics/artifact.py`'s docstring is explicit that percentiles
    are computed at analysis time over pooled raw samples (R15), and a histogram
    cannot do that. This one is for dashboards and alerts, where a bucketed
    quantile over a window is the right object and the exact tail value is not.
    """

    def __init__(self, buckets: Sequence[float] = LATENCY_BUCKETS_MS):
        self.bounds: tuple[float, ...] = tuple(sorted(float(b) for b in buckets))
        if len(set(self.bounds)) != len(self.bounds):
            raise ValueError("histogram bucket bounds must be unique")
        self.counts: list[int] = [0] * len(self.bounds)
        self.inf_count: int = 0
        self.sum: float = 0.0
        self.count: int = 0

    def observe(self, value: float) -> None:
        v = float(value)
        self.count += 1
        self.sum += v
        for i, b in enumerate(self.bounds):
            if v <= b:
                self.counts[i] += 1
                return
        self.inf_count += 1

    def samples(self, labels: tuple[tuple[str, str], ...] = ()) -> list[Sample]:
        """Cumulative buckets, then `_sum` and `_count`. `le` sorts ascending, `+Inf` last."""
        out: list[Sample] = []
        cumulative = 0
        for bound, c in zip(self.bounds, self.counts, strict=True):
            cumulative += c
            out.append(
                Sample(cumulative, "_bucket", labels + (("le", _fmt(bound)),))
            )
        out.append(Sample(self.count, "_bucket", labels + (("le", "+Inf"),)))
        out.append(Sample(self.sum, "_sum", labels))
        out.append(Sample(self.count, "_count", labels))
        return out


# ---------------------------------------------------------------------------
# Family constructors
# ---------------------------------------------------------------------------


def _with_spec(help_text: str, registry_name: str | None) -> str:
    """
    Stitch the registered (quantity, unit, source) into HELP.

    The definition has to travel with the series. A number that reaches a
    dashboard without its source is how `peak_mem_mb` became two metrics.
    """
    if registry_name is None:
        return help_text
    spec = REGISTRY.get(registry_name)
    return (
        f"{help_text} [REGISTRY {registry_name}: quantity={spec.quantity}; "
        f"unit={spec.unit}; source={spec.source}]"
    )


def counter(
    name: str,
    help_text: str,
    value: float | None = None,
    *,
    samples: Iterable[Sample] | None = None,
    registry_name: str | None = None,
) -> Family:
    """A monotonically non-decreasing count. `_total` suffix by convention."""
    if samples is None:
        samples = [Sample(float(value or 0.0))]
    return Family(name, "counter", _with_spec(help_text, registry_name),
                  list(samples), registry_name)


def gauge(
    name: str,
    help_text: str,
    value: float | None = None,
    *,
    samples: Iterable[Sample] | None = None,
    registry_name: str | None = None,
) -> Family:
    """An instantaneous value that may go up or down."""
    if samples is None:
        samples = [Sample(float(value or 0.0))]
    return Family(name, "gauge", _with_spec(help_text, registry_name),
                  list(samples), registry_name)


def histogram_family(
    name: str, help_text: str, hist: Histogram, *, registry_name: str | None = None
) -> Family:
    return Family(name, "histogram", _with_spec(help_text, registry_name),
                  hist.samples(), registry_name)


# ---------------------------------------------------------------------------
# The R16 guard
# ---------------------------------------------------------------------------


def _registry_candidates(name: str) -> list[str]:
    """Every REGISTRY name this exported name could be confused with."""
    base = name[len(PREFIX):] if name.startswith(PREFIX) else name
    cands = {name, base}
    for c in list(cands):
        if c.endswith("_total"):
            cands.add(c[: -len("_total")])
    return sorted(cands)


def assert_registry_alignment(families: Sequence[Family]) -> None:
    """
    Make a name collision with `serving/metrics/artifact.py`'s REGISTRY impossible.

    Two rules, and between them every exported name is accounted for:

      * A family that CLAIMS a registry name (`registry_name` set) must be named
        `llm_<registry_name>` and that name must be registered. It is then, by
        construction, the registered quantity with the registered unit from the
        registered source — and the HELP text says so.

      * A family that does not claim one must not resemble one. Stripping the
        `llm_` prefix and any `_total` suffix must not produce a registered name.
        This is the rule that stops a well-meaning `llm_ttft_ms` from ever being
        added: the server's TTFT is not the registry's TTFT, and the check fires
        at import/scrape time rather than in a comparison six weeks later.

    Raises `ValueError`. Not a warning — a warning in a scrape handler is a
    warning nobody reads.
    """
    for fam in families:
        if fam.registry_name is not None:
            expected = PREFIX + fam.registry_name
            if fam.name != expected:
                raise ValueError(
                    f"{fam.name!r} claims REGISTRY metric {fam.registry_name!r} but "
                    f"should then be named {expected!r}. A registry-backed export "
                    "must be mechanically recognisable as one."
                )
            REGISTRY.get(fam.registry_name)  # KeyError if unregistered
            continue
        for cand in _registry_candidates(fam.name):
            if cand in REGISTRY:
                spec = REGISTRY.get(cand)
                raise ValueError(
                    f"Exported metric {fam.name!r} collides with REGISTRY metric "
                    f"{cand!r} ({spec.quantity}, {spec.unit}, from {spec.source}) "
                    "without claiming it.\n"
                    "Either export exactly that quantity and set registry_name, or "
                    "pick a name that cannot be mistaken for it. Two quantities "
                    "under one name is the engine's peak_mem_mb bug (R16); see "
                    "NOT_EXPORTED in this module for the ones that must stay "
                    "server-local."
                )


def render(families: Sequence[Family], *, comments: Sequence[str] = ()) -> str:
    """
    Render a full exposition document. Enforces uniqueness and the name policy.

    Duplicate family names are an error rather than a merge: two `# TYPE` lines
    for one name makes the scrape ambiguous, and every Prometheus parser handles
    it differently.
    """
    seen: set[str] = set()
    for fam in families:
        if fam.name in seen:
            raise ValueError(
                f"duplicate metric family {fam.name!r}: a name must appear once, "
                "with one HELP and one TYPE"
            )
        seen.add(fam.name)
    assert_registry_alignment(families)

    out = [f"# {_escape_help(c)}" for c in comments]
    out.extend(fam.render() for fam in families)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# The server's family set
# ---------------------------------------------------------------------------


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Duck-typed read; the first name that exists wins."""
    for n in names:
        if isinstance(obj, Mapping):
            if n in obj:
                return obj[n]
        elif hasattr(obj, n):
            return getattr(obj, n)
    return default


def _cache_counters(scheduler: Any) -> tuple[float, float] | None:
    """
    Block-granularity prefix-cache hits and misses, IF a radix cache is present.

    Imported/read defensively because `serving/cache/` is Phase 4 and is being
    built in parallel: this endpoint must not fail to render because a module it
    optionally reports on does not exist yet. Absent cache -> absent metrics, not
    zeroed metrics. Zeroes would be a lie a dashboard cannot distinguish from a
    cache that is present and never hitting.

    Definitions match REGISTRY `cache_hit_rate`: blocks reused / blocks required
    at prefill, so hits / (hits + misses) is exactly that ratio.
    """
    cache = _get(scheduler, "prefix_cache", "cache", "radix_cache")
    stats: Any = None
    if cache is not None:
        stats = getattr(cache, "stats", None)
        if callable(stats):
            stats = stats()
        if stats is None:
            stats = cache.snapshot() if hasattr(cache, "snapshot") else cache
    else:
        # `Scheduler.snapshot()` prefixes the cache's own keys with `cache_`
        # rather than merging them flat, for the same R16 reason this module
        # exists. Read them back under that prefix.
        snap = scheduler.snapshot() if hasattr(scheduler, "snapshot") else {}
        if any(str(k).startswith("cache_") for k in snap):
            stats = {str(k)[len("cache_"):]: v for k, v in snap.items()
                     if str(k).startswith("cache_")}
    if stats is None:
        return None

    # PREFERRED FORM. `serving/cache/radix.py` counts blocks_reused and
    # blocks_required, which is literally the registered cache_hit_rate
    # numerator and denominator (methodology §7). Misses are DERIVED as
    # required - reused rather than counted separately, so hits + misses is
    # exactly the denominator by construction and the two counters cannot drift
    # into disagreeing about what a "miss" is.
    reused = _get(stats, "blocks_reused")
    required = _get(stats, "blocks_required")
    if reused is not None and required is not None:
        return float(reused), float(max(0.0, float(required) - float(reused)))

    hits = _get(stats, "block_hits", "cache_hits")
    misses = _get(stats, "block_misses", "cache_misses")
    if hits is None or misses is None:
        return None
    return float(hits), float(misses)


def _preemption_samples(scheduler: Any) -> list[Sample] | None:
    """
    Preemption counts, broken out by policy if the scheduler breaks them out.

    Phase 3 work; absent today. `policy` is the only label in this exporter and
    its domain is closed (`recompute`, `swap`), so cardinality is bounded by
    construction.
    """
    names = ("preemptions_by_policy", "preemptions", "preemption_counts",
             "preemptions_total")
    raw = _get(scheduler, *names)
    if raw is None:
        snap = scheduler.snapshot() if hasattr(scheduler, "snapshot") else {}
        raw = _get(snap, *names)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return [
            Sample(float(v), labels=(("policy", str(k)),))
            for k, v in sorted(raw.items())
        ]
    return [Sample(float(raw))]


def server_families(
    metrics: Any,
    scheduler: Any,
    loop: Any = None,
    *,
    guard: Any = None,
    process_memory: Mapping[str, float] | None = None,
) -> list[Family]:
    """
    Build the replica's family set from the objects the server already keeps.

    Everything here is read from `ServerMetrics`, `Scheduler.snapshot()`,
    `BlockAllocator`, `SchedulerLoop.stats()` and `CudaGuard` — no new
    instrumentation, no second source of truth. A metric with two sources
    eventually has two values.
    """
    snap = scheduler.snapshot() if hasattr(scheduler, "snapshot") else {}
    alloc = getattr(scheduler, "allocator", None)
    fams: list[Family] = []

    # -- counters -----------------------------------------------------------
    fams += [
        counter("llm_requests_received_total",
                "Requests accepted at the HTTP layer and tokenized, before admission.",
                metrics.requests_received),
        counter("llm_requests_completed_total",
                "Requests that reached a terminal generation state and were answered "
                "in full (finish_reason stop or length).",
                metrics.requests_completed),
        counter("llm_requests_failed_total",
                "Requests terminated by a server-side error, including every in-flight "
                "request failed by a fatal CUDA error.",
                metrics.requests_failed),
        counter("llm_requests_cancelled_total",
                "Requests cancelled, almost always by client disconnect. Their KV "
                "blocks are freed at the next step boundary.",
                metrics.requests_cancelled),
        counter("llm_requests_shed_total",
                "Requests rejected with HTTP 429 because the waiting queue was full. "
                "Load shedding is an answer, not a failure: unbounded queueing makes "
                "every request miss its SLO instead of some requests succeeding.",
                metrics.requests_shed_429),
        counter("llm_tokens_generated_total",
                "OUTPUT tokens emitted to clients. Output only — prefill tokens are "
                "counted separately, because a combined total can be inflated "
                "arbitrarily by lengthening prompts (methodology §2).",
                metrics.output_tokens_total),
        counter("llm_prefill_tokens_total",
                "Prompt tokens admitted for prefill. Reported separately from output "
                "tokens, never summed with them.",
                metrics.prompt_tokens_total),
        counter("llm_empty_detokenizations_total",
                "Generated tokens that detokenized to the empty string (special tokens "
                "under skip_special_tokens=True). Counted, never emitted as a visible "
                "chunk — an empty content chunk is not a first token (methodology §2).",
                metrics.empty_detokenizations_total),
    ]

    steps = _get(loop, "steps", default=None) if loop is not None else None
    if steps is None:
        steps = _get(snap, "step", default=0)
    fams.append(
        counter("llm_scheduler_steps_total",
                "Scheduler steps executed. The denominator of the registered "
                "preemption_rate (preemption events / decode iterations).",
                steps)
    )

    preempt = _preemption_samples(scheduler)
    if preempt is not None:
        fams.append(
            counter("llm_preemptions_total",
                    "Preemption events, by victim-eviction policy. Divide by "
                    "llm_scheduler_steps_total for the registered preemption_rate.",
                    samples=preempt)
        )

    cache = _cache_counters(scheduler)
    if cache is not None:
        hits, misses = cache
        fams += [
            counter("llm_cache_block_hits_total",
                    "KV blocks reused from the prefix cache at prefill. BLOCK "
                    "granularity, matching the registered cache_hit_rate definition; "
                    "hits / (hits + misses) is exactly that ratio.",
                    hits),
            counter("llm_cache_block_misses_total",
                    "KV blocks required at prefill that the prefix cache did not have. "
                    "Block granularity.",
                    misses),
        ]

    cuda_errors = 0
    if guard is not None:
        cuda_errors = getattr(guard, "errors_total", 0)
    fams.append(
        counter("llm_cuda_errors_total",
                "CUDA errors observed at a declared check point. ANY non-zero value "
                "means this replica is poisoned and its output since the previous "
                "successful check is not trustworthy: an unchecked launch failure "
                "produces plausible text at full throughput (R10). Fatal — the "
                "replica stops stepping and reports unhealthy.",
                cuda_errors)
    )

    # -- gauges -------------------------------------------------------------
    healthy = 1.0
    if loop is not None:
        alive = getattr(loop, "running", True) and getattr(loop, "healthy", True)
        healthy = 1.0 if alive else 0.0
    fams += [
        gauge("llm_requests_running",
              "Requests currently in the running batch.",
              _get(snap, "running", default=0)),
        gauge("llm_requests_waiting",
              "Requests admitted to the waiting queue but not yet in a batch.",
              _get(snap, "waiting", default=0)),
        gauge("llm_scheduler_healthy",
              "1 if the scheduler task is running and has not hit a fatal error; 0 "
              "otherwise. 0 means quarantine this replica — it accepts TCP "
              "connections and answers nothing.",
              healthy),
        gauge("llm_uptime_seconds", "Process uptime.", metrics.uptime_s),
    ]

    if alloc is not None:
        fams += [
            gauge("llm_blocks_free", "Free KV blocks in the pool.", alloc.num_free),
            gauge("llm_blocks_used", "Allocated KV blocks.", alloc.num_used),
            gauge("llm_block_utilization",
                  "Allocated blocks / total blocks, 0..1. Sustained values near 1 are "
                  "what precede preemption and admission failure.",
                  alloc.utilization),
            gauge("llm_blocks_total", "Size of the KV block pool.", alloc.num_blocks),
            gauge("llm_blocks_watermark",
                  "Reserved free blocks below which admission stops, so the running "
                  "set stays steppable.",
                  alloc.watermark_blocks),
        ]
    else:
        fams += [
            gauge("llm_blocks_free", "Free KV blocks in the pool.",
                  _get(snap, "blocks_free", default=0)),
            gauge("llm_blocks_used", "Allocated KV blocks.",
                  _get(snap, "blocks_used", default=0)),
            gauge("llm_block_utilization", "Allocated blocks / total blocks, 0..1.",
                  _get(snap, "block_utilization", default=0.0)),
        ]

    # -- registry-backed gauges (R16: name claimed, spec attached) ----------
    fams.append(
        gauge("llm_output_tok_s",
              "Output tokens per second, lifetime average over process uptime. "
              "A published throughput number comes from the load generator over a "
              "declared steady-state window, not from this lifetime average.",
              metrics.output_tokens_total / metrics.uptime_s,
              registry_name="output_tok_s")
    )
    for name, value in dict(process_memory or {}).items():
        if name in ("host_rss_mb", "gpu_mem_mb"):
            fams.append(
                gauge(PREFIX + name,
                      "Two memory numbers, two names, on purpose: the engine's "
                      "peak_mem_mb meant host RSS in one harness and CUDA "
                      "max_memory_allocated in another (BENCHMARKS.md:247).",
                      value, registry_name=name)
            )

    # -- histograms ---------------------------------------------------------
    hists = getattr(metrics, "histograms", None) or {}
    if "ttft_from_arrival_ms" in hists:
        fams.append(histogram_family(
            "llm_ttft_from_arrival_ms",
            "Server-observed time from request arrival at the HTTP handler to the "
            "first NON-EMPTY content chunk. NOT the registered ttft_ms, which starts "
            "at the client's intended dispatch and therefore includes queueing the "
            "server cannot see (R1/R16). Diagnostic; published TTFT comes from "
            "bench/loadgen.py.",
            hists["ttft_from_arrival_ms"]))
    if "itl_server_ms" in hists:
        fams.append(histogram_family(
            "llm_itl_server_ms",
            "Server-side gap between consecutive non-empty content chunks being "
            "written. NOT the registered itl_ms, which is measured on the client "
            "wall clock and includes the network.",
            hists["itl_server_ms"]))
    if "e2e_from_arrival_ms" in hists:
        fams.append(histogram_family(
            "llm_e2e_from_arrival_ms",
            "Server-observed arrival to final chunk. NOT the registered e2e_ms, "
            "which starts at intended dispatch.",
            hists["e2e_from_arrival_ms"]))
    if "step_duration_host_ms" in hists:
        fams.append(histogram_family(
            "llm_step_duration_host_ms",
            "HOST-CLOCK duration of Scheduler.step(), diagnostic only (R2). Valid as "
            "a measure of execution only while the step contains a CUDA "
            "synchronisation — today the argmax device->host copy, and the per-step "
            "CUDA error check when enabled. Without one it measures kernel-launch "
            "queueing, gets faster, and stays plausible.",
            hists["step_duration_host_ms"]))

    return fams


def render_server_metrics(
    metrics: Any,
    scheduler: Any,
    loop: Any = None,
    *,
    guard: Any = None,
    process_memory: Mapping[str, float] | None = None,
) -> str:
    """The `/metrics/prometheus` body."""
    fams = server_families(
        metrics, scheduler, loop, guard=guard, process_memory=process_memory
    )
    comments = [
        "llm serving layer — replica metrics. Definitions: "
        "docs/BENCHMARK_METHODOLOGY.md §2.",
        "Names matching serving/metrics/artifact.py's REGISTRY carry that exact "
        "(quantity, unit, source) in their HELP; everything else is server-local "
        "and named so it cannot be mistaken for one (R16).",
    ]
    comments += [
        f"NOT EXPORTED: {name} — {why}" for name, why in sorted(NOT_EXPORTED.items())
    ]
    return render(fams, comments=comments)
