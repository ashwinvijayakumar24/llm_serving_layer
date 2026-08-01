"""
Shared discipline for the phase benchmark drivers (P3, P4, P5).

WHY THIS FILE EXISTS
--------------------
`bench/run_p2.py` is the template, and the three things it does that matter are
not the measurement at all:

  1. it captures provenance and SAYS whether the run is publishable,
  2. it refuses to benchmark a server that is not healthy,
  3. it computes every percentile from RAW SAMPLES at analysis time.

(3) is here because of a specific bug rather than a principle: an earlier version
of the P2 table did `scalars.get('ttft_ms_p50', 0)`, a key that never existed,
silently got the default, and printed `0.0` in every row of a seven-rate sweep.
Nothing errored. The artifact was correct the whole time — the percentile simply
was not in it, because the schema deliberately stores samples and derives
percentiles later (methodology §5, R15).

So `samples_percentile()` is the ONLY way any driver in this repo reads a
percentile: it takes the metric name, goes to `artifact.samples`, and returns NaN
loudly if the samples are absent. There is no default-to-zero path. A missing
percentile renders as `n/a` in the table, which is a visible hole rather than an
invisible zero.

WHAT ELSE IS SHARED
-------------------
* `open_loop_dispatch` — the open-loop dispatch loop for a CALLER-SUPPLIED
  schedule. `bench.loadgen.run_load` builds its own prompts, which is right for
  P2 and useless for P4/P5, where the prompts ARE the experiment (prefix-sharing
  structure from `bench/workloads/generator.py`). This reuses `stream_one`,
  `RequestSpec`, `Phase` and hands back a real `LoadGenRun`, so `analyze()`,
  `validate_run()` and `render()` — including the R1 coordinated-omission guard
  and the R11 stationarity check — apply unchanged. loadgen.py is not modified.
* `Verdict` — validity as a composable value. A phase driver adds checks that
  loadgen cannot know about (a preemption run with zero preemptions, a
  zero-sharing control that shows sharing), and those must invalidate a run just
  as hard as dispatch drift does.
* `find_knee` / `find_crossover` — because a point value is a choice of
  operating point. Every driver here prints a curve and the place where it turns.
* `render_table` / `fmt_num` — NaN, None and empty tables render as `n/a` and
  `(no rows)`. Never as 0.0.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from bench.loadgen import (
    LoadGenConfig,
    LoadGenRun,
    Phase,
    RequestSpec,
    arrival_schedule,
    percentile,
    stream_one,
)
from serving.metrics.artifact import REGISTRY, Artifact, MetricSpec, Provenance

__all__ = [
    "NA",
    "Verdict",
    "banner",
    "base_url_of",
    "counter_delta",
    "find_crossover",
    "find_knee",
    "fmt_num",
    "get_json",
    "open_loop_dispatch",
    "prompt_text_from_ids",
    "render_table",
    "sample_count",
    "require_healthy",
    "samples_percentile",
    "scalar",
    "scheduler_snapshot",
    "specs_from_prompts",
    "write_summary",
]

# ---------------------------------------------------------------------------
# Metric definitions for the quantities P3/P4/P5 add.
#
# R16: a name means exactly one thing. Names already carrying a meaning —
# ttft_ms, e2e_ms, itl_ms, goodput_rps, offered_load_rps, output_tok_s,
# slo_attainment, cache_hit_rate, preemption_rate, dispatch_drift_ms — are
# REUSED here, never redefined. The specs below are the genuinely new ones, and
# each states its source because two numbers in the same unit measured from
# different places are not comparable (the engine's `peak_mem_mb`).
# ---------------------------------------------------------------------------

for _spec in [
    MetricSpec(
        "preemption_latency_tax_ms", "preemption latency tax", "ms",
        "client wall clock; preempted run minus matched unpreempted control",
        "A DIFFERENCE of two pooled percentiles measured in the same allocation "
        "at matched offered load, one against a KV pool small enough to force "
        "preemption and one against a pool that fits. Meaningless without both, "
        "so it is emitted only when the control run exists and is itself valid.",
    ),
    MetricSpec(
        "tokens_recomputed", "KV tokens discarded and later re-prefilled", "count",
        "server instrumentation (PreemptionStats.tokens_recomputed)",
        "The cost side of RECOMPUTE. Zero under a pure SWAP policy; a non-zero "
        "value on a swap arm means swap space ran out and the policy silently "
        "degraded, which invalidates the head-to-head.",
    ),
    MetricSpec(
        "bytes_swapped", "KV bytes copied between GPU and host", "bytes",
        "server instrumentation (PreemptionStats.bytes_swapped_out + _in)",
        "The cost side of SWAP: both directions, because resume pays the copy "
        "back. Zero under RECOMPUTE.",
    ),
    MetricSpec(
        "resume_latency_ms", "preemption resume latency", "ms",
        "server instrumentation (PreemptionStats.resume_seconds_*)",
        "Wall time of the resume operation itself. Reported next to resume STEPS "
        "because the stall a client perceives is measured in scheduler steps, "
        "which is what recompute mostly pays.",
    ),
    MetricSpec(
        "resume_steps", "scheduler steps spent preempted", "count",
        "server instrumentation (PreemptionStats.resume_steps_*)",
        "How long a victim sat preempted, in steps. The client-visible stall.",
    ),
    MetricSpec(
        "partial_hit_depth_blocks", "blocks matched before divergence", "count",
        "server instrumentation (RadixCache.stats.partial_hit_depths)",
        "Methodology §7 asks for partial-hit depth explicitly: it is where "
        "block-boundary and copy-on-write bugs show up, and a block-granularity "
        "hit rate alone cannot distinguish a deep match from a shallow one.",
    ),
    MetricSpec(
        "cache_evictions", "radix cache block evictions", "count",
        "server instrumentation (RadixCache.stats.evictions)",
        "Named for its source. NOT the same quantity as a preemption, which also "
        "frees blocks — R16 is the reason these do not share a name.",
    ),
    MetricSpec(
        "shared_blocks", "cached blocks with refcount > 1", "count",
        "server instrumentation (RadixCache.shared_blocks)",
        "Blocks genuinely shared by more than one live sequence right now. A "
        "gauge, sampled at the end of the window, not a counter.",
    ),
    MetricSpec(
        "cache_overhead_ttft_ms", "prefix cache TTFT cost when it never helps", "ms",
        "client wall clock; cache-on minus cache-off at matched workload and seed",
        "The zero-sharing control (§7: 'what does the cache cost when it never "
        "helps?'). Published whether or not it flatters; a positive value is a "
        "real cost and an argument for making the cache adaptive.",
    ),
    MetricSpec(
        "goodput_delta_rps", "goodput difference against the B5 baseline", "requests/s",
        "client aggregate; policy under test minus least-outstanding at matched load",
        "Signed on purpose. Negative is the §10 result the plan committed to "
        "publishing in advance, not a failure of the run.",
    ),
]:
    REGISTRY.register(_spec)


NA = "n/a"


# ---------------------------------------------------------------------------
# Formatting — a hole must LOOK like a hole
# ---------------------------------------------------------------------------


def fmt_num(x: Any, prec: int = 1) -> str:
    """
    Format a number for a table cell. None, NaN and inf render as `n/a`.

    The point is that a missing measurement never renders as `0.0`. A zero is a
    claim about the system; `n/a` is a claim about the measurement, and printing
    the first when you mean the second is how a sweep of seven rates published
    seven zeros without anybody noticing.
    """
    if x is None:
        return NA
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v) or math.isinf(v):
        return NA
    return f"{v:.{prec}f}"


def render_table(
    headers: list[str],
    rows: list[list[Any]],
    precisions: list[int] | None = None,
    empty_note: str = "(no rows — nothing was measured)",
) -> str:
    """
    Fixed-width table. Empty input renders the note, not an empty string, so a
    sweep that produced nothing says so instead of vanishing between headings.
    """
    if not rows:
        return empty_note
    prec = precisions or [1] * len(headers)
    if len(prec) < len(headers):
        prec = list(prec) + [1] * (len(headers) - len(prec))
    cells = [[fmt_num(v, prec[i]) if i < len(prec) else fmt_num(v) for i, v in enumerate(row)]
             for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, c in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(c))
            else:
                widths.append(len(c))
    out = ["  ".join(h.rjust(widths[i]) for i, h in enumerate(headers))]
    out.append("  ".join("-" * w for w in widths[:len(headers)]))
    for row in cells:
        out.append("  ".join(c.rjust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Reading numbers out of an artifact — samples only
# ---------------------------------------------------------------------------


def samples_percentile(art: Artifact, metric: str, q: float) -> float:
    """
    Percentile of `metric` computed HERE, from `artifact.samples[metric]`.

    Never reads `artifact.scalars`, and has NO default value: absent samples
    return NaN, which renders as `n/a`. This is the enforcement point for the
    P2 bug described in the module docstring.
    """
    xs = art.samples.get(metric) or []
    if not xs:
        return float("nan")
    return percentile(list(xs), q)


def sample_count(art: Artifact, metric: str) -> int:
    return len(art.samples.get(metric) or [])


def scalar(art: Artifact, name: str) -> float:
    """
    Read an AGGREGATE scalar (goodput, attainment) — quantities that are counts
    over a window, not percentiles. Missing returns NaN, never 0.0.
    """
    v = art.scalars.get(name)
    return float(v) if v is not None else float("nan")


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    """
    Composable run validity. `bench.loadgen.RunValidity` covers what the harness
    can see (drift, stationarity, a binding concurrency cap); a phase driver adds
    what only it knows — that a preemption benchmark recorded zero preemptions,
    that a zero-sharing control showed sharing, that a swap arm silently ran as
    recompute. Both kinds invalidate identically: excluded from any claim.
    """

    valid: bool = True
    reasons: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls) -> Verdict:
        return cls(True, [])

    @classmethod
    def bad(cls, *reasons: str) -> Verdict:
        return cls(False, list(reasons))

    def merge(self, other: Verdict | None) -> Verdict:
        if other is None:
            return self
        reasons = self.reasons + other.reasons
        return Verdict(valid=self.valid and other.valid and not reasons, reasons=reasons)

    def add(self, condition: bool, reason: str) -> Verdict:
        """Fail the verdict if `condition` holds."""
        if condition:
            return Verdict(False, self.reasons + [reason])
        return self

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "reasons": self.reasons}

    @property
    def label(self) -> str:
        return "OK" if self.valid else "INVALID"


# ---------------------------------------------------------------------------
# Curves — the knee and the crossover
# ---------------------------------------------------------------------------


def find_knee(
    loads: list[float],
    goodputs: list[float],
    attainments: list[float] | None = None,
    tracking_tolerance: float = 0.90,
    attainment_min: float = 0.95,
) -> dict[str, Any]:
    """
    The knee: the largest offered load at which goodput still TRACKS offered load
    and SLO attainment is still high (methodology §3, regions 1-3).

    Reported rather than a single goodput number, because a goodput figure quoted
    without the load it was taken at is a choice of operating point. If the
    highest load measured still tracks, the knee was NOT FOUND — the sweep did
    not go far enough, and saying so is the result.
    """
    pts = [
        (x, g, (attainments[i] if attainments else float("nan")))
        for i, (x, g) in enumerate(zip(loads, goodputs, strict=False))
        if not (math.isnan(x) or math.isnan(g))
    ]
    pts.sort(key=lambda p: p[0])
    if not pts:
        return {"knee_load_rps": None, "found": False,
                "reason": "no valid (load, goodput) points"}

    knee = None
    for x, g, a in pts:
        tracks = g >= tracking_tolerance * x
        attains = math.isnan(a) or a >= attainment_min
        if tracks and attains:
            knee = x
        else:
            break

    if knee is None:
        return {
            "knee_load_rps": pts[0][0], "found": True, "below_lowest_rate": True,
            "reason": (
                f"even the lowest offered load {pts[0][0]:g} req/s failed the "
                "tracking test; the system is already above its knee at the "
                "bottom of this sweep"
            ),
        }
    if knee == pts[-1][0]:
        return {
            "knee_load_rps": None, "found": False, "highest_tested_rps": knee,
            "reason": (
                f"goodput still tracks offered load at the highest rate tested "
                f"({knee:g} req/s). THE KNEE WAS NOT FOUND — the sweep is too "
                "short, and no capacity number may be quoted from it."
            ),
        }
    return {
        "knee_load_rps": knee, "found": True,
        "tracking_tolerance": tracking_tolerance, "attainment_min": attainment_min,
        "reason": (
            f"goodput tracks offered load up to {knee:g} req/s and stops "
            "tracking above it"
        ),
    }


def find_crossover(xs: list[float], advantage: list[float]) -> dict[str, Any]:
    """
    Where does an advantage disappear?

    `advantage[i] > 0` means the arm under test is AHEAD at `xs[i]`. The
    crossover is the first x at which it stops being ahead, linearly interpolated
    between the bracketing points. Returns `crossed=False` when the advantage
    never changes sign — including the case where it was never positive at all,
    which is a real and publishable answer ("it never won").

    This is the shape of both interesting P3 and P5 questions: does recompute
    stop beating swap at some sequence length, and does prefix-aware routing stop
    beating B5 at some offered load (methodology §10 case 3 predicts it does).
    """
    pts = [(x, a) for x, a in zip(xs, advantage, strict=False)
           if not (math.isnan(x) or math.isnan(a))]
    pts.sort(key=lambda p: p[0])
    if len(pts) < 2:
        return {"crossed": False, "crossover_x": None,
                "reason": f"only {len(pts)} usable point(s); a crossover needs at least 2"}

    if pts[0][1] <= 0:
        return {
            "crossed": False, "crossover_x": None, "ahead_at_start": False,
            "reason": (
                f"never ahead: the advantage is {pts[0][1]:.4g} at the lowest x "
                f"({pts[0][0]:g}). There is no crossover to report because there "
                "was no lead to lose."
            ),
        }

    for (x0, a0), (x1, a1) in zip(pts, pts[1:], strict=False):
        if a1 <= 0:
            span = a0 - a1
            x = x0 if span == 0 else x0 + (x1 - x0) * (a0 / span)
            return {
                "crossed": True, "crossover_x": x, "ahead_at_start": True,
                "bracket": [x0, x1], "advantage_bracket": [a0, a1],
                "reason": (
                    f"advantage falls from {a0:.4g} at x={x0:g} to {a1:.4g} at "
                    f"x={x1:g}; linear interpolation puts the crossing at {x:.4g}"
                ),
            }
    return {
        "crossed": False, "crossover_x": None, "ahead_at_start": True,
        "highest_tested_x": pts[-1][0],
        "reason": (
            f"still ahead at the largest x tested ({pts[-1][0]:g}, advantage "
            f"{pts[-1][1]:.4g}). NO CROSSOVER WAS FOUND — it may exist beyond "
            "the swept range, and this sweep cannot say it does not."
        ),
    }


# ---------------------------------------------------------------------------
# Server interaction
# ---------------------------------------------------------------------------


def base_url_of(url: str) -> str:
    """`http://h:p/v1/chat/completions` -> `http://h:p`. Same rule as run_p2."""
    return url.rsplit("/v1/", 1)[0]


async def get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as c:
        resp = await c.get(url)
        return dict(resp.json())


async def require_healthy(url: str, label: str) -> dict[str, Any]:
    """
    Refuse to benchmark an unhealthy server. FATAL, not advisory.

    Job 11599044 collected a whole sweep against a server whose scheduler loop
    was dead: HTTP 200, well-formed, EMPTY SSE bodies, `/health` saying
    `unhealthy` the entire time. A harness that cannot tell a broken server from
    a fast one is worse than no harness.
    """
    health_url = base_url_of(url) + "/health"
    try:
        h = await get_json(health_url)
    except Exception as exc:  # noqa: BLE001 — unreachable server is fatal, not data
        raise SystemExit(f"FATAL [{label}]: cannot reach {health_url}: {exc}") from exc
    loop_info = h.get("loop", {}) or {}
    if h.get("status") != "ok" or not loop_info.get("healthy", True):
        raise SystemExit(
            f"FATAL [{label}]: server at {health_url} is not healthy; refusing to "
            f"benchmark it.\n  status:     {h.get('status')}\n"
            f"  loop:       running={loop_info.get('running')} "
            f"healthy={loop_info.get('healthy')}\n"
            f"  last_error: {loop_info.get('last_error')}\n"
            f"  fatal:      {h.get('fatal')}"
        )
    return h


async def scheduler_snapshot(url: str) -> dict[str, Any]:
    """
    `/metrics` -> `scheduler`, which carries the preemption counters and the
    `cache_*` block-granularity counters. Server-side truth; latency stays
    client-side (R1).
    """
    m = await get_json(base_url_of(url) + "/metrics")
    snap = m.get("scheduler") or {}
    return dict(snap)


def counter_delta(
    before: dict[str, Any], after: dict[str, Any], keys: list[str]
) -> dict[str, float]:
    """
    after - before for monotone counters. A key missing on either side yields NaN
    rather than 0: 'the server never reported this' and 'nothing happened' are
    different statements and only one of them is a measurement.
    """
    out: dict[str, float] = {}
    for k in keys:
        b, a = before.get(k), after.get(k)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)) \
                and not isinstance(b, bool) and not isinstance(a, bool):
            out[k] = float(a) - float(b)
        else:
            out[k] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Open-loop dispatch of a CALLER-SUPPLIED schedule
# ---------------------------------------------------------------------------


def prompt_text_from_ids(token_ids: list[int] | tuple[int, ...], prefix: str = "") -> str:
    """
    Render generator token ids to a prompt string, prefix-preservingly.

    The workload generator's whole point is that sharing is exact over TOKEN IDS
    (bench/workloads/generator.py). The chat endpoint takes text, so ids have to
    be rendered — and a decode/re-encode round trip is NOT guaranteed to return
    the same ids, which would make the realized sharing structure a property of
    the tokenizer rather than of the generator.

    This mapping avoids the worst of that: each id becomes a distinct
    space-separated word, so two prompts sharing an id prefix share a character
    prefix that ends at a whitespace boundary, and a BPE tokenizer re-encodes a
    shared word-boundary prefix to a shared token prefix. It is not a proof.
    The measured, server-side block hit rate is the ground truth; the generator's
    oracle sharing rate is published next to it precisely so the gap is visible.
    """
    body = " ".join(f"w{t}" for t in token_ids)
    return f"{prefix}{body}" if prefix else body


def specs_from_prompts(
    cfg: LoadGenConfig,
    prompts: list[str],
    max_tokens: list[int],
    times: list[float] | None = None,
) -> list[RequestSpec]:
    """
    Build the schedule from prompts the caller generated, with intended dispatch
    times from the SAME arrival process `bench.loadgen.build_schedule` uses.

    Times are computed in advance and are the origin for every latency (R1);
    phases are assigned from the config's window boundaries so warmup and drain
    are discarded at analysis exactly as in a P2 run (R11).
    """
    if times is None:
        times = arrival_schedule(cfg.rate_rps, cfg.horizon_s, cfg.process, cfg.seed)
    n = min(len(times), len(prompts), len(max_tokens))
    specs = []
    for i in range(n):
        t = times[i]
        if t < cfg.steady_start_s:
            phase = Phase.WARMUP
        elif t < cfg.steady_end_s:
            phase = Phase.STEADY
        else:
            phase = Phase.DRAIN
        specs.append(
            RequestSpec(
                request_id=i,
                intended_send_time=t,
                prompt=prompts[i],
                max_tokens=max(1, int(max_tokens[i])),
                phase=phase,
            )
        )
    return specs


async def open_loop_dispatch(
    cfg: LoadGenConfig,
    specs: list[RequestSpec],
    client: Any = None,
) -> LoadGenRun:
    """
    Dispatch a precomputed schedule OPEN LOOP. The loop never awaits a response.

    A copy of `bench.loadgen.run_load`'s dispatch discipline for the case where
    the schedule is the experiment (P4/P5 prefix structures). Same guarantees,
    and it returns a real `LoadGenRun` so `analyze()` applies unchanged: latency
    from intended dispatch, drift recorded, in-flight sampled for the
    stationarity check.

    `client` is injectable so the whole path is testable against a mock transport
    with no server anywhere.
    """
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            timeout=cfg.request_timeout_s,
        )
    sem = asyncio.Semaphore(cfg.max_concurrency) if cfg.max_concurrency > 0 else None
    inflight = [0]
    inflight_samples: list[tuple[float, float]] = []
    stop = asyncio.Event()

    t0 = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    async def sampler() -> None:
        while not stop.is_set():
            inflight_samples.append((time.perf_counter() - t0, float(inflight[0])))
            try:
                await asyncio.wait_for(stop.wait(), timeout=cfg.inflight_sample_interval_s)
            except TimeoutError:
                continue

    sampler_task = asyncio.create_task(sampler())
    tasks: list[asyncio.Task[Any]] = []
    try:
        for spec in specs:
            delay = spec.intended_send_time - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(stream_one(client, spec, cfg, t0, inflight, sem)))
        results = list(await asyncio.gather(*tasks))
    finally:
        stop.set()
        await sampler_task
        if owns_client:
            await client.aclose()

    inflight_samples.append((time.perf_counter() - t0, float(inflight[0])))
    return LoadGenRun(
        cfg=cfg,
        schedule=list(specs),
        results=results,
        inflight_samples=inflight_samples,
        wall_seconds=time.perf_counter() - t0,
        started_utc=started_utc,
    )


# ---------------------------------------------------------------------------
# Provenance banner and summary writing
# ---------------------------------------------------------------------------


def banner(prov: Provenance, title: str) -> tuple[bool, list[str]]:
    """
    Print provenance and SURFACE the publishability verdict.

    `Provenance.is_publishable()` already encodes the rules — dirty tree, missing
    SHA, non-`inferno` QOS, unpinned engine, no seed. What a driver owes is
    making the answer impossible to miss, since a blocker printed once at the top
    of a long run is a blocker nobody reads at the bottom.
    """
    ok, blockers = prov.is_publishable()
    print("=" * 78)
    print(title)
    print("-" * 78)
    print(f"allocation   {prov.allocation_id}")
    print(f"node/host    {prov.slurm_node or prov.hostname}")
    print(f"gpu          {prov.gpu_name} x{prov.gpu_count}")
    print(f"qos          {prov.slurm_qos}")
    print(f"repo         {prov.repo_sha}  dirty={prov.repo_dirty}")
    print(f"engine       {prov.engine_tag or prov.engine_sha}")
    print(f"seed         {prov.seed}")
    print(f"publishable  {ok}")
    if not ok:
        for b in blockers:
            print(f"  !! NOT PUBLISHABLE: {b}")
    print("=" * 78)
    return ok, blockers


def write_summary(results_dir: str | Path, filename: str, payload: dict[str, Any]) -> Path:
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path
