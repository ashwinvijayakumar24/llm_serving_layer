"""
Phase 5 benchmark driver: prefix-aware routing across N replicas (claim S5).

WHAT THIS MEASURES
------------------
Goodput and TTFT for `prefix_aware` against **B5 least-outstanding** — the real
baseline — and against B4 round-robin, across sharing structures and offered
loads, with N replicas behind the router.

B5 IS THE BASELINE. B4 IS TABLE STAKES.
---------------------------------------
methodology §6: B5 is load-aware and cache-blind, so the gap between B5 and
prefix-aware isolates EXACTLY the value of cache awareness. B4 is deliberately
weak — it cannot see load at all — so beating it demonstrates load balancing and
nothing else. A run that beats B4 but not B5 is reported in those words, because
that is what it means.

THE LOSING CASES ARE FIRST-CLASS RESULTS
-----------------------------------------
methodology §10 names, in advance, four regimes where prefix-aware routing should
be expected to lose or tie, and the phase plan (§8) makes publishing them
mandatory:

  §10.1  zero sharing          nothing to be cache-aware about
  §10.2  uniform sharing       every request shares the SAME prefix, so every
                               replica caches it and affinity degenerates to
                               "pick any replica" — the naive mental model
  §10.3  above-the-knee load   affinity and load balance conflict directly; the
                               replica holding the prefix is busy BECAUSE it
                               holds the popular prefix
  §10.7  hot-prefix skew       affinity self-inflicts a hotspot

They are measured here by default, printed in their own section BEFORE the wins,
and their verdicts are stated as predictions kept or broken. `--scenarios` can
narrow the sweep, but dropping a losing case is recorded loudly in the output and
in the summary, because — §10 again — *a results table containing only wins is
evidence of workload selection, and a reader who knows this field will assume
exactly that.*

§10.3 is a property of the LOAD axis, not of a workload, which is why every
scenario is swept over offered load and why the deliverable is the CROSSOVER: the
offered load at which prefix-aware stops beating B5. Publishing that crossover is
a better result than publishing a win.

USAGE
-----
    python3 -m bench.run_p5 \\
        --router prefix_aware=http://127.0.0.1:9000/v1/chat/completions \\
        --router least_outstanding=http://127.0.0.1:9001/v1/chat/completions \\
        --router round_robin=http://127.0.0.1:9002/v1/chat/completions \\
        --replicas http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003 \\
        --loads 2,4,8,16,32 --duration 60
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any

from bench.driver_common import (
    Verdict,
    banner,
    base_url_of,
    find_crossover,
    find_knee,
    fmt_num,
    get_json,
    open_loop_dispatch,
    prompt_text_from_ids,
    render_table,
    require_healthy,
    sample_count,
    samples_percentile,
    scalar,
    specs_from_prompts,
    write_summary,
)
from bench.loadgen import LoadGenConfig, analyze, arrival_schedule
from bench.workloads.generator import LengthSpec, Workload, WorkloadConfig, generate
from serving.metrics.artifact import Provenance

POLICY_PREFIX = "prefix_aware"
POLICY_B5 = "least_outstanding"
POLICY_B4 = "round_robin"


@dataclass(frozen=True)
class Scenario:
    """
    One workload regime, with the §10 prediction attached BEFORE it is measured.

    `expected` is not decoration. It is what makes a losing row a confirmed
    prediction rather than an embarrassment, and it is what makes a WINNING row
    in a losing case interesting enough to investigate.
    """

    name: str
    structure: str
    sharing_rate: float
    n_shared_prefixes: int = 4
    prefix_popularity: str = "uniform"
    losing_case: bool = False
    methodology_ref: str = ""
    expected: str = ""


SCENARIOS: dict[str, Scenario] = {
    s.name: s for s in [
        Scenario(
            name="system_prompt_sharing", structure="system", sharing_rate=0.8,
            n_shared_prefixes=4, methodology_ref="§4 (the common production shape)",
            expected="prefix_aware SHOULD beat B5 here: several distinct prefixes, "
                     "so affinity carries real information and does not collapse "
                     "onto one replica.",
        ),
        Scenario(
            name="deep_conversational", structure="conversational", sharing_rate=0.8,
            methodology_ref="§4 (MOST FAVOURABLE structure for a radix cache)",
            expected="prefix_aware should win by the largest margin here, and the "
                     "margin is NOT comparable to the system-prompt rows. Labeled "
                     "as the favourable case wherever it appears.",
        ),
        Scenario(
            name="zero_sharing", structure="zero", sharing_rate=0.0, losing_case=True,
            methodology_ref="§10.1",
            expected="TIE with B5, or a slight LOSS: there is nothing to be "
                     "cache-aware about, and any deviation from load-optimal "
                     "placement is pure loss.",
        ),
        Scenario(
            name="uniform_prefix", structure="system", sharing_rate=1.0,
            n_shared_prefixes=1, losing_case=True, methodology_ref="§10.2",
            expected="B5 WINS. Every request shares the same prefix, so every "
                     "replica caches it after warmup and prefix awareness "
                     "degenerates to 'pick any replica'. This is the naive mental "
                     "model of prefix caching and the case where the clever router "
                     "is worth nothing.",
        ),
        Scenario(
            name="hot_prefix_skew", structure="system", sharing_rate=0.9,
            n_shared_prefixes=8, prefix_popularity="zipf", losing_case=True,
            methodology_ref="§10.7",
            expected="prefix_aware LOSES unless affinity is blended with load: one "
                     "very hot prefix sends a disproportionate share of traffic to "
                     "whichever replica owns it. §10 calls this the most likely "
                     "place for a genuinely bad result, and therefore the most "
                     "valuable one to measure.",
        ),
    ]
}

# §10.3 is not a workload — it is the load axis, present in every scenario.
ABOVE_KNEE_NOTE = (
    "§10.3 (above-the-knee load) is measured by the LOAD SWEEP itself, not by a "
    "separate workload: affinity and load balance conflict directly near "
    "saturation, so the prediction is that prefix_aware wins at low-to-moderate "
    "load and loses above the knee unless the policy blends affinity with load. "
    "The crossover column below is that prediction's test."
)


@dataclass
class Cell:
    """One (scenario, offered load, policy) measurement."""

    scenario: str
    load: float
    policy: str
    artifact: Any = None
    verdict: Verdict = field(default_factory=Verdict.ok)
    policy_stats: dict[str, Any] = field(default_factory=dict)
    path: str | None = None

    def ttft(self, q: float) -> float:
        """From RAW SAMPLES. There is no scalar shortcut in this driver (R15)."""
        return samples_percentile(self.artifact, "ttft_ms", q) if self.artifact else float("nan")

    @property
    def goodput(self) -> float:
        return scalar(self.artifact, "goodput_rps") if self.artifact else float("nan")

    @property
    def attainment(self) -> float:
        return scalar(self.artifact, "slo_attainment") if self.artifact else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "offered_load_rps": self.load,
            "policy": self.policy,
            "valid": self.verdict.valid,
            "reasons": self.verdict.reasons,
            "goodput_rps": self.goodput,
            "slo_attainment": self.attainment,
            "ttft_p50": self.ttft(50),
            "ttft_p99": self.ttft(99),
            "n_ttft_samples": sample_count(self.artifact, "ttft_ms") if self.artifact else 0,
            "router_policy_stats": self.policy_stats,
            "artifact": self.path,
        }


# ---------------------------------------------------------------------------
# Pure analysis
# ---------------------------------------------------------------------------


def lookup(cells: list[Cell], scenario: str, load: float, policy: str) -> Cell | None:
    for c in cells:
        if c.scenario == scenario and c.load == load and c.policy == policy:
            return c
    return None


def advantage_series(
    cells: list[Cell], scenario: str, loads: list[float], policy: str, baseline: str
) -> tuple[list[float], list[float]]:
    """
    (loads, goodput advantage over the baseline) for the valid cells only.

    A cell that failed a validity check contributes NOTHING — not a zero, not an
    optimistic carry-forward. An invalid run may not back a claim, and a
    difference computed against one is a claim.
    """
    xs, adv = [], []
    for load in loads:
        a, b = lookup(cells, scenario, load, policy), lookup(cells, scenario, load, baseline)
        if a is None or b is None or not a.verdict.valid or not b.verdict.valid:
            continue
        ga, gb = a.goodput, b.goodput
        if ga != ga or gb != gb:
            continue
        xs.append(load)
        adv.append(ga - gb)
    return xs, adv


def scenario_verdict(
    scenario: Scenario, xs: list[float], adv: list[float], crossover: dict[str, Any]
) -> str:
    """
    Did §10's prediction hold for this scenario? One sentence, in the direction
    the numbers actually came out.
    """
    if not xs:
        return (
            f"{scenario.name}: NOT MEASURED — no offered load produced a valid "
            f"{POLICY_PREFIX} cell and a valid {POLICY_B5} cell. The prediction "
            f"({scenario.methodology_ref}) is untested, which is not the same as "
            "confirmed."
        )
    wins = [x for x, a in zip(xs, adv, strict=False) if a > 0]
    losses = [x for x, a in zip(xs, adv, strict=False) if a < 0]
    ties = [x for x, a in zip(xs, adv, strict=False) if a == 0]
    if scenario.losing_case:
        if not wins:
            return (
                f"{scenario.name}: PREDICTION HELD ({scenario.methodology_ref}). "
                f"prefix_aware does not beat B5 at any load tested "
                f"({len(losses)} loss(es), {len(ties)} tie(s)) — as predicted "
                "before the run."
            )
        return (
            f"{scenario.name}: PREDICTION DID NOT HOLD ({scenario.methodology_ref}). "
            f"prefix_aware BEAT B5 at {sorted(wins)} req/s despite being predicted "
            "to lose or tie. Either the policy's affinity/load blend is doing more "
            "than the prediction assumed, or the workload is not the regime it is "
            "labelled with — worth investigating before it is quoted as a win."
        )
    if not losses:
        return (
            f"{scenario.name}: prefix_aware beats B5 at every load tested "
            f"({min(xs):g}..{max(xs):g} req/s). No crossover inside the swept "
            f"range: {crossover.get('reason')}"
        )
    return (
        f"{scenario.name}: prefix_aware beats B5 at {sorted(wins)} and loses at "
        f"{sorted(losses)} req/s. {crossover.get('reason')}"
    )


def b4_only_warning(cells: list[Cell], scenario: str, loads: list[float]) -> str | None:
    """
    The result methodology §6 demands be published in its own words: beating B4
    while not beating B5 demonstrates load balancing, not prefix awareness.
    """
    _, adv_b5 = advantage_series(cells, scenario, loads, POLICY_PREFIX, POLICY_B5)
    xs4, adv_b4 = advantage_series(cells, scenario, loads, POLICY_PREFIX, POLICY_B4)
    if not xs4 or not adv_b5:
        return None
    beats_b4 = all(a > 0 for a in adv_b4)
    beats_b5 = any(a > 0 for a in adv_b5)
    if beats_b4 and not beats_b5:
        return (
            f"{scenario}: prefix_aware beats B4 (round-robin) at every load and "
            "beats B5 (least-outstanding) at none. Per methodology §6 that result "
            "is LOAD BALANCING DEMONSTRATED, PREFIX AWARENESS NOT — and it is "
            "published in those words rather than as a routing win."
        )
    return None


def build_workload(sc: Scenario, n_requests: int, args: argparse.Namespace) -> Workload:
    return generate(WorkloadConfig(
        n_requests=n_requests,
        structure=sc.structure,
        sharing_rate=sc.sharing_rate,
        block_size=args.block_size,
        seed=args.seed,
        prompt=LengthSpec(dist="lognormal", mean=args.prompt_mean, sigma=args.sigma),
        output=LengthSpec(dist="lognormal", mean=args.output_mean, sigma=args.sigma),
        n_shared_prefixes=sc.n_shared_prefixes,
        shared_prefix_tokens=args.shared_prefix_tokens,
        prefix_popularity=sc.prefix_popularity,
        name=f"p5_{sc.name}",
    ))


def parse_routers(pairs: list[str]) -> dict[str, str]:
    """`--router name=url`, repeatable. Unknown names are rejected, not defaulted."""
    out: dict[str, str] = {}
    for raw in pairs:
        if "=" not in raw:
            raise SystemExit(
                f"--router expects NAME=URL (e.g. {POLICY_PREFIX}=http://host:9000"
                f"/v1/chat/completions); got {raw!r}"
            )
        name, url = raw.split("=", 1)
        out[name.strip()] = url.strip()
    if POLICY_B5 not in out:
        raise SystemExit(
            f"--router {POLICY_B5}=URL is REQUIRED. B5 is the real baseline "
            "(methodology §6): it is load-aware and cache-blind, so the gap "
            "between it and prefix-aware is the only thing that isolates cache "
            "awareness. A P5 run without it cannot support claim S5."
        )
    if POLICY_PREFIX not in out:
        raise SystemExit(f"--router {POLICY_PREFIX}=URL is required; it is the system under test.")
    return out


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


async def run_cell(
    url: str, policy: str, sc: Scenario, load: float,
    workload: Workload, args: argparse.Namespace,
) -> Cell:
    cfg = LoadGenConfig(
        url=url, rate_rps=load, duration_s=args.duration, warmup_s=args.warmup,
        drain_s=args.drain, seed=args.seed, slo_ttft_ms=args.slo_ttft_ms,
        slo_itl_ms=args.slo_itl_ms, request_timeout_s=args.timeout, process="poisson",
        name=f"p5_{sc.name}_{policy}_load{load:g}",
        extra_config={"routing_policy": policy, "scenario": sc.name,
                      "losing_case": sc.losing_case,
                      "methodology_ref": sc.methodology_ref},
    )
    specs = specs_from_prompts(
        cfg,
        [prompt_text_from_ids(r.token_ids) for r in workload.requests],
        [r.max_tokens for r in workload.requests],
    )
    run = await open_loop_dispatch(cfg, specs)
    a = analyze(run)
    art = a.artifact
    art.config["routing_policy"] = policy
    art.realized_workload["prefix_workload"] = workload.artifact_fields()
    art.notes.append(f"scenario {sc.name} — {sc.methodology_ref}: {sc.expected}")

    stats: dict[str, Any] = {}
    try:
        h = await get_json(base_url_of(url) + "/metrics")
        stats = dict(h.get("policy") or {})
        art.realized_workload["router_metrics"] = h
    except Exception:  # noqa: BLE001 — router metrics are context, not the measurement
        stats = {}

    verdict = Verdict(a.validity.valid, list(a.validity.reasons))
    verdict = verdict.merge(
        Verdict(False, [f"WORKLOAD DEGENERACY (R14): {w}" for w in workload.degeneracy_warnings])
        if workload.degeneracy_warnings else Verdict.ok()
    )
    if not verdict.valid:
        art.notes.insert(0, "RUN INVALID: " + " | ".join(verdict.reasons))
    path = art.write(args.results_dir)
    return Cell(scenario=sc.name, load=load, policy=policy, artifact=art,
                verdict=verdict, policy_stats=stats, path=str(path))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m bench.run_p5",
        description=(
            "Phase 5 (S5): prefix-aware routing vs B5 least-outstanding and B4 "
            "round-robin across N replicas, swept over sharing structure and "
            "offered load. The §10 LOSING CASES are measured and printed first."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--router", action="append", default=[], metavar="NAME=URL",
                   help=f"repeatable. {POLICY_PREFIX} and {POLICY_B5} are required; "
                        f"{POLICY_B4} is table stakes and strongly recommended")
    p.add_argument("--replicas", default="",
                   help="comma-separated replica BASE urls behind the routers "
                        "(the Slurm script starts several; N is reported with every "
                        "result because §10.6 says the benefit grows with N)")
    p.add_argument("--scenarios", default=",".join(SCENARIOS),
                   help="workload regimes to run; dropping a §10 losing case is "
                        "recorded loudly in the output")
    p.add_argument("--loads", default="2,4,8,16,32",
                   help="offered loads, req/s — sweep PAST the knee or §10.3 is untested")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--warmup", type=float, default=15.0)
    p.add_argument("--drain", type=float, default=10.0)
    p.add_argument("--prompt-mean", type=int, default=512)
    p.add_argument("--output-mean", type=int, default=128)
    p.add_argument("--sigma", type=float, default=0.8)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--shared-prefix-tokens", type=int, default=128)
    p.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    p.add_argument("--slo-itl-ms", type=float, default=100.0)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=20260801)
    p.add_argument("--results-dir", default="results/p5")
    p.add_argument("--label", default="s5")
    return p


async def _amain(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    routers = parse_routers(args.router)
    loads = [float(x) for x in args.loads.split(",") if x.strip()]
    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario(s) {unknown}; known: {sorted(SCENARIOS)}")
    scenarios = [SCENARIOS[n] for n in names]
    replicas = [r.strip() for r in args.replicas.split(",") if r.strip()]

    prov = Provenance.capture(repo_root=".", seed=args.seed)
    pub_ok, blockers = banner(prov, "PHASE 5 / S5 — prefix-aware routing across N replicas")

    # ---- 0. pre-flight -----------------------------------------------------
    print("\n### FLEET PRE-FLIGHT\n")
    for policy, url in routers.items():
        h = await require_healthy(url, policy)
        reported = str((h.get("policy") or {}).get("policy", "")).lower()
        # An arm may be named `<policy>@<variant>` to sweep one policy's
        # configuration -- e.g. prefix_aware@blend0.25. Only the part before '@'
        # is a policy claim; the suffix is a label for the tables.
        #
        # This does NOT weaken the guard, which exists to catch an artifact
        # saying prefix_aware while B5 is actually serving. That check compares
        # the BASE name, so prefix_aware@blend0.00 pointed at a least_outstanding
        # router still raises. What it now permits is the honest case the guard
        # had no vocabulary for: several arms that really are the same policy,
        # differing only in how it is configured.
        base = policy.split("@", 1)[0]
        print(f"  router {policy:22} reports policy={reported or 'unreported'}   {url}")
        if reported and reported != base:
            raise SystemExit(
                f"FATAL: the {policy!r} arm points at a router running "
                f"{reported!r}. Benchmarking B5 while the artifact says "
                f"{POLICY_PREFIX} is the worst-shaped error this project can make "
                "(the same class the SERVING_STATIC_BATCHING banner exists for)."
            )
    for i, r in enumerate(replicas):
        h = await require_healthy(r + "/v1/chat/completions", f"replica{i}")
        print(f"  replica {i:<2} {h.get('status')}   {r}")
    n_replicas = len(replicas)
    print(f"\n  N = {n_replicas} replicas. §10.6: routing benefit is expected to GROW "
          "with N,")
    print("  so this N is part of every number below and not a footnote. At N<=2 there")
    print("  is little routing freedom and a small benefit understates the design.")
    print(f"\n  {ABOVE_KNEE_NOTE}")

    dropped = [n for n, s in SCENARIOS.items() if s.losing_case and n not in names]
    if dropped:
        print(f"\n  !! LOSING CASES DROPPED FROM THIS RUN: {dropped}")
        print("     §10 commits to publishing cases 1, 2, 3 and 7 whether or not they")
        print("     flatter. A table missing them is evidence of workload selection.")

    # ---- 1. sweep ----------------------------------------------------------
    cells: list[Cell] = []
    print(f"\n### SWEEP — {len(scenarios)} scenarios x {len(loads)} loads x "
          f"{len(routers)} policies\n")
    for sc in scenarios:
        print(f"  {sc.name}  ({sc.methodology_ref}"
              f"{', PREDICTED LOSING CASE' if sc.losing_case else ''})")
        for load in loads:
            times = arrival_schedule(load, args.warmup + args.duration + args.drain,
                                     "poisson", args.seed)
            w = build_workload(sc, len(times), args)
            for policy, url in routers.items():
                cell = await run_cell(url, policy, sc, load, w, args)
                cells.append(cell)
                print(f"    load {load:6.1f}  {policy:18} goodput "
                      f"{fmt_num(cell.goodput, 2):>7}  attain "
                      f"{fmt_num(100 * cell.attainment, 1):>6}%  ttft_p50 "
                      f"{fmt_num(cell.ttft(50)):>9}  ttft_p99 "
                      f"{fmt_num(cell.ttft(99)):>10}  {cell.verdict.label}")

    # ---- 2. LOSING CASES FIRST --------------------------------------------
    print("\n" + "=" * 78)
    print("### THE LOSING CASES (methodology §10) — PUBLISHED FIRST, NOT AS FOOTNOTES")
    print("=" * 78)
    crossovers: dict[str, Any] = {}
    verdicts: list[str] = []
    for sc in scenarios:
        if not sc.losing_case:
            continue
        print(f"\n  {sc.name}  [{sc.methodology_ref}]")
        print(f"  predicted in advance: {sc.expected}")
        xs, adv = advantage_series(cells, sc.name, loads, POLICY_PREFIX, POLICY_B5)
        crossovers[sc.name] = find_crossover(xs, adv)
        print("")
        print(_scenario_table(cells, sc.name, loads, list(routers)))
        v = scenario_verdict(sc, xs, adv, crossovers[sc.name])
        verdicts.append(v)
        print(f"\n  -> {v}")

    # ---- 3. the rest of the sweep -----------------------------------------
    print("\n" + "=" * 78)
    print("### THE REMAINING WORKLOADS")
    print("=" * 78)
    for sc in scenarios:
        if sc.losing_case:
            continue
        print(f"\n  {sc.name}  [{sc.methodology_ref}]")
        print(f"  expected before the run: {sc.expected}")
        xs, adv = advantage_series(cells, sc.name, loads, POLICY_PREFIX, POLICY_B5)
        crossovers[sc.name] = find_crossover(xs, adv)
        print("")
        print(_scenario_table(cells, sc.name, loads, list(routers)))
        v = scenario_verdict(sc, xs, adv, crossovers[sc.name])
        verdicts.append(v)
        print(f"\n  -> {v}")

    # ---- 4. crossover and knee --------------------------------------------
    print("\n### CROSSOVER — the offered load at which prefix_aware stops beating B5\n")
    cross_rows = []
    for sc in scenarios:
        c = crossovers.get(sc.name, {})
        cross_rows.append([
            sc.name, "yes" if sc.losing_case else "no",
            c.get("crossover_x"), c.get("reason", ""),
        ])
    print(render_table(["scenario", "losing_case", "crossover_rps", "basis"],
                       cross_rows, precisions=[0, 0, 2, 0]))

    print("\n### KNEE OF THE B5 BASELINE (the capacity the crossover should be read against)\n")
    knee_rows = []
    knees: dict[str, Any] = {}
    for sc in scenarios:
        pts = [(load, lookup(cells, sc.name, load, POLICY_B5)) for load in loads]
        valid = [(x, c) for x, c in pts if c is not None and c.verdict.valid]
        knee = find_knee([x for x, _ in valid], [c.goodput for _, c in valid],
                         [c.attainment for _, c in valid])
        knees[sc.name] = knee
        knee_rows.append([sc.name, knee.get("knee_load_rps"), knee.get("reason", "")])
    print(render_table(["scenario", "b5_knee_rps", "basis"], knee_rows, precisions=[0, 2, 0]))

    # ---- 5. the sentences that have to be said -----------------------------
    print("\n### VERDICTS\n")
    for v in verdicts:
        print(f"  - {v}")
    for sc in scenarios:
        w = b4_only_warning(cells, sc.name, loads)
        if w:
            print(f"  - {w}")
    print("\n  A table containing only wins is evidence of workload selection. The")
    print("  losing cases above were named in docs/BENCHMARK_METHODOLOGY.md §10 BEFORE")
    print("  this run and are printed ahead of every win for that reason.")

    invalid = [c for c in cells if not c.verdict.valid]
    if invalid:
        print(f"\n!! {len(invalid)} cell(s) INVALID and excluded from every claim "
              "and every difference:")
        for c in invalid:
            print(f"   {c.scenario} load={c.load:g} {c.policy}:")
            for r in c.verdict.reasons:
                print(f"      - {r}")
    if not pub_ok:
        print(f"\n!! NOT PUBLISHABLE: {blockers}")

    path = write_summary(args.results_dir, f"p5_{args.label}_summary.json", {
        "n_replicas": n_replicas,
        "replicas": replicas,
        "routers": routers,
        "loads": loads,
        "scenarios": {sc.name: {
            "structure": sc.structure, "sharing_rate": sc.sharing_rate,
            "losing_case": sc.losing_case, "methodology_ref": sc.methodology_ref,
            "predicted": sc.expected,
        } for sc in scenarios},
        "dropped_losing_cases": dropped,
        "cells": [c.as_dict() for c in cells],
        "crossovers": crossovers,
        "b5_knee": knees,
        "verdicts": verdicts,
        "publishable": pub_ok,
        "blockers": blockers,
        "allocation_id": prov.allocation_id,
    })
    print(f"\nsummary -> {path}")
    return 2 if (invalid or not cells) else 0


def _scenario_table(cells: list[Cell], scenario: str, loads: list[float],
                    policies: list[str]) -> str:
    """
    Per-scenario table: one row per (load, policy), with the signed difference
    against B5 in its own column so the sign is impossible to overlook.
    """
    rows: list[list[Any]] = []
    for load in loads:
        b5 = lookup(cells, scenario, load, POLICY_B5)
        for policy in policies:
            c = lookup(cells, scenario, load, policy)
            if c is None:
                continue
            d = (c.goodput - b5.goodput) if (b5 is not None and b5.verdict.valid
                                             and c.verdict.valid) else float("nan")
            rows.append([
                load, policy, c.goodput, 100 * c.attainment, c.ttft(50), c.ttft(99),
                d, sample_count(c.artifact, "ttft_ms") if c.artifact else 0,
                c.verdict.label,
            ])
    return render_table(
        ["load", "policy", "goodput", "attain%", "ttft_p50", "ttft_p99",
         "d_goodput_vs_B5", "n", "valid"],
        rows, precisions=[1, 0, 2, 1, 1, 1, 3, 0, 0],
    )


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    sys.exit(main())
