"""
Phase 4 benchmark driver: the radix prefix cache (claim S4).

WHAT THIS MEASURES
------------------
Block-granularity hit rate and TTFT AS A FUNCTION OF SHARING RATE, swept across
ALL FOUR prefix-sharing structures in `bench/workloads/generator.py`, with the
cache on and off at matched workload and seed.

ZERO SHARING IS THE CONTROL AND IT IS MANDATORY
-----------------------------------------------
methodology §7: *"what does the cache cost when it never helps?"* The
zero-sharing structure runs every request unique from token 0, so the radix walk,
the insert, and the refcount bookkeeping all happen for no benefit whatsoever.
Whatever that costs in TTFT is the honest floor and it is published in its own
section, ahead of any win, whether or not it flatters. `--structures` cannot
exclude it: `zero` is prepended if omitted, because a cache benchmark without its
control is a cache advertisement.

A hit rate is also not a number on its own (§7). Every hit rate printed here has
the workload's realized sharing structure printed beside it — the generator's
ORACLE block-sharing rate, which is what an infinite never-evicting cache would
hit. The gap between oracle and measured is eviction plus implementation, and
without the bound the measured number has nothing to be judged against.

WHY A SEPARATE cache-off SERVER
-------------------------------
The cache is wired at server construction, so on/off is two servers, launched by
the same job into the same allocation. The driver checks that they actually
differ — a matched pair that is not matched would report a null result and look
like a careful measurement.

RE-TOKENIZATION CAVEAT, STATED RATHER THAN BURIED
-------------------------------------------------
The generator's sharing is exact over token IDS; the endpoint takes text. Prompts
are rendered word-per-id so a shared id prefix becomes a shared word-boundary
character prefix, which a BPE tokenizer re-encodes to a shared token prefix in
practice. That is a strong argument, not a proof. The SERVER-side block hit rate
is the ground truth here and the oracle rate is the bound; if the two diverge far
more than eviction explains, suspect the round trip first.

USAGE
-----
    python3 -m bench.run_p4 \\
        --url          http://127.0.0.1:8001/v1/chat/completions \\
        --cache-off-url http://127.0.0.1:8002/v1/chat/completions \\
        --sharing-rates 0,0.25,0.5,0.75,1.0 --rate 8 --duration 60
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
    counter_delta,
    find_crossover,
    fmt_num,
    open_loop_dispatch,
    prompt_text_from_ids,
    render_table,
    require_healthy,
    sample_count,
    samples_percentile,
    scalar,
    scheduler_snapshot,
    specs_from_prompts,
    write_summary,
)
from bench.loadgen import LoadGenConfig, analyze, arrival_schedule
from bench.workloads.generator import (
    SHARING_STRUCTURES,
    LengthSpec,
    Workload,
    WorkloadConfig,
    generate,
)
from serving.metrics.artifact import Provenance

# Counters merged into `/metrics` -> `scheduler` by Scheduler.snapshot(), which
# prefixes the RadixCache snapshot with `cache_`. Deltas only: a lifetime hit
# rate includes warmup and every earlier cell of the sweep.
CACHE_COUNTERS = [
    "cache_lookups",
    "cache_requests_with_a_hit",
    "cache_blocks_reused",
    "cache_blocks_required",
    "cache_partial_block_truncations",
    "cache_tokens_matched_but_recomputed",
    "cache_inserts",
    "cache_evictions",
    "cache_cow_copies",
]
# Gauges: read AFTER the window, not differenced. `shared_blocks_refcount_gt_1`
# is a live count of blocks held by more than one sequence, so a delta of it
# would be meaningless.
CACHE_GAUGES = [
    "cache_cached_blocks",
    "cache_shared_blocks_refcount_gt_1",
    "cache_node_count",
    "cache_max_node_depth",
    "cache_max_partial_hit_depth",
]


@dataclass
class CacheCell:
    """One (structure, sharing rate) measurement, cache on and cache off."""

    structure: str
    sharing_rate: float
    workload: Workload | None = None
    on_art: Any = None
    off_art: Any = None
    delta: dict[str, float] = field(default_factory=dict)
    gauges: dict[str, Any] = field(default_factory=dict)
    verdict: Verdict = field(default_factory=Verdict.ok)
    partial_hit_depth: float = float("nan")
    paths: dict[str, str] = field(default_factory=dict)

    # -- every percentile below comes from RAW SAMPLES, at read time ---------
    def ttft(self, q: float, arm: str = "on") -> float:
        art = self.on_art if arm == "on" else self.off_art
        return samples_percentile(art, "ttft_ms", q) if art is not None else float("nan")

    def ttft_delta(self, q: float) -> float:
        """cache-on minus cache-off. NEGATIVE is the cache helping."""
        return self.ttft(q, "on") - self.ttft(q, "off")

    @property
    def block_hit_rate(self) -> float:
        """blocks_reused / blocks_required over THIS window (methodology §7)."""
        reused = self.delta.get("cache_blocks_reused", float("nan"))
        required = self.delta.get("cache_blocks_required", float("nan"))
        if required != required or required <= 0 or reused != reused:
            return float("nan")
        return reused / required

    @property
    def request_hit_rate(self) -> float:
        hits = self.delta.get("cache_requests_with_a_hit", float("nan"))
        lookups = self.delta.get("cache_lookups", float("nan"))
        if lookups != lookups or lookups <= 0 or hits != hits:
            return float("nan")
        return hits / lookups

    @property
    def oracle_rate(self) -> float:
        if self.workload is None:
            return float("nan")
        s = self.workload.realized.get("sharing", {})
        return float(s.get("realized_block_sharing_rate", float("nan")))

    def as_dict(self) -> dict[str, Any]:
        return {
            "structure": self.structure,
            "sharing_rate": self.sharing_rate,
            "valid": self.verdict.valid,
            "reasons": self.verdict.reasons,
            "oracle_block_sharing_rate": self.oracle_rate,
            "measured_block_hit_rate": self.block_hit_rate,
            "measured_request_hit_rate": self.request_hit_rate,
            "mean_partial_hit_depth_blocks": self.partial_hit_depth,
            "ttft_p50_cache_on": self.ttft(50, "on"),
            "ttft_p99_cache_on": self.ttft(99, "on"),
            "ttft_p50_cache_off": self.ttft(50, "off"),
            "ttft_p99_cache_off": self.ttft(99, "off"),
            "ttft_p50_delta": self.ttft_delta(50),
            "ttft_p99_delta": self.ttft_delta(99),
            "goodput_on": scalar(self.on_art, "goodput_rps") if self.on_art else float("nan"),
            "goodput_off": scalar(self.off_art, "goodput_rps") if self.off_art else float("nan"),
            "cache_counters_delta": self.delta,
            "cache_gauges_after": self.gauges,
            "workload_fingerprint": self.workload.fingerprint() if self.workload else None,
            "degeneracy_warnings": (
                self.workload.degeneracy_warnings if self.workload else []
            ),
            "artifacts": self.paths,
            "n_ttft_on": sample_count(self.on_art, "ttft_ms") if self.on_art else 0,
            "n_ttft_off": sample_count(self.off_art, "ttft_ms") if self.off_art else 0,
        }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def window_mean(
    before: dict[str, Any], after: dict[str, Any], mean_key: str, count_key: str
) -> float:
    """
    Recover a WINDOW mean from two lifetime means and their counts.

    `RadixCache.snapshot()` exposes `mean_partial_hit_depth` over the process's
    whole history, which after a warmup and three earlier sweep cells is not this
    cell's number. Since the count is exposed too, the window mean is recoverable
    exactly:  (mean_a*n_a - mean_b*n_b) / (n_a - n_b).

    Returns NaN when no lookups happened in the window — not 0, which would read
    as "every request matched nothing" rather than "nothing was asked".
    """
    ma, mb = after.get(mean_key), before.get(mean_key)
    na, nb = after.get(count_key), before.get(count_key)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in (ma, mb, na, nb)):
        return float("nan")
    dn = float(na) - float(nb)
    if dn <= 0:
        return float("nan")
    return (float(ma) * float(na) - float(mb) * float(nb)) / dn


def zero_sharing_validity(cell: CacheCell) -> Verdict:
    """
    The control must show no sharing, and it must show it on BOTH sides.

    If the zero-sharing structure produces a non-trivial measured hit rate, then
    either the generator is not producing unique prompts or the cache is matching
    things it should not — and in either case no hit rate anywhere in this run is
    interpretable, so the whole control is INVALID rather than caveated.
    """
    v = Verdict.ok()
    oracle = cell.oracle_rate
    v = v.add(oracle == oracle and oracle > 0.0, (
        f"the zero-sharing CONTROL has a non-zero oracle block sharing rate "
        f"({oracle:.4f}). The generator did not produce the control it was asked "
        "for, so no cache number in this run is interpretable."
    ))
    measured = cell.block_hit_rate
    v = v.add(measured == measured and measured > 0.05, (
        f"the zero-sharing control measured a {measured:.3f} block hit rate. "
        "Unique prompts cannot hit; either the text round trip collapsed distinct "
        "prompts onto a shared token prefix or the cache is matching wrongly."
    ))
    return v


def workload_validity(w: Workload) -> Verdict:
    """
    R14: a workload that silently degenerated measures nothing while looking fine
    in the config. The generator already computes the warnings; refusing to
    publish on them is this driver's job.
    """
    warnings = w.degeneracy_warnings
    if not warnings:
        return Verdict.ok()
    return Verdict(False, [f"WORKLOAD DEGENERACY (R14): {msg}" for msg in warnings])


def overhead_statement(zero_cells: list[CacheCell]) -> list[str]:
    """
    The zero-sharing overhead, stated in words, in the direction it came out.

    Written so the sentence cannot be reused as a win: it names the metric, the
    sign, and the fact that this is the cost of a cache that never helped.
    """
    lines = []
    for c in zero_cells:
        d50, d99 = c.ttft_delta(50), c.ttft_delta(99)
        hit = c.block_hit_rate
        if d50 != d50:
            lines.append(
                "  zero sharing: NOT MEASURED — one of the two arms produced no "
                "TTFT samples, so the cache's cost when it never helps is unknown "
                "for this run. It is not zero; it is unmeasured."
            )
            continue
        sign = "COSTS" if d50 > 0 else "saves"
        lines.append(
            f"  zero sharing: block hit rate {fmt_num(hit, 4)} (expected ~0). "
            f"Turning the cache on {sign} {fmt_num(abs(d50), 2)} ms at TTFT p50 "
            f"and {fmt_num(abs(d99), 2)} ms at p99, for zero benefit."
        )
        if d50 > 0:
            lines.append(
                "    That is the honest floor: radix walk, insert and refcount "
                "bookkeeping on every request with nothing to reuse (§7). It is "
                "published here whether or not it flatters, and if it is material "
                "it is an argument for making the cache adaptive."
            )
    return lines or ["  zero sharing: NOT RUN — which makes this a cache advertisement."]


def build_workload(
    structure: str, rate: float, n_requests: int, args: argparse.Namespace
) -> Workload:
    """
    One workload from `bench/workloads/generator.py`. The four structures and the
    length distributions are the generator's; nothing new is invented here.
    """
    return generate(WorkloadConfig(
        n_requests=n_requests,
        structure=structure,
        sharing_rate=rate,
        block_size=args.block_size,
        seed=args.seed,
        prompt=LengthSpec(dist="lognormal", mean=args.prompt_mean, sigma=args.sigma),
        # Heavy-tailed outputs are P3's business; here output length is a
        # lognormal at a stated mean so the TTFT comparison is not confounded by
        # decode work that differs between the two arms.
        output=LengthSpec(dist="lognormal", mean=args.output_mean, sigma=args.sigma),
        n_shared_prefixes=args.n_shared_prefixes,
        shared_prefix_tokens=args.shared_prefix_tokens,
        prefix_popularity=args.prefix_popularity,
        name=f"p4_{structure}_share{rate:.2f}",
    ))


def cfg_for(
    url: str, arm: str, structure: str, rate: float, args: argparse.Namespace
) -> LoadGenConfig:
    """
    Identical load configuration for both arms. Only the URL differs, which is
    what "matched workload and seed" has to mean if the comparison is to isolate
    the cache rather than the run.
    """
    return LoadGenConfig(
        url=url, rate_rps=args.rate, duration_s=args.duration,
        warmup_s=args.warmup, drain_s=args.drain, seed=args.seed,
        slo_ttft_ms=args.slo_ttft_ms, slo_itl_ms=args.slo_itl_ms,
        request_timeout_s=args.timeout, process="poisson",
        name=f"p4_{structure}_share{rate:.2f}_cache_{arm}",
        extra_config={"cache": arm, "structure": structure, "sharing_rate": rate},
    )


async def run_arm(
    url: str, cfg: LoadGenConfig, workload: Workload
) -> tuple[Any, Verdict, dict[str, Any], dict[str, Any]]:
    """Run one arm against one server, returning (artifact, verdict, before, after)."""
    before = await scheduler_snapshot(url)
    specs = specs_from_prompts(
        cfg,
        [prompt_text_from_ids(r.token_ids) for r in workload.requests],
        [r.max_tokens for r in workload.requests],
    )
    run = await open_loop_dispatch(cfg, specs)
    a = analyze(run)
    after = await scheduler_snapshot(url)

    art = a.artifact
    art.config["workload_structure"] = workload.config.structure
    art.config["requested_sharing_rate"] = workload.config.sharing_rate
    art.realized_workload["prefix_workload"] = workload.artifact_fields()
    art.realized_workload["server_cache_before"] = before
    art.realized_workload["server_cache_after"] = after
    art.notes.append(
        "Prompts were rendered from generator TOKEN IDS one word per id; the "
        "server re-tokenizes them. Realized sharing above is the id-level ORACLE "
        "and is an upper bound on any measured hit rate."
    )
    return art, Verdict(a.validity.valid, list(a.validity.reasons)), before, after


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m bench.run_p4",
        description=(
            "Phase 4 (S4): radix prefix cache. Hit rate and TTFT as a FUNCTION OF "
            "SHARING RATE across all four sharing structures, cache on vs off at "
            "matched workload and seed. Zero sharing is a mandatory control."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions",
                   help="server with the radix prefix cache ENABLED")
    p.add_argument("--cache-off-url", default=None,
                   help="matched server with the cache DISABLED. Without it the "
                        "cache's cost and benefit are both n/a, never 0")
    p.add_argument("--structures", default=",".join(SHARING_STRUCTURES),
                   help="sharing structures to sweep; `zero` is added if omitted "
                        "because it is the control")
    p.add_argument("--sharing-rates", default="0,0.25,0.5,0.75,1.0",
                   help="THE SWEPT PARAMETER (§4: never a single favourable point)")
    p.add_argument("--rate", type=float, default=8.0, help="offered load, req/s")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--warmup", type=float, default=15.0)
    p.add_argument("--drain", type=float, default=10.0)
    p.add_argument("--prompt-mean", type=int, default=512)
    p.add_argument("--output-mean", type=int, default=128)
    p.add_argument("--sigma", type=float, default=0.8)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--n-shared-prefixes", type=int, default=4)
    p.add_argument("--shared-prefix-tokens", type=int, default=128)
    p.add_argument("--prefix-popularity", choices=["uniform", "zipf"], default="uniform")
    p.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    p.add_argument("--slo-itl-ms", type=float, default=100.0)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=20260801)
    p.add_argument("--results-dir", default="results/p4")
    p.add_argument("--label", default="s4")
    return p


def resolve_structures(raw: str) -> list[str]:
    """`zero` first, always. The control is not optional (§7)."""
    want = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in want if s not in SHARING_STRUCTURES]
    if unknown:
        raise SystemExit(
            f"unknown sharing structure(s) {unknown}; known: {list(SHARING_STRUCTURES)}"
        )
    return ["zero"] + [s for s in want if s != "zero"]


async def _amain(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    structures = resolve_structures(args.structures)
    rates = [float(x) for x in args.sharing_rates.split(",") if x.strip()]

    prov = Provenance.capture(repo_root=".", seed=args.seed)
    pub_ok, blockers = banner(prov, "PHASE 4 / S4 — radix prefix cache")

    # ---- 0. pre-flight: healthy, and actually a matched on/off PAIR --------
    print("\n### SERVER PRE-FLIGHT\n")
    await require_healthy(args.url, "cache-on")
    on_snap = await scheduler_snapshot(args.url)
    cache_on = bool(on_snap.get("cache_enabled", "cache_lookups" in on_snap))
    print(f"  cache-on   cache_enabled={cache_on}   {args.url}")
    if not cache_on:
        raise SystemExit(
            "FATAL: the --url server reports no prefix cache. Measuring a cache "
            "against itself would report a null result and look like a careful "
            "experiment."
        )
    if args.cache_off_url:
        await require_healthy(args.cache_off_url, "cache-off")
        off_snap = await scheduler_snapshot(args.cache_off_url)
        cache_off_enabled = bool(off_snap.get("cache_enabled", "cache_lookups" in off_snap))
        print(f"  cache-off  cache_enabled={cache_off_enabled}   {args.cache_off_url}")
        if cache_off_enabled:
            raise SystemExit(
                "FATAL: the --cache-off-url server reports the cache ENABLED. The "
                "two arms are not a matched pair, and the comparison would be a "
                "null result dressed as a measurement."
            )
    else:
        print("  cache-off  NOT PROVIDED — every TTFT delta below is n/a, including")
        print("             the zero-sharing overhead, which is then UNMEASURED, not 0.")

    # ---- 1. sweep ----------------------------------------------------------
    times = arrival_schedule(args.rate, args.warmup + args.duration + args.drain,
                             "poisson", args.seed)
    n_requests = len(times)
    print(f"\n### SWEEP — {len(structures)} structures x {len(rates)} sharing rates, "
          f"{n_requests} requests per cell ({args.duration:g}s steady state)\n")

    cells: list[CacheCell] = []
    for structure in structures:
        # `zero` ignores sharing_rate by construction, so it is run ONCE. Running
        # it five times would pad the table with five identical controls.
        cell_rates = [0.0] if structure == "zero" else rates
        for rate in cell_rates:
            w = build_workload(structure, rate, n_requests, args)
            cell = CacheCell(structure=structure, sharing_rate=rate, workload=w)
            cell.verdict = workload_validity(w)

            art_on, v_on, before, after = await run_arm(
                args.url, cfg_for(args.url, "on", structure, rate, args), w
            )
            cell.on_art = art_on
            cell.delta = counter_delta(before, after, CACHE_COUNTERS)
            cell.gauges = {k: after.get(k) for k in CACHE_GAUGES}
            cell.partial_hit_depth = window_mean(
                before, after, "cache_mean_partial_hit_depth", "cache_lookups"
            )
            art_on.set_scalar("cache_hit_rate", cell.block_hit_rate)
            art_on.set_scalar("cache_evictions", cell.delta.get("cache_evictions", float("nan")))
            art_on.set_scalar(
                "shared_blocks",
                float(cell.gauges.get("cache_shared_blocks_refcount_gt_1") or float("nan")),
            )
            art_on.set_scalar("partial_hit_depth_blocks", cell.partial_hit_depth)
            cell.verdict = cell.verdict.merge(v_on)
            cell.paths["cache_on"] = str(art_on.write(args.results_dir))

            if args.cache_off_url:
                art_off, v_off, _, _ = await run_arm(
                    args.cache_off_url,
                    cfg_for(args.cache_off_url, "off", structure, rate, args),
                    w,
                )
                cell.off_art = art_off
                cell.verdict = cell.verdict.merge(v_off)
                if structure == "zero":
                    art_off.set_scalar("cache_overhead_ttft_ms", -cell.ttft_delta(50))
                cell.paths["cache_off"] = str(art_off.write(args.results_dir))

            if structure == "zero":
                cell.verdict = cell.verdict.merge(zero_sharing_validity(cell))

            cells.append(cell)
            print(f"  {structure:15} share {rate:4.2f}  oracle "
                  f"{fmt_num(cell.oracle_rate, 3):>6}  measured "
                  f"{fmt_num(cell.block_hit_rate, 3):>6}  ttft_p50 "
                  f"{fmt_num(cell.ttft(50), 1):>9}  delta "
                  f"{fmt_num(cell.ttft_delta(50), 1):>9}  {cell.verdict.label}")

    # ---- 2. the control, first and on its own ------------------------------
    zero_cells = [c for c in cells if c.structure == "zero"]
    print("\n### WHAT THE CACHE COSTS WHEN IT NEVER HELPS (the mandatory control, §7)\n")
    for line in overhead_statement(zero_cells):
        print(line)

    # ---- 3. hit rate and TTFT as a function of sharing rate ----------------
    print("\n### HIT RATE AND TTFT vs SHARING RATE, BY STRUCTURE\n")
    headers = ["structure", "share", "oracle", "hit_rate", "req_hits", "depth",
               "evict", "cow", "shared>1", "ttft_p50", "ttft_p99", "d_p50", "d_p99",
               "n", "valid"]
    rows: list[list[Any]] = []
    for c in cells:
        rows.append([
            c.structure, c.sharing_rate, c.oracle_rate, c.block_hit_rate,
            c.request_hit_rate, c.partial_hit_depth,
            c.delta.get("cache_evictions"), c.delta.get("cache_cow_copies"),
            c.gauges.get("cache_shared_blocks_refcount_gt_1"),
            c.ttft(50), c.ttft(99), c.ttft_delta(50), c.ttft_delta(99),
            sample_count(c.on_art, "ttft_ms") if c.on_art else 0,
            c.verdict.label,
        ])
    print(render_table(headers, rows,
                       precisions=[0, 2, 3, 3, 3, 2, 0, 0, 0, 1, 1, 1, 1, 0, 0]))
    print("\n  oracle    = generator's infinite-cache block sharing rate, the UPPER BOUND.")
    print("  hit_rate  = server blocks_reused / blocks_required over THIS window only.")
    print("  depth     = mean partial-hit depth in blocks, over this window.")
    print("  d_p50/99  = cache-on minus cache-off TTFT; NEGATIVE is the cache helping.")
    print("  Percentiles come from the artifacts' raw ttft_ms samples at print time,")
    print("  never from a stored scalar (methodology §5, R15).")
    print("\n  A hit rate is meaningless without the structure beside it (§7): the")
    print("  conversational rows are the MOST FAVOURABLE shape for a radix cache and")
    print("  their numbers are not comparable to the system-prompt rows.")

    # ---- 4. the break-even sharing rate, per structure ---------------------
    print("\n### BREAK-EVEN SHARING RATE — where the cache starts paying for itself\n")
    breakevens: dict[str, Any] = {}
    for structure in structures:
        if structure == "zero":
            continue
        pts = [(c.sharing_rate, c.ttft_delta(50)) for c in cells
               if c.structure == structure and c.verdict.valid]
        pts.sort()
        # advantage = cost, i.e. positive while the cache is still LOSING; the
        # crossover is the sharing rate at which it stops losing.
        crossing = find_crossover([x for x, _ in pts], [d for _, d in pts])
        breakevens[structure] = crossing
        print(f"  {structure:15} {crossing.get('reason')}")
    print("\n  Read this as a curve, not a point. A TTFT improvement quoted at one")
    print("  sharing rate is a choice of workload; the rate at which the sign flips is")
    print("  the number that survives a follow-up question.")

    # ---- 5. validity -------------------------------------------------------
    invalid = [c for c in cells if not c.verdict.valid]
    if invalid:
        print(f"\n!! {len(invalid)} cell(s) INVALID and excluded from every claim:")
        for c in invalid:
            print(f"   {c.structure} share={c.sharing_rate}:")
            for r in c.verdict.reasons:
                print(f"      - {r}")
    if not pub_ok:
        print(f"\n!! NOT PUBLISHABLE: {blockers}")

    path = write_summary(args.results_dir, f"p4_{args.label}_summary.json", {
        "cells": [c.as_dict() for c in cells],
        "breakeven_sharing_rate": breakevens,
        "zero_sharing_overhead_statement": overhead_statement(zero_cells),
        "structures": structures,
        "sharing_rates": rates,
        "offered_load_rps": args.rate,
        "publishable": pub_ok,
        "blockers": blockers,
        "allocation_id": prov.allocation_id,
    })
    print(f"\nsummary -> {path}")
    return 2 if (invalid or not cells) else 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    sys.exit(main())
