"""
Phase 3 benchmark driver: preemption under forced memory pressure (claim S6).

WHAT THIS MEASURES
------------------
Recompute versus swap, head to head, at matched conditions, SWEPT OVER SEQUENCE
LENGTH. The sweep is the whole point. Either policy can be made to look good at
one length; the question the phase plan asks (§6, "swept over sequence length to
find (or fail to find) a crossover") is whether a crossover exists at all for
this model, and "no crossover in the swept range" is an answer, not a failure.

Reported per (policy, length): preemption rate, the latency tax against an
unpreempted control, tokens recomputed, bytes swapped, and resume latency in both
seconds and scheduler steps.

FORCED PRESSURE, AND WHY A ZERO IS FATAL HERE
---------------------------------------------
Preemption only fires when the KV pool cannot hold the running set. That is a
property of how the SERVER was launched — a small `--num-blocks` pool — not of
anything this script can set over HTTP. So the script cannot make preemption
happen; it can only refuse to pretend it did.

A run whose server-side preemption counter did not move measured a system that
never preempted. Its latency numbers are real and its preemption numbers are
vacuous, and the dangerous version of that outcome is not an error but a table
full of zeros that reads as "preemption is free". Therefore: `preemptions_total`
delta <= 0 marks the run INVALID, it is excluded from every claim, and the driver
exits non-zero. Same for a swap arm that silently degraded to recompute because
host swap space ran out — that arm is no longer the policy it says it is.

THE PREDICTION IS ON RECORD, AND IT IS PRINTED BEFORE THE RESULT
----------------------------------------------------------------
docs/ARCHITECTURE.md §5.2 predicts, in advance, that for Llama 3.2 1B
(32 KB KV/token) recompute wins at nearly all lengths, and §9.2 adds that a radix
cache makes recompute cheaper still because the victim's own prefix may still be
resident. This script prints that prediction BEFORE any measured number and
states plainly whether it held. A wrong prediction is a finding.

USAGE
-----
    python3 -m bench.run_p3 \\
        --recompute-url http://127.0.0.1:8001/v1/chat/completions \\
        --swap-url      http://127.0.0.1:8002/v1/chat/completions \\
        --control-url   http://127.0.0.1:8003/v1/chat/completions \\
        --lengths 128,512,1024,2048,4096 --rate 8 --duration 60
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
    get_json,
    render_table,
    require_healthy,
    sample_count,
    samples_percentile,
    scalar,
    scheduler_snapshot,
    write_summary,
)
from bench.loadgen import LoadGenConfig, analyze, run_load
from serving.metrics.artifact import Provenance

# ---------------------------------------------------------------------------
# THE PREDICTION, QUOTED FROM docs/ARCHITECTURE.md §5.2 / §9.2, PRINTED FIRST
# ---------------------------------------------------------------------------

PREDICTION = """\
docs/ARCHITECTURE.md §5.2, written before any of this was measured:

  "For Llama 3.2 1B specifically, I expect RECOMPUTE TO WIN AT NEARLY ALL
   LENGTHS, because the model is tiny - prefill is cheap and KV per token is
   small (8 kv_heads x 64 dim x 16 layers x 2 x 2 bytes = 32 KB/token), so
   neither side is under pressure."

  Mechanism, from §5.2 and §9.2:
    recompute cost grows with sequence length as PREFILL COMPUTE, paid later,
      and frees 100% of the victim's blocks immediately;
    swap cost grows with the same length as 2x PCIe TRANSFER, and consumes
      pinned host memory;
    and §9.2 step 5: with a radix cache present, "re-prefill" often means
      re-prefilling only the GENERATED TAIL, because the victim's own prefix
      blocks may still be resident - which makes recompute cheaper than the
      naive analysis suggests.

  Falsifiable form used below:
    P1  recompute's E2E p99 is lower than swap's at EVERY swept length, i.e.
        the crossover is absent from the swept range;
    P2  if a crossover exists, it lies at a LONG sequence length, and swap wins
        only above it.

  A wrong prediction is a finding. It is printed here, before the numbers,
  precisely so it cannot be quietly reinterpreted afterwards."""

POLICY_RECOMPUTE = "recompute"
POLICY_SWAP = "swap"

# Counters read from `/metrics` -> `scheduler` (Scheduler.snapshot() merges
# PreemptionStats.as_dict()). Deltas across the run window, never lifetime
# totals: a server that already preempted during warmup would otherwise credit
# those events to this run.
PREEMPTION_COUNTERS = [
    "step",
    "preemptions_total",
    "tokens_recomputed",
    "bytes_swapped_out",
    "bytes_swapped_in",
    "resumes",
    "starvation_fallbacks",
    "swap_space_exhausted",
]


@dataclass
class ArmResult:
    """One (policy, length) cell, or a control cell when `policy == 'control'`."""

    policy: str
    length: int
    artifact: Any = None
    delta: dict[str, float] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    verdict: Verdict = field(default_factory=Verdict.ok)
    artifact_path: str | None = None

    # -- percentiles, computed from RAW SAMPLES at read time --------------
    def pct(self, metric: str, q: float) -> float:
        if self.artifact is None:
            return float("nan")
        return samples_percentile(self.artifact, metric, q)

    @property
    def preemption_rate(self) -> float:
        """Preemption events / decode iterations (REGISTRY: preemption_rate)."""
        steps = self.delta.get("step", float("nan"))
        total = self.delta.get("preemptions_total", float("nan"))
        if not steps or steps != steps or total != total:
            return float("nan")
        return total / steps

    @property
    def bytes_swapped(self) -> float:
        out = self.delta.get("bytes_swapped_out", float("nan"))
        back = self.delta.get("bytes_swapped_in", float("nan"))
        return out + back

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "length": self.length,
            "valid": self.verdict.valid,
            "reasons": self.verdict.reasons,
            "preemption_rate": self.preemption_rate,
            "delta": self.delta,
            "ttft_p50": self.pct("ttft_ms", 50),
            "ttft_p99": self.pct("ttft_ms", 99),
            "e2e_p50": self.pct("e2e_ms", 50),
            "e2e_p99": self.pct("e2e_ms", 99),
            "n_ttft_samples": sample_count(self.artifact, "ttft_ms") if self.artifact else 0,
            "goodput_rps": scalar(self.artifact, "goodput_rps") if self.artifact else float("nan"),
            "artifact": self.artifact_path,
            "resume_seconds_mean": self.after.get("resume_seconds_mean"),
            "resume_seconds_max": self.after.get("resume_seconds_max"),
            "resume_steps_mean": self.after.get("resume_steps_mean"),
            "resume_steps_max": self.after.get("resume_steps_max"),
        }


# ---------------------------------------------------------------------------
# Validity — the part that refuses to publish a silent zero
# ---------------------------------------------------------------------------


def preemption_validity(policy: str, delta: dict[str, float]) -> Verdict:
    """
    Is this arm a measurement of preemption at all?

    The first clause is the one that matters. Preemption is the SUBJECT of this
    benchmark, so a run in which it never fired has no preemption content, and a
    row of zeros in a results table reads as "preemption costs nothing" rather
    than as "preemption did not happen". INVALID, excluded, non-zero exit.
    """
    v = Verdict.ok()
    total = delta.get("preemptions_total", float("nan"))
    if total != total:  # NaN: the counter was never reported
        return v.add(True, (
            "server did not report `preemptions_total`; this build cannot "
            "substantiate any preemption claim"
        ))
    v = v.add(total <= 0, (
        f"ZERO PREEMPTIONS ({total:.0f}) during the measurement window. Preemption "
        "is the subject of this benchmark, so this run measured a system that "
        "never preempted. Shrink the server's KV pool or raise offered load; the "
        "numbers here must not be read as 'preemption is cheap'."
    ))

    recomputed = delta.get("tokens_recomputed", float("nan"))
    swapped = delta.get("bytes_swapped_out", float("nan"))
    if policy == POLICY_SWAP:
        v = v.add(delta.get("swap_space_exhausted", 0) > 0, (
            f"host swap space ran out {delta.get('swap_space_exhausted', 0):.0f} "
            "time(s) and the policy DEGRADED TO RECOMPUTE mid-run. The fallback is "
            "correct behaviour and fatal to a head-to-head: this arm is no longer "
            "the policy it is labelled with."
        ))
        v = v.add(recomputed == recomputed and recomputed > 0 and total > 0, (
            f"swap arm recomputed {recomputed:.0f} tokens; a pure swap policy "
            "recomputes nothing. The arm is a mixture, not a policy."
        ))
    if policy == POLICY_RECOMPUTE:
        v = v.add(swapped == swapped and swapped > 0, (
            f"recompute arm moved {swapped:.0f} swap bytes; a pure recompute "
            "policy transfers nothing. The arm is a mixture, not a policy."
        ))

    fallbacks = delta.get("starvation_fallbacks", 0)
    if fallbacks == fallbacks and fallbacks > 0:
        # R40 / ARCHITECTURE §5.2 step 3: this is an ADMISSION-CONTROL BUG report,
        # not a measurement. Surfaced rather than absorbed, but it does not by
        # itself invalidate the latency numbers, so it is a loud note.
        v = Verdict(v.valid, v.reasons + [
            f"ADMISSION-CONTROL ALARM: the starvation fallback fired "
            f"{fallbacks:.0f} time(s). Per ARCHITECTURE §5.2 that means the "
            "watermark admitted work the running set could not step, which is a "
            "bug in admission, not evidence that the guard works."
        ])
        v = Verdict(False, v.reasons)
    return v


def control_validity(delta: dict[str, float], control_blocks: float, arm_blocks: float) -> Verdict:
    """
    A control that preempted is not a control.

    The latency tax is a DIFFERENCE against an unpreempted run at a batch size
    that fits. If the control preempted too, the difference measures nothing and
    would understate the tax — in the flattering direction, which is the one that
    matters.
    """
    v = Verdict.ok()
    total = delta.get("preemptions_total", float("nan"))
    v = v.add(total == total and total > 0, (
        f"the CONTROL preempted {total:.0f} time(s). A control must not preempt, "
        "or the latency tax it anchors is a difference between two preempted runs "
        "and understates the tax."
    ))
    if control_blocks == control_blocks and arm_blocks == arm_blocks:
        v = v.add(control_blocks <= arm_blocks, (
            f"the control server's KV pool ({control_blocks:.0f} blocks) is not "
            f"larger than the pressure server's ({arm_blocks:.0f}); the two "
            "configurations are not a pressure/no-pressure pair."
        ))
    return v


def latency_tax(arm: ArmResult, control: ArmResult | None, metric: str, q: float) -> float:
    """
    Arm minus control, both percentiles computed from RAW SAMPLES here.

    NaN when there is no control, or the control is invalid, or either side has
    no samples. NaN renders as `n/a`; it never renders as 0.0, because "no tax"
    and "no control" are different statements.
    """
    if control is None or not control.verdict.valid:
        return float("nan")
    return arm.pct(metric, q) - control.pct(metric, q)


def prediction_verdict(
    lengths: list[int],
    recompute_p99: list[float],
    swap_p99: list[float],
    crossover: dict[str, Any],
) -> list[str]:
    """
    Did §5.2's prediction hold? Stated plainly, in the same units as the claim.

    `advantage = swap - recompute`, so positive means recompute is ahead (lower
    E2E p99). The prediction survives iff recompute is ahead at every swept
    length; a crossover inside the range falsifies P1 and is reported as such.
    """
    pairs = [(x, s - r) for x, r, s in zip(lengths, recompute_p99, swap_p99, strict=False)
             if r == r and s == s]
    if len(pairs) < 2:
        return [
            "PREDICTION VERDICT: UNTESTED. Fewer than two lengths produced a "
            "usable recompute/swap pair, so neither P1 nor P2 can be evaluated. "
            "This is not evidence for the prediction.",
        ]
    ahead = [x for x, adv in pairs if adv > 0]
    behind = [x for x, adv in pairs if adv <= 0]
    lines = []
    if not behind:
        lines.append(
            f"PREDICTION VERDICT: P1 HELD over the swept range. Recompute has the "
            f"lower E2E p99 at all {len(pairs)} lengths tested "
            f"({min(ahead)}..{max(ahead)} prompt tokens). No crossover exists "
            "inside the range; the sweep cannot say one does not exist beyond it."
        )
    elif not ahead:
        lines.append(
            f"PREDICTION VERDICT: P1 DID NOT HOLD, and it failed everywhere. Swap "
            f"has the lower E2E p99 at all {len(pairs)} lengths tested. The §5.2 "
            "reasoning — tiny model, cheap prefill, 32 KB/token KV — is wrong for "
            "this system as measured, and the measurement, not the prediction, is "
            "the result."
        )
    else:
        lines.append(
            f"PREDICTION VERDICT: P1 DID NOT HOLD. Recompute wins at "
            f"{sorted(ahead)} and loses at {sorted(behind)}: a crossover exists "
            "INSIDE the swept range, which §5.2 did not expect for a 1B model."
        )
        lines.append(
            "  P2 (crossover exists but only at long lengths) is "
            + ("CONSISTENT with this: " if min(behind) > max(ahead) else "NOT supported: ")
            + f"recompute wins up to {max(ahead)} and loses from {min(behind)}."
        )
    lines.append(f"  crossover: {crossover.get('reason')}")
    return lines


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


async def _pool_blocks(url: str) -> float:
    """KV pool size in blocks, so 'forced pressure' is a number and not a claim."""
    try:
        m = await get_json(url.rsplit("/v1/", 1)[0] + "/metrics")
    except Exception:  # noqa: BLE001 — absence is data; it renders as n/a
        return float("nan")
    alloc = m.get("allocator") or {}
    v = alloc.get("num_blocks")
    return float(v) if isinstance(v, (int, float)) else float("nan")


async def run_arm(
    url: str,
    policy: str,
    length: int,
    args: argparse.Namespace,
    rate: float,
) -> ArmResult:
    """
    One (policy, length) cell: snapshot the server counters, run open loop,
    snapshot again, and attribute only the delta to this run.
    """
    before = await scheduler_snapshot(url)
    cfg = LoadGenConfig(
        url=url,
        rate_rps=rate,
        duration_s=args.duration,
        warmup_s=args.warmup,
        drain_s=args.drain,
        seed=args.seed,
        # Length is the INDEPENDENT VARIABLE of this sweep, so its variance is
        # removed on purpose (sigma=0). Methodology §4 requires skewed lengths
        # for a headline workload; this is not one — it is a controlled sweep,
        # and saying so is the difference between a control and a convenience.
        prompt_mean_tokens=length,
        prompt_sigma=0.0,
        output_mean_tokens=args.output_tokens,
        output_sigma=0.0,
        output_max_tokens=args.output_tokens,
        slo_ttft_ms=args.slo_ttft_ms,
        slo_itl_ms=args.slo_itl_ms,
        process="poisson",
        request_timeout_s=args.timeout,
        name=f"p3_{policy}_len{length}",
    )
    run = await run_load(cfg)
    a = analyze(run)
    after = await scheduler_snapshot(url)
    delta = counter_delta(before, after, PREEMPTION_COUNTERS)

    art = a.artifact
    art.config["arm"] = policy
    art.config["sequence_length_tokens"] = length
    art.config["length_variance"] = "sigma=0 — length is the swept variable"
    art.realized_workload["server_preemption_delta"] = delta
    art.realized_workload["server_scheduler_after"] = after

    harness = Verdict(a.validity.valid, list(a.validity.reasons))
    # The control's own check needs the two pool sizes, which only the caller
    # knows, so it is applied there; here the control carries the harness verdict
    # alone rather than a half-applied one.
    verdict = harness if policy == "control" else harness.merge(
        preemption_validity(policy, delta)
    )
    if not verdict.valid:
        art.notes.insert(0, "RUN INVALID: " + " | ".join(verdict.reasons))

    if delta.get("preemptions_total", float("nan")) == delta.get("preemptions_total", float("nan")):
        art.set_scalar("preemption_rate", _safe_rate(delta))
        art.set_scalar("tokens_recomputed", delta.get("tokens_recomputed", float("nan")))
        art.set_scalar(
            "bytes_swapped",
            delta.get("bytes_swapped_out", 0.0) + delta.get("bytes_swapped_in", 0.0),
        )
    path = art.write(args.results_dir)

    return ArmResult(
        policy=policy, length=length, artifact=art, delta=delta, after=after,
        verdict=verdict, artifact_path=str(path),
    )


def _safe_rate(delta: dict[str, float]) -> float:
    steps = delta.get("step", float("nan"))
    total = delta.get("preemptions_total", float("nan"))
    if steps != steps or total != total or steps <= 0:
        return float("nan")
    return total / steps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m bench.run_p3",
        description=(
            "Phase 3 (S6): preemption under FORCED memory pressure. Recompute vs "
            "swap head-to-head, swept over sequence length, against an "
            "unpreempted control. A run with zero preemptions is INVALID."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--recompute-url", default="http://127.0.0.1:8001/v1/chat/completions",
                   help="server launched with preemption_policy=recompute and a SMALL KV pool")
    p.add_argument("--swap-url", default=None,
                   help="server launched with preemption_policy=swap and the SAME small pool; "
                        "omit to run the recompute arm alone (the comparison is then NOT "
                        "MEASURED, and is reported as such rather than skipped silently)")
    p.add_argument("--control-url", default=None,
                   help="server with a KV pool that FITS the batch, so it never preempts. "
                        "Without it the latency tax is n/a, never 0")
    p.add_argument("--lengths", default="128,512,1024,2048,4096",
                   help="prompt lengths in tokens — THE SWEPT VARIABLE")
    p.add_argument("--rate", type=float, default=8.0, help="offered load, req/s")
    p.add_argument("--control-rate", type=float, default=None,
                   help="offered load for the control (default: same as --rate, which is "
                        "what 'matched conditions' means)")
    p.add_argument("--duration", type=float, default=60.0, help="steady-state seconds per cell")
    p.add_argument("--warmup", type=float, default=15.0)
    p.add_argument("--drain", type=float, default=10.0)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--slo-ttft-ms", type=float, default=2000.0,
                   help="carried into the artifact for goodput; the FROZEN P2 SLO belongs here")
    p.add_argument("--slo-itl-ms", type=float, default=100.0)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=20260801)
    p.add_argument("--results-dir", default="results/p3")
    p.add_argument("--label", default="s6")
    return p


async def _amain(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    control_rate = args.control_rate if args.control_rate is not None else args.rate

    prov = Provenance.capture(repo_root=".", seed=args.seed)
    pub_ok, blockers = banner(prov, "PHASE 3 / S6 — preemption under forced memory pressure")

    # ---- 0. the prediction, before any measurement --------------------------
    print("\n### PREDICTION ON RECORD (printed BEFORE the result)\n")
    print(PREDICTION)

    # ---- 1. refuse to benchmark an unhealthy or mis-configured server -------
    arms: list[tuple[str, str]] = [(POLICY_RECOMPUTE, args.recompute_url)]
    if args.swap_url:
        arms.append((POLICY_SWAP, args.swap_url))
    print("\n### SERVER PRE-FLIGHT\n")
    pool: dict[str, float] = {}
    for policy, url in arms:
        await require_healthy(url, policy)
        snap = await scheduler_snapshot(url)
        reported = str(snap.get("preemption_policy", "")).lower()
        if reported and policy not in reported:
            raise SystemExit(
                f"FATAL: the {policy!r} arm points at a server reporting "
                f"preemption_policy={reported!r}. Benchmarking one policy while "
                "labelling it another is the worst error this driver can make; "
                "it is fatal rather than a warning."
            )
        pool[policy] = await _pool_blocks(url)
        print(f"  {policy:10} pool {fmt_num(pool[policy], 0)} blocks   "
              f"policy={reported or 'unreported'}   {url}")
    if args.control_url:
        await require_healthy(args.control_url, "control")
        pool["control"] = await _pool_blocks(args.control_url)
        print(f"  {'control':10} pool {fmt_num(pool['control'], 0)} blocks   {args.control_url}")
    else:
        print("  control    NOT PROVIDED — the latency tax will be reported as n/a. "
              "A tax without an unpreempted reference is not a number.")

    # ---- 2. sweep ----------------------------------------------------------
    controls: dict[int, ArmResult] = {}
    results: list[ArmResult] = []
    print(f"\n### SWEEP — {len(lengths)} lengths x {len(arms)} policies"
          f"{' + control' if args.control_url else ''}, "
          f"{args.duration:g}s steady state each\n")

    for length in lengths:
        if args.control_url:
            c = await run_arm(args.control_url, "control", length, args, control_rate)
            c.verdict = c.verdict.merge(
                control_validity(c.delta, pool.get("control", float("nan")),
                                 pool.get(POLICY_RECOMPUTE, float("nan")))
            )
            controls[length] = c
            print(f"  len {length:6d}  control    preemptions "
                  f"{fmt_num(c.delta.get('preemptions_total'), 0):>6}  "
                  f"e2e_p99 {fmt_num(c.pct('e2e_ms', 99)):>10}  {c.verdict.label}")
        for policy, url in arms:
            r = await run_arm(url, policy, length, args, args.rate)
            results.append(r)
            print(f"  len {length:6d}  {policy:10} preemptions "
                  f"{fmt_num(r.delta.get('preemptions_total'), 0):>6}  "
                  f"rate {fmt_num(r.preemption_rate, 4):>8}  "
                  f"e2e_p99 {fmt_num(r.pct('e2e_ms', 99)):>10}  {r.verdict.label}")

    # ---- 3. tables ---------------------------------------------------------
    print("\n### PREEMPTION COST BY POLICY AND SEQUENCE LENGTH\n")
    headers = ["length", "policy", "preempt", "rate", "tok_recomp", "MiB_swapped",
               "resume_ms", "resume_steps", "n"]
    rows: list[list[Any]] = []
    for r in results:
        rows.append([
            r.length, r.policy,
            r.delta.get("preemptions_total"), r.preemption_rate,
            r.delta.get("tokens_recomputed"),
            r.bytes_swapped / (1024 * 1024) if r.bytes_swapped == r.bytes_swapped else None,
            (r.after.get("resume_seconds_mean") or float("nan")) * 1e3,
            r.after.get("resume_steps_mean"),
            sample_count(r.artifact, "ttft_ms") if r.artifact else 0,
        ])
    print(render_table(headers, rows, precisions=[0, 0, 0, 5, 0, 2, 2, 2, 0]))

    print("\n### LATENCY TAX vs THE UNPREEMPTED CONTROL (ms; positive = preemption cost)\n")
    tax_headers = ["length", "policy", "ttft_p50", "ttft_p99", "e2e_p50", "e2e_p99",
                   "goodput", "valid"]
    tax_rows: list[list[Any]] = []
    for r in results:
        c = controls.get(r.length)
        tax_rows.append([
            r.length, r.policy,
            latency_tax(r, c, "ttft_ms", 50), latency_tax(r, c, "ttft_ms", 99),
            latency_tax(r, c, "e2e_ms", 50), latency_tax(r, c, "e2e_ms", 99),
            scalar(r.artifact, "goodput_rps") if r.artifact else None,
            r.verdict.label,
        ])
    print(render_table(tax_headers, tax_rows, precisions=[0, 0, 1, 1, 1, 1, 2, 0]))
    print("\n  Percentiles above are computed from the artifacts' RAW ttft_ms / e2e_ms")
    print("  samples at print time (methodology §5, R15). No scalar shortcut exists in")
    print("  this driver, which is why a missing measurement prints n/a and not 0.0.")

    # ---- 4. the crossover, which is the actual deliverable -----------------
    valid = [r for r in results if r.verdict.valid]
    rec = {r.length: r.pct("e2e_ms", 99) for r in valid if r.policy == POLICY_RECOMPUTE}
    swp = {r.length: r.pct("e2e_ms", 99) for r in valid if r.policy == POLICY_SWAP}
    common = sorted(set(rec) & set(swp))
    advantage = [swp[x] - rec[x] for x in common]      # > 0 means recompute ahead
    crossover = find_crossover([float(x) for x in common], advantage)

    print("\n### CROSSOVER — recompute vs swap over sequence length\n")
    if not common:
        print("  NOT MEASURED: no length produced a valid recompute AND a valid swap arm.")
        print("  Without both arms at a matched length there is no head-to-head, and the")
        print("  §5.2 prediction is untested rather than confirmed.")
    else:
        print(render_table(
            ["length", "recompute_e2e_p99", "swap_e2e_p99", "advantage(swap-recompute)"],
            [[x, rec[x], swp[x], swp[x] - rec[x]] for x in common],
            precisions=[0, 1, 1, 1],
        ))
        print(f"\n  crossover: {crossover.get('reason')}")

    print("")
    for line in prediction_verdict(
        common, [rec[x] for x in common], [swp[x] for x in common], crossover
    ):
        print("  " + line)

    if not args.swap_url:
        print("\n  !! SWAP ARM NOT MEASURED (no --swap-url). The head-to-head that S6")
        print("     claims does not exist in this run, and the phase claim degrades to")
        print("     'implemented preemption with recompute' until it does.")

    # ---- 5. what may not back a claim --------------------------------------
    invalid = [r for r in results if not r.verdict.valid] + \
              [c for c in controls.values() if not c.verdict.valid]
    if invalid:
        print(f"\n!! {len(invalid)} run(s) INVALID and excluded from every claim:")
        for r in invalid:
            print(f"   {r.policy} len={r.length}:")
            for reason in r.verdict.reasons:
                print(f"      - {reason}")
    if not pub_ok:
        print(f"\n!! NOT PUBLISHABLE: {blockers}")

    path = write_summary(args.results_dir, f"p3_{args.label}_summary.json", {
        "prediction": PREDICTION,
        "prediction_verdict": prediction_verdict(
            common, [rec[x] for x in common], [swp[x] for x in common], crossover),
        "crossover": crossover,
        "pool_blocks": pool,
        "arms": [r.as_dict() for r in results],
        "controls": [c.as_dict() for c in controls.values()],
        "publishable": pub_ok,
        "blockers": blockers,
        "allocation_id": prov.allocation_id,
        "offered_load_rps": args.rate,
        "lengths": lengths,
    })
    print(f"\nsummary -> {path}")
    return 2 if (invalid or not results) else 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    sys.exit(main())
