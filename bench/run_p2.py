"""
Phase 2 benchmark driver: calibrate the SLO, freeze it, then sweep offered load.

THE ORDER IS THE POINT
----------------------
An SLO chosen after seeing results is not an SLO. So this script:

  1. measures UNLOADED batch-1 performance against the live server,
  2. derives the SLO from it by a multiplier DECLARED BELOW, in code, before any
     loaded run happens,
  3. prints and freezes that SLO,
  4. only then sweeps offered load.

Everything runs inside ONE Slurm allocation, because a comparison spanning two
allocations measures node assignment rather than the change under test — the
engine's own throughput moved ~79 -> ~60 tok/s across nodes for identical code
(R12, `BENCHMARKS.md:17`).

WHAT IS AND IS NOT MEASURED HERE
--------------------------------
Measured: goodput under a frozen SLO as a function of offered load, against the
continuous-batching scheduler, open loop, with the coordinated-omission guard
active (R1).

NOT measured here: the B2 static-batching baseline. That needs a second server
configuration and is run by the same job, separately, so both land in one
allocation and `assert_comparable` will accept them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

import httpx

from bench.loadgen import (
    LoadGenConfig,
    analyze,
    build_schedule,
    percentile,
    render,
    run_load,
    stream_one,
)
from serving.metrics.artifact import Provenance

# ---------------------------------------------------------------------------
# THE SLO, DECLARED BEFORE ANY MEASUREMENT
# ---------------------------------------------------------------------------
#
# Anchored to physical unloaded performance rather than to a round number, so it
# stays meaningful on a different GPU. Both multipliers are chosen here, in
# advance, and are reported alongside every result.
#
# TTFT budget = 10x unloaded TTFT.
#   Unloaded TTFT is one prefill with an empty queue. Under load a request also
#   waits for admission and shares each step with the rest of the batch, so a
#   1x budget would be unmeetable at any concurrency and a 100x budget would be
#   met by a system that is obviously too slow. 10x says: queueing may dominate
#   service time by up to an order of magnitude before we call it a violation.
#
# ITL budget = 3x unloaded p50 ITL.
#   Inter-token latency is what a streaming user perceives as smoothness. At
#   batch B every decode step serves B sequences, so per-sequence ITL rises
#   roughly with batch size; 3x permits real batching while still failing a
#   system that has become a slideshow.
#
# p95 rather than mean ITL within a request, because a single long stall is what
# users notice and a mean hides it.
SLO_TTFT_MULTIPLIER = 10.0
SLO_ITL_MULTIPLIER = 3.0


async def calibrate(url: str, prompt_tokens: int, max_tokens: int, n: int) -> dict:
    """
    Unloaded batch-1 reference. CLOSED LOOP, and labelled as such.

    Closed loop is CORRECT here and only here: the goal is service time with an
    empty queue, so exactly one request must be in flight at a time. Using it for
    a loaded measurement is the mistake this project refuses to make (R1); using
    it for calibration is the only way to get a clean unloaded number.
    """
    # Reuse the harness's own schedule builder so the calibration prompt is
    # constructed exactly like a benchmark prompt. Hand-rolling a RequestSpec
    # here would calibrate against a different workload than the one measured.
    cfg = LoadGenConfig(
        url=url, rate_rps=float(n), duration_s=float(n), warmup_s=0.0, drain_s=0.0,
        prompt_mean_tokens=prompt_tokens, prompt_sigma=0.0,
        output_mean_tokens=max_tokens, output_sigma=0.0,
        output_max_tokens=max_tokens, process="deterministic", name="calibration",
    )
    specs = build_schedule(cfg)[:n]
    if not specs:
        raise SystemExit("calibration produced no requests; check rate/duration")

    ttfts, itls = [], []
    async with httpx.AsyncClient(timeout=cfg.request_timeout_s) as client:
        for i, spec in enumerate(specs):
            # t0 is set per request and the schedule is not honoured: each request
            # is sent only after the previous one completes. Intended == actual by
            # construction, so there is no drift to guard against, and the number
            # this produces is service time with an empty queue — which is the
            # only thing the SLO should be anchored to.
            # t0 is the run's ZERO POINT, and stream_one measures latency from
            # `t0 + spec.intended_send_time`. build_schedule spreads intended
            # times across the schedule, so passing a fresh "now" as t0 places
            # every intended dispatch in the FUTURE and yields NEGATIVE latency.
            # That is what job 11599377 did: TTFT p50 came out at -402 ms, the
            # derived SLO became "TTFT < -4023 ms", no request could satisfy it,
            # and goodput was 0.00 at every rate.
            #
            # Anchoring t0 so that `t0 + intended == now` makes intended == actual,
            # which is precisely the property claimed for calibration.
            t0 = asyncio.get_running_loop().time() - spec.intended_send_time
            res = await stream_one(client, spec, cfg, t0)
            if res.ttft_ms is None:
                raise SystemExit(f"calibration request {i} produced no first token: {res.error}")
            ttfts.append(res.ttft_ms)
            itls.extend(res.itls_ms or [])

    return {
        "n_requests": n,
        "ttft_ms_p50": statistics.median(ttfts),
        "ttft_ms_mean": statistics.fmean(ttfts),
        "itl_ms_p50": statistics.median(itls) if itls else None,
        "itl_ms_mean": statistics.fmean(itls) if itls else None,
        "n_itl_samples": len(itls),
        "loop": "closed (deliberate: unloaded service time needs exactly one in flight)",
    }


def _sanity_check(cal: dict) -> None:
    """
    Refuse to derive an SLO from an impossible measurement.

    Job 11599377 propagated a negative TTFT into a negative SLO, produced
    goodput 0.00 at all seven rates, and reported success. Nothing in the chain
    asked whether the input was physically possible. These bounds are loose on
    purpose — they exist to catch nonsense, not to encode expectations.
    """
    problems = []
    if cal["ttft_ms_p50"] <= 0:
        problems.append(f"TTFT p50 is {cal['ttft_ms_p50']:.1f} ms — latency cannot be <= 0")
    if cal["itl_ms_p50"] is None or cal["itl_ms_p50"] <= 0:
        problems.append(f"ITL p50 is {cal['itl_ms_p50']} — must be > 0")
    elif cal["itl_ms_p50"] < 0.5:
        problems.append(
            f"ITL p50 is {cal['itl_ms_p50']:.3f} ms. One decode step of a 1B model "
            "cannot take under 0.5 ms; this indicates buffered reads rather than "
            "per-token arrivals."
        )
    if cal["n_itl_samples"] < cal["n_requests"]:
        problems.append(
            f"only {cal['n_itl_samples']} ITL samples from {cal['n_requests']} requests"
        )
    if problems:
        raise SystemExit(
            "FATAL: calibration is not physically plausible; refusing to derive an "
            "SLO from it.\n  " + "\n  ".join(problems)
        )


def freeze_slo(cal: dict) -> dict:
    slo = {
        "ttft_ms": round(cal["ttft_ms_p50"] * SLO_TTFT_MULTIPLIER, 3),
        "itl_p95_ms": round(cal["itl_ms_p50"] * SLO_ITL_MULTIPLIER, 3),
        "ttft_multiplier": SLO_TTFT_MULTIPLIER,
        "itl_multiplier": SLO_ITL_MULTIPLIER,
        "anchored_to": {"unloaded_ttft_p50_ms": cal["ttft_ms_p50"],
                        "unloaded_itl_p50_ms": cal["itl_ms_p50"]},
        "declared": "multipliers fixed in bench/run_p2.py before any loaded run",
    }
    return slo


async def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2: calibrate SLO, then sweep offered load")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--rates", default="1,2,4,8,16,32", help="offered load, requests/sec")
    ap.add_argument("--duration", type=float, default=60.0, help="steady-state seconds per rate")
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--drain", type=float, default=10.0)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--calibration-requests", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--results-dir", default="results/p2")
    ap.add_argument("--label", default="continuous",
                    help="config being measured, e.g. continuous|static")
    args = ap.parse_args()

    prov = Provenance.capture(repo_root=".", seed=args.seed)
    ok, blockers = prov.is_publishable()
    print("=" * 78)
    print(f"allocation   {prov.allocation_id}")
    print(f"gpu          {prov.gpu_name}")
    print(f"qos          {prov.slurm_qos}")
    print(f"engine       {prov.engine_tag or prov.engine_sha}")
    print(f"publishable  {ok}" + ("" if ok else f"  blockers={blockers}"))
    print("=" * 78)

    # ---- 0. refuse to benchmark an unhealthy server ----
    #
    # Job 11599044 collected a full sweep against a server whose scheduler loop
    # was dead: every streaming request returned HTTP 200 with a well-formed,
    # EMPTY SSE body. /health said "unhealthy" the whole time and nothing looked.
    # A benchmark that cannot tell a broken server from a fast one is worse than
    # no benchmark, so this check is fatal rather than advisory.
    health_url = args.url.rsplit("/v1/", 1)[0] + "/health"
    async with httpx.AsyncClient(timeout=30.0) as hc:
        try:
            h = (await hc.get(health_url)).json()
        except Exception as exc:
            raise SystemExit(f"FATAL: cannot reach {health_url}: {exc}") from exc
    loop_info = h.get("loop", {})
    if h.get("status") != "ok" or not loop_info.get("healthy", True):
        raise SystemExit(
            f"FATAL: server is not healthy; refusing to benchmark it.\n"
            f"  status:     {h.get('status')}\n"
            f"  loop:       running={loop_info.get('running')} healthy={loop_info.get('healthy')}\n"
            f"  last_error: {loop_info.get('last_error')}\n"
            f"  scheduler:  {h.get('scheduler')}"
        )
    print(f"\nhealth OK — loop running, {loop_info.get('steps_total', 0)} steps so far")

    # ---- 1. calibrate ----
    print("\n### CALIBRATION — unloaded batch-1, closed loop (deliberate)\n")
    cal = await calibrate(args.url, args.prompt_tokens, args.max_tokens, args.calibration_requests)
    for k, v in cal.items():
        print(f"  {k:20} {v}")

    # ---- 2. freeze ----
    _sanity_check(cal)
    slo = freeze_slo(cal)
    print("\n### SLO — FROZEN. Multipliers were fixed in source before any loaded run.\n")
    print(f"  TTFT      < {slo['ttft_ms']} ms   "
          f"({SLO_TTFT_MULTIPLIER}x unloaded p50 {cal['ttft_ms_p50']:.1f})")
    print(f"  p95 ITL   < {slo['itl_p95_ms']} ms   "
          f"({SLO_ITL_MULTIPLIER}x unloaded p50 {cal['itl_ms_p50']:.1f})")

    # ---- 3. sweep ----
    rates = [float(r) for r in args.rates.split(",")]
    print(f"\n### OPEN-LOOP SWEEP — {len(rates)} rates, {args.duration}s steady state each\n")

    rows, invalid = [], []
    for rate in rates:
        cfg = LoadGenConfig(
            url=args.url, rate_rps=rate, duration_s=args.duration,
            warmup_s=args.warmup, drain_s=args.drain, seed=args.seed,
            prompt_mean_tokens=args.prompt_tokens, output_mean_tokens=args.max_tokens,
            output_max_tokens=args.max_tokens,
            slo_ttft_ms=slo["ttft_ms"], slo_itl_ms=slo["itl_p95_ms"],
            slo_itl_percentile=95.0, process="poisson",
            name=f"p2_{args.label}_rate{rate:g}",
        )
        run = await run_load(cfg)
        a = analyze(run)
        art, verdict = a.artifact, a.validity
        art.config["slo"] = slo
        art.config["calibration"] = cal
        art.notes.append(f"config={args.label}")
        if not verdict.valid:
            invalid.append((rate, verdict.reasons))
            art.notes.append(f"INVALID: {verdict.reasons}")
        print(render(a))

        path = art.write(args.results_dir)
        # Percentiles are computed HERE from the raw samples, not read from
        # scalars. The artifact deliberately stores samples rather than
        # pre-computed percentiles (R15) — an earlier version of this table did
        # `scalars.get('ttft_ms_p50', 0)`, silently got the default, and printed
        # 0.0 in every row of a seven-rate sweep.
        s = dict(art.scalars)
        tt = art.samples.get("ttft_ms", [])
        s["ttft_p50"] = percentile(tt, 50) if tt else float("nan")
        s["ttft_p99"] = percentile(tt, 99) if tt else float("nan")
        dr = art.samples.get("dispatch_drift_ms", [])
        s["drift_p99"] = percentile(dr, 99) if dr else float("nan")
        s["n_ttft"] = len(tt)
        rows.append((rate, s, verdict.valid, path))
        print(f"  rate {rate:6.1f}  goodput {s.get('goodput_rps', 0):7.2f}  "
              f"tok/s {s.get('output_tok_s', 0):8.1f}  "
              f"ttft_p50 {s['ttft_p50']:7.1f}  ttft_p99 {s['ttft_p99']:8.1f}  "
              f"n={s['n_ttft']:4d}  {'OK' if verdict.valid else 'INVALID'}")

    # ---- 4. report ----
    print("\n### GOODPUT vs OFFERED LOAD\n")
    hdr = (f"{'offered':>9} {'goodput':>9} {'attain%':>8} {'tok/s':>9} "
           f"{'ttft_p50':>9} {'ttft_p99':>10} {'drift_p99':>10}")
    print(hdr)
    for rate, s, valid, _ in rows:
        print(f"{rate:9.1f} {s.get('goodput_rps',0):9.2f} {100*s.get('slo_attainment',0):8.1f} "
              f"{s.get('output_tok_s',0):9.1f} {s['ttft_p50']:9.1f} "
              f"{s['ttft_p99']:10.1f} {s['drift_p99']:10.2f}"
              + ("" if valid else "   <-- INVALID"))

    print("\nThe KNEE is the result — the offered load at which goodput stops tracking")
    print("it and SLO attainment starts falling. A single goodput number without this")
    print("curve is a choice of operating point, not a measurement.")

    if invalid:
        print(f"\n!! {len(invalid)} rate(s) INVALID and must not back a claim:")
        for rate, reasons in invalid:
            print(f"   rate {rate}: {reasons}")

    if not ok:
        print(f"\n!! NOT PUBLISHABLE: {blockers}")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    summary = Path(args.results_dir) / f"p2_{args.label}_summary.json"
    summary.write_text(json.dumps({
        "slo": slo, "calibration": cal,
        "rows": [{"rate": r, "scalars": s, "valid": v} for r, s, v, _ in rows],
        "publishable": ok, "blockers": blockers,
        "allocation_id": prov.allocation_id,
    }, indent=2, sort_keys=True, default=str))
    print(f"\nsummary -> {summary}")

    return 2 if invalid else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
