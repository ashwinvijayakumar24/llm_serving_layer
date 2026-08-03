# Phase 5 re-run — routing swept around each scenario's own knee

**Job `11653158`, 4 × dedicated NVIDIA H200, engine `v0.2.1`, seed 20260802.**
Ladder 1: `system_prompt_sharing`, `uniform_prefix`, `hot_prefix_skew` at
0.5/1/2/4/8 req/s. Ladder 2: `zero_sharing` at 4/8/16/32. 120 s steady state,
20 s warmup, 10 s drain, three policies against the same four replicas.

**S5 is still NOT EARNED. The re-run produced something better than a win.**

---

## 1. What the re-run fixed

Job `11610306` swept 4–48 req/s for every scenario and got no usable point out of
three of them: the ladder was derived from `zero_sharing`'s capacity and reused
for workloads whose requests cost several times more prefill, so every cell sat
far above the knee.

The knee for these workloads is **≈1 req/s**, visible directly in SLO
attainment: 94–100% at 0.5 and 1.0, 69–90% at 2.0, 51–69% at 4.0, 26–38% at 8.0.
The old ladder started at 4× the knee and went to 48×. This one brackets it.

That fix worked — there are now matched valid pairs at both ends of the ladder.
It is also the whole reason the result below exists.

## 2. The result: affinity routing is measurably harmful above the knee

Matched pairs, both arms passing the steady-state check at the same offered load:

| scenario | load | prefix_aware | least_outstanding (B5) | Δ goodput vs B5 | n |
|---|---|---|---|---|---|
| hot_prefix_skew | 0.5 (½× knee) | 0.49 | 0.51 | −0.017 | 61 |
| hot_prefix_skew | **8.0 (8× knee)** | **2.11** | **2.74** | **−0.633 (−23%)** | 946 |
| system_prompt_sharing | 0.5 | 0.50 | 0.51 | −0.008 | 61 |
| system_prompt_sharing | 8.0 | 2.88 | 2.85 | +0.025 (+0.9%) | 947 |

**The −23% is the finding.** On a workload with one very hot prefix, at eight
times the saturation knee, routing by cache affinity costs nearly a quarter of
the fleet's goodput against a policy that ignores caches entirely and only looks
at load. SLO attainment falls with it: 26.4% versus 34.4%.

The mechanism is the point and it is not subtle: one hot prefix lives on one
replica. Affinity keeps sending its traffic there. Below the knee that replica
absorbs it and the cache hit is free. Above the knee it is already the busiest
replica in the fleet, and every additional request routed to it by affinity is a
request routed *away* from an idle peer. The cache saves prefill work; the queue
charges more for it than it saves.

This is methodology §10.3 — "affinity and load balance conflict directly near
saturation" — tested rather than asserted, with a number on it.

## 3. Predictions, including the one I got wrong

| scenario | predicted | outcome |
|---|---|---|
| `uniform_prefix` (§10.2) | B5 wins — every replica caches the same prefix after warmup, so affinity degenerates to "pick any replica" | **HELD** — 1 loss, 0 ties |
| `hot_prefix_skew` (§10.7) | prefix_aware **loses** unless affinity is blended with load | **HELD** — 2 losses, 0 ties |
| `zero_sharing` (§10.1) | tie or slight loss | **NOT MEASURED** on this ladder — see §5 |

**I predicted `hot_prefix_skew` would be prefix-aware's best case and it was its
worst.** That prediction is written into `scripts/p5_knee.sbatch` before the run:
*"A skewed hot prefix is the case where cache locality is worth more than perfect
load balance. If it loses here it loses everywhere."*

It lost here, by the largest margin in the run.

The project's own benchmark methodology had it right and I contradicted it while
writing the job script. §10.7 says, in text written weeks earlier: *"one very hot
prefix sends a disproportionate share of traffic to whichever replica owns it.
§10 calls this the most likely place for a genuinely bad result, and therefore
the most valuable one to measure."*

Recorded rather than quietly aligned to the outcome afterwards. The reason to
write predictions down before a run is so that being wrong costs something.

## 4. Why `system_prompt_sharing` at 8 req/s is NOT a win

prefix_aware 2.88 vs B5 2.85 req/s, attainment 36.1% vs 35.7%. That is
**+0.9% goodput and +0.4 percentage points of attainment**, and it is a tie
dressed as a victory. Two matched cells at 0.5 and 8.0 req/s, one negative and
one marginally positive, do not establish a direction.

Claiming it would mean quoting a 0.9% difference from a single cell as evidence
that prefix-aware routing beats a load-aware baseline. The driver's own verdict
on this scenario is *"never ahead: the advantage is −0.008333 at the lowest x.
There is no crossover to report because there was no lead to lose."*

prefix_aware does beat round_robin at this load (2.88 vs 1.98). That is table
stakes — it proves load balancing, which B5 already does, and says nothing about
cache awareness.

## 5. `zero_sharing` lost its valid cells

Ladder 2 (4/8/16/32 req/s) produced **no valid matched pair**: attainment was
65–68% at load 4 and collapsed after. The previous job got valid cells at load 4
with a 45 s window; this one used 120 s, and the longer window exposed a trend
that the shorter one averaged over.

That is the steady-state check doing its job, not a regression. The §10.1
prediction remains confirmed **from job `11610306`**, where it was measured
cleanly at loads 4 and 16 (Δ −0.111 and −0.311 vs B5) — and it is quoted from
that job, not from this one.

## 6. What this earns and what it does not

**Does not earn:** any claim that prefix-aware routing improves serving
performance. It was not shown to beat a load-aware baseline on any workload at
any load in either attempt.

**Does earn:** a measured, quantified statement about *when cache-aware routing
is the wrong choice*, with the mechanism, on real hardware, against the correct
baseline — plus two §10 losing-case predictions confirmed and one of my own
falsified.

The design consequence follows directly and is now supported rather than
asserted: **affinity must be blended with load, not applied on its own.** The
router already implements this — `score = blend·affinity − (1−blend)·min(1,
effective_load/load_scale)` — and `blend=0` is asserted in tests to be exactly
B5. What this run measured is the `blend=1` extreme, which is the one worth
knowing the cost of. Sweeping `blend` is the obvious next experiment and has not
been run.

## 7. Provenance

- Log: `p5_knee_11653158.log` · 60 artifacts in this directory
- 4 dedicated H200s, one replica per GPU, all three policies against the same
  fleet in the same allocation (R12)
- Cells failing the steady-state check are excluded from every difference; the
  driver refuses to interpolate a crossover from fewer than two usable points
