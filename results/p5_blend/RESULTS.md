# Phase 5 — affinity/load blend sweep

**Job `11660337`, 4 × dedicated NVIDIA H200, `repo_sha e7c681f`, `dirty=False`,
seed 20260803, `publishable=True`.** Six arms against one fleet: `blend` ∈
{0.00, 0.25, 0.50, 0.70, 1.00} plus `least_outstanding`, on `hot_prefix_skew` at
0.5 / 2 / 8 req/s. 120 s steady state, 20 s warmup, 10 s drain.

**The sweep did not produce the crossover curve it was built to produce.**
13 of 18 cells failed the steady-state check. Read §2 before §3.

---

## 1. What it did establish

### blend = 0 really is the load-aware baseline, in a live fleet

| arm | goodput | attainment | n | valid |
|---|---|---|---|---|
| `prefix_aware@blend0.00` | **0.41** | **100.0%** | 49 | OK |
| `least_outstanding` | **0.41** | **100.0%** | 49 | OK |

Identical to the printed precision on both metrics, at load 0.5, both cells
valid. `blend=0` collapsing to B5 was previously asserted only by a unit test on
the scoring function; this is the same claim exercised through the whole stack —
router process, hint table, HTTP, four replicas — and it holds.

That check gates everything else in this document. If those two arms had
disagreed, every advantage computed against B5 in this project's routing work
would have been suspect.

### Pure affinity costs 16% of goodput at 8× the knee

| arm | goodput | attainment | Δ vs B5 | n | valid |
|---|---|---|---|---|---|
| `prefix_aware@blend1.00` | 2.27 | 29.1% | **−0.433 (−16%)** | 932 | OK |
| `least_outstanding` | 2.71 | 34.6% | 0.000 | 932 | OK |

A matched valid pair at the same offered load in the same allocation. This is
the `blend=1.0` extreme — routing purely by cache affinity, ignoring load
entirely — which had never been measured before this run. It loses, in the
direction and for the reason job `11653158` predicted.

### Below the knee, even a small amount of affinity costs

| arm | goodput | attainment | Δ vs B5 | valid |
|---|---|---|---|---|
| `prefix_aware@blend0.25` | 0.37 | 89.8% | **−0.042** | OK |

At load 0.5, 25% affinity weighting is already worse than none. Three valid
comparisons in this run, three losses or ties for affinity, zero wins.

## 2. Why the curve was not measured

The driver's own knee analysis:

> `hot_prefix_skew  b5_knee_rps 0.50` — *"even the lowest offered load 0.5 req/s
> failed the tracking test; the system is already above its knee at the bottom of
> this sweep"*

Every cell in the sweep sat at or above saturation, so 13 of 18 failed the
steady-state check with *"In-flight count TRENDS across the window: this is not
steady state... percentiles from it describe a ramp."*

**This is the third time this benchmark has failed by putting the ladder above
the knee**, and each time the correction has been smaller than the error:

| attempt | ladder | outcome |
|---|---|---|
| `11610306` | 4 – 48 req/s | 3 of 4 scenarios produced no usable point |
| `11653158` | 0.5 – 8 req/s | knee found at ~1; matched pairs at 0.5 and 8 |
| `11660337` | 0.5 – 2 – 8 | knee is actually ≤0.5; 13 of 18 cells invalid |

I read "knee ≈ 1 req/s" off `11653158`'s attainment column, where the 0.5 cells
showed 96–100% and looked comfortable. Attainment is not the tracking test.
`11660337` delivered 0.41 goodput against 0.5 offered — 82%, below the 90%
tracking tolerance — so the fleet was already saturated at the bottom of the
ladder while still meeting its SLO. **A workload can satisfy its latency target
and still be past the knee, and I used the wrong column to decide.**

A real curve needs loads around 0.1 / 0.2 / 0.3 / 0.4. That is a fourth attempt
at the same measurement, on a workload where four H200s have almost no headroom,
and it has not been run.

## 3. The trend, reported as directional and not claimed

Goodput at load 8, all six arms — **four of these cells are INVALID** and appear
here only because refusing to show them would misrepresent what the run
contains:

| blend | goodput | valid |
|---|---|---|
| 0.00 | 2.74 | INVALID |
| 0.25 | 2.79 | INVALID |
| 0.50 | 2.57 | INVALID |
| 0.70 | 2.29 | INVALID |
| 1.00 | **2.27** | **OK** |
| B5 | **2.71** | **OK** |

The shape is what the mechanism predicts — flat from 0 to 0.25, then falling
monotonically as affinity rises. **It is not a result.** Two of six cells are
valid and they are the two endpoints, so the interior of the curve is
unmeasured and the monotonicity claim is untested.

The `blend=0.25` cell at 2.79 is the only hint in either routing job that a
small affinity weight might beat pure load-awareness. It is invalid, it is a
single cell, and it is contradicted by the valid `blend=0.25` cell at load 0.5
(−0.042). It is noted here so that a future run knows where to look, and it is
**not** evidence of anything.

## 4. Predictions vs outcome

Recorded in `scripts/p5_blend.sbatch` before submission:

| prediction | outcome |
|---|---|
| Above the knee: monotonic decline as blend rises | **UNTESTED** — 4 of 6 cells invalid at load 8 |
| Below the knee: flat or shallow optimum | **UNTESTED** — the run had no below-knee region at all |
| Best blend above the knee is 0.0 or 0.25 | **CONSISTENT, not confirmed** — no blend beat B5 in any of the three valid comparisons, but only two blends were validly measured |

The pessimistic prediction was not vindicated so much as left standing. Nothing
in this run showed affinity winning; nothing in it measured the region where it
might.

## 5. Effect on the claims

**No claim changes.** Bullet 5 continues to rest on job `11653158` (−23% at
`blend=0.7`, matched valid pair, n=946), which this run neither confirms nor
contradicts — its `blend=0.7` arm was invalid at every load, so there is no
within-allocation comparison to make, and comparing across allocations is
forbidden by R12 for exactly this reason.

What this run adds is supporting rather than headline:

- `blend=0` ≡ B5 verified end-to-end in a live fleet, which is what licenses
  calling B5 "the same policy with affinity turned off".
- Pure affinity (`blend=1.0`) measured for the first time: −16% at 8× the knee.
- The crossover remains **unmeasured**, and the reason is documented rather than
  the number estimated.

## 6. Provenance

- Log: `p5_blend_11660337.log` · 20 artifacts · `publishable=True`, `blockers=[]`
- `repo_sha e7c681f`, `repo_dirty=False`, engine `v0.2.1`, 4 × H200, seed 20260803
- All six arms ran against the same four replicas in the same allocation (R12)
- 13 invalid cells are excluded from every difference above and listed in the log
