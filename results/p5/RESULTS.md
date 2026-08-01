# Phase 5 results — prefix-aware routing across 4 GPUs

**Date:** 2026-08-01 · **Job:** `11610306` · **4 × dedicated NVIDIA H200** · **Engine `v0.2.1`**
**Claim S5 (prefix-aware beats B5 on shared-prefix workloads): NOT EARNED.**
**One §10 losing-case prediction CONFIRMED.**

---

## 1. What was actually measured

Four replicas, one per **dedicated** H200 (`CUDA_VISIBLE_DEVICES` pinning), three
router processes over the same fleet: `prefix_aware`, `least_outstanding` (B5),
`round_robin` (B4). Five offered loads × four scenarios × three policies.

An earlier attempt (job `11609657`) put all four replicas on **one** H200. They
contended for SMs, the fleet saturated by load 2, and 45 of 48 cells failed the
steady-state check. That was a limitation of the setup rather than a finding
about routing, and it is why this run uses four GPUs.

## 2. The one scenario with valid data — and the prediction held

`zero_sharing`, all three policies valid at loads 4 and 16:

| load | policy | goodput | attain % | TTFT p50 | TTFT p99 | Δ goodput vs B5 | n |
|---|---|---|---|---|---|---|---|
| 4 | prefix_aware | 3.18 | 82.7 | 73.9 | 210.2 | **−0.111** | 173 |
| 4 | least_outstanding (B5) | 3.29 | 85.5 | 74.4 | 203.2 | — | 173 |
| 4 | round_robin (B4) | 3.02 | 78.6 | 73.6 | 180.7 | −0.267 | 173 |
| 16 | prefix_aware | 0.47 | 3.0 | 137.7 | 2962.6 | **−0.311** | 708 |
| 16 | least_outstanding | 0.78 | 4.9 | 170.1 | 2890.0 | — | 707 |
| 16 | round_robin | 0.51 | 3.2 | 183.3 | 3054.9 | −0.267 | 702 |

The driver's verdict:

> `zero_sharing`: **PREDICTION HELD (§10.1).** prefix_aware does not beat B5 at
> any load tested (2 losses, 0 ties) — as predicted before the run.

**What was predicted, in writing, before any measurement:**

> TIE with B5, or a slight LOSS: there is nothing to be cache-aware about, and
> any deviation from load-optimal placement is pure loss.

That is what happened: −0.111 goodput at load 4, −0.311 at load 16. Prefix
awareness with no prefixes to be aware of is overhead, and the overhead is small
but real.

**Note also that prefix_aware beats round_robin at load 4** (3.18 vs 3.02) while
losing to B5. That is precisely the situation methodology §6 warns about — a
router that beats B4 but not B5 has demonstrated *load balancing*, not prefix
awareness. Reported as such rather than as a win over "the baseline".

## 3. The three scenarios that could not be measured

`system_prompt_sharing`, `uniform_prefix` and `hot_prefix_skew` produced **no
usable point** — the driver reports:

> only 0 usable point(s); a crossover needs at least 2
> the prediction is **untested, which is not the same as confirmed**

Their cells collapsed at every load: goodput 0.00, TTFT p50 in the 10–65 second
range. Those workloads carry a long shared prefix, so each request costs far more
prefill than `zero_sharing`'s, and four H200s saturate below the lowest offered
load tested. The knee sits under 4 req/s for them, and nothing was measured
below that.

**This is a workload-design failure, not a routing result.** The loads were
chosen from `zero_sharing`'s capacity and never re-derived for the heavier
scenarios. Fixing it means measuring each scenario's own knee first and sweeping
around it — the harness supports this; it was not run.

## 4. What this earns

**No S5 claim.** Prefix-aware routing was not shown to beat the real baseline on
any workload where it should, because those workloads were never measured inside
their valid range.

**One confirmed losing-case prediction**, which is a genuine methodological
result: the case where the optimisation *should* lose was written down in advance
and then lost, by the predicted margin, in the predicted direction.

The router itself is built and CPU-tested (57 tests): prefix-aware, B5, B4 and
consistent-hash policies behind one interface, a TTL'd hint table, dual-signal
health detection, quarantine with hint purge, three-way in-flight
classification, jittered retry, quantized ramp-in for recovered replicas, and
graceful drain. `blend=0` is asserted to be exactly B5. None of that is a
measured serving claim.

## 5. Validity

- 4 dedicated H200s, one replica per GPU, all policies against the same fleet in
  the same allocation (R12).
- Cells failing the steady-state check are excluded from every difference and
  from every crossover; the driver refuses to interpolate a crossover from fewer
  than two usable points.
- Predictions for untested scenarios are reported as **untested**, explicitly
  distinguished from confirmed.
