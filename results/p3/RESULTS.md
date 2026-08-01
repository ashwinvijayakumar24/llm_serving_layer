# Phase 3 results — preemption under memory exhaustion

**Date:** 2026-08-01 · **Job:** `11609161` · **H200** · **Engine `v0.2.1`**
**Correctness gate: PASSED** (job `11608158`) · **Benchmark: recompute measured, swap NOT measured**

---

## 1. Correctness — both policies, bit-identical

Greedy output under forced memory pressure is **bit-identical to an unpreempted
control**, independently for recompute and for swap. Forced by a genuinely small
block pool, not a mocked trigger: a companion test disables preemption on the
same workload and asserts `AllocationError`.

This settles the one thing that could not be shown on CPU. Recompute re-prefills
as a **single chunk** what was originally many one-token decode passes, so its
fp16 stability was a real open question — different reduction shapes, same
arithmetic. Swap is immune by construction, so the fact that both match is the
stronger result.

Also covered: a request preempted twice, preemption *during* a chunked prefill,
the starvation-guard fallback (R40), full block reclamation, and allocator
invariants throughout.

---

## 2. Recompute — measured

Control pool large enough that it **preempted 0 times at every length**, verified
rather than assumed. 60 s steady-state windows, 248 completions per cell.

| prompt len | preemptions | TTFT p50 tax | TTFT p99 tax | e2e p50 tax | tokens recomputed | valid |
|---|---|---|---|---|---|---|
| 64 | 2 | **+0.7 ms** | +0.4 ms | +30.4 ms | 586 | OK |
| 128 | 2 | **+2.7 ms** | +6.6 ms | +85.0 ms | 970 | OK |
| 256 | 49 | **+4.2 ms** | +7.8 ms | +101.7 ms | 42,254 | OK |

"Tax" is the delta against the unpreempted control on the same node in the same
allocation.

**Recompute is cheap here, and the reason is the model.** Llama 3.2 1B holds
32 KB of KV per token and prefills a few hundred tokens in milliseconds, so
redoing that work costs little. The tax grows with length exactly as predicted —
recompute cost scales with the prefill it must redo — but even at 256 tokens with
49 preemptions it is under 5 ms at TTFT p50.

---

## 3. Swap — NOT measured, and why

Every swap cell was marked **INVALID** by the driver and excluded from all
claims:

```
len  64  swap  preemptions 44  e2e_p99 179052 ms   n=5    INVALID
len 128  swap  preemptions 47  e2e_p99 179012 ms   n=3    INVALID
len 256  swap  preemptions 331 e2e_p99 179985 ms   n=5    INVALID
```

179 s is the **client timeout**, not a latency. Three to five requests completed
per cell against 248 for recompute. Two validity checks fired: steady state not
verified, and the **admission-control alarm** (the starvation fallback fired 2–3
times, which per ARCHITECTURE §5.2 means admission let in work the running set
could not step).

**Resumption itself is healthy** — mean resume latency 1.0–1.3 ms, exactly 1 step
— so swapped requests that get their memory back recover immediately. The
problem is that most never do.

### One fix found and verified, one hypothesis rejected

**Fixed:** `_resume_swapped` gated on `can_allocate()`, which reserves the
*admission* watermark. A swapped request is not new work — it was admitted, ran,
and had its memory taken away — so holding it behind the admission watermark
starves it. The codebase already states this rule one function away, in
`_reuse_cached_prefix`. Regression test **verified to fail on the prior code**.

**Rejected:** the natural next hypothesis was that admission spends freed blocks
on new work before a swapped request — which needs its *entire* block count back
at once — can reclaim them. A guard (`if self.swapped: return 0` in `_admit`) was
written to test it and **could not be made to fail on the unfixed code**: with
the CPU model both orderings complete every request in the same number of steps.
It was reverted. A guard that changes nothing measurable is a guess, and it would
have cost throughput to buy an unproven benefit.

**Swap under sustained load is an open problem**, recorded as one. It is a
liveness/throughput issue, not a correctness one — the gate passes for both
policies.

---

## 4. The §5.2 prediction: UNTESTED, not confirmed

The architecture predicted, in writing and before measurement, that **recompute
would win at nearly all lengths** for a 1B model.

The driver's verdict:

> PREDICTION VERDICT: UNTESTED. Fewer than two lengths produced a usable
> recompute/swap pair, so neither P1 nor P2 can be evaluated. **This is not
> evidence for the prediction.**

Recompute's numbers are consistent with it, and it is tempting to call that a
confirmation. It is not one. A head-to-head needs both arms valid at a matched
length, and no length produced that. The prediction stands untested.

---

## 5. Validity

- Control preempted **0 times at every length** — a control that preempts is not
  a control, and an earlier run (job `11608501`) failed this check with 4/79/853
  preemptions before the pool was enlarged.
- Percentiles are computed from raw `ttft_ms`/`e2e_ms` samples at print time
  (§5, R15). The driver has no scalar shortcut, which is why a missing
  measurement prints `n/a` rather than `0.0`.
- Swap cells excluded from every table difference and from the crossover.
- Run reported `dirty=True` from untracked artifacts; the recompute numbers are
  therefore indicative rather than publication-grade.

---

## 6. What this earns

**A correctness claim, not a performance one.** Preemption is implemented with
two policies and verified bit-identical under forced pressure for both. Recompute
is measured and cheap. Swap is built, passes correctness, and does not hold up
under sustained load for reasons that are diagnosed but not resolved.
