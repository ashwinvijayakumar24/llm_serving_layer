# Phase 2 results — continuous batching vs static batching

**Date:** 2026-08-01 · **Job:** `11608159` · **GPU:** NVIDIA H200 80GB (sm_90) · **QOS:** `inferno`
**Engine:** `v0.2.1` · **Both arms in one allocation, back to back** (R12)

---

## The SLO, frozen before any loaded run

Multipliers were fixed in source (`bench/run_p2.py`) before the first
measurement; the thresholds are derived from unloaded batch-1 performance
measured against the live server in the same allocation.

| | measured unloaded | multiplier | **SLO** |
|---|---|---|---|
| TTFT p50 | 43.4 ms | 10× | **< 434 ms** |
| TPOT p50 | 9.3 ms | 3× | **< 27.8 ms** |

**Why TPOT and not p95 ITL.** Client-observed ITL is bimodal — SSE chunks
arrive in bursts, so the ITL p50 collapses toward 0.08 ms while the mean stays
correct at ~9.3 ms. Reproduced on CPU against a fake model doing a fixed 10 ms
per token, and identical whether timestamps are taken at line-parse time or at
network-read time, so it is the server's emission pattern rather than a client
artefact. An SLO anchored on ITL p50 would be unmeetable by any real system.
TPOT averages *within* a request and is immune to how tokens were bunched on the
wire. The ITL distribution is still recorded in every artifact.

---

## Goodput vs offered load

Open loop, Poisson arrivals, 120 s steady-state windows, warmup and drain
discarded. ✓ marks rows where the steady-state check passed in **both** arms —
those are the directly comparable ones.

| offered rps | continuous goodput | attain % | static goodput | attain % | |
|---|---|---|---|---|---|
| 1 | 0.93 | 94.1 | 0.80 | 81.4 | |
| 2 | 1.68 | 83.8 | 1.14 | 57.1 | |
| **3** | **2.14** | 70.8 | **1.27** | 41.9 | ✓ |
| 4 | **2.29** ← peak | 60.3 | 0.95 | 25.0 | |
| 6 | 2.17 | 38.1 | 0.46 | 8.0 | |
| **8** | **1.91** | 24.6 | **0.14** | 1.8 | ✓ |
| 12 | 0.83 | 7.2 | 0.03 | 0.2 | |

**Peak goodput: 2.29 rps (continuous) vs 1.27 rps (static) = 1.8×.**
**At offered 8 rps, both rows valid: 1.91 vs 0.14 = 13.6×.**

### Tail latency

| offered rps | continuous TTFT p99 | static TTFT p99 |
|---|---|---|
| 1 | 89 ms | 530 ms |
| 3 | 95 ms | 620 ms |
| 8 | 106 ms | 679 ms |
| 12 | 113 ms | 801 ms |

Continuous batching holds TTFT p99 between **89 and 113 ms across a 12×
range of offered load.** Static batching starts at 530 ms and degrades to
801 ms.

### Throughput is not the story

Raw output throughput is **identical** between the two arms at every rate
(32.4 → 394.6 tok/s). Static batching does not lose throughput — it loses
*latency*, by making every request wait for a whole wave to drain before the
next is admitted. That is precisely why goodput under an SLO is the headline
metric and raw tok/s is not: a system can be at full throughput and serving
almost nobody within their latency budget.

### Where the knee is

Continuous batching's goodput peaks at **offered 4 rps** and declines after.
Static batching peaks at **offered 3 rps**. Above the knee both degrade, which
is the expected shape — offered load exceeds capacity, the queue grows without
bound, and SLO attainment collapses even as throughput keeps rising.

---

## Validity

**Coordinated omission (R1):** dispatch drift p99 stayed at **1.6–2.6 ms**
across every run. Latency is measured from *intended* dispatch, so any harness
lateness is included in the reported numbers rather than hidden.

**Steady state (R11):** rows without ✓ failed the stationarity check —
in-flight count trended across the window, which above the knee is expected
(steady state cannot exist there) and below it means the window was too short.
Those rows are marked INVALID in the artifacts and excluded from the headline
comparison. The ✓ rows at offered 3 and 8 are valid in both arms and carry the
claim.

**Provenance:** every artifact records allocation id
`11608159@atl1-1-01-007-7-0`, GPU, QOS `inferno`, seed `20260801`, repo SHA and
engine tag `v0.2.1`, plus raw per-request and per-token samples rather than
pre-computed percentiles.

---

## What makes the comparison meaningful

Static batching (baseline **B2**) is a **one-line change to admission** in the
same server: `if self.config.static_batching and self.running: return 0`.
Kernels, paged memory manager, HTTP stack, tokenizer and model are byte-for-byte
identical between arms. The only difference is *when* a request is allowed to
join a batch, so the entire delta is attributable to scheduling.

A separate static-batching server would have been easier to write and worthless
to compare against — every other difference would confound the result.
`tests/test_scheduler_static.py` includes a test asserting the two
configurations actually diverge, guarding the failure mode where a baseline
silently behaves like the system under test and the comparison reads as a null
result.

---

## Correctness gate

Batch invariance (R4), job `11608158`, 9/9 passed: mixed prompt lengths, batch
sizes 2/3/4, chunked prefill sharing a batch with decodes, staggered mid-flight
arrival, cancellation isolation, leak check.

**Narrowed claim, stated here rather than discovered later:** greedy output is
token-identical to single-sequence output *on the measured workloads* (prompts
5–36 tokens, batch 2–4). It is **not** bit-invariant to batch shape in general —
see `results/p4/FINDING_batch_shape_numerics.md`, which measures logit drift up
to 0.1745 at larger packed-batch sizes and locates the cause in the engine's
GEMM rather than in this layer.

---

## Bugs found in the benchmark itself

Four, none of which raised an error, each caught by an assertion of a positive
property rather than by an absence of failures:

1. **Negative TTFT (−402 ms).** Calibration passed a fresh `now` as `t0` while
   latency is measured from `t0 + intended_send_time`, placing every intended
   dispatch in the future. The derived SLO became `TTFT < −4023 ms` and goodput
   read 0.00 at every rate.
2. **No plausibility check**, so that negative value propagated into an SLO
   unchallenged. Now fatal.
3. **Summary table read scalar keys that do not exist.** Artifacts store raw
   samples by design (R15), so `scalars.get('ttft_ms_p50', 0)` silently returned
   the default and printed 0.0 in all seven rows.
4. **Threshold and evaluation used different statistics** — anchored on TPOT,
   evaluated on p95 ITL. Every request failed the per-token clause while its
   TTFT sat comfortably inside budget, so a server sustaining 1073 tok/s
   reported goodput ≈ 0.
