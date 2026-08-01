# Phase 0 results — GPU correctness gate

**Date:** 2026-07-31 · **Status:** PASSED · **Engine:** `v0.2.0` (`0eff403`)

This is a **correctness gate, not a benchmark.** No performance number here backs
any claim, and none should — the run was a single job on a shared node with no
warmup discipline, no repetition, and no baseline. It exists to answer one
question: *does the GPU path produce correct output?* Until today, nothing in
either repo did.

## Run identity

| | |
|---|---|
| Slurm job | `11596894` |
| Node | `atl1-1-02-018-25-0` |
| Partition / QOS | `gpu-a100` / `inferno` |
| GPU | NVIDIA A100-PCIE-40GB, compute capability 8.0, 40960 MiB |
| Wall time | 00:17:02 |
| Engine commit | `0eff403`, tagged `v0.2.0` |
| Python / torch | 3.11.15 / (conda env `llm`) |
| Raw log | [`gpu_oracle_11596894.log`](gpu_oracle_11596894.log) |

## Why this mattered

Before this run:

- `tests/test_gpu_model.py:42-45` asserted only that logits were finite,
  correctly shaped, and had an argmax inside the vocabulary. A model returning
  confident nonsense passed all three.
- Every real correctness claim in the engine — `tests/test_forward.py:113,137`
  (logits < 1e-3 vs HuggingFace), `tests/test_decode.py:30-48` (32 greedy tokens
  bit-identical) — described the **CPU fp32 NumPy** path.
- Every published performance number described the **GPU fp16** path.

So correctness and performance were measured on different code. Tolerable for a
batch-1 benchmark harness; not tolerable as the foundation for a paged, batched,
preemptible serving layer, where a later disagreement between batched and
single-sequence output would be unattributable across the allocator, the
batching, the paged attention, and a pre-existing GPU-port defect.

This closes **R5**, the register's foundation risk. Every downstream gate — batch
invariance (R4), preemption equality (R3), cache on/off equality (R6), the
FlashInfer differential (R9) — compares against *something*, and this is that
something.

## Results

All 6 gate tests passed (933.52 s).

| Test | Result |
|---|---|
| `test_gpu_greedy_matches_hf_oracle[short]` | 16 greedy tokens match fp32 HF oracle **exactly** |
| `test_gpu_greedy_matches_hf_oracle[medium]` | 16 greedy tokens match fp32 HF oracle **exactly** |
| `test_gpu_matches_cpu_reference[short]` | GPU fp16 == CPU fp32, same engine |
| `test_gpu_matches_cpu_reference[medium]` | GPU fp16 == CPU fp32, same engine |
| `test_gpu_prefill_logits_track_oracle` | max abs logit diff **0.0111**; top-10 overlap **10/10**; argmax matches |
| `test_gpu_chunked_prefill_positions` | chunked == single-shot; max abs logit diff **0.02344** |

Regression, confirming the additive-change guarantee held:

- `tests/test_gpu_model.py` + `tests/test_components_gpu.py` — **10 passed**
- `tests/test_attention_backend.py` — **20 passed** (CPU, BatchMeta invariants)

Oracle fixtures were generated on the node (they had never existed on PACE), and
the oracle's own determinism check passed: two identical HF forwards differed by
exactly `0.0`.

### Reading the numbers

`max |logit diff| = 0.0111` between GPU fp16 and CPU fp32 is consistent with
fp16 accumulation across 16 layers — the existing GPU component tests already
require `atol=1e-2` (`tests/test_components_gpu.py:83`). The structural check is
the more informative one: **10/10 top-10 overlap** means the two paths agree
about which tokens are plausible at all, which fp16 rounding does not disturb and
a real defect would.

The 16-token greedy prefix was chosen as the gate rather than all 32 stored by
the oracle, because fp16-vs-fp32 argmax can legitimately flip once two logits
fall within fp16 resolution. **It matched at 16 with no divergence observed**, so
the threshold was not exercised; if a future run diverges, the test reports the
true divergence index rather than silently loosening.

## Cost — and R28 closed

| | |
|---|---|
| SU before | 999.93 |
| SU after | 999.85 |
| Delta | **0.08 SU** for 17:02 on one A100 |
| **Measured rate** | **0.282 SU per A100-GPU-hour** |

This *confirms* the earlier two-sample inference (~0.28) that the PRD flagged as
unsafe to budget against. It is now measured from a known job duration.

Practical consequence: ~3,500 A100-GPU-hours remain. An 8×L40S run for 8 hours
costs ~14 SU. **SUs are not a constraint on this project** — R28 closed.

## What this run corrected in the plan

**`embers` is not the iteration QOS.** The first submission of this job ran under
`embers` (free, preemptible) and received `Priority=21` behind ~50 queued jobs on
`gpu-a100`, with `StartTime=Unknown` — the scheduler could not estimate it at
all. Resubmitted unchanged on `inferno`, it started in ~8 minutes.

The *validity* rule is unchanged and still correct: no published number may come
from an `embers` run, because preemption silently truncates a measurement window
into something indistinguishable from a completed short run (R13). What was wrong
was the workflow advice. `embers` is for work you can leave overnight. At
0.282 SU/GPU-hour, use `inferno` for anything you are waiting on.

## Not done in Phase 0

**The B1 baseline** (engine batch-1 measured through this repo's harness) requires
the open-loop load harness, which is Phase 2 work. Recorded as outstanding rather
than approximated here — a baseline measured by a different method than the thing
it baselines is worse than no baseline.
