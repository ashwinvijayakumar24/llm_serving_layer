# Phase 1 results — paged KV cache and block allocator

**Date:** 2026-08-01 · **Gate:** PASSED · **Engine:** `v0.2.1` (`a3d0ab8`) · **Repo:** `374ceb5`

Two separate things are recorded here and they have different standing:

1. **The correctness gate** — paged output is bit-identical to the contiguous
   reference path. This is a *correctness* result and it is the reason Phase 1
   is done.
2. **The S1 capacity measurement** — publishable, with full provenance. This is
   a *claim* and it comes with a mandatory caveat about sequence length.

## Run identity

| | |
|---|---|
| Slurm job | `11598444` |
| Node | `atl1-1-01-006-19-0` |
| Partition / QOS | `gpu-h100` / `inferno` |
| GPU | NVIDIA H100 80GB HBM3, compute capability 9.0 |
| Allocation id | `11598444@atl1-1-01-006-19-0` |
| Repo SHA | `374ceb5`, **clean** |
| Engine | tag `v0.2.1` |
| Seed | `20260731` |
| Raw log | [`p1_gate_11598444.log`](p1_gate_11598444.log) |

## 1. The gate — 16/16 passed

```
BlockAllocator -> SequenceBlocks -> build_batch_meta
    -> PagedTorchBackend -> LlamaModelGPU.forward_varlen
```

must equal

```
KVCacheGPU -> LlamaModelGPU.prefill / decode_step      (P0-validated vs fp32 HF oracle)
```

Bit-identical, not "close". Every component can be individually correct and the
composition still wrong — a position off by one, a page boundary mishandled, a
GQA head reading the wrong KV head. All of those produce fluent text and no
error, which is why the gate is token equality.

| Test | Result |
|---|---|
| `test_paged_matches_contiguous[short]` | 24 tokens identical |
| `test_paged_matches_contiguous[medium]` | 24 tokens identical |
| `test_block_boundary_prompt_lengths[1,15,16,17,31,32,33,47,48]` | 9/9 pass |
| `test_generation_across_many_block_boundaries` | pass — boundaries crossed *during decode*, not just prefill |
| `test_blocks_are_fully_reclaimed` | pass — free list returns to its initial count |
| `test_abandoned_generator_frees_blocks` | pass — client disconnect does not leak |
| `test_fragmented_pool_gives_identical_output` | pass — output independent of *which* physical blocks |
| `test_eos_ids_match_engine` | pass |

Plus 269 CPU tests on the same node.

The three worth understanding:

- **Block-boundary sweep (R8).** A prompt of exactly 16 or 32 tokens is where
  `last_page_len` must report `block_size`, never `0`. Reporting 0 silently
  drops a page of keys; the tail of a partially-filled page is whatever the
  previous owner of that block left behind.
- **Fragmented pool.** The pool is deliberately holed so the sequence receives
  scattered, non-monotonic block ids. An implementation that assumes page
  contiguity passes every clean-pool test and fails only here.
- **Abandoned generator.** Blocks are freed in a `finally`, so a generator
  closed mid-stream still returns its block table. The engine's own loop cannot
  do this — it builds a fresh `KVCacheGPU` per call and relies on GC
  (`engine/scheduler.py:26-27`). With a finite shared pool, a disconnect-heavy
  workload that leaked would drain it with no error anywhere.

## 2. S1 — concurrent-sequence capacity

**Measured**, by driving the real allocator to admission failure — not computed
from the sizing arithmetic. Baseline B3 is computed, because the engine is
single-request and its per-request `max_seq=2048` reservation
(`engine/scheduler.py:16`) is what its allocation strategy *would* support.

| requested mean | realized mean | paged seqs | contiguous | ratio |
|---|---|---|---|---|
| 32 | 32.1 | 29,618 | 571 | **51.9×** |
| 64 | 64.0 | 16,353 | 571 | 28.6× |
| 128 | 128.1 | 8,615 | 571 | 15.1× |
| 256 | 257.8 | 4,406 | 571 | 7.7× |
| 512 | 505.2 | 2,280 | 571 | 4.0× |
| 1024 | 908.5 | 1,278 | 571 | 2.2× |
| 2048 | 1420.2 | 820 | 571 | **1.4×** |

At the 256-token operating point: realized mean 257.8 (stdev 232.9), p50/p90/p99
= 194/524/1261, 0.11% clipped at `max_seq`. Block utilization at stop 0.9998,
internal fragmentation 2.86% (slots allocated but never written).

### How this may and may not be stated

**Publish the table, not a row.** A single row is a choice of workload. The
harness prints its own caveat and it is the honest framing:

> The ratio is approximately `max_seq / realized_padded_length`. It is NOT a
> property of the allocator alone, and it falls to ~1× when sequences actually
> use all 2048 slots.

Quoting "51.9×" without its length distribution would be indefensible under one
follow-up question. The defensible form names the distribution:

> **7.7× concurrent-sequence capacity at fixed VRAM** for a lognormal length
> distribution with realized mean 258 tokens (p90 524), against a baseline
> reserving `max_seq=2048` per request.

**What this does not claim.** Not a throughput result. Not a latency result. The
engine is single-request today, so this compares *allocation strategies*, not
two running servers. Throughput arrives in Phase 2 with batching.

## 3. The provenance gate caught its own author

The first clean-looking run reported:

```
publishable as S1: False
  - Working tree is dirty — the measured code is not any commit.
```

Two causes, both mine: untracked `logs/` from the job, and reaching the model
weights by **symlinking into `vendor/`**, which dirties the submodule. Fixed by
gitignoring `logs/` and pointing at weights with `LLM_WEIGHTS_PATH`. The rerun
recorded `dirty=False`, `engine_tag=v0.2.1`, and a valid allocation id.

This is the intended behaviour and worth stating plainly: the check was written
to stop a future careless run, and the first thing it stopped was a careless run
by the person who wrote it.

## 4. A silent-skip hole, found and closed

Job `11598374` ran this gate on a Tesla V100 (sm_70) under a torch built for
CUDA 13, which dropped Volta support. `torch.cuda.is_available()` returned
False, **all 16 gate tests skipped, and the job exited 0**. In a job log a
skipped gate and a passing gate are indistinguishable.

Closed three ways: `REQUIRE_GPU=1` turns an unusable GPU into a hard failure
instead of a skip; the gate rejects `sm_<80` explicitly rather than trusting
`is_available()`; and a preflight block fails the job before any work runs.

Contributing cause worth remembering: **`--gres=gpu:1` was routed by PACE to a
V100 even though the partition list excluded `gpu-v100`.** The site remaps
generic gres, so only a *typed* gres (`--gres=gpu:h100:1`) actually pins
hardware.

## 5. Risk register changes

- **R8** (`slot_mapping` off-by-one at block boundaries) — exercised by the
  9-length sweep and the fragmented-pool test. Detection confirmed working.
- **R7** (eviction freeing a live block) — allocator leak test and
  `check_invariants()` pass over 5 generations.
- **R9** (FlashInfer layout mismatch) — **still open.** `FlashInferBackend` is
  not built yet; the differential test against `PagedTorchBackend` is what
  retires it.

## 6. Outstanding in Phase 1

- **`FlashInferBackend`** and the differential test. `PagedTorchBackend` was
  written first precisely to be the oracle for it.
- **B1 baseline** — still deferred to Phase 2, where the open-loop harness lives.

## Cost

Job `11598444`: 40 s wall on one H100. At the H100 rate (2.43× A100, measured
A100 rate 0.282 SU/GPU-hour) this is roughly 0.008 SU. Queue strategy mattered
far more than cost: `gpu-a100` estimated a ~90 minute wait behind 33 jobs while
`gpu-h100` started immediately.
