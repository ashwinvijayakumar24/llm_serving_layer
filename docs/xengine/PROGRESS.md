# Cross-engine study — running notes

Living log of what was done and what was measured.
Rules: every number here links to an artifact under `results/xengine/` (or is
marked TODO); entries are dated; nothing is estimated. The point of this file
is that a result recorded here never has to be re-run just to remember it.

## Status

| phase | state |
|---|---|
| 0 — decisions (ADR-025, SPEC) | done 2026-10-08 |
| 1 — harness, renderer, docs, source notes (local, CPU-tested) | done 2026-10-08 (870 CPU tests pass) |
| 2 — vLLM/SGLang env setup on PACE | done 2026-10-09: envs + pilot job 13918362 passed (H200) |
| 3 — runs (W1–W4, ≥3 reps, + W5 appendix) | done 2026-10-10: W1 13931401, W2 13931402, W3 13931403, W4 v2 13933751 (W4 v1 13931404 superseded) |
| 4 — analysis + BENCHMARKS.md | done 2026-10-10: data verified against artifacts (143 claims, 8 fixed; `VERIFICATION.md`); per-finding mechanism status stated; F-004/F-006/F-007 located in code and reproduced on CPU |

## Log

### 2026-10-08
- Scoped the study: Llama-3.2-1B fp16 (our engine only supports 1B), greedy,
  one frozen SLO from our P2 calibration, all engines on one PACE node per job.
- Decided to supersede ADR-013 with ADR-025: raw numbers are published only
  alongside an attribution of the gap; vLLM/SGLang runs with features disabled
  are diagnostics of those engines, never the comparison baseline.
- TensorRT-LLM deliberately out of scope for now.
- Wrote `docs/xengine/SPEC.md` (e391b50). CPU test baseline: 774 passed.
- Known issues in our server found during scoping (logged, not fixed in-study):
  `/health` reports stale feature flags; FlashInfer → PagedTorch fallback is
  silent, so the harness must verify the active backend per run.

- Merged ADR-025 (supersedes ADR-013), methodology/risk amendments, OSS
  candidates template, FINDINGS_OURS.md (F-001..F-005).
- Merged renderer + BENCHMARKS.md skeleton + Makefile + env/sbatch scripts.
  Added a labelled DERIVED table (throughput ÷ own batch-1) for finding (e).
- Source research at vLLM 0.31.0 (`db9527a4`) and SGLang 0.5.21 (`e00930c5`),
  latest stable on 2026-10-08. Verified flags and metric names in
  `ENGINE_FLAGS.md`. Findings that changed the plan:
  - `--enforce-eager` disables torch.compile too → added `vllm-nograph`.
  - vLLM overlaps CPU scheduling by default → added `vllm-noasync`.
  - SGLang defaults to FCFS, not longest-prefix-match → added opt-in `sglang-lpm`;
    hypothesis (b) restated.
  - Our prefix cache matches whole 16-token blocks (like vLLM); SGLang matches
    single tokens.
  - Preemption victims differ: ours newest arrival, vLLM last admitted, SGLang
    fewest generated tokens. vLLM V1 has no swap.
- Matrix cut to 23 cells: diagnostic arms run only on the workload they
  attribute.
- Ten candidate doc/source discrepancies in vLLM/SGLang noted in
  SOURCE_NOTES.md §8 — not yet reproduced, so not yet in oss-pr-candidates.md.

- Harness merged: engine adapters for all 14 arms, closed-loop runner (W1),
  open-loop via the existing loadgen (W2–W4), artifact writer + strict
  validator, anomaly detectors (bimodality, warmup, rep spread, GPU drift).
- Source-verified the harness against vLLM 0.31.0 / SGLang 0.5.21 and fixed
  three bugs before any GPU time: deprecated vLLM launch module, vLLM
  attention-backend log regex, SGLang dict-repr settings dump.
- W5 int8 unblocked without touching `serving/`: the harness factory swaps in
  the engine's own quantizing loader; KV pool pinned to the fp16 pool; a run is
  invalid unless the log confirms quantized linears > 0.
- Sample size: W1 now measures ≥100 requests per point; any p99 cell from <100
  samples is flagged in the doc.
- OSS candidates OSS-001..003 logged (vLLM deprecation message names a
  nonexistent `vllm server` command; missing space in a warning; auto-derived
  scheduler limits never logged). Status: candidate until seen at runtime.
- PACE: checkout fast-forwarded to main (an untracked `results/p5_blend/` that
  was byte-identical to the committed copy was moved to
  `~/ps-simpliearn-0/p5_blend_untracked_backup_20261008`, not deleted).
  `llm` env has every harness dependency except matplotlib, so charts are drawn
  locally after syncing results.

### 2026-10-09
- PACE envs installed cleanly (no pip errors or conflicts):
  vLLM 0.31.0 + torch 2.13.0+cu130 + flashinfer 0.7.0.post1;
  SGLang 0.5.21 + torch 2.13.0+cu130 + flashinfer 0.6.18. Record:
  `results/xengine/_env/engine_versions.txt` on PACE.
- Pilot submitted: job 13910146, embers QOS, H200/H100, W1 points 1 and 16,
  arms ours/vllm/sglang, 1 rep, into `results/xengine_pilot/` (setup check only,
  not reported as results).

- Pilot attempt 1 (job 13910146) landed on a V100 because the pilot asked for an
  untyped GPU; the sbatch sm_80 guard stopped it after 1m41s. Fixed by keeping
  the H200 request and only switching QOS.
- Pilot attempt 2 (job 13918362, embers, H200 node atl1-1-02-012-23-0, driver
  615.71.09) waited ~5.5 h in queue, then ran 20 min. All 6 artifacts valid.
  **Setup checks passed:** ours on FlashInfer (pre-flight + log); vLLM on
  FLASH_ATTN, SGLang on FA3; versions recorded; per-request usage read.
- **Pilot findings fixed before the real runs:**
  1. Every artifact was `repo_dirty` because the harness's own untracked
     results count as dirty in `Provenance`. Harness now judges dirtiness on
     code (untracked `results/` ignored) and keeps the raw verdict for audit.
  2. SGLang's retraction/eviction counters are labeled and exported only after
     the first event, so "absent" was recorded as "not exposed". Now: absent on
     a live endpoint = 0, with the rule written into the artifact (OSS-004).
  3. SGLang `page_size` was parsed as 16 from `c128_page_size`; real value 1.
     All resolved-default patterns now require a word boundary.
  4. The harness stopped servers by signalling the whole process group, which
     made SGLang log a scheduler crash and get SIGKILLed. Now parent-first.
  5. New anomaly detector `tail_outliers_*`: SGLang at concurrency 1 had 2/100
     TTFTs ~10x the median mid-run, which set its p99 alone.
- Startup cost observed (not a benchmark result, n=1): ours healthy after 19 s,
  vLLM 452 s, SGLang 268 s (CUDA-graph capture / compile). Matters for job
  wall-clock limits: one server start per arm per workload.
- OSS-003 confirmed at runtime (vLLM never logs `max_num_seqs`); OSS-004 added.
- Pilot numbers are single-rep setup checks on a preemptible QOS and are **not**
  results; they are kept in `results/xengine_pilot/` only as setup evidence.

- Full matrix submitted on inferno (user approved): W1 13931401, W2 13931402,
  W3 13931403, W4 13931404 (1×H200 each; W1 and W2 also run the W5 int8 cells).
- Two pre-start fixes pushed while the jobs were queued:
  (1) in-job `make render` now writes an untracked copy under
  `results/xengine/_render/`, so parallel jobs never modify a tracked file;
  (2) `vllm-matched` matches only `--max-num-seqs 32` when the workload sets no
  pool, because ours derives a larger pool (4.44M tokens) than vLLM can allocate
  (4.08M) and copying it would fail vLLM's startup.
- **Provenance note for W1 (13931401):** the job started 22:51:15 EDT at
  `614cd1a`; the PACE checkout was fast-forwarded to `3a91fd7` at 22:52:32. Only
  the first arm (`ours`) had started, and its process loaded `614cd1a`; all later
  W1 arms run `3a91fd7`. The diff between the two commits touches only
  matched-arm pool resolution (`run.py:resolve_kv_pool_tokens`,
  `engines.py` vLLM matched flags) and tests, which the `ours` arm does not
  execute. The `ours` artifacts nevertheless record `3a91fd7` because the SHA is
  read at write time. Rule adopted: no PACE checkout changes while jobs run.

## Measured results

All rows: NVIDIA H200, inferno QOS, Llama-3.2-1B-Instruct fp16, greedy,
512-token prompts / 128-token outputs (W1, closed loop), 3 reps per cell,
repo `3a91fd7` clean. Values are cell means from `docs/xengine/BENCHMARKS.md`.

| date | workload | what | value | artifacts | job |
|---|---|---|---|---|---|
| 2026-10-10 | W1 c=1 | output tok/s: ours / vLLM / SGLang | 117 / 730 / 757 | `results/xengine/W1/{ours,vllm,sglang}/concurrency1_rep*.json` | 13931401 |
| 2026-10-10 | W1 c=32 | output tok/s: ours / vLLM / SGLang | 1984 / 10970 / 11957 | `.../concurrency32_rep*.json` | 13931401 |
| 2026-10-10 | W1 c=1 | TPOT p99 ms: ours / vLLM / SGLang | 8.27 / 1.30 / 1.25 | same | 13931401 |
| 2026-10-10 | W1 c=1 | vLLM ladder tok/s: default / no-async / no-graph / eager | 730 / 477 / 185 / 154 | `results/xengine/W1/vllm*/concurrency1_rep*.json` | 13931401 |
| 2026-10-10 | W1 c=1 | SGLang ladder tok/s: default / no-overlap / eager | 757 / 486 / 133 | `results/xengine/W1/sglang*/concurrency1_rep*.json` | 13931401 |
| 2026-10-10 | W1 c=64 | ours flattens: tok/s 1984→2110 (c32→64), TTFT p99 602→2233 ms | — | `results/xengine/W1/ours/` | 13931401 |
| 2026-10-10 | W5 on W1 | ours int8 vs fp16 tok/s at c=1 | 82 vs 117 (−30%) | `results/xengine/W1/ours-int8/` | 13931401 |

| 2026-10-10 | W3 rate 8 | prefix hit rate: ours / vLLM / SGLang | 0.626 / 0.629 / 0.635 (ceiling ≈0.64) | `results/xengine/W3/{ours,vllm,sglang}/rate8_rep*.json` | 13931403 |
| 2026-10-10 | W3 rate 8 | TTFT p50 ms, cache on vs off: vLLM / SGLang | 15.1 vs 17.1 / 14.8 vs 15.7 | `results/xengine/W3/*/rate8_rep*.json` | 13931403 |
| 2026-10-10 | W3 rate 16 | ours TTFT p50 ms, cache on vs off | 728 vs 1,160 | `results/xengine/W3/ours*/rate16_rep*.json` | 13931403 |
| 2026-10-10 | W2 | ours free KV blocks at successive cell starts (0 requests running) | 277,590 → … → 0, then goodput 0 | `results/xengine/W2/ours/*.json` (before-snapshots) | 13931402 |
| 2026-10-10 | W2 rate 32 | goodput rps: vLLM / SGLang (SLO attainment 1.00) | 32.8 / 33.2 | `results/xengine/W2/{vllm,sglang}/rate32_rep*.json` | 13931402 |
| 2026-10-10 | W4 v2 c=16 | preemptions per run: ours / ours-noprefix / vLLM / SGLang | ≈1,850 / ≈970 / 22 / 5 | `results/xengine/W4/*/concurrency16_rep*.json` | 13933751 |
| 2026-10-10 | W4 v2 c=16 | output tok/s: ours / ours-noprefix / vLLM / SGLang | 592 / 874 / 5,786 / 5,551 | same | 13933751 |
| 2026-10-10 | W4 v2 c=4 | preemptions with pool not short: ours (cache on) vs ours-noprefix | 111–143 vs 0 | `results/xengine/W4/ours*/concurrency4_rep*.json` | 13933751 |

GPU health: every job ended at the 1980 MHz max SM clock, 34–45 °C
(`results/xengine/_env/gpu_*_{start,end}.csv`).

Coverage caveats: W2 and W3 have 1–3 valid reps per cell (steady-state rule);
W2 `ours` cells after its pool filled are flagged; three W5-on-W2 int8 cells
were cut by the time limit.

### 2026-10-10 (later)
- W4 v2 all 45 cells valid in 32 min. F-004 reproduced (ours preempts at
  concurrency 4 only with the cache on). New F-007: with the cache on, 12–18 of
  48 requests "complete" with no output.
- OSS candidates: 5 entries, all reproduced on the pinned versions (OSS-005:
  `vllm serve --help` crashes on a GPU-less host).

## Open questions / decisions pending

- vLLM 0.31.0 targets CUDA 13.0 in source (docs say 12.9); PACE module is
  cuda/12.9.1 — may need a cu129 wheel. Resolve at env setup.
- Our KV pool knob for W4 is `SERVING_KV_BLOCKS` (no `serving/` change needed).
