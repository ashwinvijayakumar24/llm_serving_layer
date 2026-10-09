# Cross-engine study — running notes

Living log of what was done, what was measured, and what is resume-usable.
Rules: every number here links to an artifact under `results/xengine/` (or is
marked TODO); entries are dated; nothing is estimated. The point of this file
is that a result recorded here never has to be re-run just to remember it.

## Status

| phase | state |
|---|---|
| 0 — decisions (ADR-025, SPEC) | done 2026-10-08 |
| 1 — harness, renderer, docs, source notes (local, CPU-tested) | docs, renderer, source notes merged; harness in progress |
| 2 — vLLM/SGLang env setup on PACE | not started |
| 3 — runs (W1–W4, ≥3 reps, + W5 appendix) | not started |
| 4 — analysis + BENCHMARKS.md | not started |

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

## Measured results

None yet. (Each entry: date, workload, arms, headline number with condition,
artifact path, job id, GPU.)

## Resume-usable facts

None yet — draft bullets stay as placeholders until the rows above exist.

## Open questions / decisions pending

- vLLM 0.31.0 targets CUDA 13.0 in source (docs say 12.9); PACE module is
  cuda/12.9.1 — may need a cu129 wheel. Resolve at env setup.
- Our KV pool knob for W4 is `SERVING_KV_BLOCKS` (no `serving/` change needed).
