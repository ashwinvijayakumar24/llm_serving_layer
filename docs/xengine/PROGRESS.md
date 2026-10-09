# Cross-engine study — running notes

Living log of what was done, what was measured, and what is resume-usable.
Rules: every number here links to an artifact under `results/xengine/` (or is
marked TODO); entries are dated; nothing is estimated. The point of this file
is that a result recorded here never has to be re-run just to remember it.

## Status

| phase | state |
|---|---|
| 0 — decisions (ADR-025, SPEC) | done 2026-10-08 |
| 1 — harness, renderer, docs, source notes (local, CPU-tested) | in progress |
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

## Measured results

None yet. (Each entry: date, workload, arms, headline number with condition,
artifact path, job id, GPU.)

## Resume-usable facts

None yet — draft bullets stay as placeholders until the rows above exist.

## Open questions / decisions pending

- Pinned vLLM / SGLang versions (source-research agent choosing latest stable).
- Whether our engine exposes a KV-pool-size knob for W4 without modifying `serving/`.
