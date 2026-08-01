# results/

**This directory is deliberately NOT gitignored.**

The engine does the same — `bench/results/` there holds 24 tracked files, and its
`.gitignore:29` records the decision explicitly. This repo inherits the practice.

What differs is the *content* of an artifact. The engine's harness stores
per-run p50/p99 and discards the raw samples (`bench/harness.py:136-137,178`),
which makes correct pooling across requests impossible after the fact. For a
serving benchmark that is disqualifying: percentiles must be computed over the
pooled request set, not averaged across per-run percentiles.

Every published number must resolve to a committed file here containing:

- **raw** per-request and per-token samples (NOT pre-computed percentiles)
- the full run config and RNG seed
- realized workload distributions (not requested parameters)
- git SHA of this repo + pinned engine tag (`v0.1.0`)
- Slurm allocation id, node name, and **QOS**

The last one is a validity gate, not bookkeeping: `embers` is preemptible, and a
preempted run truncates its measurement window into something that looks like a
completed short run. No published number may come from an `embers` run.

The CI `artifact-schema` job enforces these fields. Provenance cannot be
retrofitted onto results already collected, which is why it lands in Phase 0.
