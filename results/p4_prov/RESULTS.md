# Phase 4 — the four earned cells, re-run from a committed tree

**Job `11653157`, H200, engine `v0.2.1`, seed 20260801, `repo_sha ac5d227`.**
Fresh server pair per cell. 60 s steady state, 20 s warmup, 10 s drain, 2 req/s.

**All four cells: `publishable=True`, `blockers=[]`, `repo_dirty=False`.**

---

## 1. Why this job existed

Every artifact from job `11617299` — the run the S4 claim rests on — was stamped:

```
NOT PUBLISHABLE: Working tree is dirty — the measured code is not any commit.
```

The measurement was sound. Its reproducibility stamp was not, and a number that
cannot be tied to a commit cannot be re-derived by anyone.

## 2. The cause was the harness, not stray files

`Provenance.capture` (`serving/metrics/artifact.py:266`) runs
`git status --porcelain`, which counts **untracked** files. `logs/` was never in
`.gitignore`, and every sbatch in this repo writes `logs/srv_*.log` before the
driver captures provenance.

**So every run in this project dirtied its own working tree by starting.** The
code always was a commit. The guard was reporting on itself, and it had been
reporting on itself for the entire project.

That is the seventh instance of this repo's recurring failure shape — something
reporting a problem or a success that has nothing to do with what it was
watching. It is the worst of the seven in one specific way: a provenance guard
that fires on every run teaches you to ignore the stamp, which is precisely the
opposite of what it exists to do. A guard with a 100% false-positive rate is
worse than no guard.

Two fixes, both applied in `ac5d227`:

1. `logs/` is gitignored.
2. This job writes artifacts **outside the repo** and copies them in after the
   last cell, so a partially written `results/` directory cannot dirty the tree
   mid-run either.

And a preflight that refuses to start on a dirty tree, because otherwise the job
reproduces the exact defect it exists to fix:

```
=== provenance preflight ===
  HEAD: ac5d22736a2518a37029e40cfbe071f0f8ce8437
  tree: CLEAN
...
  tree during run: CLEAN
```

## 3. The numbers reproduce

Same settings, same seed, clean tree. `11617299` → `11653157`:

| cell | block hit rate | Δ TTFT p50 | Δ TTFT p99 | evictions | n |
|---|---|---|---|---|---|
| 512 tok, system prefix, 50% share | 0.137 → **0.137** | −6.3 → **−4.8** | −37.7 → **−38.1 (−23.4%)** | 0 | 118 |
| 512 tok, system prefix, 100% share | 0.272 → **0.272** | −5.1 → **−7.2** | −38.0 → **−37.1 (−22.6%)** | 0 | 118 |
| 150 tok, conversational, 50% share | 0.536 → **0.536** | −8.5 → **−9.2** | — | 0 | 118 |
| 150 tok, conversational, 100% share | 0.762 → **0.762** | −17.3 → **−20.7** | — | 0 | 118 |

Block hit rates reproduce to three decimal places, which is what should happen at
a fixed seed and confirms the workload is identical between runs. TTFT deltas
move by 1–3 ms on 59–79 ms baselines — ordinary run-to-run variation, and the p99
result that carries the claim is stable to within 1 ms.

**The claim is unchanged and now carries a commit.** `−23% TTFT p99 at 512-token
prompts with a shared system preamble` is `−38.1 ms on a 165.2 ms baseline` at
`repo_sha ac5d227`, `engine v0.2.1`, H200, seed 20260801.

## 4. One thing the job could not confirm

The per-cell eviction audit printed:

```
eviction audit: UNKNOWN — no evictions key; audit did NOT run
```

for all four cells. That is the **corrected** three-outcome audit working as
intended: the metrics endpoint does not expose an `evictions` key, so the audit
says it did not measure rather than reporting `clean`. The earlier two-outcome
version printed `clean` in exactly this situation, which is how 36 contaminated
cells were passed as clean in `11617299`.

The eviction count is confirmed **zero** from the benchmark driver's own summary
table, which reads it through a different path. The audit is redundant here; it
is kept because it is not redundant at longer prompt lengths, where the pool does
saturate.
