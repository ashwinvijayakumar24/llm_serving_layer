# P4 clean crossover — does the radix prefix cache actually make serving faster?

**Job `11617299`, H200, one fresh server pair per cell (36 pairs), seed 20260801,
rate 2 req/s, 60 s steady state + 20 s warmup + 10 s drain, output mean 32 tok,
pool `SERVING_KV_BLOCKS=40000`.**

This is the third attempt at this measurement. The first two are kept in
`results/p4/` because the way they failed is the most useful thing in this
directory. Read §4 before quoting any number from the earlier runs.

---

## 1. Answer

**Yes — where there is real prefix sharing, and by 5–17 ms of TTFT p50 on a
57–79 ms baseline. Where there is not, it is free.**

Both halves of that sentence are load-bearing. A cache that helps 20% on shared
traffic but taxes unshared traffic is not obviously worth enabling by default;
this one is measured at ≤1.3 ms of cost on the zero-sharing control at every
length where the cell was valid, so it can be left on.

## 2. Valid cells

Valid means the driver's own steady-state and plausibility checks passed:
in-flight count not trending, `n = 118/118` requests inside the window,
**evictions 0**, and the sharing oracle within tolerance of the measured block
hit rate.

### 150-token prompts — every cell valid

| structure | share | oracle | measured hit | TTFT p50 | Δ p50 | Δ p99 | n |
|---|---|---|---|---|---|---|---|
| zero (control) | 0.00 | 0.000 | 0.035 | 57.2–59.8 | **+0.4 … +1.0** | −1.5 … −6.0 | 118 |
| system | 0.00 | 0.000 | 0.070 | 58.5 | +0.3 | +2.7 | 118 |
| system | 0.50 | 0.397 | 0.405 | 58.7 | −0.0 | +3.4 | 118 |
| system | 1.00 | 0.763 | 0.723 | 57.7 | −0.5 | +482.5 † | 118 |
| conversational | 0.50 | 0.533 | 0.536 | 61.1 | **−8.5** | — | 118 |
| conversational | 1.00 | 0.765 | 0.762 | 64.8 | **−17.3** | — | 118 |

† A single stall, not a trend: p50 is −0.5 ms in the same cell. Reported because
suppressing a p99 outlier that contradicts a p50 win is exactly the kind of
selective reporting the rest of this document exists to avoid.

### 512-token prompts — `zero` and `system` cells valid

| structure | share | oracle | measured hit | TTFT p50 | Δ p50 | Δ p99 | n |
|---|---|---|---|---|---|---|---|
| zero (control) | 0.00 | 0.000 | 0.011 | 77.2–79.3 | **−1.2 … +1.1** | +1.8 … +10.3 | 118 |
| system | 0.00 | 0.000 | 0.022 | 76.5 | −0.7 | +4.3 | 118 |
| system | 0.50 | 0.123 | 0.137 | 71.1 | **−6.3 (−8.1%)** | **−37.7 (−23.1%)** | 118 |
| system | 1.00 | 0.270 | 0.272 | 71.3 | **−5.1 (−6.7%)** | **−38.0 (−23.1%)** | 118 |

The 512-token `conversational` cells at share 0.50 and 1.00 were marked INVALID
by the driver and are excluded.

**The p99 result is the stronger one.** At 512 tokens with a system prefix the
cache takes 23% off the tail at both sharing rates while barely moving the
median. That is the shape you expect: a prefix hit does not speed up the request
that misses, it removes prefill work from the queue, which shortens everyone
else's wait. The tail is where queueing shows up.

## 3. Cells with no conclusion

**1024 and 2048 tokens: all 24 cells INVALID.** Not "the cache lost" — no
measurement exists. Two independent checks fired:

- `n` collapsed from 118 to 48–58 at 1024 and to 9–15 at 2048. Most requests
  fell outside the steady-state window.
- Evictions fired **inside a single cell**: 7,525–10,788 at 1024, 10,281–10,788
  at 2048, against a 40,000-block pool.

The second one falsifies the sizing argument written into
`scripts/p4_clean.sbatch`, which claimed a single cell could not saturate 40,000
blocks. It can, at 1024+ tokens with a lognormal length distribution whose tail
is long and whose blocks the cache holds for the entire 90-second cell. Fixing
this needs a bounded `max_cached_blocks` — which measures a *different* cache —
or a shorter cell. Neither was run. **These lengths are untested, and the
project claims nothing about them.**

Note the zero-sharing controls at 1024/2048 are +0.9 … +4.4 ms, i.e. still
nearly free. That is the direct refutation of the earlier +416/+907 ms figure
(§4), but the cells are invalid so it is corroboration, not evidence.

## 4. What the two earlier runs got wrong

**Run 1 (`11611626`) — all 36 cells shared one server.** A cache holds a
reference to every block it caches, so cached blocks are never freed as requests
retire. Across four prompt lengths the trie consumed the pool; from the
1024-token cells onward every request evicted before it could allocate. Reported
+416 ms at 1024 and +907 ms at 2048 on the **zero-sharing control**, which was
read at the time as "prefix caching makes serving much slower". It was not a
property of the cache. It was 11,673 evictions in a cell whose own arithmetic
said eviction could not legitimately occur.

**Run 2 (`11616318`) — fresh per LENGTH.** Still nine cells and ~1,620 requests
per server. The 512-token cells logged 4,759 and 8,968 evictions. Its 150-token
cells were clean and are the source of the break-even numbers in §5.

**Run 3 (this one) — fresh per CELL.** Clean at 150 and 512.

**And the audit built to catch exactly this was itself broken.** The per-cell
check in `scripts/p4_clean.sbatch` printed `clean` for all 36 cells. It printed
`eviction audit: None  clean` — the metrics key was absent, `find()` returned
`None`, and the guard `ev not in (None, 0)` classifies absent as clean. A missing
measurement was reported as a passing one. The evictions in §3 were caught by
the driver's own table, not by the audit written to catch them.

That is the sixth instance in this project of something reporting success while
doing nothing, and the first where the thing reporting falsely was a check added
in response to the previous instance. `docs/BUILD_LOG.md` §6.

## 5. Break-even sharing rate

From run 2's 150-token cells (clean: evictions 0, n=118, all sharing rates on
one server, pool never saturated):

| structure | break-even sharing rate |
|---|---|
| conversational | **0.058** |
| system | **0.452** |

Below that rate the cache costs more than it saves. Conversational prefixes pay
off almost immediately because a multi-turn prompt shares a long prefix with its
own predecessor; a shared system preamble is only ~30 tokens and has to clear
block granularity before it returns anything, so it needs 45% of traffic to
share before it breaks even.

This run cannot recompute break-even: one fresh server per cell means one
sharing rate per driver invocation, and a crossover needs at least two points.
The driver says so explicitly rather than interpolating
(`only 1 usable point(s); a crossover needs at least 2`).

## 6. Provenance

- Log: `p4_clean_11617299.log` (this directory)
- 157 per-cell artifacts: `p4_clean_{len}_{struct}_{share}_summary.json`
- All runs marked `NOT PUBLISHABLE: Working tree is dirty` — the measured code is
  not a commit. Reproducing requires the tree state, not a SHA. This is a real
  limitation of the run and is not being papered over.
