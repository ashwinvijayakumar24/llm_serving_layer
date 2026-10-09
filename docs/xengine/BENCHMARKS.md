# Cross-engine benchmarks: this serving layer vs vLLM vs SGLang

This document compares this serving layer against vLLM and SGLang on the same
GPU, the same model (Llama-3.2-1B-Instruct, fp16) and byte-identical request
streams. It is a **measurement study, not an optimization effort**: the goal is
to measure where the gaps are and explain each one by mechanism. Losing with a
correct explanation counts as success.

- The binding contract (arms, workloads, artifact schema, hard rules) is
  [`SPEC.md`](SPEC.md).
- The governing decision is ADR-025 in [`docs/ADR.md`](../ADR.md), which
  supersedes ADR-013 (the earlier "no throughput comparison, scaling shape only"
  position).
- Source-code citations for vLLM and SGLang live in
  [`SOURCE_NOTES.md`](SOURCE_NOTES.md).

## How to read this document

Every number below is rendered by `bench/xengine/render.py` from the JSON
artifacts under `results/xengine/`. Run `make render` to refresh it. Only the
regions between `<!-- BEGIN GENERATED: … -->` and `<!-- END GENERATED: … -->`
markers are rewritten; everything else is hand-written and survives re-rendering.

Three rules govern the generated tables:

1. **`TODO` means no valid artifact exists for that cell.** The renderer never
   estimates, interpolates or extrapolates a value.
2. **Flags are printed, not smoothed away.** A cell with fewer than three
   repetitions, a large run-to-run spread, excluded invalid runs, or a
   harness-reported anomaly carries a bold bracketed flag.
3. **Diagnostic arms are not baselines.** Arms such as `vllm-eager` turn a
   feature off to measure that feature's contribution. They appear in separate
   tables and are never "the competitor".

## Setup

The block below lists the hardware, driver, CUDA and engine versions exactly as
the artifacts recorded them. If any of these differ across artifacts, a warning
appears at the top, because a comparison across two environments measures the
environment rather than the engines (SPEC hard rule 5).

<!-- BEGIN GENERATED: metadata -->
TODO: GPU, driver, CUDA and engine versions render from artifacts once they exist.
<!-- END GENERATED: metadata -->

## W1 — closed-loop concurrency sweep

**Workload.** A fixed number of requests is kept in flight (1, 2, 4, 8, 16, 32,
64), each with a 512-token prompt and exactly 128 output tokens. Closed loop
means a new request starts only when one finishes, so this measures raw
throughput and latency at a given level of parallelism.

<!-- BEGIN GENERATED: w1_table -->
Cells: `mean ± sample stdev (min–max)` across valid repetitions. `TODO` = no valid artifact. Bold brackets are flags: `n=k<3` too few reps, `CV x%` spread above the threshold, `k invalid excluded` runs dropped from the aggregate (listed in the run inventory), `anomaly: kind` reported by the harness. `n/a (not exposed)` = the engine reported `null` for that counter.

**W1 — output tok/s — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| concurrency=1 | TODO | TODO | TODO |
| concurrency=2 | TODO | TODO | TODO |
| concurrency=4 | TODO | TODO | TODO |
| concurrency=8 | TODO | TODO | TODO |
| concurrency=16 | TODO | TODO | TODO |
| concurrency=32 | TODO | TODO | TODO |
| concurrency=64 | TODO | TODO | TODO |

**W1 — output tok/s — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| concurrency=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=16 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=32 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=64 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W1 — TTFT p99 (ms) — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| concurrency=1 | TODO | TODO | TODO |
| concurrency=2 | TODO | TODO | TODO |
| concurrency=4 | TODO | TODO | TODO |
| concurrency=8 | TODO | TODO | TODO |
| concurrency=16 | TODO | TODO | TODO |
| concurrency=32 | TODO | TODO | TODO |
| concurrency=64 | TODO | TODO | TODO |

**W1 — TTFT p99 (ms) — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| concurrency=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=16 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=32 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=64 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W1 — TPOT p99 (ms) — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| concurrency=1 | TODO | TODO | TODO |
| concurrency=2 | TODO | TODO | TODO |
| concurrency=4 | TODO | TODO | TODO |
| concurrency=8 | TODO | TODO | TODO |
| concurrency=16 | TODO | TODO | TODO |
| concurrency=32 | TODO | TODO | TODO |
| concurrency=64 | TODO | TODO | TODO |

**W1 — TPOT p99 (ms) — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| concurrency=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=16 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=32 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| concurrency=64 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- END GENERATED: w1_table -->

**Figure W1-a: output tokens per second vs concurrency.**

<!-- BEGIN GENERATED: fig_w1_throughput -->
TODO: figure `w1_output_tok_s_vs_concurrency.png` renders once valid artifacts exist.
<!-- END GENERATED: fig_w1_throughput -->

**Figure W1-b: TTFT p99 vs concurrency.** TTFT is time to first token.

<!-- BEGIN GENERATED: fig_w1_ttft -->
TODO: figure `w1_ttft_p99_vs_concurrency.png` renders once valid artifacts exist.
<!-- END GENERATED: fig_w1_ttft -->

## W2 — open-loop mixed lengths

**Workload.** Prompt lengths follow a lognormal distribution and output lengths
have a long tail. Requests arrive at a fixed offered rate regardless of how fast
the server answers (open loop), swept across rates. This is the workload where
goodput (requests per second that meet the frozen SLO) is the headline metric.

<!-- BEGIN GENERATED: w2_table -->
Cells: `mean ± sample stdev (min–max)` across valid repetitions. `TODO` = no valid artifact. Bold brackets are flags: `n=k<3` too few reps, `CV x%` spread above the threshold, `k invalid excluded` runs dropped from the aggregate (listed in the run inventory), `anomaly: kind` reported by the harness. `n/a (not exposed)` = the engine reported `null` for that counter.

**W2 — goodput rps — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

**W2 — goodput rps — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W2 — SLO attainment (fraction) — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

**W2 — SLO attainment (fraction) — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W2 — TTFT p99 (ms) — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

**W2 — TTFT p99 (ms) — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W2 — TPOT p99 (ms) — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

**W2 — TPOT p99 (ms) — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- END GENERATED: w2_table -->

## W3 — shared system prefix

**Workload.** 80% of requests share one long common system prefix. Every engine
runs once with its prefix cache on and once with it off, so the cache's effect
is measured inside each engine rather than across engines.

<!-- BEGIN GENERATED: w3_table -->
Cells: `mean ± sample stdev (min–max)` across valid repetitions. `TODO` = no valid artifact. Bold brackets are flags: `n=k<3` too few reps, `CV x%` spread above the threshold, `k invalid excluded` runs dropped from the aggregate (listed in the run inventory), `anomaly: kind` reported by the harness. `n/a (not exposed)` = the engine reported `null` for that counter.

**W3 — TTFT p99 (ms) — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

**W3 — TTFT p99 (ms) — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W3 — goodput rps — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

**W3 — goodput rps — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W3 — prefix hit rate — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

**W3 — prefix hit rate — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- END GENERATED: w3_table -->

**Figure W3: TTFT p99 with the prefix cache on vs off, per engine.**

<!-- BEGIN GENERATED: fig_w3_prefix -->
TODO: figure `w3_ttft_p99_prefix_on_off.png` renders once valid artifacts exist.
<!-- END GENERATED: fig_w3_prefix -->

## W4 — forced preemption at an equal KV pool

**Workload.** Many long sequences run concurrently while every engine is given
the same KV-cache pool size in tokens. The pool is too small to hold them all,
so each engine must preempt (pause and evict) some sequences. This isolates the
preemption policy and its cost.

<!-- BEGIN GENERATED: w4_table -->
Cells: `mean ± sample stdev (min–max)` across valid repetitions. `TODO` = no valid artifact. Bold brackets are flags: `n=k<3` too few reps, `CV x%` spread above the threshold, `k invalid excluded` runs dropped from the aggregate (listed in the run inventory), `anomaly: kind` reported by the harness. `n/a (not exposed)` = the engine reported `null` for that counter.

**W4 — goodput rps — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

**W4 — goodput rps — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W4 — preemptions — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

**W4 — preemptions — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W4 — TTFT p99 (ms) — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

**W4 — TTFT p99 (ms) — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W4 — failed requests — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

**W4 — failed requests — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- END GENERATED: w4_table -->

**Figure W4: goodput and preemption count vs offered load.**

<!-- BEGIN GENERATED: fig_w4_goodput -->
TODO: figure `w4_goodput_and_preemptions.png` renders once valid artifacts exist.
<!-- END GENERATED: fig_w4_goodput -->

## Appendix — W5: our engine, int8 vs fp16

This appendix is about our engine only. It is not part of the cross-engine
comparison.

<!-- BEGIN GENERATED: w5_table -->
Arm ids assumed: `ours` (fp16) vs `ours-int8` (int8), run on W1 and W2. SPEC does not yet name the int8 arm; if the harness uses another id, update `INT8_ARM` in `bench/xengine/render.py`.

Cells: `mean ± sample stdev (min–max)` across valid repetitions. `TODO` = no valid artifact. Bold brackets are flags: `n=k<3` too few reps, `CV x%` spread above the threshold, `k invalid excluded` runs dropped from the aggregate (listed in the run inventory), `anomaly: kind` reported by the harness. `n/a (not exposed)` = the engine reported `null` for that counter.

**W5 on W1 — output tok/s**

| point | ours | ours-int8 |
|---|---|---|
| concurrency=1 | TODO | TODO |
| concurrency=2 | TODO | TODO |
| concurrency=4 | TODO | TODO |
| concurrency=8 | TODO | TODO |
| concurrency=16 | TODO | TODO |
| concurrency=32 | TODO | TODO |
| concurrency=64 | TODO | TODO |

**W5 on W1 — TTFT p99 (ms)**

| point | ours | ours-int8 |
|---|---|---|
| concurrency=1 | TODO | TODO |
| concurrency=2 | TODO | TODO |
| concurrency=4 | TODO | TODO |
| concurrency=8 | TODO | TODO |
| concurrency=16 | TODO | TODO |
| concurrency=32 | TODO | TODO |
| concurrency=64 | TODO | TODO |

**W5 on W2 — goodput rps**

| point | ours | ours-int8 |
|---|---|---|
| TODO | TODO | TODO |
<!-- END GENERATED: w5_table -->

## Run inventory

Every artifact the renderer found, including invalid runs. Invalid runs are
excluded from every aggregate above but are listed here with the reason.

<!-- BEGIN GENERATED: inventory -->
TODO: no artifacts under `results/xengine/` yet.
<!-- END GENERATED: inventory -->

## Analysis

Each subsection below follows the same shape. The **Observation** states what
the tables show and stays `TODO` until the data exists. The **Mechanistic
hypothesis** is the proposed cause. The **Source evidence** cites the engine
code that supports or refutes it. Every hypothesis stays marked UNVERIFIED until
Ashwin has checked it against both the data and the source.

### (a) Raw throughput gap and its attribution

**Observation.** TODO — fill from the W1 tables and Figure W1-a once artifacts
exist. State the gap between `ours` and each baseline at each concurrency.

**Mechanistic hypothesis.** The gap is attributed with an ablation ladder: each
diagnostic arm turns off one feature of a baseline engine, and the change in
throughput is that feature's contribution. Three causes are expected:

1. **CUDA graphs.** A CUDA graph records a sequence of GPU kernel launches once
   and replays it, removing per-step CPU launch overhead. Measured by `vllm` vs
   `vllm-eager` and `sglang` vs `sglang-eager`.
2. **Fused kernels.** vLLM and SGLang combine several operations (for example,
   normalization plus a matrix multiply) into one kernel. Our forward pass uses
   separate PyTorch ops. There is no off-switch for this, so it is estimated
   only as the residual gap after the other causes are removed, and labelled as
   such.
3. **Python overhead in our forward pass.** Per-step scheduling and tensor
   bookkeeping in Python add CPU time that is visible at small batch sizes.

**Source evidence.** TODO: cite (`SOURCE_NOTES.md` — vLLM CUDA-graph capture,
SGLang CUDA-graph runner, fused kernel entry points).

**Status: UNVERIFIED — Ashwin to confirm.**

### (b) Prefix cache: our radix trie vs SGLang RadixAttention vs vLLM hash-block caching

**Observation.** TODO — fill from the W3 tables and Figure W3.

**Mechanistic hypothesis.** All three engines reuse the KV cache of a shared
prefix, but they index it differently. Our engine and SGLang's RadixAttention
both keep a radix trie (a tree keyed by token sequences, where shared prefixes
share nodes). vLLM hashes fixed-size blocks of tokens, so a prefix is reusable
only in whole blocks. The expected consequence is that the trie can match at
token granularity while the hash scheme matches at block granularity; on W3's
single long prefix this difference may be small, and the larger effect may come
from eviction policy under memory pressure.

**Source evidence.** TODO: cite (`SOURCE_NOTES.md` — SGLang radix cache, vLLM
block hashing and prefix-cache lookup; our trie in `serving/`).

**Status: UNVERIFIED — Ashwin to confirm.**

### (c) Preemption policy and its cost under W4

**Observation.** TODO — fill from the W4 tables and Figure W4.

**Mechanistic hypothesis.** When the KV pool is full, an engine must choose a
victim sequence and either discard its KV cache (recompute it later) or copy it
out (swap). The choice of victim and the recompute-vs-swap decision determine
how much work is thrown away. The cost shows up as lower goodput and a heavier
TTFT tail as preemption count rises.

**Source evidence.** TODO: cite (`SOURCE_NOTES.md` — vLLM scheduler preemption
path, SGLang retraction policy; our preemption in `serving/`).

**Status: UNVERIFIED — Ashwin to confirm.**

### (d) Scheduler and batching policy effects on the TTFT tail

**Observation.** TODO — fill from the TTFT p99 tables for W1 and W2.

**Mechanistic hypothesis.** Two scheduling features are expected to shape the
TTFT tail. **Chunked prefill** splits a long prompt into pieces processed
alongside ongoing decodes, so one long prompt does not stall everyone else.
**SGLang's overlap scheduler** prepares the next batch on the CPU while the GPU
runs the current one, hiding scheduling time. The `sglang-nooverlap` arm
measures the second directly.

**Source evidence.** TODO: cite (`SOURCE_NOTES.md` — vLLM chunked prefill
budget, SGLang overlap event loop).

**Status: UNVERIFIED — Ashwin to confirm.**

### (e) Scaling shape normalized to each engine's own batch-1 capacity

<!-- BEGIN GENERATED: w1_scaling -->
**DERIVED, not measured.** Each cell is `mean(output tok/s at c) / mean(output tok/s at c=1)` for the same arm, both means aggregated from W1 artifacts in the tables above. It is a ratio of two measured cells and nothing else: no interpolation, no fitting. `TODO` = either source cell has no valid artifact. Flags from either source cell are carried over.

| point | ours | vllm | sglang |
|---|---|---|---|
| concurrency=1 | TODO | TODO | TODO |
| concurrency=2 | TODO | TODO | TODO |
| concurrency=4 | TODO | TODO | TODO |
| concurrency=8 | TODO | TODO | TODO |
| concurrency=16 | TODO | TODO | TODO |
| concurrency=32 | TODO | TODO | TODO |
| concurrency=64 | TODO | TODO | TODO |
<!-- END GENERATED: w1_scaling -->

**Observation.** TODO — fill from the derived table above once artifacts exist.

**Mechanistic hypothesis.** This comparison is retained from ADR-013. Dividing
each engine's throughput by its own batch-1 throughput removes kernel quality
from the comparison and leaves the scheduling design: does each engine's curve
bend at the same relative load? The table above is the one place in this
document where a value is derived rather than aggregated; it is labelled as such
and is a plain ratio of two measured cells.

**Source evidence.** TODO: cite (ADR-013 in `docs/ADR.md`; `SOURCE_NOTES.md`
for each engine's batch-size limits).

**Status: UNVERIFIED — Ashwin to confirm.**

## Five interview questions

Each question points at the section, table or figure that will answer it.

1. **Why is vLLM faster, and how much of the gap comes from each cause?**
   See Analysis (a), the W1 tables (baseline and diagnostic arms), and
   Figure W1-a. The `vllm-eager` and `sglang-eager` columns give the CUDA-graph
   share.
2. **How does our prefix cache compare with RadixAttention, and when does each
   win?** See Analysis (b), the W3 tables (including `prefix hit rate`) and
   Figure W3.
3. **How does each engine choose preemption victims, and what does it cost?**
   See Analysis (c), the W4 tables (`preemptions`, `goodput`) and Figure W4.
4. **How was the comparison kept fair?** See the Setup metadata block (one GPU,
   one allocation, recorded versions), the Run inventory, [`SPEC.md`](SPEC.md)
   fixed parameters (one frozen SLO, same seed, same model files), and the
   separation of diagnostic arms from baselines in every table.
5. **What was broken or underdocumented in vLLM and SGLang?** See
   `bench/oss-pr-candidates.md` (friction log from setting the engines up) and
   [`SOURCE_NOTES.md`](SOURCE_NOTES.md).
