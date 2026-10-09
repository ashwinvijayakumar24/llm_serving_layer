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
Cells: `mean ± sample stdev (min–max)` across valid repetitions. `TODO` = no valid artifact. Bold brackets are flags: `n=k<3` too few reps, `CV x%` spread above the threshold, `p99 from n=k<100` tail backed by too few samples to be stable, `k invalid excluded` runs dropped from the aggregate (listed in the run inventory), `anomaly: kind` reported by the harness. `n/a (not exposed)` = the engine reported `null` for that counter.

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
Cells: `mean ± sample stdev (min–max)` across valid repetitions. `TODO` = no valid artifact. Bold brackets are flags: `n=k<3` too few reps, `CV x%` spread above the threshold, `p99 from n=k<100` tail backed by too few samples to be stable, `k invalid excluded` runs dropped from the aggregate (listed in the run inventory), `anomaly: kind` reported by the harness. `n/a (not exposed)` = the engine reported `null` for that counter.

**W2 — goodput rps — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| rate=1 | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO |
| rate=32 | TODO | TODO | TODO |

**W2 — goodput rps — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| rate=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=32 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W2 — SLO attainment (fraction) — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| rate=1 | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO |
| rate=32 | TODO | TODO | TODO |

**W2 — SLO attainment (fraction) — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| rate=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=32 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W2 — TTFT p99 (ms) — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| rate=1 | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO |
| rate=32 | TODO | TODO | TODO |

**W2 — TTFT p99 (ms) — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| rate=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=32 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W2 — TPOT p99 (ms) — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| rate=1 | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO |
| rate=32 | TODO | TODO | TODO |

**W2 — TPOT p99 (ms) — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| rate=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=32 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- END GENERATED: w2_table -->

## W3 — shared system prefix

**Workload.** 80% of requests share one long common system prefix. Every engine
runs once with its prefix cache on and once with it off, so the cache's effect
is measured inside each engine rather than across engines.

<!-- BEGIN GENERATED: w3_table -->
Cells: `mean ± sample stdev (min–max)` across valid repetitions. `TODO` = no valid artifact. Bold brackets are flags: `n=k<3` too few reps, `CV x%` spread above the threshold, `p99 from n=k<100` tail backed by too few samples to be stable, `k invalid excluded` runs dropped from the aggregate (listed in the run inventory), `anomaly: kind` reported by the harness. `n/a (not exposed)` = the engine reported `null` for that counter.

**W3 — TTFT p99 (ms) — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| rate=1 | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO |

**W3 — TTFT p99 (ms) — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| rate=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W3 — goodput rps — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| rate=1 | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO |

**W3 — goodput rps — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| rate=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W3 — prefix hit rate — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| rate=1 | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO |

**W3 — prefix hit rate — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| rate=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=16 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
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
Cells: `mean ± sample stdev (min–max)` across valid repetitions. `TODO` = no valid artifact. Bold brackets are flags: `n=k<3` too few reps, `CV x%` spread above the threshold, `p99 from n=k<100` tail backed by too few samples to be stable, `k invalid excluded` runs dropped from the aggregate (listed in the run inventory), `anomaly: kind` reported by the harness. `n/a (not exposed)` = the engine reported `null` for that counter.

**W4 — goodput rps — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| rate=1 | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO |

**W4 — goodput rps — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| rate=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W4 — preemptions — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| rate=1 | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO |

**W4 — preemptions — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| rate=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W4 — TTFT p99 (ms) — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| rate=1 | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO |

**W4 — TTFT p99 (ms) — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| rate=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**W4 — failed requests — baseline arms**

| point | ours | vllm | sglang |
|---|---|---|---|
| rate=1 | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO |

**W4 — failed requests — diagnostic arms** (attribution only; never a competitor baseline)

| point | ours-noprefix | vllm-eager | vllm-nograph | vllm-noasync | vllm-noprefix | vllm-matched | sglang-noradix | sglang-nooverlap | sglang-eager | sglang-lpm |
|---|---|---|---|---|---|---|---|---|---|---|
| rate=1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| rate=8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- END GENERATED: w4_table -->

**Figure W4: goodput and preemption count vs offered load.**

<!-- BEGIN GENERATED: fig_w4_goodput -->
TODO: figure `w4_goodput_and_preemptions.png` renders once valid artifacts exist.
<!-- END GENERATED: fig_w4_goodput -->

## Appendix — W5: our engine, int8 vs fp16

This appendix is about our engine only. It is not part of the cross-engine
comparison.

<!-- BEGIN GENERATED: w5_table -->
`ours` (fp16) vs `ours-int8` (int8 weight-only, KV pool pinned to the fp16 pool), both on W1 and W2 (`make bench-w5`).

Cells: `mean ± sample stdev (min–max)` across valid repetitions. `TODO` = no valid artifact. Bold brackets are flags: `n=k<3` too few reps, `CV x%` spread above the threshold, `p99 from n=k<100` tail backed by too few samples to be stable, `k invalid excluded` runs dropped from the aggregate (listed in the run inventory), `anomaly: kind` reported by the harness. `n/a (not exposed)` = the engine reported `null` for that counter.

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
| rate=1 | TODO | TODO |
| rate=2 | TODO | TODO |
| rate=4 | TODO | TODO |
| rate=8 | TODO | TODO |
| rate=16 | TODO | TODO |
| rate=32 | TODO | TODO |
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
throughput is that feature's contribution to *that engine's* speed. Expected
causes:

1. **CUDA graphs.** A CUDA graph records a sequence of GPU kernel launches once
   and replays it, removing per-step CPU launch overhead. `vllm` vs
   `vllm-nograph` isolates graphs alone; `sglang` vs `sglang-eager` does the
   same for SGLang.
2. **torch.compile fusions (vLLM).** `--enforce-eager` turns off compile *and*
   graphs together, so `vllm-nograph` vs `vllm-eager` isolates the compiled
   fusions.
3. **Hand-fused kernels.** SGLang ships fused RMSNorm and activation kernels;
   ours runs separate PyTorch ops with separate Q/K/V and gate/up matmuls. There
   is no off-switch, so this is only the residual after the measured causes, and
   is labelled as a residual, not a measurement.
4. **Python and synchronization overhead in ours.** A per-layer Python loop, a
   blocking `.tolist()` every step, and the step running synchronously on the
   server's event loop. Expected to matter most at low concurrency.
5. **CPU/GPU overlap.** Both vLLM (async scheduling) and SGLang (overlap loop)
   prepare step N+1 on the CPU while the GPU runs step N; ours does not.
   `vllm-noasync` and `sglang-nooverlap` measure it.

**Source evidence.** `SOURCE_NOTES.md` §6(a), §2.5, §3.6, §4.5. Ours: no graphs
(`serving/backends/flashinfer_backend.py:148`), unfused ops
(`engine/components_gpu.py:27-31`, `:140-147`, `:214-223` in the vendored
engine), Python layer loop (`engine/model_gpu.py:102-170`), blocking
`.tolist()` (`serving/scheduler/scheduler.py:757`), synchronous step
(`serving/server/app.py:498`). vLLM 0.31.0: compile and graphs on by default,
and `--enforce-eager` disables both (`vllm/config/vllm.py:316`, `:1694-1696`,
`:1774-1778`). SGLang 0.5.21: graphs for decode and prefill
(`cuda_graph_config.py:157-161`), fused ops (`layernorm.py:102`,
`activation.py:62`). The *mechanism differences* are verified in source; that
they *account for* the gap is what the arms measure.

**Status: UNVERIFIED — Ashwin to confirm.**

### (b) Prefix cache: our radix trie vs SGLang RadixAttention vs vLLM hash-block caching

**Observation.** TODO — fill from the W3 tables and Figure W3.

**Mechanistic hypothesis.** All three engines reuse a shared prefix's KV cache
by default, but at different granularity. Ours (a trie with one 16-token block
per edge) and vLLM (a chained hash per full 16-token block) both match only
whole blocks; SGLang's radix tree uses page size 1 and matches single tokens.
On W3's single long system prefix, block alignment wastes at most 15 tokens per
request, so granularity should matter little. The larger differences are
expected from (1) **when eviction runs** — ours only at admission, vLLM on every
block allocation, SGLang on demand including before retraction — and (2)
**schedule order**: SGLang's longest-prefix-match ordering is *off by default*
(FCFS); `sglang-lpm` turns it on to measure it separately.

A correction to the original plan: "SGLang's radix cache plus longest-prefix
scheduling beats both" assumed LPM is the default. Source shows it is not, so
the default-vs-default comparison is FCFS everywhere.

**Source evidence.** `SOURCE_NOTES.md` §6(b), §2.3, §3.3, §3.4, §4.3, §5.
SGLang default policy FCFS (`fields/schedule.py:82-98`); LPM sort and its
fallback to FCFS above 128 waiting requests (`schedule_policy.py:331-342`,
`:373-431`); token granularity (`overrides.py:1278-1302`). vLLM block hashing
(`v1/core/kv_cache_utils.py:684-714`) and FCFS queue (`request_queue.py:75`).
Ours: `serving/cache/radix.py` (LRU eviction `_evict_one` at `:546`). Caveat on
the hit-rate column: vLLM and SGLang count tokens, ours counts blocks, and
vLLM's counters skip re-lookups by resumed (preempted) requests.

**Status: UNVERIFIED — Ashwin to confirm.**

### (c) Preemption policy and its cost under W4

**Observation.** TODO — fill from the W4 tables and Figure W4.

**Mechanistic hypothesis.** When the KV pool is full an engine picks a victim
and discards its KV cache to recompute later (all three default to recompute;
only ours also offers swap). Victim choice differs: ours takes the newest
arrival with a starvation guard, vLLM the last-admitted running request, SGLang
the request with the fewest generated tokens. Discarded work, and therefore
goodput loss, should scale with how much KV the victim had built. A second
expected effect is specific to ours (F-004, unreproduced): preemption may fire
while unreferenced cached blocks could have been evicted instead, which W4 would
show as more preemptions for `ours` than `ours-noprefix` would need.

**Source evidence.** `SOURCE_NOTES.md` §6(c), §2.4, §3.5, §4.2. vLLM victim
selection (`v1/core/sched/scheduler.py:763-771`) and recompute-only preemption
(`:1538-1581`; no swap mode exists in V1) — **verified**. SGLang retraction
(`schedule_batch.py:2239-2274`). Ours: LIFO with starvation guard (ADR-024,
`serving/scheduler/preemption.py:130`).

**Status: UNVERIFIED — Ashwin to confirm** (vLLM's victim rule is verified in
source; the cost comparison is not).

### (d) Scheduler and batching policy effects on latency tails

**Observation.** TODO — fill from the TTFT p99 and TPOT p99 tables for W1 and W2.

**Mechanistic hypothesis.** **Chunked prefill** splits a long prompt into
pieces processed alongside ongoing decodes. All three engines use it, with very
different budgets: ours 512 tokens, vLLM 2048–16384, SGLang 4096–16384. Its
main effect is expected on *decode* latency (TPOT/ITL), not TTFT — vLLM's own
docs say smaller budgets can worsen TTFT. So the prediction is that ours shows a
tighter TPOT tail but a heavier TTFT tail under long prompts than its raw speed
alone would suggest. SGLang also runs prefill and decode as separate batches by
default, while ours and vLLM mix them in one step.

**Source evidence.** `SOURCE_NOTES.md` §6(d), §2.1, §3.7, §4.1. vLLM running
requests first and chunk sizing (`v1/core/sched/scheduler.py:629-630`, `:1110`,
`:1123`); SGLang prefill-first and one chunked request at a time
(`scheduler.py:3746-3748`, `schedule_policy.py:1516`); ours decodes first with a
512-token prefill budget (`serving/scheduler/scheduler.py:612-694`). Mechanisms
verified; effect on tails unverified.

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

**Source evidence.** ADR-013 and ADR-025 in `docs/ADR.md`. Each engine's
batch ceiling bounds where its curve can bend: ours `max_batch_size=32`
(`serving/scheduler/scheduler.py:209`); vLLM and SGLang derive `max_num_seqs` /
`max_running_requests` from GPU memory (`SOURCE_NOTES.md` §2.7, §3.8), so the
resolved values recorded per run are needed to read this table. Ours capping at
32 means its curve must flatten by concurrency 32 regardless of kernel speed.

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
