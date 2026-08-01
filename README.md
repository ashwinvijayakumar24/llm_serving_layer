# llm_serving_layer

Concurrent serving over [`llm_inference_engine`](../llm_inference_engine) — paged block KV cache, continuous batching, preemption under memory pressure, radix prefix caching, and prefix-aware routing across GPU replicas.

> **Status: planning complete, implementation not started.** Everything below the Design section describes what will be built. No number appears in this README until it resolves to a committed artifact in `results/`.

The engine is a correct, benchmarked, **single-request** Llama 3.2 1B implementation. It is batch-1 by construction — no batch dimension in the tensors, the KV cache, the causal mask, or the CUDA kernel — and its HTTP server serializes concurrent requests. This repo is everything that has to exist between "a model that runs" and "a system that serves."

---

## Design

| Document | What it settles |
|---|---|
| [`docs/PRD.md`](docs/PRD.md) | Problem, goals, non-goals, success criteria, feature tiers, PACE hardware envelope |
| [`docs/BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md) | Open-loop load generation, goodput under SLO, baselines, **where prefix-aware routing is predicted to lose** |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Component decomposition, the engine↔serving interface, state ownership, concurrency, failure domains, sequence walkthroughs |
| [`docs/PHASE_PLAN.md`](docs/PHASE_PLAN.md) | Dependency-ordered phases, per-phase DoD + benchmark + published claim, cut order |
| [`docs/RISK_REGISTER.md`](docs/RISK_REGISTER.md) | 39 risks ordered by **detectability** — the silent invalidators are the dangerous ones |

### Citation conventions

Design docs cite `file:line` into the engine so every claim is checkable. Two caveats:

- **Line numbers are valid against engine tag `v0.1.0` (`6ff40a1`)** and will drift as the engine changes. The submodule pin is what keeps them meaningful.
- **Some cited engine documents are intentionally not public.** `.gitignore:41-50` in the engine excludes `CLAIMS_AUDIT.md`, `SERVING_INTERFACE.md`, `docs/PACE_RUNBOOK.md`, `docs/PRD.md` and others — they contain account specifics and internal audit notes. Citations to those files **will not resolve in a submodule checkout**. They are recorded for the author's traceability, not as public evidence. Every load-bearing claim in these docs is additionally supported by a citation into tracked source (`engine/*.py`, `kernels/*`, `README.md`, `BENCHMARKS.md`, `docs/BUILD_LOG.md`) or by a direct reading of the code.

Three load-bearing decisions, stated up front because they are the ones worth arguing with:

**The batched forward pass and the paged KV cache are the same change.** With a flattened variable-length token layout, every position-independent op — all linears, both norms, the MLP, the embedding lookup — is untouched, and RoPE is already batch-ready because positions are a passed tensor. Only attention, the cache write, the causal mask, and the last-token gather change. See `ARCHITECTURE.md` §2.5.

**The custom CUDA kernel is not in the paged path.** The engine's nanobind ABI takes no stride, block-table, or block-size arguments, so a paged cache is inexpressible across it. The kernel stays the engine's single-replica contiguous reference path. This project writes a PyTorch paged-attention reference as the correctness oracle and integrates FlashInfer as the fast path — **never claiming authorship of a paged kernel.**

**The router holds hints; the replica holds truth.** Router state is advisory. A stale hint costs cache locality, never correctness — which is why there is no consensus layer, no shared cache, and no distributed transaction anywhere in the design.

---

## Setup

Pins the engine at **`v0.1.0`** (commit `6ff40a1`).

```bash
git init
git submodule add https://github.com/ashwinvijayakumar24/llm_inference_engine.git vendor/llm_inference_engine
cd vendor/llm_inference_engine && git checkout v0.1.0 && cd -

python -m venv .venv && source .venv/bin/activate
pip install -e vendor/llm_inference_engine
pip install -e ".[dev,bench]"
```

The submodule (rather than a plain editable install) is deliberate: the engine's compiled kernel `.so` lands in a gitignored `build/` directory, which sits at a known relative path under a submodule and is genuinely awkward to locate in an installed package. A submodule also pins an exact commit, which is what keeps benchmarks reproducible.

### On PACE

```bash
module load cuda/12.9.1
module load anaconda3          # BEFORE conda activate, every session
conda activate llm

# Development / correctness — FREE, preemptible, 8h cap
salloc --partition=gpu-l40s --gres=gpu:l40s:2 \
       --account=paceship-simpliearn --qos=embers --time=4:00:00

# Published benchmarks — charged, up to 32 GPUs, 3-day limit
salloc --partition=gpu-l40s --gres=gpu:l40s:8 \
       --account=paceship-simpliearn --qos=inferno --time=8:00:00
```

**No published number may come from an `embers` run** — it is preemptible, and preemption truncates a measurement window into something that looks like a completed short run.

---

## Layout

```
serving/
  engine_iface/   BatchMeta construction, varlen batch assembly
  memory/         block allocator, block tables, watermark policy
  cache/          radix prefix trie, refcounting, LRU, copy-on-write
  scheduler/      admission, continuous batching, preemption
  backends/       PagedTorchBackend (authored) | FlashInferBackend (integrated)
  server/         replica HTTP surface, SSE, cancellation
  router/         prefix-aware routing, health, drain, failover
  metrics/        metric definitions with (quantity, unit, source)
bench/
  workloads/      arrival process, length + prefix-sharing distributions
  baselines/      B1..B6
results/          COMMITTED artifacts — deliberately not gitignored
vendor/           engine submodule, pinned to a tag
```

---

## Benchmarking

Read `docs/BENCHMARK_METHODOLOGY.md` before running anything. The short version:

- **Open loop.** A closed-loop harness cannot create a queue deeper than its client count, so it hides the saturation knee entirely — and the knee is the result.
- **Goodput under a declared SLO is the headline**, reported as a curve against offered load, not as a point. Raw throughput is gameable in both directions.
- **Latency is measured from intended dispatch time**, never actual send time. Otherwise a saturated system reports excellent p99 and nothing errors.
- **Every A/B runs back-to-back in one Slurm allocation.** The engine's own throughput moved ~79 → ~60 tok/s across nodes for identical code.
- **Losing results are published**, including the workload classes where prefix-aware routing is expected to lose — predicted in writing before measurement.

---

## Correctness

Every gate exists because a specific failure mode is **silent** — it produces plausible output, degrades no metric, and raises nothing.

| Gate | Catches |
|---|---|
| GPU model oracle | The engine has no model-level GPU correctness test at all; everything else compares against this |
| Batch invariance | Batched output diverging from single-sequence output |
| Preemption equality | Preemption dropping or duplicating tokens under load |
| Cache on/off equality | A prefix cache that is faster *and* wrong |
| Allocator leak | Blocks never returned; free list must return to its initial count exactly |
| Backend differential | FlashInfer layout misunderstanding |

---

## Relationship to the engine

The engine keeps its scope: model internals, the custom CUDA decode kernel, quantization, and a single-request reference server. This repo owns the production serving surface. Engine changes made for this project are **additive** — `prefill()` and `decode_step()` keep byte-identical behavior, so the engine's existing benchmarks and correctness claims are unaffected and its test suite doubles as the regression gate.

## License

MIT
