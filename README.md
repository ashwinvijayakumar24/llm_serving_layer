# llm_serving_layer

A from-scratch LLM serving system — paged KV cache, continuous batching, preemption under memory pressure, radix prefix caching, and prefix-aware routing across GPU replicas — built over [`llm_inference_engine`](https://github.com/ashwinvijayakumar24/llm_inference_engine), a single-request Llama 3.2 1B implementation.

Benchmarked on NVIDIA A100 40GB and H100 80GB (Georgia Tech PACE Phoenix). Every number below resolves to a committed artifact in [`results/`](results/) carrying its Slurm allocation id, GPU, seed, git SHA, and pinned engine tag.

---

## Why this exists

The engine underneath is correct and fast for **one request at a time**, and structurally incapable of anything else:

- **Batch 1 by construction** — no batch dimension in the tensors, the KV cache, the causal mask, or the CUDA kernel (`BENCHMARKS.md:248` states this as a known gap).
- **Its HTTP server serializes** — a blocking synchronous generator inside an `async def` (`engine/server.py:69`), so request 2's TTFT contains request 1's entire decode.
- **Memory sized for the worst case** — a fresh `max_seq=2048` KV cache per request, dropped on return (`engine/scheduler.py:26-27`). A 30-token request reserves 2048 slots × 16 layers.

Everything between "a model that runs" and "a system that serves" is this repo.

---

## Results

> **Read the tables, not the headline numbers.** Several results here are functions of workload, and a single row is a choice of operating point rather than a measurement. Where that is true it is said explicitly.

### Concurrent-sequence capacity — paged vs contiguous (S1)

Measured by driving the real allocator to admission failure. Baseline reserves `max_seq=2048` per request, as the engine does.

| realized mean length | paged | contiguous | ratio |
|---|---|---|---|
| 32 | 29,618 | 571 | **51.9×** |
| 128 | 8,615 | 571 | 15.1× |
| 258 | 4,406 | 571 | 7.7× |
| 505 | 2,280 | 571 | 4.0× |
| 1,420 | 820 | 571 | **1.4×** |

The ratio is approximately `max_seq / realized_padded_length`. **It is not a property of the allocator alone**, and it falls to ~1× when sequences genuinely use all 2048 slots. The defensible single statement names its distribution: *7.7× at realized mean 258 tokens (p90 524)*.

*Artifacts: [`results/p1/`](results/p1/) · job `11598444`, H100, engine `v0.2.1`.*

### Correctness chain

Every layer is verified against the one below it, on real weights, before the next is built:

```
FlashInferBackend  ==  PagedTorchBackend     28 differential tests   job 11598894
PagedTorchBackend  ==  contiguous engine     16 tests                job 11598444
contiguous engine  ==  fp32 HuggingFace      6 tests                 job 11596894
batched output     ==  single-sequence       9 tests                 job 11598894
```

Before this project, the engine had **no model-level correctness test for its GPU path at all** — `tests/test_gpu_model.py:42-45` asserted only that logits were finite, correctly shaped, and had an in-range argmax. Correctness was measured on the CPU fp32 path; performance on the GPU fp16 path. Closing that gap was Phase 0.

*Further results — continuous batching goodput, preemption, cache hit rate, routing — in [`results/`](results/) and summarised in [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md).*

---

## Design

| Document | What it settles |
|---|---|
| [`docs/PRD.md`](docs/PRD.md) | Goals, non-goals, success criteria, hardware envelope |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Components, the engine↔serving interface, state ownership, concurrency, failure domains, sequence walkthroughs |
| [`docs/BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md) | Open-loop load generation, goodput under SLO, baselines, **and where prefix-aware routing is predicted to lose** |
| [`docs/RISK_REGISTER.md`](docs/RISK_REGISTER.md) | 41 risks ordered by **detectability**, because a silent wrong number is worse than a crash |
| [`docs/ADR.md`](docs/ADR.md) | 24 decision records with alternatives and revisit triggers |
| [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) | How it was built, including what went wrong |
| [`docs/LEARNING_MAP.md`](docs/LEARNING_MAP.md) | Concepts, canonical papers, review questions per phase |

Three decisions worth arguing with:

**Batched decode and paged KV are one change, not two.** With a flattened variable-length token layout, every position-independent operation — all linears, both norms, the MLP, the embedding gather — is *literally unchanged*, and RoPE is already batch-ready because positions arrive as a tensor. Only attention, the cache write, the causal mask, and the last-token gather differ.

**The custom CUDA kernel is not in the paged path.** The engine's nanobind ABI takes no stride, block-table, or block-size arguments, so a paged cache cannot be described across it. The kernel remains the single-replica contiguous reference path. This project **integrates** FlashInfer's paged kernels behind a pluggable backend and **wrote** a PyTorch reference implementation as the correctness oracle — never the other way around.

**The router holds hints; the replica holds truth.** All router state is advisory. A stale hint costs cache locality, never correctness — which is why there is no consensus layer, no shared cache, and no distributed transaction anywhere in the design.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
   clients ────────▶│  ROUTER          prefix-aware selection  │
   (OpenAI API)     │                  health · drain · failover│
                    └───────┬──────────┬──────────┬────────────┘
              ┌─────────────▼──┐  ┌────▼───────┐  ┌▼─────────────┐
              │ REPLICA 0      │  │ REPLICA 1  │  │ REPLICA N-1  │
              │ 1 GPU          │  │ 1 GPU      │  │ 1 GPU        │
              └────────────────┘  └────────────┘  └──────────────┘

   inside a replica
   ┌──────────────────────────────────────────────────────────┐
   │ HTTP ingress (FastAPI, async) · SSE · cancellation        │
   │ ADMISSION CONTROL   queue depth · memory headroom · shed  │
   │ SCHEDULER           continuous batching · preemption      │
   │ RADIX PREFIX CACHE  │  BLOCK ALLOCATOR  free list · COW   │
   │ ATTENTION BACKEND   PagedTorch (oracle) | FlashInfer      │
   │ ENGINE              LlamaModelGPU (pinned dependency)     │
   └──────────────────────────────────────────────────────────┘
```

```
serving/
  memory/      block allocator, block tables, KV pool sizing
  cache/       radix prefix trie, refcounting, LRU, copy-on-write
  scheduler/   continuous batching, admission, preemption
  backends/    PagedTorchBackend (authored) | FlashInferBackend (integrated)
  engine_iface/  BatchMeta assembly, varlen batching
  server/      replica HTTP surface
  router/      routing policies, health, drain, failover
  metrics/     artifact schema with provenance, Prometheus, CUDA guard
bench/         open-loop load generator, workloads, per-phase drivers
results/       COMMITTED artifacts — every published number resolves here
```

---

## Benchmark methodology

The parts that decide whether a number means anything:

- **Open loop.** A closed-loop harness cannot create a queue deeper than its client count, so it hides the saturation knee — and the knee is the result.
- **Latency measured from *intended* dispatch**, never actual send. Otherwise a saturated system reports excellent p99 and nothing errors. The guard is mutation-tested: rewriting it to use actual send time fails exactly the two coordinated-omission tests and nothing else.
- **Goodput under a declared SLO** is the headline; raw throughput is gameable in both directions. SLO thresholds are anchored to measured unloaded performance by multipliers fixed in source *before* any loaded run.
- **Artifacts store raw samples**, not pre-computed percentiles, so percentiles can be pooled correctly at analysis time.
- **Every A/B runs back-to-back in one Slurm allocation.** The engine's own throughput moved ~79 → ~60 tok/s across nodes for identical code; a cross-allocation delta measures node assignment. The tooling *refuses* to compare across allocations rather than warning.
- **Losing cases are published.** The workloads where prefix-aware routing should lose were predicted in writing before measurement.

---

## Setup

```bash
git clone --recurse-submodules https://github.com/ashwinvijayakumar24/llm_serving_layer.git
cd llm_serving_layer
python -m venv .venv && source .venv/bin/activate
pip install -e vendor/llm_inference_engine
pip install -e ".[dev,bench]"

pytest -m "not gpu"          # CPU suite — no GPU or weights needed
```

The engine is a submodule pinned to a tag. Model weights are gated and live outside the repo; point at them with `LLM_WEIGHTS_PATH`.

Running on Slurm: see [`scripts/`](scripts/).

---

## Testing

CPU tests run without a GPU, weights, or network — the allocator, radix trie, routing policy, batch assembly, workload generation, and metric schema are all pure logic by design, so cluster queue time never blocks development.

Every GPU gate uses `REQUIRE_GPU=1`, which turns an unusable GPU into a **hard failure rather than a skip**. This is not defensive styling: an early run landed on a V100 under a CUDA-13 build, skipped all 16 gate tests, and exited 0 — a skipped gate and a passing gate are indistinguishable in a job log.

## License

MIT
