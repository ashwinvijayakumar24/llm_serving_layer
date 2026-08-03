# LLM Serving Layer

A from-scratch LLM serving system — paged KV cache, continuous batching, preemption under memory pressure, radix prefix caching, and prefix-aware routing across GPU replicas — built over [`llm_inference_engine`](https://github.com/ashwinvijayakumar24/llm_inference_engine), a single-request Llama 3.2 1B implementation.

Benchmarked on NVIDIA H100 80GB and H200 (Georgia Tech PACE Phoenix). Every number below resolves to a committed artifact in [`results/`](results/) carrying its Slurm allocation id, GPU, seed, git SHA, and pinned engine tag.

**Five claims were scoped. Four are earned; the fifth is not, and says so.** The routing benchmark never measured its shared-prefix workloads inside their valid range, so prefix-aware routing is *not* claimed to beat a load-aware baseline — see [Routing](#prefix-aware-routing--not-earned-s5). Publishing that is the point, not an apology for it.

| | claim | result |
|---|---|---|
| **S1** | paged KV capacity | **7.7×** concurrent sequences at fixed VRAM (realized mean 258 tok) |
| **S2** | continuous batching | **1.8×** peak goodput under SLO; TTFT p99 89–113 ms vs 530–801 ms |
| **S3** | preemption | **bit-identical** greedy output under forced memory pressure, both policies |
| **S4** | radix prefix cache | **−23%** TTFT p99 at 27% block reuse; ≤1.3 ms cost at zero sharing |
| **S5** | prefix-aware routing | **NOT EARNED** — one losing-case prediction confirmed instead |

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

### Continuous batching — goodput under SLO (S2)

| | continuous | static batching | ratio |
|---|---|---|---|
| peak goodput (req/s) | **2.29** | 1.27 | **1.8×** |
| goodput at 8 req/s offered | — | — | **13.6×** |
| TTFT p99 across a 12× load range | 89–113 ms | 530–801 ms | — |
| raw throughput | — | — | **identical** |

**Raw tok/s is the same in both arms.** Static batching does not lose throughput, it loses latency, by making every request wait for a whole wave to drain. The baseline is a **one-line change to admission in the same server**, so kernels, memory manager, HTTP stack and model are byte-identical between arms and the entire delta is attributable to scheduling.

*Artifacts: [`results/p2/`](results/p2/) · job `11608159`, H200.*

### Preemption under memory exhaustion (S3)

Recompute and swap-to-host policies, LIFO victim selection, starvation guard. **Greedy output verified bit-identical to an unpreempted control under forced memory pressure, for both policies.** The benchmark also found a real resume-starvation bug: swapped sequences were gated on the admission watermark, which by definition does not apply to them — `serving/scheduler/scheduler.py::_resume_swapped`, with a regression test verified to fail on the prior code.

The **correctness** claim above comes from the gate. The **cost** sweep is only half-valid: recompute measured cleanly at three sequence lengths (latency tax +0.7 to +4.2 ms TTFT p50 versus an unpreempted control, 42,254 tokens recomputed at length 256), but every swap arm failed the steady-state check and is excluded. So the recompute-vs-swap crossover predicted in the methodology is **untested, not confirmed** — no length produced a valid arm of each.

*Artifacts: [`results/p3/`](results/p3/) · gate job `11608158`.*

### Radix prefix cache — TTFT (S4)

Job `11617299`, H200, **one fresh server pair per cell** (36 pairs). Valid cells only: evictions 0, n = 118/118, steady state verified.

| prompt | structure | sharing | block hit rate | Δ TTFT p50 | Δ TTFT p99 |
|---|---|---|---|---|---|
| 512 | system prefix | 50% | 0.137 | −6.3 ms | **−37.7 ms (−23%)** |
| 512 | system prefix | 100% | 0.272 | −5.1 ms | **−38.0 ms (−23%)** |
| 150 | conversational | 100% | 0.762 | **−17.3 ms** (on 64.8 ms) | — |
| 150 | conversational | 50% | 0.536 | −8.5 ms | — |
| any | **zero sharing (control)** | 0% | ~0.01 | **+0.4 … +4.4 ms** | — |

**The win lands in the tail, not the median**, and that is the mechanism rather than a quirk: a prefix hit does not speed up the request that misses — it removes prefill work from the queue, which shortens everyone else's wait. p99 moves 23% in a cell where p50 moves 7%.

**Break-even sharing rate: 0.058 conversational, 0.452 system.** Below it the cache costs more than it saves. A shared system preamble is only ~30 tokens and must clear block granularity before it returns anything, so it needs 45% of traffic to share before it pays for itself. **A speedup quoted without its sharing rate is a choice of workload, not a measurement.**

**1024- and 2048-token prompts are untested.** All 24 cells at those lengths saturated the block pool *within a single cell* and were marked INVALID by the driver. Two earlier runs of this benchmark reported +416 ms and +907 ms there and were wrong — cells shared a server, the trie consumed the pool, and every later request evicted. The failure and its diagnosis are kept in [`results/p4/`](results/p4/) rather than deleted.

*Artifacts: [`results/p4_clean/`](results/p4_clean/) · correctness gate job `11608501`.*

### Prefix-aware routing — NOT EARNED (S5)

Four replicas on four dedicated H200s, three routers over one fleet. **S5 is not claimed.** Three of four scenarios carry a long shared prefix, saturate below the lowest offered load, and produced no usable point — the loads were derived from a lighter scenario's capacity and never re-derived. That is a workload-design failure, not a routing result, and `untested` is not `confirmed`.

What *was* earned is a **losing-case prediction confirmed**: `zero_sharing` was predicted in writing, before measurement, to lose to a load-aware baseline — "there is nothing to be cache-aware about, and any deviation from load-optimal placement is pure loss." It lost, in the predicted direction: **Δgoodput −0.111 at load 4, −0.311 at load 16** vs `least_outstanding`.

Note that prefix-aware *beats* round-robin at load 4 (3.18 vs 3.02) while losing to least-outstanding (3.29). Quoting the round-robin comparison would have produced a win — a win about load balancing, which the real baseline already does, and nothing about prefix awareness.

*Artifacts: [`results/p5/`](results/p5/) · job `11610306`, 4 × H200.*

---

## Design

| Document | What it settles |
|---|---|
| [`docs/PRD.md`](docs/PRD.md) | Goals, non-goals, success criteria, hardware envelope |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Components, the engine↔serving interface, state ownership, concurrency, failure domains, sequence walkthroughs |
| [`docs/BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md) | Open-loop load generation, goodput under SLO, baselines, **and where prefix-aware routing is predicted to lose** |
| [`docs/RISK_REGISTER.md`](docs/RISK_REGISTER.md) | 41 risks ordered by **detectability**, because a silent wrong number is worse than a crash |
| [`docs/ADR.md`](docs/ADR.md) | 24 decision records with alternatives and revisit triggers |
| [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) | How it was built in the order it happened, including what went wrong and what it cost to find out |

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

**774 CPU tests, 100 GPU-gated.** CPU tests run without a GPU, weights, or network — the allocator, radix trie, routing policy, batch assembly, workload generation, and metric schema are all pure logic by design, so cluster queue time never blocks development.

Every GPU gate uses `REQUIRE_GPU=1`, which turns an unusable GPU into a **hard failure rather than a skip**. This is not defensive styling: an early run landed on a V100 under a CUDA-13 build, skipped all 16 gate tests, and exited 0 — a skipped gate and a passing gate are indistinguishable in a job log.

That was the first of **six** failures in this project with the same shape: *something reported success while doing nothing.* A fragmentation test that passed without ever fragmenting the pool. A server returning HTTP 200 and well-formed SSE with zero content. An SLO calibration that produced goodput 0.00 at every rate from a negative TTFT. A summary table printing 0.0 in all seven rows because it read scalar keys that don't exist. And an eviction audit that printed `clean` for all 36 cells because the metrics key was absent and the guard classified *absent* as *zero* — a check added in response to the previous instance, failing the same way.

None raised. Every one was caught by an assertion of a **positive property** — "the GPU is usable", "these pages are non-contiguous", "this response has content", "this latency is physically possible" — never by the absence of an error. It is why [`docs/RISK_REGISTER.md`](docs/RISK_REGISTER.md) orders risks by *detectability* rather than impact, and why the benchmark drivers mark cells `INVALID` and refuse to interpolate rather than reporting a number with a caveat.

All six are written up with their diagnoses in [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) §10.

## License

MIT
