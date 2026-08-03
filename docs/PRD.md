# PRD — LLM Serving Layer

**Status:** draft 1, 2026-07-30. Planning only; no implementation.
**Depends on:** `llm_inference_engine` @ tag `v0.1.0` (`6ff40a1`).
**Companion docs (not yet written):** benchmark methodology, architecture, phase plan, risk register, ADR log.

Claims about the engine cite `file:line` in that repo. Statements marked **[inference]** are judgment, not read from code.

---

## 1. Problem

The engine at `../llm_inference_engine` is a correct, benchmarked, single-request Llama 3.2 1B implementation. It is also, by construction, unable to serve anyone.

Three facts, from source:

1. **Batch 1 everywhere.** `x` is `(seq, hidden)` throughout (`engine/model_gpu.py:64`, `engine/components_gpu.py:127-130`); the KV cache is `(n_layers, max_seq, n_kv_heads, head_dim)` with no sequence axis (`engine/cache.py:28`); the causal mask is 2-D (`engine/components_gpu.py:173`); the CUDA kernel takes `Q` as `[n_heads, head_dim]` with no batch dimension (`kernels/attention_decode.cu:30-32`). `BENCHMARKS.md:248` states this as a known gap: *"There is no batch dimension in the tensors, the KV cache, the causal mask, or the CUDA kernel. No claim in this document describes behaviour under concurrent load."*

2. **The HTTP server cannot serve the GPU engine and serializes under load.** `engine/server.py:24-30` imports `engine.model.LlamaModel` — the fp32 NumPy CPU path. There is no flag or code path to `LlamaModelGPU`. `engine/server.py:69` iterates the blocking synchronous generator `generate()` inside an `async def` event stream, pinning the event loop for the whole generation. Request 2's TTFT includes request 1's entire decode.

3. **Memory is allocated as if one request exists.** `engine/scheduler.py:16,26-27` constructs a fresh `KVCacheGPU(max_seq=2048)` per `generate()` call and drops it on return. A 30-token request reserves 2048 slots × 16 layers. There is no free, no refcount, no reuse, no eviction.

Every consequence of concurrent serving — queueing, admission control, memory pressure, preemption, cache sharing across requests, load distribution across replicas — is therefore unexplored territory. That territory is the subject of this project.

### Why this project, for this person

The project's lane is **systems engineering applied to AI**: serving infrastructure, agent harnesses, evals, tooling. Not ML research, not model development, not kernel engineering. The gap it exists to close is distributed systems.

This project exists to close that gap with something benchmarkable. It must survive hostile technical review at roughly five distinct claims — which is why every claim in this repo is paired with the condition that makes it true, and why claims that could not be measured cleanly are published as not earned.

---

## 2. Goals

**G1 — Serve concurrent requests with memory that scales with actual use, not worst case.**
A paged block KV cache with an allocator, block tables, and a free list, replacing the per-request `max_seq` over-allocation at `engine/scheduler.py:26-27`. Success is measured as concurrent-sequence capacity at fixed VRAM versus the contiguous baseline.

**G2 — Batch the forward pass across independent sequences at different positions.**
Without this, continuous batching is bookkeeping with no throughput behind it — the engine's own build log says as much: *"true scaling needs a batched-GEMM step"* (`engine:docs/BUILD_LOG.md:1164`). Success is throughput scaling with batch size, reported as a curve, not a point.

**G3 — Schedule at iteration granularity, with preemption that is correct under memory exhaustion.**
Admit and retire requests between decode steps. When blocks run out, evict — and be able to defend recompute-versus-swap with measurements from this system, not from a paper.

**G4 — Share prefixes across requests via a radix cache with reference-counted eviction.**
Success is cache hit rate and its effect on TTFT under a workload with a realistic, *stated* prefix-sharing distribution.

**G5 — Route across N replicas with prefix awareness, beating round-robin on a workload where it should, and losing where it should.**
The losing case must be predicted in writing before it is measured, and published alongside the winning case.

**G6 — Measure all of the above honestly enough that the numbers survive an adversarial reader.**
Open-loop load generation, goodput under a declared SLO as the headline metric, committed result artifacts, a published known-gaps list. This inherits the standard the engine already set in `BENCHMARKS.md:243-250`, including its committed artifacts (24 files under `bench/results/`). The improvement is in *content*, not practice: raw per-request samples rather than pre-computed percentiles, realized workload distributions, and provenance fields sufficient to reject an invalid comparison.

**G7 — Depth of understanding over shipped features.**
Every component is labeled *author-written* or *assistant-written*. A phase is not done until the author can derive its design and tradeoffs from first principles, unaided. Anything that fails that gate is not published as a claim, regardless of whether the code works.

---

## 3. Non-goals

| Not doing | Why |
|---|---|
| **Training, fine-tuning, or model architecture work** | Off-lane. The model is a fixed dependency. |
| **Writing a paged-attention CUDA kernel** | The engine already answers "can he write CUDA" (`kernels/attention_decode.cu`, 311 lines, three staged versions, warp-shuffle reductions, flash combine rule). A second, harder kernel costs the longest pole in the project for a marginal signal in a deprioritized lane. See §7 and the forthcoming ADR. |
| **Forking or vendoring the engine** | It is consumed as a pinned dependency. Changes to it are small, upstreamed, and enumerated in §6. |
| **Replacing the engine's HTTP server** | It stays a single-request reference path, as its README already scopes it (`README.md:144`). The production surface lives here. |
| **SQL / relational persistence** | Deliberately out of scope. Not in the hot path, and nothing in this project's claim set depends on it. One legitimate non-hot-path use exists — persisting benchmark runs for cross-session comparison — and even that is a CSV/JSON directory first, matching `bench/harness.py:187-202`. Revisit only if run comparison becomes genuinely painful. |
| **Kubernetes as a required phase** | Target hardware is GT PACE Phoenix under Slurm. K8s + a cache-aware inference gateway (llm-d, Gateway API Inference Extension) is a *late optional* phase, evaluated on systems depth alone. See §5, Tier 4. |
| **Multi-GPU tensor parallelism** | Different problem (intra-model sharding). This project is inter-request systems work. Would dilute the claim set. |
| **Beating vLLM on absolute throughput** | Not achievable and not the claim. Fairness of any vLLM comparison is a benchmark-methodology question, handled there. |

---

## 4. Success criteria

Measurable, phrased as outcomes rather than features. Numbers are placeholders until measured; **the methodology that makes each one defensible is deliverable #3, not this document.**

| # | Criterion | Measured how | Baseline |
|---|---|---|---|
| S1 | Concurrent sequences served at fixed VRAM increases by ≥ N× | Max admitted sequences before OOM, paged vs contiguous, same GPU, same model | `engine/scheduler.py:26-27` contiguous `max_seq=2048` per request |
| S2 | Throughput scales with batch size | tok/s vs batch size curve, saturation point identified | Batch-1 engine, ~60-79 tok/s A100 (`BENCHMARKS.md:29,140` — note cross-session non-comparability, `BENCHMARKS.md:17`) |
| S3 | Goodput under a stated SLO exceeds a static-batching baseline | Requests/sec meeting `TTFT < X ms AND p95 ITL < Y ms`, open-loop, at each arrival rate | Static batching at the same batch size |
| S4 | Prefix cache hit rate ≥ H% on a workload with declared prefix-sharing structure; TTFT reduction quantified | Hit rate instrumented at block granularity; TTFT p50/p95 with cache on vs off | Cache disabled, same workload, same seed |
| S5 | Prefix-aware routing beats round-robin on shared-prefix workloads **and the predicted losing workload is published and confirmed** | Goodput and TTFT vs round-robin and least-outstanding-requests, ≥3 workload mixes | Round-robin; least-outstanding-requests |
| S6 | Preemption is correct and its cost is characterized | Output token equality vs unpreempted run (bit-identical greedy); preemption rate and latency tax under forced memory pressure | No-preemption run at a batch size that fits |
| S7 | A replica killed mid-benchmark is detected, drained, and re-routed with zero dropped requests | Fault injection during a load run; request completion accounting | — |
| S8 | Every claim above has a committed artifact carrying **raw samples + full provenance**, and a stated known-gaps list | `results/` tracked in git; artifact-schema check in CI | The engine's `bench/results/` (24 committed files) — same practice, but its artifacts store per-run percentiles rather than raw samples (`bench/harness.py:136-137,178`), which makes correct pooling impossible after the fact |
| S9 | Author passes the explainability gate for every shipped phase | Self-assessment against the phase's review question set | — |

S8 and S9 are not decoration. S8 is the difference between a benchmark and a claim; S9 is the difference between a project and a liability.

---

## 5. Feature inventory

Priority tiers. **Time and risk are annotated per feature in the phase plan (deliverable #5), not here** — this is the inventory, not the schedule. "Author" / "Assistant" is the write-ownership label required by G7.

### Tier 0 — Prerequisite engine changes (blocking; see §6)

| Feature | Owner | Notes |
|---|---|---|
| Fix `tests/test_generate.py` oracle key (`"input_ids"` → `"token_ids"`) | Author | All 5 tests currently `KeyError`. The KV-cache-vs-no-cache correctness gate is not actually enforced today. `tests/test_generate.py:42,52,63,78,89` vs `tests/oracle.py:170`. |
| `AttentionBackend` protocol + injection, varlen-batched signature from day one | Author | Generalizes the existing `decode_kernel=` hook (`engine/components_gpu.py:114`) from math-only to cache-owning. Default impl wraps today's cache, bit-identical. |
| Chunked-prefill position fix | Author | `engine/model_gpu.py:65` uses `torch.arange(seq)` starting at 0 regardless of cache position. Latent bug, invisible while prefill is called once. Radix prefix caching depends on it. |
| GPU-path correctness oracle at model level | Author | **Does not exist.** `tests/test_gpu_model.py:42-45` asserts only finite/shape/argmax-in-range. GPU fp16 tokens are never compared to HF or the CPU path. This must exist before any serving claim rests on GPU output. |
| Git tag `v0.1.0` on the engine | Author | **DONE 2026-07-31** — tagged at `6ff40a1` and pushed. The submodule has something to pin to. |
| `--backend gpu` flag on the engine's reference server | Assistant | `engine/server.py:24-30` hardcodes the CPU path. Nice-to-have, not blocking — the serving layer builds its own HTTP surface. |

### Tier 1 — Core (the project is not the project without these)

Paged block KV cache · block allocator with free list · block tables · PyTorch paged-attention reference path (**Author** — the correctness oracle and the hand-written reference implementation) · FlashInfer paged-attention fast path behind the same interface (**Assistant** integration, **Author** must understand the layout contract) · **batched varlen forward pass** (Author) · continuous batching / iteration-level scheduling (Author) · request queue and admission control (Author) · **preemption on KV exhaustion, recompute vs swap** (Author — deepest systems content in the project) · production HTTP surface with SSE streaming (Assistant) · client-disconnect cancellation propagating into the scheduler (Author for the scheduler half) · core metrics: TTFT, TPOT/ITL, throughput, goodput, queue depth, GPU memory, preemption rate (Assistant wiring, Author defines what each means) · open-loop benchmark harness (Assistant scaffold, Author designs the workload model).

### Tier 2 — Differentiating

Radix prefix cache with LRU eviction and reference counting (Author) · copy-on-write for shared prefix blocks (Author) · cache-aware scheduling (Author) · hit-rate instrumentation at block granularity (Assistant) · chunked prefill (Author) · prefix-aware routing across N replicas (Author — routing policy specifically) · round-robin and least-outstanding-requests baselines (Assistant) · health checking, failover, graceful draining (Assistant, Author reviews) · backpressure and load shedding (Author) · fault injection harness (Assistant) · Prometheus + Grafana (Assistant).

### Tier 3 — Strong if reached

Tiered KV offload to CPU · priority and fairness scheduling with starvation prevention · consistent hashing for replica selection · timeouts, jittered retries, circuit breakers, idempotency keys · OpenTelemetry tracing across router → replica → scheduler · speculative decoding with acceptance-rate instrumentation · structured / constrained decoding (the engine's logits-before-sampling seam at `engine/model_gpu.py:90,158` + injected sampler at `engine/scheduler.py:15,34` makes this cheap) · containerization and CI.

### Tier 4 — Optional, evaluated on systems depth alone

Prefill/decode disaggregation · multi-LoRA serving · Kubernetes + cache-aware inference gateway (llm-d / Gateway API Inference Extension as prior art).

**[inference]** My current read: Tier 4 is where scope goes to die. Disaggregation is a genuinely deep systems idea but needs ≥2 GPUs held simultaneously (see §8, open question O1) and lands on top of everything else. K8s is off-lane for Slurm hardware and reads as padding unless the gateway's *cache-aware routing* is the point — in which case it duplicates the router already built in Tier 2. Argued properly in the phase plan.

---

## 6. Engine changes: strictly prerequisite vs nice to have

**Strictly prerequisite** — the serving layer cannot start correctly without these:

1. `test_generate.py` key fix. Restores the only cache-correctness gate.
2. `AttentionBackend` protocol and injection point, with a **varlen-batched signature from day one**. The seam must not be designed batch-1 and widened later; that means doing the surgery twice and shipping a Phase-1 interface that has to break.
3. Chunked-prefill position fix (`engine/model_gpu.py:65`).
4. A model-level GPU correctness oracle. Currently absent.
5. Git tag on the engine.

**Nice to have** — real value, not blocking:

- `--backend gpu` on the engine's reference server (`engine/server.py:24-30`).
- Packaging cleanup: `sys.path` manipulation at `engine/model_gpu.py:45-53` breaks under non-editable install; the built `.so` lands in gitignored `build/`; CUDA arch hardcoded to sm_80 (`scripts/build_kernels.sh:10`), so no H100. The last one becomes prerequisite the moment H100 benchmarking is on the table — see §8, O1.
- Store the KV cache in kernel layout to isolate the transpose cost (`BENCHMARKS.md:151` names this as unmeasured). Interesting; superseded by paging, which removes the whole-cache transpose rather than optimizing it.

**Explicitly not an engine change:** continuous batching, the scheduler, the allocator, eviction, routing. Those are this repo's headline. Putting them upstream would collapse the two independent claim sets into one (`SERVING_INTERFACE.md:284`).

---

## 7. Key architectural commitments

Stated here as PRD-level constraints; each gets a full ADR with alternatives.

**C1 — The engine is a pinned dependency, never forked.** Git submodule at a tag for phase 1 (the compiled `.so` lives at a known relative path, which is trivial for a submodule and awkward for an installed package — `SERVING_INTERFACE.md:234`); reassess before the repo goes public.

**C2 — The attention/cache seam lives in the engine; every implementation behind it lives here.** The engine's `.cu` file is frozen. This keeps the engine's "0.98–0.99× SDPA" claim (`BENCHMARKS.md:102`) intact and untouched by serving work.

*Worth knowing when that claim is questioned: two measurement biases in `bench/bench_attn_kernel.py` both run **against** the custom kernel, so 0.98–0.99× is conservative. (a) `:69-71` materializes SDPA's GQA expansion **outside** the timed lambda at `:72`, so SDPA's 4× KV expansion is free in the comparison while the kernel handles GQA internally inside its timed call at `:66`. (b) The kernel synchronizes on every call (`kernels/bindings.cpp:39,52,65`) while SDPA queues asynchronously — disclosed at `BENCHMARKS.md:119`. Bias (a) is **not** disclosed in the engine's docs. Neither weakens the claim; both are worth stating before someone finds them.*

**C3 — The custom CUDA kernel is not in the paged path.** Say it before being asked. `kernels/bindings.cpp:29-31` has no stride, block-table, or block-size arguments — a paged cache is inexpressible across that ABI. The custom kernel becomes the *single-replica, contiguous-cache reference path*. This is the honest cost of C2.

**C4 — Two paged-attention implementations, both shipped.** A pure-PyTorch block-gather path written by the author (correctness oracle, hand-written reference implementation, ~150 lines) and FlashInfer as the fast path behind the same interface. Benchmarked against each other. This yields an honest A/B, a fallback if FlashInfer will not build on PACE, and a real answer to *"why did you use a library here."*
**Attribution, fixed wording:** *integrated FlashInfer's paged-attention kernels behind a pluggable attention backend; wrote a PyTorch reference implementation as the correctness oracle.* Never "wrote a paged kernel."
✅ **Verified 2026-07-31** against `flashinfer-python==0.6.16` source on PACE (`ARCHITECTURE.md:§2.3.1`). The block layout and `page_size=16` choice were correct; the page-addressing metadata was not — FlashInfer uses a CSR triple (`kv_indptr`/`kv_indices`/`kv_last_page_len`), not a padded `block_tables` matrix. Corrected in `BatchMeta`. The mismatch cost an adapter, not a redesign, which is the outcome C2's seam was chosen to produce.

**C5 — Block layout: `[num_blocks, block_size, n_kv_heads, head_dim]`, block size 16 to start.** Block-contiguous, cheap append. The engine's per-token whole-cache transpose (`engine/components_gpu.py:153-154`, ~67 MB/token at kv_seq 2048 per `BENCHMARKS.md:149`) exists because a layout mismatch forces a *whole-cache* copy; a paged kernel reading block tables directly performs no gather and no transpose. Paging dissolves the problem rather than optimizing it. Block size is a tunable and should be swept, not asserted.

**C6 — Goodput under a declared SLO is the headline metric.** Raw throughput is reported but is not the claim. Load generation is open-loop. Rationale belongs to the benchmark methodology doc, but the commitment is made here so the phase DoDs can depend on it.

**C7 — Claim boundary.** Engine = model internals, kernels, quantization. Serving = paged KV, continuous batching, radix prefix caching, prefix-aware routing. The words *"OpenAI-compatible server"* are spent on the engine line and must not reappear on the serving line (`SERVING_INTERFACE.md:157`).

---

## 8. Open questions — blocking

*Resolved 2026-07-31: O2 answered (late-August freeze), O4 answered (C4 confirmed). O1 open, verification checklist owed. O3 and O5 remain.*

**Consequence of the late-August freeze — recorded here because it changes the goal set, not just the schedule.** Roughly four weeks. Tier 1 does not fit in four weeks, and Tier 1 does not contain prefix-aware routing (Tier 2). G5 and the routing half of C7's claim-set split are therefore **not** freeze-date deliverables under any honest reading. The phase plan will carry two cut lines — a freeze-date bullet set and a fall-rolling bullet set — rather than one plan aimed at full completion. G1–G3 plus G6 are the realistic freeze-date target; G4 and G5 are fall work.

**O1 — PACE GPU concurrency. ~~Blocks all routing work.~~ RESOLVED 2026-07-31, measured on the cluster.**

The runbook's "one A100, 4 hours" (`docs/PACE_RUNBOOK.md:24-32`) is **not a policy ceiling**. It is an artifact of using the `interactive-cpu2` partition, whose `interactive-gpu` QOS caps at `gres/gpu=2, MaxJobsPU=1`. Using the `gpu-*` partitions under the `inferno` QOS instead:

| Fact | Value | Source |
|---|---|---|
| Max concurrent GPUs per user | **32** (`inferno` QOS `MaxTRESPU=cpu=6000,gres/gpu=32`) | `sacctmgr show qos` |
| Max jobs per user | 500 submitted / 250000 queued (`inferno`) | `sacctmgr show qos` |
| Wall time, all `gpu-*` partitions | **3-00:00:00** (3 days) | `scontrol show partition` |
| Single-node GPU counts | `gpu-a100`: 2/node, **one node with 8**; `gpu-h100`: **8/node** (4 nodes); `gpu-h200`: **8/node** (12 nodes); `gpu-l40s`: **8/node** (10 nodes); `gpu-v100`: 2/node; `gpu-rtx6000`: 4/node | `sinfo -o "%P %l %G"` |
| QOS available to the account | `inferno` (charged), `embers` (free, preemptible, `MaxWall=08:00:00`, `MaxJobsPU=50`) | `sacctmgr show assoc` |
| SU balance | **999.93** available, 0.00 reserved | `pace-quota` |
| Relative GPU-hour cost | A100 = 10261, **H100/H200 = 24940 (2.43× A100)**, L40S = 8030 (0.78× A100) | `TRESBillingWeights` per partition |

**Consequences — several planning assumptions change:**

1. **Multi-replica routing is fully viable and needs no inter-node networking.** An 8-GPU H200 or L40S node gives 8 genuine, physically separate replicas inside one allocation. MIG partitioning and multi-replica-on-one-GPU are no longer necessary compromises, which removes the benchmark-phrasing problem entirely — "N replicas" means N GPUs, with no asterisk. **G5 is unblocked.**
2. **The binding constraint on this project is the late-August freeze date, not hardware.** That reverses the assumption this PRD was drafted under.
3. **A two-QOS workflow is now the plan:** `embers` (free, preemptible, 8h) for development, iteration, and correctness runs; `inferno` (charged) reserved for final published benchmarks. This protects the SU balance without slowing iteration. Preemption under `embers` must be assumed — long runs need checkpointing or must be split.
4. **L40S is the best value for routing work** — 8 per node, 0.78× the A100 SU rate, and routing/scheduling results are about *scheduling behavior*, not peak FLOPs. Reserve H100/H200 (2.43× cost) for results where absolute throughput is the claim.
5. **H100/H200/L40S require the CUDA-arch fix.** `scripts/build_kernels.sh:10` hardcodes `-DCMAKE_CUDA_ARCHITECTURES=80`. H100/H200 are sm_90, L40S is sm_89. This promotes the "nice to have" CUDA-arch parameterization in §6 to **prerequisite** for any non-A100 run.
6. ⚠️ **Absolute SU burn rate is unverified.** Only two prior jobs exist in the 30-day window (both `gpu-a100`, `inferno`, ~7.6 min, 1 GPU), and the balance reads 999.93 — consistent with roughly 0.28 SU per A100-GPU-hour, but that derivation assumes the balance started at exactly 1000.00 and is a two-sample inference. The **ratios** above are read directly from `TRESBillingWeights` and are reliable. Calibrate the absolute rate with one short instrumented job (record balance before/after) before committing to a multi-GPU benchmark budget.

**Still open:** whether MIG is enabled (irrelevant now that whole GPUs are available, but worth a `nvidia-smi -L` when first on a node), and real queue-wait distributions per partition — `gpu-h200` and `gpu-l40s` showed several `allocated`/`drained` nodes, so contention is real even though capacity exists.

**O2 — Scope freeze date.** "Time is not a constraint for this plan" governs the plan's completeness, but the phase plan still needs a marked line: which phase must be *done and benchmarked* by the date applications open. Without it, "impressive if I stop after phase N" has no N.

**O3 — Model choice for multi-replica.** Llama 3.2 1B at fp16 is ~2.36 GB of weights (`BENCHMARKS.md:161`). Multiple replicas on one 40 GB A100 is feasible; whether that constitutes an honest "N replicas" claim, versus MIG partitioning, versus a genuine multi-GPU job, is a benchmark-framing decision that depends on O1.

**O4 — Confirmation on C4.** The two-implementation paged-attention plan is my recommendation, not a settled decision. It is the single largest scope call in the project.

**O5 — Does the engine's `weights/` directory contain the Llama 3.2 1B checkpoint locally, and is HF gating going to be a per-machine friction point?** `docs/PACE_RUNBOOK.md:164`: *"Weights are HF-gated. If `weights/` is missing, `huggingface-cli login` then re-download."* Affects CI design and whether correctness tests can run without weights.

---

## 9. What this document does not decide

Phase ordering and the incremental-value property; per-feature time and risk; benchmark methodology (open vs closed loop mechanics, workload design, warmup, percentile computation, the vLLM fairness question, and the *predicted losing case for prefix-aware routing*); component decomposition and the precise engine↔serving interface; sequence walkthroughs for partial prefix hit, preemption under pressure, and replica failure mid-request; the risk register; the ADR set; and the repo scaffold.

Those are deliverables 3–9, in that order.
