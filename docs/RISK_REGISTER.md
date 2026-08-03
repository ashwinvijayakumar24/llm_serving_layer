# Risk Register

**Status:** draft 1, 2026-07-31. Planning only.
**Depends on:** `docs/PRD.md`, `docs/BENCHMARK_METHODOLOGY.md`, `docs/ARCHITECTURE.md`, the phase plan.

---

## 1. The category that matters

A crash is not a risk worth a register entry. It announces itself, it blocks the commit, and it gets fixed the same day.

The risks that end projects like this one are the ones that **produce a plausible number that is wrong**. The system runs, the tests pass, the throughput graph looks great, the claim gets published — and months later one question reveals the measurement never meant what it claimed. There is no error message for that, and no amount of care after the fact recovers it.

So this register is ordered by **detectability**, not by impact. Everything in §2 is something that fails *silently*. §3 and §4 are ordinary risks, handled ordinarily.

**Severity = Impact × Silence.** A silent risk that invalidates a published claim is CRITICAL regardless of how likely it is, because the cost of it landing is a retracted claim.

| Rating | Meaning |
|---|---|
| **CRITICAL** | Silently invalidates a published claim. Detection must exist *before* the claim is measured. |
| **HIGH** | Silently corrupts output or state; caught only by a gate built for it. |
| **MEDIUM** | Degrades results or schedule; visible with normal attention. |
| **LOW** | Annoyance, contained. |

Every CRITICAL and HIGH entry names a **detection mechanism** and **the phase that must build it**. A risk without a detection mechanism is a hope.

---

## 2. Silent invalidators

### R1 — Coordinated omission in the load generator
**CRITICAL · Phase 2**

If the load generator is at all synchronous, then when the server stalls it stops issuing requests that *should* have been issued. Those slow requests never enter the latency distribution. **A saturated system reports excellent p99.** Nothing errors, nothing looks wrong, and the number is worse than useless — it is confidently backwards.

*Detection:* measure latency from **intended dispatch time** derived from the arrival schedule, never from actual send time. Track intended-vs-actual drift as a first-class harness metric, emitted into every artifact.
*Mitigation:* runs whose drift exceeds a declared threshold are **invalid and discarded**, not caveated. Drift is plotted alongside every goodput curve so a reader can check it.
*Owner:* **[A]** — the author must understand this, because it is the single most likely way this project publishes a wrong number.

### R2 — Host-clock timing with no GPU synchronization
**CRITICAL · Phase 2**

The engine's host-clock timings are valid only by side effect: `engine/model_gpu.py:158` copies logits to host every step, forcing a CUDA sync (`BENCHMARKS.md:60` states this explicitly). `ARCHITECTURE.md:§2.6` **removes that copy on the batched path** — `forward_varlen` returns a device tensor deliberately, because at batch 32 a per-step host round-trip is on the critical path for every request in the batch.

The moment that copy goes, `time.perf_counter()` around the forward pass measures **kernel-launch queueing, not execution**. Timings get faster. Everything still runs. The throughput number becomes fiction.

*Detection:* server-side GPU timing uses CUDA events (the pattern already exists at `bench/bench_attn_kernel.py:32-44`) or an explicitly declared sync point. Add a startup assertion that the timed path contains a sync, so removing one later fails loudly.
*Mitigation:* a sanity check that measured step time is within a plausible band of a known-synced measurement, run once per benchmark session.
*Note:* client-side timings stay host-clock — they measure real network-observable behavior, which is what they should measure. Only server-side internal timing is at risk.

### R3 — Preemption silently dropping or duplicating tokens
**CRITICAL · Phase 3**

A preemption bug produces *plausible text*. Output stays fluent, no metric degrades, no exception is raised. It is load-dependent and rare, so it will not appear in casual testing — and it invalidates every correctness claim about the system simultaneously.

*Detection:* greedy output under **forced** memory pressure must be **bit-identical** to an unpreempted run, per request. This is the single most important test in the project (the phase plan §6). It runs in CI and before every published benchmark, not once at implementation time.
*Mitigation:* per-request token accounting — count tokens emitted vs tokens expected, assert on every retire. Recompute and swap tested independently; a bug in one must not be masked by the other being the default.

### R4 — Batched output diverging from single-sequence output
**CRITICAL → CONFIRMED REAL, MITIGATION CLAUSE INVOKED 2026-08-01 · Phase 2**

> **The risk materialised, and the register's own mitigation clause applies:** exact bit-identity is NOT achievable, so the divergence rate and conditions are published rather than the gate being loosened.
>
> Measured (`results/p4/FINDING_batch_shape_numerics.md`, jobs `11602281`, `11602316`): logits move with the PACKED BATCH TOKEN COUNT, because `linear()`'s `M` dimension changes cuBLAS kernel and split-K selection and therefore fp16 reduction order. A byte-identical target sequence at identical positions shows max|Δlogit| up to **0.1745**, and independent verification without the cache or scheduler reproduces **0.01953** at 361+ packed tokens.
>
> **Token-level invariance is therefore probabilistic, not structural.** The Phase 2 gate passes because its prompts are 5-36 tokens with comfortable argmax margins; the drift is present there too and simply never flips a decision. Phase 4's longer prefix-shared prompts flip roughly one argmax per 12-request workload.
>
> Forcing uniform batch shapes drives the drift to exactly zero, which localises the cause precisely. The fix — pinned cuBLAS algorithms, fp32 accumulation, or deterministic split-K — lives in the ENGINE's forward pass and costs throughput.

> The gate exists and passes on H100 (job `11598894`, 9 tests): mixed prompt lengths, batch sizes 2/3/4, a chunked prefill sharing a batch with decodes, staggered mid-flight arrival, cancellation isolation, and a leak check. This was the first thing in the project to run `n_seqs > 1` on real weights — Phase 1 was batch-1 throughout. Bit-identity held; the mitigation clause (publish the divergence rate instead of loosening the gate) was not needed.
>
> The risk does not retire — it reappears whenever batching or attention changes — but the gate that catches it now runs on every GPU job.

Batching changes numerics through reduction order, kernel selection, and attention masking. Divergence may appear only for *specific batch compositions* — a long sequence next to short ones, or a prefill chunk mixed with decodes. Every existing correctness test in the engine is batch-1 (`tests/test_forward.py`, `tests/test_decode.py`), so nothing catches it.

*Detection:* **batch-invariance test** — a prompt must produce identical greedy output alone and inside a mixed batch with sequences at different positions. Parameterized over batch compositions including the adversarial ones.
*Mitigation:* if exact bit-identity proves unachievable due to legitimate reduction-order effects, that is a *result to publish*, not a gate to loosen — quantify the divergence rate and the conditions, and say so.

> **REOPENED, WITH A MEASUREMENT, 2026-08-01 (job `11602281`, H200).** The gate
> passes on TOKENS and fails on the KV underneath them. Same 105-token prompt,
> cache off, prefilled in one chunk of 105 either way, the only difference being
> that 286 tokens belonging to other requests shared the forward pass:
> **max|Δ| = 0.1745 on the per-position K/V fingerprint, at 105 of 105 positions**
> (`scripts/debug_radix.py`, batch-shape self-test). Every per-sequence attention
> shape is held fixed by that test, so the sensitivity is not in `attend` — it is
> in the projections and FFN, whose `M` is the packed token count of the whole
> batch, and whose fp16 GEMM kernel selection moves with `M`.
>
> `tests/test_batch_invariance.py` does not catch this because its prompts are
> 5–36 tokens: the drift is real there too and simply never flips an argmax. This
> is the mitigation clause coming due — the divergence is quantified above rather
> than the gate being loosened — and it is the direct cause of the R6 failure
> below.

### R5 — The GPU path has no correctness oracle at all
**CRITICAL → RETIRED 2026-07-31 · Phase 0 · Inherited**

> **Closed.** `tests/test_gpu_oracle.py` exists and passes on A100-PCIE-40GB (Slurm job `11596894`, engine `v0.2.0`): GPU fp16 greedy output matches the HF fp32 oracle for 16 tokens exactly on both fixture prompts, matches the CPU fp32 path of the same engine, and prefill logits agree to max|Δ| 0.0111 with 10/10 top-10 overlap. Every downstream gate (R3, R4, R6, R9) now has a known-good reference to bisect against.

`tests/test_gpu_model.py:42-45` asserts only that outputs are finite, correctly shaped, and have an in-range argmax. **GPU fp16 tokens are never compared to HuggingFace or to the CPU reference path.** Every correctness claim in the engine (`tests/test_forward.py:113,137`, `tests/test_decode.py:30-48`) is about the **CPU fp32** path.

This is the foundation risk. Building a batched, paged, preemptible scheduler on an unverified forward pass means any divergence is unattributable across allocator / batching / paged attention / pre-existing GPU bug — with no way to bisect. Every gate in R3, R4, R6, and R9 compares against *something*, and this is that something.

*Detection:* Phase 0 builds it. Greedy tokens from `LlamaModelGPU` vs `tests/oracle.py`'s `greedy_ids`, token equality (not logit distance — GPU component tests already run at `atol=1e-2`, `tests/test_components_gpu.py:83`).
*Mitigation:* none. This is a hard gate; Phase 1 does not start without it.

### R6 — Radix cache returning wrong tokens
**HIGH → CLOSED AT MATCHED GEMM SHAPE 2026-08-01 · Phase 4**

> **The cache does not return wrong tokens.** Job `11611626`, H200: cache-on output is bit-identical to cache-off across all four sharing structures — including `adversarial`, which sweeps the divergence point across every offset within a block — at 38-48% block reuse. Gate exit code 0.
>
> The gate holds GEMM shapes constant by capping prefill at one block per step, because a cache hit otherwise changes the packed token count and with it cuBLAS kernel selection and fp16 reduction order. Batch size 1 does NOT achieve this (tried, failed identically): cache-off computes M=105 while cache-on computes M=41 for the uncached remainder. The shape difference is intrinsic to caching.
>
> **Still open:** bit-identical output under production chunk sizes and concurrent batching, which is blocked on R4 and is a property of `engine/components_gpu.py:linear`, not of this cache. The batched gate fails on the same three structures in the same job on the same code — one difference, opposite outcomes, which is what attributes the divergence.

A prefix cache bug makes the system **faster and wrong** — the most dangerous possible combination, because every performance metric improves. The specific hazard is block-boundary truncation: reusing a block that matches only partially (`ARCHITECTURE.md:§9.1`, the divergence-mid-block case).

*Detection:* bit-identical greedy output with cache on vs off, run on every benchmark. Explicit test at divergence points that straddle block boundaries — not just aligned ones.
*Mitigation:* the adversarial near-miss workload from methodology §4 exists specifically for this, with divergence points swept across all offsets within a block.

> **GATE RED 2026-08-01, AND THE CAUSE IS NOT THE CACHE (jobs `11602081`,
> `11602216`, `11602244`, `11602281`, H200).** `tests/test_radix_gpu.py` is
> 19/22: the block-boundary sweep passes at all 16 offsets, chunked-prefill-
> under-a-hit passes, the leak/eviction test passes, and the `zero` control
> reports zero hits. The three `system` / `conversational` / `adversarial`
> structure cases fail, each on ONE request in twelve.
>
> It is not a wrong block. Instrumenting every position's K and V at every layer
> (`scripts/debug_radix.py`) shows the reused prefix carries the right tokens at
> the right positions, differing from the recomputed prefix only in the low bits:
> **max|Δ| 0.119–0.210 over the reused region**, of which the whole workload
> shows exactly one argmax flip. Three controls close it:
>
> 1. Replaying the divergent request with the cache OFF at prefill budgets
>    2^20 / 64 / 32 / 16 reproduces the CORRECT output every time — chunking
>    alone does not do this.
> 2. Forcing every prefill chunk to exactly one block (`max_prefill_tokens=16`),
>    so publisher and consumer compute each position inside an identically
>    shaped forward pass, drives the drift to **exactly 0 at every position of
>    every request** and all 12 outputs match.
> 3. The batch-shape self-test under R4 above reproduces the same magnitude with
>    no cache in the picture at all.
>
> So the radix cache hands over the correct block; what it cannot do is make the
> publisher's forward pass the same *shape* as the consumer's, and this engine's
> KV is not invariant to that shape. R6's premise — "reuse changes no arithmetic"
> — is false for a reason that lives in R4, in `linear()`, in vendored engine
> code. **Phase 4 does not earn its correctness claim, and the fix is batch-
> invariant GEMMs, not anything in `serving/cache/radix.py`.**
>
> One genuine R6 bug WAS found and fixed on the way, by reading rather than by
> the gate: `Scheduler._cache_insert` published `prefill_ids + output_ids[:-1]`,
> and after a RECOMPUTE resume `prefill_ids` already contains `output_ids`, so
> every block past the resume point was filed in the trie under duplicated
> tokens — right KV, lying key. It now publishes
> `(prompt_ids + output_ids)[:blocks.num_tokens]`, which is the sequence by
> definition on every path. Regression:
> `test_a_recompute_resumed_request_publishes_the_tokens_its_kv_actually_holds`
> and `test_a_resumed_request_is_still_reusable_by_its_own_continuation`.

### R7 — Eviction freeing a live block
**HIGH · Phase 4 · partial detection live since Phase 1**

> The allocator half is built and exercised: `check_invariants()` (O(num_blocks), catches a block that is both referenced and on the free list), double-free raises rather than being tolerated, and the leak test confirms the free list returns to its initial count. 125 allocator tests, mutation-tested to confirm they bite. The EVICTION half arrives with the radix cache in Phase 4 and is still open.

Refcount bug → a block still referenced by a running sequence is freed and reallocated. The victim sequence's attention silently reads another sequence's KV. Output stays fluent. Cross-request contamination, no error.

*Detection:* assert `refcount == 0` on every free, in debug builds enabled during benchmarks. Allocator leak test: after N sequential requests, the free list returns to its initial count exactly.
*Mitigation:* leaf-first eviction with an invariant check that no evicted node has live descendants.

### R8 — `slot_mapping` off-by-one at block boundaries
**HIGH → DETECTION CONFIRMED 2026-08-01 · Phase 1**

> Detection is live and passing. The block-straddle sweep runs at prompt lengths 1/15/16/17/31/32/33/47/48 plus a fragmented-pool test, on real weights (job `11598444`, H100). Exact multiples of `block_size` correctly report `kv_last_page_len == block_size`, not 0. The risk itself does not retire — it reappears every time addressing changes — but the gate that would catch it exists.

`slot = block_tables[seq][pos // block_size] * block_size + (pos % block_size)`. An error here writes KV to the wrong physical slot. At batch 1 with short prompts it may never cross a boundary and never show.

*Detection:* bit-identical gate at ≥3 sequence lengths **chosen to straddle block boundaries** — this is why those lengths are in Phase 1's DoD explicitly rather than left to chance.

### R9 — FlashInfer layout mismatch
**HIGH → RETIRED 2026-08-01 · Phase 1**

> **Closed.** `FlashInferBackend` matches `PagedTorchBackend` across 28 differential
> tests on H100 (job `11598894`): pure decode at batch 1/2/5, mixed prefill+decode,
> page-boundary sweep 1/15/16/17/31/32/33, fragmented non-contiguous pages, multi-
> sequence isolation, GQA 8→{1,2,4,8}, and end-to-end greedy token equality. Since
> `PagedTorchBackend` equals the contiguous engine path (job `11598444`) and that
> equals the fp32 HF oracle (job `11596894`), the chain closes.
>
> Three contract facts were read out of the 0.6.16 **kernels**, not the docstrings,
> which do not state them: `causal=True` is **bottom-right aligned**
> (`prefill.cuh:1461`), matching the protocol exactly; `plan()` stores the page
> tables on the wrapper while `run()` takes none, so **a missing `plan()` silently
> attends over the previous step's CSR**; and `run()` returns
> `(tokens, n_heads, head_dim)`.
>
> **Stated limit:** the differential is **token-exact, not bit-exact.**
> `PagedTorchBackend` casts `probs` to fp16 before the PV matmul (mirroring
> `components_gpu.py:205`) while FlashInfer uses a fused fp32 online softmax and
> never materialises `probs` — they cannot agree bit-for-bit by construction.
> Tensors are compared at `atol=4e-3, rtol=1e-2`; **tokens are compared exactly**,
> which is the claim the system makes. Defensible because every R9 failure mode is
> an O(1) error, not an O(1e-3) one.

### R10 — Unchecked CUDA errors
**HIGH · Phase 6 (checking), Phase 0 (awareness) · Inherited**

`CLAIMS_AUDIT.md:299`: *"No CUDA error checking anywhere... A launch failure silently produces garbage rather than raising."* The kernel bindings call `cudaDeviceSynchronize()` (`kernels/bindings.cpp:39,52,65`) and never check the return.

Tolerable for a benchmark that would show a visibly wrong token. **In a serving system, a launch failure under memory pressure produces plausible text at full throughput with every metric green.**

*Detection:* check CUDA errors at declared points on the serving path.
*Mitigation:* any CUDA error is **fatal to the replica** — a poisoned context cannot be recovered in-process. Router quarantines and restarts.

### R11 — Measurement window including ramp-up or drain
**HIGH · Phase 2**

p99 improves as the queue empties at the end of a run. Including the drain tail publishes a better tail latency than the system has, and the run looks completely normal.

*Detection:* explicit window boundaries recorded per run; stationarity check on queue depth and in-flight count across the window.
*Mitigation:* above the knee, steady state does not exist by definition — those runs are labeled *unsaturated-window measurement*, never *steady state*.

### R12 — Cross-allocation A/B comparison
**HIGH · Phase 1 onward**

Node contention and clocks vary enough to move absolute throughput ~25% for identical code — the engine measured ~79 tok/s on one node and ~60 on another (`BENCHMARKS.md:17`, `docs/PACE_RUNBOOK.md:161`). A delta computed across two allocations is measuring node assignment.

*Detection:* Slurm allocation id, node name, and QOS recorded in every artifact. The analysis tooling **refuses to render a comparison** across differing allocation ids rather than warning about it.
*Mitigation:* every A/B runs back-to-back in one allocation. Comparisons that cannot be are published as two separate absolute measurements.

### R13 — Benchmark run preempted by `embers` QOS
**HIGH · Phase 1 onward**

`embers` is free and preemptible with an 8h cap. A preempted run truncates the measurement window mid-flight. The partial artifact looks like a completed short run.

*Detection:* QOS recorded in every artifact; the analysis tooling rejects any published number sourced from an `embers` run.
*Mitigation:* two-QOS discipline — `embers` for development and correctness, `inferno` for every published number.

### R14 — Degenerate workload
**MEDIUM-HIGH · Phase 2 onward**

A length or prefix-sharing distribution that collapses toward constant (bad parameters, a seeding bug, a truncation) produces a benchmark that measures nothing while looking fine. Uniform prompts remove exactly the phenomena the project claims to study.

*Detection:* publish the **realized** distributions from each run — actual length histogram, actual sharing rate, actual hit rate — not the requested parameters. Divergence between requested and realized is itself the alarm.

### R15 — Percentiles from too few samples
**MEDIUM · Phase 2**

A p99 over 127 samples is interpolated between the top two values and carries roughly one sample of resolution — that is the engine's exact situation at `--max-tokens 128` with NumPy's linear-interpolation default (`bench/harness.py:50-51`). Quoting it as p99 implies precision that is not there.

*Detection:* sample count reported alongside every published percentile; minimum sample counts declared per metric.
*Mitigation:* raw per-request and per-token samples are stored so percentiles are computed at analysis time over the pooled set — the engine's harness stores only per-run percentiles (`bench/harness.py:136-137,178`), making correct pooling impossible after the fact.

### R16 — Metric name collisions
**MEDIUM · Phase 2**

The engine's own known gap #1: `bench/harness.py:44-47` reports host RSS as `peak_mem_mb` while `bench/baseline_hf.py:108` reports `torch.cuda.max_memory_allocated()` under the **same column name** (`BENCHMARKS.md:247`). Two quantities, one name, silently compared.

*Detection:* metric schema carries unit and source per field; a name may map to exactly one (quantity, unit, source) triple.

### R17 — EOS handling changing token counts between compared runs
**MEDIUM · Phase 2**

If one side of a comparison stops early on EOS and the other does not, throughput is computed over different token counts. Precedent in the engine: `bench/baseline_hf.py:91-92` breaks on EOS while the engine harness has no equivalent break in `time_generate`.

*Detection:* per-run token counts recorded and asserted equal across compared configurations.
*Mitigation:* benchmark runs control output length explicitly (methodology §4) rather than letting the model decide.

---

## 3. Loud technical risks

| # | Risk | Sev | Detection | Mitigation |
|---|---|---|---|---|
| R18 | **FlashInfer won't build on PACE** (CUDA version, arch, no compute-node internet — `docs/PACE_RUNBOOK.md:64-66`) | MEDIUM | Build fails | `PagedTorchBackend` is the fallback and is written first. Memory claim (S1) does not depend on FlashInfer at all. Build on a login node with internet. |
| R19 | **CUDA arch hardcoded to sm_80** (`scripts/build_kernels.sh:10`) blocks L40S (sm_89) and H100/H200 (sm_90) | MEDIUM | Runtime failure on non-A100 | Parameterize in Phase 0. Promoted from "nice to have" by PRD §8/O1. |
| R20 | **Engine packaging breaks under non-editable install** — `sys.path` manipulation at `engine/model_gpu.py:45-53`; `.so` lands in gitignored `build/` | MEDIUM | Import error | Git submodule for Phase 1 keeps `build/` at a known relative path (`SERVING_INTERFACE.md:234`). |
| R21 | **Multi-process GPU orchestration under Slurm is fiddly** | MEDIUM | Phase 5 blocked | Routing policy is pure logic — develop against **mock CPU replicas**, so PACE queue time never blocks router work. |
| R22 | **Python-side batch assembly becomes a material fraction of step time** | MEDIUM | Profile | Escape hatch stated in advance (`ARCHITECTURE.md:§7`): move the scheduler to a thread with `run_coroutine_threadsafe` handoff. Contained change, not a redesign. |
| R23 | **Router SPOF** | LOW (accepted) | Obvious | Documented limitation with the exact fix. Hint-only design makes multi-router cheap if ever needed. |
| R40 | **Starvation guard deadlocks the scheduler.** If the guard excludes sequences preempted K times *absolutely*, and every running sequence reaches K, victim selection returns nothing while the batch is non-empty — the scheduler can neither step nor make room | MEDIUM | Loud under load (throughput → 0, requests hang). But the **fallback firing is silent** if absorbed without instrumentation | Guard is a preference ordering, not an exclusion (`ARCHITECTURE.md:§5.2`). Test drives every running sequence past K. Fallback firings counted and surfaced as an **admission-control alarm** — if the fallback ever fires, the watermark is wrong |
| R24 | **Weight-memory double counting** via tied embeddings | LOW | — | The engine already solves this — `bench/harness.py:59-61` de-dupes by `id()`. Reuse it rather than reimplementing. |

---

## 4. Project and schedule risks

| # | Risk | Sev | Notes |
|---|---|---|---|
| R25 | **Freeze date arrives mid-phase**, leaving a half-built feature that makes an earlier claim untrue | HIGH | This is phase-plan property 4. Cut order is pre-decided (the phase plan §6 freeze-line box): swap policy first, then FlashInfer. **Phase 2 has no safe cuts.** |
| R26 | **Effort assumption wrong.** ~100h to the freeze line assumes ~30/week | HIGH | Recompute from the real number in week 1, not week 3. Every phase carries a cheaper alternative so cuts stay informed. |
| R27 | **PACE queue wait** — `gpu-h200`/`gpu-l40s` showed allocated and drained nodes | MEDIUM | Two-QOS workflow; `embers` for iteration. Everything that can be developed CPU-side (router policy, allocator logic, radix trie, harness) is developed CPU-side. |
| R28 | ~~SU exhaustion~~ — **CLOSED 2026-07-31.** Burn rate MEASURED, not inferred: job `11596894` used 0.08 SU for 17:02 on one A100 = **0.282 SU/A100-GPU-hour**, confirming the earlier two-sample estimate. 999.85 SU remaining ≈ 3,500 A100-GPU-hours; an 8×L40S 8-hour run costs ~14 SU | CLOSED | Not a constraint on this project. Prefer `gpu-l40s` (0.78× rate) for scheduling/routing work regardless |
| R29 | **Explainability gate fails** — a phase ships but can't be explained cold | MEDIUM | Then it is not published as a claim, per PRD §G7. The gate is the point; failing it is the system working. |
| R30 | **Scope creep into Tier 3/4** | MEDIUM | Tier 4 is explicitly where scope dies (PRD §5). K8s recommended against permanently. |
| R31 | **vLLM comparison proves unfair and gets quietly dropped** | LOW | Position committed in advance (methodology §8): no throughput claim vs vLLM, ever. If even the shape comparison fails, say so in writing — that is a stronger answer than a table. |

---

## 5. Inherited risks — engine state as of `6ff40a1`

Recorded because they are this project's foundation and none are this project's fault.

| # | Finding | Status |
|---|---|---|
| R32 | `tests/test_generate.py` — all 5 tests raise `KeyError` (`"input_ids"` at `:42,52,63,78,89` vs `"token_ids"` at `oracle.py:170`). **The KV-cache-vs-no-cache correctness gate is not enforced today.** | Phase 0 fix |
| R33 | No model-level GPU correctness oracle | R5, Phase 0 |
| R34 | No CUDA error checking anywhere | R10 |
| R35 | Chunked-prefill position bug — `model_gpu.py:65` uses `arange(seq)` from 0 regardless of cache position. Latent; invisible because prefill is called once (`scheduler.py:33`) | Phase 0 fix |
| R36 | Server runs the CPU NumPy model with no GPU path (`server.py:24-30`); blocking generator inside `async def` (`:69`) serializes concurrent requests | Not fixed — the serving layer owns the production surface. Engine server stays a reference path (`README.md:144`). |
| R37 | ~~No benchmark artifacts committed~~ — **WITHDRAWN 2026-07-31.** `CLAIMS_AUDIT.md:6` is stale; the engine committed its artifacts in `c24afaf` and `bench/results/` holds 24 tracked files (`.gitignore:29` records the decision). The residual issue is narrower: those artifacts store per-run percentiles, not raw samples (`bench/harness.py:136-137,178`) | Not a risk. This repo stores raw samples so percentiles are computed at analysis time (R15) |
| R41 | **Engine's editable install resolves to a stale path.** The `__editable__` finder maps `engine` → `.../Personal Projects/llm_inference_egine/engine` — a previous location, **with a typo**. `import engine` from any directory other than the repo root raises `ModuleNotFoundError`; it appears to work in-repo only because Python puts CWD on `sys.path` | MEDIUM · **quiet** — looks fine until code runs from another CWD, e.g. a Slurm job or the serving layer's own tests | Reinstall from the current location. The serving layer installs the engine from `vendor/` so it resolves independently — but P0's DoD must verify `import engine` from **outside** the repo root, not from inside it |
| R38 | Microbench pre-expands GQA outside SDPA's timed region (`bench/bench_attn_kernel.py:69-72`), biasing toward SDPA | Engine's number is conservative. No action. |
| R39 | `cudaMalloc`/`cudaFree` per kernel invocation inside the timed path (`attention_decode.cu:297-300,311`) | Engine-side, disclosed at `BENCHMARKS.md:120`. Custom kernel is not on the paged path anyway. |

---

## 6. Detection infrastructure — build order

The register is only worth something if the detection exists before the risk lands. Consolidated build order:

**Phase 0**
- GPU correctness oracle (R5) — hard gate, blocks Phase 1
- `test_generate.py` fix (R32), chunked-prefill position fix (R35)
- Artifact schema: allocation id, node, QOS, seed, git SHA, engine tag, units + source per metric (R12, R13, R16)

**Phase 1**
- Bit-identical gate at block-straddling lengths (R8)
- Allocator leak test (R7)
- Differential oracle: `PagedTorchBackend` vs `FlashInferBackend` (R9)

**Phase 2**
- Batch-invariance test (R4) — hard gate
- CUDA-event server-side timing + sync assertion (R2)
- Intended-dispatch timing + drift metric (R1)
- Window boundaries + stationarity check (R11)
- Realized-distribution reporting (R14)
- Sample-count reporting (R15), token-count assertions (R17)

**Phase 3**
- Bit-identical-under-forced-preemption gate (R3) — the most important test in the project
- Per-request token accounting

**Phase 4**
- Cache-on vs cache-off equality (R6)
- Refcount assertions on free (R7)
- Adversarial block-boundary workload

**Phase 6**
- CUDA error checking at declared points (R10)

---

## 7. Review cadence

- **Per phase:** before publishing any number from that phase, walk its CRITICAL and HIGH entries and confirm each detection mechanism ran. Record the confirmation in the artifact.
- **Per benchmark session:** allocation id, QOS, and drift metric checked before results are considered valid.
- **Register updates:** any new silent failure mode discovered during implementation is added here immediately, with its detection mechanism, before the work continues. A discovered-but-unregistered silent risk is the same as an undetected one.

---

## 8. The three that would actually sink this

If attention is scarce, these are the ones:

1. **R5 — no GPU correctness oracle.** Everything else in this register compares against *something*. This is that something. Without it, nothing below it can be trusted, and the failure is total rather than partial.
2. **R1 — coordinated omission.** The most likely path to publishing a confidently backwards number, on the project's headline metric, with no error anywhere.
3. **R3 — silent preemption corruption.** Load-dependent, rare, produces fluent text, degrades no metric. It would survive to publication.
