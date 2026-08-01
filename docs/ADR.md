# Architecture Decision Log

**Status:** draft 1, 2026-07-31. Planning only; no implementation.
**Depends on:** `docs/PRD.md`, `docs/BENCHMARK_METHODOLOGY.md`, `docs/ARCHITECTURE.md`, `docs/PHASE_PLAN.md`. Engine @ `6ff40a1`.

One file, one decision per section. Numbering is stable and referenced from `ARCHITECTURE.md` (ADR-001 … ADR-007) — those numbers must not be reused or renumbered.

Claims about the engine cite `file:line` in `../llm_inference_engine`. **[inference]** marks judgment rather than something read from code. Numbers not attributed to a source are placeholders, not measurements; no ADR here asserts a benchmark result.

**Index**

| # | Decision | Status |
|---|---|---|
| ADR-001 | Consume FlashInfer's paged-attention kernel; write a PyTorch reference as oracle | Accepted |
| ADR-002 | No shared state between replicas — router holds hints, replica holds truth | Accepted |
| ADR-003 | Flattened varlen token layout, not padded `(batch, seq, hidden)` | Accepted |
| ADR-004 | `AttentionBackend` is varlen-batched from day one | Accepted |
| ADR-005 | Paged block layout `[num_blocks, block_size, n_kv_heads, head_dim]`, block_size 16 | Accepted (revisit after the block-size sweep and FlashInfer verification) |
| ADR-006 | Implement both recompute and swap preemption behind a policy flag | Accepted (revisit at the freeze line) |
| ADR-007 | Scheduler as a cooperative asyncio task, not a dedicated thread | Accepted (revisit if Python-side assembly is material in a profile) |
| ADR-008 | Separate repo + git submodule, not a monorepo | Accepted (revisit before the repo goes public) |
| ADR-009 | Every engine change is strictly additive | Accepted |
| ADR-010 | The serving layer owns the production HTTP surface; the engine's server stays a reference path | Accepted |
| ADR-011 | Goodput under a declared SLO is the primary metric | Accepted |
| ADR-012 | Open-loop load generation | Accepted |
| ADR-013 | No vLLM throughput comparison; shape comparison only | Accepted (revisit after one attempt) |
| ADR-014 | No SQL / relational persistence; JSON+CSV artifacts on the filesystem | Accepted (revisit if run comparison becomes painful) |
| ADR-015 | Kubernetes + cache-aware inference gateway | Rejected |
| ADR-016 | Phase ordering — preemption (P3) before the radix cache (P4) | Accepted |
| ADR-017 | Least-outstanding-requests is the real routing baseline; round-robin is table stakes | Accepted |
| ADR-018 | Benchmark artifacts are committed to git | Accepted |
| ADR-019 | The custom CUDA kernel is not in the paged path | Accepted |
| ADR-020 | A model-level GPU correctness oracle blocks all serving work | Accepted |
| ADR-021 | Replicas are whole physical GPUs on one node; two-QOS benchmark discipline | Accepted |
| ADR-022 | The router is a single point of failure, and that is accepted | Accepted (revisit if HA becomes a claim) |
| ADR-023 | `forward_varlen` returns a device tensor; server-side timing uses CUDA events | Accepted |
| ADR-024 | LIFO victim selection with a starvation guard | Accepted (revisit if the fairness result demands it) |

---

## ADR-001 — Consume FlashInfer's paged-attention kernel rather than writing one; ship a PyTorch reference as the correctness oracle

**Status:** Accepted
**Date:** 2026-07-31

**Context**

A paged KV cache needs an attention kernel that can read K/V through a block table instead of from one contiguous buffer. The engine already contains a hand-written CUDA decode kernel (`kernels/attention_decode.cu`, 311 lines, three staged versions, warp-shuffle reductions, flash combine rule), so "write another one" is the reflexive move.

The engine's kernel cannot be extended into a paged one. Its pybind ABI has no stride, block-table, or block-size arguments (`kernels/bindings.cpp:29-31`) — a paged cache is not *expressible* across that boundary, not merely awkward. `Q` is `[n_heads, head_dim]` with no batch axis (`kernels/attention_decode.cu:30-32`), so a paged *batched* kernel is a rewrite, not an extension. Rewriting also re-opens the 100-input numerical gate (`tests/test_attention_kernel.py:72-85`) and the "0.98–0.99× PyTorch SDPA" claim (`BENCHMARKS.md:102`) that the existing kernel currently owns.

The project's lane is systems engineering applied to AI, not kernel engineering (PRD §1). "Can he write CUDA" is already answered by the engine and is a *deprioritized lane*. Meanwhile PRD §8 fixes a late-August freeze at roughly four weeks / ~120 engineer-hours (PHASE_PLAN §0), against which a paged kernel is estimated at 15–25 hours of pointer-arithmetic debugging (ARCHITECTURE §10/A3).

**Decision**

Two paged-attention implementations behind one `AttentionBackend` protocol (PRD §C4):

1. **`PagedTorchBackend`** — pure PyTorch block-gather into SDPA, ~150 lines, written by the author. This is the correctness oracle and the hand-written reference. Written **first**, because it is layout-independent (PHASE_PLAN §4).
2. **`FlashInferBackend`** — FlashInfer's paged kernels as the fast path, integrated behind the same protocol.

Benchmarked head-to-head at matched sequence lengths. Attribution wording is fixed and non-negotiable (PRD §C4, PHASE_PLAN §11): *"integrated FlashInfer's paged-attention kernels behind a pluggable attention backend; wrote a PyTorch reference implementation as the correctness oracle."* Never "wrote a paged kernel."

**Alternatives considered**

- **Write a paged-attention CUDA kernel (ARCHITECTURE §10/A3).** Rejected. It is the longest pole in the project for the smallest signal: the marginal claim is "can write a *second*, harder CUDA kernel," in a lane the author has explicitly deprioritized, at the cost of the phase that carries the actual claim set. It is also not an extension of existing work — the ABI (`kernels/bindings.cpp:29-31`) and the batch-less `Q` shape (`attention_decode.cu:30-32`) mean starting over. And the failure mode is bad: a subtly wrong paged kernel produces plausible text, so debugging it competes for the same attention as the scheduler. **[inference]** The honest cost accounting is that this trade buys ~20 hours of scheduler work with ~20 hours of kernel work that duplicates an answered question.
- **FlashInfer only, no PyTorch reference.** Rejected. Three separate losses. (a) No oracle: with FlashInfer as the only paged path, a divergence cannot be attributed between the allocator, the block tables, and the kernel. (b) No fallback: FlashInfer's build on PACE is unverified, and a build failure would block the entire project's critical path. (c) No answer to *"why did you use a library here?"* other than "it was there" — whereas having written the reference makes the layout contract something the author can defend cold.
- **PyTorch reference only, no FlashInfer.** Rejected as the plan, retained as the explicit Phase 1 cut (PHASE_PLAN §4). The memory claim (S1) does not depend on FlashInfer at all; the throughput headroom and the library-vs-hand-written A/B do.
- **`torch.nn.functional.scaled_dot_product_attention` over a materialized contiguous cache, with paging only in the allocator.** Rejected. It reintroduces exactly the whole-cache copy that ADR-005 eliminates, and it makes the "paged attention" claim false — the paging would be bookkeeping with no kernel behind it.

**Consequences**

- The strongest kernel-level claim in this repo is an *integration* claim. That is a real cost, and it is why the wording is fixed in advance rather than allowed to drift under questioning.
- Two implementations to keep numerically in sync. The bit-identical greedy gate applies to both.
- Hard dependency on an unverified third-party layout contract. ADR-005's block layout is asserted as reasonable, **not** as FlashInfer-compatible (ARCHITECTURE §11); it is verified before Phase 2, not discovered in Phase 2 (PHASE_PLAN §12).
- The PyTorch path will be slower. That is fine and is itself the published A/B — a measured number justifying the library is stronger than an assumption.
- Makes harder: any future claim about paged-kernel *internals*. The author owns the block-gather logic, not the FlashInfer kernel's inner loop, and must say so.

**Revisit if:** FlashInfer will not build on the target hardware and `PagedTorchBackend` turns out to be the throughput bottleneck by a wide margin *after* the scheduler work is done and the freeze has passed — i.e. only when a paged kernel is the top remaining item, which it currently is not.

---

## ADR-002 — No shared state between replicas: the router holds hints, the replica holds truth

**Status:** Accepted
**Date:** 2026-07-31

**Context**

Prefix-aware routing (G5) needs the router to know something about where cached prefixes live. The obvious reading is that the router needs a *consistent* view of each replica's radix cache — which immediately imports replica→router state streaming, invalidation on eviction, and some story about what happens when the two disagree.

The engine offers no help here: it has zero thread safety, a single global model, mutable cache state, and no locks anywhere (`engine/server.py`, `engine/scheduler.py`). Anything resembling distributed cache coordination would be built entirely from scratch, on a four-week budget, by someone whose acknowledged skill gap is distributed systems (PRD §1).

**Decision**

Replicas share nothing. No shared memory, no shared cache, no cross-replica coordination on the request path (ARCHITECTURE §1). Every piece of router state — replica health, load estimate, prefix→replica hint table — is an **approximation that may be stale** (ARCHITECTURE §6).

The invariant: a wrong hint costs a cache miss and slightly higher latency. It never produces a wrong answer, never corrupts cache state, and never requires a distributed transaction.

The hint table is best-effort and lossy. Its "if lost" entry in the state table is literally **nothing** (ARCHITECTURE §6). Health and load estimates are TTL'd in seconds and rebuilt by probing.

**Alternatives considered**

- **Shared KV cache across replicas (ARCHITECTURE §10/A6).** Rejected. Requires RDMA or a distributed cache tier, and — the argument that actually matters — its benefit is precisely what prefix-aware *routing* already provides at a fraction of the complexity. **Route the request to the cache instead of moving the cache to the request.** That reframing is not just the rejection of A6; it is the affirmative argument for the routing feature existing at all.
- **Strongly consistent router view of replica cache state.** Rejected. Needs a coordination service (etcd/consensus) or synchronous replica→router invalidation on every eviction. Cost: a new failure domain, a consensus dependency on the request path, and eviction latency coupled to network round-trips. Benefit: slightly better hit rate. The trade is absurd at this scale, and the whole point of the design is that the benefit was never worth consistency in the first place.
- **Replica-pull instead of router-push** — the router routes blind and the replica fetches missing prefix blocks from a peer that has them. Rejected. This is a distributed cache with extra steps, adds cross-replica RPC to the prefill path, and creates a transfer-vs-recompute decision at the worst possible moment (during admission).
- **No routing state at all — pure least-outstanding-requests.** Not rejected; it *is* baseline B5 (METHODOLOGY §6), and prefix-aware routing must beat it to have demonstrated anything (ADR-017). The decision recorded here is that cache awareness is added as *advisory* state on top of B5, never as authoritative state.

**Consequences**

- The most valuable property of this architecture is what it does not contain: no consensus, no distributed transactions, no shared mutable state (ARCHITECTURE §6). That is the claim.
- Replica failure loses in-flight requests outright — there is no KV replication, so partially-streamed requests cannot be transparently retried and are failed with an explicit SSE error event (ARCHITECTURE §9.3).
- Hint staleness must be handled as a *normal* case, not an error case: a stale hint sends a request to a replica that simply misses.
- Purging hints is now a correctness-adjacent operation on quarantine — a dead replica's stale hints actively pull traffic toward it (ARCHITECTURE §9.3, step 2).
- Makes harder: any claim about global cache utilization across the fleet. There is no global view to measure, only per-replica hit rates plus routing outcomes.
- The settled answer is prepared in advance: *"How do you keep the router's view of the cache consistent?" — you don't, and that's the design.*

**Revisit if:** measured hit rate under prefix-aware routing is close to B5's (i.e. the hints are so stale they carry no signal), which would indicate the TTL/update model — not the sharing model — needs work. Genuine shared state is only reconsidered if replicas ever span nodes with an RDMA fabric, which is out of scope.

---

## ADR-003 — Flattened varlen token layout, not padded `(batch, seq, hidden)`

**Status:** Accepted
**Date:** 2026-07-31

**Context**

The engine is batch-1 throughout: `x` is `(seq, hidden)` (`engine/model_gpu.py:64`, `engine/components_gpu.py:127-130`), the KV cache has no sequence axis (`engine/cache.py:28`), the causal mask is 2-D (`engine/components_gpu.py:173`). Batching requires choosing how multiple sequences occupy one tensor.

The reflexive assumption — stated as such in ARCHITECTURE §2.5, *"mine too, before reading the source"* — is that batching touches everything. Reading the GPU forward path shows it does not, **given the right layout choice**.

Also relevant: METHODOLOGY §4 deliberately commits to skewed and heavy-tailed length distributions, because a benchmark of uniform 128-token prompts "has removed every interesting phenomenon it claims to study." Length skew is a *designed-in* property of every workload this project will run.

**Decision**

All sequences are packed along a single token axis. `tokens == sum(query_lens)`; `x` stays 2-D `(tokens, hidden)`. Sequence boundaries live in `cu_query_lens` (exclusive prefix sum) and `kv_lens`, not in a tensor dimension (ARCHITECTURE §2.2, §2.3).

```
Batch mixing a chunked prefill (D, 4 tokens) with 2 decodes:
tokens axis:   [ d0 d1 d2 d3 | a0 | b0 ]             shape (6, hidden)
```

**Alternatives considered**

- **Padded `(batch, seq, hidden)` (ARCHITECTURE §10/A5).** Rejected on three counts.
  1. *Wasted compute proportional to length skew* — and the workloads are deliberately heavy-tailed (METHODOLOGY §4), so the waste is not a corner case, it is the common case. A batch containing one 2000-token prefill and seven 1-token decodes pads to 2000×8.
  2. *Every op becomes batch-aware.* Under varlen, the entire position-independent half of the model is untouched: embedding lookup (`model_gpu.py:64`, a gather over an any-length id vector), `linear` (`components_gpu.py:14-24`, `(tokens,hidden) @ (hidden,out)`), `rms_norm_gpu` (`components_gpu.py:27-31`, reduces over the last dim only), `swiglu_ffn_gpu` (`components_gpu.py:185-194`, elementwise plus linears). RoPE is *already* batch-ready because `positions` is a passed tensor indexed as `cos[positions]` (`components_gpu.py:132-135`). None of that survives padding.
  3. *A padding mask threads everywhere,* including through code that currently has no idea sequences exist.
  This is not luck. A varlen layout is *chosen* precisely so that every position-independent op sees a plain 2-D matrix and cannot tell that multiple sequences are present.
- **Ragged/nested tensors.** Rejected. Immature support across the ops in use, and it would hide the exact index arithmetic (`cu_query_lens`, `slot_mapping`) that the author needs to own for the explainability gate. The manual flattening *is* the artifact.
- **One forward pass per sequence, batching only the scheduling.** Rejected — this is what the engine effectively does today, and it is exactly the failure the build log names: *"true scaling needs a batched-GEMM step"* (`docs/BUILD_LOG.md:1164`). Continuous batching without a batched forward pass is bookkeeping with no throughput behind it (PRD §G2).

**Consequences**

- **The batched forward and the paged cache are the same change to the same surface, not two projects** (ARCHITECTURE §2.5). This is the structural finding that makes the four-week freeze survivable at all.
- The genuinely batch-sensitive surface shrinks to four things: attention + KV (`components_gpu.py:137-182`, replaced by the backend), the causal mask (`components_gpu.py:171-174`, becomes the backend's job via `cu_query_lens`/`kv_lens`), position construction (`model_gpu.py:65,133`, moved to the caller), and last-token logits (`model_gpu.py:88` `x[-1:]`, replaced by a `last_token_ix` gather).
- Cost: index arithmetic moves into CPU-side batch assembly, and off-by-one bugs there are silent. Mitigated by the batch-invariance gate (PHASE_PLAN §5 DoD: a prompt must produce identical greedy output alone and inside a mixed batch).
- Cost: `x[-1:]` becoming a gather means "the last token" is no longer a slice — every place that assumed it was must be found.
- Makes harder: any op that genuinely wants a sequence dimension (e.g. certain sliding-window or block-sparse schemes) must reconstruct it from `cu_query_lens`.

**Revisit if:** a required attention implementation only accepts padded input and cannot be adapted — in which case the padding lives inside that backend, behind the protocol, not in the model.

---

## ADR-004 — The `AttentionBackend` protocol is varlen-batched from day one, even though Phase 1 only passes `n_seqs == 1`

**Status:** Accepted
**Date:** 2026-07-31

**Context**

Phase 1 ships paged KV with batch deliberately held at 1, so that a paged-attention bug cannot hide behind a batching bug (PHASE_PLAN §4). Batching arrives in Phase 2. The natural inference is to design the seam batch-1 now and widen it later, and `SERVING_INTERFACE.md:258` recommends exactly that deferral.

The seam itself is a widening of an existing hook. `engine/components_gpu.py:114` already takes `decode_kernel=`, threaded from `LlamaModelGPU.__init__` (`engine/model_gpu.py:44-53`) to the call sites at `:79,:147`. Today that hook injects **only the math** — `(q,k,v,scale) -> out` at `components_gpu.py:150-158` — while the engine keeps owning the cache read/write at `:137-142`.

**Decision**

`AttentionBackend` is varlen-batched in its signature from the first commit. Both methods take a `BatchMeta` describing `n_seqs` sequences; Phase 1 simply always passes `n_seqs == 1`.

```python
class AttentionBackend(Protocol):
    def append_kv(self, layer_idx: int, k, v, meta: BatchMeta) -> None: ...
    def attend(self, q, layer_idx: int, scale: float, meta: BatchMeta): ...
```

The hook widens by exactly one notch: the injected object owns *both* the cache read/write and the math, instead of the math alone. `KVCacheGPU` (`engine/cache.py:23-33`) is untouched and still serves the reference path (ARCHITECTURE §2.4).

`SERVING_INTERFACE.md:258` is right about **sequencing the work** and wrong about **sequencing the interface**. Those are separable, and this ADR separates them.

**Alternatives considered**

- **Batch-1 protocol now, widen in Phase 2 (`SERVING_INTERFACE.md:258`).** Rejected. It means doing the same surgery twice — once to introduce the seam, once to change it — and shipping a Phase-1 interface that is *known in advance* to break. The specific cost is that the seam spans two repos (ADR-008): a breaking change to it is a coordinated two-repo change with a submodule pin bump, which is the most expensive kind of change this project has. Paying that twice for a design already known is pure waste. PRD §6 states the constraint directly: *"The seam must not be designed batch-1 and widened later."*
- **A batch-1 protocol plus a separate batched protocol, chosen by the caller.** Rejected. Two protocols means two implementations of every backend and two correctness surfaces, and the batch-1 one would be dead code the moment Phase 2 lands.
- **Keep the existing `decode_kernel=` math-only hook and let the engine keep owning the cache.** Rejected. The engine's cache is `(n_layers, max_seq, n_kv_heads, head_dim)` with no sequence axis (`engine/cache.py:28`); a math-only hook cannot express a paged cache, because the *cache access pattern* is the entire thing being changed. The read path (`block_tables`) and the write path (`slot_mapping`) both live below the math.
- **Design the protocol batched but not paged** (batched attention over contiguous per-sequence caches). Rejected — that reintroduces the per-request `max_seq` over-allocation (`engine/scheduler.py:16,26-27`) the project exists to remove (PRD §G1).

**Consequences**

- Phase 1 carries interface complexity it does not use: `cu_query_lens` is always `[0, n]`, `last_token_ix` always `[n-1]`, `block_tables` always one row. Accepted deliberately.
- **[inference]** There is a real hazard in a protocol whose batched path is unexercised for a whole phase: `n_seqs == 1` can pass while `n_seqs > 1` is quietly wrong. Phase 2's batch-invariance gate is what catches it, and it is in the DoD for exactly this reason (PHASE_PLAN §5).
- Phase 0 ships a pure-interface change with no behavior (`engine/attention_backend.py`), which is why Phase 0 earns no published claim and is still a hard gate (PHASE_PLAN §3).
- Positive: the seam is stable across the whole project, so the submodule pin can advance monotonically rather than in coordinated breaking steps.
- Makes harder: reviewing Phase 1 in isolation, since the interface is justified by work that has not landed yet. This ADR is that justification.

**Revisit if:** FlashInfer's verified API demands metadata that `BatchMeta` cannot carry (e.g. a precomputed plan/handle object). That is an *additive* field, not a reshape, and is expected to be absorbable — but it is the one thing that would touch the frozen contract.

---

## ADR-005 — Paged block layout `[num_blocks, block_size, n_kv_heads, head_dim]`, `block_size = 16`; and paging dissolves the transpose rather than optimizing it

**Status:** Accepted (revisit after the block-size sweep and FlashInfer verification)
**Date:** 2026-07-31

**Context**

The engine stores KV as `(kv_seq, n_kv_heads, head_dim)` (`engine/cache.py:28`) while its CUDA decode kernel demands `(n_kv_heads, kv_seq, head_dim)` (`kernels/attention_decode.cu:31`). The bridge is `components_gpu.py:153-154`, which does `.transpose(0,1).contiguous()` on the **entire** K and V, every layer, every token — roughly 67 MB of copy traffic per token at kv_seq 2048 (`BENCHMARKS.md:149`). `BENCHMARKS.md:151` names the split between this and other causes as unmeasured. PRD §6 lists "store the KV cache in kernel layout" as a nice-to-have optimization.

Separately, the paged design needs a physical pool layout, and the choice interacts with what the consuming kernel expects.

**Decision**

Per layer: `[num_blocks, block_size, n_kv_heads, head_dim]`, fp16. `block_size = 16` initially, treated as a swept tunable, not an assertion (PRD §C5, ARCHITECTURE §3.1).

Block-contiguous, so appending a token is a single scatter through `slot_mapping`, computed CPU-side during batch assembly as
`slot = block_tables[seq][pos // block_size] * block_size + (pos % block_size)` (ARCHITECTURE §2.3).

**And the framing that matters:** the transpose cost exists because a layout mismatch forces a **whole-cache** copy. It is not a paging problem and paging does not inherit it. A paged kernel reads `block_tables` and gathers only the blocks it needs — or, in FlashInfer's case, reads the pool directly with no gather at all. Choose the layout the consuming kernel wants and the copy disappears rather than being optimized. **Paging didn't make the transpose cheaper; it made the transpose unnecessary** (ARCHITECTURE §3.2).

Sizing arithmetic: `2 (K,V) × 16 layers × 16 tokens × 8 kv_heads × 64 head_dim × 2 bytes = 512 KB per block` across all layers. **[inference]** On a 40 GB A100 with ~35 GB usable for KV that is roughly 70k blocks ≈ 1.1 M tokens of KV, versus 2048 tokens *per request* reserved up front today (`engine/scheduler.py:16,26-27`). That is the shape of the S1 claim and it must be **measured, not computed**, for publication.

**Alternatives considered**

- **Fix the transpose in place — store the contiguous cache in kernel layout (`BENCHMARKS.md:151`, PRD §6).** Rejected as a serving decision, and this is the interesting rejection: it is a *correct* optimization that this project renders moot. It would remove the copy while leaving the per-request `max_seq` over-allocation untouched. Paging supersedes it. Keeping both would mean maintaining two cache layouts for one benefit.
- **Head-major block layout `[num_blocks, n_kv_heads, block_size, head_dim]`.** Not rejected on merit — deferred to verification. It is a plausible kernel preference, and the whole point of §3.1 being asserted "as reasonable, not as FlashInfer-compatible" (ARCHITECTURE §11) is that the *consuming kernel* decides this, not aesthetics. `PagedTorchBackend` is written first precisely because it is layout-independent, so a flip here costs an adapter rather than a redesign.
- **Token-granular allocation (`block_size = 1`).** Rejected. Maximal memory efficiency and maximal prefix-match precision, but block tables become as long as sequences, per-token metadata dominates, and the scatter/gather degenerates. It also destroys the copy-on-write story, which only exists because blocks are shared units.
- **Large blocks (64, 128).** Not rejected — it is one end of the sweep. The tradeoff is stated: **block size trades cache-hit precision against block-table overhead** (ARCHITECTURE §4). Larger blocks mean shorter block tables and fewer allocator operations, but coarser prefix matching, since a partially matching block is *not* reusable — a 20-token match at `block_size=16` yields **one** reusable block, not 20 reusable tokens.
- **fp8 or quantized KV.** Rejected as out of scope. It is an engine-lane concern (quantization is on the engine's claim set, PRD §C7) and it would confound every memory-capacity comparison in this repo.

**Consequences**

- The whole-cache transpose does not appear anywhere on the paged path. The engine's contiguous path keeps it, unchanged, because that path is frozen (ADR-009).
- Internal fragmentation of at most `block_size - 1` tokens per sequence. At 16, that is bounded and small relative to the 2048-slot reservation it replaces.
- Cache hit rate must be reported at **block** granularity (METHODOLOGY §7), because that is the granularity at which work is actually saved. A hit rate reported per token would be a different, flattering number.
- Block-boundary handling becomes the highest-density bug region in the project — the divergence-mid-block case (ARCHITECTURE §9.1), COW on partially-filled shared blocks, `slot_mapping` off-by-ones. This is why the adversarial near-miss workload exists (METHODOLOGY §4) and why Phase 1's DoD requires bit-identical output at ≥3 lengths that straddle block boundaries (PHASE_PLAN §4).
- Makes harder: comparing memory numbers against the engine's, since the units differ (blocks vs reserved slots). Every S1 figure must state both.

**Revisit if:** FlashInfer's verified layout contract disagrees (then §3.1 changes and nothing above it does, per ARCHITECTURE §11), or the block-size sweep shows a different size dominating on hit rate *and* overhead.

---

## ADR-006 — Implement **both** recompute and swap preemption behind a policy flag rather than picking one

**Status:** Accepted (revisit at the freeze line)
**Date:** 2026-07-31

**Context**

When KV blocks run out mid-batch, a victim sequence must be evicted. There are two ways to make room, and the literature has a house answer that is easy to recite and hard to defend.

**Recompute** — free the victim's blocks entirely; on reschedule, re-prefill its prompt *and its generated-so-far tokens*. Cost: O(current length) of prefill compute, paid later. Frees 100% of the victim's blocks immediately. No host memory, no transfer bandwidth.

**Swap** — copy the victim's KV blocks to pinned host memory, free the GPU blocks, copy back on resume. Cost: 2× PCIe transfer of the victim's KV. Frees the GPU blocks; consumes pinned host memory.

The comparison is genuinely not universal. It turns on the ratio of recompute cost to transfer cost, which depends on model size, sequence length, and PCIe bandwidth. Recompute is favored when the sequence is short, the model is small relative to KV volume, or host memory/PCIe is contended — prefill is highly parallel and compute-bound, and a 1B model's prefill is fast. Swap is favored when sequences are long or the model is large (ARCHITECTURE §5.2).

PRD §5 names preemption the **deepest systems content in the project**.

**Decision**

Both are implemented, selected by a policy flag, and benchmarked head-to-head under forced memory pressure, swept over sequence length to find (or fail to find) a crossover (PHASE_PLAN §6).

The prediction is recorded **before** measurement. **[inference]** For Llama 3.2 1B specifically, recompute is expected to win at nearly all lengths: the model is tiny, prefill is cheap, and KV per token is small — `8 kv_heads × 64 head_dim × 16 layers × 2 (K,V) × 2 bytes = 32 KB/token` — so neither side is under real pressure.

That prediction is exactly why it is worth measuring. A measured result showing recompute dominating on a small model, with the crossover *projected* for larger ones, is a better answer than reciting the vLLM paper.

**Alternatives considered**

- **Recompute only.** Rejected as the plan; retained as the explicit freeze-date cut (~12h instead of ~23h, PHASE_PLAN §6). Recompute alone is sufficient for the system to be *honest* above the knee, which is all that PHASE_PLAN's property 4 demands. What is lost is the comparison — and the comparison, not the mechanism, is what makes this the best systems topic in the project. The bullet degrades from "measured [policy] faster by [N]% at [length]-token sequences and published the crossover analysis" to "implemented preemption with recompute."
- **Swap only.** Rejected outright. It is the strictly worse single choice here: it needs pinned host memory management and a transfer path, and **[inference]** it is predicted to *lose* on this model. Implementing only the policy expected to lose, and then having no measurement to say so, is the worst of both.
- **Pick one by reading the vLLM paper and citing it.** Rejected, and this is the real alternative being rejected. Citing someone else's crossover is precisely the answer that collapses under one follow-up question — *"does that hold for your model?"* The project's premise (PRD §G3) is to *"defend recompute-versus-swap with measurements from this system, not from a paper."*
- **A dynamic/adaptive policy that picks per victim** (e.g. by sequence length against a measured threshold). Deferred, not rejected. It is the obvious follow-on, but it presupposes the crossover measurement this ADR exists to produce. Building the adaptive policy first would mean hard-coding a threshold nobody has measured.

**Consequences**

- ~23h in Phase 3, sitting directly on the freeze line (~100h cumulative). This is the decision most exposed to schedule risk, which is why the cut is pre-planned rather than improvised.
- Two preemption paths means two ways to silently corrupt output. The gate is the same for both and is non-negotiable: **greedy output under forced preemption must be bit-identical to greedy output without preemption** (ARCHITECTURE §5.2, PHASE_PLAN §6). This is the single most important test in the project, because a preemption bug produces plausible text, degrades no metric, and would be discovered by nobody.
- Swap requires pinned host memory management and a GPU↔host transfer path that exists for no other reason. Real complexity, carried for a comparison.
- Preemption metrics must break out by policy (METHODOLOGY §2), so the harness carries a dimension it would not otherwise have.
- **The interaction worth knowing cold:** recompute is cheaper than the naive analysis suggests once a radix cache exists (P4), because the victim's own prefix blocks may still be resident — so "re-prefill" often means re-prefilling only the generated tail (ARCHITECTURE §9.2, step 5). This is a real reason to expect recompute to win here, and it is a feature-interaction answer rather than a recited one.
- Makes harder: reporting a single preemption cost. Every preemption number is now two numbers plus a length sweep.

**Revisit if:** Phase 2 overruns and the freeze line is threatened — then swap is cut first (PHASE_PLAN §6, and the cut order in the freeze-line section: swap and FlashInfer, ~17h together, before anything in Phase 2). Also revisit if the sweep finds no crossover in the reachable length range, in which case the honest publication is the null result plus the projection, not a manufactured crossover.

---

## ADR-007 — The scheduler runs as a cooperative asyncio task, not a dedicated thread

**Status:** Accepted (revisit if Python-side batch assembly is material in a profile)
**Date:** 2026-07-31

**Context**

The engine has **zero thread safety**: a single global model, mutable cache state, no locks anywhere (`engine/server.py`, `engine/scheduler.py`). Any design that touches the model from more than one thread is wrong by construction.

The engine's server also has a structural bug this project must not inherit: `engine/server.py:69` iterates a blocking synchronous generator inside an `async def`, pinning the event loop for an entire generation. Request 2's TTFT includes request 1's full decode, which makes any p99 claim about that server meaningless (`SERVING_INTERFACE.md:130`).

**Decision**

Per replica: one asyncio event loop handles HTTP ingress, SSE streaming, and client-disconnect detection. The scheduler runs as a **cooperative asyncio task**, `await`ing between iterations, where each iteration is one bounded forward pass. The model executes on one CUDA stream; there is no intra-replica parallelism across requests — **batching *is* the parallelism**. Per-request `asyncio.Queue` carries output tokens from the scheduler to the SSE writer (ARCHITECTURE §7).

The escape hatch is stated in advance rather than discovered: if profiling shows Python-side batch assembly is a material fraction of step time, the scheduler moves to a thread with `asyncio.run_coroutine_threadsafe` handoff. That is a contained change, not a redesign.

**Alternatives considered**

- **Dedicated scheduler thread.** Rejected for now. It needs thread-safe queues in both directions, and it would *still* serialize on the GIL for the Python-side batch assembly — buying only the ability to overlap Python bookkeeping with GPU execution. Since a bounded step already keeps the loop responsive, the added complexity buys little (ARCHITECTURE §7).
- **Separate scheduler process with IPC** (the vLLM-style split). Rejected. Correct at scale and wrong here: it adds serialization cost on the token path, a second process to supervise, and a whole failure mode, in exchange for GIL relief this workload has not been shown to need. It is also a much larger blast radius than the stated escape hatch.
- **Thread pool over requests.** Rejected outright — it is the design the engine's zero thread safety forbids, and it misunderstands the problem: the GPU is the serialized resource, so concurrency across requests is achieved by *batching them into one forward pass*, not by running them on separate threads.
- **Keep the engine's blocking-generator-in-`async def` shape.** Rejected; it is the exact bug being fixed (`engine/server.py:69`).

**Consequences**

- The event loop is released between iterations, so HTTP work interleaves with decode at ~10–30 ms granularity (ARCHITECTURE §7).
- **Load-bearing dependency:** a step is only non-blocking if a step is *short*. Chunked prefill is what bounds step duration — a 2000-token prefill run as one unit would pin the loop for its full duration. **[inference]** The throughput feature and the concurrency model are therefore the same mechanism (ARCHITECTURE §5.1). This is why chunked prefill's step-duration bound is a *measured* item in Phase 4's DoD (PHASE_PLAN §7), not an assumption.
- Python-side batch assembly (building `BatchMeta`, `slot_mapping`, block tables) is on the critical path and cannot overlap GPU execution. That is the accepted cost.
- **Cancellation becomes a memory-correctness requirement.** Client disconnect sets a flag on the sequence; the scheduler retires it at the next iteration boundary and explicitly frees its blocks and decrements refcounts. The engine gets abandonment right by accident (`engine/scheduler.py:11-45` stops at the next `yield`) but has *no way to reclaim the cache*, because `KVCacheGPU` is constructed per call and garbage-collected (`:26-27`). Here, **cancellation must free memory, or a disconnect-heavy workload leaks the entire KV pool** (ARCHITECTURE §7).
- Makes harder: any future CPU-heavy work on the replica (tokenization of very long prompts, complex admission policies) — it lands directly on the loop.

**Revisit if:** a profile shows Python-side batch assembly is a material fraction of step time, or if step duration cannot be bounded tightly enough for TTFT targets under chunked prefill. Migration path is already chosen: thread + `run_coroutine_threadsafe`.

---

## ADR-008 — Separate repo consuming the engine as a pinned git submodule, not a monorepo

**Status:** Accepted (revisit before the repo goes public)
**Date:** 2026-07-31

**Context**

The serving layer sits directly on top of `../llm_inference_engine` and requires changes to it (ARCHITECTURE §2.7). The engine's compiled kernel `.so` lands in a gitignored `build/` directory, and `engine/model_gpu.py:45-53` manipulates `sys.path`, which breaks under a non-editable install (PRD §6).

The project also has a non-technical constraint that is a real constraint: **two repos carry two independent claim sets**, split as engine = model internals, kernels, quantization; serving = paged KV, continuous batching, radix prefix caching, prefix-aware routing (PRD §C7).

**Decision**

Two repos. The engine is consumed as a **git submodule pinned to a tag** (`v0.1.0`, created 2026-07-31 at HEAD `6ff40a1`). Changes to the engine are small, additive, upstreamed, and enumerated (PRD §C1, ARCHITECTURE §2.7).

Submodule specifically, rather than a published package, because the compiled `.so` must live at a known relative path — trivial for a submodule, awkward for an installed package (`SERVING_INTERFACE.md:234`).

**Alternatives considered**

- **Fork the engine into a monorepo (ARCHITECTURE §10/A1).** Rejected. Two reasons, one technical and one not, and both are real.
  - *Technical:* the dependency boundary is what **forces a real interface**. Inside one repo, the fastest path to any problem is reaching into engine internals, and within a week there would be no seam to defend — `BatchMeta` and `AttentionBackend` exist as a *contract* precisely because crossing them is expensive.
  - *Non-technical:* the claim sets collapse into one. That is not vanity; the project's stated purpose (PRD §1) is to be a defensible line about serving systems, distinct from a line about model internals.
  - Cost, stated plainly: cross-repo changes are slower during Phase 1. Mitigated by the submodule pin, so the engine tracks a known SHA and the `.so` sits at a known relative path.
- **Vendor the engine's source into this repo.** Rejected (PRD §3). It is a fork with worse provenance — upstream fixes must be manually re-applied, and the engine's own test suite stops being this project's regression gate.
- **Put batching and paging *in* the engine (ARCHITECTURE §10/A2).** Rejected. It collapses the two independent claim sets, and puts the engine's clean batch-1 benchmark claims at risk of numeric drift (`SERVING_INTERFACE.md:284`). PRD §6 states the boundary flatly: continuous batching, the scheduler, the allocator, eviction, and routing are **explicitly not engine changes.** The engine exposes hooks; the serving layer owns the scheduler.
- **Bypass the engine's attention entirely from the serving layer (ARCHITECTURE §10/A4)** — reimplement the layer loop here. Rejected as strictly worse than either the hook or a kernel rewrite: it means forking `model_gpu.py` into this repo, duplicating code without owning it.
- **Publish the engine as a versioned wheel and depend on it normally.** Rejected for now, and it is the alternative most likely to win later. It is the cleaner distribution story, but today the `.so` is gitignored and `sys.path` is manipulated (`engine/model_gpu.py:45-53`), so a non-editable install does not work. That packaging cleanup is listed as nice-to-have, not prerequisite (PRD §6).

**Consequences**

- Every seam change is a two-repo change plus a pin bump. This is the direct reason ADR-004 refuses to design the protocol twice.
- Contributors (and CI) must remember `--recurse-submodules`, and CI must build the kernel or skip GPU tests. Phase 0's DoD requires `pip install -e` + submodule to resolve on a clean PACE allocation (PHASE_PLAN §3).
- The engine's existing test suite becomes this project's free regression gate — but only because of ADR-009's additive rule.
- Every published artifact must record **both** the serving-repo git SHA and the pinned engine SHA/tag (METHODOLOGY §11), because a number is meaningless without knowing which engine produced it.
- Makes harder: rapid iteration on the seam during Phase 0–1, exactly when the seam is least settled.

**Revisit if:** the repo goes public (PRD §C1 explicitly schedules a reassessment there), or if packaging cleanup lands and a versioned wheel becomes viable.

---

## ADR-009 — Every engine change is strictly additive; no existing code path changes numerics

**Status:** Accepted
**Date:** 2026-07-31

**Context**

The engine's published claims are all statements about the **contiguous, batch-1 path**: `BENCHMARKS.md:102` (v3 kernel ≈ 0.98–0.99× PyTorch SDPA), `tests/test_forward.py:113,137` (logits within 1e-3), `tests/test_decode.py:30-48` (32 greedy tokens bit-identical to HF). Those claims are on a resume. This project needs to modify the same files those claims live in.

ARCHITECTURE §2 calls this boundary load-bearing: *"Everything else can be refactored; this boundary cannot, because it spans two repos and two independent claim sets."*

**Decision**

> **Every engine change is additive. No existing code path is modified in a way that changes its numerics.**

Concretely (ARCHITECTURE §2.7):
- `prefill()` and `decode_step()` (`engine/model_gpu.py:58,127`) keep their exact current behavior and remain the engine's reference path. The batched path is a **new sibling method**, `forward_varlen`.
- `gqa_attention_gpu` gains `backend=None, meta=None` and a new branch placed *before* the existing cache logic at `components_gpu.py:137`. The existing path stays byte-for-byte unchanged.
- `KVCacheGPU` (`engine/cache.py:23-33`) is untouched and still serves the reference path. The engine never touches `.k`, `.v`, or `.pos` on the paged path.
- `engine/attention_backend.py` is a new file containing pure interface and no behavior.

One narrow exception, taken deliberately: the chunked-prefill position fix at `engine/model_gpu.py:65`. `torch.arange(seq)` starts at 0 regardless of cache position, so a second prefill chunk would receive positions `0..n` instead of `pos..pos+n`. It is invisible today only because prefill is called exactly once (`engine/scheduler.py:33`). The fix — `torch.arange(kv_cache.pos, kv_cache.pos + seq)` — is applied to the reference path too, with a test, because it is a latent bug regardless of this project.

Also fixed: `tests/test_generate.py`'s `"input_ids"` → `"token_ids"` key (`:42,52,63,78,89` vs `tests/oracle.py:170`). All 5 tests currently raise `KeyError`, meaning the KV-cache-vs-no-cache correctness gate **is not enforced today**. That is a test fix, not a numerics change.

**Alternatives considered**

- **Refactor the engine's forward path to be batch-native and delete the batch-1 special case.** Rejected. It is the cleaner end state and it destroys the thing being protected: every existing benchmark and correctness claim would need re-validation, and any drift would be discovered on the engine's claim set rather than here. It also removes the free regression gate — with the old path intact, the engine's existing tests *are* the guarantee.
- **Additive but with the old path reimplemented as a thin wrapper over the new one** (`prefill()` calls `forward_varlen` with `n_seqs=1`). Rejected, and this is the tempting one. It removes duplication, but it silently makes every existing claim a claim about the *new* code — fp16 reduction order can differ between a batched kernel and the original, and "bit-identical to HF for 32 tokens" (`tests/test_decode.py:30-48`) is exactly the kind of assertion that would break. Duplication is the cheaper price.
- **Fix the position bug only on the new path.** Rejected. It is a real latent bug in shipped code; leaving it because it is currently unreachable means shipping a known defect. **[inference]** The reference-path fix is low risk specifically because prefill is called exactly once today (`engine/scheduler.py:33`), so `kv_cache.pos` is 0 and the new expression is numerically identical to the old one — the change is a no-op on every path the existing tests exercise, which is why it is compatible with this ADR rather than an exception to it.

**Consequences**

- The engine's existing test suite becomes the regression gate for free. Phase 0's DoD requires all engine tests green **including the 5 that currently error** (PHASE_PLAN §3).
- Two attention paths coexist in `components_gpu.py` forever, and two forward entry points in `model_gpu.py`. Accepted duplication.
- The engine's contiguous path keeps its whole-cache transpose (`components_gpu.py:153-154`, ~67 MB/token at kv_seq 2048 per `BENCHMARKS.md:149`). It is not optimized; it is simply not on the serving path (ADR-005).
- Total prerequisite engine work is bounded at ~6–10 hours (ARCHITECTURE §2.7), which is what makes Phase 0 a 16-hour gate rather than a project.
- Makes harder: any change that genuinely *requires* touching the old path — there is no approved route for one, and it would need a new ADR.

**Revisit if:** the two paths diverge enough that maintaining both is a real cost, which would only plausibly happen well after the engine's claims are no longer load-bearing on a resume.

---

## ADR-010 — The serving layer owns the production HTTP surface; the engine's server stays a single-request reference path

**Status:** Accepted
**Date:** 2026-07-31

**Context**

The engine already has an HTTP server, and its README scopes it as single-request (`README.md:144`). Two facts make it unusable as a production surface:

1. **It cannot reach the GPU.** `engine/server.py:24-30` imports `engine.model.LlamaModel` — the fp32 NumPy CPU path. There is no flag or code path to `LlamaModelGPU`.
2. **It serializes under load.** `engine/server.py:69` iterates a blocking synchronous generator inside an `async def`, pinning the event loop for the whole generation. Request 2's TTFT includes request 1's entire decode (PRD §1, `SERVING_INTERFACE.md:130`).

There is also a claim-boundary constraint: the phrase *"OpenAI-compatible server"* is already spent on the engine line and must not reappear on the serving line (PRD §C7, `SERVING_INTERFACE.md:157`). One protocol-compatibility claim across both repos.

**Decision**

The production HTTP surface — FastAPI ingress, SSE streaming, admission control, client-disconnect propagation, cancellation into the scheduler — lives in this repo (PRD §3, ARCHITECTURE §1). The engine's server stays as it is: a single-request reference path.

`--backend gpu` on the engine's reference server is listed as **nice-to-have, explicitly not blocking** (PRD §5 Tier 0, §6), because the serving layer builds its own surface regardless.

**Alternatives considered**

- **Fix and extend the engine's server into the production surface.** Rejected. Fixing the blocking-generator bug at `engine/server.py:69` properly means introducing the scheduler, the admission controller, and the async token queues — i.e. building this project inside the engine repo, which ADR-008 rejects for its own reasons. It would also violate ADR-009's additive rule on the engine's most user-visible file.
- **Delete the engine's server since this one supersedes it.** Rejected. It is the engine's own published claim (`README.md:144`) and its single-request reference behavior is useful for A/B sanity checks.
- **Serve the OpenAI protocol from the router only, with a private protocol replica-side.** Considered, and partially adopted: the router exposes the OpenAI-compatible ingress to clients (PHASE_PLAN §8). Replicas still speak the same shape so a single replica can be benchmarked directly without the router — which baseline B1 requires (METHODOLOGY §6).

**Consequences**

- Baseline B1 has a subtlety that must be stated in every result: the engine's server cannot reach the GPU path, so B1 is measured *either* against a serving-layer configuration with batching and paging disabled, *or* against a patched engine server — and **which one is used is stated** (METHODOLOGY §6). Preference is the former, so the HTTP stack is held constant and only the scheduler differs.
- TTFT's definition must guard against a quiet trap the engine's server demonstrates: it emits a chunk per token id and detokenizes with `skip_special_tokens=True` (`engine/server.py:70`), so leading special tokens produce chunks with **empty `content`**. TTFT is therefore defined against the first chunk with non-empty `delta.content` (METHODOLOGY §2). Timing to the first chunk would understate TTFT.
- The engine's server remains a known-serializing path. It is not benchmarked as a serving system and no p99 claim is made about it.
- Claim wording constraint carries forward: the serving line does not say "OpenAI-compatible server" (PHASE_PLAN §11).
- Makes harder: keeping two HTTP surfaces roughly protocol-compatible as the API surface grows.

**Revisit if:** never, realistically. The `--backend gpu` nice-to-have may land opportunistically; it changes nothing here.

---

## ADR-011 — Goodput under a declared SLO is the primary metric; raw throughput is reported but is not the claim

**Status:** Accepted
**Date:** 2026-07-31

**Context**

Raw throughput is trivially gameable: batch harder, tolerate worse latency, throughput goes up. A serving system achieving high tok/s while every request violates its latency target has served no one. Conversely a latency-only metric ignores that the system exists to do volume (METHODOLOGY §3).

The commitment is made at PRD level (§C6) rather than only in the methodology doc, so that phase definitions-of-done can depend on it.

**Decision**

**Goodput** — completed requests/sec that met the SLO, over the steady-state window — is the headline metric. Throughput (output tokens/sec) is reported but is not the claim.

The SLO is declared **in advance**, subject to one revision after a calibration run; if revised, both the original and the revision are published with the reason:

```
A request meets the SLO iff:
    TTFT      <  [TBD_ttft] ms
AND p95 ITL   <  [TBD_itl]  ms   (computed within that single request)
AND the request completed (not dropped, not timed out, not errored)
```

Placeholders are deliberate. Thresholds are set from a calibration run against the **unloaded batch-1 engine on the target GPU in the same allocation**, as a stated multiple of unloaded performance, so the SLO is anchored to something physical rather than to a round number. Reference points, with their own caveat: A100 unloaded ITL p50 ~12.5–12.6 ms, p99 ~12.8–13.2 ms at ~79 tok/s (`BENCHMARKS.md:37-39`), or ~16.1–16.8 ms at ~60–62 tok/s in later sessions (`BENCHMARKS.md:139-140`) — and **those two sessions are not comparable to each other** (`BENCHMARKS.md:17`).

The headline result is a **curve, not a point**: goodput (y) vs offered load (x), SLO-violation fraction on a secondary axis, with three named regions — below capacity, the knee, above capacity. **A single goodput number without the curve it came from is not publishable in this project** (METHODOLOGY §3).

**Alternatives considered**

- **Raw throughput (tok/s) as the headline.** Rejected. Gameable in one direction and it is the number a reader will discount first. It is still reported, because a system that meets the SLO at trivial volume has also not served anyone.
- **Latency percentiles as the headline.** Rejected for the mirror reason: optimize p99 by admitting almost nothing and the number looks excellent. Goodput penalizes both directions, which is exactly the property being bought.
- **Requests/sec completed, no SLO filter.** Rejected — it is throughput with worse units and it hides whether anyone was served acceptably.
- **Set the SLO after seeing results.** Rejected explicitly: *"An SLO chosen after seeing results is not an SLO"* (METHODOLOGY §3). This is the single most common way a goodput number becomes meaningless, and the mitigation is a pre-registered threshold plus a published revision trail.
- **A round-number SLO (e.g. TTFT < 500 ms).** Rejected. Anchoring to a round number invites *"why 500?"* with no answer. Anchoring to k× unloaded batch-1 performance in the same allocation makes the threshold defensible and makes it travel across hardware.

**Consequences**

- Every goodput result carries an SLO definition and the calibration run that produced it. Results without both are not publishable.
- The SLO must be frozen after Phase 2's calibration (PHASE_PLAN §5) and cannot be re-tuned when a later phase's numbers disappoint.
- Cross-hardware comparison gets harder, since the SLO is anchored per-allocation. Accepted — METHODOLOGY §5 forbids cross-session comparison anyway (ADR-021).
- The reported number is the **knee**, which is a curve feature and requires a load sweep at every configuration. That multiplies benchmark runtime substantially versus reporting a single point.
- Above the knee, load shedding is what holds goodput flat rather than letting it collapse — which is the entire argument for having load shedding, and makes it a measurable feature rather than a checkbox.
- Makes harder: quick comparisons. There is no single headline number to quote out of context, by design.

**Revisit if:** the calibration run shows the anchoring multiple produces an SLO that essentially no configuration meets (or that every configuration meets), in which case one revision is permitted and both versions are published.

---

## ADR-012 — Open-loop load generation, with intended-dispatch timing

**Status:** Accepted
**Date:** 2026-07-31

**Context**

**Closed loop:** N clients, each sends a request, waits for the full response, sends the next. Offered load is a *function of system speed* — if the system slows, clients send less. Concurrency is capped at the client count by construction.

**Open loop:** requests arrive on a schedule independent of the system. If the system slows, arrivals keep coming and the queue grows. Offered load is a *parameter*.

A closed-loop harness **cannot measure queueing delay**, because it cannot create a queue deeper than its client count. Under a closed loop latency degrades gracefully and monotonically and the system appears to have no saturation point. Under an open loop the same system shows a knee: flat below capacity, divergent above. **The knee is the interesting result and closed-loop testing hides it** (METHODOLOGY §1).

The engine's existing harness is strictly closed-loop, one client, zero think time: `bench/harness.py:156-180` is a nested `for prompt` × `for run`; `:106-139` consumes one generator to completion; there is no arrival process, no queue, no concurrency, no HTTP client at all (`CLAIMS_AUDIT.md:259`). It is a correct microbenchmark for batch-1 decode and structurally incapable of measuring anything this project claims.

**Decision**

Primary results are open-loop. Requests are generated by an arrival process at a configured rate and submitted regardless of outstanding work, over real HTTP with SSE parsing. **Poisson arrivals** are the default (genuine bursts rather than the artificial smoothness of fixed-interval arrivals, and burstiness is what stresses admission control); a deterministic fixed-rate mode also exists, because comparing Poisson vs deterministic at the same mean rate isolates *burstiness* from *load* (METHODOLOGY §4).

Closed-loop runs are permitted **only** as a labeled secondary result, for measuring per-request service time in isolation, and must be labeled `closed-loop, N clients` wherever they appear.

**Coordinated omission must be avoided.** Latency is measured from the **intended dispatch time** derived from the arrival schedule, not from actual send time. Drift between intended and actual send time is recorded and reported as a harness health metric; above a threshold the run is **invalid and discarded, not published** (METHODOLOGY §1).

**Alternatives considered**

- **Closed loop with N clients** (the default in most quick harnesses, and what the engine already has at `bench/harness.py:156-180`). Rejected. *"A closed-loop p99 at 32 clients is a statement about 32 clients, not about the system. It is not comparable to anyone else's number, and it will be dismissed by anyone who has built a serving system"* (METHODOLOGY §1). It also cannot produce the goodput-vs-offered-load curve ADR-011 requires, because offered load is not a controllable input.
- **Extend the engine's harness rather than rewriting.** Partially adopted, deliberately split (METHODOLOGY §9). Reused as-is: `_percentile` (`bench/harness.py:50-51`), `_hw_metadata` (`:69-99`), `write_results` (`:187-202`), `_weight_mem_mb` (`:54-66`), and `_time_cuda` from `bench/bench_attn_kernel.py:32-44` — which is the pattern for server-side GPU timing. Rewritten: `time_generate` (`:106-139`) and `run_benchmark` (`:156-180`), because they are closed-loop, single-stream, in-process, and have no HTTP client. **It is reused for components, not as a harness.**
- **Open loop measuring latency from actual send time.** Rejected — this is coordinated omission with an open-loop label. If the generator ever blocks, the requests that *should* have been issued during a stall never appear in the distribution, and a saturated system looks fast. **[inference]** This is the single most likely way this project publishes a wrong number without noticing, and it is in the risk register as a *silent* invalidator (METHODOLOGY §1, §12).
- **In-process load generation (no HTTP).** Rejected. It skips the SSE framing, the event loop contention, and the connection handling that are part of what is being measured — and TTFT is defined against SSE chunk arrival (METHODOLOGY §2).

**Consequences**

- The harness must maintain a dispatch schedule independent of response handling, which means genuinely concurrent request issuance and an SSE client that does not serialize.
- Above the knee, **steady state does not exist by definition** — the queue grows without bound. Those runs are measured over a fixed window and explicitly labeled *unsaturated-window measurement*, not *steady state* (METHODOLOGY §4). **[inference]**
- Runs can be invalidated by harness health rather than system behavior, and discarded runs cost allocation time.
- Measurement windows must exclude ramp-up and drain, with boundaries recorded per run — including the drain tail is a way to accidentally publish a better p99 than the system has (METHODOLOGY §4).
- The open-loop harness is Phase 2 scope and is explicitly the thing **not** to cut: *"If something must go, cut Tier-3 metrics and Grafana, not the load model"* (PHASE_PLAN §5).
- Makes harder: comparing against anyone who published closed-loop numbers. That comparison is simply not made.

**Revisit if:** never for primary results. Closed-loop is already permitted, labeled, for service-time isolation.

---

## ADR-013 — No vLLM throughput comparison; vLLM is used as a scaling-shape and correctness reference only

**Status:** Accepted (revisit after one attempt)
**Date:** 2026-07-31

**Context**

vLLM is the obvious comparison for a paged-KV continuous-batching serving layer, and "how does it compare to vLLM" is a question this project will be asked. PRD §3 already lists *"beating vLLM on absolute throughput"* as a non-goal — *"not achievable and not the claim"* — and defers the fairness question to the methodology.

The confounds are structural, not tunable (METHODOLOGY §8):

- **Kernels.** vLLM ships fused, tuned kernels across the whole forward pass. This project's forward pass is the engine's — hand-written PyTorch ops whose linear layers are plain `x @ w.T` (`engine/components_gpu.py:24`), on an engine that already measures ~5× slower than llama.cpp at batch 1 (`BENCHMARKS.md:31`). Any throughput comparison is dominated by kernel quality, which is not what this project is about.
- **Runtime.** CUDA graphs, custom allocators, prefix-cache implementation details, tokenizer path, continuous-batching heuristics — all differ.
- **Configuration surface.** vLLM has many knobs with tuned defaults; comparing against defaults measures tuning effort, not design.

A fair comparison would require holding the forward pass constant, which is impossible across two engines.

**Decision**

Committed **in advance**, so the position is a stated methodology rather than something discovered mid-benchmark and quietly dropped (METHODOLOGY §8):

1. **Do not publish "we beat vLLM" or "we are X% of vLLM" as a throughput claim.** Dishonest in either direction — flattering if the workload favors this system, self-flagellating if it does not, uninformative either way.
2. **Do publish vLLM as a *scaling-shape* reference where the shapes are comparable.** Does goodput-vs-offered-load knee in the same place *relative to each system's own batch-1 capacity*? Does cache hit rate as a function of sharing rate follow the same curve? These are design comparisons and they normalize away kernel quality.
3. **Do use vLLM as a correctness and sanity reference.** If this system's hit rate at a given sharing rate is wildly different from vLLM's on the same workload, one of them has a bug, and finding out which is valuable regardless of the answer.
4. **If even the shape comparison proves unfair, say so and drop it**, in writing, with the reason.

**Alternatives considered**

- **Publish an absolute throughput comparison anyway, with caveats.** Rejected. Caveats do not survive being screenshotted, and the number would be read as the claim regardless of the prose around it. The comparison would measure kernel quality — a lane this project explicitly deprioritized (ADR-001, ADR-019).
- **Normalize by making both systems use the same kernels.** Rejected as impossible in practice: it would mean porting the engine's forward pass into vLLM or vLLM's kernels into the engine, either of which is a larger project than this one.
- **Compare against vLLM with vLLM detuned to a comparable configuration** (e.g. eager mode, CUDA graphs off, default block size). Rejected. It manufactures a favorable baseline, which is the exact failure mode METHODOLOGY §10 warns about for routing workloads. Handicapping the competitor is worse than not comparing.
- **Ignore vLLM entirely.** Rejected. Dropping it silently is indistinguishable from having tried and lost. Point 4 exists so that even the *failure* of the shape comparison is a published result — and **[inference]** *"I tried to benchmark against vLLM, here is specifically why the comparison isn't meaningful, and here is what I compared instead"* is a stronger settled answer than a comparison table.

**Consequences**

- The strongest baselines available are internal and precisely specified: B1 (unmodified engine, batch 1, over HTTP), B2 (static batching — same kernels, same memory manager, *only* scheduling differs, which is what makes it the honest comparison), B3 (contiguous per-sequence allocation), B4/B5 (routing). All are defined precisely enough to be reimplemented by a skeptic (METHODOLOGY §6).
- The vLLM cross-check may catch real bugs in prefix-cache hit-rate accounting, which is a benefit independent of any published number.
- Cost: the project cannot answer *"how fast is it, really, versus production systems?"* with a number. It answers with a shape or with a stated refusal.
- Makes harder: positioning the project against production systems for a reader who wants one number.

**Revisit if:** the shape comparison is attempted and either succeeds (publish it) or fails (publish the reason and drop it). METHODOLOGY §13 lists this as decided after one attempt, documented either way. The *throughput* comparison is not revisited.

---

## ADR-014 — No SQL or relational persistence; benchmark artifacts are JSON + CSV on the filesystem

**Status:** Accepted (revisit if run comparison becomes genuinely painful)
**Date:** 2026-07-31

**Context**

Serving systems commonly acquire a database — for request logs, for run history, for a results dashboard. The project has exactly one legitimate non-hot-path use: persisting benchmark runs for cross-session comparison (PRD §3).

Nothing on the request path needs durability. The state table (ARCHITECTURE §6) shows every piece of request-path state is process- or request-lifetime: KV blocks, allocator free list and refcounts, radix trie, waiting queue, running batch, per-sequence state. Losing any of it kills in-flight requests and nothing more.

**Decision**

No SQL, no relational store, nothing on the request path. Persistence is deliberately absent (PRD §3, ARCHITECTURE §6).

Benchmark runs go to git-tracked JSON + CSV, matching the engine's existing `write_results` shape (`bench/harness.py:187-202`, `{timestamp}_{hostname}_{backend}` naming), which is reused as-is (METHODOLOGY §9).

One schema change from the engine's version, and it is not optional: **raw per-request and per-token samples are stored**, not per-run percentiles. The engine computes p50/p99 per run and stores only the percentiles, discarding the raw samples (`bench/harness.py:136-137,178`) — which makes correct pooling impossible after the fact (METHODOLOGY §5, §9).

**Alternatives considered**

- **SQLite for run history.** Rejected for now, and it is the alternative most likely to become right. It would make cross-run queries pleasant. But it is not in the hot path, it is not a published claim, and a directory of JSON+CSV is diffable, greppable, reviewable in a PR, and readable without tooling. PRD §3: *"even that is a CSV/JSON directory first."* Adding it later is a pure import job, since the raw samples are retained.
- **Postgres / a real database.** Rejected. Operational weight (a service to run under Slurm, a schema to migrate) for zero request-path benefit, and it would read as padding next to measured systems claims — the same argument that rejects ADR-015.
- **A time-series database for metrics.** Partially deferred rather than rejected: Prometheus + Grafana are Tier 2 / Phase 6 for *live* observability (PRD §5, PHASE_PLAN §9). That is monitoring, not the artifact of record. The artifact of record stays a committed file, because a Grafana screenshot is not reproducible (ADR-018).
- **Request logging to durable storage on the serving path.** Rejected. It adds I/O to the hot path, and the failure model already says in-flight requests are lost on replica death (ADR-002) — durable request logs would imply a recovery story that does not exist.

**Consequences**

- Cross-run comparison is filesystem-and-script work rather than a query. Accepted; the number of runs is small.
- Raw-sample retention makes artifacts substantially larger than the engine's percentile-only files. That is the point — percentiles are computed at analysis time, so pooling is correct and the percentile method can be changed retroactively (METHODOLOGY §5).
- No request-path durability at all, which is consistent with the rest of the design and must be stated as a known limitation rather than discovered.
- Makes harder: any future dashboard over historical runs, and any claim about production-grade observability of request history.

**Revisit if:** run comparison becomes genuinely painful (PRD §3 states this trigger explicitly) — most likely when the routing phase multiplies configurations by replicas by sharing rates by offered loads.

---

## ADR-015 — Kubernetes plus a cache-aware inference gateway is not a phase

**Status:** Rejected
**Date:** 2026-07-31

**Context**

llm-d and the Gateway API Inference Extension are the current prior art for cache-aware LLM routing at cluster scope, and "did you deploy it on Kubernetes" is a question a reader may ask.

Target hardware is GT PACE Phoenix under **Slurm**, not Kubernetes (PRD §3). PRD §8/O1 (resolved 2026-07-31, measured on the cluster) establishes that up to 32 concurrent GPUs are available under the `inferno` QOS, with 8-GPU single nodes on `gpu-h100`, `gpu-h200`, and `gpu-l40s` — so multi-replica work needs no orchestration layer beyond a launcher on one node.

**Decision**

Rejected, and recorded as *"deferred, probably permanently"* (ARCHITECTURE §10/A8) / *"recommend against, permanently"* (PHASE_PLAN §10). It sits at Tier 4, evaluated on systems depth alone (PRD §5).

The substitute is explicit: **know the prior art and be able to place your own work against it.** If asked about llm-d or the Gateway API Inference Extension, *"that's the same idea as my router, at cluster scope"* is worth more than a half-built deployment (PHASE_PLAN §10).

**Alternatives considered**

- **Build the K8s deployment as a late phase.** Rejected on three independent grounds, any one of which suffices. (a) *Wrong hardware* — the cluster runs Slurm, so a K8s deployment would be built somewhere other than where every benchmark runs, and would therefore be unmeasured. (b) *Duplicates the router* — the gateway's headline feature is cache-aware routing, which is exactly what Phase 5 builds; deploying it would replace the project's own differentiating work with a config file. (c) *Reads as padding* next to five measured systems claims (PHASE_PLAN §10).
- **Use the gateway instead of building a router.** Rejected decisively. Prefix-aware routing is the distributed-systems claim and the acknowledged skill gap (PRD §1, PHASE_PLAN §8). Delegating it to a gateway deletes the reason the project exists.
- **Containerize without orchestration.** Adopted, at Phase 6 (PHASE_PLAN §9), Tier 3 (PRD §5). Docker gives reproducibility and CI value without importing an orchestration layer.
- **Benchmark against the gateway as a routing baseline.** Rejected for the same reason as ADR-013: the confounds (different engine, different kernels, different runtime) make it a comparison of implementations rather than of routing policies. B4 and B5 are implemented in-tree precisely so the comparison isolates policy (ADR-017).

**Consequences**

- The project has no deployment story beyond containers plus a Slurm launcher. This is a genuine gap for roles that screen on Kubernetes, and it is stated rather than hidden.
- Multi-replica orchestration is a launcher script on one 8-GPU node (PHASE_PLAN §8), which is simpler and — importantly — makes "N replicas" mean N whole GPUs with no asterisk (ADR-021).
- Preserves ~all of Phase 5's budget for routing policy rather than infrastructure.
- Makes harder: any claim about operating this system in a production cluster environment. The honest answer is that it has been operated under Slurm.

**Revisit if:** target hardware changes to a Kubernetes cluster, or a specific role makes K8s a hard filter. Neither is currently true.

---

## ADR-016 — Phase ordering: preemption (P3) ships before the radix prefix cache (P4)

**Status:** Accepted
**Date:** 2026-07-31

**Context**

Every phase must satisfy four properties (PHASE_PLAN §0): depend only on earlier phases; leave the system working and benchmarkable; earn at least one defensible published claim on its own; and **never leave a half-built feature that makes an earlier claim untrue.**

Property 4 is the one that does the real work. The radix prefix cache is the flashier feature and the more commonly recognized one. Preemption is less flashy and is named the deepest systems content in the project (PRD §5, Tier 1).

The freeze is late August, roughly four weeks / ~120 engineer-hours at an assumed ~30/week, with the freeze line at ~100h cumulative (PHASE_PLAN §0). PRD §8 already records the consequence: G1–G3 plus G6 are the realistic freeze-date target; G4 (prefix cache) and G5 (routing) are fall work.

**Decision**

Order: P0 prereqs (~16h) → P1 paged KV + allocator (~28h) → P2 batched varlen + continuous batching + HTTP + open-loop harness (~33h) → **P3 preemption (~23h) ← freeze line** → P4 radix cache + chunked prefill (~28h) → P5 multi-replica routing (~36h) → P6 observability/reliability (~20h).

**The reason P3 precedes P4:** without preemption, a continuous-batching claim measured under open-loop load *above the knee* is untrue — the system OOMs or hard-rejects rather than degrading. Since ADR-012 commits to open-loop load and ADR-011 commits to publishing the knee, the system will be driven above capacity **by design**, every run. A system that falls over there invalidates the Phase 2 claim retroactively. That is property 4 (PHASE_PLAN §0, §6).

Related ordering choices, same logic:
- **P1 keeps batch at 1 all phase.** The claim is memory, not throughput. Keeping batching out means a paged-attention bug cannot hide behind a batching bug (PHASE_PLAN §4).
- **P0 earns no published claim and is kept separate anyway**, so its DoD is a hard gate rather than something quietly descoped when P1 runs long (PHASE_PLAN §3).

**Alternatives considered**

- **Radix cache before preemption.** Rejected. It is the flashier bullet and the more familiar feature, so this is the ordering the project would drift into under no discipline. It fails property 4: shipping a prefix cache on top of a batching system that collapses at KV exhaustion means the *earlier* throughput claim is dishonest under exactly the load conditions the methodology requires measuring. It also worsens the failure — a radix cache increases the number of resident blocks and their sharing structure, so KV exhaustion arrives with more state to get wrong.
- **Preemption folded into Phase 2.** Rejected on risk concentration. Phase 2 is already the highest-risk phase, with three independent hard things (batched forward, scheduler, honest load harness), each capable of silently producing a plausible wrong number (PHASE_PLAN §5). Adding preemption gives four, with no clean bisect between them.
- **Routing (P5) before the radix cache (P4).** Explicitly left open rather than decided (PHASE_PLAN §12). P5 is the stronger differentiator and the acknowledged skill gap; P4 is a hard dependency for *prefix-aware* routing but **not** for routing itself — a load-aware router with health/drain/failover is buildable without a radix cache. If the fall is compressed, "routing without prefix awareness" may beat "prefix cache without routing." Decided with a fall-length estimate in hand, not now.
- **Chunked prefill earlier than P4.** Considered and left where it is, with a caveat: ARCHITECTURE §5.1 makes chunked prefill load-bearing for the *concurrency model* (it bounds step duration, which is what keeps ADR-007's cooperative scheduler responsive). **[inference]** If P2 shows long prefills starving the event loop, a minimal token cap may need to be pulled forward — the full chunked-prefill implementation with cache interaction stays in P4.

**Consequences**

- At freeze: paged KV + allocator, batched varlen forward, continuous batching, preemption, an open-loop harness with goodput-under-SLO, committed artifacts, and a published known-gaps list — **three serving bullets, all measured, all defensible** (PHASE_PLAN §6).
- **Not** at freeze: the radix prefix cache and prefix-aware routing. PRD §C7's serving claim set names both and **must be recut** to claim only what exists (PHASE_PLAN §11): *"Serving layer — paged block KV cache, continuous batching, and preemption under memory pressure, benchmarked open-loop on goodput under SLO."*
- Cut order is pre-planned, so cuts are informed rather than panicked: if behind at week 3, cut swap (ADR-006) and FlashInfer (ADR-001) — ~17h together — **before** cutting anything from Phase 2. Phase 2 is where the throughput claim lives and it has no safe cuts (PHASE_PLAN §6).
- The recompute/radix interaction (ARCHITECTURE §9.2) can only be *measured* after P4, so the preemption analysis published at freeze is missing that term and must say so.
- Makes harder: demoing the project before P4, since prefix caching is the feature a non-specialist recognizes.

**Revisit if:** weekly engineer-hours differ materially from the ~30 assumed — the freeze line moves a phase in either direction and should be recomputed from the real number (PHASE_PLAN §0). Also revisit the P4/P5 order once fall length is known.

---

## ADR-017 — Least-outstanding-requests is the real routing baseline; round-robin is table stakes

**Status:** Accepted
**Date:** 2026-07-31

**Context**

G5 requires prefix-aware routing to beat a baseline "on a workload where it should, and lose where it should" (PRD §2). The choice of baseline determines what a win *means*.

Two candidate baselines, both defined precisely enough to be reimplemented by a skeptic (METHODOLOGY §6):

- **B4 — round-robin.** *Requests are assigned to replicas in strict rotation, independent of replica load, queue depth, or cache state; no re-routing after assignment.*
- **B5 — least-outstanding-requests.** *Assign to the replica with the fewest in-flight requests; ties broken by rotation.*

**Decision**

Both are implemented, and **B5 is the baseline the claim is made against.** B4 is published but labeled as what it is: *a genuinely weak baseline — beating round-robin is table stakes, not a result* (METHODOLOGY §6).

The reasoning is the load-bearing part. B5 is **load-aware but cache-blind**, so the delta between B5 and prefix-aware routing isolates **exactly the value of cache awareness** — which is the actual claim. Against B4, a prefix-aware router wins partly because it happens to balance load, and the two effects are not separable.

The corollary is pre-committed: **a prefix-aware router that beats B4 but not B5 has demonstrated load balancing, not prefix awareness, and that result is published as such if it occurs** (METHODOLOGY §6). Phase 5's DoD requires beating B5, not B4 (PHASE_PLAN §8).

**Alternatives considered**

- **Round-robin only.** Rejected. It is the baseline that makes any router look good, and choosing it is indistinguishable from choosing a baseline to win against. Any reader who has built a load balancer will ask about B5 immediately.
- **Random assignment.** Rejected as weaker than B4 and less standard. Still occasionally useful as a sanity floor but not published as a baseline.
- **Consistent hashing on prefix.** Deferred to Tier 3 (PRD §5). It is a *different prefix-aware policy*, not a baseline — comparing against it would be comparing two implementations of the same idea, which is a later and finer question than "is cache awareness worth anything."
- **Least-loaded by queue depth or estimated work rather than in-flight count.** Considered. In-flight count is chosen because it is unambiguous, needs no server-side estimation, and is the standard formulation — which matters, since the baseline's value comes from being one a skeptic recognizes without argument.

**Consequences**

- Both B4 and B5 are implemented (PHASE_PLAN §8, `[C]`), and the results table has three routing policies at every point.
- The claim is harder to win. That is the intent.
- The **losing cases are mandatory and pre-registered** (METHODOLOGY §10), which is what makes any win credible. Predicted in writing before measurement, and at minimum cases 1, 2, 3, and 7 are measured and published whether or not they flatter:
  1. *Zero/near-zero sharing* — nothing to be cache-aware about; expected to tie B5, lose slightly if the policy sacrifices any load balance.
  2. *Uniformly shared prefixes* — when every request shares the same prefix, every replica caches it after warmup and prefix awareness degenerates to "pick any replica." **This is important because it is the naive mental model of prefix caching**, and it is where the fancy router is worthless.
  3. *High load near or above capacity* — affinity and balance conflict directly: the replica holding the right prefix is often busy *because* it holds the popular prefix. Expected to win at low-to-moderate load and lose above the knee unless the policy explicitly blends affinity with load. **Publishing the crossover point is a better result than publishing a win.**
  7. *Highly skewed prefix popularity* — a hot prefix creates a self-inflicted hotspot. **[inference]** The most likely place for a genuinely bad result, and therefore the most valuable to measure.
  (Cases 4–6 — short prompts, high cache turnover, few replicas — are predicted and measured where budget allows.)
- The affinity/load blending function is a real open design item, not a detail: METHODOLOGY §10 predicts prefix-aware routing *loses* above the knee without it, and the blend depends on measuring where the knee is (ARCHITECTURE §11).
- Makes harder: producing a clean single-number routing win. There may not be one, and the crossover is the honest deliverable.

**Revisit if:** never for the baseline choice. The blending function is expected to change based on measurement.

---

## ADR-018 — Benchmark artifacts are committed to git, and every published number resolves to a file

**Status:** Accepted
**Date:** 2026-07-31

**Context**

The engine's `BENCHMARKS.md` sets the standard: declares batch-1 scope (`:19`), publishes the row where its kernel *loses* (`:94`), names its own unmeasured splits (`:151`), maintains a known-gaps list (`:243-250`), and commits its result artifacts (24 tracked files under `bench/results/`, decision recorded at `.gitignore:29`).

> **Correction, 2026-07-31.** This ADR originally cited `CLAIMS_AUDIT.md:6` (*"`git ls-files` returns zero result files"*) as a failure to correct. That audit predates commit `c24afaf` and the claim is **false**. The decision below stands, but its justification is "inherit and extend a good practice," not "fix a failure."

The residual gap is narrower and real: the engine's artifacts store per-run p50/p99 and discard raw samples (`bench/harness.py:136-137,178`), so percentiles cannot be correctly pooled across requests after the fact.

That is the failure this project is explicitly built not to repeat (PRD §G6, S8; METHODOLOGY §8, §11).

**Decision**

`results/` is git-tracked and **explicitly not gitignored** — called out in Phase 0's scaffold item (PHASE_PLAN §3).

Every published number resolves to a committed file containing: the **raw samples** (not just percentiles), the config, the seed, the workload parameters, the git SHA of this repo, the pinned engine SHA/tag, and the Slurm allocation identity (METHODOLOGY §11).

The supporting rules, adopted as a set (METHODOLOGY §11):
1. **Publish the losing row** — the standard the engine set at `BENCHMARKS.md:94`, where it kept a row showing its kernel 39% slower than SDPA because *"omitting it would make the kernel look uniformly competitive when it is not."*
2. **Name what was not measured** — unmeasured splits are stated, not implied away (`BENCHMARKS.md:151`).
3. **Maintain a known-gaps section**, written by the author before a reader finds the gap.
4. **Commit the artifacts.**
5. **Superseded numbers stay visible** — both versions and the reason, as the engine does for its perplexity figure, superseding +0.14 with +0.044 and explaining that the earlier eval text was not WikiText (`BENCHMARKS.md:191-198`). *Silently replacing a number is indistinguishable from hiding one.*
6. **A claim whose artifact cannot be regenerated is deleted, not softened.**

And the resume rule that depends on all of it: placeholders `[N]` are filled from committed artifacts only; **every bullet must resolve to a file in `results/`** (PHASE_PLAN §11).

**Alternatives considered**

- **Report numbers in prose only.** Rejected — a number with no regenerable artifact behind it cannot be defended under questioning, and the first follow-up ("how did you measure that?") has no answer. The engine already rejected this too.
- **Commit only summary statistics, as the engine does.** Rejected — per-run percentiles cannot be pooled across requests after the fact (`bench/harness.py:136-137,178`), and pooled percentiles are exactly what a serving benchmark needs.
- **Store artifacts outside git** (object storage, a results server, Grafana screenshots). Rejected. It breaks the property that matters: a reader can `git log` a number back to the commit and config that produced it. A screenshot proves nothing and cannot be re-analyzed.
- **Commit percentiles only, not raw samples.** Rejected. It is what the engine does (`bench/harness.py:136-137,178`) and it makes correct pooling impossible after the fact — percentiles must be pooled across all requests in the window, not averaged across per-run percentiles (METHODOLOGY §5). It also forecloses recomputing with a different percentile method, which matters because different tools disagree and the difference is visible in the tail.
- **Commit only the final published runs.** Rejected as an invitation to selection bias — the whole point of rule 5 (superseded numbers stay visible) is that the trail is the credibility.

**Consequences**

- Repo size grows with raw per-token samples across many configurations. Accepted cost; it is text and it compresses.
- A number cannot be quoted before its artifact exists, which constrains claim writing to what has actually been measured (PHASE_PLAN §11).
- Every artifact must carry Slurm allocation identity, because METHODOLOGY §5 forbids cross-allocation comparison (ADR-021) — which means the artifact schema is not optional metadata, it is what makes comparison legal.
- The engine's `_hw_metadata` (`bench/harness.py:69-99`) is reused and **extended** with Slurm job id, node, allocation id, and QOS (METHODOLOGY §9).
- One naming hazard is fixed rather than inherited: `bench/harness.py:44-47` reports host RSS under the column name `peak_mem_mb` while `bench/baseline_hf.py:108` reports `torch.cuda.max_memory_allocated()` under **the same name** — the engine's known gap #1 (`BENCHMARKS.md:247`). This project uses GPU allocator metrics for GPU memory and names the column unambiguously, with unit and source recorded in the schema (METHODOLOGY §9, §12).
- Makes harder: quietly correcting an embarrassing number. By design.

**Revisit if:** repo size becomes genuinely unmanageable, in which case raw samples move to git-lfs — not out of version control.

---

## ADR-019 — The custom CUDA kernel is not in the paged path, and this is said before being asked

**Status:** Accepted
**Date:** 2026-07-31

**Context**

The engine's hand-written decode kernel is its strongest single artifact: `kernels/attention_decode.cu`, 311 lines, three staged versions, warp-shuffle reductions, a flash combine rule, measured at ≈0.98–0.99× PyTorch SDPA (`BENCHMARKS.md:102`) and gated on 100 inputs (`tests/test_attention_kernel.py:72-85`).

It cannot serve the paged path. The pybind ABI takes no stride, block-table, or block-size arguments (`kernels/bindings.cpp:29-31`) — a paged cache is *inexpressible* across it — and `Q` is `[n_heads, head_dim]` with no batch axis (`kernels/attention_decode.cu:30-32`).

This creates an easily-misread situation: a project containing a custom CUDA kernel and a paged-attention serving layer, where the two do not touch.

**Decision**

Stated as a first-class commitment rather than a footnote (PRD §C3): **the custom CUDA kernel is not in the paged path.** It becomes the *single-replica, contiguous-cache reference path*. This is the honest cost of freezing the engine's `.cu` file (PRD §C2), which is what keeps the 0.98–0.99× SDPA claim intact and untouched by serving work.

Attribution rule, non-negotiable and repeated in the phase plan (PHASE_PLAN §11): **say this before being asked.**

**Alternatives considered**

- **Extend the existing kernel to accept block tables.** Rejected — see ADR-001. The ABI (`kernels/bindings.cpp:29-31`) has no way to describe a paged cache, and the batch-less `Q` shape means a paged *batched* kernel is a rewrite. Attempting it would also re-open the numerical gate at `tests/test_attention_kernel.py:72-85` and put `BENCHMARKS.md:102` at risk.
- **Retire the kernel from the project narrative** since it is not on the serving path. Rejected. It is the engine's claim, on the engine's claim set (PRD §C7), and it is a legitimate one. The two lines are separate by design (ADR-008).
- **Let the ambiguity stand** — mention both the kernel and paged attention without explicitly disconnecting them. Rejected as the worst option. A reader who works out the disconnect independently will assume it was concealed, which costs more than the disconnect itself. Volunteering it converts a liability into evidence of precision about one's own work.
- **Use the custom kernel for the contiguous B1/B3 baselines only.** Adopted — that is exactly what "single-replica, contiguous-cache reference path" means. It keeps the kernel exercised and gives the baselines the engine's best available performance, which makes the baselines *harder* to beat, which is the right direction.

**Consequences**

- The serving path's attention is FlashInfer or `PagedTorchBackend`, never the custom kernel (ADR-001).
- The contiguous baseline (B3) runs with the custom kernel, so the paged path is compared against the engine at its best, not a strawman.
- The whole-cache transpose (`components_gpu.py:153-154`, ~67 MB/token at kv_seq 2048, `BENCHMARKS.md:149`) stays on the kernel path, since that path is frozen (ADR-009) — and is absent from the paged path entirely (ADR-005).
- Makes harder: claiming end-to-end ownership of the serving stack down to the kernel. That claim is not made.

**Revisit if:** ADR-001's revisit trigger fires — i.e. a paged kernel becomes the top remaining item after the freeze.

---

## ADR-020 — A model-level GPU correctness oracle is a blocking prerequisite for all serving work

**Status:** Accepted
**Date:** 2026-07-31

**Context**

Every correctness claim in the engine is about the **CPU fp32** path: `tests/test_forward.py:113,137` (logits within 1e-3), `tests/test_decode.py:30-48` (32 greedy tokens bit-identical to HF). The GPU path has no output-level oracle at all — `tests/test_gpu_model.py:42-45` asserts only that outputs are finite, correctly shaped, and have an in-range argmax. GPU fp16 tokens are never compared to HuggingFace or to the CPU path.

Additionally, the one cache-correctness gate that does exist is not running: `tests/test_generate.py` uses the key `"input_ids"` where `tests/oracle.py:170` provides `"token_ids"` (`:42,52,63,78,89`), so all 5 tests raise `KeyError`. The KV-cache-vs-no-cache gate **is not enforced today**.

This project then stacks a paged cache, a batched forward pass, a scheduler, preemption, a radix cache, and copy-on-write on top of that GPU path.

**Decision**

The GPU oracle is built in Phase 0 and is a hard gate: greedy tokens from `LlamaModelGPU` must match `tests/oracle.py`'s `greedy_ids` for both fixture prompts (PHASE_PLAN §3 DoD). It is Tier 0 in the PRD (§5) and a prerequisite for every benchmark in the methodology (METHODOLOGY §12).

**Token equality, not logit distance**, is the comparison — GPU fp16 against CPU fp32, where GPU component tests already need `atol=1e-2` (`tests/test_components_gpu.py:83`). Token equality is the tolerance that actually means something across that precision gap.

The `test_generate.py` key fix ships in the same phase, restoring the cache-correctness gate.

**Alternatives considered**

- **Skip it; rely on the engine's existing CPU tests.** Rejected, and this is the item that *looks* skippable and is not (PHASE_PLAN §3). The CPU tests say nothing about the fp16 GPU path that every serving claim runs on. Building a batched, paged, preemptible scheduler on a forward pass with no output-level oracle means **any divergence is unattributable** — allocator, batching, paged attention, or a pre-existing GPU-path bug, with no way to bisect (ARCHITECTURE §2.7, METHODOLOGY §12).
- **Build it later, when something breaks.** Rejected. By then there are four candidate causes and no bisect point; the oracle's value is precisely that it exists *before* the first layer is added. It is also cheap to build, which removes the usual argument for deferral.
- **Compare logits with a tolerance instead of tokens.** Rejected. Choosing a tolerance across fp16-GPU vs fp32-CPU is guesswork, and the failure mode is a tolerance quietly widened until the test passes. Token equality is binary and un-negotiable-with.
- **Compare against HuggingFace directly rather than the CPU path.** Both are available (`tests/oracle.py`). The CPU path is the tighter gate because it isolates *this* implementation's GPU-vs-CPU divergence rather than folding in framework differences; HF remains the outer check the CPU path already passes (`tests/test_decode.py:30-48`).

**Consequences**

- Phase 0 grows and still earns no published claim (PHASE_PLAN §3). Accepted — it is a gate, not a bullet.
- The correctness gate runs **every phase, not once** (PHASE_PLAN §2), because every phase adds a new way to silently corrupt output. The specific gates that descend from this one: bit-identical greedy through `PagedTorchBackend` at block-straddling lengths (P1), batch-invariance — same prompt alone vs in a mixed batch (P2), bit-identical under forced preemption (P3), bit-identical with cache on vs off plus partial-hit correctness at block boundaries (P4).
- Every benchmark run carries an output-equality check, so a *faster and wrong* result cannot be published (METHODOLOGY §12).
- The oracle depends on weights being available, which interacts with PRD §O5 (HF-gated weights) and constrains CI design — CI runs the CPU-only subset (PHASE_PLAN §3).
- Makes harder: nothing. This is the cheapest high-value item in the plan.

**Revisit if:** never.

---

## ADR-021 — Replicas are whole physical GPUs on a single node; benchmarks follow a two-QOS discipline and never compare across allocations

**Status:** Accepted
**Date:** 2026-07-31

**Context**

PRD §O3 originally asked whether "N replicas" could honestly mean multiple model instances on one 40 GB A100 (Llama 3.2 1B is ~2.36 GB fp16, `BENCHMARKS.md:161`), or MIG partitions, or genuine multi-GPU. That framing problem is now moot.

PRD §8/O1, **resolved 2026-07-31 with measurements from the cluster**: the runbook's "one A100, 4 hours" (`docs/PACE_RUNBOOK.md:24-32`) is not a policy ceiling — it is an artifact of the `interactive-cpu2` partition, whose `interactive-gpu` QOS caps at `gres/gpu=2, MaxJobsPU=1`. Under the `gpu-*` partitions with the `inferno` QOS: max 32 concurrent GPUs per user, 3-day wall time, and single nodes with 8 GPUs on `gpu-h100` (4 nodes), `gpu-h200` (12 nodes), and `gpu-l40s` (10 nodes). Relative GPU-hour cost from `TRESBillingWeights`: A100 = 10261, H100/H200 = 24940 (2.43× A100), L40S = 8030 (0.78× A100). Also available: `embers` QOS — free, preemptible, `MaxWall=08:00:00`.

Separately, the engine documents that absolute numbers are not portable across sessions: its own throughput moved from ~79 to ~60 tok/s across nodes for **identical code**, due to contention and clocks (`BENCHMARKS.md:17`, `docs/PACE_RUNBOOK.md:161`).

**Decision**

1. **Replicas are whole, physically separate GPUs inside one allocation.** No MIG, no co-tenancy, no asterisk on the phrase "N replicas" (ARCHITECTURE §1, METHODOLOGY §13).
2. **Hardware by purpose** (PHASE_PLAN §2): `gpu-l40s` (8/node, 0.78× A100 SU) for scheduling and routing results, because those results are about *scheduling behavior*, not peak FLOPs. `gpu-h100`/`gpu-h200` (2.43×) only where absolute throughput is the claim. A100 for continuity with the engine's existing numbers.
3. **Two-QOS discipline:** `embers` (free, preemptible, 8h) for development, iteration, and correctness runs; `inferno` (charged) for published benchmarks. **No published number comes from an `embers` run** — preemption mid-benchmark would silently truncate a measurement window (METHODOLOGY §13, PHASE_PLAN §2).
4. **Cross-session comparisons are forbidden by default.** Every A/B runs back-to-back in a single Slurm allocation, and the allocation/node/QOS identity is recorded in the artifact. A comparison that cannot be run back-to-back is reported as two separate absolute measurements, **not as a delta** (METHODOLOGY §5).

**Alternatives considered**

- **Multiple replicas per GPU (co-tenancy).** Rejected now that whole GPUs are available. It was the pre-O1 fallback and it carried a permanent benchmark-phrasing problem: "N replicas" would have needed an asterisk in every result, and a skeptical reader would rightly discount the routing claim.
- **MIG partitioning.** Rejected for the same reason, plus its availability is unverified (PRD §8 leaves `nvidia-smi -L` as a first-on-node check). Whole GPUs make the question irrelevant.
- **Multi-node replicas.** Rejected. It would add inter-node networking to the request path for no gain — an 8-GPU node supplies enough replicas — and it would import a class of failure this project has not budgeted for.
- **H100/H200 for everything.** Rejected on cost: 2.43× the A100 SU rate for results whose claim is scheduling behavior, not FLOPs. **[inference]** L40S is the better instrument for routing work precisely because the interesting variable is the scheduler, not the arithmetic.
- **Develop on `inferno` for convenience.** Rejected — it burns the SU balance on iteration. The balance reads 999.93 with 0.00 reserved, but the absolute burn rate is a **two-sample inference** from two ~7.6-minute A100 jobs and explicitly should not be budgeted against; the *ratios* from `TRESBillingWeights` are read directly and are reliable. Calibration with one short instrumented job (record `pace-quota` before/after) precedes any multi-GPU budget (PRD §8/O1 item 6, PHASE_PLAN §2).
- **Allow cross-session A/B with a caveat.** Rejected. The engine's own ~25% swing for identical code (`BENCHMARKS.md:17`) shows the effect is larger than most deltas this project will claim. METHODOLOGY §12 lists cross-session A/B as a *silent* threat, and the detection mechanism is mechanical: allocation id in every artifact, and comparisons refuse to render across allocations.

**Consequences**

- **G5 is unblocked** and multi-replica routing needs no inter-node networking (PRD §8/O1 item 1).
- The binding constraint on the project is the **freeze date, not hardware** — which reverses the assumption the PRD was drafted under (PRD §8/O1 item 2).
- Every A/B must be planned as a single allocation, which constrains run scheduling and means a failed run costs the whole allocation's worth of comparisons.
- Queue wait is real and excluded from author-hour estimates (PHASE_PLAN §0); `gpu-h200` and `gpu-l40s` both showed allocated/drained nodes, so contention exists even though capacity does.
- `embers` preemption must be assumed during development — long dev runs need checkpointing or splitting (PRD §8/O1 item 3).
- CUDA arch parameterization is **promoted from nice-to-have to prerequisite**: `scripts/build_kernels.sh:10` hardcodes `-DCMAKE_CUDA_ARCHITECTURES=80`; H100/H200 are sm_90 and L40S is sm_89 (PRD §8/O1 item 5, ARCHITECTURE §2.7 item 7).
- Makes harder: any longitudinal claim across the project's lifetime. Numbers from July and September are not comparable unless re-run.

**Revisit if:** cluster policy changes, or a result genuinely requires more than 8 co-located GPUs.

---

## ADR-022 — The router is a single point of failure, and that is accepted and published

**Status:** Accepted (revisit if HA becomes a claim)
**Date:** 2026-07-31

**Context**

One router process fronts N replicas (ARCHITECTURE §1). The failure-domain table is explicit: router crash → blast radius *everything* (ARCHITECTURE §8).

Making it highly available requires one of two things. Either a shared consistent view of replica state — a coordination service, and the entire point of ADR-002 is not having one — or multiple independent routers with divergent hint tables.

**Decision**

Accepted as a known limitation and **stated rather than hidden** (ARCHITECTURE §8.1). The prepared answer is specific: *"it's a SPOF, here's exactly what would be required to fix it, here's why the hint-only design makes that fix cheap, and here's why I didn't."*

**Alternatives considered**

- **Multiple independent routers with divergent hint tables.** Rejected for scope, **not on merit** — and that distinction is the interesting part. It is *actually acceptable* under this architecture: hints are advisory (ADR-002), so two routers disagreeing costs cache locality, not correctness. Each would maintain its own hint table and its own health view, and neither would need to know about the other. The reason it is out of scope is that it adds no learning this project needs and does not fit before the freeze — not that it would not work. **The fact that the fix is cheap is itself a consequence of ADR-002, and that is the point worth making.**
- **A coordination service (etcd, Consul, leader election).** Rejected. It reintroduces exactly the consensus dependency ADR-002 exists to avoid, in exchange for availability of a component whose state is disposable by design.
- **Client-side load balancing — no router at all.** Rejected. It moves prefix-aware routing into every client, which is where it cannot be maintained or measured, and it deletes the router that carries the distributed-systems claim.
- **Router as a thin stateless proxy behind an external LB.** This is effectively the multi-router option with the hint tables kept local; same rejection, same reason.

**Consequences**

- Router crash means total outage. Recorded in the failure-domain table with that blast radius (ARCHITECTURE §8).
- Fault-injection testing (S7) targets *replica* failure, not router failure — the router failure case has no interesting handling to test.
- The explainability gate requires being able to explain why the router is a SPOF, exactly what would fix it, and why the hint-only design makes the fix cheap (PHASE_PLAN §8).
- Makes harder: any availability claim about the system as a whole. None is made.

**Revisit if:** the project ever needs an availability claim, or if router-level HA becomes a cheap add-on after Phase 5 — the design already permits it.

---

## ADR-023 — `forward_varlen` returns a device tensor; server-side timing therefore uses CUDA events

**Status:** Accepted
**Date:** 2026-07-31

**Context**

The engine's `prefill`/`decode_step` return CPU numpy (`engine/model_gpu.py:90,158`) — a deliberate engine choice so the numpy sampler works unchanged (`docs/BUILD_LOG.md:823`), costing ~0.1 ms against a ~12 ms step.

That copy has a second, undocumented-until-now role: it forces a CUDA sync every step, which is **the only reason the engine's host-clock timings are valid** (`BENCHMARKS.md:60`).

**Decision**

`forward_varlen` returns a **device tensor** `(n_seqs, vocab)` fp16 (ARCHITECTURE §2.6). At batch 32 the device→host copy is on the critical path for every request in the batch, and sampling should happen on-GPU.

The consequence is accepted together with the decision: **server-side GPU timings use CUDA events or an explicit sync at a declared point** (METHODOLOGY §5). Client-side timings remain host-clock, because they measure real network-observable behavior, which is what they should measure.

`bench/bench_attn_kernel.py:32-44` `_time_cuda` — CUDA-event timing generic over a callable — is reused as the pattern (METHODOLOGY §9).

**Alternatives considered**

- **Keep returning host numpy, as the engine does.** Rejected. It preserves timing validity for free and costs a synchronous host round-trip per step, scaled by batch size, on the critical path of every request in the batch. It would also force batch-32 numpy sampling round-trips per step, which is measurable (ARCHITECTURE §11 leaves GPU-vs-numpy sampling open precisely so this stays decidable later — returning a device tensor is what keeps it open).
- **Return a device tensor but keep an unconditional sync for timing.** Rejected. It reintroduces the cost the change exists to remove, and it makes the production path pay for the benchmark path.
- **Return a device tensor and keep using host-clock timing.** Rejected, and this is the dangerous option: host-side `perf_counter` would silently become a measurement of **kernel-launch queueing, not execution** (ARCHITECTURE §2.6). Nothing errors; the numbers just get plausibly better. This is in the risk register as a *silent invalidator* and in METHODOLOGY §12's threat table with its detection mechanism: CUDA events server-side, plus asserting a sync point exists on any timed path.

**Consequences**

- Sampling can move to GPU; the choice between GPU sampling and the engine's numpy sampler (`engine/sampler.py`) stays open by construction (ARCHITECTURE §11).
- All server-side timing instrumentation must be CUDA-event based from the start. Retrofitting it after publishing a number is how the number becomes wrong.
- The engine's reference path keeps its host copy and its valid host-clock timings — unchanged, per ADR-009.
- **[inference]** Batch-level ITL measured server-side without events would be the most likely place this bites, since per-token host timing is the natural thing to instrument first.
- Makes harder: casual timing. Any `perf_counter` around GPU work on the serving path is suspect by default.

**Revisit if:** profiling shows the device tensor return is not on the critical path at realistic batch sizes — unlikely, and it would not restore host-clock validity anyway.

---

## ADR-024 — LIFO victim selection for preemption, with a starvation guard

**Status:** Accepted (revisit if the fairness result demands it)
**Date:** 2026-07-31

**Context**

ADR-006 establishes *how* to preempt (recompute or swap). This is *whom* to preempt. When the pre-step block check fails, the scheduler must pick a victim from the running batch (ARCHITECTURE §5.2, §9.2).

The workloads deliberately include heavy-tailed output lengths, because *"a single very long generation holding blocks while short requests queue behind it is the exact scenario that motivates preemption and fairness"* (METHODOLOGY §4). So victim selection is exercised under exactly the conditions that make it matter.

**Decision**

**Last-arrived-first (LIFO).** Preempt the most recently admitted sequence. Rationale: it preserves the progress of older requests and bounds worst-case latency for requests already deep into generation.

**Starvation guard:** a sequence preempted K times becomes ineligible for preemption, forcing forward progress. Demonstrated in Phase 3's DoD — a sequence preempted K times must complete (PHASE_PLAN §6).

On recompute, the victim returns to the **front** of the waiting queue with its prompt plus tokens-generated-so-far as its new prompt (ARCHITECTURE §9.2).

**Alternatives considered**

- **FIFO (preempt the oldest).** Rejected. It repeatedly punishes the oldest request, which is both unfair and **unbounded** — the same request can be re-victimized indefinitely while newer, shorter requests complete around it. Under the heavy-tailed workloads this project runs, that is not a corner case (ARCHITECTURE §5.2).
- **Preempt the longest sequence** (frees the most blocks per eviction). Considered, and it has a real argument: fewest evictions to clear the deficit. Rejected because it is maximally expensive under recompute — recompute cost is O(current length), so this picks precisely the sequence most expensive to restore. It also systematically penalizes long generations, which is the population most likely to be the *valuable* request.
- **Preempt the shortest sequence** (cheapest to recompute). Rejected as roughly LIFO with extra bookkeeping — newest sequences are usually shortest — while adding a scan and losing LIFO's clean fairness story.
- **Priority-based selection.** Deferred to Tier 3 (PRD §5, priority and fairness scheduling). It presupposes a priority signal the API does not currently carry, and it is orthogonal — it changes the ordering key, not the mechanism.

**Consequences**

- Newly admitted requests bear most of the preemption cost, which is the right place for it: they have the least sunk work and their clients have waited least.
- The starvation guard introduces a tunable K and a state bit per sequence, plus a degenerate case that must be handled: if *every* running sequence is ineligible, the scheduler cannot preempt at all. **[inference]** That case needs a defined behavior — most plausibly refusing admission and stalling rather than admitting into a state that cannot be relieved — and it is not yet specified anywhere in the docs. Flagged here as an open implementation detail for Phase 3.
- LIFO plus recompute interacts favorably with the radix cache: the victim's own prefix blocks may still be resident, so re-prefill often covers only the generated tail (ARCHITECTURE §9.2, step 5) — and a *newly admitted* victim has a short tail by definition.
- Client-visible effect is a stall, not an error, and output must be bit-identical to an unpreempted run (ADR-006).
- Makes harder: any fairness claim beyond "no request starves." Latency variance for recently admitted requests is deliberately higher.

**Revisit if:** the preemption-rate measurements show LIFO producing pathological repeat-victimization under some workload the starvation guard handles only crudely, or if priority scheduling (Tier 3) lands and supplies a real ordering key.
