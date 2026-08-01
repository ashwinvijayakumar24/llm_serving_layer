# Architecture

**Status:** draft 1, 2026-07-31. Planning only; no implementation.
**Depends on:** `docs/PRD.md`, `docs/BENCHMARK_METHODOLOGY.md`, engine @ `6ff40a1`.

Engine claims cite `file:line` in `../llm_inference_engine`. **[inference]** marks judgment rather than something read from code. Decisions marked **→ ADR** get a full alternatives write-up in the decision log.

---

## 1. The shape of the system

Two process types. That is the whole topology.

```
                    ┌──────────────────────────────────────────┐
   clients ────────▶│  ROUTER  (1 process, CPU only)           │
   (OpenAI API)     │  · prefix-aware replica selection        │
                    │  · health checks, draining, failover     │
                    │  · backpressure, load shedding           │
                    │  · approximate prefix→replica hint table │
                    └───────┬──────────┬──────────┬────────────┘
                            │          │          │
              ┌─────────────▼──┐  ┌────▼───────┐  ┌▼─────────────┐
              │ REPLICA 0      │  │ REPLICA 1  │  │ REPLICA N-1  │
              │ (1 GPU)        │  │ (1 GPU)    │  │ (1 GPU)      │
              └────────────────┘  └────────────┘  └──────────────┘
```

Each replica owns exactly one GPU and one model instance. Replicas are independent and share nothing — no shared memory, no shared cache, no cross-replica coordination on the request path. Per PRD §8/O1, replicas are whole physical GPUs on a single 8-GPU node (`gpu-l40s` preferred), so "N replicas" carries no asterisk.

**The organizing principle of this design: the router holds hints, the replica holds truth.** Every piece of router state is an approximation that may be stale. If it is wrong, a request lands on a suboptimal replica and runs slightly slower. It never produces a wrong answer, never corrupts cache state, and never requires a distributed transaction. This is the property that makes the routing layer safe to build, and it is the first thing to say when asked "how do you keep the router's view of the cache consistent?" — **you don't, and that's the design.**

### Inside a replica

```
   HTTP ingress (FastAPI, async)
        │  Request → per-request asyncio.Queue for output tokens
        ▼
   ┌───────────────────────────────────────────────────────────┐
   │ ADMISSION CONTROLLER   queue depth, memory headroom,      │
   │                        timeout budget, load shedding      │
   ├───────────────────────────────────────────────────────────┤
   │ SCHEDULER  (iteration-level / continuous batching)        │
   │   waiting queue · running batch · preemption policy       │
   ├──────────────────────┬────────────────────────────────────┤
   │ RADIX PREFIX CACHE   │  BLOCK ALLOCATOR                   │
   │  token-trie, refcnt  │   free list, block tables,         │
   │  LRU eviction        │   copy-on-write                    │
   ├──────────────────────┴────────────────────────────────────┤
   │ MODEL RUNNER   builds BatchMeta, calls engine forward     │
   ├───────────────────────────────────────────────────────────┤
   │ ATTENTION BACKEND   PagedTorchBackend | FlashInferBackend │
   ├───────────────────────────────────────────────────────────┤
   │ ENGINE (pinned dependency)  LlamaModelGPU, sampler        │
   └───────────────────────────────────────────────────────────┘
        │
        ▼  GPU: model weights (~2.36 GB fp16) + KV block pool (the rest)
```

Ownership per PRD §G7 — **Author** writes: admission controller, scheduler, preemption policy, block allocator, radix cache, `PagedTorchBackend`, routing policy, model runner's batch assembly. **Assistant** writes: HTTP ingress, SSE framing, metrics wiring, health-check plumbing, FlashInfer integration glue, benchmark harness scaffold, CI, Docker.

---

## 2. The engine ↔ serving interface

This is the load-bearing specification. Everything else can be refactored; this boundary cannot, because it spans two repos and two independent claim sets.

### 2.1 Design constraint

The engine's existing claims must survive untouched. `BENCHMARKS.md:102` ("v3 ≈ 0.98–0.99× PyTorch SDPA"), `tests/test_forward.py:113,137` (logits < 1e-3), `tests/test_decode.py:30-48` (32 greedy tokens bit-identical to HF) are all statements about the **contiguous, batch-1 path**. Therefore:

> **Every engine change is additive. No existing code path is modified in a way that changes its numerics.**

`prefill()` and `decode_step()` (`engine/model_gpu.py:58,127`) keep their exact current behavior and remain the engine's reference path. The batched path is a new sibling method. The existing test suite becomes the regression gate for free.

### 2.2 The varlen batch layout

All sequences are packed along one token axis. This is the choice that makes batching cheap — see §2.5 for why.

```
Batch of 3 sequences, decoding: seq A (1 new token), B (1), C (1)
tokens axis:   [ a0 | b0 | c0 ]                      shape (3, hidden)

Batch mixing a chunked prefill (D, 4 tokens) with 2 decodes:
tokens axis:   [ d0 d1 d2 d3 | a0 | b0 ]             shape (6, hidden)
```

### 2.3 `BatchMeta` — the contract

```python
@dataclass(frozen=True)
class BatchMeta:
    """Describes one forward pass over a ragged batch of sequences.

    Token layout is flattened varlen: all sequences' new tokens are packed
    along axis 0 in sequence order. `tokens` == sum(query_lens).
    """
    query_lens:    Tensor  # (n_seqs,)      int32  new tokens this step, per sequence
    cu_query_lens: Tensor  # (n_seqs + 1,)  int32  exclusive prefix sum of query_lens
    kv_lens:       Tensor  # (n_seqs,)      int32  total KV length AFTER this append
    positions:     Tensor  # (tokens,)      int64  RoPE position of each token
    # Paged KV addressing — CSR form, NOT a padded matrix. See §2.3.1.
    kv_indptr:     Tensor  # (n_seqs + 1,)  int32  offsets into kv_indices, per sequence
    kv_indices:    Tensor  # (total_pages,) int32  flat physical page ids, concatenated
    kv_last_page_len: Tensor # (n_seqs,)    int32  occupancy of each seq's last page, 1..page_size
    # Write addressing
    batch_indices: Tensor  # (tokens,)      int32  which sequence each new token belongs to
    slot_mapping:  Tensor | None  # (tokens,) int32  flat physical KV slot per new token
                           #                       (PagedTorchBackend path; see §2.3.1)
    last_token_ix: Tensor  # (n_seqs,)      int32  index into tokens axis of each seq's
                           #                       last token — where logits are needed
    page_size:     int = 16
    is_prefill:    bool = False
```

*This sketch is kept in sync with the authoritative definition in
`engine/attention_backend.py`; on any disagreement the code wins.*

Two fields carry the whole paged design:

- **`slot_mapping`** is the write path. For each new token, the flat index into the physical KV pool where its K and V go. The backend does one scatter; it never needs to know about blocks when writing. `slot = block_tables[seq][pos // block_size] * block_size + (pos % block_size)`, computed on the CPU during batch assembly.
- **`block_tables`** is the read path. Per-sequence logical→physical block map. The attention kernel walks it to find KV.

`positions` being an explicit per-token tensor is what fixes the chunked-prefill bug: `engine/model_gpu.py:65` currently computes `torch.arange(seq)` starting at 0 regardless of cache position, so a second prefill chunk would receive positions `0..n` instead of `pos..pos+n`. Invisible today because prefill is called exactly once (`engine/scheduler.py:33`). **The fix is not "add an offset" — it is "stop computing positions inside the model."** The caller knows them; the model should not guess.

`last_token_ix` replaces `x[-1:]` (`engine/model_gpu.py:88`). With a ragged batch, "the last token" is a gather, not a slice.

#### 2.3.1 FlashInfer verification — resolved 2026-07-31

`§11` listed FlashInfer's API and layout contract as unverified, and R9 named a layout misunderstanding as a HIGH silent risk. Verified against `flashinfer-python==0.6.16` source on PACE. **The earlier draft of `BatchMeta` was wrong in one respect and right in another.**

| Design element | Verdict |
|---|---|
| Block layout `[num_pages, page_size, n_kv_heads, head_dim]`, separate K and V | **Correct.** `append_paged_kv_cache` accepts `paged_kv_cache: Union[Tensor, Tuple[Tensor, Tensor]]` (`flashinfer/page.py:408`) — a combined `[pages, 2, page_size, heads, dim]` tensor *or* a (K, V) tuple. The tuple form matches the design. |
| `page_size = 16` | **Correct**, matches FlashInfer's documented example. |
| `NHD` layout = `[kv_len, n_kv_heads, head_dim]` | **Correct**, and it is FlashInfer's default (`flashinfer/decode.py:403,422-430`). `HND` is also supported. |
| `block_tables: (n_seqs, max_blocks)`, `-1` padded | **WRONG.** FlashInfer uses a **CSR triple**: `kv_indptr` (per-sequence offsets), `kv_indices` (flat concatenated page ids), `kv_last_page_len` (`decode.py` plan example, `page.py:409-411`). A padded matrix is vLLM's convention, not FlashInfer's. |
| `slot_mapping` as the sole write addressing | **Incomplete.** `append_paged_kv_cache` takes `batch_indices` + `positions` (`page.py:406-407`), not a flat slot index. `slot_mapping` is still the natural form for `PagedTorchBackend`'s own scatter, so `BatchMeta` carries both; each backend uses what it needs. |

**This is the design working as intended.** The protocol sits above FlashInfer, so a contract mismatch cost a metadata-field change and an adapter — not a redesign of the allocator, the radix cache, or the scheduler. The allocator's internal representation stays a per-sequence `list[int]` of page ids; CSR is produced at batch-assembly time by concatenation plus a prefix sum, which is cheap and is the only place that has to know FlashInfer exists.

**Note the `kv_last_page_len` invariant:** FlashInfer documents `1 <= kv_last_page_len <= page_size`, so a sequence whose length is an exact multiple of `page_size` reports `page_size`, **not** `0`. That off-by-one is precisely the class of bug R8's block-straddle test exists to catch, and it must be asserted at batch assembly.

✅ **Resolved 2026-08-01 by reading the 0.6.16 kernels** (not the docstrings, which do not say):

- **`run()` returns `q.shape[:-1] + (head_dim,)` in `q.dtype`** — i.e. `(tokens, n_heads, head_dim)`, matching the protocol.
- **`causal=True` is BOTTOM-RIGHT aligned.** `prefill.cuh:1461` masks iff `kv_idx + qo_len > kv_len + q_idx`, i.e. keeps `kv_idx <= kv_len - qo_len + q_idx`; corroborated at `scheduler.cuh:954` (`kv_len_init = kv_len - qo_len; // right aligned`). That is exactly the protocol's `[0, kv_len - q_len + j]` and exactly `PagedTorchBackend`'s `triu(diagonal=kv_len-q_len+1)`. SDPA's top-left `is_causal` would have been wrong — the same trap `PagedTorchBackend` avoided by not using SDPA.
- **`plan()` stores state; `run()` does not take it.** `plan()` writes the page tables onto the wrapper (`decode.py:1467-1470`, `prefill.py:2355-2365`) along with `sm_scale` and `causal`; `run(q, paged_kv_cache)` receives no CSR at all. **A missing `plan()` therefore attends over the PREVIOUS step's page table — silently.** Rule: plan once per forward pass, run once per layer. Planning per layer merely wastes a host CSR copy; skipping a plan corrupts output with no error.

⚠️ **The differential is token-exact, not bit-exact, and that is a deliberate documented limit.** `PagedTorchBackend` casts `probs` to fp16 before the PV matmul (mirroring `components_gpu.py:205`); FlashInfer runs a fused fp32 online softmax and never materialises `probs`. The two cannot agree bit-for-bit by construction. Tensors are therefore compared at `atol=4e-3, rtol=1e-2` (~4× the fp16 rounding floor) plus a mean-abs-error guard, while **output tokens are compared exactly** — which is the claim the system actually makes. This is defensible because every R9 failure mode (wrong layout, wrong causal alignment, stale page table, wrong GQA mapping) is an O(1) error, not an O(1e-3) one; nothing hides in that gap.

### 2.4 The `AttentionBackend` protocol

```python
class AttentionBackend(Protocol):
    def append_kv(self, layer_idx: int, k: Tensor, v: Tensor, meta: BatchMeta) -> None:
        """Scatter this step's K/V into the physical pool via meta.slot_mapping."""

    def attend(self, q: Tensor, layer_idx: int, scale: float, meta: BatchMeta) -> Tensor:
        """Attention over the full KV history. q is (tokens, n_heads, head_dim);
        returns (tokens, n_heads, head_dim). Causality is the backend's job."""
```

This generalizes the hook the engine already has. `engine/components_gpu.py:114` takes `decode_kernel=` and threads it from `LlamaModelGPU.__init__` (`engine/model_gpu.py:44-53`) down to the call sites at `:79,:147`. Today that hook injects **only the math** — `(q,k,v,scale) -> out` at `components_gpu.py:150-158` — while the engine keeps owning the cache read/write at `:137-142`. Widening it one notch so the injected object owns *both* is the entire change.

In `gqa_attention_gpu`, lines 137–158 are replaced by:

```python
if backend is not None:
    backend.append_kv(layer_idx, k, v, meta)
    out = backend.attend(q, layer_idx, scale, meta)
    return linear(out.reshape(tokens, n_heads * head_dim), o_w)
# ... existing contiguous path unchanged below ...
```

The engine never touches `.k`, `.v`, or `.pos` on the paged path. `KVCacheGPU` (`engine/cache.py:23-33`) is untouched and still serves the reference path.

**Critical design rule: this protocol is varlen-batched from day one, even though Phase 1 will only ever pass `n_seqs == 1`.** A batch-1 protocol widened later means doing the same surgery twice and shipping a Phase-1 interface that has to break. `SERVING_INTERFACE.md:258` recommends deferring batching; that is right about *sequencing the work* and wrong about *sequencing the interface*. **→ ADR-004**

### 2.5 Why varlen makes batching a contained change

The reflexive assumption — mine too, before reading the source — is that batching touches everything. Read the GPU forward path and it does not. With a flattened token axis:

| Component | Location | Batched? |
|---|---|---|
| Embedding lookup | `model_gpu.py:64` `w[...][ids_t]` | **unchanged** — gather over any-length id vector |
| `linear` (all projections, MLP) | `components_gpu.py:14-24` `x @ w.T` | **unchanged** — `(tokens, hidden) @ (hidden, out)` |
| `rms_norm_gpu` | `components_gpu.py:27-31` | **unchanged** — reduces over last dim only |
| `swiglu_ffn_gpu` | `components_gpu.py:185-194` | **unchanged** — elementwise + linears |
| RoPE | `components_gpu.py:132-135` | **already batch-ready** — `positions` is a passed tensor, indexed `cos[positions]`. Pass the right vector and it works. |
| Attention + KV | `components_gpu.py:137-182` | **replaced** by the backend |
| Causal mask | `components_gpu.py:171-174` (2-D `triu`) | **replaced** — becomes the backend's job, expressed via `cu_query_lens`/`kv_lens` |
| Last-token logits | `model_gpu.py:88` `x[-1:]` | **replaced** by `last_token_ix` gather |
| Position construction | `model_gpu.py:65,133` | **moved to the caller** |

Every genuinely batch-sensitive line is inside attention, cache, positions, and the final gather. **The batched forward and the paged cache are the same change to the same surface, not two projects.** That is the most important structural finding in this document, and it is why the four-week freeze is survivable at all.

The dense MLP path being untouched is not luck. A varlen layout is *chosen* precisely so that every position-independent op sees a plain 2-D matrix and cannot tell that multiple sequences are present. The alternative — a padded `(batch, seq, hidden)` layout — would make every op batch-aware, waste compute on padding proportional to length skew, and require a padding mask threaded everywhere. **→ ADR-003**

### 2.6 Engine method surface, after

```python
class LlamaModelGPU:
    # unchanged, still the reference path, still covered by existing tests
    def prefill(self, token_ids, kv_cache) -> np.ndarray: ...
    def decode_step(self, token_id, kv_cache) -> np.ndarray: ...
    def forward_all(self, token_ids) -> np.ndarray: ...
    def make_cache(self, max_seq=2048) -> KVCacheGPU: ...

    # NEW — the serving path
    def forward_varlen(
        self,
        token_ids: Tensor,          # (tokens,) int64, on device
        meta: BatchMeta,
        backend: AttentionBackend,
    ) -> Tensor:                    # (n_seqs, vocab) fp16, ON DEVICE
        ...
```

Note the return type. `prefill`/`decode_step` return CPU numpy (`model_gpu.py:90,158`) — a deliberate engine choice so the numpy sampler works unchanged (`docs/BUILD_LOG.md:823`), costing ~0.1 ms against a ~12 ms step. **`forward_varlen` returns a device tensor**, because at batch 32 that copy is on the critical path for every request in the batch, and because sampling should happen on-GPU.

This has a consequence the benchmark methodology already flags (§5): the per-token device→host copy at `model_gpu.py:158` is what currently forces a CUDA sync and makes the engine's host-clock timings valid (`BENCHMARKS.md:60`). Removing it on the batched path means **host-side timing silently becomes a measurement of kernel-launch queueing, not execution.** Server-side timing on the batched path must use CUDA events or an explicit declared sync point. This is in the risk register as a silent invalidator.

### 2.7 Prerequisite engine changes, precisely

Additive, ~6–10 hours total. Cross-referenced to PRD §6 Tier 0.

1. `engine/attention_backend.py` — `AttentionBackend` Protocol + `BatchMeta` dataclass. New file, no behavior.
2. `engine/components_gpu.py` — `gqa_attention_gpu` gains `backend=None, meta=None`; new branch before the existing cache logic. Existing path byte-for-byte unchanged.
3. `engine/model_gpu.py` — add `forward_varlen`. Existing methods untouched.
4. `engine/model_gpu.py:65` — chunked-prefill position fix on the reference path too (`torch.arange(kv_cache.pos, kv_cache.pos + seq)`), plus a test. One line; latent bug regardless of this project.
5. `tests/test_generate.py` — fix the `"input_ids"` → `"token_ids"` key (`:42,52,63,78,89` vs `oracle.py:170`). All 5 tests currently raise `KeyError`, so the KV-cache-vs-no-cache gate is **not enforced today**.
6. **New: model-level GPU correctness oracle.** Does not exist. `tests/test_gpu_model.py:42-45` asserts only finite/shape/argmax-in-range; GPU fp16 tokens are never compared to HF or to the CPU path. Every existing correctness claim is about the **CPU fp32** path.
7. `scripts/build_kernels.sh:10` — parameterize `-DCMAKE_CUDA_ARCHITECTURES` (currently hardcoded `80`). Required for L40S (sm_89) and H100/H200 (sm_90). Promoted to prerequisite by PRD §8/O1.
8. ~~Git tag `v0.1.0`.~~ **DONE 2026-07-31** — tagged at `6ff40a1` and pushed.

**Item 6 deserves the emphasis.** Building a batched, paged, preemptible scheduler on a forward pass with no output-level oracle means any divergence is unattributable — allocator, batching, paged attention, or a pre-existing GPU-path bug, with no way to bisect. It is cheap to build (greedy tokens from `LlamaModelGPU` vs `tests/oracle.py`'s `greedy_ids`, GPU fp16 vs CPU fp32, so the tolerance is token equality rather than logit distance) and it is the foundation every later claim stands on.

---

## 3. Memory: block allocator and block tables

### 3.1 Layout

```
KV pool per layer:  [num_blocks, block_size, n_kv_heads, head_dim]  fp16
```

`block_size = 16` initially, swept as a tunable. Block-contiguous so appending a token is a single scatter, and it is the layout paged-attention kernels expect.

Sizing: total VRAM − model weights (~2.36 GB fp16, `BENCHMARKS.md:161`) − activation headroom, divided by per-block bytes. `2 (K,V) × 16 layers × 16 tokens × 8 kv_heads × 64 head_dim × 2 bytes = 512 KB per block` across all layers. **[inference]** On a 40 GB A100 with ~35 GB usable for KV, that is roughly 70k blocks ≈ 1.1 M tokens of KV — versus the current design's 2048 tokens *per request* reserved up front regardless of use (`engine/scheduler.py:16,26-27`). That ratio is the S1 claim, and it must be measured rather than computed for publication.

### 3.2 The transpose problem dissolves

The engine pays a full KV-cache transpose per layer per token on the kernel path: cache is stored `(kv_seq, n_kv_heads, head_dim)` (`engine/cache.py:28`), the kernel demands `(n_kv_heads, kv_seq, head_dim)` (`kernels/attention_decode.cu:31`), so `components_gpu.py:153-154` does `.transpose(0,1).contiguous()` on the **entire** K and V every layer every step — ~67 MB of copy traffic per token at kv_seq 2048 (`BENCHMARKS.md:149`).

That cost exists because a layout mismatch forces a **whole-cache** copy. It is not a paging problem and paging does not inherit it: a paged kernel reads `block_tables` and gathers only the blocks it needs, or in FlashInfer's case reads the pool directly with no gather at all. Choose the layout the consuming kernel wants and the copy disappears rather than being optimized. **The correct framing plainly: paging didn't make the transpose cheaper, it made the transpose unnecessary.** **→ ADR-005**

### 3.3 Allocator

Free list over physical block ids. Per-sequence `block_table: list[int]`. Reference count per block, because a block shared by K sequences via the radix cache must not be freed until all K release it.

Copy-on-write: when a sequence writes into a block whose refcount > 1 (it shares a prefix but is now diverging mid-block), allocate a fresh block, copy the shared prefix portion, decrement the old block's refcount, and point the sequence's block table at the new one. Only ever triggered on the *last, partially-filled* block of a shared prefix — full shared blocks are never written to.

**Watermark policy.** The allocator reserves headroom rather than allocating to exhaustion, because a batch that fits at admission may not fit after every sequence in it grows by one token. Running to zero free blocks and *then* preempting means preempting every step. Concretely: admission stops while `free_blocks < watermark`, where the watermark covers one decode step for the entire running batch.

---

## 4. Radix prefix cache

A trie over token ids where each edge spans a block's worth of tokens and each node holds a physical block id, a refcount, and a last-access timestamp.

- **Lookup:** walk the incoming prompt's tokens down the trie; the walk ends at the deepest matching node. Matched blocks are reused (refcount incremented); the remainder is fresh prefill work.
- **Insert:** after prefill, the newly computed blocks are added as trie nodes.
- **Eviction:** LRU over nodes with `refcount == 0`, leaf-first — an internal node cannot be evicted while a descendant is live, because its blocks are part of that descendant's prefix.
- **Granularity:** block, not token. A prefix that matches 20 tokens with `block_size=16` yields **one** reusable block, not 20 reusable tokens. This is why hit rate is reported at block granularity (methodology §7) and why block size trades cache-hit precision against block-table overhead.

**Refcount and LRU interact in the one non-obvious way that is worth understanding cold:** a block can be simultaneously "in the LRU list" and "in use." Eviction must check refcount, not just LRU position, and a block whose refcount drops to zero re-enters eviction eligibility with its *original* access time, not a refreshed one — otherwise a long-running request artificially protects a prefix nobody else wants.

---

## 5. Scheduler

Iteration-level (continuous batching). Per step:

1. **Retire** finished sequences (EOS in `{128001, 128008, 128009}` — `engine/model_gpu.py:19` — or `max_tokens` hit, or client disconnected). Free their blocks, decrement refcounts.
2. **Admit** from the waiting queue while blocks are available above the watermark and the batch-size cap is not hit. Admission performs the radix lookup and reserves blocks.
3. **Preempt** if the running batch cannot be stepped within the block budget.
4. **Assemble** `BatchMeta`, run `forward_varlen`, sample, scatter tokens to per-request output queues.

Prefill and decode are mixed in one batch via chunked prefill: a prefill chunk contributes many tokens on the query axis, a decoding sequence contributes one. This is what `cu_query_lens` exists for.

### 5.1 Chunked prefill is a latency mechanism, not just a throughput one

A 2000-token prefill run as one unit stalls every decoding sequence in the batch for its full duration — a visible ITL spike for every concurrent user, caused by one unrelated request. Capping tokens-per-step bounds that stall.

It also has a concurrency consequence specific to this design: **it is what keeps the event loop responsive.** §7 runs the scheduler as a cooperative task; a step is only non-blocking if a step is short. Chunked prefill is what bounds step duration. The throughput feature and the concurrency model are the same mechanism. **[inference]**

**Phasing consequence — this splits across two phases.** The cooperative scheduler ships in Phase 2, so its responsiveness argument cannot depend on a Phase 4 feature. The work is therefore split:

- **Phase 2 — a minimal prefill token cap.** A long prefill is split into fixed-size chunks purely to bound step duration. No cache interaction, no scheduling policy: split, run in order, done. This is what makes the Phase 2 concurrency claim true on its own.
- **Phase 4 — chunked prefill as a scheduling feature.** Mixing prefill chunks and decodes in one batch, tuning the token cap against the prefill/decode ratio, and the interaction with partial radix-cache hits (a chunk boundary and a cache-hit boundary are different things and must not be conflated).

The Phase 2 half needs `positions` to already be a caller-supplied per-token tensor (§2.3) — which it is, because the same fix makes chunked prefill correct at all. One change, two payoffs.

### 5.2 Preemption — recompute vs swap

The highest-value systems topic in the project (PRD §5, Tier 1), so the design must be defensible in both directions rather than picking one.

When blocks run out, a victim sequence is evicted. Two ways to make room:

**Recompute** — free the victim's blocks entirely; when rescheduled, re-prefill its prompt *and its generated-so-far tokens*. Cost is O(current length) of prefill compute, paid later. Frees 100% of the victim's blocks immediately. No CPU memory needed, no transfer bandwidth.

**Swap** — copy the victim's KV blocks to CPU memory, free the GPU blocks, copy back on resume. Cost is 2× PCIe transfer of the victim's KV. Frees the GPU blocks; consumes pinned host memory.

The comparison is not universal — it depends on the ratio of recompute cost to transfer cost, which depends on model size, sequence length, and PCIe bandwidth:

- **Recompute favored when** the sequence is short (little to redo), the model is small relative to KV volume, or host memory/PCIe is contended. Prefill is highly parallel and compute-bound; a 1B model's prefill is fast.
- **Swap favored when** the sequence is long (recompute cost grows with length while transfer cost grows with the *same* length but at memory-bandwidth rather than compute cost), or when the model is large.

**[inference]** For Llama 3.2 1B specifically, I expect **recompute to win at nearly all lengths**, because the model is tiny — prefill is cheap and KV per token is small (8 kv_heads × 64 dim × 16 layers × 2 × 2 bytes = 32 KB/token), so neither side is under pressure. **That prediction is exactly why it's worth measuring:** a result showing recompute dominating a small model, with the crossover projected for larger ones, is a better answer than reciting the vLLM paper. Both are implemented, chosen by policy flag, and benchmarked head-to-head under forced memory pressure. **→ ADR-006**

**Victim selection:** last-arrived-first (LIFO). Preempting the newest request preserves the progress of older ones and bounds worst-case latency for requests already deep into generation. FIFO preemption would repeatedly punish the oldest request, which is both unfair and unbounded.

**Starvation prevention, and its degenerate case.** A sequence preempted K times becomes *deprioritized* as a victim, forcing forward progress. The naive version — "becomes **ineligible**" — deadlocks: if every running sequence reaches K, nothing is preemptible, and the scheduler can neither step nor make room. The guard is therefore a **preference ordering, not an absolute exclusion**:

1. Prefer victims with preemption count < K, newest first.
2. If all running sequences are at or above K, fall back to preempting the newest anyway. Forward progress for *someone* beats deadlock for everyone.
3. Admission is the real defense: the watermark (§3.3) must stop admitting before the system can reach a state where the running set cannot be stepped at all. If step 2 ever fires, that is an **admission-control bug**, and it is instrumented as such rather than silently absorbed.

The invariant worth stating plainly: *preemption policy must never be able to return "no victim" while the batch is non-empty.*

**Correctness gate:** greedy output under forced preemption must be **bit-identical** to greedy output without preemption. This is the single most important test in the project, because a preemption bug is silent — it produces plausible text, degrades no metric, and would be discovered by nobody.

---

## 6. Where state lives

| State | Owner | Durability | If lost |
|---|---|---|---|
| KV blocks | Replica GPU | Process lifetime | Requests on that replica die; recoverable by re-prefill elsewhere |
| Block allocator free list, refcounts | Replica CPU | Process lifetime | — |
| Radix trie | Replica CPU (indices) + GPU (blocks) | Process lifetime | Cold cache, correctness unaffected |
| Waiting queue, running batch | Replica CPU | Process lifetime | In-flight requests lost |
| Per-sequence state (tokens, block table, position, sampling params) | Replica CPU | Request lifetime | — |
| Replica health, load estimate | Router CPU | Seconds (TTL) | Router probes and rebuilds |
| Prefix → replica hint table | Router CPU | Best-effort, lossy | **Nothing.** Suboptimal routing only. |
| Model weights | Replica GPU | Process lifetime | — |
| Benchmark artifacts | Filesystem, git-tracked | Permanent | — |

**No shared mutable state between replicas. No consensus. No distributed transactions.** The most valuable property of this architecture is what it does *not* contain, and the reason it can omit those things is that the router's state is advisory. **→ ADR-002**

Persistence is deliberately absent from the request path (PRD §3). Benchmark runs go to git-tracked JSON/CSV, matching `bench/harness.py:187-202`.

---

## 7. Concurrency model

The engine has **zero thread safety** — a single global model, mutable cache state, no locks anywhere (`engine/server.py`, `engine/scheduler.py`). Any design that touches the model from more than one thread is wrong.

**Per replica:**
- One asyncio event loop handles HTTP ingress, SSE streaming, and client-disconnect detection.
- The scheduler runs as a cooperative asyncio task, `await`ing between iterations. Each iteration is one bounded forward pass.
- The model executes on one CUDA stream. No intra-replica parallelism across requests — batching *is* the parallelism.
- Per-request `asyncio.Queue` carries output tokens from the scheduler to the SSE writer.

This directly fixes the engine's structural bug. `engine/server.py:69` iterates a blocking synchronous generator inside an `async def`, pinning the event loop for an entire generation — request 2's TTFT includes request 1's full decode, which makes any p99 claim about that server meaningless (`SERVING_INTERFACE.md:130`). Here the loop is released between iterations, so HTTP work interleaves with decode at ~10–30 ms granularity.

**Why a cooperative task rather than a dedicated scheduler thread. →** A thread would need thread-safe queues in both directions and would still serialize on the GIL for the Python-side batch assembly, buying only the ability to overlap Python bookkeeping with GPU execution. Since a bounded step keeps the loop responsive, the added complexity buys little. **The escape hatch is stated in advance:** if profiling shows Python-side batch assembly is a material fraction of step time, the scheduler moves to a thread with `asyncio.run_coroutine_threadsafe` handoff. That is a contained change and not a redesign. **→ ADR-007**

**Cancellation.** A generator abandoned mid-iteration stops at the next `yield` — the engine gets this right by accident (`engine/scheduler.py:11-45`), but has *no way to reclaim the cache* because `KVCacheGPU` is constructed per call and garbage-collected (`:26-27`). Here, client disconnect sets a flag on the sequence; the scheduler retires it at the next iteration boundary and explicitly frees its blocks and decrements refcounts. **Cancellation must free memory, or a disconnect-heavy workload leaks the entire KV pool.**

**Router:** stateless per request, async, no locks on the request path. Health-check and hint-table updates are background tasks.

---

## 8. Failure domains

| Domain | Trigger | Blast radius | Handling |
|---|---|---|---|
| Request | Bad input, context overflow, timeout | One request | 4xx/5xx, no side effects |
| Sequence | KV exhaustion | One sequence, transiently | Preemption (§5.2); transparent to client |
| Replica | Crash, hang, OOM, CUDA fault | All in-flight on that replica | Router detects, drains, re-routes; §9.3 |
| Router | Crash | Everything | **Single point of failure.** Accepted. §8.1 |
| GPU | CUDA error | Entire replica, permanently | Restart the replica. §8.2 |

### 8.1 The router is a SPOF, and that is a deliberate choice

Making it highly available requires either a shared consistent view of replica state (a coordination service, and the whole point of §6 is not having one) or multiple independent routers with divergent hint tables. The second is actually acceptable — hints are advisory, so two routers disagreeing costs cache locality, not correctness — but it is out of scope for the freeze date and adds no learning this project needs. **Stated as a known limitation rather than hidden.** The honest settled answer is: *"it's a SPOF, here's exactly what would be required to fix it, here's why the hint-only design makes that fix cheap, and here's why I didn't."*

### 8.2 CUDA errors are a correctness hazard, not just an availability one

`CLAIMS_AUDIT.md:299` — *"No CUDA error checking anywhere... A launch failure silently produces garbage rather than raising."* The custom kernel's bindings call `cudaDeviceSynchronize()` (`kernels/bindings.cpp:39,52,65`) but never check its return.

For a benchmark harness that produced a visibly wrong token, this was tolerable. **For a serving system it is not:** a launch failure under memory pressure would produce plausible-looking text at full throughput, and every metric would look healthy. Mitigation: check CUDA errors at declared points on the serving path, and treat any CUDA error as fatal to the replica — a poisoned CUDA context cannot be recovered in-process. Goes in the risk register as a silent invalidator.

---

## 9. Sequence walkthroughs

### 9.1 Partial prefix cache hit

Request: 100 tokens. Blocks 0–3 (tokens 0–63) already cached from an earlier request; divergence at token 70, mid-block-4.

```
1. Router          hint table suggests replica 2 holds this prefix → route there.
                   (If the hint is stale, replica 2 simply misses. No error.)
2. Admission       tokenize → 100 ids. Radix walk matches blocks 0–3 (64 tokens).
                   Blocks 4–6 needed for tokens 64–99. Check free ≥ 3 + watermark.
3. Cache reuse     refcount++ on blocks 0–3. Sequence block_table = [b0,b1,b2,b3,·,·,·].
4. Allocate        3 fresh blocks appended → [b0,b1,b2,b3,b7,b8,b9].
5. Prefill         ONLY tokens 64–99 (36 tokens) are computed.
                   positions = arange(64, 100)  ← the chunked-prefill fix.
                   query_lens=[36], kv_lens=[100].
                   slot_mapping maps the 36 tokens into blocks b7,b8,b9.
6. Attention       attends over ALL 100 KV positions via block_tables — the 64 cached
                   plus the 36 just written. The kernel cannot tell which were reused.
7. Insert          blocks b7,b8,b9 inserted into the radix trie under the divergence.
8. Decode          joins the running batch; one token per step.
9. Retire          refcount-- on every block. Blocks 0–3 stay resident (may still be
                   shared); b7–b9 drop to refcount 0 and become LRU-evictable.
```

**Saved: 64 tokens of prefill. Cost: one trie walk.** Which is why the win is TTFT, not decode throughput — and why the methodology reports TTFT as the prefix-cache metric.

**The subtle case is step 5's boundary.** If divergence had fallen at token 70 with `block_size=16`, block 4 would hold tokens 64–79: shared for 64–69, divergent from 70. That block **cannot** be reused, because reuse is only valid for a *fully* matching block. The match is truncated to block 3 and block 4 is recomputed. This is the block-granularity property from §4, it is why the adversarial near-miss workload exists in the methodology (§4), and it is where off-by-one bugs live.

### 9.2 Preemption under memory pressure

Running batch of 8 decoding sequences. Free blocks fall below the watermark; the batch cannot be stepped.

```
1. Detect      scheduler pre-step check: blocks needed for one step across the
               running batch > free blocks. Do NOT enter the forward pass.
2. Select      victim = last-admitted sequence (LIFO), skipping any sequence
               already preempted K times (starvation guard).
3a. RECOMPUTE  free victim's blocks, refcount-- (shared prefix blocks survive).
               Victim returns to the FRONT of the waiting queue with its
               prompt + tokens-generated-so-far as its new prompt.
3b. SWAP       copy victim's blocks GPU→pinned host; free GPU blocks; record the
               host handle. On resume, copy back into freshly allocated blocks.
4. Step        batch of 7 proceeds. Client sees a stall, not an error.
5. Resume      when blocks free up (a sequence retires), the victim is re-admitted.
               RECOMPUTE: re-prefill (radix cache may serve much of it back —
               its own prefix blocks may still be resident, which makes recompute
               cheaper than the naive analysis suggests).
               SWAP: copy back, resume decoding at the exact prior position.
6. Invariant   output tokens are IDENTICAL either way, and identical to a run that
               never preempted. This is the gate from §5.2.
```

Step 5's parenthetical is worth internalizing: **recompute is cheaper than it looks when a radix cache is present**, because the victim's own prefix may still be cached, so "re-prefill" often means re-prefilling only the generated tail. That interaction between two features is a good settled answer and a real reason to expect recompute to win here.

### 9.3 Replica failure mid-request

Replica 3 is serving 12 requests, 5 of them streaming. It dies (OOM, CUDA fault, or hang).

```
1. Detect      (a) active health check fails N consecutive times, or
               (b) an in-flight request's connection errors / times out.
               (b) is faster and is what actually protects latency;
               (a) is what catches a hung-but-connected replica.
2. Quarantine  mark replica 3 unhealthy. Router stops routing new requests to it
               IMMEDIATELY. Purge its entries from the prefix hint table — its
               cache is gone, and stale hints would send requests toward a
               dead replica's prefixes.
3. In-flight   12 requests are lost. Their KV is gone; there is no replication.
   - NOT started (queued at replica): safe to retry on another replica.
   - Started, no tokens emitted yet: safe to retry — client has seen nothing.
   - PARTIALLY STREAMED: cannot transparently retry. The client has already
     received tokens; re-generating from scratch would either duplicate output
     or produce a different continuation. Terminate the SSE stream with an
     explicit error event.
4. Retry       eligible requests re-routed with jittered backoff. Jitter matters:
               12 simultaneous retries hitting the least-loaded replica is a
               self-inflicted thundering herd.
5. Recover     replica restarts, weights reload (~seconds), health check passes,
               router re-admits it — but with an EMPTY cache. A cold replica
               that the prefix router treats as equal will receive traffic it
               serves slowly. Ramp it in gradually rather than at full weight.
6. Drain       (graceful case, not a crash) stop accepting new requests, let
               in-flight ones finish, then exit. This is what makes deploys and
               benchmark teardown clean.
```

**Step 3's third bullet is the honest part.** Mid-stream failure is not transparently recoverable without KV replication, which is not in scope. The correct engineering answer is to fail explicitly rather than silently produce a discontinuous response. **Step 5 is the non-obvious one** — a recovered replica is a *performance* hazard to a cache-aware router, because "healthy" and "warm" are different properties and only one of them is health-checked.

---

## 10. Alternatives considered

Each becomes a full ADR. Summarized here so the tradeoffs are on record.

**A1 — Fork the engine into a monorepo.** Rejected. Two claim sets require two repos, and the dependency boundary is what forces a real interface rather than reaching into internals. Cost: cross-repo changes are slower during Phase 1. Mitigation: git submodule at a pinned commit (`SERVING_INTERFACE.md:234`), so the engine tracks a known SHA and the compiled `.so` sits at a known relative path.

**A2 — Put batching and paging *in* the engine.** Rejected. Collapses the two independent claim sets into one, and puts the engine's clean batch-1 benchmark claims at risk of numeric drift. The engine exposes hooks; the serving layer owns the scheduler.

**A3 — Write a paged-attention CUDA kernel.** Rejected. `kernels/bindings.cpp:29-31` has no stride, block-table, or block-size arguments — the ABI forbids describing a paged cache at all — and `Q` is `[n_heads, head_dim]` with no batch axis (`attention_decode.cu:30-32`), so a paged *batched* kernel is a rewrite, not an extension. 15–25 hours of pointer-arithmetic debugging in a deprioritized lane, plus re-validating the 100-input gate (`tests/test_attention_kernel.py:72-85`). "Can he write CUDA" is already answered by the engine. **→ ADR-001**

**A4 — Bypass the engine's attention entirely from the serving layer.** Rejected. Requires reimplementing the layer loop, which means forking `model_gpu.py` into this repo — strictly worse than either the hook or the kernel rewrite, since it duplicates code without owning it.

**A5 — Padded `(batch, seq, hidden)` batching.** Rejected in favor of varlen. Wastes compute proportional to length skew — and the methodology deliberately uses heavy-tailed length distributions, so the waste would be large. Makes every op batch-aware and threads a padding mask everywhere. Varlen keeps §2.5's "unchanged" column.

**A6 — Shared KV cache across replicas.** Rejected. Requires either RDMA or a distributed cache tier, and its benefit is exactly what prefix-aware *routing* provides at a fraction of the complexity: route the request to the cache instead of moving the cache to the request. This framing is itself the argument for the routing feature.

**A7 — Swap-only or recompute-only preemption.** Rejected; both are implemented behind a policy flag. The comparison is the deliverable (§5.2).

**A8 — Kubernetes + a cache-aware inference gateway.** Deferred, probably permanently. Target hardware is Slurm; the gateway's cache-aware routing duplicates the router built here. PRD §5 Tier 4.

---

## 11. What this document leaves open

- **FlashInfer's exact API and tensor-layout contract is unverified.** Not installed, not read. The `AttentionBackend` protocol is designed so FlashInfer sits *behind* it, which means a mismatch costs an adapter rather than a redesign — but the layout in §3.1 is asserted as reasonable, not as FlashInfer-compatible. **Verify against installed source before committing.** If it disagrees, §3.1 changes and nothing above it does. This is the main reason `PagedTorchBackend` is written first: it is layout-independent and unblocks everything.
- Block size, watermark level, max batch size, chunked-prefill token cap — all tunables, all swept, none asserted.
- The routing policy's affinity/load blending function. Methodology §10 predicts prefix-aware routing *loses* above the knee without it; the blend is a design decision that depends on measuring where the knee is.
- Sampling on GPU vs the engine's numpy sampler (`engine/sampler.py`). `forward_varlen` returns a device tensor precisely to keep this open; batch-32 numpy sampling round-trips per step would be measurable.
- Phase assignment. Everything here is the full system; the four-week freeze means most of §9.3 and all of the router are fall work. That slicing is deliverable #5.
