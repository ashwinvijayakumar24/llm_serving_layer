# Source notes — vLLM, SGLang and this serving layer, read from source

This file records what the three engines actually do, read from their source at
pinned versions. It is the citation base for the analysis sections of
`docs/xengine/BENCHMARKS.md` (see `SPEC.md`, "Deliverables").

Every claim cites `path:line`. For vLLM and SGLang the path is relative to that
repository's root at the tag and commit below. For this repository the path is
relative to the repo root. Summaries are paraphrased; nothing here is quoted.

A claim marked **VERIFIED** was read in source. A claim marked **UNVERIFIED** is
either a performance claim (which only a measurement can settle) or something
that was not traced end to end.

## 1. Pinned versions

The pins are the latest stable release on PyPI on 2026-10-08.

| engine | version | PyPI upload | git tag | commit SHA |
|---|---|---|---|---|
| vLLM | 0.31.0 | 2026-10-05 | `v0.31.0` | `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| SGLang | 0.5.21 | 2026-10-01 | `v0.5.21` | `e00930c5489053f26d86b179cee0d087f846acbb` |
| ours | this repo | — | branch `xengine-source` | engine submodule `vendor/llm_inference_engine` @ `a3d0ab80f41d8ad834a385fa81744574e31b2eed` |

Runtime requirements, taken from each package's PyPI `requires_dist`:

| engine | torch | flashinfer | CUDA |
|---|---|---|---|
| vLLM 0.31.0 | `torch==2.13.0` | `flashinfer-python==0.7.0.post1` | 13.0 in source (`vllm/envs.py:92`, `docker/Dockerfile:25` sets 13.0.3). The docs say 12.9 — see friction F-V1. Which CUDA build the plain PyPI wheel targets is **UNVERIFIED**. |
| SGLang 0.5.21 | `torch==2.13.0` | `flashinfer_python[cu13]==0.6.18` | CUDA 13 (from the `[cu13]` extra and `cuda-python>=13.0`). SGLang also pins `sglang-kernel==0.4.7`. |

Two consequences follow for the harness.

- **The two engines pin different FlashInfer versions** (0.7.0.post1 vs 0.6.18).
  They cannot share one Python environment. Each needs its own venv, and our
  engine's FlashInfer version must be recorded separately.
- **Both need a CUDA 13 driver stack.** The study node's driver must support
  CUDA 13 before either engine is installed.

The clones used for this reading live at `/private/tmp/claude-501/xengine-src/`
(shallow, outside the repo).

---

## 2. vLLM 0.31.0 (V1 engine)

All paths are relative to the vLLM repo root.

### 2.1 Scheduler: how one step is built

The V1 scheduler has no separate "prefill phase" and "decode phase". Every
request just tries to catch its count of computed tokens up to its total token
count, and a prefill chunk and a decode token are the same kind of work
(`vllm/v1/core/sched/scheduler.py:562-573`, the `schedule()` docstring).

The step is built in two passes against one token budget.

1. **The budget.** The per-step budget is `max_num_scheduled_tokens`, which
   defaults to `max_num_batched_tokens` (`scheduler.py:132-134`). The cap on
   running requests is `max_num_seqs` (`scheduler.py:124`).
2. **Running requests go first** (`scheduler.py:629-630`). For each one the
   scheduler asks the KV manager for slots (`allocate_slots`). If that fails, it
   preempts (section 2.4) and retries (`scheduler.py:747-813`). Scheduled tokens
   are subtracted from the budget (`scheduler.py:825`).
3. **Waiting requests go second, and only if nothing was preempted this step**
   (`scheduler.py:872-873`). Why this matters: under memory pressure a
   preempting step admits no new work, which protects running requests but
   lengthens queueing time for new ones.
   - Admission stops once the running count reaches the cap
     (`scheduler.py:892-893`).
   - The prefix-cache lookup happens here (`get_computed_blocks`,
     `scheduler.py:1004`).
   - `scheduler_reserve_full_isl` (default True,
     `vllm/config/scheduler.py:191`) admits a request only if its whole prompt
     fits in KV (`scheduler.py:1218`).

**Chunked prefill.** A waiting request's chunk is the smaller of its remaining
tokens and the remaining budget (`scheduler.py:1123`). If chunked prefill is off
and the prompt does not fit, the waiting loop stops (`scheduler.py:1115-1121`).
`long_prefill_token_threshold` caps the chunk any one request may take
(`scheduler.py:679-680`, `:1110-1111`). It defaults to 0, meaning off
(`vllm/config/scheduler.py:80`).

**Queue policy.** The default policy is FCFS (first come, first served;
`vllm/config/scheduler.py:160`). The FCFS waiting queue is a deque
(`vllm/v1/core/sched/request_queue.py:75`); there is no prefix-aware reordering.

### 2.2 KV cache manager and block pool

- `BlockPool` (`vllm/v1/core/block_pool.py:136`) owns a free-block queue kept in
  eviction order (`:139`, `:178`) and a map from block hash to cached block
  (`:181`).
- **Allocation evicts lazily.** `get_new_blocks` pops from the front of the free
  queue. Only at that moment does it drop the popped block's cached hash
  (`block_pool.py:669-703`, `_maybe_evict_cached_block` at `:732`). So a freed
  block stays reusable as a cache hit until something actually needs its memory.
- **Freeing decides eviction order** (`block_pool.py:777-808`). Blocks without a
  hash go to the front of the queue (reused first). Blocks with a hash go to the
  back, which makes eviction least-recently-used (LRU). A request frees its
  blocks in reverse, so its tail blocks are evicted before its head blocks
  (`vllm/v1/core/single_type_kv_cache_manager.py:581-589`).
- **KV usage** is reported as one minus free over total, excluding one reserved
  null block (`block_pool.py:880-891`).

### 2.3 Prefix caching

- **On by default.** The config default is True (`vllm/config/cache.py:142`). The
  CLI default is None (`vllm/engine/arg_utils.py:572`), which resolves to "is
  prefix caching supported", True for decoder LMs (`arg_utils.py:2958-2964`).
- **Block hashing.** Each full block is hashed as (parent hash, the block's token
  tuple, extra keys). Hashes chain, so a block's hash identifies its whole prefix
  (`vllm/v1/core/kv_cache_utils.py:684-714`). The default hash function is
  sha256 over a pickle (`cache.py:144`). Only full blocks get a hash
  (`kv_cache_utils.py:836-880`).
- **The last token is always recomputed.** A hit is capped at the prompt length
  minus one, which can force the last full block to be recomputed
  (`vllm/v1/core/kv_cache_manager.py:289-296`). Our cache has the same rule
  (section 4.3).
- **What is shared:** physical blocks, by reference count. A hit calls `touch()`,
  which pulls a block with ref count 0 out of the free queue and increments it
  (`block_pool.py:755-771`).
- **Granularity for Llama-3.2-1B: full 16-token blocks.** Sub-block partial
  entries exist (`block_pool.py:449`, `cache.py:91`) but apply only to models
  with several KV-cache groups (`kv_cache_utils.py:745-760`), which a plain Llama
  is not.

### 2.4 Preemption

- **Victim selection** (`scheduler.py:763-771`).
  - Under the default FCFS policy the victim is the last entry of the running
    list, `running[-1]`. That is the most recently *admitted* request, which is
    not always the most recently *arrived* one (a resumed request is appended
    again).
  - Under the priority policy the victim is the request with the largest
    (priority, arrival time) — lowest priority, latest arrival on ties.
  - If the victim is the request being scheduled, scheduling stops for this step
    (`scheduler.py:811-813`).
- **Mode: recompute only.** `_preempt_request` frees the victim's blocks
  (`scheduler.py:1556`), resets its computed-token count to 0 (`:1560`),
  increments its preemption count (`:1575`) and puts it at the **front** of the
  waiting queue (`:1580`).
- **V1 has no swap.** No `PreemptionMode` or `swap_space` exists anywhere under
  `vllm/`, and the project's own design doc records that `--swap-space` was
  removed (`docs/design/metrics.md:528`). CPU KV offloading (`kv_offloading_size`,
  `cache.py:256`) is a separate, default-off feature.
- **Why recompute is cheaper than it sounds:** the victim's freed blocks keep
  their hashes while they sit at the LRU tail (section 2.2), so the resumed
  request often hits its own old prefix.

### 2.5 CUDA graphs and torch.compile

- **The default optimization level is O2** (`vllm/config/vllm.py:442`; levels at
  `:129-141`). O2 sets `cudagraph_mode=FULL_AND_PIECEWISE` (`vllm.py:316`) and
  turns on fusion passes (`vllm.py:300-317`). Any level above O0 compiles the
  model with torch.compile and Inductor (`vllm.py:1774-1778`).
- **FULL_AND_PIECEWISE** means: full CUDA graphs for batches that are pure
  decode, and piecewise graphs (attention runs eagerly between captured pieces)
  for prefill and mixed batches (`vllm/config/compilation.py:53-63`, docstring at
  `:606-640`). So graphs are **not decode-only**: mixed batches are partly
  captured too.
- **Captured sizes** (`vllm.py:2388-2470`). The list is 1, 2, 4, then steps of 8
  up to 256, then steps of 16 up to a maximum (`vllm.py:2400`). The maximum is
  `min(max_num_seqs × decode query length × 2, ceiling)`, where the ceiling is
  512, or 1024 on Blackwell (`vllm.py:2448-2455`).
- **`--enforce-eager` turns off both compilation and CUDA graphs**
  (`vllm.py:1694-1696`). Why this matters: the `vllm-eager` arm measures graphs
  and compiled fusions together. It cannot attribute the gap to graphs alone. To
  remove only graphs, pass a compilation config with `cudagraph_mode` NONE
  (`ENGINE_FLAGS.md`).
- Model Runner V2 is the default on CUDA (`vllm.py:701-760`).

### 2.6 Sampling, attention backend, overlap

- **Sampling runs on the GPU** (`vllm/v1/worker/gpu/sample/sampler.py:42`,
  `sample()` at `:275`).
- **Attention backend on CUDA for a standard model**
  (`vllm/platforms/cuda.py:152-180`): FlashAttention first on every GPU except
  Blackwell (SM100), where FlashInfer comes first. So on an A100 or H100 vLLM
  does **not** use FlashInfer by default, unlike our engine.
- **vLLM also overlaps CPU scheduling with the GPU by default.** "Async
  scheduling" (`vllm/config/scheduler.py:209`) resolves to True for a single-GPU
  generate model (`vllm.py:1576-1635`). It keeps two batches in flight
  (`vllm.py:592-602`, driven by `vllm/v1/engine/core.py:670`). Why this matters
  for hypothesis (e): the overlap is not unique to SGLang, so a fair diagnostic
  needs a `--no-async-scheduling` vLLM arm next to `sglang-nooverlap`.

### 2.7 Defaults (OpenAI API server, single GPU)

| knob | default | citation |
|---|---|---|
| `max_num_seqs` | 256 on GPUs under 70 GiB and on any A100; 1024 on H100/H200-class | `arg_utils.py:2858-2887`, applied `:3058-3062` |
| `max_num_batched_tokens` | 2048 / 8192 / 16384 by the same GPU classes | `arg_utils.py:2858-2887` |
| `block_size` | 16 | `cache.py:71` |
| `gpu_memory_utilization` | 0.92 | `cache.py:103` |
| `enable_prefix_caching` | True | `cache.py:142` |
| `enable_chunked_prefill` | True | `vllm/config/scheduler.py:135`; `arg_utils.py:2928-2934` |
| `long_prefill_token_threshold` | 0 (off) | `vllm/config/scheduler.py:80` |
| `async_scheduling` | True (resolved) | `vllm.py:1635` |
| `seed` | 0 | `vllm/config/model.py:178` |
| `dtype` | auto (BF16 for a BF16 checkpoint) | `model.py:168-177` |

Because the batch defaults depend on the GPU class, the harness must record the
resolved values on every run rather than assume them.

---

## 3. SGLang 0.5.21

Paths are relative to the SGLang repo root. `P/` stands for `python/sglang/srt/`.

**How SGLang's CLI is built in this version.** `server_args.py` no longer lists
flags one by one. Flags are generated from dataclasses in `P/arg_groups/fields/`
(`P/server_args.py:283-285`); each field name becomes a flag by replacing `_`
with `-` (`P/arg_groups/arg_utils.py:288-290`), and a boolean field becomes a
`store_true` flag (`arg_utils.py:393-395`). Defaults that depend on the machine
are filled in later by "resolution hooks" in `P/arg_groups/*_hook.py`.

### 3.1 Scheduler main loop

`dispatch_event_loop` picks the overlap loop when overlap is enabled, otherwise
the normal loop (`P/managers/scheduler.py:5932-5967`). Overlap is on unless
`--disable-overlap-schedule` is passed (`scheduler.py:489-491`).

- **Normal loop** (`scheduler.py:1906-1939`): receive requests, pick the next
  batch, run it, process its result — strictly in sequence.
- **Choosing the next batch** (`get_next_batch_to_run`, `scheduler.py:3641-3802`).
  It first tries to build a new prefill batch, and **if one exists it runs
  instead of decode** (`scheduler.py:3746-3748`). Otherwise it runs a decode step
  for the running batch. Prefill and decode are separate batches unless
  `--enable-mixed-chunk` is set (default False,
  `P/arg_groups/fields/schedule.py:202-205`).
- **Building a prefill batch** (`_get_new_batch_prefill_raw`,
  `scheduler.py:3831-4126`): order the waiting queue by policy, then add
  requests through a `PrefillAdder` until the token budget or the running cap is
  hit.

### 3.2 Overlap scheduler

The overlap loop (`scheduler.py:1941-2002`) runs one iteration ahead of the GPU.

1. While the GPU is still running batch N, the CPU receives requests and builds
   batch N+1.
2. Batch N+1 is launched on a separate forward stream without waiting. The batch
   and its future result are pushed onto a result queue.
3. Only then does the CPU pop and process batch N's result.

The trick that makes step 1 possible is the `FutureMap`
(`P/managers/overlap_utils.py:248`). Batch N+1 needs batch N's sampled tokens as
its input, but the CPU has not seen them yet. So batch N+1's inputs hold
placeholder indices, and the GPU resolves them in-stream
(`resolve_forward_inputs`, `overlap_utils.py:87`; publication at
`scheduler.py:4379-4419`). Sampling is deferred until after launch
(`scheduler.py:4727-4752`).

Overlap is skipped for single batches in a few special cases, for example back
to back prefills when an env var is set, or a grammar sync
(`scheduler.py:2004-2040`).

### 3.3 Prefix cache (radix tree)

**The live cache is not the Python `RadixCache` class.** The default factory
builds a `UnifiedRadixCache` (`P/mem_cache/registry.py:80-148`,
`P/mem_cache/unified_radix_cache.py:165`), whose tree operations run in a Rust
core by default (`P/environ.py:690`; `rust/sglang-radix-tree/src/unified_tree_core.rs`,
`match_prefix` at `:1131`, `insert` at `:1512`, lock refs at `:905` and
`:1004`). It falls back to Python off Linux and in a few other cases
(`P/mem_cache/unified_cache/tree_core_registry.py:44-131`).

The Python `RadixCache` implements the same algorithm and is easier to read:

- **`match_prefix`** (`P/mem_cache/radix_cache.py:354`) walks the tree with the
  key aligned to `page_size` (`:397`). A match that ends partway along an edge
  splits the node (`_split_node`, `:665`), so the matched part becomes its own
  node.
- **`insert`** (`radix_cache.py:414`) adds the new suffix as a child.
- **Lock references** (`inc_lock_ref` `:583`, `dec_lock_ref` `:598`) walk from a
  node up to the root. A node with a nonzero lock count is in use by a running
  request and cannot be evicted.
- **`evict`** (`radix_cache.py:553-581`) puts unlocked leaves in a heap ordered
  by the eviction strategy and frees them. When a parent becomes an unlocked
  leaf it joins the heap. The Rust core does the same with a min-heap of
  evictable leaves (`rust/sglang-radix-tree/src/components/full.rs:135-192`).
- **Eviction policy:** LRU by default; LFU, SLRU, priority and TLRU are also
  accepted (`P/arg_groups/fields/memory.py:25-43`, `P/arg_groups/choices.py:184`).
- **Granularity: one token by default.** `page_size` resolves to 1 on CUDA
  (`P/arg_groups/overrides.py:1278-1302`). Why this matters: SGLang can reuse a
  prefix down to the token, while vLLM and our engine reuse only whole 16-token
  blocks. On W3 the difference is at most 15 tokens per request, which is small
  next to a long shared system prefix. Some attention backends force a larger
  page (trtllm_mha 64, fa4 128; `overrides.py:1114-1118`, `:1233-1235`).
- **`--disable-radix-cache`** swaps in a `ChunkCache` when chunked prefill is on
  (`registry.py:90-94`); otherwise every cache method becomes a no-op
  (`unified_radix_cache.py:556`, `:581`, `:612`).

### 3.4 Schedule policy

- **The default policy is `fcfs`, not `lpm`** (`P/arg_groups/fields/schedule.py:82-98`).
  This contradicts the common description of SGLang as "longest-prefix-match
  scheduling by default", and it matters for hypothesis (b).
- **LPM** (longest prefix match; `P/managers/schedule_policy.py:477-487`) sorts
  the waiting queue by matched prefix length, longest first. It also does
  in-batch deduplication: if several waiting requests share a new prefix of at
  least 32 tokens, only one is scheduled now so that the rest can hit the cache
  next round (`schedule_policy.py:373-431`, thresholds `:100-109`).
- **DFS-weight** (`schedule_policy.py:532-538`) orders requests by a
  depth-first walk of the tree, weighted by how many requests sit under each
  node.
- **LPM falls back to FCFS when more than 128 requests are waiting**
  (`schedule_policy.py:331-342`), and every cache-aware policy falls back when
  the cache is disabled (`:351-371`).
- **Prefix reuse happens under every policy**, including FCFS: each request is
  matched against the tree at admission (`P/managers/schedule_batch.py:1581`).
  Only the ordering and the in-batch dedup are LPM-specific.

**Admission budget (`PrefillAdder`, `schedule_policy.py:621`).** A request is
charged its new prompt tokens plus its remaining output tokens, capped at 4096
(`schedule_policy.py:79-81`, `:1347-1463`). Running requests are reserved at
their remaining output times `new_token_ratio` (`:788-796`), the scheduler's
running estimate of how much of `max_tokens` a request will really use (section
3.5). With the radix cache disabled, `ignore_eos` requests take a separate
admission path that reserves the full `max_tokens` (`:1353-1354`,
`:1200-1345`). Why this matters: the `sglang-noradix` arm on W4 (which sets
`ignore_eos=true`) admits differently from the default arm, not only without a
cache.

### 3.5 Retraction (SGLang's preemption)

- **Trigger.** Before each decode step, `update_running_batch` checks whether
  the step fits (`scheduler.py:4191-4270`). If not, it calls `retract_decode()`
  and requeues the victims.
- **Victim order** (`schedule_batch.py:3385-3425`). With the default
  `--retraction-policy length` (`P/arg_groups/fields/schedule.py:121-134`),
  requests are sorted by (generated tokens, minus prompt length) and popped from
  the end. **So the first victim is the request that has generated the fewest
  tokens, with ties broken toward the longest prompt.** At least one request is
  always kept (`schedule_batch.py:3308`).
- **Mode: recompute.** `release_req` frees the request's KV without inserting
  its generated tokens into the tree (`schedule_batch.py:2239-2274`, the key
  call at `:2268`), then resets the request (`reset_for_retract`, `:1961`). A
  host-memory backup exists only in prefill/decode disaggregation mode
  (`:2257-2266`). That the retracted prompt can re-hit its own prefix on
  re-admission is plausible but **UNVERIFIED** (not traced end to end).
- **`new_token_ratio`** starts at 0.7 times `schedule_conservativeness`, decays
  linearly over 600 steps to 0.14 of its start, and after a retraction is reset
  from observed progress
  (`P/managers/scheduler_components/new_token_ratio_tracker.py:21-49`).

### 3.6 CUDA graphs

- **Both decode and prefill are captured by default on CUDA.** The decode
  backend defaults to `full` and the prefill backend to `breakable`
  (`P/model_executor/cuda_graph_config.py:122-131`, `:157-161`). This is a change
  from older releases, where graphs were decode-only.
- **Decode batch sizes** (`P/arg_groups/cuda_graph_hook.py:642-674`): 1, 2, 4, 8,
  12, then 16 to 256 in steps of 8, then larger steps, capped at the decode
  maximum.
- **The decode maximum depends on GPU memory** (`P/arg_groups/memory_hook.py:85-180`):
  32 on 40–60 GB cards (A100-40G, L40S), 256 on 60–160 GB cards (A100-80G,
  H100, H200). A batch larger than the maximum runs without a graph
  (`P/model_executor/runner/decode_cuda_graph_runner.py:672`). Why this matters:
  on an A100-40G, W1 at concurrency 64 would decode eagerly.
- Prefill buckets are token counts capped at `chunked_prefill_size`
  (`cuda_graph_hook.py:622-639`; `memory_hook.py:224-262`).

### 3.7 Chunked prefill, sampling, attention backend, fused kernels

- **Chunked prefill** is on by default; the size comes from GPU memory (4096 on
  40–60 GB, 8192 on 60–160 GB; `memory_hook.py:85-180`). A value of `-1` (any
  value ≤ 0) disables it (`scheduler.py:1304-1329`). Only one request may be
  mid-chunk at a time (`schedule_policy.py:1516`).
- **Sampling runs on the GPU.** Greedy is a `torch.argmax`
  (`P/layers/sampler.py:185-193`).
- **Attention backend** (`P/arg_groups/model_override_base.py:247-318`): FA3 on
  Hopper with CUDA ≥ 12.3, `trtllm_mha` on Blackwell, FlashInfer otherwise
  (falling back to Triton). The choice is logged at startup
  (`overrides.py:1058-1060`).
- **Fused kernels:** fused residual-add + RMSNorm and fused SiLU-and-multiply
  from `sgl_kernel` (`P/layers/layernorm.py:102-106`, `P/layers/activation.py:62`).

### 3.8 Defaults

| knob | static default | resolved default | citation |
|---|---|---|---|
| `mem_fraction_static` | None | computed from GPU memory minus reserves; 0.95 if memory is unknown | `fields/schedule.py:27`; `memory_hook.py:274-330` |
| `max_running_requests` | None | about 2048 for Llama-3.2-1B | `fields/schedule.py:31`; `P/mem_cache/kv_cache_configurator.py:2291-2304` |
| `max_total_tokens` | None | profiled; a user value can only lower it | `fields/schedule.py:39`; `kv_cache_configurator.py:2262-2277` |
| `page_size` | None | 1 on CUDA | `fields/schedule.py:139`; `overrides.py:1278-1302` |
| `chunked_prefill_size` | None | 4096 / 8192 / 16384 by GPU memory | `fields/schedule.py:52`; `memory_hook.py:85-180` |
| `schedule_policy` | fcfs | fcfs | `fields/schedule.py:82-98` |
| `disable_radix_cache` | False | False | `fields/memory.py:64` |
| `disable_overlap_schedule` | False | False on a plain single GPU | `fields/schedule.py:187`; auto-True cases at `overrides.py:1551-1555` |
| decode graph max batch | None | 32 / 256 / 512 by GPU memory | `fields/exec_.py:500`; `memory_hook.py:85-180` |
| `enable_metrics` | False | False | `fields/observability.py:70` |
| `random_seed` | None | a random integer | `fields/device.py:47`; `P/arg_groups/serving_hook.py:603-607` |

The random default seed matters: an SGLang run without `--random-seed` is not
reproducible even under greedy decoding with ties.

---

## 4. Our serving layer

Paths are relative to this repo. The model forward lives in the engine
submodule `vendor/llm_inference_engine` (commit above).

### 4.1 Scheduler (`serving/scheduler/scheduler.py`)

- **Step order** (`step()`, `scheduler.py:696-771`): retire finished requests,
  resume swapped ones, admit waiting ones, preempt if the next step would not
  fit, select the batch, run the forward pass, sample, apply.
- **Admission is FIFO with deliberate head-of-line blocking**
  (`_admit`, `scheduler.py:514-551`; the break at `:539-543`). Admission is
  watermark-gated and counts the running batch's next-step block need
  explicitly (`_can_admit`, `scheduler.py:461-512`, especially `:503`). Swapped
  requests block new admission entirely (`scheduler.py:527-528`).
- **Batch selection** (`_select_batch`, `scheduler.py:612-694`). Decodes are
  taken first and never refused (`:658-662`). Prefill chunks fill what is left
  of one token budget, which defaults to `max_prefill_tokens + max_batch_size`
  (`:654-656`), additionally capped at `max_prefill_tokens` (`:666`). Non-final
  chunks round down to a block boundary (`:677-685`), so chunk ends are legal
  cache publication points.
- **Sampling** is a GPU `torch.argmax` followed by `.tolist()`
  (`scheduler.py:757`). The `.tolist()` is a device-to-host copy that blocks the
  CPU until the forward pass finishes, every step.
- **No CPU/GPU overlap.** The server's loop calls `scheduler.step()`
  synchronously on the asyncio event loop (`serving/server/app.py:489-498`), so
  scheduling, HTTP handling and detokenization all wait for the GPU and the GPU
  waits for them.
- **Defaults** (`serving/server/app.py:1105-1116`): `block_size=16`,
  `max_batch_size=32`, `max_prefill_tokens=512`, `max_waiting=1024`,
  `prefer_flashinfer=True`; watermark = `max_batch_size` blocks
  (`app.py:1226-1231`). Env knobs: `SERVING_KV_BLOCKS`, `SERVING_PREEMPTION`,
  `SERVING_PREFIX_CACHE` (`app.py:1145-1157`).

### 4.2 Preemption (`serving/scheduler/preemption.py` + scheduler)

- **Trigger:** a pre-step check compares the exact block need of this step's
  batch with free blocks and preempts in a loop until it fits
  (`_preempt_if_needed`, `scheduler.py:837-884`). A lone request that cannot
  fit is failed loudly rather than livelocked (`:859-868`).
- **Victim selection: LIFO by arrival order** with a starvation guard
  (`select_victim`, `preemption.py:130-195`). Requests preempted fewer than
  `starvation_k` (default 3, `scheduler.py:328`) times are preferred, newest
  first; if all have hit the limit, the newest is taken anyway and the fallback
  is counted (`preemption.py:185-195`).
- **Mode: recompute by default, swap available** (`scheduler.py:320`,
  `_preempt` at `:886-901`). Recompute re-prefills prompt plus generated tokens
  and requeues at the front (`_preempt_recompute`, `:903-958`). Swap copies KV to
  pinned host memory and degrades to recompute when the host budget is full
  (`_preempt_swap`, `:960-987`; `KVSwapSpace`, `preemption.py:335`).

### 4.3 Radix cache (`serving/cache/radix.py`)

- **Structure:** a token trie where **each edge is exactly one 16-token block**
  (`_full_block_keys`, `radix.py:350-360`; invariant at `:735-738`). A partial
  trailing block is never a key.
- **`match`** (`radix.py:362-421`) is a pure probe used for admission sizing.
  It also measures how many tokens agreed inside the block where matching died
  (`:390-406`) — sharing the cache cannot sell — and applies the last-token rule:
  a fully cached prompt gets one block fewer (`:414-415`).
- **`acquire`** (`radix.py:423-453`) increfs matched blocks in the allocator and
  stamps their logical access time.
- **`insert`** (`radix.py:455`) publishes completed blocks. The scheduler
  publishes at every chunk boundary (`scheduler.py:800`) and at retirement,
  before freeing (`scheduler.py:1209`).
- **Eviction: LRU among unused leaves** (`_evict_one`, `radix.py:546-572`), a
  linear scan. A block is evictable only when it is a leaf and the allocator
  says the cache is its only holder (`_users`, `radix.py:338-346`).
- **When eviction runs:** only at admission (`scheduler.py:511`, `:608`) or when
  a `max_cached_blocks` budget is set (`radix.py:608-616`). See section 8,
  item O-1.

### 4.4 Allocator (`serving/memory/allocator.py`)

- A fixed pool of block ids with a **FIFO free list** (`deque`,
  `allocator.py:86`; pop from the front `:164`, return to the back `:217`) and
  per-block reference counts (`free`, `allocator.py:195-217`).
- `can_allocate` respects the watermark and is what admission calls
  (`allocator.py:133-143`). `allocate` ignores it and is what running-sequence
  growth calls (`serving/memory/block_table.py:79-95`).

### 4.5 Forward pass, attention, CUDA graphs

- **Attention backend:** FlashInfer's batch prefill and batch decode
  paged-KV wrappers (`serving/backends/flashinfer_backend.py:337-346`), with a
  silent fallback to the PyTorch `PagedTorchBackend` if FlashInfer fails to
  import or construct (`app.py:1208-1224`). The SPEC requires the harness to
  treat that fallback as an invalid run.
- **No CUDA graphs**, by design (`flashinfer_backend.py:148-149`).
- **No fused kernels in the model.** The forward is a Python loop over layers
  (`vendor/llm_inference_engine/engine/model_gpu.py:102-170`). RMSNorm is several
  separate elementwise ops in fp32 (`engine/components_gpu.py:27-31`); Q, K and
  V are three separate matmuls (`components_gpu.py:140-142`); RoPE is applied to
  Q and K separately (`:146-147`); gate and up are two separate matmuls
  (`components_gpu.py:214-223`).

---

## 5. Side-by-side comparison

Short cells; the explanation is in sections 2–4.

| aspect | ours | vLLM 0.31.0 | SGLang 0.5.21 |
|---|---|---|---|
| KV block / page size | 16 tokens | 16 tokens | 1 token (page_size 1) |
| prefix-cache structure | trie, one block per edge | chained block-hash map | radix tree (Rust core) |
| prefix-cache granularity | full 16-token blocks | full 16-token blocks | single tokens |
| last token recomputed on full hit | yes | yes | — (not checked) |
| prefix-cache default | on | on | on |
| eviction policy | LRU, unused leaves | LRU over freed hashed blocks, lazy | LRU, unlocked leaves (configurable) |
| when cache eviction runs | admission only | whenever a block is allocated | on demand, incl. before retraction |
| waiting-queue order | FIFO | FCFS | FCFS (LPM opt-in) |
| preemption mode | recompute (swap optional) | recompute only | recompute (retraction) |
| victim | newest arrival, with starvation guard | last admitted (FCFS) | fewest generated tokens |
| requeue position | front | front | back into the waiting queue |
| CUDA graphs | none | full (decode) + piecewise (mixed) | decode full + prefill breakable |
| torch.compile / fusions | none | Inductor + fusion passes | sgl_kernel fused ops |
| chunked prefill | on, 512-token cap, block-aligned | on, 2048–16384 budget | on, 4096–16384 chunk |
| prefill/decode mixing | mixed in one step | mixed in one step | separate batches (mixed opt-in) |
| CPU/GPU overlap | none | async scheduling (default on) | overlap loop (default on) |
| sampling | GPU argmax + blocking `.tolist()` | GPU | GPU |
| attention backend | FlashInfer (silent fallback) | FlashAttention (FlashInfer on SM100) | FA3 on Hopper, FlashInfer otherwise |

Notes on two cells:

- **"Requeue position" for SGLang**: retracted requests are put back into the
  waiting queue (`scheduler.py:4191-4270`); with FCFS ordering by arrival time
  they may still sort early. Exact position was not traced and is
  **UNVERIFIED**.
- **"When cache eviction runs" for SGLang**: retraction itself evicts tokens
  from the tree (`schedule_batch.py:2239-2274`). That the decode-memory check
  evicts before retracting is **UNVERIFIED**.

---

## 6. Hypotheses: which code would confirm or refute them

### (a) The throughput gap is dominated by CUDA graphs, fused kernels and Python overhead in our forward

- **Code paths.** Ours: no graphs (`flashinfer_backend.py:148`), unfused ops
  (`components_gpu.py:27-31`, `:140-147`, `:214-223`), a per-layer Python loop
  (`model_gpu.py:102-170`), a blocking `.tolist()` (`scheduler.py:757`), and a
  synchronous step on the event loop (`app.py:498`). vLLM: compile + graphs by
  default (`vllm.py:316`, `:1774-1778`). SGLang: graphs for both phases
  (`cuda_graph_config.py:157-161`), fused ops (`layernorm.py:102`,
  `activation.py:62`).
- **Status.** The *mechanism differences* are **VERIFIED** in all three sources.
  That they *dominate* the gap is **UNVERIFIED**; the eager arms will measure it.
  Caveat: `vllm-eager` removes graphs and compiled fusions together
  (`vllm.py:1694-1696`), so it bounds their combined contribution, not graphs
  alone.

### (b) SGLang's radix cache plus longest-prefix scheduling beats both on W3

- **Code paths.** SGLang default policy (`fields/schedule.py:82-98`), LPM sort
  and in-batch dedup (`schedule_policy.py:373-431`, `:477-487`), LPM fallback
  above 128 waiting (`:331-342`), token granularity (`overrides.py:1278-1302`).
  vLLM: FCFS queue (`request_queue.py:75`), block hashing
  (`kv_cache_utils.py:684-714`).
- **Status.** **Partly refuted for default configurations (VERIFIED):** SGLang's
  default policy is FCFS, so "longest-prefix scheduling" is not active unless
  `--schedule-policy lpm` is passed. All three engines reuse prefixes by default.
  Whether SGLang wins W3 is **UNVERIFIED**. If the study wants to test LPM, it
  needs an extra opt-in arm, flagged as non-default.

### (c) vLLM V1 preempts by recompute only, picking the lowest-priority / latest request

- **Code paths.** `scheduler.py:763-771` (victim), `:1538-1581` (recompute
  preemption), absence of any swap mode under `vllm/`.
- **Status.** **VERIFIED.** One refinement: under the default FCFS policy the
  victim is the last-*admitted* running request, `running[-1]`, not strictly the
  latest arrival.

### (d) Chunked prefill reduces the TTFT tail under mixed lengths

- **Code paths.** vLLM: running first (`scheduler.py:629-630`), chunk sizing
  (`:1123`), per-request cap (`:679`, `:1110`). SGLang: prefill-first
  (`scheduler.py:3746-3748`), chunk size (`memory_hook.py:85-180`), one chunked
  request at a time (`schedule_policy.py:1516`). Ours: decodes first, prefill
  budget 512 (`scheduler.py:612-694`).
- **Status.** Mechanisms **VERIFIED**; the TTFT-tail effect is **UNVERIFIED**.
  Note the direction is not obvious: chunking mainly protects *decode* latency
  (ITL), and vLLM's own docs say smaller budgets can *worsen* TTFT
  (`docs/configuration/optimization.md:64-65`). The study should expect
  chunked prefill to move ITL tails more than TTFT tails.

### (e) SGLang overlap scheduling hides CPU scheduling overhead

- **Code paths.** `scheduler.py:1941-2002` (overlap loop),
  `overlap_utils.py:87`, `:248` (future token ids), `scheduler.py:4379-4419`.
- **Status.** Mechanism **VERIFIED**; the size of the benefit is
  **UNVERIFIED**. vLLM has an equivalent default-on mechanism (section 2.6), so
  "SGLang hides CPU overhead and vLLM does not" would be wrong. Ours has none
  (`app.py:498`, `scheduler.py:757`).

---

## 7. Implications for the harness

- **Equal KV pool (W4).** vLLM and ours size the pool in 16-token blocks;
  SGLang sizes it in tokens. `--max-total-tokens` can only lower SGLang's pool
  (`kv_cache_configurator.py:2262-2277`).
- **Record resolved defaults.** vLLM's batch limits and SGLang's chunk size,
  graph maximum and memory fraction all depend on GPU memory.
- **Record attention backends.** None of the three uses the same kernel on
  Hopper: ours FlashInfer, vLLM FlashAttention, SGLang FA3.
- **Diagnostic arms.** Consider adding `vllm-noasync` (`--no-async-scheduling`)
  to pair with `sglang-nooverlap`, and note that `vllm-eager` removes compile as
  well as graphs.
- **Counters.** vLLM's prefix-cache counters skip preempted requests'
  re-lookups (F-V4); SGLang's `cache_hit_rate` gauge covers only the last prefill
  batch. Use the counters listed in `ENGINE_FLAGS.md`.

---

## 8. Candidate friction (from source reading, not yet reproduced)

Only real discrepancies are listed, each with both sides cited.

### vLLM 0.31.0

- **F-V1. Default CUDA version.** The GPU install docs say the binaries are
  built with CUDA 12.9 and the default nightly is cu129
  (`docs/getting_started/installation/gpu.cuda.inc.md:39`, `:53-54`). The
  source defaults to CUDA 13.0 (`vllm/envs.py:92`, `:618-619`) and the Dockerfile
  uses 13.0.3 (`docker/Dockerfile:25`).
- **F-V2. Preemption docs describe the removed V0 engine.** The optimization
  guide quotes a V0-style per-preemption warning naming `PreemptionMode.RECOMPUTE`
  and says the default mode is recompute "rather than swap"
  (`docs/configuration/optimization.md:34`, `:47`), implying swap is selectable.
  V1 has no preemption mode or swap at all, and logs only an aggregate
  preemption count (`vllm/v1/metrics/loggers.py:295-297`).
- **F-V3. Chunked-prefill-off behaviour.** The docs warn the server may crash at
  startup when `max_num_batched_tokens` is below `max_model_len`
  (`optimization.md:70-71`). In source, an unset value is silently raised
  (`arg_utils.py:3071-3078`) and a user-set value raises a clear ValueError
  (`vllm/config/scheduler.py:315-327`). Separately, `--no-enable-chunked-prefill`
  on a standard decoder LM logs that disabling it is not officially supported
  and may produce incorrect outputs (`arg_utils.py:2936-2943`) — relevant if
  anyone proposes a "vllm-nochunk" arm.
- **F-V4. Prefix-cache counters silently exclude preempted requests.** When a
  preempted request is looked up again, its queries and hits are kept in
  separate fields that are never exported (`vllm/v1/metrics/stats.py:132-143`;
  `loggers.py:1052-1056`). The metric help text does not mention this
  (`loggers.py:600-603`). Under W4 the exported hit rate will understate reuse.
  This is undocumented behaviour rather than a direct contradiction.
- **F-V5 (minor). `gpu_memory_utilization`.** The source default is 0.92
  (`vllm/config/cache.py:103`); sample outputs in the docs show 0.9
  (`docs/design/metrics.md:385`, `docs/usage/usage_stats.md:36`).

### SGLang 0.5.21

- **F-S1. Memory-fraction fallback.** The server-arguments doc says
  `--mem-fraction-static` falls back to 0.88 when GPU memory is unknown
  (`docs/docs/advanced_features/server_arguments.mdx:440-443`). The source uses
  0.95 (`python/sglang/srt/arg_groups/memory_hook.py:316-321`).
- **F-S2. Docs recommend a deprecated flag.** `--disable-cuda-graph` is a
  deprecated alias that warns on use (`python/sglang/srt/server_args.py:339-347`);
  the replacement is `--cuda-graph-backend-{decode,prefill} disabled`. The docs
  still recommend the old flag, for example in the multi-node tip at
  `server_arguments.mdx:70`.
- **F-S3. Help text says `full` is decode-only, but prefill accepts it.** The
  `--cuda-graph-config` help says so (`python/sglang/srt/arg_groups/fields/exec_.py:482`),
  while the prefill backend's allowed set includes `full`
  (`python/sglang/srt/model_executor/cuda_graph_config.py:61-66`;
  `exec_.py:493-498`). The docs table also omits `full` for prefill
  (`server_arguments.mdx:2422-2425`).
- **F-S4. Stale choice lists in the docs.** The `--schedule-policy` row
  (`server_arguments.mdx:488-491`) omits `hrrn` and `shortest-prefill-first`,
  which the source accepts (`fields/schedule.py:86-96`). The
  `--radix-eviction-policy` row (`server_arguments.mdx:560-563`) omits `tlru`
  (`python/sglang/srt/arg_groups/choices.py:184`).
- **F-S5 (minor, internal).** The default-attention-backend docstring says FA3 is
  chosen on Hopper unless page size is above 1
  (`python/sglang/srt/arg_groups/model_override_base.py:252`), but the code never
  checks page size (`:274-280`).

### Ours (for `docs/xengine/FINDINGS_OURS.md`, per SPEC rule 4 — not fixed here)

- **O-1. Preemption fires while unused cached blocks could be evicted.** The
  preemption trigger compares the step's block need against
  `allocator.num_free` and preempts on a shortfall
  (`serving/scheduler/scheduler.py:854-857`). Blocks held only by the prefix cache
  are not free, and the growth path never asks the cache to evict
  (`serving/memory/block_table.py:79-95`); cache eviction runs only at admission
  (`scheduler.py:511`, `:608`). The cache's own docstring warns that with no
  `max_cached_blocks` set, a server must drive eviction or the cache can starve
  running sequences (`serving/cache/radix.py:274-281`), and the server builds it
  without that budget (`serving/server/app.py:1250-1253`). Expected symptom on
  W4 with the prefix cache on: preemptions (and possibly the oversized-request
  failure at `scheduler.py:859-868`) while `llm_blocks_used` is dominated by
  cache-only blocks. Not reproduced.
- **O-2. No prompt-token usage in streaming.** The request model has no
  `stream_options` field, so `include_usage` is silently ignored
  (`serving/server/app.py:184-210`); usage is only in non-streaming responses
  (`app.py:942-946`). The harness must count prompt tokens itself for `ours`.
- **O-3. No Prometheus eviction counter.** The prefix cache counts evictions
  (`serving/cache/radix.py:544`) and the scheduler's JSON snapshot carries it as
  `cache_evictions` (`serving/scheduler/scheduler.py:1267-1274`), but
  `serving/metrics/prometheus.py` exports no eviction metric (only block hits and
  misses, `prometheus.py:611-616`). The harness should read evictions from the
  snapshot, or record `null`.
