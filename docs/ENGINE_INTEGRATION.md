# Engine Integration Report

**Deliverable #1.** The evidence base every other doc in this repo cites.

Engine under audit: `../llm_inference_engine` @ `6ff40a1` ("Clean up repo for public view: remove dead code, declare a public API"), tag `v0.1.0`.
Audit date: 2026-07-31. Read from source, not from the engine's own documentation.

**Citation convention.** Every engine claim carries `file:line` against the working tree at `6ff40a1`. Paths are relative to the engine repo root unless prefixed `serving:`, which means this repo. **[inference]** marks a judgment not directly readable from code. Where the engine's own docs or this repo's docs cite a line number that has since drifted, both the cited and the actual number are given.

**Rule applied throughout:** where the project owner's stated understanding, the engine's docs, or this repo's `ARCHITECTURE.md` / `PRD.md` disagree with the source, the source wins and the disagreement is stated explicitly.

---

## 0. Executive summary — what this report changes

Eight findings that alter the plan of record. Each is substantiated below.

| # | Finding | Effect |
|---|---|---|
| 1 | **The `v0.1.0` git tag already exists** and points at `6ff40a1`. `PRD.md:105` and `ARCHITECTURE.md:200` both assert "no tags exist." | Tier-0 prerequisite #5 is **already done**. Remove it. |
| 2 | **`bench/results/` is committed — 24 tracked files.** `PRD.md:51,86` rest on `CLAIMS_AUDIT.md:6` ("`git ls-files` returns zero result files"), which is stale as of commit `c24afaf`. | S8's stated baseline is wrong. Rewrite it. |
| 3 | **The engine's editable install is broken and points at a path that no longer exists** (`.../Personal Projects/llm_inference_egine` — note the typo). `import engine` fails from any cwd outside the repo root. | `SERVING_INTERFACE.md:163-175` ("`pip install -e .` succeeds today") does not hold on this machine. Affects the submodule decision and CI. |
| 4 | **All 5 tests in `tests/test_generate.py` fail with `KeyError: 'input_ids'`** — empirically reproduced, not inferred. | Confirms `PRD.md:101`. The KV-cache-vs-no-cache gate is not enforced. |
| 5 | **There is no model-level GPU correctness oracle.** Every bit-exactness claim in the engine is about the **CPU fp32 NumPy** path. | Confirms `PRD.md:104`. This is the single most dangerous inherited gap. |
| 6 | **`bench/bench_attn_kernel.py:69-71` hoists the GQA expansion out of SDPA's timed lambda** while `:66` times the kernel with a `cudaDeviceSynchronize` inside it. The "0.98–0.99× SDPA" comparison is not apples-to-apples in either direction. | The engine acknowledges the sync bias (`BENCHMARKS.md:119`) but **not** the GQA hoist. Do not re-cite the number without this caveat. |
| 7 | **The CUDA-arch flag is at `scripts/build_kernels.sh:10`, not `:9`.** `ARCHITECTURE.md:199`, `PRD.md:141,196` and `SERVING_INTERFACE.md:197,226` all cite `:9`. | Cosmetic, but it is a citation this repo makes four times. |
| 8 | **The Q1 finding holds under line-by-line verification**: with a flattened varlen layout, `linear`, `rms_norm_gpu`, `swiglu_ffn_gpu` and the embedding lookup are **literally unchanged**, and RoPE is **already batch-ready**. Verified per-line in §3.1. | The batched-decode change is contained. It is *smaller* than the paged-KV work, not larger. |

---

## 1. What the engine actually is

Llama 3.2 1B Instruct, implemented twice: once in NumPy fp32 as a correctness reference, once in PyTorch fp16 on `cuda:0` as the performance path, plus one CUDA C++ decode-attention kernel in three staged versions. 16 layers, hidden 2048, 32 query heads, 8 KV heads, head_dim 64, FFN 8192, vocab 128256, tied embeddings.

### 1.1 Module inventory

| Module | Lines | Owns |
|---|---|---|
| `engine/__init__.py` | 69 | Public API declaration, lazy resolution |
| `engine/loader.py` | 158 | safetensors → fp32 numpy / fp16 torch / quantized |
| `engine/components.py` | 241 | NumPy reference: RMSNorm, RoPE, GQA, SwiGLU |
| `engine/components_gpu.py` | 194 | torch fp16 versions + the quant chokepoint + **the attention injection hook** |
| `engine/model.py` | 245 | CPU forward wiring, `forward_debug`, `greedy_decode` |
| `engine/model_gpu.py` | 158 | GPU forward wiring: `prefill`, `decode_step`, `forward_all` |
| `engine/cache.py` | 33 | `KVCache` (numpy) and `KVCacheGPU` (torch) |
| `engine/quant.py` | 134 | int8 per-channel, int4 group-wise, `QuantWeight` |
| `engine/sampler.py` | 76 | greedy / temperature / top-k / top-p |
| `engine/scheduler.py` | 45 | single-request generate loop |
| `engine/server.py` | 105 | FastAPI OpenAI-compatible surface + SSE |
| `engine/cli.py` | 40 | `llm-generate` entry point |
| `kernels/attention_decode.cu` | 311 | v1/v2/v3 decode-attention kernels |
| `kernels/bindings.cpp` | 75 | nanobind module, raw device pointers |
| `kernels/attn_reference.py` | 74 | torch reference + Python wrappers |

**Observation:** `engine/components.py` (241 lines) and `engine/model.py` (245 lines) are the *largest* modules in the engine, and neither is on the GPU serving path. The GPU path — the thing the serving layer will actually drive — is 352 lines total across `model_gpu.py` and `components_gpu.py`. That is the entire surface the serving layer has to reason about.

### 1.2 The declared public API

`engine/__init__.py:22-40` declares `__all__` with exactly 12 names:

```
load_config, load_weights, load_weights_gpu, load_weights_gpu_quant,
LlamaModel, LlamaModelGPU, KVCache, KVCacheGPU,
generate, greedy, get_sampler, QuantWeight
```

`engine/__init__.py:43-56` maps each name to its defining submodule; `:59-65` resolves it on first attribute access via PEP 562 `__getattr__`; `:68-69` defines `__dir__`. `__version__ = "0.1.0"` at `:20`.

The laziness is real and load-bearing: `import engine` executes no `import torch`. `LlamaModel` resolves through `engine.model` → `engine.components` → numpy only. `LlamaModelGPU` (`engine/model_gpu.py:8`) and `QuantWeight` (`engine/quant.py:20`) are the names that pull torch in.

**What `__all__` does *not* contain, and the serving layer needs:**

- `engine.components_gpu.gqa_attention_gpu` — the function the `AttentionBackend` seam must be cut into (`engine/components_gpu.py:100-182`).
- `engine.components_gpu.linear` — the quant chokepoint (`:14-24`).
- `LlamaModelGPU.forward_all` (`engine/model_gpu.py:92-125`) exists but is documented as "not on the hot path" (`:96`).
- `engine.sampler.sample` (`engine/sampler.py:10-55`) — only `greedy` and `get_sampler` are exported.
- Anything kernel-related. `kernels/attn_reference.py` is not importable as `engine.*` at all; it is reached by `sys.path` injection (§1.4).

**[inference]** The seam this repo plans to add (`serving:ARCHITECTURE.md:193-195`) will therefore need `__all__` extended by at least `AttentionBackend` and `BatchMeta`, plus `forward_varlen` as a method. That is a two-line change to `_EXPORTS`, but it is a change to the *declared contract*, so it should ship in the same commit as the tag bump.

### 1.3 The forward pass, precisely

`LlamaModelGPU.__init__` (`engine/model_gpu.py:23-53`) reads seven fields off `config` (`:28-32`), precomputes RoPE cos/sin tables to `max_position_embeddings` (`:34-40`), and optionally installs the CUDA decode kernel as a closure (`:44-53`).

Three entry points, all with identical layer bodies:

- **`prefill(token_ids, kv_cache)`** — `:58-90`. Embeds via integer indexing (`:64`), builds `positions = torch.arange(seq)` (`:65`), runs 16 layers (`:67-85`), calls `kv_cache.advance(seq)` (`:87`), norms **only the last row** `x[-1:]` (`:88`), projects to logits and returns **CPU numpy** (`:89-90`).
- **`decode_step(token_id, kv_cache)`** — `:127-158`. Same, with `positions = torch.tensor([kv_cache.pos])` (`:133`) and `advance(1)` (`:155`).
- **`forward_all(token_ids)`** — `:92-125`. No cache, logits at every position, O(seq²) attention, for perplexity teacher-forcing (`:96-97`).

Per-layer body (`:67-85`), pre-norm residual:

```
h = rms_norm_gpu(x, input_layernorm)        # components_gpu.py:27-31
h = gqa_attention_gpu(h, q,k,v,o, cos,sin, positions, ...)
x = x + h
h = rms_norm_gpu(x, post_attention_layernorm)
h = swiglu_ffn_gpu(h, gate, up, down)       # components_gpu.py:185-194
x = x + h
```

`gqa_attention_gpu` (`engine/components_gpu.py:100-182`) is where everything interesting lives:

| Lines | What |
|---|---|
| `:126` | `seq = x.shape[0]` — **the only place batch shape is inferred** |
| `:128-130` | Q/K/V projections via `linear`, reshaped `(seq, n_heads, head_dim)` |
| `:132-135` | RoPE: `cos_pos = cos[positions]`, applied to q and k |
| `:137-142` | Cache write at `[layer_idx, pos:pos+seq]`, read back `[layer_idx, :pos+seq]` |
| `:147` | `scale = 1/sqrt(head_dim)` |
| `:150-158` | **Custom-kernel branch**, gated on `decode_kernel is not None and seq == 1` |
| `:160-167` | GQA expand via `repeat_interleave`, transpose to head-major |
| `:169` | Scores in **fp32** (`q.float() @ k.float().T`) |
| `:171-174` | Causal mask — **2-D `triu`, only when `seq > 1`** |
| `:176-182` | softmax → `@V` → transpose back → reshape → `o_proj` |

### 1.4 The kernel path

`engine/model_gpu.py:44-53` is the injection hook the whole serving plan hangs off:

```python
self._decode_kernel = None
if use_cuda_attn:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "build"))
    sys.path.insert(0, str(root / "kernels"))
    from attn_reference import attention_decode
    self._decode_kernel = lambda q, k, v, scale: attention_decode(q, k, v, scale, version=ver)
```

Passed down at `:79` (prefill) and `:147` (decode), landing at `engine/components_gpu.py:114` as `decode_kernel=None`. **The hook injects only the math** — signature `(q2d, k, v, scale) -> out2d` (`:150-158`). The engine keeps ownership of the cache read/write at `:137-142`, above the branch. This is exactly what `serving:ARCHITECTURE.md:129` says, and it is correct.

Three kernels in `kernels/attention_decode.cu`:

- **v1** (`:33-93`): one block per head, **one thread per block** (`:91`, `<<<n_heads, 1>>>`). Streaming/online softmax with O(head_dim) state (`:54-77`).
- **v2** (`:105-167`): `head_dim` threads per block, shared-memory tree reduction (`:139-146`).
- **v3** (`:193-311`): split-KV flash-decoding. Grid `(n_heads, n_splits)` (`:302-305`), warp-shuffle dot product (`:186-190`, `:229-231`), second combine kernel (`:256-282`). `CHUNK=256`, `MAX_SPLITS=16` (`:291`). **`cudaMalloc` × 3 and `cudaFree` × 3 on every call** (`:298-300`, `:310`).

The nanobind ABI (`kernels/bindings.cpp:29-31`, `:42-44`, `:55-57`) is:

```
(uintptr_t q, k, v, out, int n_heads, n_kv_heads, kv_seq, head_dim, float scale)
```

Raw device addresses as integers. `:39`, `:52`, `:65` each call `cudaDeviceSynchronize()` — and **discard its return value**.

### 1.5 KV cache

`engine/cache.py` is 33 lines. `KVCache` (`:6-20`) and `KVCacheGPU` (`:23-33`) are structurally identical: a `(n_layers, max_seq, n_kv_heads, head_dim)` dense buffer for K, one for V, and **one integer `pos`** (`:17`, `:30`) shared by all layers. `advance(n)` (`:19-20`, `:32-33`) is the only mutator.

Allocation is per-`generate()`-call: `engine/scheduler.py:26-30` builds a fresh cache with `max_seq` defaulting to 2048 (`:16`) and drops it on return. No free, no refcount, no reuse.

### 1.6 Sampling, scheduling, server

`engine/sampler.py:6-7` is `greedy`. `:10-55` is `sample` — temperature → top-k → softmax → top-p → draw, all in **float64 numpy** (`:23`). `:58-76` is `get_sampler`, which short-circuits to `greedy` when `temp == 0.0` (`:68-69`).

`engine/scheduler.py:11-45` — `generate(model, token_ids, sampler_fn, max_tokens=256, max_seq=2048)`. `sampler_fn` is parameter **`:14`** (this repo and `SERVING_INTERFACE.md` both cite `:15`, which is `max_tokens`). Cache construction branches on `hasattr(model, "make_cache")` (`:26-30`) — the one piece of duck typing already present. Prefill at `:33`, sample at `:34`, decode loop at `:40-45`. It is a Python generator, so abandoning it stops decode at the next `yield`.

`engine/server.py:19-31` loads `LlamaModel` — **the fp32 NumPy CPU model** — from `engine.model` at `:24-30`, with `weights_path = "weights"` hardcoded at `:27`. `:49-105` implements `/v1/chat/completions`; `:69` iterates the blocking synchronous `generate()` inside an `async def` event stream.

### 1.7 Packaging

`pyproject.toml:1-3` is plain setuptools — no `CMakeExtension`, no `build_ext`. `:34-36` packages only `engine*`, so `kernels/`, `tests/`, `bench/` are unpackaged. The CUDA module is built by `scripts/build_kernels.sh`, which hardcodes `-DCMAKE_CUDA_ARCHITECTURES=80` at **line 10**. `.gitignore:16` ignores `build/`, where the `.so` lands. `kernels/CMakeLists.txt:11-17` fetches nanobind `v2.1.0` from GitHub at configure time.

---

## 2. Corrections to the stated understanding

The project owner's five stated beliefs, checked against source.

### (a) "Public API declared via `__all__` with lazy imports" — **TRUE, and newer than the audit that denies it**

Verified: `engine/__init__.py:22-40` (`__all__`, 12 names), `:43-56` (`_EXPORTS`), `:59-65` (PEP 562 `__getattr__`), `:20` (`__version__`).

**The correction is to the engine's own audit, not to the owner.** `SERVING_INTERFACE.md:206` states:

> **There isn't one.** `engine/__init__.py` is **0 bytes**. Nothing is exported, nothing is marked private, there is no `__all__`, no `version`.

That is false as of `6ff40a1`, whose commit message says: *"engine/`__init__.py` was empty, so any consumer had to import from internal module paths. It now declares `__all__` with the 12 supported names and resolves them lazily (PEP 562)."* `SERVING_INTERFACE.md:3` dates itself 2026-07-30; the commit landed 2026-07-30 22:28. The audit predates it by hours.

**Consequences for this repo:**
- `SERVING_INTERFACE.md:223` (item 1, "Populate `engine/__init__.py`", 1h) is **done**.
- `SERVING_INTERFACE.md:292` (rank 3, "Public API in `engine/__init__.py` (§2c items 1-2)", 2.5h, marked STRICTLY PREREQUISITE) is **half done** — item 1 is complete, item 2 (moving `kernels/attn_reference.py` → `engine/kernel_api.py` and deleting the `sys.path` hack at `model_gpu.py:45-53`) is **not**. Re-estimate at ~1.5h, not 2.5h.
- Note also that `SERVING_INTERFACE.md` and `CLAIMS_AUDIT.md` are themselves gitignored (`.gitignore:47-48`), so they are not versioned alongside the code they audit. **[inference]** They will go stale again. Cite them only with a date and a commit.

**A caveat the owner's belief omits.** The laziness is a property of `engine/__init__.py` only. It does not extend to `engine.scheduler`, which imports `from engine.model import EOS_IDS` at `engine/scheduler.py:8` — so `from engine import generate` transitively imports the **CPU** model module even when the caller only ever uses `LlamaModelGPU`. `EOS_IDS` is defined twice with identical contents (`engine/model.py:16`, `engine/model_gpu.py:19`); the scheduler uses the CPU copy for both paths (`engine/scheduler.py:36,44`). Harmless today; a trap if the two ever diverge.

### (b) "<1e-3 logit error with exact greedy match" — **TRUE for the CPU fp32 path only. There is no GPU equivalent.**

What is actually gated:

| Gate | Location | Model under test |
|---|---|---|
| Logits vs HF oracle, atol 1e-3, short prompt | `tests/test_forward.py:104-115` | `LlamaModel` (CPU fp32) |
| Logits vs HF oracle, atol 1e-3, medium prompt | `tests/test_forward.py:128-139` | `LlamaModel` (CPU fp32) |
| Argmax exact at every position | `tests/test_forward.py:89-102`, `:117-126` | `LlamaModel` (CPU fp32) |
| 32 greedy tokens bit-identical to HF | `tests/test_decode.py:15-37`, `:39-48` | `LlamaModel` (CPU fp32) |
| Per-layer post-attn / post-ffn, atol 5e-3 | `tests/test_forward.py:28-52`, `:54-74` | `LlamaModel` (CPU fp32) |

The `model` fixture is `LlamaModel(weights, config)` at `tests/conftest.py:27-29`, built from `load_weights` (`:22-24`), which is the fp32 numpy loader (`engine/loader.py:42-79`).

**What the GPU path is actually gated on:**

- `tests/test_gpu_model.py:42-45` — `logits.shape == (128256,)`, `np.all(np.isfinite(logits))`, `0 <= argmax < 128256`. That is a **liveness check, not a correctness check**. A model with a transposed weight would pass it.
- `tests/test_gpu_model.py:51-62` — 5 generated tokens are in vocab range. Same category.
- `tests/test_components_gpu.py:154-185` — `gqa_attention_gpu` vs the NumPy `gqa_attention`, `atol=1e-2`, on **synthetic random tensors at toy dimensions** (`seq=6, H=32, NH=4, NKV=2, HD=8`, `:161`) with **no KV cache** and a non-Llama `theta=10000.0` (`:170`).
- `tests/test_attention_kernel.py:124-144` — greedy tokens identical with the CUDA kernel **on vs off**. This compares the GPU path to *itself*, not to a reference.

**So: no test anywhere compares `LlamaModelGPU` output to HF or to `LlamaModel`.** This confirms `serving:PRD.md:104` and `serving:ARCHITECTURE.md:198`, and it is the most consequential single fact in this report. The engine's own `BENCHMARKS.md:223-225` lists "Final logits vs fp32 HuggingFace oracle", "Argmax vs oracle, every position", "32 greedy tokens vs HF" in a table headed "Correctness gates" **without stating that all three are CPU-path gates**. A reader of `BENCHMARKS.md` would reasonably conclude the fp16 GPU path is validated to 1e-3. It is not.

**Additional finding — the cache gate is not running at all.** `tests/test_generate.py` reads `oracle_short["input_ids"]` / `oracle_medium["input_ids"]` at `:42`, `:52`, `:63`, `:78`, `:89`. `tests/oracle.py:170` writes the key as `captures["token_ids"] = tok_ids`. There is no `"input_ids"` key anywhere in the fixture. Reproduced empirically on this machine:

```
$ python3 -m pytest tests/test_generate.py -m "not slow" -q
FF
E   KeyError: 'input_ids'   tests/test_generate.py:78
E   KeyError: 'input_ids'   tests/test_generate.py:89
2 failed, 3 deselected in 10.26s
```

Two of the five are unmarked, so they fail in the *default fast* suite (`pytest -m "not slow"`); the other three (`:40`, `:50`, `:60`) are `@pytest.mark.slow` and fail whenever the slow suite runs. **The KV-cache-vs-no-cache identity gate — the Phase 2 correctness milestone per the module docstring at `tests/test_generate.py:1-3` — has never executed.** `BENCHMARKS.md:226` nonetheless lists "KV-cache generation vs no-cache generation — **identical tokens**" as a passing gate.

Note the fixtures do exist locally (`tests/fixtures/oracle_short.pkl`, `oracle_medium.pkl`, dated Jun 4) but are gitignored (`.gitignore:24`), so the failure mode on a clean clone is `FileNotFoundError` from `tests/oracle.py:198-200`, not `KeyError`. **[inference]** That is why this survived: on any machine without regenerated fixtures the whole file errors out for a different, more obviously-benign reason.

### (c) "Custom kernel matches SDPA at 512–2048 and loses at 128" — **The numbers are as stated. The measurement they come from is not clean, and one of its two defects is undisclosed.**

Reported (`BENCHMARKS.md:87-90`, `README.md:111-114`): v3 vs SDPA 0.61× at kv_seq 128, 0.98× at 512, 0.98× at 1024, 0.99× at 2048. Flatness of v3 (189/189/191 µs) is real and is the evidence that split-KV fixed occupancy.

Two asymmetries in `bench/bench_attn_kernel.py`:

**Disclosed — the sync.** `_time_cuda` (`:32-44`) wraps N=200 iterations in CUDA events. The kernel lambda at `:66` calls `attention_decode`, which reaches `kernels/bindings.cpp:65` where `cudaDeviceSynchronize()` blocks per call. SDPA's lambda at `:72` queues asynchronously. So kernel iterations are serialized with full launch latency each time while SDPA's 200 launches pipeline. `BENCHMARKS.md:119` names this and correctly notes it biases *against* the custom kernel — the numbers are conservative in this direction. (`BENCHMARKS.md:119` cites `kernels/bindings.cpp:67,79,91`; the actual lines are **`:39,52,65`** — the file is only 75 lines. `serving:ARCHITECTURE.md:343` cites it correctly.)

**Undisclosed — the GQA hoist.** `bench/bench_attn_kernel.py:69-71`:

```python
q_s = q.unsqueeze(1)                          # line 69
k_s = k.repeat_interleave(GROUPS, dim=0)      # line 70  — 8 -> 32 heads
v_s = v.repeat_interleave(GROUPS, dim=0)      # line 71
t["sdpa"] = _time_cuda(lambda: _sdpa(q_s, k_s, v_s))   # line 72
```

The 4× expansion of K and V from 8 KV heads to 32 query heads is done **once, outside the timed region**. The custom kernel performs the GQA mapping internally, for free, by index arithmetic (`kernels/attention_decode.cu:44`, `:117`, `:208`: `kv_head = h / groups`). So SDPA is credited with a materialization that a real decode step would have to pay every layer every token — at kv_seq=2048 that is `2 × 32 × 2048 × 64 × 2 B ≈ 16.8 MB` of expansion traffic per layer per token, excluded from its measured latency.

This biases *for* SDPA and against the kernel, so it compounds with the sync in the same direction — both make the kernel look worse. **The "0.98–0.99×" claim is therefore conservative, not inflated.** But it is not a clean comparison and the engine's methodology section (`BENCHMARKS.md:114-120`) does not mention it. Also unmentioned: `kernels/attn_reference.py:62` allocates a fresh output tensor inside every timed kernel call, and `:46` calls `.contiguous()` on all three inputs (a no-op here since the bench passes contiguous tensors, but a real cost on the model path).

**The corrected statement:** *v3 measures 0.98–0.99× of SDPA at 512–2048 under a methodology that disadvantages v3 on two independent axes (per-call device sync; SDPA's GQA expansion hoisted out of the timed region) and advantages it on none. The true ratio is at least that good and is not measured.*

**The 0.61× at kv_seq=128 is real and the cause is correctly identified** (`BENCHMARKS.md:94`): two launches plus a `cudaMalloc`/`cudaFree` triple (`kernels/attention_decode.cu:298-300`, `:310`) unamortized at low work. (`BENCHMARKS.md:120` cites `:297-300, 311`; actual `:298-300, 310`.)

**And the end-to-end number is the one that matters for this repo:** +4.2% (59.5 → 62.0 tok/s, `BENCHMARKS.md:139-141`). §4 of `BENCHMARKS.md` attributes it to Amdahl plus the transpose. See Q3.

### (d) "int8/int4 quantization" — **TRUE, with three qualifications the belief omits**

int8 per-channel: `engine/quant.py:29-43`. Symmetric absmax over each output row, scale rounded through fp16 before quantizing (`:41-42`) so quant/dequant agree. int4 group-wise: `:55-82`, `group_size=128` default, symmetric to `[-7,7]`, two nibbles packed per byte (`:78-81`). `QuantWeight` (`:110-134`) holds `q`, `scale`, `mode`, `group_size`, with `nbytes()` at `:131-134` for the memory benchmark.

Qualifications:

1. **It is weight-only, dequantize-on-the-fly.** `engine/components_gpu.py:22-24`: `if isinstance(w, QuantWeight): w = w.dequantize()` then `x @ w.T`. The fp16 tile is reconstructed at every matmul. This *costs* throughput (~79 → ~45 → ~22 tok/s, `BENCHMARKS.md:162-163`). There is no low-precision GEMM.
2. **Only 7 tensors per layer are quantized** (`engine/loader.py:111-119`): q/k/v/o proj and gate/up/down. Embedding, tied `lm_head`, and all RMSNorm weights stay fp16 (`:130-134`). At 128k vocab the embedding is 525 MB of the int8 total, which is why the reduction is 39% and not 50%.
3. **The KV cache is not quantized.** `engine/cache.py:28-29` is unconditionally fp16. **This matters directly to this repo**: the serving layer's block-pool sizing (`serving:ARCHITECTURE.md:216`) assumes fp16 KV, correctly — but "the engine supports quantization" must not be read as "KV cache quantization is available." It is not, and adding it is a change to `KVCacheGPU`, the paged pool layout, and every attention backend simultaneously.

### (e) "Server is a single-request reference path running the CPU model" — **TRUE and understated**

`engine/server.py:24-30` imports `load_config`, `load_weights` (fp32 numpy) and `LlamaModel`. There is no flag, env var, or branch reaching `LlamaModelGPU`. Every A100 number in `README.md` is unreachable over HTTP. `README.md:144` discloses this.

Understated in two ways:

1. **It does not merely lack concurrency — it actively serializes.** `engine/server.py:69` drives the blocking synchronous generator `generate()` from inside an `async def` event-stream coroutine. The event loop is held for the entire generation. Request 2's TTFT includes request 1's complete decode. This is a structural bug, not an absent feature.
2. **The weights path is hardcoded** to the relative string `"weights"` (`:27`), resolved against process CWD.

`tests/test_server.py:22-50` does spin a real uvicorn subprocess and exercise both streaming and non-streaming paths (`:53-64`, `:67-87`), so the protocol surface is genuinely tested — single-request, 120 s timeouts, `max_tokens: 5` (`:19`).

---

## 3. The four questions

### Q1 — Batched decode: what changes, is it contained, whose repo, and is it bigger than paged KV?

**Answer: with a flattened varlen token axis, five things change and nothing else does. It is contained. It belongs in the engine. It is *smaller* than the paged-KV work.**

#### 3.1 Line-by-line verification of what is unchanged

The claim under test (`serving:ARCHITECTURE.md:149-159`) is that under a varlen layout — all sequences packed along one token axis, `x` shape `(tokens, hidden)` where `tokens = sum(query_lens)` — the position-independent operators need no modification. Verified operator by operator:

**`linear` — UNCHANGED.** `engine/components_gpu.py:14-24`. Body is `x @ w.T` (with a `QuantWeight` dequant branch at `:22-23`). `w` is `(out, in)`; `x` is `(*, in)`. `torch.matmul` broadcasts over all leading dimensions. `(tokens, hidden) @ (hidden, out)` is the same operation as `(seq, hidden) @ (hidden, out)` with a different leading extent. **Zero changes.**

**`rms_norm_gpu` — UNCHANGED.** `engine/components_gpu.py:27-31`. `x.float().pow(2).mean(dim=-1, keepdim=True)` reduces over the **last** axis only; `weight` is `(hidden,)` and broadcasts. Row-independent by construction. **Zero changes.**

**`swiglu_ffn_gpu` — UNCHANGED.** `engine/components_gpu.py:185-194`. Three `linear` calls and an elementwise `F.silu(gate) * up`. All row-independent. **Zero changes.**

**Embedding lookup — UNCHANGED.** `engine/model_gpu.py:64`: `w["model.embed_tokens.weight"][ids_t]`. Advanced integer indexing on a 1-D index tensor. `ids_t` of length `tokens` returns `(tokens, hidden)`. **Zero changes** — though the *construction* of `ids_t` moves to the caller (see below).

**RoPE — ALREADY BATCH-READY.** This is the load-bearing one and it holds. `engine/components_gpu.py:132-135`:

```python
cos_pos = cos[positions]   # (seq, head_dim)
sin_pos = sin[positions]
q = apply_rope_gpu(q, cos_pos, sin_pos)
k = apply_rope_gpu(k, cos_pos, sin_pos)
```

`positions` is a **passed-in tensor parameter** (`:108`), not derived from `seq`. `cos` is the full `(max_position_embeddings, head_dim)` table (`:81-82`), and `cos[positions]` is a gather. Nothing constrains `positions` to be contiguous, monotonic, or starting at zero. A varlen batch whose `positions` is `[0,1,2,3, 57, 112]` — one 4-token prefill chunk plus two decodes at different sequence offsets — gathers correctly with **no code change at all**.

`apply_rope_gpu` (`:86-97`) then does `x_rot = cat([-x[..., d//2:], x[..., :d//2]])` and `x * cos[:,None,:] + x_rot * sin[:,None,:]`. The `[:, None, :]` broadcasts `(tokens, head_dim)` against `(tokens, n_heads, head_dim)`. Row-independent. **Zero changes.**

This is the strongest single argument that varlen is the right layout: the engine's author, building for batch 1, happened to parameterize positions as a tensor rather than computing `arange` inside the RoPE function. That one decision is what makes batched decode a contained change instead of a rewrite.

**Residual adds — UNCHANGED.** `engine/model_gpu.py:81`, `:85` (and `:149`, `:153`): elementwise `x = x + h`.

#### 3.2 What must change — the complete list, five items

**1. Attention core.** `engine/components_gpu.py:160-182`. `q.transpose(0,1)` → `(n_heads, seq, head_dim)` and `matmul` against the full `k_full` assumes **one** sequence attending to **one** contiguous history. Under varlen, token *i* attends only to its own sequence's KV range. Replaced wholesale by the backend's `attend(q, layer_idx, scale, meta)`.

**2. Cache write.** `engine/components_gpu.py:137-142`:

```python
read_len = kv_cache.pos + seq
kv_cache.k[layer_idx, kv_cache.pos:read_len] = k
```

Two assumptions, both fatal for batching: the destination is a **contiguous slice**, and it starts at a **single scalar `pos`** shared by everything in the call. Replaced by a scatter through `meta.slot_mapping`.

**3. Causal mask.** `engine/components_gpu.py:171-174`:

```python
if seq > 1:
    offset = kv_seq - seq
    mask = torch.triu(torch.full((seq, kv_seq), -inf), diagonal=offset + 1)
    scores = scores + mask.unsqueeze(0)
```

A 2-D `triu` with a single global `offset`. Under varlen, each sequence has its own `(query_len, kv_len)` relationship — one triangular mask per sequence, at different offsets, on a shared score matrix. Expressed via `cu_query_lens` / `kv_lens`; in practice it is the backend's job (`serving:ARCHITECTURE.md:126` correctly assigns causality to the backend).

Note the `if seq > 1` guard: under varlen, `seq` is `tokens`, which is `> 1` even for a pure-decode batch of 3. **The existing branch condition becomes actively wrong** — a batch of three independent decodes would take the masked path with a nonsense `offset`. This is a live correctness trap, not a missing feature.

**4. Last-token gather.** `engine/model_gpu.py:88`: `last = rms_norm_gpu(x[-1:], ...)`. With a ragged batch, "the last token of each sequence" is at `cu_query_lens[i+1] - 1`, a gather, not a slice. Replaced by `meta.last_token_ix`.

**5. Position construction.** `engine/model_gpu.py:65` (`torch.arange(seq)`) and `:133` (`torch.tensor([kv_cache.pos])`). Both must move to the caller, which is the only party that knows each sequence's true offset. **This is the same line as the chunked-prefill bug** (§4, item P3) — `arange(seq)` starts at 0 regardless of `kv_cache.pos`, so a second prefill chunk gets positions `0..n` instead of `pos..pos+n`. Invisible today because `engine/scheduler.py:33` calls `prefill` exactly once per generation. Batching and radix prefix caching both require this fixed; fixing it *is* moving position construction to the caller.

#### 3.3 Is it contained?

**Yes, and the containment is measurable.** Of the ~352 lines in the GPU path (`model_gpu.py` 158 + `components_gpu.py` 194):

- Unchanged: `linear` (11 lines), `rms_norm_gpu` (5), the Llama3 inv-freq helper (26), `precompute_rope_tables_gpu` (22), `apply_rope_gpu` (12), `swiglu_ffn_gpu` (10), Q/K/V projections (3), RoPE application (4), residuals (4), the whole `forward_all` path (34). **≈131 lines untouched.**
- Changed or replaced: `components_gpu.py:137-182` (46 lines, of which `:150-158` is the existing kernel branch that stays), `model_gpu.py:63-65` and `:88` and `:131-133` (~7 lines).

**Every batch-sensitive line sits inside attention, cache addressing, positions, and the final gather.** They are the same lines the paged cache touches. `serving:ARCHITECTURE.md:161` states this and it survives verification: *"the batched forward and the paged cache are the same change to the same surface, not two projects."*

#### 3.4 Engine or serving layer?

**Engine.** Non-negotiable, for a reason that is structural rather than stylistic: the changed lines are *inside the layer loop*. `engine/model_gpu.py:67-85` interleaves attention with RMSNorm, SwiGLU, and residuals. There is no seam between "attention" and "the rest of the layer" that a serving layer could reach through from outside. To batch from the serving repo you would have to reimplement `prefill`/`decode_step`, i.e. fork `model_gpu.py`. `serving:ARCHITECTURE.md:450` (alternative A4) reaches the same conclusion by the same argument, and it is correct.

The *implementations behind* the seam — `PagedTorchBackend`, FlashInfer glue, block allocator, batch assembly — belong in the serving repo. The seam itself, `forward_varlen`, and `BatchMeta` belong in the engine.

#### 3.5 Is it bigger than the paged KV work?

**No. It is smaller, and this contradicts the sizing that this repo currently carries.**

The engine's own audit says batched forward is *"the single largest gap. ~15-25 hours"* (`SERVING_INTERFACE.md:250`) and separately says the `AttentionBackend` seam costs *"~4-6 hours"* (`:71`). `serving:ARCHITECTURE.md:446` reuses the 15-25h figure for "put batching and paging in the engine."

That 15-25h estimate is for **adding a batch axis to the whole GPU path** — the padded `(batch, seq, hidden)` design. Under varlen it does not apply, because §3.1 shows the whole GPU path outside attention needs **zero** changes. `serving:ARCHITECTURE.md:452` (alternative A5) already rejects padding in favor of varlen; the hour estimate was not updated to match.

**[inference]** My sizing, once the `AttentionBackend` seam exists:

| Work | Hours | Basis |
|---|---|---|
| `forward_varlen` on `LlamaModelGPU` | 2–3 | A copy of `prefill` (`model_gpu.py:58-90`, 33 lines) with `positions` and `ids_t` taken as arguments and `x[-1:]` replaced by an `index_select` on `meta.last_token_ix` |
| `BatchMeta` construction + CPU-side slot math | 3–4 | Serving-side; `serving:ARCHITECTURE.md:110` specifies the arithmetic |
| Varlen causal masking inside `PagedTorchBackend` | 4–6 | The genuinely fiddly part; per-sequence triangular masks over a shared score matrix |
| Correctness: varlen batch of N vs N sequential runs, bit-identical | 3–4 | Cannot be done at all until P4 (GPU oracle) exists |

**≈12–17 hours, of which only 2–3 are engine-side.** The paged block allocator, refcounting, copy-on-write, and radix cache (`serving:ARCHITECTURE.md:226-243`) are materially more work than that.

**The recommendation this changes:** `SERVING_INTERFACE.md:258-265` advises deferring batched forward to Phase 2 and shipping paged KV + radix cache + routing in Phase 1. Its stated reason is the 15-25h cost. Under varlen that reason evaporates. `serving:ARCHITECTURE.md:143` already pushes back — *"right about sequencing the work, wrong about sequencing the interface"* — and should be strengthened: it is also wrong about the sequencing of the work, because the varlen batched forward is a 2–3 hour engine change that the paged backend needs anyway (the backend's `attend` must handle `n_seqs > 1` regardless, or it will be rewritten).

### Q2 — Where contiguity is assumed: Python vs CUDA vs the nanobind boundary

Three layers, three completely different severities.

#### Python layer — assumed, but shallowly. Fixable without touching C++ at all.

`engine/components_gpu.py:138-142`:

```python
read_len = kv_cache.pos + seq
kv_cache.k[layer_idx, kv_cache.pos:read_len] = k
kv_cache.v[layer_idx, kv_cache.pos:read_len] = v
k_full = kv_cache.k[layer_idx, :read_len]
v_full = kv_cache.v[layer_idx, :read_len]
```

Three distinct assumptions, worth separating because they fail differently:

1. The write destination is a **contiguous slice** `[pos, pos+seq)`.
2. The read source is a **prefix starting at index 0** — `[:read_len]`. This is the one that makes multi-sequence sharing impossible, and it is *stronger* than contiguity.
3. `pos` is a **single scalar** for the whole call (`engine/cache.py:17,30`), shared across all 16 layers and any sequence in flight.

All three are Python-level tensor indexing. Replacing them with a `slot_mapping` scatter and a `block_tables` gather is a pure-PyTorch change inside the backend, and the engine's side of it is deleting five lines and calling two methods.

Everything downstream of `:142` consumes `k_full`/`v_full` as plain tensors — `repeat_interleave` at `:162-163`, `transpose` at `:166-167`, `matmul` at `:169`. None of those care where the tensor came from. **[inference]** A `PagedTorchBackend` that gathers blocks into a temporary `(n_seqs, max_kv, n_kv_heads, head_dim)` tensor would work with the existing arithmetic unchanged; the performance-motivated version avoids the gather, but correctness does not require it.

#### CUDA kernel — assumed absolutely, by inline pointer arithmetic

`kernels/attention_decode.cu:29-32` declares the contract in a comment:

```
//   Q:   [n_heads, head_dim]
//   K,V: [n_kv_heads, kv_seq, head_dim]
//   out: [n_heads, head_dim]
```

and every access recomputes strides inline:

- `:46-47` — `K + (size_t)kv_head * kv_seq * head_dim`. Three-dimensional row-major stride, computed from the scalar `kv_seq`.
- `:61-62` — `Kh + (size_t)j * head_dim`. Token *j* is assumed to sit exactly `head_dim` elements after token *j−1*.
- `:210-211`, `:225-226` — identical in v3.
- `:213-214` — `j_start = s * chunk; j_end = min(j_start + chunk, kv_seq)`. The split-KV chunking assumes **token index maps linearly to memory offset**. This is the deepest assumption: paging would require the chunk boundaries themselves to be block-aware.

There is also no batch axis anywhere. `:41` is `int h = blockIdx.x` mapped directly to a query head; `:45` is `Q + h * head_dim`. There is no dimension in which a second sequence could live.

#### nanobind boundary — the hard wall, and the reason the kernel is out of scope

`kernels/bindings.cpp:29-31` (and identically `:42-44`, `:55-57`):

```cpp
void attention_decode_v1(
    uintptr_t q_ptr, uintptr_t k_ptr, uintptr_t v_ptr, uintptr_t out_ptr,
    int n_heads, int n_kv_heads, int kv_seq, int head_dim, float scale)
```

**No stride arguments. No block-table pointer. No block-size argument. No batch count. No per-sequence length array.** A paged cache is not merely inconvenient to pass across this ABI — it is **inexpressible**. Four raw addresses and five scalars cannot describe a block table.

This is a genuine design decision with an upside worth naming: because the bindings take `uintptr_t` rather than torch tensor types, the extension has **no torch C++ ABI dependency** (`SERVING_INTERFACE.md:199` credits this correctly). The `.so` survives torch upgrades. The cost is that the ABI carries no structure, so extending it means changing every signature, every launcher, and re-running the 100-input gate (`tests/test_attention_kernel.py:72-85`).

#### What a pluggable interface looks like

The hook already exists and was built for a different purpose. `engine/components_gpu.py:114` takes `decode_kernel=None`; `engine/model_gpu.py:44-53` constructs it; `:79` and `:147` pass it down. Today it injects only the *math*, at `engine/components_gpu.py:150-158`, while the engine retains cache ownership at `:137-142`. The change is to widen the injected object's responsibility by one notch so it owns **both**, and to make its signature varlen from the start:

```python
class AttentionBackend(Protocol):
    def append_kv(self, layer_idx: int, k, v, meta: BatchMeta) -> None: ...
    def attend(self, q, layer_idx: int, scale: float, meta: BatchMeta) -> Tensor: ...
```

`q` is `(tokens, n_heads, head_dim)`; the return is the same shape; causality is the backend's responsibility (because only the backend knows the block tables and per-sequence KV lengths). This matches `serving:ARCHITECTURE.md:120-126` and I concur with it, including the insistence at `:143` that the protocol be varlen from day one even while Phase 1 passes `n_seqs == 1`.

The engine-side edit is bounded: in `gqa_attention_gpu`, insert a branch after the projections and RoPE (`engine/components_gpu.py:135`) that calls `backend.append_kv(...)`, `backend.attend(...)`, then `linear(out.reshape(tokens, n_heads * head_dim), o_w)`. Lines `:137-182` stay exactly as they are, below the branch, serving the reference path. `KVCacheGPU` (`engine/cache.py:23-33`) is not touched.

**Cost of the abstraction at batch 1 — bounded from source, not guessed.** The branch already runs: `if decode_kernel is not None and seq == 1` at `engine/components_gpu.py:150` executes on every layer of every token today. Adding a backend branch replaces an inline slice with two Python method calls, 16 layers deep, so 32 extra Python dispatches per token against a decode step of ~12.7 ms at 79 tok/s (`BENCHMARKS.md:29`). **[inference]** Well under 0.1%. For scale, the transpose already paid on the kernel path (`engine/components_gpu.py:153-154`) is orders of magnitude larger. This is not a close call.

#### Engine seam vs serving-layer-owns-attention

**Engine seam. The bypass is not viable and the reason is concrete, not aesthetic.**

To own attention from the serving repo you must own the layer loop, because attention is interleaved with RMSNorm/SwiGLU/residuals inside `engine/model_gpu.py:67-85`. Owning the layer loop means copying `model_gpu.py` into this repo. That is a fork with extra steps: the engine's correctness gates stop covering the code you actually run, and every engine bugfix needs manual replay.

Three further arguments against paging the CUDA kernel itself, all verifiable:

1. **It is batch-1 by construction.** `Q` is `[n_heads, head_dim]` (`kernels/attention_decode.cu:30`); `:45` indexes `Q + h * head_dim` with no room for a batch stride. A paged *batch-1* kernel is a contradiction — paging exists to serve many sequences.
2. **The ABI forbids describing a paged cache** (`kernels/bindings.cpp:29-31`), as above.
3. **It puts the engine's best claim at risk.** The 0.98–0.99× SDPA result (`BENCHMARKS.md:102`) is measured against the current kernel. Rewriting it invalidates the measurement and forces a full re-benchmark plus re-validation of `tests/test_attention_kernel.py:72-85`.

**The honest cost of this choice, which must be stated before anyone asks:** on the serving layer's paged path, the custom CUDA kernel **is not in use**. It becomes the single-replica, contiguous-cache reference path. `serving:PRD.md:156` (C3) already says this. Do not let any claim set imply the CUDA kernel serves paged traffic.

### Q3 — The transpose problem

#### What it costs

The engine stores the cache as `(n_layers, max_seq, n_kv_heads, head_dim)` — sequence-major (`engine/cache.py:28`). The kernel demands `(n_kv_heads, kv_seq, head_dim)` — head-major (`kernels/attention_decode.cu:31`). So on every kernel-path call, `engine/components_gpu.py:153-154`:

```python
kt = k_full.transpose(0, 1).contiguous()
vt = v_full.transpose(0, 1).contiguous()
```

`.transpose` is a view; `.contiguous()` **materializes a full copy of the entire K and V history** — every layer, every decode step. Note `k_full` at `:141` is `kv_cache.k[layer_idx, :read_len]`, i.e. the whole cache prefix, not just the new token.

The magnitude, recomputed from config rather than taken on faith: at `kv_seq = 2048`, per layer, `2 (K and V) × 2048 × 8 kv_heads × 64 head_dim × 2 bytes = 4.19 MB`. Across 16 layers: **67.1 MB of copy traffic per decoded token**, read + written, that the PyTorch attention path never pays. `BENCHMARKS.md:149` states 67 MB and the arithmetic checks out.

This is one of two named causes of the kernel's disappointing +4.2% end-to-end (`BENCHMARKS.md:139-141`); the other is Amdahl (attention is a small fraction of a weight-bandwidth-bound decode step). **`BENCHMARKS.md:151` states the split between the two causes has not been measured.** So the transpose's contribution to the shortfall is unquantified — an important limit on how confidently anyone can claim "paging fixes it."

#### What block layout the serving layer should use

`[num_blocks, block_size, n_kv_heads, head_dim]` per layer, fp16, `block_size = 16` to start and swept as a tunable. This matches `serving:ARCHITECTURE.md:211-214` and I concur.

The reasoning, stated in terms of the engine's actual constraints:

- **The innermost two axes `(n_kv_heads, head_dim)` match what the engine already produces.** `engine/components_gpu.py:129-130` reshapes K and V to `(seq, n_kv_heads, head_dim)`. A token's KV slice is contiguous in this layout, so `append_kv` is a single scatter of `(tokens, n_kv_heads, head_dim)` into `slot_mapping` positions — no transpose, no reshape.
- **Block-contiguity is what paged kernels expect**, and it keeps a block's worth of tokens in one cache-friendly run.
- Per-block bytes across all layers: `2 × 16 layers × 16 tokens × 8 kv_heads × 64 head_dim × 2 B = 512 KB`, matching `serving:ARCHITECTURE.md:216`.

**One caveat to carry forward.** `serving:ARCHITECTURE.md:464` states FlashInfer's tensor-layout contract is **unverified** — not installed, not read. This layout is asserted as reasonable, not as FlashInfer-compatible. Since the whole point of the `AttentionBackend` seam is that a layout mismatch costs an adapter rather than a redesign, that is an acceptable risk — but the `PagedTorchBackend` must be written first, precisely because it is layout-independent.

#### Does paging resolve or worsen the transpose?

**It resolves it, and the framing matters: paging does not make the transpose cheaper, it makes the transpose unnecessary.**

The transpose is not a paging problem, and paging does not inherit it. It exists solely because the engine's storage layout and the custom kernel's required layout disagree, and the kernel's ABI (`kernels/bindings.cpp:29-31`) offers no stride arguments through which the mismatch could be described instead of copied. Both halves of that mismatch disappear on the paged path:

1. **The paged backend is not the custom kernel.** Per Q2, the `.cu` file is frozen and out of the paged path. Its layout demand is out of scope with it.
2. **A paged attention implementation reads through `block_tables`.** Either it gathers only the blocks a sequence actually needs (bounded by that sequence's real length, not `max_seq`), or — with FlashInfer — it reads the pool directly with no gather at all. Nothing copies the whole history.
3. **The new layout is already what the engine produces.** `(n_kv_heads, head_dim)` innermost means the write path needs no permutation either.

Could paging *worsen* it? The failure mode to watch is a naive `PagedTorchBackend` that gathers blocks into a dense `(n_seqs, max_kv_len, n_kv_heads, head_dim)` tensor per layer per step. That reintroduces a whole-history copy, and with padding to the batch's longest sequence it can be *worse* than 67 MB under length skew. **[inference]** Since `PagedTorchBackend` is explicitly scoped as the correctness oracle (`serving:PRD.md:158`), a gather-based first implementation is acceptable — but it must be benchmarked separately from FlashInfer and must never be the number quoted for the paged path.

`serving:PRD.md:142` currently lists "store the KV cache in kernel layout to isolate the transpose cost" as a nice-to-have superseded by paging. **I recommend deleting it rather than deferring it.** Its only value was quantifying the unmeasured split at `BENCHMARKS.md:151`, and once the custom kernel is off the serving path that number stops informing any decision this repo makes.

### Q4 — The kernel decision: write vs consume

**Recommendation: consume. Do not write a paged-attention CUDA kernel. Ship two implementations behind one interface — a PyTorch block-gather reference written by the author, and FlashInfer as the fast path.**

#### The case for writing one

It is not empty and should be stated fairly. The engine owns the KV cache, so whatever owns cache layout arguably belongs there. A paged split-KV kernel with block tables is close to what production engines actually run, and it would be a genuinely strong artifact. It preserves one story: "I built the engine and made it serve."

#### The case for consuming, which I find decisive

**1. It is a rewrite, not an extension.** The three obstacles are independent and each is sufficient. (a) `kernels/bindings.cpp:29-31` has no stride, block-table, or block-size parameters — the ABI cannot express a paged cache. (b) `kernels/attention_decode.cu:30` has no batch axis and `:45` has no room for one; a paged *batch-1* kernel is self-contradictory. (c) The split-KV chunking at `:213-214` assumes token index maps linearly to memory offset, which is precisely what paging breaks. Fixing (c) means rewriting the part of v3 that produced the flat-in-sequence-length result the engine's headline rests on.

**2. The cost is 15–25 hours of pointer-arithmetic debugging in a deprioritized lane**, plus re-validating the 100-input hard gate (`tests/test_attention_kernel.py:72-85`) and the GQA-routing test (`:91-113`), plus a full re-run of `bench/bench_attn_kernel.py`. Those hours buy a credential that is already earned.

**3. "Can he write CUDA" is already answered.** `kernels/attention_decode.cu` is 311 lines, three staged versions, online softmax (`:54-77`), shared-memory tree reduction (`:139-146`), warp-shuffle reduction (`:186-190`), and the flash-attention combine rule (`:256-282`). A second, harder kernel is the longest pole in the project for the smallest marginal signal.

**4. It risks the engine's strongest existing claim.** The 0.98–0.99× SDPA measurement (`BENCHMARKS.md:102`) is about the frozen kernel. A frozen, validated, measured kernel is an asset. A half-paged one is a liability.

**5. The serving layer is not judged on the arithmetic.** Block allocation, refcounting, copy-on-write, radix prefix caching, preemption policy, admission control, prefix-aware routing — that is the work that reads as infrastructure engineering, and none of it needs a custom kernel underneath. vLLM's contribution was the memory manager.

#### The one thing consuming does *not* excuse

Writing the `PagedTorchBackend` yourself is non-negotiable, for reasons that are technical rather than presentational:

- It is the only correctness oracle available for the paged path. FlashInfer cannot validate itself.
- FlashInfer's layout contract is **unverified** (`serving:ARCHITECTURE.md:464`). If it will not build on PACE, the PyTorch path is the entire project rather than a fallback.
- Given §2(b) — no model-level GPU oracle exists — a divergence on the paged path would otherwise be unattributable across the allocator, the batching, the backend, and a possible pre-existing GPU-path bug, with nothing to bisect against.

**Attribution wording, fixed:** *"integrated FlashInfer's paged-attention kernels behind a pluggable attention backend; wrote a PyTorch reference implementation as the correctness oracle."* Never "wrote a paged kernel."

---

## 4. Engine changes required

Split strictly. Hours are **[inference]** and assume familiarity with the code.

### STRICTLY PREREQUISITE

| # | Change | Hours | Why — with citation |
|---|---|---|---|
| **P1** | **Model-level GPU correctness oracle.** Greedy tokens from `LlamaModelGPU` vs `tests/oracle.py`'s `greedy_ids` (`tests/oracle.py:183`) and vs `LlamaModel` on the same prompt. Tolerance = token equality. | 2–3 | **Does not exist.** `tests/test_gpu_model.py:42-45` asserts only finite/shape/argmax-in-range. Every bit-exactness gate in the repo is CPU-path (`tests/conftest.py:27-29`, `tests/test_forward.py:104-139`, `tests/test_decode.py:15-48`). Without this, any divergence in the serving layer is unattributable across allocator / batching / backend / pre-existing GPU bug, with nothing to bisect. **Do this first — it gates everything else.** |
| **P2** | **Fix `tests/test_generate.py`:** `"input_ids"` → `"token_ids"` at `:42, :52, :63, :78, :89`. | 0.2 | `tests/oracle.py:170` writes `token_ids`. All 5 tests raise `KeyError` — reproduced empirically. The KV-cache-vs-no-cache identity gate has never run, despite `BENCHMARKS.md:226` listing it as passing. This is the only cache-correctness gate the engine has. |
| **P3** | **Chunked-prefill position fix:** `engine/model_gpu.py:65`, `torch.arange(seq)` → `torch.arange(kv_cache.pos, kv_cache.pos + seq)`, plus a test that prefills in two chunks and compares to a single-shot prefill. | 1 | `arange(seq)` starts at 0 regardless of cache position, so a second prefill chunk gets positions `0..n` instead of `pos..pos+n`. Latent only because `engine/scheduler.py:33` calls `prefill` once. **Radix prefix caching and chunked prefill both require it.** Fix the reference path too, not just the varlen path — it is a real bug independent of this project. |
| **P4** | **`engine/attention_backend.py`:** `AttentionBackend` Protocol + `BatchMeta` dataclass. New file, no behavior change. Add both to `engine/__init__.py` `__all__` (`:22-40`) and `_EXPORTS` (`:43-56`). | 1–1.5 | The seam. Nothing in the serving layer can be written against a contract that does not exist. |
| **P5** | **Widen the injection point** in `gqa_attention_gpu` (`engine/components_gpu.py:100-182`): add `backend=None, meta=None`; new branch after RoPE at `:135`; existing `:137-182` untouched below it. | 2–3 | Today the hook at `:114` injects math only (`:150-158`) while the engine owns cache read/write (`:137-142`). The backend must own both or paging is impossible. Generalizes existing machinery rather than adding new. |
| **P6** | **`forward_varlen(token_ids, meta, backend) -> Tensor (n_seqs, vocab) on device`** on `LlamaModelGPU`. Existing `prefill`/`decode_step`/`forward_all` untouched. | 2–3 | The batched entry point. Returns a **device** tensor, unlike `prefill`/`decode_step` which return CPU numpy (`engine/model_gpu.py:90,158`) — see hazard H4. |
| **P7** | **Parameterize CUDA arch:** `scripts/build_kernels.sh:10`, `-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-80}`. | 0.3 | Hardcoded `80`. H100/H200 are sm_90, L40S is sm_89. Prerequisite the moment any non-A100 node is used (`serving:PRD.md:196`). **Note: line 10, not line 9** — this repo cites `:9` in three places. |
| **P8** | **Fix the editable install.** The registered install points at `file:///Applications/Ashwin/Programming/Personal%20Projects/llm_inference_egine`, a path that no longer exists. `import engine` fails from any cwd outside the repo root. Re-run `pip install -e .` from the current location, or vendor as a submodule and rely on path. | 0.3 | Verified: `pip show llm-inference-engine` reports 0.1.0 installed; `python3 -c "import engine"` from `/tmp` raises `ModuleNotFoundError`; `dist-info/direct_url.json` names the dead path. CI and any non-cwd invocation break today. |

**Prerequisite total: ~9–12 hours.** Consistent with `serving:ARCHITECTURE.md:191`'s 6–10h once P1 and P8 are added.

**Removed from the prerequisite list:**
- ~~Git tag `v0.1.0`~~ — **already exists**, pointing at `6ff40a1`. `serving:PRD.md:105` and `serving:ARCHITECTURE.md:200` are wrong. Delete this item.
- ~~"Populate `engine/__init__.py`"~~ (`SERVING_INTERFACE.md:223,292`) — **done** at `6ff40a1`.

### NICE TO HAVE

| # | Change | Hours | Why |
|---|---|---|---|
| N1 | `--backend gpu` flag on the reference server (`engine/server.py:24-30`) | 1 | Server runs the fp32 NumPy CPU model; there is no path to `LlamaModelGPU`. Presentation only — the serving layer does not use this server. |
| N2 | Move `kernels/attn_reference.py` → `engine/kernel_api.py`; delete the `sys.path` injection at `engine/model_gpu.py:45-53` | 1.5 | The hack computes repo root from `engine/`'s own location and reaches into two sibling directories not in the package (`pyproject.toml:34-36` packages `engine*` only). Works under editable install by accident; fails under a real install. Not blocking, because the serving layer does not use the custom kernel on the paged path. |
| N3 | Persistent scratch buffers in the v3 launcher (`kernels/attention_decode.cu:298-300`, `:310`) | 1–2 | `cudaMalloc`/`cudaFree` ×3 inside every call. `BENCHMARKS.md:120` names it as known-unfixed and it is the stated cause of the 0.61× at kv_seq=128. Improves an engine number; irrelevant to serving. |
| N4 | CUDA error checking after launches (`kernels/attention_decode.cu:91,165,303,307`) and on `cudaDeviceSynchronize` returns (`kernels/bindings.cpp:39,52,65`) | 1 | `CLAIMS_AUDIT.md:299`: *"A launch failure silently produces garbage rather than raising."* Only matters if the custom kernel is on a serving path. It is not. Promote to prerequisite if that ever changes. |
| N5 | Batched sampler — a GPU-side or vectorized `sample` | 2–3 | `engine/sampler.py:10-55` is scalar float64 numpy per call. At batch 32 that is 32 device→host round-trips per step. **Deliberately deferred**: `forward_varlen` returning a device tensor keeps the option open (`serving:ARCHITECTURE.md:467`). Measure before building. |
| N6 | Fix stale `file:line` citations in `BENCHMARKS.md` | 0.3 | `:59` cites `harness.py:111` (actual `:135`); `:119` cites `bindings.cpp:67,79,91` (actual `:39,52,65` — a 75-line file); `:120` cites `attention_decode.cu:297-300, 311` (actual `:298-300, 310`). Cosmetic, but this repo cites `BENCHMARKS.md` as authoritative. |

**Explicitly NOT engine changes:** continuous batching policy, scheduler, block allocator, eviction, radix cache, routing, admission control, preemption. Upstreaming any of these moves work into the wrong repo and collapses the two-repo boundary.

---

## 5. What can be reused, and what must be rewritten

### Reusable as-is

| Asset | Location | Use |
|---|---|---|
| **HF oracle fixture generator** | `tests/oracle.py:81-192` | Hook-based capture of post-embed, per-layer post-attn/post-ffn, post-final-norm, logits, plus 32 greedy tokens (`:183`). Already generates `oracle_short` / `oracle_medium`. **This is what P1 should be built on** — the reference tokens already exist. |
| **`compare_tensors`** | `tests/oracle.py:40-58` | Max-abs + mean diff with worst-index reporting. Exactly the right primitive for a paged-vs-contiguous logit diff. |
| **CUDA kernel gate structure** | `tests/test_attention_kernel.py:72-85` | 100 random inputs, kv_seq spread 1–512, `< 1e-3` each. The *pattern* transfers directly to `PagedTorchBackend` vs the contiguous backend. |
| **GQA-routing test** | `tests/test_attention_kernel.py:91-113` | Constructs K/V where each KV head has a distinct constant, then checks each query head reads `h // groups`. **Transfers verbatim** to a paged backend — a block-table bug would show up as the wrong constant. Best test in the engine. |
| **CUDA-event timing** | `bench/bench_attn_kernel.py:32-44` | Warmup + N-iteration loop between events. Correct GPU timing. Needed for the batched path (see H4). |
| **Results writers** | `bench/harness.py:187-202` | Timestamped `{host}_{backend}.json/.csv`. Matching this format is already committed to (`serving:PRD.md:66`). |
| **Hardware/version stamping** | `bench/harness.py:69-99` | Records hostname, platform, numpy/torch/CUDA/transformers versions into every row. Reuse directly. |
| **Weight-memory accounting** | `bench/harness.py:54-66` | De-duplicates the tied `lm_head`/`embed_tokens` alias by `id()`. Without it fp16 is overstated by 525 MB. Reuse if VRAM is ever broken down by component. |
| **`generate()` as a cancellation primitive** | `engine/scheduler.py:11-45` | A Python generator; abandoning it stops decode at the next `yield`. The *mechanism* is right. |
| **Sampler seam** | `engine/scheduler.py:14,34`; logits returned raw at `engine/model_gpu.py:90,158` | Sampling is an injected callable over raw logits. Clean seam for constrained decoding later. Keep it. |

### Must be rewritten

| Asset | Why it does not transfer |
|---|---|
| **`bench/harness.py:106-139` (`time_generate`)** | **Closed-loop, single-request, one sequence at a time.** Drives `generate()` in-process at `:118` and times host-side `perf_counter` around each yielded token. Serving needs an **open-loop** harness with a fixed arrival process, N concurrent clients, and queueing delay included in TTFT. Nothing about the control flow survives. |
| **`_peak_mem_mb`** (`bench/harness.py:44-47`) | Host RSS via `getrusage`. On the GPU path this measures nothing relevant — `BENCHMARKS.md:247` flags it as a known gap where the same column name carries different quantities across files. Serving needs `torch.cuda.max_memory_allocated` plus explicit block-pool occupancy. |
| **Host-clock inter-token timing** (`bench/harness.py:119`) | Valid today **only because** `engine/model_gpu.py:158` forces a device sync via `.cpu().float().numpy()` every step (`BENCHMARKS.md:60`). `forward_varlen` returns a device tensor and removes that sync. See H4. |
| **`engine/scheduler.py`** | A single-request loop. Every serving concept — admission, batching, preemption, retirement — is absent. It is a reference path, not a starting point. |
| **`engine/server.py`** | Serializes structurally (`:69`: blocking sync generator inside `async def`). Loads the CPU model (`:24-30`). Reuse the OpenAI/SSE **schema shape** (`:34-46`, `:71-87`); rewrite the execution model entirely. |
| **`tests/test_cache.py`** | All 9 tests assert on the dense `(n_layers, max_seq, n_kv_heads, head_dim)` contract: shape (`:23-25`), dtype (`:28-30`), monotonic `pos` (`:37-50`), direct slice writes (`:53-64`), zero-fill of unwritten positions (`:81-85`). Every one is meaningless under paging. A block allocator needs different tests: free-list invariants, refcount correctness, copy-on-write, no double-free, no leak on cancellation. |
| **`tests/test_gpu_model.py`** | Asserts liveness, not correctness (`:42-45`). Superseded by P1. |
| **`bench/bench_attn_kernel.py`** | Microbenches the frozen custom kernel, which is off the paged path. The `_time_cuda` helper (`:32-44`) is reusable; the rest is not. If a paged-backend microbench is written, **do not repeat the GQA hoist at `:69-71`** — put the expansion inside the timed lambda or expand for both sides. |
| **`engine/cache.py`** | 33 lines encoding exactly the contract paging removes. Keep it for the reference path (`serving:ARCHITECTURE.md:141`); do not extend it. |

---

## 6. Inherited hazards

Things the serving layer must **not** assume are covered.

**H1 — The GPU path has never been validated against a reference.** Every bit-exactness claim in the engine is CPU fp32 (`tests/conftest.py:27-29`). The GPU model is gated only on finiteness and shape (`tests/test_gpu_model.py:42-45`). **Do not build any serving correctness claim on top of `LlamaModelGPU` output until P1 exists.** If a paged run diverges from expectation, there is currently no way to tell whether the bug is in the allocator, the batching, the backend, or the GPU forward pass itself. This is the top hazard in this document.

**H2 — The KV-cache correctness gate has never executed.** `tests/test_generate.py` — all 5 tests, `KeyError`, reproduced. `BENCHMARKS.md:226` lists it as passing. **Any engine gate cited from `BENCHMARKS.md:221-231` must be independently confirmed to actually run before it is relied upon.** Treat that table as a claim, not evidence.

**H3 — Chunked prefill is silently wrong today.** `engine/model_gpu.py:65` builds positions from 0 regardless of `kv_cache.pos`. Calling `prefill` twice on the same cache produces incorrect RoPE on the second call. It fails **silently** — plausible tokens, no exception. The radix-cache partial-hit walkthrough (`serving:ARCHITECTURE.md:353-372`) depends on prefilling only the divergent suffix, which is exactly this code path.

**H4 — Removing the per-token device sync silently invalidates host-clock timing.** `engine/model_gpu.py:158` ends with `.cpu().float().numpy()`, forcing a sync every decode step. That is *why* `bench/harness.py:119`'s `perf_counter` timings are valid (`BENCHMARKS.md:60`). `forward_varlen` returns a device tensor by design, removing the sync. **Any host-side timer on the batched path will then measure kernel-launch queueing, not execution, and will silently report absurdly good latency.** Use CUDA events or an explicitly declared sync point. This is a silent invalidator — it produces a *better* number, so it will not look like a bug.

**H5 — No CUDA error checking anywhere.** `CLAIMS_AUDIT.md:299`: *"No `cudaGetLastError()` after any launch (`kernels/attention_decode.cu:91,165,303,307`), and `cudaMalloc` return values are discarded (`:298-300`). A launch failure silently produces garbage rather than raising."* The bindings call `cudaDeviceSynchronize()` (`kernels/bindings.cpp:39,52,65`) and discard the return. For a batch-1 research path this is tolerable. For serving under memory pressure it means **a failed launch produces plausible text at full throughput with healthy metrics.** Mitigate: check CUDA errors at declared points; treat any CUDA error as fatal to the replica (a poisoned context is not recoverable in-process).

**H6 — The engine has zero thread safety.** Single global model (`engine/server.py:15`), mutable cache state (`engine/cache.py:17,30` — one shared `pos`, mutated by `advance` at `:19,32`), no locks anywhere. `KVCacheGPU` mutation happens inside `gqa_attention_gpu` (`engine/components_gpu.py:139-140`) with no synchronization. **Any design that touches the model from more than one thread is wrong.** Single-threaded cooperative scheduling is not a preference; it is the only correct option without adding locking the engine does not have.

**H7 — Cancellation does not reclaim memory.** Abandoning `generate()` stops decode (correct), but `KVCacheGPU` is constructed per call at `engine/scheduler.py:26-27` and reclaimed only by garbage collection. There is no `free`, no refcount, no reuse anywhere in `engine/cache.py`. **The serving layer's allocator must free blocks on disconnect explicitly**, or a disconnect-heavy workload leaks the entire block pool.

**H8 — The installed package is broken and the built `.so` is unfindable.** The editable install points at a nonexistent path (`llm_inference_egine`); `import engine` fails outside the repo root. Separately: the CUDA module is not built by pip at all (`pyproject.toml:1-3` is plain setuptools), lands in gitignored `build/` (`.gitignore:16`), and is located by `sys.path` injection computed from `engine/`'s own location (`engine/model_gpu.py:47-50`). `SERVING_INTERFACE.md:163-175`'s "`pip install -e .` succeeds today" does not hold. **Use a git submodule at tag `v0.1.0` and verify `import engine` in CI as an explicit step**, not as an assumption.

**H9 — Test fixtures are gitignored and expensive.** `.gitignore:24` excludes `tests/fixtures/*.pkl` (12.7 MB locally). Regenerating requires the gated Llama 3.2 1B checkpoint plus a full HF fp32 forward and a 32-token greedy decode (`tests/oracle.py:209-230`). **CI cannot run any correctness test without weights.** Design for it: gate correctness tests on a weights-present marker, or commit a small distilled fixture.

**H10 — `max_seq=2048` is a default in three places, not a validated bound.** `engine/scheduler.py:16` (default), `:27` (pass-through), `engine/model_gpu.py:55` (`make_cache` default). Nothing checks `kv_cache.pos + seq <= max_seq`; `engine/components_gpu.py:139` would raise an opaque shape error or silently truncate. Config `max_position_embeddings` sizes the RoPE tables (`engine/model_gpu.py:35`) independently. **The serving layer must enforce its own length bound at admission.**

**H11 — `EOS_IDS` is duplicated.** `engine/model.py:16` and `engine/model_gpu.py:19` both define `{128001, 128008, 128009}`. `engine/scheduler.py:8` imports the **CPU** copy and uses it for both paths (`:36,44`). Identical today. If the serving layer adds stop conditions, add them in one place and make the other import it.

**H12 — Two engine self-audits are gitignored and already stale.** `.gitignore:47-48` excludes `CLAIMS_AUDIT.md` and `SERVING_INTERFACE.md`. Both are dated 2026-07-30 and both predate `6ff40a1`. Confirmed stale: `SERVING_INTERFACE.md:206` (`__init__.py` is 0 bytes — false), `CLAIMS_AUDIT.md:6` (zero committed result files — false, 24 tracked), `CLAIMS_AUDIT.md:305` (`kernels/hello.cu` dead file — deleted in `6ff40a1`). **When this repo cites either document, cite the date and commit, and re-verify against source.**

---

## 7. Corrections owed to this repo's own documents

Every item below is a claim in `serving:ARCHITECTURE.md` or `serving:PRD.md` that source does not support. Listed so they can be fixed rather than propagated.

| Location | Claim | Reality |
|---|---|---|
| `PRD.md:105`, `ARCHITECTURE.md:200` | "git tag `v0.1.0` on the engine; **no tags exist**" | `git tag -l` returns `v0.1.0`; `git rev-list -n1 v0.1.0` = `6ff40a137bff...` = HEAD. **Already done. Remove from Tier 0.** |
| `PRD.md:86` (S8), `PRD.md:51` (G6) | Cites `CLAIMS_AUDIT.md:6`: *"`git ls-files` returns zero result files. Every A100 number ... exists only as prose."* | `git ls-files bench/results` returns **24 files** including `perplexity.csv` and `attn_kernel_microbench.csv`, committed in `c24afaf`. `.gitignore:29-31` now explicitly documents `bench/results/` as tracked on purpose. **The stated S8 baseline is false. Rewrite it** — the engine did fix this; the improvement S8 claims must come from somewhere else (e.g. artifacts for *concurrent-load* runs, which genuinely do not exist). |
| `ARCHITECTURE.md:199`, `PRD.md:141`, `PRD.md:196` | `scripts/build_kernels.sh:9` hardcodes `-DCMAKE_CUDA_ARCHITECTURES=80` | The flag is at **line 10**. Line 9 is `-DCMAKE_BUILD_TYPE=Release`. Cited three times here (and twice in `SERVING_INTERFACE.md:197,226`). |
| `PRD.md:17` | Quotes `BENCHMARKS.md:249` for *"There is no batch dimension in the tensors, the KV cache, the causal mask, or the CUDA kernel..."* | That text is at `BENCHMARKS.md:248`. `:249` is the following gap item. Quote is otherwise verbatim. |
| `ARCHITECTURE.md:343` | Attributes the CUDA-error-checking quote to `CLAIMS_AUDIT.md:305` | The quote is at `CLAIMS_AUDIT.md:299`. `:305` is a now-obsolete note about `kernels/hello.cu`, deleted in `6ff40a1`. The quoted content and the `bindings.cpp:39,52,65` citation are both correct. |
| `ARCHITECTURE.md:446`, and by inheritance `SERVING_INTERFACE.md:250,258-265` | Batched forward = "**15–25 hours**", therefore defer to Phase 2 | That estimate is for a **padded** `(batch, seq, hidden)` redesign, which `ARCHITECTURE.md:452` (A5) already rejects. Under varlen, §3.1 shows every non-attention operator is unchanged. **[inference]** ≈12–17h total, ≈2–3h of it engine-side. The deferral's stated justification does not survive. |
| `PRD.md:118`, and `SERVING_INTERFACE.md` throughout | Injected sampler at `engine/scheduler.py:15` | `sampler_fn` is the parameter at `:14`; `:15` is `max_tokens`. `:34` (the call site) is correct. |
| `ARCHITECTURE.md:70` | Lists `tests/test_forward.py:113,137` and `tests/test_decode.py:30-48` as claims that "must survive untouched" | Line numbers are right and the claims are real — but **all of them are `LlamaModel` (CPU fp32) claims**, not GPU. `ARCHITECTURE.md:198` says this correctly two hundred lines later; `:70` should say it too, or a reader concludes the fp16 path is protected. |
| `PRD.md:142` | "store the KV cache in kernel layout to isolate the transpose cost" — nice-to-have, superseded by paging | Correct that it is superseded. **Recommend deleting rather than deferring** — once the custom kernel is off the paged path (C3), the unmeasured split at `BENCHMARKS.md:151` informs no decision this repo makes. |
| — (not stated anywhere, should be) | The 0.98–0.99× SDPA comparison | `bench/bench_attn_kernel.py:69-71` hoists SDPA's GQA expansion out of the timed lambda while `:66` times the kernel with a `cudaDeviceSynchronize` inside it. Both biases run **against** the kernel, so the claim is conservative — but the GQA hoist is undisclosed in `BENCHMARKS.md:114-120`. `PRD.md:154` (C2) treats "0.98–0.99× SDPA" as a claim to be preserved intact; preserve it **with this caveat attached**. |

---

## 8. Verification log

Commands run against the engine at `6ff40a1`, for anyone re-checking this report.

```
git log --oneline -1                 -> 6ff40a1 Clean up repo for public view...
git tag -l                           -> v0.1.0
git rev-list -n1 v0.1.0              -> 6ff40a137bff99ef9571055e46f289518a6c55b8
git ls-files bench/results | wc -l   -> 24
wc -l kernels/attention_decode.cu    -> 311
wc -l kernels/bindings.cpp           -> 75
ls tests/fixtures/                   -> oracle_medium.pkl, oracle_short.pkl (gitignored)

python3 -m pytest tests/test_generate.py -m "not slow" -q
  -> 2 failed, 3 deselected
  -> KeyError: 'input_ids' at tests/test_generate.py:78 and :89

pip show llm-inference-engine        -> Version 0.1.0, site-packages (editable)
python3 -c "import engine"  (cwd=/tmp) -> ModuleNotFoundError: No module named 'engine'
dist-info/direct_url.json            -> file:///.../Personal%20Projects/llm_inference_egine  [path does not exist]
```

Files read in full: `engine/{__init__,cache,model_gpu,components_gpu,components,model,scheduler,server,sampler,loader,quant,cli}.py`; `kernels/{attention_decode.cu,bindings.cpp,attn_reference.py,CMakeLists.txt}`; `tests/{oracle,conftest,test_generate,test_gpu_model,test_forward,test_decode,test_components_gpu,test_attention_kernel,test_cache,test_server}.py`; `bench/{harness,bench_attn_kernel}.py`; `pyproject.toml`, `scripts/build_kernels.sh`, `.gitignore`, `README.md`, `BENCHMARKS.md`, and the cited passages of `SERVING_INTERFACE.md`, `CLAIMS_AUDIT.md`, `docs/BUILD_LOG.md`.
