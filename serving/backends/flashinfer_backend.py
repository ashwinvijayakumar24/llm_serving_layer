"""
FlashInferBackend — the FAST implementation of the ``AttentionBackend`` protocol
(``vendor/llm_inference_engine/engine/attention_backend.py``).

INTEGRATION GLUE, NOT AUTHORED KERNELS
--------------------------------------
Everything numerically interesting in this file happens inside
``flashinfer-python``. This module owns exactly three things: the physical KV
pool, the translation from ``BatchMeta`` to FlashInfer's argument list, and the
decision of *when* ``plan()`` must run. Attribution wording is fixed in
docs/PHASE_PLAN.md §11 — the kernels are NOT claimed as author-written.

It is drop-in interchangeable with ``PagedTorchBackend``: same constructor
shape, same pool layout, same two methods. That interchangeability is the whole
point — it is what makes ``tests/test_flashinfer_differential.py`` a *differential*
rather than an approximation, and it is what retires R9.

EVERY API DETAIL BELOW WAS READ OUT OF THE INSTALLED SOURCE
-----------------------------------------------------------
Verified against ``flashinfer-python==0.6.16``
(``_build_meta.py``: ``__git_version__ = 8da13a29``), extracted on PACE at
``~/scratch/fi_probe/pkg/flashinfer/``. File:line references below point into
that tree. Nothing here is inferred from documentation or memory.

  * ``page.py:403-411`` — ``append_paged_kv_cache(append_key, append_value,
    batch_indices, positions, paged_kv_cache, kv_indices, kv_indptr,
    kv_last_page_len, kv_layout="NHD")``.
  * ``page.py:428-436`` — ``paged_kv_cache`` may be a ``(k_cache, v_cache)``
    TUPLE of 4-D ``[max_num_pages, page_size, num_kv_heads, head_dim]`` tensors
    under ``NHD``. That is byte-for-byte ``PagedTorchBackend``'s pool shape, so
    the oracle and the subject can share physical memory and a layout
    disagreement cannot hide inside a conversion step.
  * ``decode.py:1239-1246`` — ``BatchDecodeWithPagedKVCacheWrapper.plan(indptr,
    indices, last_page_len, num_qo_heads, num_kv_heads, head_dim, page_size,
    **kwargs)``. Trailing args are keyword-only in practice: passing them
    positionally hits ``*deprecated_positional_args`` and emits a
    ``DeprecationWarning`` (``decode.py:1345-1352``).
  * ``prefill.py:2069-2084`` — ``BatchPrefillWithPagedKVCacheWrapper.plan(
    qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
    num_qo_heads, num_kv_heads, head_dim_qk, page_size, ..., causal=False,
    sm_scale=None, q_data_type=..., kv_data_type=...)``.
  * ``decode.py:777-790`` / ``prefill.py:1596-1609`` — both wrappers take
    ``float_workspace_buffer`` first, then ``kv_layout``. The decode docstring
    says the buffer "Must be initialized to 0 for its first use" and "must be
    16-byte aligned", hence ``torch.zeros(..., dtype=torch.uint8)`` below.
  * ``decode.py:1929`` / ``prefill.py:2740-2746`` — ``run()`` allocates its own
    output as ``q.shape[:-1] + (head_dim,)`` in ``q.dtype`` (unless an
    ``o_data_type`` was cached), so the return is ``(tokens, n_heads, head_dim)``
    matching ``q``. That was the last item ARCHITECTURE.md §2.3.1 listed as
    unverified; it is now read, not assumed.

CAUSALITY: BOTTOM-RIGHT ALIGNED, WHICH IS WHAT WE WANT
------------------------------------------------------
This is the subtlety ``PagedTorchBackend._attend_one`` declined ``SDPA`` over,
so it has to be right here rather than plausible.

The protocol's semantics (``attention_backend.py``, ``attend`` docstring):
query ``j`` of a sequence with ``q_len`` queries and ``kv_len`` keys attends to
keys ``[0, kv_len - q_len + j]``. ``PagedTorchBackend`` implements that as
``triu(diagonal=offset + 1)`` with ``offset = kv_len - q_len``.

FlashInfer's ``causal=True`` implements exactly the same thing. From the kernel
source, not the docstring — the docstring only says "whether to apply causal
mask" (``prefill.py:2145-2146``) and is silent on alignment:

    data/include/flashinfer/attention/prefill.cuh:1461
        (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= chunk_end))   // MASKED

    i.e. kept iff  kv_idx <= kv_len - qo_len + q_idx  — identical to the
    protocol's [0, kv_len - q_len + j].

    data/include/flashinfer/attention/scheduler.cuh:954
        int kv_len_init = kv_len - qo_len;  // right aligned

    data/include/flashinfer/attention/variants.cuh:89 (sliding window)
        mask &= (kv_idx + qo_len + window_left >= kv_len + qo_idx);

and FlashInfer's own PyTorch reference implementation agrees
(``trace/templates/attention.py:1391``: ``delta = kv_len - qo_len``;
``:1890``: ``mask[i, : kv_len - q_len_per_req + 1 + i] = 0.0``).

So: BOTTOM-RIGHT aligned, not top-left. ``torch.nn.functional.
scaled_dot_product_attention(is_causal=True)`` is top-left aligned and would be
WRONG here for any ``q_len < kv_len`` — which is every chunked prefill and every
decode. FlashInfer and the reference agree; SDPA is the odd one out. Passing
``causal=True`` unconditionally on the prefill path is therefore correct, and
passing nothing on the decode path is correct too (see ``attend`` below).

WHY TWO WRAPPERS
----------------
``BatchDecodeWithPagedKVCacheWrapper`` and ``BatchPrefillWithPagedKVCacheWrapper``
are different kernels with different schedulers, not two spellings of one thing.
The decode wrapper assumes one query row per request (``decode.py:1966``:
``q_len_per_req = q.size(0) // actual_batch_size``) and, without tensor cores,
*rejects* anything else (``decode.py:1970-1972``). It is optimised for the
memory-bound ``q_len == 1`` shape: no mask, split-K over the KV axis. The prefill
wrapper takes a ``qo_indptr`` and handles arbitrary per-sequence query lengths,
including 1 — so it is the general case, and it is what a MIXED prefill+decode
batch must use, because a batch where one sequence contributes 40 tokens and two
contribute 1 has no single ``q_len_per_req``.

Dispatch is therefore on "does any sequence contribute more than one token",
which is exactly ``BatchMeta.is_prefill``. But ``is_prefill`` is documented as a
*hint, never a correctness input*, so ``attend`` derives the answer from
``query_lens`` and raises if the hint disagrees — a disagreement is a batch
assembly bug and is otherwise silent.

WHEN plan() MUST RUN — THE ONE THING THAT IS BOTH A CORRECTNESS AND A PERF BUG
------------------------------------------------------------------------------
``plan()`` does not merely size a workspace. In the non-CUDA-graph path it
*stores the page tables on the wrapper*:

    decode.py:1467-1470   self._paged_kv_indptr_buf = indptr.to(self.device)
                          self._paged_kv_indices_buf = indices.to(...)
                          self._paged_kv_last_page_len_buf = last_page_len.to(...)
    prefill.py:2355-2365  same, plus self._qo_indptr_buf = qo_indptr.to(...)
    decode.py:1733        self._sm_scale = sm_scale
    prefill.py:2481       self._causal = causal

and ``run()`` reads those buffers, NOT its arguments — ``run(q, paged_kv_cache)``
takes no page table at all (``decode.py:1810``, ``prefill.py:2561``). The
consequences are asymmetric and both bad:

  * SKIPPING a needed ``plan()`` is a CORRECTNESS bug. The kernel silently
    attends over the PREVIOUS step's page table — stale page ids, stale
    ``last_page_len``, stale ``qo_indptr``. Every read lands in real allocated
    memory, so there is no fault, no NaN, and no metric movement: just another
    sequence's KV, fluently.
  * CALLING it every layer is a PERFORMANCE bug. ``plan()`` copies the CSR to
    host (``prefill.py:2287-2289``, ``decode.py`` equivalent), runs the host-side
    tile scheduler, and returns a ``plan_info`` — none of which depends on
    ``layer_idx``. FlashInfer's own docstring says so: "auxiliary data structures
    will be created during this call and cached for multiple run calls"
    (``decode.py:1338-1341``), and its example plans once and loops over 32
    layers (``decode.py:745-762``).

The rule that follows: **plan once per forward pass, run once per layer.** The
batch composition — which sequences, their page lists, their query lengths — is
fixed for the whole 16-layer loop by construction (``BatchMeta`` is a frozen
dataclass, and ``build_batch_meta`` mints a new one every step). So this backend
caches the plan keyed on the *identity* of the ``BatchMeta`` object plus the
softmax scale, and re-plans when either changes. Identity is safe here precisely
because the cached ``meta`` is held by a strong reference: a live object's ``id``
cannot be recycled underneath us. See ``_ensure_planned``.

WHAT THIS BACKEND DELIBERATELY DOES NOT DO
------------------------------------------
No CUDA graphs (``use_cuda_graph=True`` freezes batch size for the wrapper's
lifetime — ``decode.py:1445-1451`` — which is the opposite of continuous
batching), no FP8/NVFP4 KV, no sliding window, no logits soft-cap, no RoPE inside
the kernel (``pos_encoding_mode="NONE"``: the engine has already applied RoPE to
``q`` and ``k`` before the backend ever sees them, ``components_gpu.py:128-135``).
Each of those is a knob that would have to be verified against the oracle
separately, and none of them is needed for the Phase 1 claim.

OPTIONAL BY DESIGN
------------------
``import serving.backends.flashinfer_backend`` must succeed on a CPU-only box
with no ``flashinfer`` installed — ``PagedTorchBackend`` is the fallback and the
oracle, and the Phase 1 memory claim must not depend on this wheel resolving
(docs/RISK_REGISTER.md R18). So the import is LAZY and cached; ``is_available()``
answers the question without raising, and the constructor raises one clear,
actionable error if asked to build what it cannot.
"""

from __future__ import annotations

import torch
from engine.attention_backend import BatchMeta

__all__ = ["FlashInferBackend", "is_available", "unavailable_reason"]


# FlashInfer's default and the one this backend uses. NHD means the trailing
# axes of a page are [page_size, n_kv_heads, head_dim] — the pool shape
# PagedTorchBackend already allocates (page.py:428-431). HND exists and would
# require transposing the pool; there is no reason to.
KV_LAYOUT = "NHD"

# 128 MiB per wrapper, FlashInfer's own recommendation for the float workspace
# ("The recommended size is 128MB", decode.py:790). It backs the split-K partial
# results; too small and plan() raises rather than corrupting anything.
DEFAULT_WORKSPACE_MIB = 128


# -- lazy import ------------------------------------------------------------
#
# Cached in module globals rather than re-attempted per call: `import flashinfer`
# is not cheap (it pulls in the JIT machinery), and a machine without it should
# pay the ImportError exactly once.

_FI = None
_FI_REASON: str | None = None
_FI_TRIED = False


def _load():
    """Import flashinfer once. Returns (module | None, reason | None)."""
    global _FI, _FI_REASON, _FI_TRIED
    if not _FI_TRIED:
        _FI_TRIED = True
        try:
            import flashinfer  # noqa: PLC0415 - deliberately deferred
        except Exception as exc:  # noqa: BLE001 - any failure means "unavailable"
            # Deliberately broad. A broken CUDA toolchain, a missing nvcc, an
            # ABI mismatch against torch — all of them surface here as something
            # other than ImportError, and all of them mean the same thing to a
            # caller: use PagedTorchBackend.
            _FI, _FI_REASON = None, f"{type(exc).__name__}: {exc}"
        else:
            _FI, _FI_REASON = flashinfer, None
    return _FI, _FI_REASON


def is_available() -> bool:
    """
    True if ``flashinfer`` can be imported in this process.

    Says nothing about whether a GPU is present, whether the head_dim is one
    FlashInfer compiles a kernel for, or whether the JIT can build — those fail
    later and loudly. This is the cheap gate a serving layer uses at startup to
    choose a backend, and the gate a test uses to decide whether a differential
    can run at all.
    """
    return _load()[0] is not None


def unavailable_reason() -> str | None:
    """Why ``is_available()`` is False, or None if it is True."""
    return _load()[1]


class FlashInferBackend:
    """
    Paged KV storage plus attention, on FlashInfer's kernels.

    Constructor signature is IDENTICAL to ``PagedTorchBackend``'s (plus optional
    tuning knobs with defaults) so a test can parametrize over the two classes
    and a serving layer can swap one for the other with no call-site change.

    Stateful in the same way: owns the physical KV pool for every layer, knows
    nothing about the allocator, and is handed physical page ids through
    ``BatchMeta``.

    Additionally stateful in a way ``PagedTorchBackend`` is not: it holds two
    planned FlashInfer wrappers whose page tables are only valid for the
    ``BatchMeta`` they were planned against. That state is invisible to callers
    and managed entirely by ``_ensure_planned``; see the module docstring for why
    getting it wrong is silent.

    NOT THREAD SAFE and not reentrant. One backend, one forward pass at a time —
    two concurrent steps would interleave ``plan()`` calls on the same wrapper
    and each would attend over the other's page table. Same constraint the
    allocator carries (docs/ARCHITECTURE.md §7), for the same reason.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        n_kv_heads: int,
        n_heads: int,
        head_dim: int,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.float16,
        *,
        workspace_mib: int = DEFAULT_WORKSPACE_MIB,
        use_tensor_cores: bool = False,
    ):
        # -- validation, byte-for-byte PagedTorchBackend's so the two classes
        # -- reject the same inputs with the same messages.
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}; "
                "GQA maps query head h to KV head h // (n_heads // n_kv_heads)."
            )
        for name, val in (
            ("num_layers", num_layers),
            ("num_blocks", num_blocks),
            ("block_size", block_size),
            ("n_kv_heads", n_kv_heads),
            ("n_heads", n_heads),
            ("head_dim", head_dim),
        ):
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")

        fi, reason = _load()
        if fi is None:
            raise RuntimeError(
                "FlashInferBackend requires flashinfer-python, which is not "
                f"importable here: {reason}\n"
                "This backend is OPTIONAL by design. Use "
                "serving.backends.paged_torch.PagedTorchBackend instead — it is the "
                "correctness oracle and the supported fallback — or install the "
                "extra:  pip install -e '.[flashinfer]'\n"
                "Check availability without raising via "
                "serving.backends.flashinfer_backend.is_available()."
            )
        self._fi = fi

        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.groups = n_heads // n_kv_heads

        if self.device.type != "cuda":
            raise ValueError(
                f"FlashInferBackend requires a CUDA device, got {self.device}. "
                "PagedTorchBackend is the CPU path."
            )

        # Same pool as PagedTorchBackend: [num_blocks, block_size, n_kv_heads,
        # head_dim] per layer, K and V separate. This is FlashInfer's NHD tuple
        # form verbatim (page.py:428-431), so no adapter and no copy — the two
        # backends can be handed the same numbers and disagree only about math.
        self.k_pool: list[torch.Tensor] = [self._new_pool() for _ in range(num_layers)]
        self.v_pool: list[torch.Tensor] = [self._new_pool() for _ in range(num_layers)]

        # One workspace per wrapper rather than one shared buffer. Sharing would
        # work only as long as no two wrappers are ever planned simultaneously,
        # which is true today and is exactly the kind of invariant that stops
        # being true silently. 2 x 128 MiB against a multi-GiB KV pool is cheap
        # insurance; `workspace_mib` is there for a memory-tight configuration.
        # zeros, not empty: decode.py:783 requires the float workspace be
        # zero-initialised for its first use.
        nbytes = workspace_mib * 1024 * 1024
        self._decode_ws = torch.zeros(nbytes, dtype=torch.uint8, device=self.device)
        self._prefill_ws = torch.zeros(nbytes, dtype=torch.uint8, device=self.device)

        self._decode_wrapper = fi.BatchDecodeWithPagedKVCacheWrapper(
            self._decode_ws,
            KV_LAYOUT,
            use_tensor_cores=use_tensor_cores,
        )
        self._prefill_wrapper = fi.BatchPrefillWithPagedKVCacheWrapper(
            self._prefill_ws,
            KV_LAYOUT,
        )

        # Plan cache. See _ensure_planned. `_planned_meta` is a STRONG reference
        # on purpose: identity comparison is only sound while the object it
        # names is alive.
        self._planned_meta: BatchMeta | None = None
        self._planned_scale: float | None = None
        self._planned_is_prefill: bool = False

        # Diagnostics: a plan-per-layer regression shows up as plans ~= runs
        # instead of runs ~= num_layers * plans. Free to maintain, and the only
        # way to notice the perf bug described in the module docstring.
        self.n_plans = 0
        self.n_runs = 0

    # -- pool construction / accounting -------------------------------------
    #
    # Identical arithmetic to PagedTorchBackend, duplicated rather than shared
    # via a mixin so each backend can be read end to end on its own. If they
    # ever disagree the sizing tests catch it (tests/test_sizing.py).

    def _new_pool(self) -> torch.Tensor:
        return torch.zeros(
            (self.num_blocks, self.block_size, self.n_kv_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

    def zeros_like_pool(self) -> torch.Tensor:
        """A single zeroed pool tensor with this backend's exact shape/dtype/device."""
        return self._new_pool()

    def pool_bytes(self) -> int:
        """Total bytes of KV across all layers, K and V. Excludes the workspaces."""
        return self.num_layers * 2 * self.k_pool[0].numel() * self.k_pool[0].element_size()

    def block_bytes(self) -> int:
        """Bytes one physical block costs across all layers, K and V together."""
        return self.pool_bytes() // self.num_blocks

    def workspace_bytes(self) -> int:
        """
        Bytes the FlashInfer workspaces occupy.

        Reported separately from ``pool_bytes()`` because it is fixed overhead
        that does NOT scale with ``num_blocks`` — pool sizing must subtract it
        from usable VRAM before dividing (docs/ARCHITECTURE.md §3.1). It is the
        one line of the memory budget PagedTorchBackend does not have.
        """
        return self._decode_ws.numel() + self._prefill_ws.numel()

    def tokens_capacity(self) -> int:
        """Total tokens of KV the pool can hold. Mirrors BlockAllocator's."""
        return self.num_blocks * self.block_size

    @staticmethod
    def estimate_pool_bytes(
        num_layers: int,
        num_blocks: int,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ) -> int:
        """Same arithmetic, without allocating. For sizing before construction."""
        elem = torch.empty((), dtype=dtype).element_size()
        return num_layers * 2 * num_blocks * block_size * n_kv_heads * head_dim * elem

    # -- write path ---------------------------------------------------------

    def copy_block(self, src: int, dst: int) -> None:
        """
        Duplicate one physical block's KV, EVERY LAYER, for copy-on-write.

        The radix cache owns block INDICES; this backend owns the TENSORS. So
        when a sequence diverges inside a block it shares with another sequence,
        the cache allocates a fresh block and calls this to make the new block a
        true copy before anything writes into it.

        ALL LAYERS, not just layer 0. A per-layer copy would leave layers 1..N-1
        pointing at the sibling's KV: attention would read a mixture of two
        sequences' history and produce fluent, wrong text with no error. The
        same class of bug as a layer-0-only swap, and equally invisible.

        Raising rather than silently sharing is the contract the cache relies on
        (`RadixCache.ensure_writable` raises if this callable is absent), so a
        misconfiguration fails loudly instead of corrupting attention.
        """
        if not (0 <= src < self.num_blocks and 0 <= dst < self.num_blocks):
            raise IndexError(
                f"copy_block({src} -> {dst}) outside [0, {self.num_blocks})"
            )
        if src == dst:
            return
        for layer in range(self.num_layers):
            self.k_pool[layer][dst].copy_(self.k_pool[layer][src])
            self.v_pool[layer][dst].copy_(self.v_pool[layer][src])

    def append_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        meta: BatchMeta,
    ) -> None:
        """
        Scatter this step's K/V into the physical pool, via
        ``flashinfer.append_paged_kv_cache`` (``page.py:403``).

        ``k``/``v`` are ``(tokens, n_kv_heads, head_dim)`` with RoPE already
        applied to ``k`` — the same contract PagedTorchBackend takes.

        WHY batch_indices + positions AND NOT slot_mapping
        --------------------------------------------------
        The two backends address the write differently and ``BatchMeta`` carries
        both forms so neither has to reconstruct the other's. PagedTorchBackend
        does one indexed scatter with the flat ``slot_mapping`` and never reasons
        about pages. FlashInfer's kernel resolves the page itself: it is given
        ``batch_indices[j]`` (which sequence token j belongs to) and
        ``positions[j]`` (where in that sequence it sits) and walks the CSR to
        find the page — ``page.py:406-407``. Same destination, computed on
        opposite sides of the boundary. ``slot_mapping`` is ignored here, which
        is exactly why the differential test is meaningful: the two backends
        agree on WHERE a token goes only if both address it correctly.

        THE PRECONDITION FLASHINFER STATES AND DOES NOT CHECK
        ----------------------------------------------------
        ``page.py:513-516``: "The function assumes that the space for appended
        k/v has already been allocated, which means kv_indices, kv_indptr,
        kv_last_page_len has incorporated appended k/v." That is the same
        ordering contract ``build_batch_meta`` already enforces —
        ``blocks.append()`` runs BEFORE assembly, so ``kv_lens`` is the length
        AFTER the append (serving/engine_iface/batch.py, ORDERING CONTRACT).
        Violate it and the write lands outside the described pages: no fault,
        wrong KV.

        IDEMPOTENT, same as the reference: this is a scatter of fixed values to
        fixed slots, so a scheduler retrying a step cannot double-append.
        """
        self._check_layer(layer_idx)

        tokens = k.shape[0]
        if k.shape != (tokens, self.n_kv_heads, self.head_dim):
            raise ValueError(
                f"k has shape {tuple(k.shape)}, expected "
                f"({tokens}, {self.n_kv_heads}, {self.head_dim})"
            )
        if v.shape != k.shape:
            raise ValueError(f"v shape {tuple(v.shape)} != k shape {tuple(k.shape)}")
        if meta.n_tokens != tokens:
            raise ValueError(
                f"meta describes {meta.n_tokens} tokens but {tokens} were supplied"
            )

        # positions is int64 on BatchMeta (it indexes the RoPE tables). The
        # kernel wants int32 alongside batch_indices; the cast is a few hundred
        # elements and happens once per layer.
        self._fi.append_paged_kv_cache(
            k.to(device=self.device, dtype=self.dtype).contiguous(),
            v.to(device=self.device, dtype=self.dtype).contiguous(),
            self._i32(meta.batch_indices),
            self._i32(meta.positions),
            (self.k_pool[layer_idx], self.v_pool[layer_idx]),
            self._i32(meta.kv_indices),
            self._i32(meta.kv_indptr),
            self._i32(meta.kv_last_page_len),
            KV_LAYOUT,
        )

    # -- read path ----------------------------------------------------------

    def attend(
        self,
        q: torch.Tensor,
        layer_idx: int,
        scale: float,
        meta: BatchMeta,
    ) -> torch.Tensor:
        """
        Attention over each sequence's full KV history, including what
        ``append_kv`` just wrote. ``q`` is ``(tokens, n_heads, head_dim)``; the
        return is the same shape.

        Where ``PagedTorchBackend.attend`` runs a Python loop over sequences and
        launches O(n_seqs) small kernels per layer, this is one launch for the
        whole batch: the CSR page description the loop dereferenced by hand is
        consumed directly by the kernel. That difference is the entire reason
        this file exists.

        GQA is handled inside the kernel from ``num_qo_heads`` / ``num_kv_heads``
        passed at plan time — no ``repeat_interleave``, no materialised expanded
        K/V. FlashInfer requires ``num_qo_heads % num_kv_heads == 0``
        (``decode.py:1336-1338``), which the constructor already enforces. The
        mapping it uses is ``h // gqa_ratio`` (its own reference implementation,
        ``trace/templates/attention.py:1387``: ``kv_h = h // gqa_ratio``) —
        the same ``repeat_interleave`` semantics the engine and the oracle use,
        NOT ``h % n_kv_heads``. Both mappings run and both produce fluent output,
        which is why the differential test uses distinct per-KV-head values.
        """
        self._check_layer(layer_idx)
        tokens = q.shape[0]
        if q.shape != (tokens, self.n_heads, self.head_dim):
            raise ValueError(
                f"q has shape {tuple(q.shape)}, expected "
                f"({tokens}, {self.n_heads}, {self.head_dim})"
            )
        if meta.n_tokens != tokens:
            raise ValueError(
                f"meta describes {meta.n_tokens} tokens but q has {tokens}"
            )

        use_prefill = self._ensure_planned(meta, scale)
        q = q.to(device=self.device, dtype=self.dtype).contiguous()
        kv = (self.k_pool[layer_idx], self.v_pool[layer_idx])

        self.n_runs += 1
        wrapper = self._prefill_wrapper if use_prefill else self._decode_wrapper
        return wrapper.run(q, kv)

    # -- planning -----------------------------------------------------------

    def _ensure_planned(self, meta: BatchMeta, scale: float) -> bool:
        """
        Plan the right wrapper for this batch, at most once per forward pass.
        Returns True if the PREFILL wrapper was planned, False for decode.

        THE CACHE KEY IS OBJECT IDENTITY, AND THAT IS DEFENSIBLE
        --------------------------------------------------------
        ``BatchMeta`` is a frozen dataclass describing exactly one step, and
        ``build_batch_meta`` constructs a fresh one every step
        (serving/engine_iface/batch.py). Within a step the same object is passed
        to all ``num_layers`` calls. So "same object" is precisely "same batch
        composition", which is precisely the condition under which a plan may be
        reused (``decode.py:1338-1341``).

        Comparing tensor *contents* instead would be strictly worse: it costs a
        device-to-host copy and a comparison per layer to avoid a plan that costs
        a device-to-host copy and a host-side schedule, and it can still be
        fooled by an in-place mutation. Identity cannot be fooled that way
        either, so the residual hazard is the same one in both designs — a caller
        that mutates ``meta``'s tensors in place and reuses the object. That is a
        contract violation (the dataclass is frozen and the field docs say the
        batch is fixed for the pass); ``invalidate_plan()`` is the escape hatch
        if some future caller genuinely needs it.

        Holding ``meta`` alive is load-bearing: CPython recycles ``id`` values,
        so identity comparison against a dead object could produce a false
        positive. The strong reference makes that impossible. The cost is
        retaining a handful of ``n_seqs``-sized int32 tensors until the next
        step.

        ``scale`` is part of the key because FlashInfer takes ``sm_scale`` at
        PLAN time and caches it on the wrapper (``decode.py:1733``,
        ``prefill.py`` equivalent) — ``run()`` has no scale argument. It is
        constant for a given model, so this branch is taken once; it is in the
        key so that it cannot silently go stale if that ever stops being true.
        """
        if self._planned_meta is meta and self._planned_scale == scale:
            return self._planned_is_prefill

        # ONE host sync per step, not per layer, and only on a re-plan — plan()
        # copies the CSR to host anyway (prefill.py:2287-2289), so this adds no
        # synchronisation that was not already there.
        q_lens = meta.query_lens.tolist()
        use_prefill = any(qlen > 1 for qlen in q_lens)

        # is_prefill is documented as "a hint, never a correctness input", so it
        # is cross-checked rather than trusted. A disagreement means BatchMeta
        # was assembled by something other than build_batch_meta and is
        # internally inconsistent — which would otherwise show up only as wrong
        # attention on a batch that happens to be mixed.
        if bool(meta.is_prefill) != use_prefill:
            raise ValueError(
                f"BatchMeta.is_prefill={meta.is_prefill} disagrees with query_lens="
                f"{q_lens} (max={max(q_lens)}). is_prefill must be True iff some "
                "sequence contributes more than one token; an inconsistent BatchMeta "
                "would select the wrong FlashInfer kernel."
            )

        kv_indptr = self._i32(meta.kv_indptr)
        kv_indices = self._i32(meta.kv_indices)
        last_page = self._i32(meta.kv_last_page_len)

        if use_prefill:
            # The general case: arbitrary per-sequence query lengths, so a mixed
            # prefill+decode batch and a chunked prefill both land here.
            # causal=True is bottom-right aligned — see the module docstring.
            # cu_query_lens IS qo_indptr: both are the exclusive prefix sum of
            # query_lens with n_seqs + 1 entries (BatchMeta field docs;
            # prefill.py:2110-2111). No conversion, just a dtype cast.
            self._prefill_wrapper.plan(
                self._i32(meta.cu_query_lens),
                kv_indptr,
                kv_indices,
                last_page,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                meta.page_size,
                causal=True,
                pos_encoding_mode="NONE",
                sm_scale=scale,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
        else:
            # Every sequence contributes exactly one token, so q is
            # (n_seqs, n_heads, head_dim) and there is no mask to build: the
            # single query sits at the end of history and every key is legal.
            # The reference skips its mask for the same reason
            # (paged_torch.py, `if q_len > 1`), and the decode kernel has no
            # causal argument at all — with q_len_per_req == 1, is_causal is
            # False by construction (decode.py:1429-1430), which is equivalent.
            self._decode_wrapper.plan(
                kv_indptr,
                kv_indices,
                last_page,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                meta.page_size,
                pos_encoding_mode="NONE",
                sm_scale=scale,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )

        self._planned_meta = meta
        self._planned_scale = float(scale)
        self._planned_is_prefill = use_prefill
        self.n_plans += 1
        return use_prefill

    def invalidate_plan(self) -> None:
        """
        Force the next ``attend`` to re-plan.

        Needed only by a caller that mutates a ``BatchMeta``'s tensors in place
        and reuses the object — a contract violation, but one worth having a
        remedy for rather than a silent wrong answer. Also drops the strong
        reference to the cached meta.
        """
        self._planned_meta = None
        self._planned_scale = None

    # -- misc ---------------------------------------------------------------

    def _i32(self, t: torch.Tensor) -> torch.Tensor:
        """
        Move a metadata tensor to this backend's device as contiguous int32.

        FlashInfer's CSR arguments are int32 (``decode.py:1252-1259``) and
        ``BatchMeta`` already produces int32 for everything except ``positions``,
        which is int64 because it indexes the RoPE tables. A no-op cast on a
        tensor already in the right dtype/device returns the same tensor, so the
        common path copies nothing.
        """
        return t.to(device=self.device, dtype=torch.int32).contiguous()

    def _check_layer(self, layer_idx: int) -> None:
        if layer_idx is None or not 0 <= layer_idx < self.num_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )

    def reset(self) -> None:
        """
        Zero every pool and drop the plan. Tests only — a live pool is owned by
        the allocator.

        Dropping the plan matters: a test that resets the pool and then reuses a
        stale ``BatchMeta`` object would otherwise attend through a plan built
        for data that no longer exists.
        """
        for t in self.k_pool:
            t.zero_()
        for t in self.v_pool:
            t.zero_()
        self.invalidate_plan()

    def __repr__(self) -> str:
        mib = self.pool_bytes() / (1024 * 1024)
        ws = self.workspace_bytes() / (1024 * 1024)
        return (
            f"FlashInferBackend(layers={self.num_layers}, blocks={self.num_blocks}, "
            f"block_size={self.block_size}, kv_heads={self.n_kv_heads}, "
            f"heads={self.n_heads}, head_dim={self.head_dim}, "
            f"dtype={self.dtype}, device={self.device}, pool={mib:.1f} MiB, "
            f"workspace={ws:.0f} MiB)"
        )
