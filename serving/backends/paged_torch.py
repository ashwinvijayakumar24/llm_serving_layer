"""
PagedTorchBackend — a pure-PyTorch implementation of the ``AttentionBackend``
protocol (``vendor/llm_inference_engine/engine/attention_backend.py``).

WHAT THIS IS FOR
----------------
This backend is deliberately the SLOW one. It exists for three reasons, in
order of importance:

1. **It is the differential oracle (R9).** FlashInfer is the intended fast
   path, and a layout or addressing misunderstanding there is a HIGH-severity
   *silent* risk: wrong attention produces fluent text and moves no metric. The
   only defence is a second implementation, written independently of the
   kernel, that can be diffed against it token for token. That implementation
   has to be simple enough to trust by reading, which is why every clever
   optimisation below was declined.

2. **It is the correctness floor.** If FlashInfer is unavailable — CPU box, no
   CUDA, wrong wheel, an unsupported head_dim — the serving layer still runs.
   Every test above this layer (scheduler, radix cache, router) can then be
   written and run on CPU with no GPU at all, which is most of why those
   invariants are cheap to verify.

3. **It is the artifact the author must be able to derive unaided.** Paged
   attention is three ideas: a flat physical pool, a per-sequence page list,
   and an indirection between them. Everything here is those three ideas and
   nothing else.

So: clarity beats cleverness everywhere in this file. A Python loop over
sequences is a *deliberate* choice, not an oversight — see ``attend``.

LAYOUT
------
Per layer, separate K and V tensors::

    [num_blocks, block_size, n_kv_heads, head_dim]

This was VERIFIED compatible with FlashInfer, not assumed (docs/ARCHITECTURE.md
§2.3.1, checked against ``flashinfer-python==0.6.16``):
``append_paged_kv_cache`` accepts ``paged_kv_cache`` as either a combined
``[pages, 2, page_size, heads, dim]`` tensor *or* a ``(K, V)`` tuple
(``flashinfer/page.py:408``), and the trailing ``[..., n_kv_heads, head_dim]``
axes are FlashInfer's default ``NHD`` layout (``flashinfer/decode.py:403``).
Keeping K and V separate therefore costs nothing and means this backend's pool
can be handed to FlashInfer *as-is* when the two are diffed — the oracle and
the subject share physical memory, so a layout disagreement cannot hide in a
conversion step.

WHY THE TWO ADDRESSING FORMS
----------------------------
``BatchMeta`` carries both ``slot_mapping`` (flat per-token physical slot) and
the CSR triple ``kv_indptr``/``kv_indices``/``kv_last_page_len``. That is not
redundancy for its own sake:

  * WRITING uses ``slot_mapping``. One indexed scatter, no page arithmetic.
  * READING uses CSR, because a read needs a *range* of history, not a set of
    points, and because CSR is exactly what FlashInfer consumes.

Each backend takes what it needs; the alternative is forcing one backend to
reconstruct the other's addressing on every layer of every step.
"""

from __future__ import annotations

import torch
from engine.attention_backend import BatchMeta

__all__ = ["PagedTorchBackend"]


class PagedTorchBackend:
    """
    Paged KV storage plus attention, in pure PyTorch.

    Stateful: owns the physical KV pool for every layer. The engine holds no
    cache state on this path and never touches ``.k``, ``.v`` or ``.pos``.

    This class knows nothing about the allocator. It is handed physical block
    ids through ``BatchMeta`` and dereferences them; who owns those blocks, when
    they are freed, and whether they are shared is entirely the allocator's
    business (``serving/memory/allocator.py``). One owner per question.
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
    ):
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

        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype

        # groups == how many query heads share one KV head. The single number
        # the whole GQA mapping reduces to.
        self.groups = n_heads // n_kv_heads

        # One (K, V) pair per layer rather than one fused [layers, ...] tensor:
        # FlashInfer's append/plan APIs are per-layer, and a per-layer tensor is
        # what can be passed to them without a slice-and-hope.
        self.k_pool: list[torch.Tensor] = [self._new_pool() for _ in range(num_layers)]
        self.v_pool: list[torch.Tensor] = [self._new_pool() for _ in range(num_layers)]

    # -- pool construction / accounting -------------------------------------

    def _new_pool(self) -> torch.Tensor:
        return torch.zeros(
            (self.num_blocks, self.block_size, self.n_kv_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

    def zeros_like_pool(self) -> torch.Tensor:
        """
        A single zeroed pool tensor with this backend's exact shape/dtype/device.

        Used by tests that need a scratch pool, and by any code that wants to
        reason about pool shape without hardcoding the axis order.
        """
        return self._new_pool()

    def pool_bytes(self) -> int:
        """
        Total bytes of KV the pool occupies, across all layers, K and V.

        This is the number that makes pool sizing computable rather than
        guessed: usable VRAM minus weights minus activation headroom, divided by
        ``block_bytes()``, gives ``num_blocks``. (docs/ARCHITECTURE.md §3.1.)
        """
        return self.num_layers * 2 * self.k_pool[0].numel() * self.k_pool[0].element_size()

    def block_bytes(self) -> int:
        """Bytes one physical block costs across all layers, K and V together."""
        return self.pool_bytes() // self.num_blocks

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

    def append_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        meta: BatchMeta,
    ) -> None:
        """
        Scatter this step's K/V into the physical pool.

        ``k``/``v`` are ``(tokens, n_kv_heads, head_dim)`` with RoPE already
        applied to ``k`` — exactly what ``components_gpu.py:141-147`` computes.

        WHY THIS IS THREE LINES, AND WHY PAGES NEVER APPEAR
        --------------------------------------------------
        The pool is ``[num_blocks, block_size, n_kv_heads, head_dim]``, which is
        contiguous, so viewing it as ``[num_blocks * block_size, n_kv_heads,
        head_dim]`` costs nothing and turns the two-level (block, offset)
        address into ONE flat index. ``slot_mapping`` is precomputed on the CPU
        during batch assembly as::

            slot = block_id * block_size + (position % block_size)

        (``serving/memory/block_table.py:106``) — so by the time the tensor
        arrives here, all page structure has already been resolved into flat
        integers. The backend does one indexed write and never reasons about
        pages while WRITING at all. That asymmetry is real: a write touches a
        set of independent points, so it needs no notion of a range; a read
        needs a contiguous history, which is why ``attend`` must go back to the
        CSR page list.

        IDEMPOTENT by construction: an indexed assignment writes the same values
        to the same slots however many times it runs, so a scheduler retrying a
        step after a recoverable failure cannot double-append. (Contrast the
        engine's contiguous path, which advances ``kv_cache.pos`` — a retry
        there *would* corrupt the cache.)
        """
        self._check_layer(layer_idx)
        if meta.slot_mapping is None:
            raise ValueError(
                "PagedTorchBackend.append_kv requires meta.slot_mapping. It is "
                "optional on BatchMeta because FlashInfer uses batch_indices + "
                "positions instead; this backend uses the flat form."
            )

        tokens = k.shape[0]
        if k.shape != (tokens, self.n_kv_heads, self.head_dim):
            raise ValueError(
                f"k has shape {tuple(k.shape)}, expected "
                f"({tokens}, {self.n_kv_heads}, {self.head_dim})"
            )
        if v.shape != k.shape:
            raise ValueError(f"v shape {tuple(v.shape)} != k shape {tuple(k.shape)}")
        if meta.slot_mapping.shape[0] != tokens:
            raise ValueError(
                f"slot_mapping has {meta.slot_mapping.shape[0]} entries but "
                f"{tokens} tokens were supplied"
            )

        # .long(): advanced indexing wants an integer index tensor, and
        # slot_mapping is int32 by BatchMeta's contract.
        slots = meta.slot_mapping.to(device=self.device, dtype=torch.long)

        k_flat = self.k_pool[layer_idx].view(-1, self.n_kv_heads, self.head_dim)
        v_flat = self.v_pool[layer_idx].view(-1, self.n_kv_heads, self.head_dim)

        k_flat[slots] = k.to(device=self.device, dtype=self.dtype)
        v_flat[slots] = v.to(device=self.device, dtype=self.dtype)

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
        ``append_kv`` just wrote.

        ``q`` is ``(tokens, n_heads, head_dim)`` with RoPE applied; the return
        is the same shape.

        THE PYTHON LOOP IS DELIBERATE
        -----------------------------
        One iteration per sequence, in Python, on the host. This is CORRECT and
        it is SLOW, and both halves of that sentence matter:

          * Correct, because each sequence has its own KV length, its own page
            list and its own causal offset. Writing them out one at a time means
            there is no padding, no sentinel, no ragged-batch trickery, and
            nothing to get subtly wrong.
          * Slow, because it serialises per-sequence work and launches O(n_seqs)
            small kernels per layer per step — at 16 layers and a batch of 32
            that is 512 launches a step, which will dominate the step time.

        That cost is the entire reason FlashInfer exists on the fast path: it
        does this same computation for the whole batch in one launch by
        consuming the CSR page description directly. This backend is what
        FlashInfer is *checked against* (R9), not what it replaces.
        """
        self._check_layer(layer_idx)
        tokens = q.shape[0]
        if q.shape != (tokens, self.n_heads, self.head_dim):
            raise ValueError(
                f"q has shape {tuple(q.shape)}, expected "
                f"({tokens}, {self.n_heads}, {self.head_dim})"
            )

        out = torch.empty_like(q)

        # Pull the metadata to the host once. These are tiny (n_seqs-sized) and
        # the loop below indexes them per sequence; doing it once avoids a
        # device sync per field per sequence.
        cu_q = meta.cu_query_lens.tolist()
        q_lens = meta.query_lens.tolist()
        kv_lens = meta.kv_lens.tolist()
        kv_indptr = meta.kv_indptr.tolist()
        kv_indices = meta.kv_indices.to(device=self.device, dtype=torch.long)
        last_page = meta.kv_last_page_len.tolist()

        for i in range(meta.n_seqs):
            q_start, q_end = cu_q[i], cu_q[i + 1]
            q_len = q_lens[i]
            kv_len = kv_lens[i]
            if q_len == 0:
                continue

            pages = kv_indices[kv_indptr[i] : kv_indptr[i + 1]]
            k_seq, v_seq = self._gather(layer_idx, pages, kv_len, last_page[i])

            out[q_start:q_end] = self._attend_one(
                q[q_start:q_end], k_seq, v_seq, scale, q_len, kv_len
            )

        return out

    def _gather(
        self,
        layer_idx: int,
        pages: torch.Tensor,
        kv_len: int,
        last_page_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Materialise one sequence's KV history as a dense
        ``(kv_len, n_kv_heads, head_dim)`` pair.

        THE PARTIAL LAST PAGE IS THE WHOLE POINT OF THE TRUNCATION
        ---------------------------------------------------------
        ``pool[pages]`` yields ``n_pages * block_size`` token slots, but only
        ``kv_len`` of them were ever written. The tail of the final page is
        UNINITIALISED (in production it is whatever the previous owner of that
        block left behind — another sequence's KV, since the allocator recycles
        blocks). Attending over it does not crash and does not produce NaNs; it
        produces fluent, wrong text. So the slice to ``[:kv_len]`` is not
        tidiness, it is the fix for the failure mode R8's block-straddle sweep
        exists to catch.

        ``kv_last_page_len`` and ``kv_len`` say the same thing two ways, and the
        invariant ``1 <= kv_last_page_len <= page_size`` (an exact multiple of
        page_size reports page_size, NOT 0) is asserted here rather than
        trusted. Cheap, and the failure it catches is otherwise silent.

        Note ``pool[pages]`` is a GATHER, not a slice: ``pages`` is an arbitrary
        list of physical block ids in logical order, and they need not be
        contiguous or even increasing. Nothing below assumes they are.
        """
        n_pages = pages.shape[0]
        expected_pages = (kv_len + self.block_size - 1) // self.block_size
        if n_pages != expected_pages:
            raise ValueError(
                f"{n_pages} pages supplied but kv_len={kv_len} with "
                f"block_size={self.block_size} needs {expected_pages}"
            )
        tail = kv_len - (n_pages - 1) * self.block_size
        if last_page_len != tail:
            raise ValueError(
                f"kv_last_page_len={last_page_len} but kv_len={kv_len} over "
                f"{n_pages} pages implies {tail}. An exact multiple of "
                f"block_size must report {self.block_size}, not 0."
            )

        # (n_pages, block_size, n_kv_heads, head_dim) -> flatten the two leading
        # axes back into logical token order, then keep only real tokens.
        k = self.k_pool[layer_idx][pages].reshape(-1, self.n_kv_heads, self.head_dim)
        v = self.v_pool[layer_idx][pages].reshape(-1, self.n_kv_heads, self.head_dim)
        return k[:kv_len], v[:kv_len]

    def _attend_one(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        q_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        """
        Attention for ONE sequence. ``q`` is ``(q_len, n_heads, head_dim)``;
        ``k``/``v`` are ``(kv_len, n_kv_heads, head_dim)``.

        GQA EXPANSION
        -------------
        ``repeat_interleave(groups, dim=1)`` — byte-for-byte the same call the
        engine makes at ``components_gpu.py:191-192``. It matters that it is
        *interleave* and not ``repeat``/``tile``: interleave gives
        ``[kv0, kv0, kv1, kv1, ...]``, i.e. query head ``h`` reads KV head
        ``h // groups``; ``repeat`` would give ``[kv0, kv1, kv0, kv1, ...]``,
        i.e. ``h % n_kv_heads``. Both run, both produce plausible output, and
        only one is right — which is why the test suite uses distinct per-KV-head
        values so the wrong mapping is detectable.

        WHY EXPLICIT MATMUL + SOFTMAX RATHER THAN ``scaled_dot_product_attention``
        ------------------------------------------------------------------------
        SDPA would be faster and is *nearly* equivalent, but not exactly:

          * its ``is_causal=True`` aligns the mask to the TOP-LEFT corner, which
            is wrong here — a decode step has ``q_len=1`` against ``kv_len``
            history, and the correct mask is bottom-right aligned. Passing a
            hand-built ``attn_mask`` instead throws away the fast path anyway;
          * it chooses its own internal accumulation precision and may fuse the
            softmax, so it does not reproduce the reference's exact
            "scores in fp32 -> softmax -> cast back -> matmul in model dtype"
            rounding sequence.

        This backend's job is to be *the same number* as
        ``components_gpu.py:198-207``, not to be fast. So it does what that code
        does, in the same order, at the same precisions. Cleverness declined on
        purpose.
        """
        groups = self.groups
        k = torch.repeat_interleave(k, groups, dim=1)  # (kv_len, n_heads, head_dim)
        v = torch.repeat_interleave(v, groups, dim=1)

        qh = q.transpose(0, 1)  # (n_heads, q_len, head_dim)
        kh = k.transpose(0, 1)  # (n_heads, kv_len, head_dim)
        vh = v.transpose(0, 1)

        # fp32 for the scores regardless of pool dtype, matching
        # components_gpu.py:198. In fp16 the pre-softmax scores are the place
        # this math overflows first.
        scores = torch.matmul(qh.float(), kh.float().transpose(1, 2)) * scale

        # CAUSALITY, DERIVED FROM meta — NOT MATERIALISED WHEN UNNEEDED.
        #
        # Sequence i contributes q_len queries whose true positions END at
        # kv_len - 1, so query j attends to keys [0, kv_len - q_len + j].
        # With offset = kv_len - q_len, key index t is masked iff
        # t > offset + j — exactly `triu(diagonal=offset + 1)`, which is the
        # engine's own mask (components_gpu.py:200-203) generalised from
        # "offset = kv_seq - seq" to the paged case where kv_len came from the
        # page table instead of a contiguous cache. Same formula, same meaning.
        #
        # For a pure decode step q_len == 1 and offset == kv_len - 1, so the
        # mask would be all-zero: every key is legal, because the single query
        # sits at the end of history. Building a (1, kv_len) tensor of zeros
        # every layer of every step to add nothing is pure waste, so the decode
        # path skips it. The engine skips it for the same reason (`if seq > 1`).
        if q_len > 1:
            offset = kv_len - q_len
            mask = torch.triu(
                torch.full((q_len, kv_len), float("-inf"), device=scores.device),
                diagonal=offset + 1,
            )
            scores = scores + mask.unsqueeze(0)

        # softmax in fp32, then back to the pool dtype before the value matmul —
        # components_gpu.py:205 does `.half()`; `.to(self.dtype)` is that same
        # step written so the CPU/fp32 test path works too. With dtype=float16
        # this is literally the reference's cast.
        probs = torch.softmax(scores, dim=-1).to(self.dtype)

        out = torch.matmul(probs, vh)  # (n_heads, q_len, head_dim)
        return out.transpose(0, 1)  # (q_len, n_heads, head_dim)

    # -- misc ---------------------------------------------------------------

    def _check_layer(self, layer_idx: int) -> None:
        if layer_idx is None or not 0 <= layer_idx < self.num_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )

    def reset(self) -> None:
        """Zero every pool. Tests only — a live pool is owned by the allocator."""
        for t in self.k_pool:
            t.zero_()
        for t in self.v_pool:
            t.zero_()

    def __repr__(self) -> str:
        mib = self.pool_bytes() / (1024 * 1024)
        return (
            f"PagedTorchBackend(layers={self.num_layers}, blocks={self.num_blocks}, "
            f"block_size={self.block_size}, kv_heads={self.n_kv_heads}, "
            f"heads={self.n_heads}, head_dim={self.head_dim}, "
            f"dtype={self.dtype}, device={self.device}, pool={mib:.1f} MiB)"
        )
