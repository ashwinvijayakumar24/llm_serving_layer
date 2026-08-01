"""
KV pool sizing arithmetic — the numbers behind the S1 capacity claim.

WHY THIS IS A SEPARATE MODULE FROM THE ALLOCATOR
------------------------------------------------
The allocator hands out integers and knows nothing about tensors, layers, or
bytes (see allocator.py). That is what makes it testable on CPU. This module is
the other half: it turns a model shape and a VRAM budget into `num_blocks`, and
it is pure arithmetic, so it is testable on CPU too. Neither half needs a GPU to
be verified, which is most of why the S1 claim can be checked before any
allocation is ever made.

WHY THE DERIVATION IS A FIRST-CLASS OUTPUT
------------------------------------------
`KVPoolPlan.explain()` prints every line of the arithmetic. This is not a
convenience. S1 is a published claim — "increased concurrent-sequence capacity N× at
fixed VRAM" — and a reader is entitled to ask where N came from. A number that
can only be reproduced by rerunning the code is a number nobody can audit, and
docs/BENCHMARK_METHODOLOGY.md §11.6 says a claim whose artifact cannot be
regenerated is deleted rather than softened. So the plan carries its own
derivation and the artifact carries the plan.

THE PART THAT IS EASY TO GET DISHONEST
--------------------------------------
Paged-vs-contiguous capacity ratio is A FUNCTION OF SEQUENCE LENGTH, and the
function is steep. The contiguous baseline (B3, docs/BENCHMARK_METHODOLOGY.md §6)
reserves `max_seq=2048` tokens of KV per request regardless of what the request
uses (`vendor/llm_inference_engine/engine/scheduler.py:16`, `:26-27`). Paging
reserves `ceil(len / block_size)` blocks. So:

    at mean length 32    paging holds ~64x more concurrent sequences
    at mean length 2048  paging holds ~1x  more concurrent sequences

Both numbers are correct. Quoting the first without the length distribution
attached is the sort of thing that survives exactly one skeptical question.
`capacity_ratio()` therefore returns the length it was evaluated at, and
`CapacityRatio.claim()` refuses to render a bare "Nx".

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not measure anything. Every number here is computed from a stated model
shape and a stated VRAM budget. Measured capacity — real allocations through a
real BlockAllocator until exhaustion — is bench/capacity.py, and where the two
disagree the measurement wins. ARCHITECTURE.md §3.1 flags its own ~70k-block
figure as `[inference]` for exactly this reason.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "SizingError",
    "LLAMA_3_2_1B",
    "ModelKVShape",
    "KVPoolPlan",
    "ContiguousBaselinePlan",
    "CapacityRatio",
    "kv_bytes_per_token",
    "block_bytes",
    "plan_kv_pool",
    "plan_contiguous_baseline",
    "contiguous_baseline_capacity",
    "capacity_ratio",
    "capacity_ratio_curve",
    "DEFAULT_MAX_SEQ",
    "DEFAULT_BLOCK_SIZE",
    "A100_40GB_BYTES",
    "LLAMA_3_2_1B_FP16_WEIGHT_BYTES",
    "DEFAULT_ACTIVATION_HEADROOM_BYTES",
]

GIB = 1024**3
MIB = 1024**2

# ---------------------------------------------------------------------------
# Defaults, each traceable to a source. None of these are guesses dressed up as
# constants; where a number is inferred rather than measured it says so.
# ---------------------------------------------------------------------------

#: Llama 3.2 1B, fp16. Config: 16 layers, 8 KV heads (GQA), head_dim 64.
LLAMA_3_2_1B_N_LAYERS = 16
LLAMA_3_2_1B_N_KV_HEADS = 8
LLAMA_3_2_1B_HEAD_DIM = 64
FP16_BYTES = 2

#: Per-request KV reservation in the engine today — `max_seq` default in
#: vendor/llm_inference_engine/engine/scheduler.py:16, allocated whole at
#: :26-27 and dropped on return. This is baseline B3.
DEFAULT_MAX_SEQ = 2048

#: docs/ARCHITECTURE.md §3.1. Swept as a tunable later.
DEFAULT_BLOCK_SIZE = 16

#: Nominal A100 40GB. NOT a measured figure — `torch.cuda.get_device_properties`
#: reports somewhat less than the marketing capacity, and bench/capacity.py reads
#: the real value when a GPU is present rather than trusting this.
A100_40GB_BYTES = 40 * GIB

#: fp16 weight memory measured by the engine: 2357.1 MB
#: (vendor/llm_inference_engine/BENCHMARKS.md, Benchmark 4 table).
LLAMA_3_2_1B_FP16_WEIGHT_BYTES = int(2357.1 * MIB)

#: Activations, workspace, fragmentation, and the CUDA context. **[inference]** —
#: ARCHITECTURE.md §3.1 assumes "~35 GB usable for KV" on a 40 GB A100, which
#: implies roughly this much headroom once weights are subtracted. It is a
#: parameter on every function here precisely because it is the least certain
#: input; measure it and pass the measured value.
DEFAULT_ACTIVATION_HEADROOM_BYTES = 2 * GIB


class SizingError(ValueError):
    """
    A sizing request that cannot be satisfied.

    Raised rather than returning 0 or a negative block count. A pool of zero
    blocks is not a small pool, it is a configuration that cannot serve a single
    token, and silently returning it would push the failure into the allocator
    (which rejects `num_blocks <= 0` anyway) or, worse, into a benchmark that
    reports "capacity 0" as a result.
    """


@dataclass(frozen=True)
class ModelKVShape:
    """The four numbers that determine KV size. Nothing else about the model matters here."""

    n_layers: int
    n_kv_heads: int
    head_dim: int
    dtype_bytes: int = FP16_BYTES
    name: str = "unnamed"

    def bytes_per_token(self) -> int:
        return kv_bytes_per_token(
            self.n_layers, self.n_kv_heads, self.head_dim, self.dtype_bytes
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "n_layers": self.n_layers,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "dtype_bytes": self.dtype_bytes,
        }


LLAMA_3_2_1B = ModelKVShape(
    n_layers=LLAMA_3_2_1B_N_LAYERS,
    n_kv_heads=LLAMA_3_2_1B_N_KV_HEADS,
    head_dim=LLAMA_3_2_1B_HEAD_DIM,
    dtype_bytes=FP16_BYTES,
    name="Llama 3.2 1B (fp16)",
)


# ---------------------------------------------------------------------------
# Per-token and per-block arithmetic
# ---------------------------------------------------------------------------


def kv_bytes_per_token(
    n_layers: int = LLAMA_3_2_1B_N_LAYERS,
    n_kv_heads: int = LLAMA_3_2_1B_N_KV_HEADS,
    head_dim: int = LLAMA_3_2_1B_HEAD_DIM,
    dtype_bytes: int = FP16_BYTES,
) -> int:
    """
    Bytes of KV cache one token occupies, summed across all layers.

        2 (K and V) x n_layers x n_kv_heads x head_dim x dtype_bytes

    The leading 2 is K and V, and it is the factor most often dropped. For
    Llama 3.2 1B fp16:

        2 x 16 x 8 x 64 x 2 = 32768 bytes = 32 KiB per token

    Note `n_kv_heads`, not `n_heads`: the model is GQA (32 query heads, 8 KV
    heads), and the KV cache stores KV heads only. Using 32 here would inflate
    every capacity number in this project by 4x, and nothing would raise.
    """
    for label, value in (
        ("n_layers", n_layers),
        ("n_kv_heads", n_kv_heads),
        ("head_dim", head_dim),
        ("dtype_bytes", dtype_bytes),
    ):
        if value <= 0:
            raise SizingError(f"{label} must be positive, got {value}")
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes


def block_bytes(
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_layers: int = LLAMA_3_2_1B_N_LAYERS,
    n_kv_heads: int = LLAMA_3_2_1B_N_KV_HEADS,
    head_dim: int = LLAMA_3_2_1B_HEAD_DIM,
    dtype_bytes: int = FP16_BYTES,
) -> int:
    """
    Bytes one physical block occupies across all layers.

        block_size x kv_bytes_per_token

    For block_size=16 on Llama 3.2 1B fp16: 16 x 32768 = 524288 = 512 KiB,
    matching docs/ARCHITECTURE.md §3.1.

    This is the pool's allocation granularity, so it is also the granularity of
    internal fragmentation: a 17-token sequence pays for 32 tokens of KV.
    """
    if block_size <= 0:
        raise SizingError(f"block_size must be positive, got {block_size}")
    return block_size * kv_bytes_per_token(n_layers, n_kv_heads, head_dim, dtype_bytes)


# ---------------------------------------------------------------------------
# Pool plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVPoolPlan:
    """
    A sized KV pool plus the arithmetic that produced it.

    `explain()` prints the derivation line by line. Everything in `derivation()`
    is recomputed from the stored inputs, so the printed steps cannot drift away
    from the stored result.
    """

    shape: ModelKVShape
    total_vram_bytes: int
    model_weight_bytes: int
    activation_headroom_bytes: int
    block_size: int

    available_bytes: int
    bytes_per_token: int
    bytes_per_block: int
    num_blocks: int
    tokens_capacity: int
    bytes_used: int
    bytes_leftover: int

    notes: list[str] = field(default_factory=list)

    # -- checks the plan makes about itself ---------------------------------

    def is_self_consistent(self) -> bool:
        """`num_blocks` blocks fit; `num_blocks + 1` would not."""
        return (
            self.num_blocks * self.bytes_per_block <= self.available_bytes
            and (self.num_blocks + 1) * self.bytes_per_block > self.available_bytes
        )

    @property
    def utilization_of_available(self) -> float:
        return self.bytes_used / self.available_bytes

    def sequences_at_length(self, seq_len: int) -> int:
        """
        Concurrent sequences of exactly `seq_len` tokens the pool holds.

        Rounds UP to whole blocks: that internal fragmentation is real and paying
        it is the price of paging. Rounding down here would flatter every ratio
        in this module.
        """
        if seq_len <= 0:
            raise SizingError(f"seq_len must be positive, got {seq_len}")
        blocks_each = math.ceil(seq_len / self.block_size)
        return self.num_blocks // blocks_each

    # -- derivation ---------------------------------------------------------

    def derivation(self) -> list[str]:
        s = self.shape
        return [
            f"model                 {s.name}: "
            f"{s.n_layers} layers, {s.n_kv_heads} kv heads, head_dim {s.head_dim}, "
            f"{s.dtype_bytes} bytes/element",
            f"bytes per token       2 (K,V) x {s.n_layers} x {s.n_kv_heads} x {s.head_dim} "
            f"x {s.dtype_bytes} = {self.bytes_per_token:,} B "
            f"({self.bytes_per_token / 1024:.0f} KiB)",
            f"bytes per block       {self.block_size} tokens x {self.bytes_per_token:,} B "
            f"= {self.bytes_per_block:,} B ({self.bytes_per_block / 1024:.0f} KiB)",
            f"total VRAM            {self.total_vram_bytes:,} B "
            f"({self.total_vram_bytes / GIB:.2f} GiB)",
            f"- model weights       {self.model_weight_bytes:,} B "
            f"({self.model_weight_bytes / GIB:.2f} GiB)",
            f"- activation headroom {self.activation_headroom_bytes:,} B "
            f"({self.activation_headroom_bytes / GIB:.2f} GiB)",
            f"= available for KV    {self.available_bytes:,} B "
            f"({self.available_bytes / GIB:.2f} GiB)",
            f"num_blocks            floor({self.available_bytes:,} / {self.bytes_per_block:,}) "
            f"= {self.num_blocks:,}",
            f"tokens capacity       {self.num_blocks:,} blocks x {self.block_size} tokens "
            f"= {self.tokens_capacity:,} tokens",
            f"bytes used            {self.num_blocks:,} x {self.bytes_per_block:,} "
            f"= {self.bytes_used:,} B ({self.bytes_used / GIB:.2f} GiB)",
            f"bytes left unused     {self.bytes_leftover:,} B "
            f"(< one block, by construction)",
        ]

    def explain(self) -> str:
        lines = ["KV pool sizing", "=" * 72]
        lines += self.derivation()
        if self.notes:
            lines += ["", "notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "shape": self.shape.as_dict(),
            "total_vram_bytes": self.total_vram_bytes,
            "model_weight_bytes": self.model_weight_bytes,
            "activation_headroom_bytes": self.activation_headroom_bytes,
            "block_size": self.block_size,
            "available_bytes": self.available_bytes,
            "bytes_per_token": self.bytes_per_token,
            "bytes_per_block": self.bytes_per_block,
            "num_blocks": self.num_blocks,
            "tokens_capacity": self.tokens_capacity,
            "bytes_used": self.bytes_used,
            "bytes_leftover": self.bytes_leftover,
            "derivation": self.derivation(),
            "notes": list(self.notes),
        }


def plan_kv_pool(
    total_vram_bytes: int = A100_40GB_BYTES,
    model_weight_bytes: int = LLAMA_3_2_1B_FP16_WEIGHT_BYTES,
    activation_headroom_bytes: int = DEFAULT_ACTIVATION_HEADROOM_BYTES,
    block_size: int = DEFAULT_BLOCK_SIZE,
    shape: ModelKVShape = LLAMA_3_2_1B,
    notes: Sequence[str] | None = None,
) -> KVPoolPlan:
    """
    Size the KV pool from a VRAM budget. docs/ARCHITECTURE.md §3.1.

        available  = total_vram - model_weights - activation_headroom
        num_blocks = floor(available / block_bytes)

    Raises SizingError rather than returning a zero or negative block count: a
    pool that cannot hold one block is a misconfiguration, and the caller needs
    to see which term overran, not a plausible-looking zero.
    """
    if total_vram_bytes <= 0:
        raise SizingError(f"total_vram_bytes must be positive, got {total_vram_bytes}")
    if model_weight_bytes < 0:
        raise SizingError(f"model_weight_bytes must be non-negative, got {model_weight_bytes}")
    if activation_headroom_bytes < 0:
        raise SizingError(
            f"activation_headroom_bytes must be non-negative, got {activation_headroom_bytes}"
        )

    if model_weight_bytes > total_vram_bytes:
        raise SizingError(
            f"Model weights ({model_weight_bytes:,} B, {model_weight_bytes / GIB:.2f} GiB) "
            f"exceed total VRAM ({total_vram_bytes:,} B, {total_vram_bytes / GIB:.2f} GiB). "
            "The model does not fit; there is no KV pool to size."
        )

    available = total_vram_bytes - model_weight_bytes - activation_headroom_bytes
    if available <= 0:
        raise SizingError(
            f"No VRAM left for KV: total {total_vram_bytes:,} B "
            f"- weights {model_weight_bytes:,} B "
            f"- headroom {activation_headroom_bytes:,} B = {available:,} B. "
            "Reduce the activation headroom or use a smaller model."
        )

    per_block = block_bytes(
        block_size, shape.n_layers, shape.n_kv_heads, shape.head_dim, shape.dtype_bytes
    )
    num_blocks = available // per_block
    if num_blocks < 1:
        raise SizingError(
            f"Available KV memory ({available:,} B) is smaller than one block "
            f"({per_block:,} B at block_size={block_size}). "
            "A zero-block pool cannot serve a single token."
        )

    used = num_blocks * per_block
    return KVPoolPlan(
        shape=shape,
        total_vram_bytes=int(total_vram_bytes),
        model_weight_bytes=int(model_weight_bytes),
        activation_headroom_bytes=int(activation_headroom_bytes),
        block_size=int(block_size),
        available_bytes=int(available),
        bytes_per_token=shape.bytes_per_token(),
        bytes_per_block=int(per_block),
        num_blocks=int(num_blocks),
        tokens_capacity=int(num_blocks * block_size),
        bytes_used=int(used),
        bytes_leftover=int(available - used),
        notes=list(notes or []),
    )


# ---------------------------------------------------------------------------
# Baseline B3 — contiguous per-request allocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContiguousBaselinePlan:
    """
    Baseline B3 (docs/BENCHMARK_METHODOLOGY.md §6) with its derivation.

    The engine builds one `KVCacheGPU(n_layers, max_seq, n_kv_heads, head_dim)`
    per `generate()` call (vendor/llm_inference_engine/engine/scheduler.py:26-27)
    and drops it on return. The tensor is `zeros(...)`, so the full max_seq is
    committed at admission whether the request generates 30 tokens or 2000.

    Concurrency here is therefore hypothetical in one specific sense: the engine
    is single-request today (scheduler.py docstring: "single-request, Phase 2").
    This number is what its *allocation strategy* would support if it ran N
    requests at once, which is exactly the comparison S1 makes. That framing is
    stated in `derivation()` rather than left implicit.
    """

    shape: ModelKVShape
    total_vram_bytes: int
    model_weight_bytes: int
    activation_headroom_bytes: int
    max_seq: int

    available_bytes: int
    bytes_per_token: int
    bytes_per_sequence: int
    num_sequences: int
    bytes_used: int

    def derivation(self) -> list[str]:
        return [
            f"baseline              B3 — contiguous KVCacheGPU(max_seq={self.max_seq}) "
            f"per request (engine/scheduler.py:16,26-27)",
            f"bytes per token       {self.bytes_per_token:,} B",
            f"bytes per sequence    {self.max_seq} tokens x {self.bytes_per_token:,} B "
            f"= {self.bytes_per_sequence:,} B ({self.bytes_per_sequence / MIB:.0f} MiB) "
            f"— reserved in full at admission, regardless of tokens actually used",
            f"available for KV      {self.available_bytes:,} B "
            f"({self.available_bytes / GIB:.2f} GiB)",
            f"concurrent sequences  floor({self.available_bytes:,} / "
            f"{self.bytes_per_sequence:,}) = {self.num_sequences}",
            "caveat                the engine is single-request today; this is what its "
            "ALLOCATION STRATEGY supports concurrently, which is the S1 comparison",
        ]

    def explain(self) -> str:
        return "\n".join(["Contiguous baseline (B3)", "=" * 72] + self.derivation())

    def as_dict(self) -> dict:
        return {
            "shape": self.shape.as_dict(),
            "total_vram_bytes": self.total_vram_bytes,
            "model_weight_bytes": self.model_weight_bytes,
            "activation_headroom_bytes": self.activation_headroom_bytes,
            "max_seq": self.max_seq,
            "available_bytes": self.available_bytes,
            "bytes_per_token": self.bytes_per_token,
            "bytes_per_sequence": self.bytes_per_sequence,
            "num_sequences": self.num_sequences,
            "bytes_used": self.bytes_used,
            "derivation": self.derivation(),
        }


def plan_contiguous_baseline(
    total_vram_bytes: int = A100_40GB_BYTES,
    model_weight_bytes: int = LLAMA_3_2_1B_FP16_WEIGHT_BYTES,
    max_seq: int = DEFAULT_MAX_SEQ,
    activation_headroom_bytes: int = DEFAULT_ACTIVATION_HEADROOM_BYTES,
    shape: ModelKVShape = LLAMA_3_2_1B,
) -> ContiguousBaselinePlan:
    """B3 with its derivation attached. `contiguous_baseline_capacity` is the scalar form."""
    if max_seq <= 0:
        raise SizingError(f"max_seq must be positive, got {max_seq}")
    if total_vram_bytes <= 0:
        raise SizingError(f"total_vram_bytes must be positive, got {total_vram_bytes}")
    if model_weight_bytes > total_vram_bytes:
        raise SizingError(
            f"Model weights ({model_weight_bytes:,} B) exceed total VRAM "
            f"({total_vram_bytes:,} B). The model does not fit."
        )

    available = total_vram_bytes - model_weight_bytes - activation_headroom_bytes
    if available <= 0:
        raise SizingError(
            f"No VRAM left for KV after weights and headroom: {available:,} B."
        )

    bpt = shape.bytes_per_token()
    per_seq = max_seq * bpt
    n = available // per_seq
    if n < 1:
        raise SizingError(
            f"Available KV memory ({available:,} B) cannot hold even one contiguous "
            f"max_seq={max_seq} cache ({per_seq:,} B). The baseline cannot serve a "
            "single request at this configuration."
        )
    return ContiguousBaselinePlan(
        shape=shape,
        total_vram_bytes=int(total_vram_bytes),
        model_weight_bytes=int(model_weight_bytes),
        activation_headroom_bytes=int(activation_headroom_bytes),
        max_seq=int(max_seq),
        available_bytes=int(available),
        bytes_per_token=int(bpt),
        bytes_per_sequence=int(per_seq),
        num_sequences=int(n),
        bytes_used=int(n * per_seq),
    )


def contiguous_baseline_capacity(
    total_vram_bytes: int = A100_40GB_BYTES,
    model_weight_bytes: int = LLAMA_3_2_1B_FP16_WEIGHT_BYTES,
    max_seq: int = DEFAULT_MAX_SEQ,
    activation_headroom_bytes: int = DEFAULT_ACTIVATION_HEADROOM_BYTES,
    shape: ModelKVShape = LLAMA_3_2_1B,
) -> int:
    """
    Concurrent sequences the engine's current design supports — baseline B3, and
    the denominator of the S1 claim.

        floor((vram - weights - headroom) / (max_seq x bytes_per_token))

    Length-independent BY CONSTRUCTION, which is the entire point: a 30-token
    request costs exactly as much as a 2048-token one.
    """
    return plan_contiguous_baseline(
        total_vram_bytes=total_vram_bytes,
        model_weight_bytes=model_weight_bytes,
        max_seq=max_seq,
        activation_headroom_bytes=activation_headroom_bytes,
        shape=shape,
    ).num_sequences


# ---------------------------------------------------------------------------
# The ratio — and the length-dependence that makes or breaks the claim
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapacityRatio:
    """
    Paged-vs-contiguous concurrent-sequence capacity AT A STATED MEAN LENGTH.

    The mean length is a field, not a parameter that got consumed and forgotten,
    because the ratio is meaningless without it. `claim()` will not render a bare
    "Nx".
    """

    mean_seq_len: int
    block_size: int
    max_seq: int
    padded_tokens_per_seq: int
    blocks_per_seq: int
    paged_sequences: int
    contiguous_sequences: int
    ratio: float
    paged_bytes_per_seq: int
    contiguous_bytes_per_seq: int
    internal_fragmentation_tokens: int

    @property
    def internal_fragmentation_ratio(self) -> float:
        """Fraction of allocated KV slots the sequence never writes to."""
        return self.internal_fragmentation_tokens / self.padded_tokens_per_seq

    def claim(self) -> str:
        """
        The honest one-line form of S1. Always carries the length and the
        baseline's max_seq, because "Nx capacity" without them is not defensible.
        """
        return (
            f"{self.ratio:.1f}x concurrent-sequence capacity at fixed VRAM "
            f"for sequences of {self.mean_seq_len} tokens "
            f"({self.paged_sequences:,} paged vs {self.contiguous_sequences:,} contiguous), "
            f"against a baseline that reserves max_seq={self.max_seq} tokens per request. "
            f"The ratio is a function of sequence length and falls to ~1x as "
            f"lengths approach {self.max_seq}."
        )

    def as_dict(self) -> dict:
        return {
            "mean_seq_len": self.mean_seq_len,
            "block_size": self.block_size,
            "max_seq": self.max_seq,
            "padded_tokens_per_seq": self.padded_tokens_per_seq,
            "blocks_per_seq": self.blocks_per_seq,
            "paged_sequences": self.paged_sequences,
            "contiguous_sequences": self.contiguous_sequences,
            "ratio": self.ratio,
            "paged_bytes_per_seq": self.paged_bytes_per_seq,
            "contiguous_bytes_per_seq": self.contiguous_bytes_per_seq,
            "internal_fragmentation_tokens": self.internal_fragmentation_tokens,
            "internal_fragmentation_ratio": self.internal_fragmentation_ratio,
            "claim": self.claim(),
        }


def capacity_ratio(
    mean_seq_len: int,
    total_vram_bytes: int = A100_40GB_BYTES,
    model_weight_bytes: int = LLAMA_3_2_1B_FP16_WEIGHT_BYTES,
    activation_headroom_bytes: int = DEFAULT_ACTIVATION_HEADROOM_BYTES,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_seq: int = DEFAULT_MAX_SEQ,
    shape: ModelKVShape = LLAMA_3_2_1B,
) -> CapacityRatio:
    """
    Capacity ratio paged-vs-contiguous for sequences of `mean_seq_len` tokens.

    Both sides are given the SAME available bytes — same VRAM, same weights, same
    activation headroom — so the ratio isolates the allocation strategy and
    nothing else. Give the contiguous side less headroom and the ratio inflates
    for free; that would not be a measurement of paging.

    Analytically the ratio is approximately

        max_seq / (block_size * ceil(mean_seq_len / block_size))

    i.e. it is entirely explained by how much of the reserved 2048-token window a
    request actually uses. At mean_seq_len = max_seq it is ~1. This is a
    *computed* upper bound for a uniform-length workload; a real skewed workload
    is measured by bench/capacity.py, and the measurement is what gets published.
    """
    if mean_seq_len <= 0:
        raise SizingError(f"mean_seq_len must be positive, got {mean_seq_len}")
    if mean_seq_len > max_seq:
        raise SizingError(
            f"mean_seq_len {mean_seq_len} exceeds max_seq {max_seq}: the contiguous "
            "baseline could not serve such a request at all, so there is no ratio to "
            "report. Raise max_seq if that is the configuration being compared."
        )

    plan = plan_kv_pool(
        total_vram_bytes=total_vram_bytes,
        model_weight_bytes=model_weight_bytes,
        activation_headroom_bytes=activation_headroom_bytes,
        block_size=block_size,
        shape=shape,
    )
    baseline = plan_contiguous_baseline(
        total_vram_bytes=total_vram_bytes,
        model_weight_bytes=model_weight_bytes,
        max_seq=max_seq,
        activation_headroom_bytes=activation_headroom_bytes,
        shape=shape,
    )

    blocks_each = math.ceil(mean_seq_len / block_size)
    padded = blocks_each * block_size
    paged_n = plan.num_blocks // blocks_each

    return CapacityRatio(
        mean_seq_len=int(mean_seq_len),
        block_size=int(block_size),
        max_seq=int(max_seq),
        padded_tokens_per_seq=int(padded),
        blocks_per_seq=int(blocks_each),
        paged_sequences=int(paged_n),
        contiguous_sequences=int(baseline.num_sequences),
        ratio=paged_n / baseline.num_sequences,
        paged_bytes_per_seq=int(blocks_each * plan.bytes_per_block),
        contiguous_bytes_per_seq=int(baseline.bytes_per_sequence),
        internal_fragmentation_tokens=int(padded - mean_seq_len),
    )


def capacity_ratio_curve(
    lengths: Sequence[int],
    **kwargs,
) -> list[CapacityRatio]:
    """
    The ratio across a range of lengths — the form S1 should always be published
    in. A single point is a cherry-pick until the curve next to it says otherwise.
    """
    return [capacity_ratio(int(n), **kwargs) for n in lengths]


# ---------------------------------------------------------------------------
# CLI: print the derivation. `python3 -m serving.memory.sizing`
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - a convenience, not a benchmark
    _plan = plan_kv_pool()
    print(_plan.explain())
    print()
    print(plan_contiguous_baseline().explain())
    print()
    print("Capacity ratio vs mean sequence length (COMPUTED, not measured)")
    print("=" * 72)
    print(f"{'mean len':>9}  {'paged seqs':>12}  {'contig seqs':>12}  {'ratio':>8}")
    for _r in capacity_ratio_curve([32, 64, 128, 256, 512, 1024, 2048]):
        print(
            f"{_r.mean_seq_len:>9}  {_r.paged_sequences:>12,}  "
            f"{_r.contiguous_sequences:>12,}  {_r.ratio:>7.1f}x"
        )
    print()
    print("S1, stated honestly:")
    print("  " + capacity_ratio(128).claim())
