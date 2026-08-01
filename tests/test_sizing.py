"""
KV pool sizing tests. Pure CPU, no GPU, no model, no weights — runs in CI.

These tests exist because S1 is a published claim ("increased concurrent-sequence
capacity Nx at fixed VRAM") and every input to that number is arithmetic that can
be checked exactly. A wrong factor here does not raise anywhere: it produces a
plausible capacity figure that is wrong by a clean power of two, which is the
hardest kind of wrong to notice.

    python3 -m pytest tests/test_sizing.py -q
"""

import math

import pytest

from bench.capacity import (
    CapacityConfig,
    draw_lengths,
    run_capacity_bench,
    s1_publishable,
)
from serving.memory.sizing import (
    A100_40GB_BYTES,
    GIB,
    LLAMA_3_2_1B,
    MIB,
    SizingError,
    block_bytes,
    capacity_ratio,
    capacity_ratio_curve,
    contiguous_baseline_capacity,
    kv_bytes_per_token,
    plan_contiguous_baseline,
    plan_kv_pool,
)

KIB = 1024


# ---------------------------------------------------------------------------
# 1. Bytes per token — the number every other number here multiplies
# ---------------------------------------------------------------------------


def test_kv_bytes_per_token_llama_3_2_1b_is_exactly_32_kib():
    """
    Llama 3.2 1B, fp16, K and V, summed over all layers:

        2 (K and V)
          x 16   layers
          x 8    KV heads      <- KV heads, not the 32 query heads (GQA)
          x 64   head_dim
          x 2    bytes (fp16)
        = 32768 bytes = 32 KiB per token

    Two ways to get this wrong, neither of which raises:
      - drop the leading 2 (K and V)          -> every capacity figure 2x too big
      - use n_heads=32 instead of n_kv_heads  -> every capacity figure 4x too small

    Asserted as a literal because the point is to pin the value, not to restate
    the formula the implementation already contains.
    """
    assert kv_bytes_per_token(16, 8, 64, 2) == 32768
    assert kv_bytes_per_token(16, 8, 64, 2) == 2 * 16 * 8 * 64 * 2
    assert kv_bytes_per_token(16, 8, 64, 2) == 32 * KIB
    # The module default IS Llama 3.2 1B fp16.
    assert kv_bytes_per_token() == 32768
    assert LLAMA_3_2_1B.bytes_per_token() == 32768


def test_kv_bytes_per_token_scales_linearly_in_every_term():
    base = kv_bytes_per_token(16, 8, 64, 2)
    assert kv_bytes_per_token(32, 8, 64, 2) == 2 * base
    assert kv_bytes_per_token(16, 16, 64, 2) == 2 * base
    assert kv_bytes_per_token(16, 8, 128, 2) == 2 * base
    assert kv_bytes_per_token(16, 8, 64, 4) == 2 * base   # fp32
    assert kv_bytes_per_token(16, 8, 64, 1) == base // 2  # fp8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_layers": 0},
        {"n_kv_heads": 0},
        {"head_dim": -64},
        {"dtype_bytes": 0},
    ],
)
def test_kv_bytes_per_token_rejects_nonpositive_shape(kwargs):
    with pytest.raises(SizingError):
        kv_bytes_per_token(**kwargs)


# ---------------------------------------------------------------------------
# 2. Block bytes
# ---------------------------------------------------------------------------


def test_block_bytes_at_block_size_16_is_512_kib():
    """
    16 tokens x 32768 B/token = 524288 B = 512 KiB per block, across all layers.

    This is the figure quoted in docs/ARCHITECTURE.md §3.1, and it is the
    allocation granularity, so it is also the granularity of internal
    fragmentation.
    """
    assert block_bytes(16) == 524288
    assert block_bytes(16) == 512 * KIB
    assert block_bytes(16) == 16 * kv_bytes_per_token()


def test_block_bytes_is_linear_in_block_size():
    for bs in (1, 8, 16, 32, 64, 128):
        assert block_bytes(bs) == bs * 32768


def test_block_bytes_rejects_nonpositive_block_size():
    with pytest.raises(SizingError):
        block_bytes(0)


# ---------------------------------------------------------------------------
# 3. Pool plan self-consistency
# ---------------------------------------------------------------------------


def test_plan_kv_pool_is_self_consistent_one_more_block_would_not_fit():
    """
    The floor division is the whole plan, so both sides of it are asserted:
    num_blocks fits, and num_blocks + 1 does not. A plan that satisfies only the
    first is compatible with wasting arbitrarily much VRAM.
    """
    plan = plan_kv_pool()
    assert plan.num_blocks * plan.bytes_per_block <= plan.available_bytes
    assert (plan.num_blocks + 1) * plan.bytes_per_block > plan.available_bytes
    assert plan.is_self_consistent()

    assert plan.available_bytes == (
        plan.total_vram_bytes - plan.model_weight_bytes - plan.activation_headroom_bytes
    )
    assert plan.bytes_used == plan.num_blocks * plan.bytes_per_block
    assert plan.bytes_leftover == plan.available_bytes - plan.bytes_used
    assert 0 <= plan.bytes_leftover < plan.bytes_per_block
    assert plan.tokens_capacity == plan.num_blocks * plan.block_size


@pytest.mark.parametrize("block_size", [1, 8, 16, 32, 64, 256])
@pytest.mark.parametrize("headroom_gib", [0, 1, 2, 8])
def test_plan_kv_pool_self_consistent_across_configurations(block_size, headroom_gib):
    plan = plan_kv_pool(
        activation_headroom_bytes=headroom_gib * GIB,
        block_size=block_size,
    )
    assert plan.is_self_consistent()


def test_plan_kv_pool_40gb_a100_matches_architecture_doc_order_of_magnitude():
    """
    docs/ARCHITECTURE.md §3.1 states, marked [inference], "roughly 70k blocks
    ~= 1.1 M tokens" on a 40 GB A100 with ~35 GB usable for KV.

    Asserted as a band, not a point. The doc's figure is an estimate with an
    unstated headroom, so pinning an exact equality here would be asserting
    agreement with a guess. What matters is that this implementation lands in the
    same place rather than a factor away.
    """
    plan = plan_kv_pool()
    assert 34 * GIB < plan.available_bytes < 36 * GIB
    assert 65_000 <= plan.num_blocks <= 75_000
    assert 1_000_000 <= plan.tokens_capacity <= 1_250_000


def test_plan_kv_pool_derivation_is_printable_and_arithmetic_is_shown():
    """
    S1 has to be explainable line by line. The derivation must actually contain
    the numbers, not a prose summary of them — otherwise it is decoration.
    """
    plan = plan_kv_pool()
    text = plan.explain()
    for token in (
        f"{plan.bytes_per_token:,}",
        f"{plan.bytes_per_block:,}",
        f"{plan.available_bytes:,}",
        f"{plan.num_blocks:,}",
        f"{plan.tokens_capacity:,}",
    ):
        assert token in text, f"derivation omits {token}"
    assert len(plan.derivation()) >= 8
    # Round-trips into an artifact config without custom encoders.
    import json

    json.dumps(plan.as_dict())


def test_plan_sequences_at_length_rounds_up_to_whole_blocks():
    """Internal fragmentation is paid, not netted out. 17 tokens costs 2 blocks."""
    plan = plan_kv_pool()
    assert plan.sequences_at_length(16) == plan.num_blocks
    assert plan.sequences_at_length(17) == plan.num_blocks // 2
    assert plan.sequences_at_length(32) == plan.num_blocks // 2
    with pytest.raises(SizingError):
        plan.sequences_at_length(0)


# ---------------------------------------------------------------------------
# 4. Length-dependence of the ratio — the part of S1 that can be oversold
# ---------------------------------------------------------------------------


def test_capacity_ratio_falls_monotonically_with_mean_length_and_approaches_one():
    """
    The honest form of S1 depends on the length distribution. Asserted as a TREND
    rather than a magic constant, because the constant would be a property of the
    VRAM budget and the trend is the property of the design:

      - short sequences  -> large ratio (paging reclaims the unused 2048-window)
      - length = max_seq -> ratio ~1    (nothing left to reclaim)

    A regression that made the ratio length-independent would be an allocator
    that had quietly gone back to reserving a fixed amount per sequence.
    """
    lengths = [16, 32, 64, 128, 256, 512, 1024, 2048]
    curve = capacity_ratio_curve(lengths)
    ratios = [c.ratio for c in curve]

    assert all(a > b for a, b in zip(ratios, ratios[1:], strict=False)), (
        f"ratio must strictly decrease with mean length, got {ratios}"
    )
    assert ratios[0] > 50, "short sequences must show a large capacity win"
    assert math.isclose(ratios[-1], 1.0, rel_tol=0.02), (
        f"at mean length == max_seq the ratio must collapse to ~1, got {ratios[-1]}"
    )
    # Halving the length should roughly double the ratio: the win is entirely
    # "how much of the reserved 2048-token window went unused".
    for shorter, longer in zip(ratios, ratios[1:], strict=False):
        assert math.isclose(shorter / longer, 2.0, rel_tol=0.05)


def test_capacity_ratio_matches_the_analytic_form():
    """ratio ~= max_seq / (block_size * ceil(mean_len / block_size))."""
    for n in (7, 16, 100, 333, 1024, 2048):
        r = capacity_ratio(n)
        analytic = 2048 / (16 * math.ceil(n / 16))
        assert math.isclose(r.ratio, analytic, rel_tol=0.02), (n, r.ratio, analytic)


def test_capacity_ratio_records_the_length_it_was_evaluated_at():
    """A bare 'Nx' is not a claim. The rendered claim must carry the length."""
    r = capacity_ratio(128)
    assert r.mean_seq_len == 128
    claim = r.claim()
    assert "128 tokens" in claim
    assert "2048" in claim
    assert f"{r.ratio:.1f}x" in claim
    assert "function of sequence length" in claim


def test_capacity_ratio_reports_internal_fragmentation():
    r = capacity_ratio(17)
    assert r.blocks_per_seq == 2
    assert r.padded_tokens_per_seq == 32
    assert r.internal_fragmentation_tokens == 15
    assert math.isclose(r.internal_fragmentation_ratio, 15 / 32)

    exact = capacity_ratio(64)
    assert exact.internal_fragmentation_tokens == 0


def test_capacity_ratio_refuses_lengths_longer_than_the_baseline_can_serve():
    """
    Above max_seq the contiguous baseline cannot serve the request at all, so
    there is no ratio — a number here would be a division against a system that
    would have refused the workload.
    """
    with pytest.raises(SizingError):
        capacity_ratio(2049)
    with pytest.raises(SizingError):
        capacity_ratio(0)


def test_capacity_ratio_gives_both_sides_the_same_memory_budget():
    """
    Deliberate fairness check. Handing the paged side more available bytes than
    the baseline would inflate the ratio for free, and nothing would raise.
    """
    headroom = 3 * GIB
    r = capacity_ratio(256, activation_headroom_bytes=headroom)
    plan = plan_kv_pool(activation_headroom_bytes=headroom)
    baseline = plan_contiguous_baseline(activation_headroom_bytes=headroom)
    assert plan.available_bytes == baseline.available_bytes
    assert r.paged_sequences == plan.num_blocks // 16
    assert r.contiguous_sequences == baseline.num_sequences


# ---------------------------------------------------------------------------
# 5. Contiguous baseline B3 — hand arithmetic
# ---------------------------------------------------------------------------


def test_contiguous_baseline_matches_hand_arithmetic():
    """
    Stated configuration, worked by hand:

        total VRAM         40 GiB           = 42,949,672,960 B
        model weights      2357.1 MiB       =  2,471,598,489 B   (BENCHMARKS.md, fp16)
        activation reserve  2 GiB           =  2,147,483,648 B
        available for KV                    = 38,330,590,823 B

        per sequence  2048 tokens x 32,768 B/token = 67,108,864 B = 64 MiB
        capacity      floor(38,330,590,823 / 67,108,864) = 571 sequences

    571 concurrent sequences, on a device that could hold 1.17 M tokens of KV, is
    the whole S1 argument: B3 spends 64 MiB on a 30-token request.
    """
    total = 40 * GIB
    weights = int(2357.1 * MIB)
    headroom = 2 * GIB
    available = total - weights - headroom

    assert total == 42_949_672_960
    assert weights == 2_471_598_489
    assert available == 38_330_590_823

    per_seq = 2048 * 32768
    assert per_seq == 67_108_864 == 64 * MIB

    expected = available // per_seq
    assert expected == 571

    assert contiguous_baseline_capacity(total, weights, 2048, headroom) == 571
    # The module defaults ARE this configuration.
    assert contiguous_baseline_capacity() == 571


def test_contiguous_baseline_is_independent_of_actual_sequence_length():
    """
    The defining property of B3 and the reason paging wins: a 30-token request
    and a 2048-token request cost the same 64 MiB
    (engine/scheduler.py:16,26-27 — `KVCacheGPU(..., max_seq, ...)` is `zeros`,
    allocated whole at admission).
    """
    n = contiguous_baseline_capacity()
    plan = plan_contiguous_baseline()
    assert plan.num_sequences == n
    assert plan.bytes_per_sequence == plan.max_seq * plan.bytes_per_token
    # Nothing in the derivation depends on a realized length.
    assert not any("realized" in line for line in plan.derivation())


def test_contiguous_baseline_halves_when_max_seq_doubles():
    """
    Doubling the per-request reservation halves the capacity — exactly, up to the
    floor. `assert a == 2*b` would be wrong: 1142 -> 571 -> 285, and 4*285 is
    1140, not 1142. The lost sequences are the remainder the floor discards, so
    the assertion is stated with that tolerance rather than tuned until green.
    """
    a = contiguous_baseline_capacity(max_seq=1024)
    b = contiguous_baseline_capacity(max_seq=2048)
    c = contiguous_baseline_capacity(max_seq=4096)
    assert (a, b, c) == (1142, 571, 285)
    assert a - 2 * b == 0
    assert 0 <= a - 4 * c <= 3  # floor remainder only


# ---------------------------------------------------------------------------
# 6. Edge cases — raise loudly rather than returning a plausible zero
# ---------------------------------------------------------------------------


def test_weights_larger_than_vram_raises_and_names_the_problem():
    with pytest.raises(SizingError, match="exceed total VRAM"):
        plan_kv_pool(total_vram_bytes=2 * GIB, model_weight_bytes=8 * GIB)
    with pytest.raises(SizingError, match="exceed total VRAM"):
        plan_contiguous_baseline(total_vram_bytes=2 * GIB, model_weight_bytes=8 * GIB)


def test_headroom_consuming_all_remaining_vram_raises():
    with pytest.raises(SizingError, match="No VRAM left"):
        plan_kv_pool(
            total_vram_bytes=8 * GIB,
            model_weight_bytes=4 * GIB,
            activation_headroom_bytes=4 * GIB,
        )


def test_available_memory_smaller_than_one_block_raises_not_zero_blocks():
    """
    A zero-block pool is not a small pool. Returning 0 here would push the failure
    into BlockAllocator (which rejects num_blocks <= 0 anyway) or, worse, into a
    benchmark that would happily report "capacity: 0 sequences" as a measurement.
    """
    with pytest.raises(SizingError, match="smaller than one block"):
        plan_kv_pool(
            total_vram_bytes=4 * GIB,
            model_weight_bytes=4 * GIB - 100_000,
            activation_headroom_bytes=0,
            block_size=16,
        )


def test_available_memory_smaller_than_one_contiguous_cache_raises():
    with pytest.raises(SizingError, match="cannot hold even one"):
        plan_contiguous_baseline(
            total_vram_bytes=4 * GIB,
            model_weight_bytes=4 * GIB - 32 * MIB,
            activation_headroom_bytes=0,
            max_seq=2048,
        )


@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_vram_raises(bad):
    with pytest.raises(SizingError):
        plan_kv_pool(total_vram_bytes=bad)
    with pytest.raises(SizingError):
        plan_contiguous_baseline(total_vram_bytes=bad)


def test_negative_headroom_raises():
    with pytest.raises(SizingError):
        plan_kv_pool(activation_headroom_bytes=-1)


def test_no_sizing_path_ever_returns_a_nonpositive_block_count():
    """
    Sweep configurations that straddle the failure boundary. Every one either
    raises or returns >= 1 block; none returns 0 or a negative count.
    """
    for weights_gib in range(0, 41, 4):
        for headroom_gib in (0, 2, 16, 39):
            try:
                plan = plan_kv_pool(
                    total_vram_bytes=A100_40GB_BYTES,
                    model_weight_bytes=weights_gib * GIB,
                    activation_headroom_bytes=headroom_gib * GIB,
                )
            except SizingError:
                continue
            assert plan.num_blocks >= 1
            assert plan.tokens_capacity >= plan.block_size


# ---------------------------------------------------------------------------
# 7. The capacity harness — artifact validity and honest provenance
# ---------------------------------------------------------------------------


def small_config(**overrides) -> CapacityConfig:
    """
    A pool small enough to exhaust in a test, sized the same way as the real one.
    ~4096 blocks: enough to make the distribution mean meaningful, fast on CPU.
    """
    cfg = CapacityConfig(
        total_vram_bytes=4 * GIB,
        model_weight_bytes=2 * GIB,
        activation_headroom_bytes=0,
        dist="lognormal",
        mean_len=128,
        sigma=0.8,
        seed=1234,
        name="capacity_s1_test",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_harness_artifact_validates():
    art = run_capacity_bench(small_config()).artifact
    assert art.validate() == []
    assert art.samples["seq_len_tokens"], "raw per-sequence samples must be stored"
    assert art.window, "samples without a recorded window are rejected by the schema (R11)"


def test_harness_artifact_is_unpublishable_outside_slurm():
    """
    Documents the rule rather than working around it. S1 is a comparison, and
    docs/BENCHMARK_METHODOLOGY.md §5 requires every A/B to run back-to-back
    inside one Slurm allocation with the allocation id recorded (R12). A CI or
    laptop run has no allocation id, so it can never back the claim — the
    artifact is still written, because it is real data.

    If this test ever fails, either the run really is inside Slurm, or the
    publication gate has been loosened.
    """
    art = run_capacity_bench(small_config()).artifact
    ok, blockers = s1_publishable(art.provenance)
    assert ok is False
    assert any("allocation" in b.lower() for b in blockers)
    assert art.provenance.allocation_id is None
    assert any("NOT PUBLISHABLE AS S1" in n for n in art.notes)


def test_harness_records_the_seed_and_is_reproducible():
    a = run_capacity_bench(small_config(seed=7))
    b = run_capacity_bench(small_config(seed=7))
    c = run_capacity_bench(small_config(seed=8))
    assert a.artifact.provenance.seed == 7
    assert a.artifact.samples["seq_len_tokens"] == b.artifact.samples["seq_len_tokens"]
    assert a.paged_sequences == b.paged_sequences
    assert c.artifact.samples["seq_len_tokens"] != a.artifact.samples["seq_len_tokens"]


def test_harness_records_realized_not_requested_distribution():
    """
    R14. The requested parameters are kept, but the realized distribution is a
    separate field and it is the one the claim quotes. Clipping at max_seq makes
    them genuinely differ, which is exactly why the realized one is recorded.
    """
    cfg = small_config(mean_len=1024, sigma=1.5, seed=3)
    res = run_capacity_bench(cfg)
    rw = res.artifact.realized_workload
    realized = rw["length_distribution"]

    assert rw["requested"]["mean_len"] == 1024
    assert set(realized) >= {"n", "mean", "p50", "p90", "p99", "min", "max",
                             "clipped_at_max_seq", "percentile_method"}
    assert realized["n"] == res.paged_sequences
    assert realized["clipped_at_max_seq"] > 0, "sigma=1.5 at mean 1024 must clip"
    assert realized["mean"] < 1024, "clipping must pull the realized mean below requested"
    assert max(res.artifact.samples["seq_len_tokens"]) <= cfg.max_seq


def test_harness_measurement_matches_arithmetic_for_a_fixed_length_workload():
    """
    The degenerate control. With every sequence the same length there is no skew,
    so the measured capacity must equal the computed one exactly. If these
    disagree, the simulation and the arithmetic are not describing the same
    system, and the skewed runs cannot be trusted either.
    """
    cfg = small_config(dist="fixed", mean_len=128)
    res = run_capacity_bench(cfg)
    plan = plan_kv_pool(
        total_vram_bytes=cfg.total_vram_bytes,
        model_weight_bytes=cfg.model_weight_bytes,
        activation_headroom_bytes=cfg.activation_headroom_bytes,
        block_size=cfg.block_size,
        shape=cfg.shape,
    )
    assert res.paged_sequences == plan.num_blocks // (128 // 16)
    assert res.artifact.scalars["kv_block_utilization"] == 1.0
    assert res.artifact.scalars["kv_internal_fragmentation"] == 0.0

    computed = capacity_ratio(
        128,
        total_vram_bytes=cfg.total_vram_bytes,
        model_weight_bytes=cfg.model_weight_bytes,
        activation_headroom_bytes=cfg.activation_headroom_bytes,
    )
    assert res.paged_sequences == computed.paged_sequences
    assert math.isclose(res.ratio, computed.ratio)


def test_harness_reports_gpu_as_unavailable_rather_than_estimating():
    """
    With no GPU, VRAM fields must be absent and explicitly labelled unavailable.
    An estimated figure is indistinguishable from a measured one in a JSON file.
    """
    art = run_capacity_bench(small_config(use_gpu=True)).artifact
    gpu = art.config["gpu"]
    if gpu["available"]:
        pytest.skip("a GPU is present; the no-GPU abstention path is not exercised")
    assert "reason" in gpu
    assert "gpu_total_mem_mb" not in art.scalars
    assert "gpu_free_mem_mb" not in art.scalars
    assert any("UNAVAILABLE" in n for n in art.notes)


def test_harness_ratio_shrinks_as_the_workload_lengthens():
    """
    Same trend as the arithmetic, but measured through the real allocator: the
    S1 ratio is a function of the workload, not a constant of the system.
    """
    ratios = [
        run_capacity_bench(small_config(dist="fixed", mean_len=n)).ratio
        for n in (64, 256, 1024, 2048)
    ]
    assert all(a > b for a, b in zip(ratios, ratios[1:], strict=False)), ratios
    assert math.isclose(ratios[-1], 1.0, rel_tol=0.05)


def test_harness_notes_state_the_length_dependence():
    """The caveat travels with the artifact, not only with the terminal output."""
    art = run_capacity_bench(small_config()).artifact
    joined = " ".join(art.notes)
    assert "function of sequence length" in joined
    assert "MEASURED" in joined and "COMPUTED" in joined


def test_harness_hitting_the_sequence_cap_is_labelled_a_lower_bound():
    res = run_capacity_bench(small_config(max_sequences=50))
    assert res.paged_sequences == 50
    assert res.artifact.realized_workload["hit_max_sequences_cap"] is True
    assert any("LOWER BOUND" in n for n in res.artifact.notes)


def test_harness_respects_the_admission_watermark():
    """
    Capacity is measured through `can_allocate()`, the watermark-respecting
    admission path. Reserving headroom must reduce the measured capacity;
    if it did not, admission would be bypassing the watermark.
    """
    plan = plan_kv_pool(
        total_vram_bytes=4 * GIB, model_weight_bytes=2 * GIB, activation_headroom_bytes=0
    )
    hi = run_capacity_bench(small_config(dist="fixed", mean_len=64))
    lo = run_capacity_bench(
        small_config(dist="fixed", mean_len=64, watermark_blocks=plan.num_blocks // 2)
    )
    assert lo.paged_sequences < hi.paged_sequences


# ---------------------------------------------------------------------------
# Workload generator
# ---------------------------------------------------------------------------


def test_draw_lengths_is_clipped_seeded_and_skewed():
    import random

    xs = draw_lengths(5000, "lognormal", 256, 0.8, 2048, random.Random(5))
    assert all(1 <= x <= 2048 for x in xs)
    assert xs == draw_lengths(5000, "lognormal", 256, 0.8, 2048, random.Random(5))
    # Lognormal is right-skewed: the mean sits above the median.
    assert sum(xs) / len(xs) > sorted(xs)[len(xs) // 2]


def test_draw_lengths_rejects_unknown_distribution():
    import random

    with pytest.raises(ValueError, match="unknown distribution"):
        draw_lengths(10, "normal", 256, 0.8, 2048, random.Random(0))


def test_draw_lengths_fixed_is_degenerate_on_purpose():
    import random

    xs = draw_lengths(100, "fixed", 128, 0.8, 2048, random.Random(0))
    assert set(xs) == {128}
