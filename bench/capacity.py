"""
S1 capacity benchmark — concurrent sequences, paged vs contiguous, at fixed VRAM.

WHAT THIS MEASURES AND WHAT IT DOES NOT
---------------------------------------
serving/memory/sizing.py COMPUTES capacity from a model shape and a VRAM budget.
This harness MEASURES it: it drives real sequences through the real
BlockAllocator and real SequenceBlocks until admission fails, under a stated
length distribution, and counts what got in. docs/ARCHITECTURE.md §3.1 marks its
own ~70k-block figure `[inference]` and says the ratio "must be measured rather
than computed for publication". This is that measurement.

Two things it deliberately does NOT do:

1. It does not touch the GPU for the capacity number. The allocator is pure
   Python integers by design, so the paging arithmetic is exercised exactly as
   the scheduler will exercise it, on CPU, in CI. `--gpu` additionally reports
   real `torch.cuda` VRAM figures; without a GPU those fields are recorded as
   UNAVAILABLE, never estimated.

2. It does not run the model. Capacity is an allocation property. Bit-identical
   output through PagedTorchBackend is a separate Phase 1 gate
   (docs/PHASE_PLAN.md §4, Definition of done) and is not what this file claims.

WHY THE RATIO IS REPORTED AS A FUNCTION OF LENGTH
-------------------------------------------------
The contiguous baseline B3 reserves `max_seq=2048` tokens per request whatever
the request does (engine/scheduler.py:16,26-27). Paging reserves what the request
uses, rounded up to a block. So the ratio is essentially
`max_seq / realized_padded_length` — huge for short sequences, ~1 for sequences
that genuinely fill the window. A headline "Nx" with no length distribution
attached is not a result, it is a choice of workload. Every artifact this harness
writes carries the REALIZED length distribution (R14), and the ratio is reported
next to it.

WHY THE REALIZED DISTRIBUTION, NOT THE REQUESTED ONE
----------------------------------------------------
docs/BENCHMARK_METHODOLOGY.md §12: "workload silently degenerate" is a threat
whose detection is "publish realized distributions from each run, not just
requested parameters". Here it bites concretely — lengths are clipped to
[1, max_seq], so a lognormal asked for mean 256 with a fat sigma does not have
realized mean 256. The clipped mean is what the ratio is a function of, so the
clipped mean is what gets recorded and what the claim quotes.

USAGE
-----
    python3 bench/capacity.py                       # defaults: Llama 3.2 1B, 40GB A100
    python3 bench/capacity.py --mean-len 512 --seed 7
    python3 bench/capacity.py --gpu                 # adds real torch.cuda VRAM figures
    python3 bench/capacity.py --sweep 32,128,512,2048
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # runnable as `python3 bench/capacity.py`
    sys.path.insert(0, str(REPO_ROOT))

from serving.memory.allocator import BlockAllocator  # noqa: E402
from serving.memory.block_table import SequenceBlocks  # noqa: E402
from serving.memory.sizing import (  # noqa: E402
    A100_40GB_BYTES,
    DEFAULT_ACTIVATION_HEADROOM_BYTES,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_SEQ,
    GIB,
    LLAMA_3_2_1B,
    LLAMA_3_2_1B_FP16_WEIGHT_BYTES,
    MIB,
    ModelKVShape,
    capacity_ratio,
    plan_contiguous_baseline,
    plan_kv_pool,
)
from serving.metrics.artifact import (  # noqa: E402
    REGISTRY,
    Artifact,
    MetricSpec,
    Provenance,
)

__all__ = [
    "CapacityConfig",
    "CapacityResult",
    "run_capacity_bench",
    "draw_lengths",
    "s1_publishable",
    "main",
]


def s1_publishable(prov: Provenance) -> tuple[bool, list[str]]:
    """
    May this run back the S1 claim? Stricter than `Provenance.is_publishable()`.

    S1 is a COMPARISON (paged vs B3 at fixed VRAM), and
    docs/BENCHMARK_METHODOLOGY.md §5 requires every A/B in this project to run
    inside one Slurm allocation, with the allocation identity recorded — node
    contention alone moves the engine's own throughput ~25% (BENCHMARKS.md:17).
    `assert_comparable()` already refuses to render a comparison whose allocation
    id is missing; this makes the same rule visible at publication time instead
    of at plotting time.

    A laptop run is therefore always unpublishable, by design. It is still real
    data and still written to results/ — it just may not back a claim.
    """
    _, reasons = prov.is_publishable()
    if prov.allocation_id is None:
        reasons.append(
            "No Slurm allocation id (SLURM_JOB_ID unset): this run is outside an "
            "allocation, so it is not a valid A/B measurement (BENCHMARK_METHODOLOGY "
            "§5, R12). Real data, but it may not back the S1 claim."
        )
    return (not reasons, reasons)

# ---------------------------------------------------------------------------
# Metric definitions. Registered before use, with (quantity, unit, source), so a
# later harness cannot quietly reuse one of these names for a different quantity
# — the engine's `peak_mem_mb` collision, R16.
# ---------------------------------------------------------------------------

for _spec in [
    MetricSpec(
        "concurrent_seqs_paged", "concurrent sequences admitted", "count",
        "BlockAllocator simulation driven to admission failure",
        "Sequences admitted into the paged KV pool before can_allocate() refused, "
        "under the realized length distribution recorded in this artifact.",
    ),
    MetricSpec(
        "concurrent_seqs_contiguous", "concurrent sequences admitted", "count",
        "arithmetic over baseline B3 per-request max_seq reservation",
        "floor(available_kv_bytes / (max_seq * kv_bytes_per_token)). COMPUTED, not "
        "measured: the engine allocates this cache per generate() call "
        "(engine/scheduler.py:26-27) and is single-request today, so there is no "
        "concurrent run to observe. Length-independent by construction.",
    ),
    MetricSpec(
        "capacity_ratio_paged_over_contiguous", "capacity ratio", "ratio",
        "concurrent_seqs_paged / concurrent_seqs_contiguous",
        "THE S1 NUMBER. Meaningless without the realized length distribution in "
        "this artifact attached to it.",
    ),
    MetricSpec(
        "kv_pool_blocks", "KV pool size", "count", "sizing arithmetic from VRAM budget",
        "floor((vram - weights - activation headroom) / block_bytes).",
    ),
    MetricSpec(
        "kv_block_utilization", "KV pool block utilization", "ratio",
        "BlockAllocator.utilization at admission failure",
        "Blocks held / blocks in pool. Below 1.0 means the last admission was "
        "refused by the watermark or needed more blocks than remained.",
    ),
    MetricSpec(
        "kv_internal_fragmentation", "KV internal fragmentation", "ratio",
        "allocator block accounting vs realized token counts",
        "1 - (tokens actually held / token slots allocated). The price of block "
        "granularity; it is published rather than netted out of the ratio.",
    ),
    MetricSpec(
        "seq_len_tokens", "realized sequence length", "count",
        "workload generator, after clipping to [1, max_seq]",
        "RAW per-sequence samples. Percentiles are computed at analysis time from "
        "these, never stored pre-aggregated (BENCHMARK_METHODOLOGY.md §5).",
    ),
    MetricSpec(
        "gpu_total_mem_mb", "GPU total memory", "MB",
        "torch.cuda.get_device_properties().total_memory",
        "Device capacity as the driver reports it — distinct from the nominal "
        "marketing figure used as the CLI default. Absent without a GPU.",
    ),
    MetricSpec(
        "gpu_free_mem_mb", "GPU free memory", "MB", "torch.cuda.mem_get_info()[0]",
        "Free VRAM at harness start. Distinct from gpu_total_mem_mb and from "
        "gpu_mem_mb (which is max_memory_allocated).",
    ),
]:
    REGISTRY.register(_spec)


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


def draw_lengths(
    n: int,
    dist: str,
    mean_len: int,
    sigma: float,
    max_seq: int,
    rng: random.Random,
) -> list[int]:
    """
    Draw `n` sequence lengths, clipped to [1, max_seq].

    `lognormal` is the default because docs/BENCHMARK_METHODOLOGY.md §4 commits to
    skewed rather than fixed lengths: "a benchmark using uniform 128-token prompts
    ... has removed every interesting phenomenon it claims to study." For capacity
    specifically, skew matters because a few long sequences consume blocks that
    many short ones would have used, so the mean alone does not determine the
    answer — which is exactly why the measured number and the computed one are
    both reported below and allowed to disagree.

    `mu` is solved so the UNCLIPPED lognormal has mean `mean_len`. Clipping at
    max_seq pulls the realized mean below that, which is why the realized
    distribution is recorded and the requested parameters are not trusted (R14).

    `fixed` exists as a degenerate control: it is the one distribution for which
    the measured answer must equal the computed one exactly, which makes it the
    check that the simulation and the arithmetic agree.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if mean_len <= 0:
        raise ValueError(f"mean_len must be positive, got {mean_len}")

    if dist == "fixed":
        return [min(mean_len, max_seq)] * n
    if dist == "uniform":
        lo, hi = 1, min(2 * mean_len - 1, max_seq)
        return [rng.randint(lo, max(lo, hi)) for _ in range(n)]
    if dist == "lognormal":
        if sigma <= 0:
            raise ValueError(f"sigma must be positive for lognormal, got {sigma}")
        mu = math.log(mean_len) - (sigma**2) / 2.0
        out = []
        for _ in range(n):
            raw = rng.lognormvariate(mu, sigma)
            out.append(max(1, min(max_seq, int(round(raw)))))
        return out
    raise ValueError(f"unknown distribution {dist!r}; expected fixed|uniform|lognormal")


def _percentile(samples: list[float], q: float) -> float:
    """
    Linear-interpolation percentile, stated explicitly per §5 (tools disagree and
    the disagreement is visible in the tail).
    """
    if not samples:
        raise ValueError("no samples")
    xs = sorted(samples)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[int(pos)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def realized_distribution(lengths: list[int], max_seq: int) -> dict[str, Any]:
    """
    What the generator ACTUALLY produced. R14's detection mechanism.

    Includes `clipped_at_max_seq` because clipping is the specific way this
    workload degenerates: crank sigma high enough and a "lognormal mean 256"
    workload becomes a pile of 2048s, at which point the capacity ratio collapses
    toward 1 for a reason that has nothing to do with the allocator.
    """
    fl = [float(x) for x in lengths]
    clipped = sum(1 for x in lengths if x >= max_seq)
    return {
        "n": len(lengths),
        "mean": statistics.fmean(fl),
        "stdev": statistics.pstdev(fl) if len(fl) > 1 else 0.0,
        "min": min(lengths),
        "p50": _percentile(fl, 50),
        "p90": _percentile(fl, 90),
        "p99": _percentile(fl, 99),
        "max": max(lengths),
        "clipped_at_max_seq": clipped,
        "clipped_fraction": clipped / len(lengths),
        "percentile_method": "linear interpolation between order statistics",
    }


# ---------------------------------------------------------------------------
# Config / result
# ---------------------------------------------------------------------------


@dataclass
class CapacityConfig:
    total_vram_bytes: int = A100_40GB_BYTES
    model_weight_bytes: int = LLAMA_3_2_1B_FP16_WEIGHT_BYTES
    activation_headroom_bytes: int = DEFAULT_ACTIVATION_HEADROOM_BYTES
    block_size: int = DEFAULT_BLOCK_SIZE
    max_seq: int = DEFAULT_MAX_SEQ
    shape: ModelKVShape = LLAMA_3_2_1B

    dist: str = "lognormal"
    mean_len: int = 256
    sigma: float = 0.8
    seed: int = 0

    watermark_blocks: int = 0
    max_sequences: int = 200_000
    use_gpu: bool = False
    name: str = "capacity_s1"

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "shape"}
        d["shape"] = self.shape.as_dict()
        return d


@dataclass
class CapacityResult:
    artifact: Artifact
    paged_sequences: int
    contiguous_sequences: int
    ratio: float
    realized: dict[str, Any]
    pool_derivation: list[str] = field(default_factory=list)
    baseline_derivation: list[str] = field(default_factory=list)
    gpu: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GPU probe — reports or abstains, never estimates
# ---------------------------------------------------------------------------


def probe_gpu() -> dict[str, Any]:
    """
    Real VRAM figures from torch.cuda, or an explicit statement that they are
    unavailable and why.

    No fallback estimate. An estimated VRAM figure looks exactly like a measured
    one in a results file, and the whole point of the artifact schema is that a
    reader can tell the difference.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dep, be safe anyway
        return {"available": False, "reason": f"torch import failed: {exc}"}

    if not torch.cuda.is_available():
        return {
            "available": False,
            "reason": "torch.cuda.is_available() is False — no GPU visible to this "
                      "process. VRAM figures are UNAVAILABLE, not estimated.",
            "torch_version": torch.__version__,
        }

    props = torch.cuda.get_device_properties(0)
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "available": True,
        "device_name": props.name,
        "device_count": torch.cuda.device_count(),
        "total_memory_bytes": int(props.total_memory),
        "free_memory_bytes": int(free_b),
        "driver_total_bytes": int(total_b),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


def run_capacity_bench(cfg: CapacityConfig, repo_root: str | Path = REPO_ROOT) -> CapacityResult:
    """
    Admit sequences into a real BlockAllocator until admission fails, then report.

    Admission uses `can_allocate()` — the watermark-respecting path the scheduler
    will use — rather than `allocate()`. Measuring with `allocate()` would report
    a capacity the real admission control would never grant.

    Sequences are never freed during the run. This measures PEAK CONCURRENCY at
    fixed VRAM, which is what S1 claims; a run with retirement would be measuring
    throughput, which is Phase 2's claim and a different benchmark.
    """
    if cfg.use_gpu:
        gpu = probe_gpu()
    else:
        gpu = {"available": False, "reason": "--gpu not requested; GPU not probed."}

    plan = plan_kv_pool(
        total_vram_bytes=cfg.total_vram_bytes,
        model_weight_bytes=cfg.model_weight_bytes,
        activation_headroom_bytes=cfg.activation_headroom_bytes,
        block_size=cfg.block_size,
        shape=cfg.shape,
        notes=["Sized from CLI VRAM budget, not from a measured device."]
        if not gpu.get("available")
        else ["Sized from the CLI VRAM budget; measured device figures are in `gpu`."],
    )
    baseline = plan_contiguous_baseline(
        total_vram_bytes=cfg.total_vram_bytes,
        model_weight_bytes=cfg.model_weight_bytes,
        max_seq=cfg.max_seq,
        activation_headroom_bytes=cfg.activation_headroom_bytes,
        shape=cfg.shape,
    )

    rng = random.Random(cfg.seed)
    alloc = BlockAllocator(
        num_blocks=plan.num_blocks,
        block_size=cfg.block_size,
        watermark_blocks=cfg.watermark_blocks,
    )

    admitted: list[SequenceBlocks] = []
    lengths: list[int] = []
    refused_len: int | None = None
    t0 = time.perf_counter()

    while len(admitted) < cfg.max_sequences:
        (want,) = draw_lengths(1, cfg.dist, cfg.mean_len, cfg.sigma, cfg.max_seq, rng)
        need = math.ceil(want / cfg.block_size)
        if not alloc.can_allocate(need):
            refused_len = want
            break
        seq = SequenceBlocks(alloc, seq_id=len(admitted))
        seq.append(want)
        admitted.append(seq)
        lengths.append(want)
    else:
        refused_len = None

    elapsed = time.perf_counter() - t0
    alloc.check_invariants()  # a capacity number from a corrupt allocator is worthless

    hit_cap = len(admitted) >= cfg.max_sequences
    tokens_held = sum(lengths)
    slots_allocated = alloc.num_used * cfg.block_size
    realized = realized_distribution(lengths, cfg.max_seq) if lengths else {}

    paged_n = len(admitted)
    contig_n = baseline.num_sequences
    ratio = paged_n / contig_n if contig_n else float("nan")

    # The computed prediction at the realized mean, for comparison. Reported as a
    # separate figure, never substituted for the measurement.
    predicted = None
    if realized:
        predicted = capacity_ratio(
            mean_seq_len=max(1, int(round(realized["mean"]))),
            total_vram_bytes=cfg.total_vram_bytes,
            model_weight_bytes=cfg.model_weight_bytes,
            activation_headroom_bytes=cfg.activation_headroom_bytes,
            block_size=cfg.block_size,
            max_seq=cfg.max_seq,
            shape=cfg.shape,
        )

    prov = Provenance.capture(repo_root=repo_root, seed=cfg.seed)
    art = Artifact(
        name=cfg.name,
        provenance=prov,
        config={
            **cfg.as_dict(),
            "kv_pool_plan": plan.as_dict(),
            "contiguous_baseline": baseline.as_dict(),
            "gpu": gpu,
            "admission_path": "BlockAllocator.can_allocate (watermark-respecting)",
            "sequences_retired_during_run": False,
        },
        realized_workload={
            "length_distribution": realized,
            "requested": {
                "dist": cfg.dist,
                "mean_len": cfg.mean_len,
                "sigma": cfg.sigma,
                "max_seq": cfg.max_seq,
            },
            "first_refused_length": refused_len,
            "tokens_held": tokens_held,
            "token_slots_allocated": slots_allocated,
            "hit_max_sequences_cap": hit_cap,
        },
        window={
            # A simulation has no ramp-up or drain, so the window is the whole run.
            # Recorded anyway: the schema requires explicit boundaries for any
            # artifact carrying samples (R11), and "the whole run" is a boundary.
            "first_sequence_index": 0.0,
            "last_sequence_index": float(max(0, paged_n - 1)),
            "wall_seconds": elapsed,
        },
    )

    art.add_samples("seq_len_tokens", [float(x) for x in lengths])
    art.set_scalar("kv_pool_blocks", plan.num_blocks)
    art.set_scalar("concurrent_seqs_paged", paged_n)
    art.set_scalar("concurrent_seqs_contiguous", contig_n)
    art.set_scalar("capacity_ratio_paged_over_contiguous", ratio)
    art.set_scalar("kv_block_utilization", alloc.utilization)
    art.set_scalar(
        "kv_internal_fragmentation",
        1.0 - (tokens_held / slots_allocated) if slots_allocated else 0.0,
    )
    if gpu.get("available"):
        art.set_scalar("gpu_total_mem_mb", gpu["total_memory_bytes"] / MIB)
        art.set_scalar("gpu_free_mem_mb", gpu["free_memory_bytes"] / MIB)

    art.notes.append(
        "MEASURED: sequences admitted through BlockAllocator/SequenceBlocks until "
        "can_allocate() refused. COMPUTED: the contiguous baseline B3, which cannot "
        "be measured because the engine is single-request "
        "(engine/scheduler.py, module docstring)."
    )
    art.notes.append(
        "The ratio is a function of sequence length. It is "
        f"{ratio:.1f}x at realized mean length "
        f"{realized.get('mean', float('nan')):.0f}, and tends to 1x as lengths "
        f"approach max_seq={cfg.max_seq}. Quoting it without this distribution "
        "attached is not a defensible claim."
    )
    if predicted is not None:
        art.notes.append(
            f"Computed prediction at the realized mean ({predicted.mean_seq_len} "
            f"tokens): {predicted.paged_sequences:,} paged sequences, ratio "
            f"{predicted.ratio:.1f}x. Divergence from the measured "
            f"{paged_n:,} / {ratio:.1f}x is the effect of length SKEW, which a "
            "mean-based calculation cannot capture."
        )
        art.config["computed_prediction_at_realized_mean"] = predicted.as_dict()
    if not gpu.get("available"):
        art.notes.append(
            "GPU VRAM figures UNAVAILABLE: " + str(gpu.get("reason"))
            + " The capacity number above is an allocator simulation against a "
            "STATED VRAM budget, not a measurement of this machine."
        )
    if hit_cap:
        art.notes.append(
            f"Run stopped at --max-sequences={cfg.max_sequences} before the pool was "
            "exhausted. The capacity figure is a LOWER BOUND, not a measurement."
        )
    ok, blockers = s1_publishable(prov)
    if not ok:
        art.notes.append("NOT PUBLISHABLE AS S1: " + " | ".join(blockers))

    return CapacityResult(
        artifact=art,
        paged_sequences=paged_n,
        contiguous_sequences=contig_n,
        ratio=ratio,
        realized=realized,
        pool_derivation=plan.derivation(),
        baseline_derivation=baseline.derivation(),
        gpu=gpu,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render(result: CapacityResult, cfg: CapacityConfig) -> str:
    art = result.artifact
    ok, blockers = s1_publishable(art.provenance)
    r = result.realized
    lines = [
        "S1 — concurrent-sequence capacity at fixed VRAM",
        "=" * 72,
        "",
        "KV pool sizing (computed)",
        "-" * 72,
        *result.pool_derivation,
        "",
        "Contiguous baseline B3 (computed)",
        "-" * 72,
        *result.baseline_derivation,
        "",
        "Realized length distribution (MEASURED — R14: not the requested parameters)",
        "-" * 72,
        f"  requested            {cfg.dist}, mean {cfg.mean_len}, sigma {cfg.sigma}, "
        f"clipped to [1, {cfg.max_seq}]",
    ]
    if r:
        lines += [
            f"  realized mean        {r['mean']:.1f}  (stdev {r['stdev']:.1f})",
            f"  realized p50/p90/p99 {r['p50']:.0f} / {r['p90']:.0f} / {r['p99']:.0f}",
            f"  realized min/max     {r['min']} / {r['max']}",
            f"  clipped at max_seq   {r['clipped_at_max_seq']:,} of {r['n']:,} "
            f"({r['clipped_fraction'] * 100:.2f}%)",
        ]
    frag = art.scalars["kv_internal_fragmentation"]
    lines += [
        "",
        "Result",
        "-" * 72,
        f"  paged concurrent sequences       {result.paged_sequences:>10,}   MEASURED "
        "(allocator driven to admission failure)",
        f"  contiguous concurrent sequences  {result.contiguous_sequences:>10,}   COMPUTED "
        "(B3 reserves max_seq per request; engine is single-request)",
        f"  ratio                            {result.ratio:>10.1f}x",
        f"  block utilization at stop        {art.scalars['kv_block_utilization']:>10.4f}",
        f"  internal fragmentation           {frag:>10.4f}   "
        "(token slots allocated but never written)",
        "",
        "GPU",
        "-" * 72,
    ]
    if result.gpu.get("available"):
        lines += [
            f"  device        {result.gpu['device_name']} x{result.gpu['device_count']}",
            f"  total VRAM    {result.gpu['total_memory_bytes'] / GIB:.2f} GiB "
            "(driver-reported)",
            f"  free VRAM     {result.gpu['free_memory_bytes'] / GIB:.2f} GiB at start",
        ]
    else:
        lines += [
            f"  UNAVAILABLE — {result.gpu.get('reason')}",
            "  No VRAM figure is estimated in its place.",
        ]

    lines += [
        "",
        "The claim, stated honestly",
        "-" * 72,
    ]
    if r:
        lines.append(
            f"  {result.ratio:.1f}x concurrent-sequence capacity at fixed VRAM for a "
            f"{cfg.dist} length distribution with realized mean "
            f"{r['mean']:.0f} tokens (p90 {r['p90']:.0f}), against a baseline "
            f"reserving max_seq={cfg.max_seq} per request."
        )
        lines.append(
            "  The ratio is approximately max_seq / realized_padded_length. It is NOT a "
            "property of the allocator alone, and it falls to ~1x when sequences "
            f"actually use all {cfg.max_seq} slots."
        )
    lines += [
        "",
        "Provenance",
        "-" * 72,
        f"  seed              {art.provenance.seed}",
        f"  allocation id     {art.provenance.allocation_id}",
        f"  repo sha          {art.provenance.repo_sha} "
        f"(dirty={art.provenance.repo_dirty})",
        f"  engine            tag={art.provenance.engine_tag} "
        f"sha={art.provenance.engine_sha}",
        f"  publishable as S1 {ok}",
    ]
    for b in blockers:
        lines.append(f"    - {b}")
    return "\n".join(lines)


def render_sweep(rows: list[tuple[int, CapacityResult]], cfg: CapacityConfig) -> str:
    lines = [
        "",
        "Capacity ratio vs realized mean length (MEASURED, one run per point)",
        "=" * 72,
        f"{'requested':>10}  {'realized mean':>14}  {'paged seqs':>12}  "
        f"{'contig seqs':>12}  {'ratio':>8}",
    ]
    for requested, res in rows:
        mean = res.realized.get("mean", float("nan"))
        lines.append(
            f"{requested:>10}  {mean:>14.1f}  {res.paged_sequences:>12,}  "
            f"{res.contiguous_sequences:>12,}  {res.ratio:>7.1f}x"
        )
    lines.append(
        "\nPublish this table, not a single row. A single row is a choice of "
        f"workload; the trend to ~1x at max_seq={cfg.max_seq} is the honest picture."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench/capacity.py",
        description=(
            "S1: measure concurrent-sequence capacity of the paged KV pool against "
            "baseline B3 (contiguous max_seq-per-request), at fixed VRAM. Runs on CPU; "
            "--gpu adds real torch.cuda VRAM figures when a GPU is present."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "The capacity ratio is a function of sequence length. Every result is "
            "reported next to the REALIZED length distribution that produced it."
        ),
    )
    g = p.add_argument_group("memory budget (defaults: Llama 3.2 1B fp16 on a 40GB A100)")
    g.add_argument("--total-vram-gib", type=float, default=A100_40GB_BYTES / GIB,
                   help="Nominal device VRAM. Override with the measured value when known.")
    g.add_argument("--weights-mib", type=float, default=LLAMA_3_2_1B_FP16_WEIGHT_BYTES / MIB,
                   help="Model weight memory; engine measured 2357.1 MB fp16 (BENCHMARKS.md).")
    g.add_argument("--headroom-gib", type=float,
                   default=DEFAULT_ACTIVATION_HEADROOM_BYTES / GIB,
                   help="Activations/workspace/CUDA context reserve. [inference] — measure it.")
    g.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                   help="KV block size in tokens (ARCHITECTURE.md §3.1).")
    g.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ,
                   help="Baseline B3 per-request reservation (engine/scheduler.py:16).")
    g.add_argument("--watermark-blocks", type=int, default=0,
                   help="Admission headroom. 0 measures raw pool capacity.")

    m = p.add_argument_group("model shape")
    m.add_argument("--n-layers", type=int, default=LLAMA_3_2_1B.n_layers)
    m.add_argument("--n-kv-heads", type=int, default=LLAMA_3_2_1B.n_kv_heads,
                   help="KV heads, NOT query heads — the model is GQA.")
    m.add_argument("--head-dim", type=int, default=LLAMA_3_2_1B.head_dim)
    m.add_argument("--dtype-bytes", type=int, default=LLAMA_3_2_1B.dtype_bytes)

    w = p.add_argument_group("workload")
    w.add_argument("--dist", choices=["lognormal", "uniform", "fixed"], default="lognormal",
                   help="Length distribution. Skewed by default (METHODOLOGY §4).")
    w.add_argument("--mean-len", type=int, default=256,
                   help="Target mean sequence length before clipping to max_seq.")
    w.add_argument("--sigma", type=float, default=0.8, help="Lognormal shape parameter.")
    w.add_argument("--seed", type=int, default=0,
                   help="RNG seed. RECORDED IN THE ARTIFACT; a run without one is "
                        "not publishable.")
    w.add_argument("--max-sequences", type=int, default=200_000,
                   help="Safety cap. Hitting it makes the result a lower bound, and "
                        "says so in the artifact.")
    w.add_argument("--sweep", type=str, default=None,
                   help="Comma-separated mean lengths; runs one measurement each and "
                        "prints the ratio-vs-length table.")

    o = p.add_argument_group("output")
    o.add_argument("--gpu", action="store_true",
                   help="Additionally report real torch.cuda VRAM. Without a GPU the "
                        "fields are recorded UNAVAILABLE, never estimated.")
    o.add_argument("--write", action="store_true", help="Write the artifact to --results-dir.")
    o.add_argument("--results-dir", type=str, default=str(REPO_ROOT / "results" / "p1"))
    o.add_argument("--name", type=str, default="capacity_s1", help="Artifact name.")
    o.add_argument("--json", action="store_true",
                   help="Print the artifact JSON to stdout instead of the report.")
    return p


def _cfg_from_args(args: argparse.Namespace, mean_len: int | None = None) -> CapacityConfig:
    return CapacityConfig(
        total_vram_bytes=int(args.total_vram_gib * GIB),
        model_weight_bytes=int(args.weights_mib * MIB),
        activation_headroom_bytes=int(args.headroom_gib * GIB),
        block_size=args.block_size,
        max_seq=args.max_seq,
        shape=ModelKVShape(
            n_layers=args.n_layers,
            n_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            dtype_bytes=args.dtype_bytes,
            name=f"{args.n_layers}L/{args.n_kv_heads}kvh/{args.head_dim}d"
            f"/{args.dtype_bytes}B",
        ),
        dist=args.dist,
        mean_len=mean_len if mean_len is not None else args.mean_len,
        sigma=args.sigma,
        seed=args.seed,
        watermark_blocks=args.watermark_blocks,
        max_sequences=args.max_sequences,
        use_gpu=args.gpu,
        name=args.name if mean_len is None else f"{args.name}_len{mean_len}",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = _cfg_from_args(args)
    result = run_capacity_bench(cfg)

    if args.json:
        print(json.dumps(result.artifact.to_dict(), indent=2, sort_keys=True))
    else:
        print(render(result, cfg))

    if args.sweep:
        try:
            lengths = [int(x) for x in args.sweep.split(",") if x.strip()]
        except ValueError:
            print(f"--sweep expects comma-separated integers, got {args.sweep!r}",
                  file=sys.stderr)
            return 2
        rows = []
        for n in lengths:
            c = _cfg_from_args(args, mean_len=n)
            rows.append((n, run_capacity_bench(c)))
        if not args.json:
            print(render_sweep(rows, cfg))
        if args.write:
            for _, res in rows:
                print(f"wrote {res.artifact.write(args.results_dir)}", file=sys.stderr)

    if args.write:
        path = result.artifact.write(args.results_dir)
        print(f"wrote {path}", file=sys.stderr)
        ok, _ = s1_publishable(result.artifact.provenance)
        if not ok:
            print(
                "NOTE: artifact written but NOT publishable — see publishable_blockers. "
                "It is real data; it may not back a claim.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
