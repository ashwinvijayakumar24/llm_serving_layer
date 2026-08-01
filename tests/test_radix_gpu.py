"""
PHASE 4 GATE — radix prefix cache on the real model.

Greedy output must be IDENTICAL with the cache on and with the cache off, on
real weights, across all four prefix-sharing structures from
docs/BENCHMARK_METHODOLOGY.md §4 — plus the adversarial block-boundary sweep,
where the divergence point is walked across every offset inside a block.

WHY THE CPU SUITE IS NOT ENOUGH
-------------------------------
`tests/test_radix.py` proves the block table the cache hands over is the right
one, using a fake model that dereferences `slot_mapping` and the CSR page table
the way a kernel does. That covers the bookkeeping exhaustively and cheaply.

What it cannot cover is whether reused KV, computed under one request's forward
pass and attended over by another's, actually produces the same numbers. That is
a claim about the attention kernel and about fp16 reduction order, and only real
weights can settle it. Phase 1 proved the paged path at batch 1; Phase 2 proved
batch invariance. This file adds the third: PROVENANCE invariance — the same
token attends over the same KV whether that KV was computed for this request or
inherited from another.

WHAT COUNTS AS A FAILURE
------------------------
Exact token equality. Not a tolerance. Reuse does not change any arithmetic —
the reused KV is bit-identical to what this request would have computed, because
it was computed from the same token ids at the same positions by the same
kernel — so there is no reduction-order argument available here. A divergence
means the cache handed over the wrong block, which is R6, and it is a bug, not a
threshold to relax.

    REQUIRE_GPU=1 pytest tests/test_radix_gpu.py -v -s
"""

import os

import pytest
import torch

from bench.workloads.generator import LengthSpec, WorkloadConfig, generate
from serving.cache.radix import RadixCache
from serving.memory.allocator import BlockAllocator
from serving.scheduler.scheduler import Request, Scheduler, SchedulerConfig


def _cuda_status() -> str | None:
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is False"
    cap = torch.cuda.get_device_capability(0)
    if cap < (8, 0):
        return f"{torch.cuda.get_device_name(0)} is sm_{cap[0]}{cap[1]}; needs sm_80+"
    return None


_CUDA_REASON = _cuda_status()

# Same discipline as the Phase 1 and Phase 2 gates: a skipped gate and a passing
# gate are indistinguishable in a job log. Job 11598374 skipped all 16 P1 tests
# on a V100 and exited 0. REQUIRE_GPU=1 makes an unusable GPU a hard COLLECTION
# failure — never a skip, because a prefix-cache gate that silently did not run
# is exactly how "faster and wrong" ships.
if _CUDA_REASON and os.environ.get("REQUIRE_GPU") == "1":
    pytest.fail(
        f"REQUIRE_GPU=1 but CUDA is unusable: {_CUDA_REASON}. "
        "A skipped prefix-cache gate must not be reported as a pass.",
        pytrace=False,
    )

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_CUDA_REASON is not None, reason=_CUDA_REASON or ""),
]

WEIGHTS_PATH = os.environ.get("LLM_WEIGHTS_PATH", "vendor/llm_inference_engine/weights")
BLOCK_SIZE = 16
NUM_BLOCKS = 2048

# Synthetic ids, stated as such (methodology §4). They live well inside Llama
# 3.2's vocabulary, and the point of this gate is KV PROVENANCE, not text
# quality: what must match is the token stream produced with the cache against
# the token stream produced without it, on identical inputs.
VOCAB_LIMIT = 32_000


@pytest.fixture(scope="module")
def model():
    from engine.loader import load_config, load_weights_gpu
    from engine.model_gpu import LlamaModelGPU

    config = load_config(WEIGHTS_PATH)
    return LlamaModelGPU(load_weights_gpu(WEIGHTS_PATH, config), config), config


def make_stack(model, config, *, cache: bool, **cfg):
    """
    Identical stack either way; `cache` is the ONLY difference.

    That is the whole design of the A/B: same kernels, same allocator, same
    scheduler, one flag. A separate cache-less code path would make any
    divergence unattributable.
    """
    from serving.backends.paged_torch import PagedTorchBackend

    allocator = BlockAllocator(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    backend = PagedTorchBackend(
        num_layers=config["num_hidden_layers"], num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE, n_kv_heads=config["num_key_value_heads"],
        n_heads=config["num_attention_heads"], head_dim=config["head_dim"],
        device=model.device, dtype=torch.float16,
    )
    rc = RadixCache(allocator, enabled=True) if cache else None
    sched = Scheduler(model, backend, allocator, SchedulerConfig(**cfg), prefix_cache=rc)
    return allocator, rc, sched


def run_staged(sched, groups, max_tokens):
    """
    Submit groups in order, draining between them.

    Requests admitted in the SAME step all miss — nothing has been computed for
    any of them yet. Staging is what makes the second group actually exercise
    reuse rather than silently testing the cold path twice.
    """
    out = {}
    for group in groups:
        for rid, ids in group.items():
            sched.add_request(
                Request(request_id=rid, prompt_ids=list(ids),
                        max_tokens=max_tokens, ignore_eos=True)
            )
        sched.run_until_idle()
        out.update({r.request_id: list(r.output_ids) for r in sched.finished})
    return out


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None


def compare(model, config, groups, max_tokens=8, **cfg):
    """Run the same staged workload with the cache off, then on. Returns both."""
    m, c = model, config
    _, _, off = make_stack(m, c, cache=False, **cfg)
    expected = run_staged(off, groups, max_tokens)

    alloc, rc, on = make_stack(m, c, cache=True, **cfg)
    got = run_staged(on, groups, max_tokens)

    alloc.check_invariants()
    rc.check_invariants()
    return expected, got, rc, alloc


def assert_identical(expected, got, label):
    assert set(got) == set(expected), f"{label}: request set changed"
    failures = []
    for k in sorted(expected):
        div = first_divergence(got[k], expected[k])
        if div is not None or len(got[k]) != len(expected[k]):
            failures.append(
                f"  [{k}] diverges at token {div}\n"
                f"      cache off: {expected[k]}\n"
                f"      cache on:  {got[k]}"
            )
    assert not failures, (
        f"{label}: the prefix cache changed the output.\n" + "\n".join(failures) +
        "\nThis is R6 — the system is now faster AND wrong, and every metric "
        "it publishes improved on the way."
    )


def workload(structure, **kw):
    cfg = WorkloadConfig(
        n_requests=kw.pop("n_requests", 12),
        structure=structure,
        sharing_rate=kw.pop("sharing_rate", 0.9),
        block_size=BLOCK_SIZE,
        seed=kw.pop("seed", 3),
        prompt=LengthSpec(dist="lognormal", mean=96, sigma=0.4, min_len=32, max_len=192),
        output=LengthSpec(dist="fixed", mean=8),
        shared_prefix_tokens=kw.pop("shared_prefix_tokens", 64),
        adversarial_common_tokens=kw.pop("adversarial_common_tokens", 64),
        n_shared_prefixes=kw.pop("n_shared_prefixes", 2),
        max_turns=kw.pop("max_turns", 3),
        vocab_size=VOCAB_LIMIT,
        name=f"radix-gpu-{structure}",
    )
    return generate(cfg)


# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "structure", ["zero", "system", "conversational", "adversarial"]
)
def test_cache_correctness_at_uniform_chunk_shape(model, structure):
    """
    THE CACHE'S OWN CORRECTNESS GATE — one block per prefill step, so every
    forward pass computes an IDENTICALLY SHAPED GEMM in both arms.

    WHY NOT JUST BATCH SIZE 1
    -------------------------
    Batch 1 was tried first and failed identically. It does not equalise shapes,
    because a cache HIT changes how many tokens the request prefills: cache-off
    computes M=105 in one pass, cache-on computes M=41 for the uncached
    remainder. The engine's `linear()` takes that as its `M` dimension, cuBLAS
    selects kernels and split-K by shape, and fp16 reduction order follows. The
    shape difference is INTRINSIC TO CACHING, not to batching.

    Capping prefill at exactly one block per step fixes that. Cache-off prefills
    105 tokens as 16,16,16,16,16,16,9; cache-on with 64 tokens cached prefills
    16,16,9. Every individual GEMM has the same M in both arms — the cache
    simply performs FEWER of them. Block-granularity reuse is what makes this
    exact: the cached amount is always a multiple of the block size, so even the
    partial tail chunk aligns.

    That is what isolates the cache. Any divergence here cannot be reduction
    order, because the reductions are identically shaped; it can only be the
    cache serving wrong KV. Independently confirmed during the Phase 4
    investigation, where forcing one-block chunks drove drift to exactly 0 at
    every position of every request.

    WHAT THIS DOES NOT COVER: correctness under production chunk sizes and
    concurrent batching, where shapes necessarily differ. That remains blocked
    on the engine's numerics (results/p4/FINDING_batch_shape_numerics.md) and is
    reported as blocked rather than quietly passed.
    """
    m, config = model
    wl = workload(structure)
    groups = [{r.request_id: list(r.token_ids)} for r in wl.requests]

    # ONE BLOCK PER PREFILL STEP in both arms. This is the whole mechanism.
    expected, got, rc, _ = compare(m, config, groups, max_tokens=8,
                                   max_batch_size=1, max_prefill_tokens=BLOCK_SIZE)
    assert_identical(expected, got, f"{structure} @ uniform {BLOCK_SIZE}-token chunks")

    snap = rc.snapshot()
    assert snap["blocks_reused"] > 0 or structure == "zero", (
        f"{structure}: no blocks were reused, so this run proves nothing about the cache"
    )
    print(
        f"\n  {structure} @ uniform chunks: identical output, block hit rate "
        f"{snap['block_hit_rate']:.3f} "
        f"(reused {snap['blocks_reused']} of {snap['blocks_required']})"
    )


@pytest.mark.parametrize(
    "structure", ["zero", "system", "conversational", "adversarial"]
)
def test_greedy_output_identical_with_cache_on_and_off(model, structure):
    """
    THE GATE, on all four sharing structures.

    All four, not just the flattering ones. `zero` is the control and must show
    no hits at all; `conversational` is the structure that most favours a radix
    cache and its hit rate means nothing without that label attached (§7). A
    correctness gate run only on the favourable structure would exercise mostly
    the deep-hit path and almost none of the partial-hit path.
    """
    m, config = model
    wl = workload(structure)
    # One request per group: staged submission is what makes reuse happen at all.
    groups = [{r.request_id: list(r.token_ids)} for r in wl.requests]

    expected, got, rc, _ = compare(m, config, groups, max_tokens=8,
                                   max_batch_size=8, max_prefill_tokens=128)
    assert_identical(expected, got, structure)

    snap = rc.snapshot()
    print(
        f"\n  {structure}: block hit rate {snap['block_hit_rate']:.3f} "
        f"(reused {snap['blocks_reused']} of {snap['blocks_required']}), "
        f"mean partial-hit depth {snap['mean_partial_hit_depth']:.2f}, "
        f"nodes {snap['node_count']} depth {snap['max_node_depth']}, "
        f"evictions {snap['evictions']}"
        f"  [oracle block sharing "
        f"{wl.realized['sharing']['realized_block_sharing_rate']:.3f}]"
    )

    if structure == "zero":
        assert snap["blocks_reused"] == 0, (
            "the CONTROL structure reported hits on real weights. No hit rate "
            "from any other structure is interpretable until this is 0."
        )
    else:
        assert snap["blocks_reused"] > 0, (
            f"{structure} shares prefixes by construction but the cache found "
            "none — the gate passed without exercising reuse"
        )


@pytest.mark.parametrize("offset", list(range(BLOCK_SIZE)))
def test_adversarial_block_boundary_sweep(model, offset):
    """
    THE ADVERSARIAL WORKLOAD, on real weights, one offset per case.

    Request A holds a common region plus one straddling block. Request B agrees
    with A for `common + offset` tokens and then diverges. Reuse must truncate
    to the last FULLY matching block for every offset, so B's output must be
    byte-identical to the no-cache run at all sixteen positions.

    Offset 0 is the clean boundary. Offsets 1..15 are the ones that look
    reusable and are not, and they are where an off-by-one would show up as
    fluent, wrong text with a BETTER hit rate than the correct implementation.
    """
    m, config = model
    rng = torch.Generator().manual_seed(1000 + offset)

    def ids(n, lo=200, hi=VOCAB_LIMIT):
        return torch.randint(lo, hi, (n,), generator=rng).tolist()

    common = ids(64)                 # 4 whole blocks
    straddle = ids(BLOCK_SIZE)
    groups = [
        {"a": common + straddle + ids(48)},
        {"b": common + straddle[:offset] + ids(64, lo=100, hi=190)},
    ]

    expected, got, rc, _ = compare(m, config, groups, max_tokens=8,
                                   max_batch_size=4, max_prefill_tokens=64)
    assert_identical(expected, got, f"adversarial offset {offset}")

    assert rc.stats.blocks_reused == 4, (
        f"offset {offset}: reused {rc.stats.blocks_reused} blocks, expected 4. "
        f"B shares {64 + offset} tokens with A, but only 64 of them fill whole "
        "blocks; the straddling block must be recomputed."
    )
    if offset:
        assert rc.stats.truncated_tokens == offset


def test_chunked_prefill_mixed_with_decodes_under_a_cache_hit(model):
    """
    A chunked prefill, a cache hit, and decoding sequences in one batch.

    The three things Phase 4 changed, interacting. A chunk boundary and a
    cache-hit boundary are different things: the cache sets `prefill_pos` once,
    at admission, to a block multiple; chunking then advances it against a token
    budget that has nothing to do with the prompt's content. Conflating them
    prefills from the wrong offset, which is fluent and wrong.
    """
    m, config = model
    rng = torch.Generator().manual_seed(77)
    common = torch.randint(200, VOCAB_LIMIT, (128,), generator=rng).tolist()
    tail_a = torch.randint(200, VOCAB_LIMIT, (73,), generator=rng).tolist()
    tail_b = torch.randint(200, VOCAB_LIMIT, (91,), generator=rng).tolist()
    short = torch.randint(200, VOCAB_LIMIT, (11,), generator=rng).tolist()

    groups = [
        {"warm": common + tail_a},
        {"long": common + tail_b, "short": short},
    ]
    # A cap that is neither a multiple of the block size nor aligned to the
    # cached prefix, so no boundary coincides with another by accident.
    expected, got, rc, alloc = compare(m, config, groups, max_tokens=10,
                                       max_batch_size=8, max_prefill_tokens=40)
    assert_identical(expected, got, "chunked prefill under a cache hit")
    assert rc.stats.blocks_reused >= 8, "the long request did not reuse the common prefix"
    alloc.check_invariants()


def test_no_leak_and_no_live_block_evicted_under_pressure(model):
    """
    R7 on real weights: a pool small enough that eviction runs, invariants
    checked after every step, and the free list back to its initial count once
    the cache is cleared.
    """
    m, config = model
    from serving.backends.paged_torch import PagedTorchBackend

    num_blocks = 192
    allocator = BlockAllocator(num_blocks=num_blocks, block_size=BLOCK_SIZE)
    backend = PagedTorchBackend(
        num_layers=config["num_hidden_layers"], num_blocks=num_blocks,
        block_size=BLOCK_SIZE, n_kv_heads=config["num_key_value_heads"],
        n_heads=config["num_attention_heads"], head_dim=config["head_dim"],
        device=m.device, dtype=torch.float16,
    )
    rc = RadixCache(allocator, max_cached_blocks=48)
    sched = Scheduler(m, backend, allocator, SchedulerConfig(max_batch_size=4,
                      max_prefill_tokens=64), prefix_cache=rc)
    initial = allocator.num_free

    rng = torch.Generator().manual_seed(4242)
    common = torch.randint(200, VOCAB_LIMIT, (96,), generator=rng).tolist()
    for i in range(16):
        tail = torch.randint(200, VOCAB_LIMIT, (48,), generator=rng).tolist()
        sched.add_request(
            Request(request_id=f"r{i}", prompt_ids=common + tail,
                    max_tokens=8, ignore_eos=True)
        )

    steps = 0
    while sched.has_work and steps < 20_000:
        sched.step()
        allocator.check_invariants()
        rc.check_invariants()
        steps += 1

    assert len(sched.finished) == 16
    assert rc.stats.evictions > 0, "the pool was never tight enough to evict"
    rc.clear()
    assert allocator.num_free == initial, (
        f"leaked {initial - allocator.num_free} blocks across a cache-heavy run"
    )
    allocator.check_invariants()
    print(f"\n  {steps} steps, {rc.stats.evictions} evictions, no live block freed")
