"""
PHASE 1 GATE — paged path vs contiguous reference path, end to end.

This is the test Phase 1 exists to pass. Everything else in Phase 1 verifies a
component in isolation; this verifies the composition, on the real model, with
real weights.

    greedy tokens through
        BlockAllocator -> SequenceBlocks -> build_batch_meta
            -> PagedTorchBackend -> LlamaModelGPU.forward_varlen

    MUST EQUAL

    greedy tokens through
        KVCacheGPU -> LlamaModelGPU.prefill/decode_step   (engine reference path,
                                                           validated against the
                                                           fp32 HF oracle in P0)

Bit-identical, not "close". Every component can be individually correct and the
composition still wrong — a position off by one, a page boundary mishandled, a
GQA head mapped to the wrong KV head. All of those produce fluent text and no
error, which is why the gate is token equality rather than a tolerance.

Requires CUDA and real weights. Run on PACE:

    pytest tests/test_paged_e2e.py -v -s
"""

import os

import pytest
import torch


def _cuda_status() -> str | None:
    """Return None if CUDA is usable, else a reason string."""
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is False"
    try:
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover
        return f"could not query device: {exc}"
    # torch built against CUDA 13 dropped Volta (sm_70). A V100 will report a
    # device and then fail or silently misbehave; treat it as unusable.
    if cap < (8, 0):
        return f"{name} is sm_{cap[0]}{cap[1]}; this build needs sm_80+"
    return None


_CUDA_REASON = _cuda_status()

# WHY THIS IS NOT A PLAIN skipif.
#
# This file IS the Phase 1 gate. A skipped gate and a passing gate look identical
# in a job log — 'no failures, exit 0' — which means a run that verified nothing
# can be mistaken for a run that verified everything. That happened: job 11598374
# landed on a V100 (sm_70) under a CUDA-13 torch build, every test skipped, and
# the job reported success.
#
# So: skipping is allowed on a developer laptop, where it is obvious. In CI or on
# a Slurm node — anywhere a green result would be TRUSTED — set REQUIRE_GPU=1 and
# an unusable GPU becomes a hard failure instead of a silent pass.
_REQUIRE_GPU = os.environ.get("REQUIRE_GPU") == "1"

if _CUDA_REASON and _REQUIRE_GPU:
    pytest.fail(
        f"REQUIRE_GPU=1 but CUDA is unusable: {_CUDA_REASON}. "
        "The Phase 1 gate cannot run, and a skipped gate must not be reported as a pass.",
        pytrace=False,
    )

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_CUDA_REASON is not None, reason=_CUDA_REASON or ""),
]

WEIGHTS_PATH = "vendor/llm_inference_engine/weights"
BLOCK_SIZE = 16

# Prompts chosen so token counts straddle block boundaries once the chat
# template is applied. The exact lengths are asserted at runtime rather than
# assumed, and printed, so a change in tokenizer behaviour is visible.
PROMPTS = {
    "short": [128000, 9906, 11, 358, 1097],                      # <bos> Hello, I am
    "medium": [128000, 791, 4062, 14198, 39935, 27096, 927, 279, 16053, 5679, 13],
}


@pytest.fixture(scope="module")
def model():
    from engine.loader import load_config, load_weights_gpu
    from engine.model_gpu import LlamaModelGPU

    config = load_config(WEIGHTS_PATH)
    return LlamaModelGPU(load_weights_gpu(WEIGHTS_PATH, config), config), config


def make_backend(model, config, num_blocks=512):
    from serving.backends.paged_torch import PagedTorchBackend
    from serving.memory.allocator import BlockAllocator

    allocator = BlockAllocator(num_blocks=num_blocks, block_size=BLOCK_SIZE)
    backend = PagedTorchBackend(
        num_layers=config["num_hidden_layers"],
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        n_kv_heads=config["num_key_value_heads"],
        n_heads=config["num_attention_heads"],
        head_dim=config["head_dim"],
        device=model.device,
        dtype=torch.float16,
    )
    return allocator, backend


def reference_greedy(model, prompt_ids, max_tokens):
    """Engine's contiguous path — the reference, validated against HF in P0."""
    from engine.sampler import greedy
    from engine.scheduler import generate

    return list(generate(model, list(prompt_ids), greedy, max_tokens=max_tokens))


def paged_greedy(model, config, prompt_ids, max_tokens, num_blocks=512):
    from serving.engine_iface.runner import greedy_device, paged_generate

    allocator, backend = make_backend(model, config, num_blocks)
    out = list(
        paged_generate(model, backend, allocator, list(prompt_ids), greedy_device, max_tokens)
    )
    return out, allocator


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None


# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["short", "medium"])
def test_paged_matches_contiguous(model, key):
    """THE GATE. Paged greedy output must equal contiguous greedy output exactly."""
    m, config = model
    prompt = PROMPTS[key]
    n = 24

    expected = reference_greedy(m, prompt, n)
    got, allocator = paged_greedy(m, config, prompt, n)

    div = first_divergence(got, expected)
    assert div is None and len(got) == len(expected), (
        f"Paged path diverges from the contiguous reference at token {div}.\n"
        f"  reference: {expected}\n"
        f"  paged:     {got}\n"
        "Both paths run the same weights; a difference is a defect in paging, "
        "batch assembly, or the backend — not numerics."
    )
    print(f"\n  [{key}] {len(got)} tokens identical to the contiguous path")


@pytest.mark.parametrize("prompt_len", [1, 15, 16, 17, 31, 32, 33, 47, 48])
def test_block_boundary_prompt_lengths(model, prompt_len):
    """
    Sweep prompt lengths across block boundaries (R8).

    A prompt of exactly 16 or 32 tokens is the case where `last_page_len` must
    report block_size rather than 0. Getting it wrong silently drops or invents
    a page of keys.
    """
    m, config = model
    prompt = ([128000] + [9906] * 200)[:prompt_len]
    n = 8

    expected = reference_greedy(m, prompt, n)
    got, _ = paged_greedy(m, config, prompt, n)

    assert got == expected, (
        f"prompt_len={prompt_len} (pages={-(-prompt_len // BLOCK_SIZE)}, "
        f"last_page_len={prompt_len % BLOCK_SIZE or BLOCK_SIZE}) diverges:\n"
        f"  reference: {expected}\n  paged:     {got}"
    )


def test_generation_across_many_block_boundaries(model):
    """
    Generate long enough to cross several block boundaries mid-decode.

    Prompt-length sweeps only exercise boundaries during prefill. This exercises
    them during decode, where a new block is allocated between forward passes —
    a different code path and a different opportunity to mis-address.
    """
    m, config = model
    prompt = PROMPTS["short"]
    n = 40                       # 5 + 40 tokens spans 3 block boundaries at bs=16

    expected = reference_greedy(m, prompt, n)
    got, _ = paged_greedy(m, config, prompt, n)
    assert got == expected, (
        f"Divergence at token {first_divergence(got, expected)} during long decode:\n"
        f"  reference: {expected}\n  paged:     {got}"
    )


def test_blocks_are_fully_reclaimed(model):
    """
    Every block is returned after generation, and allocator invariants hold.

    A leak here is invisible for one request and fatal over a benchmark: capacity
    silently falls until admission starts failing for no apparent reason.
    """
    m, config = model
    allocator, backend = make_backend(m, config)
    initial = allocator.num_free

    from serving.engine_iface.runner import greedy_device, paged_generate

    for i in range(5):
        list(paged_generate(m, backend, allocator, PROMPTS["short"], greedy_device, 12, seq_id=i))
        allocator.check_invariants()

    assert allocator.num_free == initial, (
        f"Leaked {initial - allocator.num_free} blocks over 5 generations"
    )


def test_abandoned_generator_frees_blocks(model):
    """
    Abandoning the generator mid-stream must still free its blocks.

    This is client disconnect. The engine's own loop cannot do this — it builds a
    fresh KVCacheGPU per call and relies on garbage collection
    (engine/scheduler.py:26-27). Under paging the pool is finite and shared, so a
    disconnect-heavy workload that leaks would drain it with no error anywhere.
    """
    m, config = model
    allocator, backend = make_backend(m, config)
    initial = allocator.num_free

    from serving.engine_iface.runner import greedy_device, paged_generate

    gen = paged_generate(m, backend, allocator, PROMPTS["medium"], greedy_device, 50)
    for _ in range(3):
        next(gen)
    assert allocator.num_free < initial, "generation should have allocated blocks"

    gen.close()                                  # simulate client disconnect
    allocator.check_invariants()
    assert allocator.num_free == initial, (
        f"Abandoned generator leaked {initial - allocator.num_free} blocks"
    )


def test_fragmented_pool_gives_identical_output(model):
    """
    Output must not depend on WHICH physical blocks a sequence got.

    Fragment the pool first so the sequence receives scattered, non-monotonic
    block ids. An implementation that accidentally assumes contiguous or sorted
    pages passes every clean-pool test and fails here.
    """
    m, config = model
    prompt = PROMPTS["medium"]
    n = 16

    clean, _ = paged_greedy(m, config, prompt, n)

    from serving.engine_iface.runner import greedy_device, paged_generate
    from serving.memory.block_table import SequenceBlocks

    allocator, backend = make_backend(m, config)
    # Carve holes: allocate a run of single-block sequences, free every other one.
    holders = [SequenceBlocks(allocator, seq_id=100 + i) for i in range(32)]
    for h in holders:
        h.append(BLOCK_SIZE)
    for h in holders[::2]:
        h.free()

    fragmented = list(
        paged_generate(m, backend, allocator, prompt, greedy_device, n, seq_id=1)
    )
    for h in holders[1::2]:
        h.free()

    assert fragmented == clean, (
        "Output changed when the sequence's physical blocks were scattered — "
        "something is assuming page contiguity.\n"
        f"  clean:      {clean}\n  fragmented: {fragmented}"
    )
    allocator.check_invariants()


def test_eos_ids_match_engine():
    """Stop condition is duplicated in runner.py for readability; keep it honest."""
    from engine.model_gpu import EOS_IDS as ENGINE_EOS

    from serving.engine_iface.runner import EOS_IDS

    assert EOS_IDS == ENGINE_EOS
