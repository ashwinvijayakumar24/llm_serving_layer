"""
PHASE 2 GATE — batch invariance.

A prompt must produce the SAME greedy output whether it runs alone or inside a
mixed batch alongside other sequences at different positions.

WHY THIS IS THE HARD GATE
-------------------------
Everything the serving layer claims about throughput is a claim that batching
changed *when* work happened, not *what* it produced. If batched output differs
from single-sequence output, the throughput number describes a system computing
something else.

And this failure is invisible without a test built for it:

  * every correctness test in the engine is batch-1 (tests/test_forward.py,
    tests/test_decode.py), so nothing upstream catches it;
  * divergence may appear only for SPECIFIC batch compositions — a long
    sequence beside short ones, a prefill chunk mixed with decodes — so a naive
    "run two together" test passes while the real case fails;
  * the output stays fluent. There is no crash, no NaN, no metric that moves.

Phase 1 verified the paged path at batch 1 (job 11598444, 16/16). This file is
the first thing in the project to run n_seqs > 1 on real weights.

WHAT COUNTS AS A FAILURE HERE
-----------------------------
Exact token equality. Not a tolerance. If reduction order genuinely makes
bit-identity unachievable, that is a RESULT TO PUBLISH — quantify the
divergence rate and the conditions — not a gate to loosen (R4's mitigation
clause).

    REQUIRE_GPU=1 pytest tests/test_batch_invariance.py -v -s
"""

import os

import pytest
import torch

from serving.engine_iface.runner import greedy_device, paged_generate
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

# Same discipline as the Phase 1 gate: a skipped gate and a passing gate are
# indistinguishable in a job log. Job 11598374 skipped all 16 P1 tests on a V100
# and exited 0. REQUIRE_GPU=1 makes an unusable GPU a hard failure.
if _CUDA_REASON and os.environ.get("REQUIRE_GPU") == "1":
    pytest.fail(
        f"REQUIRE_GPU=1 but CUDA is unusable: {_CUDA_REASON}. "
        "A skipped batch-invariance gate must not be reported as a pass.",
        pytrace=False,
    )

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_CUDA_REASON is not None, reason=_CUDA_REASON or ""),
]

WEIGHTS_PATH = os.environ.get("LLM_WEIGHTS_PATH", "vendor/llm_inference_engine/weights")
BLOCK_SIZE = 16

# Deliberately unequal lengths. Equal-length prompts are the case where a
# batching bug is LEAST likely to show, because every sequence hits the same
# page boundaries at the same step.
PROMPTS = {
    "a": [128000, 9906, 11, 358, 1097],                                        # 5
    "b": [128000, 791, 4062, 14198, 39935, 27096, 927, 279, 16053, 5679, 13],  # 11
    "c": [128000, 3923, 374, 279, 6864, 315, 9822, 30],                        # 8
    "d": [128000] + [9906] * 35,                                               # 36 — spans 3 pages
}


@pytest.fixture(scope="module")
def model():
    from engine.loader import load_config, load_weights_gpu
    from engine.model_gpu import LlamaModelGPU

    config = load_config(WEIGHTS_PATH)
    return LlamaModelGPU(load_weights_gpu(WEIGHTS_PATH, config), config), config


def make_stack(model, config, num_blocks=1024, **cfg):
    from serving.backends.paged_torch import PagedTorchBackend

    allocator = BlockAllocator(num_blocks=num_blocks, block_size=BLOCK_SIZE)
    backend = PagedTorchBackend(
        num_layers=config["num_hidden_layers"], num_blocks=num_blocks, block_size=BLOCK_SIZE,
        n_kv_heads=config["num_key_value_heads"], n_heads=config["num_attention_heads"],
        head_dim=config["head_dim"], device=model.device, dtype=torch.float16,
    )
    return allocator, backend, Scheduler(model, backend, allocator, SchedulerConfig(**cfg))


def solo(model, config, prompt, max_tokens):
    """Batch-1 reference: the path Phase 1 proved equals the contiguous engine."""
    allocator, backend, _ = make_stack(model, config)
    return list(paged_generate(model, backend, allocator, prompt, greedy_device, max_tokens))


def batched(sched, prompts: dict[str, list[int]], max_tokens) -> dict[str, list[int]]:
    for rid, ids in prompts.items():
        sched.add_request(Request(request_id=rid, prompt_ids=list(ids), max_tokens=max_tokens))
    sched.run_until_idle()
    return {r.request_id: r.output_ids for r in sched.finished}


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None


# --------------------------------------------------------------------------


def test_batch_invariance_mixed_lengths(model):
    """THE GATE. Four prompts of different lengths, alone vs batched together."""
    m, config = model
    n = 16

    expected = {k: solo(m, config, v, n) for k, v in PROMPTS.items()}
    _, _, sched = make_stack(m, config)
    got = batched(sched, PROMPTS, n)

    assert set(got) == set(expected)
    failures = []
    for k in sorted(expected):
        div = first_divergence(got[k], expected[k])
        if div is not None or len(got[k]) != len(expected[k]):
            failures.append(
                f"  [{k}] diverges at token {div}\n"
                f"      solo:    {expected[k]}\n"
                f"      batched: {got[k]}"
            )
    assert not failures, (
        "Batched output differs from single-sequence output:\n" + "\n".join(failures) +
        "\nThe throughput claim describes a system computing something else."
    )
    print(f"\n  {len(PROMPTS)} sequences, {n} tokens each — batched == solo")


@pytest.mark.parametrize("batch_size", [2, 3, 4])
def test_invariance_across_batch_sizes(model, batch_size):
    """
    The same prompt must give the same output at every batch size.

    Catches bugs whose effect depends on batch composition — a mask built from
    the wrong sequence's length, an index computed off `n_seqs`.
    """
    m, config = model
    n = 12
    keys = list(PROMPTS)[:batch_size]
    target = keys[0]

    expected = solo(m, config, PROMPTS[target], n)
    _, _, sched = make_stack(m, config)
    got = batched(sched, {k: PROMPTS[k] for k in keys}, n)

    assert got[target] == expected, (
        f"'{target}' changed at batch_size={batch_size} "
        f"(diverges at {first_divergence(got[target], expected)})"
    )


def test_prefill_chunk_mixed_with_decodes(model):
    """
    A chunked prefill sharing a batch with decoding sequences.

    This is the batch shape continuous batching exists to produce, and the one
    where `cu_query_lens` and the causal mask do the most work: query_lens are
    ragged, one sequence contributes many tokens while others contribute one.
    """
    m, config = model
    n = 12
    long_prompt = [128000] + [9906] * 79      # 80 tokens, forced into chunks

    expected_long = solo(m, config, long_prompt, n)
    expected_short = solo(m, config, PROMPTS["a"], n)

    # Cap below the long prompt so it MUST span several steps alongside decodes.
    _, _, sched = make_stack(m, config, max_prefill_tokens=32)
    got = batched(sched, {"long": long_prompt, "short": PROMPTS["a"]}, n)

    assert got["long"] == expected_long, (
        f"chunked prefill diverged at token {first_divergence(got['long'], expected_long)}"
    )
    assert got["short"] == expected_short, "decoding sequence corrupted by a concurrent prefill"


def test_staggered_arrival(model):
    """
    Requests joining an already-running batch mid-flight.

    Continuous batching's whole point, and a distinct code path from admitting
    everything at step 0: the new sequence starts prefill while others are
    already deep into decode.
    """
    m, config = model
    n = 14
    expected = {k: solo(m, config, PROMPTS[k], n) for k in ("a", "b", "c")}

    _, _, sched = make_stack(m, config)
    sched.add_request(Request(request_id="a", prompt_ids=PROMPTS["a"], max_tokens=n))
    for _ in range(4):
        sched.step()
    sched.add_request(Request(request_id="b", prompt_ids=PROMPTS["b"], max_tokens=n))
    for _ in range(3):
        sched.step()
    sched.add_request(Request(request_id="c", prompt_ids=PROMPTS["c"], max_tokens=n))
    sched.run_until_idle()

    got = {r.request_id: r.output_ids for r in sched.finished}
    for k in expected:
        assert got[k] == expected[k], (
            f"'{k}' diverged when arriving mid-batch at token "
            f"{first_divergence(got[k], expected[k])}"
        )


def test_batch_frees_all_blocks(model):
    """No leak across a full batched run, and allocator invariants hold."""
    m, config = model
    allocator, _, sched = make_stack(m, config)
    initial = allocator.num_free

    batched(sched, PROMPTS, 10)

    allocator.check_invariants()
    assert allocator.num_free == initial, f"leaked {initial - allocator.num_free} blocks"


def test_cancellation_frees_blocks_and_spares_others(model):
    """
    Cancelling mid-batch frees that request's blocks and leaves the rest intact.

    Cancellation is the path that does NOT go through normal completion, which
    is exactly why it is the one that leaks.
    """
    m, config = model
    allocator, _, sched = make_stack(m, config)
    initial = allocator.num_free

    expected_b = solo(m, config, PROMPTS["b"], 12)

    sched.add_request(Request(request_id="a", prompt_ids=PROMPTS["a"], max_tokens=40))
    sched.add_request(Request(request_id="b", prompt_ids=PROMPTS["b"], max_tokens=12))
    for _ in range(3):
        sched.step()

    assert sched.cancel("a") is True
    sched.run_until_idle()

    allocator.check_invariants()
    assert allocator.num_free == initial, "cancellation leaked blocks"
    got = {r.request_id: r.output_ids for r in sched.finished}
    assert got["b"] == expected_b, "cancelling one request corrupted another"


def test_throughput_scales_with_batch_size(model):
    """
    Sanity check that batching does something, before any number is published.

    NOT a benchmark — one process, no warmup, no repetition, shared node. It
    asserts a direction, not a magnitude: total tokens per forward pass must
    rise with batch size. If it does not, continuous batching is bookkeeping
    with no throughput behind it and the Phase 2 claim is empty.
    """
    m, config = model
    n = 10
    per_step = {}
    for bs in (1, 4):
        keys = (list(PROMPTS) * 4)[:bs]
        _, _, sched = make_stack(m, config)
        for i, k in enumerate(keys):
            sched.add_request(Request(request_id=f"{k}{i}", prompt_ids=PROMPTS[k], max_tokens=n))
        steps = sched.run_until_idle()
        produced = sum(len(r.output_ids) for r in sched.finished)
        per_step[bs] = produced / steps
        print(f"\n  batch {bs}: {produced} tokens / {steps} steps = {per_step[bs]:.2f} tok/step")

    assert per_step[4] > per_step[1], (
        f"tokens per forward pass did not rise with batch size: {per_step}. "
        "Batching is not actually batching."
    )
