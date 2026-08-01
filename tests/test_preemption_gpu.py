"""
PHASE 3 GATE — greedy output under FORCED memory pressure must be BIT-IDENTICAL
to an unpreempted run, per request, for BOTH policies independently.

WHY THIS IS THE MOST IMPORTANT TEST IN THE PROJECT
--------------------------------------------------
A preemption bug produces PLAUSIBLE TEXT. Output stays fluent, no metric
degrades, no exception is raised, and it is load-dependent — it appears only when
the pool is genuinely exhausted, which casual testing never reaches. It also
invalidates every other correctness claim in the system simultaneously: once the
scheduler can silently corrupt a sequence, the batch-invariance result, the
throughput number and the goodput curve are all describing a system computing
something else (docs/RISK_REGISTER.md R3).

So the gate is exact token equality. Not a tolerance, not a similarity score. If
reduction order genuinely makes bit-identity unachievable, that is a RESULT TO
PUBLISH — quantify the divergence rate and the conditions — not a gate to loosen.

BOTH POLICIES, INDEPENDENTLY
----------------------------
Each policy is compared to the unpreempted reference on its own, and then to the
other. R3's mitigation clause is explicit that a bug in one must not be masked by
the other being the default, and the two fail in different directions: recompute
fails by re-prefilling the wrong token range, swap fails by copying an incomplete
or misordered set of blocks.

PRESSURE IS REAL, NOT SIMULATED
-------------------------------
The pool is constructed small (`TIGHT` blocks) and the running batch genuinely
exhausts it. `test_the_rig_actually_exhausts_memory` proves that by disabling
preemption and asserting the same workload raises `AllocationError` out of the
allocator. Without that test, a gate that silently never preempted would pass.

    REQUIRE_GPU=1 pytest tests/test_preemption_gpu.py -v -s
"""

import os

import pytest
import torch

from serving.memory.allocator import AllocationError, BlockAllocator
from serving.scheduler.preemption import PreemptionPolicy
from serving.scheduler.scheduler import (
    Request,
    RequestState,
    Scheduler,
    SchedulerConfig,
)


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
# failure — this runs at import time, before any test is selected, so there is no
# subset of `-k` that can turn it back into a skip.
if _CUDA_REASON and os.environ.get("REQUIRE_GPU") == "1":
    pytest.fail(
        f"REQUIRE_GPU=1 but CUDA is unusable: {_CUDA_REASON}. "
        "A skipped preemption gate must not be reported as a pass — it is the one "
        "test standing between this system and silent, load-dependent corruption.",
        pytrace=False,
    )

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_CUDA_REASON is not None, reason=_CUDA_REASON or ""),
]

WEIGHTS_PATH = os.environ.get("LLM_WEIGHTS_PATH", "vendor/llm_inference_engine/weights")
BLOCK_SIZE = 16

# Deliberately unequal lengths, as in the Phase 2 gate. Equal-length prompts
# cross page boundaries on the same step, which is the case where a preemption
# bug is LEAST likely to show.
PROMPTS = {
    "a": [128000, 9906, 11, 358, 1097],                                        # 5
    "b": [128000, 791, 4062, 14198, 39935, 27096, 927, 279, 16053, 5679, 13],  # 11
    "c": [128000, 3923, 374, 279, 6864, 315, 9822, 30],                        # 8
    "d": [128000] + [9906] * 35,                                               # 36
}
MAX_TOKENS = 24

# Peak demand for the workload above is ceil(29/16) + ceil(35/16) + ceil(32/16)
# + ceil(60/16) = 2 + 3 + 2 + 4 = 11 blocks. TIGHT is below that by construction,
# so the batch cannot be held in memory all at once and preemption is forced —
# not by a mocked trigger, by the allocator actually running out.
ROOMY = 1024
TIGHT = 8

# `ignore_eos` so output length is CONTROLLED. A run whose sequences stop at
# different points would change how much memory the workload demands, and the
# preempted and unpreempted runs would no longer be the same experiment.


@pytest.fixture(scope="module")
def model():
    from engine.loader import load_config, load_weights_gpu
    from engine.model_gpu import LlamaModelGPU

    config = load_config(WEIGHTS_PATH)
    return LlamaModelGPU(load_weights_gpu(WEIGHTS_PATH, config), config), config


def make_stack(model, config, num_blocks, **cfg):
    from serving.backends.paged_torch import PagedTorchBackend

    allocator = BlockAllocator(num_blocks=num_blocks, block_size=BLOCK_SIZE)
    backend = PagedTorchBackend(
        num_layers=config["num_hidden_layers"], num_blocks=num_blocks,
        block_size=BLOCK_SIZE, n_kv_heads=config["num_key_value_heads"],
        n_heads=config["num_attention_heads"], head_dim=config["head_dim"],
        device=model.device, dtype=torch.float16,
    )
    return allocator, backend, Scheduler(model, backend, allocator, SchedulerConfig(**cfg))


def run(sched, prompts=None, max_tokens=MAX_TOKENS, max_steps=4000):
    for rid, ids in (prompts or PROMPTS).items():
        sched.add_request(
            Request(request_id=rid, prompt_ids=list(ids),
                    max_tokens=max_tokens, ignore_eos=True)
        )
    steps = sched.run_until_idle(max_steps=max_steps)
    assert steps < max_steps, (
        f"run_until_idle hit its {max_steps}-step cap. The scheduler is not making "
        "progress — preemption is thrashing or deadlocked."
    )
    return {r.request_id: list(r.output_ids) for r in sched.finished}, steps


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def assert_identical(got, expected, what):
    assert set(got) == set(expected), (
        f"{what}: request set changed — got {sorted(got)}, expected {sorted(expected)}"
    )
    failures = []
    for k in sorted(expected):
        if got[k] != expected[k]:
            failures.append(
                f"  [{k}] diverges at token {first_divergence(got[k], expected[k])}\n"
                f"      unpreempted: {expected[k]}\n"
                f"      {what:<11} {got[k]}"
            )
    assert not failures, (
        f"PREEMPTION CHANGED THE OUTPUT ({what}):\n" + "\n".join(failures) +
        "\n\nThis is R3. The text is fluent, no metric moved, and every correctness "
        "claim in the system is now false."
    )


@pytest.fixture(scope="module")
def reference(model):
    """
    The unpreempted run. Same scheduler, same config, same workload — the ONLY
    difference from every run below is the size of the block pool, which is what
    makes this a comparison of one variable.
    """
    m, config = model
    _, _, sched = make_stack(m, config, ROOMY)
    out, _ = run(sched)
    assert sched.preemption.total == 0, "the reference run preempted; it is not a reference"
    assert all(len(v) == MAX_TOKENS for v in out.values())
    return out


# ---------------------------------------------------------------------------
# the rig must actually force pressure
# ---------------------------------------------------------------------------


def test_the_rig_actually_exhausts_memory(model):
    """
    THE TEST THAT MAKES THE GATE MEAN SOMETHING.

    With preemption disabled, this workload on this pool must blow up inside
    `SequenceBlocks.append`. If it does not, `TIGHT` is too large, nothing below
    is under pressure, and a completely broken preemption implementation would
    pass every remaining test in this file.
    """
    m, config = model
    _, _, sched = make_stack(m, config, TIGHT, enable_preemption=False)
    with pytest.raises(AllocationError):
        run(sched)


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", [PreemptionPolicy.RECOMPUTE, PreemptionPolicy.SWAP])
def test_bit_identical_under_forced_pressure(model, reference, policy):
    """THE PHASE 3 GATE. Each policy against the unpreempted run, on its own."""
    m, config = model
    _, _, sched = make_stack(m, config, TIGHT, preemption_policy=policy)
    got, steps = run(sched)

    assert sched.preemption.total > 0, (
        f"{TIGHT} blocks did not force a single preemption. The gate proved nothing."
    )
    assert sched.preemption.by_policy[str(policy)] > 0, (
        f"policy={policy} was configured but never used — the flag is not plumbed"
    )
    assert_identical(got, reference, str(policy))
    print(
        f"\n  {policy}: {sched.preemption.total} preemptions over {steps} steps, "
        f"{sched.preemption.tokens_recomputed} tokens recomputed, "
        f"{sched.preemption.bytes_swapped_out / 1e6:.2f} MB swapped out, "
        f"mean resume {sched.preemption.mean_resume_seconds * 1e3:.2f} ms / "
        f"{sched.preemption.mean_resume_steps:.1f} steps — output bit-identical"
    )


def test_the_two_policies_produce_identical_output(model):
    """
    Recompute and swap against EACH OTHER.

    They reach the same tokens by completely different mechanisms — one
    recomputes KV from scratch in a prefill, the other restores the exact bytes
    the original decode wrote. Agreement between them is independent evidence
    that neither is quietly reconstructing a different history.
    """
    m, config = model
    outs, totals = {}, {}
    for policy in (PreemptionPolicy.RECOMPUTE, PreemptionPolicy.SWAP):
        _, _, sched = make_stack(m, config, TIGHT, preemption_policy=policy)
        outs[policy], _ = run(sched)
        totals[policy] = sched.preemption.total
        assert totals[policy] > 0

    rec, swp = outs[PreemptionPolicy.RECOMPUTE], outs[PreemptionPolicy.SWAP]
    for k in sorted(rec):
        assert rec[k] == swp[k], (
            f"[{k}] recompute and swap disagree at token "
            f"{first_divergence(rec[k], swp[k])}\n"
            f"      recompute: {rec[k]}\n      swap:      {swp[k]}"
        )
    print(f"\n  recompute {totals[PreemptionPolicy.RECOMPUTE]} preemptions, "
          f"swap {totals[PreemptionPolicy.SWAP]} — identical output")


@pytest.mark.parametrize("policy", [PreemptionPolicy.RECOMPUTE, PreemptionPolicy.SWAP])
@pytest.mark.parametrize("num_blocks", [6, 8, 10, 12])
def test_identical_across_pressure_levels(model, reference, policy, num_blocks):
    """
    Sweep the pool size. Preemption is load-dependent, and the bug that only
    appears when the third or fourth sequence is evicted is precisely the shape
    R3 describes as surviving to production.
    """
    m, config = model
    _, _, sched = make_stack(m, config, num_blocks, preemption_policy=policy)
    got, _ = run(sched)
    assert_identical(got, reference, f"{policy}@{num_blocks}")


@pytest.mark.parametrize("policy", [PreemptionPolicy.RECOMPUTE, PreemptionPolicy.SWAP])
def test_a_sequence_preempted_multiple_times_still_completes_correctly(model, reference, policy):
    """
    Repeated eviction of the SAME request.

    Under LIFO the newest arrival keeps losing, so this is the common case, not
    the exotic one. Each round trip is another chance to duplicate or drop a
    token, and the errors compound silently.
    """
    m, config = model
    _, _, sched = make_stack(m, config, 8, preemption_policy=policy, starvation_k=99)

    # STAGGERED ARRIVALS, and the reason is a real property of the scheduler,
    # not test convenience.
    #
    # `_can_admit` accounts for `blocks_needed_for_step`, which is the
    # anti-livelock guard: without it, admit -> preempt-newest -> requeue-at-front
    # -> admit loops forever. A consequence is that a victim is NOT readmitted
    # while the pressure that evicted it persists, so a single pressure wave can
    # only preempt a given request ONCE.
    #
    # Being preempted twice therefore requires two separate waves: the victim
    # resumes when room appears, and new arrivals then take that room away
    # again. That is also the realistic shape — sustained load, not one burst.
    first = {"a": PROMPTS["a"], "b": PROMPTS["b"]}
    for rid, ids in first.items():
        sched.add_request(Request(request_id=rid, prompt_ids=list(ids),
                                  max_tokens=MAX_TOKENS, ignore_eos=True))
    for extra in ({"c": PROMPTS["c"]}, {"d": PROMPTS["d"]},
                                  {"e": PROMPTS["b"]}, {"f": PROMPTS["d"]}):
        for _ in range(3):
            sched.step()
        for rid, ids in extra.items():
            sched.add_request(Request(request_id=rid, prompt_ids=list(ids),
                                      max_tokens=MAX_TOKENS, ignore_eos=True))
    steps = sched.run_until_idle(max_steps=8000)
    assert steps < 8000, "scheduler stopped making progress under repeated waves"
    got = {r.request_id: list(r.output_ids) for r in sched.finished}

    worst = max(r.preemption_count for r in sched.finished)
    assert worst >= 2, (
        f"no request was preempted more than once (max {worst}). Either the "
        "waves did not overlap or the pool is too roomy — this test proves "
        "nothing about repeated eviction unless it actually happens."
    )
    assert all(len(v) == MAX_TOKENS for v in got.values()), (
        "a repeatedly preempted request lost or gained tokens: "
        f"{ {k: len(v) for k, v in got.items()} }"
    )
    # Reference is recomputed for THIS request set, roomy pool, no preemption.
    _, _, ref_sched = make_stack(m, config, ROOMY)
    all_prompts = {"a": PROMPTS["a"], "b": PROMPTS["b"], "c": PROMPTS["c"],
                   "d": PROMPTS["d"], "e": PROMPTS["b"], "f": PROMPTS["d"]}
    ref_out, _ = run(ref_sched, prompts=all_prompts)
    assert ref_sched.preemption.total == 0, "the reference run must not preempt"
    assert_identical(got, ref_out, f"{policy} x{worst}")
    print(f"\n  {policy}: one request survived {worst} preemptions, output unchanged")


@pytest.mark.parametrize("policy", [PreemptionPolicy.RECOMPUTE, PreemptionPolicy.SWAP])
def test_preemption_during_a_chunked_prefill(model, policy):
    """
    A victim that has NOT finished its prefill.

    The policies diverge here and both are easy to get wrong: recompute must
    restart the prompt from token 0 (nothing has been generated, so its prefill
    source is unchanged), while swap must resume at the exact `prefill_pos` it
    was interrupted at, with the KV for the chunks already done still valid.
    Getting either wrong prefills a sequence from the middle of its own prompt —
    fluent, and wrong.

    The long prompt arrives LAST so that LIFO selects it, and
    `max_prefill_tokens` is capped well below its length so it is still mid-
    prefill when the pressure arrives.
    """
    m, config = model
    workload = [
        ("hog_a", [128000, 9906, 11, 358, 1097], 40),
        ("hog_b", [128000, 3923, 374, 279, 6864, 315, 9822, 30], 40),
        ("chunky", [128000] + [9906] * 95, 8),        # 96 tokens, chunks of 16
    ]

    def go(num_blocks, **cfg):
        _, _, s = make_stack(m, config, num_blocks, max_prefill_tokens=16, **cfg)
        mid_prefill = []
        inner = s._preempt

        def spy(req):
            mid_prefill.append(not req.prefill_done)
            return inner(req)

        s._preempt = spy
        # ORDER MATTERS, and the original order could not work. The long prompt
        # was added LAST so LIFO would select it — but under pressure a request
        # that has not been ADMITTED is never preempted, it just queues. It has
        # to be admitted and mid-prefill BEFORE the pressure arrives.
        #
        # So: admit the chunky prompt first, step until it is partway through
        # its prefill (max_prefill_tokens=16 against a 96-token prompt, so a few
        # steps), then add the hogs. Now it is the resident sequence that the
        # new pressure evicts, still mid-prefill.
        rid, ids, mt = workload[-1]
        s.add_request(Request(request_id=rid, prompt_ids=list(ids),
                              max_tokens=mt, ignore_eos=True))
        for _ in range(3):
            s.step()
        for rid, ids, mt in workload[:-1]:
            s.add_request(Request(request_id=rid, prompt_ids=list(ids),
                                  max_tokens=mt, ignore_eos=True))
        steps = s.run_until_idle(max_steps=4000)
        assert steps < 4000, "deadlocked during a chunked prefill"
        return s, {r.request_id: list(r.output_ids) for r in s.finished}, mid_prefill

    ref, expected, _ = go(ROOMY)
    assert ref.preemption.total == 0

    sched, got, mid = go(7, preemption_policy=policy)
    assert sched.preemption.total > 0, "no pressure arose during the chunked prefill"
    assert any(mid), (
        "no victim was mid-prefill; this degenerated into the decode-phase case "
        "already covered above and proves nothing about chunked prefill"
    )
    assert_identical(got, expected, f"{policy} chunked")


# ---------------------------------------------------------------------------
# starvation guard (R40)
# ---------------------------------------------------------------------------


def test_starvation_guard_does_not_deadlock_and_the_batch_progresses(model, reference):
    """
    THE R40 GATE. `starvation_k=0` puts EVERY running sequence at or above the
    threshold from the first preemption onward, so the fallback is the only path
    victim selection can take.

    The naive "ineligible after K" rule hangs here: no victim, no room, no step,
    throughput zero, every request waiting forever. What is asserted is not that
    the guard exists but that the batch still makes forward progress and the
    answers are still right.
    """
    m, config = model
    _, _, sched = make_stack(m, config, TIGHT, starvation_k=0)
    got, steps = run(sched)

    assert len(sched.finished) == len(PROMPTS), (
        f"only {len(sched.finished)} of {len(PROMPTS)} requests completed"
    )
    assert sched.preemption.starvation_fallbacks > 0, (
        "the fallback path never ran, so the deadlock case was never exercised"
    )
    assert sched.preemption.admission_control_alarm is True, (
        "the fallback fired and the alarm did not — a silently absorbed fallback "
        "is exactly the failure mode R40 names"
    )
    assert_identical(got, reference, "K=0 fallback")
    print(f"\n  K=0: {sched.preemption.starvation_fallbacks} fallback firings over "
          f"{steps} steps, no deadlock, output bit-identical")


@pytest.mark.parametrize("k", [0, 1, 3])
def test_batch_progresses_at_every_k(model, reference, k):
    m, config = model
    _, _, sched = make_stack(m, config, TIGHT, starvation_k=k)
    got, _ = run(sched)
    assert_identical(got, reference, f"K={k}")


# ---------------------------------------------------------------------------
# memory hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", [PreemptionPolicy.RECOMPUTE, PreemptionPolicy.SWAP])
def test_blocks_fully_reclaimed_after_a_preemption_heavy_run(model, policy):
    """
    A leak is invisible per request and fatal over a benchmark: capacity falls
    until admission starts failing for no apparent reason. Preemption doubles
    the number of paths that free memory, which is why it is where leaks appear.
    """
    m, config = model
    allocator, _, sched = make_stack(m, config, TIGHT, preemption_policy=policy)
    initial = allocator.num_free
    run(sched)

    allocator.check_invariants()
    assert sched.preemption.total > 0
    assert allocator.num_free == initial, (
        f"leaked {initial - allocator.num_free} of {allocator.num_blocks} blocks"
    )
    if sched._swap is not None:
        assert sched._swap.bytes_in_use == 0, "host swap space leaked"


def test_allocator_invariants_hold_after_every_step(model):
    """
    Checked after EVERY step, not once at the end. A block that is both
    referenced and on the free list is transiently visible and then repaired by
    the next free — an end-of-run check misses it, and in between, one sequence
    is reading another sequence's KV.
    """
    m, config = model
    allocator, _, sched = make_stack(
        m, config, TIGHT, preemption_policy=PreemptionPolicy.SWAP
    )
    for rid, ids in PROMPTS.items():
        sched.add_request(Request(request_id=rid, prompt_ids=list(ids),
                                  max_tokens=MAX_TOKENS, ignore_eos=True))
    steps = 0
    while sched.has_work and steps < 4000:
        sched.step()
        allocator.check_invariants()
        steps += 1

    assert steps < 4000, "deadlocked"
    assert sched.preemption.total > 0
    assert allocator.num_free == allocator.num_blocks


def test_cancelling_a_swapped_request_releases_its_host_memory(model):
    """
    Cancellation does not go through normal completion, and a swapped request
    holds memory no allocator invariant covers.
    """
    m, config = model
    _, _, sched = make_stack(m, config, TIGHT, preemption_policy=PreemptionPolicy.SWAP)
    for rid, ids in PROMPTS.items():
        sched.add_request(Request(request_id=rid, prompt_ids=list(ids),
                                  max_tokens=200, ignore_eos=True))

    for _ in range(500):
        sched.step()
        if sched.swapped:
            break
    assert sched.swapped, "never reached a swapped state"

    victim = sched.swapped[0]
    assert sched.cancel(victim.request_id) is True
    sched.step()
    assert victim.state == RequestState.CANCELLED
    assert victim.swap_handle is None
    assert victim not in sched.swapped


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------


def test_metrics_distinguish_the_two_policies(model):
    """
    The Phase 3 deliverable is a head-to-head, which is only possible if each
    policy's cost is actually recorded in its own unit: tokens recomputed for
    one, bytes moved and resume latency for the other.
    """
    m, config = model

    _, _, rec = make_stack(m, config, TIGHT, preemption_policy=PreemptionPolicy.RECOMPUTE)
    run(rec)
    assert rec.preemption.tokens_recomputed > 0
    assert rec.preemption.bytes_swapped_out == 0

    _, _, swp = make_stack(m, config, TIGHT, preemption_policy=PreemptionPolicy.SWAP)
    run(swp)
    s = swp.preemption
    assert s.bytes_swapped_out > 0
    assert s.bytes_swapped_in == s.bytes_swapped_out, (
        "every swapped-out sequence must come back exactly once; a mismatch means "
        "one was dropped or restored twice"
    )
    assert s.resumes > 0
    assert s.resume_seconds_max > 0.0

    print(
        f"\n  recompute: {rec.preemption.total} preemptions, "
        f"{rec.preemption.tokens_recomputed} tokens recomputed"
        f"\n  swap:      {s.total} preemptions, {s.bytes_swapped_out / 1e6:.2f} MB out / "
        f"{s.bytes_swapped_in / 1e6:.2f} MB in, "
        f"mean resume {s.mean_resume_seconds * 1e3:.2f} ms"
    )
