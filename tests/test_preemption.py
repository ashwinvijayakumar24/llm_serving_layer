"""
PHASE 3 — preemption under KV exhaustion. CPU tests, no GPU, no weights.

WHAT IS REAL HERE AND WHAT IS FAKE
----------------------------------
Fake: the model weights. `TinyPagedModel` is 2 layers of random float64 with a
64-token vocabulary.

REAL: everything the preemption path can get wrong. A real `BlockAllocator` with
a deliberately tiny pool, real `SequenceBlocks`, real `PagedTorchBackend` KV
pools, real `build_batch_meta`, real slot mapping, real CSR page tables, real
GPU-shaped swap copies (on CPU tensors), and the real scheduler.

That distinction is the point. A preemption bug is a bug in *bookkeeping* — which
blocks, which tokens, which positions — and none of that needs trained weights to
be wrong. What it needs is a forward pass whose output actually DEPENDS on the
whole KV history, which is why the fake model is a genuine paged attention stack
rather than a stub returning a constant:

  * it attends over every prior position through the backend's page tables, so a
    block restored into the wrong slot changes the answer;
  * it has TWO layers, so a swap that copies only layer 0 changes the answer;
  * it applies RoPE from `meta.positions`, so a resume at the wrong position
    changes the answer;
  * it has 4 query heads over 2 KV heads, so a GQA-shaped copy error changes the
    answer.

A stub model would make every test below pass against a completely broken swap
implementation.

WHY MEMORY PRESSURE IS NOT SIMULATED
------------------------------------
`make_stack(num_blocks=...)` builds a genuinely small pool and the running batch
genuinely exhausts it. `test_the_rig_actually_exhausts_memory` proves it by
turning preemption OFF and asserting the same workload raises `AllocationError`
from the allocator. A forced-pressure test that never forces pressure is the
single most likely way this file could pass while the feature is broken.

The GPU gate (`tests/test_preemption_gpu.py`) runs the same shape of test on real
weights. This file is what makes a failure there diagnosable.
"""

from __future__ import annotations

import math

import pytest
import torch

from serving.backends.paged_torch import PagedTorchBackend
from serving.memory.allocator import AllocationError, BlockAllocator
from serving.memory.block_table import SequenceBlocks
from serving.scheduler.preemption import (
    KVSwapSpace,
    PreemptionPolicy,
    PreemptionStats,
    SwapSpaceFull,
    select_victim,
)
from serving.scheduler.scheduler import (
    Request,
    RequestState,
    Scheduler,
    SchedulerConfig,
)

# Single-threaded so GEMM reduction order is fixed run to run. The assertions
# below are exact token equality; a thread-count-dependent reduction would make
# them flaky for reasons that have nothing to do with preemption.
torch.set_num_threads(1)

BLOCK = 4          # small on purpose: page boundaries are crossed constantly
N_LAYERS = 2
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 8
D_MODEL = N_HEADS * HEAD_DIM
VOCAB = 64
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# the fake model — real paged attention, random weights
# ---------------------------------------------------------------------------


def _rope(t: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Rotary embedding over `(tokens, heads, head_dim)`, indexed by ABSOLUTE
    position.

    Present so that resuming a sequence at the wrong position is detectable. A
    model without positional encoding would produce identical output for a
    sequence resumed one position early, and the swap path's most plausible bug
    would be invisible.
    """
    half = t.shape[-1] // 2
    inv = 10000.0 ** (-torch.arange(half, dtype=DTYPE) / half)
    ang = positions.to(DTYPE)[:, None] * inv[None, :]
    cos, sin = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    a, b = t[..., :half], t[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class TinyPagedModel:
    """
    A `forward_varlen`-compatible model over `PagedTorchBackend`.

    Deliberately structured like `LlamaModelGPU`: embed, per layer project
    Q/K/V, RoPE, append to the paged cache, attend, residual; then logits from
    the last token of each sequence via `meta.last_token_ix`.

    float64 because these tests assert EXACT token equality. The comparison is
    between two runs of the same arithmetic in a different batch order, and the
    thing being tested is bookkeeping, not numerics — running in float64 keeps a
    legitimate last-bit reduction difference from flipping an argmax and turning
    a correctness gate into a flake.
    """

    def __init__(self, seed: int = 20260801):
        g = torch.Generator().manual_seed(seed)

        def rand(*shape):
            return torch.randn(*shape, generator=g, dtype=DTYPE) * 0.4

        self.device = torch.device("cpu")
        self.n_layers = N_LAYERS
        self.emb = rand(VOCAB, D_MODEL)
        self.wq = [rand(D_MODEL, N_HEADS * HEAD_DIM) for _ in range(N_LAYERS)]
        self.wk = [rand(D_MODEL, N_KV_HEADS * HEAD_DIM) for _ in range(N_LAYERS)]
        self.wv = [rand(D_MODEL, N_KV_HEADS * HEAD_DIM) for _ in range(N_LAYERS)]
        self.wo = [rand(N_HEADS * HEAD_DIM, D_MODEL) for _ in range(N_LAYERS)]
        self.head = rand(D_MODEL, VOCAB)
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

    def forward_varlen(self, tokens: torch.Tensor, meta, backend) -> torch.Tensor:
        t = tokens.shape[0]
        x = self.emb[tokens]
        pos = meta.positions
        for layer in range(self.n_layers):
            q = _rope((x @ self.wq[layer]).view(t, N_HEADS, HEAD_DIM), pos)
            k = _rope((x @ self.wk[layer]).view(t, N_KV_HEADS, HEAD_DIM), pos)
            v = (x @ self.wv[layer]).view(t, N_KV_HEADS, HEAD_DIM)
            backend.append_kv(layer, k, v, meta)
            o = backend.attend(q, layer, self.scale, meta)
            x = x + o.reshape(t, N_HEADS * HEAD_DIM) @ self.wo[layer]
        return x[meta.last_token_ix.long()] @ self.head


@pytest.fixture(scope="module")
def model():
    return TinyPagedModel()


# ---------------------------------------------------------------------------
# rig
# ---------------------------------------------------------------------------

ROOMY = 512      # no pressure: the unpreempted reference
TIGHT = 12       # genuinely exhausts under the workload below


def make_stack(model, num_blocks: int, **cfg):
    alloc = BlockAllocator(num_blocks=num_blocks, block_size=BLOCK)
    backend = PagedTorchBackend(
        num_layers=N_LAYERS, num_blocks=num_blocks, block_size=BLOCK,
        n_kv_heads=N_KV_HEADS, n_heads=N_HEADS, head_dim=HEAD_DIM,
        device="cpu", dtype=DTYPE,
    )
    return alloc, backend, Scheduler(model, backend, alloc, SchedulerConfig(**cfg))


# Unequal lengths on purpose: equal-length prompts hit page boundaries on the
# same step, which is the case where a preemption bug is LEAST likely to show.
PROMPTS = {
    "a": [1, 5, 9, 13, 17],
    "b": [2, 4, 8, 16, 32, 33, 34],
    "c": [3, 6, 12, 24],
    "d": [7, 14, 21, 28, 35, 42, 49, 56, 63],
}
MAX_TOKENS = 24


def run(sched, prompts=PROMPTS, max_tokens=MAX_TOKENS, max_steps=4000):
    for rid, ids in prompts.items():
        sched.add_request(
            Request(request_id=rid, prompt_ids=list(ids),
                    max_tokens=max_tokens, ignore_eos=True)
        )
    steps = sched.run_until_idle(max_steps=max_steps)
    return {r.request_id: list(r.output_ids) for r in sched.finished}, steps


@pytest.fixture(scope="module")
def reference(model):
    """Unpreempted greedy output. Everything below is compared to this."""
    _, _, sched = make_stack(model, ROOMY)
    out, _ = run(sched)
    assert sched.preemption.total == 0, "the reference run must not preempt"
    assert set(out) == set(PROMPTS)
    assert all(len(v) == MAX_TOKENS for v in out.values())
    return out


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def assert_identical(got, expected, what):
    failures = []
    for k in sorted(expected):
        if got.get(k) != expected[k]:
            failures.append(
                f"  [{k}] diverges at token {first_divergence(got.get(k, []), expected[k])}\n"
                f"      unpreempted: {expected[k]}\n"
                f"      {what:<11} {got.get(k)}"
            )
    assert not failures, (
        f"{what} output differs from an unpreempted run:\n" + "\n".join(failures) +
        "\nA preemption bug is silent: fluent text, no exception, no metric moves (R3)."
    )


# ---------------------------------------------------------------------------
# victim selection — the R40 surface
# ---------------------------------------------------------------------------


class FakeReq:
    def __init__(self, arrival_seq, preemption_count=0, terminal=False):
        self.arrival_seq = arrival_seq
        self.preemption_count = preemption_count
        self._terminal = terminal

    def is_terminal(self):
        return self._terminal

    def __repr__(self):
        return f"R(arr={self.arrival_seq}, pc={self.preemption_count})"


def test_victim_is_the_newest():
    """
    LIFO. Preempting the newest preserves older requests' progress; FIFO would
    repeatedly punish the request holding the most state.
    """
    reqs = [FakeReq(0), FakeReq(1), FakeReq(2)]
    assert select_victim(reqs, starvation_k=3) is reqs[2]


def test_victim_lifo_is_by_arrival_not_list_order():
    """List position is an implementation detail; arrival order is the policy."""
    reqs = [FakeReq(7), FakeReq(2), FakeReq(4)]
    assert select_victim(reqs, starvation_k=3) is reqs[0]


def test_guard_deprioritises_but_newest_among_the_eligible():
    """Preference ordering: newest FIRST, but only among those under K."""
    reqs = [FakeReq(0, 0), FakeReq(1, 1), FakeReq(2, 3), FakeReq(3, 5)]
    assert select_victim(reqs, starvation_k=3) is reqs[1]


def test_guard_is_a_preference_not_an_exclusion():
    """
    THE R40 CASE. Every running sequence is at or above K.

    The naive "ineligible after K" rule returns None here, victim selection
    fails while the batch is non-empty, and the scheduler can neither step nor
    make room — throughput goes to zero and every request hangs.
    """
    stats = PreemptionStats()
    reqs = [FakeReq(i, preemption_count=9) for i in range(5)]
    victim = select_victim(reqs, starvation_k=3, stats=stats)
    assert victim is reqs[-1], "fallback must still preempt the NEWEST"
    assert stats.starvation_fallbacks == 1
    assert stats.admission_control_alarm is True, (
        "a fallback firing means the watermark is wrong; absorbing it silently is "
        "the failure mode R40 describes"
    )


@pytest.mark.parametrize("n", [1, 2, 5, 32])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_never_returns_no_victim_while_the_batch_is_non_empty(n, k):
    """
    THE INVARIANT. Swept over batch sizes and every preemption count from 0 to
    well past K, including k=0 where EVERY sequence is over the threshold from
    the very first call.
    """
    for count in range(0, k + 4):
        reqs = [FakeReq(i, preemption_count=count) for i in range(n)]
        assert select_victim(reqs, starvation_k=k) is not None, (
            f"no victim with {n} running at preemption_count={count}, K={k}"
        )


def test_returns_none_only_when_nothing_is_preemptible():
    assert select_victim([], starvation_k=3) is None
    assert select_victim([FakeReq(0, terminal=True)], starvation_k=3) is None


def test_terminal_requests_are_not_victims():
    """They are about to be retired; evicting one frees memory that is leaving anyway."""
    live, dead = FakeReq(0), FakeReq(9, terminal=True)
    assert select_victim([live, dead], starvation_k=3) is live


def test_negative_k_rejected():
    with pytest.raises(ValueError):
        select_victim([FakeReq(0)], starvation_k=-1)


# ---------------------------------------------------------------------------
# swap space — the physical copy
# ---------------------------------------------------------------------------


def _fill(backend, blocks, value_base):
    """Write a distinct, position- and layer-dependent value into every slot."""
    for layer in range(backend.num_layers):
        for i, b in enumerate(blocks.block_ids):
            backend.k_pool[layer][b] = value_base + layer * 100 + i
            backend.v_pool[layer][b] = -(value_base + layer * 100 + i)


def _snapshot(backend, blocks):
    return [
        (backend.k_pool[layer][b].clone(), backend.v_pool[layer][b].clone())
        for layer in range(backend.num_layers)
        for b in blocks.block_ids
    ]


def test_swap_round_trip_restores_every_layer_into_different_blocks(model):
    """
    The core of the SWAP policy: out, free, allocate elsewhere, back.

    The restored blocks are DIFFERENT physical blocks — that is the whole point,
    and a swap that only works when it lands back on the same block ids would
    pass a naive test and fail in production on the first fragmented pool.
    """
    alloc, backend, _ = make_stack(model, 16)
    swap = KVSwapSpace(backend)

    src = SequenceBlocks(alloc, seq_id=1)
    src.append(10)                          # 3 blocks at block_size=4
    _fill(backend, src, 7.0)
    before = _snapshot(backend, src)
    original_ids = list(src.block_ids)

    handle = swap.swap_out("r", src)
    src.free()

    # Force different physical blocks: hold the freed ones out of reach, then
    # scribble over them so a restore that silently reused them is detectable.
    decoy = SequenceBlocks(alloc, seq_id=2)
    decoy.append(10)
    _fill(backend, decoy, -99.0)

    dst = SequenceBlocks(alloc, seq_id=3)
    dst.append(handle.num_tokens)
    assert set(dst.block_ids).isdisjoint(original_ids), "test did not fragment the pool"

    swap.swap_in(handle, dst)
    for i, (got, want) in enumerate(zip(_snapshot(backend, dst), before, strict=True)):
        assert torch.equal(got[0], want[0]), f"K differs at (layer, block) index {i}"
        assert torch.equal(got[1], want[1]), f"V differs at (layer, block) index {i}"


def test_swap_copies_all_layers(model):
    """
    A loop that stops at layer 0 restores a sequence whose first layer is right
    and whose rest is another sequence's history. Fluent output, no error.
    """
    alloc, backend, _ = make_stack(model, 16)
    swap = KVSwapSpace(backend)
    src = SequenceBlocks(alloc, seq_id=1)
    src.append(8)
    _fill(backend, src, 3.0)
    handle = swap.swap_out("r", src)
    src.free()

    dst = SequenceBlocks(alloc, seq_id=2)
    dst.append(8)
    for layer in range(backend.num_layers):
        for b in dst.block_ids:
            backend.k_pool[layer][b].fill_(0)
            backend.v_pool[layer][b].fill_(0)
    swap.swap_in(handle, dst)

    for layer in range(backend.num_layers):
        for i, b in enumerate(dst.block_ids):
            assert torch.all(backend.k_pool[layer][b] == 3.0 + layer * 100 + i), (
                f"layer {layer} block {i} was not restored"
            )
            assert torch.all(backend.v_pool[layer][b] == -(3.0 + layer * 100 + i)), (
                f"layer {layer} V was not restored (K and V are separate copies)"
            )


def test_swap_in_rejects_a_block_count_mismatch(model):
    """
    Host index i is written to block_ids[i]. A count mismatch would shift the
    sequence's history by a page, which attention would happily read.
    """
    alloc, backend, _ = make_stack(model, 16)
    swap = KVSwapSpace(backend)
    src = SequenceBlocks(alloc, seq_id=1)
    src.append(9)
    handle = swap.swap_out("r", src)
    src.free()

    wrong = SequenceBlocks(alloc, seq_id=2)
    wrong.append(5)
    with pytest.raises(RuntimeError, match="resume mismatch"):
        swap.swap_in(handle, wrong)

    # And the count check specifically, reached by lying about the token total.
    same_len = SequenceBlocks(alloc, seq_id=3)
    same_len.append(9)
    same_len.block_ids.pop()
    with pytest.raises(RuntimeError, match="blocks allocated"):
        swap.swap_in(handle, same_len)


def test_swap_out_of_freed_blocks_raises(model):
    alloc, backend, _ = make_stack(model, 16)
    swap = KVSwapSpace(backend)
    sb = SequenceBlocks(alloc, seq_id=1)
    sb.append(4)
    sb.free()
    with pytest.raises(RuntimeError, match="freed"):
        swap.swap_out("r", sb)


def test_swap_space_budget_is_enforced_and_release_is_idempotent(model):
    alloc, backend, _ = make_stack(model, 32)
    one_block = KVSwapSpace(backend).bytes_for(1)
    swap = KVSwapSpace(backend, max_bytes=2 * one_block)

    a = SequenceBlocks(alloc, seq_id=1)
    a.append(8)                              # 2 blocks — exactly the budget
    h = swap.swap_out("a", a)
    assert swap.bytes_in_use == 2 * one_block

    b = SequenceBlocks(alloc, seq_id=2)
    b.append(4)
    with pytest.raises(SwapSpaceFull):
        swap.swap_out("b", b)

    swap.release(h)
    assert swap.bytes_in_use == 0
    swap.release(h)                          # idempotent: retirement paths overlap
    assert swap.bytes_in_use == 0
    assert swap.num_swapped == 0


def test_swap_space_rejects_a_backend_without_a_pool():
    with pytest.raises(TypeError, match="paged backend"):
        KVSwapSpace(object())


# ---------------------------------------------------------------------------
# the rig itself
# ---------------------------------------------------------------------------


def test_the_rig_actually_exhausts_memory(model):
    """
    THE TEST THAT MAKES EVERY OTHER TEST IN THIS FILE MEAN SOMETHING.

    With preemption disabled, the same workload on the same tiny pool must blow
    up in `SequenceBlocks.append`. If it does not, the pool is not small enough,
    nothing below is under pressure, and a broken preemption implementation
    would sail through.
    """
    _, _, sched = make_stack(model, TIGHT, enable_preemption=False)
    with pytest.raises(AllocationError):
        run(sched)


def test_pressure_run_really_preempts(model, reference):
    _, _, sched = make_stack(model, TIGHT)
    run(sched)
    assert sched.preemption.total > 0, (
        f"{TIGHT} blocks did not force a single preemption; the pressure is fake"
    )


# ---------------------------------------------------------------------------
# THE GATE — bit-identical output under forced pressure, per policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", [PreemptionPolicy.RECOMPUTE, PreemptionPolicy.SWAP])
def test_output_is_identical_under_forced_pressure(model, reference, policy):
    """
    THE PHASE 3 GATE, per policy, independently.

    Each policy is run against the unpreempted reference on its own. A bug in
    one must not be masked by the other being the default (R3's mitigation).
    """
    _, _, sched = make_stack(model, TIGHT, preemption_policy=policy)
    got, _ = run(sched)

    assert sched.preemption.total > 0, "no preemption occurred; the gate proved nothing"
    assert sched.preemption.by_policy[str(policy)] > 0
    assert_identical(got, reference, str(policy))


def test_the_two_policies_agree_with_each_other(model):
    """
    Both policies against each other, not just against the reference.

    Cheap, and it closes the case where both are wrong in the same direction
    relative to the reference — which cannot happen if they are also compared to
    the reference, but this is the comparison that fails FIRST and most legibly
    when only one of the two has been touched.
    """
    outs = {}
    for policy in (PreemptionPolicy.RECOMPUTE, PreemptionPolicy.SWAP):
        _, _, sched = make_stack(model, TIGHT, preemption_policy=policy)
        outs[policy], _ = run(sched)
        assert sched.preemption.total > 0
    assert outs[PreemptionPolicy.RECOMPUTE] == outs[PreemptionPolicy.SWAP]


@pytest.mark.parametrize("policy", [PreemptionPolicy.RECOMPUTE, PreemptionPolicy.SWAP])
@pytest.mark.parametrize("num_blocks", [12, 13, 14, 16, 20])
def test_identical_across_pressure_levels(model, reference, policy, num_blocks):
    """
    Sweep the pool size. Preemption is a load-dependent code path, and the bug
    that only appears when the FOURTH sequence is evicted is exactly the shape
    of bug R3 describes as surviving to production.
    """
    _, _, sched = make_stack(model, num_blocks, preemption_policy=policy)
    got, _ = run(sched)
    assert_identical(got, reference, f"{policy}@{num_blocks}")


@pytest.mark.parametrize("policy", [PreemptionPolicy.RECOMPUTE, PreemptionPolicy.SWAP])
def test_a_sequence_preempted_many_times_still_completes_correctly(model, reference, policy):
    """
    Repeated eviction of the SAME request. Under LIFO the newest arrival is the
    one that keeps losing, so this is the common case, not the exotic one.
    """
    _, _, sched = make_stack(model, 12, preemption_policy=policy, starvation_k=99)
    got, _ = run(sched)

    worst = max(r.preemption_count for r in sched.finished)
    assert worst >= 2, f"no request was preempted more than once (max {worst})"
    assert_identical(got, reference, f"{policy} repeated")
    assert all(len(v) == MAX_TOKENS for v in got.values()), "a victim lost tokens"


@pytest.mark.parametrize("policy", [PreemptionPolicy.RECOMPUTE, PreemptionPolicy.SWAP])
def test_preemption_during_chunked_prefill(model, policy):
    """
    A victim that has NOT finished its prefill.

    The two policies diverge here and both are easy to get wrong: recompute must
    restart the prompt from token 0 (its generated-so-far is empty, so its
    prefill source is unchanged), while swap must resume at the exact
    `prefill_pos` it was interrupted at, with the KV for the chunks already
    done. Getting either wrong produces a sequence prefilled from the middle of
    its own prompt — fluent, and wrong.
    """
    # Two long-lived hogs arrive first and hold the pool; the request with the
    # 30-token prompt arrives LAST, so under LIFO it is the victim, and with
    # max_prefill_tokens=4 it is still mid-prefill when the pressure hits.
    workload = [
        ("hog_a", [1, 2, 3, 4, 5, 6], 30),
        ("hog_b", [7, 8, 9, 10, 11, 12], 30),
        ("chunky", list(range(20, 50)), 6),
    ]

    def go(num_blocks, **cfg):
        _, _, s = make_stack(model, num_blocks, max_prefill_tokens=4, **cfg)
        mid_prefill_victims = []
        inner = s._preempt

        def spy(req):
            mid_prefill_victims.append(not req.prefill_done)
            return inner(req)

        s._preempt = spy
        for rid, ids, mt in workload:
            s.add_request(Request(request_id=rid, prompt_ids=list(ids),
                                  max_tokens=mt, ignore_eos=True))
        steps = s.run_until_idle(max_steps=4000)
        assert steps < 4000
        out = {r.request_id: list(r.output_ids) for r in s.finished}
        return s, out, mid_prefill_victims

    ref, expected, _ = go(ROOMY)
    assert ref.preemption.total == 0

    sched, got, mid = go(14, preemption_policy=policy)
    assert sched.preemption.total > 0, "no pressure during the chunked prefill"
    assert any(mid), (
        "no victim was mid-prefill; this test degenerated into the ordinary "
        "decode-phase preemption case that is already covered above"
    )
    assert_identical(got, expected, f"{policy} chunked")


# ---------------------------------------------------------------------------
# the starvation guard, end to end
# ---------------------------------------------------------------------------


def test_starvation_guard_does_not_deadlock_when_everyone_is_past_k(model, reference):
    """
    THE R40 SYSTEM TEST. K=0 puts EVERY running sequence at or above the
    threshold from the first preemption onward, so the fallback path is the only
    path victim selection can take.

    The naive exclusion rule hangs here: no victim, no room, no step. What is
    asserted is not "the guard exists" but "the batch still progresses and the
    answers are still right".
    """
    _, _, sched = make_stack(model, TIGHT, starvation_k=0)
    got, steps = run(sched)

    assert steps < 4000, "run_until_idle hit its step cap — the scheduler deadlocked"
    assert len(sched.finished) == len(PROMPTS), "not every request completed"
    assert sched.preemption.starvation_fallbacks > 0, "the fallback path never ran"
    assert sched.preemption.admission_control_alarm is True
    assert_identical(got, reference, "K=0 fallback")


def test_starvation_fallback_is_silent_only_when_it_should_be(model):
    """A run with generous K must NOT raise the admission-control alarm."""
    _, _, sched = make_stack(model, TIGHT, starvation_k=1000)
    run(sched)
    assert sched.preemption.starvation_fallbacks == 0
    assert sched.preemption.admission_control_alarm is False


@pytest.mark.parametrize("k", [0, 1, 2, 5])
def test_batch_progresses_at_every_k(model, reference, k):
    _, _, sched = make_stack(model, TIGHT, starvation_k=k)
    got, steps = run(sched)
    assert steps < 4000
    assert_identical(got, reference, f"K={k}")


# ---------------------------------------------------------------------------
# memory hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", [PreemptionPolicy.RECOMPUTE, PreemptionPolicy.SWAP])
def test_blocks_fully_reclaimed_after_a_preemption_heavy_run(model, policy):
    """
    A leak here is invisible per request and fatal over a benchmark: capacity
    falls until admission fails for no apparent reason. Preemption doubles the
    number of free paths, which is exactly why it is where leaks appear.
    """
    alloc, _, sched = make_stack(model, TIGHT, preemption_policy=policy)
    initial = alloc.num_free
    run(sched)

    alloc.check_invariants()
    assert sched.preemption.total > 0
    assert alloc.num_free == initial, f"leaked {initial - alloc.num_free} blocks"


def test_allocator_invariants_hold_at_every_step(model):
    """
    Checked after EVERY step, not just at the end. A block that is both
    referenced and on the free list is transiently visible and then repaired by
    the next free — an end-of-run check would miss it, and the symptom in
    between is one sequence reading another's KV.
    """
    alloc, _, sched = make_stack(model, TIGHT, preemption_policy=PreemptionPolicy.SWAP)
    for rid, ids in PROMPTS.items():
        sched.add_request(Request(request_id=rid, prompt_ids=list(ids),
                                  max_tokens=MAX_TOKENS, ignore_eos=True))
    steps = 0
    while sched.has_work and steps < 4000:
        sched.step()
        alloc.check_invariants()
        steps += 1
    assert sched.preemption.total > 0
    assert alloc.num_free == alloc.num_blocks


def test_host_swap_space_is_fully_released(model):
    alloc, _, sched = make_stack(model, TIGHT, preemption_policy=PreemptionPolicy.SWAP)
    run(sched)
    assert sched.preemption.bytes_swapped_out > 0
    assert sched._swap is not None
    assert sched._swap.peak_bytes > 0, "nothing was ever resident on the host"
    assert sched._swap.bytes_in_use == 0, "host swap space leaked"
    assert sched._swap.num_swapped == 0


def test_cancelling_a_swapped_request_releases_it(model):
    """
    Cancellation is the path that does not go through normal completion, and a
    swapped request is the one holding memory no allocator invariant covers.
    """
    _, _, sched = make_stack(model, TIGHT, preemption_policy=PreemptionPolicy.SWAP)
    for rid, ids in PROMPTS.items():
        sched.add_request(Request(request_id=rid, prompt_ids=list(ids),
                                  max_tokens=60, ignore_eos=True))

    for _ in range(400):
        sched.step()
        if sched.swapped:
            break
    assert sched.swapped, "never reached a swapped state"

    victim = sched.swapped[0]
    assert sched.cancel(victim.request_id) is True, "cancel() did not see the swapped queue"
    sched.step()
    assert victim.state == RequestState.CANCELLED
    assert victim.swap_handle is None, "cancelled request kept its host memory"
    assert victim not in sched.swapped


def test_has_work_accounts_for_swapped_requests(model):
    """
    If `has_work` misses the swapped queue, `run_until_idle` returns while
    requests are still parked and the server reports them as never finishing.
    """
    _, _, sched = make_stack(model, TIGHT, preemption_policy=PreemptionPolicy.SWAP)
    for rid, ids in PROMPTS.items():
        sched.add_request(Request(request_id=rid, prompt_ids=list(ids),
                                  max_tokens=60, ignore_eos=True))
    for _ in range(400):
        sched.step()
        if sched.swapped:
            break
    assert sched.swapped
    sched.running = [r for r in sched.running if False] or sched.running  # no-op guard
    assert sched.has_work is True


# ---------------------------------------------------------------------------
# trigger, policy mechanics, instrumentation
# ---------------------------------------------------------------------------


def test_trigger_never_enters_a_forward_pass_it_cannot_fund(model):
    """
    The pre-step check, asserted where it matters: at the moment the batch is
    handed to the model, the blocks it needs must already be available.

    Wrapping `forward_varlen` is the only way to observe this — from outside a
    step, the check and the allocation have already both happened.
    """
    _, _, sched = make_stack(model, TIGHT)
    inner = sched.model.forward_varlen
    seen = []

    def checked(tokens, meta, backend):
        seen.append(1)
        return inner(tokens, meta, backend)

    class Wrapper:
        device = sched.model.device
        forward_varlen = staticmethod(checked)

    sched.model = Wrapper()
    # A failure here surfaces as AllocationError out of blocks.append, which is
    # precisely "entered a forward pass the pool could not fund".
    run(sched)
    assert seen, "no forward pass ran at all"
    assert sched.preemption.total > 0


def test_blocks_needed_for_step_is_exact(model):
    """The trigger's arithmetic, checked against what `append()` actually takes."""
    alloc, _, sched = make_stack(model, ROOMY)
    for rid, ids in PROMPTS.items():
        sched.add_request(Request(request_id=rid, prompt_ids=list(ids),
                                  max_tokens=MAX_TOKENS, ignore_eos=True))
    for _ in range(12):
        predicted = sched.blocks_needed_for_step()
        before = alloc.num_free
        sched.step()
        # Retirement inside the step can return blocks, so the identity only
        # holds while nothing retires — assert the direction that matters.
        if sched.last_stats.retired == 0 and sched.last_stats.admitted == 0:
            assert before - alloc.num_free == predicted


def test_recompute_requeues_at_the_front_with_prompt_plus_generated(model):
    """
    The RECOMPUTE contract, inspected directly: re-prefill `prompt + output`,
    ALL of it, and leave `output_ids` untouched so the client's stream is
    unchanged.
    """
    alloc, _, sched = make_stack(model, ROOMY)
    a = Request(request_id="a", prompt_ids=[1, 2, 3], max_tokens=40, ignore_eos=True)
    b = Request(request_id="b", prompt_ids=[4, 5, 6], max_tokens=40, ignore_eos=True)
    sched.add_request(a)
    sched.add_request(b)
    for _ in range(6):
        sched.step()

    generated = list(b.output_ids)
    assert generated, "b never generated anything"
    kv_before = b.blocks.num_tokens

    sched._preempt(b)

    assert sched.waiting[0] is b, "victim must be requeued at the FRONT"
    assert b.state == RequestState.WAITING
    assert b.blocks is None
    assert b.prefill_pos == 0
    assert b.output_ids == generated, "output_ids must never be rewritten"
    assert b.prompt_ids == [4, 5, 6], "prompt_ids must never be rewritten"
    assert b.prefill_ids == [4, 5, 6] + generated, (
        "re-prefill must cover the prompt AND every generated token, including the "
        "one sampled but never fed — otherwise the next sample duplicates it"
    )
    assert b.preemption_count == 1
    assert sched.preemption.tokens_recomputed == kv_before
    assert a.state != RequestState.WAITING, "the older request must be untouched"


def test_victim_selection_in_the_scheduler_is_lifo(model):
    """The newest request is the one that loses its memory."""
    _, _, sched = make_stack(model, 12, starvation_k=99)
    for i, (rid, ids) in enumerate(PROMPTS.items()):
        sched.add_request(Request(request_id=f"{i}{rid}", prompt_ids=list(ids),
                                  max_tokens=40, ignore_eos=True))
    for _ in range(400):
        sched.step()
        preempted = [r for r in sched.waiting + sched.swapped if r.preemption_count]
        if preempted:
            break
    assert preempted, "nothing was preempted"
    newest = max(r.arrival_seq for r in sched.running + preempted)
    assert preempted[0].arrival_seq == newest, (
        "an older request was evicted before the newest; that is FIFO, not LIFO"
    )


def test_swap_space_exhaustion_degrades_to_recompute_and_says_so(model, reference):
    """
    A budget too small for even one sequence. SWAP must fall back to RECOMPUTE
    — still correct — and COUNT it, because an uncounted degradation silently
    changes which policy a head-to-head benchmark is measuring.
    """
    _, _, sched = make_stack(model, TIGHT, preemption_policy=PreemptionPolicy.SWAP,
                             swap_space_bytes=1)
    got, _ = run(sched)

    assert sched.preemption.swap_space_exhausted > 0
    assert sched.preemption.by_policy[str(PreemptionPolicy.RECOMPUTE)] > 0
    assert sched.preemption.by_policy[str(PreemptionPolicy.SWAP)] == 0
    assert_identical(got, reference, "swap->recompute")


def test_metrics_are_populated_per_policy(model):
    """
    Every number Phase 3 claims. The recompute/swap comparison is the phase's
    deliverable, and it is only possible if both costs are actually recorded.
    """
    _, _, rec = make_stack(model, TIGHT, preemption_policy=PreemptionPolicy.RECOMPUTE)
    run(rec)
    assert rec.preemption.tokens_recomputed > 0
    assert rec.preemption.bytes_swapped_out == 0
    assert rec.snapshot()["preemptions_by_policy"]["recompute"] > 0

    _, _, swp = make_stack(model, TIGHT, preemption_policy=PreemptionPolicy.SWAP)
    run(swp)
    s = swp.preemption
    assert s.bytes_swapped_out > 0
    assert s.bytes_swapped_in > 0
    assert s.bytes_swapped_in == s.bytes_swapped_out, (
        "every swapped-out sequence must come back; a mismatch means one was "
        "dropped or restored twice"
    )
    assert s.resumes > 0
    assert s.resume_seconds_max >= 0.0
    assert s.resume_steps_max >= 1, "a resume that took zero steps never stalled"
    assert s.mean_resume_steps > 0

    snap = swp.snapshot()
    for key in ("preemptions_total", "preemptions_by_policy", "tokens_recomputed",
                "bytes_swapped_out", "bytes_swapped_in", "resume_seconds_mean",
                "resume_steps_max", "starvation_fallbacks",
                "admission_control_alarm", "swap_space_exhausted",
                "swap_bytes_in_use", "swap_bytes_peak"):
        assert key in snap, f"snapshot() lost {key}"
    assert snap["preemption_policy"] == "swap"


def test_policy_flag_accepts_a_plain_string(model):
    """A benchmark matrix or CLI flag carries strings, not enum members."""
    _, _, sched = make_stack(model, TIGHT, preemption_policy="swap")
    run(sched)
    assert sched.preemption.by_policy["swap"] > 0


def test_preemption_can_be_disabled_without_touching_anything_else(model, reference):
    """The A/B switch: off, and a roomy pool, must reproduce the reference exactly."""
    _, _, sched = make_stack(model, ROOMY, enable_preemption=False)
    got, _ = run(sched)
    assert sched.preemption.total == 0
    assert_identical(got, reference, "disabled")


def test_a_sequence_larger_than_the_pool_fails_loudly(model):
    """
    The one case preemption cannot fix. Evicting the only running request frees
    the whole pool and changes nothing, so it would be readmitted and arrive
    back here forever — a livelock with zero throughput and no exception.
    """
    _, _, sched = make_stack(model, 3)     # 12 token slots total
    sched.add_request(Request(request_id="huge", prompt_ids=[1, 2, 3, 4],
                              max_tokens=60, ignore_eos=True))
    steps = sched.run_until_idle(max_steps=500)
    assert steps < 500, "livelocked instead of failing"
    assert sched.finished[0].state == RequestState.FAILED
    assert "outgrew the KV pool" in sched.finished[0].error


# ---------------------------------------------------------------------------
# Swap resume must not be gated on the ADMISSION watermark
# ---------------------------------------------------------------------------


def test_swapped_request_resumes_even_when_the_watermark_would_block_admission():
    """
    A swapped request is not new work.

    The watermark exists to stop ADMISSION while the running set might not be
    steppable. A swapped request was already admitted, ran, and had its memory
    taken away; holding it behind the admission watermark starves it. Under
    sustained pressure the watermark is never satisfied, the request parks
    forever, the scheduler keeps stepping, and the client times out.

    Measured before the fix (job 11608501): the swap arm reported 36 preemptions
    at a preemption RATE of 0.0000 — an enormous step count with nothing
    completing — and an e2e p99 of 179,954 ms, which is a timeout rather than a
    latency.

    The rig sets a watermark large enough that `can_allocate` refuses, while
    leaving enough genuinely free blocks for the resume itself.
    """
    from serving.scheduler.preemption import PreemptionPolicy

    model = TinyPagedModel()
    alloc, _, sched = make_stack(
        model, 40, max_batch_size=4, max_prefill_tokens=64,
        preemption_policy=PreemptionPolicy.SWAP,
    )
    req = Request(request_id="v", prompt_ids=list(range(8)), max_tokens=6, ignore_eos=True)
    sched.add_request(req)
    for _ in range(4):
        sched.step()
    assert req.state is not RequestState.WAITING, "victim never started"

    sched._preempt(req)
    assert req in sched.swapped, "request was not swapped out"

    # Raise the watermark AFTER the request is swapped: admission is now blocked
    # while free blocks still exist, which is exactly the situation that starved
    # the resume.
    alloc._watermark = alloc.num_free + 1
    assert not alloc.can_allocate(1), (
        "rig is wrong: the watermark must block admission for this to test anything"
    )
    assert alloc.num_free > 0, "rig is wrong: there must be free blocks to resume into"

    resumed = sched._resume_swapped()
    assert resumed >= 1, (
        "a swapped request did not resume although free blocks existed — it is "
        "being starved by the ADMISSION watermark, which does not apply to it"
    )
    assert req not in sched.swapped
