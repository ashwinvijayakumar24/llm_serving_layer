"""
R9 GATE — FlashInferBackend vs PagedTorchBackend, differentially.

WHY THIS FILE IS THE ONLY DEFENCE THAT WORKS
--------------------------------------------
A layout or addressing misunderstanding in FlashInfer is a HIGH-severity SILENT
risk (docs/RISK_REGISTER.md R9). NHD read as HND, a CSR page list read as a
padded block table, `kv_last_page_len` off by one, GQA head `h` mapped to KV head
`h % n_kv_heads` instead of `h // groups`, a causal mask aligned top-left instead
of bottom-right — every one of those runs without error, produces finite
well-scaled numbers, generates fluent text, and moves no metric. There is no
assertion you can write about a single implementation that catches them.

So there are two implementations. `PagedTorchBackend` is written independently of
the kernel, is simple enough to trust by reading, and is itself gated against the
engine's contiguous path and the fp32 HF oracle (results/p0, results/p1). This
file diffs the fast path against it on the same inputs, the same `BatchMeta`, and
the same physical page ids.

BIT-FOR-BIT IS NOT ACHIEVABLE HERE, AND PRETENDING OTHERWISE WOULD BE WORSE
---------------------------------------------------------------------------
The two implementations do not compute in the same order or at the same
precision, by construction:

  * `PagedTorchBackend._attend_one` deliberately reproduces the engine's exact
    rounding sequence — scores in fp32, softmax in fp32, `probs.to(fp16)`, then
    the PV matmul in fp16. The cast of `probs` to fp16 is a rounding step
    FlashInfer does not perform.
  * FlashInfer runs a fused ONLINE softmax with fp32 accumulators and never
    materialises `probs` at all, so there is nothing to round. It also tiles and
    rescales the accumulator as it sweeps KV, and (for long KV) may split-K and
    merge partial states — a different summation order again.

Two different-but-correct orderings cannot be bit-identical in floating point.
Claiming a bit-exact gate here and then loosening it when it fails would be worse
than stating the real bound up front, so:

  TOKENS are compared EXACTLY. The end-to-end greedy test requires identical
  token ids, because argmax over a fp16 logit vector is a discrete decision and
  a numerically irrelevant difference does not change it. That is the claim the
  system actually makes, and it is exact.

  TENSORS are compared to a stated bound. `ATOL`/`RTOL` below.

WHY THESE TOLERANCE NUMBERS AND NOT LOOSER ONES
-----------------------------------------------
The dominant term is the reference's own `probs.to(fp16)` cast: fp16 has an
11-bit significand, so each probability carries relative error <= 2^-11 ~= 4.9e-4.
The output is a convex combination `sum_t p_t v_t`, so those errors are
weighted-averaged rather than summed — the expected relative error of an output
element is ~5e-4, plus one more 2^-11 from writing the fp16 result. Empirically
that lands around 1e-3 absolute for O(1) outputs.

`ATOL = 4e-3` is therefore ~4x the expected rounding floor: loose enough not to
flake on a different GPU or a different KV length, and tight enough to be
useless as cover. THE POINT: every failure mode this file exists to catch is
O(1), not O(1e-3). A wrong GQA mapping, a dropped page, a mis-aligned causal mask
— all of them move outputs by tens of percent. There is no bug that hides in the
gap between 1e-3 and 4e-3. If this assertion fails, the cause is a defect, not
numerics, and the max-diff printed in the failure message will say so
immediately (expect >= 1e-1).

`MAX_MAE` is the secondary, shape-of-the-error check. Rounding noise is
zero-mean and spread over every element; a real bug concentrates error in the
elements it affects. A run that passes the per-element bound but has a mean
absolute error an order of magnitude above the floor is suspicious even though
nothing individually exceeded tolerance.

RUNNING IT

    # on PACE, on a real GPU, with flashinfer installed:
    REQUIRE_GPU=1 pytest tests/test_flashinfer_differential.py -v -s
"""

import os

import pytest
import torch

# --------------------------------------------------------------------------
# Tolerances. See the module docstring for the derivation.
# --------------------------------------------------------------------------

ATOL = 4e-3
RTOL = 1e-2
MAX_MAE = 1e-3


# --------------------------------------------------------------------------
# Preflight. Mirrors tests/test_paged_e2e.py, and for the same reason.
# --------------------------------------------------------------------------


def _cuda_status() -> str | None:
    """Return None if CUDA is usable, else a reason string."""
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is False"
    try:
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover
        return f"could not query device: {exc}"
    # torch built against CUDA 13 dropped Volta (sm_70). A V100 reports a device
    # and then fails or silently misbehaves; treat it as unusable. FlashInfer's
    # prebuilt kernels also start at sm_80.
    if cap < (8, 0):
        return f"{name} is sm_{cap[0]}{cap[1]}; this build needs sm_80+"
    return None


def _flashinfer_status() -> str | None:
    """Return None if flashinfer is importable, else a reason string."""
    from serving.backends.flashinfer_backend import is_available, unavailable_reason

    if is_available():
        return None
    return f"flashinfer is not importable: {unavailable_reason()}"


def _gate_reason() -> str | None:
    """
    Why this gate cannot run, or None.

    Factored out as a pure function so the REQUIRE_GPU hard-fail path below is
    testable on a machine with no GPU — see `test_require_gpu_turns_missing_gpu_into_failure`.
    """
    return _cuda_status() or _flashinfer_status()


def _preflight(reason: str | None, require_gpu: bool) -> None:
    """
    Raise if the gate cannot run AND the caller said a green result would be
    trusted.

    WHY THIS IS NOT A PLAIN skipif.

    A skipped differential and a passing differential are indistinguishable in a
    job log: 'no failures, exit 0'. That is not a hypothetical — job 11598374
    landed on a V100 under a CUDA-13 torch build, every gate test skipped, and
    the job reported success (results/p1/RESULTS.md §4). This file is strictly
    more exposed to that failure than the Phase 1 gate was, because it has TWO
    ways to vanish: no GPU, or no flashinfer. Missing flashinfer is the more
    dangerous one, since it is the expected state on a laptop and therefore the
    one nobody looks twice at.

    So: skipping is allowed on a developer laptop, where it is obvious. Anywhere
    a green result would be TRUSTED, set REQUIRE_GPU=1 and both absences become
    hard failures.
    """
    if reason and require_gpu:
        raise AssertionError(
            f"REQUIRE_GPU=1 but the FlashInfer differential cannot run: {reason}. "
            "A skipped differential must not be reported as a pass — it verifies "
            "nothing, and R9 stays open."
        )


_GATE_REASON = _gate_reason()
_REQUIRE_GPU = os.environ.get("REQUIRE_GPU") == "1"

if _GATE_REASON and _REQUIRE_GPU:
    pytest.fail(
        f"REQUIRE_GPU=1 but the FlashInfer differential cannot run: {_GATE_REASON}. "
        "A skipped differential must not be reported as a pass — it verifies "
        "nothing, and R9 stays open.",
        pytrace=False,
    )


def gpu_test(fn):
    """
    Mark a test as requiring CUDA + flashinfer.

    Applied per test rather than as a module-level `pytestmark` on purpose: the
    handful of availability and preflight tests at the bottom of this file MUST
    run on a CPU-only laptop, since they are what verify that the hard-fail path
    and the graceful-degradation path work at all.
    """
    fn = pytest.mark.skipif(_GATE_REASON is not None, reason=_GATE_REASON or "")(fn)
    return pytest.mark.gpu(fn)


# --------------------------------------------------------------------------
# Shared configuration
# --------------------------------------------------------------------------

BLOCK_SIZE = 16
NUM_BLOCKS = 256
NUM_LAYERS = 2  # >1 so a layer-indexing bug in either backend is visible
HEAD_DIM = 128  # a size FlashInfer definitely has a kernel for
N_HEADS = 8
N_KV_HEADS = 2  # groups = 4, so GQA is exercised by default
DTYPE = torch.float16
DEVICE = "cuda:0"

WEIGHTS_PATH = os.environ.get("LLM_WEIGHTS_PATH", "vendor/llm_inference_engine/weights")


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def _make_backends(n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM):
    """One backend of each kind, identically shaped."""
    from serving.backends.flashinfer_backend import FlashInferBackend
    from serving.backends.paged_torch import PagedTorchBackend

    kwargs = dict(
        num_layers=NUM_LAYERS,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        n_kv_heads=n_kv_heads,
        n_heads=n_heads,
        head_dim=head_dim,
        device=DEVICE,
        dtype=DTYPE,
    )
    return PagedTorchBackend(**kwargs), FlashInferBackend(**kwargs)


def _rand(shape, gen):
    """Random fp16 via fp32, so the generator is reproducible across dtypes."""
    return torch.randn(shape, generator=gen, device=DEVICE, dtype=torch.float32).to(DTYPE)


def _poison_pools(ref, fi, gen):
    """
    Fill BOTH pools with the SAME random garbage before anything is appended.

    THIS IS NOT DECORATION. In production a recycled block holds the previous
    owner's KV, and the tail of a partially-filled last page is exactly that
    garbage. A backend that reads `n_pages * page_size` tokens instead of
    `kv_len` — i.e. one that mishandles `kv_last_page_len` — attends over it,
    does not crash, does not produce NaNs, and produces fluent wrong text.

    Zero-initialised pools HIDE that bug: garbage of zero contributes an
    attention row that is merely small rather than obviously wrong. Poisoning
    both pools identically keeps the differential meaningful (both backends see
    the same bytes) while making "reads past kv_len" a detectable divergence,
    because the trusted reference slices to `[:kv_len]` and the subject would
    not.
    """
    for layer in range(ref.num_layers):
        k = _rand(tuple(ref.k_pool[layer].shape), gen)
        v = _rand(tuple(ref.v_pool[layer].shape), gen)
        ref.k_pool[layer].copy_(k)
        ref.v_pool[layer].copy_(v)
        fi.k_pool[layer].copy_(k)
        fi.v_pool[layer].copy_(v)


def _step(ref, fi, meta, gen, scale, kv_scale=None):
    """
    Run one full forward step (all layers) through both backends on identical
    inputs. Returns a list of (layer_idx, ref_out, fi_out).

    Order matters and matches the protocol: append_kv for a layer must complete
    before attend for that layer, so a decoding token attends to itself. Both
    backends see the byte-identical k/v/q tensors.
    """
    tokens = meta.n_tokens
    outs = []
    for layer in range(ref.num_layers):
        k = _rand((tokens, ref.n_kv_heads, ref.head_dim), gen)
        v = _rand((tokens, ref.n_kv_heads, ref.head_dim), gen)
        q = _rand((tokens, ref.n_heads, ref.head_dim), gen)
        if kv_scale is not None:
            k = (k.float() * kv_scale).to(DTYPE)
            v = (v.float() * kv_scale).to(DTYPE)

        ref.append_kv(layer, k, v, meta)
        fi.append_kv(layer, k, v, meta)
        outs.append((layer, ref.attend(q, layer, scale, meta), fi.attend(q, layer, scale, meta)))
    return outs


def _assert_matches(ref_out, fi_out, label):
    """Compare one attention output pair, with a message that diagnoses itself."""
    assert ref_out.shape == fi_out.shape, (
        f"{label}: shape {tuple(fi_out.shape)} != reference {tuple(ref_out.shape)}. "
        "attend() must return (tokens, n_heads, head_dim), same as q."
    )
    assert ref_out.dtype == fi_out.dtype, (
        f"{label}: dtype {fi_out.dtype} != reference {ref_out.dtype}"
    )
    assert torch.isfinite(fi_out).all(), f"{label}: FlashInfer produced non-finite values"

    d = (ref_out.float() - fi_out.float()).abs()
    max_diff = float(d.max())
    mae = float(d.mean())
    denom = ref_out.float().abs().max().clamp_min(1e-6)
    tol = ATOL + RTOL * float(denom)

    assert max_diff <= tol, (
        f"{label}: FlashInfer diverges from PagedTorchBackend.\n"
        f"  max|delta| = {max_diff:.6g}   (tolerance {tol:.6g} = {ATOL} + {RTOL} * "
        f"{float(denom):.4g})\n"
        f"  mean|delta| = {mae:.6g}\n"
        f"  ref range   = [{float(ref_out.min()):.4g}, {float(ref_out.max()):.4g}]\n"
        f"  fi  range   = [{float(fi_out.min()):.4g}, {float(fi_out.max()):.4g}]\n"
        "The tolerance is ~4x the fp16 rounding floor. A difference this large is a "
        "DEFECT, not numerics — look at layout (NHD vs HND), GQA head mapping "
        "(h // groups vs h % n_kv_heads), causal alignment (bottom-right, not "
        "top-left), or kv_last_page_len (must be page_size, never 0, on an exact "
        "multiple)."
    )
    assert mae <= MAX_MAE, (
        f"{label}: per-element tolerance held but mean|delta| = {mae:.6g} > {MAX_MAE}. "
        "Rounding noise is zero-mean and diffuse; a mean error this high means the "
        "error is concentrated, which is what a real bug looks like."
    )
    return max_diff, mae


def _grow_and_assemble(blocks_list, counts, device=DEVICE):
    """
    Grow each block table by `counts[i]` and assemble the BatchMeta for that step.

    Ordering is the contract (serving/engine_iface/batch.py): blocks.append()
    runs BEFORE assembly, so kv_lens is the length AFTER the append. Sequences
    contributing 0 tokens are dropped from the batch rather than carried with an
    empty range.
    """
    from serving.engine_iface.batch import ScheduledSeq, build_batch_meta

    scheds = []
    for b, n in zip(blocks_list, counts, strict=True):
        if n == 0:
            continue
        b.append(n)
        scheds.append(ScheduledSeq(blocks=b, new_token_ids=[0] * n))
    return build_batch_meta(scheds, device=device, page_size=BLOCK_SIZE)


def _new_sequences(alloc, n, first_id=0):
    from serving.memory.block_table import SequenceBlocks

    return [SequenceBlocks(alloc, seq_id=first_id + i) for i in range(n)]


def _allocator(fragment=False):
    """
    A fresh allocator, optionally pre-fragmented.

    Fragmenting first means the sequences under test receive scattered,
    non-monotonic physical page ids. An implementation that quietly assumes
    pages are contiguous or sorted passes every clean-pool test and fails here —
    and CSR exists precisely so that assumption is never needed.
    """
    from serving.memory.allocator import BlockAllocator

    alloc = BlockAllocator(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    if fragment:
        # THE POOL MUST BE EXHAUSTED BEFORE FREEING, or this does not fragment.
        #
        # BlockAllocator uses a FIFO free list (deliberately — LIFO hands back a
        # just-freed block and makes use-after-free invisible). FIFO means freed
        # blocks go to the BACK of the queue, so a later allocation is served
        # from the still-untouched front and never sees the holes at all. An
        # earlier version of this helper allocated 48 of NUM_BLOCKS and freed
        # alternates; sequences under test then received [48, 49, 50] —
        # contiguous and sorted, i.e. no fragmentation whatsoever.
        #
        # Allocating EVERY block first means the holes are the only thing left.
        holders = _new_sequences(alloc, NUM_BLOCKS, first_id=900)
        for h in holders:
            h.append(BLOCK_SIZE)
        assert alloc.num_free == 0, "pool must be exhausted for freeing to create holes"
        for h in holders[::2]:
            h.free()
        return alloc, holders[1::2]
    return alloc, []


# --------------------------------------------------------------------------
# THE DIFFERENTIAL
# --------------------------------------------------------------------------


@gpu_test
@pytest.mark.parametrize("batch", [1, 2, 5])
def test_pure_decode_batch(batch):
    """
    Every sequence contributes exactly one token — the decode-shaped batch, which
    dispatches to BatchDecodeWithPagedKVCacheWrapper.

    This is the shape that dominates a serving workload, and the one where the
    causal mask is skipped entirely on both sides (q_len == 1, so every key is
    legal). A divergence here is pure addressing.
    """
    ref, fi = _make_backends()
    gen = torch.Generator(device=DEVICE).manual_seed(1234 + batch)
    _poison_pools(ref, fi, gen)

    alloc, _ = _allocator()
    seqs = _new_sequences(alloc, batch)
    scale = HEAD_DIM**-0.5

    # Prefill first so there is real history to decode against, then decode.
    meta = _grow_and_assemble(seqs, [7 + 5 * i for i in range(batch)])
    _step(ref, fi, meta, gen, scale)

    worst = 0.0
    for round_ix in range(6):
        meta = _grow_and_assemble(seqs, [1] * batch)
        assert not meta.is_prefill, "all query_lens are 1; this must be a decode batch"
        for layer, a, b in _step(ref, fi, meta, gen, scale):
            worst = max(worst, _assert_matches(a, b, f"decode r{round_ix} L{layer}")[0])
    print(f"\n  [decode batch={batch}] worst max|delta| = {worst:.3g}")


@gpu_test
def test_mixed_prefill_and_decode():
    """
    One sequence contributes a 20-token chunk while two contribute a single token
    each — a continuous-batching step.

    This shape has no single `q_len_per_req`, so the decode wrapper cannot
    express it and dispatch must land on the prefill wrapper. It is also where
    causal alignment actually bites: the chunk's queries need a bottom-right
    aligned mask against a longer history, while the decode rows need none. Get
    the alignment wrong and only this test fails.
    """
    ref, fi = _make_backends()
    gen = torch.Generator(device=DEVICE).manual_seed(99)
    _poison_pools(ref, fi, gen)

    alloc, _ = _allocator()
    seqs = _new_sequences(alloc, 3)
    scale = HEAD_DIM**-0.5

    # Give the two decoders some history; the third stays empty for now.
    meta = _grow_and_assemble(seqs, [33, 17, 0])
    _step(ref, fi, meta, gen, scale)

    # Now: seq2 prefills 20 tokens, seq0 and seq1 decode one each.
    meta = _grow_and_assemble(seqs, [1, 1, 20])
    assert meta.is_prefill, "a batch containing a 20-token chunk is prefill-shaped"
    for layer, a, b in _step(ref, fi, meta, gen, scale):
        _assert_matches(a, b, f"mixed L{layer}")

    # And a chunked continuation of seq2 — its second chunk starts at absolute
    # position 20, which is the case a top-left aligned mask gets wrong.
    meta = _grow_and_assemble(seqs, [1, 0, 13])
    for layer, a, b in _step(ref, fi, meta, gen, scale):
        _assert_matches(a, b, f"chunked L{layer}")


@gpu_test
@pytest.mark.parametrize("prompt_len", [1, 15, 16, 17, 31, 32, 33])
def test_page_boundary_lengths(prompt_len):
    """
    Sweep prefill lengths across page boundaries (R8's sweep, applied to R9).

    16 and 32 are the cases where `kv_last_page_len` must report `page_size`, not
    0. Report 0 and a whole page of keys silently vanishes; report page_size + 1
    and a page of the poisoned tail is attended over. Both produce plausible
    output. 1 is the degenerate single-token page; 15/17/31/33 straddle.
    """
    ref, fi = _make_backends()
    gen = torch.Generator(device=DEVICE).manual_seed(7 * prompt_len)
    _poison_pools(ref, fi, gen)

    alloc, _ = _allocator()
    seqs = _new_sequences(alloc, 1)
    scale = HEAD_DIM**-0.5

    meta = _grow_and_assemble(seqs, [prompt_len])
    pages = -(-prompt_len // BLOCK_SIZE)
    assert int(meta.kv_last_page_len[0]) == (prompt_len % BLOCK_SIZE or BLOCK_SIZE)
    for layer, a, b in _step(ref, fi, meta, gen, scale):
        _assert_matches(a, b, f"prefill len={prompt_len} pages={pages} L{layer}")

    # Then decode across the next boundary, where a new page is allocated
    # BETWEEN forward passes — a different code path from prefill.
    for round_ix in range(BLOCK_SIZE + 2):
        meta = _grow_and_assemble(seqs, [1])
        for layer, a, b in _step(ref, fi, meta, gen, scale):
            _assert_matches(a, b, f"decode len={prompt_len}+{round_ix + 1} L{layer}")


@gpu_test
def test_fragmented_non_contiguous_pages():
    """
    Output must not depend on WHICH physical pages a sequence got.

    The pool is fragmented first, so page ids are scattered and non-monotonic.
    CSR carries an arbitrary ordered list of page ids and nothing in either
    backend may assume they ascend or adjoin.
    """
    ref, fi = _make_backends()
    gen = torch.Generator(device=DEVICE).manual_seed(4242)
    _poison_pools(ref, fi, gen)

    alloc, holders = _allocator(fragment=True)
    seqs = _new_sequences(alloc, 3)
    scale = HEAD_DIM**-0.5

    meta = _grow_and_assemble(seqs, [40, 9, 64])

    ids = seqs[0].block_ids
    assert len(ids) > 2, "need several pages for the ordering assumption to matter"
    assert ids != sorted(ids) or any(
        b - a != 1 for a, b in zip(ids, ids[1:], strict=False)
    ), f"pool was not actually fragmented; page ids {ids} are contiguous and sorted"

    for layer, a, b in _step(ref, fi, meta, gen, scale):
        _assert_matches(a, b, f"fragmented prefill L{layer}")
    for round_ix in range(4):
        meta = _grow_and_assemble(seqs, [1, 1, 1])
        for layer, a, b in _step(ref, fi, meta, gen, scale):
            _assert_matches(a, b, f"fragmented decode r{round_ix} L{layer}")

    for h in holders:
        h.free()
    alloc.check_invariants()


@gpu_test
def test_multi_sequence_isolation():
    """
    A sequence's output must not depend on who else is in the batch.

    Two checks, and they catch different things. The differential against the
    oracle catches "both sequences are wrong". This ALSO re-runs each sequence
    ALONE, against the same pools, and requires the same answer — which catches
    cross-sequence contamination specifically: a CSR walk that runs off the end
    of one row into the next reads real, plausible KV belonging to a neighbour.
    """
    ref, fi = _make_backends()
    gen = torch.Generator(device=DEVICE).manual_seed(31337)
    _poison_pools(ref, fi, gen)

    alloc, _ = _allocator()
    seqs = _new_sequences(alloc, 4)
    scale = HEAD_DIM**-0.5

    meta = _grow_and_assemble(seqs, [16, 3, 47, 32])
    _step(ref, fi, meta, gen, scale)

    meta = _grow_and_assemble(seqs, [1, 1, 1, 1])
    tokens = meta.n_tokens
    q = _rand((tokens, N_HEADS, HEAD_DIM), gen)
    k = _rand((tokens, N_KV_HEADS, HEAD_DIM), gen)
    v = _rand((tokens, N_KV_HEADS, HEAD_DIM), gen)

    ref.append_kv(0, k, v, meta)
    fi.append_kv(0, k, v, meta)
    batched_ref = ref.attend(q, 0, scale, meta)
    batched_fi = fi.attend(q, 0, scale, meta)
    _assert_matches(batched_ref, batched_fi, "batched")

    # Re-attend each sequence alone. The block tables are already grown, so
    # assembly is a pure read — no append, no allocation, nothing mutated.
    from serving.engine_iface.batch import ScheduledSeq, build_batch_meta

    cu = meta.cu_query_lens.tolist()
    for i, s in enumerate(seqs):
        solo_meta = build_batch_meta(
            [ScheduledSeq(blocks=s, new_token_ids=[0])], device=DEVICE, page_size=BLOCK_SIZE
        )
        solo = fi.attend(q[cu[i] : cu[i + 1]], 0, scale, solo_meta)
        _assert_matches(batched_fi[cu[i] : cu[i + 1]], solo, f"seq {i} alone vs in batch")


@gpu_test
@pytest.mark.parametrize(("n_heads", "n_kv_heads"), [(8, 1), (8, 2), (8, 4), (8, 8)])
def test_gqa_head_mapping(n_heads, n_kv_heads):
    """
    Query head h must read KV head h // groups, not h % n_kv_heads.

    Both mappings run, both produce finite well-scaled output, and only one is
    right — `PagedTorchBackend._attend_one` says so at length. To make the
    difference DETECTABLE rather than merely present, each KV head's values are
    scaled by a distinct factor: under the wrong mapping the output for query
    head h carries the wrong head's magnitude, which is a tens-of-percent error,
    not a rounding one.

    The sweep includes n_kv_heads == n_heads (groups == 1, where the two mappings
    coincide and this test proves nothing) as a control: if that case fails too,
    the bug is not GQA.
    """
    ref, fi = _make_backends(n_heads=n_heads, n_kv_heads=n_kv_heads)
    gen = torch.Generator(device=DEVICE).manual_seed(555 + n_kv_heads)
    _poison_pools(ref, fi, gen)

    alloc, _ = _allocator()
    seqs = _new_sequences(alloc, 2)
    scale = HEAD_DIM**-0.5
    kv_scale = (1.0 + torch.arange(n_kv_heads, device=DEVICE, dtype=torch.float32)).view(
        1, n_kv_heads, 1
    )

    meta = _grow_and_assemble(seqs, [21, 34])
    for layer, a, b in _step(ref, fi, meta, gen, scale, kv_scale=kv_scale):
        _assert_matches(a, b, f"gqa {n_heads}/{n_kv_heads} prefill L{layer}")
    meta = _grow_and_assemble(seqs, [1, 1])
    for layer, a, b in _step(ref, fi, meta, gen, scale, kv_scale=kv_scale):
        _assert_matches(a, b, f"gqa {n_heads}/{n_kv_heads} decode L{layer}")


@gpu_test
def test_plan_is_called_once_per_step_not_once_per_layer():
    """
    plan() must run exactly once per forward pass, and run() once per layer.

    Both deviations are bugs and they are not the same bug (see the backend's
    module docstring): an extra plan() per layer is a PERFORMANCE defect —
    a host copy of the CSR plus a host-side schedule, num_layers times, per step
    — while a missing plan() is a CORRECTNESS defect, because run() reads the
    page tables plan() stashed on the wrapper and would silently attend over the
    previous step's page table.

    Counting is the only way to see the performance half; the correctness half is
    covered by every differential above, all of which run many steps against a
    changing page table.
    """
    ref, fi = _make_backends()
    gen = torch.Generator(device=DEVICE).manual_seed(2)
    _poison_pools(ref, fi, gen)

    alloc, _ = _allocator()
    seqs = _new_sequences(alloc, 2)
    scale = HEAD_DIM**-0.5

    steps = 5
    meta = _grow_and_assemble(seqs, [11, 20])
    _step(ref, fi, meta, gen, scale)
    for _ in range(steps - 1):
        meta = _grow_and_assemble(seqs, [1, 1])
        _step(ref, fi, meta, gen, scale)

    assert fi.n_plans == steps, (
        f"expected {steps} plan() calls (one per forward pass), got {fi.n_plans}. "
        f"{'Re-planning per layer is a performance defect.' if fi.n_plans > steps else ''}"
    )
    assert fi.n_runs == steps * NUM_LAYERS, (
        f"expected {steps * NUM_LAYERS} run() calls, got {fi.n_runs}"
    )


@gpu_test
def test_backends_satisfy_the_same_protocol():
    """
    Both classes must be interchangeable at the type level too — a serving layer
    isinstance-checks a plugin at startup rather than discovering a missing
    method mid-benchmark.
    """
    from engine.attention_backend import AttentionBackend

    ref, fi = _make_backends()
    assert isinstance(ref, AttentionBackend)
    assert isinstance(fi, AttentionBackend)
    # Same accounting, so pool sizing is backend-independent.
    assert fi.pool_bytes() == ref.pool_bytes()
    assert fi.block_bytes() == ref.block_bytes()
    assert fi.tokens_capacity() == ref.tokens_capacity()
    # ...except the workspace, which is FlashInfer-only fixed overhead and must
    # be visible to a sizing calculation rather than folded into the pool.
    assert fi.workspace_bytes() > 0


# --------------------------------------------------------------------------
# END TO END — the claim that is exact
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    from engine.loader import load_config, load_weights_gpu
    from engine.model_gpu import LlamaModelGPU

    config = load_config(WEIGHTS_PATH)
    return LlamaModelGPU(load_weights_gpu(WEIGHTS_PATH, config), config), config


def _greedy(backend_cls, model, config, prompt_ids, max_tokens, num_blocks=512):
    from serving.engine_iface.runner import greedy_device, paged_generate
    from serving.memory.allocator import BlockAllocator

    alloc = BlockAllocator(num_blocks=num_blocks, block_size=BLOCK_SIZE)
    backend = backend_cls(
        num_layers=config["num_hidden_layers"],
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        n_kv_heads=config["num_key_value_heads"],
        n_heads=config["num_attention_heads"],
        head_dim=config["head_dim"],
        device=model.device,
        dtype=DTYPE,
    )
    out = list(
        paged_generate(model, backend, alloc, list(prompt_ids), greedy_device, max_tokens)
    )
    alloc.check_invariants()
    return out


E2E_PROMPTS = {
    "short": [128000, 9906, 11, 358, 1097],
    "medium": [128000, 791, 4062, 14198, 39935, 27096, 927, 279, 16053, 5679, 13],
    "page_aligned": [128000] + [9906] * 31,  # 32 tokens: last_page_len == 16, not 0
}


@gpu_test
@pytest.mark.parametrize("key", ["short", "medium", "page_aligned"])
def test_end_to_end_greedy_tokens_are_identical(model, key):
    """
    THE R9 GATE. Same model, same weights, same prompt, same greedy sampler,
    through serving/engine_iface/runner.py paged_generate() — once with each
    backend. The token ids must be IDENTICAL.

    This is exact, not toleranced, and it is exact for a real reason rather than
    an optimistic one: argmax over a logit vector is a discrete decision, so the
    ~1e-3 numerical difference the tensor tests allow cannot change it unless the
    top two logits are within that of each other. A divergence here is a defect
    in paging, addressing, or attention — and because it accumulates
    autoregressively, the FIRST divergent index localises it far better than any
    tensor diff.
    """
    m, config = model
    prompt = E2E_PROMPTS[key]
    n = 24

    from serving.backends.flashinfer_backend import FlashInferBackend
    from serving.backends.paged_torch import PagedTorchBackend

    expected = _greedy(PagedTorchBackend, m, config, prompt, n)
    got = _greedy(FlashInferBackend, m, config, prompt, n)

    div = next((i for i, (x, y) in enumerate(zip(got, expected, strict=False)) if x != y), None)
    assert div is None and len(got) == len(expected), (
        f"[{key}] FlashInfer greedy output diverges from PagedTorchBackend at token {div}.\n"
        f"  PagedTorch (oracle): {expected}\n"
        f"  FlashInfer:          {got}\n"
        "Both paths run the same weights through the same model code and differ only "
        "in the attention backend. R9 is NOT retired."
    )
    print(f"\n  [{key}] {len(got)} tokens identical across both backends")


@gpu_test
def test_end_to_end_long_decode_crosses_page_boundaries(model):
    """
    Prompt-length sweeps only exercise page boundaries during prefill. Generating
    40 tokens from a 5-token prompt crosses three at block_size 16, each time
    allocating a new page BETWEEN forward passes — which changes the CSR the
    plan() is built from, and is therefore also the strongest available test that
    plan() is not being skipped.
    """
    m, config = model
    from serving.backends.flashinfer_backend import FlashInferBackend
    from serving.backends.paged_torch import PagedTorchBackend

    prompt = E2E_PROMPTS["short"]
    expected = _greedy(PagedTorchBackend, m, config, prompt, 40)
    got = _greedy(FlashInferBackend, m, config, prompt, 40)
    assert got == expected, (
        "Divergence during long decode across page boundaries:\n"
        f"  PagedTorch (oracle): {expected}\n  FlashInfer:          {got}"
    )


# --------------------------------------------------------------------------
# CPU-RUNNABLE: the gate's own machinery
#
# These are deliberately NOT marked gpu. They are what verifies that a machine
# without CUDA or without flashinfer degrades the way it is supposed to — which
# is the exact property results/p1/RESULTS.md §4 records us getting wrong once
# already, and which no GPU test can check.
# --------------------------------------------------------------------------


def test_backend_module_imports_without_flashinfer():
    """
    `import serving.backends.flashinfer_backend` must succeed anywhere.

    PagedTorchBackend is the fallback and the oracle; the Phase 1 memory claim
    must not depend on this wheel resolving (R18). If the import itself required
    flashinfer, a CPU box could not even ask whether flashinfer is present.
    """
    import serving.backends.flashinfer_backend as fib

    assert isinstance(fib.is_available(), bool)
    assert callable(fib.FlashInferBackend)


def test_is_available_and_reason_agree():
    """`unavailable_reason()` is None exactly when `is_available()` is True."""
    from serving.backends.flashinfer_backend import is_available, unavailable_reason

    if is_available():
        assert unavailable_reason() is None
    else:
        reason = unavailable_reason()
        assert isinstance(reason, str) and reason, "an unavailable backend must say why"


def test_constructor_error_names_the_fallback():
    """
    Without flashinfer the constructor must fail with an ACTIONABLE message, not
    a bare ImportError from ten frames down. It has to name PagedTorchBackend,
    because "use the other backend" is the actual remedy and the caller has no
    way to know that from an ImportError.
    """
    from serving.backends.flashinfer_backend import FlashInferBackend, is_available

    if is_available():
        pytest.skip("flashinfer IS available here; this checks the degraded path")

    with pytest.raises(RuntimeError) as exc:
        FlashInferBackend(
            num_layers=2,
            num_blocks=8,
            block_size=16,
            n_kv_heads=2,
            n_heads=8,
            head_dim=128,
            device="cpu",
        )
    msg = str(exc.value)
    assert "PagedTorchBackend" in msg
    assert "is_available" in msg


def test_constructor_validates_shapes_before_touching_flashinfer():
    """
    Bad arguments must be rejected identically by both backends, and the check
    must happen BEFORE the flashinfer import — otherwise the same mistake gives
    two different errors depending on the machine, and the CPU-only developer
    never sees the real one.
    """
    from serving.backends.flashinfer_backend import FlashInferBackend

    with pytest.raises(ValueError, match="divisible"):
        FlashInferBackend(2, 8, 16, n_kv_heads=3, n_heads=8, head_dim=128, device="cpu")
    with pytest.raises(ValueError, match="num_blocks"):
        FlashInferBackend(2, 0, 16, n_kv_heads=2, n_heads=8, head_dim=128, device="cpu")


def test_require_gpu_turns_missing_gpu_into_failure():
    """
    The REQUIRE_GPU hard-fail path, tested on a machine that has no GPU.

    A skipped differential and a passing differential are indistinguishable in a
    job log (results/p1/RESULTS.md §4). The preflight is what closes that hole,
    so the preflight itself needs a test that runs where the hole opens — on the
    laptop — rather than only on the cluster that is supposed to be protected.
    """
    # Unavailable + trusted => must raise, and must say why.
    with pytest.raises(AssertionError) as exc:
        _preflight("no CUDA device", require_gpu=True)
    assert "REQUIRE_GPU=1" in str(exc.value)
    assert "no CUDA device" in str(exc.value)

    # Unavailable + laptop => silent skip is fine, it is visible there.
    _preflight("no CUDA device", require_gpu=False)
    # Available => never raises, either way.
    _preflight(None, require_gpu=True)
    _preflight(None, require_gpu=False)


def test_gate_reason_covers_both_absences():
    """
    Missing flashinfer must be a gate reason in its own right, not just missing
    CUDA. It is the more dangerous of the two: it is the expected state on a
    laptop, so it is the one nobody looks twice at, and REQUIRE_GPU=1 has to
    catch it on the cluster.
    """
    reason = _gate_reason()
    if reason is None:
        assert _cuda_status() is None and _flashinfer_status() is None
    else:
        assert reason in (_cuda_status(), _flashinfer_status())
