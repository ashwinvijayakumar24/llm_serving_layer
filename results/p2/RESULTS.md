# Phase 2 results — batch invariance, and the FlashInfer differential

**Date:** 2026-08-01 · **Gates:** BOTH PASSED · **Engine:** `v0.2.1` · **Job:** `11598894`

Two gates, both correctness. **No performance claim is made here.** The one
throughput figure below is a direction check, not a measurement — Phase 2's
actual throughput and goodput numbers come later, from the open-loop harness
against the HTTP server, once the SLO is calibrated and frozen.

## Run identity

| | |
|---|---|
| Slurm job | `11598894` |
| Node / partition / QOS | `gpu-h100` / `inferno` |
| GPU | NVIDIA H100 80GB HBM3, sm_90 |
| Engine | tag `v0.2.1` |
| Raw log | [`p2_gate_11598894.log`](p2_gate_11598894.log) |

## Gate 1 — Batch invariance (R4). 9 passed.

**This was the first thing in the project to run `n_seqs > 1` on real weights.**
Phase 1 was batch-1 throughout, and every correctness test in the engine is
batch-1 (`tests/test_forward.py`, `tests/test_decode.py`), so nothing upstream
could have caught a batching divergence.

| Test | Covers |
|---|---|
| `test_batch_invariance_mixed_lengths` | 4 prompts of different lengths, alone vs batched |
| `test_invariance_across_batch_sizes[2,3,4]` | same prompt, same output at every batch size |
| `test_prefill_chunk_mixed_with_decodes` | ragged `query_lens` — the shape continuous batching exists to produce |
| `test_staggered_arrival` | requests joining an already-running batch mid-flight |
| `test_batch_frees_all_blocks` | no leak across a full batched run |
| `test_cancellation_frees_blocks_and_spares_others` | the path that does *not* go through normal completion |

**Bit-identity held.** R4's mitigation clause — if exact equality proves
unachievable through legitimate reduction-order effects, publish the divergence
rate rather than loosen the gate — was not needed and was not used.

**Direction check, not a benchmark:** 0.91 tok/step at batch 1 → 3.64 tok/step
at batch 4. Asserted as a trend only. One process, no warmup, no repetition,
shared node. It exists to catch the case where continuous batching is
bookkeeping with no throughput behind it, which would make the Phase 2 claim
empty.

## Gate 2 — FlashInfer differential (R9). 28 passed, 1 skipped. **R9 RETIRED.**

`FlashInferBackend` matches `PagedTorchBackend` across pure decode (batch
1/2/5), mixed prefill+decode, a page-boundary sweep (1/15/16/17/31/32/33),
fragmented non-contiguous pages, multi-sequence isolation, GQA 8→{1,2,4,8},
plan-call accounting, and end-to-end greedy token equality.

**The chain that makes this mean something:**

```
FlashInferBackend  ==  PagedTorchBackend   (job 11598894, this gate)
PagedTorchBackend  ==  contiguous engine   (job 11598444, Phase 1)
contiguous engine  ==  fp32 HF oracle      (job 11596894, Phase 0)
```

Each link was verified separately, on real weights, before the next was built.
That is the whole reason `PagedTorchBackend` was written first: it is the oracle,
not the fast path.

### Three contract facts read from the kernels, not the docs

The 0.6.16 docstrings do not state any of these; the source does.

- **`causal=True` is BOTTOM-RIGHT aligned.** `prefill.cuh:1461` masks iff
  `kv_idx + qo_len > kv_len + q_idx`, corroborated at `scheduler.cuh:954`
  (`kv_len_init = kv_len - qo_len; // right aligned`). That is exactly the
  protocol's `[0, kv_len - q_len + j]`. PyTorch SDPA's `is_causal` is *top-left*
  aligned and would have been silently wrong for decode — which is why
  `PagedTorchBackend` deliberately avoided SDPA.
- **`plan()` stores state; `run()` does not take it.** The page tables live on
  the wrapper (`decode.py:1467-1470`, `prefill.py:2355-2365`); `run(q, cache)`
  receives no CSR. **A missing `plan()` therefore attends over the previous
  step's page table, silently.** Rule: plan once per forward pass, run once per
  layer. Planning per layer only wastes a host copy; skipping one corrupts
  output with no error.
- **`run()` returns `(tokens, n_heads, head_dim)`** — resolves an item
  `ARCHITECTURE.md §2.3.1` previously listed as unverified.

### Stated limit: token-exact, not bit-exact

`PagedTorchBackend` casts `probs` to fp16 before the PV matmul, mirroring
`components_gpu.py:205`. FlashInfer runs a fused fp32 online softmax and never
materialises `probs`. **They cannot agree bit-for-bit by construction.**

Tensors are compared at `atol=4e-3, rtol=1e-2` (~4× the fp16 rounding floor)
with a mean-abs-error guard; **output tokens are compared exactly**, which is
the claim the system actually makes. This is defensible because every failure
mode R9 describes — wrong layout, wrong causal alignment, a stale page table, a
wrong GQA mapping — is an O(1) error, not an O(1e-3) one. Nothing hides in the
gap. Recorded rather than left as an unexplained tolerance.

## A test that had never tested anything

The differential first failed on **its own precondition**, not on a mismatch:

```
AssertionError: pool was not actually fragmented; page ids [48, 49, 50] are contiguous and sorted
```

`BlockAllocator` uses a **FIFO** free list — deliberately, because LIFO hands
back the just-freed block and makes use-after-free invisible. FIFO puts freed
blocks at the *back*, so a later allocation is served from the still-untouched
front and never sees the holes. Both fragmentation helpers allocated a small
prefix of the pool and freed alternates, then received a contiguous run of fresh
blocks.

**This means Phase 1's `test_fragmented_pool_gives_identical_output` had been
passing without testing fragmentation since it was written.** It is fixed the
same way — exhaust the pool *before* freeing — and now asserts its own
precondition. Verified off-GPU: the naive strategy yields `[32,33,34]`,
exhaust-then-free yields `[0,2,4]`.

The lesson, and the second of its kind this project (after the silent-skip hole
in `results/p1/RESULTS.md` §4): **a test that cannot detect its own setup
failing is not a test.** The agent-written version was better than the
hand-written one for exactly one reason — it checked.

## Also landed in Phase 2

- **`bench/loadgen.py`** — open-loop harness, 29 tests. The coordinated-omission
  guard (R1, the register's most likely path to a confidently wrong number) is
  **mutation-tested**: rewriting TTFT to subtract actual instead of intended
  send time fails exactly the two CO tests and nothing else. Four further
  mutants caught, including timing TTFT to the first chunk rather than the first
  non-empty one. Invalid runs `exit(2)`.
- **`bench/workloads/generator.py`** — 75 tests. Four prefix-sharing structures.
  The adversarial divergence sweep is **re-derived from the emitted tokens**,
  ignoring recorded metadata, so mislabeled offsets over correct-looking tokens
  still fail; verified across 40 seeds × 5 block sizes.
- **`serving/scheduler/scheduler.py`** — iteration-level continuous batching.

## Outstanding in Phase 2

- **HTTP surface** — in progress. Until it exists, the load harness has nothing
  to point at, and no goodput number can be produced.
- **SLO calibration** — must be measured from unloaded batch-1 in the same
  allocation, then **frozen**. An SLO chosen after seeing results is not an SLO.
- **B1 baseline** and the S2/S3 measurements, which depend on both of the above.

## Risk register changes

- **R9 — RETIRED.** The chain closes.
- **R4 — detection live and passing.** The risk itself does not retire; it
  reappears whenever batching or attention changes, and the gate now runs on
  every GPU job.
