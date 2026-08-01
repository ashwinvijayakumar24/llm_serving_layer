# Finding: batched output is not bit-invariant to batch shape

**Date:** 2026-08-01 · **Status:** measured, reproduced, and independently verified
**Consequence:** Phase 4 does not earn its correctness claim. Phase 2's claim is narrowed.

## The claim being corrected

Phase 2's gate asserts that a prompt produces **identical greedy output** alone
and inside a mixed batch, and it passes (9/9, job `11598894`). The Phase 4 gate
asserts cache-on output equals cache-off output, and it **fails** on exactly the
structures with real prefix reuse.

The natural reading is a radix-cache bug — R6, "faster and wrong". That reading
is wrong, and three controls show why.

## What was actually measured

**Control 1.** Replaying a divergent request with the cache **off**, at prefill
budgets 2²⁰ / 64 / 32 / 16, gives the correct output every time. Chunking alone
never reproduces it.

**Control 2.** Forcing every prefill chunk to exactly one block
(`max_prefill_tokens=16`), so publisher and consumer compute each position in an
identically *shaped* forward pass, drives the drift to **exactly 0 at every
position of every request**, and all 12 outputs match.

**Control 3, decisive.** Same 105-token prompt, **cache off in both runs**, one
chunk of 105 either way. The only difference: 286 other requests' tokens shared
the forward pass. Result: **max|Δlogit| = 0.1745 at 105 of 105 positions.**

Per-sequence attention shapes are held fixed in control 3, so the sensitivity is
not in `attend`. It is in `linear()` — `x @ w.T` in
`engine/components_gpu.py` — whose `M` dimension is the *packed batch token
count*. cuBLAS selects different kernels and split-K strategies by shape, and a
different reduction order gives a different fp16 sum.

## Independent verification

Re-derived without the radix cache, the scheduler, or preemption — calling the
engine's batched forward directly with the same target sequence and varying only
the amount of unrelated traffic beside it (`scripts/verify_batch_numerics.py`,
job `11602316`, H200):

| filler seqs | batch tokens | max abs Δlogit | argmax | flipped |
|---|---|---|---|---|
| 1 | 169 | 0.00000 | 9906 | no |
| 2 | 233 | 0.00000 | 9906 | no |
| 4 | 361 | **0.01953** | 9906 | no |
| 6 | 489 | **0.01953** | 9906 | no |

The target sequence is byte-identical in every run. Drift appears once the
packed batch crosses a shape threshold.

## What this means, stated plainly

**Batch invariance of the OUTPUT TOKENS is probabilistic, not structural.**
Logits move with batch shape; whether the argmax follows depends on the margin
between the top two candidates. Phase 2's gate passes because its prompts are
5–36 tokens and its margins are comfortable — the drift is present there too, it
simply never flips a decision. The Phase 4 gate uses longer, prefix-shared
prompts, and one flip per 12-request workload is enough to fail it.

So the Phase 4 gate was doing its job: it detected a real divergence. It
attributed it to the cache because that was the variable under test, but the
cause lies underneath, in the engine's GEMM, and it is **not fixable in
`serving/cache/radix.py`.**

## Honest restatement of the claims

- **Phase 2 (batching):** greedy output is token-identical to single-sequence
  output *on the measured workloads* (9 gate cases, prompts 5–36 tokens, batch
  sizes 2–4). It is **not** bit-invariant to batch shape in general, and the
  measured logit drift is up to 0.1745 at larger packed-batch sizes.
- **Phase 4 (radix cache):** **does not earn a correctness claim.** The cache is
  functionally correct — the block-boundary sweep (16/16), chunked-prefill-under-
  a-hit, eviction, leak and the zero-sharing control all pass — but the
  end-to-end equality gate cannot pass while the underlying GEMM is
  shape-sensitive. Reported as incomplete rather than passed with a widened
  tolerance.

## What would fix it

Batch-invariant GEMMs: pinning cuBLAS algorithm selection, or accumulating in
fp32, or a deterministic split-K. All of them live in the **engine's** forward
pass (`engine/components_gpu.py:linear`), not in the serving layer, and all cost
throughput. That is a real tradeoff — determinism versus speed — and it belongs
in the engine's scope, not this one.

The cheaper mitigation, if exact reproducibility is required: force uniform
batch shapes, which control 2 shows drives the drift to exactly zero. That
sacrifices most of the benefit of continuous batching, so it is a debugging tool
rather than a serving configuration.

## A real bug found on the way

Not the cause of the gate failure, but genuinely wrong and now fixed:
`Scheduler._cache_insert` published `prefill_ids + output_ids[:-1]`. After a
RECOMPUTE resume, `prefill_ids` *already is* `prompt + output_ids`, so every
block past the resume point was filed in the trie under **duplicated tokens** —
correct KV, lying key. A later request matching that key would receive KV for a
different token sequence.

The CPU suite could not see it: its simulated KV stored only the token id, so it
was blind to position and prefix. The simulator now stores a rolling hash of
(prefix, token, absolute position), and two regression tests fail on the old
code and pass on the new.
