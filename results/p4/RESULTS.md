# Phase 4 results — radix prefix cache

**Date:** 2026-08-01 · **Job:** `11608501` · **H200** · **Engine `v0.2.1`**
**Correctness: EARNED, at matched GEMM shape** (job `11611626`, section 0 below).
**Under production batching it remains blocked** on the engine's shape-dependent
numerics — `FINDING_batch_shape_numerics.md`. Both results are stated because
they are both true and they say different things.

---

## 0. Correctness — EARNED at matched GEMM shape

**Job `11611626`, H200.** Raw log: [`p4_gate_11611626.log`](p4_gate_11611626.log).

```
zero            identical output, block hit rate 0.000   (0 of 83 blocks)    PASSED
system          identical output, block hit rate 0.476   (40 of 84)          PASSED
conversational  identical output, block hit rate 0.465   (80 of 172)         PASSED
adversarial     identical output, block hit rate 0.381   (32 of 84)          PASSED
>>> uniform-chunk gate exit code: 0
```

Cache-on output is **bit-identical to cache-off** across all four sharing
structures — including `adversarial`, which sweeps the divergence point across
every offset within a block — while genuinely reusing 32–80 blocks. The `zero`
control correctly reuses nothing.

### Why this gate is stricter than the one it replaces, not weaker

The batched gate compares two runs whose forward passes have **different
shapes**, because a cache hit changes how many tokens are prefilled. The
engine's `linear()` takes the packed token count as its `M` dimension, cuBLAS
selects kernels and split-K by shape, and fp16 reduction order follows. So that
gate cannot distinguish "the cache served wrong KV" from "the GEMM summed in a
different order".

**Batch size 1 does not fix this**, which was tried first and failed
identically: cache-off computes `M=105` in one pass while cache-on computes
`M=41` for the uncached remainder. The shape difference is **intrinsic to
caching**, not to batching.

Capping prefill at exactly one block per step does fix it. Cache-off prefills
105 tokens as `16,16,16,16,16,16,9`; cache-on with 64 tokens cached prefills
`16,16,9`. Every individual GEMM has the same `M` in both arms — the cache
simply performs **fewer of them**. Block-granularity reuse is what makes this
exact: the cached amount is always a multiple of the block size, so even the
partial tail chunk aligns.

Under that configuration reduction order cannot vary, so **any divergence would
have to be the cache serving wrong KV.** There is none.

### The confirming half

The **batched** gate still fails on exactly `system`, `conversational` and
`adversarial` in the same job, on the same code, in the same allocation. Two
gates, one difference — whether GEMM shapes are held constant — and opposite
outcomes. That is what attributes the divergence to the engine's numerics rather
than to this component.

### What is and is not claimed

- **Claimed:** the radix cache returns correct KV. Verified at 38–48% block
  reuse across four structures including the block-boundary stress case.
- **Not claimed:** bit-identical output under production chunk sizes and
  concurrent batching. That is blocked on shape-dependent fp16 reduction order
  in `engine/components_gpu.py:linear`, is measured at max|Δlogit| 0.1745, and
  is reported as blocked rather than quietly passed.

---

## 1. The finding that explains the whole table

**In a chat server, "zero prefix sharing" does not exist at the token level.**

The workload generator's `zero` structure gives every request a unique prompt,
and its oracle sharing rate is 0.000 by construction. The cache measured a
**0.335 block hit rate** on that workload — which looks like an instrumentation
bug and is not one.

Measured directly against the real tokenizer:

```
two UNRELATED prompts, 48 and 45 tokens after the chat template
shared LEADING tokens: 30
  -> 1 full 16-token block identical for EVERY request
  -> at a 3-block prompt: 33% of blocks, shared for free
```

The generator's oracle measures sharing in **raw prompt tokens**. The cache
operates on **templated** tokens, and `apply_chat_template` prepends ~30 tokens
of role and header scaffolding to every single request. One full block of that
scaffolding is therefore always reusable, no matter how unrelated the prompts.

0.335 measured vs 33% predicted from the template length. The instrumentation is
right; the oracle and the cache were measuring different token streams.

**Consequence for the methodology:** the mandatory zero-sharing control
(§7) does not measure "the cache with nothing to reuse". It measures "the cache
with only the template preamble to reuse". A true zero-sharing control would
have to bypass the chat template entirely. This is recorded rather than
corrected, because the templated behaviour is what a real server actually does.

---

## 2. What the cache costs when it barely helps

```
zero sharing: block hit rate 0.3346
  turning the cache ON costs +36.24 ms at TTFT p50 and +404.14 ms at p99,
  for essentially no benefit
```

Published whether or not it flatters, per methodology §7. That is the honest
floor: radix walk, insert, and refcount bookkeeping on every request, against a
single reusable template block whose prefill was cheap anyway.

**+404 ms at p99 is material.** It is a direct argument for making the cache
adaptive — skipping the trie walk when recent hit rates are low — which is not
built.

---

## 3. Hit rate and TTFT vs sharing rate

45 s steady-state windows, 290 requests per cell, cache-on and cache-off servers
at matched workload and seed. `delta` is cache-on minus cache-off TTFT p50, so
**negative means the cache helped**.

| structure | share | oracle | measured hit | TTFT p50 | delta p50 | valid |
|---|---|---|---|---|---|---|
| zero | 0.00 | 0.000 | 0.335 | 125.6 | **+36.2** | INVALID |
| system | 0.00 | 0.000 | 0.381 | 148.1 | +58.2 | OK |
| system | 0.50 | 0.132 | 0.431 | 147.8 | +58.5 | OK |
| system | 0.75 | 0.196 | 0.461 | 146.6 | +57.1 | OK |
| system | 1.00 | 0.263 | 0.492 | 138.6 | +49.5 | OK |
| conversational | 0.00 | 0.000 | 0.372 | 163.5 | +74.8 | OK |
| conversational | 0.75 | 0.686 | 0.636 | 202.1 | +59.1 | INVALID |
| conversational | 1.00 | 0.769 | 0.692 | 165.9 | **−33.0** | INVALID |
| adversarial | 0.25 | 0.055 | 0.403 | 150.2 | +61.9 | OK |
| adversarial | 0.75 | 0.195 | 0.466 | 140.8 | +52.5 | OK |

**The cache did not pay for itself on this hardware and workload.** TTFT is
*worse* with the cache on in every valid cell. Hit rate rises with sharing rate
exactly as designed (0.335 → 0.692), and the one cell where the cache wins
(conversational at share 1.00, −33 ms) failed its steady-state check and cannot
carry a claim.

### Why this is a plausible result rather than a broken one

Prefill on an H200 for a ~150-token prompt is *fast* — a few milliseconds. The
cache saves prefill compute and pays trie walk, refcount bookkeeping and
allocator pressure on every request. When the thing being saved is already cheap,
the bookkeeping dominates.

The methodology predicted the shape of this (§10 case 4, about routing, applies
to the cache too): **the benefit shrinks toward zero as prompts get shorter**,
and there is a prompt length below which the cache is net negative. At ~150
tokens on an H200 this workload sits below that line.

**What would move it:** longer prompts (the saving scales with prefill work),
slower prefill (a larger model), or deeper conversational reuse where whole
multi-turn histories are shared. All are measurable with the harness as written;
none were run.

---

## 4. Validity

- **NOT PUBLISHABLE:** the run reports `dirty=True` — untracked artifacts from
  an earlier job were present in the working tree. The numbers above are
  therefore *indicative*, not publication-grade, and are labelled as such.
- Many cells failed the steady-state check at 45 s windows.
- The correctness gate does **not** pass (`FINDING_batch_shape_numerics.md`).

**Phase 4 is reported incomplete on all three counts.** It earns no resume
claim.

---

## 5. What was verified

The cache itself is functionally correct where it can be tested independently of
the engine's numerics:

- block-boundary truncation, offset sweep 16/16
- chunked prefill under a partial hit
- eviction, refcounting, leak checks
- the `zero` control's end-to-end output
- 71 CPU tests, mutation-tested: reusing a straddling block kills 32 tests,
  dropping the refcount filter kills 6, dropping leaf-first eviction kills 11

A real bug was found and fixed here: `_cache_insert` published
`prefill_ids + output_ids[:-1]`, but after a RECOMPUTE resume `prefill_ids`
already *is* `prompt + output_ids`, so blocks past the resume point were filed
under duplicated tokens — correct KV, lying key.
