# Build Log

How this system was built, in the order it happened, including the parts that
went wrong. Written as a companion to the design docs: those say what the system
*is*, this says what it cost to find out.

The engine it sits on has its own build log (`docs/BUILD_LOG.md` in
the `llm_inference_engine` repo). This one starts where that one stops — at a correct,
benchmarked, single-request model that could not serve anyone.

---

## 0. The starting position

Three facts about the engine, read from its source rather than its README:

1. **Batch 1 everywhere.** `x` is `(seq, hidden)` throughout
   (`engine/model_gpu.py:64`); the KV cache is
   `(n_layers, max_seq, n_kv_heads, head_dim)` with no sequence axis
   (`engine/cache.py:28`); the causal mask is 2-D
   (`engine/components_gpu.py:173`); the CUDA kernel takes `Q` as
   `[n_heads, head_dim]` with no batch dimension
   (`kernels/attention_decode.cu:30-32`).
2. **The HTTP server cannot serve the GPU model and serializes under load.**
   `engine/server.py:24-30` loads the fp32 NumPy CPU path; `:69` iterates a
   blocking synchronous generator inside an `async def`, so request 2's TTFT
   contains request 1's entire decode.
3. **Memory is allocated as if one request exists.** `engine/scheduler.py:26-27`
   builds a fresh `KVCacheGPU(max_seq=2048)` per call and drops it. A 30-token
   request reserves 2048 slots × 16 layers.

Everything below follows from those three.

---

## 1. Planning (deliverables 1–9)

Nine documents before any code: engine integration report, PRD, benchmark
methodology, architecture, phase plan, risk register, concept map, ADR log,
repo scaffold. Roughly 250 KB of prose. Six of the nine are published in
`docs/`; the planning and note-taking documents are not.

**Was that worth it?** The honest answer is that the planning documents needed
**six factual corrections and two design fixes before a line of code existed** —
found by running adversarial passes over my own output:

- A deadlocking starvation guard. The design said a sequence preempted K times
  becomes *ineligible* as a victim. If every running sequence reaches K, victim
  selection returns nothing while the batch is non-empty, and the scheduler can
  neither step nor make room. Fixed to a **preference ordering**, with the
  fallback instrumented as an admission-control alarm.
- A phase-ordering violation. The cooperative scheduler (P2) is only responsive
  because steps are short, and what bounds a step is chunked prefill — which was
  scheduled for P4. Split: a bare token cap in P2, chunked prefill as a
  *scheduling feature* in P4.
- I had written that the engine "commits no artifacts", citing its own audit.
  That audit was stale: `bench/results/` holds 24 tracked files. Corrected in six
  places with visible correction notes rather than silent edits.

The planning docs are good not because they were written carefully but because
they were **checked hostilely**.

---

## 2. Phase 0 — prerequisites and the oracle that did not exist

**The finding that shaped everything after it:** the engine had **no
model-level correctness test for the GPU path at all**.

- `tests/test_gpu_model.py:42-45` asserted only that logits were finite,
  correctly shaped, and had an argmax inside the vocabulary. A model returning
  confident nonsense passes all three.
- Every real correctness claim — logits < 1e-3, 32 greedy tokens bit-identical —
  described the **CPU fp32 NumPy** path (`tests/test_forward.py:113,137`,
  `tests/test_decode.py:30-48`).
- Every published performance number described the **GPU fp16** path.

So correctness and performance were measured on different code. Tolerable for a
batch-1 harness; not tolerable as the foundation for a paged, batched,
preemptible scheduler, where a later disagreement would be unattributable across
the allocator, the batching, the paged attention, and a pre-existing GPU defect.

Also found and fixed: **all five tests in `tests/test_generate.py` raised
`KeyError`** (`oracle_short["input_ids"]` vs the fixture's `"token_ids"`), so
the KV-cache-vs-no-cache gate had never actually run.

**Result** (job `11596894`, A100): GPU fp16 greedy output matches the fp32 HF
oracle for 16 tokens exactly on both fixture prompts, matches the CPU fp32 path,
prefill logits agree to max|Δ| 0.0111 with 10/10 top-10 overlap.

**Cost measured, not assumed:** 0.08 SU for 17:02 on one A100 =
**0.282 SU/A100-GPU-hour**. ~3,500 GPU-hours remain; SUs are not a constraint.

---

## 3. Phase 1 — paged KV, and a claim that depends on the workload

The engine's contiguous `max_seq=2048`-per-request allocation replaced with a
fixed pool of 16-token blocks, a free list, refcounting, and per-sequence block
tables.

**Three decisions worth the space they take in the code:**

- **FIFO free list, not LIFO.** LIFO hands back the just-freed block
  immediately, which makes use-after-free *invisible* — the stale reader sees
  data that still looks right. FIFO maximises delay before reuse so the bug
  surfaces as obvious garbage.
- **Refcounts live in the allocator, not the prefix cache.** If two components
  can both decide a block is free they will eventually disagree, and the failure
  is silent cross-sequence KV contamination.
- **`can_allocate()` respects the watermark; `allocate()` does not.** Admission
  must leave headroom; a sequence already admitted must not be starved by
  headroom it is itself the reason for.

**Two backends, deliberately.** `PagedTorchBackend` (pure PyTorch, block-gather
→ SDPA) was written **first**, as the correctness oracle. `FlashInferBackend`
came second and was checked against it. That ordering is the entire reason a
FlashInfer layout misunderstanding would have been attributable.

Reading FlashInfer's **kernels** rather than its docstrings produced three facts
the documentation does not state:

- `causal=True` is **bottom-right aligned** (`prefill.cuh:1461`), matching this
  system's convention exactly. PyTorch SDPA's `is_causal` is *top-left* aligned
  and would have been silently wrong for decode — which is why
  `PagedTorchBackend` avoids SDPA.
- `plan()` stores the page tables **on the wrapper**; `run()` takes none. **A
  missing `plan()` silently attends over the previous step's page table.**
- `run()` returns `(tokens, n_heads, head_dim)`.

**Stated limit:** the differential is **token-exact, not bit-exact**.
`PagedTorchBackend` casts `probs` to fp16 before the PV matmul (mirroring the
engine); FlashInfer uses a fused fp32 online softmax and never materialises
`probs`. They cannot agree bit-for-bit by construction. Tensors compare at
`atol=4e-3`; **tokens compare exactly**, which is the claim the system makes.
Defensible because every failure mode this gate exists to catch is an O(1)
error, not an O(1e-3) one.

**The S1 result, and why the table is the result.** Measured by driving the real
allocator to admission failure:

| realized mean length | paged | contiguous | ratio |
|---|---|---|---|
| 32 | 29,618 | 571 | 51.9× |
| 128 | 8,615 | 571 | 15.1× |
| 258 | 4,406 | 571 | 7.7× |
| 505 | 2,280 | 571 | 4.0× |
| 1420 | 820 | 571 | 1.4× |

The ratio is approximately `max_seq / realized_padded_length`. It is **not a
property of the allocator alone** and falls to ~1× when sequences actually use
all 2048 slots. Quoting 51.9× bare would collapse under one follow-up question.

---

## 4. Phase 2 — batching, and the bug the whole design was arranged around

Batched varlen forward, iteration-level scheduler, production HTTP surface,
open-loop load harness.

**Why varlen made this tractable.** With all sequences packed on one token axis,
every position-independent operation is *literally unchanged*: `linear`,
`rms_norm_gpu`, `swiglu_ffn_gpu`, the embedding gather. RoPE was already
batch-ready because `positions` is a passed tensor. Only attention, the cache
write, the causal mask, and the last-token gather change. **Batched decode and
paged KV are one change to one surface, not two projects.**

**The server exists to not repeat `engine/server.py:69`.** Proving the fix took
more care than writing it: `httpx.ASGITransport` buffers the whole response
body, so client-side chunk timing through it is a transport artefact and a naive
interleaving test measures nothing. The test speaks ASGI directly, logs both
requests' body messages into one timestamped transcript, and gates request 2 so
it cannot start until request 1 has emitted 5 of 200 tokens. Measured: request 2
reached its first token at **2.4% of request 1's stream span**, 192 of request
1's 200 tokens arrived after it, and the longest single-request run inside the
overlap was **1 chunk** — exact step-granularity alternation. Mutation-tested by
patching the loop to drain the batch before yielding, a literal reproduction of
the engine's bug; the assertion kills it and names that line.

---

## 5. Phase 3 — preemption

Recompute and swap behind one policy flag, LIFO victim selection, and a
starvation guard implemented as a **preference ordering rather than an
exclusion** — the naive "ineligible after K preemptions" rule deadlocks the
moment every running sequence reaches K, because victim selection returns
nothing while the batch is non-empty.

**The gate passes** (job `11608158`): greedy output under forced memory pressure
is bit-identical to an unpreempted run, for recompute *and* swap independently.
That closes the one thing that could not be shown on CPU — recompute re-prefills
as a single chunk what was originally many decode GEMVs, so its fp16 stability
was a genuine open question. Swap is immune by construction.

Two properties of the design surfaced only because tests refused to pass without
them:

**A single pressure wave can preempt a given request at most once.** LIFO
selection plus requeue-at-front plus `_can_admit` accounting for
`blocks_needed_for_step` (the anti-livelock guard) means a victim is not
readmitted while the pressure that evicted it persists — and by the time it is,
anything newer becomes the next victim. Repeated eviction of one request is
something the scheduler actively avoids. Good behaviour; a bad basis for a test,
so that test now drives `_preempt` directly and asserts the *resume path*.

**An unadmitted request is never preempted — it queues.** The original
chunked-prefill test added the long prompt last so LIFO would select it, which
cannot work: under pressure it was never admitted. It has to be resident and
mid-prefill *before* the pressure arrives.

### A real bug the benchmark found

The swap arm reported **36 preemptions at a preemption rate of 0.0000** — an
enormous step count with nothing completing — and an e2e p99 of 179,954 ms,
which is a client timeout rather than a latency. The driver marked every swap
cell INVALID and refused to publish.

`_resume_swapped` gated on `can_allocate()`, which reserves the **admission**
watermark. But a swapped request is not new work: it was admitted, ran, and had
its memory taken away. Under sustained pressure the watermark is never
satisfied, so the request parks forever while the loop keeps stepping.

The codebase already states the correct rule one function away, in
`_reuse_cached_prefix`: *"the watermark exists to stop admission, not to starve a
sequence that has already taken references."* The same principle, applied
inconsistently. The regression test was verified to fail on the old code before
being kept.

## 6. Phase 4 — radix prefix cache

**Correctness is earned; performance is not.** The cache returns correct KV —
verified bit-identical against a no-cache baseline at matched GEMM shape across
all four sharing structures at 38-48% block reuse (job `11611626`). What is not
earned is a performance claim: the measured effect at ~150-token prompts was
negative.

Getting to that took two wrong gates. The first compared runs whose forward
passes had different shapes, so it could not distinguish "the cache served wrong
KV" from "the GEMM summed in a different order". The second tried batch size 1
and failed identically — because a cache HIT changes how many tokens are
prefilled (M=105 vs M=41), the shape difference is **intrinsic to caching**, not
to batching. Capping prefill at one block per step finally isolated it: every
GEMM has the same M in both arms and the cache simply performs fewer of them.
Under that configuration reduction order cannot vary, so any divergence would
have to be the cache — and there is none. The batched gate still fails on the
same structures in the same job, which is the confirming half.

Two findings came out of it that are worth more than a passing number would have
been.

**Batched output is not bit-invariant to batch shape.** The gate failed on
exactly the structures with real prefix reuse, which reads as a cache bug and is
not one. Three controls plus an independent re-derivation without the cache,
scheduler or preemption in the loop locate the cause in `linear()`'s `M`
dimension — the packed batch token count — where cuBLAS selects different
kernels and split-K and therefore a different fp16 reduction order. Same prompt,
same positions, cache off in both runs, differing only in surrounding traffic:
max|Δlogit| 0.1745. Forcing uniform batch shapes drives it to exactly zero.
Full write-up in `results/p4/FINDING_batch_shape_numerics.md`. This narrowed
Phase 2's claim as well.

**"Zero prefix sharing" does not exist in a chat server.** The generator's oracle
reports 0.000 for unique prompts; the cache measured 0.335. Not instrumentation:
`apply_chat_template` prepends ~30 tokens of role and header scaffolding to every
request, which is one full 16-token block, and at a 3-block prompt that is
exactly 33%. The oracle counts raw prompt tokens; the cache operates on templated
ones. The mandatory zero-sharing control therefore measures the cache with *only
the preamble* to reuse.

**Then three runs to find out whether the cache is worth having, two of which
measured nothing.** The performance question took longer than the correctness
question and was wrong twice.

Run 1 put all 36 cells on one server and reported **+416 ms at 1024 tokens and
+907 ms at 2048** on the zero-sharing control. Read literally: prefix caching
makes serving dramatically slower. The run's own arithmetic said otherwise — at
1024 tokens a 40,000-block pool holds 625 requests' worth of blocks against ~90
offered, so eviction could not legitimately fire, and it fired 11,673 times. A
cache holds a reference to every block it caches, so cached blocks are never
returned as requests retire; across four lengths the trie ate the pool and every
later cell was contaminated by the earlier ones. Run 2 restarted per *length* —
still nine cells and ~1,620 requests per server, and the 512-token cells duly
logged 4,759 and 8,968 evictions.

Run 3 (`11617299`) used **one fresh server pair per cell**, 36 restarts, ~15
minutes of startup bought in exchange for cells that do not contaminate each
other. With that, at 512 tokens with a shared system prefix at 50% sharing:
**−6.3 ms TTFT p50 and −37.7 ms p99, a 23% tail reduction**, evictions 0,
n = 118/118. At 150 tokens with conversational reuse at 76% block hit rate:
**−17.3 ms p50** on a 64.8 ms baseline. The zero-sharing control costs **+0.4 …
+1.1 ms**, so the cache can be left on by default.

Two things that took the whole exercise to see. First, **the win lands in the
tail, not the median** — a prefix hit does not speed up the request that misses,
it removes prefill work from the queue, which shortens everyone else's wait; p99
moves 23% in a cell where p50 moves 7%. Second, **an earlier revision of this
log said "the cache did not pay for itself"** on the strength of the +36/+75 ms
figures. Those were the zero-sharing control — the arm where the cache is
*supposed* to do nothing — quoted as if it were the result. Reading a control as
an outcome is its own failure mode and it is recorded here rather than edited
away.

1024 and 2048 tokens are still **untested**: even a single cell saturates the
pool at those lengths, so all 24 cells are INVALID. The sizing comment in
`scripts/p4_clean.sbatch` that proved this impossible was wrong — the window is
90 s not 45, lengths are lognormal, and the cache frees nothing mid-cell.

Break-even sharing rate, from the clean 150-token cells: **0.058 conversational,
0.452 system**. A shared system preamble is only ~30 tokens and has to clear
block granularity before it returns anything.

## 7. Phase 5 — routing, and the claim that was not earned

Four replicas, one per **dedicated** H200, three router processes over the same
fleet — `prefix_aware`, `least_outstanding` (B5), `round_robin` (B4) — five
offered loads × four scenarios × three policies. Job `11610306`.

**S5 was not earned.** Prefix-aware routing was never shown to beat the real
baseline on a workload where it should, because those workloads were never
measured inside their valid range. Three of four scenarios —
`system_prompt_sharing`, `uniform_prefix`, `hot_prefix_skew` — collapsed at
*every* offered load: goodput 0.00, TTFT p50 between 10 and 65 seconds. They
carry a long shared prefix, so each request costs far more prefill than
`zero_sharing`'s, and four H200s saturate below the lowest load tested. The
loads were chosen from `zero_sharing`'s capacity and never re-derived for the
heavier scenarios. That is a workload-design failure of mine, not a routing
result, and the driver says so: `only 0 usable point(s); a crossover needs at
least 2` — *untested, which is not the same as confirmed.*

**What was earned is a confirmed losing-case prediction**, and it is worth more
than a fourth throughput number. `zero_sharing` was predicted in writing, before
any measurement, to *lose*:

> TIE with B5, or a slight LOSS: there is nothing to be cache-aware about, and
> any deviation from load-optimal placement is pure loss.

It lost, in the predicted direction, by a small margin: **Δgoodput −0.111 at load
4 and −0.311 at load 16 versus B5**, both cells valid at n = 173 and 708.

The detail that makes it honest: **prefix_aware beats round_robin at load 4**
(3.18 vs 3.02) while losing to B5 (3.29). Quoting the B4 comparison would have
produced a win. It would have been a win about *load balancing*, which B5 already
does, and nothing about prefix awareness. Methodology §6 exists to forbid exactly
that, and this is the run where it bit.

An earlier attempt (`11609657`) put all four replicas on one H200. They contended
for SMs, the fleet saturated by load 2, and 45 of 48 cells failed the
steady-state check — a property of the setup, not of routing. Hence four GPUs.

The Phase 5 benchmark is also the one place the harness caught *me* rather than
the system: `run_p5` refused to start because `--router least_outstanding=URL`
was missing, with the message *"B5 is the real baseline... it is load-aware and
cache-blind, so the gap between it and prefix-aware is the only thing that
isolates cache awareness."* I had started four replicas and no routers.

Full write-up: `results/p5/RESULTS.md`.

## 8. Phase 6 — operability

No benchmark claim, by design. A Prometheus exporter over the same metric
registry the artifacts use, so a dashboard and a published number cannot
disagree about what a name means. A CUDA guard that makes any CUDA error
**fatal to the replica** rather than recoverable — a device-side assert leaves
the context poisoned, and a replica that keeps answering after one is serving
wrong tokens with a healthy `/health`. A fault-injection harness driving replica
kill, hang, and slow-loris against the router's quarantine, hint purge and
drain paths.

## 9. Where it ended

**774 CPU tests, 100 GPU-gated**, the CPU suite running with no GPU, no weights
and no network — the allocator, radix trie, routing policy, batch assembly,
workload generation and metric schema are pure logic by design, so cluster queue
time never blocked development.

Four claims earned and defensible: paged KV capacity (S1), continuous batching
goodput (S2), preemption correctness (S3), prefix cache TTFT (S4). One not
earned (S5). The one not earned is stated as not earned in the README and here.

---

## 10. The pattern worth taking away

Six separate failures this project, all mine, all the same shape: **something
reported success while doing nothing.**

| What | How it looked |
|---|---|
| GPU gate on a V100 under CUDA-13 torch | 16 tests **skipped**, job exited 0 |
| Fragmentation test | **passed** — but the pool was never fragmented (FIFO free list put freed blocks behind untouched ones) |
| Server with a broken tokenizer | **HTTP 200**, well-formed SSE, zero content |
| SLO calibration | completed, produced **goodput 0.00 at every rate** from a negative TTFT |
| Benchmark summary table | printed **0.0** in all seven rows — read scalar keys that don't exist |
| P4 per-cell eviction audit | printed **`clean` for all 36 cells** — the metrics key was absent, `find()` returned `None`, and the guard `ev not in (None, 0)` classified *absent* as *zero* |

The sixth is the one to sit with: it was a check **added in response to the
fifth**, and it failed the same way the thing it was watching for failed. It
had two outcomes where it needed three — clean, contaminated, and *did not
look*. Any audit that cannot say "I did not measure" will eventually report a
missing measurement as a passing one. The evictions it existed to catch were
caught by the benchmark driver's own summary table instead.

None raised. None showed up as an error. Every one was caught by something
asserting a **positive property** — "the GPU is usable", "these pages are
non-contiguous", "this response has content", "this latency is physically
possible" — rather than by an absence of errors.

That is why `docs/RISK_REGISTER.md` is ordered by **detectability rather than
impact**, and why the register's severity scale is *Impact × Silence*. It was
written as a defence against hypothetical future carelessness. Every entry it
has caught so far has been my own.

The generalisable rule: **a test that cannot detect its own setup failing is not
a test.** The agent-written fragmentation test was better than the hand-written
one for exactly one reason — it asserted its precondition and failed loudly when
the precondition did not hold.
