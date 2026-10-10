# Findings in our own server (cross-engine study)

Bugs and inaccuracies found in **this** serving layer while running the
cross-engine study (ADR-025, `docs/xengine/SPEC.md`). They are logged here and
**not fixed in-study**: SPEC hard rule 4 freezes `serving/` and `vendor/` so the
system being measured is the system that was frozen. Each entry says what the
consequence is for the study, and how the harness works around it until it is
fixed after the study.

Line numbers are as of base commit `e391b50`.

## Entry format

- **Where:** `file:line`
- **What:** the observed behavior, verified in code
- **Why it matters:** consequence, especially for benchmarking
- **In-study handling:** what the harness does instead of fixing it
- **Status:** open | fixed after study (<commit>)

---

## F-001 — `/health` reports `prefix_cache: False` and `preemption: False` even though both features exist

- **Where:** `serving/server/app.py:984-985`, inside the `capabilities` block of the `/health` payload.
- **What:** both values are hardcoded literals, not read from the running configuration. Both features are present and on by default: the radix prefix cache is enabled unless `SERVING_PREFIX_CACHE=0` (`serving/server/app.py:1157`, constructed at `:1250-1253`); preemption defaults to `PreemptionPolicy.RECOMPUTE` (`serving/scheduler/scheduler.py:320`) and can be switched to swap via `SERVING_PREEMPTION` (`serving/server/app.py:1260-1261`). The same payload contradicts itself: its `scheduler` field (`Scheduler.snapshot()`, `serving/scheduler/scheduler.py:1254-1278`) reports `preemption_policy` and, when the cache is on, `cache_*` counters.
- **Why it matters:** anything that discovers capabilities from `/health` — a router, or the study harness recording engine configuration into the artifact — would record "no prefix cache, no preemption" for an arm that has both. That would mislabel the `ours` arm as equivalent to `ours-noprefix`, and make a W3 (shared-prefix) or W4 (forced-preemption) result look like it came from a system without the mechanism being measured.
- **In-study handling:** the harness does not read `/health.capabilities`. It records configuration from the launch environment (`SERVING_PREFIX_CACHE`, `SERVING_PREEMPTION`) and cross-checks it against the startup `config:` line (`serving/server/app.py:1263-1269`) and the `scheduler` snapshot counters.
- **Status:** open.

## F-002 — FlashInfer silently falls back to PagedTorch on any exception, and the backend in use is never recorded

- **Where:** `serving/server/app.py:1208-1223`.
- **What:** with `prefer_flashinfer` true (the default, `:1115`), the server constructs `FlashInferBackend` inside a bare `except Exception:` that sets `backend = None` (`:1217-1218`), then builds `PagedTorchBackend` instead (`:1219-1223`). The exception is neither logged nor re-raised. The startup `config:` line (`:1263-1269`) prints `kv_blocks`, `preemption`, `prefix_cache`, `static_batching` — not the attention backend. `Scheduler.snapshot()` and `/health` do not report it either. The fallback is intentional (the comment cites R18); the defect is that it is **silent**.
- **Why it matters:** the most dangerous item for the study. A broken FlashInfer install (wrong CUDA version, wrong GPU arch, missing JIT cache on a compute node) would not fail the run; it would produce plausible but much slower numbers from the PyTorch reference backend in an artifact that looks valid. In a cross-engine comparison that inflates the gap to vLLM/SGLang by an amount unrelated to the mechanisms being attributed — the "plausible wrong number" threat class of `docs/BENCHMARK_METHODOLOGY.md` §12.
- **In-study handling:** SPEC requires the `ours` arm to record the active attention backend and treats a PagedTorch fallback as invalid. Since the server does not expose it, the harness establishes it independently — a pre-flight that imports and constructs `FlashInferBackend` in the same environment and on the same GPU — and records the result in `validity.attention_backend`. If it cannot be established, the field is recorded as unknown and the run is not published as valid.
- **Status:** open. Post-study fix: log the caught exception; add `attention_backend` to the startup `config:` line and to `/health`.

## F-003 — `docs/RESUME_BULLETS.md` said P1 ran on H200; the artifact says H100

- **Where:** `docs/RESUME_BULLETS.md:99` ("Artifact: `results/p1/`, job `11598444`, H200.").
- **What:** every JSON under `results/p1/` carries `"gpu_name": "NVIDIA H100 80GB HBM3"` (e.g. `results/p1/20260801T030324_atl1-1-01-006-19-0_capacity_s1.json:191`); `results/p1/RESULTS.md:20` and `README.md:49` both say H100. P2/P4/P5 did run on H200, the likely source of the slip.
- **Why it matters:** ADR-018 requires every published claim to resolve to an artifact; a hardware claim contradicting its own artifact is the kind of detail an interviewer checks.
- **Status:** fixed 2026-10-08 (`H200` → `H100`). The file is gitignored (private), so the fix is local-only and has no commit.

## F-004 — Preemption fires while evictable prefix-cache blocks are still held (reproduced in W4 v2)

- **Where:** found while reading source for `docs/xengine/SOURCE_NOTES.md` (see its "Ours" section for the file:line trail).
- **What:** the scheduler's preemption check counts only free blocks; radix-cache eviction runs only at admission; and the server builds the cache with no `max_cached_blocks` bound. So a running sequence could be preempted while unreferenced cached blocks that could have been evicted are still resident.
- **Why it matters:** if real, W4 with the prefix cache on would show more preemptions than necessary, which would be a policy cost to attribute to us rather than to memory pressure.
- **In-study handling:** W4 records preemption and eviction counters per run; an `ours` vs `ours-noprefix` preemption gap on W4 is the reproduction. Not fixed in-study.
- **Reproduced (W4 v2, job 13933751).** At 4 sequences in flight — total demand ≈10k tokens, well inside the 32,768-token pool — `ours` preempted 111, 143 and 141 times in its three repetitions while evicting 5,975–8,016 cached blocks; `ours-noprefix` preempted 0 times in all three. The cache-on run also ended with `blocks_free=0`, `cache_cached_blocks=2048`, `admission_control_alarm=True`, and 108,156 tokens recomputed. With nothing else differing between the two arms, preemption here is caused by cached blocks occupying the pool, which is this finding's hypothesis.
- **Mechanism, located in code (2026-10-10).** `_preempt_if_needed` decides whether the step fits by comparing the step's block demand with `self.allocator.num_free` only (`serving/scheduler/scheduler.py:855-856`); it never asks the cache to give blocks back before choosing a victim. Admission *does* reclaim cache (`_can_admit` falls back to `prefix_cache.reserve`, `:505-511`), so the asymmetry is: admission evicts cache, decode growth preempts people. Worse, preempting does not free cached blocks: `_preempt_recompute` calls `req.blocks.free()` (`:940-942`), which drops only the request's own reference; any block the cache also references stays allocated to the cache. So a preemption can free zero usable blocks, and the loop preempts again.
- **CPU reproduction:** `docs/xengine/repro/f004_f007_cpu_repro.py` drives the unmodified scheduler, allocator and radix cache with the CPU test model. Demand fits the pool in every configuration; cache off → 0 preemptions, cache on → 27–136 preemptions.
- **Status:** reproduced 2026-10-10 on GPU (W4 v2, controlled by `ours-noprefix`) and on CPU (repro script); mechanism located; not fixed in-study.

## F-005 — No streaming `usage` and no Prometheus eviction counter

- **What:** our server ignores `stream_options.include_usage`, so prompt-token counts cannot be read from the stream as they are for vLLM/SGLang; and cache evictions are only in the JSON scheduler snapshot (`cache_evictions`), not in `/metrics/prometheus`.
- **In-study handling:** the harness counts prompt tokens with the tokenizer (or records null with a note) and reads evictions from the JSON snapshot.
- **Status:** open.

## F-006 — Under a small KV pool, the server can sit with every block held by the prefix cache and make no progress

- **Where:** observed, not yet localized in code. Evidence: `results/xengine_w4_v1/W4/ours/` (job 13931404, H200, `SERVING_KV_BLOCKS=2048`, prefix cache on, recompute preemption).
- **What:** two W4-v1 cells (`rate2_rep1`, `rate1_rep2`) completed **zero** of 186 and 86 requests; every request ended `no_content` (stream closed before a first token). The scheduler snapshot taken *before* `rate2_rep1` reads `running=0 waiting=0 blocks_free=0 blocks_used=2048`: the pool was entirely held by cached blocks with no request on the server. During the cell the scheduler advanced 772 steps and evicted 16,512 cache blocks, yet produced no first token for any of 256 requests. `starvation_fallbacks` was 1,619 before that cell and 105,769 before `rate1_rep2`. A different cell starting from the same full-cache state (`rate4_rep1`) did make progress, so the state is not a permanent deadlock.
- **Related:** the same run preempted 1,991 times at 1 request/s, where vLLM and SGLang preempted 0 times on the same pool size — although v1's open-loop design gave ours far more requests in flight (see W4 v2 rationale), so that count is not a like-for-like comparison. F-004 (preemption while evictable cached blocks are held) is a candidate mechanism, not a confirmed one.
- **Why it matters:** a serving system that can stop producing tokens under memory pressure, with no error, is the worst failure mode for availability; it also makes any W4 number for ours depend on the state left by the previous point.
- **In-study handling:** W4 v2 adds `ours-noprefix` (does the state disappear without the cache?), records the server's running/waiting/block state before every point (`server_counters.raw.pre_point_state`), and caps each closed-loop point at 1,200 s so a stuck server yields an invalid point (`point_deadline_exceeded`) instead of a hung job. Not fixed in-study.
- **Reproduced at full pool size (W2, job 13931402) — the mechanism is cache occupancy that only grows.** In W2 (unique prompts, default VRAM-sized pool of 277,590 blocks) the `before` scheduler snapshot of successive `ours` cells shows `running=0 waiting=0` every time while `blocks_free` falls (7 of the 18 cells, in run order): 277,590 → 223,571 → 167,282 → 75,304 → 49,624 → 20,106 → 0. The radix cache keeps every finished prompt's blocks; nothing evicts them until admission needs space (F-004 notes the server builds the cache with no `max_cached_blocks` bound). Once the free pool reaches zero, `ours` collapses: goodput 0.00 at every rate in repetition 3 (including rate 1, where it was 0.90 in repetition 1), 26 of 2,013 requests completed in `rate32_rep2`, and 0 of 1,920 in `rate32_rep3`. vLLM, also with prefix caching on, shows no such drift over the same 18 cells.
- **Controlled by the cache-off arm (W3, job 13931403).** Over 15 cells, `ours` went from 277,590 to 20,634 free blocks at the start of its last cell; `ours-noprefix` stayed at exactly 277,590 throughout. W1 (`ours`) never dropped below 87,166 (31% free), so W1 results are not in this state.
- **Consequence for the study:** `ours` results on W2, and the late cells of W3, depend on the order points ran in. The renderer flags every cell whose run began with <5% free blocks and no requests (`pool_held_by_cache_at_start`, read from the artifact's own before-snapshot) instead of dropping it.
- **W4 v2 (job 13933751):** a fresh `ours` server reached `blocks_free=0` with every block cached during its first point (concurrency 4, demand ≈10k of 32k tokens), and started its concurrency-16 and -32 points with the pool full of cache (flagged `pool_held_by_cache_at_start`). `ours-noprefix` started every point with all 2,048 blocks free.
- **Mechanism, located in code (2026-10-10).** The server constructs `RadixCache(allocator, block_copy=...)` with no `max_cached_blocks` (`serving/server/app.py:1250-1253`), so `_enforce_budget` never runs (`serving/cache/radix.py:609`, returns immediately when the budget is `None`). The only paths that evict are `reserve()` / `evict()` called from admission. Every finished or preempted request's full blocks stay in the trie at refcount 1, so occupancy ratchets up until the cache holds the whole pool.
- **Status:** open; mechanism located in code; reproduced on GPU (W2, W4 v2) and on CPU (`docs/xengine/repro/f004_f007_cpu_repro.py`: cache on ends with 0–2 free blocks, cache off with all blocks free); consequences in F-004 and F-007.

## F-007 — Under KV pressure with the prefix cache on, some requests "complete" with no output

- **Where:** observed, not localized. Evidence: `results/xengine/W4/ours/concurrency4_rep*.json` (job 13933751).
- **What:** at W4 concurrency 4, `ours` (cache on) had 12, 15 and 18 of its 48 measured requests end `no_content` on the client — the stream closed without a single content chunk — in repetitions 1–3. The server nevertheless counted every request as completed (`requests_completed` 56 of 56 received in repetition 1) while emitting 20,330 output tokens against 28,672 expected (56 × 512); `ours-noprefix` on the identical request stream emitted 28,486 and had 0 `no_content`. `starvation_fallbacks` rose by 13 in the same cache-on run; the fallback itself still preempts rather than terminating a request (`serving/scheduler/preemption.py:185-194`), so it is a correlate, not an established cause.
- **Why it matters:** a request that returns success with no output is a silent correctness failure, worse than an error; any goodput or throughput number for `ours` under pressure includes it.
- **Mechanism, reproduced on CPU (2026-10-10).** This is F-004's loop reaching its last branch. When every preemption frees nothing (the victims' blocks stay with the cache), `_preempt_if_needed` keeps preempting until one request is left, then calls `_fail_oversized` (`serving/scheduler/scheduler.py:861-869`, `:1090-1106`), which marks it `FAILED` with "sequence outgrew the KV pool … holds N of M, and it is the only running request". The message is false in this situation: the request holds 10 of 32 blocks and the rest are evictable cache. `docs/xengine/repro/f004_f007_cpu_repro.py` produces 9 such failures out of 48 requests with the cache on and 0 with it off, on a workload whose demand fits the pool.
- **Why the server calls it a success.** A failed request is retired through the normal finish path, so the streaming handler maps it to `finish_reason: "error"` (`serving/server/app.py:637-638`) but counts it under `requests_completed` (`:828-836`; only the separate `"error"` event kind increments `requests_failed`), and the error text is never sent to the client. A client that does not inspect `finish_reason` sees an empty, successful response — what the harness recorded as `no_content`.
- **Link to the GPU evidence:** inferred, not proven per request — the W4 artifacts do not store per-request finish reasons. The CPU reproduction shows the mechanism produces exactly this symptom; the GPU counts (12–18 no-content requests, 0 with the cache off) are consistent with it.
- **In-study handling:** reported, counted (`metrics.outcomes.no_content`), not fixed.
- **Status:** open; mechanism reproduced on CPU; fix is post-study (count reclaimable cache blocks in the preemption check, evict cache before choosing a victim, and report scheduler failures as failures).
