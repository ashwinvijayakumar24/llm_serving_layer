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
