# Open-source contribution candidates (vLLM / SGLang)

## Purpose

Setting up vLLM and SGLang for the cross-engine study (ADR-025,
`docs/xengine/SPEC.md`) will hit friction: a flag the docs don't explain, an
error message that points at the wrong cause, a crash that reproduces. Each of
those is a lead for an upstream contribution — a docs PR, a clearer error, or a
bug report with a repro. This file captures them **at the moment they happen**,
while the exact command, version, and output are still on screen.

Why it matters: friction that is not written down immediately gets worked
around and forgotten, and a vague memory ("the SGLang flags were confusing") is
not something an upstream maintainer can act on. A repro with a pinned version
is.

## Rules

- **Entries come only from friction actually encountered** while running this
  study. No speculative entries, no issues found by reading other people's bug
  reports, no "this might be confusing".
- Every entry names the **exact engine version** (pip version and, if built
  from source, the commit) and an **evidence path** — a log, terminal capture,
  or artifact committed under `results/xengine/` or `bench/xengine/logs/`.
- **Search upstream before proposing anything.** Record what was searched
  (issue tracker query, docs page) and what was found. A duplicate of an open
  issue becomes a comment on that issue, not a new one.
- An entry is a *candidate*. Nothing here is filed upstream without a fresh
  repro on the pinned version.

## Entry format

Copy this block for each entry. Number entries sequentially (`OSS-001`, …).

```markdown
### OSS-NNN — <one-line title>

- **Engine + version:** vLLM x.y.z / SGLang x.y.z (pip) — commit <sha> if from source
- **Category:** doc gap | confusing error | reproducible bug
- **Found while:** <what the study was doing, e.g. "launching the vllm-eager arm for W1">
- **Repro steps:**
  1. <exact command, with flags and env vars>
  2. ...
- **Expected:** <what the docs or a reasonable user would expect>
- **Actual:** <what happened; paste the key lines of the error verbatim>
- **Evidence:** <path to committed log/artifact>
- **Upstream search done?** yes/no — <queries run, links to related issues/PRs, or "none found">
- **Proposed fix:** <docs wording, error-message change, or patch sketch>
- **Status:** candidate | repro confirmed on pinned version | filed (<link>) | merged (<link>) | dropped (<reason>)
```

Category definitions:

- **doc gap** — the behavior is correct, but the documentation is missing,
  wrong, or out of date for this version.
- **confusing error** — the engine fails correctly, but the message points at
  the wrong cause or omits what the user must change.
- **reproducible bug** — the engine does the wrong thing, and the steps above
  make it happen every time on the stated version.

## Entries

Entries OSS-001..003 were first hit while wiring the harness's vLLM launch command
and startup-log parsing (2026-10-08), before any GPU run. Their evidence is the
source at the pinned commit; the runtime log line will be captured on the first
PACE launch and the status moved to "repro confirmed" only then.

### OSS-001 — Deprecation message tells users to run `vllm server`; the subcommand is `vllm serve`

- **Engine + version:** vLLM 0.31.0 — commit `db9527a4`
- **Category:** confusing error
- **Found while:** choosing how the harness launches vLLM under a per-engine interpreter (`python -m ...`); the obvious module is deprecated, and its warning names a command that does not exist.
- **Repro steps:**
  1. `python -m vllm.entrypoints.openai.api_server --model <path>`
- **Expected:** a DeprecationWarning pointing at the replacement, `vllm serve`.
- **Actual:** the warning says "Please use `vllm server` instead." (`vllm/entrypoints/openai/api_server.py:51-55`). The CLI subcommand is named `serve` (`vllm/entrypoints/cli/serve.py:49`); `vllm server` would fail with an invalid-choice error. Elsewhere vLLM's own messages use the correct name (`vllm/entrypoints/grpc_server.py:22`).
- **Evidence:** source at the pinned commit (lines above); runtime warning to be captured on first PACE launch.
- **Upstream search done?** no — to do before filing.
- **Proposed fix:** one-word change, `vllm server` → `vllm serve`.
- **Status:** candidate

### OSS-002 — Missing space in the module-level deprecation warning ("will likely beunsupported")

- **Engine + version:** vLLM 0.31.0 — commit `db9527a4`
- **Category:** doc gap (message text)
- **Found while:** same as OSS-001; importing the module emits this warning.
- **Repro steps:**
  1. `python -W always -c "import vllm.entrypoints.openai.api_server"`
- **Expected:** "...is deprecated and will likely be unsupported in a future version..."
- **Actual:** two adjacent string literals concatenate without a space: `"...will likely be"` + `"unsupported in a future version..."` (`vllm/entrypoints/openai/api_server.py:24-30`) → "will likely beunsupported".
- **Evidence:** source at the pinned commit; runtime warning to be captured on first PACE launch.
- **Upstream search done?** no — to do before filing.
- **Proposed fix:** add the trailing space to the first literal. Can ride in the same PR as OSS-001.
- **Status:** candidate

### OSS-003 — Auto-derived `max_num_seqs` / `max_num_batched_tokens` are never logged

- **Engine + version:** vLLM 0.31.0 — commit `db9527a4`
- **Category:** doc gap (observability)
- **Found while:** making the harness record each engine's *resolved* scheduling limits per run, which this study needs because vLLM derives them from GPU memory (`docs/xengine/SOURCE_NOTES.md` §2.7).
- **Repro steps:**
  1. `vllm serve <path>` with neither flag set; read the startup log.
- **Expected:** the values the scheduler actually uses appear somewhere in the startup log (as SGLang does for `max_running_requests` and `chunked_prefill_size`, `managers/scheduler.py:1229-1237`).
- **Actual:** the startup logs that exist are the "non-default args" dict (`entrypoints/serve/utils/api_utils.py:286`), which only lists flags the user passed; the engine-config line (`v1/engine/core.py:130`, built at `config/vllm.py:2840+`) carries `max_seq_len` but not these two; and the KV-capacity line (`v1/core/kv_cache_utils.py:2464`) gives tokens and max concurrency only. Reproducing a benchmark therefore requires knowing GPU memory and re-deriving the defaults by hand.
- **Evidence:** source at the pinned commit; harness consequence in `bench/xengine/engines.py` (`VLLM_RESOLVED_RES` comment). Runtime: the vLLM server log of pilot job 13918362 (`results/xengine_pilot/W1/vllm/server_20261009T225009Z.log`, H200) contains zero occurrences of `max_num_seqs`; it does log `max_num_batched_tokens`.
- **Upstream search done?** no — to do before filing.
- **Proposed fix:** add both resolved values to the engine-config log line, or log them once after scheduler-config resolution.
- **Status:** repro confirmed on pinned version (pilot job 13918362) for `max_num_seqs`; `max_num_batched_tokens` does appear, so the entry narrows to `max_num_seqs`.

### OSS-004 — SGLang's retraction and eviction counters are absent from `/metrics` until the first event

- **Engine + version:** SGLang 0.5.21 — commit `e00930c5`
- **Category:** doc gap (observability)
- **Found while:** reading preemption counts from the pilot run (job 13918362); the harness recorded "not exposed" for SGLang because the series did not exist.
- **Repro steps:**
  1. `python -m sglang.launch_server --model-path <path> --enable-metrics`
  2. Send any load that causes no retraction; `curl -s localhost:<port>/metrics | grep retracted`
- **Expected:** `sglang:num_retracted_requests_total{...} 0` and `sglang:evicted_tokens_total{...} 0`, so a scraper can tell "zero" from "not supported".
- **Actual:** only the old gauge `sglang:num_retracted_reqs` appears. The counters are prometheus_client Counters with label names (`observability/metrics_collector.py:474-479`, eviction counter near `:2213`), and a labeled child is created only on the first `.labels(...).inc()` (`:1270`). Until then the series does not exist.
- **Evidence:** `results/xengine_pilot/W1/sglang/concurrency1_rep1.json` → `server_counters.raw.before` (452 series, no `num_retracted_requests_total`, no `evicted_tokens_total`); source lines above.
- **Upstream search done?** no — to do before filing.
- **Proposed fix:** initialize each labeled child at collector construction (`self.num_retracted_reqs_total.labels(**labels)` with no increment), which exports a 0 series. This is the standard prometheus_client pattern for counters that should read 0 before their first event.
- **Status:** repro confirmed on pinned version (pilot job 13918362).
