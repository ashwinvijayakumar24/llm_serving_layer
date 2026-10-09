# Engine flags and metric names — verified in source

This file lists the exact launch flags and Prometheus metric names the harness
(`bench/xengine/`) should use for each engine. Every entry was checked against
the argument parser or config dataclass of the pinned version, and cites
`path:line`.

| engine | version | commit |
|---|---|---|
| vLLM | 0.31.0 (tag `v0.31.0`) | `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| SGLang | 0.5.21 (tag `v0.5.21`) | `e00930c5489053f26d86b179cee0d087f846acbb` |

Paths are relative to each engine's repo root. For SGLang, `P/` means
`python/sglang/srt/`. Background on what each knob does is in
`docs/xengine/SOURCE_NOTES.md`.

## How the flags are generated (why the names look the way they do)

- **vLLM** turns each config dataclass field into a `--kebab-case` flag. Any
  boolean field gets argparse's `BooleanOptionalAction`, so both `--X` and
  `--no-X` exist (`vllm/engine/arg_utils.py:377-379`). That is why the way to
  disable prefix caching is `--no-enable-prefix-caching`.
- **SGLang** also generates flags from dataclasses (`P/server_args.py:283-285`,
  `P/arg_groups/fields/*.py`), replacing `_` with `-`
  (`P/arg_groups/arg_utils.py:288-290`). A boolean field becomes a plain
  `store_true` flag (`arg_utils.py:393-395`), so there are no `--no-X` forms;
  disabling is spelled `--disable-X`.

## 1. Launch flags

### vLLM 0.31.0 (`vllm serve <model> ...`)

| purpose | exact flag | citation |
|---|---|---|
| fp16 | `--dtype float16` (`half` also accepted) | `vllm/engine/arg_utils.py:937`; `vllm/config/model.py:99` |
| disable CUDA graphs **and** torch.compile | `--enforce-eager` | `arg_utils.py:960`; effect at `vllm/config/vllm.py:1694-1696` |
| disable CUDA graphs only (keep compile) | `--compilation-config '{"cudagraph_mode": "NONE"}'` (short form `-cc`) | `arg_utils.py:1818`; modes at `vllm/config/compilation.py:53-63` |
| disable prefix cache | `--no-enable-prefix-caching` | `arg_utils.py:1339` (+ `:377-379`) |
| max running sequences | `--max-num-seqs N` | `arg_utils.py:1682` |
| KV pool cap | `--num-gpu-blocks-override N` (unit: blocks; tokens = N × block size) | `arg_utils.py:1336`; `vllm/config/cache.py:136` |
| KV pool cap, by bytes (alternative) | `--kv-cache-memory-bytes B` (overrides the memory fraction) | `arg_utils.py:1332`; `cache.py:247` |
| GPU memory fraction | `--gpu-memory-utilization F` (default 0.92) | `arg_utils.py:1325-1330`; `cache.py:103` |
| block size | `--block-size N` (default 16) | `arg_utils.py:1324`; `cache.py:71` |
| disable chunked prefill | `--no-enable-chunked-prefill` (logs an "unsupported, may produce incorrect outputs" warning for decoder LMs) | `arg_utils.py:1713`; warning `:2936-2943` |
| chunk / step token budget | `--max-num-batched-tokens N` | `arg_utils.py:1668` |
| per-request chunk cap | `--long-prefill-token-threshold N` (default 0 = off) | `arg_utils.py:1700`; `vllm/config/scheduler.py:80` |
| metrics endpoint | **no flag needed**: `/metrics` is always mounted on the API port | `vllm/entrypoints/serve/instrumentator/metrics.py:66`, `:77` |
| keep metrics populated | do **not** pass `--disable-log-stats` | `arg_utils.py:1846`; `vllm/v1/engine/async_llm.py:192-197` |
| seed | `--seed N` (default 0) | `arg_utils.py:938`; `model.py:178` |
| disable CPU/GPU overlap (diagnostic) | `--no-async-scheduling` | `arg_utils.py:1739` |
| attention backend | `--attention-backend NAME` | `arg_utils.py:1064` |
| eviction histograms (opt-in) | `--kv-cache-metrics` | `arg_utils.py:1630` |
| cached tokens in `usage` | `--enable-prompt-tokens-details` | `vllm/entrypoints/openai/launchers/cli_args.py:145` |
| usage on every response | `--enable-force-include-usage` | `cli_args.py:151` |

### SGLang 0.5.21 (`python -m sglang.launch_server --model-path <model> ...`)

| purpose | exact flag | citation |
|---|---|---|
| fp16 | `--dtype float16` (`half` also accepted) | `P/arg_groups/fields/model.py:152-168` |
| disable CUDA graphs (current spelling) | `--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled` | `P/arg_groups/fields/exec_.py:486-499` |
| disable CUDA graphs (convenience form) | `--disable-decode-cuda-graph --disable-prefill-cuda-graph` | `exec_.py:537-544` |
| disable CUDA graphs (deprecated) | `--disable-cuda-graph` — still works, warns, disables both phases | `P/server_args.py:339-347`; `P/arg_groups/cuda_graph_hook.py:60-62` |
| disable radix cache | `--disable-radix-cache` | `P/arg_groups/fields/memory.py:64` |
| disable overlap scheduler | `--disable-overlap-schedule` | `P/arg_groups/fields/schedule.py:187` |
| max running requests | `--max-running-requests N` | `schedule.py:31` |
| KV pool cap | `--max-total-tokens N` (unit: tokens; can only **lower** the profiled pool) | `schedule.py:39`; `P/mem_cache/kv_cache_configurator.py:2262-2277` |
| GPU memory fraction | `--mem-fraction-static F` | `schedule.py:27` |
| page (block) size | `--page-size N` (resolves to 1 on CUDA) | `schedule.py:139`; `P/arg_groups/overrides.py:1278-1302` |
| disable chunked prefill | `--chunked-prefill-size -1` | `schedule.py:52-55`; `P/managers/scheduler.py:1304-1329` |
| chunk size | `--chunked-prefill-size N` | `schedule.py:52` |
| metrics endpoint | `--enable-metrics` (`/metrics` on the same HTTP port) | `P/arg_groups/fields/observability.py:70`; `P/entrypoints/http_server.py:296-297` |
| seed | `--random-seed N` (default: random) | `P/arg_groups/fields/device.py:47`; `P/arg_groups/serving_hook.py:603-607` |
| decode CUDA-graph max batch | `--cuda-graph-max-bs-decode N` (no bare `--cuda-graph-max-bs`) | `exec_.py:500` |
| schedule policy (opt-in LPM) | `--schedule-policy lpm` (default `fcfs`) | `schedule.py:82-98` |
| cached tokens in `usage` | `--enable-cache-report` | `P/arg_groups/fields/serving.py:189` |
| usage in streams by default | `--stream-response-default-include-usage` | `serving.py:263` |

### Ours (for completeness)

| purpose | knob | citation |
|---|---|---|
| disable prefix cache | `SERVING_PREFIX_CACHE=0` | `serving/server/app.py:1157` |
| KV pool cap | `SERVING_KV_BLOCKS=N` (unit: 16-token blocks) | `app.py:1155`, `:1206` |
| preemption mode | `SERVING_PREEMPTION=recompute\|swap` | `app.py:1156` |
| max running, prefill cap, block size | `build_default_app` arguments (defaults 32 / 512 / 16) | `app.py:1105-1116` |

### Mapping the SPEC arms to flags

| arm | flags (on top of `--dtype float16` and a fixed seed) |
|---|---|
| `vllm` | none |
| `vllm-eager` | `--enforce-eager` (removes compile **and** graphs; see SOURCE_NOTES §2.5) |
| `vllm-noprefix` | `--no-enable-prefix-caching` |
| `vllm-matched` | `--max-num-seqs 32 --num-gpu-blocks-override <ours' block count>` |
| `sglang` | `--enable-metrics` |
| `sglang-noradix` | `--enable-metrics --disable-radix-cache` |
| `sglang-nooverlap` | `--enable-metrics --disable-overlap-schedule` |
| `sglang-eager` | `--enable-metrics --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled` |

`--enable-metrics` is needed on every SGLang arm, because without it no
counters exist. It is an observability switch, not a performance knob.

**Equal KV pool for W4.** With block size 16 on both, `--num-gpu-blocks-override N`
(vLLM) and `SERVING_KV_BLOCKS=N` (ours) give `16 × N` tokens. The SGLang
equivalent is `--max-total-tokens 16N`. Because SGLang's value can only lower
the profiled pool, check the startup log for the actual size.

## 2. Prometheus metric names

**A note on the `_total` suffix.** All three engines use `prometheus_client`.
That library strips a trailing `_total` from a counter's registered name and
always exposes the counter as `<name>_total`. So vLLM's counter registered as
`vllm:num_preemptions` is scraped as `vllm:num_preemptions_total`, and SGLang's
counters, which are registered with `_total` already, are scraped under exactly
their registered names. Every counter also exposes a `<name>_created` sample,
which the harness should ignore. vLLM metrics carry `model_name` and `engine`
labels (`vllm/v1/metrics/loggers.py:480`).

The "scrape as" column is the name to grep for in `/metrics` output.

| purpose | vLLM 0.31.0 (scrape as) | SGLang 0.5.21 (scrape as) | ours (scrape as) |
|---|---|---|---|
| preemptions | `vllm:num_preemptions_total` (counter) — `loggers.py:677` | `sglang:num_retracted_requests_total` (counter) — `P/observability/metrics_collector.py:474-479` | `llm_preemptions_total` — `serving/metrics/prometheus.py:601` |
| prefix cache queries | `vllm:prefix_cache_queries_total` (counter, tokens) — `loggers.py:600` | `sglang:prefill_effective_tokens_total` summed over all `mode` labels — `metrics_collector.py:913-921` | `llm_cache_block_hits_total + llm_cache_block_misses_total` (blocks) — `prometheus.py:611-616` |
| prefix cache hits | `vllm:prefix_cache_hits_total` (counter, tokens) — `loggers.py:611` | same counter, `mode` in {`device_hit`, `host_hit`, `storage_hit`} | `llm_cache_block_hits_total` (blocks) — `prometheus.py:611` |
| hit rate (gauge) | not exported; compute hits / queries | `sglang:cache_hit_rate` — **last prefill batch only**, do not use as a run total — `metrics_collector.py:305-310` | not exported; compute from counters |
| running requests | `vllm:num_requests_running` (gauge) — `loggers.py:509` | `sglang:num_running_reqs` (gauge) — `metrics_collector.py:281-286` | `llm_requests_running` — `prometheus.py:641` |
| waiting requests | `vllm:num_requests_waiting` (gauge) — `loggers.py:519` | `sglang:num_queue_reqs` (gauge) — `metrics_collector.py:287-292` | `llm_requests_waiting` — `prometheus.py:644` |
| KV usage (0–1) | `vllm:kv_cache_usage_perc` (gauge) — `loggers.py:577` | `sglang:token_usage` (gauge) — `metrics_collector.py:321-326` | `llm_block_utilization` — `prometheus.py:659` |
| eviction counter | **none**. Opt-in histograms with `--kv-cache-metrics`: `vllm:kv_block_lifetime_seconds`, `vllm:kv_block_idle_before_evict_seconds` — `loggers.py:931`, `:944` | `sglang:evicted_tokens_total` (counter, tokens) — `metrics_collector.py:2213-2220` | **none** in Prometheus; `cache_evictions` in the scheduler JSON snapshot — `serving/scheduler/scheduler.py:1267-1274` |

Caveats for the `server_counters` block of the result artifact:

- **vLLM's prefix-cache counters skip preempted requests' re-lookups.** Those go
  into fields that are never exported (`vllm/v1/metrics/stats.py:132-143`;
  `loggers.py:1052-1056`). Under W4 the exported hit rate understates reuse.
- **Units differ.** vLLM and SGLang count prefix hits in tokens; ours counts
  blocks. A block-based rate and a token-based rate agree only up to the
  partial last block.
- **SGLang's counters exist only with `--enable-metrics`**, including the
  eviction counter (`P/managers/scheduler.py:572`;
  `P/mem_cache/base_prefix_cache.py:380-392`).
- **vLLM evictions** should be recorded as `null` (SPEC: never 0 for "not
  exposed").

## 3. OpenAI chat request behaviour

### `ignore_eos` in the chat request body

| engine | honoured? | citation |
|---|---|---|
| vLLM | **Yes.** Declared on the chat request (default False) and passed into sampling params. | `vllm/entrypoints/openai/chat_completion/protocol.py:274`; `:647`, `:713` |
| SGLang | **Yes.** Declared on `ChatCompletionRequest` (default False) and passed into sampling params. | `P/entrypoints/openai/protocol.py:932`; `:1147` |
| ours | **Yes.** | `serving/server/app.py:200-208`, `:735` |

One SGLang side effect: with the radix cache disabled, `ignore_eos` requests use
a different admission path that reserves the full `max_tokens` up front
(`P/managers/schedule_policy.py:1353-1354`, `:1200-1345`). This affects
`sglang-noradix` on W4.

### Prompt-token usage while streaming

Send `"stream_options": {"include_usage": true}` with `"stream": true`.

| engine | behaviour | citation |
|---|---|---|
| vLLM | A final chunk with `choices: []` carries `usage` with `prompt_tokens`, `completion_tokens`, `total_tokens`. `continuous_usage_stats: true` puts usage on every chunk. `stream_options` without `stream` is rejected. | fields at `vllm/entrypoints/openai/generate/base/protocol.py:266-268`; final chunk `vllm/entrypoints/openai/chat_completion/serving.py:839-891`; rejection `chat_completion/protocol.py:769-775` |
| SGLang | Same shape: a final chunk with `choices: []` and `usage`; `continuous_usage_stats` attaches usage to every chunk. | fields `P/entrypoints/openai/protocol.py:218-220`, `:878`; final chunk `P/entrypoints/openai/serving_chat.py:2197-2228`; per-chunk `:911-988` |
| ours | **Not supported.** `stream_options` is not a field and is silently ignored; usage appears only in non-streaming responses. The harness must count prompt tokens with the tokenizer. | `serving/server/app.py:184-210`; `:942-946` |

## 4. Not verified

- Which CUDA build the plain `pip install vllm==0.31.0` wheel targets
  (source defaults to 13.0, docs say 12.9; see SOURCE_NOTES F-V1).
  **UNVERIFIED.**
- Exact exposition of label sets on a live server. Names were read from source;
  a one-shot `curl /metrics` on the study node should confirm them before the
  harness relies on them. **UNVERIFIED** until that scrape.
