# Cross-engine benchmark study — shared spec

Study: this serving layer vs vLLM vs SGLang on the same GPU, same model, same
request streams. A **measurement study**, not an optimization effort. Losing with
a correct, mechanistic explanation is a successful outcome. Governing decision:
ADR-025 (supersedes ADR-013).

This file is the contract between the harness (`bench/xengine/`), the renderer
(`bench/xengine/render.py`), and `docs/xengine/BENCHMARKS.md`. Change it first,
then the code.

## Hard rules

1. **No fabricated numbers.** Every number in `BENCHMARKS.md` is rendered from a
   JSON artifact under `results/xengine/`. A cell with no artifact says `TODO`.
   No estimates, no extrapolation, no hand-typed numbers.
2. **Anomalies are flagged, not smoothed.** Bimodal latency, warmup effects,
   thermal/clock variance, run-to-run spread above a threshold — the renderer
   prints a flag next to the cell.
3. **≥3 repetitions per cell.** Report mean, min, max and stdev across reps.
4. **`serving/` and `vendor/` are not modified** by this study. Harness adapters
   live in `bench/xengine/`. Bugs found in our own server are logged in
   `docs/xengine/FINDINGS_OURS.md`, not fixed in this study.
5. **All engines run on the same node in the same Slurm allocation** (ADR-021;
   PACE nodes drift ~25% for identical code).
6. **Diagnostic arms are not baselines.** vLLM/SGLang runs with features turned
   off (`--enforce-eager`, prefix caching off, …) exist only to attribute those
   engines' own performance. They are never presented as "the competitor".

## Fixed parameters

| parameter | value |
|---|---|
| model | Llama-3.2-1B-Instruct (the checkpoint at `LLM_WEIGHTS_PATH`; same files for all engines) |
| dtype | fp16 everywhere (`--dtype float16` for vLLM/SGLang) |
| sampling | greedy (`temperature=0`), `ignore_eos=true`, exact `max_tokens` |
| API | OpenAI `POST /v1/chat/completions`, `stream=true`, same chat template (HF tokenizer of the checkpoint) |
| SLO | **one frozen absolute SLO for all engines**, taken from our engine's calibration (`results/p2/RESULTS.md`: TTFT < 434 ms, per-request TPOT < 27.8 ms) unless re-calibrated on the study GPU — then frozen in `bench/xengine/configs/slo.yaml` with the calibration artifact path. Never re-calibrated per engine. |
| goodput | `bench/loadgen.py:meets_slo` semantics: COMPLETED ∧ TTFT < slo_ttft ∧ own TPOT < slo_itl, counted in the steady window |

## Engines and arms

| arm id | engine | notes |
|---|---|---|
| `ours` | this serving layer | FlashInfer backend required; the active attention backend MUST be recorded and a PagedTorch fallback marks the run invalid |
| `ours-noprefix` | ours | `SERVING_PREFIX_CACHE=0` |
| `ours-int8` | ours | W5 appendix only: int8 weight-only quantization |
| `vllm` | vLLM default | fp16, pinned version |
| `vllm-eager` | vLLM | `--enforce-eager` — disables CUDA graphs **and torch.compile** |
| `vllm-nograph` | vLLM | `-cc '{"cudagraph_mode":"NONE"}'` — CUDA graphs only; `vllm-eager` minus this isolates torch.compile |
| `vllm-noasync` | vLLM | `--no-async-scheduling` — vLLM's CPU/GPU overlap (on by default), pairs with `sglang-nooverlap` |
| `vllm-noprefix` | vLLM | `--no-enable-prefix-caching` |
| `vllm-matched` | vLLM | `--max-num-seqs 32 --num-gpu-blocks-override N --block-size 16` matched to ours |
| `sglang` | SGLang default | fp16, pinned version (default schedule policy is FCFS) |
| `sglang-noradix` | SGLang | `--disable-radix-cache` |
| `sglang-nooverlap` | SGLang | `--disable-overlap-schedule` |
| `sglang-eager` | SGLang | `--disable-decode-cuda-graph --disable-prefill-cuda-graph` |
| `sglang-lpm` | SGLang | `--schedule-policy lpm` — turns ON longest-prefix-match ordering (not default); W3 only |

Versions (2026-10-08, latest stable): vLLM 0.31.0 (`db9527a4`), SGLang 0.5.21
(`e00930c5`). Exact flags and metric names: `docs/xengine/ENGINE_FLAGS.md`.

### Arm × workload

Baseline arms (`ours`, `vllm`, `sglang`) run on every workload. Each diagnostic
arm runs only where its mechanism is exercised: W1 — `vllm-eager`,
`vllm-nograph`, `vllm-noasync`, `vllm-matched`, `sglang-eager`,
`sglang-nooverlap`; W3 — `ours-noprefix`, `vllm-noprefix`, `sglang-noradix`,
`sglang-lpm`; W4 — `ours-noprefix`, `vllm-matched`. Source of truth: the `ARMS_W*` lists in the
`Makefile`.

## Workloads (`bench/xengine/configs/workloads/*.yaml`)

| id | shape | loop |
|---|---|---|
| W1 | concurrency sweep 1,2,4,8,16,32,64; prompt 512 / output 128 fixed | closed loop (fixed in-flight count) |
| W2 | lognormal prompt lengths, long-tail outputs; offered-rate sweep | open loop (existing loadgen) |
| W3 | 80% of requests share a long common system prefix | open loop; prefix-cache on vs off for every engine |
| W4 | v2: 4 / 16 / 32 long sequences in flight (≈2.5k tokens each) with an **equal 32,768-token KV pool** on every engine; 4 fits, 16 and 32 force preemption | closed loop (v1 open loop, job 13931404, superseded: unequal pressure across engines) |
| W5 | appendix: our engine int8 vs fp16 on W1/W2 | — |

All workloads seeded; same seed → byte-identical request stream for every engine.

## Result artifact (one JSON per engine × workload × point × rep)

Path: `results/xengine/<workload>/<arm>/<point>_rep<k>.json`

```json
{
  "schema": "xengine-run/1",
  "arm": "vllm-eager",
  "engine": {"name": "vllm", "version": "x.y.z", "launch_cmd": ["..."], "flags": {}},
  "workload": {"id": "W1", "config_path": "...", "config_sha256": "...", "seed": 0, "point": {"concurrency": 16}},
  "rep": 1,
  "hardware": {"gpu_name": "...", "gpu_count": 1, "driver": "...", "cuda": "...", "node": "...", "slurm_job_id": "..."},
  "provenance": {"repo_sha": "...", "repo_dirty": false, "started_at": "...", "harness_version": "..."},
  "slo": {"ttft_ms": 0, "tpot_ms": 0, "source": "..."},
  "validity": {"valid": true, "reasons": [], "attention_backend": "flashinfer"},
  "metrics": {
    "ttft_ms": {"p50": 0, "p95": 0, "p99": 0, "n": 0},
    "itl_ms": {"p50": 0, "p95": 0, "p99": 0, "n": 0},
    "tpot_ms": {"p50": 0, "p95": 0, "p99": 0, "n": 0},
    "e2e_ms": {"p50": 0, "p95": 0, "p99": 0, "n": 0},
    "output_tok_s": 0, "request_rps": 0, "goodput_rps": 0, "slo_attainment": 0,
    "completed": 0, "failed": 0
  },
  "server_counters": {"preemptions": null, "evictions": null, "prefix_hit_rate": null, "raw": {}},
  "anomalies": [{"kind": "bimodal_ttft", "detail": "..."}],
  "samples": {"ttft_ms": [], "itl_ms": [], "tpot_ms": []}
}
```

`server_counters` values are `null` when an engine does not expose them — never 0.

## Deliverables

- `make bench-all` runs the full matrix; `make bench ENGINE=<arm> WORKLOAD=<id>` one cell.
- `bench/xengine/render.py` renders tables + 3–4 matplotlib charts into `docs/xengine/BENCHMARKS.md`.
- `docs/xengine/BENCHMARKS.md`: tables, charts, one analysis subsection per finding with
  vLLM/SGLang source citations (`docs/xengine/SOURCE_NOTES.md`); unverified hypotheses
  marked **UNVERIFIED — Ashwin to confirm**.
- `bench/oss-pr-candidates.md`: ≥3 friction entries from setting up vLLM/SGLang.
