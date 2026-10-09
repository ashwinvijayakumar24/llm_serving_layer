# Runbook — running the cross-engine study on PACE

Everything up to this point was built and tested locally on CPU against mock
servers. This file is the sequence of steps that needs a GPU. Each step says
what it produces and what to check before moving on.

## 0. Sync the repo on PACE

```bash
cd ~/ps-simpliearn-0/llm_serving_layer     # or wherever the PACE checkout lives
git pull
git submodule update --init                # vendor/llm_inference_engine
```

The working tree must be clean when the benchmark runs: artifacts record
`repo_dirty`, and the renderer flags dirty runs.

## 1. Create the vLLM and SGLang environments (once)

```bash
module load anaconda3
bash scripts/xengine/setup_envs.sh          # vLLM 0.31.0 and SGLang 0.5.21, separate envs
```

Produces `results/xengine/_env/engine_versions.txt`, one `pip freeze` per
engine, and the full pip install logs.

**Check:**
- Both `import vllm` and `import sglang` lines report the pinned version.
- Read `pip_install_*.log` for warnings and errors. Anything confusing goes into
  `bench/oss-pr-candidates.md` now, while the output is still on screen.

**Known risk:** vLLM 0.31.0 defaults to CUDA 13.0 in its build config
(`docs/xengine/SOURCE_NOTES.md` §1). If the PACE GPU driver is too old for
CUDA 13 wheels, vLLM fails at import or first kernel launch. The fix is a
CUDA 12.9 wheel variant; whichever way it goes, log it as an OSS candidate (the
vLLM docs say 12.9).

## 2. Pilot job (about 1–2 hours)

```bash
scripts/xengine/submit.sh pilot
```

Runs only the three default engines (`ours`, `vllm`, `sglang`) on W1 at
concurrency 1 and 16, with 1 repetition, into `results/xengine_pilot/`. That
directory is never rendered into `BENCHMARKS.md`.

**What the pilot is for:** proving each engine launches, is detected correctly,
and produces valid artifacts — not producing numbers.

**Check, in `logs/xengine_<job>.out` and the pilot artifacts:**
1. `ours` artifacts have `validity.attention_backend == "flashinfer"`. If it
   says `paged_torch(...)`, FlashInfer failed to load (FINDINGS_OURS F-002);
   fix the environment before anything else.
2. vLLM and SGLang artifacts have non-null `engine.version` and an
   `engine.flags.resolved` block (e.g. `kv_cache_tokens`, `max_running_requests`).
3. `server_counters.raw` contains the metric names from `ENGINE_FLAGS.md`. Run
   `curl -s localhost:<port>/metrics | grep -E "preempt|prefix|retract|evict"`
   on a live server once to confirm the label sets.
4. Note the wall-clock time per cell (start/end timestamps in the log). Set each
   workload's `--time` in `scripts/xengine/submit.sh` to about 1.5× the pilot-based
   projection.
5. Capture the startup DeprecationWarnings for OSS-001/002 and move their status
   to "repro confirmed" if they appear.

## 3. Full matrix (four jobs, one per workload)

```bash
scripts/xengine/submit.sh all
```

- W1 and W2 jobs also run the W5 int8 appendix (`ours-int8`) after the `ours`
  arm, reusing its KV pool size.
- Each job renders `docs/xengine/BENCHMARKS.md` at the end from everything in
  `results/xengine/`. Missing cells stay `TODO`.
- 3 repetitions per cell. A re-run appends rep 4, 5, … rather than overwriting.

## 4. After the runs

1. `git add results/xengine docs/xengine` and commit the artifacts with the job
   ids in the message.
2. Read every flagged cell in `BENCHMARKS.md` (bold brackets). Investigate
   flags before writing any observation.
3. Fill each **Observation** in the Analysis section from the tables, citing
   table and cell. Keep **UNVERIFIED** until you have checked the mechanism
   against both the data and `SOURCE_NOTES.md`.
4. Add measured headlines to `docs/xengine/PROGRESS.md` → "Measured results",
   each with artifact path, job id and GPU.

## Things that need your hands

| step | why it can't be done locally |
|---|---|
| 1 | needs PACE network + conda |
| 2–3 | needs a GPU allocation |
| 4.3 | the analysis sign-off is yours (UNVERIFIED → verified) |
