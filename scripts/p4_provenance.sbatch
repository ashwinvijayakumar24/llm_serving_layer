#!/bin/bash
# PHASE 4 — RE-RUN THE FOUR EARNED CELLS FROM A COMMITTED TREE.
#
# WHY THIS EXISTS
# ---------------
# Job 11617299 produced the numbers the S4 claim rests on: -37.7 ms TTFT p99 at
# 512-token prompts with a shared system preamble, -17.3 ms p50 on multi-turn
# traffic. Every one of its artifacts is stamped:
#
#     NOT PUBLISHABLE: Working tree is dirty — the measured code is not any commit.
#
# The measurement is sound; its reproducibility stamp is not. A number that
# cannot be tied to a commit cannot be re-derived by anyone, including me.
#
# THE ROOT CAUSE, WHICH WAS NOT STRAY ARTIFACTS
# ----------------------------------------------
# Provenance.capture (serving/metrics/artifact.py:266) runs
# `git status --porcelain`, which counts UNTRACKED files, and `logs/` was never
# in .gitignore. Every sbatch in this repo writes logs/srv_*.log before the
# driver captures provenance, so every run dirtied its own tree by starting.
# The code was always a commit. The harness was reporting on itself.
#
# Two fixes, both applied:
#   1. logs/ is gitignored (this commit).
#   2. This job writes artifacts OUTSIDE the repo and copies them in at the end,
#      so a partially-written results/ directory cannot dirty the tree mid-run
#      either. Belt and braces, because the failure mode is silent.
#
# WHAT IT RE-RUNS
# ---------------
# Only the four cells that carry the claim, at identical settings and seed to
# 11617299. This is a provenance fix, not a new measurement: if the numbers move
# materially, that is itself a finding and the claim needs re-examining, not
# re-labelling.
#
#SBATCH --job-name=p4_prov
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/p4_prov_%j.out
#SBATCH --error=logs/p4_prov_%j.out

set -e
mkdir -p logs
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load cuda/12.9.1; module load anaconda3; conda activate llm
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export LLM_WEIGHTS_PATH="${LLM_WEIGHTS_PATH:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/vendor/llm_inference_engine:${PYTHONPATH}"
SLO_TTFT=434
SLO_ITL=27.8
POOL=40000

# THE WHOLE POINT OF THE JOB. If the tree is dirty here, the run reproduces the
# exact defect it exists to fix, so it refuses to start and says what is dirty.
echo "=== provenance preflight ==="
echo "  HEAD: $(git rev-parse HEAD)"
DIRT="$(git status --porcelain)"
if [ -n "$DIRT" ]; then
    echo "FATAL: working tree is dirty — this job would reproduce the defect it fixes."
    echo "$DIRT" | head -20
    exit 1
fi
echo "  tree: CLEAN"

# Artifacts land outside the repo so that writing them cannot dirty the tree
# while later cells are still measuring. Copied back in after the last cell.
OUT="${SLURM_SUBMIT_DIR}/../p4_prov_out_${SLURM_JOB_ID}"
mkdir -p "$OUT"
echo "  artifacts -> $OUT (copied into results/p4_prov/ at the end)"

PIDS=()
start_one () {
    local port=$1 label=$2; shift 2
    env "$@" python3 -m uvicorn serving.server.app:build_default_app --factory \
        --host 127.0.0.1 --port "$port" --log-level warning \
        > "logs/srv_${label}_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=($!)
    for i in $(seq 1 180); do
        curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && { echo "  up: $label (${i}s)"; return 0; }
        sleep 1
    done
    echo "FATAL: $label unhealthy"; tail -25 "logs/srv_${label}_${SLURM_JOB_ID}.log"; return 1
}
stop_all () { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; PIDS=(); sleep 5; }
trap stop_all EXIT
U () { echo "http://127.0.0.1:$1/v1/chat/completions"; }

# The four cells the S4 claim rests on. Fresh server pair per cell, because
# sharing a server across cells is what invalidated the first two attempts at
# this measurement (results/p4/RESULTS.md §4).
run_cell () {   # $1=prompt_len  $2=structure  $3=sharing_rate
    local PLEN=$1 STRUCT=$2 SHARE=$3
    echo
    echo "########## len=${PLEN} struct=${STRUCT} share=${SHARE} ##########"
    start_one 9401 "on_${PLEN}_${STRUCT}_${SHARE}"  SERVING_KV_BLOCKS=$POOL
    start_one 9402 "off_${PLEN}_${STRUCT}_${SHARE}" SERVING_KV_BLOCKS=$POOL SERVING_PREFIX_CACHE=0

    set +e
    python3 -m bench.run_p4 \
        --url "$(U 9401)" --cache-off-url "$(U 9402)" \
        --structures "$STRUCT" --sharing-rates "$SHARE" \
        --prompt-mean "$PLEN" --output-mean 32 \
        --rate 2 --duration 60 --warmup 20 --drain 10 \
        --slo-ttft-ms $SLO_TTFT --slo-itl-ms $SLO_ITL --seed 20260801 \
        --label "prov_${PLEN}_${STRUCT}_${SHARE}" --results-dir "$OUT" 2>&1
    set -e

    # Absent is not clean. See scripts/p4_clean.sbatch for why this has three
    # outcomes instead of two.
    curl -s "http://127.0.0.1:9401/metrics" 2>/dev/null | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('  eviction audit: UNKNOWN — metrics unavailable'); raise SystemExit
def find(o,k):
    if isinstance(o,dict):
        for a,b in o.items():
            if a==k: return b
            r=find(b,k)
            if r is not None: return r
    return None
ev=find(d,'evictions')
if ev is None:   print('  eviction audit: UNKNOWN — no evictions key; audit did NOT run')
elif ev == 0:    print('  eviction audit: 0  clean')
else:            print(f'  eviction audit: {ev}  !! CONTAMINATED')
"
    stop_all
}

run_cell 512 system         0.5
run_cell 512 system         1.0
run_cell 150 conversational 0.5
run_cell 150 conversational 1.0

echo
echo "=== provenance postflight ==="
echo "  HEAD still: $(git rev-parse HEAD)"
echo "  tree during run: $([ -z "$(git status --porcelain)" ] && echo CLEAN || echo DIRTY)"

echo
echo "=== publishable check ==="
python3 - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
files = sorted(out.glob("*_summary.json"))
if not files:
    print("  NO ARTIFACTS WRITTEN"); raise SystemExit(1)
for f in files:
    d = json.loads(f.read_text())
    prov = d.get("provenance", d)
    dirty = prov.get("repo_dirty")
    sha = (prov.get("repo_sha") or prov.get("git_sha") or "?")[:8]
    print(f"  {f.name:52s} sha={sha} dirty={dirty}")
PY

mkdir -p results/p4_prov
cp -f "$OUT"/* results/p4_prov/ 2>/dev/null || true
echo
echo "=== artifacts: $(ls results/p4_prov 2>/dev/null | wc -l) in results/p4_prov ==="
echo "=== done ==="
