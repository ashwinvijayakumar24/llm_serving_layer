#!/bin/bash
# PHASE 4 crossover, RUN CLEANLY — a fresh server pair per cell.
#
# WHY THIS EXISTS
# ---------------
# Job 11611626 measured +416 ms at 1024 tokens and +907 ms at 2048, which read as
# "prefix caching makes inference dramatically slower". It was an experimental
# artefact and the arithmetic in that run's own output said so: at 1024 tokens the
# 40,000-block pool holds 625 requests' worth of blocks while the workload offered
# about 90, so eviction could not legitimately fire — and it fired 11,673 times.
#
# Cause: all 36 cells ran against ONE long-lived server. A cache holds a reference
# to every block it caches, so cached blocks are never returned as requests
# retire; that is what a cache is. Across four prompt lengths the trie filled the
# pool, and from the 1024-token cells onward every request had to evict before it
# could allocate. Cells were not independent and the later ones were contaminated
# by the earlier ones.
#
# THE FIX: one fresh server pair per prompt length, torn down in between. Startup
# is ~20 s and buys independence between cells, which is the only thing that makes
# a cross-length comparison mean anything.
#
# A bounded max_cached_blocks would also stop the pool being consumed, but it
# would change what is being measured — a bounded cache is a different cache. A
# fresh server measures the thing itself.
#
#SBATCH --job-name=p4_clean
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/p4_clean_%j.out
#SBATCH --error=logs/p4_clean_%j.out

set -e
mkdir -p logs results/p4_clean
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load cuda/12.9.1; module load anaconda3; conda activate llm
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export LLM_WEIGHTS_PATH="${LLM_WEIGHTS_PATH:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/vendor/llm_inference_engine:${PYTHONPATH}"
SLO_TTFT=434
SLO_ITL=27.8

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

# Pool sized so it CANNOT saturate within a cell: at 2048 tokens a request needs
# 128 blocks and a 45 s window at 2 req/s offers ~90 requests, so ~11,520 blocks
# are live at most. 40,000 leaves the cache room to hold every block it ever sees
# within one cell without a single eviction. Any eviction in this run is
# therefore a signal, not background noise — and is reported.
POOL=40000

for PLEN in 150 512 1024 2048; do
    echo
    echo "################ prompt_mean=${PLEN}  (FRESH SERVERS) ################"
    start_one 9401 "on_${PLEN}"  SERVING_KV_BLOCKS=$POOL
    start_one 9402 "off_${PLEN}" SERVING_KV_BLOCKS=$POOL SERVING_PREFIX_CACHE=0

    set +e
    python3 -m bench.run_p4 \
        --url "$(U 9401)" --cache-off-url "$(U 9402)" \
        --structures zero,system,conversational \
        --sharing-rates 0.0,0.5,1.0 \
        --prompt-mean "$PLEN" --output-mean 32 \
        --rate 2 --duration 60 --warmup 20 --drain 10 \
        --slo-ttft-ms $SLO_TTFT --slo-itl-ms $SLO_ITL --seed 20260801 \
        --label "clean_len${PLEN}" --results-dir results/p4_clean 2>&1
    set -e

    # The check that would have caught the original mistake. Any eviction here
    # means the pool saturated inside a single cell, which invalidates the cell.
    echo "--- eviction audit for prompt_mean=${PLEN} ---"
    curl -s "http://127.0.0.1:9401/metrics" 2>/dev/null \
        | python3 -c "
import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('  (metrics unavailable)'); raise SystemExit
def find(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key: return v
            r = find(v, key)
            if r is not None: return r
    return None
ev = find(d, 'evictions')
print(f'  evictions this cell: {ev}')
if ev not in (None, 0):
    print('  !! NON-ZERO: the pool saturated inside this cell and the numbers are contaminated')
else:
    print('  clean: the cache never had to evict, so cells are independent')
"
    stop_all
done

echo; echo "=== artifacts: $(ls results/p4_clean 2>/dev/null | wc -l) ==="
echo "=== done ==="
