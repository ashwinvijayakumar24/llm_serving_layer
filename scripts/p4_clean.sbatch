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
#SBATCH --time=05:00:00
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
#
# THIS ARITHMETIC IS WRONG ABOVE 512 TOKENS AND JOB 11617299 PROVED IT. The
# 1024- and 2048-token cells each logged 7,525-10,788 evictions against this
# 40,000-block pool. Three things the estimate above omits: the window is 90 s
# (20 warmup + 60 steady + 10 drain), not 45, so ~180 requests arrive, not 90;
# prompt length is LOGNORMAL, so a mean of 2048 has a long right tail; and the
# cache holds a reference to every block it caches for the whole cell, so
# nothing is returned as requests retire. ~180 x 128 = 23,040 is the floor, not
# the ceiling, and the tail carries it past 40,000.
#
# Consequence: the 150- and 512-token cells are clean and valid; every 1024 and
# 2048 cell is INVALID and the project claims nothing at those lengths. Fixing
# it needs a bounded max_cached_blocks — which measures a DIFFERENT cache — or a
# shorter cell. Neither has been run. See results/p4_clean/RESULTS.md §3.
POOL=40000

# ONE FRESH SERVER PAIR PER CELL — per (length, structure, sharing rate).
#
# Fresh-per-LENGTH was not enough, and the arithmetic says why: nine cells shared
# one server and each issues ~180 requests once warmup and drain are counted.
#   512 tok:  9 x 180 x  32 =  51,840 blocks against a 40,000 pool
#  1024 tok:  9 x 180 x  64 = 103,680
#  2048 tok:  9 x 180 x 128 = 207,360
# The 512-token cells duly logged 4,759 and 8,968 evictions. Per CELL the worst
# case is 180 x 128 = 23,040 blocks, comfortably inside 40,000, so the cache can
# hold everything it sees and never evict.
#
# 36 restarts at ~25 s is ~15 minutes of startup. That is the price of cells that
# do not contaminate each other, and it is cheap next to a wrong conclusion —
# the previous run's +416 ms was exactly this artefact.
for PLEN in 150 512 1024 2048; do
  for STRUCT in zero system conversational; do
    for SHARE in 0.0 0.5 1.0; do
      echo
      echo "########## len=${PLEN} struct=${STRUCT} share=${SHARE} (FRESH PAIR) ##########"
      start_one 9401 "on_${PLEN}_${STRUCT}_${SHARE}"  SERVING_KV_BLOCKS=$POOL
      start_one 9402 "off_${PLEN}_${STRUCT}_${SHARE}" SERVING_KV_BLOCKS=$POOL SERVING_PREFIX_CACHE=0

      set +e
      python3 -m bench.run_p4 \
          --url "$(U 9401)" --cache-off-url "$(U 9402)" \
          --structures "$STRUCT" --sharing-rates "$SHARE" \
          --prompt-mean "$PLEN" --output-mean 32 \
          --rate 2 --duration 60 --warmup 20 --drain 10 \
          --slo-ttft-ms $SLO_TTFT --slo-itl-ms $SLO_ITL --seed 20260801 \
          --label "clean_${PLEN}_${STRUCT}_${SHARE}" --results-dir results/p4_clean 2>&1
      set -e

      # ANY eviction here means the pool saturated inside a single cell, which is
      # the failure this whole restructuring exists to prevent. Reported per cell
      # rather than inferred later from a suspicious total.
      curl -s "http://127.0.0.1:9401/metrics" 2>/dev/null | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('  eviction audit: metrics unavailable'); raise SystemExit
def find(o,k):
    if isinstance(o,dict):
        for a,b in o.items():
            if a==k: return b
            r=find(b,k)
            if r is not None: return r
    return None
ev=find(d,'evictions')
# ABSENT IS NOT CLEAN. The first version of this audit read
#   '!! CONTAMINATED' if ev not in (None,0) else 'clean'
# and printed 'eviction audit: None  clean' for all 36 cells of job 11617299 —
# the key was never in /metrics at all, so a MISSING measurement was reported as
# a PASSING one. The driver's own summary table showed 7,525-10,788 evictions
# inside single 1024/2048-token cells at the same time. An audit that cannot
# distinguish 'zero' from 'did not look' is worse than no audit, because it
# converts an unknown into a green tick. Three outcomes, never two.
if ev is None:   print('  eviction audit: UNKNOWN — no evictions key in /metrics; audit did NOT run')
elif ev == 0:    print('  eviction audit: 0  clean')
else:            print(f'  eviction audit: {ev}  !! CONTAMINATED')
"
      stop_all
    done
  done
done

echo; echo "=== artifacts: $(ls results/p4_clean 2>/dev/null | wc -l) ==="
echo "=== done ==="
