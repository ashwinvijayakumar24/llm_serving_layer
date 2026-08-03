#!/bin/bash
# PHASE 5 — ROUTING, SWEPT AROUND EACH SCENARIO'S OWN KNEE.
#
# WHY THIS EXISTS
# ---------------
# Job 11610306 measured four scenarios at loads 4,8,16,32,48 and got a usable
# comparison out of exactly one of them. The other three -- system_prompt_sharing,
# uniform_prefix, hot_prefix_skew -- returned goodput 0.00 at EVERY load, with
# TTFT p50 between 10 and 65 seconds. Every cell was far above the knee, so no
# matched policy pair was valid at any load and the driver correctly refused to
# interpolate a crossover from zero usable points.
#
# The cause was mine and it was not subtle: the load ladder was derived from
# zero_sharing's capacity and reused unchanged for scenarios whose requests cost
# several times more prefill. A ladder that starts at 4 req/s cannot measure a
# fleet that saturates below 1.
#
# THE FIX: each scenario gets a ladder sized for its own cost.
#   heavy (long shared prefix):  0.5, 1, 2, 4, 8
#   zero_sharing (cheap):        4, 8, 16, 32
# Both ladders sweep PAST the expected knee, because §10.3 -- the case where
# affinity and load balance conflict -- only exists above it. A sweep that stops
# at the knee leaves the most interesting prediction untested.
#
# WHAT WOULD MAKE THIS FAIL AGAIN, STATED IN ADVANCE
# ---------------------------------------------------
# If the heavy scenarios still collapse at 0.5 req/s, the fleet is prefill-bound
# to a degree that no routing policy can address, and the honest conclusion is
# that this workload cannot be measured on four H200s rather than that routing
# does not help. That would be a third negative result, and it gets published
# the same way the first two did.
#
# PREDICTIONS, RECORDED BEFORE THE RUN (methodology §10):
#   hot_prefix_skew        prefix_aware SHOULD WIN. A skewed hot prefix is the
#                          case where cache locality is worth more than perfect
#                          load balance. If it loses here it loses everywhere.
#   system_prompt_sharing  prefix_aware should win modestly. The shared preamble
#                          is short, so there is less locality to exploit.
#   uniform_prefix         B5 WINS or ties. Every replica caches the same prefix
#                          after warmup, so prefix awareness degenerates to
#                          "pick any replica" and any deviation from load-optimal
#                          placement is pure loss.
#   zero_sharing           B5 WINS. Already confirmed in 11610306; re-run here as
#                          a control on the new ladder.
#
#SBATCH --job-name=p5_knee
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --output=logs/p5_knee_%j.out
#SBATCH --error=logs/p5_knee_%j.out

set -e
mkdir -p logs
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  qos: ${SLURM_JOB_QOS} ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

module load cuda/12.9.1; module load anaconda3; conda activate llm
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export LLM_WEIGHTS_PATH="${LLM_WEIGHTS_PATH:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/vendor/llm_inference_engine:${PYTHONPATH}"

# Inherited from the Phase 2 calibration (job 11602207), not re-derived: routing
# changes placement, not decode speed, so re-anchoring would move the goalposts
# between this run and every earlier comparison.
SLO_TTFT=434
SLO_ITL=27.8

echo "=== provenance preflight ==="
echo "  HEAD: $(git rev-parse HEAD)"
DIRT="$(git status --porcelain)"
if [ -n "$DIRT" ]; then
    echo "WARNING: working tree dirty — artifacts will be stamped NOT PUBLISHABLE:"
    echo "$DIRT" | head -10
else
    echo "  tree: CLEAN"
fi

OUT="${SLURM_SUBMIT_DIR}/../p5_knee_out_${SLURM_JOB_ID}"
mkdir -p "$OUT"

PIDS=()
start_one () {
    local port=$1 label=$2; shift 2
    env "$@" python3 -m uvicorn serving.server.app:build_default_app --factory \
        --host 127.0.0.1 --port "$port" --log-level warning \
        > "logs/srv_${label}_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=($!)
    for i in $(seq 1 180); do
        curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && { echo "  up: $label :$port (${i}s)"; return 0; }
        sleep 1
    done
    echo "FATAL: $label did not become healthy"; tail -30 "logs/srv_${label}_${SLURM_JOB_ID}.log"; return 1
}
start_router () {   # $1=port $2=policy
    ROUTER_POLICY=$2 ROUTER_REPLICAS="$REPLICAS" \
    python3 -m uvicorn scripts._router_entry:app --factory \
        --host 127.0.0.1 --port "$1" --log-level warning \
        > "logs/router_${2}_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=($!)
    for i in $(seq 1 60); do
        curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { echo "  router up: $2 :$1 (${i}s)"; return 0; }
        sleep 1
    done
    echo "FATAL: router $2 did not become healthy"; tail -20 "logs/router_${2}_${SLURM_JOB_ID}.log"; return 1
}
stop_all () { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; PIDS=(); sleep 5; }
trap stop_all EXIT

# ONE REPLICA PER PHYSICAL GPU. Four replicas sharing one H200 contend for SMs;
# that setup produced 45 of 48 invalid cells in job 11609657 and was a property
# of the allocation rather than of routing.
for i in 1 2 3 4; do
    start_one $((9500+i)) "p5_replica$i" CUDA_VISIBLE_DEVICES=$((i-1))
done
REPLICAS="http://127.0.0.1:9501,http://127.0.0.1:9502,http://127.0.0.1:9503,http://127.0.0.1:9504"

# One router process per policy, all pointing at the SAME four replicas.
# least_outstanding is the baseline that isolates cache awareness: it is
# load-aware and cache-blind, so the gap between it and prefix_aware IS the
# claim. round_robin is table stakes and proves only load balancing.
start_router 9601 prefix_aware
start_router 9602 least_outstanding
start_router 9603 round_robin
R_ARGS="--router prefix_aware=http://127.0.0.1:9601/v1/chat/completions \
        --router least_outstanding=http://127.0.0.1:9602/v1/chat/completions \
        --router round_robin=http://127.0.0.1:9603/v1/chat/completions"

# ---------------------------------------------------------------------------
# LADDER 1 — the three scenarios that produced nothing last time.
#
# duration 120 rather than 45: at 0.5 req/s a 45 s window offers 22 requests and
# a p99 computed from 22 samples is decoration. 120 s gives 60 at the bottom of
# the ladder and ~960 at the top. The goodput comparison, which is the actual
# claim, is sound across the whole ladder.
# ---------------------------------------------------------------------------
echo; echo "################ LADDER 1: heavy scenarios, loads 0.5-8 ################"
set +e
python3 -m bench.run_p5 \
    --replicas "$REPLICAS" $R_ARGS \
    --loads 0.5,1,2,4,8 \
    --scenarios system_prompt_sharing,uniform_prefix,hot_prefix_skew \
    --duration 120 --warmup 20 --drain 10 \
    --slo-ttft-ms $SLO_TTFT --slo-itl-ms $SLO_ITL --seed 20260802 \
    --label knee_heavy --results-dir "$OUT" 2>&1
echo "  ladder 1 exit: $?"
set -e

# ---------------------------------------------------------------------------
# LADDER 2 — zero_sharing control on the cheap ladder. Already confirmed as a
# §10.1 losing case in 11610306; re-run so the control and the treatment come
# from the same allocation (R12) rather than being compared across jobs.
# ---------------------------------------------------------------------------
echo; echo "################ LADDER 2: zero_sharing control, loads 4-32 ################"
set +e
python3 -m bench.run_p5 \
    --replicas "$REPLICAS" $R_ARGS \
    --loads 4,8,16,32 \
    --scenarios zero_sharing \
    --duration 120 --warmup 20 --drain 10 \
    --slo-ttft-ms $SLO_TTFT --slo-itl-ms $SLO_ITL --seed 20260802 \
    --label knee_zero --results-dir "$OUT" 2>&1
echo "  ladder 2 exit: $?"
set -e

stop_all

mkdir -p results/p5_knee
cp -f "$OUT"/* results/p5_knee/ 2>/dev/null || true
echo
echo "=== artifacts: $(ls results/p5_knee 2>/dev/null | wc -l) in results/p5_knee ==="
echo "=== done ==="
