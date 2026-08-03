#!/bin/bash
# PHASE 5 — SWEEP THE AFFINITY/LOAD BLEND AND FIND THE CROSSOVER.
#
# WHAT JOB 11653158 ACTUALLY MEASURED, AND WHY THIS JOB EXISTS
# ------------------------------------------------------------
# That job found prefix_aware losing 23% of fleet goodput to least_outstanding
# on hot_prefix_skew at 8x the saturation knee (2.11 vs 2.74 req/s, n=946).
#
# The first write-up called that "pure affinity being punished for ignoring
# load". It was not. build_default_router defaults to blend=0.7 --
#
#     score(r) = blend * affinity(r) - (1 - blend) * min(1, inflight / 8)
#
# -- so the policy that lost was ALREADY 30% load-aware. The real statement is
# that 0.7 is too much affinity above the knee, and the useful blend lies
# somewhere below it. "Somewhere below it" is not a result. This job measures
# where.
#
# blend = 0.0 is asserted by test to be exactly least_outstanding, so the sweep
# carries its own baseline: the 0.0 arm IS B5, measured in the same allocation
# against the same replicas at the same instant as every other arm. No
# cross-job comparison is needed to compute an advantage, which is the whole
# reason to sweep rather than to compare against 11653158.
#
# WHY THIS IS CHEAP
# -----------------
# Routers are CPU processes holding advisory hints. Six of them cost six Python
# processes and no GPU. The four replicas -- the expensive part -- are shared by
# every arm, so a six-point blend sweep costs the same GPU time as a two-policy
# comparison and every arm sees an identical fleet.
#
# PREDICTIONS, RECORDED BEFORE THE RUN
# -------------------------------------
#   At 8x knee (above saturation): monotonic decline in goodput as blend rises.
#     The mechanism from 11653158 says every unit of affinity is a unit of
#     load-blindness, and above the knee load-blindness is what costs goodput.
#     If this is NOT monotonic, the mechanism I published is wrong.
#   At 0.5x knee (below saturation): flat, or a shallow optimum at moderate
#     blend. Below the knee there is no queue to misjudge, so affinity is
#     nearly free -- but it was still slightly negative (-0.017) at blend=0.7,
#     so "free" is not the same as "beneficial".
#   Crossover: I expect the best blend above the knee to be 0.0 or 0.25, i.e.
#     for cache-aware routing to have no positive operating point on this
#     workload. I have already been wrong once on this benchmark in exactly the
#     direction of over-crediting affinity (see results/p5_knee/RESULTS.md §3),
#     so this prediction is deliberately the pessimistic one.
#
#SBATCH --job-name=p5_blend
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:30:00
#SBATCH --output=logs/p5_blend_%j.out
#SBATCH --error=logs/p5_blend_%j.out

set -e
mkdir -p logs
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  qos: ${SLURM_JOB_QOS} ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

module load cuda/12.9.1; module load anaconda3; conda activate llm
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export LLM_WEIGHTS_PATH="${LLM_WEIGHTS_PATH:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/vendor/llm_inference_engine:${PYTHONPATH}"
SLO_TTFT=434
SLO_ITL=27.8

echo "=== provenance preflight ==="
echo "  HEAD: $(git rev-parse HEAD)"
DIRT="$(git status --porcelain)"
if [ -n "$DIRT" ]; then
    echo "FATAL: working tree is dirty — artifacts would be stamped NOT PUBLISHABLE."
    echo "$DIRT" | head -20
    exit 1
fi
echo "  tree: CLEAN"

OUT="${SLURM_SUBMIT_DIR}/../p5_blend_out_${SLURM_JOB_ID}"
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
start_router () {   # $1=port $2=policy $3=blend-or-empty $4=label
    ROUTER_POLICY=$2 ROUTER_BLEND=$3 ROUTER_REPLICAS="$REPLICAS" \
    python3 -m uvicorn scripts._router_entry:app --factory \
        --host 127.0.0.1 --port "$1" --log-level warning \
        > "logs/router_${4}_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=($!)
    for i in $(seq 1 60); do
        curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { echo "  router up: $4 :$1 (${i}s)"; return 0; }
        sleep 1
    done
    echo "FATAL: router $4 did not become healthy"; tail -20 "logs/router_${4}_${SLURM_JOB_ID}.log"; return 1
}
stop_all () { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; PIDS=(); sleep 5; }
trap stop_all EXIT

for i in 1 2 3 4; do
    start_one $((9500+i)) "p5b_replica$i" CUDA_VISIBLE_DEVICES=$((i-1))
done
REPLICAS="http://127.0.0.1:9501,http://127.0.0.1:9502,http://127.0.0.1:9503,http://127.0.0.1:9504"

# Six arms against one fleet. The 0.70 arm is named `prefix_aware` rather than
# `blend_0.70` for one reason: run_p5 hard-requires that name (it refuses to
# start without it, bench/run_p5.py:340) and computes its d_goodput_vs_B5 column
# from it. 0.70 IS build_default_router's default, so the arm the driver treats
# as "the system under test" is exactly the configuration that produced the
# -23% in 11653158. The other blends ride alongside as extra named policies.
#
# blend_0.00 is B5 by construction and by test;
# least_outstanding is ALSO started, as a check that the assertion holds in a
# live fleet and not only in a unit test. If those two arms disagree materially,
# the blend=0 equivalence is broken and every advantage in this run is suspect.
start_router 9610 prefix_aware      0.00 blend_0.00
start_router 9611 prefix_aware      0.25 blend_0.25
start_router 9612 prefix_aware      0.50 blend_0.50
start_router 9613 prefix_aware      0.70 prefix_aware   # 0.70 IS the library default
start_router 9614 prefix_aware      1.00 blend_1.00
start_router 9615 least_outstanding ""   least_outstanding
C () { echo "http://127.0.0.1:$1/v1/chat/completions"; }
R_ARGS="--router blend_0.00=$(C 9610) \
        --router blend_0.25=$(C 9611) \
        --router blend_0.50=$(C 9612) \
        --router prefix_aware=$(C 9613) \
        --router blend_1.00=$(C 9614) \
        --router least_outstanding=$(C 9615)"

# hot_prefix_skew only. It produced the largest effect in 11653158 (-0.633
# req/s) and it is the workload where affinity and load conflict hardest, so it
# is where a crossover, if one exists, is most visible. Two loads that both
# produced matched valid pairs last time -- 0.5 (below the knee) and 8.0 (8x
# above it) -- plus 2.0 to see whether the sign flips between them.
echo; echo "################ BLEND SWEEP: hot_prefix_skew ################"
set +e
python3 -m bench.run_p5 \
    --replicas "$REPLICAS" $R_ARGS \
    --loads 0.5,2,8 \
    --scenarios hot_prefix_skew \
    --duration 120 --warmup 20 --drain 10 \
    --slo-ttft-ms $SLO_TTFT --slo-itl-ms $SLO_ITL --seed 20260803 \
    --label blend_sweep --results-dir "$OUT" 2>&1
echo "  sweep exit: $?"
set -e

stop_all
mkdir -p results/p5_blend
cp -f "$OUT"/* results/p5_blend/ 2>/dev/null || true
echo
echo "=== artifacts: $(ls results/p5_blend 2>/dev/null | wc -l) in results/p5_blend ==="
echo "=== done ==="
