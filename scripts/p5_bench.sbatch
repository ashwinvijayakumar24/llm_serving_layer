#!/bin/bash
# PHASE 3, 4 and 5 BENCHMARKS — one allocation, one node, back to back (R12).
#
# Every arm of every comparison runs on the same GPU in the same job, because a
# delta computed across allocations measures node assignment rather than the
# change under test.
#
# The SLO is inherited from the Phase 2 calibration rather than re-derived:
# these phases change scheduling and memory behaviour, not decode speed, so
# re-anchoring per phase would move the goalposts between comparisons.
#
#SBATCH --job-name=p5_bench
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=logs/p5_bench_%j.out
#SBATCH --error=logs/p5_bench_%j.out

set -e
mkdir -p logs results/p5

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  qos: ${SLURM_JOB_QOS} ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

module load cuda/12.9.1; module load anaconda3; conda activate llm
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export LLM_WEIGHTS_PATH="${LLM_WEIGHTS_PATH:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/vendor/llm_inference_engine:${PYTHONPATH}"

# Frozen in the Phase 2 calibration (job 11602207): unloaded TTFT p50 43.4 ms,
# unloaded TPOT p50 9.3 ms, multipliers 10x and 3x fixed in source beforehand.
SLO_TTFT=434
SLO_ITL=27.8

# EVERY server must be given an explicit pool size. plan_kv_pool sizes from
# TOTAL VRAM, so an unconstrained server takes essentially the whole GPU and the
# next one OOMs while loading weights — which is exactly how job 11608239 died.
PIDS=()
start_one () {   # $1=port  $2=label  rest=env assignments
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
stop_all () { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; PIDS=(); sleep 4; }
trap stop_all EXIT

U () { echo "http://127.0.0.1:$1/v1/chat/completions"; }

# ---------------------------------------------------------------------------
# PHASE 5 — routing. FOUR replicas on ONE GPU.
#
# STATED PLAINLY BECAUSE IT LIMITS THE CLAIM: these replicas share one H200.
# They are separate processes with separate schedulers, block pools and prefix
# caches, so the ROUTING logic is exercised exactly as it would be across
# machines — but they contend for the same SMs, so absolute throughput is not
# what four dedicated GPUs would give. The comparison between policies is still
# valid because every policy runs against the same four replicas on the same
# GPU; the absolute numbers are not a four-GPU result and are not reported as
# one.
# ---------------------------------------------------------------------------
echo; echo "################ P5: PREFIX-AWARE ROUTING ################"
for i in 1 2 3 4; do start_one $((9500+i)) "p5_replica$i" SERVING_KV_BLOCKS=4000; done
REPLICAS="http://127.0.0.1:9501,http://127.0.0.1:9502,http://127.0.0.1:9503,http://127.0.0.1:9504"

# ONE ROUTER PROCESS PER POLICY, all pointing at the SAME four replicas. The
# driver requires least_outstanding (B5) and refuses to run without it, because
# B5 is the only baseline that isolates cache awareness: it is load-aware and
# cache-blind, so the gap between it and prefix_aware is the claim. Beating
# round_robin (B4) is table stakes and proves only load balancing.
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
start_router 9601 prefix_aware
start_router 9602 least_outstanding
start_router 9603 round_robin

python3 -m bench.run_p5 \
    --replicas "$REPLICAS" \
    --router prefix_aware=http://127.0.0.1:9601/v1/chat/completions \
    --router least_outstanding=http://127.0.0.1:9602/v1/chat/completions \
    --router round_robin=http://127.0.0.1:9603/v1/chat/completions \
    --loads 2,4,8,16 --scenarios zero_sharing,system_prompt_sharing,uniform_prefix,hot_prefix_skew \
    --duration 45 --warmup 15 --drain 10 \
    --slo-ttft-ms $SLO_TTFT --slo-itl-ms $SLO_ITL --seed 20260801 \
    --results-dir results/p5 2>&1 || echo "P5 driver exited non-zero (invalid cells)"
stop_all

echo; echo "=== artifacts ==="
for d in p3 p4 p5; do echo "results/$d: $(ls results/$d 2>/dev/null | wc -l) files"; done
echo "=== done ==="
