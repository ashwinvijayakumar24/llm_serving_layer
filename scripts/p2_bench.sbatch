#!/bin/bash
# PHASE 2 BENCHMARK — goodput under a frozen SLO vs offered load.
#
# ONE ALLOCATION, ONE NODE, EVERYTHING. Calibration, the continuous-batching
# sweep, and the static-batching baseline all run here, back to back, because a
# comparison spanning two allocations measures node assignment rather than the
# change under test (R12). The engine's own throughput moved ~79 -> ~60 tok/s
# across nodes for identical code.
#
# Order is deliberate and enforced by bench/run_p2.py:
#   1. measure UNLOADED batch-1 service time against the live server
#   2. derive the SLO from it using multipliers fixed IN SOURCE beforehand
#   3. freeze and print the SLO
#   4. only then sweep offered load
# An SLO chosen after seeing results is not an SLO.
#
#SBATCH --job-name=p2_bench
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
# PACE enforces a max 8:1 CPU:GPU ratio on gpu-h100; 16 with one GPU is rejected
# at submission with 'Invalid generic resource (gres) specification'.
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/p2_bench_%j.out
#SBATCH --error=logs/p2_bench_%j.out

set -e
mkdir -p logs results/p2

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  qos: ${SLURM_JOB_QOS} ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

module load cuda/12.9.1
module load anaconda3
conda activate llm

cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export LLM_WEIGHTS_PATH="${LLM_WEIGHTS_PATH:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/vendor/llm_inference_engine:${PYTHONPATH}"

python3 - <<'PREFLIGHT'
import sys, torch
if not torch.cuda.is_available(): sys.exit("FATAL: no CUDA")
cap = torch.cuda.get_device_capability(0)
if cap < (8, 0): sys.exit(f"FATAL: sm_{cap[0]}{cap[1]} < sm_80")
print(f"  {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}  torch {torch.__version__}")
PREFLIGHT

PORT=8${SLURM_JOB_ID: -3}
URL="http://127.0.0.1:${PORT}/v1/chat/completions"

start_server () {   # $1 = label, $2.. = extra env
    echo "--- starting server ($1) on :${PORT} ---"
    env "${@:2}" python3 -m uvicorn serving.server.app:build_default_app \
        --factory --host 127.0.0.1 --port "${PORT}" --log-level warning \
        > "logs/server_${1}_${SLURM_JOB_ID}.log" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            echo "--- server up after ${i}s ---"; return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "FATAL: server died during startup"; tail -40 "logs/server_${1}_${SLURM_JOB_ID}.log"; exit 1
        fi
        sleep 1
    done
    echo "FATAL: server did not become healthy in 120s"; tail -40 "logs/server_${1}_${SLURM_JOB_ID}.log"; exit 1
}

stop_server () {
    [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 3
}
trap stop_server EXIT

# ---------------------------------------------------------------------------
# A — continuous batching (the system under test)
# ---------------------------------------------------------------------------
start_server continuous
echo
echo "############ A: CONTINUOUS BATCHING ############"
python3 -m bench.run_p2 --url "$URL" --label continuous \
    --rates "${RATES:-1,2,4,8,16,24,32}" \
    --duration "${DURATION:-45}" --warmup 15 --drain 10 \
    --prompt-tokens 128 --max-tokens 64 --seed 20260801 || true
stop_server

# ---------------------------------------------------------------------------
# B — B2 static-batching baseline, SAME allocation, SAME node, back to back.
#
# Approximated by forcing the scheduler to admit a full batch and run it out:
# max_batch_size unchanged, but admission is blocked until the running set
# drains. Kernels, memory manager and HTTP stack are held constant, so the ONLY
# difference is scheduling — which is what makes the comparison meaningful.
# ---------------------------------------------------------------------------
if [ "${SKIP_STATIC:-0}" != "1" ]; then
    start_server static SERVING_STATIC_BATCHING=1
    echo
    echo "############ B: STATIC BATCHING (baseline B2) ############"
    python3 -m bench.run_p2 --url "$URL" --label static \
        --rates "${RATES:-1,2,4,8,16,24,32}" \
        --duration "${DURATION:-45}" --warmup 15 --drain 10 \
        --prompt-tokens 128 --max-tokens 64 --seed 20260801 || true
    stop_server
fi

echo
echo "=== artifacts ==="
ls -1 results/p2/ | tail -30
echo "=== done ==="
