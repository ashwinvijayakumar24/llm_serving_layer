#!/bin/bash
# PHASE 4 — radix prefix cache, redone on two axes.
#
# 1. THE CACHE'S OWN CORRECTNESS GATE, at batch size 1.
#    The batched gate fails for a reason outside the cache: the engine's
#    linear() takes the packed batch token count as its M dimension, cuBLAS
#    picks kernels and split-K by shape, and fp16 reduction order therefore
#    depends on how many OTHER requests share the forward pass (max|dlogit|
#    0.1745 with the cache entirely out of the loop). A cache HIT changes batch
#    shape, so it trips that property. At batch 1 both arms compute
#    identically-shaped GEMMs and any divergence IS the cache's fault — a
#    stricter test of the cache, not a weaker one.
#
# 2. THE CROSSOVER. The first run measured ~150-token prompts and found the
#    cache COSTS +36 to +75 ms TTFT p50. methodology §10 predicted exactly that
#    shape: the saving is prefill work avoided, so it shrinks toward zero as
#    prompts shorten, and there is a length below which the cache is net
#    negative. Prefill for 150 tokens on an H200 is a few milliseconds while the
#    trie walk and refcounting are charged on every request.
#    Measuring at 512/1024/2048 tokens tests that prediction rather than
#    shopping for a better number — and BOTH results are published.
#
#SBATCH --job-name=p4_bench
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/p4_bench_%j.out
#SBATCH --error=logs/p4_bench_%j.out

set -e
mkdir -p logs results/p4
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load cuda/12.9.1; module load anaconda3; conda activate llm
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export LLM_WEIGHTS_PATH="${LLM_WEIGHTS_PATH:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/vendor/llm_inference_engine:${PYTHONPATH}"
export REQUIRE_GPU=1
SLO_TTFT=434
SLO_ITL=27.8

echo; echo "################ P4 GATE: cache correctness at batch=1 ################"
# `|| true` deliberately: a FAILING gate must not abort the script under `set -e`
# and take the crossover sweep with it, which is what killed job 11610459. The
# exit status is captured explicitly instead — `echo "$?"` on the next line would
# report the echo's own status, not pytest's.
set +e
python3 -m pytest tests/test_radix_gpu.py -v -s -k "uniform_chunk_shape" 2>&1
GATE_RC=$?
set -e
echo ">>> uniform-chunk gate exit code: ${GATE_RC}"

echo; echo "################ P4 GATE: full batched (expected to fail, engine numerics) ################"
set +e; python3 -m pytest tests/test_radix_gpu.py -q -k "identical_with_cache_on_and_off" 2>&1 | tail -6; set -e

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
stop_all () { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; PIDS=(); sleep 4; }
trap stop_all EXIT
U () { echo "http://127.0.0.1:$1/v1/chat/completions"; }

start_one 9401 p4_on  SERVING_KV_BLOCKS=40000
start_one 9402 p4_off SERVING_KV_BLOCKS=40000 SERVING_PREFIX_CACHE=0

# The crossover sweep. 150 tokens is the length already measured as NEGATIVE and
# is included so both sides of the prediction appear in one table.
for PLEN in 150 512 1024 2048; do
    echo; echo "########## prompt_mean=${PLEN} ##########"
    python3 -m bench.run_p4 \
        --url "$(U 9401)" --cache-off-url "$(U 9402)" \
        --structures zero,system,conversational \
        --sharing-rates 0.0,0.5,1.0 \
        --prompt-mean "$PLEN" --output-mean 32 \
        --rate 2 --duration 45 --warmup 15 --drain 10 \
        --slo-ttft-ms $SLO_TTFT --slo-itl-ms $SLO_ITL --seed 20260801 \
        --label "len${PLEN}" --results-dir results/p4 2>&1 || echo "(invalid cells at ${PLEN})"
done
stop_all
echo; echo "=== artifacts: $(ls results/p4 | wc -l) ==="
echo "=== done ==="
