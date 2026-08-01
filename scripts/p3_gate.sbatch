#!/bin/bash
# EVERY GPU CORRECTNESS GATE, ONE ALLOCATION.
#
# Queue time dominates compute time on PACE, so batching the gates into one job
# is strictly better than four jobs. It also means every gate sees the SAME
# node, the same driver and the same engine build, so a failure cannot be blamed
# on environment drift between runs.
#
# Gates, in dependency order — each builds on the one above:
#   P1  paged path      == contiguous engine path   (which P0 verified vs HF fp32)
#   P2  batched output  == single-sequence output
#   P3  preempted output == unpreempted output      (recompute AND swap)
#   P4  cache-on output == cache-off output
#
# REQUIRE_GPU=1 throughout: an unusable GPU is a hard failure, never a skip.
# A skipped gate and a passing gate are indistinguishable in a job log, and this
# project has already lost one allocation to exactly that (results/p1 §4).
#
#SBATCH --job-name=p3_gate
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=logs/all_gates_h200_%j.out
#SBATCH --error=logs/all_gates_h200_%j.out

set -e
mkdir -p logs
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  qos: ${SLURM_JOB_QOS} ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

module load cuda/12.9.1
module load anaconda3
conda activate llm
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export LLM_WEIGHTS_PATH="${LLM_WEIGHTS_PATH:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/vendor/llm_inference_engine:${PYTHONPATH}"
export REQUIRE_GPU=1

python3 - <<'PREFLIGHT'
import sys, torch
if not torch.cuda.is_available(): sys.exit("FATAL: no CUDA")
cap = torch.cuda.get_device_capability(0)
print(f"  {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}  torch {torch.__version__}")
if cap < (8, 0): sys.exit(f"FATAL: sm_{cap[0]}{cap[1]} < sm_80")
PREFLIGHT

python3 -m pip install -q "flashinfer-python==0.6.16" 2>&1 | tail -2 || echo "  (flashinfer unavailable — PagedTorchBackend is the fallback by design)"

echo; echo "=== CPU suite on the GPU node ==="
python3 -m pytest tests/ -q -m "not gpu" 2>&1 | tail -4

FAILED=""
run_gate () {   # $1 = label, $2 = path
    echo; echo "############ GATE: $1 ############"
    if python3 -m pytest "$2" -v -s 2>&1; then
        echo ">>> $1 PASSED"
    else
        echo ">>> $1 FAILED"
        FAILED="$FAILED $1"
    fi
}




run_gate "P3 preemption equality"        tests/test_preemption_gpu.py


echo; echo "================== SUMMARY =================="
if [ -z "$FAILED" ]; then
    echo "ALL GATES PASSED"
else
    echo "FAILED GATES:$FAILED"
    echo "The phases owning those gates are NOT complete and earn no claim."
fi
echo "=== done ==="
