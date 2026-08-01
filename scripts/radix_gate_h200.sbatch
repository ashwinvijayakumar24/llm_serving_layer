#!/bin/bash
# P4 ONLY — the radix prefix-cache gate, plus the R6 KV diagnostic.
# Same environment as scripts/all_gates_h200.sbatch; narrower gate list so the
# iteration loop is queue time plus two minutes rather than queue time plus an hour.
#
#SBATCH --job-name=radix_gate
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=logs/radix_gate_h200_%j.out
#SBATCH --error=logs/radix_gate_h200_%j.out

set -e
mkdir -p logs
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

module load cuda/12.9.1
module load anaconda3
conda activate llm
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export LLM_WEIGHTS_PATH="${LLM_WEIGHTS_PATH:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/vendor/llm_inference_engine:${PYTHONPATH}"
export REQUIRE_GPU=1

python3 -m pip install -q "flashinfer-python==0.6.16" 2>&1 | tail -1 || true

if [ "${RADIX_DIAGNOSE:-0}" = "1" ]; then
    echo; echo "############ R6 DIAGNOSTIC ############"
    for s in ${RADIX_STRUCTURES:-system conversational adversarial}; do
        for b in ${RADIX_BUDGETS:-128}; do
            python3 scripts/debug_radix.py "$s" "$b" 2>&1 || echo ">>> errored on $s/$b"
        done
    done
fi

if [ "${RADIX_SKIP_GATE:-0}" = "1" ]; then
    echo "=== done ==="
    exit 0
fi

echo; echo "############ CPU suite ############"
python3 -m pytest tests/ -q -m "not gpu" 2>&1 | tail -5

echo; echo "############ GATE: P4 radix cache equality ############"
if python3 -m pytest tests/test_radix_gpu.py -vv 2>&1; then
    echo ">>> P4 PASSED"
else
    echo ">>> P4 FAILED"
fi
echo "=== done ==="
