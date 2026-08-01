#!/bin/bash
# Short diagnostic: one request through the real stack, every layer printed.
# Lives in the repo rather than /tmp because PACE has several login nodes and a
# file written on one is not visible from another.
#SBATCH --job-name=dbg
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/dbg_%j.out
#SBATCH --error=logs/dbg_%j.out
set -e
mkdir -p logs
module load cuda/12.9.1; module load anaconda3; conda activate llm
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1
export LLM_WEIGHTS_PATH="${LLM_WEIGHTS_PATH:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/vendor/llm_inference_engine"
python3 scripts/debug_p2.py 2>&1
