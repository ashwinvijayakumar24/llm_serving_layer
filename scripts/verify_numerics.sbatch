#!/bin/bash
#SBATCH --job-name=verify_num
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=logs/verify_num_%j.out
#SBATCH --error=logs/verify_num_%j.out
set -e
mkdir -p logs
module load cuda/12.9.1; module load anaconda3; conda activate llm
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1
export LLM_WEIGHTS_PATH="${LLM_WEIGHTS_PATH:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/vendor/llm_inference_engine"
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/verify_batch_numerics.py 2>&1
