#!/bin/bash
# PHASE 1 GATE — paged path vs contiguous reference path, on real weights.
#
# Phase 1 is not done until this passes. Everything else in Phase 1 tests a
# component in isolation; this tests the composition:
#
#   BlockAllocator -> SequenceBlocks -> build_batch_meta -> PagedTorchBackend
#       -> LlamaModelGPU.forward_varlen
#
# must produce greedy tokens BIT-IDENTICAL to the engine's contiguous path,
# which P0 validated against the fp32 HuggingFace oracle.
#
#     sbatch scripts/p1_gate.sbatch
#
# QOS is `inferno`, not `embers`. Measured in P0: an embers job sat at
# Priority=21 behind ~50 queued jobs with StartTime unknown; the same job on
# inferno started in ~8 minutes. At 0.282 SU/A100-GPU-hour the cost is noise.
# The validity rule is unaffected — this is a correctness gate, not a published
# number, and no benchmark may come from embers regardless.
#
# GPU TYPE IS PINNED, AND THE REASON IS A BUG THIS SCRIPT ALREADY HIT.
#
# A generic `--gres=gpu:1` was routed by PACE to a Tesla V100 (sm_70) even
# though the partition list excluded gpu-v100 — the site remaps generic gres.
# This conda env runs torch 2.12+cu130, and CUDA 13 dropped Volta, so
# torch.cuda.is_available() came back False, every gate test SKIPPED, and the
# job exited 0. A skipped gate is indistinguishable from a passing one.
#
# Typed gres is what actually pins the hardware. sm_80+ is required.
#
#SBATCH --job-name=p1_gate
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/p1_gate_%j.out
#SBATCH --error=logs/p1_gate_%j.out

set -e
mkdir -p logs

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  qos: ${SLURM_JOB_QOS} ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

module load cuda/12.9.1
module load anaconda3
conda activate llm

cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/vendor/llm_inference_engine:${PYTHONPATH}"

echo
echo "=== versions ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
git -C vendor/llm_inference_engine describe --tags --always
git rev-parse --short HEAD

echo
echo "=== PREFLIGHT: the GPU must be usable, or fail now ==="
# Fail here rather than letting the gate skip. Job 11598374 skipped all 16 gate
# tests on a V100 and reported success; that must never happen silently again.
python3 - <<'PREFLIGHT'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("FATAL: torch.cuda.is_available() is False")
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"  {name}  sm_{cap[0]}{cap[1]}  torch {torch.__version__} cuda {torch.version.cuda}")
if cap < (8, 0):
    sys.exit(f"FATAL: {name} is sm_{cap[0]}{cap[1]}; this torch build requires sm_80+")
print("  preflight OK")
PREFLIGHT

export REQUIRE_GPU=1      # turns a skipped gate into a hard failure

echo
echo "=== CPU suite (must still pass on the GPU node) ==="
python3 -m pytest tests/ -q -m "not gpu" 2>&1 | tail -5

echo
echo "=== THE PHASE 1 GATE: paged == contiguous ==="
python3 -m pytest tests/test_paged_e2e.py -v -s 2>&1

echo
echo "=== capacity measurement (S1) ==="
# --sweep reports the ratio ACROSS mean sequence lengths, because a single
# capacity ratio without its length distribution is not a defensible claim.
# --write emits an Artifact with full provenance into results/.
python3 bench/capacity.py --seed 20260731 --gpu --sweep 32,64,128,256,512,1024,2048 --write 2>&1 | tail -50

echo
echo "=== done ==="
