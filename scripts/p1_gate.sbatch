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
# PARTITION AND GRES ARE DELIBERATELY UNPINNED.
#
# This gate needs a CUDA device and nothing else: PagedTorchBackend is pure
# PyTorch, and the contiguous reference path runs with use_cuda_attn=False, so
# no custom kernel and therefore no sm_80 dependency. Naming one partition just
# means queueing behind whoever wants that exact GPU — measured: gpu-a100 had 33
# jobs pending while gpu-h100 had 10. Listing several lets Slurm take whichever
# frees first, and `--gres=gpu:1` (generic, not `gpu:a100:1`) is what makes that
# legal across GPU types.
#
# BENCHMARKS MUST NOT DO THIS. A performance number has to name its hardware and
# stay on one allocation (R12). Unpinning is correct for correctness runs only.
#
#SBATCH --job-name=p1_gate
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h100,gpu-a100,gpu-l40s,gpu-h200
#SBATCH --gres=gpu:1
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
python3 bench/capacity.py --seed 20260731 --gpu --sweep --write 2>&1 | tail -50

echo
echo "=== done ==="
