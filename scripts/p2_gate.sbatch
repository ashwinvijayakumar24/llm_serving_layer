#!/bin/bash
# PHASE 2 GATE (+ the outstanding Phase 1 FlashInfer differential).
#
# Two independent gates, both of which must pass:
#
#   1. BATCH INVARIANCE (R4, Phase 2). A prompt must produce identical greedy
#      output alone and inside a mixed batch. Nothing in this project has run
#      n_seqs > 1 on real weights before this — Phase 1 was batch-1 throughout.
#      Every throughput claim is a claim that batching changed WHEN work
#      happened, not WHAT it produced.
#
#   2. FLASHINFER DIFFERENTIAL (R9, Phase 1). FlashInferBackend must match
#      PagedTorchBackend, which the Phase 1 gate proved equals the contiguous
#      engine path. This is what retires R9 and what makes the "integrated
#      FlashInfer" attribution honest.
#
#     sbatch scripts/p2_gate.sbatch
#
#SBATCH --job-name=p2_gate
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/p2_gate_%j.out
#SBATCH --error=logs/p2_gate_%j.out

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

echo
echo "=== PREFLIGHT ==="
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

export REQUIRE_GPU=1     # a skipped gate must never be reported as a pass

echo
echo "=== CPU suite ==="
python3 -m pytest tests/ -q -m "not gpu" 2>&1 | tail -4

echo
echo "=== GATE 1: BATCH INVARIANCE (R4) ==="
python3 -m pytest tests/test_batch_invariance.py -v -s 2>&1

echo
echo "=== regression: Phase 1 gate must still pass ==="
python3 -m pytest tests/test_paged_e2e.py -q 2>&1 | tail -4

echo
echo "=== flashinfer availability ==="
# Pinned to the version the source was actually read at. Installing here rather
# than assuming it is present; if it fails, the differential is reported as
# UNAVAILABLE and the run does NOT claim R9 is retired.
python3 -m pip install -q "flashinfer-python==0.6.16" 2>&1 | tail -3 || echo "  install failed"
python3 -c "
try:
    import flashinfer; print('  flashinfer', flashinfer.__version__)
except Exception as e:
    print('  flashinfer UNAVAILABLE:', type(e).__name__, e)
"

echo
echo "=== GATE 2: FLASHINFER DIFFERENTIAL (R9) ==="
# Not fatal to the job if flashinfer will not install — PagedTorchBackend is the
# fallback by design and the Phase 1 memory claim never depended on FlashInfer.
# But R9 stays OPEN unless this actually runs and passes.
python3 -m pytest tests/test_flashinfer_differential.py -v -s 2>&1 || echo "DIFFERENTIAL DID NOT PASS — R9 REMAINS OPEN"

echo
echo "=== done ==="
