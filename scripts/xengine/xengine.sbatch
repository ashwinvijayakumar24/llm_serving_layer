#!/bin/bash
# CROSS-ENGINE STUDY — this serving layer vs vLLM vs SGLang (docs/xengine/SPEC.md).
#
# UNTESTED ON PACE. Written from the conventions of scripts/p2_bench.sbatch and
# scripts/p345_bench.sbatch; it has not been submitted yet. Expect to fix paths,
# the wall-clock limit and env activation on the first run.
#
# ONE ALLOCATION, ONE GPU, EVERY ARM (SPEC hard rule 5, ADR-021). PACE nodes
# drift ~25% for identical code, so a delta between two engines measured in two
# allocations measures node assignment, not the engines. All arms therefore run
# sequentially here, on the same GPU, and the renderer flags any artifact whose
# node, GPU, driver or job id differs from the rest.
#
# Prerequisite: scripts/xengine/setup_envs.sh has created the vLLM and SGLang
# envs under $ENV_ROOT. Our server and the harness client run in conda `llm`.
#
# Submit from the repo root:   sbatch scripts/xengine/xengine.sbatch
# Subset:                      ARMS="ours vllm sglang" WORKLOADS=W1 sbatch scripts/xengine/xengine.sbatch
#
#SBATCH --job-name=xengine
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
# PACE enforces a max 8:1 CPU:GPU ratio; 16 CPUs with one GPU is rejected.
#SBATCH --cpus-per-task=8
# 10 arms x 4 workloads x 3 reps, plus one server start per cell. Not yet
# measured: shorten once one real run gives the per-cell time.
#SBATCH --time=12:00:00
#SBATCH --output=logs/xengine_%j.out
#SBATCH --error=logs/xengine_%j.out

set -e
mkdir -p logs results/xengine/_env

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  qos: ${SLURM_JOB_QOS} ==="

module load cuda/12.9.1
module load anaconda3
conda activate llm

cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export LLM_WEIGHTS_PATH="${LLM_WEIGHTS_PATH:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/vendor/llm_inference_engine:${PYTHONPATH}"

ENV_ROOT="${ENV_ROOT:-$HOME/ps-simpliearn-0/xengine_envs}"
OURS_PY="$(command -v python3)"
VLLM_PY="${VLLM_PY:-$ENV_ROOT/xengine-vllm/bin/python}"
SGLANG_PY="${SGLANG_PY:-$ENV_ROOT/xengine-sglang/bin/python}"
for py in "$OURS_PY" "$VLLM_PY" "$SGLANG_PY"; do
    [ -x "$py" ] || { echo "FATAL: interpreter missing: $py (run scripts/xengine/setup_envs.sh)"; exit 1; }
done

# ---------------------------------------------------------------------------
# Record the GPU state before anything runs. Clocks, temperature and power are
# captured again at the end so a thermal or clock change across the job is
# visible next to the results rather than discovered later.
# ---------------------------------------------------------------------------
ENVDIR="results/xengine/_env"
GPU_Q="name,driver_version,compute_cap,memory.total,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit"
nvidia-smi                                   > "${ENVDIR}/nvidia-smi_${SLURM_JOB_ID}_start.txt"
nvidia-smi -q                               >> "${ENVDIR}/nvidia-smi_${SLURM_JOB_ID}_start.txt"
nvidia-smi --query-gpu="$GPU_Q" --format=csv | tee "${ENVDIR}/gpu_${SLURM_JOB_ID}_start.csv"

{
    echo "job ${SLURM_JOB_ID} node $(hostname)"
    echo "ours:   $("$OURS_PY"   -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)')"
    echo "vllm:   $("$VLLM_PY"   -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)')"
    echo "sglang: $("$SGLANG_PY" -c 'import sglang, torch; print("sglang", sglang.__version__, "torch", torch.__version__)')"
} | tee "${ENVDIR}/versions_${SLURM_JOB_ID}.txt"

"$OURS_PY" - <<'PREFLIGHT'
import sys, torch
if not torch.cuda.is_available(): sys.exit("FATAL: no CUDA")
cap = torch.cuda.get_device_capability(0)
if cap < (8, 0): sys.exit(f"FATAL: sm_{cap[0]}{cap[1]} < sm_80")
print(f"  {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}  torch {torch.__version__}")
PREFLIGHT

# ---------------------------------------------------------------------------
# The matrix. A failed cell does not abort the job: bench-all lists failures and
# returns non-zero at the end, and render still runs so the document shows what
# exists (missing cells render as TODO, never as a guess).
# ---------------------------------------------------------------------------
MAKEVARS=(OURS_PY="$OURS_PY" VLLM_PY="$VLLM_PY" SGLANG_PY="$SGLANG_PY" REPS="${REPS:-3}")
[ -n "${ARMS:-}" ]      && MAKEVARS+=(ARMS="$ARMS")
[ -n "${WORKLOADS:-}" ] && MAKEVARS+=(WORKLOADS="$WORKLOADS")

set +e
make bench-all "${MAKEVARS[@]}"
BENCH_RC=$?
if [ "${RUN_W5:-0}" = "1" ]; then
    make bench-w5 "${MAKEVARS[@]}" || BENCH_RC=1
fi
set -e

nvidia-smi --query-gpu="$GPU_Q" --format=csv | tee "${ENVDIR}/gpu_${SLURM_JOB_ID}_end.csv"

make render OURS_PY="$OURS_PY"

echo
echo "=== artifacts ==="
find results/xengine -name '*.json' | wc -l
echo "=== done (bench rc=${BENCH_RC}) ==="
exit "$BENCH_RC"
