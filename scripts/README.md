# scripts/

Slurm batch scripts and PACE helpers.

Hardware selection by purpose (docs/PRD.md §8/O1):
  gpu-l40s   8/node, 0.78x A100 SU rate  -> scheduling + routing results
  gpu-a100   2/node (one 8-GPU node)     -> continuity with engine numbers
  gpu-h100   8/node, 2.43x A100 SU rate  -> absolute-throughput claims only
  gpu-h200   8/node, 2.43x A100 SU rate  -> absolute-throughput claims only

QOS:
  embers   free, preemptible, 8h cap   -> development and correctness ONLY
  inferno  charged, up to 32 GPUs, 3d  -> every published benchmark

Calibrate SU burn with one instrumented job (pace-quota before/after) before
budgeting a multi-GPU campaign. The ~0.28 SU/A100-GPU-hr figure in the PRD is a
two-sample inference and should not be planned against.
