#!/bin/bash
# Submit the cross-engine study to PACE (docs/xengine/RUNBOOK.md).
#
#   scripts/xengine/submit.sh pilot     one short job on embers (free): 3 baseline arms, W1 points
#                                       1,16, 1 rep, into results/xengine_pilot (never rendered)
#   scripts/xengine/submit.sh all       one job per workload W1..W4 (+ W5 appendix with W1/W2's job)
#   scripts/xengine/submit.sh W3        one workload
#
# Why one job per workload: comparisons are only ever made WITHIN a workload, so
# every arm of a workload shares one allocation and one GPU (SPEC rule 5). Four
# shorter jobs also queue faster than one long one and a failure costs one
# workload, not the study. The renderer flags any cross-node mixing it sees.
#
# Wall-clock limits: checked against pilot job 13918362 (server start: vLLM ~7.5 min,
# SGLang ~4.5 min, ours ~20 s per arm per workload; W1 points ~20 s-2 min each).

set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

SB=scripts/xengine/xengine.sbatch

# Extra sbatch options for every submission (e.g. QOS override), set by a case below.
SBATCH_EXTRA=()

submit() {  # name time env...
    local name=$1 time=$2; shift 2
    echo "+ sbatch --job-name=$name --time=$time ${SBATCH_EXTRA[*]:-} ($*)"
    env "$@" sbatch --job-name="$name" --time="$time" "${SBATCH_EXTRA[@]}" --export=ALL "$SB"
}

case "${1:-}" in
    pilot)
        # Setup check, not a measurement: free preemptible QOS, same H200 as the
        # real matrix (which stays on inferno, non-preemptible). Only the QOS is
        # overridden: an untyped --gres=gpu:1 let embers place job 13910146 on a
        # V100 (sm_70), which the sbatch preflight correctly rejected.
        SBATCH_EXTRA=(--qos=embers)
        submit xengine-pilot 02:00:00 \
            WORKLOADS=W1 ARMS="ours vllm sglang" REPS=1 POINTS=1,16 OUT=results/xengine_pilot
        ;;
    all)
        submit xengine-W1 06:00:00 WORKLOADS=W1 RUN_W5=1 W5_WORKLOADS=W1
        submit xengine-W2 04:00:00 WORKLOADS=W2 RUN_W5=1 W5_WORKLOADS=W2
        submit xengine-W3 05:00:00 WORKLOADS=W3
        submit xengine-W4 04:00:00 WORKLOADS=W4
        ;;
    W1|W2|W3|W4)
        submit "xengine-$1" 06:00:00 WORKLOADS="$1"
        ;;
    *)
        echo "usage: $0 pilot | all | W1|W2|W3|W4" >&2
        exit 2
        ;;
esac
