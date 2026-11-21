#!/bin/bash
# Submit a stage of the study as a SLURM job array, one GPU per run.
# Usage, from experiments/cluster/:
#     ./submit.sh budget            all runs of the stage
#     ./submit.sh budget 0          only run 0, to time one run first
# Runs whose result file exists are skipped, so resubmitting is safe.
set -e

ACCOUNT="def-ankush"        # the allocation account; list with: sshare -U $USER
declare -A TIME=([budget]=5:00:00 [seeds]=4:00:00 [control_variate]=3:00:00 [width]=4:00:00 [damping]=4:00:00 [linear]=3:00:00 [closed_loop]=4:00:00)   # per run

STAGE=$1
[ -n "${TIME[$STAGE]}" ] || { echo "stage must be budget, seeds, control_variate, width, damping, linear or closed_loop"; exit 1; }
module load StdEnv/2023 python/3.11
if [ -n "$2" ]; then
    ARRAY=$2
else
    ARRAY=0-$(( $(python ../runs.py "$STAGE" --count) - 1 ))
fi

mkdir -p logs
sbatch --account="$ACCOUNT" --time="${TIME[$STAGE]}" --array="$ARRAY" \
       --job-name="pig-$STAGE" --export=ALL,STAGE="$STAGE" job.sh
