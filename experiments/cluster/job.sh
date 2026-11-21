#!/bin/bash
# One task of a SLURM job array: runs run number $SLURM_ARRAY_TASK_ID of stage $STAGE.
# Submitted by submit.sh, which sets the account, time limit, array range and STAGE.
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%A_%a.out

module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR/.."                             # experiments/
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python runs.py "$STAGE" --index "$SLURM_ARRAY_TASK_ID"
