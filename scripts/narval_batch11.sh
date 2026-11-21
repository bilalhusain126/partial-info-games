#!/bin/bash
# Eleventh batch (GPU): the three 128k runs of batch 6 that ran out of GPU memory when eleven runs shared one GPU. Same settings.
# Submit from scripts/: sbatch narval_batch11.sh
#SBATCH --account=def-ankush
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=pig-batch11gpu
#SBATCH --output=logs/%x_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate
export THREADS=1; export DEVICE=cuda
export RESULTS_DATA="$SLURM_SUBMIT_DIR/../results"
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
LAM=0 CV=1 M_PATHS=131072 STEPS=6000 BATCH=2048 AVG_FROM=6 SAVE=../results/b11_pt_lam0_128k_cv1.npz python pt_game.py 16 > ../results/b11_pt_lam0_128k_cv1.log 2>&1 &
LAM=1 CV=1 M_PATHS=131072 STEPS=6000 BATCH=2048 AVG_FROM=6 REF_NPZ=../references/pt_private_lam1.0.npz SAVE=../results/b11_pt_lam1_128k_cv1.npz python pt_game.py 16 > ../results/b11_pt_lam1_128k_cv1.log 2>&1 &
H=0 CV=1 M_PATHS=131072 STEPS=6000 BATCH=2048 AVG_FROM=6 ITERS=16 SAVE=../results/b11_wj_h0_128k_cv1.npz python npg_wj.py > ../results/b11_wj_h0_128k_cv1.log 2>&1 &
wait
echo "batch11 done"
