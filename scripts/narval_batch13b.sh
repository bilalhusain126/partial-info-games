#!/bin/bash
# Batch 13b (GPU): seeds 1 and 2 at 64k for Firoozi-Caines and the Markov-chain game at lam = 0, same settings as batch 9 (seed 0),
# and one width-64 run (Markov chain, lam = 0). Submit from scripts/: sbatch narval_batch13b.sh
#SBATCH --job-name=pig-batch13b
#SBATCH --account=def-ankush
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate
export THREADS=1; export DEVICE=cuda
export RESULTS_DATA="$SLURM_SUBMIT_DIR/../results"
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
for sd in 1 2; do
  CV=1 SEED=$sd M_PATHS=65536 STEPS=3000 BATCH=2048 AVG_FROM=6 SAVE=../results/b13_fc_64k_s$sd.npz python fc_game.py 16 > ../results/b13_fc_64k_s$sd.log 2>&1 &
  LAM=0 CV=1 CV_JUMP=1 RHO_DAMP=0.5 SEED=$sd M_PATHS=65536 STEPS=3000 BATCH=2048 AVG_FROM=6 SAVE=../results/b13_pm_lam0_64k_s$sd.npz python pm_game.py 16 > ../results/b13_pm_lam0_64k_s$sd.log 2>&1 &
done
HID=64 LAM=0 CV=1 CV_JUMP=1 RHO_DAMP=0.5 M_PATHS=65536 STEPS=3000 BATCH=2048 AVG_FROM=6 SAVE=../results/b13_pm_lam0_64k_hid64.npz python pm_game.py 16 > ../results/b13_pm_lam0_64k_hid64.log 2>&1 &
wait
echo "batch13b done"
