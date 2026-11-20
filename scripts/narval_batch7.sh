#!/bin/bash
# Seventh batch (GPU): the non-Gaussian two-trader game (private Markov-chain signals). The control variate now includes the
# compensated chain increments (CV_JUMP=1), which cut the regression loss by a factor of four in a local test.
# lam = 0 against the exact solution (control variate off / price only / with jumps at 32k, with jumps at 128k);
# lam = 1, three seeds at 64k, with the first-order-condition residual. Submit from scripts/: sbatch narval_batch7.sh
#SBATCH --account=def-ankush
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=pig-batch7gpu
#SBATCH --output=logs/%x_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate
export THREADS=1; export DEVICE=cuda
export RESULTS_DATA="$SLURM_SUBMIT_DIR/../results"
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
for v in "0 0 none" "1 0 price" "1 1 jump"; do
  set -- $v
  LAM=0 CV=$1 CV_JUMP=$2 RHO_DAMP=0.5 M_PATHS=32768 STEPS=1500 BATCH=2048 AVG_FROM=6 SAVE=../results/b7_pm_lam0_32k_$3.npz python pm_game.py 16 > ../results/b7_pm_lam0_32k_$3.log 2>&1 &
done
LAM=0 CV=1 CV_JUMP=1 RHO_DAMP=0.5 M_PATHS=131072 STEPS=6000 BATCH=2048 AVG_FROM=6 SAVE=../results/b7_pm_lam0_128k_jump.npz python pm_game.py 16 > ../results/b7_pm_lam0_128k_jump.log 2>&1 &
for sd in 1 2 3; do
  LAM=1 CV=1 CV_JUMP=1 SEED=$sd M_PATHS=65536 STEPS=3000 BATCH=2048 AVG_FROM=6 FOC_CHECK=1 SAVE=../results/b7_pm_lam1_64k_s$sd.npz python pm_game.py 16 > ../results/b7_pm_lam1_64k_s$sd.log 2>&1 &
done
wait
echo "batch7 done"
