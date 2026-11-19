#!/bin/bash
# Fifth batch (GPU): the non-Gaussian two-trader game (private Markov-chain signals, games/pm_game.py), plus the lam=1 seeds of the Gaussian game.
# Submit from scripts/: sbatch narval_batch5.sh
#SBATCH --account=def-ankush
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=pig-batch5gpu
#SBATCH --output=logs/%x_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate
export THREADS=2; export DEVICE=cuda
export RESULTS_DATA="$SLURM_SUBMIT_DIR/../results"          # on the cluster, npz next to the logs
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
# (1) Markov-chain signals, lam = 0 (exact reference), two budgets, with the two checks
for m in 8192 32768; do
  LAM=0 CV=1 RHO_DAMP=0.5 M_PATHS=$m STEPS=1500 BATCH=2048 AVG_FROM=4 FOC_CHECK=1 GATEAUX=1 SAVE=../results/pm_lam0_$m.npz python pm_game.py 12 > ../results/pm_lam0_$m.log 2>&1 &
done
# (2) Markov-chain signals, private, lam = 0.5 and lam = 1 (three seeds), checks
LAM=0.5 CV=1 M_PATHS=16384 STEPS=1500 BATCH=2048 AVG_FROM=4 FOC_CHECK=1 GATEAUX=1 SAVE=../results/pm_priv_lam0.5.npz python pm_game.py 12 > ../results/pm_priv_lam0.5.log 2>&1 &
for sd in 1 2 3; do
  LAM=1 CV=1 SEED=$sd M_PATHS=16384 STEPS=1500 BATCH=2048 AVG_FROM=4 FOC_CHECK=1 GATEAUX=1 SAVE=../results/pm_priv1_s$sd.npz python pm_game.py 12 > ../results/pm_priv1_s$sd.log 2>&1 &
done
# (3) Gaussian game, headline lam = 1, three seeds (never run), checks
for sd in 1 2 3; do
  LAM=1 CV=1 SEED=$sd M_PATHS=16384 STEPS=1500 BATCH=2048 AVG_FROM=4 FOC_CHECK=1 GATEAUX=1 SAVE=../results/pt_priv1_cv_fix_s$sd.npz python pt_game.py 12 > ../results/pt_priv1_cv_fix_s$sd.log 2>&1 &
done
wait
echo "batch5 done"
