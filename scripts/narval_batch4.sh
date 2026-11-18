#!/bin/bash
# Fourth batch: two-trader runs at lam > 0 redone with the corrected first-order condition (the lam q_i(t) term). Submit from scripts/: sbatch narval_batch4.sh
#SBATCH --account=def-ankush
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=pig-batch4
#SBATCH --output=logs/%x_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate
export THREADS=2
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
# private lam=2, three seeds, FOC check
for sd in 1 2 3; do
  LAM=2 CV=1 SEED=$sd M_PATHS=16384 STEPS=1500 BATCH=2048 AVG_FROM=4 FOC_CHECK=1 SAVE=../results/pt_priv2_cv_fix_s$sd.npz python pt_game.py 12 > ../results/pt_priv2_cv_fix_s$sd.log 2>&1 &
done
# continuity: private lam = 0.25, 0.5, 1
for lam in 0.25 0.5 1; do
  LAM=$lam CV=1 M_PATHS=16384 STEPS=1500 BATCH=2048 AVG_FROM=4 FOC_CHECK=1 SAVE=../results/pt_priv_fix_lam$lam.npz python pt_game.py 12 > ../results/pt_priv_fix_lam$lam.log 2>&1 &
done
# full information lam=2 (exact by linear Picard, now also by the kernel benchmark), FOC check
INFO=full LAM=2 CV=1 M_PATHS=16384 STEPS=1500 BATCH=2048 AVG_FROM=4 FOC_CHECK=1 SAVE=../results/pt_full2_cv_fix.npz python pt_game.py 10 > ../results/pt_full2_cv_fix.log 2>&1 &
# bias tests at lam=0 (unchanged FOC): last iterate vs average is in every log; split-sample variant not implemented yet
wait
echo "batch4 done"
