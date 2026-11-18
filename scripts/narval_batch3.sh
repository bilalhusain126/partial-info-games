#!/bin/bash
# Third batch: the strong-coupling Wu-Jaimungal runs that diverged in batch 2, with heavy damping. Submit from scripts/: sbatch narval_batch3.sh
#SBATCH --account=def-ankush
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=pig-batch3
#SBATCH --output=logs/%x_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate
export THREADS=2
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
# a = b = 0.1: the best-response gain scales like 1/b, so the plain iteration is far from a contraction; damping 0.2 and 0.1, more iterations
for r in 0.2 0.1; do
  A_COST=0.1 B_COST=0.1 BROKER=gru  RHO_DAMP=$r M_PATHS=16384 FIT_STEPS=1500 BATCH=2048 SAVE=../results/wj_strong_h0_gru_rho$r.npz python wj_game.py 0 0.3 40 > ../results/wj_strong_h0_gru_rho$r.log 2>&1 &
  A_COST=0.1 B_COST=0.1 BROKER=gru  RHO_DAMP=$r M_PATHS=16384 FIT_STEPS=1500 BATCH=2048 SAVE=../results/wj_strong_h1_gru_rho$r.npz python wj_game.py 1 0.3 40 > ../results/wj_strong_h1_gru_rho$r.log 2>&1 &
done
wait
echo "batch3 done"
