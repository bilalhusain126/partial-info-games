#!/bin/bash
# First batch on Narval (CPU only; runs parallelise across experiments). Submit from scripts/ with: sbatch narval_batch1.sh
#SBATCH --account=def-ankush
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=pig-batch1
#SBATCH --output=logs/%x_%j.out
set -e
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate                    # environment from latent-factors-control/experiments/cluster/setup.sh
export THREADS=2                                  # torch threads per run (npg.py reads THREADS); 8 runs x 2 = 16 cpus
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
# budget scaling at lam=0 (exact reference), control variate on
for m in 2048 4096 8192 16384 32768; do
  LAM=0 CV=1 RHO_DAMP=0.5 M_PATHS=$m STEPS=$(( m/8 > 2000 ? 2000 : m/8 )) BATCH=$(( m/4 > 2048 ? 2048 : m/4 )) AVG_FROM=4 SAVE=../results/pt_scale_$m.npz python pt_game.py 10 > ../results/pt_scale_$m.log 2>&1 &
done
# private two-way case, three seeds with exploitability, and the full-information comparison
for sd in 1 2 3; do
  LAM=2 CV=1 SEED=$sd M_PATHS=16384 STEPS=1500 BATCH=2048 AVG_FROM=4 EXPLOIT=1 SAVE=../results/pt_priv2_cv_s$sd.npz python pt_game.py 12 > ../results/pt_priv2_cv_s$sd.log 2>&1 &
done
wait
INFO=full LAM=2 CV=1 M_PATHS=16384 STEPS=1500 BATCH=2048 AVG_FROM=4 SAVE=../results/pt_full2_cv.npz python pt_game.py 10 > ../results/pt_full2_cv.log 2>&1 &
# finer grid for Firoozi-Caines with the control variate
Q0=3 A0=-1 NOISE=asym SV2=0.05 CV=1 N_STEPS=200 M_PATHS=16384 STEPS=1500 BATCH=2048 RHO_DAMP=0.5 AVG_FROM=4 SAVE=../results/fc_cv_N200.npz python fc_game.py 10 > ../results/fc_cv_N200.log 2>&1 &
wait
