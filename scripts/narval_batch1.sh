#!/bin/bash
# First batch on Narval (CPU only). Submit from scripts/ with: sbatch narval_batch1.sh
# 14 runs in parallel, 2 torch threads each. Every run writes its own log and npz in results/.
#SBATCH --account=def-ankush
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --job-name=pig-batch1
#SBATCH --output=logs/%x_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate                    # environment from latent-factors-control/experiments/cluster/setup.sh
export THREADS=2
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
# (1) budget scaling at lam=0 (exact reference), control variate on, 128 epochs per iteration at every budget, FOC check for calibration
for m in 2048 4096 8192 16384 32768; do
  b=$(( m/4 > 2048 ? 2048 : m/4 )); st=$(( 128*m/b ))
  LAM=0 CV=1 RHO_DAMP=0.5 M_PATHS=$m STEPS=$st BATCH=$b AVG_FROM=4 FOC_CHECK=1 SAVE=../results/pt_scale_$m.npz python pt_game.py 10 > ../results/pt_scale_$m.log 2>&1 &
done
# (2) private two-way case, lam=2: three seeds, FOC check and exploitability
for sd in 1 2 3; do
  LAM=2 CV=1 SEED=$sd M_PATHS=16384 STEPS=1500 BATCH=2048 AVG_FROM=4 FOC_CHECK=1 EXPLOIT=1 EXPLOIT_ITERS=4 SAVE=../results/pt_priv2_cv_s$sd.npz python pt_game.py 12 > ../results/pt_priv2_cv_s$sd.log 2>&1 &
done
# (3) full-information lam=2 at the same budget (exact reference), for the private-vs-full comparison
INFO=full LAM=2 CV=1 M_PATHS=16384 STEPS=1500 BATCH=2048 AVG_FROM=4 SAVE=../results/pt_full2_cv.npz python pt_game.py 10 > ../results/pt_full2_cv.log 2>&1 &
# (4) Firoozi-Caines on a grid twice as fine, control variate on (discrete-time exact reference at any grid)
Q0=3 A0=-1 NOISE=asym SV2=0.05 CV=1 N_STEPS=200 M_PATHS=16384 STEPS=1500 BATCH=2048 RHO_DAMP=0.5 AVG_FROM=4 SAVE=../results/fc_cv_N200.npz python fc_game.py 10 > ../results/fc_cv_N200.log 2>&1 &
# (5) continuity in lam: private case at small impact, compared offline with the exact lam=0 solution (gap should be O(lam))
for lam in 0.25 0.5; do
  LAM=$lam CV=1 M_PATHS=16384 STEPS=1500 BATCH=2048 AVG_FROM=4 FOC_CHECK=1 SAVE=../results/pt_priv_lam$lam.npz python pt_game.py 12 > ../results/pt_priv_lam$lam.log 2>&1 &
done
# (6) Wu-Jaimungal h=0 on the engine at a proper budget, control variate on (exact reference)
H=0 T=0.3 CV=1 STEPS=1500 M_PATHS=16384 ITERS=12 AVG_FROM=4 SAVE=../results/wj_h0_cv.npz python npg_wj.py > ../results/wj_h0_cv.log 2>&1 &
wait
echo "batch1 done"
