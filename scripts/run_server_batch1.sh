#!/bin/zsh
# First server batch: budget scaling (lam=0, exact reference), seeds of the private case, finer grid, FOC-residual check.
cd "$(dirname "$0")/../games"
for m in 2048 4096 8192 16384 32768; do
  LAM=0 CV=1 RHO_DAMP=0.5 M_PATHS=$m STEPS=$((m/8 > 2000 ? 2000 : m/8)) BATCH=$((m/4 > 2048 ? 2048 : m/4)) AVG_FROM=4 SAVE=../results/pt_scale_$m.npz python3 pt_game.py 10 > ../results/pt_scale_$m.log 2>&1 &
done
wait
for sd in 1 2 3; do
  LAM=2 CV=1 SEED=$sd M_PATHS=16384 STEPS=1500 BATCH=2048 AVG_FROM=4 EXPLOIT=1 SAVE=../results/pt_priv2_cv_s$sd.npz python3 pt_game.py 12 > ../results/pt_priv2_cv_s$sd.log 2>&1 &
done
INFO=full LAM=2 CV=1 M_PATHS=16384 STEPS=1500 BATCH=2048 AVG_FROM=4 SAVE=../results/pt_full2_cv.npz python3 pt_game.py 10 > ../results/pt_full2_cv.log 2>&1 &
Q0=3 A0=-1 NOISE=asym SV2=0.05 CV=1 N_STEPS=200 M_PATHS=16384 STEPS=1500 BATCH=2048 RHO_DAMP=0.5 AVG_FROM=4 SAVE=../results/fc_cv_N200.npz python3 fc_game.py 10 > ../results/fc_cv_N200.log 2>&1 &
wait
