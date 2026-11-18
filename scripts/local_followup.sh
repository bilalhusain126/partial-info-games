#!/bin/zsh
# Reviewer follow-ups, laptop, two runs at a time: (a) three seeds at lam=0, 8k paths (bias floor vs variance); (b) full-info lam=2 with FOC check.
cd "$(dirname "$0")/../games"; export THREADS=3
(LAM=0 CV=1 RHO_DAMP=0.5 SEED=1 M_PATHS=8192 STEPS=512 BATCH=2048 AVG_FROM=4 SAVE=../results/pt_lam0_8k_s1.npz python3 pt_game.py 10 > ../results/pt_lam0_8k_s1.log 2>&1;
 LAM=0 CV=1 RHO_DAMP=0.5 SEED=3 M_PATHS=8192 STEPS=512 BATCH=2048 AVG_FROM=4 SAVE=../results/pt_lam0_8k_s3.npz python3 pt_game.py 10 > ../results/pt_lam0_8k_s3.log 2>&1) &
(LAM=0 CV=1 RHO_DAMP=0.5 SEED=2 M_PATHS=8192 STEPS=512 BATCH=2048 AVG_FROM=4 SAVE=../results/pt_lam0_8k_s2.npz python3 pt_game.py 10 > ../results/pt_lam0_8k_s2.log 2>&1;
 INFO=full LAM=2 CV=1 M_PATHS=8192 STEPS=1000 BATCH=1024 AVG_FROM=4 FOC_CHECK=1 SAVE=../results/pt_full2_cv_foc.npz python3 pt_game.py 10 > ../results/pt_full2_cv_foc.log 2>&1) &
wait
