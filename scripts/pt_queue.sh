#!/bin/zsh
cd "$(dirname "$0")/../games"
# (wait removed)
LAM=0 M_PATHS=8192 STEPS=1000 BATCH=1024 AVG_FROM=4 SAVE=wj/pt_lam0.npz python3 pt_game.py 10 > ../results/pt_lam0.log 2>&1
INFO=full LAM=2 M_PATHS=8192 STEPS=1000 BATCH=1024 AVG_FROM=4 SAVE=wj/pt_full2.npz python3 pt_game.py 10 > ../results/pt_full2.log 2>&1
LAM=2 CV=1 M_PATHS=8192 STEPS=1000 BATCH=1024 AVG_FROM=4 EXPLOIT=1 EXPLOIT_ITERS=6 SAVE=wj/pt_priv2_cv.npz python3 pt_game.py 12 > ../results/pt_priv2_cv.log 2>&1
LAM=2 CV=0 M_PATHS=8192 STEPS=1000 BATCH=1024 AVG_FROM=4 EXPLOIT=1 EXPLOIT_ITERS=6 SAVE=wj/pt_priv2.npz python3 pt_game.py 12 > ../results/pt_priv2.log 2>&1
