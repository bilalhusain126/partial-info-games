#!/bin/zsh
# Model-free optimality check on three games after a laptop-budget solve (sequential).
cd "$(dirname "$0")/../games"; export THREADS=4 DEVICE=cpu
LAM=1 CV=1 M_PATHS=8192 STEPS=1000 BATCH=1024 AVG_FROM=4 GATEAUX=1 SAVE=../results/pt_priv1_gateaux.npz python3 pt_game.py 12 > ../results/pt_priv1_gateaux.log 2>&1
Q0=3 A0=-1 NOISE=asym SV2=0.05 CV=1 M_PATHS=8192 STEPS=1000 BATCH=1024 RHO_DAMP=0.5 AVG_FROM=4 GATEAUX=1 SAVE=../results/fc_gateaux.npz python3 fc_game.py 10 > ../results/fc_gateaux.log 2>&1
H=0 T=0.3 CV=1 STEPS=1000 M_PATHS=8192 ITERS=10 AVG_FROM=4 GATEAUX=1 SAVE=../results/wj_h0_gateaux.npz python3 npg_wj.py > ../results/wj_h0_gateaux.log 2>&1
echo "gateaux chain done" >> ../results/wj_h0_gateaux.log
