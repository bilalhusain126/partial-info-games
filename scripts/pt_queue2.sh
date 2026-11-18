#!/bin/zsh
cd "$(dirname "$0")/../games"
until ! pgrep -f "pt_queue.sh" >/dev/null; do sleep 60; done
# chain A: budget scaling at lam=0 (exact reference), control variate on
(
for cfg in "2048:500:512" "4096:1000:1024" "8192:1500:2048" "16384:2000:2048"; do
  IFS=: read m st b <<< "$cfg"
  LAM=0 CV=1 RHO_DAMP=0.5 M_PATHS=$m STEPS=$st BATCH=$b AVG_FROM=4 SAVE=wj/pt_scale_$m.npz python3 pt_game.py 10 > ../results/pt_scale_$m.log 2>&1
done
) &
# chain B: private lam=2 with CV, seeds 1 and 2, then full-info lam=2 saved for the economic comparison
(
for sd in 1 2; do
  LAM=2 CV=1 SEED=$sd M_PATHS=8192 STEPS=1000 BATCH=1024 AVG_FROM=4 SAVE=wj/pt_priv2_cv_s$sd.npz python3 pt_game.py 12 > ../results/pt_priv2_cv_s$sd.log 2>&1
done
INFO=full LAM=2 CV=1 M_PATHS=8192 STEPS=1000 BATCH=1024 AVG_FROM=4 SAVE=wj/pt_full2_cv.npz python3 pt_game.py 10 > ../results/pt_full2_cv.log 2>&1
) &
wait
