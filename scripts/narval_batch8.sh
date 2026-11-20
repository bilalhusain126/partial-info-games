#!/bin/bash
# Eighth batch (GPU): closed-loop Nash equilibrium of the private two-trader game by gradient play (npg.run_direct). Each player
# takes Adam steps on its own realised cost, backpropagated through the simulated game with the other player's network inside, so
# the gradient includes the other player's reaction through the price. Warm start: the open-loop Picard solution (8 iterations).
# The open-loop variant of the same scheme (other player's controls detached) is the control experiment. Exact closed-loop and
# open-loop references from the exact-class solver are in references/ (they differ by 8.5% at lam = 1.5 and by 149% at lam = 2).
# A local test at 4k paths and lam = 1.5 moved from 12% to 7% of the closed-loop solution (closed) and stayed at 7% of the open-loop one (open).
# Submit from scripts/: sbatch narval_batch8.sh
#SBATCH --account=def-ankush
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=pig-batch8gpu
#SBATCH --output=logs/%x_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate
export THREADS=1; export DEVICE=cuda
export RESULTS_DATA="$SLURM_SUBMIT_DIR/../results"
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
COMMON="CV=1 M_PATHS=32768 STEPS=1500 BATCH=2048 AVG_FROM=4 DIRECT_ITERS=30 DIRECT_STEPS=200 DIRECT_LR=1e-4"
# lam = 1.5: closed loop, three seeds; open loop, one seed
for sd in 1 2 3; do
  env $COMMON LAM=1.5 SEED=$sd DIRECT=closed REF_NPZ=../references/pt_private_lam1.5.npz SAVE=../results/b8_closed_lam1.5_s$sd.npz python pt_game.py 8 > ../results/b8_closed_lam1.5_s$sd.log 2>&1 &
done
env $COMMON LAM=1.5 SEED=1 DIRECT=open REF_NPZ=../references/pt_private_lam1.5.npz SAVE=../results/b8_open_lam1.5_s1.npz python pt_game.py 8 > ../results/b8_open_lam1.5_s1.log 2>&1 &
# lam = 2 (edge of convexity; the exact closed-loop iteration needed 96 damped steps): closed loop at two learning rates
for lr in 1e-4 3e-5; do
  env $COMMON LAM=2 SEED=1 DIRECT=closed DIRECT_LR=$lr DIRECT_ITERS=40 REF_NPZ=../references/pt_private_lam2.0.npz SAVE=../results/b8_closed_lam2_lr$lr.npz python pt_game.py 8 > ../results/b8_closed_lam2_lr$lr.log 2>&1 &
done
wait
echo "batch8 done"
