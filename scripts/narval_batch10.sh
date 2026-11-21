#!/bin/bash
# Tenth batch (GPU): closed-loop Picard (CLOSED=1: targets from the derivative of the realised cost with the other player's network
# inside the simulation) with stronger damping, against the exact closed-loop solution. With damping 0.6 this diverged at lam = 2;
# small damping makes each iteration a small step, as in gradient play, while keeping the regression. Two traders, private, 32k paths.
# Submit from scripts/: sbatch narval_batch10.sh
#SBATCH --account=def-ankush
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=pig-batch10gpu
#SBATCH --output=logs/%x_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate
export THREADS=1; export DEVICE=cuda
export RESULTS_DATA="$SLURM_SUBMIT_DIR/../results"
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
COMMON="CLOSED=1 CV=1 M_PATHS=32768 STEPS=1500 BATCH=2048"
# lam = 1.5: damping 0.6 (as before), 0.2 and 0.1; more iterations for smaller damping
env $COMMON LAM=1.5 RHO_DAMP=0.6 AVG_FROM=8 REF_NPZ=../references/pt_private_lam1.5.npz SAVE=../results/b10_cpicard_lam1.5_rho0.6.npz python pt_game.py 16 > ../results/b10_cpicard_lam1.5_rho0.6.log 2>&1 &
env $COMMON LAM=1.5 RHO_DAMP=0.2 AVG_FROM=20 REF_NPZ=../references/pt_private_lam1.5.npz SAVE=../results/b10_cpicard_lam1.5_rho0.2.npz python pt_game.py 40 > ../results/b10_cpicard_lam1.5_rho0.2.log 2>&1 &
env $COMMON LAM=1.5 RHO_DAMP=0.1 AVG_FROM=30 REF_NPZ=../references/pt_private_lam1.5.npz SAVE=../results/b10_cpicard_lam1.5_rho0.1.npz python pt_game.py 60 > ../results/b10_cpicard_lam1.5_rho0.1.log 2>&1 &
# lam = 2 (diverged with damping 0.6): damping 0.2 and 0.1
env $COMMON LAM=2 RHO_DAMP=0.2 AVG_FROM=20 REF_NPZ=../references/pt_private_lam2.0.npz SAVE=../results/b10_cpicard_lam2_rho0.2.npz python pt_game.py 40 > ../results/b10_cpicard_lam2_rho0.2.log 2>&1 &
env $COMMON LAM=2 RHO_DAMP=0.1 AVG_FROM=30 REF_NPZ=../references/pt_private_lam2.0.npz SAVE=../results/b10_cpicard_lam2_rho0.1.npz python pt_game.py 60 > ../results/b10_cpicard_lam2_rho0.1.log 2>&1 &
wait
echo "batch10 done"
