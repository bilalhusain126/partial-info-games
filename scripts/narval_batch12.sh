#!/bin/bash
# Twelfth batch (GPU): Wu-Jaimungal h = 0, the broker's error. In batches 6 and 9 the broker's per-iteration error moved between 4% and 26%
# at damping 0.7 (trader settled at 2-4%). Test: smaller damping, with more iterations and averaging over the second half.
# Submit from scripts/: sbatch narval_batch12.sh
#SBATCH --account=def-ankush
#SBATCH --time=5:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=pig-batch12gpu
#SBATCH --output=logs/%x_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate
export THREADS=1; export DEVICE=cuda
export RESULTS_DATA="$SLURM_SUBMIT_DIR/../results"
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
H=0 CV=1 RHO_DAMP=0.3 M_PATHS=32768 STEPS=1500 BATCH=2048 ITERS=30 AVG_FROM=15 SAVE=../results/b12_wj_h0_32k_rho0.3.npz python npg_wj.py > ../results/b12_wj_h0_32k_rho0.3.log 2>&1 &
H=0 CV=1 RHO_DAMP=0.15 M_PATHS=32768 STEPS=1500 BATCH=2048 ITERS=40 AVG_FROM=20 SAVE=../results/b12_wj_h0_32k_rho0.15.npz python npg_wj.py > ../results/b12_wj_h0_32k_rho0.15.log 2>&1 &
H=0 CV=1 RHO_DAMP=0.3 M_PATHS=65536 STEPS=3000 BATCH=2048 ITERS=30 AVG_FROM=15 SAVE=../results/b12_wj_h0_64k_rho0.3.npz python npg_wj.py > ../results/b12_wj_h0_64k_rho0.3.log 2>&1 &
wait
echo "batch12 done"
