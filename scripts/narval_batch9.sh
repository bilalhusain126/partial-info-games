#!/bin/bash
# Ninth batch (GPU): the 64k middle point of every error-versus-paths curve in batches 6 and 7 (control variate on, settings as there),
# so that each curve has three budgets (32k, 64k, 128k). Submit from scripts/: sbatch narval_batch9.sh
#SBATCH --account=def-ankush
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=pig-batch9gpu
#SBATCH --output=logs/%x_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate
export THREADS=1; export DEVICE=cuda
export RESULTS_DATA="$SLURM_SUBMIT_DIR/../results"
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
LAM=0 CV=1 M_PATHS=65536 STEPS=3000 BATCH=2048 AVG_FROM=6 SAVE=../results/b9_pt_lam0_64k_cv1.npz python pt_game.py 16 > ../results/b9_pt_lam0_64k_cv1.log 2>&1 &
LAM=1 CV=1 M_PATHS=65536 STEPS=3000 BATCH=2048 AVG_FROM=6 REF_NPZ=../references/pt_private_lam1.0.npz SAVE=../results/b9_pt_lam1_64k_cv1.npz python pt_game.py 16 > ../results/b9_pt_lam1_64k_cv1.log 2>&1 &
CV=1 M_PATHS=65536 STEPS=3000 BATCH=2048 AVG_FROM=6 SAVE=../results/b9_fc_64k_cv1.npz python fc_game.py 16 > ../results/b9_fc_64k_cv1.log 2>&1 &
H=0 CV=1 M_PATHS=65536 STEPS=3000 BATCH=2048 AVG_FROM=6 ITERS=16 SAVE=../results/b9_wj_h0_64k_cv1.npz python npg_wj.py > ../results/b9_wj_h0_64k_cv1.log 2>&1 &
LAM=0 CV=1 CV_JUMP=1 RHO_DAMP=0.5 M_PATHS=65536 STEPS=3000 BATCH=2048 AVG_FROM=6 SAVE=../results/b9_pm_lam0_64k_jump.npz python pm_game.py 16 > ../results/b9_pm_lam0_64k_jump.log 2>&1 &
wait
echo "batch9 done"
