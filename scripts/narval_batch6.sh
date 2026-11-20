#!/bin/bash
# Sixth batch (GPU): larger budgets for the Gaussian games with exact references, to see where the error levels off, and a clean
# test of the control variate (same settings with CV=0 and CV=1). STEPS is scaled with the number of paths so that every run sees
# about 90 passes over its paths per iteration. Validation controls are averaged from iteration AVG_FROM on.
# Submit from scripts/: sbatch narval_batch6.sh
#SBATCH --account=def-ankush
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=pig-batch6gpu
#SBATCH --output=logs/%x_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate
export THREADS=1; export DEVICE=cuda
export RESULTS_DATA="$SLURM_SUBMIT_DIR/../results"
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
# (1) two traders, lam = 0, exact reference: control variate on/off at 32k, and 128k with it on
for cv in 0 1; do
  LAM=0 CV=$cv M_PATHS=32768 STEPS=1500 BATCH=2048 AVG_FROM=6 SAVE=../results/b6_pt_lam0_32k_cv$cv.npz python pt_game.py 16 > ../results/b6_pt_lam0_32k_cv$cv.log 2>&1 &
done
LAM=0 CV=1 M_PATHS=131072 STEPS=6000 BATCH=2048 AVG_FROM=6 SAVE=../results/b6_pt_lam0_128k_cv1.npz python pt_game.py 16 > ../results/b6_pt_lam0_128k_cv1.log 2>&1 &
# (2) two traders, private, lam = 1, exact open-loop reference (exact-class solver): 32k and 128k
LAM=1 CV=1 M_PATHS=32768 STEPS=1500 BATCH=2048 AVG_FROM=6 REF_NPZ=../references/pt_private_lam1.0.npz SAVE=../results/b6_pt_lam1_32k_cv1.npz python pt_game.py 16 > ../results/b6_pt_lam1_32k_cv1.log 2>&1 &
LAM=1 CV=1 M_PATHS=131072 STEPS=6000 BATCH=2048 AVG_FROM=6 REF_NPZ=../references/pt_private_lam1.0.npz SAVE=../results/b6_pt_lam1_128k_cv1.npz python pt_game.py 16 > ../results/b6_pt_lam1_128k_cv1.log 2>&1 &
# (3) Firoozi-Caines major-minor game, exact reference: control variate on/off at 32k, and 128k with it on
for cv in 0 1; do
  CV=$cv M_PATHS=32768 STEPS=1500 BATCH=2048 AVG_FROM=6 SAVE=../results/b6_fc_32k_cv$cv.npz python fc_game.py 16 > ../results/b6_fc_32k_cv$cv.log 2>&1 &
done
CV=1 M_PATHS=131072 STEPS=6000 BATCH=2048 AVG_FROM=6 SAVE=../results/b6_fc_128k_cv1.npz python fc_game.py 16 > ../results/b6_fc_128k_cv1.log 2>&1 &
wait
echo "batch6 done"
