#!/bin/bash
# Second batch on Narval (CPU only): breadth on the other examples. Submit from scripts/ with: sbatch narval_batch2.sh
# 9 runs in parallel, 2 torch threads each.
#SBATCH --account=def-ankush
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --job-name=pig-batch2
#SBATCH --output=logs/%x_%j.out
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/lfc/bin/activate
export THREADS=2
cd "$SLURM_SUBMIT_DIR/../games"; mkdir -p ../results
# (1) Wu-Jaimungal, strong coupling (a = b = 0.1, h = 1): partial information (GRU broker), full information (vs linear Picard),
#     and h = 0 at the same costs (vs exact). Tests the iteration beyond the small-T contraction regime.
A_COST=0.1 B_COST=0.1 BROKER=gru  M_PATHS=16384 FIT_STEPS=1500 BATCH=2048 SAVE=../results/wj_strong_h1_gru.npz  python wj_game.py 1 0.3 15 > ../results/wj_strong_h1_gru.log 2>&1 &
A_COST=0.1 B_COST=0.1 BROKER=full M_PATHS=16384 FIT_STEPS=1500 BATCH=2048 SAVE=../results/wj_strong_h1_full.npz python wj_game.py 1 0.3 15 > ../results/wj_strong_h1_full.log 2>&1 &
A_COST=0.1 B_COST=0.1 BROKER=gru  M_PATHS=16384 FIT_STEPS=1500 BATCH=2048 SAVE=../results/wj_strong_h0_gru.npz  python wj_game.py 0 0.3 15 > ../results/wj_strong_h0_gru.log 2>&1 &
# (2) Wu-Jaimungal, longer horizon T = 1 at h = 1 (default costs), partial and full information
BROKER=gru  M_PATHS=16384 FIT_STEPS=1500 BATCH=2048 SAVE=../results/wj_T1_h1_gru.npz  python wj_game.py 1 1.0 15 > ../results/wj_T1_h1_gru.log 2>&1 &
BROKER=full M_PATHS=16384 FIT_STEPS=1500 BATCH=2048 SAVE=../results/wj_T1_h1_full.npz python wj_game.py 1 1.0 15 > ../results/wj_T1_h1_full.log 2>&1 &
# (3) Bergault-Sanchez-Betancourt: gentle parameters at a larger population and budget; the paper's own (stiff) parameters on a fine grid
PARAMS=gentle M_PATHS=8192 M_POP=128 FIT_STEPS=2000 BATCH=2048 AVG_FROM=6 SAVE=../results/bsb_gentle_pop128.npz python bsb_game.py 20 > ../results/bsb_gentle_pop128.log 2>&1 &
PARAMS=paper  N_STEPS=1000 M_PATHS=2048 M_POP=32 FIT_STEPS=2000 BATCH=512 AVG_FROM=6 SAVE=../results/bsb_paper_N1000.npz python bsb_game.py 20 > ../results/bsb_paper_N1000.log 2>&1 &
# (4) Firoozi-Caines at two other noise levels (exact discrete reference), control variate on
Q0=3 A0=-1 NOISE=high SV2=0.05 CV=1 M_PATHS=16384 STEPS=1500 BATCH=2048 RHO_DAMP=0.5 AVG_FROM=4 FOC_CHECK=1 EXPLOIT=1 EXPLOIT_ITERS=4 SAVE=../results/fc_high.npz python fc_game.py 10 > ../results/fc_high.log 2>&1 &
Q0=10 A0=-1 NOISE=asym SV2=0.3 CV=1 M_PATHS=16384 STEPS=1500 BATCH=2048 RHO_DAMP=0.5 AVG_FROM=4 FOC_CHECK=1 SAVE=../results/fc_q10.npz python fc_game.py 10 > ../results/fc_q10.log 2>&1 &
wait
echo "batch2 done"
