#!/bin/zsh
# Kernel benchmark, private information, continuation in lam from the exact lam=0 solution.
cd "$(dirname "$0")/../games"
LAM=0 ITERS=2 RHO_DAMP=1 INNER=300 python3 pt_kernel.py > ../results/pt_kernel_private_lam0.log 2>&1
LAM=0.25 ITERS=100 RHO_DAMP=0.5 INNER=300 INIT=../results/pt_kernel_private_lam0.0.npz python3 pt_kernel.py > ../results/pt_kernel_private_lam0.25.log 2>&1
LAM=0.5  ITERS=100 RHO_DAMP=0.5 INNER=300 INIT=../results/pt_kernel_private_lam0.25.npz python3 pt_kernel.py > ../results/pt_kernel_private_lam0.5.log 2>&1
LAM=1    ITERS=100 RHO_DAMP=0.5 INNER=300 INIT=../results/pt_kernel_private_lam0.5.npz python3 pt_kernel.py > ../results/pt_kernel_private_lam1.0.log 2>&1
LAM=2    ITERS=150 RHO_DAMP=0.5 INNER=300 INIT=../results/pt_kernel_private_lam1.0.npz python3 pt_kernel.py > ../results/pt_kernel_private_lam2.0.log 2>&1
echo chain done >> ../results/pt_kernel_private_lam2.0.log
