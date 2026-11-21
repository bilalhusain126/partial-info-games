"""
Exact equilibria of the two-trader game at lam = 1, 1.5 and 2, the references of the learned method at lam > 0.

For each lam, the open-loop equilibrium among strategies linear in the own observed path is computed by
games.two_traders.LinearStrategies, and written to references/two_traders_lam<lam>.npz with the controls on the
validation paths ('controls', [M, 2, N]) and the coefficients ('beta'). Each solve starts from the solution at
the previous lam, and raises if it does not converge.

Usage: python references.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from partial_info_games import use_device, validation_paths
from partial_info_games.games import TwoTraders, LinearStrategies

use_device('cpu')
(HERE / 'references').mkdir(exist_ok=True)
beta = None
for lam in (1.0, 1.5, 2.0):
    game = TwoTraders(lam=lam)
    solver = LinearStrategies(game)
    beta = solver.solve(start=beta)
    controls = solver.controls(beta, validation_paths(game)).astype(np.float32)
    np.savez(HERE / 'references' / f'two_traders_lam{lam}.npz', controls=controls, beta=beta.numpy())
    print(f'lam {lam}: mean |nu| {np.abs(controls).mean():.4f}', flush=True)
