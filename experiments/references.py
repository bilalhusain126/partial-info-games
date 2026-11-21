"""
Exact linear-strategy equilibria of the two-trader game, open and closed loop, at lam = 1, 1.5 and 2.

These are the references of the learned method at lam > 0 (games.two_traders.LinearStrategies). For each lam,
writes references/two_traders_lam<lam>.npz with the controls on the validation paths ('open', 'closed',
[M, 2, N]) and the coefficients ('beta_open', 'beta_closed'). Each solve starts from the solution of the same
loop at the previous lam, and raises if it does not converge.

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
start = {'open': None, 'closed': None}
for lam in (1.0, 1.5, 2.0):
    game = TwoTraders(lam=lam)
    solver = LinearStrategies(game)
    validation = validation_paths(game)
    saved = {}
    for loop in ('open', 'closed'):
        beta = solver.solve(closed=loop == 'closed', start=start[loop])
        start[loop] = beta
        saved[loop] = solver.controls(beta, validation).astype(np.float32)
        saved[f'beta_{loop}'] = beta.numpy()
        print(f'lam {lam} {loop} loop: mean |nu| {np.abs(saved[loop]).mean():.4f}', flush=True)
    np.savez(HERE / 'references' / f'two_traders_lam{lam}.npz', **saved)
