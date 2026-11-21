"""
Independent checks of the exact references: each must satisfy the first order condition of the note.

At an open-loop equilibrium, u^i_t = (1 / 2 a_i) E[ Lambda^i_t | F^i_t ] for every player.
For each reference, the exact controls are fed to Solver.targets on fresh paths, which gives
Lambda^i / (2 a_i) along each path. The difference target - control is regressed at each t_j on the
player's information (1, current observation, all past observation increments). Its projection is the
first order condition residual, reported relative to the size of the control: zero up to sampling error
at an exact equilibrium, and the sampling error falls as n_paths grows (about 5-10% at 20000 paths, with
up to 2N + 4 features per time). The regression uses a richer set of features than any exact solution,
so the check does not presuppose its form.

Checked: the exact solutions of the four games, and the two-trader references at lam = 1, 1.5 and 2.

Usage: python checks.py [n_paths]   (default 20000)
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from partial_info_games import Solver, use_device
from partial_info_games.games import TwoTraders, MarkovTraders, BrokerTrader, MajorMinor, LinearStrategies

use_device('cpu')
n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 20000


def residual(game, exact_controls) -> list:
    """
    First order condition residual of each player.

    Args:
        game: The game.
        exact_controls: Function exogenous -> controls [M, n_players, N] to be checked.

    Returns:
        Per player, the relative L2 size of the projection of target - control on the player's information.
    """
    solver = Solver(game, steps=0, n_paths=8)          # used only for simulate and targets
    exogenous = game.sample(n_paths, 31)
    exact = torch.tensor(exact_controls(exogenous), dtype=torch.float32)
    controls = [exact[:, i] for i in range(game.n_players)]
    state, _, _ = solver.simulate(exogenous, fixed=dict(enumerate(controls)))
    targets = solver.targets(exogenous, controls)
    residuals = []
    for i in range(game.n_players):
        past = []
        projection = torch.zeros_like(controls[i])
        for j in range(game.N):
            features = torch.cat([torch.ones(n_paths, 1), game.observe(i, state, exogenous, j)] + past, 1).double()
            difference = (targets[i][:, j] - controls[i][:, j]).double()
            coefficients = torch.linalg.lstsq(features, difference[:, None], rcond=1e-8, driver='gelsd').solution
            projection[:, j] = (features @ coefficients)[:, 0].float()
            if game.increment_dims[i] > 0:
                past.append(game.increment(i, state, exogenous, j))
        residuals.append(float(projection.norm() / controls[i].norm()))
    return residuals


def show(label: str, residuals: list):
    print(f'{label:40s} residual ' + ' '.join(f'{r:.4f}' for r in residuals), flush=True)


for game in (TwoTraders(lam=0.0), MarkovTraders(lam=0.0), BrokerTrader(), MajorMinor()):
    show(f'{type(game).__name__}, exact', residual(game, game.exact))

for lam in (1.0, 1.5, 2.0):
    game = TwoTraders(lam=lam)
    linear_solver = LinearStrategies(game)
    beta = np.load(HERE / 'references' / f'two_traders_lam{lam}.npz')['beta']
    show(f'TwoTraders lam {lam}, reference', residual(game, lambda exogenous: linear_solver.controls(beta, exogenous)))
