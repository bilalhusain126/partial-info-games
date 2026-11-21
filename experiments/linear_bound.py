"""
Best fit of the exact Markov-chain controls at lam = 0 by strategies linear in the own observed path.

At each t_j, each trader's exact control is regressed on

    (1, alpha^i_{t_j}, q^i_{t_j}, S_{t_j}, all increments of alpha^i and S before t_j)

on half of the paths, and the relative L2 error of the fit is measured on the other half. These features
span every linear function of the observed path, so no strategy linear in the observed path is closer to
the exact control than this error, up to sampling error.

Usage: python linear_bound.py [n_paths]   (default 65536)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from partial_info_games import use_device
from partial_info_games.games import MarkovTraders

use_device('cpu')
game = MarkovTraders(lam=0.0)
n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
N, dt = game.N, game.dt
half = n_paths // 2

noise = game.sample(n_paths, 99)
exact = game.exact(noise)                                       # [M, 2, N]
alpha = np.stack([noise['alpha1'].numpy(), noise['alpha2'].numpy()], 1).astype(np.float64)
price_drift = alpha[:, 0, :-1] + alpha[:, 1, :-1]               # lam = 0: the price does not depend on the trades
S = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(price_drift * dt + game.sigma * noise['dWS'].numpy(), 1)], 1)

for i in (0, 1):
    q = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(exact[:, i] * dt, 1)], 1)
    fit = np.zeros((n_paths, N))
    for j in range(N):
        columns = [np.ones(n_paths), alpha[:, i, j], q[:, j], S[:, j]]
        columns += [alpha[:, i, k + 1] - alpha[:, i, k] for k in range(j)]
        columns += [S[:, k + 1] - S[:, k] for k in range(j)]
        features = np.stack(columns, 1)
        coefficients = np.linalg.lstsq(features[:half], exact[:half, i, j], rcond=1e-10)[0]
        fit[:, j] = features @ coefficients
    error = np.sqrt(((fit[half:] - exact[half:, i]) ** 2).mean() / (exact[half:, i] ** 2).mean())
    print(f'trader {i + 1}: best linear fit of the exact control, out of sample: relative L2 error {error:.3f}')
