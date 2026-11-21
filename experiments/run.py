"""
One run of the study: build the game, compute the open-loop equilibrium by the damped Picard iteration, compare it
with the reference, save the result.

The reference of each game:
    two traders, λ = 0           exact solution (TwoTraders.exact)
    two traders, λ > 0           exact equilibrium from references/ (LinearStrategies)
    Markov chain, λ = 0          exact solution (MarkovTraders.exact)
    Markov chain, λ > 0          none; the first order condition residual is the check (--residual)
    broker-trader, major-minor     exact solution of the discretised game

Writes results/<name>.json (all settings, errors, timing) and results/<name>.npz (the validation controls).
A run whose JSON file exists is skipped.

Usage: run.py --game two_traders --λ 1.0 --paths 50000 [--seed 0] [...]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from partial_info_games import Solver, first_order_residual, relative_l2, use_device
from partial_info_games.games import TwoTraders, MarkovTraders, BrokerTrader, MajorMinor

DAMPING = dict(two_traders=0.6, markov_traders=0.5, broker_trader=0.7, major_minor=0.7)


def name(config: dict) -> str:
    """File name of a run, from the settings given on the command line."""
    return '_'.join(str(value) if key == 'game' else f'{key}{value}' for key, value in config.items())


def settings(config: dict) -> dict:
    """All settings of a run: the command-line settings with the defaults filled in."""
    run = dict(lam=None, seed=0, width=32, damping=DAMPING[config['game']], batch=2000,
               steps=3 * config['paths'] // 50,                # 120 passes over the paths per iteration
               control_variate=True, residual=False, iterations=16)
    run.update(config)
    run['average_from'] = 6 if run['iterations'] <= 16 else run['iterations'] // 2
    return run


def build_game(run: dict):
    """The game of a run."""
    if run['game'] == 'two_traders':
        return TwoTraders(lam=run['lam'])
    if run['game'] == 'markov_traders':
        return MarkovTraders(lam=run['lam'])
    if run['game'] == 'broker_trader':
        return BrokerTrader()
    return MajorMinor()


def reference(run: dict, game, validation: dict):
    """Reference controls on the validation paths, one [M, N] tensor per player, or None (see the module docstring)."""
    try:
        controls = game.exact(validation)
    except NotImplementedError:
        if run['game'] != 'two_traders':
            return None
        controls = np.load(HERE / 'references' / f"two_traders_lam{run['lam']}.npz")['controls']
    return [torch.tensor(controls[:, i], dtype=torch.float32) for i in range(controls.shape[1])]


def main(config: dict):
    path = HERE / 'results' / name(config)
    json_file = path.parent / f'{path.name}.json'
    npz_file = path.parent / f'{path.name}.npz'
    if json_file.exists():
        print('exists:', json_file.name)
        return
    device = use_device()
    print('device', device, '|', path.name, flush=True)

    run = settings(config)
    game = build_game(run)
    start = time.time()
    solver = Solver(game, width=run['width'], damping=run['damping'], steps=run['steps'], batch=run['batch'],
                    n_paths=run['paths'], control_variate=run['control_variate'], seed=run['seed'])
    target = reference(run, game, solver.validation)
    controls, history = solver.run(run['iterations'], run['average_from'], reference=target)

    result = dict(settings=run, seconds=time.time() - start, history=history,
                  mean_abs_control=[float(c.abs().mean()) for c in controls])
    if target is not None:
        result['error'] = [relative_l2(c, r) for c, r in zip(controls, target)]
    if run['residual']:
        result['first_order_residual'] = [first_order_residual(solver, i) for i in range(game.n_players)]
    print(json.dumps({key: value for key, value in result.items() if key != 'history'}), flush=True)

    path.parent.mkdir(exist_ok=True)
    np.savez(npz_file, controls=torch.stack(controls, 1).cpu().numpy())
    json_file.write_text(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', required=True, choices=list(DAMPING))
    parser.add_argument('--lam', type=float, help='price impact (two-trader and Markov-chain games)')
    parser.add_argument('--paths', type=int, default=50000, help='training paths (default 50000); Adam steps scale with paths')
    parser.add_argument('--seed', type=int, default=0, help='seed of the networks and training paths (default 0)')
    parser.add_argument('--damping', type=float, help='Picard damping (default per game)')
    parser.add_argument('--iterations', type=int, help='Picard iterations (default 16)')
    parser.add_argument('--no-control-variate', dest='control_variate', action='store_false', default=None)
    parser.add_argument('--residual', action='store_true', default=None, help='compute the first order condition residuals')
    arguments = {key: value for key, value in vars(parser.parse_args()).items() if value is not None}
    if arguments['game'] in ('two_traders', 'markov_traders') and 'lam' not in arguments:
        parser.error('--lam is required for the two-trader and Markov-chain games')
    main(arguments)
