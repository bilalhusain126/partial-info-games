"""
One run of the study: build the game, compute the equilibrium, compare it with the reference, save the result.

The reference of each game:
    two traders, lam = 0           exact solution (TwoTraders.exact)
    two traders, lam > 0           exact linear-class equilibrium from references/ (open or closed loop,
                                   matching the method)
    Markov chain, lam = 0          exact solution (MarkovTraders.exact)
    Markov chain, lam > 0          none
    broker-trader, major-minor     exact solution of the discretised game

Methods:
    picard               damped Picard iteration, open loop (the method of the note)
    closed_picard        damped Picard iteration with the closed-loop targets
    gradient_play        closed-loop gradient play, started from 8 Picard iterations
    gradient_play_open   open-loop gradient play, started from 8 Picard iterations
    gradient_play_cold   closed-loop gradient play from untrained networks
    linear               the Picard iteration among controls linear in the observed path

Writes results/<name>.json (all settings, errors, timing) and results/<name>.npz (the validation controls).
A run whose JSON file exists is skipped.

Usage: run.py --game two_traders --lam 1.0 --paths 65536 [--seed 0] [--method picard] [...]
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
from partial_info_games import (Solver, gradient_play, linear_equilibrium, first_order_residual, relative_l2,
                                use_device)
from partial_info_games.games import TwoTraders, MarkovTraders, BrokerTrader, MajorMinor

DAMPING = dict(two_traders=0.6, markov_traders=0.5, broker_trader=0.7, major_minor=0.7)
ITERATIONS = dict(picard=16, closed_picard=16, gradient_play=30, gradient_play_open=30, gradient_play_cold=60, linear=30)
CLOSED_LOOP = ('closed_picard', 'gradient_play', 'gradient_play_cold')


def name(config: dict) -> str:
    """File name of a run, from the settings given on the command line."""
    return '_'.join(str(value) if key == 'game' else f'{key}{value}' for key, value in config.items())


def settings(config: dict) -> dict:
    """All settings of a run: the command-line settings with the defaults filled in."""
    method = config.get('method', 'picard')
    run = dict(lam=None, seed=0, method=method, width=32, damping=DAMPING[config['game']], batch=2048,
               steps=1500 * config['paths'] // 32768,          # about 90 passes over the paths per iteration
               control_variate=True, jump_variate=True, residual=False,
               iterations=ITERATIONS[method], lr=1e-4 if method.startswith('gradient_play') else None)
    run.update(config)
    short_picard = method in ('picard', 'closed_picard') and run['iterations'] <= 16
    run['average_from'] = 6 if short_picard else run['iterations'] // 2
    return run


def build_game(run: dict):
    """The game of a run."""
    if run['game'] == 'two_traders':
        return TwoTraders(lam=run['lam'])
    if run['game'] == 'markov_traders':
        return MarkovTraders(lam=run['lam'], jump_variate=run['jump_variate'])
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
        loop = 'closed' if run['method'] in CLOSED_LOOP else 'open'
        controls = np.load(HERE / 'references' / f"two_traders_lam{run['lam']}.npz")[loop]
    return [torch.tensor(controls[:, i], dtype=torch.float32) for i in range(controls.shape[1])]


def solve(run: dict, solver: Solver, target):
    """Computes the equilibrium with the method of the run. Returns the validation controls and the error history."""
    method = run['method']
    if method == 'linear':
        return linear_equilibrium(solver, run['iterations'], run['average_from'], reference=target), []
    if method in ('gradient_play', 'gradient_play_open'):
        solver.run(8, average_from=4)
        return gradient_play(solver, run['iterations'], lr=run['lr'], closed=method == 'gradient_play', reference=target)
    if method == 'gradient_play_cold':
        return gradient_play(solver, run['iterations'], lr=run['lr'], reference=target)
    return solver.run(run['iterations'], run['average_from'], reference=target, closed=method == 'closed_picard')


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
    controls, history = solve(run, solver, target)

    result = dict(settings=run, seconds=time.time() - start, history=history,
                  mean_abs_control=[float(c.abs().mean()) for c in controls])
    if target is not None:
        result['error'] = [relative_l2(c, r) for c, r in zip(controls, target)]
        result['slope'] = [float(np.polyfit(r.cpu().numpy().ravel(), c.cpu().numpy().ravel(), 1)[0])
                           for c, r in zip(controls, target)]
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
    parser.add_argument('--paths', type=int, default=65536, help='training paths (default 65536); Adam steps scale with paths')
    parser.add_argument('--seed', type=int, default=0, help='seed of the networks and training paths (default 0)')
    parser.add_argument('--method', default='picard', choices=list(ITERATIONS), help='default picard')
    parser.add_argument('--damping', type=float, help='Picard damping (default per game)')
    parser.add_argument('--iterations', type=int, help='Picard or gradient-play iterations (default per method)')
    parser.add_argument('--lr', type=float, help='gradient-play learning rate (default 1e-4)')
    parser.add_argument('--width', type=int, help='GRU width (default 32)')
    parser.add_argument('--no-control-variate', dest='control_variate', action='store_false', default=None)
    parser.add_argument('--no-jump-variate', dest='jump_variate', action='store_false', default=None,
                        help='Markov-chain game: control variate on the price noise only')
    parser.add_argument('--residual', action='store_true', default=None, help='compute the first order condition residuals')
    arguments = {key: value for key, value in vars(parser.parse_args()).items() if value is not None}
    if arguments['game'] in ('two_traders', 'markov_traders') and 'lam' not in arguments:
        parser.error('--lam is required for the two-trader and Markov-chain games')
    main(arguments)
