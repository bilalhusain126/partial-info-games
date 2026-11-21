"""One run: build the game, solve it, compare with the reference, and save the result.

Writes results/<name>.json (settings, errors, timing) and results/<name>.npz (the validation controls) and skips the
run if the JSON file exists. The reference is the exact solution where one is known (two traders at lam = 0, the
Markov-chain game at lam = 0, the broker-trader and major-minor games) and the exact linear-strategy equilibrium
from references/ for the two-trader game at lam > 0; the Markov-chain game at lam > 0 has none.

Usage: run.py --game two_traders --lam 1 --paths 65536 --seed 0 [--method picard] [...]   (see arguments below)
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
import pig
from pig.games import (TwoTraders, MarkovTraders, BrokerTrader, MajorMinor, two_traders_exact_lam0,
                       markov_traders_exact_lam0, broker_trader_exact, major_minor_exact, linear_equilibrium)

DAMPING = dict(two_traders=0.6, markov_traders=0.5, broker_trader=0.7, major_minor=0.7)


def name(config: dict) -> str:
    """File name of a run, from its settings."""
    return '_'.join(f'{key}{value}' if key != 'game' else str(value) for key, value in config.items())


def build_game(config: dict):
    """The game of config."""
    if config['game'] == 'two_traders':
        return TwoTraders(lam=config['lam'])
    if config['game'] == 'markov_traders':
        return MarkovTraders(lam=config['lam'], jump_variate=config.get('jump_variate', True))
    return BrokerTrader() if config['game'] == 'broker_trader' else MajorMinor()


def reference(config: dict, game, validation: dict):
    """Reference controls on the validation paths, one [M, N] tensor per player, or None."""
    lam, method = config.get('lam'), config.get('method', 'picard')
    if config['game'] == 'two_traders':
        if lam == 0:
            controls = two_traders_exact_lam0(game, validation)
        else:
            kind = 'closed' if method in ('closed_picard', 'gradient_play', 'gradient_play_cold') else 'open'
            controls = np.load(HERE / 'references' / f'two_traders_lam{lam}.npz')[kind]
    elif config['game'] == 'markov_traders':
        if lam != 0:
            return None
        controls = markov_traders_exact_lam0(game, validation)
    else:
        controls = (broker_trader_exact if config['game'] == 'broker_trader' else major_minor_exact)(game, validation)
    return [torch.tensor(controls[:, i], dtype=torch.float32) for i in range(2)]


def main(config: dict):
    path = HERE / 'results' / name(config)
    if path.with_suffix('.json').exists():
        print('exists:', path.name); return
    device = pig.use_device(); print('device', device, '|', name(config), flush=True)
    game = build_game(config); paths = config['paths']; start = time.time()
    solver = pig.Solver(game, width=config.get('width', 32), damping=config.get('damping', DAMPING[config['game']]),
                        steps=1500 * paths // 32768, batch=2048, n_paths=paths, control_variate=config.get('control_variate', True),
                        seed=config.get('seed', 0))
    target = reference(config, game, solver.validation); method = config.get('method', 'picard')
    if method == 'linear':
        controls = linear_equilibrium(solver, reference=target, log=pig.solver.report); history = []
    elif method in ('gradient_play', 'gradient_play_open'):
        solver.run(8, average_from=4, reference=None)
        controls, history = pig.gradient_play(solver, config.get('iterations', 30), lr=config.get('lr', 1e-4),
                                              closed=method == 'gradient_play', reference=target)
    elif method == 'gradient_play_cold':
        controls, history = pig.gradient_play(solver, config.get('iterations', 60), lr=config.get('lr', 1e-4), reference=target)
    else:
        iterations = config.get('iterations', 16)
        average_from = 6 if iterations <= 16 else iterations // 2
        controls, history = solver.run(iterations, average_from, reference=target, closed=method == 'closed_picard')
    result = dict(config=config, seconds=time.time() - start, history=history,
                  mean_abs_control=[float(c.abs().mean()) for c in controls])
    if target is not None:
        result['error'] = [pig.relative_l2(c, r) for c, r in zip(controls, target)]
        result['slope'] = [float(np.polyfit(r.cpu().numpy().ravel(), c.cpu().numpy().ravel(), 1)[0]) for c, r in zip(controls, target)]
    if config.get('residual'):
        result['first_order_residual'] = [pig.first_order_residual(solver, i) for i in range(game.n_players)]
    print(json.dumps({key: value for key, value in result.items() if key != 'history'}), flush=True)
    path.parent.mkdir(exist_ok=True)
    np.savez(path.with_suffix('.npz'), controls=torch.stack(controls, 1).cpu().numpy())
    path.with_suffix('.json').write_text(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', required=True, choices=list(DAMPING))
    parser.add_argument('--lam', type=float, help='price impact (two-trader and Markov-chain games)')
    parser.add_argument('--paths', type=int, default=65536, help='training paths (default 65536); Adam steps scale with paths')
    parser.add_argument('--seed', type=int, default=0, help='seed of the networks and training paths (default 0)')
    parser.add_argument('--method', default='picard', help='picard (default), closed_picard, gradient_play, gradient_play_open, '
                        'gradient_play_cold, linear')
    parser.add_argument('--damping', type=float, help='Picard damping (default per game)')
    parser.add_argument('--iterations', type=int, help='Picard (default 16) or gradient-play iterations')
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
