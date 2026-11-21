"""The runs of each stage of the study. run.py skips a run whose result file exists, so a stage can be resumed.

Stages:
    budget           the cases with an exact reference at 32768, 65536 and 131072 paths, seed 0
    seeds            the same cases at 65536 paths, seeds 1 and 2; the Markov-chain game at lam = 1, seeds 0-2, with
                     the first order condition residuals
    control_variate  the cases with an exact reference at 32768 paths without the control variate; the Markov-chain
                     game with the price noise only
    width            two traders at lam = 1 and the Markov-chain game at lam = 0, GRU width 64, 65536 paths
    damping          the broker-trader game with damping 0.3 and 0.15
    linear           the Markov-chain game at lam = 0, strategies linear in the observed path, 32768 and 131072 paths
    closed_loop      two traders at lam = 1.5 and 2: gradient play (closed, three seeds at 1.5; open; from untrained
                     networks) and the closed-loop Picard iteration with damping 0.6, 0.2 and 0.1

Usage: runs.py <stage> [--dry-run]      all runs of the stage, one after another
       runs.py <stage> --count          number of runs in the stage
       runs.py <stage> --index i        only run i (0-based), as in a SLURM array task
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = [dict(game='two_traders', lam=0.0), dict(game='two_traders', lam=1.0), dict(game='markov_traders', lam=0.0),
         dict(game='broker_trader'), dict(game='major_minor')]


def command(**config):
    """Command line of one run: command(game='two_traders', lam=1.0, paths=65536)."""
    flags = []
    for key, value in config.items():
        flag = f"--{key.replace('_', '-')}"
        flags += [flag] if value is True else [flag, str(value)]
    return [sys.executable, str(HERE / 'run.py')] + flags


def closed_loop():
    runs = [command(game='two_traders', lam=1.5, paths=32768, seed=seed, method='gradient_play') for seed in (0, 1, 2)]
    runs += [command(game='two_traders', lam=1.5, paths=32768, method='gradient_play_open'),
             command(game='two_traders', lam=1.5, paths=32768, method='gradient_play_cold')]
    runs += [command(game='two_traders', lam=1.5, paths=32768, method='closed_picard', damping=damping, iterations=iterations)
             for damping, iterations in ((0.6, 16), (0.2, 40), (0.1, 60))]
    runs += [command(game='two_traders', lam=2.0, paths=32768, method='gradient_play', iterations=40, lr=lr) for lr in (1e-4, 3e-5)]
    runs += [command(game='two_traders', lam=2.0, paths=32768, method='closed_picard', damping=damping, iterations=iterations)
             for damping, iterations in ((0.2, 40), (0.1, 60))]
    return runs


STAGES = dict(
    budget=lambda: [command(**case, paths=paths) for case in CASES for paths in (32768, 65536, 131072)],
    seeds=lambda: [command(**case, paths=65536, seed=seed) for case in CASES for seed in (1, 2)]
                  + [command(game='markov_traders', lam=1.0, paths=65536, seed=seed, residual=True) for seed in (0, 1, 2)],
    control_variate=lambda: [command(**case, paths=32768, no_control_variate=True) for case in CASES]
                            + [command(game='markov_traders', lam=0.0, paths=32768, no_jump_variate=True)],
    width=lambda: [command(game='two_traders', lam=1.0, paths=65536, width=64), command(game='markov_traders', lam=0.0, paths=65536, width=64)],
    damping=lambda: [command(game='broker_trader', paths=paths, damping=damping, iterations=iterations)
                     for paths, damping, iterations in ((32768, 0.3, 30), (32768, 0.15, 40), (65536, 0.3, 30))],
    linear=lambda: [command(game='markov_traders', lam=0.0, paths=paths, method='linear') for paths in (32768, 131072)],
    closed_loop=closed_loop,
)

if __name__ == '__main__':
    stage = sys.argv[1]; runs = STAGES[stage]()
    if '--count' in sys.argv:
        print(len(runs)); sys.exit()
    if '--index' in sys.argv:
        runs = [runs[int(sys.argv[sys.argv.index('--index') + 1])]]
    for run in runs:
        print(' '.join(run[1:]), flush=True)
        if '--dry-run' not in sys.argv:
            subprocess.run(run, check=True)
