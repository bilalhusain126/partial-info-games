"""The runs of each stage of the study. run.py skips a run whose result file exists, so a stage can be resumed.

Stages:
    budget           the cases with an exact reference at 25000, 50000 and 100000 paths, seed 0
    seeds            the same cases at 50000 paths, seeds 1 and 2; the Markov-chain game at lam = 1, seeds 0-2, with
                     the first order condition residuals
    control_variate  the cases with an exact reference at 25000 paths without the control variate; the Markov-chain
                     game with the price noise only
    width            two traders at lam = 1 and the Markov-chain game at lam = 0, GRU width 64, 50000 paths
    damping          the broker-trader game with damping 0.3 and 0.15
    coupling         two traders at lam = 1.5 and 2 (stronger coupling), three seeds

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
    """Command line of one run: command(game='two_traders', lam=1.0, paths=50000)."""
    flags = []
    for key, value in config.items():
        flag = f"--{key.replace('_', '-')}"
        flags += [flag] if value is True else [flag, str(value)]
    return [sys.executable, str(HERE / 'run.py')] + flags


STAGES = dict(
    budget=lambda: [command(**case, paths=paths) for case in CASES for paths in (25000, 50000, 100000)],
    seeds=lambda: [command(**case, paths=50000, seed=seed) for case in CASES for seed in (1, 2)]
                  + [command(game='markov_traders', lam=1.0, paths=50000, seed=seed, residual=True) for seed in (0, 1, 2)],
    control_variate=lambda: [command(**case, paths=25000, no_control_variate=True) for case in CASES]
                            + [command(game='markov_traders', lam=0.0, paths=25000, no_jump_variate=True)],
    width=lambda: [command(game='two_traders', lam=1.0, paths=50000, width=64), command(game='markov_traders', lam=0.0, paths=50000, width=64)],
    damping=lambda: [command(game='broker_trader', paths=paths, damping=damping, iterations=iterations)
                     for paths, damping, iterations in ((25000, 0.3, 30), (25000, 0.15, 40), (50000, 0.3, 30))],
    coupling=lambda: [command(game='two_traders', lam=lam, paths=25000, seed=seed) for lam in (1.5, 2.0) for seed in (0, 1, 2)],
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
