"""Loads the saved runs for the results notebook."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / 'results'
DAMPING = dict(two_traders=0.6, markov_traders=0.5, broker_trader=0.7, major_minor=0.7)       # as in run.py
ITERATIONS = dict(picard=16, closed_picard=16, gradient_play=30, gradient_play_open=30, gradient_play_cold=60, linear=30)
DEFAULTS = dict(lam=np.nan, seed=0, method='picard', width=32, control_variate=True, jump_variate=True, residual=False)


def settings(config: dict) -> dict:
    """The settings of a run with the defaults of run.py filled in."""
    row = {**DEFAULTS, **config}
    row.setdefault('damping', DAMPING[row['game']])
    row.setdefault('iterations', ITERATIONS[row['method']])
    row.setdefault('lr', 1e-4 if row['method'].startswith('gradient_play') else np.nan)
    return row


def load(results=RESULTS) -> pd.DataFrame:
    """One row per run: its settings with the defaults filled in, errors and mean absolute controls per player,
    first order condition residuals, seconds and file name."""
    rows = []
    for path in sorted(Path(results).glob('*.json')):
        result = json.loads(path.read_text()); row = {**settings(result['config']), 'seconds': result['seconds'], 'name': path.stem}
        for i in range(2):
            row[f'error{i}'] = result.get('error', [np.nan, np.nan])[i]
            row[f'mean_abs{i}'] = result['mean_abs_control'][i]
            row[f'residual{i}'] = result.get('first_order_residual', [np.nan, np.nan])[i]
        rows.append(row)
    return pd.DataFrame(rows)


def controls(name: str, results=RESULTS) -> np.ndarray:
    """Validation controls of a run, [M, 2, N]."""
    return np.load(Path(results) / f'{name}.npz')['controls']
