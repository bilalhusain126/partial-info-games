"""Loads the saved runs for the results notebook."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / 'results'


def load(results=RESULTS) -> pd.DataFrame:
    """One row per run: its settings, and per player the error, the regression slope on the reference, the mean
    absolute control and the first order condition residual; the run time in seconds and the file name."""
    rows = []
    for path in sorted(Path(results).glob('*.json')):
        result = json.loads(path.read_text()); row = {**result['settings'], 'seconds': result['seconds'], 'name': path.stem}
        for i in range(2):
            row[f'error{i}'] = result.get('error', [np.nan, np.nan])[i]
            row[f'mean_abs{i}'] = result['mean_abs_control'][i]
            row[f'residual{i}'] = result.get('first_order_residual', [np.nan, np.nan])[i]
            row[f'slope{i}'] = result.get('slope', [np.nan, np.nan])[i]
        rows.append(row)
    if not rows:
        print(f'no saved runs in {results}: run the study first (experiments/runs.py)')
        return pd.DataFrame(columns=['game', 'lam', 'paths', 'seed', 'width', 'iterations', 'control_variate',
                                     'jump_variate', 'name'] + [f'{key}{i}' for key in ('error', 'mean_abs', 'residual', 'slope') for i in range(2)])
    return pd.DataFrame(rows)


def controls(name: str, results=RESULTS) -> np.ndarray:
    """Validation controls of a run, [M, 2, N]."""
    return np.load(Path(results) / f'{name}.npz')['controls']
