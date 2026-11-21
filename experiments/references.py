"""Exact linear-strategy equilibria of the two-trader game, open and closed loop, at lam = 1, 1.5 and 2, and the
spectrum of the best-response map at each.

Writes references/two_traders_lam<lam>.npz with the controls on the validation paths ('open', 'closed', [M, 2, N])
and the coefficients ('beta_open', 'beta_closed'), and references/spectrum.json with, for each lam and loop, the
spectral radius of the undamped best-response map, its leading eigenvalues, the factor of the map damped by 0.5,
and the smallest nonzero and largest eigenvalues of trader 1's Hessian. Each solve starts from the solution of the
same loop at the previous lam.

Usage: python references.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
import pig
from pig.games import TwoTraders, LinearStrategies

pig.use_device('cpu')
(HERE / 'references').mkdir(exist_ok=True)
spectrum = {}; start = {'open': None, 'closed': None}
for lam in (1.0, 1.5, 2.0):
    game = TwoTraders(lam=lam); solver = LinearStrategies(game); validation = game.sample(2048, 7); saved = {}
    for loop in ('open', 'closed'):
        beta = solver.solve(closed=loop == 'closed', start=start[loop]); start[loop] = beta
        saved[loop] = solver.controls(beta, validation).astype(np.float32); saved[f'beta_{loop}'] = beta.numpy()
        eigenvalues, hessian = solver.spectrum(beta, closed=loop == 'closed')
        leading = eigenvalues[np.argsort(-np.abs(eigenvalues))[:6]]
        spectrum[f'lam{lam}_{loop}'] = dict(radius=float(np.abs(eigenvalues).max()),
                                            leading=[[float(z.real), float(z.imag)] for z in leading],
                                            damped_factor=float(np.abs(0.5 + 0.5 * eigenvalues).max()),
                                            hessian_smallest_nonzero=float(hessian[hessian > 1e-9 * hessian.max()].min()),
                                            hessian_largest=float(hessian.max()))
        spectrum[f'lam{lam}_{loop}']['final_change'] = solver.final_change
        print(f'lam {lam} {loop}: mean |nu| {np.abs(saved[loop]).mean():.4f}, ' + json.dumps(spectrum[f'lam{lam}_{loop}']), flush=True)
    np.savez(HERE / 'references' / f'two_traders_lam{lam}.npz', **saved)
(HERE / 'references' / 'spectrum.json').write_text(json.dumps(spectrum, indent=1))
