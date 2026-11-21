"""Linear-strategy equilibrium of the Markov-chain two-trader game (pm_game.py), as a baseline for the learned solution.
Same damped Picard iteration and the same targets (automatic-differentiation first order condition, open loop) as the learned solver,
but the control of trader i at t_k is restricted to a linear function of its own observed path:
    u^i_k = c^i_k . (1, alpha^i_k, q^i_k, S_k, increments of (alpha^i, S) over steps 0..k-1),
and the regression step is ordinary least squares at each time step, which gives the best linear predictor of the target.
At lam = 0 the exact (nonlinear, Wonham-filter) solution is the reference.
Usage: LAM=0 python3 pm_linear.py [iters]   (env: LAM, M_PATHS, RHO_DAMP, AVG_FROM, SEED, SAVE)"""
import os, sys, math, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pm_game as P
from npg import Solver

M_ = P.PMModel(); N = P.N
S = Solver(M_, hid=8, m_paths=int(os.environ.get('M_PATHS', '32768')), m_val=2048, seed=int(os.environ.get('SEED', '0')))
rho = float(os.environ.get('RHO_DAMP', '0.5')); avg_from = int(os.environ.get('AVG_FROM', '6'))

def features(exo, coef=None):
    """Roll the game forward. With coef (list over players of (N, d_k) lists), the controls are the linear strategies; returns the
    controls, the feature matrices per player and step, and the state."""
    m = exo['a1'].shape[0]; st = M_.init_state(m); ctrl = [torch.zeros(m, N) for _ in range(2)]
    feats = [[None] * N for _ in range(2)]; incs = [[] for _ in range(2)]
    for k in range(N):
        u = {}
        for i in range(2):
            now = M_.obs_now(i, exo, st, k)
            F = torch.cat([torch.ones(m, 1), now] + incs[i], 1); feats[i][k] = F
            u[i] = F @ coef[i][k] if coef is not None else torch.zeros(m); ctrl[i][:, k] = u[i]
        M_.step(st, u, k)
        for i in range(2): incs[i].append(M_.obs_inc(i, exo, st, k))
    return ctrl, feats, st

ref = None
if P.LAM == 0.0:
    r = P.reference_lam0(); nu_ex = P.reference_paths_lam0(r, S.exo_v['_nz']); ref = [torch.tensor(nu_ex[:, i], dtype=torch.float32) for i in range(2)]
rel = lambda a, b: float(((a - b) ** 2).mean().sqrt() / (b ** 2).mean().sqrt())
coef = None; avg = None; n_avg = 0; t0 = time.time()
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
for it in range(ITERS):
    ctrl, feats, st = features(S.exo_tr, coef)
    targets = S.gradient_targets(S.exo_tr, ctrl)
    new = [[None] * N for _ in range(2)]
    for i in range(2):
        tg = (1 - rho) * ctrl[i] + rho * targets[i]
        for k in range(N):
            F = feats[i][k].double(); y = tg[:, k].double()
            new[i][k] = torch.linalg.lstsq(F, y.unsqueeze(1)).solution.squeeze(1).float()
    coef = new
    vctrl, _, _ = features(S.exo_v, coef)
    if it >= avg_from:
        n_avg += 1; avg = [c.clone() for c in vctrl] if avg is None else [a + (c - a) / n_avg for a, c in zip(avg, vctrl)]
    msg = f'linear it {it:2d}'
    if ref is not None:
        msg += ' | rel L2 vs exact: ' + ' '.join(f'u{i} {rel(vctrl[i], ref[i]):.3f}' for i in range(2))
        if avg is not None: msg += f' | avg({n_avg}): ' + ' '.join(f'u{i} {rel(avg[i], ref[i]):.3f}' for i in range(2))
    print(msg + f' ({time.time()-t0:.0f}s)', flush=True)
out = avg if avg is not None else vctrl
np.savez(os.environ.get('SAVE', os.path.join(os.environ.get('RESULTS_DATA', os.path.expanduser('~/Desktop/Bilal Saleh Husain/research-data/partial-info-games')), 'pm_linear_last.npz')),
         nu1=out[0].cpu().numpy(), nu2=out[1].cpu().numpy(), nu_ex=(ref[0].cpu().numpy() if ref else np.zeros(1)))
