"""Recompute the full-information linear reference in float64 on a saved run's validation paths and report corrected errors."""
import sys, os, numpy as np, torch
f, h, T = sys.argv[1], sys.argv[2], sys.argv[3]; sys.argv = ['x', h, T]; os.environ['BROKER'] = 'full'
import wj_game as W
d = np.load(f); al_v = torch.tensor(d['alpha'], dtype=torch.float32); N = W.N; DT = W.DT; H = W.H
def rollout(al, cB, cI):
    m = al.shape[0]; nu = torch.zeros(m, N); eta = torch.zeros(m, N); QB = torch.zeros(m, N + 1); QI = torch.zeros(m, N + 1); Y = torch.zeros(m, N + 1)
    for k in range(N):
        X = torch.stack([torch.ones(m), al[:, k], QB[:, k], QI[:, k], Y[:, k]], 1); nu[:, k] = X @ cB[k]; eta[:, k] = X @ cI[k]
        QB[:, k + 1] = QB[:, k] + (nu[:, k] - eta[:, k]) * DT; QI[:, k + 1] = QI[:, k] + eta[:, k] * DT; Y[:, k + 1] = Y[:, k] + (H * nu[:, k] - W.P_ * Y[:, k]) * DT
    return nu, eta, QB, QI, Y
al_big = torch.tensor(W.simulate(65536, 3)[0], dtype=torch.float32); cB = torch.zeros(N, 5); cI = torch.zeros(N, 5); rho = float(os.environ.get('RHO', '0.3')); prev = None
for it in range(int(os.environ.get('ITERS', '150'))):
    nu, eta, QB, QI, Y = rollout(al_big, cB, cI); tI, tB = W.targets(al_big, nu, eta, QB, QI, Y)
    for k in range(N):
        X = torch.stack([torch.ones(al_big.shape[0]), al_big[:, k], QB[:, k], QI[:, k], Y[:, k]], 1).double()
        Xn = X.numpy(); ls = lambda y: torch.tensor(np.linalg.lstsq(Xn, y.double().numpy(), rcond=1e-6)[0], dtype=torch.float32)   # rank-truncated: early-time features are collinear
        cB[k] = (1 - rho) * cB[k] + rho * ls(tB[:, k])
        cI[k] = (1 - rho) * cI[k] + rho * ls(tI[:, k])
    chg = float(((nu - rollout(al_big, cB, cI)[0]) ** 2).mean().sqrt()) if True else 0
    if it % 10 == 0 or chg > 1: print(f'  lin-Picard it {it} change {chg:.2e} rms nu {float((nu**2).mean().sqrt()):.4f}', flush=True)
    if chg < 1e-5: break
nu_ex, eta_ex, _, _, _ = rollout(al_v, cB, cI); nu_ex, eta_ex = nu_ex.numpy(), eta_ex.numpy()
rel = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
print(f'{f}: corrected full-info reference (float64) -> solver error eta {rel(d["eta"], eta_ex):.3f} nu {rel(d["nu"], nu_ex):.3f} | old (float32) reference vs new: eta {rel(d["eta_ex"], eta_ex):.3f} nu {rel(d["nu_ex"], nu_ex):.3f}')
np.savez(f.replace('.npz', '_ref64.npz'), nu_ex=nu_ex, eta_ex=eta_ex)