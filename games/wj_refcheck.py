"""Reference-vs-reference check at h = 0: linear-regression Picard with the broker regressed on the EXACT filtered features
(Qhat^B, alphahat, Qhat^I) and the trader on the full state, against the Riccati/Kalman/LQR solution in wj_game.exact_h0."""
import sys, os; sys.argv = ["x", "0", "0.3"]; os.environ["BROKER"] = "gru"
import numpy as np, torch
import wj_game as W
al, dz = W.simulate(65536, 11); eta_ex, nu_ex, QI_ex, QB_ex, XI = W.exact_h0(al, dz)
alT, XIT = torch.tensor(al, dtype=torch.float32), torch.tensor(XI, dtype=torch.float32)
N, DT, m = W.N, W.DT, al.shape[0]
nu = torch.zeros(m, N); eta = torch.zeros(m, N)
for it in range(int(os.environ.get("REF_ITERS", "60"))):
    QB = torch.zeros(m, N + 1); QI = torch.zeros(m, N + 1); Y = torch.zeros(m, N + 1)
    for k in range(N):
        QB[:, k + 1] = QB[:, k] + (nu[:, k] - eta[:, k]) * DT; QI[:, k + 1] = QI[:, k] + eta[:, k] * DT
    tI, tB = W.targets(alT, nu, eta, QB, QI, Y); nn_, ne_ = torch.zeros_like(nu), torch.zeros_like(eta)
    for k in range(N):
        XB = torch.cat([torch.ones(m, 1), XIT[:, k]], 1); XIf = torch.stack([torch.ones(m), alT[:, k], QI[:, k]], 1)
        nn_[:, k] = (XB @ torch.linalg.lstsq(XB, tB[:, k:k + 1]).solution).squeeze(-1)
        ne_[:, k] = (XIf @ torch.linalg.lstsq(XIf, tI[:, k:k + 1]).solution).squeeze(-1)
    nu = 0.3 * nu + 0.7 * nn_; eta = 0.3 * eta + 0.7 * ne_
rel = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
print('linear Picard on exact features vs exact_h0:  eta rel err', round(rel(eta.numpy(), eta_ex), 4), ' nu rel err', round(rel(nu.numpy(), nu_ex), 4))
# note: Qhat^B in XI was built with the exact trader; in the Picard loop the trader is re-estimated, so small discrepancy is expected