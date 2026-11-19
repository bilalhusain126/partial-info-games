"""Audit check: at h = 0, compare on the same paths
  (A) wj_game.exact_h0            continuous-time Riccati / Kalman-Bucy / LQR, Euler-discretised (the repo's reference)
  (B) discrete-time exact solution of the Euler game actually simulated (discrete Riccati, discrete Kalman filter, discrete LQR)
  (C) fixed point of the code's bracket (targets(), sum over s >= t) by linear regression on exact features (wj_refcheck.py logic)
  (D) same as (C) but with the discrete-game first-order condition (sum over s > t)
"""
import sys, os
sys.argv = ['x', '0', os.environ.get('T', '0.3')]; os.environ['BROKER'] = 'gru'
sys.path.insert(0, '/Users/bhusain/Desktop/Bilal Saleh Husain/Academics and Research/Repos/partial-info-games/games')
import numpy as np, torch
import wj_game as W
N, DT = W.N, W.DT; m = int(os.environ.get('M', '32768'))
al, dz = W.simulate(m, 11)
eta_A, nu_A, QI_A, QB_A, XI_A = W.exact_h0(al, dz)
rel = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))

# ---------- (B) discrete-time exact ----------
c = 1 - W.KAP * DT
# trader: x = (alpha, Q^I), x_{k+1} = A x + B eta + noise; reward sum_k (x'M x - b eta^2) DT - psi Q_N^2
A = np.array([[c, 0.0], [0.0, 1.0]]); B = np.array([0.0, DT]); Mq = np.array([[0.0, 0.5], [0.5, -W.RI]])
S = np.zeros((N + 1, 2, 2)); S[N] = np.diag([0.0, -W.PSI]); L = np.zeros((N, 2))
for k in range(N - 1, -1, -1):
    Sn = S[k + 1]; BSB = B @ Sn @ B - W.B_ * DT; BSA = B @ Sn @ A
    L[k] = -BSA / BSB                                   # eta_k = L[k] @ x_k
    S[k] = Mq * DT + A.T @ Sn @ A - np.outer(BSA, BSA) / BSB
QI = np.zeros((m, N + 1)); eta_B = np.zeros((m, N))
for k in range(N):
    eta_B[:, k] = L[k, 0] * al[:, k] + L[k, 1] * QI[:, k]; QI[:, k + 1] = QI[:, k] + eta_B[:, k] * DT
# broker: discrete Kalman filter of x_k = (alpha_k, Q^I_k) given dz_0..dz_{k-1}; observation dz_k = alpha_k DT + sig sqrt(DT) zeta
Hm = np.array([1.0, 0.0]); G = np.diag([W.SA ** 2 * DT, 0.0])
Sig = np.diag([W.SA ** 2 / (2 * W.KAP), 0.0]); Ks = []; Fs = []
for k in range(N):
    K = Sig @ Hm / (Hm @ Sig @ Hm * DT + W.SIG ** 2); Ks.append(K)
    F = A + np.outer(B, L[k]); Fs.append(F)
    Sp = Sig - np.outer(K, Hm @ Sig) * DT; Sig = F @ Sp @ F.T + G
# LQR on xi = (Qhat^B, alphahat, Qhat^I): xi_{k+1} = Axi_k xi_k + Bxi nu_k + noise
Bxi = np.array([DT, 0.0, 0.0]); Mxi = np.zeros((3, 3)); Mxi[0, 0] = -W.RB; Mxi[0, 1] = Mxi[1, 0] = 0.5
Sx = np.zeros((N + 1, 3, 3)); Sx[N] = np.diag([-W.PHI, 0, 0]); Lx = np.zeros((N, 3))
for k in range(N - 1, -1, -1):
    F = Fs[k]
    Axi = np.array([[1.0, -DT * L[k, 0], -DT * L[k, 1]], [0.0, F[0, 0], F[0, 1]], [0.0, F[1, 0], F[1, 1]]])
    Sn = Sx[k + 1]; BSB = Bxi @ Sn @ Bxi - W.A_ * DT; BSA = Bxi @ Sn @ Axi
    Lx[k] = -BSA / BSB; Sx[k] = Mxi * DT + Axi.T @ Sn @ Axi - np.outer(BSA, BSA) / BSB
xh = np.zeros((m, 2)); intnu = np.zeros(m); nu_B = np.zeros((m, N)); XI_B = np.zeros((m, N, 3))
for k in range(N):
    xi = np.c_[intnu - xh[:, 1], xh]; XI_B[:, k] = xi; nu_B[:, k] = xi @ Lx[k]
    inn = dz[:, k] - xh[:, 0] * DT
    xh = (xh + np.outer(inn, Ks[k])) @ Fs[k].T; intnu = intnu + nu_B[:, k] * DT

print(f'N={N} T={W.T} DT={DT:.4g}  paths {m}')
print(f'(A) continuous-Euler reference vs (B) discrete-exact:   eta {rel(eta_A, eta_B):.4f}   nu {rel(nu_A, nu_B):.4f}')

# ---------- (C)/(D) linear-regression Picard on exact features with the code's bracket ----------
def picard(conv, feats_B, iters=80, rho=0.7, dtype=torch.float64):
    alT = torch.tensor(al, dtype=torch.float32); fB = torch.tensor(feats_B, dtype=torch.float32)   # fB[:, k] = (alphahat_k, Qhat^I_k)
    nu = torch.zeros(m, N); eta = torch.zeros(m, N)
    for it in range(iters):
        QB = torch.zeros(m, N + 1); QI = torch.zeros(m, N + 1); Y = torch.zeros(m, N + 1)
        for k in range(N):
            QB[:, k + 1] = QB[:, k] + (nu[:, k] - eta[:, k]) * DT; QI[:, k + 1] = QI[:, k] + eta[:, k] * DT
        if conv == 'ge':
            tI, tB = W.targets(alT, nu, eta, QB, QI, Y)
        else:   # discrete-game FOC at h = 0: eta_k = (1/2b)[-2 psi QI_N + sum_{j>k}(alpha_j - 2 rI QI_j) DT]; nu_k similar with QB
            gI = alT[:, :-1] - 2 * W.RI * QI[:, :-1]; gB = alT[:, :-1] - 2 * W.RB * QB[:, :-1]
            tailI = torch.flip(torch.cumsum(torch.flip(gI, [1]), 1), [1]) * DT; tailI = torch.cat([tailI[:, 1:], torch.zeros(m, 1)], 1)
            tailB = torch.flip(torch.cumsum(torch.flip(gB, [1]), 1), [1]) * DT; tailB = torch.cat([tailB[:, 1:], torch.zeros(m, 1)], 1)
            tI = (-2 * W.PSI * QI[:, -1:] + tailI) / (2 * W.B_); tB = (-2 * W.PHI * QB[:, -1:] + tailB) / (2 * W.A_)
        intnu = torch.cat([torch.zeros(m, 1), torch.cumsum(nu, 1)[:, :-1]], 1) * DT
        nn_, ne_ = torch.zeros_like(nu), torch.zeros_like(eta)
        for k in range(N):
            XB = torch.stack([torch.ones(m), intnu[:, k] - fB[:, k, 1], fB[:, k, 0], fB[:, k, 1]], 1).to(dtype); XIf = torch.stack([torch.ones(m), alT[:, k], QI[:, k]], 1).to(dtype)
            nn_[:, k] = (XB @ torch.linalg.lstsq(XB, tB[:, k:k + 1].to(dtype)).solution).squeeze(-1).float()
            ne_[:, k] = (XIf @ torch.linalg.lstsq(XIf, tI[:, k:k + 1].to(dtype)).solution).squeeze(-1).float()
        chg = float(((nu - nn_) ** 2).mean().sqrt()); nu = (1 - rho) * nu + rho * nn_; eta = (1 - rho) * eta + rho * ne_
    print(f'   picard final change {chg:.2e}')
    return nu.numpy(), eta.numpy()

nu_C, eta_C = picard('ge', XI_A[:, :-1, 1:])
print(f'(C) code bracket (s>=t) fixed point vs (A): eta {rel(eta_C, eta_A):.4f} nu {rel(nu_C, nu_A):.4f}   vs (B): eta {rel(eta_C, eta_B):.4f} nu {rel(nu_C, nu_B):.4f}')
nu_D, eta_D = picard('gt', XI_B[:, :, 1:])
print(f'(D) discrete FOC (s>t) fixed point   vs (A): eta {rel(eta_D, eta_A):.4f} nu {rel(nu_D, nu_A):.4f}   vs (B): eta {rel(eta_D, eta_B):.4f} nu {rel(nu_D, nu_B):.4f}')
nu_E, eta_E = picard('ge', XI_A[:, :-1, 1:], dtype=torch.float32)
print(f'(E) = (C) with float32 lstsq (as wj_refcheck.py) vs (A): eta {rel(eta_E, eta_A):.4f} nu {rel(nu_E, nu_A):.4f}')
print('scales: rms eta', float(np.sqrt((eta_B**2).mean())), 'rms nu', float(np.sqrt((nu_B**2).mean())))
