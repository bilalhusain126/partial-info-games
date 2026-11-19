"""Audit checks for games/bsb_game.py (read-only on the repo).
(A) exact discrete-time Picard fixed point in coefficient space, for the code's bracket convention (sum over j >= k)
    and the strict convention (j > k), compared with the continuous-time ODE coefficients on the same grid;
(B) finite-population floor of the code's rel-L2 metric: roll the closed-form linear policies on a population of
    MPOP traders and compare with the alpha-only mean-field reference (reference_paths)."""
import os, sys, numpy as np
os.environ.setdefault('DEVICE', 'cpu')
sys.path.insert(0, '/Users/bhusain/Desktop/Bilal Saleh Husain/Academics and Research/Repos/partial-info-games/games')
import bsb_game as G
T, KA, SA, KI, SI, ETA_I, ETA_B, B_, A_I, A_B, PHI_I, PHI_B, N, DT = G.T, G.KA, G.SA, G.KI, G.SI, G.ETA_I, G.ETA_B, G.B_, G.A_I, G.A_B, G.PHI_I, G.PHI_B, G.N, G.DT
cf = G.closed_form()
e = np.eye(3); E = np.eye(5)

def picard_mf(conv, iters=4000, rho=0.5):
    """MF state X=(alpha, Qbar, QbarB). nubar_k = g_k.X_k, nuB_k = h_k.X_k. Returns g, h (N,3) at the discrete fixed point."""
    g = np.stack([cf['ga'][:-1], cf['gb'][:-1], cf['gc'][:-1]], 1); h = np.stack([cf['ha'][:-1], cf['hb'][:-1], cf['hc'][:-1]], 1)
    for it in range(iters):
        A = np.zeros((N, 3, 3))
        for k in range(N):
            A[k, 0, 0] = 1 - KA * DT; A[k, 1] = e[1] + DT * g[k]; A[k, 2] = e[2] + DT * (h[k] - g[k])
        cI = B_ * h + e[0] - 2 * PHI_I * e[1]; cB = B_ * g + e[0] - 2 * PHI_B * e[2]            # integrands as linear forms on X_j
        WI = -2 * A_I * e[1]; WB = (B_ - 2 * A_B) * e[2]; gn = np.zeros_like(g); hn = np.zeros_like(h)
        for k in range(N - 1, -1, -1):
            WI_ge = cI[k] * DT + WI @ A[k]; WB_ge = cB[k] * DT + WB @ A[k]                     # bracket with j >= k
            gn[k] = (WI_ge if conv == 'ge' else WI @ A[k]) / (2 * ETA_I); hn[k] = (WB_ge if conv == 'ge' else WB @ A[k]) / (2 * ETA_B)
            WI, WB = WI_ge, WB_ge
        d = max(np.abs(gn - g).max(), np.abs(hn - h).max()); g = (1 - rho) * g + rho * gn; h = (1 - rho) * h + rho * hn
        if d < 1e-12: break
    return g, h, it, d

def picard_trader(conv, g, h, iters=4000, rho=0.5):
    """Y=(alpha, alpha_i, Qbar, Q_i, QbarB); nu_i = f.Y. g, h fixed (MF). Returns f (N,5)."""
    f = np.stack([cf['fa'][:-1], cf['faI'][:-1], cf['fb'][:-1], cf['fbI'][:-1], cf['fc'][:-1]], 1)
    gY = np.zeros((N, 5)); gY[:, 0] = g[:, 0]; gY[:, 2] = g[:, 1]; gY[:, 4] = g[:, 2]
    hY = np.zeros((N, 5)); hY[:, 0] = h[:, 0]; hY[:, 2] = h[:, 1]; hY[:, 4] = h[:, 2]
    for it in range(iters):
        A = np.zeros((N, 5, 5))
        for k in range(N):
            A[k, 0, 0] = 1 - KA * DT; A[k, 1, 1] = 1 - KI * DT; A[k, 2] = E[2] + DT * gY[k]; A[k, 3] = E[3] + DT * f[k]; A[k, 4] = E[4] + DT * (hY[k] - gY[k])
        c = B_ * hY + E[1] + E[0] - 2 * PHI_I * E[3]; W = -2 * A_I * E[3]; fn = np.zeros_like(f)
        for k in range(N - 1, -1, -1):
            W_ge = c[k] * DT + W @ A[k]; fn[k] = (W_ge if conv == 'ge' else W @ A[k]) / (2 * ETA_I); W = W_ge
        d = np.abs(fn - f).max(); f = (1 - rho) * f + rho * fn
        if d < 1e-12: break
    return f, it, d

def roll_mf(al, g, h, f, alI):
    m = al.shape[0]; Qb = np.zeros(m); QB = np.zeros(m); Qi = np.zeros(m); nuB = np.zeros((m, N)); nubar = np.zeros((m, N)); nui = np.zeros((m, N))
    for k in range(N):
        nubar[:, k] = g[k, 0] * al[:, k] + g[k, 1] * Qb + g[k, 2] * QB; nuB[:, k] = h[k, 0] * al[:, k] + h[k, 1] * Qb + h[k, 2] * QB
        nui[:, k] = f[k, 0] * al[:, k] + f[k, 1] * alI[:, k] + f[k, 2] * Qb + f[k, 3] * Qi + f[k, 4] * QB
        Qb = Qb + nubar[:, k] * DT; QB = QB + (nuB[:, k] - nubar[:, k]) * DT; Qi = Qi + nui[:, k] * DT
    return nuB, nubar, nui

rel = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean()))
print(f'PARAMS={G.PAR} N={N}')
gc_ = np.stack([cf['ga'][:-1], cf['gb'][:-1], cf['gc'][:-1]], 1); hc_ = np.stack([cf['ha'][:-1], cf['hb'][:-1], cf['hc'][:-1]], 1)
fc_ = np.stack([cf['fa'][:-1], cf['faI'][:-1], cf['fb'][:-1], cf['fbI'][:-1], cf['fc'][:-1]], 1)
al, alI = G.simulate(4000, 3); alI1 = alI[:, 0]
ref = roll_mf(al, gc_, hc_, fc_, alI1)
for conv in ['ge', 'gt']:
    g, h, it, d = picard_mf(conv); f, it2, d2 = picard_trader(conv, g, h)
    print(f'--- convention {conv}: MF Picard converged in {it} iters (d={d:.1e}); trader in {it2} (d={d2:.1e})')
    print('  max|coef diff| vs ODE: g', np.abs(g - gc_).max(0).round(4), ' h', np.abs(h - hc_).max(0).round(4), ' f', np.abs(f - fc_).max(0).round(4))
    print('  coef at k=0:   g', g[0].round(4), ' h', h[0].round(4), ' f', f[0].round(4))
    print('  ODE  at k=0:   g', gc_[0].round(4), ' h', hc_[0].round(4), ' f', fc_[0].round(4))
    print('  identities at discrete fixed point: |fa-ga|', np.abs(f[:, 0] - g[:, 0]).max().round(5), '|fc-gc|', np.abs(f[:, 4] - g[:, 2]).max().round(5), '|fb+fbI-gb|', np.abs(f[:, 2] + f[:, 3] - g[:, 1]).max().round(5))
    out = roll_mf(al, g, h, f, alI1)
    print(f'  rel L2 (discrete fixed point vs continuous ODE reference on same paths): broker {rel(out[0], ref[0]):.4f} avg flow {rel(out[1], ref[1]):.4f} trader {rel(out[2], ref[2]):.4f}')

# (B) population floor: closed-form linear policies run on a finite population vs the alpha-only MF reference
print('--- finite-population floor (closed-form policies on a population vs alpha-only reference), 2000 paths')
import torch
for mpop in [32, 64, 128, 512, 2048]:
    G.MPOP = mpop; al, alI = G.simulate(2000, 7); nuB_ex, nubar_ex, nui_ex = G.reference_paths(cf, al, alI)
    m = al.shape[0]; Qb = np.zeros(m); QB = np.zeros(m); Qi = np.zeros((m, mpop)); nuB = np.zeros((m, N)); nubar = np.zeros((m, N)); nui = np.zeros((m, mpop, N))
    for k in range(N):
        nuB[:, k] = cf['ha'][k] * al[:, k] + cf['hb'][k] * Qb + cf['hc'][k] * QB
        nui[:, :, k] = cf['fa'][k] * al[:, k, None] + cf['faI'][k] * alI[:, :, k] + cf['fb'][k] * Qb[:, None] + cf['fbI'][k] * Qi + cf['fc'][k] * QB[:, None]
        nubar[:, k] = nui[:, :, k].mean(1); Qb = Qb + nubar[:, k] * DT; QB = QB + (nuB[:, k] - nubar[:, k]) * DT; Qi = Qi + nui[:, :, k] * DT
    print(f'  MPOP={mpop:5d}: broker {rel(nuB, nuB_ex):.4f} avg flow {rel(nubar, nubar_ex):.4f} trader_i {rel(nui, nui_ex):.4f}   (1/sqrt(MPOP) = {1/np.sqrt(mpop):.3f})')
