"""Wu-Jaimungal broker-trader game as an npg.Model (engine sanity check against the earlier prototype and the h=0 exact solution).
Players: 0 = informed trader (full info), 1 = broker (prices only)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import sys, os, math, numpy as np, torch
sys.argv = ['x', os.environ.get('H', '0'), os.environ.get('T', '0.3')]
import wj_game as W          # reuse constants, simulate(), kalman_alpha(), exact_h0()
from npg import Solver, gateaux_check, foc_residual

class WJModel:
    n_players = 2; N = W.N; DT = W.DT; d_now = [5, 2]; d_seq = [0, 2]
    d_full = 5; d_w = 2; cv_gamma = [1.0, 1.0]
    def full_obs(s, exo, st): return torch.stack([exo['al'][:, :-1], st['QB'][:, :-1], st['QI'][:, :-1], st['Y'][:, :-1], exo['ah'][:, :-1]], -1)
    def dW(s, exo):                                       # recover the two Brownian increments (signal, price noise), scaled by 1/sqrt(dt)
        al, dz = exo['al'], exo['dz']; dWa = (al[:, 1:] - al[:, :-1] + W.KAP * al[:, :-1] * s.DT) / W.SA; dWz = (dz - al[:, :-1] * s.DT) / W.SIG
        return torch.stack([dWa, dWz], -1) / math.sqrt(s.DT)
    def exo(s, m, seed):
        al, dz = W.simulate(m, seed); ah = W.kalman_alpha(dz)
        T_ = lambda x: torch.tensor(x, dtype=torch.float32); return dict(al=T_(al), dz=T_(dz), ah=T_(ah))
    def init_state(s, m): return dict(QB=torch.zeros(m, s.N + 1), QI=torch.zeros(m, s.N + 1), Y=torch.zeros(m, s.N + 1))
    def step(s, st, u, k):
        nu, eta = u[1], u[0]
        st['QB'][:, k + 1] = st['QB'][:, k] + (nu - eta) * s.DT; st['QI'][:, k + 1] = st['QI'][:, k] + eta * s.DT
        st['Y'][:, k + 1] = st['Y'][:, k] + (W.H * nu - W.P_ * st['Y'][:, k]) * s.DT
    def obs_seq(s, i, exo, st):
        if i == 0: return None
        return torch.stack([exo['dz'] / math.sqrt(s.DT), exo['ah'][:, :-1]], -1)          # broker: price increments + Kalman alphahat
    def obs_now(s, i, exo, st, k):
        if i == 0: return torch.stack([exo['al'][:, k], st['QB'][:, k], st['QI'][:, k], st['Y'][:, k], exo['ah'][:, k]], -1)
        return torch.stack([st['Y'][:, k], exo['ah'][:, k]], -1)
    def target_weight(s, i): return torch.full((s.N,), 2 * (W.B_ if i == 0 else W.A_) * s.DT)
    def cost(s, i, exo, st, ctrl):                       # minus realised performance of trader (i=0) or broker (i=1), Wu-Jaimungal objectives
        al, Y, QB, QI = exo['al'][:, :-1], st['Y'][:, :-1], st['QB'], st['QI']; eta, nu = ctrl[0], ctrl[1]
        if i == 0: perf = ((QI[:, :-1] * (al + W.H * nu - W.P_ * Y) - W.B_ * eta ** 2 - W.RI * QI[:, :-1] ** 2) * s.DT).sum(1) - W.PSI * QI[:, -1] ** 2
        else: perf = ((QB[:, :-1] * (al + W.H * nu - W.P_ * Y) - W.A_ * nu ** 2 - W.RB * QB[:, :-1] ** 2) * s.DT).sum(1) - W.PHI * QB[:, -1] ** 2
        return -perf
    def bracket(s, i, exo, st, ctrl):
        tI, tB = W.targets(exo['al'], ctrl[1], ctrl[0], st['QB'], st['QI'], st['Y']); return tI if i == 0 else tB

def exact_discrete_h0(al, dz):
    """Exact equilibrium of the discrete-time game actually simulated, at h = 0: the trader's discrete LQR on (alpha, Q^I); the
    broker's discrete Kalman filter of (alpha, Q^I) from the price increments, then the discrete LQR on (Qhat^B, alphahat, Qhat^I).
    Same construction as audit/checks/wj_discrete_check.py, part (B). Inputs (m, N+1) and (m, N) arrays; returns eta, nu (m, N)."""
    N, DT = W.N, W.DT; m = al.shape[0]; c = 1 - W.KAP * DT
    A = np.array([[c, 0.0], [0.0, 1.0]]); B = np.array([0.0, DT]); Mq = np.array([[0.0, 0.5], [0.5, -W.RI]])
    S = np.zeros((N + 1, 2, 2)); S[N] = np.diag([0.0, -W.PSI]); L = np.zeros((N, 2))
    for k in range(N - 1, -1, -1):
        Sn = S[k + 1]; BSB = B @ Sn @ B - W.B_ * DT; BSA = B @ Sn @ A
        L[k] = -BSA / BSB; S[k] = Mq * DT + A.T @ Sn @ A - np.outer(BSA, BSA) / BSB
    QI = np.zeros((m, N + 1)); eta = np.zeros((m, N))
    for k in range(N):
        eta[:, k] = L[k, 0] * al[:, k] + L[k, 1] * QI[:, k]; QI[:, k + 1] = QI[:, k] + eta[:, k] * DT
    Hm = np.array([1.0, 0.0]); G = np.diag([W.SA ** 2 * DT, 0.0]); Sig = np.diag([W.SA ** 2 / (2 * W.KAP), 0.0]); Ks, Fs = [], []
    for k in range(N):
        K = Sig @ Hm / (Hm @ Sig @ Hm * DT + W.SIG ** 2); Ks.append(K); F = A + np.outer(B, L[k]); Fs.append(F)
        Sp = Sig - np.outer(K, Hm @ Sig) * DT; Sig = F @ Sp @ F.T + G
    Bxi = np.array([DT, 0.0, 0.0]); Mxi = np.zeros((3, 3)); Mxi[0, 0] = -W.RB; Mxi[0, 1] = Mxi[1, 0] = 0.5
    Sx = np.zeros((N + 1, 3, 3)); Sx[N] = np.diag([-W.PHI, 0, 0]); Lx = np.zeros((N, 3))
    for k in range(N - 1, -1, -1):
        F = Fs[k]; Axi = np.array([[1.0, -DT * L[k, 0], -DT * L[k, 1]], [0.0, F[0, 0], F[0, 1]], [0.0, F[1, 0], F[1, 1]]])
        Sn = Sx[k + 1]; BSB = Bxi @ Sn @ Bxi - W.A_ * DT; BSA = Bxi @ Sn @ Axi
        Lx[k] = -BSA / BSB; Sx[k] = Mxi * DT + Axi.T @ Sn @ Axi - np.outer(BSA, BSA) / BSB
    xh = np.zeros((m, 2)); intnu = np.zeros(m); nu = np.zeros((m, N))
    for k in range(N):
        xi = np.c_[intnu - xh[:, 1], xh]; nu[:, k] = xi @ Lx[k]
        inn = dz[:, k] - xh[:, 0] * DT; xh = (xh + np.outer(inn, Ks[k])) @ Fs[k].T; intnu = intnu + nu[:, k] * DT
    return eta, nu

if __name__ == '__main__':
    M = WJModel(); S = Solver(M, steps=int(os.environ.get('STEPS', '800')), batch=int(os.environ.get('BATCH', '1024')), m_paths=int(os.environ.get('M_PATHS', '8192')))
    ref = None
    if W.H == 0:                                  # REF=discrete (default): exact solution of the simulated discrete game; REF=continuous: Euler-discretised continuous-time solution
        al_v, dz_v = S.exo_v['al'].cpu().numpy().astype(np.float64), S.exo_v['dz'].cpu().numpy().astype(np.float64)
        if os.environ.get('REF', 'discrete') == 'discrete': eta_ex, nu_ex = exact_discrete_h0(al_v, dz_v)
        else: eta_ex, nu_ex, _, _, _ = W.exact_h0(al_v, dz_v)
        ref = [torch.tensor(eta_ex, dtype=torch.float32), torch.tensor(nu_ex, dtype=torch.float32)]
        print(f"h = 0 reference: {os.environ.get('REF', 'discrete')}", flush=True)
    vstate, vctrl, hist = S.run(int(os.environ.get('ITERS', '8')), ref=ref)
    if os.environ.get('GATEAUX'):
        for i in range(2): gateaux_check(S, i)
    np.savez(os.environ.get('SAVE', os.path.join(os.environ.get('RESULTS_DATA', os.path.expanduser('~/Desktop/Bilal Saleh Husain/research-data/partial-info-games')), f'wj_last.npz')), eta=vctrl[0].cpu().numpy(), nu=vctrl[1].cpu().numpy(), eta_ex=(ref[0].cpu().numpy() if ref else np.zeros(1)), nu_ex=(ref[1].cpu().numpy() if ref else np.zeros(1)), hist=np.array(hist))
