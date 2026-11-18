"""Firoozi-Caines (arXiv 1810.04369) major-minor LQG game with partial observations of ALL agents, "Instance 1" (scalar, one type,
H_0 = H_2 = 0 so the paper's solution is exact): the major tracks eta_0 and observes its own state through noise; the minor tracks
H_1 x_0 + eta, observes (x_i, x_0) through noise, and must estimate the major's ESTIMATE of x_0 (the major's control is a feedback on
it). Information is non-nested: neither player sees the other's observation path.
    dx_0 = (A0 x_0 + B0 u_0) dt + D0 dw_0,      dy_0 = l0 x_0 dt + sv0 dv_0                     (major)
    dx_i = (A x_i + G x_0 + B u_i) dt + D dw_i,  dy_i = (l1 x_i, l2 x_0) dt + sv dv_i             (minor)
    J_0 = E int e^{-rho t} [Q0 (x_0 - eta0)^2 + R0 u_0^2] dt,   J_i = E int e^{-rho t} [Q (x_i - H1 x_0 - eta)^2 + R u_i^2] dt
Exact reference (separation theorem): major = Kalman filter on x_0 + discounted Riccati; minor = Kalman filter on (x_i, x_0, xhat_0)
with the joint closed-loop dynamics of the major's state and filter (paper's Prop. 2, eq. (43)) + discounted Riccati.
Solver: npg deep Picard regression, each player's GRU reads only its own observation increments.
Usage: python3 fc_game.py [iters]   (env: M_PATHS, STEPS, BATCH, N_STEPS, NOISE=mid|low|high, CHECK_REF, SAVE)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import os, sys, math, time, numpy as np, torch
from npg import Solver

T = 5.0; N = int(os.environ.get('N_STEPS', '100')); DT = T / N; tg = np.linspace(0, T, N + 1); RHO = 0.5
A0, B0, L0, Q0, R0, ETA0 = float(os.environ.get('A0', '-1.0')), 1.0, 1.0, float(os.environ.get('Q0', '1.0')), 1.0, 0.25
A_, B_, G_, L1, L2, Q_, R_, H1, ETA = -0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.6, 0.25
NOISE = os.environ.get('NOISE', 'mid')
D0, D_, SV0, SV = dict(low=(0.1, 0.1, 0.1, 0.1), mid=(0.5, 0.5, 0.3, 0.3), high=(1.0, 1.0, 0.5, 0.5), asym=(0.5, 0.5, 1.0, 0.3))[NOISE]
SV2 = float(os.environ.get('SV2', str(SV)))                 # minor's observation noise on the x_0 channel (default = SV)
VX0 = 1.0                                                   # Var of x_0(0), x_i(0)

# ---------------- exact reference (discrete-time LQG of the Euler-discretised system, so no O(dt) mismatch) ----------------
BETA = math.exp(-RHO * DT)
def dlqg(F, G, c, Qz, qz, R):
    """Discounted discrete-time LQ: cost sum_k beta^k [(z-zeta)'Q(z-zeta) + R u^2] dt with Q=Qz, Q zeta = qz; z_{k+1} = F_k z_k + G u_k + c_k + noise.
    Returns feedback u_k = -(Gain_k z_k + g_k) (certainty equivalence applies to the filtered state)."""
    n = len(G); P = np.zeros((n, n)); sv = np.zeros(n); Gain = np.zeros((N, n)); g = np.zeros(N)
    for k in range(N - 1, -1, -1):
        Fk, ck = F(k), c(k); Mi = 1.0 / (R * DT + BETA * G @ P @ G)
        Gain[k] = Mi * BETA * (G @ P @ Fk); g[k] = Mi * BETA * (G @ (P @ ck + sv))
        Pn = Qz * DT + BETA * Fk.T @ P @ Fk - BETA ** 2 * np.outer(Fk.T @ P @ G, G @ P @ Fk) * Mi
        sv = -qz * DT + BETA * Fk.T @ (P @ ck + sv) - BETA ** 2 * (Fk.T @ P @ G) * Mi * (G @ (P @ ck + sv)); P = Pn
    return Gain, g
def dkalman(F, Sig, C, Rv, V0):
    """Discrete Kalman gains for prior estimates: observation increment dy_k = C z_k dt + noise(Rv dt); returns K_k (prior -> posterior) and prior variances."""
    n = V0.shape[0]; V = V0.copy(); K = np.zeros((N, n, C.shape[0])); Vs = np.zeros((N + 1, n, n)); Vs[0] = V
    for k in range(N):
        K[k] = V @ C.T @ np.linalg.inv(C @ V @ C.T * DT + Rv); Vp = V - K[k] @ C * DT @ V
        V = F(k) @ Vp @ F(k).T + Sig(k); Vs[k + 1] = V
    return K, Vs
def reference():
    # major: scalar system x_{k+1} = (1 + A0 dt) x + B0 dt u + D0 dw
    F0 = lambda k: np.array([[1 + A0 * DT]]); G0 = np.array([B0 * DT]); c0 = lambda k: np.zeros(1)
    Gain0, g0 = dlqg(F0, G0, c0, np.array([[Q0]]), np.array([Q0 * ETA0]), R0)
    K0, V0 = dkalman(F0, lambda k: np.array([[D0 ** 2 * DT]]), np.array([[L0]]), np.array([[SV0 ** 2]]), np.array([[VX0]]))
    K0 = K0[:, 0, 0]
    # minor: z = (x_i, x_0, xhat_0) with the major's exact filter recursion xhat_{k+1} = (1+A0 dt)[xhat + K_k (dy_0 - L0 xhat dt)] + B0 dt u_0
    def Fz(k):
        a0 = 1 + A0 * DT
        return np.array([[1 + A_ * DT, G_ * DT, 0.0], [0.0, a0, -B0 * DT * Gain0[k, 0]], [0.0, a0 * K0[k] * L0 * DT, a0 * (1 - K0[k] * L0 * DT) - B0 * DT * Gain0[k, 0]]])
    def cz(k): return np.array([0.0, -B0 * DT * g0[k], -B0 * DT * g0[k]])
    def Sz(k): return np.diag([D_ ** 2 * DT, D0 ** 2 * DT, ((1 + A0 * DT) * K0[k] * SV0) ** 2 * DT])
    Gz = np.array([B_ * DT, 0.0, 0.0]); Cz = np.array([[L1, 0, 0], [0, L2, 0]]); Cc = np.array([1.0, -H1, 0.0])
    Gain1, g1 = dlqg(Fz, Gz, cz, np.outer(Cc, Cc) * Q_, Cc * Q_ * ETA, R_)
    K1, V1 = dkalman(Fz, Sz, Cz, np.diag([SV ** 2, SV2 ** 2]), np.diag([VX0, VX0, 0.0]))
    return dict(Gain0=Gain0, g0=g0, K0=K0, V0=V0, Gain1=Gain1, g1=g1, K1=K1, V1=V1, Fz=Fz, cz=cz, Gz=Gz, Cz=Cz)

def simulate_noise(m, seed):
    rng = np.random.default_rng(seed)
    return dict(x0_0=rng.standard_normal(m) * math.sqrt(VX0), xi_0=rng.standard_normal(m) * math.sqrt(VX0),
                dw0=rng.standard_normal((m, N)) * math.sqrt(DT), dwi=rng.standard_normal((m, N)) * math.sqrt(DT),
                dv0=rng.standard_normal((m, N)) * math.sqrt(DT), dvi=rng.standard_normal((m, N, 2)) * math.sqrt(DT))

def reference_paths(ref, nz):
    """Equilibrium under the exact strategies along given noise: returns controls (u0, ui), states, prior filter states."""
    m = nz['x0_0'].shape[0]; x0 = nz['x0_0'].copy(); xi = nz['xi_0'].copy(); xh0 = np.zeros(m); zh = np.zeros((m, 3))
    u0 = np.zeros((m, N)); ui = np.zeros((m, N)); X0 = np.zeros((m, N + 1)); XI = np.zeros((m, N + 1)); XH0 = np.zeros((m, N + 1)); ZH = np.zeros((m, N + 1, 3))
    X0[:, 0], XI[:, 0] = x0, xi
    for k in range(N):
        u0[:, k] = -(ref['Gain0'][k, 0] * xh0 + ref['g0'][k]); ui[:, k] = -(zh @ ref['Gain1'][k] + ref['g1'][k])
        dy0 = L0 * x0 * DT + SV0 * nz['dv0'][:, k]; dyi = np.stack([L1 * xi, L2 * x0], -1) * DT + np.array([SV, SV2]) * nz['dvi'][:, k]
        x0n = x0 + (A0 * x0 + B0 * u0[:, k]) * DT + D0 * nz['dw0'][:, k]
        xin = xi + (A_ * xi + G_ * x0 + B_ * ui[:, k]) * DT + D_ * nz['dwi'][:, k]
        xh0 = (1 + A0 * DT) * (xh0 + ref['K0'][k] * (dy0 - L0 * xh0 * DT)) + B0 * DT * u0[:, k]
        zp = zh + (dyi - zh @ ref['Cz'].T * DT) @ ref['K1'][k].T; zh = zp @ ref['Fz'](k).T + ref['cz'](k) + ref['Gz'] * ui[:, k, None]
        x0, xi = x0n, xin; X0[:, k + 1], XI[:, k + 1], XH0[:, k + 1], ZH[:, k + 1] = x0, xi, xh0, zh
    return u0, ui, X0, XI, XH0, ZH

def kernel(A, B):
    """W[j, k] = e^{-rho (t_j - t_k)} (1 + A dt)^{j-k-1} B dt for j > k: sensitivity of the discounted cost at j to the control at k."""
    W = np.zeros((N, N)); j, k = np.indices((N, N)); n = j - k; mask = n > 0
    W[mask] = np.exp(-RHO * n[mask] * DT) * (1 + A * DT) ** (n[mask] - 1) * B * DT; return torch.tensor(W, dtype=torch.float32)

def brackets_np(X0, XI):
    """Pathwise first-order conditions: u = -(1/R) sum_j W[j,k] Q (x_j - target_j)."""
    W0, Wi = kernel(A0, B0).numpy().astype(np.float64), kernel(A_, B_).numpy().astype(np.float64)
    g0 = Q0 * (X0[:, :-1] - ETA0); gi = Q_ * (XI[:, :-1] - H1 * X0[:, :-1] - ETA)   # cost integrand on [t_j, t_{j+1}) uses x_j
    return -(g0 @ W0) / R0, -(gi @ Wi) / R_

# ---------------- npg model ----------------
class FCModel:
    n_players = 2; N = N; DT = DT; d_now = [0, 0]; d_seq = [1, 2]
    d_full = 2; d_w = 2; cv_gamma = [BETA * (1 + A0 * DT), BETA * (1 + A_ * DT)]       # bracket kernel w(j,k) = c gamma^{j-k}
    def full_obs(s, exo, st): return torch.stack([st['x0'][:, :-1], st['xi'][:, :-1]], -1)
    def dW(s, exo): return torch.stack([exo['dw0'], exo['dwi']], -1) / math.sqrt(DT)
    def __init__(s): s.W0 = kernel(A0, B0); s.Wi = kernel(A_, B_)
    def exo(s, m, seed): return {k: torch.tensor(v, dtype=torch.float32) for k, v in simulate_noise(m, seed).items()}
    def init_state(s, m): return dict(x0=torch.zeros(m, N + 1), xi=torch.zeros(m, N + 1), _init=False)
    def step(s, st, u, k):
        if not st['_init']: st['_init'] = True
        x0, xi = st['x0'][:, k], st['xi'][:, k]
        st['x0'][:, k + 1] = x0 + (A0 * x0 + B0 * u[0]) * DT + D0 * s._e['dw0'][:, k]
        st['xi'][:, k + 1] = xi + (A_ * xi + G_ * x0 + B_ * u[1]) * DT + D_ * s._e['dwi'][:, k]
    def obs_now(s, i, exo, st, k):
        if k == 0 and not st['_init']: s._e = exo; st['x0'][:, 0] = exo['x0_0']; st['xi'][:, 0] = exo['xi_0']
        return torch.zeros(st['x0'].shape[0], 0)
    def obs_inc(s, i, exo, st, k):                       # observation increment over [k, k+1], scaled by 1/sqrt(dt)
        x0, xi = st['x0'][:, k], st['xi'][:, k]
        if i == 0: return ((L0 * x0 * DT + SV0 * exo['dv0'][:, k]) / math.sqrt(DT))[:, None]
        return (torch.stack([L1 * xi, L2 * x0], -1) * DT + torch.tensor([SV, SV2]) * exo['dvi'][:, k]) / math.sqrt(DT)
    def cost(s, i, exo, st, ctrl):                        # realised discounted cost of player i (lower is better)
        disc = torch.tensor(np.exp(-RHO * tg[:-1]), dtype=torch.float32)
        if i == 0: return ((Q0 * (st['x0'][:, :-1] - ETA0) ** 2 + R0 * ctrl[0] ** 2) * disc).sum(1) * DT
        return ((Q_ * (st['xi'][:, :-1] - H1 * st['x0'][:, :-1] - ETA) ** 2 + R_ * ctrl[1] ** 2) * disc).sum(1) * DT
    def bracket(s, i, exo, st, ctrl):
        if i == 0: return -(Q0 * (st['x0'][:, :-1] - ETA0)) @ s.W0 / R0
        return -(Q_ * (st['xi'][:, :-1] - H1 * st['x0'][:, :-1] - ETA)) @ s.Wi / R_

if __name__ == '__main__':
    ref = reference(); nz = simulate_noise(int(os.environ.get('M_CHECK', '2048')), 7); u0_ex, ui_ex, X0, XI, XH0, ZH = reference_paths(ref, nz)
    print(f'NOISE={NOISE} SV2={SV2} Q0={Q0} A0={A0} N={N} | reference scales: |u0| {np.abs(u0_ex).mean():.3f} |ui| {np.abs(ui_ex).mean():.3f} | Gain0(0) {ref["Gain0"][0,0]:.3f} Gain1(0) {ref["Gain1"][0].round(3)} K0(0) {ref["K0"][0]:.3f}', flush=True)
    if os.environ.get('CHECK_REF'):
        # FOC residual: regress the pathwise bracket on the player's own filter state (sufficient statistic) and compare with the control
        b0, bi = brackets_np(X0, XI); rel = lambda a, b: np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean())
        f0 = np.zeros_like(u0_ex); fi = np.zeros_like(ui_ex)
        for k in range(N):
            F = np.stack([np.ones(len(X0)), XH0[:, k]], -1); f0[:, k] = F @ np.linalg.lstsq(F, b0[:, k], rcond=1e-8)[0]
            F = np.concatenate([np.ones((len(X0), 1)), ZH[:, k]], -1); fi[:, k] = F @ np.linalg.lstsq(F, bi[:, k], rcond=1e-8)[0]
        print(f'FOC check (E[bracket|filter] vs control): major {rel(f0, u0_ex):.4f} minor {rel(fi, ui_ex):.4f}')
        print(f'naive minor (ignores the estimate-of-estimate: regress on (xhat_i, xhat_0) only): ', end='')
        for k in range(N):
            F = np.concatenate([np.ones((len(X0), 1)), ZH[:, k, :2]], -1); fi[:, k] = F @ np.linalg.lstsq(F, bi[:, k], rcond=1e-8)[0]
        print(f'{rel(fi, ui_ex):.4f}'); sys.exit()
    M = FCModel(); S = Solver(M, hid=32, rho=float(os.environ.get('RHO_DAMP', '0.7')), lr=1e-3, batch=int(os.environ.get('BATCH', '1024')), steps=int(os.environ.get('STEPS', '1500')),
                              m_paths=int(os.environ.get('M_PATHS', '8192')), m_val=2048)
    u0_ex, ui_ex, *_ = reference_paths(ref, {k: v.numpy().astype(np.float64) for k, v in S.exo_v.items()})
    if os.environ.get('EXPLOIT'):
        from npg import exploitability, foc_residual
        S.run(int(os.environ['EXPLOIT']), ref=[torch.tensor(u0_ex, dtype=torch.float32), torch.tensor(ui_ex, dtype=torch.float32)])
        if os.environ.get('FOC_CHECK'):
            for i in range(2): foc_residual(S, i)
        for i in range(2): g, J0 = exploitability(S, i, iters=int(os.environ.get('EXPLOIT_ITERS', '6'))); print(f'player {i}: exploitability {g:+.5f} relative to cost {J0:.4f} ({100 * g / J0:+.2f}%)', flush=True)
        sys.exit()
    vstate, vctrl, hist = S.run(int(sys.argv[1]) if len(sys.argv) > 1 else 10, ref=[torch.tensor(u0_ex, dtype=torch.float32), torch.tensor(ui_ex, dtype=torch.float32)])
    np.savez(os.environ.get('SAVE', '../results/fc_last.npz'), t=tg[:-1], u0=vctrl[0].numpy(), u0_ex=u0_ex, ui=vctrl[1].numpy(), ui_ex=ui_ex, hist=np.array(hist))
