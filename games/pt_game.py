"""Two traders with PRIVATE signals and a common price (two-way private information; the belief hierarchy does not close).
    dalpha_i = -kappa alpha_i dt + sa dW^i      (private to trader i)
    dS = (alpha_1 + alpha_2 + lam (nu_1 + nu_2)) dt + sig dW^S,   dq_i = nu_i dt
    performance_i = int_0^T q_i mu_t dt - a int nu_i^2 dt - phi int q_i^2 dt - psi q_i(T)^2,   mu = alpha_1 + alpha_2 + lam (nu_1 + nu_2)
    First-order condition:  2a nu_i(t) = E[ -2 psi q_i(T) + int_t^T (mu_s - 2 phi q_i(s)) ds | F^i_t ],   F^i = sigma(alpha_i, S).
Information modes (INFO env):  private (default): trader i sees alpha_i and S;  full: both see both signals (exact LQ Nash by linear Picard).
Exact reference at lam = 0 (game decouples): single-agent LQG with state (q_i, alpha_i, alphahat_j), alphahat_j = Kalman estimate of the
other signal from dS - alpha_i dt. For lam > 0 private: no closed form; use exploitability (npg.exploitability) and the two limits.
Usage: python3 pt_game.py [iters]  (env: LAM, A_COST, INFO, M_PATHS, STEPS, BATCH, N_STEPS, CV, EXPLOIT, CHECK_REF)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import os, sys, math, time, numpy as np, torch
from npg import Solver, exploitability, foc_residual

T = 1.0; N = int(os.environ.get('N_STEPS', '50')); DT = T / N; tg = np.linspace(0, T, N + 1)
LAM = float(os.environ.get('LAM', '1.0')); A_ = float(os.environ.get('A_COST', '0.5')); PHI, PSI = 0.1, 1.0
KAP, SA, SIG = 1.0, 1.0, 0.5; VA0 = SA ** 2 / (2 * KAP)                      # stationary signal variance
INFO = os.environ.get('INFO', 'private')

def simulate_noise(m, seed):
    rng = np.random.default_rng(seed)
    return dict(a1_0=rng.standard_normal(m) * math.sqrt(VA0), a2_0=rng.standard_normal(m) * math.sqrt(VA0),
                dw1=rng.standard_normal((m, N)) * math.sqrt(DT), dw2=rng.standard_normal((m, N)) * math.sqrt(DT), dws=rng.standard_normal((m, N)) * math.sqrt(DT))

def signals(nz):
    m = nz['a1_0'].shape[0]; a1 = np.zeros((m, N + 1)); a2 = np.zeros((m, N + 1)); a1[:, 0], a2[:, 0] = nz['a1_0'], nz['a2_0']
    for k in range(N):
        a1[:, k + 1] = a1[:, k] - KAP * a1[:, k] * DT + SA * nz['dw1'][:, k]; a2[:, k + 1] = a2[:, k] - KAP * a2[:, k] * DT + SA * nz['dw2'][:, k]
    return a1, a2

# ---------------- exact reference at lam = 0 (discrete-time LQG, trader i alone with state z = (q_i, alpha_i, alphahat_j)) ----------------
def reference_lam0():
    assert LAM == 0.0
    # Kalman for alpha_j from dY = alpha_j dt + sig dW^S (prior estimates)
    V = VA0; K = np.zeros(N); Vs = np.zeros(N + 1); Vs[0] = V
    for k in range(N):
        K[k] = V / (V * DT + SIG ** 2); Vp = V - K[k] * DT * V; V = (1 - KAP * DT) ** 2 * Vp + SA ** 2 * DT; Vs[k + 1] = V
    # z_{k+1} = F_k z_k + G nu_k + noise;  alphahat_{k+1} = (1 - kappa dt)[alphahat + K (dY - alphahat dt)],  dY = alpha_j dt + sig dW
    Fz = lambda k: np.array([[1.0, 0, 0], [0, 1 - KAP * DT, 0], [0, 0, (1 - KAP * DT) * (1 - K[k] * DT)]])
    Gz = np.array([DT, 0.0, 0.0]); cz = lambda k: np.zeros(3)
    # cost to MINIMISE: -(q (alpha_i + alphahat_j)) + phi q^2 + a nu^2 per unit time, terminal psi q^2
    Qz = np.array([[PHI, -0.5, -0.5], [-0.5, 0, 0], [-0.5, 0, 0]]); qz = np.zeros(3)
    # dlqg has zero terminal weight; fold psi q_T^2 in by extending: use a modified terminal via one extra fictitious step is messy -> implement directly
    P = np.diag([PSI, 0.0, 0.0]); sv = np.zeros(3); Gain = np.zeros((N, 3))
    for k in range(N - 1, -1, -1):
        Fk = Fz(k); Mi = 1.0 / (A_ * DT + Gz @ P @ Gz); Gain[k] = Mi * (Gz @ P @ Fk)
        P = Qz * DT + Fk.T @ P @ Fk - np.outer(Fk.T @ P @ Gz, Gz @ P @ Fk) * Mi
    return dict(K=K, Gain=Gain)

def reference_paths_lam0(ref, nz):
    a1, a2 = signals(nz); m = a1.shape[0]; q = np.zeros((m, 2)); ah = np.zeros((m, 2)); nu = np.zeros((m, 2, N))
    for k in range(N):
        for i, j in [(0, 1), (1, 0)]:
            ai = a1[:, k] if i == 0 else a2[:, k]
            nu[:, i, k] = -(ref['Gain'][k] @ np.stack([q[:, i], ai, ah[:, i]]))
        dY = np.stack([a2[:, k], a1[:, k]], -1) * DT + SIG * nz['dws'][:, k, None]      # trader i's observation of alpha_j (lam = 0)
        ah = (1 - KAP * DT) * (ah + ref['K'][k] * (dY - ah * DT)); q = q + nu[:, :, k] * DT
    return nu

# ---------------- npg model ----------------
class PTModel:
    n_players = 2; N = N; DT = DT
    d_now = [3, 3] if INFO == 'private' else [4, 4]; d_seq = [2, 2] if INFO == 'private' else [0, 0]
    d_full = 5; d_w = 3; cv_gamma = [1.0, 1.0]
    def __init__(s): j, k = np.indices((N, N)); s.W = torch.tensor(np.where(j > k, DT, 0.0), dtype=torch.float32)   # integral over s > t
    def exo(s, m, seed):
        nz = simulate_noise(m, seed); a1, a2 = signals(nz)
        return dict(a1=torch.tensor(a1, dtype=torch.float32), a2=torch.tensor(a2, dtype=torch.float32), dws=torch.tensor(nz['dws'], dtype=torch.float32), _nz=nz)
    def init_state(s, m): return dict(q1=torch.zeros(m, N + 1), q2=torch.zeros(m, N + 1), S=torch.zeros(m, N + 1), mu=torch.zeros(m, N))
    def step(s, st, u, k):
        mu = s._e['a1'][:, k] + s._e['a2'][:, k] + LAM * (u[0] + u[1]); st['mu'][:, k] = mu
        st['q1'][:, k + 1] = st['q1'][:, k] + u[0] * DT; st['q2'][:, k + 1] = st['q2'][:, k] + u[1] * DT; st['S'][:, k + 1] = st['S'][:, k] + mu * DT + SIG * s._e['dws'][:, k]
    def obs_now(s, i, exo, st, k):
        s._e = exo; q = st['q1'] if i == 0 else st['q2']; a = exo['a1'] if i == 0 else exo['a2']
        if INFO == 'private': return torch.stack([a[:, k], q[:, k], st['S'][:, k]], -1)
        return torch.stack([exo['a1'][:, k], exo['a2'][:, k], st['q1'][:, k], st['q2'][:, k]], -1)
    def obs_inc(s, i, exo, st, k):                       # private: own signal increment and price increment, scaled
        a = exo['a1'] if i == 0 else exo['a2']
        return torch.stack([(a[:, k + 1] - a[:, k]) / math.sqrt(DT), (st['S'][:, k + 1] - st['S'][:, k]) / math.sqrt(DT)], -1)
    def bracket(s, i, exo, st, ctrl):
        q = st['q1'] if i == 0 else st['q2']
        return (-2 * PSI * q[:, -1:] + (st['mu'] - 2 * PHI * q[:, :-1]) @ s.W) / (2 * A_)
    def cost(s, i, exo, st, ctrl):                        # minus realised performance (lower is better)
        q = st['q1'] if i == 0 else st['q2']
        return -((q[:, :-1] * st['mu'] - A_ * ctrl[i] ** 2 - PHI * q[:, :-1] ** 2).sum(1) * DT - PSI * q[:, -1] ** 2)
    def full_obs(s, exo, st): return torch.stack([exo['a1'][:, :-1], exo['a2'][:, :-1], st['q1'][:, :-1], st['q2'][:, :-1], st['S'][:, :-1]], -1)
    def dW(s, exo):
        nz = exo['_nz']; return torch.tensor(np.stack([nz['dw1'], nz['dw2'], nz['dws']], -1) / math.sqrt(DT), dtype=torch.float32)

def linear_picard_full(exo, iters=30, rho=0.5):
    """Full-information LQ Nash by linear regression Picard on features (1, a1, a2, q1, q2), float64 (exact for LQ up to grid)."""
    M = PTModel(); m = exo['a1'].shape[0]; a1, a2 = exo['a1'].numpy().astype(np.float64), exo['a2'].numpy().astype(np.float64); dws = exo['dws'].numpy().astype(np.float64)
    coef = [np.zeros((N, 5)) for _ in range(2)]; W = M.W.numpy().astype(np.float64)
    for it in range(iters):
        q = np.zeros((m, 2, N + 1)); nu = np.zeros((m, 2, N)); mu = np.zeros((m, N))
        for k in range(N):
            feats = np.stack([np.ones(m), a1[:, k], a2[:, k], q[:, 0, k], q[:, 1, k]], -1)
            for i in range(2): nu[:, i, k] = feats @ coef[i][k]
            mu[:, k] = a1[:, k] + a2[:, k] + LAM * (nu[:, 0, k] + nu[:, 1, k]); q[:, :, k + 1] = q[:, :, k] + nu[:, :, k] * DT
        new = []
        for i in range(2):
            br = (-2 * PSI * q[:, i, -1:] + (mu - 2 * PHI * q[:, i, :-1]) @ W) / (2 * A_); tgt = (1 - rho) * nu[:, i] + rho * br; c = np.zeros((N, 5))
            for k in range(N):
                feats = np.stack([np.ones(m), a1[:, k], a2[:, k], q[:, 0, k], q[:, 1, k]], -1); c[k] = np.linalg.lstsq(feats, tgt[:, k], rcond=1e-6)[0]
            new.append(c)
        chg = max(np.abs(new[i] - coef[i]).max() for i in range(2)); coef = new
        if chg < 1e-6: break
    return nu, chg

if __name__ == '__main__':
    ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    M = PTModel(); S = Solver(M, hid=32, rho=float(os.environ.get('RHO_DAMP', '0.6')), batch=int(os.environ.get('BATCH', '1024')), steps=int(os.environ.get('STEPS', '1000')),
                              m_paths=int(os.environ.get('M_PATHS', '8192')), m_val=2048, seed=int(os.environ.get('SEED', '0')))
    ref = None
    if LAM == 0.0:
        r = reference_lam0(); nu_ex = reference_paths_lam0(r, S.exo_v['_nz']); ref = [torch.tensor(nu_ex[:, i], dtype=torch.float32) for i in range(2)]
        print(f'lam=0 exact reference: |nu| {np.abs(nu_ex).mean():.3f}', flush=True)
    elif INFO == 'full':
        nu_ex, chg = linear_picard_full(S.exo_v); ref = [torch.tensor(nu_ex[:, i], dtype=torch.float32) for i in range(2)]
        print(f'full-info linear Picard reference: |nu| {np.abs(nu_ex).mean():.3f}, final change {chg:.1e}, |lam nu|/|alpha| {LAM * np.abs(nu_ex).mean() / np.abs(S.exo_v["a1"].numpy()).mean():.2f}', flush=True)
    print(f'INFO={INFO} LAM={LAM} a={A_} N={N} M={S.m}', flush=True)
    vstate, vctrl, hist = S.run(ITERS, ref=ref)
    if os.environ.get('FOC_CHECK'):
        for i in range(2): foc_residual(S, i)
    if os.environ.get('EXPLOIT'):
        for i in range(2): g, J0 = exploitability(S, i, iters=int(os.environ.get('EXPLOIT_ITERS', '6'))); print(f'player {i}: exploitability {g:+.5f} relative to |cost| {abs(J0):.4f} ({100 * g / abs(J0):+.2f}%)', flush=True)
    np.savez(os.environ.get('SAVE', '../results/pt_last.npz'), t=tg[:-1], nu1=vctrl[0].numpy(), nu2=vctrl[1].numpy(), a1=S.exo_v['a1'].numpy(), a2=S.exo_v['a2'].numpy(), q1=vstate['q1'].numpy(), q2=vstate['q2'].numpy(), nu_ex=(ref[0].numpy() if ref else np.zeros(1)), hist=np.array(hist))
