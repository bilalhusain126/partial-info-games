"""Two traders with PRIVATE MARKOV-CHAIN signals and a common price with impact (the non-Gaussian version of pt_game.py).
    alpha_i in {-A, +A}, a continuous-time Markov chain with symmetric switching rate KAP, private to trader i; the two chains are independent
    dS = (alpha_1 + alpha_2 + lam (nu_1 + nu_2)) dt + sig dW^S,   dq_i = nu_i dt
    performance_i = int q_i mu dt - a int nu_i^2 dt - phi int q_i^2 dt - psi q_i(T)^2,   mu = alpha_1 + alpha_2 + lam (nu_1 + nu_2)
    First-order condition: 2a nu_i(t) = E[ lam q_i(t) - 2 psi q_i(T) + int_t^T (mu_s - 2 phi q_i(s)) ds | F^i_t ],  F^i = sigma(alpha_i, S).
Nothing is Gaussian: trader i's conditional expectation of alpha_j given the price path is the Wonham filter, nonlinear in the observations,
so the linear-strategy fixed point of pt_kernel.py does not apply. The learned solver is unchanged (the bracket does not depend on the signal law).
Exact reference at lam = 0 (decoupled): trader i alone with signal m = alpha_i + E[alpha_j | F^i]; the discrete-time problem is linear-quadratic in
the inventory with an exogenous adapted signal, solved by a backward recursion whose random coefficient is a sum of conditional expectations of the
future signal, E[alpha(t_l) | F(t_k)] = e^{-2 KAP (t_l - t_k)} E[alpha(t_k) | F(t_k)]; the filter is the exact discrete Bayes (HMM) filter.
Usage: LAM=1 python3 pm_game.py [iters]   (env as pt_game: LAM, A_COST, M_PATHS, STEPS, BATCH, N_STEPS, CV, CV_JUMP, FOC_CHECK, GATEAUX, SEED, SAVE)"""
import os, sys, math, time, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from npg import Solver, exploitability, foc_residual, gateaux_check

T = 1.0; N = int(os.environ.get('N_STEPS', '50')); DT = T / N; tg = np.linspace(0, T, N + 1)
LAM = float(os.environ.get('LAM', '1.0')); A_ = float(os.environ.get('A_COST', '0.5')); PHI, PSI = 0.1, 1.0
KAP, AMP, SIG = 1.0, 1.0, 0.5                       # switching rate, chain amplitude (states -AMP, +AMP), price noise
P_STAY = 0.5 * (1 + math.exp(-2 * KAP * DT))        # exact transition probability of staying over one step (two-state symmetric chain)
CV_JUMP = bool(int(os.environ.get('CV_JUMP', '0')))  # add the compensated chain increments to the control variate

def simulate_noise(m, seed):
    rng = np.random.default_rng(seed)
    return dict(s1_0=rng.integers(0, 2, m), s2_0=rng.integers(0, 2, m), u1=rng.random((m, N)), u2=rng.random((m, N)), dws=rng.standard_normal((m, N)) * math.sqrt(DT))

def signals(nz):
    """Chain states as +-AMP paths (m, N+1), simulated exactly with the one-step transition probability."""
    m = nz['s1_0'].shape[0]; out = []
    for s0, u in [(nz['s1_0'], nz['u1']), (nz['s2_0'], nz['u2'])]:
        s = np.zeros((m, N + 1), dtype=int); s[:, 0] = s0
        for k in range(N): s[:, k + 1] = np.where(u[:, k] < P_STAY, s[:, k], 1 - s[:, k])
        out.append(AMP * (2 * s - 1).astype(float))
    return out[0], out[1]

# ---------------- exact reference at lam = 0 ----------------
def hmm_filter(dY, a_self=None):
    """Exact discrete Bayes filter for the other trader's chain from observation increments dY = alpha_j dt + sig dW (m, N); returns
    pi_k = P(alpha_j(t_k) = +AMP | increments before step k) for k = 0..N-1 (prior at k, before seeing dY_k)."""
    m = dY.shape[0]; pi = np.full(m, 0.5); out = np.zeros((m, N))
    for k in range(N):
        out[:, k] = pi
        lp = np.exp(-(dY[:, k] - AMP * DT) ** 2 / (2 * SIG ** 2 * DT)); lm = np.exp(-(dY[:, k] + AMP * DT) ** 2 / (2 * SIG ** 2 * DT))
        post = pi * lp / (pi * lp + (1 - pi) * lm)                       # update with the increment at step k
        pi = P_STAY * post + (1 - P_STAY) * (1 - post)                   # predict one step
    return out

def reference_lam0():
    """Backward recursion for the single-trader problem with an exogenous adapted signal m_k (cost = sum [-q m + a nu^2 + phi q^2] dt + psi q_N^2):
    V_k(q) = A_k q^2 + B_k q + C_k with B_k random; nu_k = -(2 A_{k+1} q + E[B_{k+1} | F_k]) / (2a + 2 A_{k+1} dt).
    B_k = -m_k dt + E[B_{k+1}|F_k] (1 - A_{k+1} dt / (a + A_{k+1} dt)). With E[m_l | F_k] = e^{-2 KAP (l-k) dt} mhat_k, E[B_{k+1}|F_k] = -w_k mhat_k
    for deterministic weights w_k computed backward."""
    A = np.zeros(N + 1); A[N] = PSI; w = np.zeros(N + 1); D = np.zeros(N)   # w_k: E[B_k | F_{k-1}] = -w_k * E[m_k | F_{k-1}] /e^{-2KAP dt} handled below
    # Let B_k = -sum_{l>=k} c_{k,l} m_l with E[m_l|F_k] = r^{l-k} m^hat_k, r = e^{-2 KAP dt}. Then E[B_{k+1}|F_k] = -S_{k+1} r mhat_k where S_{k+1} = sum_{l>=k+1} c_{k+1,l} r^{l-k-1}.
    r = math.exp(-2 * KAP * DT); S = np.zeros(N + 1); G = np.zeros(N); g = np.zeros(N)
    for k in range(N - 1, -1, -1):
        den = 2 * A_ + 2 * A[k + 1] * DT
        G[k] = 2 * A[k + 1] / den; g[k] = -S[k + 1] * r / den                            # nu_k = -G_k q_k - g_k * mhat_k  (sign: E[B_{k+1}|F_k] = -S r mhat)
        A[k] = PHI * DT + A[k + 1] - (2 * A[k + 1]) ** 2 * DT / (2 * den)              # A_k = phi dt + A_{k+1} - A_{k+1}^2 dt / (a + A_{k+1} dt)
        S[k] = DT + S[k + 1] * r * (1 - A[k + 1] * DT / (A_ + A[k + 1] * DT))           # c_{k,k} = dt; c_{k,l} = c_{k+1,l} (1 - A dt/(a + A dt)) for l > k
    return dict(G=G, g=g, A=A, S=S)

def reference_paths_lam0(ref, nz):
    a1, a2 = signals(nz); m = a1.shape[0]
    dY1 = a2[:, :-1] * DT + SIG * nz['dws']; dY2 = a1[:, :-1] * DT + SIG * nz['dws']                 # what trader 1 / 2 sees of the other (lam = 0)
    pi1 = hmm_filter(dY1); pi2 = hmm_filter(dY2); ah1 = AMP * (2 * pi1 - 1); ah2 = AMP * (2 * pi2 - 1)
    nu = np.zeros((m, 2, N)); q = np.zeros((m, 2))
    for k in range(N):
        mh = np.stack([a1[:, k] + ah1[:, k], a2[:, k] + ah2[:, k]], 1)
        nu[:, :, k] = -ref['G'][k] * q - ref['g'][k] * mh; q = q + nu[:, :, k] * DT
    return nu

# ---------------- npg model (identical to pt_game except the signals and the control-variate noise) ----------------
class PMModel:
    n_players = 2; N = N; DT = DT; d_now = [3, 3]; d_seq = [2, 2]
    d_full = 5; d_w = 3 if CV_JUMP else 1; cv_gamma = [1.0, 1.0]
    def __init__(s): j, k = np.indices((N, N)); s.W = torch.tensor(np.where(j > k, DT, 0.0), dtype=torch.float32)
    def exo(s, m, seed):
        nz = simulate_noise(m, seed); a1, a2 = signals(nz)
        return dict(a1=torch.tensor(a1, dtype=torch.float32), a2=torch.tensor(a2, dtype=torch.float32), dws=torch.tensor(nz['dws'], dtype=torch.float32), _nz=nz)
    def init_state(s, m): return dict(q1=torch.zeros(m, N + 1), q2=torch.zeros(m, N + 1), S=torch.zeros(m, N + 1), mu=torch.zeros(m, N))
    def step(s, st, u, k):
        mu = s._e['a1'][:, k] + s._e['a2'][:, k] + LAM * (u[0] + u[1]); st['mu'][:, k] = mu
        st['q1'][:, k + 1] = st['q1'][:, k] + u[0] * DT; st['q2'][:, k + 1] = st['q2'][:, k] + u[1] * DT; st['S'][:, k + 1] = st['S'][:, k] + mu * DT + SIG * s._e['dws'][:, k]
    def obs_now(s, i, exo, st, k):
        s._e = exo; q = st['q1'] if i == 0 else st['q2']; a = exo['a1'] if i == 0 else exo['a2']
        return torch.stack([a[:, k], q[:, k], st['S'][:, k]], -1)
    def obs_inc(s, i, exo, st, k):
        a = exo['a1'] if i == 0 else exo['a2']
        return torch.stack([(a[:, k + 1] - a[:, k]), (st['S'][:, k + 1] - st['S'][:, k]) / math.sqrt(DT)], -1)
    def bracket(s, i, exo, st, ctrl):
        q = st['q1'] if i == 0 else st['q2']
        return (LAM * q[:, :-1] - 2 * PSI * q[:, -1:] + (st['mu'] - 2 * PHI * q[:, :-1]) @ s.W) / (2 * A_)
    def target_weight(s, i): return torch.full((N,), 2 * A_ * DT)
    def cost(s, i, exo, st, ctrl):
        q = st['q1'] if i == 0 else st['q2']
        return -((q[:, :-1] * st['mu'] - A_ * ctrl[i] ** 2 - PHI * q[:, :-1] ** 2).sum(1) * DT - PSI * q[:, -1] ** 2)
    def full_obs(s, exo, st): return torch.stack([exo['a1'][:, :-1], exo['a2'][:, :-1], st['q1'][:, :-1], st['q2'][:, :-1], st['S'][:, :-1]], -1)
    def dW(s, exo):
        """Martingale increments for the control variate: the price noise and, with CV_JUMP=1 (default 0), the compensated chain
        increments alpha(t_{k+1}) - r alpha(t_k), r = e^{-2 KAP dt}, which have zero mean given the full information at t_k."""
        out = [exo['dws'] / math.sqrt(DT)]
        if CV_JUMP:
            r = math.exp(-2 * KAP * DT); out += [(a[:, 1:] - r * a[:, :-1]) / math.sqrt(DT) for a in (exo['a1'], exo['a2'])]
        return torch.stack(out, -1)

if __name__ == '__main__':
    ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    M = PMModel(); S = Solver(M, hid=int(os.environ.get('HID', '32')), rho=float(os.environ.get('RHO_DAMP', '0.6')), batch=int(os.environ.get('BATCH', '1024')), steps=int(os.environ.get('STEPS', '1000')),
                              m_paths=int(os.environ.get('M_PATHS', '8192')), m_val=2048, seed=int(os.environ.get('SEED', '0')))
    ref = None
    if LAM == 0.0:
        r = reference_lam0(); nu_ex = reference_paths_lam0(r, S.exo_v['_nz']); ref = [torch.tensor(nu_ex[:, i], dtype=torch.float32) for i in range(2)]
        print(f'lam=0 exact reference (HMM filter + backward recursion): |nu| {np.abs(nu_ex).mean():.3f}', flush=True)
    print(f'Markov-chain signals: LAM={LAM} a={A_} N={N} M={S.m} KAP={KAP} AMP={AMP}', flush=True)
    vstate, vctrl, hist = S.run(ITERS, ref=ref)
    if os.environ.get('FOC_CHECK'):
        for i in range(2): foc_residual(S, i)
    if os.environ.get('GATEAUX'):
        for i in range(2): gateaux_check(S, i)
    np.savez(os.environ.get('SAVE', os.path.join(os.environ.get('RESULTS_DATA', os.path.expanduser('~/Desktop/Bilal Saleh Husain/research-data/partial-info-games')), f'pm_last.npz')), t=tg[:-1], nu1=vctrl[0].cpu().numpy(), nu2=vctrl[1].cpu().numpy(), nu_ex=(ref[0].cpu().numpy() if ref else np.zeros(1)), hist=np.array(hist))
