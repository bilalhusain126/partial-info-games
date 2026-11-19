"""Independent checks of games/pm_game.py's lam = 0 exact reference (read-only import of the repo file)."""
import os, sys, math, numpy as np
os.environ['LAM'] = '0'; os.environ['DEVICE'] = 'cpu'
sys.path.insert(0, '/Users/bhusain/Desktop/Bilal Saleh Husain/Academics and Research/Repos/partial-info-games/games')
import pm_game as P
N, DT, KAP, AMP, SIG, A_, PHI, PSI = P.N, P.DT, P.KAP, P.AMP, P.SIG, P.A_, P.PHI, P.PSI
rel = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))

# ---- 1. chain: transition probability and the decay of the conditional mean
Pmat = np.array([[P.P_STAY, 1 - P.P_STAY], [1 - P.P_STAY, P.P_STAY]])
C = KAP * np.array([[-1.0, 1.0], [1.0, -1.0]])
from scipy.linalg import expm
print('1. P_STAY vs expm(C dt):', P.P_STAY, expm(C * DT)[0, 0])
m = 200000; nz = P.simulate_noise(m, 3); a1, a2 = P.signals(nz)
for l in (1, 5, 20):
    emp = (a1[:, l] * a1[:, 0]).mean() / (a1[:, 0] ** 2).mean()
    print(f'   E[alpha_l alpha_0]/E[alpha_0^2] at l={l}: empirical {emp:.4f} vs e^(-2 kap l dt) {math.exp(-2 * KAP * l * DT):.4f}')
print('   stationary start: mean alpha_0', a1[:, 0].mean(), ' P(+)', (a1[:, 0] > 0).mean())

# ---- 2. HMM filter: pi_k should be P(alpha_j(t_k) = + | dY_0..dY_{k-1}); calibration test E[alpha_j(t_k) | pi_k]
dY1 = a2[:, :-1] * DT + SIG * nz['dws']; pi1 = P.hmm_filter(dY1); ah1 = AMP * (2 * pi1 - 1)
for k in (1, 10, 25, 49):
    c = np.polyfit(ah1[:, k], a2[:, k], 1)
    # bin calibration
    bins = np.quantile(ah1[:, k], np.linspace(0, 1, 11)); idx = np.clip(np.searchsorted(bins, ah1[:, k]) - 1, 0, 9)
    cal = max(abs(a2[idx == b, k].mean() - ah1[idx == b, k].mean()) for b in range(10))
    print(f'2. k={k}: regression of alpha_j on ahat: slope {c[0]:.4f} intercept {c[1]:+.4f}; max decile calibration gap {cal:.4f}; std(ahat) {ah1[:, k].std():.3f}')
# brute-force check of the filter on one path by enumerating chain paths over a short horizon (K steps)
K = 8; dY = dY1[0, :K]; sp = a2[0, :K]
states = np.array(np.meshgrid(*[[0, 1]] * K, indexing='ij')).reshape(K, -1).T   # all 2^K paths
logp = np.log(0.5) * np.ones(states.shape[0])
for k in range(1, K): logp += np.log(np.where(states[:, k] == states[:, k - 1], P.P_STAY, 1 - P.P_STAY))
alpha = AMP * (2 * states - 1)
for k in range(K - 1):   # likelihood of dY_0..dY_{k-1}, then P(state_k = 1 | ...)
    ll = logp.copy()
    for j in range(k): ll += -(dY[j] - alpha[:, j] * DT) ** 2 / (2 * SIG ** 2 * DT)
    w = np.exp(ll - ll.max()); post = (w * (states[:, k] == 1)).sum() / w.sum()
    print(f'   brute-force pi_{k} = {post:.6f}  code {pi1[0, k]:.6f}')

# ---- 3. backward recursion vs an independent deterministic LQ solve (certainty equivalence)
ref = P.reference_lam0(); r = math.exp(-2 * KAP * DT)
def det_opt(k0, q0, mpath):
    """Minimise sum_{k>=k0} [-q_k m_k + a nu_k^2 + phi q_k^2] dt + psi q_N^2 over nu_{k0..N-1} given q_{k0} = q0 (q_k left endpoint); returns nu_{k0}."""
    n = N - k0; L = np.tril(np.ones((n + 1, n)), -1) * DT          # q_{k0+j} = q0 + sum_{l<j} nu_l dt, j = 0..n
    qv = lambda nu: q0 + L @ nu
    # cost = -sum_{j<n} q_j m_j dt + a |nu|^2 dt + phi sum_{j<n} q_j^2 dt + psi q_n^2  -> quadratic in nu; solve by normal equations
    Dm = np.diag(np.r_[np.full(n, DT), 0.0]); Dphi = np.diag(np.r_[np.full(n, PHI * DT), PSI])
    H = 2 * (A_ * DT * np.eye(n) + L.T @ Dphi @ L); g = -L.T @ Dm @ np.r_[mpath, 0.0] + 2 * L.T @ Dphi @ (q0 * np.ones(n + 1))
    return np.linalg.solve(H, -g)[0]
for k0 in (0, 10, 30, 48):
    mh = 0.7; q0 = 0.4; mpath = mh * r ** np.arange(N - k0)   # forecast E[m_l | F_k] = r^{l-k} mhat
    nu_rec = -ref['G'][k0] * q0 - ref['g'][k0] * mh; nu_det = det_opt(k0, q0, mpath)
    print(f'3. k={k0}: recursion nu {nu_rec:+.6f}  deterministic LQ with forecast path {nu_det:+.6f}  (G {ref["G"][k0]:.4f}, -g {-ref["g"][k0]:.4f})')

# ---- 4. FOC check of the reference on simulated paths: 2a nu_k = E[-2 psi q_N + sum_{l>k} (m_l - 2 phi q_l) dt | F_k] with m = a_i + a_j (true), regressed on (1, a_i, q, ahat)
nu = P.reference_paths_lam0(ref, nz); q = np.concatenate([np.zeros((m, 2, 1)), np.cumsum(nu, 2) * DT], 2)
W = np.where(np.indices((N, N))[0] > np.indices((N, N))[1], DT, 0.0); mu = (a1 + a2)[:, :-1]
for i in (0, 1):
    br = (-2 * PSI * q[:, i, -1:] + (mu - 2 * PHI * q[:, i, :-1]) @ W) / (2 * A_); ai = (a1 if i == 0 else a2)[:, :-1]
    ah = ah1 if i == 0 else AMP * (2 * P.hmm_filter(a1[:, :-1] * DT + SIG * nz['dws']) - 1)
    fit = np.zeros_like(br)
    for k in range(N):
        F = np.stack([np.ones(m), ai[:, k], q[:, i, k], ah[:, k]], -1); fit[:, k] = F @ np.linalg.lstsq(F, br[:, k], rcond=None)[0]
    print(f'4. trader {i+1}: E[bracket | 1, a_i, q_i, ahat_j] (linear regression) vs reference nu: rel L2 {rel(fit, nu[:, i]):.4f}; mean |nu| {np.abs(nu[:, i]).mean():.4f}')
    # pathwise Gateaux at the reference along admissible directions: random linear functions of (a_i, q, ahat) -> derivative of the mean cost should be ~0
    def cost(nuv):
        qq = np.concatenate([np.zeros((m, 1)), np.cumsum(nuv, 1) * DT], 1)
        return (-(qq[:, :-1] * mu - A_ * nuv ** 2 - PHI * qq[:, :-1] ** 2).sum(1) * DT + PSI * qq[:, -1] ** 2).mean()
    rng = np.random.default_rng(0); outs = []
    for d in range(4):
        c = rng.standard_normal(4); v = c[0] + c[1] * ai + c[2] * np.sin(3 * ah) + c[3] * ah * ai; v *= 0.1 / np.abs(v).mean()
        # directions as functions of the observed features only (open-loop; q not included to keep v exogenous; q-dependent directions are also admissible)
        eps = 1e-3; outs.append((cost(nu[:, i] + eps * v) - cost(nu[:, i] - eps * v)) / (2 * eps))
    sc = (cost(0.8 * nu[:, i] + 1e-3 * v) - cost(0.8 * nu[:, i] - 1e-3 * v)) / 2e-3
    print(f'   Gateaux derivatives at reference (4 dirs): {np.round(outs, 5)}; at 0.8 x reference (last dir): {sc:+.5f}; cost at ref {cost(nu[:, i]):.5f}, at 0.8 ref {cost(0.8 * nu[:, i]):.5f}')

# ---- 5. continuous-time closed form (Casgrain-Jaimungal 5.12 with beta = 0, two-state chain): nu = (h2/a) Q + h1/(2a),
# h2' = h2^2/a - phi... derive: HJB for value h0 + Q h1 + Q^2 h2 with running -phi Q^2, cost a nu^2, drift A: sup_nu [ Q A + nu (h1 + 2 h2 Q) - a nu^2 ] = Q A + (h1 + 2 h2 Q)^2/(4a)
# => h2' + (h2^2)/a - phi = 0, h2(T) = -psi;  h1' + L h1 + (h2/a) h1 + A = 0, h1(T) = 0, with L the generator acting on the posterior (here A-hat decays at rate 2 kap)
# => h2(t) = -sqrt(a phi) (zeta e^{g(T-t)} - e^{-g(T-t)}) / (zeta e^{g(T-t)} + e^{-g(T-t)}), g = sqrt(phi/a), zeta = (psi + sqrt(a phi)) / (psi - sqrt(a phi))
# => h1(t) = ahat_t * int_t^T e^{-2 kap (u-t)} exp(int_t^u h2(s)/a ds) du
g = math.sqrt(PHI / A_); sq = math.sqrt(A_ * PHI); zeta = (PSI + sq) / (PSI - sq)
h2 = lambda t: -sq * (zeta * math.exp(g * (1 - t)) + math.exp(-g * (1 - t))) / (zeta * math.exp(g * (1 - t)) - math.exp(-g * (1 - t)))   # (5.10a), beta = 0
tt = np.linspace(0, 1, 4001); h2v = np.array([h2(t) for t in tt])
print('5. h2(T) =', h2(1.0), ' (should be -psi =', -PSI, ');  check ODE residual at t=0.3:', (h2(0.3 + 1e-5) - h2(0.3 - 1e-5)) / 2e-5 + h2(0.3) ** 2 / A_ - PHI)
cum = np.concatenate([[0.0], np.cumsum(0.5 * (h2v[1:] + h2v[:-1]) * (tt[1] - tt[0]))]) / A_    # int_0^t h2/a
def h1w(t):   # weight multiplying ahat_t
    j = np.searchsorted(tt, t); u = tt[j:]; integ = np.exp(-2 * KAP * (u - t)) * np.exp(cum[j:] - cum[j]); return np.trapezoid(integ, u)
for Nn in (50, 200, 800):
    os.environ['N_STEPS'] = str(Nn)
    import importlib; importlib.reload(P); refn = P.reference_lam0(); tk = np.arange(Nn) / Nn
    Gc = np.array([-h2(t) / A_ for t in tk]); gc = np.array([h1w(t) / (2 * A_) for t in tk])
    print(f'   N={Nn}: inventory gain G vs -h2/a rel L2 {rel(refn["G"], Gc):.4f}; signal gain -g vs h1-weight/(2a) rel L2 {rel(-refn["g"], gc):.4f}; '
          f'at k=0: G {refn["G"][0]:.4f} vs {Gc[0]:.4f}, -g {-refn["g"][0]:.4f} vs {gc[0]:.4f}')
