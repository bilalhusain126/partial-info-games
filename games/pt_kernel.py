"""Independent benchmark for the two-trader private-signal game (pt_game.py), in the spirit of Babichenko (arXiv 2603.12140):
a deterministic fixed point over linear strategies, with every process represented by its loading vector on the Gaussian shocks.
No Monte Carlo, no neural networks.

Discrete game on the grid t_j = j dt, j = 0..N (same Euler scheme and conventions as pt_game.py). Shocks e in R^M, M = 2 + 3N:
e = (xi_1, xi_2, eps_1^{a1}, eps_1^{a2}, eps_1^S, ..., eps_N^{a1}, eps_N^{a2}, eps_N^S), iid N(0,1). Every zero-mean process value at
t_j is a loading vector in R^M, so E[X Y] = X . Y. Trader i's strategy is a linear map of its own observed path,
    nu_i(t_j) = sum_{k<=j} beta_i[j,k]^a alpha_i(t_k) + sum_{1<=k<=j} beta_i[j,k]^S S(t_k),
with deterministic coefficients beta_i (the admissible linear class; by Babichenko's Cor. 4.5 a Nash in this class is a Nash of the
unrestricted game). Player i's cost (to minimise) J_i = sum_j dt [ -q_i[j].mu[j] + a |nu_i[j]|^2 + phi |q_i[j]|^2 ] + psi |q_i[N]|^2,
mu = alpha_1 + alpha_2 + lam (nu_1 + nu_2). With beta_j fixed, all loadings are linear in beta_i, so J_i is quadratic in beta_i and the
best response is its minimiser (L-BFGS on the autodiff gradient; the dependence of nu_j on beta_i through S, the belief-manipulation
channel, is included). Nash = fixed point of the damped best-response iteration beta <- (1-rho) beta + rho BR(beta).
Usage: LAM=2 python3 pt_kernel.py   (env: LAM, N_STEPS, ITERS, RHO_DAMP, INNER, SAVE)
Output: equilibrium coefficients and the controls along pt_game's validation paths (seed 7), compared with the learned solutions."""
import os, sys, math, time, numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_default_device('cpu')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pt_game as P
N, DT, LAM, A_, PHI, PSI, KAP, SA, SIG, VA0 = P.N, P.DT, P.LAM, P.A_, P.PHI, P.PSI, P.KAP, P.SA, P.SIG, P.VA0
M = 2 + 3 * N
def idx_a(i, s): return 2 + 3 * (s - 1) + i          # eps_s^{a i}, s = 1..N
def idx_S(s): return 2 + 3 * (s - 1) + 2            # eps_s^S

AL = torch.zeros(2, N + 1, M)                        # signal loadings (Euler OU as in pt_game.signals)
for i in range(2):
    AL[i, 0, i] = math.sqrt(VA0)
    for k in range(N): AL[i, k + 1] = (1 - KAP * DT) * AL[i, k]; AL[i, k + 1, idx_a(i, k + 1)] += SA * math.sqrt(DT)
SN = torch.zeros(N, M)                               # price noise increments sig sqrt(dt) eps^S_{k+1}
for k in range(N): SN[k, idx_S(k + 1)] = SIG * math.sqrt(DT)
TRIL = torch.tril(torch.ones(N, N))                  # beta[j, k] active for k <= j
INFO = os.environ.get('INFO', 'private'); NB = 3 if INFO == 'full' else 2   # coefficient channels: (alpha_i, S) or (alpha_1, alpha_2, S)

def forward(beta):
    """beta: (2, N, N, 2) coefficients on (alpha_i(t_k), S(t_k)). Causal recursion of all loadings. Returns nu (2,N,M), q (2,N+1,M), S (N+1,M), mu (N,M)."""
    nu = []; S = [torch.zeros(M)]; q = [torch.zeros(2, M)]
    for j in range(N):
        Sst = torch.stack(S)                                          # (j+1, M): S(t_0..t_j), S(t_0) = 0
        if INFO == 'full': n_j = torch.stack([beta[i, j, :j + 1, 0] @ AL[0, :j + 1] + beta[i, j, :j + 1, 1] @ AL[1, :j + 1] + beta[i, j, :j + 1, 2] @ Sst for i in range(2)])
        else: n_j = torch.stack([beta[i, j, :j + 1, 0] @ AL[i, :j + 1] + beta[i, j, :j + 1, 1] @ Sst for i in range(2)])   # (2, M)
        nu.append(n_j); mu_j = AL[0, j] + AL[1, j] + LAM * (n_j[0] + n_j[1])
        S.append(S[-1] + mu_j * DT + SN[j]); q.append(q[-1] + n_j * DT)
    nu = torch.stack(nu, 1); q = torch.stack(q, 1); S = torch.stack(S); mu = AL[0, :N] + AL[1, :N] + LAM * (nu[0] + nu[1])
    return nu, q, S, mu

def cost(i, beta):
    nu, q, S, mu = forward(beta)
    return ((-(q[i, :-1] * mu).sum(-1) + A_ * (nu[i] ** 2).sum(-1) + PHI * (q[i, :-1] ** 2).sum(-1)).sum() * DT + PSI * (q[i, -1] ** 2).sum())

def best_response(beta, i, inner):
    """Minimise J_i over beta_i with beta_j fixed (J_i is quadratic in beta_i); L-BFGS from the current beta_i."""
    b = beta[i].detach().clone().requires_grad_(True); other = beta[1 - i].detach()
    opt = torch.optim.LBFGS([b], max_iter=inner, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); full = torch.stack([b, other]) if i == 0 else torch.stack([other, b]); J = cost(i, full * TRIL[None, :, :, None]); J.backward(); return J
    opt.step(closure)
    full = torch.stack([b, other]) if i == 0 else torch.stack([other, b]); J = cost(i, (full * TRIL[None, :, :, None]).detach())
    return (b * TRIL[:, :, None]).detach(), float(J)

def solve(iters, rho, inner, init=None, log=lambda m: print(m, flush=True)):
    beta = torch.zeros(2, N, N, NB) if init is None else init.clone(); t0 = time.time()
    for it in range(iters):
        new = beta.clone(); Js = []
        for i in (0, 1): bi, J = best_response(beta, i, inner); new[i] = bi; Js.append(J)         # simultaneous update
        chg = float((new - beta).norm() / max(1.0, float(new.norm()))); beta = (1 - rho) * beta + rho * new
        log(f'it {it:3d} J1 {Js[0]:.6f} J2 {Js[1]:.6f} rel change of best response {chg:.2e} |beta| {float(beta.norm()):.3f} ({time.time()-t0:.0f}s)')
        if chg < 1e-6: break
    return beta

def controls_on_paths(beta, nz):
    """Evaluate nu_i(t_j) along given noise paths (pt_game.simulate_noise conventions): e from the path, then loadings . e."""
    m = nz['a1_0'].shape[0]; e = np.zeros((m, M)); e[:, 0] = nz['a1_0'] / math.sqrt(VA0); e[:, 1] = nz['a2_0'] / math.sqrt(VA0)
    for s in range(1, N + 1):
        e[:, idx_a(0, s)] = nz['dw1'][:, s - 1] / math.sqrt(DT); e[:, idx_a(1, s)] = nz['dw2'][:, s - 1] / math.sqrt(DT); e[:, idx_S(s)] = nz['dws'][:, s - 1] / math.sqrt(DT)
    with torch.no_grad(): nu, q, S, mu = forward(beta)
    return np.einsum('ijm,pm->pij', nu.numpy(), e)                      # (m, 2, N)

if __name__ == '__main__':
    iters = int(os.environ.get('ITERS', '100')); rho = float(os.environ.get('RHO_DAMP', '0.5')); inner = int(os.environ.get('INNER', '200'))
    print(f'kernel benchmark: INFO={INFO} LAM={LAM} N={N} M={M} iters={iters} rho={rho} inner={inner}', flush=True)
    init = None
    if os.environ.get('INIT') and os.path.exists(os.environ['INIT']): init = torch.tensor(np.load(os.environ['INIT'])['beta']); print('continuation from', os.environ['INIT'])
    beta = solve(iters, rho, inner, init)
    nz = P.simulate_noise(2048, 7); nu = controls_on_paths(beta, nz)
    rel = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
    if LAM == 0.0:
        ex = P.reference_paths_lam0(P.reference_lam0(), nz)
        print(f'lam=0 check vs closed form: trader1 {rel(nu[:, 0], ex[:, 0]):.4f} trader2 {rel(nu[:, 1], ex[:, 1]):.4f}  (|nu| {np.abs(nu).mean():.3f} exact {np.abs(ex).mean():.3f})')
    np.savez(os.environ.get('SAVE', f'../results/pt_kernel_{INFO}_lam{LAM}.npz'), beta=beta.numpy(), nu1=nu[:, 0], nu2=nu[:, 1])
    for f in [f'../results/pt_priv2_cv_s{s}.npz' for s in (1, 2, 3)] + ['../results/pt_scale_32768.npz', '../results/pt_priv_lam0.25.npz', '../results/pt_priv_lam0.5.npz', '../results/pt_full2_cv.npz', '../results/pt_full2_cv_foc.npz']:
        if os.path.exists(f):
            d = np.load(f)
            if d["nu1"].shape == nu[:, 0].shape: print(f"vs {os.path.basename(f)}: trader1 {rel(d['nu1'], nu[:, 0]):.4f} trader2 {rel(d['nu2'], nu[:, 1]):.4f}")
