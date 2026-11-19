"""Independent audit: open-loop vs closed-loop linear-class Nash for the two-trader game (pt_game), and comparison with learned seeds.
Open loop: trader i's best response holds the OTHER trader's control fixed as a process (its loading vector), as in the FOC the
learned solver regresses on. Closed loop (pt_kernel.solve): the other's control is a fixed function of its observed path and reacts through S."""
import os, sys, math, glob, time, numpy as np, torch
os.environ.setdefault('LAM', '1.0'); os.environ.setdefault('INFO', 'private'); os.environ['DEVICE'] = 'cpu'
sys.path.insert(0, '/Users/bhusain/Desktop/Bilal Saleh Husain/Academics and Research/Repos/partial-info-games/games')
import pt_kernel as K; import pt_game as P
N, DT, LAM, A_, PHI, PSI, M, AL, SN, TRIL = K.N, K.DT, K.LAM, K.A_, K.PHI, K.PSI, K.M, K.AL, K.SN, K.TRIL
torch.set_default_dtype(torch.float64)

def forward_open(bi, i, nu_other):
    """Loadings of nu_i, q_i, mu when the other's control loadings nu_other (N, M) are held fixed."""
    S = [torch.zeros(M)]; q = [torch.zeros(M)]; nus = []
    for j in range(N):
        Sst = torch.stack(S); n = bi[j, :j + 1, 0] @ AL[i, :j + 1] + bi[j, :j + 1, 1] @ Sst
        mu = AL[0, j] + AL[1, j] + LAM * (n + nu_other[j]); nus.append(n)
        S.append(S[-1] + mu * DT + SN[j]); q.append(q[-1] + n * DT)
    nu = torch.stack(nus); q = torch.stack(q); mu = AL[0, :N] + AL[1, :N] + LAM * (nu + nu_other)
    return nu, q, mu
def cost_open(bi, i, nu_other):
    nu, q, mu = forward_open(bi, i, nu_other)
    return ((-(q[:-1] * mu).sum(-1) + A_ * (nu ** 2).sum(-1) + PHI * (q[:-1] ** 2).sum(-1)).sum() * DT + PSI * (q[-1] ** 2).sum())
def best_response_open(beta, i, inner=200):
    with torch.no_grad(): nu_all, _, _, _ = K.forward(beta * TRIL[None, :, :, None])
    nu_other = nu_all[1 - i].detach(); b = beta[i].detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([b], max_iter=inner, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); J = cost_open(b * TRIL[:, :, None], i, nu_other); J.backward(); return J
    opt.step(closure); return (b * TRIL[:, :, None]).detach(), float(cost_open((b * TRIL[:, :, None]).detach(), i, nu_other))
def solve_open(iters=100, rho=0.5, inner=200):
    beta = torch.zeros(2, N, N, 2); t0 = time.time()
    for it in range(iters):
        new = beta.clone(); Js = []
        for i in (0, 1): bi, J = best_response_open(beta, i, inner); new[i] = bi; Js.append(J)
        chg = float((new - beta).norm() / max(1.0, float(new.norm()))); beta = (1 - rho) * beta + rho * new
        print(f'open it {it:3d} J {Js[0]:.6f} {Js[1]:.6f} change {chg:.2e} ({time.time()-t0:.0f}s)', flush=True)
        if chg < 1e-6: break
    return beta
rel = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
nz = P.simulate_noise(2048, 7)
t0 = time.time(); b_open = solve_open(); nu_open = K.controls_on_paths(b_open, nz); print(f'open-loop kernel done ({time.time()-t0:.0f}s)', flush=True)
t0 = time.time(); b_closed = K.solve(100, 0.5, 200, log=lambda m: None); nu_closed = K.controls_on_paths(b_closed, nz); print(f'closed-loop kernel done ({time.time()-t0:.0f}s)', flush=True)
print(f'closed vs open: trader1 {rel(nu_closed[:, 0], nu_open[:, 0]):.4f} trader2 {rel(nu_closed[:, 1], nu_open[:, 1]):.4f}; mean|nu| open {np.abs(nu_open).mean():.4f} closed {np.abs(nu_closed).mean():.4f}')
# FOC consistency: with the open-loop equilibrium loadings, is 2a nu_i = E[bracket | F^i]?  Check via the learned-solver bracket on sampled paths.
files = sorted(glob.glob('/Users/bhusain/Desktop/Bilal Saleh Husain/research-data/partial-info-games/pt_priv2_cv_fix_s*.npz') + glob.glob('/Users/bhusain/Desktop/Bilal Saleh Husain/Academics and Research/Repos/partial-info-games/results/pt_priv2_cv_fix_s*.npz'))
for f in files:
    d = np.load(f)
    if d['nu1'].shape != nu_open[:, 0].shape: print('skip', f, d['nu1'].shape); continue
    print(f'{os.path.basename(f)}: vs OPEN trader1 {rel(d["nu1"], nu_open[:, 0]):.4f} trader2 {rel(d["nu2"], nu_open[:, 1]):.4f} | vs CLOSED trader1 {rel(d["nu1"], nu_closed[:, 0]):.4f} trader2 {rel(d["nu2"], nu_closed[:, 1]):.4f} | mean|nu| learned {np.abs(d["nu1"]).mean():.4f}')
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernel_open_closed_lam2.npz'), nu_open=nu_open, nu_closed=nu_closed, b_open=b_open.numpy(), b_closed=b_closed.numpy())
