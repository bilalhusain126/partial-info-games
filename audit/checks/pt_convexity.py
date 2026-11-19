"""Convexity audit: Hessian of trader i's discrete cost over its linear strategy coefficients (closed loop: other's rule fixed; open loop: other's
control process fixed), at lam in {1, 2}. The cost is quadratic in beta_i, so the Hessian is constant; its minimum eigenvalue decides convexity.
Open-loop cost is a quadratic functional with a deterministic kernel, so PSD on the linear class implies PSD on all admissible controls."""
import os, sys, math, time, numpy as np, torch
os.environ['DEVICE'] = 'cpu'; os.environ.setdefault('INFO', 'private')
sys.path.insert(0, '/Users/bhusain/Desktop/Bilal Saleh Husain/Academics and Research/Repos/partial-info-games/games')
for lam in ['1.0', '2.0']:
    os.environ['LAM'] = lam
    for mod in ['pt_game', 'pt_kernel']: sys.modules.pop(mod, None)
    import pt_kernel as K
    N, M, TRIL = K.N, K.M, K.TRIL
    act = TRIL.bool()                                             # active (j, k) pairs, k <= j
    n_act = int(act.sum()); idx = act.nonzero()
    def unpack(x, i, beta_other):                                  # x: (2*n_act,) -> full beta (2, N, N, 2)
        bi = torch.zeros(N, N, 2); bi[idx[:, 0], idx[:, 1], 0] = x[:n_act]; bi[idx[:, 0], idx[:, 1], 1] = x[n_act:]
        return torch.stack([bi, beta_other]) if i == 0 else torch.stack([beta_other, bi])
    rng = torch.Generator().manual_seed(0); beta_other = 0.1 * torch.randn(N, N, 2, generator=rng) * TRIL[:, :, None]
    for i in (0,):
        # closed loop: other's rule (beta_other) fixed, nu_other reacts through S
        f_closed = lambda x: K.cost(i, unpack(x, i, beta_other))
        # open loop: other's control loadings fixed
        with torch.no_grad(): nu_all, _, _, _ = K.forward(unpack(torch.zeros(2 * n_act), i, beta_other))
        nu_other = nu_all[1 - i]
        def f_open(x):
            b = unpack(x, i, beta_other)[i]; S = [torch.zeros(M)]; q = [torch.zeros(M)]; nus = []
            for j in range(N):
                Sst = torch.stack(S); n = b[j, :j + 1, 0] @ K.AL[i, :j + 1] + b[j, :j + 1, 1] @ Sst
                mu = K.AL[0, j] + K.AL[1, j] + K.LAM * (n + nu_other[j]); nus.append(n); S.append(S[-1] + mu * K.DT + K.SN[j]); q.append(q[-1] + n * K.DT)
            nu = torch.stack(nus); q = torch.stack(q); mu = K.AL[0, :N] + K.AL[1, :N] + K.LAM * (nu + nu_other)
            return ((-(q[:-1] * mu).sum(-1) + K.A_ * (nu ** 2).sum(-1) + K.PHI * (q[:-1] ** 2).sum(-1)).sum() * K.DT + K.PSI * (q[-1] ** 2).sum())
        for name, f in [('open', f_open), ('closed', f_closed)]:
            t0 = time.time(); H = torch.autograd.functional.hessian(f, torch.zeros(2 * n_act)); H = 0.5 * (H + H.T)
            ev = torch.linalg.eigvalsh(H); print(f'lam={lam} {name}-loop trader {i+1}: Hessian {H.shape[0]}x{H.shape[0]}, min eig {float(ev.min()):.3e}, max eig {float(ev.max()):.3e}, #eig<-1e-8: {int((ev < -1e-8).sum())} ({time.time()-t0:.0f}s)', flush=True)
