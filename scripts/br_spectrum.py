"""Best-response spectrum of the private two-trader game in the linear-strategy class (pt_kernel), at the exact equilibria.
For each player i, the best response BR_i solves grad_{b_i} J_i(b_i, beta) = 0, so by implicit differentiation
    d BR_i / d beta = - H_ii^+ C_i,   H_ii = d grad / d b_i (own Hessian),  C_i = d grad / d beta.
Closed loop: J_i(b_i, beta) = cost with beta_i replaced by b_i (the other's rule fixed). Open loop: the other's control process
(its loadings under beta) is held fixed. Prints the spectral radius of the undamped iteration map beta -> (BR_1, BR_2), its eigenvalue
of smallest real part (below -1: the undamped iteration oscillates out; near +1: slow), and the smallest nonzero eigenvalue of H_ii.
Usage, from scripts/: python3 br_spectrum.py   (env: LAMS, default 1.0,1.5,2.0)"""
import os, sys, time, numpy as np, torch
torch.set_default_dtype(torch.float64)
os.environ['DEVICE'] = 'cpu'; os.environ['INFO'] = 'private'
D = os.environ.get('RESULTS_DATA', os.path.expanduser('~/Desktop/Bilal Saleh Husain/research-data/partial-info-games'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'games'))

def equilibria(lam):
    if lam == '1.5':
        d = np.load(f'{D}/kernel_open_closed_lam1.5.npz'); return d['b_open'], d['b_closed']
    return np.load(f'{D}/pt_kernel_openloop_lam{lam}.npz')['beta'], np.load(f'{D}/pt_kernel_private_lam{lam}.npz')['beta']

for lam in os.environ.get('LAMS', '1.0,1.5,2.0').split(','):
    os.environ['LAM'] = lam
    for mod in ['pt_game', 'pt_kernel']: sys.modules.pop(mod, None)
    import pt_kernel as K
    N, M, TRIL, AL, SN, DT, A_, PHI, PSI, LAMV = K.N, K.M, K.TRIL, K.AL, K.SN, K.DT, K.A_, K.PHI, K.PSI, K.LAM
    act = TRIL.bool().unsqueeze(-1).expand(N, N, 2)               # active coefficients of one player
    P = int(act.sum())
    def unpack(v):                                                 # active vector of one player -> (N, N, 2)
        b = torch.zeros(N, N, 2); b[act] = v; return b
    def full(b1, b2): return torch.stack([b1, b2])
    def forward_open(bi, i, nu_other):                             # as audit/checks/pt_kernel_openloop.py
        S = [torch.zeros(M)]; q = [torch.zeros(M)]; nus = []
        for j in range(N):
            Sst = torch.stack(S); n = bi[j, :j + 1, 0] @ AL[i, :j + 1] + bi[j, :j + 1, 1] @ Sst
            mu = AL[0, j] + AL[1, j] + LAMV * (n + nu_other[j]); nus.append(n)
            S.append(S[-1] + mu * DT + SN[j]); q.append(q[-1] + n * DT)
        nu = torch.stack(nus); q = torch.stack(q); mu = AL[0, :N] + AL[1, :N] + LAMV * (nu + nu_other)
        return ((-(q[:-1] * mu).sum(-1) + A_ * (nu ** 2).sum(-1) + PHI * (q[:-1] ** 2).sum(-1)).sum() * DT + PSI * (q[-1] ** 2).sum())
    b_open, b_closed = equilibria(lam)
    for kind, beta0 in [('open', b_open), ('closed', b_closed)]:
        t0 = time.time(); beta0 = torch.tensor(beta0) * TRIL[None, :, :, None]
        vec = torch.cat([beta0[0][act], beta0[1][act]])            # (2P,)
        rows = []; hmin = []
        for i in (0, 1):
            def grad_i(bi_vec, all_vec):
                bi = unpack(bi_vec); b = [unpack(all_vec[:P]), unpack(all_vec[P:])]
                if kind == 'closed':
                    b[i] = bi; J = K.cost(i, full(*b))
                else:
                    nu_all, _, _, _ = K.forward(full(*b)); J = forward_open(bi, i, nu_all[1 - i])
                return torch.autograd.grad(J, bi_vec, create_graph=True)[0]
            bi0 = vec[i * P:(i + 1) * P].clone().requires_grad_(True); allv = vec.clone().requires_grad_(True)
            g = grad_i(bi0, allv)
            H = torch.zeros(P, P); C = torch.zeros(P, 2 * P)
            for r in range(P):
                gr = torch.autograd.grad(g[r], [bi0, allv], retain_graph=True, allow_unused=True)
                H[r] = gr[0]; C[r] = gr[1] if gr[1] is not None else 0.0
            ev = np.linalg.eigvalsh(((H + H.T) / 2).numpy()); tol = 1e-9 * ev.max()
            hmin.append((ev[ev > tol].min(), int((ev <= tol).sum()), ev.max()))
            rows.append(-np.linalg.pinv(H.numpy(), rcond=1e-10) @ C.numpy())
        Jmap = np.vstack(rows); eig = np.linalg.eigvals(Jmap)
        rho = np.abs(eig).max(); emin = eig[np.argmin(eig.real)]; emax = eig[np.argmax(eig.real)]
        top = eig[np.argsort(-np.abs(eig))[:6]]; d = float(os.environ.get('DAMP', '0.5')); damped = np.abs(1 - d + d * eig).max()
        print(f'lam={lam} {kind}: leading eigenvalues ' + ', '.join(f'{z.real:+.3f}{z.imag:+.3f}i' for z in top) + f'; damped map (damping {d}) factor {damped:.4f}', flush=True)
        print(f'lam={lam} {kind:6s}: spectral radius {rho:.3f}; eigenvalue of smallest real part {emin.real:+.3f}{emin.imag:+.3f}i, largest real part {emax.real:+.3f}; '
              f'own Hessian smallest nonzero eig {hmin[0][0]:.2e} / max {hmin[0][2]:.2e} ({hmin[0][1]} zero) ({time.time()-t0:.0f}s)', flush=True)
