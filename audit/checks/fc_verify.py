"""Independent checks of games/fc_game.py (read-only import). Run with the same env as the Gateaux log:
NOISE=asym SV2=0.05 Q0=3.0 A0=-1.0 DEVICE=cpu python3 fc_verify.py
(a) dlqg vs brute-force least-squares LQ on the deterministic (certainty-equivalent) problem;
(b) Kalman filters vs linear regression of the true state on past observation increments (Monte Carlo);
(c) brackets vs autograd derivative of the realised cost;
(d) Gateaux ratio of npg.gateaux_check evaluated at the EXACT reference strategy (noise-floor calibration)."""
import os, sys, math, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/Users/bhusain/Desktop/Bilal Saleh Husain/Academics and Research/Repos/partial-info-games/src')
sys.path.insert(0, '/Users/bhusain/Desktop/Bilal Saleh Husain/Academics and Research/Repos/partial-info-games/games')
import fc_game as F
from npg import Solver, gateaux_check
N, DT, BETA = F.N, F.DT, F.BETA
ref = F.reference()

# ---------------- (a) dlqg vs least squares ----------------
def lsq_lq(Fk, G, c, Qz, qz, R, z0):
    """Deterministic finite-horizon LQ: minimise sum_k beta^k[(z_k - zeta)'Q(z_k - zeta) + R u_k^2] dt over u in R^N. Returns u."""
    n = len(z0); Z0 = np.zeros((N, n)); Zu = np.zeros((N, n, N))      # z_k = Z0[k] + Zu[k] @ u
    z, Su = z0.copy(), np.zeros((n, N))
    for k in range(N):
        Z0[k], Zu[k] = z, Su
        Fk_ = Fk(k); z = Fk_ @ z + c(k); Su = Fk_ @ Su; Su[:, k] += G
    # quadratic in u: sum_k beta^k [(Z0+Zu u - zeta)'Q(...) + R u_k^2] dt ; solve normal equations
    H = np.zeros((N, N)); b = np.zeros(N)
    for k in range(N):
        w = BETA ** k * DT
        H += w * (Zu[k].T @ Qz @ Zu[k]); b += w * (Zu[k].T @ (Qz @ Z0[k] - qz))
        H[k, k] += w * R
    return np.linalg.solve(H, -b)
rng = np.random.default_rng(0)
for name, _ in ([] if os.environ.get('SKIP_AB') else [1]):
  pass
for name, (Fk, G, c, Qz, qz, R, Gain, g, n) in dict(
        major=(lambda k: np.array([[1 + F.A0 * DT]]), np.array([F.B0 * DT]), lambda k: np.zeros(1), np.array([[F.Q0]]), np.array([F.Q0 * F.ETA0]), F.R0, ref['Gain0'], ref['g0'], 1),
        minor=(ref['Fz'], ref['Gz'], ref['cz'], np.outer([1, -F.H1, 0], [1, -F.H1, 0]) * F.Q_, np.array([1, -F.H1, 0]) * F.Q_ * F.ETA, F.R_, ref['Gain1'], ref['g1'], 3)).items():
    errs = []
    for trial in range(3):
        z0 = rng.standard_normal(n); u = lsq_lq(Fk, G, c, Qz, qz, R, z0)
        z = z0.copy(); ufb = np.zeros(N)
        for k in range(N):
            ufb[k] = -(Gain[k] @ z + g[k]); z = Fk(k) @ z + G * ufb[k] + c(k)
        errs.append(np.abs(u - ufb).max() / np.abs(u).max())
    print(f'(a) dlqg {name}: max rel diff feedback vs least squares over 3 random z0: {max(errs):.2e}')

# ---------------- (b) Kalman filters vs regression on past increments ----------------
m = 2000 if os.environ.get('SKIP_AB') else 200000; nz = F.simulate_noise(m, 3); u0, ui, X0, XI, XH0, ZH = F.reference_paths(ref, nz)
# rebuild the observation increments exactly as reference_paths does
DY0 = F.L0 * X0[:, :-1] * DT + F.SV0 * nz['dv0']; DYI = np.stack([F.L1 * XI[:, :-1], F.L2 * X0[:, :-1]], -1) * DT + np.array([F.SV, F.SV2]) * nz['dvi']
for k in (() if os.environ.get('SKIP_AB') else (5, 30, 99)):
    R0 = np.concatenate([np.ones((m, 1)), DY0[:, :k]], 1); xh_reg = R0 @ np.linalg.lstsq(R0, X0[:, k], rcond=None)[0]
    print(f'(b) k={k} major: rel L2 (filter - regression)/sd(x0) {np.sqrt(((XH0[:, k] - xh_reg) ** 2).mean() / X0[:, k].var()):.3e}  (E|x0-xh|^2 filter {((X0[:, k] - XH0[:, k]) ** 2).mean():.4f} vs V0 {ref["V0"][k, 0, 0]:.4f})')
    Ri = np.concatenate([np.ones((m, 1)), DYI[:, :k].reshape(m, -1)], 1); Z = np.stack([XI[:, k], X0[:, k], XH0[:, k]], -1)
    zreg = Ri @ np.linalg.lstsq(Ri, Z, rcond=None)[0]
    for j, nm in enumerate(['x_i', 'x_0', 'xhat_0']):
        print(f'      minor {nm}: rel L2 (filter - regression)/sd {np.sqrt(((ZH[:, k, j] - zreg[:, j]) ** 2).mean() / Z[:, j].var()):.3e}  (E err^2 filter {((Z[:, j] - ZH[:, k, j]) ** 2).mean():.4f} vs V1 {ref["V1"][k, j, j]:.4f})')
# certainty-equivalence sanity: control = feedback on prior estimate (what reference_paths does); cost of reference vs perturbed
# ---------------- (c) brackets vs autograd ----------------
torch.set_default_device('cpu'); M = F.FCModel(); mm = 64; exo = M.exo(mm, 5)
st = M.init_state(mm); M.obs_now(0, exo, st, 0)
U = [torch.tensor(u0[:mm], dtype=torch.float32, requires_grad=True), torch.tensor(ui[:mm], dtype=torch.float32, requires_grad=True)]
for k in range(N): M.step(st, {0: U[0][:, k], 1: U[1][:, k]}, k)
ctrl = [U[0], U[1]]
for i in range(2):
    J = M.cost(i, exo, st, ctrl).sum(); gJ, = torch.autograd.grad(J, U[i], retain_graph=True)
    disc = torch.tensor(np.exp(-F.RHO * F.tg[:-1]), dtype=torch.float32); R = F.R0 if i == 0 else F.R_
    br_from_grad = U[i].detach() - gJ / (2 * disc * DT * R)        # dJ/du_k = 2 beta^k dt R (u_k - bracket_k)
    br = M.bracket(i, exo, {k: v.detach() if torch.is_tensor(v) else v for k, v in st.items()}, [c.detach() for c in ctrl])
    print(f'(c) bracket player {i}: max |bracket - (u - dJ/du / (2 beta^k dt R))| = {(br - br_from_grad).abs().max():.2e} (|bracket| max {br.abs().max():.3f})')
    # also: cross-player derivative (does player i's control enter the other player's cost through the state?)
    Jo = M.cost(1 - i, exo, st, ctrl).sum(); gJo, = torch.autograd.grad(Jo, U[i], retain_graph=True, allow_unused=True)
    print(f'    d(cost of player {1 - i})/d(u_{i}) max |.| = ' + ('UNUSED in graph (exactly zero)' if gJo is None else f'{gJo.abs().max():.2e}'))

# ---------------- (d) Gateaux check at the exact reference strategy ----------------
class RefGRU:
    """Stands in for nn.GRU: hidden state h (1, m, 32) carries [filter state (n), step k] in its first n+1 channels."""
    def __init__(s, i): s.i = i; s.n = 1 if i == 0 else 3
    def __call__(s, seq, h):
        h = h.clone(); m = seq.shape[0]; k = int(round(float(h[0, 0, s.n]))); dy = seq[:, 0] * math.sqrt(DT)          # obs_inc is scaled by 1/sqrt(dt)
        if s.i == 0:
            xh = h[0, :, 0]; u = -(ref['Gain0'][k, 0] * xh + ref['g0'][k])
            h[0, :, 0] = (1 + F.A0 * DT) * (xh + ref['K0'][k] * (dy[:, 0] - F.L0 * xh * DT)) + F.B0 * DT * u
        else:
            zh = h[0, :, :3].double().numpy(); u = -(zh @ ref['Gain1'][k] + ref['g1'][k])
            zp = zh + (dy.double().numpy() - zh @ ref['Cz'].T * DT) @ ref['K1'][k].T
            h[0, :, :3] = torch.tensor(zp @ ref['Fz'](k).T + ref['cz'](k) + ref['Gz'] * u[:, None], dtype=torch.float32)
        h[0, :, s.n] = k + 1; return None, h
class RefPolicy(nn.Module):
    def __init__(s, i): super().__init__(); s.i = i; s.hid = 32; s.d_seq = M.d_seq[i]; s.gru = RefGRU(i)
    def mem(s, seq): raise NotImplementedError
    def forward(s, t, now, memk):
        k = int(round(float(t[0, 0]) / DT))
        if s.i == 0: return -(ref['Gain0'][k, 0] * memk[:, 0] + ref['g0'][k])
        return -(memk[:, :3] @ torch.tensor(ref['Gain1'][k], dtype=torch.float32) + ref['g1'][k])
S = Solver(M, hid=32, m_paths=256, m_val=2048); S.pols = [RefPolicy(0), RefPolicy(1)]
# sanity: rollout under RefPolicy reproduces reference_paths on the validation noise
st, ctrl, _ = S.rollout(S.exo_v); u0v, uiv, *_ = F.reference_paths(ref, {k: v.numpy().astype(np.float64) for k, v in S.exo_v.items()})
print(f'(d) RefPolicy rollout vs reference_paths: max |diff| u0 {np.abs(ctrl[0].numpy() - u0v).max():.2e} ui {np.abs(ctrl[1].numpy() - uiv).max():.2e}')
for i in range(2): gateaux_check(S, i)
# the same check with the reference strategy corrupted by 6% (rel L2) of information-adapted noise: a different random net added
class Noisy(nn.Module):
    def __init__(s, base, v, c): super().__init__(); s.base, s.v, s.c, s.gru, s.hid, s.d_seq = base, v, c, base.gru, base.hid, base.d_seq
    def forward(s, t, now, memk): return s.base(t, now, memk) + s.c * s.v(t, now, memk)
from npg import Policy
for i in range(2):
    torch.manual_seed(99 + i); v = Policy(M.d_now[i], M.d_seq[i], 32)
    class VG:                                                    # v needs its own hidden state: run v's GRU on the same increments
        pass
    # simpler: perturb along a direction that only depends on time (adapted): c * sin(3 t)
    class TimeDir(nn.Module):
        def forward(s, t, now, memk): return torch.sin(3 * t[:, 0])
    base = S.pols[i]; st, ctrl, _ = S.rollout(S.exo_v); scale = float(ctrl[i].pow(2).mean().sqrt())
    S.pols[i] = Noisy(base, TimeDir(), 0.06 * scale / float(torch.sin(3 * S.t).pow(2).mean().sqrt()))
    _, c2, _ = S.rollout(S.exo_v); print(f'(d) player {i}: reference + 6% deterministic error (rel L2 {float(((c2[i]-ctrl[i])**2).mean().sqrt()/ctrl[i].pow(2).mean().sqrt()):.3f}):')
    gateaux_check(S, i); S.pols[i] = base
