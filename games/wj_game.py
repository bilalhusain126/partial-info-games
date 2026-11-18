"""Prototype: deep Picard-regression solver for the Wu-Jaimungal broker-trader Nash equilibrium (arXiv 2412.17712), K = 1 asset.
Model (Sec. 2): unimpacted price dz = alpha dt + sigma dW, OU signal d alpha = -kappa alpha dt + sigma_a dB (trader sees alpha; broker sees z only);
impact dY = (h nu - p Y) dt; Q^B' = nu - eta, Q^I' = eta.  Best responses (Props 3.8, 3.9):
  eta_t = (1/2b) E[ -2 psi Q^I_T + int_t^T (alpha_s + h nu_s - p Y_s - 2 r_I Q^I_s) ds | F_t ]
  nu_t  = (1/2a) E[ -(2phi-h) Q^B_T + int_t^T (alpha_s + h eta_s - p Y_s - (h p e^{-(s-t)p} + 2 r_B) Q^B_s) ds | G_t ]
Solver: Picard iteration. Given current policies, simulate forward, compute the pathwise bracket, regress it on each player's OWN
information (trader: full state; broker: GRU over the price-increment path), damp, repeat. Regression on restricted inputs = projection.
Exact reference at h = 0: trader = Riccati ODEs (P, R); broker = Kalman-Bucy on (alpha, Q^I) + LQR Riccati on (Qhat^B, alphahat, Qhat^I).
Usage: python3 wj_game.py <h> <T> [iters]"""
import sys, os, math, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(4); torch.manual_seed(0)
A_, B_, PHI, PSI, RB, RI, P_, KAP, SA, SIG, QB0, QI0 = float(os.environ.get('A_COST', '1.0')), float(os.environ.get('B_COST', '1.0')), 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0
H = float(sys.argv[1]); T = float(sys.argv[2]); ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 15
MODE = os.environ.get('BROKER', 'gru'); N = int(os.environ.get('N_STEPS', '40')); DT = T / N; M = int(os.environ.get('M_PATHS', '4096')); M_VAL = 2048; STEPS = int(os.environ.get('FIT_STEPS', '300'))
tgrid = np.linspace(0, T, N + 1)

def simulate(m, seed):
    rng = np.random.default_rng(seed)
    al = np.zeros((m, N + 1)); al[:, 0] = math.sqrt(SA ** 2 / (2 * KAP)) * rng.standard_normal(m)
    dz = np.zeros((m, N))
    for k in range(N):
        dz[:, k] = al[:, k] * DT + SIG * math.sqrt(DT) * rng.standard_normal(m)
        al[:, k + 1] = al[:, k] - KAP * al[:, k] * DT + SA * math.sqrt(DT) * rng.standard_normal(m)
    return al, dz

def kalman_alpha(dz):                                  # scalar Kalman-Bucy for alpha from z (control-independent); returns alphahat on grid
    m = dz.shape[0]; ah = np.zeros((m, N + 1)); S = SA ** 2 / (2 * KAP)
    for k in range(N):
        K = S / SIG ** 2
        ah[:, k + 1] = ah[:, k] + (-KAP * ah[:, k]) * DT + K * (dz[:, k] - ah[:, k] * DT)
        S = S + (-2 * KAP * S + SA ** 2 - S ** 2 / SIG ** 2) * DT
    return ah

# ---------------- exact reference at h = 0 ----------------
def exact_h0(al, dz):
    # trader: eta = (-P Q^I + R alpha)/(2b);  P' = P^2/(2b) - 2 r_I, P(T) = 2 psi;  R' = (P/(2b) + kappa) R - 1, R(T) = 0
    P = np.zeros(N + 1); R = np.zeros(N + 1); P[N] = 2 * PSI
    for k in range(N, 0, -1):
        P[k - 1] = P[k] - DT * (P[k] ** 2 / (2 * B_) - 2 * RI); R[k - 1] = R[k] - DT * ((P[k] / (2 * B_) + KAP) * R[k] - 1)
    m = al.shape[0]; QI = np.zeros((m, N + 1)); eta = np.zeros((m, N))
    for k in range(N):
        eta[:, k] = (-P[k] * QI[:, k] + R[k] * al[:, k]) / (2 * B_); QI[:, k + 1] = QI[:, k] + eta[:, k] * DT
    # broker: Kalman-Bucy on (alpha, Q^I) with dQ^I = eta dt;  then LQR on xi = (Qhat^B, alphahat, Qhat^I)
    Fm = lambda k: np.array([[-KAP, 0.0], [R[k] / (2 * B_), -P[k] / (2 * B_)]])
    Hm = np.array([1.0, 0.0]); Gc = np.diag([SA ** 2, 0.0])
    Sig = np.diag([SA ** 2 / (2 * KAP), 0.0]); Ks = []
    for k in range(N):
        Ks.append(Sig @ Hm / SIG ** 2); F = Fm(k)
        Sig = Sig + DT * (F @ Sig + Sig @ F.T + Gc - np.outer(Sig @ Hm, Sig @ Hm) / SIG ** 2)
    # LQR: xi' = A(t) xi + B nu + noise;  reward xi'M xi - a nu^2, terminal -phi Qhat^B_T^2;  S' = -(M + A'S + SA + S B B' S / a)
    Mq = np.zeros((3, 3)); Mq[0, 0] = -RB; Mq[0, 1] = Mq[1, 0] = 0.5; Bv = np.array([1.0, 0, 0])
    S = np.zeros((N + 1, 3, 3)); S[N] = np.diag([-PHI, 0, 0])
    for k in range(N, 0, -1):
        Am = np.array([[0, -R[k] / (2 * B_), P[k] / (2 * B_)], [0, -KAP, 0], [0, R[k] / (2 * B_), -P[k] / (2 * B_)]])
        S[k - 1] = S[k] + DT * (Mq + Am.T @ S[k] + S[k] @ Am + S[k] @ np.outer(Bv, Bv) @ S[k] / A_)
    mh = np.zeros((m, 2)); QBh = np.zeros(m); nu = np.zeros((m, N)); QB = np.zeros((m, N + 1)); XI = np.zeros((m, N + 1, 3))
    for k in range(N):
        xi = np.c_[QBh, mh]; XI[:, k] = xi; nu[:, k] = (xi @ S[k][0]) / A_
        inn = dz[:, k] - mh[:, 0] * DT; etah = (-P[k] * mh[:, 1] + R[k] * mh[:, 0]) / (2 * B_)
        mh = mh + (mh @ Fm(k).T) * DT + np.outer(inn, Ks[k]); QBh = QBh + (nu[:, k] - etah) * DT - Ks[k][1] * inn
        QB[:, k + 1] = QB[:, k] + (nu[:, k] - eta[:, k]) * DT
    XI[:, N] = np.c_[QBh, mh]
    return eta, nu, QI, QB, XI

# ---------------- deep Picard-regression solver ----------------
def mlp(i, o, hdim=64): return nn.Sequential(nn.Linear(i, hdim), nn.Tanh(), nn.Linear(hdim, hdim), nn.Tanh(), nn.Linear(hdim, o))
class Broker(nn.Module):                                  # nu_t = f(t, Y_t, alphahat_t, feats_t); feats = GRU(dz_{0:t}) | exact filter | full state
    def __init__(s, hid=32):
        super().__init__(); s.hid = hid if MODE == 'gru' else (3 if MODE == 'exact' else 3)
        s.gru = nn.GRU(2, hid, batch_first=True) if MODE == 'gru' else None; s.head = mlp(3 + s.hid, 1); s.extra = None
    def features(s, dz, ah):                              # (m, N+1, hid): information available before step k
        if MODE == 'gru':
            x = torch.stack([dz / math.sqrt(DT), ah[:, :-1]], -1); out, _ = s.gru(x)
            return torch.cat([torch.zeros(dz.shape[0], 1, out.shape[-1]), out], 1)
        return s.extra if s.extra.shape[0] == dz.shape[0] else s.extra[s._idx]   # set externally: exact filtered (QBhat, alphahat, QIhat) or full (alpha, QB, QI)
    def forward(s, t, Y, ah, hb): return s.head(torch.cat([t, Y, ah, hb], -1)).squeeze(-1)
class Trader(nn.Module):                                  # eta_t = g(t, alpha, Q^B, Q^I, Y, GRU features of broker)
    def __init__(s, hid=32): super().__init__(); s.net = mlp(5 + hid, 1)
    def forward(s, t, al, QB, QI, Y, hb): return s.net(torch.cat([t, al, QB, QI, Y, hb], -1)).squeeze(-1)

def rollout(brk, trd, al, dz, ah, grad=False):
    m = al.shape[0]; t_ = torch.tensor(tgrid[:-1], dtype=torch.float32).expand(m, N)
    with torch.set_grad_enabled(grad):
        hb = None if MODE == 'full' else brk.features(dz, ah)[:, :-1]
    QB = torch.zeros(m, N + 1); QI = torch.zeros(m, N + 1); Y = torch.zeros(m, N + 1); nu = torch.zeros(m, N); eta = torch.zeros(m, N)
    with torch.no_grad():
        for k in range(N):
            tk = t_[:, k:k + 1]
            hbk = torch.cat([al[:, k:k + 1], QB[:, k:k + 1], QI[:, k:k + 1]], -1) if MODE == 'full' else hb[:, k]
            nu[:, k] = brk(tk, Y[:, k:k + 1], ah[:, k:k + 1], hbk)
            eta[:, k] = trd(tk, al[:, k:k + 1], QB[:, k:k + 1], QI[:, k:k + 1], Y[:, k:k + 1], hbk)
            QB[:, k + 1] = QB[:, k] + (nu[:, k] - eta[:, k]) * DT; QI[:, k + 1] = QI[:, k] + eta[:, k] * DT
            Y[:, k + 1] = Y[:, k] + (H * nu[:, k] - P_ * Y[:, k]) * DT
    return nu, eta, QB, QI, Y

def targets(al, nu, eta, QB, QI, Y):
    a = al[:, :-1]; yk = Y[:, :-1]
    gI = a + H * nu - P_ * yk - 2 * RI * QI[:, :-1]                                  # trader integrand
    tI = (-2 * PSI * QI[:, -1:] + torch.flip(torch.cumsum(torch.flip(gI, [1]), 1), [1]) * DT) / (2 * B_)
    base = a + H * eta - P_ * yk - 2 * RB * QB[:, :-1]
    tail = torch.flip(torch.cumsum(torch.flip(base, [1]), 1), [1]) * DT
    kern = H * P_ * torch.exp(-P_ * DT * (torch.arange(N)[None, :] - torch.arange(N)[:, None]).clamp(min=0).float()) * (torch.arange(N)[None, :] >= torch.arange(N)[:, None]).float()
    conv = (QB[:, :-1] @ kern.T) * DT                                             # sum_{s>=t} h p e^{-(s-t)p} Q^B_s dt
    tB = (-(2 * PHI - H) * QB[:, -1:] + tail - conv) / (2 * A_)
    return tI, tB

def fit(brk, trd, al, dz, ah, hb_detached, tI, tB, steps=300, lr=None):
    lr = lr or float(os.environ.get('LR', '2e-3')); BATCH = int(os.environ.get('BATCH', '512'))
    opt = torch.optim.Adam(list(brk.parameters()) + list(trd.parameters()), lr=lr); m = al.shape[0]
    t_ = torch.tensor(tgrid[:-1], dtype=torch.float32).expand(m, N)[..., None]
    nu, eta, QB, QI, Y = rollout(brk, trd, al, dz, ah)   # state paths fixed during the fit (current policies)
    for it in range(steps):
        idx = torch.randint(0, m, (512,)); brk._idx = idx
        hb = torch.cat([al[idx, :-1, None], QB[idx, :-1, None], QI[idx, :-1, None]], -1) if MODE == 'full' else brk.features(dz[idx], ah[idx])[:, :-1]
        pb = brk(t_[idx], Y[idx, :-1, None], ah[idx, :-1, None], hb)
        pi = trd(t_[idx], al[idx, :-1, None], QB[idx, :-1, None], QI[idx, :-1, None], Y[idx, :-1, None], hb.detach())
        loss = ((pb - tB[idx]) ** 2).mean() + ((pi - tI[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

if __name__ == '__main__':
    t0 = time.time()
    al_tr, dz_tr = simulate(M, 1); al_v, dz_v = simulate(M_VAL, 7)
    ah_tr, ah_v = kalman_alpha(dz_tr), kalman_alpha(dz_v)
    T_ = lambda x: torch.tensor(x, dtype=torch.float32)
    al_tr, dz_tr, ah_tr, al_v, dz_v, ah_v = map(T_, (al_tr, dz_tr, ah_tr, al_v, dz_v, ah_v))
    brk = Broker(); trd = Trader(hid=brk.hid); rho = float(os.environ.get('RHO_DAMP', '0.7'))
    if H == 0:
        eta_ex, nu_ex, _, _, XIv = exact_h0(al_v.numpy(), dz_v.numpy()); eta_ex, nu_ex = T_(eta_ex), T_(nu_ex)
        if MODE == 'exact':
            _, _, _, _, XItr = exact_h0(al_tr.numpy(), dz_tr.numpy())
            feats_tr, feats_v = T_(XItr), T_(XIv)
    if MODE == 'full':                                     # reference: linear-regression Picard on the full state (exact for this LQ game)
        def lin_rollout(al, cB, cI):
            m = al.shape[0]; nu = torch.zeros(m, N); eta = torch.zeros(m, N); QB = torch.zeros(m, N + 1); QI = torch.zeros(m, N + 1); Y = torch.zeros(m, N + 1)
            for k in range(N):
                X = torch.stack([torch.ones(m), al[:, k], QB[:, k], QI[:, k], Y[:, k]], 1); nu[:, k] = X @ cB[k]; eta[:, k] = X @ cI[k]
                QB[:, k + 1] = QB[:, k] + (nu[:, k] - eta[:, k]) * DT; QI[:, k + 1] = QI[:, k] + eta[:, k] * DT; Y[:, k + 1] = Y[:, k] + (H * nu[:, k] - P_ * Y[:, k]) * DT
            return nu, eta, QB, QI, Y
        def lin_picard(al, iters=60, rho=float(os.environ.get('RHO_REF', '0.7'))):
            cB = torch.zeros(N, 5); cI = torch.zeros(N, 5)
            for it in range(iters):
                nu, eta, QB, QI, Y = lin_rollout(al, cB, cI); tI, tB = targets(al, nu, eta, QB, QI, Y)
                for k in range(N):
                    X = torch.stack([torch.ones(al.shape[0]), al[:, k], QB[:, k], QI[:, k], Y[:, k]], 1).double()   # float64 lstsq (float32 gave ~6% artefacts)
                    Xn = X.numpy(); ls = lambda y: torch.tensor(np.linalg.lstsq(Xn, y.double().numpy(), rcond=1e-6)[0], dtype=torch.float32)
                    cB[k] = (1 - rho) * cB[k] + rho * ls(tB[:, k])
                    cI[k] = (1 - rho) * cI[k] + rho * ls(tI[:, k])
            return cB, cI
        al_big, _ = map(T_, simulate(65536, 3)); cB, cI = lin_picard(al_big)
        nu_ex, eta_ex, _, _, _ = lin_rollout(al_v, cB, cI); print('linear reference fitted', flush=True)
    prev = None
    for it in range(ITERS):
        if MODE == 'exact': brk.extra = feats_tr
        nu, eta, QB, QI, Y = rollout(brk, trd, al_tr, dz_tr, ah_tr)
        tI, tB = targets(al_tr, nu, eta, QB, QI, Y)
        tI = (1 - rho) * eta + rho * tI; tB = (1 - rho) * nu + rho * tB                  # damped targets
        loss = fit(brk, trd, al_tr, dz_tr, ah_tr, None, tI, tB, steps=STEPS)
        if MODE == 'exact': brk.extra = feats_v
        nu_v, eta_v, QBv, QIv, Yv = rollout(brk, trd, al_v, dz_v, ah_v)
        chg = 0.0 if prev is None else float(((nu_v - prev[0]) ** 2 + (eta_v - prev[1]) ** 2).mean().sqrt())
        # fixed-point residual on validation paths (how far current policies are from their own best responses)
        tIv, tBv = targets(al_v, nu_v, eta_v, QBv, QIv, Yv)
        resI = float(((eta_v - tIv) ** 2).mean().sqrt() / (tIv ** 2).mean().sqrt()); resB = float(((nu_v - tBv) ** 2).mean().sqrt() / (tBv ** 2).mean().sqrt())
        msg = f'it {it:2d} loss {loss:.2e} change {chg:.4f} | BR residual (pathwise) trader {resI:.3f} broker {resB:.3f}'
        if H == 0 or MODE == 'full':
            eI = float(((eta_v - eta_ex) ** 2).mean().sqrt() / (eta_ex ** 2).mean().sqrt()); eB = float(((nu_v - nu_ex) ** 2).mean().sqrt() / (nu_ex ** 2).mean().sqrt())
            msg += f' | rel L2 error vs exact: eta {eI:.3f} nu {eB:.3f}'
        print(msg + f' ({time.time()-t0:.0f}s)', flush=True); prev = (nu_v, eta_v)
    print('nu scale', float(nu_v.abs().mean()), 'eta scale', float(eta_v.abs().mean()))
    out = dict(t=tgrid[:-1], nu=nu_v.numpy(), eta=eta_v.numpy(), alpha=al_v.numpy(), QB=QBv.numpy(), QI=QIv.numpy(), Y=Yv.numpy())
    if H == 0 or MODE == 'full': out.update(nu_ex=nu_ex.numpy(), eta_ex=eta_ex.numpy())
    np.savez(os.environ.get('SAVE', '../results/last.npz'), **out)
