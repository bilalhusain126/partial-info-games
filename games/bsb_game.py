"""Bergault & Sanchez-Betancourt (arXiv 2401.05257): mean-field game between informed traders (private signals) and a broker.
Deep Picard-regression solver with a finite population of M_POP traders per path approximating the mean field, checked against
the paper's Riccati closed form (Sec. 3.7).
Model: d alpha = -k_a alpha dt + s_a dW ; d alpha^i = -k_I alpha^i dt + s_I dW^i ; dQ^i = nu^i dt ; dQbarB = (nu^B - nubar) dt,
nubar = population-average client rate; Qbar = population-average inventory.
Best responses (Thms 3.4, 3.7):
  2 eta_I nu^i_t = E[ -2 a_I Q^i_T + int_t^T ( b nu^B + alpha^i + alpha - 2 phi_I Q^i ) du | F^i_t ]
  2 eta_B nu^B_t = E[ (b - 2 a_B) QbarB_T + int_t^T ( b nubar + alpha - 2 phi_B QbarB ) du | F^B_t ]
Information: trader i sees (alpha, alpha^i, Q^i) and the F^alpha-adapted mean-field states (Qbar, QbarB); broker sees (alpha, Qbar, QbarB).
Usage: python3 bsb_game.py [iters]   (env: PARAMS=paper|gentle, M_PATHS, M_POP, FIT_STEPS, BATCH, N_STEPS)"""
import os, sys, math, time, numpy as np, torch, torch.nn as nn
from scipy.integrate import solve_ivp
torch.set_num_threads(int(os.environ.get('THREADS', '4'))); torch.manual_seed(0)
DEVICE = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')   # DEVICE=cuda|mps|cpu; all new tensors go there
torch.set_default_device(DEVICE); print(f'device {DEVICE}', flush=True)

PAR = os.environ.get('PARAMS', 'gentle')
if PAR == 'paper':  T, KA, SA, KI, SI, ETA_I, ETA_B, B_, A_I, A_B, PHI_I, PHI_B = 1.0, 5.0, 1.0, 5.0, 0.5, 1e-3, 1.2e-3, 1e-3, 1.0, 1.0, 1e-2, 1e-2
else:               T, KA, SA, KI, SI, ETA_I, ETA_B, B_, A_I, A_B, PHI_I, PHI_B = 1.0, 1.0, 1.0, 1.0, 0.5, 0.1, 0.1, 0.05, 1.0, 1.0, 0.1, 0.1
N = int(os.environ.get('N_STEPS', '100')); DT = T / N; tg = np.linspace(0, T, N + 1)
M = int(os.environ.get('M_PATHS', '2048')); MPOP = int(os.environ.get('M_POP', '32')); M_VAL = 512

# ---------------- closed form (Sec. 3.7): six Riccati ODEs for (g, h), linear ODE for (h^a, g^a), then trader f's ----------------
def closed_form():
    abar, phibar = A_I, PHI_I
    def rhs(t, y):                                        # y = (gb, hb, gc, hc); dy/dt = -[...]
        gb, hb, gc, hc = y
        return [-(gb * gb + gc * (hb - gb) + (B_ * hb - 2 * phibar) / (2 * ETA_I)), -(hb * gb + hc * (hb - gb) + B_ * gb / (2 * ETA_B)),
                -(gb * gc + gc * (hc - gc) + B_ * hc / (2 * ETA_I)), -(hb * gc + hc * (hc - gc) + (B_ * gc - 2 * PHI_B) / (2 * ETA_B))]
    s1 = solve_ivp(rhs, [T, 0], [-abar / ETA_I, 0.0, 0.0, -(2 * A_B - B_) / (2 * ETA_B)], t_eval=tg[::-1], method='Radau', rtol=1e-10, atol=1e-12)
    gb, hb, gc, hc = [s1.y[i][::-1] for i in range(4)]
    from scipy.interpolate import interp1d
    I = lambda v: interp1d(tg, v, kind='cubic')
    Igb, Ihb, Igc, Ihc = I(gb), I(hb), I(gc), I(hc)
    def rhs2(t, y):                                       # (ga, ha)
        ga, ha = y
        return [-(-KA * ga + Igb(t) * ga + Igc(t) * (ha - ga) + (B_ * ha + 1) / (2 * ETA_I)), -(-KA * ha + Ihb(t) * ga + Ihc(t) * (ha - ga) + (B_ * ga + 1) / (2 * ETA_B))]
    s2 = solve_ivp(rhs2, [T, 0], [0.0, 0.0], t_eval=tg[::-1], method='Radau', rtol=1e-10, atol=1e-12); ga, ha = s2.y[0][::-1], s2.y[1][::-1]
    Iga, Iha = I(ga), I(ha)
    def rhs3(t, y):                                       # trader: (fa, faI, fb, fbI, fc)   (3.18)-(3.22)
        fa, faI, fb, fbI, fc = y
        return [-(-KA * fa + fb * Iga(t) + fbI * fa + fc * (Iha(t) - Iga(t)) + (B_ * Iha(t) + 1) / (2 * ETA_I)), -(-KI * faI + fbI * faI + 1 / (2 * ETA_I)),
                -(fb * Igb(t) + fbI * fb + fc * (Ihb(t) - Igb(t)) + B_ * Ihb(t) / (2 * ETA_I)), -(fbI * fbI - PHI_I / ETA_I),
                -(fb * Igc(t) + fbI * fc + fc * (Ihc(t) - Igc(t)) + B_ * Ihc(t) / (2 * ETA_I))]
    s3 = solve_ivp(rhs3, [T, 0], [0.0, 0.0, 0.0, -A_I / ETA_I, 0.0], t_eval=tg[::-1], method='Radau', rtol=1e-10, atol=1e-12)
    fa, faI, fb, fbI, fc = [s3.y[i][::-1] for i in range(5)]
    return dict(ga=ga, ha=ha, gb=gb, hb=hb, gc=gc, hc=hc, fa=fa, faI=faI, fb=fb, fbI=fbI, fc=fc)

def simulate(m, seed):
    rng = np.random.default_rng(seed); al = np.zeros((m, N + 1)); alI = np.zeros((m, MPOP, N + 1))
    for k in range(N):
        al[:, k + 1] = al[:, k] - KA * al[:, k] * DT + SA * math.sqrt(DT) * rng.standard_normal(m)
        alI[:, :, k + 1] = alI[:, :, k] - KI * alI[:, :, k] * DT + SI * math.sqrt(DT) * rng.standard_normal((m, MPOP))
    return al, alI

def reference_paths(cf, al, alI):
    """Closed-form equilibrium along given paths: MF states from the alpha path, then each trader via (3.17)."""
    m = al.shape[0]; Qb = np.zeros(m); QB = np.zeros(m); Qi = np.zeros((m, MPOP)); nuB = np.zeros((m, N)); nubar = np.zeros((m, N)); nui = np.zeros((m, MPOP, N))
    for k in range(N):
        nubar[:, k] = cf['ga'][k] * al[:, k] + cf['gb'][k] * Qb + cf['gc'][k] * QB; nuB[:, k] = cf['ha'][k] * al[:, k] + cf['hb'][k] * Qb + cf['hc'][k] * QB
        nui[:, :, k] = cf['fa'][k] * al[:, k, None] + cf['faI'][k] * alI[:, :, k] + cf['fb'][k] * Qb[:, None] + cf['fbI'][k] * Qi + cf['fc'][k] * QB[:, None]
        Qb = Qb + nubar[:, k] * DT; QB = QB + (nuB[:, k] - nubar[:, k]) * DT; Qi = Qi + nui[:, :, k] * DT
    return nuB, nubar, nui

# ---------------- solver ----------------
def mlp(i, o, h=64): return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
SC = 1.0 / max(1.0, abs(A_I / ETA_I) / 10)              # output scale so that the network's raw output is O(1)
class Nets(nn.Module):
    def __init__(s): super().__init__(); s.trd = mlp(6, 1); s.brk = mlp(4, 1)
    def nu_i(s, t, al, alI, Qi, Qb, QB): return s.trd(torch.stack([t, al, alI, Qi, Qb, QB], -1)).squeeze(-1) / SC
    def nu_B(s, t, al, Qb, QB): return s.brk(torch.stack([t, al, Qb, QB], -1)).squeeze(-1) / SC

def rollout(nets, al, alI):
    m = al.shape[0]; Qi = torch.zeros(m, MPOP, N + 1); Qb = torch.zeros(m, N + 1); QB = torch.zeros(m, N + 1)
    nui = torch.zeros(m, MPOP, N); nuB = torch.zeros(m, N); nubar = torch.zeros(m, N)
    with torch.no_grad():
        for k in range(N):
            t = torch.full((m,), tg[k]); nuB[:, k] = nets.nu_B(t, al[:, k], Qb[:, k], QB[:, k])
            tt = t[:, None].expand(m, MPOP); nui[:, :, k] = nets.nu_i(tt, al[:, k, None].expand(m, MPOP), alI[:, :, k], Qi[:, :, k], Qb[:, k, None].expand(m, MPOP), QB[:, k, None].expand(m, MPOP))
            nubar[:, k] = nui[:, :, k].mean(1); Qi[:, :, k + 1] = Qi[:, :, k] + nui[:, :, k] * DT; Qb[:, k + 1] = Qb[:, k] + nubar[:, k] * DT; QB[:, k + 1] = QB[:, k] + (nuB[:, k] - nubar[:, k]) * DT
    return nui, nuB, nubar, Qi, Qb, QB

def brackets(al, alI, nui, nuB, nubar, Qi, Qb, QB):
    rc0 = lambda x: torch.flip(torch.cumsum(torch.flip(x, [-1]), -1), [-1]) * DT
    rc = (lambda x: rc0(x) - x * DT) if os.environ.get('FOC', 'discrete') == 'discrete' else rc0   # discrete: exact first order condition of the discrete game (sum over j > k)
    gI = B_ * nuB[:, None, :] + alI[:, :, :-1] + al[:, None, :-1] - 2 * PHI_I * Qi[:, :, :-1]
    tI = (-2 * A_I * Qi[:, :, -1:] + rc(gI)) / (2 * ETA_I)
    gB = B_ * nubar + al[:, :-1] - 2 * PHI_B * QB[:, :-1]
    tB = ((B_ - 2 * A_B) * QB[:, -1:] + rc(gB)) / (2 * ETA_B)
    return tI, tB

def fit(nets, al, alI, Qi, Qb, QB, tI, tB, steps, batch, lr):
    opt = torch.optim.Adam(nets.parameters(), lr=lr); m = al.shape[0]; tt = torch.tensor(tg[:-1], dtype=torch.float32)
    for it in range(steps):
        i = torch.randint(0, m, (batch,)); j = torch.randint(0, MPOP, (batch,))
        pI = nets.nu_i(tt.expand(batch, N), al[i, :-1], alI[i, j, :-1], Qi[i, j, :-1], Qb[i, :-1], QB[i, :-1]); pB = nets.nu_B(tt.expand(batch, N), al[i, :-1], Qb[i, :-1], QB[i, :-1])
        loss = ((pI - tI[i, j]) ** 2).mean() * SC ** 2 + ((pB - tB[i]) ** 2).mean() * SC ** 2
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(nets.parameters(), 10.0); opt.step()
    return float(loss)

if __name__ == '__main__':
    ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12; STEPS = int(os.environ.get('FIT_STEPS', '600')); BATCH = int(os.environ.get('BATCH', '256')); LR = float(os.environ.get('LR', '1e-3')); rho = 0.6
    t0 = time.time(); cf = closed_form()
    if os.environ.get('CHECK_CF'):
        for k, tt_ in [(0, 0.0), (N // 2, 0.5), (int(0.95 * N), 0.95)]: print(f"t={tt_}: ga {cf['ga'][k]:.4g} ha {cf['ha'][k]:.4g} gb {cf['gb'][k]:.4g} hb {cf['hb'][k]:.4g} gc {cf['gc'][k]:.4g} hc {cf['hc'][k]:.4g} | fa {cf['fa'][k]:.4g} faI {cf['faI'][k]:.4g} fb {cf['fb'][k]:.4g} fbI {cf['fbI'][k]:.4g} fc {cf['fc'][k]:.4g}")
        print('identities (fa=ga, fc=gc, fb+fbI=gb): ', np.abs(cf['fa'] - cf['ga']).max(), np.abs(cf['fc'] - cf['gc']).max(), np.abs(cf['fb'] + cf['fbI'] - cf['gb']).max()); sys.exit()
    T_ = lambda x: torch.tensor(x, dtype=torch.float32)
    al, alI = simulate(M, 1); alv, alIv = simulate(M_VAL, 7); nuB_ex, nubar_ex, nui_ex = map(T_, reference_paths(cf, alv, alIv)); al, alI, alv, alIv = map(T_, (al, alI, alv, alIv))
    nets = Nets(); prev = None; AVG_FROM = int(os.environ.get('AVG_FROM', '6')); avg = None; n_avg = 0   # Polyak average of validation controls from iteration AVG_FROM on
    print(f'PARAMS={PAR} N={N} M={M} MPOP={MPOP} | reference scales: |nuB| {float(nuB_ex.abs().mean()):.3f} |nubar| {float(nubar_ex.abs().mean()):.3f} |nu_i| {float(nui_ex.abs().mean()):.3f}', flush=True)
    rel = lambda a, b: float(((a - b) ** 2).mean().sqrt() / (b ** 2).mean().sqrt())
    for it in range(ITERS):
        nui, nuB, nubar, Qi, Qb, QB = rollout(nets, al, alI); tI, tB = brackets(al, alI, nui, nuB, nubar, Qi, Qb, QB)
        loss = fit(nets, al, alI, Qi, Qb, QB, (1 - rho) * nui + rho * tI, (1 - rho) * nuB + rho * tB, STEPS, BATCH, LR)
        nuiv, nuBv, nubarv, _, _, _ = rollout(nets, alv, alIv)
        chg = 0.0 if prev is None else float(((nuBv - prev[0]) ** 2).mean().sqrt() + ((nuiv - prev[1]) ** 2).mean().sqrt())
        msg = f'it {it:2d} loss {loss:.2e} change {chg:.4f} | rel L2 vs closed form: broker {rel(nuBv, nuB_ex):.3f} avg flow {rel(nubarv, nubar_ex):.3f} trader_i {rel(nuiv, nui_ex):.3f}'
        if it >= AVG_FROM:
            n_avg += 1; avg = [x.clone() for x in (nuBv, nubarv, nuiv)] if avg is None else [a + (x - a) / n_avg for a, x in zip(avg, (nuBv, nubarv, nuiv))]
            msg += f' | avg({n_avg}): broker {rel(avg[0], nuB_ex):.3f} avg flow {rel(avg[1], nubar_ex):.3f} trader_i {rel(avg[2], nui_ex):.3f}'
        print(msg + f' ({time.time()-t0:.0f}s)', flush=True); prev = (nuBv, nuiv)
    np.savez(os.environ.get('SAVE', os.path.join(os.environ.get('RESULTS_DATA', os.path.expanduser('~/Desktop/Bilal Saleh Husain/research-data/partial-info-games')), f'bsb_last.npz')), t=tg[:-1], nuB=nuBv.cpu().numpy(), nuB_ex=nuB_ex.cpu().numpy(), nubar=nubarv.cpu().numpy(), nubar_ex=nubar_ex.cpu().numpy(), nui=nuiv[:, 0].cpu().numpy(), nui_ex=nui_ex[:, 0].cpu().numpy())
