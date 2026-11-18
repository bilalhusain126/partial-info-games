"""Generic deep Picard-regression solver for N-player stochastic differential games with asymmetric information.
A Model supplies:
  n_players, N (steps), DT
  exo(m, seed)               -> dict of exogenous paths (numpy), each (m, N+1) or (m, N)
  init_state(m)              -> dict of endogenous state paths initialised at k=0 (torch, (m, N+1))
  step(state, ctrl, k)       -> writes state at k+1 given controls at k (dict of (m,) tensors)
  obs_seq(i, exo, state)     -> (m, N, d_i) tensor: player i's observation sequence (what its recurrent encoder sees up to each k;
                                 entry k must use information available BEFORE acting at k)
  obs_now(i, exo, state, k)  -> (m, d'_i) tensor: player i's current observables (Markov features) at k
  bracket(i, exo, state, ctrl) -> (m, N) tensor: pathwise best-response functional of player i (Props 3.8/3.9-type)
Solver: simulate under current policies, evaluate brackets, damp, regress each player's policy on its own information, repeat."""
import os, math, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(int(os.environ.get('THREADS', '4')))
DEVICE = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')   # DEVICE=cuda|mps|cpu; all new tensors go there
torch.set_default_device(DEVICE); print(f'device {DEVICE}', flush=True)


def mlp(i, o, h=64): return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
class Policy(nn.Module):
    """u_i(k) = head(t_k, obs_now, GRU(obs_seq up to k)); set d_seq = 0 for a purely Markov (no-memory) policy."""
    def __init__(s, d_now, d_seq, hid=32):
        super().__init__(); s.d_seq = d_seq; s.hid = hid if d_seq > 0 else 0
        s.gru = nn.GRU(d_seq, hid, batch_first=True) if d_seq > 0 else None; s.head = mlp(1 + d_now + s.hid, 1)
    def mem(s, seq):                                  # (m, N, d_seq) -> (m, N, hid), entry k uses seq[:, :k]  (zero at k=0)
        if s.gru is None: return None
        out, _ = s.gru(seq); return torch.cat([torch.zeros(seq.shape[0], 1, s.hid), out[:, :-1]], 1)
    def forward(s, t, now, memk):
        x = [t, now] + ([memk] if memk is not None else []); return s.head(torch.cat(x, -1)).squeeze(-1)

class Solver:
    """Options (env): CV=1 adds a martingale control variate per player: target_i(k) = bracket_i(k) - sum_{j>=k} gamma_i^{j-k} zeta_i(t_j, full_j) . dW_j,
    with zeta_i a net on FULL information (model.full_obs(exo, state) -> (m, N, d_f), model.dW(exo) -> (m, N, d_w), model.cv_gamma[i]);
    the term has zero conditional mean given any player's information, so the regression's fixed point is unchanged and only its variance drops.
    DAMP=fp uses the fictitious-play schedule rho_n = max(rho, 1/(n+1)) instead of the constant rho."""
    def __init__(s, model, hid=32, rho=0.7, lr=1e-3, batch=2048, steps=1500, m_paths=16384, m_val=2048, seed=0):
        s.M = model; torch.manual_seed(seed); s.rho, s.lr, s.batch, s.steps, s.m, s.m_val = rho, lr, batch, steps, m_paths, m_val
        s.pols = [Policy(model.d_now[i], model.d_seq[i], hid) for i in range(model.n_players)]
        s.cv = bool(int(os.environ.get('CV', '0'))) and hasattr(model, 'full_obs'); s.damp = os.environ.get('DAMP', 'const')
        if s.cv:
            s.zetas = [mlp(1 + model.d_full, model.d_w) for _ in range(model.n_players)]
            j, k = np.indices((model.N, model.N)); s.Wg = [torch.tensor(np.where(j >= k, g ** (j - k), 0.0), dtype=torch.float32) for g in model.cv_gamma]
        s.exo_tr = model.exo(m_paths, 1 + 100 * seed); s.exo_v = model.exo(m_val, 7)
        s.t = torch.tensor(np.linspace(0, model.N * model.DT, model.N + 1)[:-1], dtype=torch.float32)
    def rollout(s, exo, grad=False):
        """Returns state, ctrl, seqs. If the model defines obs_inc(i, exo, state, k) (observation increment over [k, k+1], available
        after step k), the observation sequences are built inside the loop and the GRUs run incrementally; else obs_seq is used."""
        M = s.M; m = next(iter(exo.values())).shape[0]; state = M.init_state(m)
        ctrl = [torch.zeros(m, M.N) for _ in range(M.n_players)]; inc = hasattr(M, 'obs_inc')
        with torch.no_grad():
            if inc:
                seqs = [None if M.d_seq[i] == 0 else torch.zeros(m, M.N, M.d_seq[i]) for i in range(M.n_players)]
                h = [None if p.gru is None else torch.zeros(1, m, p.hid) for p in s.pols]
            else:
                seqs = [M.obs_seq(i, exo, state) for i in range(M.n_players)]; mems = [p.mem(q) for p, q in zip(s.pols, seqs)]
            for k in range(M.N):
                tk = s.t[k].expand(m, 1); uk = {}
                for i, p in enumerate(s.pols):
                    if inc: memk = None if p.gru is None else h[i][0]
                    else: memk = None if mems[i] is None else mems[i][:, k]
                    uk[i] = p(tk, M.obs_now(i, exo, state, k), memk); ctrl[i][:, k] = uk[i]
                M.step(state, uk, k)
                if inc:
                    for i, p in enumerate(s.pols):
                        if p.gru is not None:
                            seqs[i][:, k] = M.obs_inc(i, exo, state, k); _, h[i] = p.gru(seqs[i][:, k:k + 1], h[i])
        return state, ctrl, seqs
    def fit(s, exo, state, ctrl, targets, seqs):
        M = s.M; m = next(iter(exo.values())).shape[0]
        params = [q for p in s.pols for q in p.parameters()] + ([q for z in s.zetas for q in z.parameters()] if s.cv else [])
        opt = torch.optim.Adam(params, lr=s.lr)
        nows = [torch.stack([M.obs_now(i, exo, state, k) for k in range(M.N)], 1) for i in range(M.n_players)]   # (m, N, d)
        tt = s.t.expand(m, M.N)[..., None]
        if s.cv: full = torch.cat([tt, M.full_obs(exo, state)], -1); dW = M.dW(exo)                                 # (m, N, 1+d_f), (m, N, d_w)
        for it in range(s.steps):
            idx = torch.randint(0, m, (s.batch,)); loss = 0.0
            for i, p in enumerate(s.pols):
                memk = p.mem(seqs[i][idx]) if p.gru is not None else None
                pred = p(tt[idx], nows[i][idx], memk); tgt = targets[i][idx]
                if s.cv: tgt = tgt - s.rho_n * ((s.zetas[i](full[idx]) * dW[idx]).sum(-1) @ s.Wg[i])           # M_k = sum_{j>=k} gamma^{j-k} zeta_j.dW_j  (Wg[j,k] = gamma^{j-k}, j>=k)
                loss = loss + ((pred - tgt) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        return float(loss.detach())
    def run(s, iters, ref=None, log=lambda m: print(m, flush=True)):
        M = s.M; prev = None; t0 = time.time(); hist = []; avg_from = int(os.environ.get('AVG_FROM', '4')); avg = None; n_avg = 0
        for it in range(iters):
            state, ctrl, seqs = s.rollout(s.exo_tr); s.rho_n = max(s.rho, 1.0 / (it + 1)) if s.damp == 'fp' else s.rho
            tg = [(1 - s.rho_n) * ctrl[i] + s.rho_n * M.bracket(i, s.exo_tr, state, ctrl) for i in range(M.n_players)]
            loss = s.fit(s.exo_tr, state, ctrl, tg, seqs)
            vstate, vctrl, _ = s.rollout(s.exo_v)
            chg = 0.0 if prev is None else float(sum(((vctrl[i] - prev[i]) ** 2).mean() for i in range(M.n_players)).sqrt())
            msg = f'it {it:2d} rho {s.rho_n:.2f} loss {loss:.2e} change {chg:.4f}'
            if ref is not None:
                errs = [float(((vctrl[i] - ref[i]) ** 2).mean().sqrt() / (ref[i] ** 2).mean().sqrt()) for i in range(M.n_players)]
                msg += ' | rel L2 vs ref: ' + ' '.join(f'u{i} {e:.3f}' for i, e in enumerate(errs)); hist.append(errs)
                if it >= avg_from:                      # Polyak average of the validation controls over iterations
                    n_avg += 1; avg = [c.clone() for c in vctrl] if avg is None else [a + (c - a) / n_avg for a, c in zip(avg, vctrl)]
                    msg += f' | avg({n_avg}): ' + ' '.join(f'u{i} {float(((avg[i] - ref[i]) ** 2).mean().sqrt() / (ref[i] ** 2).mean().sqrt()):.3f}' for i in range(M.n_players))
            log(msg + f' ({time.time()-t0:.0f}s)'); prev = vctrl
        return vstate, (avg if avg is not None else vctrl), hist

def exploitability(S, i, iters=6, steps=None, m_eval=4096, log=lambda m: print(m, flush=True)):
    """Freeze every policy except player i's, re-solve player i's single-agent problem by the same Picard regression with a larger fit
    budget (steps, default 2x), and return J_i(frozen others, re-solved) - J_i(frozen others, original) estimated on fresh paths.
    Requires model.cost(i, exo, state, ctrl) -> (m,) tensor of player i's realised cost (lower is better). Zero gain = best response."""
    import copy
    M = S.M; orig = [copy.deepcopy(p) for p in S.pols]; exo_e = M.exo(m_eval, 11)
    st, ctrl, _ = S.rollout(exo_e); J0 = M.cost(i, exo_e, st, ctrl).mean()
    steps0 = S.steps; S.steps = steps or 2 * steps0
    for p in S.pols: p.requires_grad_(False)
    S.pols[i].requires_grad_(True)
    for it in range(iters):                                     # Picard regression for player i only; others fixed
        state, c, seqs = S.rollout(S.exo_tr)
        tg = [(1 - S.rho) * c[j] + S.rho * M.bracket(j, S.exo_tr, state, c) if j == i else c[j] for j in range(M.n_players)]
        S.rho_n = S.rho; loss = S.fit(S.exo_tr, state, c, tg, seqs)
        st, ctrl, _ = S.rollout(exo_e); J = M.cost(i, exo_e, st, ctrl).mean()
        log(f'  exploit player {i} it {it}: loss {loss:.2e} J_i {float(J):.5f} (original {float(J0):.5f}, gain {float(J0 - J):+.5f})')
    gain = float(J0 - J); S.steps = steps0
    for p, q in zip(S.pols, orig): p.load_state_dict(q.state_dict()); p.requires_grad_(True)
    return gain, float(J0)

def foc_residual(S, i, steps=None, m_fresh=None, log=lambda m: print(m, flush=True)):
    """First-order-condition check for player i: on fresh paths simulated under the current controls, fit a new, independently
    initialised network to player i's undamped target E[bracket | own path] (no Picard feedback), and return the relative L2
    distance between this independent regression and the current control on the validation paths. Its floor is the regression
    noise, which the exact-reference cases calibrate."""
    import copy
    M = S.M; m = m_fresh or S.m; exo_f = M.exo(m, 23 + i)
    state, ctrl, seqs = S.rollout(exo_f)
    target = M.bracket(i, exo_f, state, ctrl)
    saved = [copy.deepcopy(p) for p in S.pols]; fresh = Policy(M.d_now[i], M.d_seq[i], S.pols[i].hid if S.pols[i].gru is not None else 32)
    S.pols[i] = fresh; steps0, rho0 = S.steps, getattr(S, 'rho_n', S.rho); S.steps = steps or 2 * steps0; S.rho_n = 1.0
    targets = [target if j == i else ctrl[j] for j in range(M.n_players)]
    for p in S.pols: p.requires_grad_(False)
    fresh.requires_grad_(True); loss = S.fit(exo_f, state, ctrl, targets, seqs)
    _, vfresh, _ = S.rollout(S.exo_v); S.pols[i] = saved[i]
    for p, q in zip(S.pols, saved): p.load_state_dict(q.state_dict()); p.requires_grad_(True)
    _, vcur, _ = S.rollout(S.exo_v); S.steps, S.rho_n = steps0, rho0
    d = float(((vfresh[i] - vcur[i]) ** 2).mean().sqrt() / (vcur[i] ** 2).mean().sqrt())
    log(f'  FOC residual player {i}: independent regression vs current control, rel L2 {d:.4f} (fit loss {loss:.2e})'); return d
