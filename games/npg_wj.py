"""Wu-Jaimungal broker-trader game as an npg.Model (engine sanity check against the earlier prototype and the h=0 exact solution).
Players: 0 = informed trader (full info), 1 = broker (prices only)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import sys, os, math, numpy as np, torch
sys.argv = ['x', os.environ.get('H', '0'), os.environ.get('T', '0.3')]
import wj_game as W          # reuse constants, simulate(), kalman_alpha(), exact_h0()
from npg import Solver

class WJModel:
    n_players = 2; N = W.N; DT = W.DT; d_now = [5, 2]; d_seq = [0, 2]
    d_full = 5; d_w = 2; cv_gamma = [1.0, 1.0]
    def full_obs(s, exo, st): return torch.stack([exo['al'][:, :-1], st['QB'][:, :-1], st['QI'][:, :-1], st['Y'][:, :-1], exo['ah'][:, :-1]], -1)
    def dW(s, exo):                                       # recover the two Brownian increments (signal, price noise), scaled by 1/sqrt(dt)
        al, dz = exo['al'], exo['dz']; dWa = (al[:, 1:] - al[:, :-1] + W.KAP * al[:, :-1] * s.DT) / W.SA; dWz = (dz - al[:, :-1] * s.DT) / W.SIG
        return torch.stack([dWa, dWz], -1) / math.sqrt(s.DT)
    def exo(s, m, seed):
        al, dz = W.simulate(m, seed); ah = W.kalman_alpha(dz)
        T_ = lambda x: torch.tensor(x, dtype=torch.float32); return dict(al=T_(al), dz=T_(dz), ah=T_(ah))
    def init_state(s, m): return dict(QB=torch.zeros(m, s.N + 1), QI=torch.zeros(m, s.N + 1), Y=torch.zeros(m, s.N + 1))
    def step(s, st, u, k):
        nu, eta = u[1], u[0]
        st['QB'][:, k + 1] = st['QB'][:, k] + (nu - eta) * s.DT; st['QI'][:, k + 1] = st['QI'][:, k] + eta * s.DT
        st['Y'][:, k + 1] = st['Y'][:, k] + (W.H * nu - W.P_ * st['Y'][:, k]) * s.DT
    def obs_seq(s, i, exo, st):
        if i == 0: return None
        return torch.stack([exo['dz'] / math.sqrt(s.DT), exo['ah'][:, :-1]], -1)          # broker: price increments + Kalman alphahat
    def obs_now(s, i, exo, st, k):
        if i == 0: return torch.stack([exo['al'][:, k], st['QB'][:, k], st['QI'][:, k], st['Y'][:, k], exo['ah'][:, k]], -1)
        return torch.stack([st['Y'][:, k], exo['ah'][:, k]], -1)
    def bracket(s, i, exo, st, ctrl):
        tI, tB = W.targets(exo['al'], ctrl[1], ctrl[0], st['QB'], st['QI'], st['Y']); return tI if i == 0 else tB

if __name__ == '__main__':
    M = WJModel(); S = Solver(M, steps=int(os.environ.get('STEPS', '800')), batch=1024, m_paths=int(os.environ.get('M_PATHS', '8192')))
    ref = None
    if W.H == 0:
        eta_ex, nu_ex, _, _, _ = W.exact_h0(S.exo_v['al'].numpy(), S.exo_v['dz'].numpy()); ref = [torch.tensor(eta_ex), torch.tensor(nu_ex)]
    S.run(int(os.environ.get('ITERS', '8')), ref=ref)
