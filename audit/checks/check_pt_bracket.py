"""Finite-difference / autograd check of pt_game's bracket: is it the open-loop gradient of the discrete cost?"""
import os, sys, math, numpy as np, torch
os.environ.setdefault('LAM', '1.0'); os.environ['DEVICE'] = 'cpu'
sys.path.insert(0, '.'); import pt_game as P
torch.manual_seed(0); M = P.PTModel(); m = 6
exo = M.exo(m, 3); M._e = exo
nu = [torch.randn(m, P.N, dtype=torch.float32) * 0.3 for _ in range(2)]
nu[0].requires_grad_(True)
st = M.init_state(m)
for k in range(P.N): M.step(st, {0: nu[0][:, k], 1: nu[1][:, k]}, k)     # open loop: nu_1 is a fixed tensor
J = M.cost(0, exo, st, nu)                                               # (m,) cost to minimise
grad = torch.autograd.grad(J.sum(), nu[0])[0]                            # dJ/dnu_0[k], per path
with torch.no_grad():
    br = M.bracket(0, exo, st, nu)                                       # = E-free pathwise bracket /(2a)
    pred = 2 * P.A_ * P.DT * (nu[0] - br)                                # FOC: dJ/dnu_0[k] = dt (2a nu - 2a bracket)
print('LAM', P.LAM, ' max |autograd - predicted| =', float((grad - pred).abs().max()), ' scale', float(grad.abs().max()))
# closed-loop check: what the gradient would be if trader 2 reacted linearly to S (slope c): extra term
print('(the bracket is the open-loop derivative; any reaction of nu_1 to S is not in it)')
