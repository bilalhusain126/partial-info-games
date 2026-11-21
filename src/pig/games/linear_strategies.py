"""Exact equilibria of the two-trader game in the class of strategies linear in the own observed path.

The shocks of the discretised game form a vector e in R^K, K = 2 + 3N: the two initial signals and, at each step,
the increments of the two signal noises and of the price noise, all scaled to unit variance. Trader i uses
    nu^i_j = sum_{k <= j} ( beta^i[j, k, 0] alpha^i_k + beta^i[j, k, 1] S_k ),
so every process is linear in e and is represented by its loading vector. The cost of trader i is quadratic in
beta^i given the other trader's coefficients, and its best response is computed by L-BFGS. In this Gaussian
linear-quadratic game a best response to a linear strategy is linear, so an equilibrium in this class is an
equilibrium of the game.
Open loop: the other trader's control is held fixed as a process (its loading vector). Closed loop: the other
trader's coefficients are held fixed, so its control responds to trader i through the price.
"""

import numpy as np
import torch

CPU = dict(dtype=torch.float64, device='cpu')


class LinearStrategies:
    """Exact solver for a TwoTraders game; all computations in float64 on the CPU."""

    def __init__(self, game):
        """Args: game: a TwoTraders instance."""
        self.game = game; N, dt = game.N, game.dt
        self.K = 2 + 3 * N
        self.alpha = torch.zeros(2, N + 1, self.K, **CPU)            # loadings of the signals
        for i in range(2):
            self.alpha[i, 0, i] = game.signal_variance ** 0.5
            for k in range(N):
                self.alpha[i, k + 1] = (1 - game.kappa * dt) * self.alpha[i, k]
                self.alpha[i, k + 1, 2 + 3 * k + i] += game.sigma_alpha * dt ** 0.5
        self.price_noise = torch.zeros(N, self.K, **CPU)
        for k in range(N):
            self.price_noise[k, 2 + 3 * k + 2] = game.sigma * dt ** 0.5
        self.causal = torch.tril(torch.ones(N, N, **CPU))[:, :, None]   # beta[j, k] is used for k <= j only

    def forward(self, beta, own=None, other_control=None):
        """Loadings of the controls [2, N, K], inventories [2, N+1, K] and drift [N, K] under coefficients beta
        [2, N, N, 2]. With own = i and other_control [N, K], trader i uses beta[i] and the other trader's control
        is held fixed at other_control (open loop)."""
        game = self.game; N, dt = game.N, game.dt
        S = [torch.zeros(self.K, **CPU)]; controls = []
        for j in range(N):
            prices = torch.stack(S)
            u = [beta[i, j, :j + 1, 0] @ self.alpha[i, :j + 1] + beta[i, j, :j + 1, 1] @ prices for i in range(2)]
            if own is not None:
                u[1 - own] = other_control[j]
            controls.append(torch.stack(u))
            S.append(S[-1] + (self.alpha[0, j] + self.alpha[1, j] + game.lam * (u[0] + u[1])) * dt + self.price_noise[j])
        controls = torch.stack(controls, 1)
        q = torch.cat([torch.zeros(2, 1, self.K, **CPU), torch.cumsum(controls, 1) * dt], 1)
        mu = self.alpha[0, :N] + self.alpha[1, :N] + game.lam * (controls[0] + controls[1])
        return controls, q, mu

    def cost(self, i, beta, other_control=None):
        """Expected cost of trader i, E[x y] = x . y for loadings x, y."""
        game = self.game
        controls, q, mu = self.forward(beta * self.causal, None if other_control is None else i, other_control)
        running = -(q[i, :-1] * mu).sum(-1) + game.a * (controls[i] ** 2).sum(-1) + game.phi * (q[i, :-1] ** 2).sum(-1)
        return running.sum() * game.dt + game.psi * (q[i, -1] ** 2).sum()

    def best_response(self, beta, i, closed: bool, inner: int = 200):
        """Coefficients of trader i minimizing its cost against beta[1 - i] (closed) or against the other's control
        process under beta (open)."""
        other_control = None if closed else self.forward(beta * self.causal)[0][1 - i].detach()
        own = beta[i].detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([own], max_iter=inner, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            full = torch.stack([own, beta[1]]) if i == 0 else torch.stack([beta[0], own])
            J = self.cost(i, full, other_control); J.backward(); return J
        optimizer.step(closure)
        return (own * self.causal).detach()

    def solve(self, closed: bool, iterations: int = 150, damping: float = 0.5, start=None, tolerance: float = 1e-6, log=None):
        """Damped best-response iteration from start (default zero coefficients); stops when the relative change
        of the best responses falls below tolerance. Returns beta [2, N, N, 2]; the last change is kept in final_change."""
        N = self.game.N
        beta = torch.zeros(2, N, N, 2, **CPU) if start is None else torch.as_tensor(start, **CPU).clone()
        for iteration in range(iterations):
            new = torch.stack([self.best_response(beta, i, closed) for i in range(2)])
            change = float((new - beta).norm() / max(1.0, float(new.norm())))
            beta = (1 - damping) * beta + damping * new
            if log:
                log(f'iteration {iteration:3d}  change {change:.2e}')
            if change < tolerance:
                break
        self.final_change = change
        return beta

    def controls(self, beta, noise: dict) -> np.ndarray:
        """Controls [M, 2, N] along the paths of noise (TwoTraders.sample)."""
        game = self.game; N, dt = game.N, game.dt
        to_numpy = lambda key: noise[key].cpu().numpy().astype(np.float64)
        M = noise['dWS'].shape[0]; e = np.zeros((M, self.K))
        e[:, 0] = to_numpy('alpha1_0') / game.signal_variance ** 0.5; e[:, 1] = to_numpy('alpha2_0') / game.signal_variance ** 0.5
        for k in range(N):
            for offset, key in enumerate(['dW1', 'dW2', 'dWS']):
                e[:, 2 + 3 * k + offset] = to_numpy(key)[:, k] / dt ** 0.5
        with torch.no_grad():
            loadings = self.forward(torch.as_tensor(beta, **CPU) * self.causal)[0].numpy()
        return np.einsum('ijk,mk->mij', loadings, e)

    def spectrum(self, beta, closed: bool):
        """Eigenvalues of the Jacobian of the undamped best-response map at beta, and the eigenvalues of trader 1's
        Hessian in its own active coefficients. By implicit differentiation, d BR_i / d beta = -H_ii^+ C_i, with
        H_ii and C_i the derivatives of trader i's gradient in its own and in all coefficients."""
        N = self.game.N
        active = (self.causal.bool().expand(N, N, 1)).expand(N, N, 2)
        P = int(active.sum()); beta = torch.as_tensor(beta, **CPU) * self.causal
        vector = torch.cat([beta[0][active], beta[1][active]])

        def unpack(v):
            b = torch.zeros(N, N, 2, **CPU); b[active] = v; return b
        rows = []; hessian_eigenvalues = None
        for i in range(2):
            own = vector[i * P:(i + 1) * P].clone().requires_grad_(True); all_coefficients = vector.clone().requires_grad_(True)
            b = [unpack(all_coefficients[:P]), unpack(all_coefficients[P:])]
            if closed:
                b[i] = unpack(own); J = self.cost(i, torch.stack(b))
            else:
                other_control = self.forward(torch.stack(b))[0][1 - i]
                b[i] = unpack(own); J = self.cost(i, torch.stack(b), other_control)
            gradient = torch.autograd.grad(J, own, create_graph=True)[0]
            H = torch.zeros(P, P, **CPU); C = torch.zeros(P, 2 * P, **CPU)
            for r in range(P):
                g_own, g_all = torch.autograd.grad(gradient[r], [own, all_coefficients], retain_graph=True, allow_unused=True)
                H[r] = g_own
                if g_all is not None:
                    C[r] = g_all
            if i == 0:
                hessian_eigenvalues = np.linalg.eigvalsh(((H + H.T) / 2).numpy())
            rows.append(-np.linalg.pinv(H.numpy(), rcond=1e-10) @ C.numpy())
        return np.linalg.eigvals(np.vstack(rows)), hessian_eigenvalues
