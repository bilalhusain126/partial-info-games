"""
Two traders with private Ornstein--Uhlenbeck signals and a common price.

The game of base.TradingGame with the signals

    dalpha^i_t = -kappa alpha^i_t dt + sigma_alpha dW^i_t,   alpha^i_0 ~ N(0, sigma_alpha^2 / (2 kappa)),

each observed by trader i only. Trader i infers the other signal from the price, which also moves with
the other trader's trades, and those trades depend on the other trader's inference of alpha^i. No
finite-dimensional filter closes this hierarchy.

Exact solutions, used to validate the learned method:
    TwoTraders.exact     at lam = 0 the traders decouple, and each solves a discrete LQG problem with the
                         Kalman filter of the other signal.
    LinearStrategies     at lam > 0, the equilibrium among strategies linear in the own observed path,
                         open or closed loop. In this Gaussian game a best response to a linear strategy is
                         linear, so this is the equilibrium of the game.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from .base import TradingGame

FLOAT64_CPU = dict(dtype=torch.float64, device='cpu')


@dataclass
class TwoTraders(TradingGame):
    """Model parameters and the game interface (see base.TradingGame for the price and the rewards)."""

    kappa: float = 1.0        # signal mean reversion
    sigma_alpha: float = 1.0  # signal volatility

    noise_dim = 3             # the control variate uses dW^1, dW^2, dW^S

    @property
    def signal_variance(self) -> float:
        """Stationary variance sigma_alpha^2 / (2 kappa) of each signal, the law of alpha^i_0."""
        return self.sigma_alpha ** 2 / (2 * self.kappa)

    def sample(self, n_paths, seed):
        """
        Simulates the signals by the Euler--Maruyama scheme.

        Returns:
            alpha1_0, alpha2_0: initial signals [M]
            dW1, dW2, dWS: Brownian increments of the signals and of the price [M, N]
            alpha1, alpha2: signal paths [M, N + 1]
        """
        rng = np.random.default_rng(seed)
        dt, N = self.dt, self.N
        noise = dict(alpha1_0=rng.standard_normal(n_paths) * math.sqrt(self.signal_variance),
                     alpha2_0=rng.standard_normal(n_paths) * math.sqrt(self.signal_variance),
                     dW1=rng.standard_normal((n_paths, N)) * math.sqrt(dt),
                     dW2=rng.standard_normal((n_paths, N)) * math.sqrt(dt),
                     dWS=rng.standard_normal((n_paths, N)) * math.sqrt(dt))
        for i in (1, 2):
            alpha = np.zeros((n_paths, N + 1))
            alpha[:, 0] = noise[f'alpha{i}_0']
            for j in range(N):
                alpha[:, j + 1] = alpha[:, j] - self.kappa * alpha[:, j] * dt + self.sigma_alpha * noise[f'dW{i}'][:, j]
            noise[f'alpha{i}'] = alpha
        return {key: torch.tensor(value, dtype=torch.float32) for key, value in noise.items()}

    def signal_increment(self, alpha, j):
        """(alpha_{t_{j+1}} - alpha_{t_j}) / sqrt(dt)."""
        return (alpha[:, j + 1] - alpha[:, j]) / math.sqrt(self.dt)

    def martingale_increments(self, noise):
        """(dW^1, dW^2, dW^S) / sqrt(dt)."""
        return torch.stack([noise['dW1'], noise['dW2'], noise['dWS']], -1) / math.sqrt(self.dt)

    def exact(self, noise):
        """
        Exact equilibrium at lam = 0, [M, 2, N].

        At lam = 0 the price does not depend on the trades, so trader i solves a single-agent problem.
        Its state is (q^i, alpha^i, alphahat^j), where alphahat^j is the Kalman filter of the other signal
        from the price increments dS - alpha^i dt = alpha^j dt + sigma dW^S. The control is the discrete
        LQG feedback nu^i_{t_j} = -L_j (q^i, alpha^i, alphahat^j), with L_j from the backward Riccati
        recursion of the discretised problem.

        Raises:
            NotImplementedError: If lam != 0.
        """
        if self.lam != 0.0:
            return super().exact(noise)
        dt, N, kappa = self.dt, self.N, self.kappa

        # Kalman gains of the other signal, from its stationary prior.
        variance = self.signal_variance
        filter_gain = np.zeros(N)
        for j in range(N):
            filter_gain[j] = variance / (variance * dt + self.sigma ** 2)
            variance = (1 - kappa * dt) ** 2 * (variance - filter_gain[j] * dt * variance) + self.sigma_alpha ** 2 * dt

        # Backward Riccati recursion for the state z = (q, alpha^i, alphahat^j): z_{j+1} = F z_j + G nu_j + noise,
        # with running cost (z' Q z + a nu^2) dt and terminal cost psi q^2.
        F = np.diag([1.0, 1 - kappa * dt, 1 - kappa * dt])
        G = np.array([dt, 0.0, 0.0])
        Q = np.array([[self.phi, -0.5, -0.5], [-0.5, 0, 0], [-0.5, 0, 0]])
        P = np.diag([self.psi, 0.0, 0.0])
        feedback = np.zeros((N, 3))
        for j in range(N - 1, -1, -1):
            inverse = 1.0 / (self.a * dt + G @ P @ G)
            feedback[j] = inverse * (G @ P @ F)
            P = Q * dt + F.T @ P @ F - np.outer(F.T @ P @ G, G @ P @ F) * inverse

        # Controls along the paths.
        alpha = np.stack([noise['alpha1'].cpu().numpy(), noise['alpha2'].cpu().numpy()], 1).astype(np.float64)
        dWS = noise['dWS'].cpu().numpy().astype(np.float64)
        M = alpha.shape[0]
        q = np.zeros((M, 2))
        estimate = np.zeros((M, 2))                # estimate[:, i] = alphahat^j of trader i
        nu = np.zeros((M, 2, N))
        for j in range(N):
            for i in (0, 1):
                nu[:, i, j] = -(feedback[j] @ np.stack([q[:, i], alpha[:, i, j], estimate[:, i]]))
            observed = alpha[:, ::-1, j] * dt + self.sigma * dWS[:, j, None]       # alpha^j dt + sigma dW^S
            estimate = (1 - kappa * dt) * (estimate + filter_gain[j] * (observed - estimate * dt))
            q = q + nu[:, :, j] * dt
        return nu


class LinearStrategies:
    """
    Exact equilibria of a TwoTraders game among strategies linear in the own observed path,

        nu^i_{t_j} = sum_{k <= j} ( beta^i[j, k, 0] alpha^i_{t_k} + beta^i[j, k, 1] S_{t_k} ).

    The shocks of the discretised game form a vector e in R^K, K = 2 + 3N: the two initial signals and,
    at each step, the increments of the two signal noises and of the price noise, at unit variance. Every
    process is linear in e and is represented by its loading vector, so E[x y] = x . y and the expected
    cost of each trader is computed exactly, without simulation. Each best response is computed by L-BFGS
    over the trader's coefficients, and the equilibrium by a damped best-response iteration.

    Open loop: in each best response the other trader's control is held fixed as a process, as in the
    open-loop equilibrium of the note. Closed loop: the other trader's coefficients are held fixed, so
    its control responds to trader i through the price.

    All computations are in float64 on the CPU.
    """

    def __init__(self, game: TwoTraders):
        """
        Args:
            game: The TwoTraders game.
        """
        self.game = game
        N, dt = game.N, game.dt
        self.K = 2 + 3 * N

        # Loadings of the signals: alpha^i_{t_j} = signal_loadings[i, j] . e.
        self.signal_loadings = torch.zeros(2, N + 1, self.K, **FLOAT64_CPU)
        for i in range(2):
            self.signal_loadings[i, 0, i] = game.signal_variance ** 0.5
            for j in range(N):
                self.signal_loadings[i, j + 1] = (1 - game.kappa * dt) * self.signal_loadings[i, j]
                self.signal_loadings[i, j + 1, 2 + 3 * j + i] += game.sigma_alpha * dt ** 0.5

        # Loadings of the price noise sigma dW^S over [t_j, t_{j+1}].
        self.price_noise = torch.zeros(N, self.K, **FLOAT64_CPU)
        for j in range(N):
            self.price_noise[j, 2 + 3 * j + 2] = game.sigma * dt ** 0.5

        self.causal = torch.tril(torch.ones(N, N, **FLOAT64_CPU))[:, :, None]   # beta[j, k] is used for k <= j only

    def forward(self, beta, own: int = None, other_control=None):
        """
        Loadings of the processes under the coefficients beta.

        Args:
            beta: Coefficients [2, N, N, 2].
            own: With other_control, the trader whose control follows beta; the other trader's control is
                held fixed at other_control (open loop). Default none.
            other_control: Loadings of the other trader's control [N, K].

        Returns:
            controls [2, N, K], inventories [2, N + 1, K] and price drift mu [N, K].
        """
        game = self.game
        N, dt = game.N, game.dt
        prices = [torch.zeros(self.K, **FLOAT64_CPU)]
        controls = []
        for j in range(N):
            price_path = torch.stack(prices)
            u = [beta[i, j, :j + 1, 0] @ self.signal_loadings[i, :j + 1] + beta[i, j, :j + 1, 1] @ price_path for i in range(2)]
            if own is not None:
                u[1 - own] = other_control[j]
            controls.append(torch.stack(u))
            drift = self.signal_loadings[0, j] + self.signal_loadings[1, j] + game.lam * (u[0] + u[1])
            prices.append(prices[-1] + drift * dt + self.price_noise[j])
        controls = torch.stack(controls, 1)
        q = torch.cat([torch.zeros(2, 1, self.K, **FLOAT64_CPU), torch.cumsum(controls, 1) * dt], 1)
        mu = self.signal_loadings[0, :N] + self.signal_loadings[1, :N] + game.lam * (controls[0] + controls[1])
        return controls, q, mu

    def cost(self, i: int, beta, other_control=None):
        """
        Expected cost of trader i (minus the reward J^i), computed from the loadings.

        Args:
            i: The trader.
            beta: Coefficients [2, N, N, 2].
            other_control: If given, the other trader's control is held fixed at these loadings (open loop).
        """
        game = self.game
        controls, q, mu = self.forward(beta * self.causal, None if other_control is None else i, other_control)
        running = -(q[i, :-1] * mu).sum(-1) + game.a * (controls[i] ** 2).sum(-1) + game.phi * (q[i, :-1] ** 2).sum(-1)
        return running.sum() * game.dt + game.psi * (q[i, -1] ** 2).sum()

    def best_response(self, beta, i: int, closed: bool, inner: int = 200):
        """
        Coefficients of trader i that minimize its expected cost, by L-BFGS from beta[i].

        Args:
            beta: Current coefficients [2, N, N, 2].
            i: The trader.
            closed: Closed loop (the other trader's coefficients fixed) or open loop (its control fixed).
            inner: Maximum number of L-BFGS iterations. Default 200.

        Returns:
            Coefficients [N, N, 2].
        """
        other_control = None if closed else self.forward(beta * self.causal)[0][1 - i].detach()
        coefficients = beta[i].detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([coefficients], max_iter=inner, tolerance_grad=1e-9, tolerance_change=1e-12,
                                      line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            both = torch.stack([coefficients, beta[1]]) if i == 0 else torch.stack([beta[0], coefficients])
            cost = self.cost(i, both, other_control)
            cost.backward()
            return cost
        optimizer.step(closure)
        return (coefficients * self.causal).detach()

    def solve(self, closed: bool, iterations: int = 150, damping: float = 0.5, start=None, tolerance: float = 1e-6):
        """
        Damped best-response iteration beta <- (1 - damping) beta + damping BR(beta).

        Args:
            closed: Closed-loop or open-loop equilibrium.
            iterations: Maximum number of iterations. Default 150.
            damping: Weight of the best response. Default 0.5.
            start: Initial coefficients [2, N, N, 2]. Default zero.
            tolerance: Stop when the relative change of the best responses falls below. Default 1e-6.

        Returns:
            The equilibrium coefficients [2, N, N, 2].

        Raises:
            RuntimeError: If the iteration does not converge within `iterations`.
        """
        N = self.game.N
        beta = torch.zeros(2, N, N, 2, **FLOAT64_CPU) if start is None else torch.as_tensor(start, **FLOAT64_CPU).clone()
        for _ in range(iterations):
            responses = torch.stack([self.best_response(beta, i, closed) for i in range(2)])
            change = float((responses - beta).norm() / max(1.0, float(responses.norm())))
            beta = (1 - damping) * beta + damping * responses
            if change < tolerance:
                return beta
        raise RuntimeError(f'best-response iteration not converged after {iterations} iterations (change {change:.1e})')

    def controls(self, beta, noise: dict) -> np.ndarray:
        """
        Controls of the linear strategies along simulated paths.

        Args:
            beta: Coefficients [2, N, N, 2].
            noise: Noise of the paths (TwoTraders.sample).

        Returns:
            Controls [M, 2, N].
        """
        game = self.game
        N, dt = game.N, game.dt
        to_numpy = lambda key: noise[key].cpu().numpy().astype(np.float64)
        M = noise['dWS'].shape[0]
        shocks = np.zeros((M, self.K))                   # the shock vector e of each path
        shocks[:, 0] = to_numpy('alpha1_0') / game.signal_variance ** 0.5
        shocks[:, 1] = to_numpy('alpha2_0') / game.signal_variance ** 0.5
        for j in range(N):
            for offset, key in enumerate(['dW1', 'dW2', 'dWS']):
                shocks[:, 2 + 3 * j + offset] = to_numpy(key)[:, j] / dt ** 0.5
        with torch.no_grad():
            loadings = self.forward(torch.as_tensor(beta, **FLOAT64_CPU) * self.causal)[0].numpy()
        return np.einsum('ijk,mk->mij', loadings, shocks)
