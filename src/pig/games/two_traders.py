"""Two traders with private signals and a common price.

    dalpha^i_t = -kappa alpha^i_t dt + sigma_alpha dW^i_t       signal of trader i, observed by trader i only
    dS_t = mu_t dt + sigma dW^S_t,   mu_t = alpha^1_t + alpha^2_t + lam (nu^1_t + nu^2_t)
    dq^i_t = nu^i_t dt                                           inventory of trader i
Trader i observes (alpha^i, q^i, S) and minimizes
    J^i = -E[ int_0^T ( q^i_t mu_t - a (nu^i_t)^2 - phi (q^i_t)^2 ) dt - psi (q^i_T)^2 ].
The game is solved on the Euler--Maruyama grid. At lam = 0 the traders decouple and the exact solution is the
discrete LQG solution of each trader, with the Kalman filter of the other trader's signal (exact_lam0).
"""

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TwoTraders:
    """Model parameters and the game interface of pig.solver."""

    lam: float = 1.0          # permanent price impact
    a: float = 0.5            # temporary impact (cost of trading speed)
    phi: float = 0.1          # running inventory penalty
    psi: float = 1.0          # terminal inventory penalty
    kappa: float = 1.0        # signal mean reversion
    sigma_alpha: float = 1.0  # signal volatility
    sigma: float = 0.5        # price volatility
    T: float = 1.0
    N: int = 50

    n_players = 2
    observation_dims = [3, 3]
    increment_dims = [2, 2]
    full_dim = 5
    noise_dim = 3
    cv_decay = [1.0, 1.0]

    @property
    def dt(self) -> float:
        return self.T / self.N

    @property
    def signal_variance(self) -> float:
        """Stationary variance of each signal, the law of alpha^i_0."""
        return self.sigma_alpha ** 2 / (2 * self.kappa)

    def sample(self, n_paths: int, seed: int) -> dict:
        """Initial signals and Brownian increments; the signal paths alpha1, alpha2 [M, N+1]."""
        rng = np.random.default_rng(seed); dt, N = self.dt, self.N
        noise = dict(alpha1_0=rng.standard_normal(n_paths) * math.sqrt(self.signal_variance),
                     alpha2_0=rng.standard_normal(n_paths) * math.sqrt(self.signal_variance),
                     dW1=rng.standard_normal((n_paths, N)) * math.sqrt(dt), dW2=rng.standard_normal((n_paths, N)) * math.sqrt(dt),
                     dWS=rng.standard_normal((n_paths, N)) * math.sqrt(dt))
        for i in (1, 2):
            alpha = np.zeros((n_paths, N + 1)); alpha[:, 0] = noise[f'alpha{i}_0']
            for k in range(N):
                alpha[:, k + 1] = alpha[:, k] - self.kappa * alpha[:, k] * dt + self.sigma_alpha * noise[f'dW{i}'][:, k]
            noise[f'alpha{i}'] = alpha
        return {key: torch.tensor(value, dtype=torch.float32) for key, value in noise.items()}

    def initial_state(self, noise: dict) -> dict:
        M = noise['dWS'].shape[0]
        return dict(q1=torch.zeros(M, self.N + 1), q2=torch.zeros(M, self.N + 1), S=torch.zeros(M, self.N + 1), mu=torch.zeros(M, self.N))

    def step(self, state, noise, controls, k):
        mu = noise['alpha1'][:, k] + noise['alpha2'][:, k] + self.lam * (controls[0] + controls[1])
        state['mu'][:, k] = mu
        state['q1'][:, k + 1] = state['q1'][:, k] + controls[0] * self.dt
        state['q2'][:, k + 1] = state['q2'][:, k] + controls[1] * self.dt
        state['S'][:, k + 1] = state['S'][:, k] + mu * self.dt + self.sigma * noise['dWS'][:, k]

    def observe(self, i, state, noise, k):
        return torch.stack([noise[f'alpha{i + 1}'][:, k], state[f'q{i + 1}'][:, k], state['S'][:, k]], -1)

    def increment(self, i, state, noise, k):
        alpha = noise[f'alpha{i + 1}']; S = state['S']
        return torch.stack([alpha[:, k + 1] - alpha[:, k], S[:, k + 1] - S[:, k]], -1) / math.sqrt(self.dt)

    def cost(self, i, state, noise, controls):
        q = state[f'q{i + 1}']
        running = (q[:, :-1] * state['mu'] - self.a * controls[i] ** 2 - self.phi * q[:, :-1] ** 2).sum(1) * self.dt
        return -(running - self.psi * q[:, -1] ** 2)

    def curvature(self, i):
        return torch.full((self.N,), 2 * self.a * self.dt)

    def full_state(self, state, noise):
        return torch.stack([noise['alpha1'][:, :-1], noise['alpha2'][:, :-1], state['q1'][:, :-1], state['q2'][:, :-1], state['S'][:, :-1]], -1)

    def martingale_increments(self, noise):
        return torch.stack([noise['dW1'], noise['dW2'], noise['dWS']], -1) / math.sqrt(self.dt)


def exact_lam0(game: TwoTraders, noise: dict) -> np.ndarray:
    """Exact equilibrium controls at lam = 0 on the given paths, [M, 2, N]. Each trader solves a discrete LQG
    problem in (q^i, alpha^i, alphahat^j), with alphahat^j the Kalman filter of the other signal from the price."""
    assert game.lam == 0.0
    dt, N, kappa = game.dt, game.N, game.kappa
    variance = game.signal_variance; gain_filter = np.zeros(N)
    for k in range(N):
        gain_filter[k] = variance / (variance * dt + game.sigma ** 2)
        variance = (1 - kappa * dt) ** 2 * (variance - gain_filter[k] * dt * variance) + game.sigma_alpha ** 2 * dt
    F = np.diag([1.0, 1 - kappa * dt, 1 - kappa * dt]); G = np.array([dt, 0.0, 0.0])
    Q = np.array([[game.phi, -0.5, -0.5], [-0.5, 0, 0], [-0.5, 0, 0]])
    P = np.diag([game.psi, 0.0, 0.0]); gain = np.zeros((N, 3))
    for k in range(N - 1, -1, -1):
        inverse = 1.0 / (game.a * dt + G @ P @ G); gain[k] = inverse * (G @ P @ F)
        P = Q * dt + F.T @ P @ F - np.outer(F.T @ P @ G, G @ P @ F) * inverse
    alpha = np.stack([noise['alpha1'].cpu().numpy(), noise['alpha2'].cpu().numpy()], 1).astype(np.float64)
    dWS = noise['dWS'].cpu().numpy().astype(np.float64)
    M = alpha.shape[0]; q = np.zeros((M, 2)); estimate = np.zeros((M, 2)); nu = np.zeros((M, 2, N))
    for k in range(N):
        for i in (0, 1):
            nu[:, i, k] = -(gain[k] @ np.stack([q[:, i], alpha[:, i, k], estimate[:, i]]))
        observed = alpha[:, ::-1, k] * dt + game.sigma * dWS[:, k, None]           # trader i's observation of alpha^j
        estimate = (1 - kappa * dt) * (estimate + gain_filter[k] * (observed - estimate * dt))
        q = q + nu[:, :, k] * dt
    return nu
