"""The broker-trader game of Wu and Jaimungal (arXiv 2412.17712), one asset, without transient impact (h = 0).

    dz_t = alpha_t dt + sigma dW_t                   unimpacted price
    dalpha_t = -kappa alpha_t dt + sigma_alpha dB_t  trading signal
    dQ^I_t = eta_t dt,   dQ^B_t = (nu_t - eta_t) dt  inventories of the informed trader and of the broker
The informed trader (player 0) observes (alpha, Q^I); the broker (player 1) observes the price path only. They minimize
    J^I = -E[ int_0^T ( Q^I_t alpha_t - b eta_t^2 - r_I (Q^I_t)^2 ) dt - psi (Q^I_T)^2 ],
    J^B = -E[ int_0^T ( Q^B_t alpha_t - a nu_t^2 - r_B (Q^B_t)^2 ) dt - phi (Q^B_T)^2 ].
exact computes the equilibrium of the discretised game: the trader's discrete LQR in (alpha, Q^I), and the broker's
discrete Kalman filter of (alpha, Q^I) from the price followed by a discrete LQR.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BrokerTrader:
    """Model parameters and the game interface of pig.solver."""

    a: float = 1.0            # broker's trading cost
    b: float = 1.0            # trader's trading cost
    phi: float = 1.0          # broker's terminal inventory penalty
    psi: float = 1.0          # trader's terminal inventory penalty
    r_B: float = 0.1          # broker's running inventory penalty
    r_I: float = 0.1          # trader's running inventory penalty
    kappa: float = 1.0        # signal mean reversion
    sigma_alpha: float = 1.0  # signal volatility
    sigma: float = 0.5        # price volatility
    T: float = 0.3
    N: int = 40

    n_players = 2
    observation_dims = [2, 0]
    increment_dims = [0, 1]
    full_dim = 3
    noise_dim = 2
    cv_decay = [1.0, 1.0]

    @property
    def dt(self) -> float:
        return self.T / self.N

    def sample(self, n_paths: int, seed: int) -> dict:
        """Signal paths alpha [M, N+1] and price increments dz [M, N]."""
        rng = np.random.default_rng(seed); N, dt = self.N, self.dt
        alpha = np.zeros((n_paths, N + 1)); alpha[:, 0] = math.sqrt(self.sigma_alpha ** 2 / (2 * self.kappa)) * rng.standard_normal(n_paths)
        dz = np.zeros((n_paths, N))
        for k in range(N):
            dz[:, k] = alpha[:, k] * dt + self.sigma * math.sqrt(dt) * rng.standard_normal(n_paths)
            alpha[:, k + 1] = alpha[:, k] - self.kappa * alpha[:, k] * dt + self.sigma_alpha * math.sqrt(dt) * rng.standard_normal(n_paths)
        return dict(alpha=torch.tensor(alpha, dtype=torch.float32), dz=torch.tensor(dz, dtype=torch.float32))

    def initial_state(self, noise: dict) -> dict:
        M = noise['dz'].shape[0]
        return dict(QI=torch.zeros(M, self.N + 1), QB=torch.zeros(M, self.N + 1))

    def step(self, state, noise, controls, k):
        eta, nu = controls[0], controls[1]
        state['QI'][:, k + 1] = state['QI'][:, k] + eta * self.dt
        state['QB'][:, k + 1] = state['QB'][:, k] + (nu - eta) * self.dt

    def observe(self, i, state, noise, k):
        if i == 0:
            return torch.stack([noise['alpha'][:, k], state['QI'][:, k]], -1)
        return torch.zeros(noise['dz'].shape[0], 0)

    def increment(self, i, state, noise, k):
        return noise['dz'][:, k:k + 1] / math.sqrt(self.dt)

    def cost(self, i, state, noise, controls):
        alpha = noise['alpha'][:, :-1]
        if i == 0:
            Q, cost, running_penalty, terminal_penalty = state['QI'], self.b, self.r_I, self.psi
        else:
            Q, cost, running_penalty, terminal_penalty = state['QB'], self.a, self.r_B, self.phi
        running = (Q[:, :-1] * alpha - cost * controls[i] ** 2 - running_penalty * Q[:, :-1] ** 2).sum(1) * self.dt
        return -(running - terminal_penalty * Q[:, -1] ** 2)

    def curvature(self, i):
        return torch.full((self.N,), 2 * (self.b if i == 0 else self.a) * self.dt)

    def full_state(self, state, noise):
        return torch.stack([noise['alpha'][:, :-1], state['QB'][:, :-1], state['QI'][:, :-1]], -1)

    def martingale_increments(self, noise):
        """Signal and price Brownian increments recovered from the paths, scaled by 1/sqrt(dt)."""
        alpha, dz = noise['alpha'], noise['dz']
        dB = (alpha[:, 1:] - alpha[:, :-1] + self.kappa * alpha[:, :-1] * self.dt) / self.sigma_alpha
        dW = (dz - alpha[:, :-1] * self.dt) / self.sigma
        return torch.stack([dB, dW], -1) / math.sqrt(self.dt)


def exact(game: BrokerTrader, noise: dict) -> np.ndarray:
    """Exact equilibrium controls of the discretised game on the given paths, [M, 2, N]: trader, broker."""
    N, dt = game.N, game.dt; c = 1 - game.kappa * dt
    alpha = noise['alpha'].cpu().numpy().astype(np.float64); dz = noise['dz'].cpu().numpy().astype(np.float64); M = alpha.shape[0]
    # trader: x = (alpha, Q^I), x_k+1 = A x_k + B eta_k + noise
    A = np.array([[c, 0.0], [0.0, 1.0]]); B = np.array([0.0, dt]); reward = np.array([[0.0, 0.5], [0.5, -game.r_I]])
    S = np.diag([0.0, -game.psi]); L = np.zeros((N, 2))
    for k in range(N - 1, -1, -1):
        BSB = B @ S @ B - game.b * dt; BSA = B @ S @ A
        L[k] = -BSA / BSB; S = reward * dt + A.T @ S @ A - np.outer(BSA, BSA) / BSB
    # broker: Kalman filter of (alpha, Q^I) from the price increments, then an LQR on (Qhat^B, alphahat, Qhat^I)
    H = np.array([1.0, 0.0]); signal_noise = np.diag([game.sigma_alpha ** 2 * dt, 0.0])
    covariance = np.diag([game.sigma_alpha ** 2 / (2 * game.kappa), 0.0]); gains, transitions = [], []
    for k in range(N):
        gain = covariance @ H / (H @ covariance @ H * dt + game.sigma ** 2); gains.append(gain)
        transition = A + np.outer(B, L[k]); transitions.append(transition)
        covariance = transition @ (covariance - np.outer(gain, H @ covariance) * dt) @ transition.T + signal_noise
    B_broker = np.array([dt, 0.0, 0.0]); reward_broker = np.zeros((3, 3)); reward_broker[0, 0] = -game.r_B; reward_broker[0, 1] = reward_broker[1, 0] = 0.5
    S = np.diag([-game.phi, 0.0, 0.0]); L_broker = np.zeros((N, 3))
    for k in range(N - 1, -1, -1):
        F = transitions[k]
        A_broker = np.array([[1.0, -dt * L[k, 0], -dt * L[k, 1]], [0.0, F[0, 0], F[0, 1]], [0.0, F[1, 0], F[1, 1]]])
        BSB = B_broker @ S @ B_broker - game.a * dt; BSA = B_broker @ S @ A_broker
        L_broker[k] = -BSA / BSB; S = reward_broker * dt + A_broker.T @ S @ A_broker - np.outer(BSA, BSA) / BSB
    QI = np.zeros(M); estimate = np.zeros((M, 2)); bought = np.zeros(M); controls = np.zeros((M, 2, N))
    for k in range(N):
        controls[:, 0, k] = L[k, 0] * alpha[:, k] + L[k, 1] * QI; QI = QI + controls[:, 0, k] * dt
        controls[:, 1, k] = np.c_[bought - estimate[:, 1], estimate] @ L_broker[k]
        innovation = dz[:, k] - estimate[:, 0] * dt
        estimate = (estimate + np.outer(innovation, gains[k])) @ transitions[k].T; bought = bought + controls[:, 1, k] * dt
    return controls
