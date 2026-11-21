"""
The broker-trader game, one asset, without transient impact.

Dynamics:
    unimpacted price:  dz_t = alpha_t dt + sigma dW_t
    trading signal:    dalpha_t = -kappa alpha_t dt + sigma_alpha dB_t
    inventories:       dQ^I_t = eta_t dt (informed trader),   dQ^B_t = (nu_t - eta_t) dt (broker)

Information. The informed trader (player 0) observes (alpha, Q^I). The broker (player 1) observes the
price path only; it must infer both the signal and the trader's inventory from the prices.

Rewards (the code minimizes their negatives):
    J^I = E[ int_0^T ( Q^I_t alpha_t - b eta_t^2 - r_I (Q^I_t)^2 ) dt - psi (Q^I_T)^2 ],
    J^B = E[ int_0^T ( Q^B_t alpha_t - a nu_t^2 - r_B (Q^B_t)^2 ) dt - phi (Q^B_T)^2 ].

Exact solution of the discretised game (BrokerTrader.exact): the trader's problem does not depend on the
broker, and is a discrete LQR in (alpha, Q^I). The broker filters (alpha, Q^I) from the price increments
with a discrete Kalman filter, then solves a discrete LQR in (Qhat^B, alphahat, Qhat^I).
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from .base import Game


@dataclass
class BrokerTrader(Game):
    """Model parameters and the game interface."""

    a: float = 1.0            # broker's cost of the trading rate
    b: float = 1.0            # trader's cost of the trading rate
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
    observation_dims = [2, 0]     # trader: (alpha, Q^I); broker: none beyond the price increments
    increment_dims = [0, 1]       # trader: no memory; broker: the price increments
    full_dim = 3
    noise_dim = 2
    cv_decay = [1.0, 1.0]

    def sample(self, n_paths, seed):
        """
        Simulates the signal and the price by the Euler--Maruyama scheme.

        Returns:
            alpha: signal paths [M, N + 1]
            dz: price increments [M, N]
        """
        rng = np.random.default_rng(seed)
        N, dt = self.N, self.dt
        alpha = np.zeros((n_paths, N + 1))
        alpha[:, 0] = math.sqrt(self.sigma_alpha ** 2 / (2 * self.kappa)) * rng.standard_normal(n_paths)
        dz = np.zeros((n_paths, N))
        for j in range(N):
            dz[:, j] = alpha[:, j] * dt + self.sigma * math.sqrt(dt) * rng.standard_normal(n_paths)
            alpha[:, j + 1] = alpha[:, j] - self.kappa * alpha[:, j] * dt + self.sigma_alpha * math.sqrt(dt) * rng.standard_normal(n_paths)
        return dict(alpha=torch.tensor(alpha, dtype=torch.float32), dz=torch.tensor(dz, dtype=torch.float32))

    def initial_state(self, noise):
        M = noise['dz'].shape[0]
        return dict(QI=torch.zeros(M, self.N + 1), QB=torch.zeros(M, self.N + 1))

    def step(self, state, noise, controls, j):
        eta, nu = controls[0], controls[1]
        state['QI'][:, j + 1] = state['QI'][:, j] + eta * self.dt
        state['QB'][:, j + 1] = state['QB'][:, j] + (nu - eta) * self.dt

    def observe(self, i, state, noise, j):
        """Trader: (alpha, Q^I) at t_j. Broker: empty (it acts on the price increments through its GRU)."""
        if i == 0:
            return torch.stack([noise['alpha'][:, j], state['QI'][:, j]], -1)
        return torch.zeros(noise['dz'].shape[0], 0)

    def increment(self, i, state, noise, j):
        """Price increment over [t_j, t_{j+1}], divided by sqrt(dt)."""
        return noise['dz'][:, j:j + 1] / math.sqrt(self.dt)

    def cost(self, i, state, noise, controls):
        """Minus the realised reward of the trader (i = 0) or the broker (i = 1)."""
        alpha = noise['alpha'][:, :-1]
        if i == 0:
            Q, trading_cost, running_penalty, terminal_penalty = state['QI'], self.b, self.r_I, self.psi
        else:
            Q, trading_cost, running_penalty, terminal_penalty = state['QB'], self.a, self.r_B, self.phi
        running = (Q[:, :-1] * alpha - trading_cost * controls[i] ** 2 - running_penalty * Q[:, :-1] ** 2).sum(1) * self.dt
        return -(running - terminal_penalty * Q[:, -1] ** 2)

    def curvature(self, i):
        """2 b dt for the trader, 2 a dt for the broker."""
        return torch.full((self.N,), 2 * (self.b if i == 0 else self.a) * self.dt)

    def full_state(self, state, noise):
        """X_{t_j} = (alpha, Q^B, Q^I) at t_j."""
        return torch.stack([noise['alpha'][:, :-1], state['QB'][:, :-1], state['QI'][:, :-1]], -1)

    def martingale_increments(self, noise):
        """(dB, dW) / sqrt(dt), the signal and price Brownian increments recovered from the paths."""
        alpha, dz = noise['alpha'], noise['dz']
        dB = (alpha[:, 1:] - alpha[:, :-1] + self.kappa * alpha[:, :-1] * self.dt) / self.sigma_alpha
        dW = (dz - alpha[:, :-1] * self.dt) / self.sigma
        return torch.stack([dB, dW], -1) / math.sqrt(self.dt)

    def exact(self, noise):
        """Exact equilibrium controls of the discretised game on the given paths, [M, 2, N]: trader, broker."""
        N, dt = self.N, self.dt
        alpha = noise['alpha'].cpu().numpy().astype(np.float64)
        dz = noise['dz'].cpu().numpy().astype(np.float64)

        # Trader: state x = (alpha, Q^I), x_{j+1} = A x_j + B eta_j + noise; feedback eta_j = L_j x_j.
        A = np.array([[1 - self.kappa * dt, 0.0], [0.0, 1.0]])
        B = np.array([0.0, dt])
        reward = np.array([[0.0, 0.5], [0.5, -self.r_I]])
        S = np.diag([0.0, -self.psi])
        L = np.zeros((N, 2))
        for j in range(N - 1, -1, -1):
            BSB = B @ S @ B - self.b * dt
            BSA = B @ S @ A
            L[j] = -BSA / BSB
            S = reward * dt + A.T @ S @ A - np.outer(BSA, BSA) / BSB

        # Broker, filter: Kalman filter of x = (alpha, Q^I) from the price increments dz = alpha dt + sigma dW,
        # with the transition of x under the trader's feedback.
        H = np.array([1.0, 0.0])
        signal_noise = np.diag([self.sigma_alpha ** 2 * dt, 0.0])
        covariance = np.diag([self.sigma_alpha ** 2 / (2 * self.kappa), 0.0])
        gains, transitions = [], []
        for j in range(N):
            gain = covariance @ H / (H @ covariance @ H * dt + self.sigma ** 2)
            transition = A + np.outer(B, L[j])
            covariance = transition @ (covariance - np.outer(gain, H @ covariance) * dt) @ transition.T + signal_noise
            gains.append(gain)
            transitions.append(transition)

        # Broker, control: LQR in xi = (Qhat^B, alphahat, Qhat^I); feedback nu_j = L^B_j xi_j.
        B_broker = np.array([dt, 0.0, 0.0])
        reward_broker = np.array([[-self.r_B, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        S = np.diag([-self.phi, 0.0, 0.0])
        L_broker = np.zeros((N, 3))
        for j in range(N - 1, -1, -1):
            F = transitions[j]
            A_broker = np.array([[1.0, -dt * L[j, 0], -dt * L[j, 1]], [0.0, F[0, 0], F[0, 1]], [0.0, F[1, 0], F[1, 1]]])
            BSB = B_broker @ S @ B_broker - self.a * dt
            BSA = B_broker @ S @ A_broker
            L_broker[j] = -BSA / BSB
            S = reward_broker * dt + A_broker.T @ S @ A_broker - np.outer(BSA, BSA) / BSB

        # Controls along the paths.
        M = alpha.shape[0]
        QI = np.zeros(M)
        estimate = np.zeros((M, 2))                     # (alphahat, Qhat^I) before step j
        bought = np.zeros(M)                            # the broker's cumulative trades, so Qhat^B = bought - Qhat^I
        controls = np.zeros((M, 2, N))
        for j in range(N):
            controls[:, 0, j] = L[j, 0] * alpha[:, j] + L[j, 1] * QI
            controls[:, 1, j] = np.c_[bought - estimate[:, 1], estimate] @ L_broker[j]
            QI = QI + controls[:, 0, j] * dt
            bought = bought + controls[:, 1, j] * dt
            innovation = dz[:, j] - estimate[:, 0] * dt
            estimate = (estimate + np.outer(innovation, gains[j])) @ transitions[j].T
        return controls
