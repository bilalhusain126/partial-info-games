"""
The major-minor LQG game of Firoozi and Caines, with one minor player and partial observations of both players.

States and observations (the player index is a superscript: 0 the major player, 1 the minor player):
    major:  dX⁰_t = (A₀ X⁰_t + B₀ u⁰_t) dt + D₀ dW⁰_t,              dO⁰_t = L₀ X⁰_t dt + σ₀ dV⁰_t
    minor:  dX¹_t = (A X¹_t + G X⁰_t + B u¹_t) dt + D dW¹_t,        dO¹_t = (L₁ X¹_t, L₂ X⁰_t) dt + σ₁ dV¹_t
W⁰, W¹, V⁰ and V¹ are independent Brownian motions, and V¹ is two-dimensional.

Information. Neither player observes its state directly. The major player observes O⁰, a noisy measurement of
its own state. The minor player observes O¹, noisy measurements of its own state and of the major player's state.
Neither player observes the other's observations, so the information is not nested.

Rewards, with discount rate δ:
    J⁰ = -𝔼 ∫₀ᵀ e^{-δ t} [ Q₀ (X⁰_t − η₀)^2 + R₀ (u⁰_t)^2 ] dt,
    J¹ = -𝔼 ∫₀ᵀ e^{-δ t} [ Q (X¹_t − H₁ X⁰_t − η)^2 + R (u¹_t)^2 ] dt.

Exact solution of the discretised game (MajorMinor.exact). The major player's control is linear in its Kalman
filter X̂⁰ of its own state, so the minor player's dynamics depend on X̂⁰. The minor player therefore filters
(X¹, X⁰, X̂⁰) and controls by a discounted LQR on that filter.

Differences from the paper: one minor player instead of many coupled through their average (the coupling terms
H₀ and H₂ are zero); a finite horizon T with discounting instead of an infinite horizon; scalar states; and
the coefficients L₀, L₁, L₂ in place of the paper's observation matrices.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from .base import Game


# ==============================================================================
# Discrete LQR and Kalman filter, for the exact solution
# ==============================================================================

def discounted_lqr(game, F, B, c, Q, q, R):
    """
    Discounted discrete LQR on the grid of game.

    Minimizes Σ_j β^j [ (z_j - ζ)' Q (z_j - ζ) + R u_j^2 ] dt, with Q ζ = q and
    β = exp(-δ dt), subject to z_{j+1} = F(j) z_j + B u_j + c(j) + noise.

    Returns:
        gain [N, n], offset [N] of the feedback u_j = -(gain_j z_j + offset_j).
    """
    N, dt, beta = game.N, game.dt, game.discount
    n = len(B)
    P = np.zeros((n, n))                 # quadratic coefficient of the value function
    s = np.zeros(n)                      # linear coefficient of the value function
    gain = np.zeros((N, n))
    offset = np.zeros(N)
    for j in range(N - 1, -1, -1):
        Fj, cj = F(j), c(j)
        inverse = 1.0 / (R * dt + beta * B @ P @ B)
        gain[j] = inverse * beta * (B @ P @ Fj)
        offset[j] = inverse * beta * (B @ (P @ cj + s))
        P_next = Q * dt + beta * Fj.T @ P @ Fj - beta ** 2 * np.outer(Fj.T @ P @ B, B @ P @ Fj) * inverse
        s = -q * dt + beta * Fj.T @ (P @ cj + s) - beta ** 2 * (Fj.T @ P @ B) * inverse * (B @ (P @ cj + s))
        P = P_next
    return gain, offset


def kalman_gains(game, F, noise_covariance, C, observation_covariance, initial):
    """
    Gains of the discrete Kalman filter of z_j from the increments dy_j = C z_j dt + noise.

    Args:
        F: Transition F(j), with z_{j+1} = F(j) z_j + noise of covariance noise_covariance(j).
        C, observation_covariance: Observation matrix and covariance per unit time.
        initial: Prior covariance of z_0.

    Returns:
        gains [N, n, m]: posterior estimate = prior estimate + gain_j (dy_j - C prior dt).
    """
    N, dt = game.N, game.dt
    V = initial.copy()
    gains = np.zeros((N, initial.shape[0], C.shape[0]))
    for j in range(N):
        gains[j] = V @ C.T @ np.linalg.inv(C @ V @ C.T * dt + observation_covariance)
        V = F(j) @ (V - gains[j] @ C * dt @ V) @ F(j).T + noise_covariance(j)
    return gains


# ==============================================================================
# The game
# ==============================================================================

@dataclass
class MajorMinor(Game):
    """Model parameters and the game interface."""

    A0: float = -1.0
    B0: float = 1.0
    L0: float = 1.0
    Q0: float = 1.0
    R0: float = 1.0
    eta0: float = 0.25
    D0: float = 0.5
    sigma0: float = 0.3
    A: float = -0.5
    B: float = 1.0
    G: float = 0.5
    L1: float = 1.0
    L2: float = 1.0
    Q: float = 1.0
    R: float = 1.0
    H1: float = 0.6
    eta: float = 0.25
    D: float = 0.5
    sigma1: float = 0.3
    delta: float = 0.5                 # discount rate
    initial_variance: float = 1.0    # variance of X⁰_0 and X¹_0
    T: float = 1.0
    N: int = 50

    n_players = 2
    observation_dims = [1, 2]        # O⁰, O¹
    full_dim = 2
    noise_dim = 2

    @property
    def discount(self) -> float:
        """Discount factor per step, exp(-δ dt)."""
        return math.exp(-self.delta * self.dt)

    @property
    def cv_decay(self) -> list:
        """Decay per step of each player's reward sensitivity to the state, which weights the control variate."""
        return [self.discount * (1 + self.A0 * self.dt), self.discount * (1 + self.A * self.dt)]

    def sample(self, n_paths, seed):
        """
        Returns:
            X0_0, X1_0: initial states [M]
            dW0, dW1: state noise increments [M, N]
            dV0 [M, N], dV1 [M, N, 2]: observation noise increments
        """
        rng = np.random.default_rng(seed)
        N, dt = self.N, self.dt
        s = math.sqrt(self.initial_variance)
        exogenous = dict(X0_0=rng.standard_normal(n_paths) * s, X1_0=rng.standard_normal(n_paths) * s,
                     dW0=rng.standard_normal((n_paths, N)) * math.sqrt(dt),
                     dW1=rng.standard_normal((n_paths, N)) * math.sqrt(dt),
                     dV0=rng.standard_normal((n_paths, N)) * math.sqrt(dt),
                     dV1=rng.standard_normal((n_paths, N, 2)) * math.sqrt(dt))
        return {key: torch.tensor(value, dtype=torch.float32) for key, value in exogenous.items()}

    def initial_state(self, exogenous):
        M = exogenous['dW0'].shape[0]
        state = dict(X0=torch.zeros(M, self.N + 1), X1=torch.zeros(M, self.N + 1),
                     O0=torch.zeros(M, self.N + 1, 1), O1=torch.zeros(M, self.N + 1, 2))
        state['X0'][:, 0] = exogenous['X0_0']
        state['X1'][:, 0] = exogenous['X1_0']
        return state

    def step(self, state, exogenous, controls, j):
        X0, X1, dt = state['X0'][:, j], state['X1'][:, j], self.dt
        state['X0'][:, j + 1] = X0 + (self.A0 * X0 + self.B0 * controls[0]) * dt + self.D0 * exogenous['dW0'][:, j]
        state['X1'][:, j + 1] = X1 + (self.A * X1 + self.G * X0 + self.B * controls[1]) * dt + self.D * exogenous['dW1'][:, j]
        state['O0'][:, j + 1] = state['O0'][:, j] + (self.L0 * X0 * dt + self.sigma0 * exogenous['dV0'][:, j])[:, None]
        state['O1'][:, j + 1] = state['O1'][:, j] + torch.stack([self.L1 * X1, self.L2 * X0], -1) * dt + self.sigma1 * exogenous['dV1'][:, j]

    def observe(self, i, state, exogenous, j):
        """Oⁱ at t_j."""
        return state[f'O{i}'][:, j]

    def discounting(self):
        """Discount factors exp(-δ t_j), [N]."""
        return torch.tensor(np.exp(-self.delta * np.arange(self.N) * self.dt), dtype=torch.float32)

    def reward(self, i, state, exogenous, controls):
        """Realised reward of player i: minus its discounted cost."""
        X0, X1 = state['X0'][:, :-1], state['X1'][:, :-1]
        if i == 0:
            running = self.Q0 * (X0 - self.eta0) ** 2 + self.R0 * controls[0] ** 2
        else:
            running = self.Q * (X1 - self.H1 * X0 - self.eta) ** 2 + self.R * controls[1] ** 2
        return -(running * self.discounting()).sum(1) * self.dt

    def curvature(self, i):
        """2 R_i exp(-δ t_j) dt."""
        return 2 * (self.R0 if i == 0 else self.R) * self.discounting() * self.dt

    def full_state(self, state, exogenous):
        """X_{t_j} = (X⁰, X¹) at t_j."""
        return torch.stack([state['X0'][:, :-1], state['X1'][:, :-1]], -1)

    def martingale_increments(self, exogenous):
        """(dW⁰, dW¹) / √dt."""
        return torch.stack([exogenous['dW0'], exogenous['dW1']], -1) / math.sqrt(self.dt)

    def exact(self, exogenous):
        """Exact equilibrium controls of the discretised game on the given paths, [M, 2, N]: major, minor."""
        N, dt = self.N, self.dt
        a0 = 1 + self.A0 * dt

        # Major player: Kalman filter of X⁰ from O⁰, and LQR on its estimate.
        gain0, offset0 = discounted_lqr(self, lambda j: np.array([[a0]]), np.array([self.B0 * dt]), lambda j: np.zeros(1),
                                        np.array([[self.Q0]]), np.array([self.Q0 * self.eta0]), self.R0)
        filter0 = kalman_gains(self, lambda j: np.array([[a0]]), lambda j: np.array([[self.D0 ** 2 * dt]]),
                               np.array([[self.L0]]), np.array([[self.sigma0 ** 2]]), np.array([[self.initial_variance]]))[:, 0, 0]

        # Minor player: state z = (X¹, X⁰, X̂⁰), where X̂⁰ is the major player's estimate, which
        # follows the major player's filter recursion under its feedback control.
        def F1(j):
            return np.array([[1 + self.A * dt, self.G * dt, 0.0],
                             [0.0, a0, -self.B0 * dt * gain0[j, 0]],
                             [0.0, a0 * filter0[j] * self.L0 * dt, a0 * (1 - filter0[j] * self.L0 * dt) - self.B0 * dt * gain0[j, 0]]])

        def c1(j):
            return np.array([0.0, -self.B0 * dt * offset0[j], -self.B0 * dt * offset0[j]])

        def noise1(j):
            return np.diag([self.D ** 2 * dt, self.D0 ** 2 * dt, (a0 * filter0[j] * self.sigma0) ** 2 * dt])

        B1 = np.array([self.B * dt, 0.0, 0.0])
        C1 = np.array([[self.L1, 0, 0], [0, self.L2, 0]])
        target = np.array([1.0, -self.H1, 0.0])          # X¹ - H X⁰
        gain1, offset1 = discounted_lqr(self, F1, B1, c1, np.outer(target, target) * self.Q, target * self.Q * self.eta, self.R)
        filter1 = kalman_gains(self, F1, noise1, C1, np.diag([self.sigma1 ** 2, self.sigma1 ** 2]),
                               np.diag([self.initial_variance, self.initial_variance, 0.0]))

        # Controls along the paths.
        to_numpy = lambda key: exogenous[key].cpu().numpy().astype(np.float64)
        X0, X1 = to_numpy('X0_0'), to_numpy('X1_0')
        dW0, dW1, dV0, dV1 = to_numpy('dW0'), to_numpy('dW1'), to_numpy('dV0'), to_numpy('dV1')
        M = X0.shape[0]
        estimate0 = np.zeros(M)                          # the major player's estimate of X⁰
        estimate1 = np.zeros((M, 3))                     # the minor player's estimate of z
        controls = np.zeros((M, 2, N))
        for j in range(N):
            u0 = -(gain0[j, 0] * estimate0 + offset0[j])
            u1 = -(estimate1 @ gain1[j] + offset1[j])
            controls[:, 0, j], controls[:, 1, j] = u0, u1
            dO0 = self.L0 * X0 * dt + self.sigma0 * dV0[:, j]
            dO1 = np.stack([self.L1 * X1, self.L2 * X0], -1) * dt + self.sigma1 * dV1[:, j]
            X0, X1 = (X0 + (self.A0 * X0 + self.B0 * u0) * dt + self.D0 * dW0[:, j],
                      X1 + (self.A * X1 + self.G * X0 + self.B * u1) * dt + self.D * dW1[:, j])
            estimate0 = a0 * (estimate0 + filter0[j] * (dO0 - self.L0 * estimate0 * dt)) + self.B0 * dt * u0
            posterior = estimate1 + (dO1 - estimate1 @ C1.T * dt) @ filter1[j].T
            estimate1 = posterior @ F1(j).T + c1(j) + B1 * u1[:, None]
        return controls
