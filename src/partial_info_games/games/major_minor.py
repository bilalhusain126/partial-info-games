"""
The major-minor LQG game, scalar, one minor player, with partial observations of both players.

Dynamics and observations:
    major:  dx^0_t = (A0 x^0_t + B0 u^0_t) dt + D0 dw^0_t,             dy^0_t = L0 x^0_t dt + s0 dv^0_t
    minor:  dx^1_t = (A1 x^1_t + G x^0_t + B1 u^1_t) dt + D1 dw^1_t,   dy^1_t = (L1 x^1_t, L2 x^0_t) dt + s1 dv^1_t

Information. The major player (0) observes y^0 and the minor player (1) observes y^1. Neither observes the
other's observation path, and the minor player must estimate the major player's own estimate of x^0.

Costs, with discount rate rho (each player's reward is minus its cost):
    J^0 = E int_0^T e^{-rho t} [ Q0 (x^0_t - eta0)^2 + R0 (u^0_t)^2 ] dt,
    J^1 = E int_0^T e^{-rho t} [ Q1 (x^1_t - H x^0_t - eta1)^2 + R1 (u^1_t)^2 ] dt.

Exact solution of the discretised game (MajorMinor.exact): the major player's discrete Kalman filter and
discounted LQR; the minor player's discrete Kalman filter of (x^1, x^0, xhat^0), with the major player's
filter in its dynamics, and discounted LQR.
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

    Minimizes sum_j beta^j [ (z_j - zeta)' Q (z_j - zeta) + R u_j^2 ] dt, with Q zeta = q and
    beta = exp(-rho dt), subject to z_{j+1} = F(j) z_j + B u_j + c(j) + noise.

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
    s0: float = 0.3
    A1: float = -0.5
    B1: float = 1.0
    G: float = 0.5
    L1: float = 1.0
    L2: float = 1.0
    Q1: float = 1.0
    R1: float = 1.0
    H: float = 0.6
    eta1: float = 0.25
    D1: float = 0.5
    s1: float = 0.3
    rho: float = 0.5                 # discount rate
    initial_variance: float = 1.0    # variance of x^0_0 and x^1_0
    T: float = 5.0
    N: int = 100

    n_players = 2
    observation_dims = [1, 2]        # y^0, y^1
    full_dim = 2
    noise_dim = 2

    @property
    def discount(self) -> float:
        """Discount factor per step, exp(-rho dt)."""
        return math.exp(-self.rho * self.dt)

    @property
    def cv_decay(self) -> list:
        """Decay per step of each player's reward sensitivity to the state, which weights the control variate."""
        return [self.discount * (1 + self.A0 * self.dt), self.discount * (1 + self.A1 * self.dt)]

    def sample(self, n_paths, seed):
        """
        Returns:
            x0_0, x1_0: initial states [M]
            dw0, dw1: state noise increments [M, N]
            dv0 [M, N], dv1 [M, N, 2]: observation noise increments
        """
        rng = np.random.default_rng(seed)
        N, dt = self.N, self.dt
        s = math.sqrt(self.initial_variance)
        exogenous = dict(x0_0=rng.standard_normal(n_paths) * s, x1_0=rng.standard_normal(n_paths) * s,
                     dw0=rng.standard_normal((n_paths, N)) * math.sqrt(dt),
                     dw1=rng.standard_normal((n_paths, N)) * math.sqrt(dt),
                     dv0=rng.standard_normal((n_paths, N)) * math.sqrt(dt),
                     dv1=rng.standard_normal((n_paths, N, 2)) * math.sqrt(dt))
        return {key: torch.tensor(value, dtype=torch.float32) for key, value in exogenous.items()}

    def initial_state(self, exogenous):
        M = exogenous['dw0'].shape[0]
        state = dict(x0=torch.zeros(M, self.N + 1), x1=torch.zeros(M, self.N + 1),
                     y0=torch.zeros(M, self.N + 1, 1), y1=torch.zeros(M, self.N + 1, 2))
        state['x0'][:, 0] = exogenous['x0_0']
        state['x1'][:, 0] = exogenous['x1_0']
        return state

    def step(self, state, exogenous, controls, j):
        x0, x1, dt = state['x0'][:, j], state['x1'][:, j], self.dt
        state['x0'][:, j + 1] = x0 + (self.A0 * x0 + self.B0 * controls[0]) * dt + self.D0 * exogenous['dw0'][:, j]
        state['x1'][:, j + 1] = x1 + (self.A1 * x1 + self.G * x0 + self.B1 * controls[1]) * dt + self.D1 * exogenous['dw1'][:, j]
        state['y0'][:, j + 1] = state['y0'][:, j] + (self.L0 * x0 * dt + self.s0 * exogenous['dv0'][:, j])[:, None]
        state['y1'][:, j + 1] = state['y1'][:, j] + torch.stack([self.L1 * x1, self.L2 * x0], -1) * dt + self.s1 * exogenous['dv1'][:, j]

    def observe(self, i, state, exogenous, j):
        """y^i at t_j."""
        return state[f'y{i}'][:, j]

    def discounting(self):
        """Discount factors exp(-rho t_j), [N]."""
        return torch.tensor(np.exp(-self.rho * np.arange(self.N) * self.dt), dtype=torch.float32)

    def reward(self, i, state, exogenous, controls):
        """Realised reward of player i: minus its discounted cost."""
        x0, x1 = state['x0'][:, :-1], state['x1'][:, :-1]
        if i == 0:
            running = self.Q0 * (x0 - self.eta0) ** 2 + self.R0 * controls[0] ** 2
        else:
            running = self.Q1 * (x1 - self.H * x0 - self.eta1) ** 2 + self.R1 * controls[1] ** 2
        return -(running * self.discounting()).sum(1) * self.dt

    def curvature(self, i):
        """2 R_i exp(-rho t_j) dt."""
        return 2 * (self.R0 if i == 0 else self.R1) * self.discounting() * self.dt

    def full_state(self, state, exogenous):
        """X_{t_j} = (x^0, x^1) at t_j."""
        return torch.stack([state['x0'][:, :-1], state['x1'][:, :-1]], -1)

    def martingale_increments(self, exogenous):
        """(dw^0, dw^1) / sqrt(dt)."""
        return torch.stack([exogenous['dw0'], exogenous['dw1']], -1) / math.sqrt(self.dt)

    def exact(self, exogenous):
        """Exact equilibrium controls of the discretised game on the given paths, [M, 2, N]: major, minor."""
        N, dt = self.N, self.dt
        a0 = 1 + self.A0 * dt

        # Major player: Kalman filter of x^0 from y^0, and LQR on its estimate.
        gain0, offset0 = discounted_lqr(self, lambda j: np.array([[a0]]), np.array([self.B0 * dt]), lambda j: np.zeros(1),
                                        np.array([[self.Q0]]), np.array([self.Q0 * self.eta0]), self.R0)
        filter0 = kalman_gains(self, lambda j: np.array([[a0]]), lambda j: np.array([[self.D0 ** 2 * dt]]),
                               np.array([[self.L0]]), np.array([[self.s0 ** 2]]), np.array([[self.initial_variance]]))[:, 0, 0]

        # Minor player: state z = (x^1, x^0, xhat^0), where xhat^0 is the major player's estimate, which
        # follows the major player's filter recursion under its feedback control.
        def F1(j):
            return np.array([[1 + self.A1 * dt, self.G * dt, 0.0],
                             [0.0, a0, -self.B0 * dt * gain0[j, 0]],
                             [0.0, a0 * filter0[j] * self.L0 * dt, a0 * (1 - filter0[j] * self.L0 * dt) - self.B0 * dt * gain0[j, 0]]])

        def c1(j):
            return np.array([0.0, -self.B0 * dt * offset0[j], -self.B0 * dt * offset0[j]])

        def noise1(j):
            return np.diag([self.D1 ** 2 * dt, self.D0 ** 2 * dt, (a0 * filter0[j] * self.s0) ** 2 * dt])

        B1 = np.array([self.B1 * dt, 0.0, 0.0])
        C1 = np.array([[self.L1, 0, 0], [0, self.L2, 0]])
        target = np.array([1.0, -self.H, 0.0])          # x^1 - H x^0
        gain1, offset1 = discounted_lqr(self, F1, B1, c1, np.outer(target, target) * self.Q1, target * self.Q1 * self.eta1, self.R1)
        filter1 = kalman_gains(self, F1, noise1, C1, np.diag([self.s1 ** 2, self.s1 ** 2]),
                               np.diag([self.initial_variance, self.initial_variance, 0.0]))

        # Controls along the paths.
        to_numpy = lambda key: exogenous[key].cpu().numpy().astype(np.float64)
        x0, x1 = to_numpy('x0_0'), to_numpy('x1_0')
        dw0, dw1, dv0, dv1 = to_numpy('dw0'), to_numpy('dw1'), to_numpy('dv0'), to_numpy('dv1')
        M = x0.shape[0]
        estimate0 = np.zeros(M)                          # the major player's estimate of x^0
        estimate1 = np.zeros((M, 3))                     # the minor player's estimate of z
        controls = np.zeros((M, 2, N))
        for j in range(N):
            u0 = -(gain0[j, 0] * estimate0 + offset0[j])
            u1 = -(estimate1 @ gain1[j] + offset1[j])
            controls[:, 0, j], controls[:, 1, j] = u0, u1
            dy0 = self.L0 * x0 * dt + self.s0 * dv0[:, j]
            dy1 = np.stack([self.L1 * x1, self.L2 * x0], -1) * dt + self.s1 * dv1[:, j]
            x0, x1 = (x0 + (self.A0 * x0 + self.B0 * u0) * dt + self.D0 * dw0[:, j],
                      x1 + (self.A1 * x1 + self.G * x0 + self.B1 * u1) * dt + self.D1 * dw1[:, j])
            estimate0 = a0 * (estimate0 + filter0[j] * (dy0 - self.L0 * estimate0 * dt)) + self.B0 * dt * u0
            posterior = estimate1 + (dy1 - estimate1 @ C1.T * dt) @ filter1[j].T
            estimate1 = posterior @ F1(j).T + c1(j) + B1 * u1[:, None]
        return controls
