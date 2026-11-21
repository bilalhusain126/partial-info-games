"""The major-minor LQG game of Firoozi and Caines (arXiv 1810.04369), scalar, one minor player, with partial
observations of both players.

    dx^0_t = (A0 x^0_t + B0 u^0_t) dt + D0 dw^0_t,             dy^0_t = L0 x^0_t dt + s0 dv^0_t
    dx^1_t = (A1 x^1_t + G x^0_t + B1 u^1_t) dt + D1 dw^1_t,   dy^1_t = (L1 x^1_t, L2 x^0_t) dt + s1 dv^1_t
The major player (0) observes y^0 and the minor player (1) observes y^1; neither observes the other's observation path.
With discount rate rho they minimize
    J^0 = E int_0^T e^{-rho t} [ Q0 (x^0_t - eta0)^2 + R0 (u^0_t)^2 ] dt,
    J^1 = E int_0^T e^{-rho t} [ Q1 (x^1_t - H x^0_t - eta1)^2 + R1 (u^1_t)^2 ] dt.
exact computes the equilibrium of the discretised game: the major player's Kalman filter and discounted LQR; the
minor player's Kalman filter of (x^1, x^0, xhat^0), with the major player's filter in its dynamics, and discounted LQR.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MajorMinor:
    """Model parameters and the game interface of pig.solver."""

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
    rho: float = 0.5           # discount rate
    initial_variance: float = 1.0
    T: float = 5.0
    N: int = 100

    n_players = 2
    observation_dims = [0, 0]
    increment_dims = [1, 2]
    full_dim = 2
    noise_dim = 2

    @property
    def dt(self) -> float:
        return self.T / self.N

    @property
    def discount(self) -> float:
        return math.exp(-self.rho * self.dt)

    @property
    def cv_decay(self) -> list:
        """Decay of each player's cost sensitivity per step, which weights the control variate."""
        return [self.discount * (1 + self.A0 * self.dt), self.discount * (1 + self.A1 * self.dt)]

    def sample(self, n_paths: int, seed: int) -> dict:
        rng = np.random.default_rng(seed); N, dt = self.N, self.dt; s = math.sqrt(self.initial_variance)
        noise = dict(x0_0=rng.standard_normal(n_paths) * s, x1_0=rng.standard_normal(n_paths) * s,
                     dw0=rng.standard_normal((n_paths, N)) * math.sqrt(dt), dw1=rng.standard_normal((n_paths, N)) * math.sqrt(dt),
                     dv0=rng.standard_normal((n_paths, N)) * math.sqrt(dt), dv1=rng.standard_normal((n_paths, N, 2)) * math.sqrt(dt))
        return {key: torch.tensor(value, dtype=torch.float32) for key, value in noise.items()}

    def initial_state(self, noise: dict) -> dict:
        M = noise['dw0'].shape[0]
        state = dict(x0=torch.zeros(M, self.N + 1), x1=torch.zeros(M, self.N + 1))
        state['x0'][:, 0] = noise['x0_0']; state['x1'][:, 0] = noise['x1_0']
        return state

    def step(self, state, noise, controls, k):
        x0, x1, dt = state['x0'][:, k], state['x1'][:, k], self.dt
        state['x0'][:, k + 1] = x0 + (self.A0 * x0 + self.B0 * controls[0]) * dt + self.D0 * noise['dw0'][:, k]
        state['x1'][:, k + 1] = x1 + (self.A1 * x1 + self.G * x0 + self.B1 * controls[1]) * dt + self.D1 * noise['dw1'][:, k]

    def observe(self, i, state, noise, k):
        return torch.zeros(noise['dw0'].shape[0], 0)

    def increment(self, i, state, noise, k):
        x0, x1, dt = state['x0'][:, k], state['x1'][:, k], self.dt
        if i == 0:
            return ((self.L0 * x0 * dt + self.s0 * noise['dv0'][:, k]) / math.sqrt(dt))[:, None]
        return (torch.stack([self.L1 * x1, self.L2 * x0], -1) * dt + self.s1 * noise['dv1'][:, k]) / math.sqrt(dt)

    def _discounting(self):
        return torch.exp(-self.rho * torch.arange(self.N, dtype=torch.float32) * self.dt)

    def cost(self, i, state, noise, controls):
        x0, x1 = state['x0'][:, :-1], state['x1'][:, :-1]
        if i == 0:
            running = self.Q0 * (x0 - self.eta0) ** 2 + self.R0 * controls[0] ** 2
        else:
            running = self.Q1 * (x1 - self.H * x0 - self.eta1) ** 2 + self.R1 * controls[1] ** 2
        return (running * self._discounting()).sum(1) * self.dt

    def curvature(self, i):
        return 2 * (self.R0 if i == 0 else self.R1) * self._discounting() * self.dt

    def full_state(self, state, noise):
        return torch.stack([state['x0'][:, :-1], state['x1'][:, :-1]], -1)

    def martingale_increments(self, noise):
        return torch.stack([noise['dw0'], noise['dw1']], -1) / math.sqrt(self.dt)


def _lqr(game, F, B, c, Q, q, R):
    """Discounted discrete LQR: minimize sum_k beta^k [(z - zeta)' Q (z - zeta) + R u^2] dt, Q zeta = q, with
    z_k+1 = F(k) z_k + B u_k + c(k) + noise. Returns the feedback u_k = -(gain_k z_k + offset_k)."""
    N, dt, beta = game.N, game.dt, game.discount; n = len(B)
    P = np.zeros((n, n)); s = np.zeros(n); gain = np.zeros((N, n)); offset = np.zeros(N)
    for k in range(N - 1, -1, -1):
        Fk, ck = F(k), c(k); inverse = 1.0 / (R * dt + beta * B @ P @ B)
        gain[k] = inverse * beta * (B @ P @ Fk); offset[k] = inverse * beta * (B @ (P @ ck + s))
        P_next = Q * dt + beta * Fk.T @ P @ Fk - beta ** 2 * np.outer(Fk.T @ P @ B, B @ P @ Fk) * inverse
        s = -q * dt + beta * Fk.T @ (P @ ck + s) - beta ** 2 * (Fk.T @ P @ B) * inverse * (B @ (P @ ck + s)); P = P_next
    return gain, offset


def _kalman(game, F, noise_covariance, C, observation_covariance, initial):
    """Gains of the discrete Kalman filter of z_k from the increments dy_k = C z_k dt + noise."""
    N, dt = game.N, game.dt; V = initial.copy(); gains = np.zeros((N, initial.shape[0], C.shape[0]))
    for k in range(N):
        gains[k] = V @ C.T @ np.linalg.inv(C @ V @ C.T * dt + observation_covariance)
        V = F(k) @ (V - gains[k] @ C * dt @ V) @ F(k).T + noise_covariance(k)
    return gains


def exact(game: MajorMinor, noise: dict) -> np.ndarray:
    """Exact equilibrium controls of the discretised game on the given paths, [M, 2, N]: major, minor."""
    N, dt = game.N, game.dt; a0 = 1 + game.A0 * dt
    gain0, offset0 = _lqr(game, lambda k: np.array([[a0]]), np.array([game.B0 * dt]), lambda k: np.zeros(1),
                          np.array([[game.Q0]]), np.array([game.Q0 * game.eta0]), game.R0)
    filter0 = _kalman(game, lambda k: np.array([[a0]]), lambda k: np.array([[game.D0 ** 2 * dt]]), np.array([[game.L0]]),
                      np.array([[game.s0 ** 2]]), np.array([[game.initial_variance]]))[:, 0, 0]

    def F1(k):
        return np.array([[1 + game.A1 * dt, game.G * dt, 0.0], [0.0, a0, -game.B0 * dt * gain0[k, 0]],
                         [0.0, a0 * filter0[k] * game.L0 * dt, a0 * (1 - filter0[k] * game.L0 * dt) - game.B0 * dt * gain0[k, 0]]])

    def c1(k):
        return np.array([0.0, -game.B0 * dt * offset0[k], -game.B0 * dt * offset0[k]])

    def noise1(k):
        return np.diag([game.D1 ** 2 * dt, game.D0 ** 2 * dt, (a0 * filter0[k] * game.s0) ** 2 * dt])
    B1 = np.array([game.B1 * dt, 0.0, 0.0]); C1 = np.array([[game.L1, 0, 0], [0, game.L2, 0]]); target = np.array([1.0, -game.H, 0.0])
    gain1, offset1 = _lqr(game, F1, B1, c1, np.outer(target, target) * game.Q1, target * game.Q1 * game.eta1, game.R1)
    filter1 = _kalman(game, F1, noise1, C1, np.diag([game.s1 ** 2, game.s1 ** 2]), np.diag([game.initial_variance, game.initial_variance, 0.0]))
    to_numpy = lambda key: noise[key].cpu().numpy().astype(np.float64)
    x0, x1 = to_numpy('x0_0'), to_numpy('x1_0'); dw0, dw1, dv0, dv1 = to_numpy('dw0'), to_numpy('dw1'), to_numpy('dv0'), to_numpy('dv1')
    M = x0.shape[0]; estimate0 = np.zeros(M); estimate1 = np.zeros((M, 3)); controls = np.zeros((M, 2, N))
    for k in range(N):
        u0 = -(gain0[k, 0] * estimate0 + offset0[k]); u1 = -(estimate1 @ gain1[k] + offset1[k]); controls[:, 0, k], controls[:, 1, k] = u0, u1
        dy0 = game.L0 * x0 * dt + game.s0 * dv0[:, k]
        dy1 = np.stack([game.L1 * x1, game.L2 * x0], -1) * dt + game.s1 * dv1[:, k]
        x0, x1 = (x0 + (game.A0 * x0 + game.B0 * u0) * dt + game.D0 * dw0[:, k],
                  x1 + (game.A1 * x1 + game.G * x0 + game.B1 * u1) * dt + game.D1 * dw1[:, k])
        estimate0 = a0 * (estimate0 + filter0[k] * (dy0 - game.L0 * estimate0 * dt)) + game.B0 * dt * u0
        posterior = estimate1 + (dy1 - estimate1 @ C1.T * dt) @ filter1[k].T
        estimate1 = posterior @ F1(k).T + c1(k) + B1 * u1[:, None]
    return controls
