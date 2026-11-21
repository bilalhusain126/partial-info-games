"""
Two traders with private Markov-chain signals and a common price.

The game of base.TradingGame with each signal αⁱ a two-state Markov chain on {-A, A}, switching at
rate κ, simulated exactly on the grid and observed by trader i only. The conditional expectation of
the other signal given the own observations is the Wonham filter, which is nonlinear in the observed path.
The linear-strategy solver of two_traders therefore does not apply.

Exact solution, used to validate the learned method:
    MarkovTraders.exact     at λ = 0 the traders decouple. Each trader faces the signal
                            αⁱ + 𝔼[αʲ | ℱⁱ], with the conditional expectation from the exact
                            discrete Bayes filter of the other chain, and solves a single-agent problem
                            by a backward recursion.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from .base import TradingGame


@dataclass
class MarkovTraders(TradingGame):
    """Model parameters and the game interface (see base.TradingGame for the price and the rewards)."""

    kappa: float = 1.0          # switching rate of each chain
    amplitude: float = 1.0      # states of each chain: -amplitude, +amplitude

    noise_dim = 3               # the control variate uses dW^S and the two compensated chain increments

    @property
    def decay(self) -> float:
        """exp(-2 κ dt) = 𝔼[alpha_{t_{j+1}} | alpha_{t_j}] / alpha_{t_j} for the two-state chain."""
        return math.exp(-2 * self.kappa * self.dt)

    def sample(self, n_paths, seed):
        """
        Simulates the chains exactly on the grid, from the uniform initial law.

        Returns:
            alpha1, alpha2: chain paths [M, N + 1]
            dWS: Brownian increments of the price [M, N]
        """
        rng = np.random.default_rng(seed)
        N = self.N
        starts = [rng.integers(0, 2, n_paths), rng.integers(0, 2, n_paths)]
        uniforms = [rng.random((n_paths, N)), rng.random((n_paths, N))]
        exogenous = dict(dWS=rng.standard_normal((n_paths, N)) * math.sqrt(self.dt))
        stay = 0.5 * (1 + self.decay)                # probability that a chain keeps its state over one step
        for i, (start, uniform) in enumerate(zip(starts, uniforms), 1):
            state = np.zeros((n_paths, N + 1), dtype=int)
            state[:, 0] = start
            for j in range(N):
                state[:, j + 1] = np.where(uniform[:, j] < stay, state[:, j], 1 - state[:, j])
            exogenous[f'alpha{i}'] = self.amplitude * (2 * state - 1).astype(float)
        return {key: torch.tensor(value, dtype=torch.float32) for key, value in exogenous.items()}

    def martingale_increments(self, exogenous):
        """
        dW^S / √dt and the compensated chain increments (αⁱ_{t_{j+1}} - decay αⁱ_{t_j}) / √dt,
        which have zero mean given αⁱ_{t_j}.
        """
        increments = [exogenous['dWS'] / math.sqrt(self.dt)]
        for i in (1, 2):
            alpha = exogenous[f'alpha{i}']
            increments.append((alpha[:, 1:] - self.decay * alpha[:, :-1]) / math.sqrt(self.dt))
        return torch.stack(increments, -1)

    def exact(self, exogenous):
        """
        Exact equilibrium at λ = 0, [M, 2, N].

        Trader i faces the signal m_{t_j} = αⁱ_{t_j} + 𝔼[αʲ_{t_j} | ℱⁱ_{t_j}], whose future
        conditional means decay by exp(-2 κ dt) per step. Its value function is quadratic in the
        inventory, V_j(q) = A_j q² + B_j q + C_j, with B_j linear in the signal estimate, which gives the
        control nu_{t_j} = -G_j q_{t_j} - g_j m_{t_j}. The conditional expectation of the other chain is the
        discrete Bayes filter from the price increments dS - αⁱ dt = αʲ dt + σ dW^S.

        Raises:
            NotImplementedError: If λ ≠ 0.
        """
        if self.lam != 0.0:
            raise NotImplementedError('exact solution at lam = 0 only')
        dt, N, a, r = self.dt, self.N, self.a, self.decay

        # Backward recursion for the feedback coefficients G_j, g_j.
        A = np.zeros(N + 1)
        A[N] = self.psi
        S = np.zeros(N + 1)
        G = np.zeros(N)
        g = np.zeros(N)
        for j in range(N - 1, -1, -1):
            denominator = 2 * a + 2 * A[j + 1] * dt
            G[j] = 2 * A[j + 1] / denominator
            g[j] = -S[j + 1] * r / denominator
            A[j] = self.phi * dt + A[j + 1] - (2 * A[j + 1]) ** 2 * dt / (2 * denominator)
            S[j] = dt + S[j + 1] * r * (1 - A[j + 1] * dt / (a + A[j + 1] * dt))

        # Controls along the paths, with the Bayes filter of the other chain.
        alpha = np.stack([exogenous['alpha1'].cpu().numpy(), exogenous['alpha2'].cpu().numpy()], 1).astype(np.float64)
        dWS = exogenous['dWS'].cpu().numpy().astype(np.float64)
        M = alpha.shape[0]
        stay = 0.5 * (1 + r)
        amplitude, sigma = self.amplitude, self.sigma
        probability = np.full((M, 2), 0.5)              # probability[:, i] = P(αʲ = +A | ℱⁱ) before step j
        q = np.zeros((M, 2))
        nu = np.zeros((M, 2, N))
        for j in range(N):
            estimate = amplitude * (2 * probability - 1)
            nu[:, :, j] = -G[j] * q - g[j] * (alpha[:, :, j] + estimate)
            q = q + nu[:, :, j] * dt
            observed = alpha[:, ::-1, j] * dt + sigma * dWS[:, j, None]          # αʲ dt + σ dW^S
            likelihood_up = np.exp(-(observed - amplitude * dt) ** 2 / (2 * sigma ** 2 * dt))
            likelihood_down = np.exp(-(observed + amplitude * dt) ** 2 / (2 * sigma ** 2 * dt))
            posterior = probability * likelihood_up / (probability * likelihood_up + (1 - probability) * likelihood_down)
            probability = stay * posterior + (1 - stay) * (1 - posterior)
        return nu
