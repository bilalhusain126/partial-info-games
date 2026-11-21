"""
Abstract base class for the games, and the shared part of the two trading games.

A game is defined on the grid t_j = j dt, j = 0, ..., N, with dt = T / N (note, Section 1.4). It
supplies the simulator of the state X (note, Section 1.1), the observations O^i of each player, the
realised cost of each player (minus the reward of the note), and the curvature 2 a_i dt of the cost in
the player's own control, from which the solver builds the targets Lambda^i / (2 a_i).
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch


class Game(ABC):
    """
    Abstract base class representing a game of n players on the Euler--Maruyama grid.

    Class attributes set by subclasses:
        n_players:         number of players
        T, N:              horizon and number of time steps
        observation_dims:  per player, the dimension of the current observation O^i_t
        increment_dims:    per player, the dimension of the observation increments (the GRU input)
        full_dim:          dimension of the full state X_t (the input of the control variate)
        noise_dim:         number of martingale increments per step (the factors of the control variate)
        cv_decay:          per player, the decay per step of the cost sensitivity: Lambda^i_{t_j} depends on
                           the increment at t_{j'} > t_j with weight proportional to decay^(j' - j)
    """

    @property
    def dt(self) -> float:
        """Time step T / N."""
        return self.T / self.N

    @abstractmethod
    def sample(self, n_paths: int, seed: int) -> dict:
        """
        Simulates the exogenous noise.

        Args:
            n_paths: Number of paths M.
            seed: Seed of the random number generator.

        Returns:
            Dictionary of float32 tensors on the default device (Brownian increments [M, N], signal paths
            [M, N + 1], ...).
        """

    @abstractmethod
    def initial_state(self, noise: dict) -> dict:
        """Returns the state paths on the time grid, as tensors [M, N + 1] initialised at t_0."""

    @abstractmethod
    def step(self, state: dict, noise: dict, controls: dict, j: int):
        """Writes the state at t_{j+1} from the state at t_j and the controls {player: [M]} at t_j."""

    @abstractmethod
    def observe(self, i: int, state: dict, noise: dict, j: int) -> torch.Tensor:
        """Current observation O^i_{t_j} of player i, [M, observation_dims[i]]."""

    @abstractmethod
    def increment(self, i: int, state: dict, noise: dict, j: int) -> torch.Tensor:
        """Observation increment of player i over [t_j, t_{j+1}], rescaled to order one, [M, increment_dims[i]]."""

    @abstractmethod
    def cost(self, i: int, state: dict, noise: dict, controls: list) -> torch.Tensor:
        """Realised cost of player i along each path (minus the reward of the note), [M]."""

    @abstractmethod
    def curvature(self, i: int) -> torch.Tensor:
        """Second derivative of the realised cost in player i's control at each t_j (2 a_i dt), [N]."""

    @abstractmethod
    def full_state(self, state: dict, noise: dict) -> torch.Tensor:
        """Full state X_{t_j} at each t_j, [M, N, full_dim]."""

    @abstractmethod
    def martingale_increments(self, noise: dict) -> torch.Tensor:
        """Increments over [t_j, t_{j+1}] with zero conditional mean given the full state at t_j, [M, N, noise_dim]."""

    def exact(self, noise: dict) -> np.ndarray:
        """
        Exact equilibrium controls on the given paths, [M, n_players, N].

        Raises:
            NotImplementedError: If the game has no exact solution at its parameters.
        """
        raise NotImplementedError(f'{type(self).__name__} has no exact solution at these parameters')


# ==============================================================================
# Two traders with private signals and a common price
# ==============================================================================

@dataclass
class TradingGame(Game):
    """
    Two traders with private signals alpha^1, alpha^2 and a common price S.

    Dynamics:
        price:        dS_t = mu_t dt + sigma dW^S_t,   mu_t = alpha^1_t + alpha^2_t + lam (nu^1_t + nu^2_t)
        inventories:  dq^i_t = nu^i_t dt
    The signals come from the subclass (Ornstein--Uhlenbeck in TwoTraders, Markov chains in MarkovTraders).
    They are private: trader i observes O^i = (alpha^i, q^i, S).

    Reward of trader i (the code minimizes its negative):
        J^i = E[ int_0^T ( q^i_t mu_t - a (nu^i_t)^2 - phi (q^i_t)^2 ) dt - psi (q^i_T)^2 ].
    The first order condition (note, Section 1.2, with a_i = a) is
        2 a nu^i_t = E[ lam q^i_t - 2 psi q^i_T + int_t^T ( mu_s - 2 phi q^i_s ) ds | F^i_t ],
    so Lambda^i_t = lam q^i_t - 2 psi q^i_T + int_t^T ( mu_s - 2 phi q^i_s ) ds.
    """

    lam: float = 1.0          # permanent price impact
    a: float = 0.5            # temporary impact (cost of the trading rate)
    phi: float = 0.1          # running inventory penalty
    psi: float = 1.0          # terminal inventory penalty
    sigma: float = 0.5        # price volatility
    T: float = 1.0
    N: int = 50

    n_players = 2
    observation_dims = [3, 3]
    increment_dims = [2, 2]
    full_dim = 5
    cv_decay = [1.0, 1.0]

    @abstractmethod
    def signal_increment(self, alpha: torch.Tensor, j: int) -> torch.Tensor:
        """Rescaled increment of a signal path alpha [M, N + 1] over [t_j, t_{j+1}], [M]."""

    def initial_state(self, noise):
        M = noise['dWS'].shape[0]
        return dict(q1=torch.zeros(M, self.N + 1), q2=torch.zeros(M, self.N + 1),
                    S=torch.zeros(M, self.N + 1), mu=torch.zeros(M, self.N))

    def step(self, state, noise, controls, j):
        mu = noise['alpha1'][:, j] + noise['alpha2'][:, j] + self.lam * (controls[0] + controls[1])
        state['mu'][:, j] = mu
        state['q1'][:, j + 1] = state['q1'][:, j] + controls[0] * self.dt
        state['q2'][:, j + 1] = state['q2'][:, j] + controls[1] * self.dt
        state['S'][:, j + 1] = state['S'][:, j] + mu * self.dt + self.sigma * noise['dWS'][:, j]

    def observe(self, i, state, noise, j):
        """O^i_{t_j} = (alpha^i, q^i, S) at t_j."""
        return torch.stack([noise[f'alpha{i + 1}'][:, j], state[f'q{i + 1}'][:, j], state['S'][:, j]], -1)

    def increment(self, i, state, noise, j):
        """Increments of (alpha^i, S) over [t_j, t_{j+1}], the price increment divided by sqrt(dt)."""
        S = state['S']
        price_increment = (S[:, j + 1] - S[:, j]) / math.sqrt(self.dt)
        return torch.stack([self.signal_increment(noise[f'alpha{i + 1}'], j), price_increment], -1)

    def cost(self, i, state, noise, controls):
        """Minus the realised reward of trader i."""
        q = state[f'q{i + 1}']
        running = (q[:, :-1] * state['mu'] - self.a * controls[i] ** 2 - self.phi * q[:, :-1] ** 2).sum(1) * self.dt
        return -(running - self.psi * q[:, -1] ** 2)

    def curvature(self, i):
        """2 a dt."""
        return torch.full((self.N,), 2 * self.a * self.dt)

    def full_state(self, state, noise):
        """X_{t_j} = (alpha^1, alpha^2, q^1, q^2, S) at t_j."""
        return torch.stack([noise['alpha1'][:, :-1], noise['alpha2'][:, :-1],
                            state['q1'][:, :-1], state['q2'][:, :-1], state['S'][:, :-1]], -1)
