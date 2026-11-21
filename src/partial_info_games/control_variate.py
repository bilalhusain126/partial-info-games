"""
Martingale control variate for the regression targets.

The target Λⁱ_{t_j} / (2 aᵢ) contains the future noise of the game, which has zero conditional mean given
the player's information but makes the regression noisy. The control variate subtracts a learned estimate of that
noise,

    Σ_{j' ≥ j} decayᵢ^{j' - j} ζⁱ(t_{j'}, X_{t_{j'}}) · dW_{j'},

where X is the full state, dW_{j'} are increments over [t_{j'}, t_{j'+1}] with zero conditional mean given the
full state at t_{j'}, and ζⁱ is a network fitted jointly with the control. Each term has zero conditional
mean given ℱⁱ_{t_j}, so the conditional expectation, and hence the fixed point, are unchanged. Only the variance
of the target drops. The game supplies full_state, martingale_increments and cv_decay.
"""

import numpy as np
import torch

from .utils import mlp


class ControlVariate:
    """The networks ζⁱ and the weights decayᵢ^{j' - j} of the martingale term of each player."""

    def __init__(self, game):
        """
        Args:
            game: The game; it must implement full_state and martingale_increments, and set full_dim, noise_dim
                and cv_decay.
        """
        self.game = game
        self.zetas = [mlp(1 + game.full_dim, game.noise_dim) for _ in range(game.n_players)]
        later, now = np.indices((game.N, game.N))
        self.weights = [torch.tensor(np.where(later >= now, decay ** (later - now), 0.0), dtype=torch.float32)
                        for decay in game.cv_decay]

    def parameters(self, players) -> list:
        """Parameters of the networks ζⁱ of the given players."""
        return [p for i in players for p in self.zetas[i].parameters()]

    def inputs(self, state: dict, exogenous: dict, t: torch.Tensor):
        """
        Inputs of the martingale term along the simulated paths.

        Args:
            state, exogenous: The simulated paths.
            t: Times [M, N, 1].

        Returns:
            (t, X_t) [M, N, 1 + full_dim] and the martingale increments [M, N, noise_dim].
        """
        return torch.cat([t, self.game.full_state(state, exogenous)], -1), self.game.martingale_increments(exogenous)

    def term(self, i: int, full_state: torch.Tensor, increments: torch.Tensor) -> torch.Tensor:
        """
        The martingale term of player i at each t_j, [M, N], from the inputs of the paths in a minibatch.
        """
        return (self.zetas[i](full_state) * increments).sum(-1) @ self.weights[i]
