"""
The control of one player: the network N^i of the note (Section 1.4).
"""

import torch
import torch.nn as nn

from .utils import mlp


class Control(nn.Module):
    """
    The network N^i of player i (note, Section 1.4).

    The control at t_j is a function of the observed path O^i on [0, t_j]:

        u^i_{t_j} = head( t_j, O^i_{t_j}, h_j ),

    where h_j is the state of a GRU that takes the increments of O^i over [t_0, t_1], ..., [t_{j-1}, t_j]
    as input, with h_0 = 0. Thus u^i_{t_j} is F^i_{t_j}-measurable. A player whose observation increments
    have dimension 0 has no GRU, and its control is a function of (t_j, O^i_{t_j}) only.
    """

    def __init__(self, observation_dim: int, increment_dim: int, width: int = 32):
        """
        Args:
            observation_dim: Dimension of the current observation O^i_t.
            increment_dim: Dimension of the observation increments (0 for no memory).
            width: Hidden size of the GRU. Default 32.
        """
        super().__init__()
        self.width = width if increment_dim > 0 else 0
        self.gru = nn.GRU(increment_dim, width, batch_first=True) if increment_dim > 0 else None
        self.head = mlp(1 + observation_dim + self.width, 1)

    def memory(self, increments: torch.Tensor):
        """
        GRU states before each step, from a whole path of increments.

        Args:
            increments: Observation increments [M, N, increment_dim].

        Returns:
            Tensor [M, N, width] whose entry j is h_j, the state after the increments 0, ..., j - 1
            (zero at j = 0), or None without a GRU.
        """
        if self.gru is None:
            return None
        states, _ = self.gru(increments)
        return torch.cat([torch.zeros(increments.shape[0], 1, self.width), states[:, :-1]], 1)

    def forward(self, t: torch.Tensor, observation: torch.Tensor, memory) -> torch.Tensor:
        """
        Args:
            t: Time [M, 1].
            observation: Current observation [M, observation_dim].
            memory: GRU state h_j [M, width], or None.

        Returns:
            Controls [M].
        """
        inputs = [t, observation] + ([memory] if memory is not None else [])
        return self.head(torch.cat(inputs, -1)).squeeze(-1)
