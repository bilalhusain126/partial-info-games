"""
The control of one player: the network N^i of the note.
"""

import torch
import torch.nn as nn

from .utils import mlp


class Control(nn.Module):
    """
    The network N^i of player i.

    The control at t_j is a function of the observed path O^i on [0, t_j]:

        u^i_{t_j} = f( t_j, O^i_{t_j}, h_j ),

    where h_j is a learned summary of the observed path up to t_j and f is a feedforward network. The
    summary is computed by a GRU (a recurrent network) that takes the increments of O^i over
    [t_0, t_1], ..., [t_{j-1}, t_j] as input, one at a time, with h_0 = 0. Thus u^i_{t_j} depends only on
    the observations up to t_j, so it is F^i_{t_j}-measurable. A player without observation increments has
    no summary, and its control is a function of (t_j, O^i_{t_j}) only.
    """

    def __init__(self, observation_dim: int, increment_dim: int, width: int = 32):
        """
        Args:
            observation_dim: Dimension of the current observation O^i_t.
            increment_dim: Dimension of the observation increments (0 for no summary).
            width: Dimension of the summary h_j. Default 32.
        """
        super().__init__()
        self.width = width if increment_dim > 0 else 0
        self.gru = nn.GRU(increment_dim, width, batch_first=True) if increment_dim > 0 else None
        self.feedforward = mlp(1 + observation_dim + self.width, 1)

    def summaries(self, increments: torch.Tensor):
        """
        The summaries h_0, ..., h_{N-1} along whole paths of increments.

        Args:
            increments: Observation increments [M, N, increment_dim].

        Returns:
            Tensor [M, N, width] whose entry j is h_j, the summary of the increments before t_j (zero at
            j = 0), or None without a GRU.
        """
        if self.gru is None:
            return None
        states, _ = self.gru(increments)
        return torch.cat([torch.zeros(increments.shape[0], 1, self.width), states[:, :-1]], 1)

    def forward(self, t: torch.Tensor, observation: torch.Tensor, summary) -> torch.Tensor:
        """
        Args:
            t: Time [M, 1].
            observation: Current observation O^i_{t_j} [M, observation_dim].
            summary: Summary h_j of the observed path [M, width], or None.

        Returns:
            Controls [M].
        """
        inputs = [t, observation] + ([summary] if summary is not None else [])
        return self.feedforward(torch.cat(inputs, -1)).squeeze(-1)
