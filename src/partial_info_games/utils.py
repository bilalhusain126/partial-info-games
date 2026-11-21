"""
Utilities shared by the solvers: the device, the validation paths, the relative L2 error, the running
average of the controls over iterations, and the feedforward network used by the controls and by the
control variate.
"""

import torch
import torch.nn as nn

# The validation paths, on which every method and every exact reference is compared.
VALIDATION_SEED = 7
N_VALIDATION = 2000


def report(message: str):
    """Prints a progress line immediately (unbuffered, for cluster logs)."""
    print(message, flush=True)


def use_device(name: str = None) -> str:
    """
    Sets the default torch device.

    Args:
        name: 'cuda', 'mps' or 'cpu'. Default: the first available in that order.

    Returns:
        The name of the device set.
    """
    if name is None:
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_default_device(name)
    return name


def validation_paths(game) -> dict:
    """Noise of the validation paths of a game: N_VALIDATION paths from VALIDATION_SEED."""
    return game.sample(N_VALIDATION, VALIDATION_SEED)


def relative_l2(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """Relative L2 error ||estimate - reference|| / ||reference|| over all paths and times."""
    return float(((estimate - reference) ** 2).mean().sqrt() / (reference ** 2).mean().sqrt())


def format_errors(controls: list, reference: list) -> str:
    """Relative L2 errors per player, formatted for a log line."""
    return ' '.join(f'{relative_l2(c, r):.3f}' for c, r in zip(controls, reference))


def subset(exogenous: dict, index: torch.Tensor) -> dict:
    """The exogenous processes of the paths in index."""
    return {key: value[index] for key, value in exogenous.items()}


def mlp(n_in: int, n_out: int, width: int = 64) -> nn.Module:
    """Feedforward network with two hidden layers of the given width and tanh activations."""
    return nn.Sequential(nn.Linear(n_in, width), nn.Tanh(),
                         nn.Linear(width, width), nn.Tanh(),
                         nn.Linear(width, n_out))


class RunningAverage:
    """
    Average of the validation controls over Picard iterations (iterate averaging).

    The average over the last iterates is reported instead of the last iterate. It reduces the
    iteration-to-iteration fluctuation caused by the regression noise.
    """

    def __init__(self):
        self.value = None
        self.count = 0

    def add(self, controls: list):
        """Adds one iterate: a list of [M, N] tensors, one per player."""
        self.count += 1
        if self.value is None:
            self.value = [c.clone() for c in controls]
        else:
            self.value = [v + (c - v) / self.count for v, c in zip(self.value, controls)]
