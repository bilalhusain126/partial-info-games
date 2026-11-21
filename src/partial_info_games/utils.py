"""
Utilities: the device, the validation paths, the relative L2 error, and the feedforward network used by the
controls and by the control variate.
"""

import torch
import torch.nn as nn

# The validation paths, on which every method and every exact reference is compared.
VALIDATION_SEED = 7
N_VALIDATION = 2000


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
    """Exogenous processes of the validation paths of a game: N_VALIDATION paths from VALIDATION_SEED."""
    return game.sample(N_VALIDATION, VALIDATION_SEED)


def relative_l2(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """Relative L2 error ||estimate - reference|| / ||reference|| over all paths and times."""
    return float(((estimate - reference) ** 2).mean().sqrt() / (reference ** 2).mean().sqrt())


def mlp(n_in: int, n_out: int, width: int = 64) -> nn.Module:
    """Feedforward network with two hidden layers of the given width and tanh activations."""
    return nn.Sequential(nn.Linear(n_in, width), nn.Tanh(),
                         nn.Linear(width, width), nn.Tanh(),
                         nn.Linear(width, n_out))
