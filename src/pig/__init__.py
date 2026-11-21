"""Deep Picard regression for Nash equilibria of stochastic differential games with private information."""

from .solver import Solver, Policy, gradient_play, first_order_residual, relative_l2, use_device
from .plotting import use_style, LEARNED, LINEAR
