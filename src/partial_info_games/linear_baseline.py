"""
The equilibrium among controls linear in the observed path, a baseline for the learned equilibrium.
"""

import torch

from .solver import Solver
from .utils import RunningAverage, format_errors, report


def linear_equilibrium(solver: Solver, iterations: int = 30, average_from: int = 15, reference: list = None, log=report):
    """
    The damped Picard iteration restricted to controls linear in the observed path.

    The regression of Solver.fit is replaced by ordinary least squares at each t_j on the features
    (1, O^i_{t_j}, all increments of O^i before t_j). This gives the best linear predictor of the target,
    and the iteration converges to the equilibrium among strategies linear in the observed path, a
    baseline for the learned equilibrium in games where the conditional expectation is nonlinear.

    Args:
        solver: A Solver, used for its game, paths, damping and targets.
        iterations: Number of iterations. Default 30.
        average_from: First iteration included in the average of the validation controls. Default 15.
        reference: Reference validation controls for the error log. Default none.
        log: Function that receives one progress line per iteration.

    Returns:
        The averaged validation controls, list of [M, N] tensors.
    """
    game = solver.game
    N = game.N
    players = range(game.n_players)

    def simulate(exogenous, coefficients):
        """Simulates the game under the linear controls; returns the controls and the features at each step."""
        M = next(iter(exogenous.values())).shape[0]
        state = game.initial_state(exogenous)
        controls = [torch.zeros(M, N) for _ in players]
        features = [[None] * N for _ in players]
        past = [[] for _ in players]
        for j in range(N):
            u = {}
            for i in players:
                features[i][j] = torch.cat([torch.ones(M, 1), game.observe(i, state, exogenous, j)] + past[i], 1)
                u[i] = features[i][j] @ coefficients[i][j] if coefficients else torch.zeros(M)
                controls[i][:, j] = u[i]
            game.step(state, exogenous, u, j)
            for i in players:
                if game.increment_dims[i] > 0:
                    past[i].append(game.increment(i, state, exogenous, j))
        return controls, features

    def least_squares(features, target):
        """Least-squares coefficients in float64, with a rank cutoff because the features can be collinear."""
        features, target = features.cpu().double(), target.cpu().double()[:, None]
        solution = torch.linalg.lstsq(features, target, rcond=1e-6, driver='gelsd').solution
        return solution[:, 0].float().to(torch.get_default_device())

    coefficients = None
    average = RunningAverage()
    for k in range(iterations):
        controls, features = simulate(solver.training, coefficients)
        targets = solver.targets(solver.training, controls)
        damped = [(1 - solver.damping) * u + solver.damping * target for u, target in zip(controls, targets)]
        coefficients = [[least_squares(features[i][j], damped[i][:, j]) for j in range(N)] for i in players]
        validation, _ = simulate(solver.validation, coefficients)
        if k >= average_from:
            average.add(validation)
        if reference is not None:
            log(f'linear iteration {k:2d}  error ' + format_errors(average.value or validation, reference))
    return average.value or validation
