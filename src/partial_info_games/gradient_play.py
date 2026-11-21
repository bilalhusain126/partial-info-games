"""
Gradient play on the realised costs, an alternative solver for closed-loop Nash equilibria.
"""

import time

import torch
import torch.nn as nn

from .solver import Solver
from .utils import RunningAverage, format_errors, relative_l2, report, subset


def gradient_play(solver: Solver, iterations: int, steps: int = 200, lr: float = 1e-4, closed: bool = True,
                  reference: list = None, log=report):
    """
    Gradient play on the realised costs, an alternative solver for closed-loop equilibria.

    At each step, every player takes an Adam step on its own expected cost, estimated on a minibatch of
    simulated paths, with respect to its own network parameters. All players step together. A fixed point
    is a Nash equilibrium within the network class. In the closed loop, the game is simulated with every
    network in the graph, so player i's gradient includes the response of the other players' controls to
    its actions through their observations. In the open loop, the other players' controls are detached.

    Args:
        solver: A Solver whose controls are the starting point (for example after Solver.run).
        iterations: Number of iterations; each takes `steps` Adam steps and logs one line.
        steps: Adam steps per iteration. Default 200.
        lr: Learning rate. Default 1e-4.
        closed: Closed loop (default) or open loop.
        reference: Reference validation controls for the error log. Default none.
        log: Function that receives one progress line per iteration.

    Returns:
        controls: The validation controls averaged over the second half of the iterations.
        history: Per iteration, the relative L2 error of each player against reference.
    """
    game = solver.game
    M = next(iter(solver.training.values())).shape[0]
    players = range(game.n_players)
    optimizers = [torch.optim.Adam(c.parameters(), lr=lr) for c in solver.controls]
    average = RunningAverage()
    history = []
    start = time.time()
    for k in range(iterations):
        for _ in range(steps):
            part = subset(solver.training, torch.randint(0, M, (solver.batch,)))
            if closed:
                state, controls, _ = solver.simulate(part, live=set(players))
                costs = [game.cost(i, state, part, controls).mean() for i in players]
                gradients = [torch.autograd.grad(costs[i], list(solver.controls[i].parameters()),
                                                 retain_graph=i < game.n_players - 1) for i in players]
            else:
                gradients = []
                for i in players:
                    state, controls, _ = solver.simulate(part, live={i})
                    cost = game.cost(i, state, part, controls).mean()
                    gradients.append(torch.autograd.grad(cost, list(solver.controls[i].parameters())))
            for control, optimizer, gradient in zip(solver.controls, optimizers, gradients):
                for parameter, g in zip(control.parameters(), gradient):
                    parameter.grad = g
                nn.utils.clip_grad_norm_(control.parameters(), 5.0)
                optimizer.step()
        _, validation, _ = solver.simulate(solver.validation)
        if k >= iterations // 2:
            average.add(validation)
        message = f'gradient play iteration {k:2d}'
        if reference is not None:
            history.append([relative_l2(v, r) for v, r in zip(validation, reference)])
            message += '  error ' + format_errors(validation, reference)
        log(message + f'  ({time.time() - start:.0f}s)')
    return (average.value or validation), history
