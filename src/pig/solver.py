"""Deep Picard regression for open-loop Nash equilibria of games with private information, its closed-loop
variant, and gradient play.

A game supplies:
    n_players, N, dt                     players, time steps, step size
    observation_dims, increment_dims     per player: current observation, observation increment
    full_dim, noise_dim, cv_decay        control variate: full state, martingale increments, kernel decay per player
    sample(n_paths, seed)                dict of noise tensors
    initial_state(noise)                 dict of state tensors on the time grid
    step(state, noise, controls, k)      writes the state at k + 1 given the controls at k
    observe(i, state, noise, k)          current observation of player i at k, [M, observation_dims[i]]
    increment(i, state, noise, k)        observation increment of player i over [t_k, t_k+1], [M, increment_dims[i]]
    cost(i, state, noise, controls)      realised cost of player i, [M]
    curvature(i)                         second derivative of the cost in the control at each t_k, [N]
    full_state(state, noise)             [M, N, full_dim], the full state at each t_k
    martingale_increments(noise)         [M, N, noise_dim], zero conditional mean given the full state at t_k
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn


def report(message: str):
    """Print a progress line immediately."""
    print(message, flush=True)


def use_device(name: str = None) -> str:
    """Set the default torch device: name, or cuda, mps or cpu in that order of availability."""
    if name is None:
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_default_device(name)
    return name


def mlp(n_in: int, n_out: int, width: int = 64) -> nn.Module:
    """Two hidden layers of the given width with tanh activations."""
    return nn.Sequential(nn.Linear(n_in, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, n_out))


class Policy(nn.Module):
    """Control of one player: u_k = head(t_k, observation_k, h_k), where h_k is the state of a GRU that takes the
    observation increments before t_k as input (h_0 = 0). Without increments the policy is a function of
    (t_k, observation_k) only."""

    def __init__(self, observation_dim: int, increment_dim: int, width: int = 32):
        """Args: observation_dim, increment_dim: input sizes; width: GRU hidden size (default 32)."""
        super().__init__()
        self.width = width if increment_dim > 0 else 0
        self.gru = nn.GRU(increment_dim, width, batch_first=True) if increment_dim > 0 else None
        self.head = mlp(1 + observation_dim + self.width, 1)

    def memory(self, increments: torch.Tensor):
        """GRU states before each step, [M, N, width], from the increments [M, N, increment_dim]."""
        if self.gru is None:
            return None
        out, _ = self.gru(increments)
        return torch.cat([torch.zeros(increments.shape[0], 1, self.width), out[:, :-1]], 1)

    def forward(self, t, observation, memory):
        inputs = [t, observation] + ([memory] if memory is not None else [])
        return self.head(torch.cat(inputs, -1)).squeeze(-1)


def subset(noise: dict, index) -> dict:
    """The noise of the paths in index."""
    return {key: value[index] for key, value in noise.items()}


class Solver:
    """Damped Picard iteration over all players. At each iteration the game is simulated under the current
    controls, each player's target is computed from the derivative of its realised cost, and each policy is
    fitted to the damped target by minimizing a mean squared error over its own observations."""

    def __init__(self, game, width: int = 32, damping: float = 0.6, steps: int = 1000, batch: int = 1024,
                 lr: float = 1e-3, n_paths: int = 8192, n_validation: int = 2048, control_variate: bool = False,
                 seed: int = 0):
        """Args: game: the game; width: GRU width (default 32); damping: weight of the new target (default 0.6);
        steps, batch, lr: Adam steps, minibatch size and learning rate of each fit (defaults 1000, 1024, 1e-3);
        n_paths: training paths (default 8192); n_validation: validation paths, common to all seeds (default 2048);
        control_variate: subtract a learned martingale term from the targets (default False); seed: seed of the
        networks and of the training paths (default 0)."""
        torch.manual_seed(seed)
        self.game, self.damping, self.steps, self.batch, self.lr = game, damping, steps, batch, lr
        self.policies = [Policy(game.observation_dims[i], game.increment_dims[i], width) for i in range(game.n_players)]
        self.control_variate = control_variate
        if control_variate:
            self.zetas = [mlp(1 + game.full_dim, game.noise_dim) for _ in range(game.n_players)]
            j, k = np.indices((game.N, game.N))
            self.cv_kernels = [torch.tensor(np.where(j >= k, decay ** (j - k), 0.0), dtype=torch.float32)
                               for decay in game.cv_decay]
        self.noise = game.sample(n_paths, 1 + 100 * seed)
        self.validation = game.sample(n_validation, 7)
        self.t = torch.arange(game.N, dtype=torch.float32) * game.dt

    def simulate(self, noise: dict, fixed: dict = None, live=()):
        """Simulate the game. Player j uses fixed[j] ([M, N]) if given, else its policy; policy outputs are
        detached unless j is in live. Returns the state, the controls and the observation increments."""
        game = self.game; M = next(iter(noise.values())).shape[0]; fixed = fixed or {}
        state = game.initial_state(noise)
        hidden = [None if p.gru is None else torch.zeros(1, M, p.width) for p in self.policies]
        controls = [[] for _ in self.policies]; increments = [[] for _ in self.policies]
        with torch.set_grad_enabled(bool(live) or any(u.requires_grad for u in fixed.values())):
            for k in range(game.N):
                u = {}
                for j, policy in enumerate(self.policies):
                    if j in fixed:
                        u[j] = fixed[j][:, k]
                    else:
                        memory = None if policy.gru is None else hidden[j][0]
                        u[j] = policy(self.t[k].expand(M, 1), game.observe(j, state, noise, k), memory)
                        u[j] = u[j] if j in live else u[j].detach()
                    controls[j].append(u[j])
                game.step(state, noise, u, k)
                for j, policy in enumerate(self.policies):
                    if policy.gru is not None and j not in fixed:
                        increment = game.increment(j, state, noise, k); increments[j].append(increment)
                        hidden[j] = policy.gru(increment[:, None], hidden[j])[1]
                        if j not in live:
                            hidden[j] = hidden[j].detach()
        controls = [torch.stack(c, 1) for c in controls]
        increments = [torch.stack(c, 1) if c else None for c in increments]
        return state, controls, increments

    def targets(self, noise: dict, controls: list, closed: bool = False, chunk: int = 4096) -> list:
        """Best-response targets u_i - (dJ_i/du_i) / curvature_i, with J_i the realised cost, by automatic
        differentiation. Open loop (default): the other players' controls are held fixed as given. Closed loop:
        they act through their policies, so they respond to player i through their observations."""
        game = self.game; M = controls[0].shape[0]; out = []
        for i in range(game.n_players):
            target = torch.zeros_like(controls[i])
            for a in range(0, M, chunk):
                index = torch.arange(a, min(a + chunk, M)); part = subset(noise, index)
                u_i = controls[i][index].detach().clone().requires_grad_(True)
                fixed = {i: u_i} if closed else {j: (u_i if j == i else c[index].detach()) for j, c in enumerate(controls)}
                state, used, _ = self.simulate(part, fixed, live=set(range(game.n_players)) - {i} if closed else ())
                J = game.cost(i, state, part, used).sum()
                gradient, = torch.autograd.grad(J, u_i)
                target[index] = (u_i - gradient / game.curvature(i)).detach()
            out.append(target)
        return out

    def fit(self, noise: dict, state: dict, increments: list, targets: list, damping: float, players=None) -> float:
        """Fit the policies of players (default all) to the targets by Adam on the mean squared error; with the
        control variate, damping times the martingale term is subtracted from each target. Returns the last loss."""
        game = self.game; M = targets[0].shape[0]; players = range(game.n_players) if players is None else players
        observations = [torch.stack([game.observe(i, state, noise, k) for k in range(game.N)], 1) for i in range(game.n_players)]
        t = self.t.expand(M, game.N)[..., None]
        parameters = [q for i in players for q in self.policies[i].parameters()]
        if self.control_variate:
            full = torch.cat([t, game.full_state(state, noise)], -1); dW = game.martingale_increments(noise)
            parameters += [q for i in players for q in self.zetas[i].parameters()]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        for _ in range(self.steps):
            index = torch.randint(0, M, (self.batch,)); loss = 0.0
            for i in players:
                policy = self.policies[i]
                memory = policy.memory(increments[i][index]) if policy.gru is not None else None
                target = targets[i][index]
                if self.control_variate:
                    target = target - damping * ((self.zetas[i](full[index]) * dW[index]).sum(-1) @ self.cv_kernels[i])
                loss = loss + ((policy(t[index], observations[i][index], memory) - target) ** 2).mean()
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(parameters, 5.0); optimizer.step()
        return float(loss.detach())

    def run(self, iterations: int, average_from: int = 4, reference: list = None, closed: bool = False, log=report):
        """Damped Picard iteration. Returns the validation controls averaged over the iterations from average_from
        on, and the relative L2 errors per iteration against reference (validation controls per player), if given.
        closed=True uses the closed-loop targets."""
        average = None; count = 0; history = []; start = time.time()
        for iteration in range(iterations):
            state, controls, increments = self.simulate(self.noise)
            targets = self.targets(self.noise, controls, closed)
            damped = [(1 - self.damping) * c + self.damping * b for c, b in zip(controls, targets)]
            loss = self.fit(self.noise, state, increments, damped, self.damping)
            _, validation, _ = self.simulate(self.validation)
            if iteration >= average_from:
                count += 1
                average = [v.clone() for v in validation] if average is None else [a + (v - a) / count for a, v in zip(average, validation)]
            message = f'iteration {iteration:2d}  loss {loss:.2e}'
            if reference is not None:
                errors = [relative_l2(v, r) for v, r in zip(validation, reference)]; history.append(errors)
                message += '  error ' + ' '.join(f'{e:.3f}' for e in errors)
                if average is not None:
                    message += '  averaged ' + ' '.join(f'{relative_l2(a, r):.3f}' for a, r in zip(average, reference))
            log(message + f'  ({time.time() - start:.0f}s)')
        return (average if average is not None else validation), history


def relative_l2(estimate, reference) -> float:
    """Relative L2 error of estimate against reference, over all paths and times."""
    return float(((estimate - reference) ** 2).mean().sqrt() / (reference ** 2).mean().sqrt())


def gradient_play(solver: Solver, iterations: int, steps: int = 200, lr: float = 1e-4, closed: bool = True,
                  reference: list = None, log=report):
    """Gradient play on the realised costs: at each Adam step every player descends its own expected cost,
    estimated on a minibatch, with respect to its own policy parameters. Closed loop (default): the other players
    act through their policies inside the simulation. Open loop: their controls are detached. Each of the
    iterations takes steps Adam steps with learning rate lr. Returns the validation controls averaged over the
    second half of the iterations, and the errors per iteration against reference."""
    game = solver.game; M = next(iter(solver.noise.values())).shape[0]
    optimizers = [torch.optim.Adam(p.parameters(), lr=lr) for p in solver.policies]
    average = None; count = 0; history = []; start = time.time()
    for iteration in range(iterations):
        for _ in range(steps):
            part = subset(solver.noise, torch.randint(0, M, (solver.batch,)))
            for i, (policy, optimizer) in enumerate(zip(solver.policies, optimizers)):
                state, controls, _ = solver.simulate(part, live=set(range(game.n_players)) if closed else {i})
                gradients = torch.autograd.grad(game.cost(i, state, part, controls).mean(), list(policy.parameters()))
                for parameter, gradient in zip(policy.parameters(), gradients):
                    parameter.grad = gradient
                nn.utils.clip_grad_norm_(policy.parameters(), 5.0); optimizer.step()
        _, validation, _ = solver.simulate(solver.validation)
        if iteration >= iterations // 2:
            count += 1
            average = [v.clone() for v in validation] if average is None else [a + (v - a) / count for a, v in zip(average, validation)]
        message = f'gradient play iteration {iteration:2d}'
        if reference is not None:
            errors = [relative_l2(v, r) for v, r in zip(validation, reference)]; history.append(errors)
            message += '  error ' + ' '.join(f'{e:.3f}' for e in errors)
        log(message + f'  ({time.time() - start:.0f}s)')
    return (average if average is not None else validation), history


def first_order_residual(solver: Solver, i: int, n_paths: int = None) -> float:
    """Relative L2 distance, on the validation paths, between player i's control and an independently initialised
    policy fitted (twice the solver's steps) to player i's undamped target on fresh paths (default as many as the
    solver's). Zero up to regression error at an equilibrium."""
    game = solver.game
    noise = game.sample(n_paths or next(iter(solver.noise.values())).shape[0], 23 + i)
    state, controls, increments = solver.simulate(noise)
    target = solver.targets(noise, controls)[i]
    saved_policy, saved_steps = solver.policies[i], solver.steps
    saved_zeta = copy.deepcopy(solver.zetas[i]) if solver.control_variate else None
    try:
        solver.policies[i] = Policy(game.observation_dims[i], game.increment_dims[i], saved_policy.width or 32)
        solver.steps = 2 * saved_steps
        solver.fit(noise, state, increments, [target if j == i else controls[j] for j in range(game.n_players)], 1.0, players=[i])
        _, fresh, _ = solver.simulate(solver.validation)
    finally:
        solver.policies[i], solver.steps = saved_policy, saved_steps
        if solver.control_variate:
            solver.zetas[i] = saved_zeta
    _, current, _ = solver.simulate(solver.validation)
    return relative_l2(fresh[i], current[i])
