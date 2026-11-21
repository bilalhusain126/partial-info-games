"""Two traders with private Markov-chain signals and a common price.

The game of two_traders with the signals replaced by independent two-state Markov chains alpha^i in {-A, A},
switching at rate kappa, simulated exactly on the grid. The conditional expectation of the other trader's signal
is the Wonham filter, which is nonlinear in the observed path, so the linear-strategy solver does not apply.
At lam = 0 the exact solution combines the discrete Bayes filter of the other chain with a backward recursion for
the single-trader problem (exact_lam0). linear_equilibrium computes the equilibrium among strategies linear in
the own observed path, as a baseline.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..solver import relative_l2


@dataclass
class MarkovTraders:
    """Model parameters and the game interface of pig.solver."""

    lam: float = 1.0          # permanent price impact
    a: float = 0.5            # temporary impact
    phi: float = 0.1          # running inventory penalty
    psi: float = 1.0          # terminal inventory penalty
    kappa: float = 1.0        # switching rate of each chain
    amplitude: float = 1.0    # states of each chain: -amplitude, +amplitude
    sigma: float = 0.5        # price volatility
    jump_variate: bool = True # include the compensated chain increments in the control variate
    T: float = 1.0
    N: int = 50

    n_players = 2
    observation_dims = [3, 3]
    increment_dims = [2, 2]
    full_dim = 5
    cv_decay = [1.0, 1.0]

    @property
    def dt(self) -> float:
        return self.T / self.N

    @property
    def noise_dim(self) -> int:
        return 3 if self.jump_variate else 1

    @property
    def decay(self) -> float:
        """E[alpha_k+1 | alpha_k] / alpha_k = exp(-2 kappa dt)."""
        return math.exp(-2 * self.kappa * self.dt)

    def sample(self, n_paths: int, seed: int) -> dict:
        """Chain paths alpha1, alpha2 [M, N+1] (uniform initial state, exact transitions) and price increments dWS."""
        rng = np.random.default_rng(seed); N = self.N
        start1, start2 = rng.integers(0, 2, n_paths), rng.integers(0, 2, n_paths)
        uniform1, uniform2 = rng.random((n_paths, N)), rng.random((n_paths, N))
        dWS = rng.standard_normal((n_paths, N)) * math.sqrt(self.dt)
        stay = 0.5 * (1 + self.decay); noise = {}
        for i, (start, uniform) in enumerate([(start1, uniform1), (start2, uniform2)], 1):
            s = np.zeros((n_paths, N + 1), dtype=int); s[:, 0] = start
            for k in range(N):
                s[:, k + 1] = np.where(uniform[:, k] < stay, s[:, k], 1 - s[:, k])
            noise[f'alpha{i}'] = self.amplitude * (2 * s - 1).astype(float)
        noise['dWS'] = dWS
        return {key: torch.tensor(value, dtype=torch.float32) for key, value in noise.items()}

    def initial_state(self, noise: dict) -> dict:
        M = noise['dWS'].shape[0]
        return dict(q1=torch.zeros(M, self.N + 1), q2=torch.zeros(M, self.N + 1), S=torch.zeros(M, self.N + 1), mu=torch.zeros(M, self.N))

    def step(self, state, noise, controls, k):
        mu = noise['alpha1'][:, k] + noise['alpha2'][:, k] + self.lam * (controls[0] + controls[1])
        state['mu'][:, k] = mu
        state['q1'][:, k + 1] = state['q1'][:, k] + controls[0] * self.dt
        state['q2'][:, k + 1] = state['q2'][:, k] + controls[1] * self.dt
        state['S'][:, k + 1] = state['S'][:, k] + mu * self.dt + self.sigma * noise['dWS'][:, k]

    def observe(self, i, state, noise, k):
        return torch.stack([noise[f'alpha{i + 1}'][:, k], state[f'q{i + 1}'][:, k], state['S'][:, k]], -1)

    def increment(self, i, state, noise, k):
        alpha = noise[f'alpha{i + 1}']; S = state['S']
        return torch.stack([alpha[:, k + 1] - alpha[:, k], (S[:, k + 1] - S[:, k]) / math.sqrt(self.dt)], -1)

    def cost(self, i, state, noise, controls):
        q = state[f'q{i + 1}']
        running = (q[:, :-1] * state['mu'] - self.a * controls[i] ** 2 - self.phi * q[:, :-1] ** 2).sum(1) * self.dt
        return -(running - self.psi * q[:, -1] ** 2)

    def curvature(self, i):
        return torch.full((self.N,), 2 * self.a * self.dt)

    def full_state(self, state, noise):
        return torch.stack([noise['alpha1'][:, :-1], noise['alpha2'][:, :-1], state['q1'][:, :-1], state['q2'][:, :-1], state['S'][:, :-1]], -1)

    def martingale_increments(self, noise):
        """Price noise and, with jump_variate, the compensated chain increments alpha_k+1 - decay alpha_k."""
        increments = [noise['dWS'] / math.sqrt(self.dt)]
        if self.jump_variate:
            increments += [(noise[f'alpha{i}'][:, 1:] - self.decay * noise[f'alpha{i}'][:, :-1]) / math.sqrt(self.dt) for i in (1, 2)]
        return torch.stack(increments, -1)


def exact_lam0(game: MarkovTraders, noise: dict) -> np.ndarray:
    """Exact equilibrium controls at lam = 0 on the given paths, [M, 2, N]. Trader i faces the signal
    m_k = alpha^i_k + E[alpha^j_k | F^i_k], with the conditional expectation from the exact discrete Bayes filter, and
    the control nu_k = -G_k q_k - g_k E[m_k | F^i_k] from a backward recursion."""
    assert game.lam == 0.0
    dt, N, a = game.dt, game.N, game.a; r = game.decay
    A = np.zeros(N + 1); A[N] = game.psi; S = np.zeros(N + 1); G = np.zeros(N); g = np.zeros(N)
    for k in range(N - 1, -1, -1):
        denominator = 2 * a + 2 * A[k + 1] * dt
        G[k] = 2 * A[k + 1] / denominator; g[k] = -S[k + 1] * r / denominator
        A[k] = game.phi * dt + A[k + 1] - (2 * A[k + 1]) ** 2 * dt / (2 * denominator)
        S[k] = dt + S[k + 1] * r * (1 - A[k + 1] * dt / (a + A[k + 1] * dt))
    alpha = np.stack([noise['alpha1'].cpu().numpy(), noise['alpha2'].cpu().numpy()], 1).astype(np.float64)
    dWS = noise['dWS'].cpu().numpy().astype(np.float64); M = alpha.shape[0]
    stay = 0.5 * (1 + r); amplitude, sigma = game.amplitude, game.sigma
    probability = np.full((M, 2), 0.5); q = np.zeros((M, 2)); nu = np.zeros((M, 2, N))
    for k in range(N):
        estimate = amplitude * (2 * probability - 1)                               # E[alpha^j_k | F^i_k], i = 0, 1
        nu[:, :, k] = -G[k] * q - g[k] * (alpha[:, :, k] + estimate)
        q = q + nu[:, :, k] * dt
        observed = alpha[:, ::-1, k] * dt + sigma * dWS[:, k, None]                # trader i's observation of alpha^j
        likelihood_up = np.exp(-(observed - amplitude * dt) ** 2 / (2 * sigma ** 2 * dt))
        likelihood_down = np.exp(-(observed + amplitude * dt) ** 2 / (2 * sigma ** 2 * dt))
        posterior = probability * likelihood_up / (probability * likelihood_up + (1 - probability) * likelihood_down)
        probability = stay * posterior + (1 - stay) * (1 - posterior)
    return nu


def linear_equilibrium(solver, iterations: int = 30, average_from: int = 15, reference: list = None, log=None):
    """Equilibrium among strategies linear in the own observed path: the damped Picard iteration of solver, with
    each regression replaced by least squares at each time step on (1, current observation, all past observation
    increments). Returns the validation controls averaged from average_from on."""
    game = solver.game; N = game.N

    def simulate(noise, coefficients):
        M = noise['dWS'].shape[0]; state = game.initial_state(noise)
        controls = [torch.zeros(M, N) for _ in range(2)]; features = [[None] * N for _ in range(2)]; past = [[], []]
        for k in range(N):
            u = {}
            for i in range(2):
                features[i][k] = torch.cat([torch.ones(M, 1), game.observe(i, state, noise, k)] + past[i], 1)
                u[i] = features[i][k] @ coefficients[i][k] if coefficients else torch.zeros(M)
                controls[i][:, k] = u[i]
            game.step(state, noise, u, k)
            for i in range(2):
                past[i].append(game.increment(i, state, noise, k))
        return controls, features

    coefficients = None; average = None; count = 0
    for iteration in range(iterations):
        controls, features = simulate(solver.noise, coefficients)
        targets = solver.targets(solver.noise, controls)
        coefficients = [[torch.linalg.lstsq(features[i][k].cpu().double(),
                                            ((1 - solver.damping) * controls[i][:, k] + solver.damping * targets[i][:, k]).cpu().double()[:, None]
                                            ).solution[:, 0].float().to(controls[i].device) for k in range(N)] for i in range(2)]
        validation, _ = simulate(solver.validation, coefficients)
        if iteration >= average_from:
            count += 1
            average = [v.clone() for v in validation] if average is None else [a + (v - a) / count for a, v in zip(average, validation)]
        if reference is not None and log:
            current = average if average is not None else validation
            log(f'linear iteration {iteration:2d}  error ' + ' '.join(f'{relative_l2(c, r):.3f}' for c, r in zip(current, reference)))
    return average if average is not None else validation
