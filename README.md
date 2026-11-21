# partial-info-games

Nash equilibria of stochastic differential games in which each player observes different information,
computed by deep Picard regression. The method is described in `notes/method.tex` (the note).

## Overview

Player i chooses a control u^i adapted to the filtration of its own observations O^i. In the quadratic case,
the first order condition of each player is

    u^i_t = (1 / 2 a_i) E[ Lambda^i_t | F^i_t ],

where Lambda^i_t depends on the future state and on the controls of all players. The open-loop Nash
equilibrium is the fixed point of the operator Psi that maps the controls to the right-hand side for every
player. It is computed by the damped Picard iteration

    u^(k+1) = (1 - rho) u^(k) + rho Psi(u^(k)).

Each conditional expectation is approximated by a recurrent network of the player's own observed path,
fitted by minimizing a mean squared error over simulated paths. No filter is computed.

## Examples

Each example has an exact solution of the discretised game, used only to validate the method, except the
Markov-chain game at lam > 0.

- **Two traders with private signals** (`games.two_traders`). Each trader observes a private
  Ornstein-Uhlenbeck signal and the common price, which both traders move. No finite filter exists. The
  exact equilibrium at lam > 0, open and closed loop, is computed among strategies linear in the observed
  path (`LinearStrategies`), which contains the equilibrium in this Gaussian game.
- **Two traders with private Markov-chain signals** (`games.markov_traders`). The same game with two-state
  chains, whose filter is nonlinear. Exact solution at lam = 0, and the equilibrium among linear
  strategies as a baseline.
- **Broker and informed trader** (`games.broker_trader`), without transient impact.
- **Major and minor player** (`games.major_minor`), both players partially observed.

## Notebooks

- `two_traders.ipynb`: the exact linear-class equilibria, the learned open-loop equilibrium, and gradient
  play for the closed-loop one.
- `markov_traders.ipynb`: learned, linear-strategy and exact equilibria at lam = 0, and the game at lam = 1.
- `benchmarks.ipynb`: the broker-trader and major-minor games against their exact solutions.
- `results.ipynb`: the tables and figures of the study, read from the saved runs in `experiments/results/`.

The first three use a small budget and run on a laptop.

## Layout

```
src/partial_info_games/     the package
  solver.py                 the control network, the damped Picard iteration (open and closed loop),
                            gradient play, the linear-strategy baseline, the first order condition check
  plotting.py               plot style
  games/                    base.py, the game interface; one module per game with its exact solution
notebooks/
experiments/
  run.py                    one run: solve, compare with the reference, save results/<name>.json and .npz
  runs.py                   the runs of each stage of the study
  references.py             the exact linear-class equilibria of the two-trader game, in references/
  checks.py                 checks that every exact reference satisfies its first order condition
  linear_bound.py           the best linear fit of the exact Markov-chain controls
  collect.py                loads the saved runs for the results notebook
  cluster/                  SLURM scripts for an Alliance cluster: job array, submission
  results/                  per run: a JSON file and the validation controls (not tracked)
notes/                      the note, the audit, the results log
```

## Usage

```python
import sys; sys.path.insert(0, 'src')
from partial_info_games import Solver
from partial_info_games.games import TwoTraders

game = TwoTraders(lam=1.0)
solver = Solver(game, n_paths=16384, steps=750, batch=2048)
controls, _ = solver.run(10)
```

Requires PyTorch 2.0 or later, NumPy, pandas and matplotlib.

To run a stage of the study on a local machine:

```
cd experiments
python references.py                 # once: the exact references of the two-trader game
python runs.py budget --dry-run      # list the runs
python runs.py budget                # run them; finished runs are skipped
```

On an Alliance cluster (Narval), with the environment `~/envs/lfc`:

```
cd partial-info-games/experiments/cluster
./submit.sh budget 0                 # one run first, to check the time limit
./submit.sh budget                   # the whole stage as a job array, one GPU per run
```

The stages are `budget`, `seeds`, `control_variate`, `width`, `damping`, `linear` and `closed_loop`. Job
logs go to `experiments/cluster/logs/`. To build the tables and figures, copy the results back and run
`results.ipynb`:

```
rsync -av <cluster>:<repo>/experiments/results/ experiments/results/
```
