# partial-info-games

Nash equilibria of stochastic differential games in which each player observes different information, computed by
deep Picard regression. Each player's best response is the conditional expectation, given that player's own
observations, of a target obtained by differentiating the player's realised cost. The method simulates the game
under the current controls, fits one recurrent network per player to its target by minimizing a mean squared error
with the player's own observation path as the only input, and iterates with damping. No filter is computed. The
method is described in `notes/method.tex`.

## Examples

Each example has an exact solution of the discretised game, used only to validate the method, except the
Markov-chain game at lam > 0.

- **Two traders with private signals** (`pig.games.two_traders`). Each trader observes a private Ornstein-Uhlenbeck
  signal and the common price, which both traders move. No finite filter exists. The exact equilibrium, open and
  closed loop, is computed in the class of strategies linear in the observed path (`linear_strategies`), which
  contains the equilibrium in this Gaussian game.
- **Two traders with private Markov-chain signals** (`markov_traders`). The same game with two-state chains; the
  filter is nonlinear. Exact solution at lam = 0, and the equilibrium among linear strategies as a baseline.
- **Broker and informed trader** (`broker_trader`), the game of Wu and Jaimungal (2024) without transient impact.
- **Major and minor player** (`major_minor`), the game of Firoozi and Caines (2020), both players partially observed.

## Notebooks

- `two_traders.ipynb`: the exact linear-class equilibria, the learned open-loop equilibrium, and gradient play for
  the closed-loop one.
- `markov_traders.ipynb`: learned, linear-strategy and exact equilibria at lam = 0, and the game at lam = 1.
- `benchmarks.ipynb`: the broker-trader and major-minor games against their exact solutions.
- `results.ipynb`: the tables and figures of the study, read from the saved runs in `experiments/results/`.

The first three use a small budget and run on a laptop.

## Layout

```
src/pig/                the package
  solver.py             policy network, damped Picard iteration (open and closed loop), gradient play,
                        first order condition residual
  plotting.py           plot style
  games/                one module per game with its exact solution; linear_strategies.py, the exact
                        linear-class solver of the two-trader game and its best-response spectrum
notebooks/
experiments/
  run.py                one run: solve, compare with the reference, save results/<name>.json and .npz
  runs.py               the runs of each stage of the study
  references.py         the exact linear-class equilibria at lam = 1, 1.5, 2 and their spectra, in references/
  collect.py            loads the saved runs for the results notebook
  cluster/              SLURM scripts for an Alliance cluster: job array, submission
  results/              per run: a JSON file and the validation controls (not tracked)
notes/                  method note, audit, results log
```

## Usage

```python
import sys; sys.path.insert(0, 'src')
import pig
from pig.games import TwoTraders
game = TwoTraders(lam=1.0)
solver = pig.Solver(game, n_paths=16384, steps=750, batch=2048, control_variate=True)
controls, _ = solver.run(10)
```

Requires PyTorch 2.0 or later, NumPy, SciPy, pandas and matplotlib.

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

The stages are `budget`, `seeds`, `control_variate`, `width`, `damping`, `linear` and `closed_loop`. Job logs go to
`experiments/cluster/logs/`. To build the tables and figures, copy the results back and run `results.ipynb`:

```
rsync -av <cluster>:<repo>/experiments/results/ experiments/results/
```
