# partial-info-games

Deep Picard regression for Nash equilibria of stochastic differential games in which the players observe different information. Each player's best response is a conditional expectation of a pathwise functional given that player's own observations. The solver simulates the game under the current controls, fits one recurrent network per player to that player's functional by minimizing a mean squared error with the player's own observation path as the only input, and iterates with damping. No filter is computed. See `notes/method.tex`.

## Layout

- `src/npg.py` engine: `Policy` (GRU over observation increments + feedforward head), `Solver` (simulate, evaluate functionals, damped regression, iterate averaging), optional martingale control variate (`CV=1`), `exploitability` check.
- `games/` one script per example, each with its exact reference where one exists:
  - `wj_game.py` Wu and Jaimungal (2024), broker and informed trader, nested information. Standalone prototype; `npg_wj.py` is the same game on the engine.
  - `bsb_game.py` Bergault and Sanchez-Betancourt (2024), broker and a mean field of traders with private signals. Standalone.
  - `fc_game.py` Firoozi and Caines (2020), major and minor agents, both partially observed, non-nested, scalar instance with an exact discrete-time LQG reference.
  - `pt_game.py` two traders with private signals and a common price with impact, two-way private information, no closed form; exact limits at `LAM=0` and `INFO=full`.
- `scripts/` run queues used so far.
- `notes/` method note (`method.tex`) and the results report (`overnight_report.md`).
- `results/` logs and saved paths (`npz`) of the runs reported in the notes.

## Running

From `games/`:

    LAM=2 CV=1 M_PATHS=8192 STEPS=1000 BATCH=1024 AVG_FROM=4 EXPLOIT=1 python3 pt_game.py 12
    Q0=3 A0=-1 NOISE=asym SV2=0.05 CV=1 M_PATHS=16384 STEPS=1000 BATCH=2048 RHO_DAMP=0.5 python3 fc_game.py 10
    PARAMS=gentle M_PATHS=4096 M_POP=64 FIT_STEPS=1500 BATCH=1024 python3 bsb_game.py 20
    BROKER=gru M_PATHS=8192 FIT_STEPS=800 python3 wj_game.py 0 0.3 10      # args: h, T, iterations

Each script prints, per iteration, the regression loss, the change between iterates, and the relative L2 error of each control against the reference when one exists. Environment variables are documented in each script's docstring. Outputs go to `results/`.

Requirements: Python 3, numpy, scipy, torch. CPU is sufficient; runs parallelise across experiments, so cores matter more than a GPU.

## Status

See `notes/overnight_report.md` for the results table. Variable names in the game scripts are still the prototype's short names; renaming to the repository's descriptive-name convention is pending.
