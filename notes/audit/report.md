# Audit of partial-info-games (2026-11-19, commit 31ffd60)

Scope: every file in `src/` and `games/`, the method note, `results/summary.md`, the logs, and the saved controls in the data folder. Each game was checked against its paper by an independent reviewer; the engine, the two-trader game and its exact-class benchmark were checked directly. Every number below can be rerun from `audit/checks/`; the per-game notes with file and line references are in `audit/notes/`.

## 1. Verdict

The code is correct in everything checked. The algorithm computes what it claims, with one qualification that the paper must state: it computes the open-loop Nash equilibrium (controls as adapted processes; a deviation by one player leaves the other players' control processes unchanged). This is the equilibrium that Wu and Jaimungal's FBSDE characterizes, so it is the right object, but the exact-class benchmark and two of the repository's checks use the closed-loop concept, and the difference is measurable.

The project is a legitimate contribution. One conclusion in the summary is wrong in the method's favour: the lam = 2 case is not a failure (Section 5). What the project can claim today is otherwise narrower than the summary suggests, for the reasons in Sections 4 and 6.

## 2. What was verified

### Engine (`src/npg.py`)

- Information. Each policy reads only its own observations. The GRU memory at step k contains increments 0..k-1, both in the rollout and in the regression (`npg.py:24-26, 58-65`). No lookahead in any game.
- Targets and damping. The target is (1 - rho) u + rho Lambda, and the warm start is the previous network; this is the damped Picard iteration of the note.
- Control variate. `sum_{j >= k} gamma^{j-k} zeta_j . dW_j` with zeta_j measurable at t_j and dW_j the increment over [t_j, t_{j+1}], so its conditional mean given any player's information is zero; the fixed point is unchanged. The earlier transposed-kernel bug is fixed.
- Iterate averaging averages the validation controls, not the parameters; the reported errors are of that average.

### Two-trader game (`games/pt_game.py`, `games/pt_kernel.py`)

- First-order condition. The bracket equals the gradient of the discrete cost with the other trader's control held fixed, to 1e-8 by automatic differentiation, at lam = 0 and lam = 1 (`checks/check_pt_bracket.py`). This includes the lam q_i(t) term that was missing before 2026-11-18.
- Exact-class benchmark. The shock loadings reproduce the simulator step for step and the cost is the exact expectation. The best response is closed-loop: the other trader's control is a fixed rule of its observed path, so it reacts through the price.
- Convexity. The Hessian of each trader's cost over its 2550 linear-strategy coefficients is positive semidefinite at lam = 1 and lam = 2, for both equilibrium concepts (minimum eigenvalue 0 to rounding; the null directions are the coefficients on S(t_0) = 0) (`checks/pt_convexity.py`, `checks/convexity.log`). For the open-loop cost, which is a quadratic functional with a deterministic kernel, positive semidefiniteness on the linear class implies it on all admissible controls. Hence the first-order condition is sufficient and the fixed points are equilibria. The summary states this result but no script in the repository computes it; `checks/pt_convexity.py` now does.
- Open-loop against closed-loop at lam = 1 (`checks/pt_kernel_openloop.py`, `checks/kernel_open_closed.log`): the two equilibria differ by 1.92% in relative L2; mean |nu| is 0.1738 (open) and 0.1766 (closed). The three corrected seeds at 16k paths are 4.4-6.1% from the open-loop solution and 4.8-6.1% from the closed-loop one; their average is 3.0% / 3.1% from the open-loop solution (3.5% / 3.7% from the closed-loop one), with correlation 0.9996 and slope 0.998; their mean |nu| is 0.1736-0.1747. The learned method computes the open-loop equilibrium, as the summary says. The error is concentrated at the end of the horizon (2.3% in the first fifth, 7.3% in the last), where the controls are small.

### Markov-chain game (`games/pm_game.py`)

- Chain simulation exact (P_stay = expm(C dt)); the discrete Bayes filter reproduces a brute-force enumeration to 6 decimals and is calibrated (slope 0.997-1.001); the backward recursion agrees to 6 digits with a direct minimisation; the bracket is the same open-loop condition as in the Gaussian game.
- The cross-check against Casgrain and Jaimungal (eqs 5.10a-b, beta = 0) reproduces the summary's numbers (0.2%; 5.3% / 1.4% / 0.3% at N = 50 / 200 / 800), but only by rebuilding it: commit 397e4e1 changed the summary and nothing else. The remark about the sign of beta/2 is right.

### Firoozi-Caines (`games/fc_game.py`)

- Correct throughout. The discrete-time reference reproduces a brute-force LQ optimum to 1e-15; Kalman timing matches the information the GRU has; the brackets match autograd to 2e-6. The exactness claim holds because the major's cost and dynamics do not depend on the minor at all, so the major has a dominant strategy and the minor faces a single-agent problem; open-loop and closed-loop coincide here.
- Wording: the code solves the finite-horizon, Euler-discretised version with time-varying gains. It tests the paper's construction, not the paper's stationary solution.

### Bergault-Sanchez-Betancourt (`games/bsb_game.py`)

- Correct: dynamics and information sets are the mean-field model's; both brackets match Theorems 3.4 and 3.7 term by term (the broker's own-impact term b Qbar^B nu^B is what produces (b - 2 a_B) Qbar^B_T, and the code has it); all eleven ODE right-hand sides and terminal conditions match the paper, identities to 2e-6; open-loop, matching the paper.
- The "average flow" error column is mostly the finite-population approximation of the mean field: its floor is 0.48 / sqrt(population), i.e. 8.5%, 6.1% and 4.3% at 32, 64 and 128 traders. The broker and trader columns (3.9%, 3.1-3.7% at 128) are genuine learning error.

### Wu-Jaimungal (`games/wj_game.py`, `games/npg_wj.py`)

- Model and first-order conditions match the paper (Props 3.8 and 3.9; the own-impact term is absorbed by the identity h Q^B_t = h Q^B_T - h int_t^T (nu - eta)). Open-loop, matching the paper's Definition 2.3. No lookahead.
- The h = 0 reference is the continuous-time solution stepped by Euler. Against the exact discrete-time optimum of the simulated game it is 5.1% (trader) and 7.3% (broker) off at N = 40, halving at N = 80 (`checks/wj_discrete_check.py`). The solver's bracket uses the same left-point convention (sums over j >= k), so solver and reference are mutually consistent to 0.6% / 1.7%, and the logged 5-10% errors are regression error, not reference bias. The 6.7% discrepancy in `wj_refcheck.py` is a float32 least-squares artefact (0.6% / 1.7% in float64).
- `npg_wj.py:34`: the broker's cost uses h * eta where the paper has h * nu. It affects only the cost-based checks for the broker at h > 0, which were not run; training is unaffected. In `npg_wj.py` the trader has no memory and no access to the broker's filter, which its conditional expectation needs at h > 0; the published h-sweep used the prototype, which passes it.

## 3. Conceptual points the paper must address

1. Equilibrium concept. State that the method computes the open-loop Nash equilibrium, as in Wu and Jaimungal, Casgrain and Jaimungal and the FBSDE literature. The exact-class solver as written computes the closed-loop one; the two differ by 1.9% at lam = 1. Report the learned errors against the open-loop solution (`checks/pt_kernel_openloop.py` computes it), or add the opponents' reaction to the target by differentiating through their networks, and say which.
2. The cost-based checks test the wrong condition at lam > 0. `exploitability` and `gateaux_check` perturb player i while the other players' networks react through the price, so they measure the closed-loop derivative. At the exact open-loop equilibrium this is nonzero, by the reaction term. The lam = 2 "stationarity test" (9.5% gain from an exact best response to the learned opponent) used the closed-loop best response, and Section 5 shows that this, not the method, produced the apparent failure. Keep the FOC residual and the exact references as the accuracy measures; use the cost-based checks only as a safeguard, which is what the summary already concludes.
3. Discretisation conventions differ between games. The two-trader, Markov-chain and Firoozi-Caines games use the exact first-order condition of the discrete game (sums over j > k, exact discrete references). Wu-Jaimungal and Bergault-Sanchez-Betancourt use the left Riemann sum of the continuous condition (j >= k) against continuous-time references. Each is internally consistent. For the paper, define the discrete game once, use its exact first-order condition everywhere, and rebuild the Wu-Jaimungal reference in discrete time as was done for Firoozi-Caines.

## 4. Reporting corrections

- Bergault-Sanchez-Betancourt: the paper quotes no numerical values, so "matches the paper's quoted values (ga(0) = 63.2, hb(0.95) = -19.4)" cannot be right; the values are consistent with the figure ranges, which is all one can say. `bsb_paper_N1000` diverged (relative error above 100), while the report says the stiff parameters were "untested". `bsb_game.py` is standalone and has no GRU; the report's sections 1 and 5 say the game runs on the engine.
- Markov-chain game: at 8k paths and lam = 0 the last iterate drifts from 11% to 26% over iterations 4-11, which the Polyak average hides. At lam = 1 the seeds agree only to 17-21% and the per-iteration change stays at 0.02-0.03 (about 10% of |nu|); "converges" is generous. The optimality-check ratios 0.1-0.6 do not support "within 5-12% of optimal" (point 2 above).
- Wu-Jaimungal: `wj_analyse.py` calls the full-information linear Picard an exact reference; it is the same bracket under a linear ansatz, a consistency check.
- Firoozi-Caines: "the paper's solution is exact" should read "the paper's construction".
- Two-trader game at lam = 2: the "range limit", the stationarity test and the "private traders trade 80% more" statement are withdrawn (Section 5). The continuity-in-lam table compares learned open-loop solutions with the exact lam = 0 solution and is unaffected.
- Reproducibility: the Hessian check, the lam = 2 stationarity test, the seeds-against-kernel comparisons, the Casgrain-Jaimungal cross-check and the open-loop/closed-loop comparison exist only as numbers in the summary. Each needs a script in the repository; `audit/checks/` supplies four of them.

## 5. The lam = 2 case: the summary's conclusion is wrong

The summary reads the lam = 2 runs as a failure of the method ("the learned iteration stops short of the equilibrium", "a range limit"). That reading rests on comparisons with the closed-loop equilibrium. With the open-loop equilibrium computed (`checks/pt_kernel_openloop_lam2.py`, `checks/kernel_open_closed_lam2.log`):

- The two equilibria differ by 149% in relative L2 at lam = 2. Mean |nu| is 0.358 (open-loop) and 0.792 (closed-loop). At this parameter the own-impact revenue cancels the terminal penalty (psi - lam/2 = 0), and the reaction channel, the other trader's response through the price, more than doubles trading.
- The three corrected learned seeds are 4.5-6.9% from the open-loop equilibrium (trader 1: 5.6%, 5.8%, 4.5%; trader 2: 5.6%, 6.9%, 4.2%), with mean |nu| 0.351-0.362 against 0.358, and 62-63% from the closed-loop one, which is the figure the summary reports. The open-loop cost at equilibrium, -0.156, is the summary's "learned-pair cost -0.154"; the "kernel-pair cost -0.356" is the closed-loop equilibrium's.
- The open-loop best-response iteration converged in 30 damped iterations; the closed-loop one needed 96, and that count was read as the map being nearly non-contractive.

So at lam = 2 the method works as well as at lam = 1, and the stationarity test, the 9.5% gain and the "working range set by the gain of the best-response map" paragraph should be withdrawn. The economic statement for lam = 2 also reverses: it was the closed-loop equilibrium in which private traders trade more than full-information ones, and the learned solution never computed that equilibrium. The result worth reporting is the size of the reaction channel itself: at this parameter, letting the opponent respond to one's orders through the price more than doubles equilibrium trading. This is Babichenko's belief-manipulation effect, computed exactly.

Which concept to adopt is a modelling decision. The open-loop equilibrium is the one the FBSDE literature characterizes (Wu and Jaimungal, Definition 2.3) and the one the learned method computes without change. The closed-loop one is what the exact-class solver computes, and what a trader who anticipates the other's reaction would play. The paper should state the choice, note that both exist and can differ by a large factor, and compare like with like.

## 6. Assessment of the contribution

Combined with the literature review of 2026-11-18, the position is this.

- The problem class is open by the published statements of Casgrain-Jaimungal, Bergault-Sanchez-Betancourt, Firoozi-Caines and Guan-Shames-Summers, and the architecture (damped Picard, regression on a sub-filtration) is in print for the common-noise mean field game in Antunes-Saporito-Jaimungal. The finite-player, non-nested, filter-free version is new.
- In linear-Gaussian games the exact-class solver settles the equilibrium in seconds, so the learned method's claim there is validation only. The validation is now sound: 3-7% against the open-loop equilibrium at lam = 1 and lam = 2, converging with the number of paths, with the error concentrated where the controls are small. The exact solver also gives a result of its own: the size of the reaction channel, which more than doubles equilibrium trading at lam = 2.
- The method's own claim therefore rests on the Markov-chain game, where no exact method exists. There the evidence is 6% at lam = 0 with 32k paths, and at lam = 1 only internal checks with a 17-21% spread between seeds. That is not yet enough to carry the paper. The obvious fixes are the ones the summary names: a control variate for the chain jumps (compensated jump martingale), and a larger budget.
- Everything is scalar, two-player and finite-horizon; there is no convergence statement. The literature review ranks a non-LQ example first, a small-horizon contraction second, and more players third.

## 7. Recommended order of work

1. Decide the equilibrium concept and make every check consistent with it (open-loop: compare with the open-loop kernel, which `checks/pt_kernel_openloop.py` computes; closed-loop: add the reaction term to the target by differentiating through the opponents' networks). Rewrite the lam = 2 section of the summary accordingly.
2. Commit scripts for every number in the summary.
3. Markov-chain game: jump control variate, 32k paths and three seeds at lam = 1; report the FOC residual and the seed spread as the accuracy statement.
4. Rebuild the Wu-Jaimungal reference in discrete time; fix `npg_wj.py:34`.
5. Fix the reporting items in Section 4, and state the average-flow floor next to the Bergault-Sanchez-Betancourt numbers.
6. Then the write-up, with the method presented as validated in LQG against the exact-class solver and applied beyond it.
