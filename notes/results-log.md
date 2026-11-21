Results of the code before the rewrite into src/partial_info_games (last commit f2be1f7, scripts in games/ and scripts/). The rewrite
reproduces that code exactly on the same seeds; the study is rerun with experiments/runs.py.

# Batch summary

## Corrections after the audit of 2026-11-19 (audit/REPORT.md); these supersede the statements below

- Equilibrium concept. The learned method computes the open-loop Nash equilibrium (other players' orders held fixed as processes), as in Wu and Jaimungal. The exact-class solver `games/pt_kernel.py` computes the closed-loop one (other players' rules held fixed, so their orders react through the price); `audit/checks/pt_kernel_openloop.py` computes the open-loop one. The two differ by 1.9% at lam = 1 and by 149% at lam = 2 (mean |nu| 0.358 open-loop vs 0.792 closed-loop).
- lam = 2 is not a failure of the method. The three corrected learned seeds are 4.5-6.9% from the open-loop equilibrium (mean |nu| 0.351-0.362 vs 0.358) and 62-63% from the closed-loop one. The "range limit", the lam = 2 stationarity test (9.5% gain, which used the closed-loop best response) and "private traders trade 80% more at lam = 2" are withdrawn. What the exact solver shows at lam = 2 is the size of the reaction channel: letting the opponent respond through the price more than doubles equilibrium trading.
- Headline at lam = 1: the learned three-seed average is 3.0 / 3.1% from the open-loop equilibrium (3.5 / 3.7% from the closed-loop one).
- Cost-based checks (exploitability, model-free optimality check) perturb one player while the others' networks react through the price, so at lam > 0 they test the closed-loop condition and are not zero at the open-loop equilibrium. Accuracy measures: the exact references and the FOC residual only.
- Markov-chain game: at 8k paths and lam = 0 the last iterate drifts from 11% to 26% over iterations 4-11, hidden by the averaging; at lam = 1 the seeds agree only to 17-21% and the per-iteration change stays at about 10% of |nu|, so "converges" is too strong; the optimality-check ratios do not support "within 5-12% of optimal".
- Bergault-Sanchez-Betancourt: the paper quotes no numerical values, so the reference is "consistent with the paper's figure ranges", not "matches quoted values". Most of the average-flow error is the finite-population floor. The run at the paper's stiff parameters (bsb_paper_N1000) diverged; it was not "untested". bsb_game.py is a standalone script, not on the engine.
- Wu-Jaimungal: the h = 0 reference is continuous-time and 5-7% from the discrete optimum; the full-information linear Picard is a consistency check under a linear ansatz, not an exact reference.
- Firoozi-Caines: "the paper's solution is exact" should read "the paper's construction is exact in this instance".
- Reproducibility: the Hessian check, the stationarity test, the seeds-vs-kernel comparisons, the Casgrain-Jaimungal cross-check and the open/closed-loop comparison lacked scripts; audit/checks/ supplies four of them, the rest are to be added.

## Scaling at lam = 0 (exact reference, recomputed after the reference fix of 2026-11-18), control variate on, 128 epochs per iteration

| paths | trader 1 | trader 2 | FOC residual 1 | FOC residual 2 |
|---|---|---|---|---|
| 2048 | 0.078 | 0.078 | 0.1321 | 0.1222 |
| 4096 | 0.064 | 0.062 | 0.0871 | 0.0935 |
| 8192 | 0.055 | 0.050 | 0.0828 | 0.0987 |
| 16384 | 0.046 | 0.045 | 0.1123 | 0.0753 |
| 32768 | 0.037 | 0.032 | 0.0437 | 0.0574 |

Figure: results/scaling.png


## Private two-way case, lam = 2, control variate on [pre-fix runs with the wrong first-order condition; superseded]

| seed | final change | FOC residual 1 | FOC residual 2 | exploitability 1 | exploitability 2 |
|---|---|---|---|---|---|
| 1 | 0.0092 | 0.0760 | 0.0629 | -1.25% | -1.01% |
| 2 | 0.0098 | 0.0713 | 0.0646 | -2.51% | +0.98% |
| 3 | 0.0104 | 0.0599 | 0.0696 | -1.61% | +0.32% |

Seed agreement (rel L2 between seeds on the common validation paths): s1-s2: trader1 0.062, trader2 0.063, s1-s3: trader1 0.073, trader2 0.075, s2-s3: trader1 0.085, trader2 0.081

Trading intensity, mean |nu| over paths and time: private 0.167, full information 0.199 (exact full-info 0.197); rel L2 between private and full-information controls on the same paths 0.638

## Continuity in the impact parameter (private case vs the exact lam = 0 solution on the same validation paths)

| lam | rel L2 of trader 1 to the exact lam=0 control | mean abs control |
|---|---|---|
| 0 (learned, 16k) | 0.046 | 0.148 (exact 0.146) |
| 0.25 | 0.089 | 0.154 |
| 0.5 | 0.149 | 0.159 |
| 2 | 0.459 | 0.167 |

## Other runs (final averaged relative L2 vs exact, from the logs)

- bsb_gentle_pop128: broker 0.039, avg flow 0.049, trader_i 0.031
- bsb_paper_N1000: broker 112.013, avg flow 161.644, trader_i 144.873
- fc_cv_N200: u0 0.062, u1 0.044
- fc_high: u0 0.035, u1 0.036
    checks: FOC0 0.1118 FOC1 0.0574 
- fc_q10: u0 23.964, u1 1.453
    checks: FOC0 1.7588 FOC1 4.8652 
- pt_full2_cv: u0 0.041, u1 0.040
- wj_T1_h1_full: eta 0.098, nu 0.065
- wj_T1_h1_gru: it 14 loss 2.37e-02 change 0.0188 | BR residual (pathwise) trader 0.679 broker 0.981 (2578s)
- wj_h0_cv: u0 0.052, u1 0.071
- wj_strong_h0_gru: eta 0.739, nu 3.456
- wj_strong_h1_full: eta 0.089, nu 0.073
- wj_strong_h1_gru: it 14 loss 2.71e-01 change 0.0968 | BR residual (pathwise) trader 0.473 broker 1.009 (5284s)

## Follow-ups (laptop, 2026-11-18 evening)

- Bergault-SB reference check (regress bracket on the exact features along closed-form paths, 16k paths, population 32): broker 1.9%, trader 3.2%. Reference consistent.
- Seed-average test, lam = 0, 8k paths, CV: single-seed errors trader 1 [5.2, 4.7, 4.7]%, trader 2 [5.8, 5.4, 5.3]%; three-seed averaged control 4.2% / 4.8%; pure variance would give ~2.8%. A finite-sample bias of ~4% at 8k, which the 32k single-seed result (3.7% / 3.2%) shows is not a fixed floor.
- FOC residual vs true error under strong coupling, full information lam = 2, 8k paths, CV: residual 1.7% / 1.9%, true error 4.0% / 4.2% (factor ~2.3). The residual underestimates the error at strong coupling; at lam = 0 the factor was ~1.5. Applied to the private lam = 2 case, residuals of 6-8% suggest a true error of 10-15%.
- Open: the full-information lam = 2 run has negligible regression noise (loss 5e-5) yet a 4% systematic gap to the linear-Picard reference.
- Strong-coupling Wu-Jaimungal (a = b = 0.1), damping 0.2 and 0.1, 40 iterations, GPU (batch 3): trader converges to 7.3% vs exact at h = 0 (both dampings); broker's error drifts from 20% (iteration 9) to 70-80% (iteration 39) with its best-response residual at 1.0 throughout, i.e. the broker's regression never fits its target. Full-information broker (linear regression) reached 7.3%, so this is a regression failure at high gain (target ten times larger relative to the control than at a = 1; standalone prototype has no control variate and a fixed output scale), not an iteration failure. To try: engine version with control variate and rescaled outputs.

## Corrected two-trader runs on the GPU (batch 4, 16k paths, 1500 steps, control variate), vs the exact-class kernel solution, 2026-11-19

| run | lam | trader 1 | trader 2 | mean |nu| learned / kernel |
|---|---|---|---|---|
| private | 0.25 | 4.4% | 4.7% | 0.151 / 0.151 |
| private | 0.5 | 4.6% | 4.7% | 0.157 / 0.157 |
| private (headline) | 1 | 5.2% | 4.3% | 0.174 / 0.177 |
| full information | 2 | 3.2% | 3.3% | 0.434 / 0.442 |
| private, seeds 1-3 | 2 | 63% each vs the closed-loop solution; 4.5-6.9% vs the open-loop one (see corrections) | | 0.36 / 0.79 (closed-loop); 0.358 (open-loop) |

The private-information equilibrium is computed to 4-5% at lam <= 1 against an independent solver; the 4% shrinkage of the controls seen at 8k paths is 1.7% at 16k. The full-information gap of the earlier runs was the first-order-condition bug. At the edge case lam = 2 the learned seeds agree with each other but not with the kernel branch (test of stationarity below).
- [WITHDRAWN, see corrections: compared against the closed-loop equilibrium] Stationarity test at lam = 2 (private): the learned strategy, fitted as a linear map of own observations (5% fit error), is NOT an equilibrium under the exact cost: each trader gains 9.4-9.7% of its cost by an exact best response to the learned opponent (kernel-pair cost -0.356 vs learned-pair -0.154). So at lam = 2 the learned iteration (12 iterations, damping 0.6) stops short of the equilibrium rather than finding a second one; the kernel iteration itself needed 96 damped iterations there against 18 at lam = 1, i.e. the best-response map is close to non-contractive. Same failure type as the strong-coupling Wu-Jaimungal broker: the method's working range is set by the gain of the best-response map, and the edge cases need continuation or many more damped iterations.

## Non-Gaussian two-trader game (private Markov-chain signals, games/pm_game.py), 2026-11-19

- Signals: independent two-state chains (+-1, switching rate 1), private; price and costs as in the Gaussian game. The conditional expectation of the other trader's signal is the (nonlinear) Wonham filter, so the linear-strategy exact solver does not apply.
- Exact lam = 0 reference: exact discrete Bayes filter plus a backward recursion for the single-trader problem with an exogenous adapted signal. Checks: filter slope 1.00; regress-on-exact-features test 1.7% / 1.7% (statistical floor); agreement with Casgrain and Jaimungal's closed form (arXiv 1806.04472, eq. 5.12 with the sign of beta/2 corrected, two-state chain, pure diffusion): inventory gain 0.2%, signal gain 5.3% at 50 steps, 1.4% at 200, 0.3% at 800, i.e. discretisation only.
- Learned runs: batch 5 on the GPU (lam = 0 at 8k and 32k paths vs exact; private lam = 0.5 and lam = 1 with three seeds, FOC and model-free optimality checks).

## Batch 5 (GPU), 2026-11-19

Gaussian two-trader game, headline lam = 1, three seeds at 16k paths vs the exact-class solution: trader 1 5.1 / 5.1 / 5.2%, trader 2 4.8 / 6.1 / 4.9%; seeds agree within 6-7%; the three-seed average is 3.5 / 3.7% from the exact solution; mean |nu| 0.174 (seeds) vs 0.177 (exact). FOC residuals 6-7%.

Markov-chain two-trader game (non-Gaussian):
- lam = 0 vs the exact solution: 13.7 / 9.0% at 8k paths, 6.2 / 6.1% at 32k (correlation 0.998, slope 0.98, mean |nu| 0.184 vs 0.187). The error falls with budget as in the Gaussian case but from a higher level: the regression target is noisier (loss 0.12 vs 0.03 in the Gaussian game), because the control variate removes only the price noise, not the jumps of the chains.
- private, lam = 0.5 and lam = 1 (three seeds, 16k paths): the iteration converges (change 0.02-0.03 per iteration); seeds agree within 17-21%; FOC residuals 13-19%; model-free optimality check ratios 0.1-0.6 (0 = optimal, 1 = as far from optimal as a 20% rescaling), i.e. the strategies are within roughly 5-12% of optimal by that measure. Trading intensity rises with the impact parameter (mean |nu| 0.187, 0.197, 0.217 at lam = 0, 0.5, 1).
Reading: the method runs unchanged on the non-Gaussian game and is near-optimal there, with accuracy at lam = 1 of order 10% at this budget rather than 5%; a control variate for the chain jumps (compensated jump martingale) and a larger budget are the obvious improvements.

## Model-free optimality check on the three older games (laptop, 8k paths; applied to the LAST iterate, not the averaged controls)

| game | player | last-iterate error vs exact | check ratio (0 = optimal, 1 = like a 20% rescaling) |
|---|---|---|---|
| Wu-Jaimungal h = 0 | trader | 9.7% | 0.71 |
| Wu-Jaimungal h = 0 | broker | 15.0% | 1.13 |
| Firoozi-Caines | major | 8.3% | 0.05 |
| Firoozi-Caines | minor | 5.9% | 0.76 |
| two-trader lam = 1 (Gaussian) | trader 1 | ~7% (vs kernel, averaged) | 0.99 |
| two-trader lam = 1 (Gaussian) | trader 2 | ~5% | 0.31 |

Reading: with six random directions and 4k paths the check is coarse. It scales roughly with the error (a ratio of 1 corresponds to an error of order 15-20%), it would flag a gross error such as the 63% of the lam = 2 case, but it cannot resolve 5% from 15%, and it varies by a factor of several between players with similar errors. It is a safeguard against derivation mistakes, not an accuracy measure; the FOC residual and the exact references remain the accuracy measures. More directions and paths would tighten it.

## Equilibrium concept (audit, 2026-11-19)

The learned method's first-order condition holds the other players' orders fixed as processes (open-loop Nash, as in Wu-Jaimungal). The exact-class solver as written holds the other players' rules fixed, so their orders react through the price (closed-loop Nash; Babichenko's "belief manipulation"). At lam = 1 with private information the two equilibria differ by 1.9% in relative L2 (mean |nu| 0.1766 closed-loop vs 0.1738 open-loop); the learned three-seed average is 3.0 / 3.1% from the open-loop solution and 3.5 / 3.7% from the closed-loop one, consistent with the method computing the open-loop equilibrium. The reaction effect is therefore small at these parameters but should be stated; computing the closed-loop equilibrium with the learned method would require the derivative of the opponents' rules, which is available by automatic differentiation through their networks but is not implemented.

## Closed-loop targets and automatic-differentiation targets (2026-11-19)

- Closed loop (target = derivative of the realised cost with the opponent's network in the graph, CLOSED=1), 8k paths, laptop: lam = 1: 6.7 / 5.1% from the exact closed-loop solution and 6.4 / 5.5% from the open-loop one (the two equilibria differ by 1.9%, so inconclusive); lam = 2: diverged (per-iteration change 0.2 -> 1.5 -> 3.4, strategies ten times too large). The reaction term through the opponent's network slopes destabilises the iteration at strong coupling; the exact closed-loop iteration itself needed 96 damped steps there. Closed loop is future work.
- Automatic-differentiation targets are now the default in the engine (TARGET=bracket restores the hand-derived ones). Open loop: the other players' controls are fixed tensors, so the target is the exact first order condition of the discrete game. One-time check at the start of each run against the hand-derived functional: two-trader Gaussian 1.4e-7, Markov-chain 1.5e-7, Firoozi-Caines 2.1e-7; Wu-Jaimungal 4.4% at 40 steps, 2.2% at 80, 1.1% at 160, i.e. exactly the continuous-convention term of the hand-derived functional (after fixing the broker cost, h * nu, audit item). Bergault-Sanchez-Betancourt (standalone script): hand-derived target switched to the exact discrete convention (FOC=discrete, default). Reruns of both at laptop budget pending.

## Batches 6-10 (GPU, 2026-11-20/21): budgets, control variate, Markov-chain lam = 1, closed loop

Averaged controls, relative L2 error per player (player 1 / player 2), against exact solutions on the validation paths.

| case | 32k | 64k | 128k |
|---|---|---|---|
| two traders, lam = 0 (closed form) | 3.0 / 2.9% | 2.5 / 2.6% | out of GPU memory, rerun in batch 11 |
| two traders, private, lam = 1 (exact open-loop) | 2.8 / 2.9% | 2.4 / 2.6% | rerun in batch 11 |
| Firoozi-Caines | 2.2 / 1.9% | 1.7 / 1.6% | 1.1 / 1.5% |
| Wu-Jaimungal h = 0 (exact discrete), trader / broker | 4.7 / 7.1% | 3.1 / 8.8% | rerun in batch 11 |
| Markov chain, lam = 0 (exact), jump control variate | 3.5 / 4.4% | 2.8 / 2.6% | 2.4 / 2.3% |

Errors still fall with the budget except the Wu-Jaimungal broker (7-9%). Two traders at lam = 1: mean |nu| 0.1732 vs exact 0.1738, regression slope 1.00 (no shrinkage).

Control variate, 32k, off / on: two traders lam = 0 4.9/4.5% -> 3.0/2.9%; Firoozi-Caines 2.6/3.0% -> 2.2/1.9%; Wu-Jaimungal 4.9/6.8% -> 4.7/7.1% (no effect); Markov chain none 6.8/6.0%, price only 6.6/5.9%, price and jumps 3.5/4.4% (regression loss 0.117 -> 0.025). Kept.

Markov chain, private, lam = 1, three seeds at 64k with the jump control variate and averaging: seeds agree within 3.5-3.6% (16k, last iterate, no jump control variate: 18-19%); first order condition residuals 5.5-7.9%; mean |nu| 0.214-0.215.

Closed loop, two traders, private, lam = 1.5 (exact open- and closed-loop equilibria differ by 8.5%):

| run | vs exact closed loop | vs exact open loop |
|---|---|---|
| gradient play, closed, seeds 1-3 | 3.3 / 3.1 / 3.2% | 8.8 / 8.3 / 9.3% |
| gradient play, open (control) | 8.8% | 3.3% |
| gradient play, closed, from untrained networks | 5.8% | 10.3% |
| closed-loop Picard, damping 0.6 / 0.2 / 0.1 | 3.0 / 3.2 / 3.9% | 8.6 / 8.8 / 8.1% |

Both closed-loop methods find the closed-loop equilibrium at lam = 1.5 and separate it from the open-loop one. At lam = 2 (edge of convexity, psi - lam/2 = 0; exact closed and open equilibria differ by 149%) every closed-loop run fails: gradient play 48-51% from the exact solution, closed-loop Picard diverges at damping 0.2 and 0.1.

## Linear-strategy baseline (laptop, 2026-11-21)

Markov chain, lam = 0, against the exact (Wonham-filter) solution (same damped iteration and targets, controls restricted to linear functions of the own observed path, least squares per time step):

| | 32k | 128k |
|---|---|---|
| linear-strategy equilibrium | 14.2 / 14.4% | 13.4 / 13.7% |
| learned (GRU), batches 7 and 9 | 3.5 / 4.4% | 2.4 / 2.3% |

Best linear fit of the exact control on the own observed path (the current own signal, inventory and price plus every past increment of the own signal and the price), out of sample (65k paths): 11.5% for both traders. No strategy linear in the observed path gets below about 11.5%.

## Batches 11-13 (GPU, 2026-11-21)

Seeds at 64k (mean +- spread over seeds 0-2, player 1 / player 2): two traders lam = 0 2.4 +- 0.1 / 2.5 +- 0.1%; private lam = 1 2.3 +- 0.1 / 2.4 +- 0.2%; Firoozi-Caines 1.5 +- 0.2 / 1.6 +- 0.1%; Markov chain lam = 0 2.7 +- 0.2 / 2.7 +- 0.1%. At 128k: two traders lam = 0 2.7 / 2.5%, lam = 1 2.3 / 2.4%. Width 64 is worse (4.5 / 4.3% two traders, 3.7 / 4.2% Markov chain). Broker-trader broker: 7.1 -> 8.8 -> 17.5% at 32k -> 64k -> 128k (damping 0.7); damping 0.15 at 32k 5.8%.

## Changes in the rewrite

- The broker-trader game: the broker observes the price increments only (before: also the Kalman estimate of the signal and the transient impact); the trader observes (alpha, Q^I) (before: also Q^B, Y and the Kalman estimate). The broker numbers above are not comparable with the rewrite.
- The Bergault and Sanchez-Betancourt game and the strong-coupling Wu-Jaimungal runs were dropped; their scripts are at the last commit before the rewrite.
- The closed-loop solvers (gradient play, closed-loop Picard), the closed-loop references and the linear-strategy baseline were removed from the repository to keep it to the open-loop method and the benchmarks; they are at commit cc97a64.

## 2026-11-21: broker-trader horizon

- The broker-trader default horizon changed from T = 0.3, N = 40 to T = 1, N = 50, matching the other games. T = 0.3 came from the first prototype; the short-horizon restriction of Wu and Jaimungal applies only to h > 0.
- Fraction of Var(α) explained by the broker's Kalman filter (current parameters): 29% at t = 0.3, 35% at t = 0.5, 38% at t = 1 (steady state). A longer horizon makes the broker's targets only slightly less noisy.
- First order condition check of the exact reference (experiments/checks.py, 20000 paths): residual 0.029 (trader), 0.078 (broker) at T = 1, against 0.013, 0.075 at T = 0.3. Both at the sampling level of the check.
- Cluster runs of batch 2026-11-21 (stages budget, seeds, control_variate, damping) used T = 0.3.

## 2026-11-21: cluster batch with fixed training paths (T = 0.3 broker)

- 42 runs (stages budget, seeds, control_variate, damping, coupling), all with one set of training paths reused at every iteration.
- Final averaged errors (player 1 / player 2) at 25k, 50k (mean of 3 seeds), 100k paths: two traders λ = 0: 2.9/3.3, 2.8/2.8, 2.7/2.9; λ = 1: 2.8/3.1, 2.8/2.9, 2.6/2.5; Markov λ = 0: 3.7/3.6, 3.2/3.0, 2.8/2.7; major-minor: 3.1/2.4, 1.7/1.9, 1.1/1.8; broker-trader: 1.3/7.7, 0.8/7.4, 0.6/12.4 (%).
- Control variate off at 25k roughly doubles the errors (two traders λ = 1: 6.9/6.5; Markov: 7.6/7.4). Broker damping 0.3 and 0.15 does not help (8.5 to 15.5% broker).
- Two traders λ = 1.5: 2.8 to 3.5% over 3 seeds; λ = 2: 3.4, 3.9, 6.1%.
- Per-iteration histories: the error reaches its minimum early and then drifts upward (broker at 100k: 6.2% at iteration 2, 26% at iteration 15; two traders λ = 1 at 100k: 2.3% at iteration 5, 4.4/5.3% at iteration 15). Diagnosis: the warm-started networks are trained on the same paths at every iteration and fit their noise.
- Change: new training paths at every iteration (Solver.run), seed 1000 (seed + 1) + k. The study is rerun with this change; the broker-trader game now has T = 1, N = 50, σ = 0.5.
- Iterate averaging removed: Solver.run returns the controls of the last iterate, i.e. the trained networks. Averaging the network weights over the last iterates (Polyak averaging) is an option for later.
- Major-minor default horizon changed from T = 5, N = 100 to T = 1, N = 50, so that every game uses T = 1, N = 50. First order condition check of its exact reference at 20000 paths: 0.023 (major), 0.032 (minor).

## 2026-11-22: cluster batch with new paths at every iteration (last iterate reported)

- Same 42 runs; broker-trader at T = 1, N = 50 (was T = 0.3), major-minor at T = 1, N = 50 (was T = 5), so those two games are not like-for-like with the previous batch.
- Last-iterate errors (player 1 / player 2, %), 25k / 50k seeds 0-2 / 100k: two traders λ = 0: 2.3/2.1; 1.7/1.8, 2.8/3.3, 1.7/1.6; 2.3/3.2. λ = 1: 2.0/1.9; 1.3/1.5, 1.4/1.3, 2.7/2.4; 2.3/2.5. Markov λ = 0: 3.0/2.4; 2.0/1.9, 2.3/2.6, 1.9/1.9; 2.2/2.8. Major-minor: 2.0/2.0; 1.5/1.9, 2.3/2.4, 2.4/2.2; 0.9/1.0. Broker-trader: 1.0/2.6; 0.4/5.6, 0.5/1.8, 1.0/5.2; 0.2/3.4.
- No upward drift: errors fall over the first 4-6 iterations (about 10 at λ = 2) and then fluctuate around a level (two traders about 2-4%, broker about 3-7%). The final number therefore depends on which iterate is last; the seed spread reflects this.
- Errors barely fall from 25k to 100k paths in the two-trader games, so the level is not set by the number of paths alone. Candidates: optimisation noise of each regression (constant learning rate), network capacity.
- Control variate off (25k): two traders λ = 1 4.9/4.0 against 2.0/1.9; Markov 6.4/4.8 against 3.0/2.4; broker 3.1/8.0 against 1.0/2.6.
- Broker damping 0.3 / 0.15: 2.9-4.0% broker, no better than 0.7.
- Coupling (25k): λ = 1.5: 2.3/2.2, 3.9/4.1, 3.7/2.4; λ = 2: 2.1/1.9, 4.1/5.0, 4.7/4.5.
- Markov λ = 1 first order condition residuals (50k): 0.088/0.102, 0.090/0.114, 0.085/0.083. No calibration of the residual at λ = 0 yet.
