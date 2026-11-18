# Overnight report, 2026-11-17 (new examples for the asymmetric-information Nash solver)

Scope: Bilal asked for new example problems, not more Wu-Jaimungal compute. Four candidates were examined. Two were implemented and run against exact references, two were parked with reasons. All files are in the session scratchpad (nothing copied to the project folder).

## Summary table

| Problem | Information structure | Fits the method? | Exact reference | Result (rel L2 of controls vs reference) |
|---|---|---|---|---|
| Bergault-Sanchez-Betancourt 2401.05257 (broker + mean field of informed traders) | traders have private signals; broker sees aggregate flow | yes (FOCs are E[bracket \| own info]) | 6 Riccati + linear ODEs, verified against paper values | broker 5%, average flow 7%, individual trader 5% (iterate-averaged, population 64) |
| Firoozi-Caines 1810.04369 (major-minor LQG MFG, all agents partially observed), scalar instance | non-nested: each player sees only its own noisy observation path | yes | discrete-time LQG (Kalman + Riccati), FOC residual 1-2% | major 6.3%, minor 5.1% (iterate-averaged, 16k paths) |
| Aqsha-Drissi-Sanchez-Betancourt 2412.20847 | trader filters broker speed, broker filters signal | no: not a Nash fixed point | - | parked |
| Chen-Choi-Larsen-Seppi 2105.13401 | non-nested private targets | no: market clearing, no backward equation | - | parked |

## 1. Engine

`npg.py`: generic N-player Picard-regression solver. A model supplies dynamics, each player's observation increments and current observables, and each player's pathwise best-response functional (the "bracket"). Each player's policy is a GRU over its own observation increments plus an MLP head. One iteration = simulate under current policies, evaluate brackets, damp, regress each policy on its own information. Tonight's extension: observation sequences may depend on the endogenous state (needed when a player observes the other's state through noise), so the GRUs now run inside the simulation loop. Sanity: `npg_wj.py` reproduces the Wu-Jaimungal prototype at h=0.

## 2. Bergault-Sanchez-Betancourt (`bsb_game.py`)

Model: one broker, a continuum of informed traders of one type; trader i sees the common signal alpha, its private signal alpha^i, its inventory, the mean-field inventory and the broker inventory; the broker sees alpha and the two aggregate inventories. Both FOCs (Thms 3.4, 3.7) are of the form control = (1/2 eta) E[pathwise bracket | own information]. The mean field is approximated by a finite population (32 or 64 traders per path).

Reference: closed-form ODEs (g, h for the mean field and broker; f's for the trader, eqs. 3.18-3.22). My ODE solver matches the paper's quoted values (ga(0) = 63.2, hb(0.95) = -19.4) and the identities fa = ga, fc = gc, fb + fbI = gb hold to 2e-3.

Parameters: the paper's set is stiff (gb -> -1000 near T, needs ~1e4 time steps), so a "gentle" set was used (T = 1, k_alpha = k_I = 1, sigma_alpha = 1, sigma_I = 0.5, eta_I = eta_B = 0.1, b = 0.05, a = 1, phi = 0.1), N = 100 steps.

Runs:
- bsb_a: 2048 paths, population 32, 600 fit steps, batch 256, 14 iterations: broker 7.7%, average flow 12.4%, trader 9.4%.
- bsb_b: 4096 paths, population 32, 1500 steps, batch 512: broker 5-7%, flow 10-12%, trader 5-9%, swinging +-5% between iterations.
- bsb_c: 4096 paths, population 64, batch 1024, 20 iterations: single iterations 5-8% / 6-9% / 4-7%; controls averaged over iterations 6-19: broker 5.3%, average flow 6.7%, trader 4.2% (stable from iteration 12 on).

Reading: the regression loss stays at ~0.4 in control units while the controls are ~0.4 in size, so the pathwise brackets have noise comparable to the signal; iteration swings are fit noise, not method failure, and averaging removes most of it. Remaining ~5% is a mix of fit noise and the finite-population approximation of the mean field. Untested: the paper's stiff parameters (needs a finer grid), heterogeneous trader types.

## 3. Firoozi-Caines (`fc_game.py`)

Model (scalar instance, the subagent's "Instance 1", where the paper's solution is exact): major agent tracks a target and observes its own state through noise; minor agent tracks H1 x_0 + eta and observes (x_i, x_0) through noise. The major's control is a feedback on its own filter x_hat_0, so the minor's forecast of x_0 depends on the major's estimate, which the minor cannot see. Information is non-nested. Cost: discounted quadratic tracking with control penalty, horizon T = 5, N = 100.

Reference: exact discrete-time LQG of the Euler-discretised system (discrete Kalman filters + discrete discounted Riccati, major first, then minor on the extended state (x_i, x_0, x_hat_0) with the major's filter recursion as part of the dynamics, as in the paper's Prop. 2). Check: regressing the pathwise bracket on the filter state reproduces the reference control to 1.7-2.4% with 32k paths (statistical floor). Lesson from tonight: a continuous-time Riccati reference on a dt = 0.05 grid was 7-16% off the discrete optimum (Euler Kalman gains ~11 with K dt = 0.55), so the reference was rebuilt in discrete time.

Caveat on the instance: the coefficient of the "estimate of the major's estimate" in the minor's control is small (0.03 vs 0.48 on its own state), and regressing the bracket on (x_hat_i, x_hat_0) alone does as well as the full filter state. So this instance tests the solver on a non-nested partially observed game, but it does not exercise the paper's estimates-of-estimates feature numerically. A two-way coupled instance (H_0 != 0, major tracks the mean field) would, but there the paper's closed form is heuristic (subagent flags: the claim E[mean estimation error | F_i] = 0 fails when the major's state noise is nonzero), so the solver would have no exact reference, which is also the point where it would go beyond the paper.

Run fc_a: 8192 paths, 1500 fit steps, batch 1024, 12 iterations, Q0 = 3, A0 = -1, noise D0 = D = 0.5, major observation noise 1.0, minor observation noise (0.3 on x_i, 0.05 on x_0):
- minor 6-8% from iteration 4 on (correlation with the exact control 0.99, slope 0.97);
- major reaches 10% at iteration 5, then drifts to 34% by iteration 11. Diagnosis from the saved paths: the mean control is right at every time, the spread is 1.7x too large mid-horizon (sd 0.09 vs 0.05). The major's bracket has about 12x the control's spread, the GRU fits that noise, and the noise feeds back through the state into the next iteration. The minor's bracket has better signal-to-noise and does not show this.
- fc_b (launched 06:30): 16384 paths, 1000 fit steps, batch 2048, damping 0.5, controls averaged from iteration 4: stable; single iterations major 6.6-8.2%, minor 4.8-7.1% (iterations 4-9); averaged controls major 6.3%, minor 5.1%. The variance blow-up is gone.

## 4. Parked

Aqsha-Drissi-Sanchez-Betancourt 2412.20847: the informed trader assumes the broker's speed is an exogenous OU process and best-responds; the broker then takes the trader's feedback form, substitutes her own filter and her own control (a reaction term), and solves an HJB. One pass, no fixed point; the paper calls it Stackelberg-type and "strategic to first order". Each player's problem on its own is a single-agent partial-information LQ problem (same class as Chapter 1), so there is no Nash equilibrium to test a game solver against. A Nash iteration under the true law gives a different pair (at the paper's parameters the difference is O(f_2) ~ 3%). Side finding from the full read: the paper's ODEs (A.8)-(A.10) carry g_2/(2b) where the HJB gives g_2/b; check their GitHub code before using those ODEs as a reference.

Chen-Choi-Larsen-Seppi 2105.13401: equilibrium is a market-clearing price with linear conjectures and a pointwise FOC; no backward equation. Would need a clearing step inside the loop.

## 5. What this says about the method

- Three games now solved by the same code path (WJ, Bergault-SB, Firoozi-Caines): nested (WJ), mean-field with private signals (BSB), non-nested two-player (FC). Accuracy 3-7% against exact references at moderate budgets, limited by regression noise of pathwise brackets rather than by the representation.
- Common bottleneck: the pathwise bracket is a noisy target (variance grows with the cost weight and horizon). Iterate averaging helps; a control-variate or a BSDE-style target (regress Y, then differentiate) would help more. Worth a line in any write-up.
- Open: an example where asymmetric information changes the equilibrium qualitatively and the solver has no closed form to lean on (two-way coupled Firoozi-Caines; strong-coupling WJ). Those are the cases where the method would be the only available tool.

Figure `wj/overnight_examples.png`: learned (colour) vs exact (dashed black) controls on one validation path for each problem (bsb_c, fc_b).
