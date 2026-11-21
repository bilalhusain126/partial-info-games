# Audit of games/fc_game.py (Firoozi-Caines scalar instance)

Repository: /Users/bhusain/Desktop/Bilal Saleh Husain/Academics and Research/Repos/partial-info-games (read only).
Files: games/fc_game.py, src/npg.py, notes/overnight_report.md section 3, results/fc_gateaux.log, results/logs/fc_exploit_cv{0,1}.log.
Paper: Firoozi and Caines, arXiv 1810.04369v4 (equation numbers below are from v4).
Verification script: scratchpad/audit/fc_verify.py (independent checks (a)-(d), output quoted below).
Run settings of the Gateaux log (scripts/gateaux_chain.sh line 5): Q0=3, A0=-1, NOISE=asym (D0=D=0.5, SV0=1.0, SV=0.3), SV2=0.05, CV=1, 8192 paths, 1000 fit steps, damping 0.5, 10 iterations.

## 1. Model. Verdict: correct (one wording issue)

Code (fc_game.py:17-23, 102-117) vs paper:
- Major dynamics dx0 = (A0 x0 + B0 u0) dt + D0 dw0: paper (1). Code fc_game.py:105.
- Minor dynamics dxi = (A xi + G x0 + B ui) dt + D dwi: paper (2). Code :106.
- Major observation dy0 = L0 x0 dt + SV0 dv0: paper (5)-(6), L_0 = [l_0^1, 0] (own state only). Code :112.
- Minor observation (L1 xi, L2 x0) dt + (SV, SV2) dvi: paper (7)-(8) with p = 2, l_k^1 = [L1; 0], l_k^2 = [0; L2]. Code :113. A special case of the paper's form.
- Costs: paper (3) with H_0 = 0 gives Q0 (x0 - eta0)^2 + R0 u0^2; paper (4) with H_2 = 0 gives Q (xi - H1 x0 - eta)^2 + R ui^2; both discounted by e^{-rho t}. Code :116-117, :7.
- Initial states N(0, 1), independent: paper Assumption 1 (mean zero, independent). Code :63.

Difference from the paper, not an error: the paper is infinite horizon with stationary gains (eqs (33), (54), Assumptions 4-5 restrict to time-invariant linear feedback). The code solves the finite-horizon T = 5 version with time-varying gains (Gain_k -> 0 near T; e^{-rho T} = 0.08, so the horizon matters). The docstring's "the paper's solution is exact" should read "the paper's structure (Prop. 2 extended-state filter + separation) applied to the finite-horizon, Euler-discretised game is exact". Also the "exact reference" is exact for the discretised game, not for the continuous-time one (O(dt) difference); the comparison with the learned controls is apples to apples since the solver plays the same discrete game.

Exactness of the Nash equilibrium with one minor and H_0 = H_2 = 0: justified. The major's cost (3), dynamics (1) and observation (5) contain no x_i, so the major's problem is a single-agent partially observed LQG problem whatever the minor does (dominant strategy). Given the major's strategy, the minor's problem is a single-agent partially observed LQG problem on the extended state (x_i, x_0, xhat_0) (paper Prop. 2, eqs (43)-(45)). Both best responses are unique among all adapted controls (strict convexity), so the pair is the unique exact Nash equilibrium of the finite-population game; the infinite-population approximation plays no role since the mean field enters nowhere. The paper's Prop. 2 is stated for exactly this uncoupled case ("the simple case where ... not coupled with the mean field", p. 13). Numerical confirmation in check (c): autograd finds the major's cost does not depend on u_1 at all (tensor unused in the graph), while d(cost_1)/d(u_0) is nonzero (max 4e-2).

## 2. Reference. Verdict: correct

Discrete discounted Riccati, dlqg (fc_game.py:27-36). Derivation: V_k(z) = z'P z + 2 s'z + const, stage cost beta^k [(z - zeta)'Q(z - zeta) + R u^2] dt, continuation beta V_{k+1}(F z + G u + c + noise). The minimiser is u = -(R dt + beta G'PG)^{-1} beta G'(P F z + P c + s); P and s recursions match lines 34-35 term by term. Discount: beta = e^{-rho dt} multiplies the continuation, which matches cost() (:115, disc = e^{-rho t_k} = beta^k). Terminal condition P_N = 0, s_N = 0: correct, since cost() has no terminal term and sums the integrand over k = 0..N-1 with x_k (left-point rule), the same convention as dlqg's stage cost. Check (a): the feedback -(Gain_k z + g_k) reproduces the brute-force least-squares optimum of the deterministic problem to 3e-15 (major, 1-d) and 2e-15 (minor, 3-d) for random z_0.

Discrete Kalman filter, dkalman (:37-43). Measurement dy_k = C z_k dt + noise, var Rv dt, i.e. H = C dt; standard gain V H'(H V H' + Rv dt)^{-1} = V C'(C V C' dt + Rv)^{-1}: line 41. Posterior V - K C dt V, predict F Vp F' + Sig: lines 41-42. Timing: zh_k is the prior E[z_k | dy_0..dy_{k-1}]; the control u_k = -(Gain_k zh_k + g_k) uses it (reference_paths :73), then dy_k (which uses z_k, :74) updates it (:77-78). This is the information the learned policy has at k (item 4), and certainty equivalence applies because the estimation error does not depend on the control (linear dynamics). Check (b), 200k paths: the filter state equals the linear regression of the true state on (1, dy_0..dy_{k-1}) to within the regression's own sampling error (rel L2 7e-3 at k = 5 to 3e-2 at k = 99 with 200 regressors), and E[err^2] matches the computed prior variances V0, V1 for all three minor components at k = 5, 30, 99 (e.g. xhat_0: 0.0736 vs 0.0735, 0.0069 vs 0.0069, 0.0043 vs 0.0043).

Minor's extended state (x_i, x_0, xhat_0), Fz/cz/Sz (:51-56). Expanding the major's filter recursion with u0 = -(Gain0 xhat + g0): xhat_{k+1} = a0 K0 L0 dt x0 + [a0(1 - K0 L0 dt) - B0 dt Gain0] xhat - B0 dt g0 + a0 K0 SV0 dv0. Row 3 of Fz, cz[2], Sz[2,2] match. Row 2 (x0 under the major's feedback) and cz[1] match. This is the discrete analogue of the paper's (43). Noises (dwi, dw0, dv0) are mutually independent and independent of dvi, so Sz is diagonal and no cross-term appears; this is the "key property" the paper cites after Prop. 2.

Separation principle: valid. u_i enters only the x_i row; (x_0, xhat_0) is an autonomous linear Gaussian system; observations are linear in z with independent Gaussian noise. So (z, y_i) is jointly Gaussian, the Kalman filter is the exact conditional mean, and the LQG separation theorem applies for any finite horizon. Observability of xhat_0 is not needed for the conditional mean to be well defined (finite horizon; the xhat_0 channel is driven by the x_0 observation through F row 3).

Observation 4 of the paper (E[mean-field estimation error | F_i] = 0, eq. (75)): the author's suspicion is reasonable. The second term in (75) is E[int Phi D~ (dw_0, dnu_0) | F^y_i]; the paper sets it to zero "due to Assumption 3" (v_i independent of w_0), but F^y_i also contains y_i, which depends on x_0 and hence on w_0, so independence of v_i from w_0 does not give zero. The claim holds when D_0 = 0 and the major has full observations. It does not matter in this instance: the mean field appears in no cost, dynamics or observation (H_0 = H_2 = 0, no x-bar in (2)), and the minor's filter for (x_0, xhat_0) is a full joint Kalman filter (Prop. 2 / code) that needs no such claim.

## 3. Brackets. Verdict: correct

J_i = sum_j beta^j [Q (x_j - target_j)^2 + R u_j^2] dt with x_j affine in u_k for j > k: dx_j/du_k = (1 + A dt)^{j-k-1} B dt (j > k), zero for j <= k (the stage cost at j = k uses x_k). So dJ/du_k = 2 beta^k dt [R u_k + sum_{j>k} e^{-rho (j-k) dt} (1 + A dt)^{j-k-1} B dt Q (x_j - target_j)], and the stationarity condition is u_k = -(1/R) E[sum_{j>k} W[j,k] Q (x_j - target_j) | F^i_k]. kernel (:82-85) is W with mask n = j - k > 0; bracket (:118-120) is -(Q (x - target)) @ W / R. The factor 2 and the stage dt cancel between the two terms; the discount inside the sum is e^{-rho (t_j - t_k)} (the e^{-rho t_k} factor on the u^2 term cancels). Check (c), autograd of cost(): max |bracket - (u - dJ/du / (2 beta^k dt R))| = 2e-6 (major) and 7e-7 (minor), against |bracket| up to 3.1.

No own-impact term: correct. The control enters the cost only through R u^2 and the state only linearly through the dynamics, so no term with u multiplying the state appears.

Cross-player reaction terms are legitimately absent: the minor's deviation does not change x_0 or y_0 (autograd: J_0 unused w.r.t. u_1), and the major's deviation changes x_i and u_i but J_0 does not depend on them.

The control variate (npg.py:31-33, 79), used in the log's run (CV=1): M_k = sum_{j>=k} gamma^{j-k} zeta(t_j, x_j).dW_j with dW the state noises; E[zeta_j dW_j | F^i_k] = 0 for j >= k since dW_j is independent of everything at t_j. Fixed point unchanged. cv_gamma = beta (1 + A dt) matches W[j,k] = (beta a)^{j-k} B dt / a.

## 4. Information. Verdict: correct

d_now = [0, 0] (:95), so each head sees only (t, GRU hidden). obs_inc (:110-113): player 0 gets (L0 x0_k dt + SV0 dv0_k)/sqrt(dt); player 1 gets ((L1 xi_k, L2 x0_k) dt + (SV, SV2) dvi_k)/sqrt(dt). Both read the state at index k (not k+1) although called after step k (npg.py:65). The increment over [k, k+1] is fed to the GRU after step k and its hidden state is first used at step k+1, so the control at k is a function of dy_0..dy_{k-1} only; at k = 0 it is a constant (hidden = 0), as is the reference's u_0 = -g_0 (prior mean 0). In fit() (npg.py:77), Policy.mem shifts the GRU output by one step (npg.py:26), the same convention. The reference strategy uses exactly the same information (prior estimate at k). No lookahead; no access to the other player's observation path or to the true states.

## 5. Equilibrium concept. Verdict: correct, and the distinction is immaterial here

The solver's fixed point is u_i = E[bracket_i | F^i] with the other player's policy held fixed as a feedback map of its own observations (closed loop in the other's policy, but the bracket ignores the other's reaction). In this instance: the minor cannot influence x_0, y_0, or u_0 (verified by autograd), and the major's cost is independent of x_i, u_i. Hence for both players the open-loop and closed-loop best-response conditions coincide and equal the single-agent FOC that the reference solves. The major's control being a feedback on an estimate the minor cannot see affects only the minor's filtering problem (handled by the extended state), not the equilibrium concept.

## 6. Gateaux log. Verdict: test valid; the minor's ratio 0.76 is uninformative, not a failure

The log (results/fc_gateaux.log): last-iterate rel L2 errors u0 8.3%, u1 5.9%; Polyak average u0 6.6%, u1 5.1%. Gateaux: major mean |dJ/d eps| 2.6e-3 vs 5.7e-2 at 0.8 x (ratio 0.046); minor 8.5e-3 vs 1.1e-2 (ratio 0.76).

Validity: gateaux_check (npg.py:141-176) perturbs player i by eps v with v = random MLP of (t, learned GRU hidden), the hidden being a function of own observations only, so the directions are admissible; the other player is held fixed as a feedback map; common random numbers; central difference. The cost is exactly quadratic in eps (linear dynamics), so the central difference has no bias and the derivative at an exact optimum is pure Monte Carlo noise. Note the check runs on the last iterate (S.pols), not the averaged controls that give the headline numbers.

Calibration (check (d), same procedure with the exact reference strategy implemented as a policy, same 4096 paths, directions reading the filter state instead of the GRU hidden): at the exact optimum the "learned" numerator is 2.7e-3 (major) and 3.7e-3 (minor), i.e. the noise floor at these settings; the ratio is 0.044 and 0.075, never 0. With a deliberate 6% deterministic error added to the reference, the numerator becomes 5.6e-3 (major) and 8.6e-3 (minor). The log's minor numerator 8.5e-3 is therefore consistent with a ~6% error, in line with the 5-6% rel L2 reported. The log's minor denominator 1.1e-2 is 4.5x smaller than the denominator in the calibration (4.9e-2): it depends on how aligned the random directions are with u*, and directions built from a random MLP of a 32-dim GRU hidden are less aligned than directions built from the filter state. So the ratio is direction-family dependent and not a calibrated statistic; the absolute numerator against the noise floor is the informative quantity. The exploitability runs (results/logs/fc_exploit_cv0.log, cv1.log) agree: re-solving the minor's best response gains at most 0.02% of cost, the major's at most 0.6% (negative gains, i.e. within re-solve noise).

Reading of the log: the major's derivative is at the noise floor and the minor's is 2-3x the floor, consistent with a 5-6% control error of mostly low-curvature (noise-like) type that costs little. The comment "0 = optimal" overstates the resolution of the test; a floor computed at the reference (or a reported standard error over directions and paths) should accompany the ratio.

## Reproduced author checks

CHECK_REF at the log's settings with 32768 paths: E[bracket | filter state] vs reference control, rel L2 major 0.0244, minor 0.0165; regression on (xhat_i, xhat_0) only 0.0152. Matches the report (1.7-2.4%) and confirms the caveat that the estimate-of-estimate coefficient (0.031 vs 0.484 on xhat_i) is not exercised numerically in this instance.

## Summary of issues (all low severity)

1. Wording: "the paper's solution is exact" (fc_game.py:2, report section 3) should say the finite-horizon, discrete-time version of the paper's Prop. 2 construction; the paper's gains are stationary infinite-horizon. No numerical consequence.
2. The Gateaux ratio is not calibrated; the minor's 0.76 reflects a small denominator for the chosen direction family, and the numerator is consistent with the reported 5-6% error. Report the noise floor at the reference alongside it, and run the check on the averaged controls if they are the headline.
3. Model-state initialisation through a side effect of obs_now (fc_game.py:108, s._e = exo) is fragile but correct in every call path in npg.py (obs_now(0, ...) always precedes step at k = 0 within a fresh init_state).
No correctness errors found in the model, reference, brackets, information structure or equilibrium concept.
