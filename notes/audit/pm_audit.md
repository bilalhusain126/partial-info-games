# Audit of games/pm_game.py (private Markov-chain signals), 2026-11-19

Repo: /Users/bhusain/Desktop/Bilal Saleh Husain/Academics and Research/Repos/partial-info-games (read-only).
Files: games/pm_game.py, games/pt_game.py, src/npg.py, results/pm_*.log, results/summary.md, scripts/narval_batch5.sh.
Check script: scratchpad/audit/pm_check.py (imports pm_game read-only; all numbers below come from it unless a log is cited).
Paper: Casgrain and Jaimungal, arXiv 1806.04472 (Theorem 3.1, Proposition 5.1, eq. 5.12, Appendix C.1).

## 1. Model, filter, control at lam = 0 — CORRECT

Chain (pm_game.py:18-32). P_STAY = (1 + e^{-2 KAP dt})/2 equals expm(C dt)[0,0] for the generator C = KAP [[-1, 1], [1, -1]] (0.98039472 both). Simulated autocorrelation E[alpha_l alpha_0]/E[alpha_0^2] at l = 1, 5, 20: 0.9614, 0.8181, 0.4499 vs e^{-2 KAP l dt} 0.9608, 0.8187, 0.4493 (200k paths). Initial state uniform (P(+) = 0.4999) = stationary law; filter prior 0.5 (pm_game.py:38) matches. So E[alpha(t_l) | F_k] = r^{l-k} E[alpha(t_k) | F_k], r = e^{-2 KAP dt}, as used at pm_game.py:53.

Observation. Price S_k and own state alpha_i(k) in obs_now (pm_game.py:83-85); own increment and price increment in obs_inc (86-88). At lam = 0 trader i's observation of the other chain is dS - alpha_i dt = alpha_j dt + sig dW, which is what the reference filters (pm_game.py:63, dY1 = a2 dt + SIG dws).

Filter (pm_game.py:35-44). Discrete Bayes filter: pi_k = P(alpha_j(t_k) = + | dY_0..dY_{k-1}) (prior at k), Gaussian likelihood update with the left-endpoint drift, then one-step prediction with P_STAY. Checks: (a) brute-force enumeration of all 2^8 chain paths on one simulated path reproduces pi_0..pi_6 to 6 decimals; (b) calibration, regression of alpha_j(t_k) on ahat_j(t_k) has slope 0.997-1.001, intercept < 2e-3, max decile gap 0.011, at k = 1, 10, 25, 49. Timing is right: the control at k uses pi_k, which is measurable w.r.t. S_0..S_k. Note this is the exact filter for the DISCRETE model the engine simulates (drift frozen at the left endpoint, pm_game.py:81-82), which is the right comparison for the learned solver; it is the Euler-grid version of the Wonham filter (Theorem 3.1 with C != 0), not the continuous one.

Control (pm_game.py:46-69). Backward recursion nu_k = -G_k q_k - g_k mhat_k with mhat = alpha_i + ahat_j. Independent check by certainty equivalence: for q_k = 0.4, mhat = 0.7 and the forecast path m_l = r^{l-k} mhat, the direct minimisation of the deterministic quadratic cost sum_{l>=k} [-q_l m_l + a nu_l^2 + phi q_l^2] dt + psi q_N^2 over (nu_k..nu_{N-1}) returns nu_k = -0.082221, -0.134394, -0.317185, -0.729272 at k = 0, 10, 30, 48; the recursion returns the same six digits. Signs: G > 0 (liquidate), -g > 0 (buy on positive signal). Verified on paths: (i) FOC, E[(1/2a)(-2 psi q_N + sum_{l>k} (mu_l - 2 phi q_l) dt) | 1, alpha_i, q_i, ahat_j] (linear regression, 200k paths) vs reference nu: rel L2 0.0071 / 0.0094 (the author reports 1.7%, consistent with fewer paths); (ii) Gateaux derivative of the realised mean cost at the reference along four random admissible directions: |dJ| <= 7e-5, vs +1.6e-3 at 0.8 x reference; cost at reference -0.0356 vs -0.0342 at 0.8 x.

Continuous closed form (paper eq. 5.12, beta = 0). h_2 from (5.10a) with zeta = (psi + sqrt(a phi))/(psi - sqrt(a phi)): h_2(T) = -psi, Riccati residual 2e-11. h_1 = ahat_t int_t^T e^{-2 KAP (u-t)} exp(int_t^u h_2/a) du (C.10 with E_t[A_u] = e^{-2 KAP (u-t)} ahat_t for the symmetric two-state chain; here A = alpha_i + alpha_j, both components decay at the same rate, so the same weight multiplies alpha_i(t) and ahat_j(t)). Discrete G_k vs -h_2(t_k)/a: rel L2 0.0022 (N=50), 0.0006 (200), 0.0001 (800). Discrete -g_k vs h_1-weight/(2a): 0.0531, 0.0135, 0.0034. O(dt) convergence, as expected from the one-step offset (the discrete sum starts at l = k+1).

## 2. Bracket — CORRECT, identical to pt_game.py

pm_game.py:91 is character-for-character pt_game.py:83: (LAM q_k - 2 PSI q_N + (mu - 2 PHI q) @ W)/(2 A_), W_{jk} = dt for j > k. Rederived from the discrete performance sum_l (q_l mu_l - a nu_l^2 - phi q_l^2) dt - psi q_N^2 with q_l the left endpoint: d/dnu_k gives dt [lam q_k - 2 a nu_k - 2 psi q_N + sum_{l>k} (mu_l - 2 phi q_l) dt]; the lam q_k term is from mu_k = ... + lam nu_k. The other trader's nu_j inside mu is held fixed (open-loop). cost (pm_game.py:92-94) is minus the same performance.

## 3. Information — CORRECT, no lookahead

obs_now at k: alpha_i(k), q_i(k), S_k. obs_inc at k (computed after step k, npg.py:63-65) holds the increments over [k, k+1] and feeds the GRU state used at step k+1 only (npg.py:58: memk = h[i][0] before step k). In fit, Policy.mem shifts the GRU outputs by one with zero at k=0 (npg.py:26), so entry k uses seq[:, :k]. Player i's regression uses seqs[i] only. Nothing of the other chain or of nu_j enters except through S. Minor: the own-chain increment is not scaled by 1/sqrt(dt) as in pt_game.py:80 (harmless; it is +-2 or 0).

## 4. Cross-check against Casgrain-Jaimungal (commit 397e4e1) — CORRECT but undocumented

The commit changes only results/summary.md; no script or log in the repo contains the check. What was checked (from the summary text, confirmed by reproduction): the deterministic coefficients G_k and g_k of the discrete recursion against -h_2/a and the h_1 weight/(2a) of eqs (5.10a)-(5.10b), beta = 0, two-state chain. My independent implementation reproduces the stated numbers exactly: inventory gain 0.2%, signal gain 5.3% (N=50), 1.4% (200), 0.3% (800). The remark "sign of beta/2 corrected" is right: with +beta/2 in (5.10a), 2 h_2(T) + beta = 3 beta - 2 alpha instead of the terminal value beta - 2 alpha required by (C.9); irrelevant at beta = 0. Not checked by the author: the filter itself against the paper (the paper gives only the SDE (3.3)); my brute-force and calibration checks cover that. Recommendation: commit the cross-check script.

## 5. Results — CLAIMS SUPPORTED; interpretation issues

pm_lam0_32768.log: averaged controls (saved npz) vs reference: rel L2 0.0624 (trader 1), corr 0.998, slope 0.983, mean |nu| 0.184 vs 0.187, i.e. the summary's "6.2 / 6.1%, 0.998, 0.98" is exact. Last iterate stable (0.066-0.072 over it 6-11). Error by time decile 0.042 ... 0.127 (worst near T). FOC residuals 0.11 / 0.14.

pm_lam0_8192.log: averaged 0.137 / 0.090 as reported, BUT the last iterate drifts away monotonically from it 4 to it 11 (u0 0.108 -> 0.262, u1 0.086 -> 0.138; change rises 0.012 -> 0.026). At 8k paths the Picard iteration is not stable; the Polyak average hides this. "Error falls with budget" is true, "converges" at 8k is not.

pm_priv1_s1-3.log (lam = 1): per-iteration change settles at 0.02-0.03 and rises slightly over the last iterations (s2 0.017 -> 0.028, s3 0.011 -> 0.031); relative to RMS |nu| ~ 0.27 that is ~10% jitter per iterate. Pairwise seed distances of the saved averaged controls: 0.197-0.207 (trader 1), 0.171-0.180 (trader 2); each seed is 10-11% from the three-seed average. "Converges" is generous; "stationary up to 10% regression noise" is accurate. FOC residuals 0.16-0.19.

Gateaux ratios: only 7 values exist (s1: 0.292 player 0 only; s2: 0.115, 0.211; s3: 0.607, 0.576; lam 0.5: 0.307, 0.318). pm_lam0_32768.log has NO Gateaux output and pm_lam0_8192.log / pm_priv1_s1.log have player 0 only: the jobs were cut off (8 concurrent jobs under one 4h SLURM allocation, scripts/narval_batch5.sh). The summary's "0.1-0.6" matches the numbers that exist.

Validity of the Gateaux check for the equilibrium concept (npg.py:141-176): the perturbed u_i + eps v is rolled out with the OTHER player's network running as a feedback rule on the perturbed price (npg.py:154, 60-65), so the measured derivative is d/d eps J_i(u_i + eps v, u_j[S(u_i + eps v)]) = (open-loop Gateaux derivative) + (reaction term through nu_j's dependence on S). The method's first-order condition sets the first term to zero (open-loop Nash, as stated in the equilibrium-concept audit of 2026-11-19); the reaction term is nonzero whenever lam > 0, even at the exact open-loop equilibrium. So at lam = 0.5, 1 the ratio measures learning error plus the open/closed-loop gap, and a nonzero ratio is expected by construction; it cannot be read as "within 5-12% of optimal". Its size is bounded by the 1.9% concept gap found in the Gaussian game, so it is probably a minor part of 0.1-0.6, but the check is not a test of the condition the method solves. At lam = 0 the check is valid (no coupling); there it was applied to the last iterate of the 8k run (error 26% / 14%), giving 0.334, which also shows the ratio's scale is loose (the summary's rule "ratio 1 ~ 15-20% error" does not hold here).

## 6. Other — nothing wrong found

Seeds: training noise seed 1 + 100 SEED (1, 101, 201, 301), validation 7, FOC 23+i, Gateaux 29+i, exploitability 11; no overlap; torch.manual_seed(SEED) for the networks. Reference evaluated on the same validation noise (pm_game.py:104), so the comparison is pathwise. float32 only in the torch side; the reference and filter are float64 numpy on the same noise. Control variate at lam = 0 uses the price noise only (pm_game.py:96); zeta_j reads (t, alpha, q, S) at j, all prior to dW_j, so the term has zero conditional mean given F^i_k (valid). Cosmetic: unused variables w, D (pm_game.py:51) and argument a_self (35); docstring line 10 says "E[alpha(t_l) | F(t_k)] = e^{-2 KAP (t_l - t_k)} E[alpha(t_k) | F(t_k)]" which is right.

## Verdicts

1 model/filter/control: correct.  2 bracket: correct.  3 information: correct.  4 cross-check: correct (reproduced), artefact missing from the repo.  5 results: numbers supported; "converges" at 8k (lam 0) and at lam 1 overstated; Gateaux ratios at lam > 0 do not test the open-loop condition.  6 nothing suspicious.
