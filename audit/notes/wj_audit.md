# Audit: Wu-Jaimungal broker-trader game (partial-info-games)

Repository: `/Users/bhusain/Desktop/Bilal Saleh Husain/Academics and Research/Repos/partial-info-games` (not modified).
Files: `src/npg.py`, `games/wj_game.py`, `games/npg_wj.py`, `games/wj_fullref.py`, `games/wj_refcheck.py`, `games/wj_analyse.py`.
Paper: Wu and Jaimungal, arXiv 2412.17712v2 (1 Apr 2025). Page numbers below refer to the arXiv PDF.
Check script (new, in the scratchpad): `audit/wj_discrete_check.py`.

## Summary of verdicts

| Item | Verdict | Severity |
|---|---|---|
| 1 Model and instance | correct (the instance is the repo's own; the paper has no numerical section) | - |
| 2 First-order conditions / brackets | correct term by term; no missing own-impact term (it is absorbed into `-(2 phi - h) Q^B_T`); discretisation is a left Riemann sum with `s >= t`, which approximates the continuous FOC, not the discrete game's | medium (see item 3) |
| 3 References | continuous-time Riccati/Kalman-Bucy/LQR, Euler-discretised; 5.1% (trader) and 7.3% (broker) from the exact solution of the discrete game at N = 40, halving at N = 80. The code's bracket fixed point tracks the continuous reference to 0.6% / 1.7%, so solver and reference are mutually consistent. `wj_refcheck.py`'s 6.7% is a float32 artefact | medium |
| 4 Information structure | correct, no lookahead. One restriction: in `npg_wj.py` the trader has no memory and does not see the broker's hidden state, which it needs for h > 0 | low at h = 0, medium at h > 0 |
| 5 Equilibrium concept | open-loop, matching Definition 2.3 of the paper. The cost-based checks (`exploitability`, `gateaux_check`) are closed-loop deviations | low at h = 0, medium at h > 0 |
| 6 Other | broker cost in `npg_wj.py` uses `h*eta` where the paper has `h*nu`; wrong Gateaux/exploitability diagnostics for the broker at h > 0 (training unaffected). Seeds fine. Some runs lie outside the paper's small-time condition | medium |

## 1. Model

Paper (Sec. 2, p. 3-5): `dz_t = alpha_t dt + sigma_t dW_t` (2.1); impact `dY = (h nu - p Y) dt`, `Y_0 = y`; midprice `S = Y + z` (2.4); `dQ^B = (nu - eta) dt` (2.5); `dQ^I = eta dt`; cash `dX^B = (-(S + a nu) nu + (S + b eta) eta) dt` (2.6), `dX^I = -(S + b eta) eta dt`; criteria (2.9), (2.10)
`J_I = E[X^I_T + (S_T - psi Q^I_T) Q^I_T - int_0^T r_I (Q^I_t)^2 dt]`, `J_B = E[X^B_T + (S_T - phi Q^B_T) Q^B_T - int_0^T r_B (Q^B_t)^2 dt]`.
Information: the trader uses `H^2_F` (full information), the broker uses `H^2_G`, `G` the filtration of the lit price `z` (p. 3).

Code: `wj_game.py:20-27` (OU signal `alpha`, `dz = alpha dt + sigma dW`), `wj_game.py:95-96` and `npg_wj.py:22-24` (`QB += (nu - eta) dt`, `QI += eta dt`, `Y += (h nu - p Y) dt`). Sign of the impact, which inventory moves with which control, and who pays what all match. The paper's `alpha` is a general `H^2_F` process; the code specialises to a stationary OU signal (`kappa = 1`, `sigma_a = 1`), which is allowed.

Observations: the broker in the paper sees `S = z + Y`; since `Y` is a function of the broker's own past `nu`, this is the same sigma-algebra as `z`. The code gives the broker `dz` increments, the Kalman estimate `alphahat(dz)` and `Y` (`npg_wj.py:25-30`), all `G`-measurable. The trader sees `alpha, Q^B, Q^I, Y, alphahat` (and in the prototype also the broker's GRU state), all `F`-measurable.

Instance: the paper has no numerical experiments or parameter values (confirmed by searching the full text; Sec. 5 is an asymptotic expansion in h, Sec. 6 the conclusion). The instance `a = b = phi = psi = 1, r_B = r_I = 0.1, p = kappa = sigma_a = 1, sigma = 0.5, q_B = q_I = y = 0` (`wj_game.py:15`) is the repo's own choice. T = 0.3 satisfies the small-time condition of Theorem 4.1: `C(T) = T^2 max{5, 9.5, 1, 0.02, 0.02} = 0.855 < 1` at h = 0 (unique equilibrium guaranteed for T < 0.324). At h = 0.2 the bound gives T < 0.298, so `hsweep_0.2` at T = 0.3 is marginally outside; the full-information figure at h = 0.2, T = 1 (`C = 11.3`) and the strong-coupling runs `a = b = 0.1, h = 1, T = 0.3` (`C ~ 180`, scripts/narval_batch2.sh:16) are far outside. The condition is only sufficient, but the batch-2 divergence noted in `scripts/narval_batch3.sh:2` is consistent with it.

## 2. First-order conditions

Paper, Proposition 3.8 (p. 16):
`eta_t = (1/2) b^{-1} E[ -2 psi Q^I_T + int_t^T (alpha_s + h nu_s - p Y_s - 2 r_I Q^I_s) ds | F_t ]` (3.13).
Proposition 3.9 (p. 17):
`nu_t = (1/2) a^{-1} E[ -(2 phi - h) Q^B_T + int_t^T (alpha_s + h eta_s - p Y_s - (h e^{(t-s)p} p + 2 r_B) Q^B_s) ds | G_t ]` (3.16).
Theorem 4.1 (p. 18) defines `Phi_{t,1}`, `Phi_{t,2}` as the same two expressions with `Q^B_T = q_B + int_0^T (nu - eta)` and `Q^I_T = q_I + int_0^T eta` written out.

Own-impact term. Differentiating `J_B = E[int (Q^B (alpha + h nu - p Y) - a nu^2 - r_B (Q^B)^2) dt - phi (Q^B_T)^2]` in the direction `zeta` gives, besides the integral terms, the direct term `h Q^B_t` (the control multiplies the state) and the terminal `-2 phi Q^B_T`. With `h Q^B_t = h Q^B_T - h int_t^T (nu_s - eta_s) ds` the direct term and the `h nu_s` integrand combine to `-(2 phi - h) Q^B_T + int_t^T h eta_s ds`, which is exactly the paper's form. So the paper's bracket already contains the own-impact term; nothing is missing. The trader's reward has no `eta`-times-state product (`Q^I` multiplies `alpha + h nu - p Y`), so no such term arises for the trader.

Code (`wj_game.py:99-108`, used by `npg_wj.py:36-37`):
- trader: `gI = alpha + h nu - p Y - 2 r_I Q^I` at the left grid point; `tI = (-2 psi Q^I_N + sum_{j >= k} gI_j dt) / (2 b)`. Matches (3.13).
- broker: `base = alpha + h eta - p Y - 2 r_B Q^B`; `conv_k = sum_{j >= k} h p e^{-p dt (j-k)} Q^B_j dt`; `tB = (-(2 phi - h) Q^B_N + sum_{j >= k} base_j dt - conv_k) / (2 a)`. Matches (3.16) term by term, including the sign of the memory kernel and the `(2 phi - h)` terminal coefficient.

Discretisation. All sums run over `j >= k` (flip-cumsum-flip at `wj_game.py:102,104`, kernel mask `>=` at line 105): a left Riemann sum of `int_t^T` that includes the current step. The exact first-order condition of the Euler game that is simulated is instead
`eta_k = (1/2b)[-2 psi Q^I_N + sum_{j > k} (...) dt]`, `nu_k = (1/2a)[h Q^B_k + sum_{j > k}(alpha_j + h nu_j - p Y_j - 2 r_B Q^B_j) dt - h p sum_{j > k} (1 - p dt)^{j-k-1} Q^B_j dt - 2 phi Q^B_N]`
(from `dQ_j / d u_k = dt` for `j > k` only, and `dY_j / d nu_k = h dt (1 - p dt)^{j-k-1}`). The difference is one integrand term times `dt` per control, i.e. O(dt). Numerically (item 3) the `j >= k` convention lands close to the continuous-time equilibrium and 4.6% / 6.8% away from the discrete game's equilibrium at N = 40.

## 3. References

`exact_h0` (`wj_game.py:38-66`). Trader: `P' = P^2/(2b) - 2 r_I, P(T) = 2 psi`; `R' = (P/(2b) + kappa) R - 1, R(T) = 0`; `eta = (-P Q^I + R alpha)/(2b)`. Verified against the HJB with `V = -P Q^2/2 + R Q alpha + ...`: correct continuous-time solution. Broker: Kalman-Bucy on `(alpha, Q^I)` with `dQ^I = eta dt` and `eta` the exact linear trader (`Fm`, `Hm = [1, 0]`, gain `Sigma H / sigma^2`, Riccati `Sigma' = F Sigma + Sigma F' + G - Sigma H H' Sigma / sigma^2`), then LQR on `xi = (Qhat^B, alphahat, Qhat^I)` with `A`, `M`, `S' = -(M + A'S + SA + S B B' S / a)`, `nu = B' S xi / a`. The separation is valid at h = 0: `E[Q^B alpha | G] = Qhat^B alphahat + Sigma_12(t)` and `E[(Q^B)^2 | G] = (Qhat^B)^2 + Sigma_22(t)` with deterministic covariances independent of `nu`, so the broker's problem is an LQR in the filtered state driven by innovations. The filter is the right one: the broker's information is the price path, equivalently `dz`, and `Q^I` is unobserved but correlated with `alpha` through the trader's feedback. All three pieces are the continuous-time equations stepped by forward/backward Euler (`wj_game.py:42, 52, 58, 63`), not the discrete Riccati, discrete Kalman filter and discrete LQR of the Euler game.

Numerical check (`audit/wj_discrete_check.py`; same paths, seed 11; (A) = `exact_h0`, (B) = exact discrete-time solution of the simulated game: discrete Riccati for the trader, discrete Kalman filter with gain `Sigma H / (H Sigma H' dt + sigma^2)`, discrete LQR for the broker; (C) = fixed point of the code's bracket by float64 linear regression on the exact features; (D) = same with the discrete-game FOC `j > k`; (E) = (C) with float32 `lstsq` as in `wj_refcheck.py`):

| N | paths | (A) vs (B) eta / nu | (C) vs (A) | (C) vs (B) | (D) vs (B) | (E) vs (A) |
|---|---|---|---|---|---|---|
| 40 | 16384 | 0.051 / 0.073 | 0.008 / 0.028 | 0.046 / 0.072 | 0.004 / 0.027 | 0.013 / 0.051 |
| 40 | 65536 | 0.051 / 0.073 | 0.006 / 0.017 | 0.046 / 0.068 | 0.002 / 0.016 | 0.067 / 0.067 |
| 80 | 16384 | 0.025 / 0.036 | 0.005 / 0.030 | 0.024 / 0.045 | 0.005 / 0.030 | 0.012 / 0.052 |

Reading: (A) vs (B) halves from N = 40 to N = 80, so the reference is the continuous-time solution with an O(dt) error of 5-7% at the grid used (the known pitfall). (D) vs (B) shrinks with paths like regression noise, which validates (B). (C) vs (A) is 0.6% / 1.7%, so the code's `j >= k` bracket converges to (approximately) the continuous equilibrium, and the solver's reported errors against `exact_h0` (7% trader, 9-12% broker in `results/logs/npgwj_cv*.log`) are regression error, not reference bias. The pair (bracket, reference) is internally consistent, but neither is the equilibrium of the discrete game whose cost `npg_wj.cost` evaluates; see items 5 and 6.

`wj_refcheck.py` reports `eta 0.067, nu 0.067` (reproduced). Row (E) reproduces exactly this number with float32 `torch.linalg.lstsq` at 65536 paths; in float64 the same computation gives 0.006 / 0.017. The script's headline number is a float32 artefact (the same artefact `wj_fullref.py:17` already notes for the full-information reference). Its comment at line 22 attributes the gap to the wrong cause.

`wj_fullref.py` / `MODE=full` in `wj_game.py:136-154`: a linear-regression Picard on the full state `(1, alpha, Q^B, Q^I, Y)` using the same `targets()`. It is a consistency check of the same bracket under a linear ansatz (exact for the LQ structure), not an independent reference; it inherits the `j >= k` convention. `lstsq` in float64 with `rcond = 1e-6` is fine. At h = 0.2, T = 1 the contraction constant is 11, so convergence of this Picard is not guaranteed by Theorem 4.1 (it did converge, `change < 1e-5`).

## 4. Information structure

Engine (`npg.py:24-26`): `Policy.mem` returns `cat(zeros, out[:, :-1])`, so the memory used at step `k` is the GRU run on `seq[:, :k]`, entries `0..k-1`. With `obs_seq` entry `k = (dz_k / sqrt(dt), alphahat_k)` (`npg_wj.py:27`), the broker at `k` uses `dz_0..dz_{k-1}` and `alphahat_0..alphahat_{k-1}`: no lookahead. `obs_now` for the broker at `k` is `(Y_k, alphahat_k)` (`npg_wj.py:30`); `alphahat_k` is built from `dz_0..dz_{k-1}` (`wj_game.py:33`, `ah[k+1]` uses `dz[k]`), and `Y_k` depends on `nu_0..nu_{k-1}`. In `fit` the same `seqs` and `nows` are used (`npg.py:71, 77`). Prototype (`wj_game.py:74-78`): `features = cat(zeros, GRU(x))` then `[:, :-1]` in `rollout`, same shift; the trader receives `hb` (the broker's GRU state) which is `F`-measurable. The rollout loop (`npg.py:55-61`, `wj_game.py:90-96`) computes the controls at `k` before `step(k)`. All inputs are measurable with respect to the right filtration at the right time.

Restriction (not a lookahead): in `npg_wj.py` the trader has `d_seq = 0` and `obs_now = (alpha, Q^B, Q^I, Y, alphahat)`. At h = 0 the trader's bracket does not involve `nu`, so this is sufficient. At h > 0 the trader's conditional expectation of `h nu_s - p Y_s` given `F_t` depends on the broker's state (its GRU hidden state, or `Qhat^B, Qhat^I` in the linear case), which the trader cannot reconstruct from `alphahat` alone. The regression then projects on a smaller sigma-algebra than `F_t`, and the fixed point is an equilibrium in a restricted trader class. The prototype avoids this by passing `hb` to the trader (`wj_game.py:94, 119`). The h-sweep and full-information results (`wj_analyse.py`) come from the prototype (saved keys `t, nu, eta, alpha, QB, QI, Y`), so published numbers are not affected; `npg_wj.py` at h > 0 would be.

Note on "no filter is computed" (README): the broker's inputs include the exact Kalman estimate `alphahat` (`npg_wj.py:27, 30`; `wj_game.py:76, 93`). This is legitimate `G`-information, but it hands the broker the optimal signal filter rather than letting the GRU learn it.

## 5. Equilibrium concept

Paper, Definition 2.3 (p. 7): `(nu, eta) in H^2_G x H^2_F` is a Nash equilibrium if `nu` maximises `J_B(., eta)` over `H^2_G` and `eta` maximises `J_I(nu, .)` over `H^2_F`. Strategies are adapted processes and the other player's process is held fixed under a deviation: open-loop. Theorem 4.1's map `Phi` and the FBSDE of Theorem 4.2 (p. 20) are stated for processes.

Code: `bracket(i)` is evaluated on the realised paths with the other player's realised control path entering as data (`h nu_s` in the trader's target, `h eta_s` and `Q^B` in the broker's), and the Picard target is `(1 - rho) u_i + rho bracket_i` regressed on player i's information (`npg.py:87`). This is the open-loop fixed-point map `Phi`. The policies are feedback maps, but at the fixed point the realised processes satisfy the open-loop conditions, so the computed object is the paper's equilibrium. Verdict: matches.

The cost-based diagnostics are closed-loop: `exploitability` (`npg.py:101-119`) and `gateaux_check` (`npg.py:141-176`) re-simulate with the other player's feedback policy fixed, so the other player's realised control changes with the deviation. For the broker this makes no difference (its inputs `dz, alphahat, Y` do not depend on `eta`). For a broker deviation the trader's learned map reacts through its `Q^B` and `Y` inputs (and `hb` in the prototype); at h = 0 the true trader does not depend on these, so only spurious learned dependence enters; at h > 0 the two notions genuinely differ. These checks therefore measure closed-loop optimality of the broker, which is not the paper's notion.

## 6. Other findings

(a) Broker cost mismatch, `npg_wj.py:34`:
`perf = sum(QB * (al + W.H * eta - W.P_ * Y) - W.A_ * nu**2 - W.RB * QB**2) * dt - W.PHI * QB_N**2`.
The paper's reduced criterion (Lemma 2.2, p. 7) is `J_B = const + E[int (-a nu^2 + b eta^2 + (alpha + (h - 2 phi) nu - p Y + 2 phi eta - r_B Q^B)' Q^B) dt]`, i.e. with the terminal `-phi (Q^B_T)^2` kept separate, `Q^B (alpha + h nu - p Y)`. The code has `h eta` in place of `h nu` (it copied the integrand of the FOC bracket, where the `h eta` form is only valid together with the `(2 phi - h)` terminal coefficient). The Gateaux derivative of the code's cost has terminal coefficient `-2 phi Q^B_T` instead of `-(2 phi - h) Q^B_T`, a 10% change of the dominant term at h = 0.2. Consequences: `gateaux_check` and `exploitability` for the broker are wrong at h > 0 (`gateaux_chain.sh:6` ran only h = 0, where the two agree). Training is unaffected (it uses `bracket`). The omitted `+ b eta^2` is constant in `nu` and harmless for the checks, but `cost` is then not minus `J_B`.

(b) Discrete-vs-continuous mismatch in the diagnostics: `cost` is the discrete game's objective, whose optimum is (B) above, while the solver's fixed point approximates the continuous equilibrium (A), 5-7% away at N = 40. A perfectly converged solver would still show a nonzero Gateaux derivative and a positive exploitability gain of the corresponding size. The ratio reported by `gateaux_check` (against a 20% rescaling) has a floor of roughly 0.3 from this alone.

(c) Seeds: training `exo(m, 1 + 100 seed)`, validation `exo(m_val, 7)`, exploitability 11, FOC residual `23 + i`, Gateaux `29 + i` (`npg.py:42, 106, 127, 148`); prototype train 1, validation 7, linear reference 3 (`wj_game.py:126, 153`); `wj_refcheck` 11. No reuse between training, validation and reference. The h = 0 reference is evaluated on the validation paths (`npg_wj.py:43`), as it should be for a pathwise comparison.

(d) Float32: rollouts, targets and the 40-step cumsums in float32 are fine. The linear references use float64 `lstsq` (`wj_game.py:148`, `wj_fullref.py:16`); `wj_refcheck.py:17-18` does not and is wrong by 6.7% as a result (item 3). The `rcond = 1e-6` truncation is needed because the regressors `Q^B, Q^I, Y` are all zero at k = 0 and nearly collinear early on.

(e) Cost sign: `cost = -perf`, `exploitability` returns `J0 - J` with `J` the re-solved cost, positive = improvement, consistent with "lower is better".

(f) `wj_analyse.py:11` labels the full-information linear-regression Picard "exact ref"; it is the same bracket under a linear ansatz (item 3), so the 0.6-1.7% agreement there is a consistency check, not an independent validation.

(g) The "7% trader error" at h = 0 in the engine logs is large for a Markov full-information control with five inputs; given (C) vs (A) is 0.6%, it is regression error, plausibly from the irrelevant inputs `Q^B, Y, alphahat` at h = 0 and the noisy pathwise target. Not a correctness issue.
