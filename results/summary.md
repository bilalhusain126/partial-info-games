# Batch summary

## Scaling at lam = 0 (exact reference, recomputed after the reference fix of 2026-11-18), control variate on, 128 epochs per iteration

| paths | trader 1 | trader 2 | FOC residual 1 | FOC residual 2 |
|---|---|---|---|---|
| 2048 | 0.078 | 0.078 | 0.1321 | 0.1222 |
| 4096 | 0.064 | 0.062 | 0.0871 | 0.0935 |
| 8192 | 0.055 | 0.050 | 0.0828 | 0.0987 |
| 16384 | 0.046 | 0.045 | 0.1123 | 0.0753 |
| 32768 | 0.037 | 0.032 | 0.0437 | 0.0574 |

Figure: results/scaling.png


## Private two-way case, lam = 2, control variate on

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
