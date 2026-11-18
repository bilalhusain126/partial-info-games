"""Diagnose the lam = 0 plateau of pt_game: (A) is the exact reference right? Regress the pathwise bracket on the exact sufficient
statistic (alpha_i, q_i, alphahat_j) along reference paths and compare with the reference control (noise floor only if the reference
is the discrete optimum). (B) is the GRU the bottleneck? Same regression on the learned solution's paths with the exact alphahat_j as
a feature. Usage, from scripts/: LAM=0 python3 pt_diag.py"""
import os, sys, numpy as np
os.environ.setdefault('LAM', '0'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'games'))
import pt_game as P

ref = P.reference_lam0(); m = int(os.environ.get('M', '32768')); nz = P.simulate_noise(m, 7); a1, a2 = P.signals(nz)
# reference paths with the filter states kept
q = np.zeros((m, 2)); ah = np.zeros((m, 2)); nu = np.zeros((m, 2, P.N)); AH = np.zeros((m, 2, P.N)); Q = np.zeros((m, 2, P.N)); mu = np.zeros((m, P.N))
for k in range(P.N):
    AH[:, :, k] = ah; Q[:, :, k] = q
    for i in (0, 1):
        ai = a1[:, k] if i == 0 else a2[:, k]; nu[:, i, k] = -(ref['Gain'][k] @ np.stack([q[:, i], ai, ah[:, i]]))
    mu[:, k] = a1[:, k] + a2[:, k]
    dY = np.stack([a2[:, k], a1[:, k]], -1) * P.DT + P.SIG * nz['dws'][:, k, None]
    ah = (1 - P.KAP * P.DT) * (ah + ref['K'][k] * (dY - ah * P.DT)); q = q + nu[:, :, k] * P.DT
W = P.PTModel().W.numpy().astype(np.float64)
rel = lambda a, b: np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean())
for i in (0, 1):
    qT = Q[:, i, -1] + nu[:, i, -1] * P.DT; ai = (a1 if i == 0 else a2)[:, :-1]
    br = (-2 * P.PSI * qT[:, None] + (mu - 2 * P.PHI * Q[:, i]) @ W) / (2 * P.A_)
    fit = np.zeros_like(br)
    for k in range(P.N):
        F = np.stack([np.ones(m), ai[:, k], Q[:, i, k], AH[:, i, k]], -1); fit[:, k] = F @ np.linalg.lstsq(F, br[:, k], rcond=1e-8)[0]
    print(f'(A) trader {i+1}: E[bracket | alpha_i, q_i, alphahat_j] vs reference control: rel L2 {rel(fit, nu[:, i]):.4f}  (|nu| {np.abs(nu[:, i]).mean():.3f})', flush=True)
    # discretisation check of the Kalman filter: alphahat_j should be the regression of alpha_j on it with slope 1
    aj = (a2 if i == 0 else a1)[:, :-1]
    for k in (10, 25, 45):
        c = np.polyfit(AH[:, i, k], aj[:, k], 1); print(f'      k={k}: E[alpha_j | alphahat_j] slope {c[0]:.3f} intercept {c[1]:+.4f}; var(alphahat) {AH[:, i, k].var():.4f} var(alpha) {aj[:, k].var():.4f}')
# (B) learned solution: bracket regressed on the exact statistic along the learned paths
f = os.environ.get('NPZ', '../results/pt_scale_32768.npz')
if os.path.exists(f):
    d = np.load(f); print(f'(B) learned ({os.path.basename(f)}): trader 1 vs exact {rel(d["nu1"], d["nu_ex"]):.4f}; mean |learned| {np.abs(d["nu1"]).mean():.3f}, |exact| {np.abs(d["nu_ex"]).mean():.3f}; corr {np.corrcoef(d["nu1"].ravel(), d["nu_ex"].ravel())[0, 1]:.4f}; slope {np.polyfit(d["nu_ex"].ravel(), d["nu1"].ravel(), 1)[0]:.3f}')
    N = P.N; print('      error by time decile:', [round(rel(d['nu1'][:, j * N // 10:(j + 1) * N // 10], d['nu_ex'][:, j * N // 10:(j + 1) * N // 10]), 3) for j in range(10)])
    # regress learned-minus-exact on the exact statistic reconstructed from the saved paths (same validation noise seed 7)
    nzv = P.simulate_noise(d['nu1'].shape[0], 7); a1v, a2v = P.signals(nzv); assert np.allclose(a1v, d['a1'], atol=1e-5)
    ahv = np.zeros(d['nu1'].shape[0]); AHv = np.zeros_like(d['nu1'])
    for k in range(N):
        AHv[:, k] = ahv; dY = a2v[:, k] * P.DT + P.SIG * nzv['dws'][:, k]; ahv = (1 - P.KAP * P.DT) * (ahv + ref['K'][k] * (dY - ahv * P.DT))
    err = d['nu1'] - d['nu_ex']; expl = np.zeros_like(err)
    for k in range(N):
        F = np.stack([np.ones(len(err)), a1v[:, k], d['q1'][:, k], AHv[:, k]], -1); expl[:, k] = F @ np.linalg.lstsq(F, err[:, k], rcond=1e-8)[0]
    print(f'      fraction of the error explained by a linear function of (alpha_i, q_i, alphahat_j): {1 - ((err - expl) ** 2).mean() / (err ** 2).mean():.3f}')
    for k in (10, 25, 45):
        c = np.linalg.lstsq(np.stack([np.ones(len(err)), a1v[:, k], d['q1'][:, k], AHv[:, k]], -1), err[:, k], rcond=1e-8)[0]; g = ref['Gain'][k]
        print(f'      k={k}: error coefficients on (1, alpha_i, q_i, alphahat_j) {np.round(c, 4)} vs exact gains {np.round(-g[[1, 0, 2]], 4)} (alpha_i, q_i, alphahat_j)')
