import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(9, 5.6))
for col, (f, title) in enumerate([('../results/fig_h0.npz', 'h = 0, T = 0.3, partial information (vs Riccati/Kalman exact)'), ('../results/fig_full.npz', 'h = 0.2, T = 1, full information (vs linear-regression exact)')]):
    d = np.load(f); t = d['t']
    for row, (key, lab) in enumerate([('eta', r'informed trader $\eta_t$'), ('nu', r'broker $\nu_t$')]):
        a = ax[row, col]
        for i, c in zip(range(3), ['C0', 'C1', 'C2']):
            a.plot(t, d[key + '_ex'][i], 'k--', lw=1, label='exact' if i == 0 else None)
            a.plot(t, d[key][i], color=c, lw=1.2, label='learned' if i == 0 else None)
        err = np.sqrt(((d[key] - d[key + '_ex']) ** 2).mean() / (d[key + '_ex'] ** 2).mean())
        a.set_title((title + '\n' if row == 0 else '') + f'{lab}, rel. L2 error {100*err:.1f}%', fontsize=9)
        a.set_xlabel('t'); a.spines[['top', 'right']].set_visible(False)
        if row == 0 and col == 0: a.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig('../results/wj_prototype.png', dpi=160); print('saved ../results/wj_prototype.png')