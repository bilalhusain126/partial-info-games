import numpy as np, glob
rel = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
print('--- fine grid (N=80, 32k paths) h=0 vs exact:'); d = np.load('../results/fine_h0.npz'); print('  eta', round(rel(d['eta'], d['eta_ex']), 3), 'nu', round(rel(d['nu'], d['nu_ex']), 3))
print('--- continuity in h (partial info, T=0.3): distance of solution from the h=0 EXACT solution, rel L2')
d0 = np.load('../results/fig_h0.npz'); e0, n0 = d0['eta_ex'], d0['nu_ex']
for h in ['0.05', '0.1', '0.2']:
    d = np.load(f'../results/hsweep_{h}.npz'); print(f'  h={h}: eta {rel(d["eta"], e0):.3f}  nu {rel(d["nu"], n0):.3f}   (solver at h=0 itself: eta {rel(d0["eta"], e0):.3f} nu {rel(d0["nu"], n0):.3f})')
print('--- information enlargement at h=0.2, T=0.3: partial-info solution vs full-info solution (same validation paths), rel L2')
dp, df = np.load('../results/hsweep_0.2.npz'), np.load('../results/full_h02_T03.npz')
print('  broker  nu_partial vs nu_full:', round(rel(dp['nu'], df['nu']), 3), ' | trader eta_partial vs eta_full:', round(rel(dp['eta'], df['eta']), 3))
print('  full-info solver vs its exact ref: eta', round(rel(df['eta'], df['eta_ex']), 3), 'nu', round(rel(df['nu'], df['nu_ex']), 3))
# payoff proxy: variance of inventories (risk) and mean |control|
for name, d in [('partial', dp), ('full', df)]: print(f'  {name}: mean|nu| {np.abs(d["nu"]).mean():.4f} mean|eta| {np.abs(d["eta"]).mean():.4f} std QB_T {d["QB"][:, -1].std():.4f}')