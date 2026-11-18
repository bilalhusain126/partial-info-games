"""Summarise the Narval batches from results/: scaling curve at lam = 0, seed agreement and checks for the private case, private vs
full-information trading intensity, and the final errors of every run with an exact reference. Writes results/summary.md and
results/scaling.png. Usage, from scripts/: python3 analyse_batches.py"""
import os, re, glob, numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'); out = []
def last(pattern, log):
    m = [l for l in open(log) if re.match(pattern, l)]; return m[-1].strip() if m else None
def avg_errs(log):
    l = last(r'^it', log)
    if not l: return None
    m = re.findall(r'avg\(\d+\): (.*?) \(', l) or re.findall(r'rel L2 (?:vs ref|vs closed form|error vs exact): (.*?)(?: \||$| \()', l)
    return dict(re.findall(r'(u\d|broker|avg flow|trader_i|eta|nu) ([\d.]+)', m[0])) if m else None
def rel(a, b): return float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))

out.append('# Batch summary\n')
# 1. scaling at lam = 0
rows = []
for log in sorted(glob.glob(f'{R}/pt_scale_*.log'), key=lambda f: int(re.search(r'_(\d+)\.log', f).group(1))):
    m = int(re.search(r'_(\d+)\.log', log).group(1)); e = avg_errs(log); foc = re.findall(r'FOC residual player (\d).*?rel L2 ([\d.]+)', open(log).read())
    rows.append((m, e, dict(foc)))
if rows:
    out.append('## Scaling at lam = 0 (exact reference), control variate on, 128 epochs per iteration\n')
    out.append('| paths | trader 1 | trader 2 | FOC residual 1 | FOC residual 2 |\n|---|---|---|---|---|')
    for m, e, foc in rows: out.append(f"| {m} | {e['u0'] if e else '-'} | {e['u1'] if e else '-'} | {foc.get('0', '-')} | {foc.get('1', '-')} |")
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        ms = [r[0] for r in rows if r[1]]; e0 = [float(r[1]['u0']) for r in rows if r[1]]; e1 = [float(r[1]['u1']) for r in rows if r[1]]
        fig, ax = plt.subplots(figsize=(4.2, 3.2)); ax.loglog(ms, e0, 'o-', color='C0', label='trader 1'); ax.loglog(ms, e1, 's-', color='C1', label='trader 2')
        ref = e0[0] * np.sqrt(ms[0] / np.array(ms)); ax.loglog(ms, ref, 'k--', lw=1, label='1/sqrt(M)'); ax.set_xlabel('paths M'); ax.set_ylabel('relative L2 error'); ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(); fig.savefig(f'{R}/scaling.png', dpi=130); out.append('\nFigure: results/scaling.png\n')
    except Exception as ex: out.append(f'\n(no figure: {ex})\n')
# 2. private case, lam = 2: seeds, checks, agreement
seeds = sorted(glob.glob(f'{R}/pt_priv2_cv_s*.npz'))
if seeds:
    out.append('\n## Private two-way case, lam = 2, control variate on\n')
    out.append('| seed | final change | FOC residual 1 | FOC residual 2 | exploitability 1 | exploitability 2 |\n|---|---|---|---|---|---|')
    for f in seeds:
        log = f.replace('.npz', '.log'); txt = open(log).read() if os.path.exists(log) else ''
        chg = re.findall(r'change ([\d.]+)', txt); foc = dict(re.findall(r'FOC residual player (\d).*?rel L2 ([\d.]+)', txt)); ex = dict(re.findall(r'player (\d): exploitability \S+ relative to \|?cost\|? [\d.]+ \(([+-][\d.]+%)\)', txt))
        out.append(f"| {re.search(r'_s(\d)', f).group(1)} | {chg[-1] if chg else '-'} | {foc.get('0', '-')} | {foc.get('1', '-')} | {ex.get('0', '-')} | {ex.get('1', '-')} |")
    if len(seeds) > 1:
        d = [np.load(f) for f in seeds]
        out.append('\nSeed agreement (rel L2 between seeds on the common validation paths): ' + ', '.join(f"s{re.search(r'_s(\d)', seeds[a]).group(1)}-s{re.search(r'_s(\d)', seeds[b]).group(1)}: trader1 {rel(d[a]['nu1'], d[b]['nu1']):.3f}, trader2 {rel(d[a]['nu2'], d[b]['nu2']):.3f}" for a in range(len(d)) for b in range(a + 1, len(d))))
    full = f'{R}/pt_full2_cv.npz'
    if os.path.exists(full):
        fu = np.load(full); pr = np.load(seeds[0])
        out.append(f"\nTrading intensity, mean |nu| over paths and time: private {np.abs(pr['nu1']).mean():.3f}, full information {np.abs(fu['nu1']).mean():.3f} (exact full-info {np.abs(fu['nu_ex'][:, 0]).mean():.3f}); rel L2 between private and full-information controls on the same paths {rel(pr['nu1'], fu['nu1']):.3f}")
# 3. continuity in lam
cont = sorted(glob.glob(f'{R}/pt_priv_lam*.npz'))
if cont and os.path.exists(f'{R}/pt_scale_16384.npz'):
    ex0 = np.load(f'{R}/pt_scale_16384.npz')
    out.append('\n## Continuity in the impact parameter (private case vs the exact lam = 0 solution on the same validation paths)\n')
    out.append('| lam | rel L2 to exact lam=0, trader 1 | trader 2 |\n|---|---|---|')
    out.append(f"| 0 (learned) | {rel(ex0['nu1'], ex0['nu_ex'][:, 0]):.3f} | {rel(ex0['nu2'], ex0['nu_ex'][:, 1]):.3f} |")
    for f in cont:
        d = np.load(f); lam = re.search(r'lam([\d.]+)', f).group(1); out.append(f"| {lam} | {rel(d['nu1'], ex0['nu_ex'][:, 0]):.3f} | {rel(d['nu2'], ex0['nu_ex'][:, 1]):.3f} |")
# 4. every other run with a reference
out.append('\n## Other runs (final averaged relative L2 vs exact, from the logs)\n')
for log in sorted(glob.glob(f'{R}/*.log')):
    name = os.path.basename(log)[:-4]
    if name.startswith(('pt_scale', 'pt_priv2_cv_s', 'pt_priv_lam')): continue
    e = avg_errs(log); l = last(r'^it', log)
    if e: out.append(f"- {name}: " + ', '.join(f'{k} {v}' for k, v in e.items()))
    elif l: out.append(f"- {name}: {l[:110]}")
    foc = re.findall(r'FOC residual player (\d).*?rel L2 ([\d.]+)', open(log).read()); ex = re.findall(r'player (\d): exploitability.*?\(([+-][\d.]+%)\)', open(log).read())
    if foc or ex: out.append('    checks: ' + ' '.join(f'FOC{p} {v}' for p, v in foc) + ' ' + ' '.join(f'exploit{p} {v}' for p, v in ex))
open(f'{R}/summary.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
