"""Appendix figures drawn from the new compute jobs (compute_app_*.py) -> figs_appendix/.

Run from experiments/ once the jobs are done:
    python makefig_appendix_computed.py [figure_name ...]
Also writes figs_appendix/tab_patch_examples.tex.
"""

import os
import sys

import numpy as np

from appendix_style import (COL, CASE2, LIGHT, Y, U, V2, PERT, SEQ_CM, SCAT, LABEL_FS,
                            TICK_FS, FIGS_DIR, RESULTS_DIR, plt, load, style, letter, legend, save,
                            mse, aggregate, align_sign, fit_logx, cbar, wilson)

U_LAYERS = [1, 3, 5, 7, 9, 11]


def have(name):
    return os.path.exists(os.path.join(RESULTS_DIR, name))


def pos_ticklabels(labels):
    """Label each operand's last token and the final prompt token; leave the rest as bare ticks."""
    out = [Y[int(str(l)[1])] if str(l) in ('y1', 'y2', 'y3') else '' for l in labels]
    out[-1] = 'last'
    return out


def layer_position_panels(mats, labels, cmap, vmin, vmax, cb_label, name, xlabels=None):
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 0.8, 6.2), squeeze=False)
    axes = axes[0]
    for i, (ax, M) in enumerate(zip(axes, mats)):
        im = ax.imshow(np.clip(M, vmin, vmax), origin='lower', aspect='auto', cmap=cmap, vmin=vmin,
                       vmax=vmax, interpolation='nearest')
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels(pos_ticklabels(labels), fontsize=TICK_FS - 1)
        ax.set_yticks(range(0, M.shape[0], 3))
        style(ax, xlabel=None if xlabels is None else xlabels[i],
              ylabel='layer' if i == 0 else None)
        if i:
            ax.set_yticklabels([])
        letter(ax, 'abcdef'[i], x=-0.16 if i == 0 else -0.06)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
    cb.set_label(cb_label, fontsize=LABEL_FS - 2)
    cb.ax.tick_params(labelsize=TICK_FS - 2)
    cb.outline.set_visible(False)
    save(fig, name)


# --------------------------------------------------------------------------------------------------
def fig_behaviour():
    A = load('multicompare2_ablate_2digit.npz')
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ks = [int(k) for k in A['ks']]
    acc = [A[f'intact_k{k}'].mean() for k in ks]
    ci = [wilson(A[f'intact_k{k}'].sum(), len(A[f'intact_k{k}'])) for k in ks]
    ax.errorbar(ks, acc, yerr=[[a - lo for a, (lo, _) in zip(acc, ci)],
                               [hi - a for a, (_, hi) in zip(acc, ci)]],
                fmt='-o', color=COL['h14'], lw=2, ms=6, capsize=3)
    ax.set_xscale('log')
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    style(ax, xlabel='number of operands, $k$', ylabel='accuracy')
    fig.tight_layout()
    save(fig, 'app_behaviour')


def patch_categories(P, k):
    """0 = r, 1 = b, 2 = a, 3 = r's leading digit followed by a's last digit, 4 = other."""
    cat = P[f'{k}_cat'].copy()
    cat[cat == 3] = 4
    clean, corr, gen = P[f'{k}_clean'], P[f'{k}_corr'], P[f'{k}_gen']
    for i in np.where(cat == 4)[0]:
        w = int(np.argmax(clean[i]))                   # the winning slot of the clean prompt
        hyb = int(str(corr[i][w])[:-1] + str(clean[i][w])[-1])
        if gen[i] == hyb:
            cat[i] = 3
    return cat


def fig_patch_outcomes():
    P = load('app_patch_outcomes.npz')
    conds = [('none', 'no patch'), ('u_at_y1', f'{U} at {Y[1]}'), ('full_at_y2', f'full at {Y[2]}')]
    cats = [(r'$r$', COL['h14']), (r'$b$', COL['full']), (r'$a$', COL['das']),
            (r'leading digit of $r$, last digit of $a$', COL['pc1']), ('other', LIGHT)]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (k, _) in enumerate(conds):
        cat, left = patch_categories(P, k), 0.0
        for j, (_, col) in enumerate(cats):
            f = float(np.mean(cat == j))
            ax.barh(i, f, left=left, color=col, height=0.62, label=cats[j][0] if i == 0 else None)
            left += f
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([c[1] for c in conds], fontsize=TICK_FS + 1)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    style(ax, xlabel='fraction of generated answers')
    legend(ax, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
    fig.tight_layout()
    save(fig, 'app_patch_outcomes')
    write_patch_table(P)


def write_patch_table(P):
    rows = []
    for k, lab, want, n in (('u_at_y1', r'$\mathbf{u}$ at $y_1$', 0, 3),
                            ('full_at_y2', r'full at $y_2$', 3, 3)):
        ok = np.where(patch_categories(P, k) == want)[0][:n]
        for i in ok:
            c, r = P[f'{k}_clean'][i], P[f'{k}_corr'][i]
            rows.append(f'{lab} & ({c[0]}, {c[1]}) & {max(c)} & ({r[0]}, {r[1]}) & {max(r)} & '
                        f'{P[f"{k}_gen"][i]} \\\\')
    body = '\n'.join(rows)
    tex = ('\\begin{tabular}{lccccc}\n\\toprule\n'
           'intervention & clean $(y_1, y_2)$ & max & corrupted $(y_1, y_2)$ & max & '
           'patched answer \\\\\n\\midrule\n'
           f'{body}\n\\bottomrule\n\\end{{tabular}}\n')
    path = os.path.join(FIGS_DIR, 'tab_patch_examples.tex')
    with open(path, 'w') as f:
        f.write(tex)
    print(f'saved {path}', flush=True)


def fig_trace_k2():
    Z = load('app_trace_k2.npz')
    mats = [Z[f'posrec_{c}'].mean(2) for c in ('n1_larger', 'n2_larger')]
    vmax = max(float(m.max()) for m in mats)
    layer_position_panels(mats, Z['labels'], SEQ_CM, 0, vmax, 'PR', 'app_trace_k2',
                          xlabels=[f'token, {PERT[c]}' for c in ('n1_larger', 'n2_larger')])


def fig_probes_k2():
    Z = load('app_probes_k2.npz')
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    specs = [(Z['r2_logy1'], 0, 1, rf'$R^2$, $\log {Y[1][1:-1]}$'),
             (Z['r2_logy2'], 0, 1, rf'$R^2$, $\log {Y[2][1:-1]}$'),
             (Z['acc_y1gty2'], 0.5, 1, r'accuracy, $y_1>y_2$')]
    for i, (ax, (M, lo, hi, lab)) in enumerate(zip(axes, specs)):
        im = ax.imshow(np.clip(M, lo, hi), origin='lower', aspect='auto', cmap=SEQ_CM, vmin=lo,
                       vmax=hi, interpolation='nearest')
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels(pos_ticklabels(Z['labels']), fontsize=TICK_FS - 1)
        ax.set_yticks(range(0, M.shape[0], 3))
        style(ax, xlabel='token', ylabel='layer' if i == 0 else None)
        if i:
            ax.set_yticklabels([])
        cbar(fig, im, ax, label=lab)
        letter(ax, 'abc'[i], x=-0.16 if i == 0 else -0.06)
    fig.tight_layout()
    save(fig, 'app_probes_k2')


def fig_trace_k3():
    mats, labels, xl = [], None, []
    for c in ('y1', 'y2', 'y3'):
        Z = load(f'app_trace_k3_{c}.npz')
        mats.append(Z['posrec'].mean(2))
        labels = Z['labels']
        xl.append(f'token, {PERT[c]}')
    vmax = max(float(m.max()) for m in mats)
    layer_position_panels(mats, labels, SEQ_CM, 0, vmax, 'PR', 'app_trace_k3', xlabels=xl)


def fig_u_layers():
    R = load('numberreps_2digit.npz')
    rows = {l: load(f'app_u_layers_L{l}.npz') for l in U_LAYERS if have(f'app_u_layers_L{l}.npz')}
    layers = sorted(rows) + [13]

    def metric(l, key):
        if l == 13:
            nr = {'iia_u_n1_larger': 'iia_das_n1_larger_first_number',
                  'iia_full_n1_larger': 'iia_full_n1_larger_first_number',
                  'iia_pc1a_n1_larger': 'iia_pc1_n1_larger_first_number',
                  'iia_full_n2_larger': 'iia_full_n2_larger_second_number',
                  'iia_pc1b_n2_larger': 'iia_pc1b_n2_larger_second_number'}[key]
            return R[nr]
        return rows[l][key]

    fig = plt.figure(figsize=(18, 9.4))
    gs = fig.add_gridspec(2, 6, hspace=0.42, wspace=0.9)
    for s, (sl, arms) in enumerate(((slice(0, 3), [('iia_full_n1_larger', 'full', COL['full']),
                                                   ('iia_u_n1_larger', U, COL['das']),
                                                   ('iia_pc1a_n1_larger', 'PC1', COL['pc1'])]),
                                    (slice(3, 6), [('iia_full_n2_larger', 'full', COL['full']),
                                                   ('iia_pc1b_n2_larger', V2, COL['pc1'])]))):
        ax = fig.add_subplot(gs[0, sl])
        for key, lab, col in arms:
            ms = [mse(metric(l, key)) for l in layers]
            ax.errorbar(layers, [m for m, _ in ms], yerr=[e for _, e in ms], fmt='-o', color=col,
                        lw=2, ms=5, capsize=2, label=lab)
        ax.set_xticks(layers)
        style(ax, xlabel=f'layer, patched at {Y[1 + s]}', ylabel='IIA')
        legend(ax, loc='best')
        letter(ax, 'ab'[s], x=-0.13)
    show = [l for l in (3, 7) if l in rows] + [13]
    for i, l in enumerate(show):
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        if l == 13:
            y1, c = R['a_vals'], R['das_comp_a']
        else:
            y1, c = rows[l]['a_vals'], rows[l]['comp_u']
        c = align_sign(np.asarray(c, dtype=np.float64), y1)
        xs, m, se = aggregate(y1, c)
        ax.errorbar(xs, m, yerr=se, fmt='o', ms=3.5, color=COL['das'], elinewidth=0.8)
        p, q = fit_logx(y1, c)
        xx = np.linspace(y1.min(), y1.max(), 200)
        ax.plot(xx, p * np.log(xx) + q, ':', color='0.15', lw=1.6)
        style(ax, xlabel=Y[1], ylabel=f'component along {U}, L{l}')
        letter(ax, 'cde'[i], x=-0.28)
    save(fig, 'app_u_layers')


def fig_v2_shortcut():
    Z = load('app_v2_shortcut.npz')
    y1, y2 = Z['a_vals'], Z['b_vals']
    big2 = y2 > y1
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.9), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    for i, (key, lab) in enumerate((('comp_das', f'DAS component at {Y[2]}'),
                                    ('comp_pc1b', f'component along {V2}'))):
        ax = axes[i]
        c = align_sign(Z[key], np.log(y2))
        for m, col, l in ((~big2, CASE2['n1_larger'], r'$y_1>y_2$'),
                          (big2, CASE2['n2_larger'], r'$y_2>y_1$')):
            ax.scatter(y2[m], c[m], color=col, label=l, **SCAT)
        style(ax, xlabel=Y[2], ylabel=lab)
        if i == 0:
            legend(ax, loc='best', markerscale=2.5)
        letter(ax, 'ab'[i])
    ax = axes[2]
    arms = [('full', 'full', COL['full']), ('das', 'DAS', COL['h14']), ('pc1b', V2, COL['pc1'])]
    ms = [mse(Z[f'iia_{a}']) for a, _, _ in arms]
    ax.bar(range(3), [m for m, _ in ms], 0.65, yerr=[e for _, e in ms], capsize=3,
           color=[c for _, _, c in arms], alpha=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([a[1] for a in arms], fontsize=LABEL_FS)
    style(ax, ylabel=f'IIA, {PERT["n2_larger"]}')
    letter(ax, 'c')
    fig.tight_layout()
    save(fig, 'app_v2_shortcut')


FIGURES = {f.__name__[4:]: f for f in (fig_behaviour, fig_patch_outcomes, fig_trace_k2,
                                         fig_probes_k2, fig_trace_k3, fig_u_layers,
                                         fig_v2_shortcut)}
NEEDS = {'behaviour': [], 'patch_outcomes': ['app_patch_outcomes.npz'],
         'trace_k2': ['app_trace_k2.npz'], 'probes_k2': ['app_probes_k2.npz'],
         'trace_k3': [f'app_trace_k3_{c}.npz' for c in ('y1', 'y2', 'y3')],
         'u_layers': [], 'v2_shortcut': ['app_v2_shortcut.npz']}

if __name__ == '__main__':
    for n in sys.argv[1:] or list(FIGURES):
        missing = [f for f in NEEDS[n] if not have(f)]
        if missing:
            print(f'skip {n}: missing {missing}', flush=True)
            continue
        FIGURES[n]()
