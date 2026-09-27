"""Appendix figures that need no new model computation: every panel is drawn from results/*.npz.

Run from experiments/:  python makefig_appendix.py [figure_name ...]
With no arguments every figure is drawn; figures land in figs_appendix/<name>.pdf.
"""

import sys

import numpy as np
from matplotlib.lines import Line2D

from appendix_style import (COL, NCOL, CASE2, ORDER_COL, ORDER_LAB, LIGHT, Y, U, V1, V2, V3, PERT,
                            SEQ_CM, DIV_CM, RF_CM, VAL_CM, SCAT, LABEL_FS, TICK_FS, ANNOT_FS, plt,
                            load, style, letter, legend, save, mse, aggregate, align_sign, fit_logx,
                            grouped_bars, heat, cbar)

CASES2 = ['n1_larger', 'n2_larger']
CASES3 = ['y1', 'y2', 'y3']
GROUPS3 = ['s1gts2', 's1gts3', 's2gts3']
D_FF = 18944
HATCH = '//'
_cache = {}


def get(name):
    if name not in _cache:
        _cache[name] = load(name)
    return _cache[name]


def zscore(x):
    return (x - x.mean()) / x.std()


def errline(ax, x, y, color, label=None, ms=4, **kw):
    xs, m, se = aggregate(x, y, **kw)
    ax.errorbar(xs, m, yerr=se, fmt='o', ms=ms, color=color, ecolor=color, elinewidth=0.8,
                capsize=0, alpha=0.9, label=label)
    return xs, m


# --------------------------------------------------------------------------------------------------
# Individual number representations
# --------------------------------------------------------------------------------------------------
def fig_u_vs_pc1():
    R = get('numberreps_2digit.npz')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    names, cols = ['full', 'das', 'pc1'], [COL['full'], COL['das'], COL['pc1']]
    labs = ['full', U, 'PC1']
    for ax, met, lab, s in ((axes[0, 0], 'iia', 'IIA', 'a'), (axes[0, 1], 'posrec', 'PR', 'b')):
        ms = [mse(R[f'{met}_{n}_n1_larger_first_number']) for n in names]
        ax.bar(range(3), [m for m, _ in ms], 0.65, yerr=[e for _, e in ms], capsize=3, color=cols,
               alpha=0.9)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labs, fontsize=LABEL_FS)
        ax.axhline(0, color='0.6', lw=0.8)
        style(ax, ylabel=lab)
        letter(ax, s)

    ax = axes[1, 0]
    y1 = R['a_vals']
    cu = zscore(align_sign(R['das_comp_a'], y1))
    cp = zscore(align_sign(R['pca3_coords_a'][:, 0].astype(np.float64), y1))
    errline(ax, y1, cu, COL['das'], label=U)
    errline(ax, y1, cp, COL['pc1'], label='PC1')
    style(ax, xlabel=Y[1], ylabel='standardized component')
    legend(ax, loc='lower right')
    letter(ax, 'c')

    ax = axes[1, 1]
    X = R['resid_a'].astype(np.float64)
    X -= X.mean(0)
    tot = (X ** 2).sum() / len(X)
    evr_u = float(np.var(X @ R['das_dir'].astype(np.float64)) / tot)
    spec = R['pca_evr_spectrum_a']
    ax.bar(np.arange(1, len(spec) + 1), spec, 0.7,
           color=[COL['pc1']] + [LIGHT] * (len(spec) - 1))
    ax.axhline(evr_u, color=COL['das'], lw=2.2, ls='--', label=U)
    ax.set_xticks([1, 5, 10, 15, 20])
    style(ax, xlabel='principal component', ylabel='variance explained')
    legend(ax, loc='upper right')
    letter(ax, 'd')
    fig.tight_layout()
    save(fig, 'app_u_vs_pc1')


def fig_three_digit():
    R = get('numberreps_3digit.npz')
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    bins = np.arange(100, 1001, 25)

    ax = axes[0]
    y1 = R['a_vals']
    cu = align_sign(R['das_comp_a'], y1)
    errline(ax, y1, cu, COL['das'], bins=bins)
    p, q = fit_logx(y1, cu)
    xx = np.linspace(y1.min(), y1.max(), 200)
    ax.plot(xx, p * np.log(xx) + q, ':', color='0.15', lw=1.6)
    style(ax, xlabel=Y[1], ylabel=f'component along {U}')
    letter(ax, 'a')

    ax = axes[1]
    y2 = R['b_vals']
    Xb = R['resid_b'].astype(np.float64)
    cv = align_sign((Xb - Xb.mean(0)) @ R['pc1b_dir'].astype(np.float64), y2)
    errline(ax, y2, cv, COL['pc1'], bins=bins)
    p, q = fit_logx(y2, cv)
    ax.plot(xx, p * np.log(xx) + q, ':', color='0.15', lw=1.6)
    style(ax, xlabel=Y[2], ylabel=f'component along {V2}')
    letter(ax, 'b')

    ax = axes[2]
    bars = [('iia_full_n1_larger_first_number', 'full', COL['full']),
            ('iia_das_n1_larger_first_number', U, COL['das']),
            ('iia_pc1_n1_larger_first_number', 'PC1', COL['pc1']),
            ('iia_full_n2_larger_second_number', 'full', COL['full']),
            ('iia_pc1b_n2_larger_second_number', V2, COL['pc1'])]
    xs = [0, 1, 2, 3.6, 4.6]
    for x, (k, _, c) in zip(xs, bars):
        m, e = mse(R[k])
        ax.bar(x, m, 0.75, yerr=e, capsize=3, color=c, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[1] for b in bars], fontsize=TICK_FS + 1)
    for xc, j in ((1.0, 1), (4.1, 2)):
        ax.text(xc, -0.2, Y[j], transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=LABEL_FS)
    style(ax, ylabel='IIA')
    letter(ax, 'c')
    fig.tight_layout()
    save(fig, 'app_three_digit')


def mean_path(coords, values, bins=14, smooth=3):
    order = np.argsort(values, kind='stable')
    groups = np.array_split(order, bins)
    pts = np.array([coords[g].mean(0) for g in groups])
    if smooth and smooth > 1:
        k = np.ones(smooth) / smooth
        pad = np.pad(pts, ((smooth // 2, smooth // 2), (0, 0)), mode='edge')
        pts = np.stack([np.convolve(pad[:, d], k, mode='valid') for d in range(pts.shape[1])], 1)
    return pts


def manifold_panel(ax, coords, values, direction, color, label, s=None):
    """A cloud in two principal components, coloured by `values`, with its equal-count mean path and
    a causal direction drawn as a unit arrow from the cloud centre. Returns the scatter handle."""
    kw = dict(SCAT) if s is None else dict(s=s, edgecolors='none', rasterized=True)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=VAL_CM, zorder=2, **kw)
    path = mean_path(coords, values)
    ax.plot(path[:, 0], path[:, 1], '-', color='0.2', lw=2.4, zorder=3)
    ax.plot(path[:, 0], path[:, 1], 'o', color='0.2', ms=4, zorder=3)
    span = np.ptp(coords, axis=0)
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d / span) * 0.34            # a third of the cloud, in plotted units
    o = coords.mean(0)
    ax.annotate('', xy=o + d, xytext=o, zorder=5,
                arrowprops=dict(arrowstyle='-|>,head_width=0.45,head_length=0.9', lw=3.2,
                                color=color, shrinkA=0, shrinkB=0))
    from matplotlib import patheffects
    ax.text(*(o + 1.22 * d), label, color=color, fontsize=LABEL_FS + 2, ha='center', va='center',
            fontweight='bold', zorder=6,
            path_effects=[patheffects.withStroke(linewidth=4, foreground='white')])
    return sc


def fig_u_manifold():
    R = get('numberreps_2digit.npz')
    y1, c = R['a_vals'], R['pca3_coords_a'].astype(np.float64)
    vals, first = np.unique(y1, return_index=True)          # one residual per value at y1's position
    pts = c[first][:, [0, 2]]
    u = R['das_dir'].astype(np.float64)
    if np.corrcoef(R['das_comp_a'], y1)[0, 1] < 0:
        u = -u
    cos = (R['pca3_components_a'].astype(np.float64) @ u)[[0, 2]]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    sc = manifold_panel(ax, pts, vals, cos, COL['das'], U, s=34)
    style(ax, xlabel='PC1', ylabel='PC3')
    cbar(fig, sc, ax, label=Y[1])
    letter(ax, 'a')

    ax = axes[1]
    cu = align_sign(R['das_comp_a'], y1)
    errline(ax, y1, cu, COL['das'])
    p, q = fit_logx(y1, cu)
    xx = np.linspace(y1.min(), y1.max(), 200)
    ax.plot(xx, p * np.log(xx) + q, ':', color='0.15', lw=1.6)
    style(ax, xlabel=Y[1], ylabel=f'component along {U}')
    letter(ax, 'b')
    fig.tight_layout()
    save(fig, 'app_u_manifold')


def fig_k3_manifolds():
    from sklearn.decomposition import PCA
    Dd = get('multicompare_dirs_2digit.npz')
    y = Dd['cloud_y']
    items = [('cloud_head14', 'v1', 1, 'H14 output'), ('cloud_head18', 'v2', 2, 'H18 output'),
             ('cloud_l13', 'v3', 3, 'L13 residual')]
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 4.9))
    for i, (ax, (ck_, dk, j, site)) in enumerate(zip(axes, items)):
        X = Dd[ck_].astype(np.float64)
        pca = PCA(n_components=2).fit(X)
        coords = pca.transform(X)
        v = Dd[dk].astype(np.float64)
        v = v / np.linalg.norm(v)
        if np.corrcoef((X - X.mean(0)) @ v, y[:, j - 1])[0, 1] < 0:
            v = -v
        sc = manifold_panel(ax, coords, y[:, j - 1], pca.components_ @ v, NCOL[j],
                            [V1, V2, V3][j - 1])
        ev = pca.explained_variance_ratio_
        style(ax, xlabel=f'PC1 of {site}', ylabel=f'PC2 of {site}')
        cbar(fig, sc, ax, label=Y[j])
        print(f'  {dk}: top-2 EVR {ev.round(3).tolist()}, '
              f'norm in plane {np.linalg.norm(pca.components_ @ v):.3f}', flush=True)
        letter(ax, 'abc'[i])
    fig.tight_layout()
    save(fig, 'app_k3_manifolds')


def fig_transport():
    S = get('sharedrep_2digit.npz')
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    ax = axes[0]
    c, y1 = S['h14_pca_coords'], S['a_vals']
    sc = ax.scatter(c[:, 0], c[:, 1], c=y1, cmap=VAL_CM, **SCAT)
    path = mean_path(c[:, :2], y1)
    ax.plot(path[:, 0], path[:, 1], '-', color='0.2', lw=2.4)
    ax.plot(path[:, 0], path[:, 1], 'o', color='0.2', ms=4)
    v = S['h14_das_dir'].astype(np.float64)
    if np.corrcoef(S['h14_das_comp'], y1)[0, 1] < 0:
        v = -v                                        # point the arrow toward larger y1
    d = S['h14_pca_components'][:2].astype(np.float64) @ v
    d = d / np.linalg.norm(d) * 0.34 * np.ptp(c[:, 0])
    o = c[:, :2].mean(0)
    ax.annotate('', xy=o + d, xytext=o, zorder=5,
                arrowprops=dict(arrowstyle='-|>,head_width=0.45,head_length=0.9', lw=3.2,
                                color=COL['h14'], shrinkA=0, shrinkB=0))
    ax.text(*(o + 0.35 * d + np.array([0.0, -0.8])), V1, color=COL["h14"], fontsize=LABEL_FS + 2,
            ha="center", va="center",
            fontweight='bold', zorder=6)
    style(ax, xlabel='PC1 of H14 output', ylabel='PC2 of H14 output')
    cbar(fig, sc, ax, label=Y[1])
    letter(ax, 'a')

    ax = axes[1]
    errline(ax, y1, S['h14_attn_to_n1'], COL['h14'])
    style(ax, xlabel=Y[1], ylabel=f'H14 attention to {Y[1]}')
    letter(ax, 'b')

    ax = axes[2]
    errline(ax, y1, align_sign(S['h14_das_comp'].astype(np.float64), y1), COL['h14'])
    style(ax, xlabel=Y[1], ylabel=f'component along {V1}')
    letter(ax, 'c')
    fig.tight_layout()
    save(fig, 'app_transport')


# --------------------------------------------------------------------------------------------------
# Shared representation
# --------------------------------------------------------------------------------------------------
def fig_shared_probes():
    S = get('sharedrep_2digit.npz')
    te = S['probe_test_idx']
    ys = {1: S['a_vals'][te], 2: S['b_vals'][te]}
    comps = {1: S['probe_logA_pre_comp'][te], 2: S['probe_logB_pre_comp'][te]}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), sharey='row')
    for i, j in enumerate((1, 2)):
        cj = align_sign(comps[j], np.log(ys[j]))
        for k, t in enumerate((1, 2)):
            ax = axes[i, k]
            ax.scatter(ys[t], cj, color=NCOL[j], **SCAT)
            style(ax, xlabel=Y[t] if i == 1 else None,
                  ylabel=rf'component along $\mathbf{{p}}_{j}$' if k == 0 else None)
            letter(ax, 'abcd'[2 * i + k])
    fig.tight_layout()
    save(fig, 'app_shared_probes')


def fig_shared_posrec():
    S = get('sharedrep_2digit.npz')
    conds = [('L13_full_atn2', 'L13 full', False), ('L13_PC1_atn2', V2, False),
             ('H14_DAS', V1, False), ('H14_DAS_and_L13_PC1_atn2', f'{V1} & {V2}', True),
             ('L14_uv_plane_preMLP_atn2', f'{V1}, {V2} plane', True),
             ('L14_full_preMLP_atn2', 'L14 full', False)]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    w = 0.38
    for si, (pre, case) in enumerate((('', 'n1_larger'), ('alt_', 'n2_larger'))):
        for gi, (k, _, joint) in enumerate(conds):
            m, e = mse(S[f'{pre}cond_posrec_{k}'])
            ax.bar(gi + (si - 0.5) * w, m, w, yerr=e, capsize=2.5, color=CASE2[case], alpha=0.9,
                   hatch=HATCH if joint else None, edgecolor='white', linewidth=0.6,
                   label=PERT[case] if gi == 0 else None)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[1] for c in conds], fontsize=TICK_FS, rotation=20, ha='right')
    ax.axhline(0, color='0.6', lw=0.8)
    style(ax, xlabel='patched component', ylabel='PR')
    legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, 'app_shared_posrec')


# --------------------------------------------------------------------------------------------------
# Comparator neurons (two numbers)
# --------------------------------------------------------------------------------------------------
def fig_mlp_freeze():
    C = get('comparator_data_2digit.npz')
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    groups = ['none', 'MLP14', 'MLP15', 'MLP16', 'attn15', 'attn16', 'MLP14+15']
    vals = {c: {g: C[f'mod_iia_{c}_{g}'] for g in groups} for c in CASES2}
    grouped_bars(ax, groups, CASES2, vals, [CASE2[c] for c in CASES2],
                 labels=[PERT[c] for c in CASES2])
    ax.set_xticklabels(groups, rotation=20, ha='right')
    style(ax, xlabel='frozen sub-block', ylabel='IIA')
    legend(ax, loc='upper right', bbox_to_anchor=(1.0, 1.12), ncol=2)
    letter(ax, 'a', x=-0.12)

    ax = axes[1]
    groups = ['none', 'MLP14', 'MLP14+MLP15', 'MLP14+MLP15+MLP16']
    vals = {c: {g: C[f'inj_iia_nc_{c}_{g}'] for g in groups} for c in CASES2}
    grouped_bars(ax, groups, CASES2, vals, [CASE2[c] for c in CASES2])
    ax.set_xticklabels(['none', 'MLP14', 'MLP14, 15', 'MLP14, 15, 16'], rotation=20, ha='right')
    style(ax, xlabel='injected MLP outputs', ylabel='IIA')
    letter(ax, 'b')
    fig.tight_layout()
    save(fig, 'app_mlp_freeze')


def fig_attribution():
    C = get('comparator_data_2digit.npz')
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.0), gridspec_kw=dict(width_ratios=[1, 1.1, 1]))
    ax = axes[0]
    s1 = np.concatenate([C['attr_n1_larger_L14'], C['attr_n1_larger_L15']])
    s2 = np.concatenate([C['attr_n2_larger_L14'], C['attr_n2_larger_L15']])
    ov = np.asarray(C['overlap_k20'])
    ax.scatter(s1, s2, s=5, color=LIGHT, alpha=0.5, edgecolors='none', rasterized=True,
               label='all neurons')
    ax.scatter(s1[ov], s2[ov], s=46, color=COL['h14'], edgecolors='white', linewidths=0.6,
               label='shared top 20', zorder=3)
    allv = np.abs(np.concatenate([s1, s2]))
    lt = float(np.percentile(allv, 99))
    dec = 10.0 ** np.arange(np.ceil(np.log10(lt)) + 1, np.floor(np.log10(allv.max())) + 1)
    ticks = sorted(-dec) + [0.0] + list(dec)
    for set_scale, set_ticks in ((ax.set_xscale, ax.set_xticks), (ax.set_yscale, ax.set_yticks)):
        set_scale('symlog', linthresh=lt)
        set_ticks(ticks)
    ax.minorticks_off()
    style(ax, xlabel=f'attribution, {PERT["n1_larger"]}', ylabel=f'attribution, {PERT["n2_larger"]}')
    legend(ax, loc='upper left', markerscale=1.2)
    letter(ax, 'a')

    ax = axes[1]
    ks = [1, 3, 10, 30, 100, 300, 1000, 3000]
    for c in CASES2:
        top = [C[f'topk_iia_{c}_top_k{k}'].mean() for k in ks]
        rnd = [0.5 * (C[f'topk_iia_{c}_rand0_k{k}'].mean() + C[f'topk_iia_{c}_rand1_k{k}'].mean())
               for k in ks]
        ax.plot(ks, top, '-o', color=CASE2[c], lw=2, ms=5)
        ax.plot(ks, rnd, '--', color=CASE2[c], lw=1.6, alpha=0.8)
    ax.set_xscale('log')
    style(ax, xlabel='neurons frozen, $k$', ylabel='IIA')
    handles = [Line2D([], [], color=CASE2[c], lw=2, label=PERT[c]) for c in CASES2] + \
              [Line2D([], [], color='0.3', lw=2, ls='-', marker='o', label='top $k$'),
               Line2D([], [], color='0.3', lw=1.6, ls='--', label='random $k$')]
    legend(ax, handles=handles, loc='lower left')
    letter(ax, 'b')

    ax = axes[2]
    groups = ['none', 'shared', 'own', 'random']
    vals = {c: {'none': C[f'mod_iia_{c}_none'], 'shared': C[f'ov20_iia_{c}_shared'],
                'own': C[f'ov20_iia_{c}_own'],
                'random': np.concatenate([C[f'ov20_iia_{c}_rand0'], C[f'ov20_iia_{c}_rand1']])}
            for c in CASES2}
    grouped_bars(ax, groups, CASES2, vals, [CASE2[c] for c in CASES2])
    ax.set_xticklabels(['none', 'shared 12', 'own top 20', 'random 12'], rotation=20, ha='right')
    style(ax, xlabel='frozen neurons', ylabel='IIA')
    letter(ax, 'c')
    fig.tight_layout()
    save(fig, 'app_attribution')


def rf_field(C, pooled):
    layer = 14 if pooled < D_FF else 15
    within = pooled if layer == 14 else pooled - D_FF
    col = int(np.where(C[f'cloud_idx_L{layer}'] == within)[0][0])
    g = C['rf_grid']
    a = C[f'rf_acts_L{layer}'][:, col].reshape(len(g), len(g)).T    # pairs are y1-major
    return a / (np.abs(a).max() + 1e-9), g, layer, within


def rf_grid_figure(rows, row_labels, name):
    C = get('comparator_data_2digit.npz')
    nr, nc = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(nr, nc, figsize=(2.75 * nc + 1.2, 2.75 * nr), squeeze=False)
    im = None
    for i, row in enumerate(rows):
        for j in range(nc):
            ax = axes[i, j]
            if j >= len(row):
                ax.axis('off')
                continue
            a, g, layer, within = rf_field(C, row[j])
            ext = [g[0], g[-1], g[0], g[-1]]
            im = ax.imshow(a, origin='lower', extent=ext, cmap=RF_CM, vmin=-1, vmax=1,
                           interpolation='nearest', aspect='equal')
            ax.plot(ext[:2], ext[:2], '--', color='0.3', lw=0.9)
            ax.text(0.04, 0.96, f'L{layer} #{within}', transform=ax.transAxes, ha='left', va='top',
                    fontsize=ANNOT_FS, color='0.1')
            ax.set_xticks([20, 50, 80])
            ax.set_yticks([20, 50, 80])
            style(ax, xlabel=Y[1] if i == nr - 1 else None, ylabel=Y[2] if j == 0 else None)
            if i != nr - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
        if row_labels is not None:
            axes[i, 0].text(-0.62, 0.5, row_labels[i], transform=axes[i, 0].transAxes, rotation=90,
                            ha='center', va='center', fontsize=LABEL_FS)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cb.set_label('activation / peak', fontsize=LABEL_FS - 2)
    cb.ax.tick_params(labelsize=TICK_FS - 2)
    cb.outline.set_visible(False)
    save(fig, name)


def _worst_rank(C, pooled):
    layer = 14 if pooled < D_FF else 15
    within = pooled if layer == 14 else pooled - D_FF
    i = int(np.where((C['prof_layer'] == layer) & (C['prof_within'] == within))[0][0])
    return max(C['prof_rank_n1_larger'][i], C['prof_rank_n2_larger'][i])


def fig_rf_shared():
    C = get('comparator_data_2digit.npz')
    ov = sorted(C['overlap_k20'].tolist(), key=lambda p: _worst_rank(C, p))
    rows = [[p for p in ov if p < D_FF], [p for p in ov if p >= D_FF]]
    rf_grid_figure(rows, None, 'app_rf_shared')


def fig_rf_exclusive():
    C = get('comparator_data_2digit.npz')
    ov = set(C['overlap_k20'].tolist())
    rows = [[int(p) for p in C[f'top20_{c}'] if int(p) not in ov][:6] for c in CASES2]
    rf_grid_figure(rows, [PERT[c] for c in CASES2], 'app_rf_exclusive')


def fig_connectivity():
    C = get('comparator_data_2digit.npz')
    ov = C['overlap_k20']
    r14 = [int(np.where(C['cloud_idx_L14'] == p)[0][0]) for p in ov if p < D_FF]
    r15 = [int(np.where(C['cloud_idx_L15'] == p - D_FF)[0][0]) for p in ov if p >= D_FF]
    yl = [f'#{C["cloud_idx_L14"][r]}' for r in r14]
    xl = [f'#{C["cloud_idx_L15"][c]}' for c in r15]
    mats = [(C['edge_n1_larger'], f'causal edge (SD), {PERT["n1_larger"]}'),
            (C['edge_n2_larger'], f'causal edge (SD), {PERT["n2_larger"]}'),
            (C['path_gate'], 'virtual weight (gate)')]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    for i, (ax, (M, lab)) in enumerate(zip(axes, mats)):
        Ms = M[np.ix_(r14, r15)]
        v = float(np.abs(Ms).max())
        im = heat(ax, Ms, DIV_CM, -v, v, xt=xl, yt=yl)
        ax.set_xticklabels(xl, rotation=45, ha='right', fontsize=TICK_FS - 2)
        ax.set_yticklabels(yl, fontsize=TICK_FS - 2)
        style(ax, xlabel='MLP15 neuron', ylabel='MLP14 neuron' if i == 0 else None)
        cbar(fig, im, ax, label=lab)
        letter(ax, 'abc'[i], x=-0.28 if i == 0 else -0.12)
    fig.tight_layout()
    save(fig, 'app_connectivity')


def fig_l15_direction():
    C = get('comparator_data_2digit.npz')
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    ax = axes[0]
    other = {'n1_larger': 'n2_larger', 'n2_larger': 'n1_larger'}
    series = ['same', 'other', 'full']
    vals = {'same': {c: C[f'das15_iia_{c}_from_{c}_k1'] for c in CASES2},
            'other': {c: C[f'das15_iia_{c}_from_{other[c]}_k1'] for c in CASES2},
            'full': {c: C[f'das15_iia_{c}_full'] for c in CASES2}}
    grouped_bars(ax, CASES2, series, vals, [COL['h14'], '#8CC7C7', COL['full']],
                 labels=['fitted, same case', 'fitted, other case', 'full rank'])
    ax.set_xticklabels([PERT[c] for c in CASES2])
    style(ax, xlabel='evaluated with', ylabel='IIA')
    legend(ax, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=TICK_FS - 2)
    letter(ax, 'a')

    shared15 = [int(p) - D_FF for p in C['overlap_k20'] if p >= D_FF]
    for i, c in enumerate(CASES2):
        ax = axes[i + 1]
        rho = C[f'dasfrac_L15_{c}_k1']
        ax.hist(rho, bins=70, color=LIGHT, log=True)
        for j in shared15:
            ax.axvline(rho[j], color=COL['h14'], lw=1.8)
        style(ax, xlabel=rf'$\rho_j$, direction fitted with {PERT[c]}',
              ylabel='MLP15 neurons' if i == 0 else None)
        letter(ax, 'bc'[i])
    fig.tight_layout()
    save(fig, 'app_l15_direction')


# --------------------------------------------------------------------------------------------------
# Three numbers
# --------------------------------------------------------------------------------------------------
def fig_k3_heads():
    H = get('multicompare_heads_2digit.npz')
    means = np.stack([H[f'posrec_{g}'].mean(1) for g in GROUPS3])      # (3, 28)
    top = np.argsort(-means.max(0))[:8]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.8 / 3
    site = {'s1gts2': Y[2], 's1gts3': Y[3], 's2gts3': Y[3]}
    for gi, g in enumerate(GROUPS3):
        ms = [mse(H[f'posrec_{g}'][h]) for h in top]
        xp = np.arange(len(top)) + (gi - 1) * w
        ax.bar(xp, [m for m, _ in ms], w, yerr=[e for _, e in ms], capsize=2, color=ORDER_COL[gi],
               alpha=0.9, label=f'{ORDER_LAB[g]}, at {site[g]}')
        ax.axhline(H[f'base_{g}_L13_posrec'].mean(), color=ORDER_COL[gi], ls='--', lw=1.3)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([f'H{h}' for h in top])
    ax.axhline(0, color='0.6', lw=0.8)
    style(ax, xlabel='layer-14 head', ylabel='PR')
    legend(ax, loc='upper right')
    fig.tight_layout()
    save(fig, 'app_k3_heads')


def fig_k3_directions():
    Dd = get('multicompare_dirs_2digit.npz')
    Uu = get('multicompare_ucos_2digit.npz')
    y = Dd['cloud_y']

    def comp(cloud, d):
        X = cloud.astype(np.float64)
        return (X - X.mean(0)) @ d.astype(np.float64)

    items = [(r'$\mathbf{u}_1$', Uu['u1_comp'], 1), (r'$\mathbf{u}_2$', Uu['u2_comp'], 2),
             (V1, comp(Dd['cloud_head14'], Dd['v1']), 1),
             (V2, comp(Dd['cloud_head18'], Dd['v2']), 2),
             (V3, comp(Dd['cloud_l13'], Dd['v3']), 3)]
    fig, axes = plt.subplots(1, 5, figsize=(23, 4.4))
    for i, (ax, (lab, c, j)) in enumerate(zip(axes, items)):
        errline(ax, y[:, j - 1], align_sign(np.asarray(c, dtype=np.float64), y[:, j - 1]), NCOL[j],
                ms=3.5)
        style(ax, xlabel=Y[j], ylabel=f'component along {lab}')
        letter(ax, 'abcde'[i], x=-0.26)
    fig.tight_layout()
    save(fig, 'app_k3_directions')


def fig_k3_dissociation():
    Sp = get('multicompare_space_2digit.npz')
    Uu = get('multicompare_ucos_2digit.npz')
    rows = [(r'$\mathbf{u}_1$', lambda c: Uu[f'posrec_{c}_u1']),
            (r'$\mathbf{u}_2$', lambda c: Uu[f'posrec_{c}_u2']),
            (V1, lambda c: Sp[f'sp_posrec_{c}_v1_only']),
            (V2, lambda c: Sp[f'sp_posrec_{c}_v2_only']),
            (V3, lambda c: Sp[f'sp_posrec_{c}_v3_only'])]
    M = np.array([[f(c).mean() for c in CASES3] for _, f in rows])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = heat(ax, np.clip(M, 0, 1), SEQ_CM, 0, 1, xt=[Y[i] for i in (1, 2, 3)],
              yt=[r[0] for r in rows])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=ANNOT_FS + 1,
                    color='white' if M[i, j] > 0.55 else '0.15')
    style(ax, xlabel='perturbed number', ylabel='patched direction')
    cbar(fig, im, ax, label='PR')
    fig.tight_layout()
    save(fig, 'app_k3_dissociation')


def fig_k3_neurons():
    F = get('multicompare_freeze_2digit.npz')
    A = get('multicompare_attr_2digit.npz')
    T = get('multicompare_transfer_2digit.npz')
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.9), gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
    cols = [NCOL[j] for j in (1, 2, 3)]
    ax = axes[0]
    groups = ['none', 'MLP14', 'MLP15', 'MLP16', 'attn15', 'attn16', 'MLP14+15']
    vals = {c: {g: F[f'posrec_{c}_{g}'] for g in groups} for c in CASES3}
    grouped_bars(ax, groups, CASES3, vals, cols, labels=[PERT[c] for c in CASES3])
    ax.set_xticklabels(groups, rotation=20, ha='right')
    style(ax, xlabel='frozen sub-block', ylabel='PR')
    legend(ax, loc='upper right', bbox_to_anchor=(1.0, 1.16), ncol=3, fontsize=TICK_FS - 2)
    letter(ax, 'a', x=-0.13)

    ax = axes[1]
    ks = [1, 3, 10, 30, 100, 300, 1000, 3000]
    for c, col in zip(CASES3, cols):
        ax.plot(ks, [A[f'topk_posrec_{c}_top_k{k}'].mean() for k in ks], '-o', color=col, lw=2,
                ms=5)
        ax.plot(ks, [0.5 * (A[f'topk_posrec_{c}_rand0_k{k}'].mean() +
                            A[f'topk_posrec_{c}_rand1_k{k}'].mean()) for k in ks], '--',
                color=col, lw=1.6, alpha=0.8)
    ax.set_xscale('log')
    handles = [Line2D([], [], color='0.3', lw=2, marker='o', label='top $k$'),
               Line2D([], [], color='0.3', lw=1.6, ls='--', label='random $k$')]
    legend(ax, handles=handles, loc='center right')
    style(ax, xlabel='neurons frozen, $k$', ylabel='PR')
    letter(ax, 'b')

    ax = axes[2]
    series = ['none', 'two', 'rand']
    vals = {'none': {c: F[f'posrec_{c}_none'] for c in CASES3},
            'two': {c: T[f'posrec_{c}_two_number'] for c in CASES3},
            'rand': {c: np.concatenate([T[f'posrec_{c}_rand0'], T[f'posrec_{c}_rand1']])
                     for c in CASES3}}
    grouped_bars(ax, CASES3, series, vals, [COL['ceil'], COL['h14'], '#8CC7C7'],
                 labels=['none frozen', 'two-number 12', 'random 12'])
    ax.set_xticklabels([PERT[c] for c in CASES3], rotation=20, ha='right')
    style(ax, ylabel='PR')
    legend(ax, loc='upper center', fontsize=TICK_FS - 2)
    letter(ax, 'c')
    fig.tight_layout()
    save(fig, 'app_k3_neurons')


def fig_k3_l15():
    X3 = get('multicompare_das15_2digit.npz')
    X2 = get('multicompare2_das15_2digit.npz')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for i, (ax, X, cases, site) in enumerate(((axes[0], X3, CASES3, Y[3]),
                                              (axes[1], X2, CASES3[:2], Y[2]))):
        M = np.array([[X[f'posrec_{e}_from_{f}_k1'].mean() for e in cases] for f in cases] +
                     [[X[f'posrec_{e}_full'].mean() for e in cases]])
        vmax = 2.0
        im = heat(ax, np.clip(M, 0, vmax), SEQ_CM, 0, vmax, xt=[PERT[c] for c in cases],
                  yt=[Y[int(c[1])] for c in cases] + ['full rank'])
        ax.set_xticklabels([Y[int(c[1])] for c in cases])
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax.text(b, a, f'{M[a, b]:.2f}', ha='center', va='center', fontsize=ANNOT_FS + 1,
                        color='white' if M[a, b] > 1.1 else '0.15')
        ax.axhline(len(cases) - 0.5, color='white', lw=3)
        style(ax, xlabel=f'perturbed number, at {site}',
              ylabel='direction fitted with' if i == 0 else None)
        cbar(fig, im, ax, label='PR')
        letter(ax, 'ab'[i], x=-0.3 if i == 0 else -0.25)
    fig.tight_layout()
    save(fig, 'app_k3_l15')


def fig_k3_readout_pca():
    L = get('multicompare_late_2digit.npz')
    slot = L['late_y'].argmax(1) + 1
    layers = [int(l) for l in L['late_layers']]
    fig, axes = plt.subplots(1, len(layers), figsize=(4.4 * len(layers), 4.2))
    for i, (ax, l) in enumerate(zip(axes, layers)):
        c = L[f'pca_coords_L{l}']
        for j in (1, 2, 3):
            m = slot == j
            ax.scatter(c[m, 0], c[m, 1], color=NCOL[j], label=f'max {Y[j]}', **SCAT)
        style(ax, xlabel=f'PC1, L{l}', ylabel='PC2' if i == 0 else None)
        letter(ax, 'abcde'[i], x=-0.2)
    legend(axes[-1], loc='upper right', markerscale=2.5, handletextpad=0.1)
    fig.tight_layout()
    save(fig, 'app_k3_readout_pca')


def fig_k3_readout_causal():
    L = get('multicompare_late_2digit.npz')
    LH = get('multicompare_latehead20_2digit.npz')
    layers = [int(l) for l in L['late_layers']]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axes[0]
    for gi, g in enumerate(GROUPS3):
        ms = [mse(L[f'resid_posrec_{g}_L{l}']) for l in layers]
        ax.errorbar(layers, [m for m, _ in ms], yerr=[e for _, e in ms], fmt='-o', color=ORDER_COL[gi],
                    lw=2, ms=5, capsize=2, label=ORDER_LAB[g])
    ax.set_xticks(layers)
    style(ax, xlabel='layer (last token)', ylabel='PR')
    legend(ax, loc='upper left')
    letter(ax, 'a')

    ax = axes[1]
    means = np.stack([LH[f'posrec_{g}'].mean(1) for g in GROUPS3])
    top = np.argsort(-means.max(0))[:8]
    w = 0.8 / 3
    for gi, g in enumerate(GROUPS3):
        ms = [mse(LH[f'posrec_{g}'][h]) for h in top]
        ax.bar(np.arange(len(top)) + (gi - 1) * w, [m for m, _ in ms], w, yerr=[e for _, e in ms],
               capsize=2, color=ORDER_COL[gi], alpha=0.9)
        ax.axhline(LH[f'attnfull_posrec_{g}'].mean(), color=ORDER_COL[gi], ls='--', lw=1.3)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([f'H{h}' for h in top])
    ax.axhline(0, color='0.6', lw=0.8)
    style(ax, xlabel='layer-20 head (last token)', ylabel='PR')
    letter(ax, 'b', x=-0.12)
    fig.tight_layout()
    save(fig, 'app_k3_readout_causal')


def fig_ablation():
    B = get('multicompare2_ablate_2digit.npz')
    Bl = get('multicompare2_ablate_lastpos_2digit.npz')
    ks = [int(k) for k in B['ks']]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    curves = [('intact', lambda k: B[f'intact_k{k}'], COL['ceil'], '-'),
              ('12 zeroed, all tokens', lambda k: B[f'killed_k{k}'], COL['das'], '-'),
              ('12 zeroed, last token', lambda k: Bl[f'killed_k{k}'], COL['pc1'], '-'),
              ('random 12 zeroed', lambda k: np.concatenate([B[f'rand0_k{k}'], B[f'rand1_k{k}']]),
               COL['full'], '--')]
    for lab, f, c, ls in curves:
        ms = [mse(f(k)) for k in ks]
        ax.errorbar(ks, [m for m, _ in ms], yerr=[e for _, e in ms], fmt='o', ls=ls, color=c, lw=2,
                    ms=5, capsize=2, label=lab)
    ax.set_xscale('log')
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    style(ax, xlabel='number of operands, $k$', ylabel='accuracy')
    legend(ax, loc='lower left')
    fig.tight_layout()
    save(fig, 'app_ablation')


FIGURES = {f.__name__[4:]: f for f in (
    fig_u_manifold, fig_u_vs_pc1, fig_three_digit, fig_transport, fig_shared_probes, fig_shared_posrec,
    fig_mlp_freeze, fig_attribution, fig_rf_shared, fig_rf_exclusive, fig_connectivity,
    fig_l15_direction, fig_k3_heads, fig_k3_directions, fig_k3_dissociation, fig_k3_neurons,
    fig_k3_l15, fig_k3_manifolds, fig_k3_readout_pca, fig_k3_readout_causal, fig_ablation)}

if __name__ == '__main__':
    names = sys.argv[1:] or list(FIGURES)
    for n in names:
        FIGURES[n]()
