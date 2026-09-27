"""Shared style and small plotting helpers for the appendix figures (makefig_appendix*.py)."""

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RESULTS_DIR, FIGS_DIR = 'results', 'figs_appendix'
os.makedirs(FIGS_DIR, exist_ok=True)

LABEL_FS, TICK_FS, LEGEND_FS, PANEL_FS, ANNOT_FS = 18, 15, 14, 20, 12
plt.rcParams.update({'font.size': TICK_FS, 'axes.linewidth': 1.0, 'pdf.fonttype': 42,
                     'mathtext.fontset': 'dejavusans'})

# Same CVD-checked palette as the main-paper makefig notebooks.
COL = {'full': '#2166AC', 'das': '#C97B00', 'pc1': '#C06A9B', 'h14': '#117777', 'ceil': '#6E6E6E'}
NCOL = {1: '#C97B00', 2: '#2166AC', 3: '#2E8B57'}          # y1, y2, y3 (as in the Fig. 5 notebook)
CASE2 = {'n1_larger': COL['das'], 'n2_larger': COL['full']}  # y1 perturbed, y2 perturbed
ORDER_COL = ['#2166AC', '#C97B00', '#117777']              # the three value orders, when not a number
LIGHT = '#B8B8B8'

Y = {1: r'$y_1$', 2: r'$y_2$', 3: r'$y_3$'}
U, V1, V2, V3 = r'$\mathbf{u}$', r'$\mathbf{v}_1$', r'$\mathbf{v}_2$', r'$\mathbf{v}_3$'
PERT = {'n1_larger': r'$y_1$ perturbed', 'n2_larger': r'$y_2$ perturbed',
        'y1': r'$y_1$ perturbed', 'y2': r'$y_2$ perturbed', 'y3': r'$y_3$ perturbed'}
ORDER_LAB = {'s1gts2': r'$y_1>y_2>y_3$', 's1gts3': r'$y_1>y_3>y_2$', 's2gts3': r'$y_2>y_3>y_1$'}


def trim_cmap(name, lo, hi, n=256):
    """`name` restricted to [lo, hi] of its range; drops the harshest ends."""
    base = matplotlib.colormaps[name]
    return LinearSegmentedColormap.from_list(f'{name}_{lo}_{hi}', base(np.linspace(lo, hi, n)))


SEQ_CM = trim_cmap('YlGnBu', 0.05, 0.9)       # sequential maps (scores, accuracies)
DIV_CM = trim_cmap('PuOr_r', 0.08, 0.92)      # signed maps (cosines, edges)
RF_CM = trim_cmap('PRGn', 0.16, 0.84)         # receptive fields, as in the Fig. 4 notebook
VAL_CM = trim_cmap('plasma', 0.0, 0.9)        # a cloud coloured by a number's value
SCAT = dict(s=10, alpha=0.6, edgecolors='none', rasterized=True)


def load(name):
    return dict(np.load(os.path.join(RESULTS_DIR, name), allow_pickle=True))


def style(ax, xlabel=None, ylabel=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def letter(ax, s, x=-0.2, y=1.03):
    ax.text(x, y, f'({s})', transform=ax.transAxes, fontsize=PANEL_FS, fontweight='bold',
            ha='left', va='bottom')


def legend(ax, **kw):
    kw.setdefault('fontsize', LEGEND_FS)
    kw.setdefault('frameon', False)
    return ax.legend(**kw)


def save(fig, name):
    path = os.path.join(FIGS_DIR, f'{name}.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'saved {path}  ({os.path.getsize(path) / 1e6:.2f} MB)', flush=True)


def mse(a):
    a = np.asarray(a, dtype=np.float64)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def aggregate(x, y, bins=None):
    """Mean and SE of y per distinct x (or per bin of x when `bins` is given)."""
    x, y = np.asarray(x), np.asarray(y, dtype=np.float64)
    if bins is not None:
        idx = np.digitize(x, bins)
        keys = [k for k in np.unique(idx) if 0 < k < len(bins)]
        xs = np.array([0.5 * (bins[k - 1] + bins[k]) for k in keys])
        groups = [y[idx == k] for k in keys]
    else:
        xs = np.unique(x)
        groups = [y[x == v] for v in xs]
    m = np.array([g.mean() for g in groups])
    se = np.array([g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0 for g in groups])
    return xs, m, se


def align_sign(comp, target):
    """Flip a sign-indeterminate component so that it rises with `target`."""
    return comp if np.corrcoef(comp, target)[0, 1] >= 0 else -comp


def fit_logx(x, y):
    p, q = np.polyfit(np.log(x), y, 1)
    return p, q


def grouped_bars(ax, groups, series, values, colors, width=0.8, hatches=None, err=True,
                 labels=None):
    """values[s][g] is an array of per-example scores; bars show mean +/- SE."""
    x = np.arange(len(groups))
    w = width / len(series)
    for si, s in enumerate(series):
        ms = [mse(values[s][g]) for g in groups]
        xp = x + (si - (len(series) - 1) / 2) * w
        ax.bar(xp, [a for a, _ in ms], w, yerr=[b for _, b in ms] if err else None, capsize=2.5,
               color=colors[si], alpha=0.9, edgecolor='white', linewidth=0.6,
               hatch=None if hatches is None else hatches[si],
               label=s if labels is None else labels[si], error_kw=dict(lw=1.0))
    ax.set_xticks(x)
    return x


def heat(ax, M, cmap, vmin, vmax, xt=None, yt=None, annot=False, fmt='{:.2f}'):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
    if xt is not None:
        ax.set_xticks(range(len(xt)))
        ax.set_xticklabels(xt, fontsize=TICK_FS)
    if yt is not None:
        ax.set_yticks(range(len(yt)))
        ax.set_yticklabels(yt, fontsize=TICK_FS)
    if annot:
        mid = 0.5 * (vmin + vmax)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    dark = M[i, j] > mid if cmap is not DIV_CM else abs(M[i, j]) > 0.6 * vmax
                    ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center', fontsize=ANNOT_FS,
                            color='white' if dark else '0.15')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return im


def cbar(fig, im, ax, label=None, **kw):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, **kw)
    cb.ax.tick_params(labelsize=TICK_FS - 2)
    cb.outline.set_visible(False)
    if label:
        cb.set_label(label, fontsize=LABEL_FS - 2)
    return cb


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
