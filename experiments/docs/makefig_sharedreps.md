# `makefig_sharedreps.ipynb` — the shared-representation figures

Draws the five paper figures from `results/sharedrep_{tag}.npz` + `results/sharedrep_meta.json`.
It loads arrays only — no model, no GPU — so it is safe to re-run and re-style freely. See
`docs/data_sharedrep.md` for how each array was produced.

Figures are saved to `figs/makefig_sharedreps_{regime}_{name}.png` at 300 dpi. Set `SAVE = False`
in the style cell to preview without writing.

---

## Cell map

| cell | what it does |
|---|---|
| load | reads the npz + meta JSON, prints the setup line and the probe R² values |
| style | all the constants every figure reads: font sizes, palette, colormaps, 3D convention, `VIEW` |
| `Arrow3D` | a 3D arrow with a real filled head |
| Fig 1 | H14 DAS component vs $y_1$, log fits, inset vs the L13 DAS component |
| Fig 2 | L14 3D PCA, two panels (by $y_1$ with $\mathbf{v}_1$, by $y_2$ with $\mathbf{v}_2$) |
| angle picker | the same cloud from a grid of `elev`/`azim`, to choose `VIEW` |
| Fig 3 | per-head sweep across layer 14 |
| Fig 4 | direction cosine heatmap |
| Fig 5 | every interchange condition in the file |
| combined | the 2x3 paper figure, panels (a)-(f) |
| Fig 6 | the L14 cloud inside span($\mathbf{v}_1$, $\mathbf{v}_2$), with a true-angle compass |
| Fig 7 | rank-1 / rank-2 / full-rank patching of L14 at both sites |
| Fig 8 | panels (a)/(b) redrawn on the pre-MLP L14 cloud (`SITE` switches sites) |
| Fig 9 | the panel-(c) cosine heatmap at both probe read-out points, side by side |
| Fig 10 | dose-response sweep of the plane, with an additivity decomposition (`SWEEP_CASE`) |
| Fig 11 | the headline conditions in both counterfactual cases |

**Notation.** Both directions this notebook is about are read at the **second-number**
position, so both are $\mathbf{v}$'s and the subscript says which number each carries:
$\mathbf{v}_1$ = the H14 DAS 1D direction (the first number, written in at $n_2$ by the
head) · $\mathbf{v}_2$ = L13 PC1 at $n_2$ (the second number, already in place).
$\mathbf{v}_1$ is the first number's direction at its *own* position $n_1$ and belongs to
`makefig_numberreps.ipynb`. $y_1$ / $y_2$ = the first / second number.

(Earlier drafts called the H14 direction $\mathbf{v}_1$ and numberreps' $n_1$ direction
$\mathbf{v}_1$; the two were swapped so that the two directions sharing a location share a
letter. The code variables `V1`, `V2`, `U` follow the current names.)

---

## Style cell

Everything visual lives here; the figure cells contain no ad-hoc constants.

* **Palette.** The same CVD-validated categorical triple as `makefig_numberreps.ipynb`
  (blue `#2166AC` / amber `#C97B00` / orchid `#C06A9B`) plus teal `#117777` for H14 scopes and grey
  for the L14-full ceiling. Reused rather than re-picked so the two paper figures read as one set.
  Joint (H14 + L13) conditions are the *same* colour as their single-scope counterpart, hatched —
  scope is carried by colour, jointness by texture, so no fifth hue is needed.
* **Colormaps.** `CMAP_A = 'plasma'` for $y_1$, `CMAP_B = 'viridis'` for $y_2$ — two perceptually
  uniform sequential ramps, one per number, as requested. Never a rainbow map.
* **3D convention.** `PC_ORDER = (1, 2, 0)` puts $(x,y,z) = (\text{PC2},\text{PC3},\text{PC1})$, so
  PC1 is vertical and increases upward, matching `makefig_numberreps.ipynb`.
* **`VIEW`** is the shared `elev`/`azim`/`roll` for the 3D panels. Use the angle-picker cell to
  choose, then edit it here and re-run Figure 2.
* **`aggregate(x, y)`** groups by number value and returns (distinct value, mean, SE). At each
  value the partner number ranges over the whole cloud, so this averages that nuisance variation
  out rather than showing it as scatter. A value seen once gets SE 0 rather than a nan.

### `Arrow3D`

mpl's 3D `quiver` draws its head as two thin line segments, which all but vanish when an arrow is
near-vertical or seen edge-on, and whose size is only settable as a fraction of the shaft.
`Arrow3D` subclasses `FancyArrowPatch` and overrides `do_3d_projection` to re-project the endpoints
each draw, so the head is a filled polygon sized in absolute points (`mutation_scale`). Copied from
`makefig_numberreps.ipynb`.

---

## Figure 1 — H14 DAS component vs the first number

Arrays: `h14_das_comp`, `a_vals`, `l13_das_comp_n1`. Fits: `logfit_h14_das_a` from the meta JSON.

**Sign convention.** A DAS direction's sign is arbitrary, so the component is multiplied by
`sgn = -1 if pearson_r < 0 else 1`, which makes the curve rise with the number it encodes. The same
`sgn` is applied to the fit line and to the printed inset correlation, so all three stay consistent.

**Two fits.** `FIT1 = 'both'` draws the all-points fit faintly and the masked fit ($y_1 \geq$
`fit_lo`) heavily, over its own range. This is not decoration: on the real data the component
departs from the log law at the bottom of the range ($R^2$ 0.75 all points vs 0.90 for
$y_1 \geq 25$), and showing only the all-points fit would hide that. `'all'` or `'masked'` draw
just one.

**Inset** (`INSET_BOX`, axes-fraction coordinates): the H14 DAS component at $n_2$ against the L13
DAS component at $n_1$, coloured by $y_1$. Both come from the same forward pass over the same
prompts, so the rows line up exactly. The inset has an opaque white background and `zorder=5`
because the main fit line runs through that corner.

**Toggles:** `AGG1` (`'mean'` | `'raw'`), `FIT1`, `INSET_BOX`.

---

## Figure 2 — L14 residual, top-3 PCA

Two separate figures from one `pca3_panel()` call each, driven by the `PANELS` list. Arrays:
`pca14_coords`, `pca14_evr`, `a_vals` / `b_vals`, `*_in_pca14`, `resid14`.

**Arrow direction.** `{dkey}_in_pca14` is the direction's cosines with PC1/2/3. Two adjustments:

1. *Sign.* Flipped when the direction's component over the cloud anti-correlates with the number,
   so the arrow always points toward increasing $y$. The component is recomputed here as
   `resid14 @ dir` rather than read from a saved correlation, so the check is exact for both
   directions without either needing its own saved log fit.
2. *Length.* The projection is normalized before drawing. `‖proj‖` is the fraction of the unit
   direction that lives inside the 3D basis (`*_pca14_captured`, printed by the cell: 0.86 for
   $\mathbf{v}_2$, 0.13 for $\mathbf{v}_1$); leaving it un-normalized would shrink the arrow by that
   factor on top of the length scale. The length is then set by the cloud's extent *along* the
   arrow, not by an isotropic spread, so an elongated cloud does not leave it stubby.

Worth a caption note: the arrow is a projection of a 3584-d vector, and `*_pca14_captured` says how
faithful that projection is.

**Axes** are deliberately bare (no panes, no grid, grey axis lines) so the cloud carries the depth
cue. Axis labels include each PC's variance explained.

**Toggles:** `ARROW_SPAN`, `ARROW_LW`, `ARROW_HEAD`, `ARROW_LABEL_OFF`, and `VIEW` in the style
cell. `tight_layout` does not handle 3D axes, so margins are set with `subplots_adjust`.

### Angle picker

Re-renders the same coordinates over a grid of `ELEVS` × `AZIMS`. No recomputation — pick a pair,
set `VIEW`, re-run Figure 2.

---

## Figure 3 — per-head sweep across layer 14

Two stacked panels (position recovery, IIA), grouped bars over all 28 heads. Arrays:
`head_posrec_{arm}`, `head_iia_{arm}`, shape `(n_heads, n_eval)` — per example, so the error bars
are computed here as `v.std(axis=1) / sqrt(n_eval)` rather than trusted from a saved summary.

**`ARMS`** selects which arms to show. `'alone'` is the head by itself; `'with_L13_PC1_n2'` and
`'with_L13_full_n2'` add the rank-1 and full-rank L13 interchange at the same position. All three
are in `results/`; the default shows `alone` + the rank-1 joint arm.

**Head ordering.** Heads are ranked by effect, not by index: with 28 heads and one that matters,
index order buries the result in a flat row. `SORT_ARM` / `SORT_METRIC` pick the ranking key
(default: the last arm in `ARMS`, by position recovery); `order` is the resulting permutation and
**both panels share it**, so a column is one head. The x tick labels carry the head numbers, so
nothing is lost. The copy head's shaded column is located via `np.where(order == H_HEAD)`, not its
index, so the shading follows it wherever it lands.

**The reference line** is the key reading aid. Every joint bar carries the L13 patch's own effect,
which on the real data is ≈0.027 position recovery for *every* head — flat, and nothing to do with
the heads. `ARM_REF` maps each joint arm to its matching single-scope condition
(`cond_*_L13_PC1_atn2` / `cond_*_L13_full_atn2`) and draws it as a dashed line in that arm's
colour, so a bar is read as what the head adds on top of it.

The copy head's column is shaded (`axvspan`) so it is findable without counting ticks. The legend
is drawn once, on the top panel.

---

## Figure 4 — direction alignment

`cos_matrix` with `meta['cos_names']` as labels, on a diverging `RdBu_r` scale fixed to
$[-1, +1]$ — fixed, not data-scaled, because the meaningful reference points (0 and $\pm 1$) are
absolute. Values are printed in each cell, in white where the fill is dark.

`SHOW_H14_PC1 = False` drops the fifth row/column, leaving the four the figure asks for (H14 DAS,
L13 PC1 @n2, probe log A, probe log B). Set it `True` to include H14 PC1. The title carries
$1/\sqrt{d_{\text{model}}} \approx 0.017$, the chance cosine between two random unit vectors, which
is what "these are unrelated" looks like.

---

## Figure 5 — interchange conditions

Bars for every condition in `meta['condition_names']`, IIA and (with `SHOW_POSREC = True`)
position recovery. `cond_key()` reproduces the name → array-key transform `data_sharedrep.py`
applies when saving, so the two files cannot drift apart silently.

Colour = **site** (L13 blue/orchid, H14 teal, everything patched at L14 grey); hatching = **rank
restriction**, listed in `HATCHED`, so a hatched bar always means "a subspace, not the whole
thing" rather than specifically "joint". The position-recovery panel carries a dashed line at
1.0, the full-restoration reference.

**The two metrics, as equations for the main text.** With $\ell$ a last-position logit vector,
$t_{\mathrm{nc}}$ the counterfactual-correct answer token and $t_b$ the clean answer token, write
the answer logit difference as

$$\Delta(\ell)\;=\;\ell_{t_{\mathrm{nc}}}-\ell_{t_{b}} .$$

Then, for an intervention producing $\ell_{\text{patched}}$,

$$\mathrm{PR}\;=\;
  \frac{\Delta(\ell_{\text{patched}})-\Delta(\ell_{\text{corr}})}
       {\Delta(\ell_{\text{clean}})-\Delta(\ell_{\text{corr}})},
  \qquad
  \mathrm{IIA}\;=\;\mathbb{1}\!\left[\arg\max_{t}\,\ell_{\text{patched},\,t}
  = t_{\mathrm{nc}}\right],$$

both averaged over the `N_EVAL` evaluation triples; the reported error bars are SEs over that
average. $\Delta(\ell_{\text{clean}})$ and $\Delta(\ell_{\text{corr}})$ are the un-patched
anchors, saved as `pld_clean` / `pld_corr`. $\mathrm{PR}=0$ means the patch did nothing,
$\mathrm{PR}=1$ means it fully restored the clean behaviour, and $\mathrm{PR}>1$ means it
over-shot. This is the normalized-restoration convention of Meng et al. 2022 (ROME) and the IOI
literature; IIA is Geiger et al. 2021's interchange-intervention accuracy.

The two metrics say different things and both are worth keeping: position recovery is continuous
and can exceed 1 (over-restoration), IIA is the discrete "did the model actually answer the
counterfactual number" and saturates at 1.

**Toggles:** `SHOW_POSREC`, `COND_SHORT` (the two-line x-tick labels), `COND_COL`, `HATCHED`.

---

## Combined paper figure — the joint representation

Six panels, and **every one is a claim about the pair**. Provenance, each direction's encoding
curve, and each direction's own causal role now live in `makefig_numberreps.ipynb`; this figure
starts from them rather than re-establishing them. That split is what the figure is organised
around — see *Why this split* below.

| panel | content | claim |
|---|---|---|
| (a) | the pre-MLP L14 cloud in span($\mathbf{v}_1$, $\mathbf{v}_2$), by $y_1$ | structure |
| (b) | the same plane, by $y_2$ | structure — both numbers, one location |
| (c) | the same plane, by $y_1 > y_2$ | the comparison is a diagonal here |
| (d) | cosines among $\mathbf{u}$, $\mathbf{v}_1$, $\mathbf{v}_2$, lower triangle | near-orthogonality of the pair |
| (e) | IIA in **both** counterfactual cases | the double dissociation, and the pair reaching the ceiling |
| (f) | the sweep, as contours over the same plane, with the cloud underneath | the boundary, continuously |

**No prose on the figure.** Angles, chance levels and scope notes were removed from the axis
labels and are printed under the cell instead — a paper figure shows the result, the caption
explains it. What is left is axis labels, tick labels, legends, and measured values (the compass
angle, the contour levels). Bar heights are no longer printed on (e) -- they are in the table
below and in the cell's printout.

Saved as `figs/makefig_sharedreps_{regime}_paperfig.png`. It is self-contained (it no longer
reads `COND_SHORT` / `HATCHED` from the Figure 5 cell), and the Figure 6 / 9 / 10 sections below document the standalone
versions of (a)–(c), (d) and (f) in full — including every design note (the Gram–Schmidt basis,
the compass inset, the lower-triangle masking, the boxed pairs, what the sweep scores).

**Axes of (a)-(c) and (f).** The basis is Gram-Schmidt on the pair: $\mathbf{e}_1=\mathbf{v}_1$,
$\mathbf{e}_2=\mathbf{v}_2^{\perp}\propto\mathbf{v}_2-(\mathbf{v}_2\cdot\mathbf{v}_1)\mathbf{v}_1$.
The horizontal coordinate is exactly the (centred) projection on unit $\mathbf{v}_1$, i.e. the
component $f_1(y_1)$ H14 writes, so it is labelled "$f_1(y_1)$, comp. along $\mathbf{v}_1$". The
vertical one is the projection on $\mathbf{v}_2^{\perp}$, not on $\mathbf{v}_2$ -- the two are not
orthogonal (94.7 deg) -- so it is labelled "comp. along $\mathbf{v}_2^{\perp}$" (it is 99.7%
$\mathbf{v}_2$, but not exactly $f_2(y_2)$). Every panel carries the $\mathbf{v}_1$/$\mathbf{v}_2$
compass (upper right, `COMPASS_BOX`; (f) uses a slightly smaller, higher box to clear the 0.75
contour). No panel draws a grid; legends are frameless.

**Colours.** (c) uses the two PuOr ends (orange = $y_1$ larger, purple = $y_2$ larger), the same
map as (f)'s colorbar, so orange always means "$y_1$ wins". (d) uses `RdBu_r` trimmed to
[0.12, 0.88] so the diagonal is not saturated. (f)'s colorbar carries a black line at 0.5, the
level of the heavy black boundary, which is itself labelled "0.50".

**Panel (d)** is a bar plot of the three signed cosines cos($\mathbf{u}$,$\mathbf{v}_1$),
cos($\mathbf{u}$,$\mathbf{v}_2$), cos($\mathbf{v}_1$,$\mathbf{v}_2$), replacing the earlier
lower-triangle heatmap (three informative cells out of six, the diagonal drew the eye). $\mathbf{u}$
is `l13_das_dir` (L13 DAS at $n_1$); every direction is signed to increase with its own number, so
the signs are meaningful. The grey band is $\pm 2/\sqrt{d}$ ($d$ = d_model = 3584), i.e. $\pm 2$ sd of
the cosine between two independent random unit vectors in $\mathbb{R}^d$, whose distribution has
mean 0 and sd $\approx 1/\sqrt{d}$ -- the standard concentration-of-measure baseline. Bars are the palette teal (`COL['h14']`), the one validated series
colour not otherwise used in this figure (amber/blue are (e), orange/purple are (c)/(f)). Whiskers on
the $\mathbf{u}$ bars are the **range** over $\mathbf{u}$'s DAS seeds (`das_seed_dirs` in
`numberreps_*.npz`; only 2 seeds, seed 0 is $\mathbf{u}$ itself, so a range rather than an SE).
$\mathbf{v}_1$ has one seed, so the $\mathbf{v}_1$,$\mathbf{v}_2$ bar has no whisker.

Values (2digit): cos($\mathbf{u}$,$\mathbf{v}_1$) = -0.005 [seeds -0.005, +0.004],
cos($\mathbf{u}$,$\mathbf{v}_2$) = +0.428 [+0.427, +0.428], cos($\mathbf{v}_1$,$\mathbf{v}_2$) = -0.082;
random-pair sd = 0.017. Caveats: $\mathbf{v}_1$,$\mathbf{v}_2$ lies *outside* the band (about
$-4.8\sigma$) -- near-orthogonal (94.7 deg) but not random-like; and the residual stream is
anisotropic, so the isotropic band is a lenient baseline (a within-cloud top-PC baseline would be
stricter; not computed).

**Panel (f)** shares its axes with (a)–(c): the sweep grid is mapped out of the raw
`uv_plane_basis` the patch used and into the panel's centred, sign-oriented frame (exactly, via
`uv_plane_basis` and the cloud mean — not re-estimated), so the implanted grid and the observed
cloud can be read against each other. The cloud is grey underneath; the sweep is drawn as labelled
contours of $P(\text{names the first-slot number})$, with the 0.5 level heavy. That level *is* the
measured decision boundary, and it runs diagonally across the cloud.

**One site throughout.** `UV_SITE_FIG` defaults to `'pre'`, so the plane and the sweep in (f) are
both read at the pre-MLP residual — the only site where the
rank-2 patch does anything. Set either to `'post'` for the contrast; the Figure 6 section has the
numbers that justify the default.

**Panel (e) shows both counterfactual cases.** Colour is *which number the counterfactual
perturbs* — warm for $y_1$, cool for $y_2$, the same convention (a) and (b) use for the two
numbers. All bars share one style (no hatching). The full-residual scopes are labelled
$\mathbf{z}^2_{l=13}$ and $\mathbf{z}^2_{l=14}$ (the residual at layer $l$, second-number
position). Naming the arms by what is perturbed
rather than by `n1 larger` / `n2 larger` is what makes the panel readable in one look:

| patched | $y_1$ perturbed | $y_2$ perturbed |
|---|---|---|
| L13 full @ $n_2$ | 0.00 | **0.98** |
| $\mathbf{v}_2$ | 0.00 | **0.79** |
| $\mathbf{v}_1$ | **0.46** | 0.00 |
| $\mathbf{v}_1$ & $\mathbf{v}_2$ | 0.68 | 0.88 |
| $\mathbf{v}_1,\mathbf{v}_2$ plane, pre-MLP | 0.81 | 0.95 |
| L14 full | 0.82 | 0.99 |

**Each direction moves the answer exactly when the number it carries is the one that changed**, and
the plane reaches the full-rank ceiling either way. The zeros are the other half of the
dissociation, not failures: `make_batch` varies only the winning slot's number, so when $y_2$ is
perturbed H14's input is unchanged and the $\mathbf{v}_1$ patch writes back a value it already
had. Fig 11 is the same data with position recovery alongside, and
`docs/data_sharedrep.md` §6c has the design that would test $\mathbf{v}_1$ when $y_2$ wins *and*
$y_1$ changes.

This is why the panel is not captioned as "these directions are causal" — that claim belongs to
`makefig_numberreps.ipynb`. What this panel adds is the dissociation and the rank-2 sufficiency.

### Why this split between the two paper figures

The earlier version of this figure carried the H14 encoding curve and the per-head sweep, and
they looked out of place for a structural reason: $\mathbf{v}_2$ arrived already established
(loaded from `numberreps_*.npz`) while $\mathbf{v}_1$ was *born* here, so one figure was
simultaneously establishing one of its two objects and using both.

Moving both panels to `makefig_numberreps.ipynb` removes the asymmetry and gives each figure a
one-sentence thesis:

* **numberreps** — each number is encoded on a low-rank, log-scaled direction at its own position,
  each is causal there, and L14.H14 transports the first one to the second position.
* **sharedreps** — at that destination the two sit on near-orthogonal directions, and the 2D pair,
  neither half alone, decides the comparison.

The cosine panel stays here because orthogonality is a property of the *pair*, not of either
direction. The 3D PCA panels went to numberreps because a PCA view is only honest where the
direction is close to principal: the L13 DAS direction holds 0.59 of its norm in numberreps' own
3D basis, whereas $\mathbf{v}_1$ holds 0.13 in the L14 basis (0.22 pre-MLP) — it is the
low-variance, high-causal-impact direction PCA cannot see, which is the point but makes 3D the
wrong view for it.

## Figure 6 — the L14 cloud inside span($\mathbf{v}_1$, $\mathbf{v}_2$)

The residual stream leaving layer 14 at $n_2$, mean-centred and projected onto the 2D plane the
two causal directions span. The basis is Gram–Schmidt, not a PCA of the pair:

$$\mathbf{e}_1=\mathbf{v}_1,\qquad
  \mathbf{e}_2=\frac{\mathbf{v}_2-(\mathbf{v}_2\cdot\mathbf{v}_1)\,\mathbf{v}_1}
                      {\lVert\mathbf{v}_2-(\mathbf{v}_2\cdot\mathbf{v}_1)\,\mathbf{v}_1\rVert}.$$

That choice matters: it makes the horizontal axis *literally* the component an $\mathbf{v}_1$ patch
writes, rather than a rotation of it, and because $\cos(\mathbf{v}_1,\mathbf{v}_2)=-0.08$ the
vertical axis is 99.7% $\mathbf{v}_2$. Signs are set by correlation with the number each
direction carries, the same convention as the 3D panels.

Three panels: coloured by $y_1$, by $y_2$, and by whether $y_1>y_2$ (`SHOW_ORDER_PANEL`).

**The compass inset.** The scatter's aspect ratio is set by the data (the $\mathbf{v}_1$ coordinate
spans ~8 units, the $\mathbf{v}_2$ one ~40), so an `equal` aspect would flatten the panel into an
unreadable strip — but with an unequal aspect the drawn angle between two arrows is wrong. The two
directions are therefore drawn in their own small equal-aspect axes in the corner
(`COMPASS_BOX`), where the 94.7° is geometrically true.

**`UV_SITE` — and why it defaults to `'pre'`.** Which residual gets projected matters more here
than anywhere else in the notebook, and the two sites tell different stories:

| along | metric | pre-MLP | end of L14 |
|---|---|---|---|
| $\mathbf{v}_1$ | corr with $\log y_1$ | $+0.85$ | $+0.46$ |
| $\mathbf{v}_1$ | per-number-mean $R^2$ | **0.87** | **0.35** |
| $\mathbf{v}_1$ | sign agrees with $y_1>y_2$ | 0.87 | 0.96 |
| $\mathbf{v}_2$ | per-number-mean $R^2$ vs $y_2$ | 0.90 | 0.88 |

At the **pre-MLP** site the $\mathbf{v}_1$ axis is $y_1$'s *magnitude*: the cloud is a continuous
sheet, $y_1$ rising left to right and $y_2$ bottom to top, and the third panel's $y_1>y_2$ split
falls on a clean diagonal — which is exactly the picture a comparison mechanism predicts, and the
setup for the Fig 10 sweep.

At the **end of layer 14** the same axis has been taken over by the comparison outcome (its sign
agrees with $y_1>y_2$ on 96% of the cloud while a per-number mean explains only 35% of it), and
the cloud breaks into two blobs with a gap. So layer 14's MLP is what converts "how big is $y_1$"
into "which one is bigger" along this direction. That is a finding, not an artifact — but it is
also why the end-of-layer version of this figure is easy to misread, and why `'pre'` is the
default. `UV_SITE = 'post'` redraws it; the printout under the figure reports both statistics
for whichever site is drawn, and the file name carries the site.

None of this contradicts panel (d) of the main figure, which plots the component of **H14's own
output** along $\mathbf{v}_1$ ($r = +0.91$ with $\log y_1$; $R^2 = 0.75$ for $\log y_1$ alone,
rising only to 0.77 when the $y_1>y_2$ indicator is added, so that curve is magnitude, not the
indicator). The L14 residual is H14's write plus everything else the stack has put there.

**Toggles:** `UV_SITE`, `COMPASS_BOX`, `SHOW_ORDER_PANEL`.

---

## Figure 7 — patching L14 inside span($\mathbf{v}_1$, $\mathbf{v}_2$)

The causal counterpart of Figure 6: how much of a full-rank interchange at layer 14 a rank-2
interchange restricted to that plane reproduces. Columns are the two **sites** (end of layer 14,
and the pre-MLP residual), rows are the two metrics, and every bar is annotated with its rank.

Grouping by site is the point of the layout: a rank-2 bar is only interpretable against the
full-rank ceiling **at the same site**, and the two sites have different ceilings. Colour encodes
rank (teal = 1, amber = 2, grey = $d_{\text{model}}$) rather than scope, since here the site is
already carried by the column.

The cell degrades gracefully — it filters `SITES` down to the conditions actually present in the
npz and prints a note if none are, so the notebook still runs against a pre-rerun results file.

**Toggles:** `SITES` (which conditions, and how they group), `RANK`, `UV_BAR_COL`.

---

## Figure 8 — the two 3D panels at the pre-MLP site

Panels (a) and (b) of the combined figure, redrawn on `resid14_pre`: the residual stream at $n_2$
**before** layer 14's MLP. Same cloud of prompts, same $\mathbf{v}_1$ and $\mathbf{v}_2$, same
probe recipe — `data_sharedrep.py` runs one `pca_and_probes` over both clouds with a shared
train/test split, so the only thing that differs between this figure and (a)/(b) is where the
residual was read.

Why it earns a figure: Fig 7 shows that a rank-2 patch inside span($\mathbf{v}_1$, $\mathbf{v}_2$)
is worth nothing at the end of layer 14 and everything just before the MLP. This is the
descriptive side of the same contrast — what the cloud and the two directions look like at the
site where the patch actually works.

`SITE = 'post'` redraws it on `resid14`, which reproduces (a)/(b) standalone; the two renders share
every styling constant (`F8`), so they can be put side by side. Keys are namespaced by site
(`pca14pre_*`, `probe_logA_pre_*`, `h14_das_in_pca14pre`, …), and the cell prints a note and skips
if the npz predates the `resid14_pre` capture.

**Toggles:** `SITE`, `F8` (arrow weight, inset box, camera zoom, colorbar padding).

---

## Figure 9 — direction cosines by probe read-out point

The panel-(c) heatmap drawn once per probe site, side by side: probes fitted on the residual
leaving layer 14, and probes fitted on the pre-MLP residual. Same directions, same lower-triangle
convention, same two boxes. `COS_SITES = ['pre']` drops to the single panel.

Each panel's x-label carries the two angles **and the probes' held-out $R^2$**, deliberately
together: a more orthogonal pair of worse probes is not a better result, and without the $R^2$
the angle alone invites that reading.

**What it shows (2digit).**

| | post-MLP | pre-MLP |
|---|---|---|
| probe pair $\cos(\mathbf{p}_1,\mathbf{p}_2)$ | $+0.212$ (78°) | $+0.166$ (80°) |
| $\cos(\mathbf{v}_1,\mathbf{p}_1)$ | $+0.05$ | $+0.16$ |
| $\cos(\mathbf{v}_2,\mathbf{p}_2)$ | $+0.34$ | $+0.37$ |
| probe held-out $R^2$ | 0.998 / 0.999 | 0.999 / 1.000 |

Three things to take from it. The probe pair is **slightly** more orthogonal before the MLP — 78°
to 80°, about 2.7° — so the non-orthogonality of the two probes is not something the MLP creates;
it is already there when H14 writes. The probes did not get worse in the move (held-out $R^2$ is
if anything marginally higher), so that 2.7° is not the worse-probe artifact the $R^2$ column is
there to rule out. And $\mathbf{v}_1$ aligns about three times better with the pre-MLP $\log y_1$
probe than with the post-MLP one ($+0.05 \to +0.16$), which is the expected direction: that is
the stream $\mathbf{v}_1$ is written into, and the site where the rank-2 patch works.

The full 7×7 `cos_matrix` also holds the cross terms the figure does not draw:
$\cos(\mathbf{p}_1,\mathbf{p}_1^{\text{pre}}) = 0.613$ and
$\cos(\mathbf{p}_2,\mathbf{p}_2^{\text{pre}}) = 0.757$. So layer 14's MLP rewrites *where*
$\log y_1$ is read from substantially more than it rewrites $\log y_2$'s read-out (52° vs 41°
of movement) — consistent with $y_1$ being the number that has only just arrived at this position.

**Toggles:** `COS_SITES`, `COS_FS`. Panel (c) of the combined figure has the same switch as
`PROBE_SITE`, defaulting to `'post'`.

---

## Figure 10 — dose-response sweep of the span($\mathbf{v}_1$, $\mathbf{v}_2$) plane

Fig 7 shows the plane is sufficient *at one counterfactual point per example*. This asks the same
question as a surface. For a grid of number pairs $(y_1^\*, y_2^\*)$ the two pre-MLP coordinates
are **set** — not copied — to the cloud-mean coordinate of each number, and the model is asked
which number it then names:

$$\mathrm{LD}(y_1^\*,y_2^\*)=\ell_{t(y_1^\*)}-\ell_{t(y_2^\*)}.$$

**Why this and not an "additivity" figure.** Three different claims hide under that word, and only
two of them are true here:

1. *Write-side additivity* — the residual in the plane is $c_u\mathbf{v}_1+c_{v_2}\mathbf{v}_2$
   from two independent sources. Close to definitional (the residual stream is a sum), with the
   cross-talk bounded by $\cos(\mathbf{v}_1,\mathbf{v}_2)=-0.08$.
2. *Subspace sufficiency* — setting both coordinates recovers the behaviour. **True**, and Fig 7
   already shows it (rank-2 pre-MLP 0.81 against a 0.82 ceiling).
3. *Read-side separability* — $f(c_u,c_{v_2}) = g(c_u)+h(c_{v_2})$. This is what "additivity"
   usually means, and it is the one you should **not** expect: `max(a, b)` is a threshold on a
   contrast between the two numbers, and additive separability is close to the definition of not
   comparing them. The existing conditions already say so — position recovery is linear in the
   logit difference, and $\mathrm{PR}(\mathbf{v}_1\ \&\ \mathbf{v}_2)=1.469$ against
   $\mathrm{PR}(\mathbf{v}_1)+\mathrm{PR}(\mathbf{v}_2)=0.986+0.028=1.014$, super-additive far
   beyond the $\pm0.05$ SEs.

So the figure measures the *form* of the dependence rather than asserting additivity.

**What is scored — the thing that is easy to get wrong.** The model only ever names a number that
is in its prompt: 99.5% of outputs here are one of the prompt's own two. So "did it say
$y_1^\*$?" is unanswerable — $t(y_1^\*)$ never wins, and a panel scored that way reads as a flat
zero with a meaningless field beside it. What the implanted coordinates *can* decide is **which
slot** the comparison picks, and that is precisely what the mechanism claims: $\mathbf{v}_1$ carries
the first-slot number, $\mathbf{v}_2$ the second-slot number, and the read-out names the winner.
Both the discrete and the continuous panels are therefore scored on the prompt's own answer pair,
$\ell_{t_{\mathrm{nc}}}-\ell_{t_b}$ — the same quantity position recovery is built from, so the
sweep and the interchange conditions measure one thing.

**Panels.** (a) $P(\text{names the first-slot number})$ among prompts that named one of their own
two. (b) the continuous version, the answer logit difference, with its $\mathrm{LD}=0$ contour and
the $y_1^\*=y_2^\*$ diagonal a comparison predicts. (c) the best additive fit, a two-way
decomposition into a grand mean plus a row and a column effect — exactly $g(y_1^\*)-h(y_2^\*)$.
(d) what that fit misses. Panel (a) needs only `sweep_argmax`; (b)–(d) need `sweep_ld_slot`.

**Result (2digit).** $P(\text{first slot})$ is $0.77$ above the diagonal and $0.14$ below, its
sign matching $y_1^\*>y_2^\*$ on 85% of cells, with the measured boundary tracking the diagonal.
Two implanted coordinates, at a site where nothing about the prompt's own numbers has changed,
decide which number the model names.

The boundary is not exactly the diagonal: it sits slightly above it at small $y_2^\*$ and flattens
at large $y_2^\*$ (at $y_1^\*=75$–$95$ the first slot still wins ~0.75 of the time even when
$y_2^\*$ is larger). Part of that is a genuine asymmetry — $\mathbf{v}_1$ has the stronger main
effect — and part is a confound worth naming in the caption: the number → coordinate map
marginalizes the partner, so an implanted "$y_1^\*=95$" carries the coordinate that in
distribution occurs when $y_1$ is large *and usually wins*.

The variance shares of (c) and (d) are printed. Two caveats to carry into the caption: the missing
diagonal makes the design non-orthogonal, so the two shares need not sum to 1 (both are printed);
and a saturating logit difference produces interaction near the corners for reasons that are about
the read-out's dynamic range, not about the mechanism.

**Design notes.** The saved field is `sweep_ld_slot`; the earlier `sweep_ld` key held the
difference between the *implanted* numbers' tokens and was renamed rather than reused, so an old
results file cannot be silently reinterpreted. The grid takes **one value per leading digit** (15, 25, … 95), because the model
names a number by emitting its first token — the same constraint `build_triples` enforces. The
diagonal is skipped and left `nan`: $y_1^\*=y_2^\*$ is one token, so the difference is
identically zero. For the contour only, those cells are filled from their two in-row neighbours,
since a contour cannot cross a hole; the drawn heatmap keeps them blank. The sweep runs at the
**pre-MLP** site because that is the only site where the rank-2 patch does anything, and because a
per-number mean is a good description of each coordinate there ($R^2 = 0.87$ / $0.90$) and not at
the end of the layer ($0.35$).

**`SWEEP_CASE`** picks `'main'` (`meta['case']`) or `'alt'` (`meta['alt_case']`). Unlike the
interchange conditions, the sweep sets *absolute* coordinates, so it asks the same question in
either case — and the two agree (see the pitfall below).

**A pitfall worth knowing about.** `make_batch` defines `tok_nc` as the number at the **winning**
slot, which is $n_1$ only in the `n1 larger` case. Scoring "did it name the first slot" straight
off `tok_nc` therefore silently *inverts* the alt case — it looked like the plane did the opposite
thing there. `n1_slot_tokens()` in the style cell picks the right array from the case; every sweep
panel goes through it.

With that fixed the two cases agree:

| case | $P(n_1\text{-slot})$, $y_1^*>y_2^*$ | $y_1^*<y_2^*$ | sign matches |
|---|---|---|---|
| `n1 larger` | 0.773 | 0.143 | 0.847 |
| `n2 larger` | 0.858 | 0.160 | 0.944 |

**Toggles:** `SWEEP_CASE`, `SW_FS`, `SW_CMAP`; the grid and prompt count are `N_SWEEP` /
`sweep_grid_values` in `data_sharedrep.py`.

**An easy extension if a reviewer wants the textbook additivity test:** sweep $c_u$ finely at 4–5
fixed levels of $c_{v_2}$ and overlay the profiles. Parallel curves = separable, converging or
crossing = interaction. That is the classic no-interaction test and it costs one 1-D sweep per
level.

---

## Figure 11 — the headline conditions in both cases

The six conditions of the paper figure's panel (e), run on `n1 larger` prompts and again on
`n2 larger` prompts with the **same directions, nothing refitted**. IIA on top, position recovery
below.

The result is a **double dissociation**:

| condition | `n1 larger` | `n2 larger` |
|---|---|---|
| L13 full @ $n_2$ | 0.00 | **0.98** |
| $\mathbf{v}_2$ | 0.00 | **0.79** |
| $\mathbf{v}_1$ | **0.46** | 0.00 |
| $\mathbf{v}_1$ & $\mathbf{v}_2$ | 0.68 | 0.89 |
| $\mathbf{v}_1,\mathbf{v}_2$ plane, pre-MLP | 0.81 | 0.95 |
| L14 full (ceiling) | 0.82 | 0.99 |

**Each direction is causal exactly when the number it carries is the one the counterfactual
changed**, and the plane reaches the ceiling either way. That is much stronger than either case
alone, and it is why this belongs in the paper rather than staying a control.

It is also why the zeros are not evidence of inertness. `make_batch` varies only the number at the
winning slot, so under `n2 larger` the first number is identical in both runs, H14's input is
unchanged, and the $\mathbf{v}_1$ patch writes back a value it already had — the 0.00 is the
dissociation's other half, not a failure. `docs/data_sharedrep.md` §6c has the batch design that
would test $\mathbf{v}_1$ when the second number wins *and* the first number changes.

The paper figure's panel (e) stays `n1 larger` only: putting a structurally-zero $\mathbf{v}_1$
bar in the headline panel needs a paragraph to explain, and the headline panel should not need one.

---

## Switching regimes

Set `REGIME = '3digit'` in the load cell — but `data_sharedrep.py` must have been run with
`'3digit'` in its `TAGS` first, or the npz will not exist. Figure file names carry the regime, so
both regimes' outputs coexist in `figs/`.
