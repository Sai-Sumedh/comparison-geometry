# makefig_comparator.ipynb — the summary figure

Covers the summary figure and the cell that feeds it. The exploratory cells above it are
documented in `docs/explore_comparatorfig.md`; the data they read is written by
`data_comparator.py` (`docs/data_comparator.md`).

---

## The direction that flips the answer (`BND_NORMAL`)

The one direction in the span($\mathbf{v}_1,\mathbf{v}_2$) plane along which pushing the residual
changes which number the model answers. Everything it needs is already on disk, so the cell is
pure post-processing — no model, no GPU.

**It is not drawn.** The panel-(a) cloud is coloured by which number is larger, and at that
panel's axis scaling the normal lands close to perpendicular to the colour split, so an arrow for
it reads as a restatement of the colours rather than as new information. What the cell is for is
the *number*: each read direction's true separation from it, printed under the figure cell where
no shear distorts it.

**Where it comes from.** `makefig_sharedreps` panel (f) is a dose-response surface over this same
plane: its sweep *sets* the two coordinates to the values a chosen pair of numbers would put
there and records which slot the model then names. `results/sharedrep_{TAG}.npz` holds
`sweep_coords` (the 9 grid values' plane coordinates), `sweep_argmax` (the answered token, 64
prompts per cell) and the two token tables. The two notebooks' `uv_plane_basis` arrays are
byte-identical and the comparator's `plane_sgn` is `[1, 1]`, so the sweep grid and
`Z['plane_pq']` are already in one frame — and a gradient is translation-invariant, so the two
files' different centrings never have to be reconciled.

**How it is estimated.** Contouring the surface and then differentiating it is the long way
round. A logistic model has straight, parallel iso-probability contours, so its weight vector
**is** the contour normal:

$$P(\text{answers }y_1 \mid p, q) = \sigma(w_p\,p + w_q\,q + b), \qquad
  \mathbf{n} = \frac{(w_p, w_q)}{\lVert (w_p, w_q) \rVert}$$

fitted by Newton/IRLS (`logit_fit`) on one row per prompt that named either implanted number —
4587 rows. The sign convention points $\mathbf{n}$ toward "the model answers $y_1$".

`tok_nc` is the number at the *winning* slot, which is the first number only in the `n1 larger`
case, so which saved table holds the first-slot token is chosen from `SWM['case']`; reading it
straight off `tok_nc` silently inverts the other case.

**Cross-check, printed by the cell.** The same normal is refitted by total least squares on the
vertices of the drawn $0.5$ contour. The two agree to $0.1°$ (cos $= 1.0000$), which is what
licenses the logistic shortcut.

**The number that matters.** The boundary normal sits at $+7.1°$ in the plane; the
**difference axis** `Z['diff_axis']` — the least-squares direction along which the natural cloud
encodes $\log y_1 - \log y_2$ — sits at $-21.3°$. They are $28°$ apart (cos $0.88$). The
direction the cloud *encodes* the comparison along is not the direction the model *acts* on, so
`diff_axis` is not a usable stand-in here and the dashed $y_1 = y_2$ line in panel (a) (drawn as
`diff_axis`'s normal) is not perpendicular to the reference arrow. That gap is a result, not a
plotting artefact.

---

## Panel-by-panel

| panel | content |
|---|---|
| (a) | read directions of the selected neurons in the plane, over the cloud coloured by which number is larger |
| (b) | the eight highest-ranked shared MLP14 receptive fields |
| (c) | the MLP15 receptive field |
| (d) | the rank-1 L15 DAS component, split by which slot holds the max |
| (e) | IIA for the plane patch, that patch with the shared neurons frozen, and the rank-1 L15 DAS patch |

### Panels (d) and (e)

(d)'s y axis counts $(y_1, y_2)$ pairs — one cloud prompt is one pair, so that is what the bar
heights are, stated rather than left as "prompts".

(e)'s bars are named by the intervention, at the level a paper reader can act on. Bar 1 is the
base patch, `(v1,v2) patch`; its site (pre-MLP) is left to the caption and the notebook header,
where the rest of the patch's definition already lives.

Bar 2's tick reads "freeze top-`n_sh` MLP neurons". Precisely, the frozen set is the neurons
appearing in **both** cases' top-`FIN_OVK` attribution lists, so `n_sh` (12 at `FIN_OVK = 20`) is
an intersection size rather than a rank depth — the true top 12 by rank would be a slightly
different set. The label deliberately does not say so: the main paper never introduces the
per-case lists, and the bar's point is that these are the neurons most responsible. The exact
membership and the L14/L15 split are printed under the cell, and `n_sh` is computed, so changing
`FIN_OVK` keeps the count right.

### Panel (a)

Every arrow is drawn at the same length *as a fraction of the axes box* by `plane_arrow`, because
the two axes are scaled independently to the cloud (a $3.8\times$ difference). That makes the
frame sheared: **no angle read off this panel is an angle in the
$(\mathbf{v}_1,\mathbf{v}_2)$ basis**, and the distortion does not even go one way — the gap
between the causal and observational normals is $25°$ true but reads as $6.7°$, while the causal
normal's departure from perpendicular-to-the-boundary is $28°$ true but reads as $58°$. Every
angle that matters is printed under the cell instead.

* **The cloud is coloured by which number is larger** — `SLOT_COL`, warm for $y_1 > y_2$ and cool
  for $y_2 > y_1$. That is `makefig_sharedreps`' convention (its $y_1$/$y_2$ ramps and its
  `PuOr_r` dose-response surface) and it is the same pair panel (d) splits its histogram with, so
  one slot is one colour across both figures. Slot is $99.5\%$ linearly separable from $(p, q)$,
  so the split is clean; the dashed $y_1 = y_2$ line stays as a guide through the few points that
  interleave at the boundary.
* **The two groups are named on the points, not in the legend** (`SLOT_LABEL_POS`, `FS_SLOT`),
  in their own colour. The cloud is a band running top-left to bottom-right, so each label sits
  at the tip of its own group's end of it: $y_2 > y_1$ top-left, $y_1 > y_2$ bottom-right. Those
  corners hold the densest part of each group, so `CLOUD_PAD_Y` (1.26, against 1.10 on $x$)
  opens the vertical margin the labels sit in — without it there is no clear space there.
  The legend then moves to `upper right`, the one large area the band never reaches.
* `CLOUD_S`, `CLOUD_ALPHA` — kept pale, because the read arrows sit on top. At the original
  `s=6, '0.80', 0.55` the cloud was invisible at print size.
* `ACOL` — the read arrows moved off amber and blue once the cloud claimed them; they are now
  black, teal, orchid and purple.
* The legend no longer carries each arrow's in-plane fraction. It still goes to stdout, with
  chance ($\sqrt{2/d_\text{model}}$) beside it.
* `ARROW_MIN_FRAC` / `ARROW_MIN_SEP` still choose *which* neurons get an arrow: above a multiple
  of chance, and at least so many degrees from an already-kept, stronger one.

**The result that is now text, not geometry.** MLP14 #6150 reads within $5°$ of the direction
that flips the answer; the other three are $116°$, $155°$ and $171°$ off. That line is printed
every run and is the one thing panel (a) would otherwise have to carry visually.

### Panels (b) and (c)

`rf_axes` draws one field, normalised to its own peak so (b) and (c) share a colour scale. The
`n1/n2` attribution rank that used to sit under each neuron name is gone from the figure; both
ranks are still printed under the cell.

**On smoothing them.** They are deliberately left blocky (`imshow` default `interpolation=
'nearest'`). The grid is $23 \times 23$ at stride 4 over $y \in [11, 99]$, and each cell is **one
deterministic forward pass on one prompt** — not a noisy average over repeats. So the speckle is
real per-number variation, not sampling noise waiting to be averaged out, and the sharp sign flip
across $y_1 = y_2$ is the claim the panels exist to make. `interpolation='bilinear'` turns
isolated cells into plus-shaped smears and reads worse than the raw grid; a Gaussian blur looks
better but destroys the one-cell-wide antisymmetric band that is the whole content of a neuron
like #6854. Both were rendered and rejected.

---

## Reproducing

Run the notebook top to bottom. The boundary-normal cell needs
`results/sharedrep_{TAG}.npz` and `results/sharedrep_meta.json` alongside the comparator's own
npz; if either is missing, `BND_NORMAL` is undefined and the figure cell skips the
separation line rather than failing.

## References

* Logistic decision boundary as the contour normal — standard; the linear-probe reading is
  Alain & Bengio (2016), *Understanding intermediate layers using linear classifier probes*.
* Interchange interventions and IIA — Geiger et al. (2021), *Causal abstractions of neural
  networks*.
* Position recovery normalisation — Meng et al. (2022), *Locating and editing factual
  associations in GPT* (the ROME/IOI convention).
