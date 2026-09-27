# `makefig_numberreps.ipynb` — single-number representations, and the transport of the first one

Draws the first of the two paper figures from `results/numberreps_*.npz` (written by
`data_numberreps.py`) and `results/sharedrep_*.npz` (written by `data_sharedrep.py`). Computes
nothing that needs the model.

---

## Notation

Three directions, named so that the two which share a *location* share a letter:

| symbol | what it is | where it lives |
|---|---|---|
| $\mathbf{u}$ | DAS 1D trained at the first-number position | L13 residual @ $n_1$ |
| $\mathbf{v}_1$ | DAS 1D inside head L14.H14's output | written into the residual @ $n_2$ |
| $\mathbf{v}_2$ | PC1 of the second-number cloud | L13 residual @ $n_2$ |

So $\mathbf{u}$ is the first number *before* it is moved and $\mathbf{v}_1$ is the same number
*after* transport. The two $\mathbf{v}$'s coexist at $n_2$, which is what
`makefig_sharedreps.ipynb` is about.

Earlier drafts called $\mathbf{u}$ "$\mathbf{v}_1$" and $\mathbf{v}_1$ "$\mathbf{u}$"; the two
were swapped. Code variables `U`, `V1`, `V2` follow the current names. Array keys in the npz files
are unchanged (`das_dir`, `h14_das_dir`, `pca3_components_b`), so nothing on disk moved.

## What this figure owns, and why

This figure establishes the **objects**: where each number is encoded, that the encoding is
low-rank and log-scaled, that each direction is causal for its own number, and that one head
moves the first number to the second position. `makefig_sharedreps.ipynb` then uses them and only
makes claims about the **pair**.

That division is the reason the H14 encoding curve (c) and the per-head sweep (d) live here rather
than in the sharedreps figure. $\mathbf{v}_2$ was always established here and loaded there; having
$\mathbf{v}_1$ established in the sharedreps figure made that figure do setup work in the middle of
its own argument. Now both directions arrive at the second figure pre-established.

One-sentence thesis: *each number is encoded on a low-rank, log-scaled direction at its own
position, each is causal there, and L14.H14 transports the first one to the second position.*

## Loading

`R` / `META` hold the numberreps results for every regime in `REGIMES`. `S` / `SMJ` hold the
sharedrep results, loaded only for the regimes that have a `sharedrep_*.npz` — panels (c), (d) and
the third group of (e) print a placeholder and skip if it is missing, so the notebook still runs
against numberreps results alone. Same model, prompt and seed in both files, so they describe one
run.

## Style cell

All font sizes come from `FS`; `FS_SCALE` grows them together. The series palette is validated
(OKLab $\Delta E$, Machado–Oliveira–Fernandes CVD simulation, WCAG contrast) and carries one
colour per direction:

| key | hex | direction |
|---|---|---|
| `full` | `#2166AC` blue | full-residual / full-output interchanges |
| `das` | `#C97B00` amber | $\mathbf{u}$ |
| `pc1` | `#C06A9B` orchid | $\mathbf{v}_2$ |
| `h14` | `#117777` teal | $\mathbf{v}_1$ |

The teal matches the colour the sharedreps notebook uses for H14, so a reader moving between the
two figures sees one direction keep one colour. `CMAP_FIG = 'plasma'` for panel (a)'s magnitude
ramp: measured min $\Delta E$ to the series colours is 7.6, against viridis's 5.7 (whose mid-blue
collides with `COL['full']`).

3D convention: $(x, y, z) = (\text{PC2}, \text{PC3}, \text{PC1})$, so PC1 is vertical and increases
upward (`PC_ORDER`, `xyz_for_plot`).

---

## Paper figure

A 2x6 gridspec read as two rows of three, each row with its own thesis. Saved as
`figs/makefig_numberreps_number_reps_fig.png`.

* **Row 1 — where each number is encoded, and what transport does to the first one's code.**
* **Row 2 — the head that does the transporting, and what each direction buys causally.**

Every font size and layout constant is a named variable in a block at the top of the cell
(`FS_AXIS`, `FS_TICK`, ..., `FIG_SIZE`, `WIDTH_RATIOS`, `ARROW_SPAN`, ...), with one comment per
entry saying which piece of text or spacing it controls. Nothing below applies a hidden offset.

**No $n_1$ / $n_2$ on the figure.** Every displayed label uses $y_1$, $y_2$, $\mathbf{u}$,
$\mathbf{v}_1$, $\mathbf{v}_2$ only, and no bar or legend says "L13" — the scope words are `full`
and the direction symbols. Array keys in the npz files still carry the old names
(`h14_attn_row_n2`, `with_L13_PC1_n2`, `iia_full_n1_larger_first_number`); those are never drawn.

### (a) the first number's cloud, with u

2D PCA (PC1 vs PC3, set by `PCS_A`) of the L13 residual at the first number's position, coloured by $y_1$, with
$\mathbf{u}$ projected into that plane. `das_in_pca3_a[PCS_A]` are unit $\mathbf{u}$'s coefficients
on PC1 and PC3; the arrow is that 2D projection, normalized, sign-flipped to point toward
increasing $y_1$, starting at the cloud mean, and `ARROW_SPAN` times the cloud's extent along it.
Its true in-plane length ($\|P_{13}\mathbf{u}\|$) is printed, not encoded.
`annotation_clip=False` keeps the arrow drawn when its head passes the axes limits.

The dark trace is the cloud's mean position per distinct $y_1$, then averaged within
`CURVE_BINS_A` equal-count bins of increasing $y_1$ (one marker per bin) -- the binned-mean
construction of `manifold_curve`, in 2D. PC1 vs PC3 is the pair that shows the arc; PC1–PC2 and PC2–PC3 show none.

The cloud has one point per distinct value: at the first number's position causal masking makes
the residual a deterministic function of that number alone.

No panel in the figure draws a grid (`style_axes(..., grid=False)` everywhere); all six columns
have equal width now that (a) is no longer 3D.

### (b) the first number, before and after transport

The same number read in two places on **twin axes**: along $\mathbf{u}$ where it sits (amber,
left) and along $\mathbf{v}_1$ after H14 copies it (teal, right). Spines, ticks and labels are
colour-matched to their series.

Twin axes rather than one scale, and rather than z-scoring: the two components live in different
spaces (the L13 residual, and H14's output) and their units are not comparable. What *is*
comparable is the shape, and the shapes differ — along $\mathbf{u}$ a clean log, along
$\mathbf{v}_1$ a curve that turns over below $y_1 \approx 25$ and saturates above $\approx 70$.
**The copy is not faithful.** Only $\mathbf{u}$ gets a log fit; $\mathbf{v}_1$ deliberately has
none, since a log law drawn across that reports $R^2 = 0.90$ while hiding both departures.

Note `errorbar` returns an `ErrorbarContainer`, and that is what carries the label — indexing
`[0]` for a legend handle gives the bare `Line2D`, whose label is `_nolegend_`.

Because (b) carries axis labels on **both** sides, `GRID_WSPACE` has to be large enough to hold
two of them; below about 0.9 its right label and (c)'s left label print on top of each other.

### (c) the second number, at its own position

$\mathbf{v}_2$'s component against $y_2$, with the log fit `data_numberreps.py` saved — always fit
on the raw cloud, never on the aggregated means. `ALIGN_SIGNS` flips a component whose fit slopes
down, since a direction's sign is arbitrary.

### (d) what the head reads

H14's attention from the second number's query across the whole prompt, mean +/- SE. Only the
first number's position is labelled; the rest are ticks, to show the axis is over tokens. The band
marks that number's whole token span, so the single label cannot be misread as off-by-one against
a peak on the number's other token. The head puts 0.71 of its weight there.

The inset is **attention to $y_1$ against $y_1$**, and it earns its space: attention climbs from
about 0.2 to 0.85 across the range ($r = +0.61$). So part of (b)'s teal curve is *how much* the
head attends, not only what it copies — worth a caption sentence rather than leaving a reader to
assume the curve is purely the copied value.

### (e) which head does the transport

Each layer-14 head's own output contribution interchanged where the second number sits, alone and
jointly with a $\mathbf{v}_2$ interchange at the same place. The joint arm is the one that
matters, because H14 writes on top of whatever L13 already holds there; the dashed line is that
patch's own effect, so a bar reads as what the head adds on top of it. Only the top
`TOP_N_HEADS` are shown — the rest sit on that line (full sweep in `makefig_sharedreps.ipynb`).

### (f) IIA: each direction is causal for its own number

Two groups, one per number. $\mathbf{v}_1$ sits in the $y_1$ group rather than a group of its
own: it is the same number, read after transport.

| group | bars |
|---|---|
| $y_1$ | full **0.07** · $\mathbf{u}$ **0.94** · $\mathbf{v}_1$ **0.46** |
| $y_2$ | full **0.98** · $\mathbf{v}_2$ **0.79** |

In two of the three direction/full pairs the rank-1 direction scores far **higher** than the full
swap: a full interchange drags in everything else at that site, the learned direction is the
surgical version of the same intervention. It is the cleanest single argument that the directions
are the right objects, and worth a caption sentence.

$\mathbf{v}_1$'s own full-scope comparator — interchanging H14's *entire* output contribution —
scores 0.00 and is **printed rather than drawn**: a second blue "full" bar inside one group, at a
different scope from the first, reads as the same control twice. It is another instance of rank-1
beating the full swap, so it belongs in the caption.

---

## Appendix figure — the transported number's manifold

H14's output cloud in its top two PCs, coloured by $y_1$ on the same map as panel (a), with the
smoothed mean position per number drawn through it. It traces an **arc**.

| | |
|---|---|
| top-2 PCs hold | **66%** of the head's output variance |
| $\mathbf{v}_1$ accounts for | **64%** of the trajectory's net displacement |
| ...but only | **21%** of its arc length |
| and that one chord gives | **IIA 0.46** |

So the transported code is a curved, low-dimensional manifold and $\mathbf{v}_1$ is one straight
chord through it — which is the point: the geometry is not 1-D, and a 1-D direction through it
still moves the model. Arc length is measured on the smoothed path; the raw per-number path
jitters, which inflates it (17-21% smoothed against 2-7% raw).

**Caveat for the caption.** An arc in PC1-PC2 is also the classic **horseshoe / arch artifact** of
PCA on data with a single monotone gradient — it appears even when the underlying structure is
exactly 1-D. Do not rest the claim on the picture. The independent evidence is panel (b)'s
$\mathbf{v}_1$ curve: that direction is found by DAS in the full 3584-d space, not in PCA
coordinates, and its readout is non-monotone. If the manifold were a straight line parametrised by
$\log y_1$, *every* direction's readout would be monotone.

For scale, all three clouds are curved — the L13 clouds bend 52% and 56% of their own chord
against the H14 output's 81% — so bending is a general property here, not something transport
creates. What is special about the head's output is how *concentrated* it is: 66% of its variance
in two PCs, against 24% and 20% for the L13 clouds, which is why a 2D picture is faithful for it
and not for them.

**Why it is not an inset.** Both (a) and (b) were tried and rejected. (b) already carries two
y-axes, two series, a fit and a legend. (a)'s margins look empty but hold the three PC labels and
the $\mathbf{u}$ arrow label, so the inset landed on the cloud; clearing it needs `BOX_ZOOM_3D`
around 1.0, which undoes the widening (a) needed in the first place.

---

## Variance explained

An exploratory cell above the paper figure: per-direction EVR for the top 3 PCs and the cumulative
spectrum, at each number position. Not a paper panel.

## Switching regimes

`FIG_REGIME` picks which regime the paper figure draws; `REGIMES` controls what is loaded. The
`3digit` numberreps results exist, but `data_sharedrep.py` runs `2digit` only by default, so
panels (c), (d) and (e)'s third group skip with a note under `FIG_REGIME = '3digit'` until
`'3digit'` is added to that script's `TAGS`.
