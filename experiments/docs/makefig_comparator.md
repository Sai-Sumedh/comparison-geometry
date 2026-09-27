# makefig_comparator.ipynb

A trimmed copy of `makefig_comparator_old.ipynb` that renders only the main-paper summary figure.
No model, no GPU: it reads `results/comparator_data_2digit.npz`, `comparator_data_meta.json`, and
`results/sharedrep_2digit.npz` / `sharedrep_meta.json`. Runs in about 3 s.

## Cells kept (in order)

| cell | what it does |
|---|---|
| header | title / data provenance (unchanged from the original) |
| load | loads the comparator npz and meta (`Z`, `M`, `CASES`, `ck`) |
| plot style | palette, `style_axes`, `savefig`, pooled-neuron tables (`PL`, `POOL_LAYER`, `RANK`, ...) |
| boundary direction | fits a logistic regression to the sharedreps dose-response sweep in span(v1, v2) and saves `BND_NORMAL`, the unit normal to the P = 0.5 contour, pointing toward "answers y1" |
| summary figure | the big figure; saves `figs/comparator_2digit_summary.png` (PDF line commented, to `figs_mainpaper/`) |

Every exploratory section of the original (conditions, module freezes, top-k curve, overlap,
injection, plane stages, receptive fields, geometry, gate boundaries, connectivity, DAS) is dropped.
The figure cell only depends on the four cells above it; this was checked by running the trimmed
notebook end to end headless.

The original notebook is untouched, and all design notes for the summary figure stay in
`docs/makefig_comparator_old.md`. Edits to the figure must be made in one notebook and copied to the
other by hand -- they are independent copies.

## Panel (a): matched to makefig_sharedreps panel (a)

Same plane and frame as the sharedreps panel (pre-MLP L14 cloud, $\mathbf{e}_1=\mathbf{v}_1$,
$\mathbf{e}_2=\mathbf{v}_2^{\perp}$, centred), so it is drawn with the same shape:

* `PLANE_BOX_ASPECT = 0.952` -- the height/width of sharedreps (a)'s axes box (measured: 4.96 x 4.72 in).
* Data-fitted limits (matplotlib's default 5% margins), as sharedreps does, instead of the old
  symmetric padding (`CLOUD_PAD`, `CLOUD_PAD_Y`, removed) that stretched y to +-30 and squashed the
  cloud. The limits are frozen before the arrows are added; the arrows and the $y_1=y_2$ marker are
  scaled by the half-ranges of those limits.
* Axis labels as in sharedreps: "$f_1(y_1)$, comp. along $\mathbf{v}_1$" and
  "comp. along $\mathbf{v}_2^{\perp}$" (the vertical axis is $\mathbf{v}_2$ orthogonalised against
  $\mathbf{v}_1$, not $\mathbf{v}_2$).
* With tight limits every corner holds cloud, so the $y_1>y_2$ / $y_2>y_1$ corner labels
  (`SLOT_LABEL_POS`, `FS_SLOT`, removed) became legend entries, and the legend moved outside the
  box on the right -- where sharedreps (a) has its colorbar. `GRID_WSPACE` went 0.34 -> 0.62 to hold it.

**Arrows in (a)** are chosen by hand: `ARROW_N14 = [6150, 9459]` (#6262 dropped: its gate arrow
contradicted its y2-selective field in (b), see the caveat below; the panel is illustrative), all MLP14 and all shown
in (b). This replaces the old automatic rule (`ARROW_MIN_FRAC`, `ARROW_MIN_SEP`), and the MLP15
neuron (#12784, in-plane 0.028 = 1.2x chance) is no longer drawn -- it stays in (c). #6150 is the
right-pointing direction (+2 deg); #682 was considered and rejected: its read direction is at
-88 deg with in-plane fraction 0.013, below chance. Every arrow is drawn at the same length; the
in-plane fractions are printed under the cell (2digit: #6262 0.215, #6150 0.305, #9459 0.033;
chance sqrt(2/d) = 0.024), so #9459's arrow is a direction, not evidence that it reads the plane.

## Publication pass

* Font sizes at the numberreps / sharedreps scale or above: `FS_AXIS` 28, `FS_TICK` 22, `FS_PANEL` 34,
  `FS_LEGEND` 22, `FS_NEURON` 20, `FS_CBAR_LABEL` 26, `FS_CBAR_TICK` 21, `FS_BAR_TICK` 22,
  `FS_ANNOT` 22. `FIG_SIZE` is (23.5, 12.6); `GRID_WSPACE` back to 0.40 now (a) has no outside legend.
* No grids anywhere ((d) and (e) had horizontal y-grids).
* (e): the values printed over the bars are gone (as in sharedreps (e)); tick labels broken onto
  short lines so neighbours do not touch; y-limit 1.2.
* (a): x ticks at -5, 0, 5 as in sharedreps (a). (d): y-label is "count" (the longer
  "number of pairs" ran into (c)'s colorbar).
* (a) has **no legend**. Labels sit on the plot, each with a white halo:
  * the neuron number at each arrow tip, in the arrow's colour (all three are MLP14; the caption
    says so). Right-pointing: beyond the tip (`ANNOT_OFF` = 1.13). Left-pointing: centred on the
    tip -- beyond it runs into the y axis -- above it if the arrow rises (#9459), below it if it is
    flat (#6262, since above would sit under the rising arrow);
  * "$y_1 = y_2$" at the upper end of the dashed line;
  * "$y_2 > y_1$" and "$y_1 > y_2$" on empty space only: $y_2 > y_1$ in a strip of headroom above
    the blue cloud (`HEADROOM_A` = 0.14 of the y range is added to the top limit; the box aspect is
    unchanged), $y_1 > y_2$ bottom right in a matching strip below the cloud (`FOOTROOM_A` = 0.10).
* (a) carries the $\mathbf{v}_1$ / $\mathbf{v}_2$ compass as in sharedreps, lower left
  (`COMPASS_BOX_A`), where the cloud leaves an empty corner. The angle comes from the saved `v1`,
  `v2` with the plot's own axis signs (`plane_sgn`): cos = -0.082, 94.7 deg, the same as sharedreps.
* (b): the three neurons drawn as arrows in (a) are marked -- frame (`HL_LW` = 3.2) and neuron
  number (bold) in that arrow's colour.

## Caveat: the (a) arrows are gate read directions, and gate is not what (b) shows

(b) shows the SwiGLU activation SiLU(gate.x) * (up.x); (a)'s arrows are the in-plane part of the
gate row only (`read_gate_prof_L14`: gate_proj row * RMSNorm weight, projected on the signed plane
basis). For these neurons the two disagree. Over the cloud (2digit):

| neuron | gate: angle, R2 on (log y1, log y2) | up: angle, R2 | activation gradient in plane (lstsq of act on pq), R2 |
|---|---|---|---|
| #6262 | -178 deg, 0.04 | -33 deg, 0.92 | -107 deg, 0.72 |
| #6150 | +2 deg, 0.87 | +177 deg, 0.00 | -179 deg, 0.74 |
| #9459 | +121 deg, 0.04 | -177 deg, 0.59 | -174 deg, 0.32 |

#6262's y2 selectivity in (b) lives in its **up** projection, not its gate. The panel's axes are
also scaled independently (y range ~3.6x x range), so any direction with a v2-perp component looks
flatter than it is. Decision: keep gate arrows as an illustration, drop #6262, keep #6150 and #9459.

**Arrow colours** (`ACOL`, in `ARROW_N14` order): #6150 black, #9459 crimson `#B2182B`. Teal was
invisible over the blue cloud; crimson has min OKLab dE >= 16 (normal, deuteranope, protanope
simulations) to both clouds, black and the PRGn ends, and is only near the dashed grey $y_1=y_2$
line (5.2), which differs in line style.

**Receptive-field colour map**: trimmed PRGn (`trim_cmap('PRGn', 0.16)`), shared with
`makefig_multicompare` (c); that figure's $y_3$ was moved from green to crimson so the two do not
collide. (A grey/purple map was tried and reverted.)

**Panel (b) neurons.** Exactly the six MLP14 members of the 12 shared comparator neurons
(`overlap_k{FIN_OVK}` = `overlap_k20`: in the top 20 of both perturbation cases; the same set (e)'s
bar 2 freezes), in a 2 x 3 grid ordered by worst-case attribution rank as in the appendix's
receptive-field figure: #6262 (1/1), #6150 (2/2), #5076 (5/5), #9459 (3/8), #6854 (10/7),
#15775 (20/16). The cell asserts 12 shared / 6 in MLP14. It previously drew the first 8 MLP14
neurons of `overlap_k30` ordered by best rank, which added #5222 (23/11) and #682 (24/13) -- neurons
outside the 12 (`FIN_N14` removed).

**Square fields in (b).** `rf_axes` draws each field with `aspect='equal'`; the colorbar anchors call
`apply_aspect()` first so they measure the shrunken square boxes. To let the squares fill (b)'s
width while keeping the rows near-equal (a much taller top row left white space under (a)):
`ROW_HEIGHTS` = (1.38, 1.0), `FIG_SIZE` = (23.5, 14.6), `RF_WSPACE, RF_HSPACE` = 0.04, 0.14. The
fields are square like (c)'s. (a), whose
box aspect is fixed to match sharedreps, is anchored to the top of its cell so its letter stays level
with (b)'s.

**One L15 DAS direction in (d) and (e).** `FIN_DASFIT = CASES[0]` (y1 perturbed) selects the single
rank-`FIN_DASRANK` layer-15 DAS direction used in both panels: (d) projects the cloud onto it
(`das15_proj_{FIN_DASFIT}_k1`), and bar 3 of (e) evaluates that same direction in both cases
(`das15_iia_{case}_from_{FIN_DASFIT}_k1`). Previously bar 3 refitted the direction per case
(`das15_iia_{case}_from_{case}_k1`, 1.00 / 1.00), so the two cases did not share a direction. With
one direction the bars are 1.000 (y1 perturbed) and 0.655 (y2 perturbed); the cell prints both. The
per-case fits and the full cross-case matrix remain in the appendix transfer figure. No new compute:
all arrays were already in `results/comparator_data_2digit.npz`.
