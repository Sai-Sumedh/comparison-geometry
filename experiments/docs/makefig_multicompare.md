# makefig_multicompare.ipynb -- main-paper figure

Pure plotting pass over the cached `results/multicompare*_2digit.npz` files (no model, no GPU). The
earlier 3x4, 12-panel version is frozen in `makefig_multicompare_appendix.ipynb` and is the
appendix figure `figs_appendix/app_k3_summary.pdf`; this notebook is the reduced main-paper figure,
saved as `figs/multicompare_summary.png` (PDF line commented, to `figs_mainpaper/`).

## Layout: 2 x 3

| | panel | content | old (3x4) panel |
|---|---|---|---|
| row 1 | (a) linear codes at $y_3$ | components along $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$ at the position of $y_3$ against the number each carries, 9 equal-width bins, mean +- SE | new |
| | (b) causal effect at $y_3$ | PR of each direction alone and of their pre-MLP span, by perturbed number (the "separate patches" and full-L14 bars are dropped) | (b) |
| | (c) receptive fields at $y_3$ | MLP14 #5076 and MLP15 #12784, the three pairwise marginals of the $(y_1,y_2,y_3)$ cube; no ticks | (c) |
| row 2 | (d) the two flags | L15 rank-1 DAS component at $y_2$ (top) and $y_3$ (bottom); the number that is the max filled in its colour, "not max" a grey outline | (h) over (d) |
| | (e) answer at last token | L20 last-token PCA coloured by the max slot | (i) |
| | (f) rank-limited vs full | PR of each flag's DAS $k=1$ alone (no full-rank bar), and of the 2 PCs of (e) vs the full L20 residual in the three orders; dashed PR = 1 | the (d), (h) insets + (l) |

(f) values (2digit): flag at $y_2$ 3.21, flag at $y_3$ 2.02 (the full-L15 values, 0.20 and 1.02,
are no longer drawn), answer $y_1>y_2$ 0.86 / 0.85,
$y_1>y_3$ 0.72 / 0.73, $y_2>y_3$ 0.08 / 0.06 (ranked / full).

## Panel (a): how the components are computed

For $j = 1, 2, 3$ the component is $(X - \bar X)\,\mathbf{v}_j$ on the 1500-prompt cloud at the
position of $y_3$, with $X$ the space $\mathbf{v}_j$ lives in: head L14.H14's output for
$\mathbf{v}_1$, head L14.H18's output for $\mathbf{v}_2$, the L13 residual for $\mathbf{v}_3$ -- the
same construction as `makefig_appendix.fig_k3_directions`. The three spaces have different scales,
so each component is **z-scored** to share one axis, then sign-aligned to rise with its number.
Points are the mean +- SE over `A_BINS` = 9 equal-width bins of $y_j$ (per-value points overplotted
into noise). The unscaled per-value components are appendix Figure `fig:app-k3-directions`.

## Colour rules

One meaning per colour: amber / blue / sand (`#DDCC77`, Paul Tol's "muted" sand) are **only**
$y_1$ / $y_2$ / $y_3$ (`NCOL` in the style cell; $y_3$ was green until the receptive fields went back
to PRGn, then briefly crimson, which read as too harsh). Sand was the only soft candidate with min
OKLab dE >= 12 to amber, blue, both PRGn ends, the teal bars and grey, under normal, deutan and protan
vision; rose/pink options fell to 2-6 against the teal or the PRGn green, and deeper sands collided
with amber. Sand is pale on white (dE 18.5), so (e) draws the $y_3$ points opaque.
(f)'s flag bars (DAS $k=1$) take the colour of the number perturbed in their experiment -- blue for
the flag at $y_2$ (case $y_2$), sand for the flag at $y_3$ (case $y_3$), matching (d) -- with a split
blue/sand legend swatch; the answer bars (2 PCs) teal `#35978F` (`RANKED_COL`), full rank
light grey (`FULL_COL`), with a three-entry legend; (d)'s
"not max" is a grey outline (`NOTMAX_COL`) drawn over the filled group so nothing blends.
(c) uses trimmed PRGn (green positive, purple negative), **the same map as `makefig_comparator`
(b)/(c)**, so the receptive fields read the same in both figures. (d)'s x label is "flag".

## Style

Fonts at the scale of the other main figures (`FS_AXIS` 26, `FS_TICK` 20, `FS_PANEL` 31), no grids,
frameless legends, no insets, and no reference lines in (b) or (f) (the zero line, the
dividers and (f)'s dashed PR = 1 were removed). `D_HEADROOM` = 1.45 opens space above each (d) histogram for its
legend and position label.
