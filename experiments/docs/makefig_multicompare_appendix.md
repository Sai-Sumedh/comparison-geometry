# makefig_multicompare_appendix.ipynb

A frozen copy of `makefig_multicompare.ipynb` as of 2026-09-24, kept so the appendix figure does not
change when the main-paper version is cut down. It is the same plotting pass (no model, no GPU) over
the cached `results/multicompare*_2digit.npz` files, and differs from the original only in where it
saves:

* `figs_appendix/app_k3_summary.pdf` -- the appendix figure, `\label{fig:app-k3-summary}` in
  `paper/appendix_results.tex` (Section "Three numbers").
* `figs/multicompare_appendix_summary.png` -- a PNG preview (renamed so it does not overwrite the
  main notebook's `multicompare_summary.png`).

Panel contents, and every number quoted in the caption, are documented in
`docs/explore_multicompare.md` (position of $y_3$, last token) and `docs/explore_multicompare2.md`
(position of $y_2$). The caption values were read from the cached results:

| panel | quantity | value |
|---|---|---|
| (b) | span pre-MLP PR, $y_1$ / $y_2$ / $y_3$ perturbed | 0.746 / 0.144 / 0.833 |
| (d) | L15 DAS $k=1$ PR vs full L15, case $y_3$ | 2.02 vs 1.02 |
| (f) | $\mathbf{w}_1$ alone PR, $y_1$ perturbed; full L14 | 0.665; -0.054 |
| (f) | $\mathbf{w}_2$ alone PR, $y_2$ perturbed | 3.08 |
| (h) | L15 DAS $k=1$ PR vs full L15, case $y_2$ | 3.20 vs 0.20 |
| (i) | L20 top-2 PC variance | 54% |
| (j) | H27 attention on the max slot ($y_1$, $y_2$, $y_3$ max) | 0.27, 0.54, 0.93 |
| (k) | H27 PR, order $s_1>s_2>s_3$ | 0.184 |
| (l) | full vs 2D-PCA PR, three orders | 0.85/0.86, 0.73/0.72, 0.06/0.08 |

The appendix text flags (f)'s $\mathbf{w}_1$ result as a possible subspace-patching illusion
(a rank-one patch succeeding where the full-rank patch fails; Makelov, Lange & Nanda, ICLR 2024,
added to `paper/appendix_methods.bib` as `makelov2024illusion`), rather than as evidence of a copy
of $y_1$ at the position of $y_2$.
