# Appendix figures: code documentation

Produces every figure of `paper/appendix_results.tex` (Appendix D) into `figs_appendix/*.pdf`, plus
`figs_appendix/tab_patch_examples.tex`. Run everything from `experiments/`.

## Files

| file | role |
|---|---|
| `appendix_style.py` | shared style and helpers for both plotting scripts |
| `makefig_appendix.py` | 19 figures drawn only from existing `results/*.npz` (no model) |
| `compute_app_behaviour.py` | GPU: behavioural accuracy and patched generations |
| `compute_app_trace_k2.py` | GPU: two-number causal trace and per-cell probes |
| `compute_app_trace_k3.py <case>` | GPU: three-number causal trace, one case per job |
| `compute_app_u_layers.py <layers>` | GPU: the layer-13 number analysis at earlier layers |
| `compute_app_v2_shortcut.py` | GPU: rank-1 DAS at the position of y2 |
| `makefig_appendix_computed.py` | 7 figures and the example table from the compute outputs |

Workflow:

```bash
python makefig_appendix.py                 # zero-compute figures, any time
python compute_app_behaviour.py            # GPU; the compute_app_* scripts are independent
python compute_app_trace_k2.py
python compute_app_trace_k3.py y1          # and y2, y3
python compute_app_u_layers.py 1 3 5 7 9 11
python compute_app_v2_shortcut.py
python makefig_appendix_computed.py        # after the jobs finish; skips figures whose data is missing
```

## Hyperparameters

Everything the compute jobs do reuses the modules in this directory and their constants, so the results sit on the
same measurement code and populations as the main figures: `SEED = 52`; the 400 held-out
counterfactuals (`build_triples(N_EVAL + N_DAS)[:400]` for two numbers, `eval_quads` of
`results/multicompare_pool_2digit.npz` for three); the 2000-prompt cloud `sample_cloud_pairs(2000,
SEED + 7)`; DAS with 100 Adam steps, learning rate 0.05, minibatch 32 and 128 fitting examples,
initialised with `init_seed = SEED` (seed 0 of `data_numberreps.py`); the 13-point ridge penalty grid
`logspace(-2, 4, 13)` with leave-one-out selection and a 20% held-out split
(`train_test_split(..., random_state=SEED)`).

## `appendix_style.py`

Palette `COL` / `NCOL` / `CASE2` copied from the main makefig notebooks (full = blue, u = amber,
PC1 / v2 = orchid, H14 = teal; y1 amber, y2 blue, y3 green). Colormaps: `SEQ_CM` (trimmed YlGnBu)
for scores, `DIV_CM` (trimmed PuOr) for signed maps, `RF_CM` (trimmed PRGn, as Fig. 4) for
receptive fields, `VAL_CM` (plasma, as Fig. 2) for clouds coloured by value. No red/green pairs.

Helpers: `style` (labels, tick sizes, removes top/right spines), `letter` (bold panel letter),
`legend` (frameless), `save` (PDF, tight bbox, 300 dpi for rasterized scatters), `mse` (mean, SE),
`aggregate` (mean and SE of y per distinct x or per bin), `align_sign` (orients a sign-arbitrary
component to rise with its number), `fit_logx`, `grouped_bars`, `heat` (annotated heatmap), `cbar`,
`wilson` (Wilson interval).

## `makefig_appendix.py`

`python makefig_appendix.py [name ...]`; with no names it draws all. Each `fig_<name>` writes
`figs_appendix/app_<name>.pdf`.

| name | data | panels |
|---|---|---|
| `u_manifold` | numberreps_2digit | L13 cloud at y1 (one point per value) in PC1–PC3 with its mean path and u as a unit arrow; component along u vs y1 with log fit |
| `k3_manifolds` | multicompare_dirs | H14 output, H18 output and L13 residual at y3, each in its own top-2 PCA (fitted here), with v1 / v2 / v3 as unit arrows |
| `u_vs_pc1` | numberreps_2digit | IIA / PR of full, u, PC1 at y1; standardized components vs y1; PCA spectrum vs variance along u (computed from `resid_a`) |
| `three_digit` | numberreps_3digit | u and v2 (PC1 of the y2 cloud, projected from `resid_b`) vs their numbers with log fits; IIA bars |
| `transport` | sharedrep | H14 output in its top 2 PCs coloured by y1 with the equal-count mean path; H14 attention to y1 vs y1 |
| `shared_probes` | sharedrep | components of the pre-MLP L14 cloud along p1, p2 (held-out prompts) vs y1, y2 |
| `shared_posrec` | sharedrep | PR version of the main Fig. 3e bars, joint conditions hatched |
| `mlp_freeze` | comparator_data | IIA under sub-block freezes (attn14 omitted, see caption); IIA under MLP injection |
| `attribution` | comparator_data | attribution scatter across cases (symlog) with the shared 12; top-k vs random-k freeze; shared / own / random freeze |
| `rf_shared`, `rf_exclusive` | comparator_data | receptive fields (`rf_acts_L*`, y1-major pairs, divided by peak); shared 12 by worst-case rank; top-6 case-exclusive per case |
| `connectivity` | comparator_data | causal edges and gate virtual weights restricted to the shared 6 x 6 neurons |
| `l15_direction` | comparator_data | L15 DAS IIA fitted same / other case vs full rank; write-alignment histograms with shared L15 neurons marked |
| `k3_heads` | multicompare_heads | per-head PR with L13 co-patch, top 8 heads, three orders, L13-alone baselines |
| `k3_directions` | multicompare_dirs, _ucos | u1, u2, v1, v2, v3 components vs their numbers (v's projected from the saved clouds) |
| `k3_dissociation` | multicompare_space, _ucos | direction x case PR heatmap |
| `k3_neurons` | multicompare_freeze, _attr, _transfer | sub-block freezes; top-k freeze; two-number neurons vs random |
| `k3_l15` | multicompare_das15, multicompare2_das15 | fit x eval PR matrices at y3 and y2 positions, with full rank |
| `k3_readout_pca` | multicompare_late | last-token PCA for L18 to L22 coloured by argmax slot |
| `k3_readout_causal` | multicompare_late, _latehead20 | last-token full-rank PR vs layer; L20 per-head PR with attention-sublayer references |
| `ablation` | multicompare2_ablate(_lastpos) | task accuracy vs k with the 12 neurons zeroed everywhere / at the last token / random 12 |

## Compute scripts

All print `PROGRESS i/N` progress lines.

- **`compute_app_behaviour.py`** -> `results/app_behaviour.npz`, `results/app_patch_outcomes.npz`.
  `greedy_numbers` decodes with a manual argmax loop, so a patch hook stays attached at every step,
  and returns the first integer of each continuation. Behaviour covers all 8,010 ordered two-digit
  pairs (3 tokens) and 4,000 random three-digit pairs (4 tokens). `patch_outcomes` decodes the 400
  corrupted prompts with no patch, with the full L13 residual patched at y2 (y2 larger), and with u
  patched at y1 (y1 larger). It then labels each answer as r (the corrupted prompt's value at the
  winning slot), b, a (clean max) or other.
- **`compute_app_trace_k2.py`** -> `results/app_trace_k2.npz`, `results/app_probes_k2.npz`. The trace
  sets `D.LAYER = l` for each layer, captures the clean residual once, and runs a full-rank patch at
  each question token (first token of y1 to the last token). The probes capture every layer's
  output at those tokens for the cloud with `output_hidden_states` (float16 on CPU). `ridge_loo` then
  fits dual-form ridge on the GPU. Each (layer, token) cell uses one eigendecomposition of the
  training Gram matrix, which gives the exact leave-one-out error for every penalty (the RidgeCV
  rule). Targets are log y1, log y2 (held-out R²) and ±1 for y1 > y2 (held-out sign accuracy).
- **`compute_app_trace_k3.py <case>`** -> `results/app_trace_k3_<case>.npz`. The same trace for one
  three-number perturbation case (`MC.CASES`), on the cached eval quadruples.
- **`compute_app_u_layers.py <layers>`** -> `results/app_u_layers_L<l>.npz`. Per layer it captures
  the cloud at y1 and y2 positions and fits PCA to each. It trains u with `train_das_1d` at y1
  (y1-larger case), evaluates full / u / PC1(y1) at y1 and full / PC1(y2) at y2, and saves the
  components and the variance fraction along u. Layer 13 is read from `numberreps_2digit.npz`.
- **`compute_app_v2_shortcut.py`** -> `results/app_v2_shortcut.npz`. It trains rank-1 DAS at y2's
  position with y2 perturbed, scores full / DAS / PC1(y2), and projects the cached y2 cloud onto it.

## `makefig_appendix_computed.py`

Figures `behaviour` (accuracy vs number of operands only: pairwise accuracy is at ceiling, 100% and
99.9%, so `app_behaviour.npz` is reported in the text rather than plotted), `patch_outcomes`
(`patch_categories` adds a category for "leading digit of r followed by the last digit of a", which
is what the full-rank patch at y2's last token produces; `tab_patch_examples.tex` lists two
examples per category and intervention), `trace_k2`, `probes_k2`, `trace_k3`, `u_layers` (layers found on disk plus 13),
`v2_shortcut`. A figure whose inputs are missing is skipped with a message. Token tick labels mark
each operand's last token as y_j; its first token is left blank; the space token is drawn as ␣.
