# `data_numberreps.py` + `makefig_numberreps.ipynb` — Documentation

Collects, for the paper figures, everything about how Qwen2.5-7B-Instruct represents the two numbers
of a `max(a, b)` prompt in the **layer-13 residual stream**, and the causal role of a learned rank-1
direction in that space. The `.py` file does all the GPU work and writes `results/`; the notebook
only loads `results/` and draws, so figure style can be iterated without re-running the model.

Files:

| file | role |
| --- | --- |
| `data_numberreps.py` | all model work; writes `results/numberreps_{2digit,3digit}.npz` and `results/meta.json` |
| `makefig_numberreps.ipynb` | loads `results/`, draws figures into `figs/` |

---

## Setup borrowed from e11

Taken verbatim from `e11_comparison_largemodels`, so the numbers here are comparable with the
figures already in that experiment:

- **Model** `Qwen/Qwen2.5-7B-Instruct`, fp16, 28 layers, `d_model = 3584`, `SEED = 52`.
- **Prompt** (`prompt_gen`) — the one-shot template from `generate_manifolds_compare.ipynb` /
  `das_linear_vs_manifold.ipynb`:
  `'Answer in the following format with a single answer. The maximum of 12 and 4 is 12. The maximum of {n1} and {n2} is '`.
  The one-shot anchor stays `max(12, 4)` in **both** digit regimes, so the only thing that differs
  between regimes is the question clause.
- **Site** — `LAYER = 13`, **zero-indexed**, hooked as `model.model.layers[13]`'s forward output,
  i.e. the residual stream *leaving* decoder block 13. This is the same hook point that produces
  `cloud_resid_a[13]` / `cloud_resid_b[13]` in `generate_manifolds_compare.ipynb` and the same layer
  index used in its `DAS_LAYERS = [13, 14, 15, 17, 20]`. (In `hidden_states` terms that is
  `hidden_states[14]`, since index 0 there is the embedding — no such offset applies here because we
  hook the block directly.)
- **Token positions** — the *last* token of each number. Qwen2.5 tokenizes numbers digit-by-digit,
  so this is the last digit. Verified layout: 2-digit prompts are `T = 36` tokens with the numbers'
  last tokens at 29 and 33; 3-digit prompts are `T = 38` with last tokens at 30 and 35.
  `regime_positions()` derives these at run time rather than hard-coding them, and every batch
  asserts it matches the regime template.

## Digit regimes

`REGIMES` runs the entire pipeline twice:

| tag | number range | triple `gap` | `fit_lo` |
| --- | --- | --- | --- |
| `2digit` | `[10, 100)` | 10 | 25 |
| `3digit` | `[100, 1000)` | 100 | 250 |

`gap` is the minimum separation inside a counterfactual triple `a > b + gap > c + 2*gap`, scaled
with the range so both regimes pose the same "how far apart are the numbers" problem. `fit_lo` is
the lower cutoff for the masked log fit — `generate_manifolds_compare.ipynb` fits the first-number
panel over `a_vals >= 25`; 250 is the same fraction of the 3-digit range.

---

## Section-by-section: what `data_numberreps.py` computes

### 1. Representation cloud (request item 1)

`sample_cloud_pairs(N_CLOUD=2000, ...)` draws 1000 distinct *unordered* pairs and mirrors each to
`(b, a)`, so the label "which slot holds the max" is exactly 50/50 by construction and the cloud has
no implicit slot confound. This is `generate_manifolds_compare.ipynb`'s cloud convention. Pairs are
additionally required to have **distinct leading digits** (`leading_digits_distinct`) — matching
`das_linear_vs_manifold.ipynb`'s `sample_pairs`, and keeping the cloud population identical in kind
to the patching population, where a distinct leading digit is what makes the first-token readout
unambiguous.

`capture_resid` runs the cloud in batches of 20 with a single forward hook on layer 13 and keeps the
residual at the two number positions:

- `resid_a` — `(2000, 3584)` float32, **first** number's last digit.
- `resid_b` — `(2000, 3584)` float32, **second** number's last digit.

> **Note on the effective sample size.** Because every unordered pair is entered in both orders,
> 2000 rows come from 1000 independent draws. The 2000 points in a log fit are not independent —
> each shares its partner number with one other row. It does not change the qualitative fits, but
> it makes nominal standard errors slightly optimistic. The size was raised from 400 mainly because
> PCA in 3584 dimensions from 400 samples gives a rank-deficient covariance and noisy eigenvector
> estimates, and `pc1_dir` / `pc1b_dir` are *used as interventions*, so that noise propagates into
> the `pc1` / `pc1b` bars.

Also saved: `cloud_nums`, `a_vals`, `b_vals`, `max_vals`, `position_label` (0 = first larger,
1 = second larger).

### 2. 3D PCA and variance explained (request item 2)

For each of the two clouds, an `sklearn.decomposition.PCA(n_components=3)` fit (mean-centred SVD):

- `pca3_coords_{a,b}` — `(2000, 3)` cloud coordinates.
- `pca3_components_{a,b}` — `(3, 3584)` the three directions.
- `pca3_evr_{a,b}` — **explained variance ratio per direction**, i.e. 3 numbers, not a sum.
- `pca3_evals_{a,b}` — the raw eigenvalues.
- `pca3_mean_{a,b}` — the cloud mean the PCA centred on.
- `pca_evr_spectrum_{a,b}` — the first 20 ratios, for a cumulative-variance curve.

`logfit_pc1_{a,b}` (in `meta.json`) carries the same log fit as section 4 applied to PC1, so the
notebook can draw the unsupervised baseline alongside DAS for free.

### 3. The DAS 1D direction (request item 3)

**Method.** Distributed Alignment Search, single-causal-variable case (Geiger, Wu, Potts et al.,
*Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations*,
2023) — the same implementation used in `e11/das_linear_vs_manifold.ipynb` and
`e11/generate_manifolds_compare.ipynb`. A `d_model × 1` orthonormal basis `Q` is parametrized by a
raw matrix through a **QR retraction** (`torch.linalg.qr`), the standard differentiable
parametrization for optimizing on the Stiefel manifold, and trained so that the rank-1 interchange

```
h' = h + Q Qᵀ (h_clean − h)
```

at layer 13 makes the corrupted run answer the clean maximum's slot value. All base-model parameters
are frozen (`requires_grad_(False)`), so only `Q` receives gradients.

**Counterfactual design** (from `das_linear_vs_manifold.ipynb`). From each triple `a > b > c` and a
slot `s` holding the clean maximum:

- clean = `a` at slot `s`, `b` at the other slot → the model answers `a`.
- corrupted = `c` at slot `s`, `b` at the other slot → the model answers `b` (= `tok_b`).
- a successful interchange makes the corrupted run answer **its own** slot-`s` value `c` (= `tok_nc`).

`tok_nc` / `tok_b` are the numbers' *first* tokens (leading digits), which is why the triples must
have distinct leading digits.

**Where it is trained.** At the **first-number position**, on the `n1 larger` case only (slot 0
holds the clean max). That is the only case in which a first-number stage exists at all — with the
maximum at slot 1 there is nothing about the first number to transport. This corresponds to the
`L0–L13 @ n1 last token` band in `das_linear_vs_manifold.ipynb`'s pipeline, where L13 is the last
layer of the first-number band.

**Training.** `DAS_STEPS = 100` Adam steps at `DAS_LR = 0.05` on `N_DAS = 128` triples, disjoint
from the `N_EVAL = 400` evaluation triples.

> **How the split is made.** A single deduplicated pool of `N_EVAL + N_DAS` triples is drawn and
> then split — disjoint by construction. Drawing the two sets independently from different seeds
> does *not* work at these sizes: the 2-digit population is only ~54,700 valid triples, so two
> independent draws of 400 and 128 have an expected overlap of ~0.9 and collide in practice. Since
> `build_triples` is a deterministic prefix in `n`, `pool[:N_EVAL]` is exactly what
> `build_triples(N_EVAL, SEED, ...)` used to return, so the evaluation set is unchanged. The loss (`cf_pair_loss`) is cross-entropy
toward `tok_nc` restricted to `{tok_b, tok_nc}` — the actual DAS objective, since IIA is an argmax
and has no gradient. A separate direction is trained per digit regime.

Gradients accumulate over minibatches of `N_DAS_BS = 32` (weighted by minibatch size, so the step
is the full-batch mean gradient). This decouples the training-set size from the activation memory
of the backward pass, which has to store every layer above `LAYER`.

> **Why 128, not 16.** e11's notebooks use 16, but that was a *GPU-memory* decision, not a
> statistical one — `docs/das_linear_vs_manifold.md` lists `DAS_N_TRAIN_PER_CASE` as "the first
> knob to turn down if the backward pass OOMs", because those notebooks held all-layer activation
> caches alongside training. Nothing here does, so the constraint does not apply. It matters:
> `Q` is a point on the 3583-sphere and the objective constrains it only through `N_DAS` scalar
> projections `(h_clean − h)·Q`, so at n = 16 the overwhelming majority of directions are
> unconstrained and many `Q` reach the training optimum. At n = 16 the earlier run showed
> train IIA 1.000 against held-out 0.688 — the signature of partial overfitting.

**Seed stability.** `DAS_N_SEEDS = 5` directions are trained from different random inits
(`init_seed = SEED + 1000*i`) on the **same** training data. Seed 0 is the direction used
everywhere else; the rest exist to answer whether the solution is determined at all. Small *n* in a
high-dimensional space does not by itself invalidate the result — what matters is whether
optimization repeatedly lands in the same place.

Reported and saved: per-seed held-out IIA / position recovery, and the pairwise cosine matrix.
Signs are arbitrary for a DAS direction, so the summary uses `|cos|`, against the
`1/sqrt(d_model) ≈ 0.017` chance level.

- **Pairwise `|cos|` near 1 with a tight IIA spread** → well-determined; "the direction" is fair.
- **Near-orthogonal directions at comparable IIA** → badly underdetermined; the singular framing
  would have to go, and the honest claim becomes "a rank-1 subspace suffices", not "this one does".

Saved: `das_dir` `(3584,)` unit vector (= seed 0), `das_loss_curve` `(100,)`, `das_train_posrec` /
`das_train_iia` (seed 0 on its own training batch), `das_train_triples`, plus
`das_seed_dirs` `(5, 3584)`, `das_seed_cos` `(5, 5)`, `das_seed_curves` `(5, 100)`,
`das_seed_iia` / `das_seed_posrec` `(5,)` (held out) and their `_train` counterparts.
`meta.json` carries `das_seed_cos_absmean`, `das_seed_iia_mean`, `das_seed_iia_std`.

**Projection into the 3D PCA space** (second half of item 3): `das_in_pca3_{a,b}` is the `(3,)`
vector of cosines `das_dir · PC_k`, i.e. the direction expressed in that cloud's PCA basis, and
`das_pca3_captured_{a,b} = ‖P Pᵀ w‖` (with `w` unit) is the fraction of the direction that lies
inside the 3D PCA subspace — 1.0 would mean the learned direction is entirely contained in the top-3
PCA space, `sqrt(3/3584) ≈ 0.029` is chance.

### 4. Component along the fixed DAS direction, and its log fit (request items 4 and 5)

Each cloud is mean-centred on its own mean and projected onto the **same fixed** `das_dir`:

- `das_comp_a` — first-number activations along the DAS direction, `(400,)`.
- `das_comp_b` — second-number activations along the same direction, `(400,)`.
- `das_comp_{a,b}_raw` — the un-centred projections, if an absolute offset is ever wanted.

`log_fit_block` then fits `y = p·log(x) + q` by least squares (`np.polyfit` on `log x`), the same
`fit_logx` used in `generate_manifolds_compare.ipynb`, against the number living at that position
(`a_vals` for `das_comp_a`, `b_vals` for `das_comp_b`). Each fit is reported twice — over **all**
points (`p_all`, `q_all`, `r2_all`) and over `x >= fit_lo` (`p_masked`, `q_masked`, `r2_masked`),
since the low end of the range departs from the log trend — plus the raw Pearson `r`. These land in
`meta.json` under `logfit_das_{a,b}` (and `logfit_pc1_{a,b}` for the PCA baseline).

*(The request numbered two items "4"; both are covered here — the component itself and its log fit.)*

### 5. Patching metrics (request item 5)

At layer 13, for **each of the two cases**, **each of two intervention scopes**, and **each of the
two number positions**:

| scope | hook | meaning |
| --- | --- | --- |
| `full` | `make_full_patch_hook` | overwrite the entire layer-13 residual at that position with the clean run's |
| `das` | `make_subspace_patch_hook(Q, …)` | overwrite only the component along `das_dir` |
| `pc1` | `make_subspace_patch_hook(Q_pc1, …)` | overwrite only the component along `pc1_dir` |

`pc1_dir` is PC1 of the **first-number** cloud (`pca3_components_a[0]`, renormalized) — the same
cloud and the same rank as DAS, so the two rank-1 conditions differ only in how the direction was
found: trained on the counterfactual objective versus the top-variance direction of the
representation. This is the PC1 baseline used in `e11/das_linear_vs_manifold.ipynb` and
`e11/generate_manifolds_compare.ipynb`. Sign is irrelevant — the intervention uses `Q Qᵀ`. Also
saved: `pc1_dir` `(3584,)` and `das_pc1_cos`, the full-space cosine between the two directions.

Metrics (`posrec_and_iia`), both per-example arrays of length `N_EVAL = 400`:

- **Position recovery** `= (PLD_patched − PLD_corr) / (PLD_clean − PLD_corr)` with
  `PLD = logit[tok_nc] − logit[tok_b]` — the normalized-restoration convention of Meng, Bau,
  Andonian & Belinkov 2022 (ROME), as used throughout e11.
- **IIA** `= 1[argmax over the full vocabulary == tok_nc]` — Interchange Intervention Accuracy
  (Geiger, Lu, Icard & Potts 2021). A full-vocabulary argmax, so it is strictly harder than the
  two-way training loss; a low loss does not imply IIA = 1.

Both cases (`n1 larger` = first number larger in the clean prompt, `n2 larger` = second number
larger) use the *same* 400 triples, just with the clean max placed at the other slot.

**Why both patch positions are saved.** The band pipeline in `das_linear_vs_manifold.ipynb` scores
layer 13 at the clean maximum's own slot — first-number position for `n1 larger`, second-number
position for `n2 larger`. That cell is the one to plot; the other position is the natural control
and costs one extra forward pass, so both are stored and the notebook's `PATCH_AT` switch picks.

Key naming: `{posrec|iia}_{full|das|pc1}_{n1_larger|n2_larger}_{first_number|second_number}`.

Also saved per case: `triples_*`, `clean_pairs_*`, `corr_pairs_*`, `pld_clean_*`, `pld_corr_*` (the
un-patched anchors, so recovery can be recomputed or reported unnormalized).

---

## Output layout and size

`save_regime` splits each regime's result dict: every `np.ndarray` goes into
`results/numberreps_<tag>.npz` (compressed), everything else (the log-fit blocks, the `meta` block)
into `results/meta.json` under `regimes.<tag>`.

The script ends with a **size report** to stdout (so it lands in `final_numberreps.out`): every
array's shape, dtype and MB, the uncompressed total per regime, the compressed on-disk size of each
`.npz`, and the grand total for `results/`. The dominant arrays are `resid_a`/`resid_b` at
`400 × 3584 × 4 B ≈ 5.7 MB` each, so expect roughly **12 MB uncompressed per regime, ~25 MB total**
before compression.

## Runtime and resources

The run used 1 GPU, 100 GB host RAM, 8 cores and a 3-hour time limit. The real cost is the DAS
backward pass — 100 steps × 4 minibatches × 5 seeds × 2 regimes, each backpropagating activations
(not weights) through layers 14–27 for a 32-prompt minibatch. That is ~40× the DAS work of the
n=16 single-seed version, so expect minutes rather than the previous ~14 s per regime. `model.config.use_cache = False` is set globally and
`gpu_mem()` prints allocated/reserved memory at each phase boundary.

If it OOMs, in order: lower `N_DAS_BS` (32 → 16 → 8; this does not change the training
set, only the accumulation granularity), lower `DAS_N_SEEDS`, lower `DAS_STEPS`, or enable
`model.gradient_checkpointing_enable()` before training.

---

## `makefig_numberreps.ipynb`

Pure plotting. Loads all regimes into `R[tag]` (npz) and `META` (json); a single style cell at the
top defines every font size, colour and marker constant that the figure cells read, plus
`style_axes` / `style_axes3d` / `savefig` helpers. Set `SAVE = False` to iterate without writing
files.

| section | figure | saved as `figs/makefig_numberreps_…` |
| --- | --- | --- |
| 1 | 3D PCA scatter, rows = regime, cols = number position, coloured by that position's own number; titles carry per-direction EVR | `pca3d.png` |
| 1 | the same scatter coloured by which slot holds the max | `pca3d_bypos.png` |
| 2 | per-direction EVR bars + cumulative-variance spectrum | `variance_explained.png` |
| 3 | DAS 1D direction drawn as an arrow inside each 3D PCA space, with the captured fraction | `das_in_pca3d.png` |
| 3 | the same three cosines printed numerically | — |
| 4 | DAS component vs the number, with the log fit overlaid (`USE_MASKED_FIT` toggles which fit) | `das_logfit.png` |
| 4 | DAS component and PC1 on the same axes, with Pearson `r` | `das_vs_pc1_component.png` |
| 5 | position recovery and IIA bars, full residual vs DAS 1D vs PC1, both cases (`PATCH_AT` selects the position; `'band'` = clean max's own slot) | `patching_band.png` |
| 5 | every (case × patch position × scope) cell printed as a table | — |
| 5 | DAS training loss curves | `das_training.png` |

### 3D plot conventions

Both 3D-PCA figures and the DAS-arrow figure share two conventions, set in the style cell:

- **`PC_ORDER = (1, 2, 0)`** — plot axes are `(x, y, z) = (PC2, PC3, PC1)`, so **PC1 is the vertical
  axis and increases upward**. `xyz_for_plot()` applies the reorder to cloud coordinates and to the
  DAS direction alike, so the arrow stays consistent with the points.
- **`manifold_curve()`** — the grey trace. Points are rank-sorted by the number living at that
  position, split into `CURVE_BINS = 14` equal-count groups, each group's mean position is taken,
  and the resulting polyline is moving-averaged over `CURVE_SMOOTH = 3` bins. It shows the path the
  representation takes as the value increases, without asserting any parametric form. Equal-count
  (rank) bins rather than equal-width value bins, so every marker carries the same number of points.
  `SHOW_CURVE = False` turns it off.

Switches worth knowing: `SAVE`, `DPI`, `SHOW_CURVE`, `CURVE_BINS`, `CURVE_SMOOTH`, `USE_MASKED_FIT`,
`PATCH_AT`, `ARROW_SCALE`, and the `COL` / `METHOD_NAME` dicts. The patching cells fall back to
whatever scopes are present in the `.npz`, so a results file written before the PC1 baseline was
added still plots (just without the PC1 bar).

---

## Notes and caveats

- The `3digit` prompts keep the 2-digit one-shot anchor (`max(12, 4)`). This is deliberate — the
  exemplar is held fixed so the regime comparison isolates the question clause — but it does mean the
  3-digit prompts are slightly out of distribution relative to their own exemplar.
- The DAS direction is trained per regime; the 2-digit direction is **not** evaluated on 3-digit
  data. Adding that transfer check would be a `das_dir` swap in the section-4/5 loops.
- `position_label` is balanced by construction in the cloud, but the patching triples are built
  independently per case from the same `a > b > c`, so the two cases are matched on values, not
  resampled.
- The cloud requires distinct leading digits, which slightly thins the population relative to
  `generate_manifolds_compare.ipynb`'s cloud (which did not). It matches
  `das_linear_vs_manifold.ipynb`'s cloud instead.


---

## Evaluation batching

Every evaluation forward pass runs in minibatches of `EVAL_BS = 48`. The binding constraint is not
activations but **logits**: `model(ids).logits` is materialized for every position before anything
is sliced, i.e. `B x T x |vocab| x 2` bytes — about 0.5 GB at `B = 48, T = 36` for Qwen2.5's 152k
vocabulary, and 4.5 GB at `B = 400`. Raising `N_EVAL` therefore required batching, not just a
larger constant.

Three functions carry it:

- **`clean_hidden(ids, bs)`** — captures the L13 residual in chunks and concatenates.
- **`pos_anchors(bat, bs)`** — reduces each chunk to the two token logits before moving on.
- **`eval_patched(factory, bat, ch, bs)`** — scores one intervention. It takes a hook **factory**,
  not a hook, because the clean-activation source has to be sliced to match each minibatch;
  `factory(ch[lo:hi])` builds the hook per chunk. Position recovery and IIA are reduced per
  minibatch and concatenated, so the full `(N_EVAL, |vocab|)` logits tensor is never held.

`run_patched` survives only inside `train_das_1d`, where the minibatch is `N_DAS_BS = 32` and the
autograd graph is needed.

## Expected cost

Forward-pass accounting per regime, at the current settings:

| stage | passes |
| --- | --- |
| cloud capture | 100 forward (bs 20) |
| **DAS training** | **2000 forward+backward (bs 32)** |
| seed-check evaluation | 60 forward (bs 48) |
| anchors + `clean_hidden` | 78 forward (bs 48) |
| patching sweeps | 144 forward (bs 48) |

The DAS seed loop (`DAS_STEPS x ceil(N_DAS/N_DAS_BS) x DAS_N_SEEDS = 100 x 4 x 5`) dominates
everything else by an order of magnitude. Extrapolating from the measured 13.7 s run at
`N_CLOUD=400, N_DAS=16, N_EVAL=48, 1 seed`, expect roughly **5-8 minutes per regime, ~10-20 minutes
total** including model load — still far inside the 3-hour wall. To cut it, lower `DAS_N_SEEDS`
first (the stability check is a one-off diagnostic, not something every run needs).

Result size: **~58 MB uncompressed per regime**, ~97% of it `resid_a` + `resid_b`
(`2 x 2000 x 3584 x 4` bytes). At the compression ratio observed on the current files (0.59, since
float32 activations compress poorly) that is **~34 MB per `.npz`, ~69 MB for `results/`**. The
script prints the exact figures at the end of every run.
