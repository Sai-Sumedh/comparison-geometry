# `data_sharedrep.py` — the shared-representation data collection

Batch-mode collection of everything the "shared representation" paper figure needs. It is the
companion to `data_numberreps.py` (L13 number representations) and the productionized version of
the interactive work in `explore_h14_copy.ipynb`.

Run it with `python data_sharedrep.py`. It writes `results/sharedrep_{tag}.npz` and
`results/sharedrep_meta.json`, and plots nothing.

---

## 1. Setting

Everything in this file lives at the **second-number position** (`POS_B`, the last token of the
second number in the question clause). The claim being measured is:

* by layer 13, the residual stream at each number's own position carries that number's magnitude
  (that is what `data_numberreps.py` established);
* head **L14.H14** then copies the *first* number into the *second* number's position;
* so the residual stream leaving layer 14 at `POS_B` holds **both** numbers, and the comparison can
  be read out from there.

Model, prompt, seed and counterfactual design are inherited unchanged from `data_numberreps.py`:
Qwen2.5-7B-Instruct, the one-shot `max(a, b)` prompt, `SEED = 52`, and triples `a > b + gap >
c + 2*gap` with distinct leading digits.

### What is *not* recomputed

The two L13 directions are **loaded** from `results/numberreps_{tag}.npz`, never retrained:

| direction | source array | meaning |
|---|---|---|
| `L13 DAS` | `das_dir` | the rank-1 DAS direction trained at the first-number position |
| `L13 PC1 @n2` | `pca3_components_b[0]` | PC1 of the L13 residual cloud at the second-number position |

That keeps the two result files describing one consistent model rather than two independently
fitted ones. `data_numberreps.py` must therefore have been run first.

The evaluation triples are built from the **same pool with the same split** as
`data_numberreps.py` (`build_triples(N_EVAL + N_H14_DAS, SEED, ...)`, then `pool[:N_EVAL]` for
evaluation and the rest for DAS training). `build_triples` is a deterministic prefix in `n`, so
`eval_triples` here is byte-identical to the evaluation set there and every number in this file
sits on the same examples as the ones already saved.

### Config knobs (top of the file)

| name | default | what it controls |
|---|---|---|
| `H_LAYER`, `H_HEAD` | 14, 14 | the head that is studied, and the layer whose heads are swept |
| `CASE` | `'n1 larger'` | the clean maximum sits at slot 0 — the only case in which a first-number copy stage exists |
| `TAGS` | `['2digit']` | digit regimes to run; add `'3digit'` to do both |
| `N_CLOUD` | 3000 | pairs in the representation cloud (both orders, balanced) |
| `N_H14_DAS` | 128 | held-out triples for training the H14-space DAS direction |
| `H14_DAS_BS` | 16 | DAS minibatch; gradients accumulate over the full `N_H14_DAS` |
| `RIDGE_ALPHAS`, `PROBE_TEST_FRAC` | logspace(-2,4,13), 0.2 | the ridge probes' CV grid and held-out split |

`N_EVAL` (400) and `EVAL_BS` (48) are imported from `data_numberreps.py`.

---

## 2. Per-head machinery

### `head_out(ctx, head)`

A head's own additive contribution to the residual stream is `z_h @ (W_O^h)^T`. `o_proj`'s output
is the sum over heads, so head `h`'s share is read off the **`o_proj` input** by slicing the
head-major block `[h*head_dim : (h+1)*head_dim]` and multiplying by the corresponding columns of
`W_O`. The `o_proj` bias is omitted, since it is not attributable to any single head
(convention from `e11/head_analysis.ipynb`).

### `head_patch_handles(head, pos, target, P=None)`

Registers a pre-hook (to capture the `o_proj` input) and a forward hook (to rewrite the output) on
layer `H_LAYER`'s `o_proj`. Because the output is a sum over heads, rewriting one head's
contribution is *exact*:

* `P is None` — full interchange: `out[:, pos] += target − current`, i.e. head `h` now writes what
  it wrote on the clean run.
* `P` given (`d_model × k`, orthonormal) — rank-`k` interchange **inside that head's own output**:
  `out[:, pos] += P Pᵀ (target − current)`.

Everything else in the forward pass — the other 27 heads, the MLP, all later layers — is untouched.

### `capture_cloud(ids, pos_a, pos_b)`

One forward pass returning three things at once: the L13 residual at `pos_a`, H14's own output at
`pos_b`, and the residual stream leaving layer 14 at `pos_b`. They are read in the *same* pass on
purpose — the inset in figure 1 plots the H14 component at n2 against the L13 DAS component at n1,
so those two must come from the same prompts.

### `clean_hidden_at(ids, layer, bs)`

`data_numberreps.clean_hidden` is pinned to `LAYER = 13`; the `L14 full @n2` condition needs layer
14's. Minibatched for the same reason as there: a forward pass materializes
`B × T × |vocab|` logits before anything is sliced, so the logits, not the activations, bound the
batch size.

### `capture_ctx(ids, pos, bs)`

The `o_proj` input at `pos`, `(B, d_model)`. Stored as the *context* rather than as per-head
outputs so that a single capture serves all 28 head conditions.

### `capture_attn(ids, pos_b, layer, head)`

The `(B, T)` attention row of `L14.H14` at query position `pos_b` — i.e. how the head's weight at
`n2` is distributed over the whole prompt. It is the *read* side of the head: the component along
**u** is what the head writes, this is what it was looking at while writing it.

Implemented with `model(ids, output_attentions=True)`. With `attn_implementation='sdpa'` (the
default for this load) transformers ≥ 4.48 detects that request and falls back to eager attention
for the pass, printing one warning; the numerics are the same. It is deliberately **its own
forward pass** rather than being folded into `capture_cloud`, so that every other saved activation
stays on exactly the kernel path it was originally computed on.

Per-number totals sum the row over each number's whole token span, not just its last token
(`n_tok_per_number` from `regime_positions`):

* `h14_attn_row_n2` `(N_CLOUD, T)` — the full row, so any other span can be re-derived
* `h14_attn_to_n1`, `h14_attn_to_n2` `(N_CLOUD,)` — the per-number totals

### `_premlp_hooks` / `capture_premlp` / `premlp_patch_handles`

A second intervention site inside layer 14: the residual stream **before the layer's MLP**, i.e.
`layer_input + self_attn(...)` at `pos`. This is the point at which H14 has just written and the
layer's own MLP has not yet read it; every other residual condition patches at the *end* of a
layer, which is after that MLP has already run on the unpatched value.

`Qwen2DecoderLayer.forward` keeps its skip connection in a local variable, so the pre-MLP residual
cannot be rewritten by hooking the layer or the layernorm. The only term the layer adds before the
MLP is the attention sublayer's output, so `_premlp_hooks` pairs

* a `with_kwargs` forward **pre**-hook on the decoder layer, storing its input at `pos`, with
* a forward hook on `self_attn`, which reconstructs `cur = layer_input + attn_out`, applies
  `on_resid(cur)`, and writes the difference back into `attn_out`.

`capture_premlp` passes `on_resid=None` and only observes; `premlp_patch_handles(target, pos, P)`
patches, full-rank when `P is None` and rank-`k` inside `P` otherwise. `capture_cloud` registers
the observe-only pair alongside its own hooks (different slots, so nothing collides) and returns
the pre-MLP residual as a fourth output, free of any extra forward pass and row-aligned with the
end-of-layer cloud by construction.

**A consistency check worth knowing about.** `L14 full preMLP @n2` and `L14 full @n2` come out
identical (IIA arrays exactly equal; position recovery agrees to fp16 rounding). That is expected,
not a bug: replacing the whole pre-MLP residual at one position makes the MLP at that position
compute on the clean input, so the end-of-layer residual there is clean too. The two conditions
are the same intervention reached through different hooks, so their agreement is what says the
pre-MLP hook is wired correctly.

---

## 3. Interventions as declarative specs

A condition is a small dict:

```python
{'resid':  [(layer, 'full'|'sub', P), ...],  # residual-stream interchanges at pos (end of layer)
 'head':   (head_index, P or None),         # rewrite one head's own output at pos
 'premlp': P or None}                       # interchange layer H_LAYER's pre-MLP residual at pos
```

`'premlp'` is detected by key presence, not by value, because `None` is a meaningful value there
(full-rank).

`build_handles` registers whatever the spec asks for, slicing the clean-run activations
(`sources`) to match the current minibatch. `eval_spec` runs the corrupted batch under those hooks
in minibatches and returns per-example position recovery and IIA.

### Metrics (unchanged from `data_numberreps.py`)

* **position recovery** = `(PLD_patched − PLD_corr) / (PLD_clean − PLD_corr)`, where
  `PLD = logit[tok_nc] − logit[tok_b]`. This is the normalized-restoration convention of Meng et
  al. 2022 (ROME) / the IOI literature: 0 = the patch did nothing, 1 = it fully restored the clean
  behaviour.
* **IIA** = `1[argmax over the full vocabulary == tok_nc]`, the interchange-intervention accuracy of
  Geiger et al. 2021.

---

## 4. `train_h14_das` — DAS inside H14's output space

Rank-1 **distributed alignment search** (Geiger et al. 2023, single-causal-variable case), with the
same QR-retraction parametrization of the Stiefel manifold and the same counterfactual-pair
cross-entropy objective as `data_numberreps.train_das_1d`. The *only* difference is the
intervention site: instead of a subspace of the L13 residual stream, the subspace is applied to
**H14's own output contribution** at `POS_B`.

Two things worth knowing:

* The basis is rebuilt (`das.basis()`) inside each minibatch, because each `backward()` frees the
  QR graph.
* Only the `head_dim`-dimensional column space of `W_O^h` is reachable — anything orthogonal to it
  leaves H14's output unchanged — so this is effectively a 128-d search embedded in 3584 dims, not
  a 3584-d one. `h14_das_reachable = ‖Q_col Q_colᵀ w‖` reports the fraction of the learned unit
  direction that lands inside that column space; chance for a random unit vector is
  `sqrt(head_dim / d_model) ≈ 0.19`. A value near 1 says the direction is genuinely a statement
  about H14's output space.

Trained on `N_H14_DAS = 128` triples held out from the evaluation set, `H14_DAS_STEPS = 100` Adam
steps at `lr = 0.05`, gradients accumulated over minibatches of 16.

---

## 5. `pca_and_probes` — one recipe, two read-out points

Two **separate** ridge probes on the L14 residual cloud at `POS_B`, one predicting `log A` and one
predicting `log B` (linear probing in the sense of Alain & Bengio 2016; ridge because the target is
continuous, matching `e11/compare_geometry.ipynb`). `RidgeCV` selects `alpha` by leave-one-out CV
on the training split, so the regularization is not an arbitrary constant. The probe *direction* is
the unit-normalized weight vector; **held-out R²** is what says whether the quantity is linearly
present at all.

Because the two probes are trained independently, `cos(probe log A, probe log B)` and the two
held-out R² values together are the quantitative form of "both numbers are readable from this one
site".

---

## 6. What each requested figure gets

All array names below are keys of `results/sharedrep_{tag}.npz`. Scalars, log-fit blocks and probe
statistics go to `results/sharedrep_meta.json` under `regimes[tag]`.

### Figure 1 — H14 DAS component vs the first number, with an inset

| what | array |
|---|---|
| x-axis | `a_vals` (and `b_vals` for the control) |
| y-axis | `h14_das_comp` (mean-centred) / `h14_das_comp_raw` (un-centred) |
| log fit | `logfit_h14_das_a`, `logfit_h14_das_b` in the meta JSON |
| inset x-axis | `l13_das_comp_n1` — the L13 DAS component read at the **first**-number position, same prompts |
| inset summary | `h14_das_vs_l13_das_r` (meta) |

Each `logfit_*` block is `log_fit_block`'s output: `p/q/r2` over all points and over
`values >= fit_lo`, plus `pearson_r`. Same convention as `data_numberreps.py`, so the fits are
directly comparable with the L13 ones.

The same set exists for H14's unsupervised direction: `h14_pc1_comp`, `h14_pc1_comp_raw`,
`logfit_h14_pc1_a/b`, `h14_pc1_vs_l13_das_r`, and `h14_das_pc1_cos` relating the two.

### Figure 2 — L14 3D PCA with direction arrows

| what | array |
|---|---|
| scatter | `pca14_coords` `(N, 3)` |
| colour by first / second number | `a_vals` / `b_vals` |
| variance explained | `pca14_evr`, `pca14_evals` |
| basis | `pca14_components` `(3, d_model)`, `pca14_mean` |
| **H14 DAS arrow** | `h14_das_in_pca14` `(3,)` — direction cosines with PC1/2/3 |
| **L13 PC1 @n2 arrow** | `l13_pc1b_in_pca14` `(3,)` |
| also available | `h14_pc1_in_pca14`, `probe_logA_in_pca14`, `probe_logB_in_pca14` |

Each arrow also has a `*_pca14_captured` scalar array = `‖proj‖`, the fraction of that unit
direction the 3D basis actually holds — worth quoting in the caption, since an arrow drawn in 3D
is a projection of a 3584-d vector.

The full clouds (`h14_cloud`, `resid14`, both `(N_CLOUD, d_model)` float32) are saved too, so the
PCA, the probes or a different view can be redone in the notebook without the GPU. They are the
bulk of the file size.

### Figure 3 — per-head patching sweep over layer 14

Three arms, each `(n_heads, N_EVAL)` per-example arrays:

| arm | arrays | what it is |
|---|---|---|
| `alone` | `head_posrec_alone`, `head_iia_alone` | head `h`'s own output fully interchanged at n2 |
| `with_L13_PC1_n2` | `head_posrec_with_L13_PC1_n2`, `head_iia_with_L13_PC1_n2` | the same, plus a rank-1 L13 interchange along `L13 PC1 @n2` at the same position |
| `with_L13_full_n2` | `head_posrec_with_L13_full_n2`, `head_iia_with_L13_full_n2` | the same, plus a **full-rank** L13 interchange at that position |

Both joint arms are computed because "with the L13 patch at n2" is ambiguous between the rank-1 and
the full-rank control; pick whichever the figure should show. The joint arms exist because H14
writes *on top of* whatever L13 already holds at n2 — the alone arm is the baseline the joint
number is read against.

Per-example arrays are kept (not just means) so the notebook can compute its own error bars; the
standard error is `x.std(axis=1) / sqrt(N_EVAL)`.

### Figure 4 — direction cosine matrix

`cos_matrix` `(5, 5)` over `cos_dirs` `(5, d_model)`, in the order given by `meta['cos_names']`:

```
['H14 DAS', 'L13 PC1 @n2', 'probe log A', 'probe log B', 'H14 PC1']
```

The four the figure asks for are rows/columns `[0, 1, 2, 3]`; `H14 PC1` rides along as a fifth so
it is there if wanted. All five are vectors in the same `d_model` residual-stream basis and compare
directly — H14's output is *added* into that stream, so its direction needs no change of basis to
be "seen in" L14's residual space. Chance for two random unit vectors is
`1/sqrt(d_model) ≈ 0.017`.

The individual direction vectors are also saved separately: `h14_das_dir`, `h14_pc1_dir`,
`l13_pc1b_dir`, `l13_das_dir`, `probe_logA_dir`, `probe_logB_dir`.

### Figure 5 — headline IIA conditions

Ten conditions, each with `cond_posrec_<key>` and `cond_iia_<key>` `(N_EVAL,)`:

| condition | key | rank | intervention at `POS_B` |
|---|---|---|---|
| L13 full @n2 | `L13_full_atn2` | full | whole L13 residual replaced |
| L13 PC1 @n2 | `L13_PC1_atn2` | 1 | rank-1 along the n2 cloud's own PC1 |
| H14 DAS | `H14_DAS` | 1 | rank-1 inside H14's output contribution |
| H14 DAS + L13 PC1 @n2 | `H14_DAS_and_L13_PC1_atn2` | 1+1 | both of the above jointly |
| L14 u @n2 | `L14_u_atn2` | 1 | end-of-L14 residual, along **u** only |
| L14 v2 @n2 | `L14_v2_atn2` | 1 | end-of-L14 residual, along **v₂** only |
| L14 uv plane @n2 | `L14_uv_plane_atn2` | 2 | end-of-L14 residual, inside span(**u**, **v₂**) |
| L14 full @n2 | `L14_full_atn2` | full | whole residual leaving layer 14 — the ceiling |
| L14 uv plane preMLP @n2 | `L14_uv_plane_preMLP_atn2` | 2 | same plane, pre-MLP site |
| L14 full preMLP @n2 | `L14_full_preMLP_atn2` | full | whole pre-MLP residual — the pre-MLP ceiling |

#### The rank-2 plane

`P_uv` is `torch.linalg.qr(stack([u, v2]).T)[0]`, `(d_model, 2)`, saved as `uv_plane_basis`. Since
`cos(u, v2) = −0.08` the QR barely changes anything — its first column is exactly **u** and its
second is **v₂** with that 8% of **u** removed — so the plane is the one the figures draw, not a
rotated cousin of it.

The four end-of-L14 rows are a rank ladder at one fixed site: 1 (**u**), 1 (**v₂**), 2 (the plane),
`d_model` (everything). Read the rank-2 bar as a fraction of the rank-`d_model` bar *at its own
site*; the pre-MLP pair is the same comparison one sublayer earlier. This is the standard
subspace-sufficiency framing of distributed interchange interventions (Geiger et al. 2023): a
hypothesis about a variable's representation is tested by asking how much of the full-rank
interchange a rank-`k` interchange inside the hypothesized subspace reproduces.

`meta['condition_names']` holds the display names in order. `pld_clean` / `pld_corr` are the
un-patched anchors the position-recovery normalization uses, and `eval_triples` `(N_EVAL, 3)` the
`(a, b, c)` triples, so any of this can be re-sliced by example.

---

## 6b. `sweep_uv_plane` — the plane as a dose-response surface

`sweep_grid_values(low, high)` takes one value per leading digit (15, 25, … 95 in the 2-digit
regime) because the model names a number by emitting its **first** token — the same constraint
`build_triples` enforces with `leading_digits_distinct`. `first_tok_of(value, slot)` derives the
answer token exactly as `make_batch` derives `tok_nc` / `tok_b`, so the sweep and the interchange
conditions score the same event.

For each grid pair the two pre-MLP coordinates are **set** to `coord_of[y]`, the cloud-mean raw
coordinate of that number over a `±SWEEP_WIN` window (marginalizing the partner), and the
corrupted eval prompts are run under that patch. Setting rather than copying is what makes it a
dose-response surface: the rank-2 condition in §8 visits one counterfactual point per example,
this visits the whole grid.

**What is scored.** Not the implanted numbers. The model only names numbers that are in its
prompt (99.5% of outputs here are one of its own two), so $t(y_1^*)$ never wins and scoring
against it measures nothing. The implanted coordinates instead decide **which slot** the
comparison picks — which is the mechanism's own claim — so the score is the prompt's answer logit
difference `logit[tok_nc] - logit[tok_b]`, the same quantity position recovery is built from.
Positive = the first slot won; the prediction is a sign flip across `y1* = y2*`.

Saved (`N_SWEEP = 64` prompts, 9×9 grid): `sweep_values`, `sweep_coords`, `sweep_tokens`,
`sweep_ld_slot` `(G, G, n)`, `sweep_argmax` `(G, G, n)`, and the prompts' own answer tokens
`sweep_tok_prompt` / `sweep_tok_prompt_nc`. The diagonal is skipped and left `nan` — it is the
boundary itself, with no predicted sign. The key is `sweep_ld_slot`, not `sweep_ld`: an earlier
version saved the difference between the implanted numbers' tokens under that name, and the
rename stops an old results file being read as if it held the new quantity.

Cost is ~81 × 64 forward passes, well under the per-head sweep's 28 × 3 × 400.

**Why pre-MLP.** Both because the rank-2 patch only works there (§8), and because the
number → coordinate map the grid depends on is only a good description there: a per-number mean
explains $R^2 = 0.87$ of the $\mathbf{u}$ coordinate pre-MLP against $0.35$ at the end of the
layer, where the MLP has replaced the magnitude with the comparison outcome.

---

## 6c. Section 9b — the same conditions in the other case

`ALT_CASE = 'n2 larger'` repeats the headline conditions (§8) and the plane sweep (§6b) on
prompts where the **second** number is the clean maximum, **reusing `Q_h14`, `P_l13b` and `P_uv`
exactly as found** — nothing is refitted. That makes it a generalization test ("do the directions
found in one case still do anything in the other?") rather than a second fit. Set `ALT_CASE = None`
to skip.

Everything lands under an `alt_` prefix — `alt_cond_iia_<key>`, `alt_cond_posrec_<key>`,
`alt_sweep_ld_slot`, `alt_sweep_argmax`, `alt_sweep_tok_prompt{,_nc}`, `alt_pld_{clean,corr}`, and
`meta['alt_case']`. No existing key changes and `condition_names` is untouched, so every figure
that does not ask for `alt_*` is unaffected.

### Read it with the counterfactual design in mind

`make_batch` varies **only the number at the winning slot**:

| case | clean | corrupted | what differs |
|---|---|---|---|
| `n1 larger` | $(a, b)$ | $(c, b)$ | the **first** number |
| `n2 larger` | $(b, a)$ | $(b, c)$ | the **second** number |

So in `n2 larger` the first number is `b` in both runs. H14 reads the first number — 0.71 of its
attention at the $n_2$ query sits on that token — so its output at $n_2$ is nearly identical in the
two runs and the $\mathbf{v}_1$ patch writes back a value it already had. It is not *exactly* a
no-op (H14 also attends a little to $n_2$ itself, and its query depends on $n_2$'s own residual),
but it is close to one by construction.

**Measured (2digit).** That is exactly what happens, and the pair of cases turns out to be a
**double dissociation** rather than just a control:

| condition | `n1 larger` | `n2 larger` |
|---|---|---|
| $\mathbf{v}_2$ | 0.00 | **0.79** |
| $\mathbf{v}_1$ | **0.46** | 0.00 |
| plane, pre-MLP | 0.81 | 0.95 |
| L14 full | 0.82 | 0.99 |

Each direction is causal exactly when the number it carries is the one the counterfactual changed,
and the plane reaches the ceiling either way. The sweep, which sets *absolute* coordinates and so
does not depend on any clean-vs-corrupted difference, agrees across both cases (sign match 0.85
and 0.94).

A near-zero $\mathbf{v}_1$ bar in the alt case is therefore a **specificity control** —
"$\mathbf{v}_1$ does nothing when the information it carries has not changed", which rules out the
1-dimensional perturbation itself doing the work — and **not** evidence that $\mathbf{v}_1$ is
inert when the second number wins. $\mathbf{v}_2$, by contrast, is the winner's own
representation here, so the alt case does test it in the regime where it decides the answer.

### A read-out trap in the saved arrays

`tok_nc` is the number at the **winning** slot, which is $n_1$ only in `n1 larger`. Anything that
scores "did the model name the first slot" must pick the right array from the case, or the alt
case comes out inverted. `makefig_sharedreps.ipynb` has `n1_slot_tokens()` for this.

### What would test the plane symmetrically

The missing experiment is: with the second number winning, does changing what the plane says about
the **first** number flip the outcome? That needs a batch `make_batch` cannot build, because the
corrupted prompt has to differ at the *losing* slot:

* clean $(b, a)$ with $a > b$ — the second number wins;
* corrupted $(a', a)$ with $a' > a$ — the first number now wins;
* patch H14's output at $n_2$ (or the plane) with the clean value, and a success is the model
  going back to naming the second slot.

That is a new batch constructor plus a sign flip in the metric, not a config change, which is why
it is not in `ALT_CASE`.

---

## 7. Output layout

```
results/sharedrep_2digit.npz     arrays (clouds, coords, directions, per-example metrics)
results/sharedrep_meta.json      scalars: meta block, log-fit blocks, probe stats, correlations
```

`save_regime` splits the result dict by type: `np.ndarray` → the `.npz`, everything else → the
JSON. The script prints a per-array size report at the end. The two clouds dominate; at
`N_CLOUD = 3000` they are ~43 MB each uncompressed.

`meta` records the token layout, the head and layer studied, every sample size and hyperparameter,
`h14_das_reachable`, the path the L13 directions were loaded from, and the wall-clock runtime.

---

## 8. Assumptions worth flagging

1. **Which L13 patch counts as "the L13 patch at n2"** was ambiguous in the request, so the per-head
   sweep computes both the rank-1 (`L13 PC1 @n2`) and the full-rank variant. The headline condition
   list in figure 5 uses the rank-1 one, following `explore_h14_copy.ipynb`.
2. **Only `2digit` runs by default** (`TAGS`). Every requested plot is a single-regime plot; add
   `'3digit'` to `TAGS` for the cross-regime version, at roughly double the runtime.
3. **One DAS seed.** `data_numberreps.py` retrains its L13 DAS from several inits as a stability
   check; that is not repeated here, since the H14 search space is only 128-d and the figure does
   not show a stability panel. Raise it if a reviewer asks for one.
4. **The DAS direction is trained on the `n1 larger` case only.** §9b now re-evaluates it in
   `n2 larger` without refitting, but read that as a specificity control rather than the symmetric
   test — see §6c.
5. **`resid14_pre` costs ~44 MB in the npz** (a third `(N_CLOUD, d_model)` float32 cloud). It is
   kept rather than reduced to coordinates so the pre-MLP panels can be re-cut — a different
   probe, a different number of PCs — without another GPU run.
6. **The rank-2 plane is not fitted.** `span(u, v2)` is assembled from two directions found
   independently (a DAS objective inside H14's output, and a PCA of the L13 cloud at `n2`), and
   then patched. It is not a rank-2 DAS subspace trained against the interchange objective, so its
   score is a *lower* bound on what rank 2 can do at that site — worth saying explicitly if a
   reviewer reads the bar as "the best rank-2 subspace".
7. **The attention row is captured under eager attention**, the rest of the file under sdpa. The
   two agree to fp16 rounding, but the attention numbers are not bit-identical to what produced
   the activations they are plotted against.
