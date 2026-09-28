# `patching_model_behavior.ipynb` — what the model actually says under each patch

Re-runs the paper's interchange interventions with **greedy generation** instead of a single
next-token argmax, so each patched run produces a whole number. IIA is then scored two ways:

* **first digit**: argmax of the last-position logits equals the leading-digit token of the target
  $y^*$. This is the metric every makefig panel uses (`posrec_and_iia` in `data_numberreps.py`).
* **whole number**: the leading integer parsed from the greedy output equals $y^*$.

$y^*$ is the corrupted prompt's number at the perturbed slot (`corr_pairs[:, win_slot]`, i.e. `c`
in the triple). A patch that flips which slot wins makes the model copy that slot's number, so a
first-digit hit that is not followed by the right second digit is a partial success, and the gap
between the two metrics measures how often that happens.

Inference only: no direction is retrained. Everything is loaded from the saved results.

---

## Data and directions

| object | source | used for |
|---|---|---|
| $\mathbf{u}$ | `numberreps_2digit.npz['das_dir']` | L13 residual @ $y_1$'s last token, rank 1 |
| $\mathbf{v}_1$ | `sharedrep_2digit.npz['h14_das_dir']` | inside H14's output @ $y_2$ (and the pre-MLP variant) |
| $\mathbf{v}_2$ | `sharedrep_2digit.npz['l13_pc1b_dir']` | L13 residual @ $y_2$'s last token, rank 1 |
| plane | `sharedrep_2digit.npz['uv_plane_basis']` | L14 pre-MLP residual @ $y_2$, rank 2 |
| L15 DAS | `comparator_data_2digit.npz['das15_basis_n1_larger_k1']` | L15 residual @ $y_2$, rank 1 |

The evaluation triples are rebuilt exactly as the data scripts do
(`build_triples(N_EVAL + N_DAS, SEED, ...)[:N_EVAL]`, 400 triples) and asserted equal to the
`eval_triples` saved in all three npz files. Batches come from `make_batch`, so the counterfactual
design is unchanged: `y1` perturbed = `n1 larger` (clean `(a, b)`, corrupted `(c, b)`), `y2`
perturbed = `n2 larger` (clean `(b, a)`, corrupted `(b, c)`).

Clean-run sources are captured with the same helpers the data scripts use
(`data_sharedrep.clean_hidden_at` for L13/L14/L15, `capture_ctx` for H14's o_proj input,
`capture_premlp` for the L14 pre-MLP residual), all in `EVAL_BS` minibatches.

## Conditions (`COND`)

Each entry is a `data_sharedrep.build_handles` spec plus a patch position, so every hook is the
one the makefig numbers were produced with.

| name | spec | position |
|---|---|---|
| `L13 full @y1` | full L13 residual | $y_1$ |
| `u` | rank 1 in L13 residual along $\mathbf{u}$ | $y_1$ |
| `L13 full @y2` | full L13 residual | $y_2$ |
| `v2` | rank 1 in L13 residual along $\mathbf{v}_2$ | $y_2$ |
| `v1` | rank 1 inside H14's own output contribution along $\mathbf{v}_1$ | $y_2$ |
| `v1 pre-MLP` | rank 1 in the L14 pre-MLP residual along $\mathbf{v}_1$ | $y_2$ |
| `v1 & v2` | `v1` and `v2` together | $y_2$ |
| `v1,v2 plane` | rank 2 in the L14 pre-MLP residual, span($\mathbf{v}_1$, $\mathbf{v}_2$) | $y_2$ |
| `L14 full` | full residual leaving L14 | $y_2$ |
| `L15 DAS` | rank 1 in the residual leaving L15 | $y_2$ |

**Two versions of $\mathbf{v}_1$.** The makefig $\mathbf{v}_1$ bar (0.46) patches $\mathbf{v}_1$
*inside H14's output* (`head_patch_handles`): only the head's contribution along $\mathbf{v}_1$ is
swapped. `v1 pre-MLP` instead swaps the whole pre-MLP residual's component along $\mathbf{v}_1$,
which also moves whatever other heads and earlier layers put on that direction. It has no saved
counterpart; it is the other reading of "$\mathbf{v}_1$ in the pre-MLP layer-14 residual".

`RUNS` is the set of (condition, case) pairs the three figures need; `SAVED` maps each to its
per-example first-digit IIA in the npz files (`n/a` for `v1 pre-MLP`).

## Functions

* **`as_basis(v)`**: numpy vector or `(d, k)` array to a float32 `(d, k)` tensor on the GPU.
* **`generate(ids, handles_fn, n_new, bs)`**: greedy decoding in `EVAL_BS` minibatches. The hooks
  from `handles_fn(lo, hi)` are registered for the **prompt pass only** and removed before
  decoding; the generated tokens then read the patched prompt positions through the KV cache.
  This is the same as re-running the whole sequence with the patch at its (fixed) prompt position
  on every step, since the clean source at that position does not depend on later tokens. The
  patch is never applied at generated positions: it is an intervention on the representation of
  an input number, not steering. `MAX_NEW = 4` (two digits, the terminator, one spare).
* **`score(gen, target)`**: decodes each generation, parses its leading integer (`-1` if it does
  not start with a digit), and returns the first-digit hit (token id of the first generated token
  vs the leading-digit token of `target`, i.e. `tok_nc` for patched runs) and the whole-number hit.
* **`mean_se(x)`**: mean and standard error over examples, for the error bars.

## Cells

1. **Setup**: loads the model (`D.load_model`), the H14 machinery (`S.init_head_machinery`), the
   token layout.
2. **Directions, triples, batches**: loads the directions, rebuilds and checks the triples, builds
   both cases' batches and clean sources, prints one clean and one corrupted prompt per case.
3. **Conditions**: `COND`, `RUNS`, `SAVED`.
4. **Greedy generation**: the functions above.
5. **Unpatched outputs**: the model's own answers on clean and corrupted prompts, first digit and
   whole number, as the reference for how often it names the max at all.
6. **Patched outputs**: runs every pair in `RUNS` and prints one table: saved first-digit IIA,
   recomputed first-digit IIA, their per-example agreement, whole-number IIA, and an outcome
   breakdown of the patched answers (`y* 1st digit only`: right leading digit, wrong number;
   `=unpatched`: the corrupted run's own answer; `other`).
7. **Example generations**: for each run, the first `N_SHOW` examples (clean / corrupted / patched
   outputs as the raw decoded strings, and $y^*$), followed by the examples where the first digit
   is right and the whole number is wrong.
8. **First digit right, whole number wrong**: per run, the number of first-digit hits, how many
   of them are the wrong whole number, and the rate (wrong / first-digit hits). `miss_kind`
   classifies each such output, first match wins: `hybrid: clean` (first digit of $y^*$ + last
   digit of the clean run's number at the perturbed slot), `hybrid: other` (first digit of $y^*$ +
   last digit of the other, unperturbed number), `12 (1-shot)` (the one-shot example's answer),
   `too long` (more digits than $y^*$, e.g. `111`), `other`. The last column, `hits a hybrid`,
   counts whole-number hits where the clean number and $y^*$ share their last digit. On those
   examples a clean-hybrid output would *also* equal $y^*$, so for the full-residual swaps it
   says how many of the whole-number successes could be coincidences.
9. **IIA: u, v1, v2**: the analogue of `makefig_numberreps` (f), one panel per metric, grouped by
   which number is perturbed.
10. **IIA: the (v1, v2) space**: the analogue of `makefig_sharedreps` (e), same six conditions, both
   cases, one panel per metric.
11. **IIA: comparator**: the L15 DAS bar of `makefig_comparator` (e), both cases, both metrics. As
    in the figure, one direction (fit with $y_1$ perturbed) is evaluated in both cases.

Palette and labels follow the makefig notebooks (`#2166AC` full, `#C97B00` u / $y_1$ perturbed,
`#C06A9B` v2, `#117777` v1; $y_2$ perturbed in blue).

## Agreement with the saved values

The first-digit column reproduces the saved IIA per example (`agree` = 1.000) in every run except
L15 DAS with $y_2$ perturbed: 0.652 against the saved 0.655, where one example flips (agree 0.998).
The prompt pass is the same forward as the original evaluation (same hooks, same minibatch size and
order) but with the KV cache on, so an fp16 near-tie can go the other way.

## Results (2digit, n = 400, run 2026-09-28)

Unpatched, the model names the max as a whole number on 100% of clean and corrupted prompts in
both cases.

| condition | perturbed | first digit (saved) | whole number | right first digit, wrong number |
|---|---|---|---|---|
| L13 full @ $y_1$ | $y_1$ | 0.072 (0.072) | **0.003** | 0.070 |
| $\mathbf{u}$ | $y_1$ | 0.940 (0.940) | 0.940 | 0 |
| $\mathbf{v}_1$ (H14 output) | $y_1$ | 0.460 (0.460) | 0.460 | 0 |
| $\mathbf{v}_1$ pre-MLP | $y_1$ | 0.710 (n/a) | 0.710 | 0 |
| L13 full @ $y_2$ | $y_2$ | 0.980 (0.980) | **0.115** | 0.865 |
| $\mathbf{v}_2$ | $y_2$ | 0.790 (0.790) | 0.785 | 0.005 |
| $\mathbf{v}_1$ & $\mathbf{v}_2$ | $y_1$ / $y_2$ | 0.675 / 0.885 | 0.675 / 0.880 | 0 / 0.005 |
| plane, pre-MLP | $y_1$ / $y_2$ | 0.810 / 0.948 | 0.810 / 0.940 | 0 / 0.007 |
| L14 full @ $y_2$ | $y_1$ / $y_2$ | 0.825 / 0.985 | 0.825 / **0.115** | 0 / 0.870 |
| L15 DAS rank 1 | $y_1$ / $y_2$ | 1.000 / 0.652 (0.655) | 1.000 / 0.647 | 0 / 0.005 |

Controls (L13 full @ $y_2$ and $\mathbf{v}_2$ with $y_1$ perturbed, $\mathbf{v}_1$ with $y_2$
perturbed) are 0 on both metrics.

**The directions are faithful at the whole-number level; the full-residual swaps are not.** For
every rank-1 and rank-2 patch the whole-number IIA equals the first-digit IIA to within 0.008. The
few misses are all $y^* \in \{10, 11\}$ turned into `111.` / `102.` / `105.`, a repeated-digit
copying slip. For the full swaps at the second number's position, the first digit is right but the
number mostly is not (0.98 to 0.115). In every printed example the model outputs a **hybrid**: the
first digit of $y^*$ and the second digit of the clean number. (That these are all hybrids is read
off the printed examples, not counted over all 346.) Example: clean max(55, 70), corrupted max(55, 33),
patched output `30`, not `33`. The patch position is the number's last token (its second digit),
so a full swap also transplants the clean run's *token identity* there. The model copies the first
digit from the unpatched position before it and the second digit from the patched one. The
directions carry magnitude and not token identity, so they never do this.

This is also why L14 full with $y_1$ perturbed shows no gap (0.825 on both): in that case the
second number is the same in the clean and corrupted prompts, so there is no foreign digit to
transplant. L13 full @ $y_1$ gets the whole number right on only 1 of its 29 first-digit hits. The
misses include hybrids (`26` for $y^* = 20$, clean first number 96) and `12`, the number in the
one-shot example.

**$\mathbf{v}_1$ pre-MLP vs inside H14.** Swapping the whole pre-MLP residual's component along
$\mathbf{v}_1$ gives 0.71, against 0.46 for swapping only H14's contribution along it. The extra
0.25 comes from what the rest of layer 14's attention (and the stream below it) writes on
$\mathbf{v}_1$ at $y_2$.

## References

* IIA: Geiger et al. 2021, *Causal Abstractions of Neural Networks*; DAS: Geiger et al. 2023,
  *Finding Alignments Between Interpretable Causal Variables and Distributed Neural
  Representations*.
* Scoring an interchange by the multi-token output the model generates, with the intervention
  fixed at the input position and generation reading it through the cache: Huang et al. 2024,
  *RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations*.
  That is the closest standard to the whole-number metric here.
