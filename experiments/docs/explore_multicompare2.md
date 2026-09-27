# `explore_multicompare2.ipynb` + `data_multicompare2.py`

The same three-number comparator (`max(n1, n2, n3)` on `Qwen/Qwen2.5-7B-Instruct`, one-shot
prompt, 2-digit operands) read at the **second** number position $n_2$ instead of the third.

Companion to `explore_multicompare.ipynb`, which does everything at $n_3$. Read
`docs/explore_multicompare.md` first — the task, prompt, token layout, counterfactual
construction, metrics and intervention machinery are all shared and are not repeated here. This
document covers only what changes when the patch site moves to $n_2$.

Same structure as the sibling notebook: every expensive block writes one `.npz` under
`results/multicompare2_<block>_2digit.npz` and draws its figure immediately, so a re-run skips
everything already on disk and becomes a pure plotting pass.

---

## 1. Why $n_2$ is a different object

Attention is causal. The residual stream at $n_2$ has seen the first two operands and nothing
else, so:

1. **The space is at most two-dimensional.** $y_3$ cannot be represented there. The notebook
   asserts this rather than assuming it: `M2.y3_is_unreachable` runs two prompts differing only in
   their third number and requires the residual at $n_2$ (after L13 and after L14) to be bitwise
   identical.
2. **There is no `y3` perturbation case.** Its counterfactual edits a token strictly to the right
   of the patch site, so the clean and corrupted runs are identical at $n_2$ and the interchange
   is a no-op by construction. This is why only two cases are carried.
3. **Receptive fields are surfaces, not cubes.** `MC.receptive_fields` sweeps a $(y_1, y_2, y_3)$
   cube because a neuron read at $n_3$ sees three numbers; at $n_2$ the third axis is pinned
   (`RF_Y3`) and the field is a $G \times G$ surface. This also makes the sweep $G$ times cheaper.

## 2. The two cases

`data_multicompare2.CASES2` is exactly `data_multicompare.CASES` minus `y3`:

| case | value order | maximum at | runner-up at | what patching at $n_2$ asks |
|---|---|---|---|---|
| `y1` | $s_1 > s_3 > s_2$ | $n_1$ | $n_3$ | is there a usable **copy** of $y_1$ at $n_2$? |
| `y2` | $s_2 > s_3 > s_1$ | $n_2$ | $n_3$ | $n_2$ is the source position of the perturbed number |

Keeping the sibling notebook's orders is deliberate. The same quadruples, the same value orders
and the same clean/corrupted construction are used; only the patch position moves. That makes
(n2 result, n3 result) a **within-design** comparison rather than two separate experiments, and it
lets the quadruple pool be loaded straight from `results/multicompare_pool_2digit.npz` — same
seed, same `N_POOL`, same order filter, so the file is identical by construction.

The two cases are not symmetric, and that asymmetry is the point. In `y1` the numbers at $n_2$ are
the same in the clean and corrupted runs (only slot 1 was edited), so a full-rank patch at $n_2$
transfers *only* whatever information about $y_1$ earlier heads wrote there. In `y2` the patched
token is the edited one, so the patch injects the winning number at its own position — the direct
analogue of $\mathbf{v}_3$ at $n_3$ and of the two-number "n2 larger" case.

### Expected shape of the result

The sibling notebook already contains one measurement at $n_2$: its section-1 head sweep runs the
`s1>s2` group ($s_1 > s_2 > s_3$, runner-up at $n_2$) there, and both the full-L13 and full-L14
interchange come out at IIA $0.000$ with position recovery around $-0.1$. So a $y_1$ perturbation
is probably **not** recoverable at $n_2$ at all: layer-14's copy heads move $n_1 \to n_3$ and
$n_2 \to n_3$, not $n_1 \to n_2$. The `y1` column of this notebook is therefore expected to be a
null result, and is run because a null measured on the same design as the positive one is worth
having. The `y2` column is expected to be strong — `explore_multicompare`'s `ucos` block already
gets IIA $0.980$ from a **rank-1** DAS direction on the L13 residual at $n_2$.

## 3. The two directions at $n_2$

| symbol | definition | analogue at $n_3$ |
|---|---|---|
| $\mathbf{w}_1$ | rank-1 DAS on the L13 residual at $n_2$, fitted on case `y1` | $\mathbf{v}_1$ |
| $\mathbf{w}_2$ | rank-1 DAS on the L13 residual at $n_2$, fitted on case `y2` | $\mathbf{v}_3$ |

Both search the **whole** L13 residual rather than one head's output, which is a strictly larger
space than the $n_3$ definition of $\mathbf{v}_1$. For $\mathbf{w}_1$ that is the strongest
available causal test of "$y_1$ is usable at $n_2$": if a rank-1 direction anywhere in the
residual carries it, DAS finds it.

$\mathbf{w}_2$ was originally PC1 of the same cloud, mirroring $\mathbf{v}_3$. That was dropped:
the DAS fit is what actually moves the answer, and the two are not interchangeable here. The fits
tell the story on their own —

| | DAS loss | \|corr\| with its own $\log y$ |
|---|---|---|
| $\mathbf{w}_1$ | 14.212 → 4.700 | 0.826 |
| $\mathbf{w}_2$ | 14.011 → 0.001 | 0.927 |

$\mathbf{w}_2$ is the same fit, same seed, same training batch as $\mathbf{u}_2$ in
`explore_multicompare`'s `ucos` block; section 10 reports their cosine as a cross-notebook
reproducibility check.

### The patch basis, and what the plane figure's axes mean

`PB` is Gram-Schmidt on $(\mathbf{w}_1, \mathbf{w}_2)$, giving an orthonormal basis **for the
plane**. This is not cosmetic: `make_subspace_patch_hook` computes $h + (c Q - h Q) Q^\top$,
which is the orthogonal projector $QQ^\top$ only when $Q^\top Q = I$. Fed the raw pair it would
not be a projector at all.

The two directions are **not** orthogonal — $\cos(\mathbf{w}_1, \mathbf{w}_2) = +0.354$, i.e.
69°, against a chance level of 0.017. Both live inside the plane; Gram-Schmidt is an isometry of
the subspace, so the cloud's geometry in the plane is exact, but axis 1 *is* $\mathbf{w}_1$ while
axis 2 is what remains of $\mathbf{w}_2$ after removing $\mathbf{w}_1$ (93.5% of it). The span
itself is basis independent, so every rank-2 interchange result is unaffected by the choice; only
coordinates are. Consequences:

* the plane scatter's y axis should be read as $\mathbf{w}_2^{\perp \mathbf{w}_1}$;
* the contour grid's $y_1^\ast, y_2^\ast$ are cloud-mean coordinates on the GS axes, not
  $\mathbf{w}_1 / \mathbf{w}_2$ components;
* the per-direction readouts in the projection cell use the raw (oblique) dot products instead,
  which is the honest answer to "what does each direction read".

## 4. Module: `data_multicompare2.py`

Everything else is imported from `data_multicompare` / `data_sharedrep` / `data_comparator` /
`data_numberreps`, so the $n_2$ numbers sit on the same measurement code as the $n_3$ ones.

| function | what it does |
|---|---|
| `CASES2`, `CASE_LIST2` | the two cases above, taken from `MC.CASES` |
| `mean_coords_k(...)` | `MC.mean_coords` with the axis count as a parameter |
| `sweep_plane(bat, P, coord_of, grid, pos)` | 2-D dose-response surface: sets both plane coordinates at `pos` to the cloud-mean coordinate of a chosen $(y_1^\ast, y_2^\ast)$ and records both the logit difference and the argmax token. Cells with $y_1^\ast = y_2^\ast$ are skipped. The 2-D counterpart of `MC.sweep_space` and the three-number counterpart of `data_sharedrep.sweep_uv_plane`. |
| `receptive_fields_2d(grid, y3_fixed, T, pos, layers, idx)` | post-SwiGLU activations over the $G^2$ $(y_1, y_2)$ plane with $y_3$ pinned, one deterministic forward pass per cell |
| `y3_is_unreachable(T, pos, low, high)` | the causal-mask sanity check of §1; must return exactly 0 |
| `make_neuron_zero_hook(idx)` | zero-ablation: sets those `down_proj` input columns to 0 at **every** position, removing the neurons' $a_j W^{\rm down}_{:,j}$ write from the residual entirely |
| `sample_max_prompts(k, n, seed, low, high)` | `n` lists of `k` distinct operands, for the `max` task at list length `k`. No leading-digit constraint — the answer is scored by decoding, not by reading one token, which is what makes $k > 9$ possible. |
| `task_accuracy(nums_list, n_digits, make_handles)` | greedy-decodes the answer with the ablation attached for the whole decode (a manual argmax loop, not `generate`, so the hooks stay on and nothing depends on a generation config) and exact-matches against the true maximum |

## 5. Block by block

Blocks run in order, each wrapped in `cached(name, fn)` and skipped if its `.npz` exists;
`FORCE = {'block'}` recomputes one.

**Neurons are not re-derived at $n_2$.** Attribution patching was dropped: the 12 neurons the
two-number comparator identified (`overlap_k20`) are carried over as `SEL`, and the question the
notebook asks is how *they* behave here. The per-head L14 sweep was dropped too — it was the
single most expensive block and its answer was already null.

| # | block | what it does |
|---|---|---|
| — | `pool` | loaded from `explore_multicompare`'s cache: 400 eval + 128 train quadruples |
| 1 | `posmap` | full-rank L13 and L14 interchange at $n_1$, $n_2$, $n_3$ — which positions carry the perturbed number, before any direction is fitted |
| 2 | `dirs` | the cloud at $n_2$ over the same 1500 triples as the sibling notebook, the two DAS fits, and ridge probes $\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3$ on the pre-MLP L14 cloud ($\mathbf{p}_3$ is a negative control: $y_3$ cannot be there, so its $R^2$ is the floor) |
| 3 | `space` | 9 arms per case: each direction alone, the two as separate rank-1 patches, the rank-2 span at L13 / pre-MLP L14 / post-MLP L14, and three full-rank ceilings |
| 4 | — | cosines over $\mathbf{w}_1, \mathbf{w}_2, \mathbf{p}_1, \mathbf{p}_2$ against $1/\sqrt{d_\text{model}}$; no compute |
| 5 | `sweep` | $9 \times 9$ contour grid, 24 prompts per cell |
| 6 | `freeze` | MLP and attention at 14, 15, 16 pinned to their corrupted output under the rank-2 pre-MLP span patch |
| 7 | `rf` | `receptive_fields_2d` over `arange(11, 100, 4)` for the 12, $y_3$ pinned at 55 |
| 8 | `transfer` | the 12 frozen under the span patch, against two random sets of 12 |
| 9 | `das15` | rank-1 DAS on the L15 residual at $n_2$ (`DAS15_RANKS`; add 2 for the 2-D scatter, ~13 min), cross-evaluated across cases, full-rank L15 ceiling, and $\rho_j = \lVert Q^\top w_j\rVert / \lVert w_j\rVert$ over **every** MLP15 neuron so the 12 sit in their own empirical null |
| 10 | — | cosines across positions: $\mathbf{u}_1, \mathbf{u}_2$ and the $n_3$ frame from the sibling notebook's caches; no compute |
| 11 | `ablate` | zero-ablation on the task itself — see below |

### Reading `freeze` and `transfer`: use position recovery, not IIA

IIA requires `tok_nc` to beat every token in the vocabulary; position recovery measures only the
`tok_nc` − `tok_b` gap. At $n_2$ they diverge sharply, and IIA only fires once the restoration
*overshoots*:

| arm (case `y2`) | IIA | posrec |
|---|---|---|
| $\mathbf{w}_2$ only | 0.980 | 3.083 |
| $\mathbf{w}_1 + \mathbf{w}_2$ separate | 0.955 | 3.016 |
| span @ L13 | 0.827 | 2.439 |
| span @ pre-MLP (the freeze base) | 0.078 | 0.944 |
| L13 full | 0.002 | 0.132 |

The pre-MLP span base restores the logit difference essentially fully (0.944) even though its IIA
is 0.078, and it is the *better* freeze baseline precisely because it sits at the clean level
rather than overshooting. Read those two sections in the right-hand panel.

Two further traps in that figure:

* **`attn14` freezing to 0.000 is an artefact, not a finding.** `S.premlp_patch_handles`
  implements the pre-MLP patch *through the attention sublayer's output* — the decoder layer keeps
  its skip connection in a local variable, so that is the only way to modify the pre-MLP
  residual. Freezing attn14's output pins exactly what the patch writes. It is a check that the
  patch lands where intended.
* **`L13 full` scoring far below a rank-1 subspace of it** is real (`posmap` and `space` measure
  it independently and agree exactly). In case `y2` the patched token *is* the edited one, so a
  full-rank patch transplants that token's whole representation, identity included, and derails
  the model; the rank-1 patch moves only the magnitude coordinate.

### 11. `ablate` — zero-ablation on the task

The counterfactual blocks measure one token. This asks the blunter question: remove the neurons
from the model and see whether it can still do `max` at all.

* The 12 neurons' post-SwiGLU activations are set to 0 at **every** position, not just $n_2$.
* Accuracy is exact match on the greedily decoded answer, `ABL_N = 200` prompts per condition.
* Swept over list lengths $k \in \{2, 3, 4, 5, 10, 20\}$; the first figure is the $k = 3$ slice,
  the second is the curve and the accuracy lost relative to the intact model.
* Two random sets of 12 are the control, drawn from the same MLP14+MLP15 pool.
* Operands stay 2-digit (the notebook's regime) and $k$ is the **number of operands**. If the
  intended sweep was over digit count instead, it is a one-line change to `sample_max_prompts`'s
  `low` / `high`.

Zero-ablation (rather than mean-ablation) of MLP neurons is the standard knockout here; 0 is the
activation's value when the SwiGLU gate is closed, so it is in-distribution for the unit.

## 6. What is deliberately not here

* **Attribution patching at $n_2$** and the top-$k$ freezing curve — replaced by the carried-over
  two-number set.
* **The per-head L14 sweep.** `explore_multicompare`'s own section 1 already runs it at $n_2$ for
  the `s1>s2` group and gets IIA 0.000 with position recovery about $-0.1$.
* **`connect`** — the MLP14 → MLP15 causal-edge and weight-path block.
* **A 3-D scatter.** There is no third variable at $n_2$.
* **Cross-position direction transfer** ($\mathbf{v}_1$ fitted at $n_3$, patched at $n_2$).
* **A second DAS seed.**

## 7. Cost

Measured from this notebook's own result-file timestamps, not estimated.

| block | measured |
|---|---|
| `posmap` | ~1 min |
| `dirs` | ~9 min (1500-prompt cloud + 2 DAS fits; was 3 fits before the head direction was dropped) |
| `space` + `sweep` + `freeze` + `rf` + `transfer` | ~2 min combined |
| `das15` | ~6 min |
| `ablate` | ~3 min (24 decode runs of 200 prompts × 3 tokens) |
| **total, cold** | **~21 min** |

Removed since the first version: the per-head sweep (~10 min) and attribution (~4 min).

The residual captures `S.clean_hidden_at` returns are whole `(B, T, d_model)` sequences and are
therefore position independent. Section 1 patches at three positions, so `get_src` caches them
per `(order, layer)` and shares them, keeping only the o_proj context and pre-MLP residual per
position; the $n_1$ and $n_3$ contexts are dropped once `posmap` is done. Without that the six
(order, position) combinations would hold ~1.6 GB of duplicated activations.

## 8. Reproducing

```bash
# from experiments/, on a GPU node, with explore_multicompare already run
jupyter nbconvert --to notebook --execute --inplace explore_multicompare2.ipynb
```

Required from the sibling notebook: `results/multicompare_pool_2digit.npz` (recomputed if
absent). Optional: its `ucos` / `dirs` caches, which section 10 drops what it cannot find.
`results/comparator_data_2digit.npz` supplies `overlap_k20` and is **not** optional — it is the
neuron set for sections 7, 8, 9 and 11.

## 9. References

* Interchange interventions, IIA, DAS — Geiger et al. (2021), *Causal abstractions of neural
  networks*; Geiger et al. (2023), *Finding alignments between interpretable causal variables and
  distributed neural representations*. The rank-1 single-causal-variable case is used throughout,
  and the contour sweep is the same intervention run as a surface rather than at a point.
* Position-recovery normalisation — Meng et al. (2022), *Locating and editing factual associations
  in GPT* (the ROME/IOI normalised-restoration convention).
* Localising by position before localising by component — Wang et al. (2022), *Interpretability in
  the wild* (IOI). The `posmap` block is the standard pass that precedes any direction-level claim,
  and the zero- vs mean-ablation contrast is discussed there and in Conmy et al. (2023), *Towards
  automated circuit discovery*.
* MLP neurons as key–value memories, and neuron-level knockout — Geva et al. (2021), *Transformer
  feed-forward layers are key-value memories*; Geva et al. (2022), *Transformer feed-forward
  layers build predictions by promoting concepts in the vocabulary space*.
* Per-head output contributions as separable terms — Elhage et al. (2021), *A mathematical
  framework for transformer circuits*.
* Linear probes — Alain & Bengio (2016). $\mathbf{p}_3$ as a probe on a variable that provably
  cannot be present is the standard control for probe-capacity artefacts (Hewitt & Liang 2019,
  *Designing and interpreting probes with control tasks*).
* Task, prompt and value orders — `e15_multi_compare`.
