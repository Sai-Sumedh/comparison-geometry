# `explore_comparatorfig.ipynb` — which components build the comparator?

Interactive notebook. Loads the model and re-runs interventions in the kernel; the directions it
patches are **loaded**, never retrained.

## 1. The question

By layer 14 at the second-number position (`n2`) the residual stream holds *both* numbers: head
L14.H14 has copied the first number there (direction `u`, the H14-space rank-1 DAS direction) on
top of the second number already present from L13 (direction `v2`, L13 PC1 read at `n2`). Patching
both at once (`H14 u + L13 v2 @ n2`) flips the model's answer on 67.5 % of held-out triples
(IIA 0.675, position recovery 1.469 — `results/sharedrep_2digit.npz`).

Something *downstream* of that point must actually compare the two magnitudes and emit the answer —
the argmax clusters visible in the middle band of layers. This notebook asks which sub-blocks
carry that step, by knocking them out one at a time, in **both** cases (`n1 larger`, `n2 larger`).

## 2. Method: freezing a sub-block at the patched token

A sub-block (a decoder layer's `mlp` or `self_attn`) is **frozen** by pinning its output at `n2`
to the value it produced on the *unpatched corrupted run*:

```
out_L[:, n2, :] := out_L_corrupted[:, n2, :]
```

Everything else runs normally. The block still receives the patched input; it simply is not allowed
to pass any consequence of the patch on. If downstream IIA / position recovery collapses when a
block is frozen, that block is on the causal path from "both numbers present at `n2`" to "the right
answer at the final token".

This is the knock-out complement of **path patching** (Goldowsky-Dill et al. 2023, *Localizing
Model Behavior with Path Patching*; the technique as used in Wang et al. 2023's IOI circuit), and
is the same manoeuvre as the "severed" / frozen-module variant of causal tracing in Meng et al.
2022 (ROME, §3.2 — MLP or attention modules held at their corrupted values while the residual
stream is restored). Freezing to the *corrupted* activations rather than to zero or to a mean is
what makes it a clean interchange: the frozen value is on-distribution, so nothing is being ablated
out of the model's normal operating range.

The metrics are the ones used throughout this project:

- **position recovery** = `(PLD_patched − PLD_corr) / (PLD_clean − PLD_corr)`, `PLD = logit[tok_nc]
  − logit[tok_b]` — the ROME/IOI normalized-restoration convention (Meng et al. 2022).
- **IIA** = `1[argmax over the vocabulary == tok_nc]` — interchange-intervention accuracy
  (Geiger et al. 2021).

Both come from `data_numberreps.posrec_and_iia`, so every number is directly comparable with those
in `results/sharedrep_2digit.npz` — the notebook asserts that its 400 evaluation triples are
byte-identical to the saved ones.

**Freeze position.** The freeze is applied at `n2` only (`FREEZE_POS = POS_B`), i.e. it tests
whether the block's *local computation at the patched token* is the comparator. See the caveat in
§6.

## 2b. The two cases, and why the live arm swaps between them

`make_batch(triples, win_slot, T)` puts the clean maximum `a` at slot `win_slot` and the corrupted
run's `c` at the same slot; the other slot holds `b` in both runs. So the contested number — the
one that differs between clean and corrupted — always sits at `win_slot`:

| case | `win_slot` | contested number is at | how it reaches `n2` | live patch arm |
| --- | --- | --- | --- | --- |
| `n1 larger` | 0 (first) | the first number token | copied there by head L14.H14 | `H14 u` |
| `n2 larger` | 1 (second) | `n2` itself | it is already there | `L13 v2 @ n2` |

In `n2 larger` the number H14 copies into `n2` is `b`, the loser — identical in the clean and
corrupted runs — so the `H14 u` patch writes back approximately what is already there and is a
**null intervention**. The mirror image holds in `n1 larger`, where `L13 v2 @ n2` was near-null
(IIA 0.000): the second number is `b` there.

This is not a defect of the design, it is the point: the same freeze sweep is being run on two
different routes into the comparator. If MLP14 is the comparator rather than part of H14's copy
path, freezing it should break the interchange in both cases, even though a *different* arm
delivers the number in each.

Cell 9 prints the quantitative version of this — the mean `|clean − corrupted|` displacement along
`u` and along `v2` at `n2`, per case. A near-zero displacement is what a null arm looks like before
any patching is run.

## 3. Cell-by-cell

| Cell | What it does |
| --- | --- |
| 1 | Imports `data_numberreps` (as `D`) and `data_sharedrep` (as `S`) — neither loads the model at import — and reads `results/sharedrep_2digit.npz` for `u` (`h14_das_dir`) and `v2` (`l13_pc1b_dir`), both re-normalized. Prints the saved reference conditions. |
| 3 | Plot style: the same palette / font sizes as `makefig_sharedreps.ipynb`. `savefig` writes `figs/comparatorfig_*.png`. |
| 5 | Loads the model, calls `S.init_head_machinery()` (binds `W_O`, `n_heads`, `head_dim` for layer 14), recomputes the token layout and asserts it matches the saved run, and puts `u`, `v2` on the GPU as `(d_model, 1)` orthonormal bases `Q_h14`, `P_l13b`. |
| 7 | Freeze machinery — see §4. |
| 9 | Rebuilds the evaluation triples (same `build_triples` pool, seed and split as `data_sharedrep.py`), then **per case** the counterfactual batch, the clean-run sources and the corrupted-run output of all six freeze sites, into `BAT` / `SRC` / `FRZ`. Also prints the patch-source displacement diagnostic (§2b). |
| 11 | Sanity check, both cases. |
| 13 | Builds the patch arms and freeze sets, then runs the 2 cases × 3 arms × 8 freeze sets grid — **unless `results/comparator_2digit.npz` already exists**, in which case the patching is skipped and cell 15 loads it (`FORCE_GRID = True` re-runs it). The arms are rebuilt either way, because the neuron cells need them as real specs. |
| 15 | Defines `key_of` and the result paths, rebuilds `RES` from the saved `.npz` if it is not already in the kernel, and prints the metric tables — so they appear whether the grid was computed or loaded. See §5b. |
| 17 | Figure 1 — grouped bars, one group per freeze set, one column per case, y shared within a row. |
| 19 | Figure 2 — the L14–L16 sweep for the joint patch, MLP vs attention, one row per case, normalized by that case's unfrozen value. |
| 21 | Saves `results/comparator_2digit.npz` (per-example arrays) and `results/comparator_meta.json` (config + condition means). |

## 4. Functions defined in the notebook

### `block(site)`
`site` is `(kind, layer)` with `kind ∈ {'mlp', 'attn'}`. Returns
`model.model.layers[layer].mlp` or `.self_attn`. A tuple key rather than a string so the same value
indexes the captured activations, the freeze sets and the saved arrays.

### `make_freeze_hook(frozen, pos)`
Returns a forward hook that replaces `out[:, pos, :]` with `frozen` (a `(B, d_model)` float32
tensor, cast to the module's dtype). Handles both output shapes in this model: `mlp` returns a bare
tensor, `self_attn` returns `(hidden_states, attn_weights)`, so the hook rebuilds the tuple with
only element 0 changed. The tensor is cloned before writing — an in-place write on a hook output
can corrupt an activation another hook is still holding.

### `capture_corrupted(sites, ids, pos, bs)`
One minibatched pass over the corrupted prompts `ids_r` with a read-only hook on every site at
once, returning `{site: (n, d_model)}`. Capturing all sites in a single pass guarantees every freeze
is drawn from the same forward pass, so a combined freeze (e.g. MLP14+MLP15) is internally
consistent. Minibatched at `EVAL_BS` for the same reason as everywhere else in this project: the
`(B, T, |vocab|)` logits, not the activations, set the memory ceiling.

### `eval_spec(spec, bat, sources, pos, freeze, frozen, bs)`
The workhorse. For each minibatch it

1. builds the **patch** handles via `S.build_handles(spec, sources, lo, hi, pos)` — `spec` is
   `data_sharedrep.py`'s declarative format, `{'resid': [(layer, 'full'|'sub', P)],
   'head': (head, P)}`;
2. appends one **freeze** hook per site in `freeze`, sliced to the minibatch;
3. runs the corrupted prompts, takes the final-position logits, and reduces to per-example position
   recovery and IIA.

Handles are always removed in a `finally`, so an exception mid-cell cannot leave the model hooked.
Returns `(posrec, iia)`, each `(n,)`.

Hook ordering matters and is relied on: `o_proj` is a submodule of `self_attn`, so the H14 patch
(registered on `o_proj`) fires *before* an `attn14` freeze (registered on `self_attn`), and within
a layer the `mlp` runs after `self_attn`. A freeze therefore always sees the patched input and
still emits the corrupted output — which is the intended semantics — and an `attn14` freeze
overwrites the H14 patch entirely (§6).

### `key_of(case, arm, fz)`
`'n1 larger', 'H14 u + L13 v2 @n2', 'MLP14'` → `'n1_larger__H14_u_and_L13_v2_atn2__MLP14'`. The
same space/`+`/`@` substitution `data_sharedrep.py` uses, so the saved `.npz` keys read the same
way.

## 5. Conditions

**Cases**: `n1 larger`, `n2 larger` — `CASES` in cell 9. Every batch, source, frozen capture and
result is keyed by case; `RES` is `{(case, arm, freeze_set): (posrec, iia)}`.

**Patch arms** (all at `n2`, rank-1 throughout, identical in both cases):

| Arm | Spec |
| --- | --- |
| `H14 u` | `head=(14, Q_h14)` — rank-1 interchange inside head L14.H14's own output contribution |
| `L13 v2 @n2` | `resid=[(13, 'sub', P_l13b)]` — rank-1 interchange in the L13 residual |
| `H14 u + L13 v2 @n2` | both — the reference condition (IIA 0.675) |

**Freeze sets**: `none`, `MLP14`, `MLP15`, `MLP16`, `attn14`, `attn15`, `attn16`, `MLP14+15`.

Reading the grid: a freeze that leaves the joint arm's IIA near its unfrozen value is *not* on the
path; one that drives it toward 0 (the corrupted baseline) is. `MLP14+15` tests whether the two
MLPs are redundant — if each alone does little but the pair kills the effect, the comparison is
distributed across them. Compare *within* a case column, since the unfrozen value differs between
cases.

## 5a. Result (n = 400 per case, 2-digit regime)

Patch-source displacement at `n2` (cell 9), confirming which arm is live in which case:

| case | mean \|clean − corr\| along `u` | along `v2` |
| --- | --- | --- |
| `n1 larger` | 5.181 | 9.588 |
| `n2 larger` | 0.802 | 25.575 |

IIA, joint arm (`H14 u + L13 v2 @ n2`):

| freeze | none | MLP14 | MLP15 | MLP16 | attn14* | attn15 | attn16 | MLP14+15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `n1 larger` | 0.675 | 0.010 | 0.207 | 0.652 | 0.000 | 0.670 | 0.685 | 0.000 |
| `n2 larger` | 0.885 | 0.300 | 0.717 | 0.900 | 0.760 | 0.865 | 0.895 | 0.035 |

position recovery, same arm:

| freeze | none | MLP14 | MLP15 | MLP16 | attn14* | attn15 | attn16 | MLP14+15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `n1 larger` | 1.469 | 0.087 | 0.495 | 1.432 | 0.028 | 1.477 | 1.516 | 0.037 |
| `n2 larger` | 1.784 | 0.574 | 1.368 | 1.788 | 1.556 | 1.738 | 1.815 | 0.164 |

The live single arms behave as §2b predicts: `H14 u` alone gives IIA 0.460 in `n1 larger` and
**0.000** in `n2 larger`; `L13 v2 @ n2` alone gives 0.000 in `n1 larger` and **0.790** in
`n2 larger`. All six no-patch sanity conditions returned posrec exactly `+0.000000`.

What it says:

- **MLP14 at `n2` is the dominant component.** Freezing it alone costs 99 % of the effect in
  `n1 larger` and 66 % in `n2 larger` — in both cases the single largest drop by a wide margin.
- **MLP15 is a secondary contributor** (down to 0.207 and 0.717), MLP16 contributes nothing
  (0.652, 0.900 — at or above the unfrozen value).
- **MLP14 + MLP15 together abolish the effect** in both cases (0.000, 0.035), and the pair is worth
  more than either alone, so the comparison is concentrated in those two MLPs at `n2` but not
  entirely in one of them.
- **No attention block above 14 matters at this position** (attn15, attn16 within noise of
  unfrozen). `attn14` in `n2 larger` is a genuine number (0.885 → 0.760, a modest real effect,
  since the `H14 u` patch it also overwrites is null in that case); `attn14` in `n1 larger` is the
  degenerate cell, see §6.
- The conclusion **does not depend on how the number reaches `n2`**: the same two MLPs carry it
  whether the contested magnitude was copied in by head L14.H14 or was native to the position.

## 5b. Reusing the results without re-running

The grid takes ~2 minutes plus a model load. Two guards make that a one-off:

- **Cell 13** checks the disk. If `results/comparator_2digit.npz` exists it prints a note and skips
  the patching entirely, leaving `RES` for cell 15 to load. `FORCE_GRID = True` re-runs it. It still
  rebuilds `ARMS` / `FREEZE_SETS`, which are cheap and which the neuron cells need as real specs.
- **Cell 15** checks the kernel. If `RES` is already in memory it keeps it (`del RES` forces a
  reload), otherwise it loads from disk. Then it prints the tables, so they show up on both paths.

Two ways to run, then:

| goal | cells | cost |
| --- | --- | --- |
| figures only, no model | 1, 3, 15, 17, 19, then 23 onward | seconds |
| recompute something | Run All, with the relevant `FORCE` / `FORCE_GRID` set | see below |

On a Run All with everything cached, what still runs is the model load (~25 s), the per-case
captures in cell 9 (~40 s) and the sanity check in cell 11 (~15 s); cell 13 and every neuron block
are skipped. Cell 9 is not guarded because it produces the `BAT` / `SRC` / `FRZ` any *compute* path
needs — if nothing needs recomputing, take the no-model path above instead.

`key_of` and the two result paths live in cell 15 rather than in the save cell, so both the load
and the save use the same key convention.

## 6. Caveats

- **`attn14` is degenerate.** The H14 `u` patch is written into layer 14's `o_proj` output, which
  the `attn14` freeze then overwrites at `n2`. Any arm containing `H14 u` therefore reduces to that
  arm minus its H14 component. It is marked with `*` in Figure 1 and should be read as a
  manipulation check (it *should* collapse), not as evidence about attention 14's role.
- **The freeze is at `n2` only.** Layer 14's attention at the *final* token already reads the
  patched L13 residual at `n2`, so MLP14/MLP15 at the final token still see a consequence of the
  patch. A block that looks innocent here may still matter at another position. Changing
  `FREEZE_POS` to a list of positions (or to all positions ≥ `n2`) is a one-line change in
  `make_freeze_hook`'s call site if that route needs closing.
- **The `H14 u` arm is near-null in `n2 larger`** and the `L13 v2 @ n2` arm is near-null in
  `n1 larger` (§2b). Those bars are specificity controls, not failures; read each case off its own
  live arm, or off the joint arm, which contains both.
- **Position recovery above 1** is expected for the joint arm (1.469 unfrozen): the patch pushes the
  logit difference *past* the clean run's value. IIA is the bounded metric and is the one to quote.
- Nothing here separates "the MLP computes the comparison" from "the MLP is a necessary conduit for
  a comparison computed elsewhere". A knock-out shows necessity of the node, not the locus of the
  computation. Establishing the locus needs the positive direction — e.g. reading a
  max/argmax-selective direction out of that MLP's output — which this notebook does not do.

## 7. Outputs

- `figs/comparatorfig_2digit_freeze_grid.png`
- `figs/comparatorfig_2digit_layer_sweep.png`
- `results/comparator_2digit.npz` — `posrec_<key>` / `iia_<key>` per-example arrays (400 each) for
  all 48 conditions (2 cases × 3 arms × 8 freeze sets), plus `eval_triples`.
- `results/comparator_meta.json` — config and the condition means (`summary[key_of(case, arm,
  freeze)] = {posrec, iia}`), which is enough to read every number in §5a without opening the
  `.npz`.

## 8. References

- Geiger, Lu, Icard & Potts (2021), *Causal Abstractions of Neural Networks* — interchange
  interventions, IIA.
- Geiger et al. (2023), *Finding Alignments Between Interpretable Causal Variables and Distributed
  Neural Representations* — DAS; the source of the `u` direction being patched.
- Meng, Bau, Andonian & Belinkov (2022), *Locating and Editing Factual Associations in GPT* —
  normalized restoration (position recovery) and severed / frozen-module causal tracing.
- Goldowsky-Dill, MacLeod, Sato & Arora (2023), *Localizing Model Behavior with Path Patching* — the
  formal treatment of freezing nodes to isolate a causal path.
- Wang, Variengien, Conmy, Shlegeris & Steinhardt (2023), *Interpretability in the Wild* (IOI) —
  path patching applied to a circuit, and the frozen-attention convention.

---

# Part II — neuron level (cells 22–35)

Appended to the same notebook. Everything here is about *which neurons inside MLP14 / MLP15* carry
the comparison, and what they read from the $\mathbf{v}_1,\mathbf{v}_2$ plane. A neuron is the
post-SwiGLU scalar feeding `down_proj` — `e11/which_neurons_compare.ipynb`'s definition, kept so the
two notebooks mean the same thing by "neuron". $d_{ff} = 18944$ per layer, so 37888 pooled
candidates.

## 9. Caching

Every block is wrapped in `cached_npz(path, compute)`: if `path` exists it is loaded and
`compute()` never runs. Delete the file, or set `FORCE = True` in the config cell, to recompute.
Four files:

| file | holds |
| --- | --- |
| `results/comparator_atp_2digit.npz` | per-neuron attribution scores, per case and layer |
| `results/comparator_topk_2digit.npz` | the top-$k$ / random-$k$ / bottom-$k$ freeze curve |
| `results/comparator_inject_2digit.npz` | the clean-MLP injection results, three metrics |
| `results/comparator_tune_2digit.npz` | plane coordinates, top-neuron cloud activations, read-in alignment |

First run ≈ 5 minutes total; afterwards the whole block is a few seconds of loading. The figure
cells read only these files, so they redraw without the model.

Note the one-time dependency: the *compute* path needs `BAT` / `SRC` from cell 9 and `ARMS` from
cell 13, so the first run goes through the model load. It does **not** re-run the 48-condition grid
— cell 13 skips that once its file is on disk (§5b) — but it does pay cell 9's captures.
`cached_npz` asserts the dependency with a readable message rather than failing on a `NameError`.

## 10. Rank — attribution patching (cell 25)

$$s_i \;\approx\; \bigl(a_i^{\mathrm{corr}} - a_i^{\mathrm{patched}}\bigr)\,
\frac{\partial\,\mathrm{PLD}}{\partial a_i}\Bigg|_{\mathrm{patched}},\qquad
\mathrm{PLD}=\mathrm{logit}[\texttt{tok\_nc}]-\mathrm{logit}[\texttt{tok\_b}]$$

One forward + backward scores all 37888 neurons. References: Nanda (2023) *Attribution Patching*;
Syed, Rager & Conmy (2023); Kramár, Lieberum, Shah & Nanda (2024) *AtP\**.

Four design points that matter:

- **Evaluated in the regime the freeze happens in.** The base run is corrupted **+ the joint
  $\mathbf{v}_1,\mathbf{v}_2$ patch** and the source is the corrupted activations, so $s_i$
  estimates the effect of *freezing* neuron $i$. Scoring the plain clean/corrupted contrast would
  answer a different question. Large **negative** = freezing it destroys the effect.
- **PLD, not IIA.** IIA is an argmax and has no gradient; PLD is the numerator of position
  recovery, so the gradient target and the reported metric are the same quantity.
- **A zero handle creates the graph.** `load_model` sets `requires_grad_(False)` on every
  parameter, so activations carry no graph by default. The hook inserts
  `h = zeros(B, d_ff, requires_grad=True)` into the `down_proj` input at $n_2$; `h.grad` is then
  exactly $\partial\,\mathrm{PLD}/\partial a$. The objective is the **sum** over the batch, not the
  mean, so each row's gradient is that example's own derivative with no $1/B$ to undo.
  `ATP_SCALE = 100` guards fp16 gradient underflow and is divided back out.
- **Ranked on a disjoint split.** The 128 held-out DAS triples, never the 400 the curve is reported
  on. With 37888 candidates, ranking and reporting on the same examples would be selection bias.

Scores from both layers are pooled into one ranking (`POOL_LAYER`, `POOL_WITHIN`, `ORDER`), so a
top-$k$ set can span layers. The cell also prints the L14/L15 split of the top 100 and the
**overlap between the two cases' top-100 sets** against its chance level — a free test of whether
the same neurons serve both routes into the comparator.

## 11. Verify — the top-$k$ freeze curve (cells 27–28)

`k ∈ {1, 3, 10, 30, 100, 300, 1000, 3000}`, four series per case:

- **top-$k$** by attribution,
- **random-$k$**, two seeds — without this the curve is uninterpretable, since freezing 3000
  arbitrary neurons also does damage,
- **bottom-$k$**, the other end of the ranking.

Plus the $k = 2 d_{ff}$ endpoint, which freezes every neuron in both layers. Qwen2's `down_proj`
has no bias, so that is *exactly* the MLP14+15 module freeze: the cell prints it next to the value
Part I already measured (`0.000` / `0.035` IIA) as a correctness check on the whole neuron path.

`make_neuron_freeze_hook` writes **in place**, unlike the module-level freeze. That is safe here and
only here: the `down_proj` input is `act_fn(gate(x)) * up(x)`, freshly allocated inside the MLP's
forward and consumed by nothing else, whereas a layer *output* is also read by the residual add.
Avoiding a `(B, T, 18944)` clone per hook call is what keeps the 66-condition sweep near two
minutes.

## 12. Inject — the sufficiency direction (cells 30–31)

The complement of freezing: the **clean** MLP output at $n_2$ written into an otherwise untouched
corrupted run, no upstream patch. Standard single-module activation patching (Vig et al. 2020; Meng
et al. 2022). It reuses `make_freeze_hook` unchanged — freeze and inject are the same hook with a
different source — and runs cumulatively (`MLP14`, `MLP14+15`, `MLP14+15+16`) so that L15/L16 cannot
quietly undo a lone L14 injection.

**Two IIA metrics, and the reason for them.** `IIA (tok_nc)` is the model naming the winning
*slot*; `IIA (tok_a)` is it naming the clean run's winning *value*. `make_batch` only stores the
corrupted run's `tok_nc` / `tok_b`, so `tok_a` is recovered from the clean prompts the same way
(leading digits are distinct inside a triple, so the first token is unambiguous). The freeze
experiment cannot tell "MLP14 writes the comparison outcome" from "MLP14 writes the answer value";
this pair of metrics can. The `none` row is the untouched corrupted baseline and should be
posrec 0.

## 13. Tune — neurons in the $\mathbf{v}_1,\mathbf{v}_2$ plane (cells 33–35)

**The frame.** $\mathbf{e}_1 = \mathbf{v}_1$, $\mathbf{e}_2 = \mathbf{v}_2$ orthogonalised against
it — `makefig_sharedreps.ipynb` Figure 6's Gram–Schmidt basis — read at the **pre-MLP L14
residual**, i.e. the input to `post_attention_layernorm`. That site matters: the MLP module's own
input is that vector already normalised and scaled by $\gamma$, a different frame, and sharedreps
found the rank-2 patch only does anything pre-MLP. Coordinates are centred on the cloud mean and
each axis is signed to rise with its own number, as in Fig 6.

**The difference axis** is measured on this notebook's own 1500-pair cloud: the least-squares
in-plane direction along which $\log y_1 - \log y_2$ grows fastest. A comparator neuron should read
along *that*, not along $\mathbf{e}_1$ or $\mathbf{e}_2$ — which is the sharp version of "is this
neuron aligned with the relevant subspace", and it connects directly to the diagonal decision
boundary sharedreps already measured with the plane sweep.

**Read-in alignment.** A SwiGLU neuron has two read-in vectors:
$a_i=\mathrm{silu}(w_{g,i}\cdot\tilde x)\,(w_{u,i}\cdot\tilde x)$ with
$\tilde x = \mathrm{RMSNorm}(x)\odot\gamma$. So both rows are multiplied by $\gamma$
(`post_attention_layernorm.weight`) before any cosine — RMSNorm's per-token rescaling is a scalar
and does not change directions, and unlike LayerNorm there is no centring, so no mean direction to
project out. Reported per neuron:

- **in-plane fraction** $\lVert P_V w\rVert / \lVert w\rVert$, chance $\sqrt{2/d} = 0.024$,
- **in-plane angle** $\mathrm{atan2}(w\cdot\mathbf{e}_2,\, w\cdot\mathbf{e}_1)$, against the
  measured difference axis.

Gate and up are kept separate throughout, because they do different jobs: the gate is the switch
(`silu` decides *whether* the unit fires), so a threshold/comparator unit's prediction is on
$w_g$. `e11/which_neurons_compare.ipynb` only looked at `up_proj`.

500 random neurons give the null for the in-plane fraction (the shaded band in the alignment
figure is its 95th percentile) — a weight-space cosine needs a null, since high-dimensional
intuitions about "small" cosines are unreliable.

**Caveat.** Read-in alignment is data-independent: a neuron can align well and never fire on this
distribution. It is a *read-side* statistic and is never used as the importance ranking — that is
what the attribution + top-$k$ verification is for. The tuning maps (cell 34) are the activation-
space version of the same question and are the stronger evidence.

**Not done here.** The strongest version would record neuron activations *during* the sharedreps
Fig 10 plane sweep, giving each neuron's causal tuning surface rather than its correlational
tuning on the natural cloud. That needs the sweep machinery from the other notebook and is the
obvious next step.

## 14. New outputs

- `figs/comparatorfig_2digit_topk_neurons.png` — IIA / posrec vs $k$, both cases, with controls.
- `figs/comparatorfig_2digit_inject_mlp.png` — injection, three metrics, both cases.
- `figs/comparatorfig_2digit_neuron_tuning.png` — top neurons over the plane, with the $y_1>y_2$
  sign map underneath for comparison.
- `figs/comparatorfig_2digit_neuron_alignment.png` — in-plane fraction vs read-in angle, gate and
  up, against the random-neuron null and the measured difference axis.
- The four `results/comparator_{atp,topk,inject,tune}_2digit.npz` files above.

## 15. Additional references

- Nanda (2023), *Attribution Patching: Activation Patching At Industrial Scale*.
- Syed, Rager & Conmy (2023), *Attribution Patching Outperforms Automated Circuit Discovery*.
- Kramár, Lieberum, Shah & Nanda (2024), *AtP\*: An efficient and scalable method for localizing
  LLM behaviour to components* — the saturation failure mode of the linear approximation, and why
  a top-$k$ verification sweep is required rather than optional.
- Vig et al. (2020), *Investigating Gender Bias in Language Models Using Causal Mediation
  Analysis* — single-module activation patching, the injection direction.
- Elhage et al. (2022), *Solu: Toy Models / neuron interpretability*; Gurnee et al. (2023),
  *Finding Neurons in a Haystack* — sparse probing, and reading a neuron's selectivity off a
  feature plane.

## 16. Result — neuron level (n = 400 per case)

### Sparsity

IIA of the joint patch as the top $k$ neurons (pooled over L14, L15; 37888 candidates) are frozen
at $n_2$:

| $k$ | 0 | 1 | 3 | 10 | 30 | 100 | 1000 | 3000 | 37888 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `n1 larger` | 0.675 | 0.498 | 0.175 | 0.013 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `n2 larger` | 0.885 | 0.675 | 0.598 | 0.130 | 0.030 | 0.010 | 0.000 | 0.000 | 0.035 |

**Ten neurons out of 37888 abolish the effect.** The effect halves at $k=3$ (`n1 larger`) and
$k=10$ (`n2 larger`).

The controls are what make that readable:

| $k$ | 100 | 300 | 1000 | 3000 |
| --- | --- | --- | --- | --- |
| random-$k$, `n1 larger` | 0.675 | 0.673 | 0.640 | 0.495 |
| bottom-$k$, `n1 larger` | 0.848 | 0.882 | 0.923 | 0.955 |

Random-$k$ is flat at the unfrozen value until $k\approx10^3$ — three orders of magnitude above
where the top-$k$ curve has already hit zero. Bottom-$k$ *raises* IIA, to 0.955 (`n1 larger`) and
0.998 (`n2 larger`): the positively-scored neurons oppose the interchange, and freezing them
removes that opposition. The signed ranking is therefore meaningful at both ends, not just as a
magnitude.

**The $k = 2d_{ff}$ endpoint reproduces the module freeze exactly** — 0.000 vs 0.000 and 0.035 vs
0.035 — confirming the neuron path, the in-place hook and the bias-free `down_proj` assumption.

### The same neurons serve both routes

Top-100 overlap between the two cases is **50/100** against a chance level of 0.26/100. Seven of
the top ten are shared, and both cases agree on ranks 1 and 2:

```
n1 larger: L14n6262 L14n6150 L15n12784 L15n8383 L14n9459 L14n5076 L14n5820 L15n3003 ...
n2 larger: L14n6262 L14n6150 L15n5795  L15n12784 L15n3003 L14n6854 L14n9459 L14n5076 ...
```

The comparator is one set of units, reached by two different routes — the neuron-level version of
the cross-case result in §5a.

Note that attribution spreads the top 100 roughly evenly over L14 and L15 (46/54 and 52/48) even
though the *module* freeze said MLP14 ≫ MLP15. Both are true: MLP15's individual neurons carry
attribution, while the module as a whole is partly redundant with MLP14.

### They are comparator units, not magnitude units

`figs/comparatorfig_2digit_neuron_tuning.png`: every one of the top six splits the
$\mathbf{v}_1,\mathbf{v}_2$ plane along the **same diagonal** as $\mathrm{sign}(\log y_1-\log y_2)$,
drawn underneath for comparison. Rank 1 (L14 n6262) and rank 2 (L14 n6150) have opposite signs —
the same boundary, read in opposite directions. None of them shows the monotone one-axis gradient a
magnitude-encoding neuron would.

(The thin gap along the diagonal in the cloud is a sampling artifact: `sample_cloud_pairs` rejects
pairs that share a leading digit, which removes most near-equal pairs.)

### Necessary but not sufficient

Injecting the clean MLP output at $n_2$ into an untouched corrupted run:

| injected | `n1 larger` posrec / IIA(tok_nc) | `n2 larger` posrec / IIA(tok_nc) |
| --- | --- | --- |
| none | 0.000 / 0.000 | 0.000 / 0.000 |
| MLP14 | 0.068 / 0.010 | 0.906 / 0.445 |
| MLP14+15 | 0.323 / 0.075 | 1.835 / 0.955 |
| MLP14+15+16 | 0.502 / 0.115 | 1.892 / 0.965 |

`n1 larger` is the clean test — the $n_2$ token is identical in the clean and corrupted runs there,
so the injection imports only the MLP's response to a different *context*. It recovers half the
position recovery but almost none of the IIA (0.115). So **MLP14/15 at $n_2$ are necessary but not
sufficient**: the comparison outcome alone does not produce the answer.

`n2 larger` reaches 0.955, but that number is confounded and should not be quoted as sufficiency:
there the $n_2$ token itself differs between the runs ($a$ vs $c$), so the injected activation
carries the clean number, not just the comparison.

### What the MLP writes is the slot, not the value

`IIA (tok_a)` is **0.000 in every condition of both cases** — no injection ever makes the model
name the clean run's winning value. In `n2 larger` the same injections drive `IIA (tok_nc)` to
0.955: the model names $c$, the value actually sitting at that position in the corrupted run, even
though the injected activation was computed with $a$ there.

So the MLP output at $n_2$ is read downstream as *which slot wins*, and the value is retrieved
separately from the token that is actually present. This is the distinction the freeze experiment
could not make.

### Read-in alignment is real but weak

| | median in-plane fraction | background median | fraction above background 95th pct |
| --- | --- | --- | --- |
| gate | 0.044 | 0.017 | 0.52 |
| up | 0.050 | 0.016 | 0.65 |

(`n1 larger`; chance for a random unit vector is 0.024.) Angular concentration within $\pm30°$ of
the measured difference axis ($-21.3°$): 0.55 (gate) / 0.65 (up) against 0.43 for random neurons,
where uniform would be 0.33.

Both enrichments are genuine and both are modest — barely half the top neurons clear the 95th
percentile of the random null. **Weight-space alignment alone would not have found these units**,
which is exactly why the causal attribution was the ranking and this is the interpretation.

**But these are medians over the top 100, and they are diluted.** The top *ten* read the plane far
more strongly — in-plane fractions of 0.21–0.40 for `gate` (§18), against a chance level of 0.024,
i.e. 9–17x chance. Alignment falls off steeply with rank, so "modest" describes the top-100
population, not the neurons that actually carry the effect.

## 17. Top-10 profile — receptive fields and read directions (cells 36–39)

Three cells appended after the tuning block, cached to `results/comparator_rf_2digit.npz`.

**Which neurons.** `RF_CASE = CASES[0]` (`n1 larger`), the same ranking the tuning maps use; set it
to `CASES[1]` for the other case. The printed table gives rank, layer, neuron index, attribution
score, the in-plane fraction of each read-in vector and its angle, then the L14/L15 split. Note
the pool is `NEURON_LAYERS = [14, 15]` — **L16 is not searched**, because the module freeze showed
it contributes nothing (§5a). Adding it means putting 16 in `NEURON_LAYERS` and rerunning the
attribution and sweep with `FORCE = True`.

**Receptive fields** (cell 38) are measured on a dense grid of prompts, `np.arange(low+1, high,
RF_STEP)` on each axis — 23 × 23 = 529 prompts at the default stride 4, about 12 forward passes.
A grid rather than the natural cloud, so every cell is measured and none is interpolated, and the
diagonal is covered (the cloud has a gap there, §16). `pairs` is built $y_1$-major, so the
activation column reshapes to `[y_1, y_2]` and is transposed for `imshow`, giving $y_1$ on x and
$y_2$ on y with `origin='lower'`. Each panel is scaled symmetrically about zero on `RdBu_r`, per
neuron, since the activation ranges differ by an order of magnitude between them; the dashed line
is $y_1=y_2$.

**Read directions** (cell 39) are the $\gamma$-folded `gate_proj` / `up_proj` rows projected onto
the same signed plane basis the tuning cell used (`TUNE['sgn']`, so the arrows and the cloud share
a frame). Drawn over the grey cloud with the measured $y_1=y_2$ boundary, numbered by rank, L14
blue and L15 amber, gate solid and up dashed. **The arrows are normalised to a common length** —
they show direction only, because the in-plane fraction is ~0.02–0.06 while the cloud spans tens of
units, so true-to-scale arrows would be invisible. The magnitudes are in the table instead.
`SHOW_LAYERS = [14]` restricts the figure to the L14 arrows.

New figures: `figs/comparatorfig_2digit_neuron_receptive_fields.png`,
`figs/comparatorfig_2digit_neuron_read_directions.png`.

## 18. SVD of the L14 read-in weights inside the plane (cells 40–42)

The L14 members of the top-10 (6 of them at the default `RF_TOP`), each read-in vector
$\gamma$-folded and projected onto span($\mathbf{v}_1,\mathbf{v}_2$), stacked into an $n\times2$
matrix $M$ and decomposed. Pure numpy over the cached `RF` and `TUNE` arrays — no model, no new
cache file.

**Row normalisation.** Each row is divided by the **full** weight norm, so its length is that
neuron's in-plane fraction: the SVD then describes how much of each neuron's reading lands in the
plane, with arbitrary per-neuron weight scale removed. `ROW_NORM = 'direction'` makes every row
unit length instead, which turns the statistic into pure angular clustering.

**Uncentred.** The origin is meaningful here — a zero row means the neuron does not read the plane
at all — so this is a second-moment decomposition, not a PCA of the 6 points. Centring would
subtract the very thing being measured.

**What the numbers mean.** $\sigma_1/\sigma_2$ is the anisotropy: large means the group reads one
shared in-plane axis rather than spanning the plane. $\sigma_1^2/(\sigma_1^2+\sigma_2^2)$ is the
share of squared in-plane weight on that axis. The first right singular vector $v_1$ is that
shared axis, sign-oriented toward the measured difference axis, and its angular deviation from that
axis is the quantity of interest — if the comparator neurons read the plane through one direction,
it should be the difference direction.

**The null is not optional.** Six random 2-vectors already give $\sigma_1/\sigma_2 \approx 2$, so a
raw ratio is uninterpretable. The cell bootstraps 4000 groups of the same size from random L14
neurons — reusing the in-plane fraction and angle of the 500-neuron alignment background the tuning
cell saved, filtered to L14 by its stored layer array — and reports the observed ratio's percentile
against that, and likewise the percentile of the angular deviation.

Reported per read-in matrix (`gate` and `up` separately, as everywhere else): both singular values,
both right singular vectors with their angles, the ratio with its null median and percentile, and
the deviation from the difference axis with its null median and percentile.

The figure draws, per read-in: the $n$ rows as numbered arrows (rank labels), the
$\sigma$-scaled ellipse whose axes are the singular vectors, the two singular vectors as heavy
arrows labelled with their values, and the difference axis dashed. The ellipse is scaled to the
panel, so compare shapes across panels, not sizes. Saved as
`figs/comparatorfig_2digit_l14_readin_svd.png`.

`SVD_LAYER = 15` runs the same analysis on the L15 members instead.

## 18a. SVD result — the L14 gates read one axis, and it is not the difference axis

Six L14 neurons (ranks 1, 2, 5, 6, 7, 9), rows normalised by the full weight norm:

| read-in | \(\sigma_1\) | \(\sigma_2\) | \(\sigma_1/\sigma_2\) | null median | pct | \(v_1\) angle | off the difference axis | pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate | 0.670 | 0.031 | **21.4** | 2.04 | 1.000 | +0.5 deg | 21.7 deg | 0.330 |
| up | 0.492 | 0.244 | 2.02 | 2.06 | 0.482 | +3.9 deg | 25.1 deg | 0.435 |

Two clean results and one puzzle.

- **The gates are almost perfectly collinear**: 99.8% of their squared in-plane weight lies on a
  single axis, an anisotropy of 21.4 against a null median of 2.04 (percentile 1.000). Five of the
  six gate vectors sit within a few degrees of \(\pm\mathbf{v}_1\). As a group these neurons read
  the plane through *one* direction.
- **The up vectors do not**: anisotropy 2.02 against a null median of 2.06, percentile 0.48 —
  indistinguishable from six random vectors. Gate and up play different roles, as §13 assumed, but
  not the roles predicted.
- **The shared gate axis is \(\mathbf{v}_1\), not the difference axis.** It sits at +0.5 deg while
  the measured difference axis is at −21.3 deg, and the 21.7 deg deviation is at percentile 0.33 of
  the null — no better than chance. The §13 prediction, that the gate would carry the difference
  direction, is **wrong**.

The puzzle: the tuning maps (§16) are unambiguously diagonal, yet the gates read \(\mathbf{v}_1\).
Both quantities are gradients in the same plane-coordinate frame, so the comparison is
apples-to-apples and the discrepancy is real. Three candidate resolutions, none tested yet:

1. the diagonal comes from the **product** \(\mathrm{silu}(w_g\!\cdot\!\tilde x)(w_u\!\cdot\!\tilde x)\),
   not from either factor — a \(\mathbf{v}_1\)-reading gate times a differently-oriented up;
2. 60–80% of each read-in is **out of plane** and carries the rest of the computation;
3. the threshold does the work: a gate that reads \(y_1\) alone still fires on a diagonal if what
   it is compared against is carried elsewhere.

§19 is the first probe of this: it shows what these neurons actually *write* into the plane.

## 19. Pushing the plane through the neurons of interest (cells 44–46)

The same cloud and the same signed plane basis, read at each stage and projected back into the
plane. Cached to `results/comparator_push_2digit.npz`, one forward pass over 1500 prompts (~10 s).

The two layers are treated differently, because the residual stream makes them different:

- **L14.** The plane is *defined* at the pre-MLP L14 residual, so MLP14's write
  \(m_{14}=\sum_{i} a_i W^{\mathrm{down}}_{:,i}\) over the 6 neurons is the direct image of that
  plane. Panels: the pre-MLP residual, the write alone, and their sum.
- **L15.** MLP15 reads the residual, which *already carries* the structure, and its write is added
  on top. Panels: the pre-MLP L15 residual, MLP15's write over its 4 neurons, and their sum.

Every stage is projected with the same `E` (signed, from `TUNE['sgn']`), so all panels share a
frame. Each panel is centred on its own mean, and the mean write is printed separately — otherwise
the MLP's large mean offset would dominate the axes and hide the shape.

**The table** gives, per stage: \(\sigma_1/\sigma_2\) of the centred 2D cloud (how squashed it is),
the \(R^2\) of a linear read of \(\log y_1\), of \(\log y_2\) and of their difference from the two
coordinates, and the AUC for \(y_1>y_2\) along the fitted difference direction.

Read it as follows. Because \(\log y_1\) and \(\log y_2\) are independent in the cloud, a stage
carrying **only** their difference scores \(R^2\approx0.5\) on each individually and \(\approx1\) on
the difference — that 0.5 is the floor, not partial information. A stage that still carries both
numbers separately scores high on all three. So the diagnostic for "the MLP has converted a 2D
magnitude code into a 1D comparison" is: \(R^2\) on the difference stays high while the two
individual \(R^2\) fall toward 0.5, with \(\sigma_1/\sigma_2\) rising.

`m14_all` and `m15_all` are the full-MLP writes, included as a control: they say whether any
squashing is specific to the neurons of interest or is what the whole MLP does.

Figure: `figs/comparatorfig_2digit_plane_through_neurons.png`, 2x3, coloured by
\(\log y_1-\log y_2\) with each panel's own \(y_1=y_2\) boundary dashed.

## 19a. Result — what the neurons write into the plane

| stage | \(\sigma_1/\sigma_2\) | \(R^2\log y_1\) | \(R^2\log y_2\) | \(R^2\) diff | AUC \(y_1>y_2\) | mean write |
| --- | --- | --- | --- | --- | --- | --- |
| pre-MLP L14 residual | 4.50 | 0.752 | 0.905 | 0.872 | 1.000 | |
| MLP14 write, 6 neurons | 1.37 | 0.706 | 0.592 | 0.783 | 0.991 | 0.59 |
| MLP14 write, all neurons | 2.73 | 0.716 | 0.661 | 0.679 | 0.869 | 0.90 |
| residual + MLP14 write | 4.96 | 0.675 | 0.904 | 0.855 | 0.999 | |
| pre-MLP L15 residual | 8.01 | 0.233 | 0.869 | 0.780 | 0.978 | |
| **MLP15 write, 4 neurons** | **25.88** | 0.327 | 0.377 | 0.607 | **1.000** | 0.18 |
| MLP15 write, all neurons | 1.82 | 0.337 | 0.346 | 0.613 | 0.878 | 0.63 |
| residual + MLP15 write | 7.10 | 0.251 | 0.869 | 0.785 | 0.982 | |

Four things stand out.

**MLP15's four neurons write a rank-1, two-cluster signal.** \(\sigma_1/\sigma_2 = 25.9\) — their
in-plane write lies on a *line* — and along that line the cloud splits into two well-separated
clusters with AUC 1.000. That is a categorical comparison outcome, not a graded magnitude. The
all-neuron control for the same layer is 1.82, so the collapse is specific to these four and is not
what MLP15 does in general.

**MLP14's six neurons do not collapse the plane** (\(\sigma_1/\sigma_2 = 1.37\)) but their write is
already **bimodal** — two colour-separated clusters, AUC 0.991, well above the all-neuron control's
0.869. So the binarisation begins at MLP14 and is made one-dimensional at MLP15.

**The individual magnitudes are discarded on the way.** Between the pre-MLP L14 residual and the
MLP15 write, \(R^2\) for \(\log y_1\) goes 0.75 → 0.33 and for \(\log y_2\) 0.91 → 0.38, while AUC
for \(y_1>y_2\) goes 1.000 → 1.000. The writes keep which number is larger and drop how large each
one is. (The \(R^2\) diff column falls too, to 0.61, because a two-cluster code is a poor *linear*
predictor of a continuous difference even when it separates the sign perfectly — AUC is the metric
to read here, not \(R^2\).)

**In the plane, the writes are small.** Mean in-plane write is 0.59 (MLP14) and 0.18 (MLP15)
against a residual spread of ±20, and the "residual + write" panels are visually identical to the
residual panels. Combined with the in-plane fractions of 0.2–0.4 in §18, this says the plane is
where these neurons **read** the comparison from, not where they write the answer to — the write
goes mostly out of plane, which is consistent with §16's finding that freezing them destroys the
behaviour while the plane itself barely moves.

This also partly answers §18a's puzzle: the gate vectors reading \(\mathbf{v}_1\) rather than the
difference axis does not prevent the group from producing a clean diagonal split, because the
selectivity is built by the gate × up product and the threshold, not by a single read-in direction.

## 19b. DAS inside the MLP's own neuron space (cells 47–50)

Inserted between the push-through plots and the L15 residual DAS, because it asks the same question
one level down: not *where in the residual* the causal variable lives, but *what the MLP itself is
computing*. Cached to `results/comparator_mlpdas_2digit.npz`, roughly 5–8 minutes for the eight
fits.

**The subspace is the neurons' own activation vector**, not the residual stream: 6 dimensions at
MLP14 and 4 at MLP15, the neurons from the top-10. The intervention is

$$a_S \leftarrow a_S + PP^{\top}\!\left(a_S^{\mathrm{clean}} - a_S\right),\qquad P\in\mathbb{R}^{k\times r}$$

applied to the `down_proj` input at $n_2$. Two advantages over a residual-space DAS: the search
space is 6- or 4-dimensional rather than 3584, so the fit is far better determined; and the learned
direction is **a weighting over named neurons**, which the third column of the figure plots
directly. Full rank ($r=k$) sets every one of those neurons to its clean value and is the ceiling
for the group.

`make_neuron_das_hook` clones the `down_proj` input, unlike `make_neuron_freeze_hook` which writes
in place — the freeze runs under `no_grad`, this one runs under autograd, where an in-place write
on a tensor the graph still needs is invalid.

**64 training triples rather than the usual 128.** The basis has at most $k\times r = 12$ free
parameters, so counterfactual pairs are not the binding constraint here, and it halves the fit
time. Everything else follows the project convention (100 steps, lr 0.05, minibatches of 32,
`cf_pair_loss`, gradient accumulation).

Trained separately per case, as in §20, so each bar is its own experiment.

**The cloud panels reuse the tuning capture** — `TUNE['acts_L14']` / `acts_L15` are the same 1500
prompts, so projecting them onto the learned bases needs no forward pass. Coloured by which slot
holds the max.

What to look for: if the rank-1 IIA already matches full rank, the group of neurons is carrying
**one** causal variable and the MLP is computing a scalar; the weight bars then say whether that
scalar is one dominant neuron or a genuine combination. If rank 1 falls well short of full rank,
these neurons carry more than the comparison.

Figures: `figs/comparatorfig_2digit_mlp_neuron_das_space.png`,
`figs/comparatorfig_2digit_mlp_neuron_das_iia.png`.

## 20. DAS at the L15 residual, rank 1/2/3 (cells 52–55)

Rank-$k$ DAS on the residual stream leaving layer 15 at $n_2$ — the site whose write §19a showed to
be a rank-1, two-cluster code. Cached to `results/comparator_das15_2digit.npz`; the fits are the
expensive part, roughly 5–8 minutes for all six.

**`train_das_at(layer, pos, k, bat, ch, seed)`** generalises the project's two existing trainers:
`data_numberreps.train_das_1d` is pinned to L13 and rank 1, `data_sharedrep.train_h14_das` to a
head's output space. Same everything else — `DASSubspace`'s QR retraction for the Stiefel
constraint, `cf_pair_loss` (cross-entropy toward `tok_nc` restricted to {`tok_b`, `tok_nc`}, the
differentiable surrogate for IIA), 128 held-out triples, 100 steps, lr 0.05, gradients accumulated
over minibatches of 32 so the training-set size is decoupled from the backward pass's activation
memory. Geiger et al. (2023), rank-$k$ single-variable case.

**Trained separately per case.** Each bar in the figure is then its own experiment rather than a
transfer result, which is what makes "IIA for first-number-larger vs second-number-larger" a fair
comparison. The cell additionally cross-evaluates every basis in the other case and prints the
transfer columns beside the own-case ones — free at evaluation time, and the interesting question
is whether one subspace serves both, since §16 found the *neurons* do.

**The full-rank interchange at the same site** (`L15 full @ n2`) is evaluated as the fourth bar. It
is the ceiling any subspace at that layer and position can reach, so the rank-$k$ bars are read as a
fraction of it, not against 1.0.

**The cloud plots** (cell 50) project a 1500-pair cloud's L15 residual at $n_2$ onto each basis and
colour by **which slot holds the maximum** — the variable DAS is being asked to control, and the
one §19a says is encoded categorically here. Rank 1 is a pair of histograms, rank 2 a scatter, rank
3 a 3D scatter. Projections are centred on the cloud mean. `DAS15_PLOT_CASE` selects whose basis is
shown.

Expect the rank-1 panel to be bimodal and the higher ranks to add spread within each cluster rather
than new separation, if the L15 code really is the one-dimensional categorical signal §19a
measured. If instead rank 2 or 3 separates the slots better than rank 1, the outcome is not purely
binary at this site.

Figures: `figs/comparatorfig_2digit_l15_das_space.png`,
`figs/comparatorfig_2digit_l15_das_iia.png`.

## 21. MLP15 alignment with the DAS direction, and L14 → L15 connectivity (cells 56–59)

Cached to `results/comparator_connect_2digit.npz`. The weight parts are instant; the causal edges
are ~1 minute. **Requires the L15 DAS block (§20) to have run**, since Q1 needs its basis.

### Q1 — which MLP15 neurons write along the DAS direction

MLP15 neuron $j$ adds $a_j W^{\mathrm{down}}_{:,j}$ to the residual, and the L15 DAS basis $Q$ is a
subspace of that same residual, so the two are directly comparable with no change of basis. The
statistic is the fraction of the write inside the subspace,

$$\rho_j=\frac{\lVert Q^{\!\top}w_j\rVert}{\lVert w_j\rVert}\qquad(\;=|\cos|\ \text{for rank }1),$$

reported for ranks 1–3 and for both cases' bases.

**Significance is empirical.** $\rho$ is computed for **all 18944** MLP15 neurons and each neuron of
interest is placed in that null as a percentile — there is no parametric null worth trusting for a
cosine against a trained direction, and the analytic $\sqrt{r/d}$ chance level ignores the fact
that `down_proj` columns are not isotropic. With four neurons tested, treat percentiles below
~0.99 as unremarkable.

A second column, $\rho_j\cdot\mathrm{sd}(a_j)$, weights the geometry by how much the neuron actually
varies over the cloud: a neuron can point along the DAS direction and barely move, or move a lot
along a direction only partly in the subspace. The first number says *where* it writes, the second
*how much* it actually pushes the residual along the causal direction.

### Q2 — which L14 neurons feed which L15 neurons

**Direct weight path** (`path_gate`, `path_up`): L14 neuron $i$ writes $W^{\mathrm{down}}_{:,i}$;
L15 neuron $j$ reads through its $\gamma$-folded `gate` and `up` rows. Their cosine is the
composition strength — the "virtual weight" of Elhage et al. (2021), here in its simplest form
because an MLP→MLP path through the residual stream is a plain inner product with no attention
pattern in between. Each entry carries the percentile of its magnitude against the null of that
same L14 write against **all** 18944 L15 reads.

Two caveats, both important: this is the **direct** residual path only, ignoring anything L15's
attention routes in from other positions; and it is pure geometry, blind to whether the upstream
neuron ever fires or whether the downstream one is in its active regime.

**Causal edge** (`edge_<case>`) is the answer to those caveats. Each L14 neuron of interest is
patched at $n_2$ to its clean-run value, and the shift in every L15 neuron of interest is measured
and divided by that neuron's own spread over the cloud, so entries read as "moves neuron $j$ by
$x$ standard deviations". The `all 6` row patches them together, which is not the column sum
whenever the path is nonlinear — comparing the two is a cheap additivity check.

This is case-dependent (the patch is a counterfactual), so it is computed for both; the weight path
is not, so it is computed once.

Figures: `figs/comparatorfig_2digit_l15_neuron_das_alignment.png` (bars with the null's 95th/99th
percentiles), `figs/comparatorfig_2digit_l14_l15_connectivity.png` (2x2 heatmaps: weight path gate
and up on top, causal edge per case below, rows = L14 neurons, columns = L15 neurons, both labelled
by global rank and index).

## 22. Each L14 neuron's switching boundary in the plane (cells 60–62)

Six panels, one per L14 neuron of interest: the cloud coloured by that neuron's activation, its
$\gamma$-folded `gate` read direction as an arrow, and two lines for where it switches on. Cached
to `results/comparator_bound_2digit.npz`; one cloud pass, ~10 s. The cell self-heals a cache
written before the slice boundary existed.

### Why there are two lines

A SwiGLU neuron is off where its gate preactivation is negative. With $w_{\mathrm{eff}} =
\gamma\odot w_g$,

$$z = w_g\cdot\tilde x = \frac{w_{\mathrm{eff}}\cdot x}{\mathrm{rms}(x)},$$

so the boundary is $\{x : w_{\mathrm{eff}}\cdot x = 0\}$ — a hyperplane through the origin.
**RMSNorm cannot move it**: dividing by a positive scalar leaves the sign unchanged. (An earlier
version of this section wrongly implied the RMS scale was part of the difficulty. It is not; it
only rescales $z$'s magnitude, which matters for a least-squares fit and nothing else.)

A hyperplane does not *project* to a line — a generic hyperplane projects onto the whole 2D plane.
What is well defined is its **trace on a slice**. Writing $x$ in terms of its in-plane coordinates
$(p,q)$ and an out-of-plane remainder,

$$z \propto a p + b q + w_{\mathrm{eff}}\cdot x_\perp,\qquad (a,b) = \text{the drawn arrow},$$

so fixing $x_\perp$ at the cloud mean gives the line $ap+bq+c_0=0$, whose normal **is** the arrow.
That is the *slice* line (solid), and the cell asserts that its gradient reproduces `RF['read_gate']`
to zero error.

The second line (dash-dot) is the least-squares fit of $z$ on $(p,q)$. Its normal is a total
regression coefficient, not a partial derivative: it absorbs out-of-plane variation *correlated*
with $p$ and $q$, and so is generally not orthogonal to the arrow. The angle between the two lines
(`slice-fit` in the table) measures exactly that correlated out-of-plane structure.

### Reading the accuracies

Every accuracy is quoted against the **majority baseline** $\max(\text{on}, 1-\text{on})$, because
a neuron that is always on is classified perfectly by a line that never cuts the cloud. An accuracy
sitting at the baseline means the boundary does not usefully separate anything, whatever angle it
was drawn at, and a low fit $R^2$ means the plane barely predicts $z$, so the fitted line's angle is
fitted to noise.

### Result (first run, fitted line only)

| rank | neuron | gate on | baseline | fit acc | fit $R^2$ | fit angle | off the $y_1{=}y_2$ line |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | n6262 | 1.000 | 1.000 | 1.000 | 0.085 | −94.0° | 17.2° |
| 2 | n6150 | 0.902 | 0.902 | 0.902 | 0.660 | +92.3° | 23.5° |
| 5 | n9459 | 0.609 | 0.609 | 0.609 | 0.018 | +129.0° | 60.2° |
| 6 | n5076 | 0.545 | 0.545 | 0.758 | 0.348 | −85.0° | 26.2° |
| 7 | n5820 | 0.201 | 0.799 | 0.887 | 0.737 | −89.9° | 21.4° |
| 9 | n4724 | 0.669 | 0.669 | 0.958 | 0.883 | +91.3° | 22.5° |

**Most of these neurons do not have a meaningful boundary in this plane.** Ranks 1, 2 and 5 sit
exactly at the majority baseline — their fitted line is a trivial "always on" classifier — and
ranks 1 and 5 additionally have $R^2$ of 0.09 and 0.02, so the plane explains essentially none of
their gate preactivation. Rank 1 is never off at all across the whole cloud, so for it the question
is not where it switches but how its graded value varies.

Only ranks 9 ($R^2$ 0.88, 0.96 vs 0.67 baseline) and 7 ($R^2$ 0.74, 0.89 vs 0.80) have a boundary
that is both planar and useful, with rank 6 marginal. For those, the boundary runs at about
$\pm90°$ — roughly along the $\mathbf{v}_2$ axis, some 21–26° off the $y_1{=}y_2$ line.

This is the clearest statement yet of §18a's puzzle: these neurons' switching is largely driven by
directions **outside** the $\mathbf{v}_1,\mathbf{v}_2$ plane, even though their activations map out
a clean diagonal in it (§16). The plane is where the comparison can be *read off*, not where the
neurons' own decision is made.

Figure: `figs/comparatorfig_2digit_l14_neuron_boundaries.png`.

## 23. Run modes and the result manifest (cells 3, 5, 9, 11, 13, and the last two)

The notebook has two ways to run, chosen automatically.

**`PLOT_ONLY`** is set in the plot-style cell: it lists the twelve `results/comparator_*.npz` files
every expensive block writes and is `True` when all of them exist. `comparator_checks_*.npz` is
deliberately **not** in that list — nothing plots from it, and gating on it would mean the first
save flipped `PLOT_ONLY` to `True` and so stopped cells 9 and 11 from ever recording their numbers.
In plot-only mode

- cell 5 skips the model load and takes the token layout and `d_model` from the saved meta,
- cell 9 skips the per-case captures (`BAT` / `SRC` / `FRZ` feed compute paths only),
- cell 11 skips the sanity check,
- cell 13 skips both the grid and the construction of `ARMS` (which needs the direction tensors),
- every `cached_npz` block loads instead of computing.

Measured: **the whole notebook runs top to bottom in ~11 seconds with no model and no GPU.** With
anything missing it prints `COMPUTE` and names the files, and only the missing blocks recompute.

Overrides, from coarse to fine: `PLOT_ONLY = False` forces the compute path back on; `FORCE = True`
recomputes every `cached_npz` block; `FORCE_GRID` does the 48-condition grid alone; deleting one
`.npz` recomputes exactly that block.

**The last cell is self-contained** — it can be run on its own in a kernel that has executed only
part of the notebook, which is the usual case after editing cells. It imports what it needs,
falls back to its own copy of `RESULT_FILES` and `ck` when those are not defined, and reads
everything else from disk.

It saves what the other cells only printed — the patch-source displacement (cell 9), the no-patch
sanity numbers (cell 11) and the read-in SVD (cell 41) — into
`results/comparator_checks_2digit.npz`. It merges rather than overwrites, so entries already on
disk survive when their producing cell has not run in this kernel, and it *names* whatever is
neither in memory nor on disk rather than skipping it silently. It then prints a manifest of every
result file with size, array count and contents, and says whether the next run will be plot-only.

Total on disk: about 2.4 MB for the whole analysis.

### Result — the slice boundary matches the fitted one

With the slice line added (§22), its sign accuracy turns out to match the fitted line's almost
exactly:

| rank | neuron | baseline | slice acc | fit acc | fit $R^2$ | slice−fit angle |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | n6262 | 1.000 | 1.000 | 1.000 | 0.085 | 6.0° |
| 2 | n6150 | 0.902 | 0.902 | 0.902 | 0.660 | 0.4° |
| 5 | n9459 | 0.609 | 0.609 | 0.609 | 0.018 | 82.0° |
| 6 | n5076 | 0.545 | 0.757 | 0.758 | 0.348 | 6.6° |
| 7 | n5820 | 0.799 | 0.886 | 0.887 | 0.737 | 0.3° |
| 9 | n4724 | 0.669 | 0.942 | 0.958 | 0.883 | 0.7° |

The two constructions agree to within a degree wherever the planar fit means anything, so the
correlated out-of-plane structure the regression could have absorbed is negligible: the honest,
orthogonal-to-the-arrow line is as good as the fitted one. Rank 5's 82° gap is the exception that
proves it — with $R^2=0.018$ the *fit's* angle is noise, while the slice angle is the true in-plane
gradient.

Ranks 6, 7 and 9 beat their majority baselines (0.76 vs 0.55, 0.89 vs 0.80, 0.94 vs 0.67); ranks 1,
2 and 5 sit exactly on theirs, so for those the boundary does not usefully cut this cloud.

## 24. Neurons that matter in both cases (inserted before the save cell)

Cached to `results/comparator_shared_2digit.npz`; about 40 seconds (a grid pass plus eight freeze
conditions).

### Why the intersection is the interesting set

The attribution ranking is per case, and **the case is the perturbation**: `make_batch` varies only
the winning slot's number, so `n1 larger` perturbs $y_1$ with $y_2$ fixed and `n2 larger` perturbs
$y_2$ with $y_1$ fixed. Intersecting the two top-$k$ lists therefore selects neurons that matter
*whichever* number is contested — the case-independent core of the comparator — and drops those
that only serve one route.

### Overlap (pool = 37888 neurons)

| top-$k$ each | overlap | chance | fold | L14 / L15 |
| --- | --- | --- | --- | --- |
| 10 | 7 | 0.003 | 2650x | 4 / 3 |
| 20 | 13 | 0.011 | 1230x | 7 / 6 |
| 30 | **19** | 0.024 | 800x | 11 / 8 |
| 50 | 31 | 0.066 | 470x | 17 / 14 |
| 100 | 50 | 0.264 | 190x | 26 / 24 |

The default `SHARED_K = 30` gives 19 neurons. Chance is $k^2/37888$, so even the loosest depth is
two orders of magnitude above it, and the enrichment *falls* with $k$ — the agreement is
concentrated at the top, exactly where it should be if the shared set is real. Ranks 1 and 2 are
the same neuron in both cases.

The top-30 overlap, with each neuron's rank in (`n1 larger` / `n2 larger`):

```
L14n6262(1/1)   L14n6150(2/2)   L15n12784(3/4)   L15n8383(4/10)   L14n9459(5/7)
L14n5076(6/8)   L15n3003(8/5)   L15n9442(13/18)  L14n6854(14/6)   L14n9341(15/19)
L15n188(16/16)  L14n5222(18/9)  L15n3616(19/12)  L15n7514(21/13)  L14n7636(23/26)
L14n682(24/11)  L15n11556(25/20) L14n4851(26/28) L14n15775(27/15)
```

Several of these are new relative to the top-10 profile in §17, which was `n1 larger` only —
n6854, n5222, n9442, n9341, n188, n3616 and the rest rank well in both cases without reaching
either top ten.

### What the cells do

The overlap itself is pure numpy over the cached attribution, so the table prints with no model and
no recomputation. `POS_IN` inverts each `ORDER` once, so a neuron's rank in either case is an O(1)
lookup rather than a scan over 37888 entries.

The cached block then measures two things for that set:

- **Receptive fields** over the same dense $(y_1,y_2)$ grid the top-10 profile uses (`RF_STEP`), so
  the panels are directly comparable with §17's, each titled with its rank in both cases.
- **IIA and position recovery with exactly that set frozen** under the joint patch, in both cases,
  against three same-size controls: **that case's own top-$n$** (is the shared set as good as a
  case-tuned one of equal size?), and **two random sets of $n$** (the null). The no-freeze and
  all-neuron values are drawn as reference lines from the already-cached grid and top-$k$ results
  rather than recomputed.

Figures: `figs/comparatorfig_2digit_shared_receptive_fields.png`,
`figs/comparatorfig_2digit_shared_freeze_iia.png`.
