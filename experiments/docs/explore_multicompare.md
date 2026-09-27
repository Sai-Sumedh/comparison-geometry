# `explore_multicompare.ipynb` + `data_multicompare.py`

The three-number version of the comparator analysis: `max(n1, n2, n3)` on
`Qwen/Qwen2.5-7B-Instruct`, one-shot prompt, 2-digit operands.

The two-number work is spread over three data modules and three notebooks
(`data_numberreps.py` → `makefig_numberreps`, `data_sharedrep.py` → `makefig_sharedreps`,
`data_comparator.py` → `explore_comparatorfig` / `makefig_comparator`). This is the three-number
counterpart of all of it, in one notebook, structured like `explore_comparatorfig.ipynb`: every
expensive block writes one `.npz` and draws its figure immediately, so results appear as they are
produced and a re-run skips everything already on disk.

**Nothing here is reimplemented.** The hooks, the interchange interventions, the metrics, the DAS
parametrization and the neuron-level machinery are all imported from the two-number modules, so
the three-number numbers sit on exactly the same measurement code. What `data_multicompare.py`
adds is only what was hard-coded to two numbers.

---

## 1. Task, prompt and token layout

Prompt (`data_multicompare.prompt_gen`), from `e15_multi_compare`:

```
Answer in the following format with a single answer. The maximum of 12, 437 and 5 is 437. The maximum of 89, 67 and 34 is
```

The one-shot example is fixed at three operands spanning one, two and three digits, so example
length never co-varies with anything being measured. All question operands are 2-digit, so every
prompt in the regime tokenizes identically — asserted in `make_batch`, not assumed.

* `num_char_spans` scans forward from past the **last** `ANCHOR = 'The maximum of '`, advancing
  its cursor past each match, so the one-shot example can never be matched and a digit string
  repeated across slots is matched once per slot in order.
* `leading_digits_distinct` — Qwen splits numbers into digits, so a number's *first* token is its
  leading digit. Every readout here is a first token, which is only unambiguous when no two
  numbers share one.
* `regime_positions` returns `(T, POS, NTOK, reference prompt)`. `POS[i]` is the **last** token of
  number `i`: the position at which the whole number has been aggregated under causal attention,
  which is where all residual-stream geometry is read.

### Value orders

An order is the tuple of slot indices in descending value order (`order_key`), the convention of
`e15/algo_multi_compare.ipynb`. `(0, 2, 1)` reads $s_1 > s_3 > s_2$.

### The counterfactual (`make_batch`)

Each **quadruple** $v_0 > v_1 > v_2 > v_3$ supplies three operands plus the replacement.

| | layout |
|---|---|
| clean | $v_0, v_1, v_2$ placed so `order` is the descending value order; the model answers $v_0$ at slot `order[0]` |
| corrupted | $v_3$ replaces $v_0$ **in place** at slot `order[0]`; every other slot untouched |

Four values rather than three, because requiring the replacement to be below *all three* operands
is what makes the corrupted winner always the clean runner-up whatever the layout — so one pool of
quadruples serves every order, and the orders are compared on the same values.

Readout tokens (e15's `make_compute_metrics`, `data_numberreps.make_batch`):

| token | is | role |
|---|---|---|
| `tok_a` | first token of $v_0$ | the clean answer |
| `tok_b` | first token of $v_1$, at slot `order[1]` | the corrupted answer; shared between the two runs |
| `tok_nc` | first token of $v_3$, at slot `order[0]` | a **value-independent** readout of "slot `order[0]` wins" — the patch never supplies this value |
| `tok_slot` | first token of each of the corrupted prompt's three numbers | used by the 3-D sweep, which scores all three slots |

$\text{PLD} = \text{logit}[\texttt{tok\_nc}] - \text{logit}[\texttt{tok\_b}]$;
position recovery $= (\text{PLD}_{\text{patched}} - \text{PLD}_{\text{corr}}) /
(\text{PLD}_{\text{clean}} - \text{PLD}_{\text{corr}})$ (the ROME/IOI normalized-restoration
convention, Meng et al. 2022); IIA $= \mathbb{1}[\arg\max = \texttt{tok\_nc}]$ over the full
vocabulary (Geiger et al. 2021).

### The pool is filtered to solved cases

Three-number `max` is *not* solved perfectly (`e15/multicompare_accuracy.ipynb`), and position
recovery divides by $\text{PLD}_{\text{clean}} - \text{PLD}_{\text{corr}}$. On a case the model
gets wrong that denominator is small and the ratio explodes. `solved_mask` is e15's
`model_solves`, batched; a quadruple is kept only if it is solved in **every** order used below,
so all orders are measured on the same values and the comparison between them is paired. The
notebook draws `N_POOL = 1400` and asserts that at least `N_EVAL + N_TRAIN = 528` survive.

`eval_quads` (400) carry every reported number; `train_quads` (128, disjoint) fit the directions
and rank the neurons. Nothing is ever ranked and scored on the same prompts.

---

## 2. The two sets of value orders, and why there are two

**`HEAD_GROUPS`** — the three "top two" arrangements the per-head sweep compares, exactly e15's
groups A / B / C. Each is patched at its **own runner-up position**.

| group | order | clean order | patch site |
|---|---|---|---|
| `s1>s2` | `(0,1,2)` | $s_1>s_2>s_3$ | $n_2$ |
| `s1>s3` | `(0,2,1)` | $s_1>s_3>s_2$ | $n_3$ |
| `s2>s3` | `(1,2,0)` | $s_2>s_3>s_1$ | $n_3$ |

`s1>s2` vs `s1>s3` holds the max at $s_1$ and moves the runner-up, so a head that tracks the
runner-up's *absolute* position must change between them and a head that tracks "the runner-up,
wherever it is" must not. `s1>s3` vs `s2>s3` holds the runner-up at $s_3$ — identical patch site,
identical token — and moves the max, so any difference is attributable to where the max sits.

**`CASES`** — the perturbation cases everything from §3 onward uses, all patched at $n_3$.
`make_batch` only ever varies the winning slot's value, so **the case is the perturbed variable**.

| case | order | clean order | perturbs | corrupted winner |
|---|---|---|---|---|
| `y1` | `(0,2,1)` | $s_1>s_3>s_2$ | $y_1$ | $s_3$ |
| `y2` | `(1,2,0)` | $s_2>s_3>s_1$ | $y_2$ | $s_3$ |
| `y3` | `(2,1,0)` | $s_3>s_2>s_1$ | $y_3$ | $s_2$ |

`y1` and `y2` **are** the head groups `s1>s3` and `s2>s3`, so their batches are shared. `y3` puts
the max at $s_3$, the patch site itself — the direct analogue of the two-number `n2 larger` case,
where the max also sits at the patch position and the runner-up is elsewhere. Its runner-up is
fixed at $s_2$ so the case has one layout and stays comparable with the other two.

Each $\mathbf{v}_j$ should be causal in exactly its own case. That dissociation is the claim the
figures in §3 and §4 test.

---

## 3. The three directions at $n_3$

$n_3$ is where all three numbers are present: L14's copy heads have written the first two on top
of what L13 already holds there. This is the three-number version of the two-number $n_2$, where
$\mathbf{v}_1$ = H14 DAS and $\mathbf{v}_2$ = L13 PC1.

| | what | where | how |
|---|---|---|---|
| $\mathbf{v}_1$ | $y_1$, copied in from $n_1$ | inside L14.**H14**'s own output contribution | rank-1 DAS, fitted on case `y1` |
| $\mathbf{v}_2$ | $y_2$, copied in from $n_2$ | inside L14.**H18**'s own output contribution | rank-1 DAS, fitted on case `y2` |
| $\mathbf{v}_3$ | $y_3$, already in place | the L13 residual at $n_3$ | PC1 of the cloud — **not** fitted |

H14 moving $s_1 \to s_3$ and H18 moving $s_2 \to s_3$ is `e15/localize_multi_compare.ipynb` and
`e15/algo_multi_compare.ipynb`'s finding; §1 of the notebook re-measures it properly (400
quadruples per group rather than 5 single examples) rather than assuming it.

**DAS** (`train_head_das`) is Geiger et al. 2023's single-causal-variable rank-1 case, generalised
from `data_sharedrep.train_h14_das` to an arbitrary head so H14 and H18 are fitted by identical
code: QR retraction on the Stiefel manifold, counterfactual-pair cross-entropy objective (IIA is
an argmax and has no gradient), gradients accumulated over minibatches. The reachable search space
is only the 128-dimensional column space of that head's $W_O$ block — anything orthogonal leaves
the head's output untouched.

$\mathbf{v}_3$ is PC1 rather than a fit, mirroring the two-number convention where the number
already *at* the read position is taken as PC1 and only the transported one is fitted. The
notebook prints its correlation with $\log y_3$ and the rank-1 interchange's IIA, which is what
says PC1 is the right object.

**Signs.** A DAS subspace and a PCA component are both sign-arbitrary. Each direction is oriented
so its component rises with its own number, so the plotted coordinates are readable.

**The basis `PB`.** Gram-Schmidt (QR) on $(\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3)$, sign-fixed so
$\operatorname{diag}(P^\top V) > 0$. Axis 1 **is** $\mathbf{v}_1$; axis 2 is $\mathbf{v}_2$
orthogonalised against it; axis 3 is $\mathbf{v}_3$ orthogonalised against both. Same display
convention as `makefig_sharedreps` Fig 6, extended by one dimension, and the same array is used as
the rank-3 patch basis, so the scatter and the interchange are in one frame. **The true pairwise
angles are not readable off the 3-D scatter** — they are in the §4 cosine heatmap.

---

## 4. Block by block

Each block is `cached(name, compute)`: load `results/multicompare_{name}_{REGIME}.npz` if present,
else compute, save, return. `FORCE = {'sweep'}` recomputes one block; deleting its file does the
same. When all ten exist, `PLOT_ONLY` is true, the model is never loaded, and the notebook is a
few-second plotting pass.

| block | file | what it holds |
|---|---|---|
| `pool` | `multicompare_pool_*.npz` | the filtered quadruples, the solve rates, `T` and `POS` |
| `heads` | `..._heads_*` | per-head position recovery / IIA, three groups, plus L13 and L14 baselines |
| `dirs` | `..._dirs_*` | the cloud, $\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3$, the DAS loss curves, the three probes |
| `space` | `..._space_*` | every interchange condition at $n_3$, both metrics, all three cases |
| `sweep` | `..._sweep_*` | the $9^3$ dose-response cube |
| `freeze` | `..._freeze_*` | MLP / attention freezes at L14–L16 under the rank-3 patch |
| `attr` | `..._attr_*` | per-neuron attribution, the top-$k$ curves, the consensus set |
| `rf` | `..._rf_*` | the dense $(y_1,y_2,y_3)$ activation cube |
| `connect` | `..._connect_*` | MLP14 → MLP15 causal edges and weight paths |
| `transfer` | `..._transfer_*` | the two-number neurons re-measured here |
| `heads_alone` | `..._heads_alone_*` | the per-head sweep with no L13 co-patch |
| `das15` | `..._das15_*` | L15 DAS (rank 1, 2), the L15 PCA, and MLP15 write alignment |
| `ucos` | `..._ucos_*` | u1, u2 at the L13 residual, and their IIA |

### §1 — which L14 head moves which number

Each L14 head's own additive contribution ($z_h W_O^{h\top}$, read off the `o_proj` input) is
interchanged at the group's runner-up position, co-patched with the **full clean L13 residual**
there. The L13 co-patch is what makes single-head effects visible at all: on its own a head adds
little on top of a residual stream that is still corrupted. Baselines are L13 alone (dashed) and
the full L14 residual.

Figure: three bars per head, one per group, on an $x$-axis fixed to the `s1>s3` ranking so a head
that only matters in another order shows up out of order. Both metrics, sharing the $x$-axis.

**The `heads_alone` variant.** The same sweep with the L13 co-patch removed — each head's output
interchanged and nothing else. The two figures answer different questions: the co-patched one asks
what a head *adds on top of* a restored L13 residual, the alone one asks what it does by itself.
The reference line changes accordingly, from the L13-alone baseline to the **full L14 residual
patch** at the same position (`base_*_L14_*`, which the `heads` block already measures without a
co-patch) — the ceiling for everything layer-14 attention can do unaided. The cell prints both
numbers side by side for the heads that rank top-3 in either version, which is the direct read of
what the co-patch buys.

### §2–3 — the directions, and patching the space

The IIA-by-number figure follows `makefig_numberreps` panel (f): one group per number, and inside
a group the full-scope control at that site against the rank-1 direction.

The space-patching figure is `makefig_sharedreps` panel (e) extended to three numbers. Conditions:
each $\mathbf{v}_j$ alone; all three as separate rank-1 patches at their own sites; the rank-3
span at the pre-MLP L14 residual; the same span post-MLP; and the full-rank ceilings (L13 full,
L14 full pre-MLP, L14 full post-MLP) at the same position. Hatching marks a joint / rank-restricted
scope; colour is which number the counterfactual perturbs.

The claim the panel exists to make is the **increment**: no single direction should move the answer
except in its own case, and the rank-3 span should reach the full-rank ceiling at the same site.

**Why the pre-MLP site is the headline one.** Same reason as `data_comparator.py`: the rank-1
triple sets three upstream sources whose effects then propagate through L14 attention (which can
rewrite $n_3$, and which also changes what the heads themselves read). The rank-3 pre-MLP patch
sets the space coordinate directly, at the consumer's input.

### §4 — angles and probes

Cosine heatmap over $\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3$ and three ridge log-magnitude probes
$\mathbf{p}_1,\mathbf{p}_2,\mathbf{p}_3$ fitted on the pre-MLP L14 cloud at $n_3$. The three
$\mathbf{v}$ pairs are boxed. Chance is $1/\sqrt{d_\text{model}} \approx 0.017$; the angles are
printed as well as the cosines, because $-0.08$ reads as "a small number" while $95°$ reads as a
right angle.

Probes follow Alain & Bengio 2016 (linear probing), ridge because the target is continuous, with
`RidgeCV` choosing $\alpha$ by leave-one-out CV on the training split so the regularisation is not
an arbitrary constant. The held-out $R^2$ is what says the quantity is linearly present at all.
The right panel is the component of the cloud along each probe against that slot's number.

### §5 — probability contours over the space

`sweep_space` is the three-number version of `data_sharedrep.sweep_uv_plane`. All three
coordinates are **set** to the cloud-mean coordinate of a chosen $(y_1^*, y_2^*, y_3^*)$ over a
$9^3$ grid (one value per leading digit, mid-decade), and the answered token is recorded over
`SWEEP_N` prompts per cell.

**What is scored, and why it is not the implanted numbers.** The model only ever emits a number
that is in its prompt, so "did it say $y_1^*$?" is unanswerable — $t(y_1^*)$ never wins. What the
implanted coordinates can decide is **which slot** the comparison picks, which is exactly the
mechanism's claim. Each cell is scored against the prompt's own three slot tokens (`tok_slot`),
giving three probability fields $P_1, P_2, P_3$. `PCOND` renormalises by the fraction that named
*any* of the three, so a cell where the model went off-prompt does not read as a weak preference.

Cells with a repeated coordinate sit on a boundary with no predicted winner; they are skipped and
masked out (the white bands in the slice panels).

This is the interchange-intervention logic of Geiger et al. 2021/2023 run as a dose-response
surface rather than at a single counterfactual point — the same move as a causal-scrubbing style
sweep over the hypothesised variable's whole range.

**The figure.** A row of $(y_1^*, y_2^*)$ slices at fixed $y_3^*$: the fill is the argmax slot,
the lines are each field's $P = 0.5$ contour. The right panel is the surface
$y_3^*(y_1^*, y_2^*)$ at which slot 3 takes over — a height field rather than a true isosurface,
which is what an isosurface of a monotone field reduces to and needs no `skimage` (not installed
in the `interp` env). The `imshow` extent is set to cell **edges** while the contours use the grid
as cell **centres**, so the two overlay correctly.

The printed check is how often the named slot matches the largest *implanted* number (chance
$1/3$).

### §6 — module freezes

A sub-block is **frozen** by pinning its output at $n_3$ to the value it takes on the unpatched
corrupted run, so the rank-3 patch cannot reach the rest of the network through it:
$\text{out}_\ell[n_3] := \text{out}^{\text{corr}}_\ell[n_3]$. MLP and attention at L14–L16
individually, and MLP14+MLP15 together. Dashed line is that case unfrozen.

### §7 — attribution and top-$k$ freezing

$s_i \approx (a_i^{\text{corr}} - a_i^{\text{patched}})\, \partial\text{PLD}/\partial a_i$,
evaluated **at the patched state** (which is why the base run is the rank-3 patch, the regime the
freeze happens in). A neuron is the post-SwiGLU scalar feeding `down_proj`, $d_{ff} = 18944$ per
layer, pooled over L14 and L15 (37888 candidates). The gradient is taken w.r.t. a zero handle
added to the `down_proj` input — the model's parameters are frozen, so that handle is what creates
a graph from the activation to the logits. Large **negative** score = freezing that neuron
destroys the effect. Attribution patching: Nanda 2023; Syed et al. 2023 (EAP); Kramár et al. 2024
(AtP\*).

Verified by freezing the top $k$ against bottom-$k$ and two random-$k$ draws, on the 400 eval
quadruples, never on the 128 the ranking used.

**The consensus set.** The two-number analysis used the *intersection* of the two cases' top-20.
With three cases a strict three-way top-20 intersection is expected to be empty by chance
($20^3/37888^2 \approx 6\times10^{-6}$), so it is computed and **reported** but not used. The set
the freeze test uses is the 12 neurons with the best **worst-case** rank across the three
perturbations — i.e. the neurons that matter whichever number moved. 12 matches the size of the
two-number set, so §10's comparison is like for like.

### §8 — receptive fields

Post-SwiGLU activation over the full $23^3$ cube of $(y_1, y_2, y_3)$ prompts at stride
`RF_STEP = 4` over $y \in [11, 99]$. Every cell is **one deterministic forward pass on one
prompt** — not an average over repeats — so structure is real per-number variation, not sampling
noise. The leading-digits-distinct guard is deliberately not applied: nothing here reads an answer
token, only activations, so the cube is complete. Left blocky
(`interpolation='nearest'`) for the same reason as `makefig_comparator` panels (b)/(c).

Shown as the **three pairwise marginals** — $(y_1,y_2)$, $(y_1,y_3)$, $(y_2,y_3)$, each averaged
over the number not on its axes. One symmetric colour scale per neuron, taken on the marginals
rather than on the cube: averaging out a number shrinks the range, and a per-panel scale would
hide which pair the neuron actually reads. The dotted diagonal is equality of the two plotted
numbers. `draw_marginals` lays each neuron out as a block of three panels separated by a gap
column, with the neuron's name centred over its own block.

The trade-off of marginals over slices: a three-way interaction averages away. If one is
suspected, slice the cube directly — `rf_cube(pooled)` returns the full $(G,G,G)$ array.

### §9 — MLP14 → MLP15 connections

Two measures of the same edge, as in `data_comparator.py` §11:

* **causal edge** — one MLP14 neuron set to its clean value at $n_3$, read as the shift in each
  MLP15 neuron, in units of that neuron's cloud SD (so an edge is in units the neuron actually
  varies over, which accounts for whether the upstream neuron ever fires);
* **weight path** — the "virtual weight" of Elhage et al. 2021; for an MLP → MLP path through the
  residual stream it is a plain inner product of the write column with the $\gamma$-folded read
  row.

The figure pairs the two heatmaps, then draws the receptive fields of the four MLP14 neurons
feeding the most strongly driven MLP15 target, with that target's own field alongside.

### §10 — the two-number neurons, here

The 12 neurons `makefig_comparator` identified as shared between its two cases
(`overlap_k20` in `results/comparator_data_2digit.npz`; 6 in MLP14, 6 in MLP15). The pooling
convention is identical — layers `[14, 15]`, $d_{ff} = 18944$, so pooled index $p$ means
L14#$p$ for $p < 18944$ and L15#$(p - 18944)$ above — so the indices transfer directly.

Two measurements:

1. **their fields over three numbers** — the same pairwise marginals, so the two-number panel and
   this one are the same object with a third number added;
2. **freezing them under the three-number rank-3 span patch**, in all three cases, against two
   random 12-neuron draws and the unfrozen reference.

Plus their rank under the three-number attribution in each case, printed as a table. A neuron that
carried the two-number comparison and still ranks high here is doing the same job; one that drops
to rank ~20000 is not.

If `results/comparator_data_2digit.npz` is absent, `TWO` is `None`, `TRANSFER` is empty and this
whole section is skipped rather than failing.

### §11 — L15, the comparison outcome

Section 3's space carries the three *numbers*; this asks what the layer above carries. Rank-1 and
rank-2 DAS on the residual leaving L15 at $n_3$ (`MC.train_resid_das`, same QR retraction and
counterfactual-pair objective as every other fit here), plus the 3-component PCA of the same cloud.
Every projection is coloured by **which slot holds the maximum**, which is the variable L15 is
expected to hold.

The cloud is regenerated with `sample_cloud_triples(N_CLOUD, SEED + 11, ...)` — deterministic, so
it is the same triples as the section-2 cloud, asserted rather than assumed. Only the L15 residual
is captured (`MC.capture_resid_cloud`), so the cached `dirs` block is never touched.

**Cross-evaluation.** A subspace is fitted on one perturbation case and scored on all three,
giving a `fit × eval` IIA matrix per rank. That is the test of whether L15 holds *one shared*
"which slot won" variable rather than three per-number ones: if it does, the off-diagonal holds
up. The full-rank patch at the same site is the ceiling.

`src_with_L15` adds the L15 clean residual into section 3's already-captured sources dict in place,
so switching the patch site up a layer costs one extra capture per order rather than a full
re-capture.

**Cost.** `len(DAS15_CASES) × len(DAS15_RANKS)` DAS fits — six by default, i.e. three times the
`dirs` block. `DAS15_CASES = ['y1']` cuts it to two.

**Separability, printed.** A 3-way logistic regression on the rank-2 DAS coordinates against the
top-2 and top-3 PCs, held-out accuracy, chance $1/3$. That is the number the colours are making
visually; DAS beating PCA at the same dimensionality is the claim.

#### MLP15 write alignment

$\rho_j = \lVert Q^\top w_j\rVert / \lVert w_j\rVert$, with $w_j$ the neuron's `down_proj` column.
Neuron $j$ adds $a_j w_j$ to the residual and $Q$ spans a subspace of that same residual, so the
two are directly comparable — no change of basis and no $\gamma$-folding, unlike the read side.
Computed by `data_comparator.das15_write_alignment` for **every** MLP15 neuron, so the profiled
ones sit in their own empirical null; the analytic $\sqrt{r/d}$ level is not trustworthy because
`down_proj` columns are not isotropic.

The top `N_ALIGN_SHOW` neurons are picked **from the attribution-profiled set only**, as asked —
so they already have receptive-field cubes from the `rf` block and no extra model time is needed
to draw them. The table prints each profiled L15 neuron's $\rho$, its percentile in the
18944-neuron null, and its attribution rank in each case, so alignment and causal importance can
be read against each other.

### §12 — $\mathbf{u}_1, \mathbf{u}_2$, and all five directions

$\mathbf{u}_j$ = rank-1 DAS on the **L13 residual at $n_j$**, fitted on case `yj`: the $j$-th
number at its own position, before a layer-14 head transports it. This is the three-number version
of `data_numberreps`' L13 DAS direction (what `data_sharedrep` calls $\mathbf{u}$).

Only `u1` and `u2` are fitted, because those are the two numbers that get transported.
$\mathbf{v}_3$ is *already* the $y_3$ counterpart of a $\mathbf{u}$ — L13 read at $n_3$, the number
at its own position — except that it is PC1 rather than a DAS fit. A `u3` would be one more call
to `train_resid_das` at `POS[2]` if the symmetry is wanted.

**The heatmap.** All five directions, lower triangle, with $(\mathbf{u}_1,\mathbf{v}_1)$ and
$(\mathbf{u}_2,\mathbf{v}_2)$ boxed: those are the same number before and after transport. Angles
and the chance level $1/\sqrt{d_\text{model}}$ are printed, since a cosine near zero reads as "a
small number" while $90°$ reads as a right angle.

Comparing directions read at *different token positions* is deliberate and is what
`data_sharedrep`'s own cosine matrix does — they live in the same residual basis, so the inner
product is defined; what it measures is whether the head writes its copy along the same direction
the source position used.

**The side panel** is each $\mathbf{u}_j$'s IIA across all three cases, against the rank-3 span at
$n_3$. Each $\mathbf{u}$ should be live only in its own case: in the other two the number sitting
at $n_j$ is identical between clean and corrupted, so patching there is a no-op and IIA should be
~0. That is the control that says the fit found $y_j$ and not something positional.


---

### §13 — layers 18–22 at the read-out token

Sections 1–12 all read a *number* position ($n_1$, $n_2$, $n_3$), where the question is which
operand a component carries. This section reads the **last prompt token** instead — the position
the answer is emitted from — and asks the different question of when, in depth, the model's
decision about *which number wins* becomes the thing that determines the output.

Three value orders, the `HEAD_GROUPS` triple (`s1>s2>s3`, `s1>s3>s2`, `s2>s3>s1`), each with the
batch §1 already uses. The counterfactual is unchanged: the clean maximum is replaced in place by
a value below all three, so the corrupted run answers the clean runner-up, and a successful
interchange moves the argmax from `tok_b` back to `tok_nc`. Both metrics are the same
`posrec_and_iia` used everywhere else, so the numbers sit on the same scale as §1–§12.

`LAST_POS = T - 1`. The prompt ends `... is `, so the final token is the read-out site and it is
the same index in every prompt of a regime (the token layout is asserted in `make_batch`).

**Block `late`** (`results/multicompare_late_2digit.npz`), two things in one file:

* *Clouds.* `MC.capture_resid_cloud(trips, T, [(l, LAST_POS) for l in LATE_LAYERS])` reads the
  residual leaving each of L18–L22 at the read-out token, over the **same** `N_CLOUD = 1500`
  triples §2 uses (`sample_cloud_triples(N_CLOUD, SEED + 11, ...)`), in one forward pass per
  minibatch so every layer's row describes the same prompt. Each layer is then centred and given
  its own 3-component PCA; `pca_coords_L{l}` (1500, 3) and `pca_evr_L{l}` are stored. The figure
  draws PC1–PC3 (3-D) above PC1–PC2 (2-D) for each layer, coloured by $\max(y_1,y_2,y_3)$. The
  colouring is the point: the triples are sampled independently per slot, so "the maximum" is not
  a function of any single slot, and a cloud that organises by it is carrying the *comparison
  outcome's value*, not a copy of one operand.
* *Full-rank residual interchange per layer.* For each layer and order,
  `eval_spec(dict(resid=[(l, 'full', None)]), bat, {l: clean_hidden}, LAST_POS)` — the whole
  residual leaving layer `l` at the read-out token replaced by its clean-run value. The clean
  capture is `S.clean_hidden_at`, a full $(n, T, d_{\text{model}})$ tensor, so layers are looped
  one at a time and the capture is freed before the next one; only one is ever resident.

The sweep figure plots IIA and position recovery against layer, one line per order. It then prints
the mean IIA over the three orders per layer and the step between consecutive layers, and sets

    JUMP_LAYER = LATE_LAYERS[argmax(diff(mean IIA)) + 1]

— the layer the largest increase lands *in*. This is deterministic given the cached `.npz`, so a
plot-only re-run picks the same layer as the compute run did.

**Block `latehead`** (`results/multicompare_latehead_2digit.npz`). At `HEAD_LAYER` (= `JUMP_LAYER`
unless overridden by hand in that cell), each head's own output contribution at the read-out token
is interchanged **alone** — no co-patch of the residual entering the layer, so the bar is what that
head does by itself, the same convention as §1's "no L13 co-patch" sweep. The ceiling drawn as a
dashed line per order is the whole attention block's output at the same layer and token, patched
with `C.capture_sites` + `C.make_out_patch_hook`; it is the matched ceiling, because patching heads
can only ever move the attention sublayer's contribution, not the layer's MLP or its skip.

`head_layer` is stored in the `.npz` and asserted against `HEAD_LAYER` on load, so a cached sweep
from one layer cannot be silently plotted as another.

#### New machinery in `data_multicompare.py`

`data_sharedrep`'s per-head functions close over module globals (`attn`, `W_O`) that
`init_head_machinery()` pins to `H_LAYER = 14`. §1–§12 never need another layer; §13 does, so four
functions repeat the same three operations with the layer as an argument. The decomposition is
unchanged — `o_proj`'s output is the sum over heads of $z_h W_O^h$ (Elhage et al. 2021), so
rewriting one head is exactly `out += target_h - current_h` at that position, and the `o_proj` bias
is left alone because it is not attributable to any one head.

* `w_o_of(layer)` — that layer's `o_proj` weight as float32, cached in `_WO_CACHE` so the
  per-head loop does not re-read it 28 times.
* `head_out_at(ctx, head, layer)` — slices the head-major `o_proj` input to head `head`'s
  `head_dim` block and maps it through that head's $W_O$ block into residual space.
* `capture_ctx_at(ids, pos, layer)` — the $(B, d_{\text{model}})$ `o_proj` input at `pos`, in
  minibatches. The *context* is stored rather than the per-head outputs, so one capture serves all
  28 head conditions of that layer.
* `head_patch_handles_at(layer, head, pos, target, P=None)` — `S.head_patch_handles` with the
  layer as an argument: a pre-hook stashing the `o_proj` input and a post-hook adding
  `target - current` (or its projection into `P`) at `pos`. Returns handles; the caller removes
  them.

The head conditions are driven through `eval_spec(None, bat, {}, LAST_POS, extra=...)`: the spec
language in `build_handles` only knows about `H_LAYER` heads, and `extra` is exactly the escape
hatch for conditions it does not cover, so no change to `build_handles` was needed.

#### Cost

`late`: one cloud pass (1500 prompts / `CLOUD_BS = 48` ≈ 32 forwards) plus
5 layers × 3 orders × (1 clean capture + 400 patched forwards). `latehead`:
3 orders × (28 heads + 1 attention-block condition) × 400 forwards, plus one context and one
attention-output capture per order. Both are forward-only — no DAS fit, no backward — so together
they are a few minutes on one A100 and far below §2's or §7's cost.


## 5. Cost and memory

Rough, at `N_EVAL = 400`, `T ≈ 45`, on one A100-class GPU. The two DAS fits dominate.

| block | scale | note |
|---|---|---|
| `pool` | ~1400 × 4 orders × 2 runs | forward only |
| `heads` | 3 groups × 30 conditions × 400 | forward only |
| `dirs` | 2 × 100 steps × 8 minibatches, fwd+bwd | **the expensive one** |
| `space` | 3 cases × 15 conditions × 400 | forward only |
| `sweep` | 504 cells × 24 prompts | forward only |
| `attr` | 3 × 8 minibatches fwd+bwd, then 3 × 32 freeze conditions × 400 | |
| `rf` | $23^3 = 12167$ prompts | forward only |
| `connect` | 3 cases × $n_{14}$ neurons × 400 | forward only |
| `heads_alone` | 3 groups × 28 heads × 400 | forward only |
| `das15` | 6 DAS fits, then 18 cross-eval conditions × 400 | **3× the `dirs` block** |
| `ucos` | 2 DAS fits, then 6 conditions × 400 | |

`SRC` caches `(400, T, d_model)` fp16 residuals — ~129 MB per layer per order — and is keyed by
`(order, pos, layers)`; four entries × two layers is ~1 GB on top of the ~15 GB of fp16 weights.
`FRZ` and `NEUR` add ~300 MB. Batch sizes: `EVAL_BS = 48` for forwards (bounded by the
$B \times T \times |\text{vocab}|$ logits tensor, not by the activations), `DAS_BS = 16` and
`ATP_BS = 16` for anything with a backward.

## 6. Reproducing

```
cd experiments
# run explore_multicompare.ipynb top to bottom on a GPU node
```

First run computes and saves all fifteen `.npz`; later runs load them and never touch the model. To
recompute one block, add its name to `FORCE` or delete its file.

`results/comparator_data_2digit.npz` is optional (only §10 needs it).

## 7. References

* Interchange interventions, IIA, DAS — Geiger et al. (2021), *Causal abstractions of neural
  networks*; Geiger et al. (2023), *Finding alignments between interpretable causal variables and
  distributed neural representations*.
* Position-recovery normalisation — Meng et al. (2022), *Locating and editing factual associations
  in GPT* (the ROME/IOI convention).
* Attribution patching — Nanda (2023); Syed et al. (2023), *Attribution patching outperforms
  automated circuit discovery*; Kramár et al. (2024), *AtP\**.
* Linear probes — Alain & Bengio (2016), *Understanding intermediate layers using linear
  classifier probes*.
* Virtual weights / composition between components — Elhage et al. (2021), *A mathematical
  framework for transformer circuits*.
* Task, prompt, value orders and the head-sweep design — `e15_multi_compare`
  (`algo_multi_compare.ipynb`, `localize_multi_compare.ipynb`, `order_patching.ipynb`).
