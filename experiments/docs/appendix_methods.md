# `paper/appendix_methods.tex` + `paper/appendix_methods.bib`

LaTeX appendix covering the experimental setup and methods for the paper. Two top-level sections
(`\section{Experimental Setup}`, `\section{Methods}`), written to be pasted directly into Overleaf.

```
paper/appendix_methods.tex   ~7,500 words, 12 pages standalone at 1in margins
paper/appendix_methods.bib   17 entries; every field verified against primary sources (see below)
```

## Preamble requirements

```latex
\usepackage{amsmath, amssymb, booktabs, multirow, bm}
\usepackage[numbers]{natbib}   % or any package providing \citep / \citet
```

`\bm` is used for the direction and activation symbols; substitute `\mathbf` throughout if the
paper's preamble does not load `bm`. `multirow` is used only in the hyperparameter table.

Verified to compile with `pdflatex` (12 pages, 4 overfull hboxes, all under 5.5pt; two of them are
the verbatim prompt strings, which will reflow under different margins). All 17 bib keys resolve,
none is unused, and `bibtex` runs warning-free.

## Bibliography verification

The `.bib` was first written from memory, then checked **entry by entry, field by field** against
primary sources (arXiv API and abstract pages, official proceedings pages, publisher records, the
publication's own site). **Three fields were fabricated and have been corrected.** Every remaining
field is confirmed against a primary source; nothing is left on trust.

### Corrected (were wrong)

| entry | field | was | is |
|---|---|---|---|
| `kramar2024atpstar` | venue | ICML 2024 | arXiv-only — retyped `@article`, arXiv:2403.00745 |
| `syed2023eap` | venue, year | BlackboxNLP 2024 | NeurIPS 2023 ATTRIB Workshop |
| `nanda2023atp` | URL | `neelnanda.io/attribution-patching` (does not exist) | `www.neelnanda.io/mechanistic-interpretability/attribution-patching` |

### Confirmed, with the source used

| entry | fields confirmed | source |
|---|---|---|
| `qwen25` | arXiv ID, title, first-10 author order, year | arXiv abs/2412.15115 |
| `geiger2021causal` | authors, venue, vol 34, pp. 9574--9586 | NeurIPS 2021 proceedings |
| `geiger2023das` | authors, CLeaR 2024, PMLR v236, pp. 160--187 | proceedings.mlr.press/v236 |
| `chan2022scrubbing` | 8-author list and order, year, URL | AI Alignment Forum post |
| `meng2022rome` | authors, venue, vol 35, pp. 17359--17372 | NeurIPS 2022 proceedings |
| `vig2020causal` | 7-author list, venue, vol 33, pp. 12388--12401 | NeurIPS 2020 proceedings |
| `wang2023ioi` | authors, ICLR 2023 | iclr.cc virtual poster page |
| `goldowskydill2023pathpatching` | arXiv ID, title, 4-author list, year | arXiv API |
| `nanda2023atp` | title, year, URL | neelnanda.io |
| `syed2023eap` | authors, arXiv ID, venue (Comments field) | arXiv abs/2310.10348 |
| `kramar2024atpstar` | authors, arXiv ID, year, absence of a venue | arXiv abs/2403.00745 |
| `conmy2023acdc` | 5-author list, NeurIPS 2023, arXiv ID | NeurIPS 2023 proceedings |
| `elhage2021framework` | title, full author order, venue, year, URL | transformer-circuits.pub |
| `alain2016probes` | arXiv ID, title, authors, year | arXiv API |
| `belinkov2022survey` | *Comp. Ling.* 48(1), pp. 207--219, DOI | ACL Anthology 2022.cl-1.7 |
| `gurnee2023space` | authors, ICLR 2024, arXiv ID | ICLR 2024 proceedings |
| `brown2001interval` | *Stat. Sci.* 16(2), pp. 101--133, DOI | Project Euclid |

### Deliberate omission

`conmy2023acdc` carries **no page range**. The only source offering one (`16318--16352`) was an
auto-generated BibTeX whose editor list was visibly corrupted, and both ACM DL (403) and the
Semantic Scholar API (429) were unreachable for an independent check. Omitting `pages` from a
NeurIPS entry is standard and cannot be wrong; inventing one can. Revisit only if a primary source
becomes available.

`vig2020causal` likewise carries no arXiv note, because its arXiv ID was never independently seen.

### What went wrong the first time

The initial check — "all keys resolve, none unused, `bibtex` runs" — is a **consistency** check. It
passed while three fields were fabricated, because a fabricated venue is internally consistent. A
bibliography written from memory needs field-level verification against primary sources, and the
compile check says nothing about it.

## Prose conventions

The appendix is written to read as ordinary academic prose. These constraints are deliberate and
should be preserved when editing, because they were applied as a pass over an earlier draft that
violated all of them.

- **No em-dash asides in the body text.** Earlier drafts used 102 of them. Subordinate the clause
  with `because`, `since`, `so that`, `although`, or split it into its own sentence.
- **No semicolons in the body text.** Earlier drafts used 73. Every remaining `;` in the file is a
  `\;` math-spacing command inside an equation.
- **No "X is what makes/says/rules out Y."** This construction appeared 9 times and reads as a tic
  when repeated. Say the thing directly: "A near-zero score rules out ..." rather than "A near-zero
  score is what rules out ...".
- **No meta-commentary about the argument's own structure.** Sentences like "Two kinds of direction
  appear, and the distinction is the organizing one for the whole analysis" or "The observation that
  dictates every metric below is ..." were removed. State the content; let the reader see the
  structure.
- **Avoid repeated "N things follow" announcements.** One or two are natural in a methods section,
  but the earlier draft opened seven paragraphs this way. Most were dissolved into the first item.
- **Complete sentences that finish a thought.** Target length is roughly 25--35 words; the current
  file averages 28.7 words per sentence with a median of 29. Avoid short punchy fragments for
  emphasis.

Quick check after editing:

```bash
grep -c -- '---' <(grep -v '^%' appendix_methods.tex)   # expect 0
grep -c 'is what' appendix_methods.tex                   # expect 0
```

## Notation contract

The appendix commits to one notation and never departs from it. Anything internal to the codebase
(`tok_nc`, `n1 larger`, `POS`, `o_proj`, array names, block names) is absent by design.

| symbol | meaning |
|---|---|
| $y_j$ | the $j$-th number (operand) of the question |
| "$y_j$'s position" | the token of $y_j$'s last digit — the readout site |
| $t(\cdot)$ | the first token (leading digit) of a number |
| $\bm{x}$ | a model activation; $\bm{x}_\ell$ = residual stream leaving decoder block $\ell$ |
| $\bm{u}$, $\bm{u}_j$ | rank-1 direction in a number's **own** representation space, at its own position |
| $\bm{v}_j$ | rank-1 direction carrying $y_j$ in the **shared** representation space |
| $a, b, c, r$ | the sampled values of a counterfactual tuple; $r$ is the replacement |

Value orders are written as inequality chains over operands ($y_1 > y_3 > y_2$), never as slot
indices. Positions are named in prose ("$y_3$'s position"), never as $n_j$.

## Section map, and where each claim came from

### Experimental Setup

| subsection | source |
|---|---|
| Model and compute | `data_numberreps.py` config + `load_model`; job resources (1 GPU, 100 GB host RAM, 8 cores); batching rationale from `docs/data_numberreps.md` §"Evaluation batching" |
| Task and prompt | `data_numberreps.prompt_gen`, `data_multicompare.prompt_gen`; exemplar rationale from `e15/docs/multicompare_accuracy.md` |
| Token layout | `regime_positions`, `num_char_spans`, `leading_digits_distinct`; realized layouts from `results/meta.json` and `e15/docs/algo_multi_compare.md` |
| Digit regimes | `data_numberreps.REGIMES` |
| Counterfactual pairs, two operands | `build_triples` / `make_batch`; **the "patch moves the position, not the value" paragraph** is the motivation for `tok_nc` as stated across `docs/data_numberreps.md` §5, `e15/docs/order_patching.md` and `e15/docs/algo_multi_compare.md`; the 99.5% on-prompt figure is from `docs/data_sharedrep.md` §6b |
| Counterfactual pairs, three operands | `data_multicompare.build_quads` / `make_batch`, `HEAD_GROUPS`, `CASES` |
| Sampling, filtering, splits | `solved_mask` + pool assertion in `explore_multicompare.ipynb` cell 9; split rationale from `docs/data_numberreps.md` §3 |
| Behavioural baseline | `e15/docs/multicompare_accuracy.md` in full |
| Hyperparameter table | configs of `data_numberreps.py`, `data_sharedrep.py`, `data_comparator.py`, `data_multicompare.py`, and `explore_multicompare.ipynb` |

### Methods

| subsection | source |
|---|---|
| Interchange interventions | `make_subspace_patch_hook` / `make_full_patch_hook` |
| Intervention sites (i)–(iv) | `head_out` / `head_patch_handles` / `_premlp_hooks` in `data_sharedrep.py`; neuron machinery in `data_comparator.py` |
| Metrics | `posrec_and_iia`, `pos_anchors`; `make_compute_metrics` in `functions/comparison_utils.py` for the value-recovery and indirect-effect variants |
| Distributed alignment search | `DASSubspace`, `cf_pair_loss`, `train_das_1d`, `train_h14_das`, `train_head_das`; the $n{=}16$ overfitting figure is from `docs/data_numberreps.md` §3 |
| The directions and their frames | `docs/makefig_sharedreps.md` §"Notation" and Fig 6; `data_multicompare` basis construction |
| Unsupervised baselines | `pca_and_probes`; `e11/docs/compare_geometry.md` for the probe conventions |
| Localization sweeps | `causal_trace_sweep3` and the co-patch rationale in `e15/docs/algo_multi_compare.md` Parts 1–3 |
| Attribution patching | `data_comparator.py` §3; consensus-set rationale from `docs/explore_multicompare.md` §7 |
| Freezing and injection | `make_out_patch_hook`, `make_neuron_freeze_hook` |
| Dose-response sweeps | `sweep_uv_plane`, `sweep_space`; scoring rationale from `docs/data_sharedrep.md` §6b |
| Receptive fields | `data_comparator.py` §8, `docs/explore_multicompare.md` §8 |
| Connectivity | `data_comparator.py` §11, `das15_write_alignment` |
| Nulls and controls | `docs/data_comparator.md` §2 "Nulls everywhere"; specificity framing from `docs/data_sharedrep.md` §6c |

## Numbers quoted in the text

Every figure quoted in the appendix was read off a results file or a config, not recalled:

| claim | value | source |
|---|---|---|
| seed-stability mean $\lvert\cos\rvert$ / IIA spread | 0.99 / 0.0 | `results/meta.json` → `das_seed_cos_absmean`, `das_seed_iia_std` |
| DAS at $n{=}16$: train vs held-out | 1.000 / 0.688 | `docs/data_numberreps.md` §3 |
| $\cos(\bm{v}_1, \bm{v}_2)$ | $-0.08$ | `docs/data_sharedrep.md` §8 |
| per-number-mean $R^2$, pre-MLP vs post-MLP | 0.87 / 0.35 | `docs/makefig_sharedreps.md` Fig 6 table |
| outputs that name an in-prompt operand | 99.5% | `docs/data_sharedrep.md` §6b |
| head-space reachable fraction chance level | 0.19 | $\sqrt{128/3584}$ |
| three-way top-20 intersection chance | $6\times10^{-6}$ | $20^3/37888^2$ |

**Note on `DAS_N_SEEDS`.** `docs/data_numberreps.md` says 5 random initializations; the config and
the executed run (`results/meta.json` → `das_n_seeds`) both say 2. The appendix reports **2**, and
the hyperparameter table says "random initializations (stability check): 2". The doc is stale on
this point.

## Deliberate omissions

Things present in the source docs but left out, because they are implementation detail rather than
method:

- Result-file layout, array key names, `.npz` sizes, caching and run-mode logic.
- GPU-memory tuning notes (which knob to lower on OOM), runtime accounting and forward-pass counts.
- Figure-level plotting conventions (colour palettes, axis ordering, panel layout, smoothing of
  display curves). The one exception is where a display choice is load-bearing for a claim —
  nearest-neighbour rendering of receptive fields, shared colour scales across trace panels, and the
  fixed head ordering on the per-head bar charts — since each of those exists to avoid creating or
  hiding an effect.
- The `explore_h14_copy` / `explore_comparatorfig` exploratory notebooks, which were superseded by
  the batch data modules.

## Caveats carried into the appendix on purpose

These are stated in the text rather than suppressed, because a reviewer will ask:

1. The multi-dimensional frame is **assembled** from independently found directions, not fitted
   jointly against the interchange objective, so its score is a lower bound on what that rank can do
   at that site (§"The directions, and the frames they span").
2. The alternative-case evaluation is a **specificity control**, not a symmetric sufficiency test;
   the symmetric test needs a counterfactual that perturbs the losing operand (§"Null distributions
   and controls").
3. Cloud rows are **not independent** — each unordered pair is entered in both orders — so nominal
   standard errors on cloud fits are slightly optimistic (§"Representation clouds").
4. Three-digit questions sit slightly **out of distribution** relative to their fixed two-digit
   exemplar (§"Task and prompt").
5. Position recovery can exceed 1 and is reported unclipped (§"Metrics").
6. For $K=3$, pairwise marginals of a receptive field **average away** genuine three-way
   interactions (§"Receptive fields").

## References cited

Geiger et al. 2021 (causal abstraction, IIA); Geiger et al. 2023 (DAS); Meng et al. 2022 (ROME,
normalized restoration, severed paths); Vig et al. 2020 (causal mediation); Wang et al. 2023 (IOI,
contrastive logit difference, per-head site); Goldowsky-Dill et al. 2023 (path patching); Nanda 2023,
Syed et al. 2024 (EAP), Kramár et al. 2024 (AtP\*) (attribution patching); Conmy et al. 2023 (ACDC);
Chan et al. 2022 (causal scrubbing); Elhage et al. 2021 (virtual weights); Alain & Bengio 2016 and
Belinkov 2022 (linear probing); Gurnee & Tegmark 2024 (unscaled PCA convention); Brown, Cai &
DasGupta 2001 (Wilson interval); Qwen2.5 technical report.
