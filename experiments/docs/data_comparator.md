# `data_comparator.py` + `makefig_comparator.ipynb`

Batch collection and plotting for the comparator circuit, with the **(v1, v2) plane as the base
patch**. `data_comparator.py` computes and saves; `makefig_comparator.ipynb` loads and draws, with
no model and no GPU.

```
python data_comparator.py         # ~14 min on one GPU
# -> results/comparator_data_2digit.npz, results/comparator_data_meta.json
# then open makefig_comparator.ipynb
```

## 1. What changed from `explore_comparatorfig.ipynb`

One substantive thing: **the base intervention**. The notebook sat every measurement on top of two
rank-1 patches (v1 inside H14's output, v2 in the L13 residual). This script sits them on a
**rank-2 interchange of the pre-MLP L14 residual inside span(v1, v2)** — `dict(premlp=P_uv)`,
using `data_sharedrep`'s `_premlp_hooks` machinery and the same `P_uv = qr([v1, v2])` basis, so the
two files describe one geometry.

The reason is the algorithmic claim being tested: *v1 and v2 are the variables, they are summed
into one plane at n2, and the comparator reads that plane*. The rank-1 pair establishes the first
two legs — each direction moves the answer exactly when its own number changes, and the residual
stream is a sum — but it never sets the **summed** coordinate to its counterfactual value. It sets
two upstream sources, whose effects then propagate through L14 attention, which can rewrite n2 and
which also changes what H14 itself reads. Measured consequence (`makefig_sharedreps` Fig 7): the
rank-1 pair reaches 0.68 / 0.88 IIA, the rank-2 pre-MLP patch 0.81 / 0.95, against a same-site
full-rank ceiling of 0.82 / 0.99.

That matters most for **attribution**, because the score is a gradient evaluated *at the patched
state*. With a base that flips 68% of examples, part of the gradient is measured in a regime where
the interchange did not take. Both bases are kept as reported conditions (§2 below), so the
rank-1 dissociation — which is what licenses calling v1 and v2 "the variables" at all, and which
the plane patch cannot supply — is still in the output.

Everything is computed for **both** cases, and the case is the perturbation: `make_batch` varies
only the winning slot's number, so `n1 larger` perturbs y1 and `n2 larger` perturbs y2.

## 2. What the script computes

| § | block | output keys | cost |
| --- | --- | --- | --- |
| 1 | headline conditions: v1, v2, the rank-1 pair, the plane pre/post-MLP, full-rank pre/post | `cond_{iia,posrec}_{case}_{arm}` | ~1 min |
| 2 | module freezes (MLP / attention, L14-L16) under the plane patch | `mod_*` | ~1 min |
| 3 | per-neuron attribution over MLP14+MLP15 (37888 neurons) | `attr_{case}_L{l}`, `order_{case}` | ~1 min |
| 4 | top-k freeze curve, with random-k and bottom-k controls | `topk_*` | ~4 min |
| 5 | top 10/20/30 per case, the overlaps, and freezing each overlap set | `top{k}_{case}`, `overlap_k{k}`, `ov{k}_*` | ~1 min |
| 6 | clean-MLP injection, scored on tok_nc **and** tok_a | `inj_*` | ~1 min |
| 7 | the cloud in the plane; every stage in one signed frame | `plane_{pq,pre14,m14,pre15,m15}`, `diff_axis` | ~1 min |
| 8 | receptive fields on a dense (y1, y2) grid | `rf_grid`, `rf_acts_L{l}` | ~30 s |
| 9 | read-in / write-out geometry and the read-in SVD with a bootstrap null | `read_*`, `readfrac_*`, `write_*`, `svd_*` | instant |
| 10 | gate-preactivation boundaries, slice and planar fit | `bnd_*` | instant |
| 11 | L14 -> L15 weight paths and causal edges | `path_*`, `edge_*` | ~1 min |
| 12 | DAS at the L15 residual, ranks 1-3, plus cloud projections | `das15_*` | ~8 min |
| 13 | DAS inside the MLP neuron space | `mlpdas_*` | ~6 min |

### Which neurons get profiled

Sections 7-11 profile the **union of both cases' top-`N_PROFILE`** (default 30), ordered by the
better of a neuron's two ranks, so nothing strong in either case is dropped. `prof_shared` marks
which of them are in the intersection. The earlier notebook profiled only `n1 larger`'s top ten,
which silently made every geometry result y1-specific.

### Conventions worth knowing

- **Attribution.** `score_i ~= (a_i^corr - a_i^patched) * dPLD/da_i`, evaluated on the patched run,
  PLD = logit[tok_nc] - logit[tok_b]. The objective is **summed** over the batch so each row's
  gradient is its own derivative; `ATP_SCALE = 100` guards fp16 underflow; a zero handle with
  `requires_grad=True` is inserted at the `down_proj` input because the model's own parameters are
  frozen and there would otherwise be no graph. Ranked on the 128 held-out triples, never on the
  400 the curves are reported on. Nanda 2023; Syed et al. 2023; Kramar et al. 2024 (AtP*).
- **Two freeze hooks.** `make_neuron_freeze_hook` writes **in place** — the `down_proj` input is a
  temporary consumed by nothing else — which is what keeps the 66-condition sweep fast. The DAS
  version clones, because it runs under autograd.
- **One signed plane frame.** e1 = v1, e2 = v2 orthogonalised against it, each axis signed to rise
  with its own number, matching `makefig_sharedreps` Fig 6. Every plane read-out (`plane_*`, all
  `read_*` / `write_*` components, the boundaries) uses it, so arrows, clouds and lines are
  comparable without further transformation.
- **Boundaries, two lines.** The gate boundary is the hyperplane w.x = 0 (RMSNorm scales by a
  positive number and cannot move it). A hyperplane has no exact 2D trace, so both are saved: the
  **slice** at the cloud's mean out-of-plane state, whose normal *is* the in-plane read gradient,
  and the **planar fit** of z on (p, q), whose normal is a total regression coefficient and so need
  not be orthogonal to it. `bnd_on_frac` lets accuracies be read against a majority baseline.
- **Nulls everywhere.** In-plane fractions get 500 random neurons per layer (`*_bg_*`); the SVD
  anisotropy gets a 4000-draw bootstrap over random L14 neurons, with s1 and s2 from the *same*
  draw; weight paths get the full 18944-neuron distribution (`path_*_null`).

## 3. The plotting notebook

`makefig_comparator.ipynb`, 14 code cells, runs in seconds. Sections: conditions, module freezes,
attribution and top-k, overlap between cases, injection, the plane, receptive fields, geometry,
boundaries, connectivity, DAS. Knobs at the top of the relevant cells: `RF_SHARED_ONLY` (show only
the neurons both cases agree on), `GEO_LAYER`, `BND_NCOL`, `PLOT_CASE`.

Figures are written to `figs/comparator_2digit_*.png`; set `SAVE = False` to preview without
writing.

## 4. Caveats

- The script is written but **has not been executed** — it needs the GPU. Its imports, syntax and
  lint are clean, and the notebook has been run end to end against synthetic data with the same
  schema, so the two sides agree on key names; the compute paths themselves are reviewed, not run.
- Runtime is dominated by the 14 DAS fits (§12-13). Cutting `DAS15_RANKS` to `[1, 2]` or
  `MLPDAS_RANKS` to `[1]` roughly halves it.
- L16 is in the module sweep but not in the neuron pool, because the module freeze showed it
  contributes nothing. Add it to `NEURON_LAYERS` to include it.
- `N_PROFILE` must be one of `OVERLAP_KS` (asserted at import), since `prof_shared` is defined
  against the overlap at that depth.

## 5. References

Geiger et al. 2021 (interchange interventions, IIA); Geiger et al. 2023 (DAS); Meng et al. 2022
(normalized restoration, severed causal tracing); Vig et al. 2020 (single-module patching);
Goldowsky-Dill et al. 2023 and Wang et al. 2023 (path patching); Nanda 2023, Syed et al. 2023,
Kramar et al. 2024 (attribution patching and AtP*); Elhage et al. 2021 (virtual weights, the
per-head decomposition).
