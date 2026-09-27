# comparison-geometry

Code for the paper **"When Models Don't Manipulate Manifolds: The Geometry of a Comparison Task"** (&lt;venue / arXiv link&gt;).

The paper is a mechanistic analysis of how **Qwen2.5-7B-Instruct** computes `max(a, b)` (and
`max(a, b, c)`) from a one-shot prompt. It demonstrates how the model uses linear representations of each number to perform comparison, despite the presence of number manifolds. It looks at where each number is represented in the
residual stream, how attention head L14.H14 moves the first number to the second number's
position, the shared two-dimensional representation that results there, the MLP14–15 comparator
neurons that read it, and how this extends to three numbers. The tools are distributed alignment
search (DAS), interchange interventions and activation patching, PCA, linear probes and
neuron-level attribution.

This repository contains every script and notebook needed to recompute the results and redraw
all main-paper and appendix figures. 

## Repository layout

```
comparison-geometry/
├── functions/
│   └── comparison_utils.py        # shared helpers: token positions, caching, patching, traces
└── experiments/                   # all paper code; run everything from here
    ├── data_*.py                  # model computation, writes results/*.npz
    ├── explore_*.ipynb            # model computation + exploratory plots, writes results/*.npz
    ├── compute_app_*.py           # model computation for appendix-only analyses
    ├── makefig_*.ipynb            # main-paper figures (plotting only, no GPU)
    ├── makefig_appendix*.py       # appendix figures (plotting only, no GPU)
    ├── appendix_style.py          # shared style for the appendix figures
    ├── docs/                      # per-file notes: methods, array contents, figure design
    ├── results/                   # (generated) cached results
    ├── figs/                      # (generated) PNG previews
    └── figs_mainpaper/            # (generated) main-paper PDFs
```

## Setup

```bash
git clone https://github.com/Sai-Sumedh/comparison-geometry.git
cd comparison-geometry
conda create -n comparison-geometry python=3.10 -y
conda activate comparison-geometry
pip install -r requirements.txt
```

Tested with Python 3.10.16, PyTorch 2.6.0 (CUDA 12.4) and Transformers 4.50.3 (full pins in
`requirements.txt`).

**Hardware.** The model runs in float16, which is about 15 GB of weights. Every model-computation
step was run on a single GPU with 100 GB of host RAM. The figure scripts only need a CPU.

**Model weights** are downloaded from the Hugging Face Hub (`Qwen/Qwen2.5-7B-Instruct`) on the
first run. `data_numberreps.py` sets `HF_HOME` to `$SCRATCHDIR/hf_cache`, or to `~/hf_cache` when
`SCRATCHDIR` is unset.

## Reproducing the results

Run all commands from `experiments/`. Scripts use paths relative to that directory (`results/`,
`figs/`, and `../` for `functions/`). Each step reads the results of
the steps before it, so run them in this order. Everything uses `SEED = 52`
(`data_numberreps.py`).

| Step | Command | Writes to `results/` |
|---|---|---|
| 1 | `python data_numberreps.py` | `numberreps_{2,3}digit.npz`, `meta.json` |
| 2 | `python data_sharedrep.py` | `sharedrep_2digit.npz`, `sharedrep_meta.json` |
| 3 | `python data_comparator.py` (about 14 min) | `comparator_data_2digit.npz`, `comparator_data_meta.json` |
| 4 | run `explore_comparatorfig.ipynb` | `comparator_*_2digit.npz` (step 5 reads `comparator_das15_2digit.npz`) |
| 5 | run `explore_multicompare.ipynb` | `multicompare_*_2digit.npz` |
| 6 | run `explore_multicompare2.ipynb` | `multicompare2_*_2digit.npz` |
| 7 | `compute_app_*.py` (see below) | `app_*.npz` |

The three `explore_*` notebooks are both computation and analysis. Each block computes its
result, caches it to `results/`, and plots it straight away. On a re-run, blocks load the cached
file instead of recomputing. `data_multicompare.py` and `data_multicompare2.py` hold the
machinery these notebooks import; they are not run on their own.

**Step 7: appendix computations.** Each takes 2–12 min on one GPU:

```bash
python compute_app_behaviour.py            # app_behaviour.npz, app_patch_outcomes.npz
python compute_app_trace_k2.py             # app_trace_k2.npz, app_probes_k2.npz
python compute_app_trace_k3.py y1          # app_trace_k3_y1.npz (repeat for y2, y3)
python compute_app_u_layers.py 1 3 5 7 9 11  # app_u_layers_L{layer}.npz
python compute_app_v2_shortcut.py          # app_v2_shortcut.npz
```

These scripts are independent of each other, so they can run in parallel on separate GPUs.

## Reproducing the figures

After the steps above, the figures need no GPU.

### Main paper

| Figure file | Content | Notebook |
|---|---|---|
| `number_reps_fig.pdf` | Single-number representations and transport of the first number | `makefig_numberreps.ipynb` |
| `sharedrep_paperfig.pdf` | The shared representation at the second number's position | `makefig_sharedreps.ipynb` |
| `comparator_fig.pdf` | The comparator | `makefig_comparator.ipynb` |
| `multicompare_fig.pdf` | Three numbers | `makefig_multicompare.ipynb` |

Each notebook always writes a PNG preview to `figs/`. The PDF export is the commented-out
`plt.savefig('./figs_mainpaper/...')` line at the end of the figure cell; uncomment it to write
the PDF.

### Appendix

```bash
python makefig_appendix.py            # 21 figures -> figs_appendix/
python makefig_appendix_computed.py   # 7 figures + tab_patch_examples.tex -> figs_appendix/
```

Then run `makefig_multicompare_appendix.ipynb` to get `figs_appendix/app_k3_summary.pdf`. To
draw a subset, pass figure names; for example `python makefig_appendix.py u_vs_pc1 three_digit`
draws `app_u_vs_pc1.pdf` and `app_three_digit.pdf`.

| Script | Figures (`figs_appendix/app_<name>.pdf`) |
|---|---|
| `makefig_appendix.py` | `u_vs_pc1`, `three_digit`, `u_manifold`, `k3_manifolds`, `transport`, `shared_probes`, `shared_posrec`, `mlp_freeze`, `attribution`, `rf_shared`, `rf_exclusive`, `connectivity`, `l15_direction`, `k3_heads`, `k3_directions`, `k3_dissociation`, `k3_neurons`, `k3_l15`, `k3_readout_pca`, `k3_readout_causal`, `ablation` |
| `makefig_appendix_computed.py` | `behaviour`, `patch_outcomes` (+ `tab_patch_examples.tex`), `trace_k2`, `probes_k2`, `trace_k3`, `u_layers`, `v2_shortcut` |
| `makefig_multicompare_appendix.ipynb` | `k3_summary` |

## Notes

- **Documentation.** `docs/<file>.md` documents the matching script or notebook: what it
  computes and how, what each saved array holds, and the design choices behind each figure.
  `docs/makefig_comparator_old.md` holds the design notes for the comparator figure. The earlier
  notebook it describes is not part of this repository.
- **`topup_dasalign.py`** is a one-off patch for results made before `data_comparator.py`
  computed the MLP15 write alignment. A fresh run does not need it.
- **Notebooks** are distributed without outputs.
- **Known issue.** In `makefig_sharedreps.ipynb`, the cells after the combined paper figure are
  exploratory. The direction-cosine cell by probe site ("fig9") needs
  `from matplotlib.patches import Rectangle` added before it runs. The paper figure itself is
  unaffected.

## Citation

```bibtex
@article{<key>,
  title   = {When Models Don't Manipulate Manifolds: The Geometry of a Comparison Task},
  author  = {Hindupur, Sai Sumedh R. and Orgad, Hadas and Fel, Thomas and Ba, Demba},
  journal = {<venue / arXiv>},
  year    = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE).
