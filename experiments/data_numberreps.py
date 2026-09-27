"""Collect the L13 number-representation results for the paper figures.

Setup borrowed from e11_comparison_largemodels (`generate_manifolds_compare.ipynb` for the cloud /
PCA / log-fit conventions, `das_linear_vs_manifold.ipynb` for the two-case counterfactual design and
the rank-1 DAS training): Qwen2.5-7B-Instruct, one-shot `max(a, b)` prompt, SEED = 52, and the
residual stream *leaving* decoder layer 13 (zero-indexed, i.e. `model.model.layers[13]`'s output --
the same hook point `cloud_resid_a[13]` / `DAS_LAYERS = [13, ...]` use there).

Everything is computed twice: once for 2-digit numbers (the usual regime) and once for 3-digit.

Outputs -> results/numberreps_2digit.npz, results/numberreps_3digit.npz, results/meta.json.
Nothing is plotted here; makefig_numberreps.ipynb loads these and draws the figures.

See docs/data_numberreps.md.
"""

import os
import gc
import json
import time
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.append('../')
from functions.comparison_utils import tok_indices_for_chars  # noqa: E402

# ----------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------
SEED       = 52
MODELNAME  = 'Qwen/Qwen2.5-7B-Instruct'
LAYER      = 13                 # zero-indexed decoder layer; residual stream leaving it
ANCHOR     = 'The maximum of '
K          = 2                  # two numbers per prompt
RESULTS_DIR = 'results'

N_CLOUD    = 2000               # pairs in the PCA / projection cloud (both orders, balanced)
CLOUD_BS   = 20                 # forward-pass batch size for the cloud capture
N_EVAL     = 400                # held-out triples for the patching sweeps
EVAL_BS    = 48                 # forward-pass minibatch for evaluation -- see _batched note below
N_DAS      = 128                # triples used to train the DAS direction (disjoint from N_EVAL)
N_DAS_BS   = 32                 # DAS minibatch size; gradients accumulate over the full N_DAS
DAS_STEPS  = 100
DAS_LR     = 0.05
DAS_N_SEEDS = 2                 # random inits retrained on the same data, for the stability check
N_SPECTRUM = 20                 # how many PCA eigenvalues to keep for the variance spectrum

# Digit regimes. `gap` is the minimum separation inside a triple a > b > c, scaled with the range so
# the two regimes pose the same "how far apart are the numbers" problem. `fit_lo` is the lower cutoff
# used for the masked log fit at the first-number position (generate_manifolds_compare.ipynb uses
# a_vals >= 25 for 2-digit; 250 is the same fraction of the range).
REGIMES = {
    '2digit': dict(low=10,  high=100,  gap=10,  fit_lo=25),
    '3digit': dict(low=100, high=1000, gap=100, fit_lo=250),
}

CASES     = ['n1 larger', 'n2 larger']
CASE_SLOT = {'n1 larger': 0, 'n2 larger': 1}

os.makedirs(RESULTS_DIR, exist_ok=True)

hfcache_dir = os.path.join(os.environ.get('SCRATCHDIR', os.path.expanduser('~')), 'hf_cache')
os.makedirs(hfcache_dir, exist_ok=True)
os.environ['HF_HOME'] = hfcache_dir
print('HF_HOME:', hfcache_dir, flush=True)

import random  # noqa: E402
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, flush=True)


def gpu_mem(tag=''):
    if device != 'cuda':
        return
    a, r = torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9
    print(f'{tag:>30}  allocated {a:.2f} GB   reserved {r:.2f} GB', flush=True)


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

model = tokenizer = None
n_layers = d_model = head_dim = None


def load_model(name=MODELNAME):
    """Load the model/tokenizer into this module's globals and return them.

    Kept out of import time so a notebook can `from data_numberreps import *` for the helpers
    alone, and decide for itself when to pay for the weights. Idempotent -- a second call is a
    no-op, so re-running a notebook cell does not reload 7B parameters."""
    global model, tokenizer, n_layers, d_model, head_dim
    if model is not None:
        return model, tokenizer

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(device)
    model.eval()
    model.config.use_cache = False      # no KV cache is wanted anywhere here
    tokenizer = AutoTokenizer.from_pretrained(name)

    n_layers = model.config.num_hidden_layers
    d_model  = model.config.hidden_size
    head_dim = d_model // model.config.num_attention_heads

    # Only the DAS basis is ever trained -- keeps the backward pass from allocating gradient
    # buffers for the ~7B base weights.
    for prm in model.parameters():
        prm.requires_grad_(False)

    print(f'{name}: {n_layers} layers, d_model={d_model}, head_dim={head_dim}; '
          f'capturing/patching layer {LAYER}', flush=True)
    gpu_mem('model loaded')
    return model, tokenizer


# ----------------------------------------------------------------------------------------------
# Prompt and token positions
# ----------------------------------------------------------------------------------------------
def prompt_gen(nums):
    """The e11 one-shot prompt. The anchor example stays `max(12, 4)` in both digit regimes so the
    only thing that changes between regimes is the question clause itself."""
    return (f'Answer in the following format with a single answer. '
            f'The maximum of 12 and 4 is 12. '
            f'The maximum of {nums[0]} and {nums[1]} is ')


def num_char_spans(prompt, nums):
    """Char span of each number in the question clause (scanned forward from past the last ANCHOR)."""
    cur = prompt.rfind(ANCHOR) + len(ANCHOR)
    spans = []
    for n in nums:
        s = prompt.index(str(n), cur)
        spans.append((s, s + len(str(n))))
        cur = s + len(str(n))
    return spans


def num_tok_span(prompt, nums, i, offsets_np):
    """Token indices covering number `i` in `prompt`."""
    s, e = num_char_spans(prompt, nums)[i]
    return tok_indices_for_chars(offsets_np, s, e)


def leading_digits_distinct(nums):
    """First-token readouts are only unambiguous when no two numbers share a leading digit."""
    return len({str(n)[0] for n in nums}) == len(nums)


def order_key(nums):
    """Slot indices in descending value order. (0, 1) means slot 0 holds the max."""
    return tuple(int(i) for i in np.argsort(-np.asarray(nums)))


def regime_positions(low, high):
    """Template token layout for a regime: prompt length T and the last-token position of each
    number. Every prompt in a regime has the same number of digits, hence the same layout."""
    ref = [low + 1, low + 2]
    p = prompt_gen(ref)
    enc = tokenizer(p, return_tensors='pt', return_offsets_mapping=True)
    offs = enc['offset_mapping'][0].numpy()
    T = enc['input_ids'].shape[1]
    pos_last = [num_tok_span(p, ref, i, offs)[-1] for i in range(K)]
    n_tok = [len(num_tok_span(p, ref, i, offs)) for i in range(K)]
    return T, pos_last, n_tok, p


# ----------------------------------------------------------------------------------------------
# Counterfactual pairs (two cases) -- das_linear_vs_manifold.ipynb's design
# ----------------------------------------------------------------------------------------------
def build_triples(n, seed, low, high, gap):
    """(a, b, c) with a > b + gap > c + 2*gap, all in [low, high), distinct leading digits."""
    g = torch.Generator().manual_seed(seed)
    out, seen = [], set()
    while len(out) < n:
        a, b, c = torch.randint(low, high, (3,), generator=g).tolist()
        if a > b + gap and b > c + gap and leading_digits_distinct([a, b, c]) and (a, b, c) not in seen:
            seen.add((a, b, c))
            out.append((a, b, c))
    return out


def sample_cloud_pairs(n, seed, low, high):
    """`n` pairs for the representation cloud: n/2 distinct unordered pairs, each in both orders, so
    "which slot holds the max" is exactly balanced (generate_manifolds_compare.ipynb's convention)."""
    g = torch.Generator().manual_seed(seed)
    base, seen = [], set()
    while len(base) < n // 2:
        a, b = torch.randint(low, high, (2,), generator=g).tolist()
        key = frozenset((a, b))
        if a != b and key not in seen and leading_digits_distinct([a, b]):
            seen.add(key)
            base.append((a, b))
    return base + [(b, a) for a, b in base]


def make_batch(triples, win_slot, T):
    """Tokenised clean/corrupted batch with the clean maximum placed at slot `win_slot`.

    clean = (a at win_slot, b elsewhere) -> model answers a.
    corrupted = (c at win_slot, b elsewhere) -> model answers b (`tok_b`).
    A successful interchange makes the corrupted run answer its own slot-`win_slot` value c (`tok_nc`).
    """
    other = 1 - win_slot
    clean_pairs, corr_pairs = [], []
    for a, b, c in triples:
        cp, rp = [None, None], [None, None]
        cp[win_slot], cp[other] = a, b
        rp[win_slot], rp[other] = c, b
        clean_pairs.append(cp)
        corr_pairs.append(rp)
    assert all(order_key(p) == (win_slot, other) for p in clean_pairs), 'clean max must sit at win_slot'
    assert all(order_key(p) == (other, win_slot) for p in corr_pairs), 'corrupted max must sit at the other slot'

    cp_s = [prompt_gen(p) for p in clean_pairs]
    rp_s = [prompt_gen(p) for p in corr_pairs]
    enc_c = tokenizer(cp_s, return_tensors='pt', return_offsets_mapping=True)
    enc_r = tokenizer(rp_s, return_tensors='pt', return_offsets_mapping=True)
    assert enc_c['input_ids'].shape[1] == T and enc_r['input_ids'].shape[1] == T, \
        'every prompt in a regime must share the token layout'
    ids_c, ids_r = enc_c['input_ids'].to(device), enc_r['input_ids'].to(device)

    nc_list, b_list = [], []
    for i in range(len(triples)):
        orr = enc_r['offset_mapping'][i].numpy()
        # First token of each number = its leading digit; unambiguous because leading digits differ.
        nc_list.append(int(ids_r[i, num_tok_span(rp_s[i], corr_pairs[i], win_slot, orr)[0]]))
        b_list.append(int(ids_r[i, num_tok_span(rp_s[i], corr_pairs[i], other, orr)[0]]))

    return dict(ids_c=ids_c, ids_r=ids_r, n=len(triples), win_slot=win_slot,
                clean_pairs=np.array(clean_pairs), corr_pairs=np.array(corr_pairs),
                tok_nc=torch.tensor(nc_list, device=device),
                tok_b=torch.tensor(b_list, device=device))


# ----------------------------------------------------------------------------------------------
# Hooks, interventions, metrics
# ----------------------------------------------------------------------------------------------
def capture_resid(ids, positions):
    """(B, len(positions), d_model) float32 residual stream leaving LAYER, one forward pass."""
    holder = {}

    def _hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        holder['h'] = hs[:, positions, :].detach().float()

    hnd = model.model.layers[LAYER].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(ids)
    finally:
        hnd.remove()
    return holder['h']


def clean_hidden(ids, bs=None):
    """(B, T, d_model) residual stream leaving LAYER for the clean batch -- the interchange source.

    Run in minibatches: a forward pass materializes logits for every position before anything is
    sliced, i.e. B x T x |vocab| x 2 bytes (~0.5 GB at B=48, T=36 for Qwen2.5's 152k vocab), so the
    logits, not the activations, are what bounds the batch size here.
    """
    bs = EVAL_BS if bs is None else bs
    holder, chunks = {}, []

    def _hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        holder['h'] = hs.detach()

    hnd = model.model.layers[LAYER].register_forward_hook(_hook)
    try:
        for lo in range(0, ids.shape[0], bs):
            with torch.no_grad():
                _ = model(ids[lo:lo + bs])
            chunks.append(holder.pop('h'))
    finally:
        hnd.remove()
    return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]


def make_full_patch_hook(clean_h, pos):
    """Full-rank interchange at `pos`: h' = h_clean."""
    def _hook(mod, inp, out, _pos=pos, _c=clean_h):
        hs = out[0] if isinstance(out, tuple) else out
        hs_new = hs.clone()
        hs_new[:, _pos, :] = _c[:, _pos, :].to(hs.dtype)
        return (hs_new,) + out[1:] if isinstance(out, tuple) else hs_new
    return _hook


def make_subspace_patch_hook(Q, clean_h, pos):
    """Rank-k interchange at `pos`: h' = h + Q Q^T (h_clean - h), Q (d_model, k) orthonormal."""
    def _hook(mod, inp, out, _pos=pos, _c=clean_h, _Q=Q):
        hs = out[0] if isinstance(out, tuple) else out
        h = hs[:, _pos, :].float()
        cl = _c[:, _pos, :].float()
        patched = h + (cl @ _Q - h @ _Q) @ _Q.T
        hs_new = hs.clone()
        hs_new[:, _pos, :] = patched.to(hs.dtype)
        return (hs_new,) + out[1:] if isinstance(out, tuple) else hs_new
    return _hook


def run_patched(hook, bat, grad=False):
    """Corrupted-batch forward with `hook` on LAYER -> (B, vocab) last-position logits."""
    hnd = model.model.layers[LAYER].register_forward_hook(hook)
    try:
        if grad:
            return model(bat['ids_r']).logits[:, -1, :].float()
        with torch.no_grad():
            return model(bat['ids_r']).logits[:, -1, :].float()
    finally:
        hnd.remove()


def eval_patched(factory, bat, ch, bs=None):
    """Per-example position recovery and IIA for one intervention, in forward-pass minibatches.

    `factory(ch_slice)` builds the hook for each minibatch, so the clean-activation source is
    sliced to match. Metrics are reduced per minibatch and concatenated, so the full
    (n, |vocab|) logits tensor is never held.
    """
    bs = EVAL_BS if bs is None else bs
    recs, iias = [], []
    for lo in range(0, bat['n'], bs):
        hi = min(lo + bs, bat['n'])
        sub = _slice_batch(bat, lo, hi)
        hnd = model.model.layers[LAYER].register_forward_hook(factory(ch[lo:hi]))
        try:
            with torch.no_grad():
                lg = model(sub['ids_r']).logits[:, -1, :].float()
        finally:
            hnd.remove()
        r, i = posrec_and_iia(lg, sub)
        recs.append(r)
        iias.append(i)
        del lg
    return np.concatenate(recs), np.concatenate(iias)


def pos_anchors(bat, bs=None):
    """Un-patched clean / corrupted position-logit-difference anchors, per example (minibatched)."""
    bs = EVAL_BS if bs is None else bs
    out_c, out_r = [], []
    for lo in range(0, bat['n'], bs):
        hi = min(lo + bs, bat['n'])
        idx = torch.arange(hi - lo, device=device)
        nc, tb = bat['tok_nc'][lo:hi], bat['tok_b'][lo:hi]
        with torch.no_grad():
            lc = model(bat['ids_c'][lo:hi]).logits[:, -1, :].float()
            lr = model(bat['ids_r'][lo:hi]).logits[:, -1, :].float()
        out_c.append(lc[idx, nc] - lc[idx, tb])
        out_r.append(lr[idx, nc] - lr[idx, tb])
        del lc, lr
    return torch.cat(out_c), torch.cat(out_r)


def posrec_and_iia(logits_p, bat):
    """Per-example position recovery and IIA.

    position recovery = (PLD_patched - PLD_corr) / (PLD_clean - PLD_corr), PLD = logit[tok_nc] -
    logit[tok_b] (the ROME/IOI normalized-restoration convention, Meng et al. 2022).
    IIA = 1[argmax over the full vocabulary == tok_nc] (Geiger et al. 2021).
    """
    idx = torch.arange(bat['n'], device=device)
    ld_p = logits_p[idx, bat['tok_nc']] - logits_p[idx, bat['tok_b']]
    rec = (ld_p - bat['pld_corr']) / (bat['pld_clean'] - bat['pld_corr'])
    iia = (logits_p.argmax(dim=-1) == bat['tok_nc']).float()
    return rec.cpu().numpy(), iia.cpu().numpy()


# ----------------------------------------------------------------------------------------------
# DAS: a learned rank-1 subspace (Geiger et al. 2023, single-causal-variable case)
# ----------------------------------------------------------------------------------------------
class DASSubspace(torch.nn.Module):
    """Learnable d_model x k orthonormal basis via a QR retraction -- the standard differentiable
    parametrization for Stiefel-manifold optimization."""

    def __init__(self, dim, k, dev):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.randn(dim, k, device=dev, dtype=torch.float32) * 0.01)

    def basis(self):
        Q, _ = torch.linalg.qr(self.raw)
        return Q


def cf_pair_loss(logits_p, bat):
    """The actual DAS training objective: cross-entropy toward the counterfactual target `tok_nc`,
    restricted to {tok_b, tok_nc}. IIA (an argmax) has no gradient; this is the differentiable
    surrogate, with IIA / position recovery kept as evaluation-only metrics."""
    idx = torch.arange(bat['n'], device=device)
    pair = torch.stack([logits_p[idx, bat['tok_b']], logits_p[idx, bat['tok_nc']]], dim=-1)
    return -torch.log_softmax(pair, dim=-1)[:, 1].mean()


def _slice_batch(bat, lo, hi):
    """A contiguous minibatch view of a counterfactual batch."""
    sub = {k: v[lo:hi] for k, v in bat.items()
           if isinstance(v, (torch.Tensor, np.ndarray, list)) and len(v) == bat['n']}
    sub['n'] = hi - lo
    return sub


def train_das_1d(bat, pos, ch=None, n_steps=DAS_STEPS, lr=DAS_LR, init_seed=None, bs=N_DAS_BS):
    """Train a rank-1 DAS basis at (LAYER, pos) on `bat`. Returns (Q (d_model,1) tensor, loss curve).

    Gradients are accumulated over minibatches of `bs` so the training-set size is decoupled from
    the activation memory of the backward pass (which stores every layer above `LAYER`).
    `init_seed` makes the random init reproducible, which is what the stability check varies;
    `ch` lets the caller hoist the clean-activation capture out of a retraining loop.
    """
    own_ch = ch is None
    if own_ch:
        ch = clean_hidden(bat['ids_c'])
    if init_seed is not None:
        torch.manual_seed(init_seed)
    das = DASSubspace(d_model, 1, device)
    opt = torch.optim.Adam([das.raw], lr=lr)
    bounds = [(i, min(i + bs, bat['n'])) for i in range(0, bat['n'], bs)]
    curve = []
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for lo, hi in bounds:
            sub = _slice_batch(bat, lo, hi)
            logits_p = run_patched(make_subspace_patch_hook(das.basis(), ch[lo:hi], pos), sub,
                                   grad=True)
            loss = cf_pair_loss(logits_p, sub) * (sub['n'] / bat['n'])   # mean over the full batch
            loss.backward()
            total += float(loss)
            del logits_p
        opt.step()
        curve.append(total)
    if own_ch:
        del ch
    torch.cuda.empty_cache()
    with torch.no_grad():
        return das.basis().detach(), np.array(curve, dtype=np.float32)


# ----------------------------------------------------------------------------------------------
# Log fit
# ----------------------------------------------------------------------------------------------
def fit_logx(x, y):
    """Least-squares fit y = p*log(x) + q; returns (p, q, r2)."""
    p, q = np.polyfit(np.log(x), y, 1)
    pred = p * np.log(x) + q
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(p), float(q), 1.0 - ss_res / ss_tot


def log_fit_block(values, comp, fit_lo):
    """Log fits of `comp` against `values`, over all points and over `values >= fit_lo`."""
    mask = values >= fit_lo
    p_a, q_a, r2_a = fit_logx(values, comp)
    p_m, q_m, r2_m = fit_logx(values[mask], comp[mask])
    return dict(p_all=p_a, q_all=q_a, r2_all=r2_a,
                p_masked=p_m, q_masked=q_m, r2_masked=r2_m,
                fit_lo=float(fit_lo), n_masked=int(mask.sum()),
                pearson_r=float(np.corrcoef(values, comp)[0, 1]))


# ----------------------------------------------------------------------------------------------
# One regime, end to end
# ----------------------------------------------------------------------------------------------
def run_regime(tag, low, high, gap, fit_lo):
    t0 = time.time()
    print(f'\n{"=" * 92}\n=== regime {tag}: numbers in [{low}, {high}), triple gap {gap}\n{"=" * 92}', flush=True)

    T, pos_last, n_tok, example_prompt = regime_positions(low, high)
    POS_A, POS_B = pos_last            # last token of the first / second number
    print(f'T={T}  tokens per number={n_tok}  first-number last tok @ {POS_A}  '
          f'second-number last tok @ {POS_B}', flush=True)
    print(f'example prompt: {example_prompt!r}', flush=True)

    out = {}

    # -- 1. Representation cloud at L13, first- and second-number last-digit positions -----------
    cloud_pairs = sample_cloud_pairs(N_CLOUD, SEED + 7, low, high)
    cloud_nums = np.array(cloud_pairs, dtype=np.int64)
    a_vals = cloud_nums[:, 0].astype(np.float64)
    b_vals = cloud_nums[:, 1].astype(np.float64)
    max_vals = cloud_nums.max(axis=1).astype(np.float64)
    position_label = (b_vals > a_vals).astype(np.int64)     # 0 = first larger, 1 = second larger

    resid_a = np.zeros((N_CLOUD, d_model), dtype=np.float32)
    resid_b = np.zeros((N_CLOUD, d_model), dtype=np.float32)
    for i0 in range(0, N_CLOUD, CLOUD_BS):
        chunk = cloud_pairs[i0:i0 + CLOUD_BS]
        enc = tokenizer([prompt_gen(list(p)) for p in chunk], return_tensors='pt')
        ids = enc['input_ids'].to(device)
        assert ids.shape[1] == T, 'cloud prompt length must match the regime template'
        h = capture_resid(ids, [POS_A, POS_B]).cpu().numpy()
        resid_a[i0:i0 + len(chunk)] = h[:, 0, :]
        resid_b[i0:i0 + len(chunk)] = h[:, 1, :]
        del h
        torch.cuda.empty_cache()
    print(f'cloud captured: resid_a {resid_a.shape}, resid_b {resid_b.shape}', flush=True)

    out.update(cloud_nums=cloud_nums, a_vals=a_vals, b_vals=b_vals, max_vals=max_vals,
               position_label=position_label, resid_a=resid_a, resid_b=resid_b)

    # -- 2. 3D PCA + variance explained ---------------------------------------------------------
    for name, cloud, vals in (('a', resid_a, a_vals), ('b', resid_b, b_vals)):
        pca3 = PCA(n_components=3).fit(cloud)
        pca_full = PCA(n_components=min(N_SPECTRUM, cloud.shape[0], d_model)).fit(cloud)
        coords = pca3.transform(cloud)                              # (N, 3)
        out[f'pca3_coords_{name}'] = coords.astype(np.float32)
        out[f'pca3_components_{name}'] = pca3.components_.astype(np.float32)          # (3, d_model)
        out[f'pca3_evr_{name}'] = pca3.explained_variance_ratio_.astype(np.float64)   # per-direction
        out[f'pca3_evals_{name}'] = pca3.explained_variance_.astype(np.float64)
        out[f'pca3_mean_{name}'] = pca3.mean_.astype(np.float32)
        out[f'pca_evr_spectrum_{name}'] = pca_full.explained_variance_ratio_.astype(np.float64)
        print(f'PCA @ {"first" if name == "a" else "second"}-number pos  '
              f'EVR per direction = {np.round(pca3.explained_variance_ratio_, 4).tolist()}  '
              f'(3D total {pca3.explained_variance_ratio_.sum():.4f})', flush=True)
        # PC1 log fit, kept alongside so the notebook can show the unsupervised baseline for free.
        out[f'logfit_pc1_{name}'] = log_fit_block(vals, coords[:, 0].astype(np.float64), fit_lo)

    # -- 3. DAS 1D direction, trained at the FIRST-number position ------------------------------
    # One deduplicated pool, then split -- disjoint by construction. Drawing the two sets
    # independently does NOT work at these sizes: the population is ~55k triples (2-digit), so two
    # independent draws of 128 and 400 collide with probability ~1 (expected overlap ~0.9).
    # build_triples is a deterministic prefix in `n`, so pool[:N_EVAL] is exactly the old
    # build_triples(N_EVAL, SEED, ...) and the evaluation set is unchanged.
    pool = build_triples(N_EVAL + N_DAS, SEED, low, high, gap)
    eval_triples, das_triples = pool[:N_EVAL], pool[N_EVAL:]
    overlap = set(map(tuple, das_triples)) & set(map(tuple, eval_triples))
    assert not overlap, f'DAS training triples must be held out from the eval set ({len(overlap)} shared)'
    assert len(das_triples) == N_DAS and len(eval_triples) == N_EVAL

    # Trained on the "clean max at slot 0" case, patching the first number's own position -- the
    # only case in which a first-number stage exists at all.
    das_bat = make_batch(das_triples, CASE_SLOT['n1 larger'], T)
    das_bat['pld_clean'], das_bat['pld_corr'] = pos_anchors(das_bat)

    # Held-out batches, built once: the seed check and the patching sweeps both score on them.
    eval_batches = {}
    for _case in CASES:
        _b = make_batch(eval_triples, CASE_SLOT[_case], T)
        _b['pld_clean'], _b['pld_corr'] = pos_anchors(_b)
        eval_batches[_case] = _b

    # Retrained DAS_N_SEEDS times from different random inits on the SAME data. Seed 0 is the
    # direction used everywhere else; the rest are the stability check. With d_model = 3584 and
    # N_DAS counterfactuals the objective constrains Q through only N_DAS scalar projections, so
    # many directions can fit the training data -- the question is whether optimization actually
    # lands in the same place. Pairwise |cos| near 1 with a tight held-out IIA spread says the
    # solution is well determined; near-orthogonal directions at equal IIA say it is not, and the
    # singular "the direction" framing would not be supported.
    ch_das = clean_hidden(das_bat['ids_c'])
    ev = eval_batches['n1 larger']
    ch_ev = clean_hidden(ev['ids_c'])

    seed_dirs, seed_curves, seed_rec, seed_iia, seed_rec_tr, seed_iia_tr = [], [], [], [], [], []
    for si in range(DAS_N_SEEDS):
        Qs, curve = train_das_1d(das_bat, POS_A, ch=ch_das, init_seed=SEED + 1000 * si)
        _fac = lambda c, _Q=Qs: make_subspace_patch_hook(_Q, c, POS_A)
        r_tr, i_tr = eval_patched(_fac, das_bat, ch_das)
        r_ev, i_ev = eval_patched(_fac, ev, ch_ev)
        if si == 0:
            Q, das_curve = Qs, curve
            rec_tr, iia_tr = r_tr, i_tr
        seed_dirs.append(Qs[:, 0].cpu().numpy().astype(np.float32))
        seed_curves.append(curve)
        seed_rec_tr.append(r_tr.mean()); seed_iia_tr.append(i_tr.mean())
        seed_rec.append(r_ev.mean()); seed_iia.append(i_ev.mean())
        print(f'  DAS seed {si}: loss {curve[0]:.4f} -> {curve[-1]:.4f}   '
              f'train IIA {i_tr.mean():.3f}   held-out IIA {i_ev.mean():.3f}  '
              f'(posrec {r_ev.mean():.3f})', flush=True)
    del ch_das, ch_ev
    torch.cuda.empty_cache()

    das_dir = seed_dirs[0]                                           # unit vector in R^d_model
    S = np.stack(seed_dirs).astype(np.float64)
    seed_cos = S @ S.T                                               # signed; DAS signs are arbitrary
    off = np.abs(seed_cos)[~np.eye(DAS_N_SEEDS, dtype=bool)]
    print(f'DAS 1D @ first-number pos, n_train={N_DAS}, {DAS_N_SEEDS} inits: '
          f'held-out IIA {np.mean(seed_iia):.3f} +/- {np.std(seed_iia):.3f} '
          f'(min {np.min(seed_iia):.3f}, max {np.max(seed_iia):.3f})', flush=True)
    print(f'  pairwise |cos| between seeds: mean {off.mean():.3f}  min {off.min():.3f}  '
          f'max {off.max():.3f}   (chance ~ {1 / np.sqrt(d_model):.3f})', flush=True)
    print(f'  seed 0 (the direction used below): train IIA {iia_tr.mean():.3f}  '
          f'held-out IIA {seed_iia[0]:.3f}', flush=True)
    gpu_mem('after DAS training')

    out.update(das_dir=das_dir, das_loss_curve=das_curve,
               das_train_posrec=rec_tr, das_train_iia=iia_tr,
               das_train_triples=np.array(das_triples, dtype=np.int64),
               das_seed_dirs=S.astype(np.float32),
               das_seed_cos=seed_cos,
               das_seed_curves=np.stack(seed_curves).astype(np.float32),
               das_seed_iia=np.array(seed_iia, dtype=np.float64),
               das_seed_posrec=np.array(seed_rec, dtype=np.float64),
               das_seed_iia_train=np.array(seed_iia_tr, dtype=np.float64),
               das_seed_posrec_train=np.array(seed_rec_tr, dtype=np.float64))

    # Projection of the DAS direction into each 3D PCA basis, plus how much of it that basis holds.
    for name in ('a', 'b'):
        comps = out[f'pca3_components_{name}']                      # (3, d_model)
        proj = (das_dir @ comps.T).astype(np.float64)               # (3,) direction cosines
        out[f'das_in_pca3_{name}'] = proj
        out[f'das_pca3_captured_{name}'] = np.array(np.linalg.norm(proj))  # ||P P^T w|| / ||w||, w unit
        print(f'DAS direction in the {name}-cloud 3D PCA basis: '
              f'{np.round(proj, 4).tolist()}  (norm captured {np.linalg.norm(proj):.4f})', flush=True)

    # -- 4. Component along the fixed DAS direction, + log fits ---------------------------------
    # Mean-centred on each cloud's own mean, so the component is comparable with the PCA coords.
    comp_a = ((resid_a - resid_a.mean(axis=0)) @ das_dir).astype(np.float64)
    comp_b = ((resid_b - resid_b.mean(axis=0)) @ das_dir).astype(np.float64)
    out.update(das_comp_a=comp_a, das_comp_b=comp_b,
               das_comp_a_raw=(resid_a @ das_dir).astype(np.float64),
               das_comp_b_raw=(resid_b @ das_dir).astype(np.float64))

    out['logfit_das_a'] = log_fit_block(a_vals, comp_a, fit_lo)
    out['logfit_das_b'] = log_fit_block(b_vals, comp_b, fit_lo)
    for name, blk in (('first', out['logfit_das_a']), ('second', out['logfit_das_b'])):
        print(f'log fit, DAS component @ {name} number:  all points  y={blk["p_all"]:+.3f}*log(x)'
              f'{blk["q_all"]:+.3f}  R2={blk["r2_all"]:.3f}   |   x>={blk["fit_lo"]:.0f}  '
              f'y={blk["p_masked"]:+.3f}*log(x){blk["q_masked"]:+.3f}  R2={blk["r2_masked"]:.3f}',
              flush=True)

    # -- 5. Patching sweeps: full L13 residual vs DAS 1D vs PC1, both cases, both positions ------
    # The band pipeline (das_linear_vs_manifold.ipynb) scores L13 at the clean max's own slot, so
    # `patch_pos == the clean max's slot` is the band-consistent cell; the other position is kept as
    # the control.
    #
    # PC1 is the unsupervised rank-1 baseline at the same rank as DAS -- taken, like the DAS
    # direction, from the FIRST-number cloud, so the two rank-1 conditions differ only in how the
    # direction was found (trained on the counterfactual objective vs. top variance direction).
    pc1_dir = out['pca3_components_a'][0].astype(np.float32)
    pc1_dir = pc1_dir / np.linalg.norm(pc1_dir)
    Q_pc1 = torch.tensor(pc1_dir, device=device, dtype=torch.float32).reshape(-1, 1)
    out['pc1_dir'] = pc1_dir
    out['das_pc1_cos'] = np.array(float(das_dir @ pc1_dir))
    # Position-matched PC1: the SECOND-number cloud's own top direction. `pc1` transfers the
    # first-number direction everywhere (the DAS-matched control); `pc1b` is the direction the
    # second-number position actually has, which is what a "PC1 at n2" figure means.
    pc1b_dir = out['pca3_components_b'][0].astype(np.float32)
    pc1b_dir = pc1b_dir / np.linalg.norm(pc1b_dir)
    Q_pc1b = torch.tensor(pc1b_dir, device=device, dtype=torch.float32).reshape(-1, 1)
    out['pc1b_dir'] = pc1b_dir
    out['das_pc1b_cos'] = np.array(float(das_dir @ pc1b_dir))

    print(f'PC1 baseline direction from the first-number cloud; cos(DAS, PC1) = '
          f'{float(das_dir @ pc1_dir):+.4f}   cos(DAS, PC1_n2) = '
          f'{float(das_dir @ pc1b_dir):+.4f}', flush=True)

    patch_positions = {'first number': POS_A, 'second number': POS_B}
    for case in CASES:
        bat = eval_batches[case]                     # built once, above
        ch = clean_hidden(bat['ids_c'])
        ckey = case.replace(' ', '_')
        out[f'triples_{ckey}'] = np.array(eval_triples, dtype=np.int64)
        out[f'clean_pairs_{ckey}'] = bat['clean_pairs']
        out[f'corr_pairs_{ckey}'] = bat['corr_pairs']
        out[f'pld_clean_{ckey}'] = bat['pld_clean'].cpu().numpy()
        out[f'pld_corr_{ckey}'] = bat['pld_corr'].cpu().numpy()
        for plabel, ppos in patch_positions.items():
            pkey = plabel.replace(' ', '_')
            # Hook FACTORIES, not hooks: eval_patched slices the clean source per minibatch.
            for mkey, factory in (('full', lambda c, _p=ppos: make_full_patch_hook(c, _p)),
                                  ('das',  lambda c, _p=ppos: make_subspace_patch_hook(Q, c, _p)),
                                  ('pc1',  lambda c, _p=ppos: make_subspace_patch_hook(Q_pc1, c, _p)),
                                  ('pc1b', lambda c, _p=ppos: make_subspace_patch_hook(Q_pc1b, c, _p))):
                rec, iia = eval_patched(factory, bat, ch)
                out[f'posrec_{mkey}_{ckey}_{pkey}'] = rec
                out[f'iia_{mkey}_{ckey}_{pkey}'] = iia
                print(f'{case:>10} | patch @ {plabel:>13} | {mkey:>4}: '
                      f'position recovery {rec.mean():.3f} +/- {rec.std() / np.sqrt(len(rec)):.3f}   '
                      f'IIA {iia.mean():.3f}', flush=True)
        del ch
        torch.cuda.empty_cache()

    out['meta'] = dict(tag=tag, low=low, high=high, gap=gap, fit_lo=fit_lo, T=T,
                       pos_first=int(POS_A), pos_second=int(POS_B), n_tok_per_number=n_tok,
                       example_prompt=example_prompt, n_cloud=N_CLOUD, n_eval=N_EVAL, n_das=N_DAS,
                       das_steps=DAS_STEPS, das_lr=DAS_LR, das_train_pos='first number',
                       das_bs=N_DAS_BS, das_n_seeds=DAS_N_SEEDS,
                       das_seed_cos_absmean=float(np.abs(seed_cos)[~np.eye(DAS_N_SEEDS, dtype=bool)].mean()),
                       das_seed_iia_mean=float(np.mean(seed_iia)),
                       das_seed_iia_std=float(np.std(seed_iia)),
                       layer=LAYER, d_model=d_model, model=MODELNAME, seed=SEED,
                       cases=CASES, patch_positions=list(patch_positions),
                       patch_methods=['full', 'das', 'pc1', 'pc1b'],
                       pc1_from='first number cloud', pc1b_from='second number cloud',
                       runtime_s=round(time.time() - t0, 1))
    print(f'regime {tag} done in {out["meta"]["runtime_s"]:.1f} s', flush=True)
    return out


# ----------------------------------------------------------------------------------------------
# Run both regimes and save
# ----------------------------------------------------------------------------------------------
def save_regime(tag, res):
    """Split the result dict: arrays -> .npz, scalars/dicts -> the returned json-able block."""
    arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    path = os.path.join(RESULTS_DIR, f'numberreps_{tag}.npz')
    np.savez_compressed(path, **arrays)
    return path, arrays, scalars


if __name__ == '__main__':
    load_model()

    all_meta = dict(model=MODELNAME, layer=LAYER, seed=SEED, n_layers=n_layers, d_model=d_model,
                    anchor=ANCHOR, regimes={})
    sizes = {}

    for tag, cfg in REGIMES.items():
        res = run_regime(tag, **cfg)
        path, arrays, scalars = save_regime(tag, res)
        all_meta['regimes'][tag] = scalars
        sizes[tag] = dict(path=path,
                          on_disk_MB=os.path.getsize(path) / 1e6,
                          in_memory_MB=sum(a.nbytes for a in arrays.values()) / 1e6,
                          arrays={k: dict(shape=list(a.shape), dtype=str(a.dtype), MB=a.nbytes / 1e6)
                                  for k, a in arrays.items()})
        del res
        gc.collect()
        torch.cuda.empty_cache()

    meta_path = os.path.join(RESULTS_DIR, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(all_meta, f, indent=2, default=float)

    # ----------------------------------------------------------------------------------------------
    # Size report
    # ----------------------------------------------------------------------------------------------
    print(f'\n{"=" * 92}\n=== RESULT SIZES\n{"=" * 92}', flush=True)
    total_disk = 0.0
    for tag, s in sizes.items():
        print(f'\n--- {s["path"]}  ({len(s["arrays"])} arrays)', flush=True)
        for k, a in sorted(s['arrays'].items(), key=lambda kv: -kv[1]['MB']):
            print(f'    {k:<34} {str(a["shape"]):>16} {a["dtype"]:>8}  {a["MB"]:9.3f} MB', flush=True)
        print(f'    {"TOTAL uncompressed":<34} {"":>16} {"":>8}  {s["in_memory_MB"]:9.3f} MB', flush=True)
        print(f'    {"ON DISK (npz, compressed)":<34} {"":>16} {"":>8}  {s["on_disk_MB"]:9.3f} MB', flush=True)
        total_disk += s['on_disk_MB']

    meta_MB = os.path.getsize(meta_path) / 1e6
    total_disk += meta_MB
    print(f'\n{meta_path}: {meta_MB:.3f} MB', flush=True)
    print(f'\nTOTAL results/ on disk: {total_disk:.3f} MB', flush=True)
    print('\ndone.', flush=True)
