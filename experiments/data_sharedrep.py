"""Collect the "shared representation" results for the next paper figure.

Companion to `data_numberreps.py` (which stays the source of the L13 directions) and a batch-mode
version of the interactive work in `explore_h14_copy.ipynb`. Everything here lives at the
**second-number position**, where head L14.H14 has written the first number on top of whatever the
L13 residual already holds, so both numbers are present in the L14 residual stream.

What is computed, per digit regime:
  1. H14 output cloud @ n2: its 3D PCA (H14 output PC1), the H14-space rank-1 DAS direction, and
     each one's component vs the first number (with log fits) and vs the L13 DAS component read
     at n1.
  2. L14 residual cloud @ n2: 3D PCA, plus the H14 DAS and L13-PC1@n2 directions expressed in that
     PCA basis (the arrows).
  3. Per-head patching sweep over all of layer 14's heads (`n1 larger` case): each head's own output
     contribution interchanged at n2, alone and jointly with an L13 interchange at the same
     position (PC1 and full variants).
  4. Cosine similarities between H14 DAS, L13 PC1 @n2, and the two log-magnitude probe directions.
  5. IIA / position recovery for: L13 full @n2, L13 PC1 @n2, H14 DAS, H14 DAS + L13 PC1 @n2,
     L14 full @n2.

The L13 DAS direction and the L13 PC1 @n2 direction are *loaded* from
`results/numberreps_{tag}.npz` rather than retrained, so the two files describe one consistent
model. The evaluation triples are built from the same pool with the same split as
`data_numberreps.py`, so the numbers here are directly comparable with the ones saved there.

Notation (the figures use this; the array keys predate it and were deliberately NOT renamed, so
`h14_das_*` is v1 and the `uv_plane` keys are the (v1, v2) plane):
  v1 = the H14 DAS 1D direction  -- the FIRST number, written into the residual at n2 by the head
  v2 = L13 PC1 at n2             -- the SECOND number, already in place there
  u  = the L13 DAS direction at n1 -- the first number at its OWN position; it lives in
       data_numberreps.py and is only loaded here.
Comments below that say "u" for the H14 direction predate the rename and mean v1.

Outputs -> results/sharedrep_{tag}.npz, results/sharedrep_meta.json.
Nothing is plotted here; a makefig notebook loads these and draws the figures.

See docs/data_sharedrep.md.
"""

import os
import gc
import json
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

import data_numberreps as D
from data_numberreps import (prompt_gen, build_triples, make_batch, sample_cloud_pairs,
                             regime_positions, pos_anchors, posrec_and_iia, num_tok_span,
                             cf_pair_loss, DASSubspace, make_full_patch_hook,
                             make_subspace_patch_hook, log_fit_block,
                             CASE_SLOT, SEED, LAYER, REGIMES, EVAL_BS, N_EVAL, device)

# ----------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------
RESULTS_DIR = 'results'

H_LAYER    = 14                 # the layer whose heads are swept / whose residual is read
H_HEAD     = 14                 # the copy head (L14.H14), e11/head_analysis.ipynb
CASE       = 'n1 larger'        # the case v1 is trained in, and the one every headline number
                                # above is measured in
ALT_CASE   = 'n2 larger'        # repeat the headline conditions and the plane sweep here too,
                                # REUSING the directions found in CASE -- a generalization test,
                                # not a second fit. Set to None to skip (section 9b).

TAGS       = ['2digit']         # regimes to run; add '3digit' to do both

N_CLOUD    = 3000               # pairs in the H14 / L14 cloud (both orders, balanced)
CLOUD_BS   = 20                 # forward-pass batch size for the cloud capture

N_H14_DAS  = 128                # held-out triples used to train the H14-space DAS direction
H14_DAS_BS = 16                 # DAS minibatch; gradients accumulate over the full N_H14_DAS
H14_DAS_STEPS = 100
H14_DAS_LR = 0.05

RIDGE_ALPHAS = np.logspace(-2, 4, 13)
PROBE_TEST_FRAC = 0.2

N_SWEEP    = 64                 # eval prompts per grid point of the (u, v2) plane sweep
SWEEP_WIN  = 2                  # +/- window, in units of the number, for a grid value's
                                # cloud-mean coordinate (smooths the per-number estimate)

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(SEED)
torch.random.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ----------------------------------------------------------------------------------------------
# Per-head machinery at layer H_LAYER
#
# A head's own additive contribution to the residual stream is `z_h @ W_O^h.T`, read off the
# `o_proj` input (`e11/head_analysis.ipynb`). The o_proj bias is omitted because it is not
# attributable to any one head. Because o_proj's output is the sum over heads, rewriting one head's
# contribution is exactly `out += (target_h - current_h)` at that position.
# ----------------------------------------------------------------------------------------------
attn = W_O = d_model = head_dim = n_heads = None


def init_head_machinery():
    global attn, W_O, d_model, head_dim, n_heads
    d_model, head_dim = D.d_model, D.head_dim
    attn = D.model.model.layers[H_LAYER].self_attn
    W_O = attn.o_proj.weight.detach().float()          # (d_model, d_model), input is head-major
    n_heads = d_model // head_dim
    print(f'layer {H_LAYER}: {n_heads} heads x head_dim {head_dim}', flush=True)


def head_slice(head):
    return head * head_dim, (head + 1) * head_dim


def head_out(ctx, head):
    """ctx: (..., d_model) o_proj input -> `head`'s own residual-space output contribution."""
    s, e = head_slice(head)
    return ctx[..., s:e].float() @ W_O[:, s:e].T


def head_patch_handles(head, pos, target, P=None):
    """Hooks on layer H_LAYER's o_proj that rewrite `head`'s contribution at `pos`.

    P is None  -> full interchange of that head's output:  h_out := target.
    P (d, k)   -> rank-k interchange inside its own output: h_out += P P^T (target - h_out).
    Returns the handles; the caller removes them.
    """
    store = {}
    s, e = head_slice(head)
    W_h = W_O[:, s:e]

    def pre(mod, inp):
        store['ctx'] = inp[0]

    def post(mod, inp, out):
        cur = store['ctx'][:, pos, s:e].float() @ W_h.T        # (B, d_model)
        delta = (target - cur) if P is None else ((target - cur) @ P) @ P.T
        out_new = out.clone()
        out_new[:, pos, :] = out[:, pos, :] + delta.to(out.dtype)
        return out_new

    return [attn.o_proj.register_forward_pre_hook(pre),
            attn.o_proj.register_forward_hook(post)]


def capture_attn(ids, pos_b, layer=H_LAYER, head=H_HEAD):
    """(B, T) attention row of `layer`.`head` at query position `pos_b` -- what the head reads
    when it writes at n2.

    `output_attentions=True` makes transformers fall back from sdpa to eager attention for this
    pass (one warning, same numerics), so it is its own forward rather than folded into
    `capture_cloud`: that keeps every other saved activation on exactly the path it was computed on.
    """
    with torch.no_grad():
        att = D.model(ids, output_attentions=True).attentions[layer]
    return att[:, head, pos_b, :].detach().float()


def _premlp_hooks(pos, on_resid):
    """Hook pair exposing layer H_LAYER's residual stream *before* its MLP, at `pos`.

    That residual is `layer_input + self_attn(...)` -- the point at which H14 has just written and
    the layer's own MLP has not yet read. The decoder layer keeps its skip connection in a local
    variable, so the only way to change the pre-MLP residual is to change the attention sublayer's
    output; `on_resid(current_resid) -> new_resid` is applied there and converted back to a delta.
    Returns (handles, store); `on_resid` may be None to only observe (the value lands in
    `store['pre']`).
    """
    store = {}
    layer = D.model.model.layers[H_LAYER]

    def layer_pre(mod, args, kwargs):
        h = args[0] if args else kwargs['hidden_states']
        store['h_in'] = h[:, pos, :].detach().float()

    def attn_post(mod, inp, out):
        a = out[0] if isinstance(out, tuple) else out
        cur = store['h_in'] + a[:, pos, :].float()
        store['pre'] = cur.detach()
        if on_resid is None:
            return out
        a_new = a.clone()
        a_new[:, pos, :] = a[:, pos, :] + (on_resid(cur) - cur).to(a.dtype)
        return ((a_new,) + tuple(out[1:])) if isinstance(out, tuple) else a_new

    return [layer.register_forward_pre_hook(layer_pre, with_kwargs=True),
            layer.self_attn.register_forward_hook(attn_post)], store


def capture_premlp(ids, pos, bs=EVAL_BS):
    """(B, d_model) clean residual stream at `pos` just before layer H_LAYER's MLP."""
    handles, store = _premlp_hooks(pos, None)
    chunks = []
    try:
        for lo in range(0, ids.shape[0], bs):
            with torch.no_grad():
                _ = D.model(ids[lo:lo + bs])
            chunks.append(store.pop('pre'))
    finally:
        for h in handles:
            h.remove()
    return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]


def premlp_patch_handles(target, pos, P=None):
    """Interchange the pre-MLP residual at `pos`. P is None -> full; P (d, k) -> rank-k inside P."""
    def on_resid(cur, _t=target, _P=P):
        return _t if _P is None else cur + ((_t - cur) @ _P) @ _P.T

    return _premlp_hooks(pos, on_resid)[0]


def capture_cloud(ids, pos_a, pos_b):
    """One forward pass -> (L13 resid @ pos_a, H14 output @ pos_b, L14 pre-MLP resid @ pos_b,
    L14 resid @ pos_b).

    All four are read in the same pass so every row lines up with the same prompt: the L13 DAS
    component at n1 (the inset's x-axis) must come from the same prompts as the H14 component at n2,
    and the pre-MLP cloud must be row-aligned with the end-of-layer one it is compared against.
    """
    store = {}

    def l13_hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        store['l13'] = hs[:, pos_a, :].detach().float()

    def l14_hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        store['l14'] = hs[:, pos_b, :].detach().float()

    def pre(mod, inp):
        store['ctx'] = inp[0][:, pos_b, :].detach().float()

    # The pre-MLP pair hooks the decoder layer's *pre* slot and self_attn; capture_cloud's own
    # hooks sit on the layer's post slot and on o_proj, so nothing collides.
    pre_handles, pre_store = _premlp_hooks(pos_b, None)
    handles = [D.model.model.layers[LAYER].register_forward_hook(l13_hook),
               D.model.model.layers[H_LAYER].register_forward_hook(l14_hook),
               attn.o_proj.register_forward_pre_hook(pre)] + pre_handles
    try:
        with torch.no_grad():
            _ = D.model(ids)
    finally:
        for h in handles:
            h.remove()
    return store['l13'], head_out(store['ctx'], H_HEAD), pre_store['pre'], store['l14']


def clean_hidden_at(ids, layer, bs=EVAL_BS):
    """(B, T, d_model) residual stream leaving `layer` for the clean batch, in minibatches.

    `data_numberreps.clean_hidden` is pinned to LAYER; the L14-full condition needs layer 14's.
    Minibatched for the same reason as there: the logits, not the activations, bound the batch.
    """
    holder, chunks = {}, []

    def _hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        holder['h'] = hs.detach()

    hnd = D.model.model.layers[layer].register_forward_hook(_hook)
    try:
        for lo in range(0, ids.shape[0], bs):
            with torch.no_grad():
                _ = D.model(ids[lo:lo + bs])
            chunks.append(holder.pop('h'))
    finally:
        hnd.remove()
    return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]


def capture_ctx(ids, pos, bs=EVAL_BS):
    """(B, d_model) o_proj input at `pos` for layer H_LAYER -- the source of every head's clean
    output contribution. Kept as the context rather than the per-head outputs so one capture serves
    all `n_heads` conditions."""
    store, chunks = {}, []
    hnd = attn.o_proj.register_forward_pre_hook(
        lambda mod, inp: store.__setitem__('ctx', inp[0][:, pos, :].detach().float()))
    try:
        for lo in range(0, ids.shape[0], bs):
            with torch.no_grad():
                _ = D.model(ids[lo:lo + bs])
            chunks.append(store.pop('ctx'))
    finally:
        hnd.remove()
    return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]


# ----------------------------------------------------------------------------------------------
# Interventions as declarative specs
# ----------------------------------------------------------------------------------------------
def sweep_grid_values(low, high):
    """One grid value per distinct leading digit, spread over [low, high).

    The model names a number by emitting its **first** token, which is its leading digit -- that is
    why `build_triples` insists on distinct leading digits. Two grid values sharing a leading digit
    would be indistinguishable in the answer logits, so the grid takes one value per digit, placed
    mid-decade (15, 25, ... 95 for the 2-digit regime).
    """
    step = 10 ** (len(str(high - 1)) - 1)
    return [v for v in (d * step + step // 2 for d in range(1, 10)) if low <= v < high]


def first_tok_of(value, slot):
    """The token the model must emit to name `value`: that number's first token, read from a prompt
    with it at `slot`. Same derivation `make_batch` uses for tok_nc / tok_b, so the sweep and the
    interchange conditions score the same event."""
    pair = [value, value]
    prompt = prompt_gen(pair)
    enc = D.tokenizer(prompt, return_tensors='pt', return_offsets_mapping=True)
    span = num_tok_span(prompt, pair, slot, enc['offset_mapping'][0].numpy())
    return int(enc['input_ids'][0, span[0]])


def sweep_uv_plane(bat, P_uv, coord_of, values, pos, bs=EVAL_BS):
    """Set the pre-MLP (u, v2) coordinates to each (y1*, y2*) on a grid and read off which number
    the model then names. Returns (answer logit difference, argmax token id), both (G, G, n).

    The continuous counterpart of the rank-2 interchange. Instead of copying one counterfactual
    example's coordinates, it *sets* the two coordinates to the cloud-mean coordinate of a chosen
    number, over a grid of number pairs, and asks what comes out. That answers two things the
    interchange cannot:

      * **sufficiency, continuously** -- does the model name the *implanted* numbers rather than
        the ones actually written in its prompt, everywhere in the plane or only near the data?
      * **the functional form** -- where the decision boundary falls, and (by a two-way
        decomposition of the resulting field) how much of it is separable in the two coordinates
        versus how much is interaction. A comparison is a threshold on a contrast, so a large
        interaction term is the expected result, not a failure.

    **What is scored, and why it is not the implanted numbers.** The model names a number by
    emitting a token, and it will only ever emit a number that is in its prompt -- 99.5% of
    outputs here are one of the prompt's own two. So "did it say y1*?" is unanswerable: t(y1*)
    never wins. What the implanted coordinates can decide is **which slot** the comparison picks,
    and that is what the mechanism claims: u carries the first-slot number, v2 the second-slot
    number, and the read-out names whichever slot won. The score is therefore the prompt's own
    answer logit difference, `logit[tok_nc] - logit[tok_b]` -- the same quantity position recovery
    is built from, so this sweep and the interchange conditions measure one thing. Positive means
    the first slot won. The prediction is a sign flip across y1* = y2*.

    The diagonal is skipped and left as nan: y1* == y2* is the boundary itself, with no predicted
    sign.

    This is the interchange-intervention logic of Geiger et al. 2021/2023 run as a dose-response
    surface rather than at a single counterfactual point -- the same move as a causal-scrubbing
    style sweep over the hypothesised variable's whole range.
    """
    n, G = bat['n'], len(values)
    toks = [(first_tok_of(v, 0), first_tok_of(v, 1)) for v in values]
    idx_all = torch.arange(n, device=device)
    ld = np.full((G, G, n), np.nan, dtype=np.float32)
    am = np.full((G, G, n), -1, dtype=np.int64)
    for i, y1 in enumerate(values):
        for j, y2 in enumerate(values):
            if y1 == y2:
                continue
            target = (coord_of[y1][0] * P_uv[:, 0] + coord_of[y2][1] * P_uv[:, 1])
            for lo in range(0, n, bs):
                hi = min(lo + bs, n)
                handles = premlp_patch_handles(target.unsqueeze(0).expand(hi - lo, -1), pos, P_uv)
                try:
                    with torch.no_grad():
                        lg = D.model(bat['ids_r'][lo:hi]).logits[:, -1, :].float()
                finally:
                    for h in handles:
                        h.remove()
                _i = idx_all[:hi - lo]
                ld[i, j, lo:hi] = (lg[_i, bat['tok_nc'][lo:hi]]
                                   - lg[_i, bat['tok_b'][lo:hi]]).cpu().numpy()
                am[i, j, lo:hi] = lg.argmax(dim=-1).cpu().numpy()
                del lg
        print(f'  sweep row y1*={y1}: done', flush=True)
    return ld, am, np.array(toks, dtype=np.int64)


def pca_and_probes(cloud, tag, sfx, dirs, idx_tr, idx_te, a_vals, b_vals, out, label):
    """3-component PCA of one residual cloud, a ridge log-magnitude probe per number, and every
    direction of interest re-expressed in that PCA basis. Everything lands in `out` under keys
    namespaced by `tag` (the PCA) and `sfx` (the probes).

    Used for both residual clouds at n2 -- the one leaving layer 14 (`tag='14'`, `sfx=''`) and the
    pre-MLP one (`tag='14pre'`, `sfx='_pre'`) -- so the two are treated identically by
    construction and the pre-MLP panels are comparable with the end-of-layer ones rather than
    being a second, subtly different recipe. `idx_tr` / `idx_te` are shared, so the probes are
    also fitted and scored on the same prompts.

    Probes follow Alain & Bengio 2016 (linear probing), ridge because the target is continuous;
    RidgeCV picks alpha by leave-one-out CV on the training split, so the regularization is not an
    arbitrary constant. The probe direction is the unit-normalized weight vector, and the held-out
    R2 is what says the quantity is linearly present at all.
    """
    pca = PCA(n_components=3).fit(cloud)
    coords = pca.transform(cloud)
    print(f'{label} PCA: EVR {np.round(pca.explained_variance_ratio_, 4).tolist()}', flush=True)
    out.update({f'pca{tag}_coords': coords.astype(np.float32),
                f'pca{tag}_components': pca.components_.astype(np.float32),
                f'pca{tag}_evr': pca.explained_variance_ratio_.astype(np.float64),
                f'pca{tag}_evals': pca.explained_variance_.astype(np.float64),
                f'pca{tag}_mean': pca.mean_.astype(np.float32)})

    probe_dirs = {}
    for lab, vals in (('logA', a_vals), ('logB', b_vals)):
        y = np.log(vals)
        reg = RidgeCV(alphas=RIDGE_ALPHAS).fit(cloud[idx_tr], y[idx_tr])
        w = (reg.coef_ / np.linalg.norm(reg.coef_)).astype(np.float32)
        probe_dirs[lab] = w
        key = f'probe_{lab}{sfx}'
        out[f'{key}_dir'] = w
        out[f'{key}_comp'] = ((cloud - pca.mean_) @ w).astype(np.float64)
        out[f'{key}_stats'] = dict(alpha=float(reg.alpha_),
                                   r2_train=float(reg.score(cloud[idx_tr], y[idx_tr])),
                                   r2_test=float(reg.score(cloud[idx_te], y[idx_te])))
        print(f'  probe {lab}{sfx}: alpha={reg.alpha_:.3g}  R2 train='
              f'{out[f"{key}_stats"]["r2_train"]:.3f}  held-out='
              f'{out[f"{key}_stats"]["r2_test"]:.3f}', flush=True)

    # The arrows for the 3D panels: each direction's cosines with the top-3 PCs, plus how much of
    # its unit norm those three directions hold (which is why some arrows look short).
    allw = list(dirs.items()) + [(f'probe_logA{sfx}', probe_dirs['logA']),
                                 (f'probe_logB{sfx}', probe_dirs['logB'])]
    for key, w in allw:
        proj = np.asarray(w, dtype=np.float64) @ pca.components_.T
        out[f'{key}_in_pca{tag}'] = proj
        out[f'{key}_pca{tag}_captured'] = np.array(np.linalg.norm(proj))
        print(f'  {key:>16} in the {label} PCA basis: {np.round(proj, 4).tolist()}  '
              f'(norm captured {np.linalg.norm(proj):.3f})', flush=True)
    return probe_dirs


def build_handles(spec, sources, lo, hi, pos):
    """Register everything one condition asks for.

    spec['resid'] : list of (layer, 'full'|'sub', P) residual-stream interchanges at `pos`.
    spec['head']  : (head, P or None) -- rewrite that head's own output contribution at `pos`.
    spec['premlp']: P or None -- interchange the pre-MLP residual of layer H_LAYER at `pos`.
    `sources` holds the clean-run activations, sliced here to match the minibatch.
    """
    handles = []
    for layer, kind, P in spec.get('resid', ()):
        src = sources[layer][lo:hi]
        hook = (make_full_patch_hook(src, pos) if kind == 'full'
                else make_subspace_patch_hook(P, src, pos))
        handles.append(D.model.model.layers[layer].register_forward_hook(hook))
    if 'premlp' in spec:
        handles += premlp_patch_handles(sources['premlp'][lo:hi], pos, spec['premlp'])
    if spec.get('head') is not None:
        head, P = spec['head']
        handles += head_patch_handles(head, pos, head_out(sources['ctx'][lo:hi], head), P)
    return handles


def eval_spec(spec, bat, sources, pos, bs=EVAL_BS):
    """Per-example position recovery and IIA for one condition, in forward-pass minibatches."""
    recs, iias = [], []
    for lo in range(0, bat['n'], bs):
        hi = min(lo + bs, bat['n'])
        sub = D._slice_batch(bat, lo, hi)
        handles = build_handles(spec, sources, lo, hi, pos)
        try:
            with torch.no_grad():
                lg = D.model(sub['ids_r']).logits[:, -1, :].float()
        finally:
            for h in handles:
                h.remove()
        r, i = posrec_and_iia(lg, sub)
        recs.append(r)
        iias.append(i)
        del lg
    return np.concatenate(recs), np.concatenate(iias)


# ----------------------------------------------------------------------------------------------
# DAS inside H14's own output space (Geiger et al. 2023, rank-1 single-variable case)
# ----------------------------------------------------------------------------------------------
def train_h14_das(tbat, ctx_clean, pos, init_seed, steps=H14_DAS_STEPS, lr=H14_DAS_LR,
                  bs=H14_DAS_BS):
    """Rank-1 DAS trained on H14's output contribution instead of the L13 residual.

    Same QR-retraction parametrization and same counterfactual-pair cross-entropy objective as
    `data_numberreps.train_das_1d`; only the intervention site differs. Gradients accumulate over
    minibatches so the training-set size is decoupled from the backward pass's activation memory.

    Note the effective search space: only the `head_dim`-dimensional column space of W_O^h is
    reachable -- anything orthogonal to it leaves H14's output untouched -- so this is a 128-d
    search embedded in 3584 dims, not a 3584-d one.
    """
    torch.manual_seed(init_seed)
    das = DASSubspace(d_model, 1, device)
    opt = torch.optim.Adam([das.raw], lr=lr)
    bounds = [(i, min(i + bs, tbat['n'])) for i in range(0, tbat['n'], bs)]
    curve = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for lo, hi in bounds:
            sub = D._slice_batch(tbat, lo, hi)
            # basis() is rebuilt per minibatch: the QR graph is freed by each backward().
            handles = head_patch_handles(H_HEAD, pos, head_out(ctx_clean[lo:hi], H_HEAD),
                                         P=das.basis())
            try:
                with torch.enable_grad():
                    logits = D.model(sub['ids_r']).logits[:, -1, :].float()
            finally:
                for h in handles:
                    h.remove()
            loss = cf_pair_loss(logits, sub) * (sub['n'] / tbat['n'])   # mean over the full batch
            loss.backward()
            total += float(loss)
            del logits
        opt.step()
        curve.append(total)
    torch.cuda.empty_cache()
    with torch.no_grad():
        return das.basis().detach(), np.array(curve, dtype=np.float32)


# ----------------------------------------------------------------------------------------------
# One regime, end to end
# ----------------------------------------------------------------------------------------------
def run_regime(tag):
    t0 = time.time()
    cfg = REGIMES[tag]
    low, high, gap, fit_lo = cfg['low'], cfg['high'], cfg['gap'], cfg['fit_lo']
    print(f'\n{"=" * 92}\n=== regime {tag}: numbers in [{low}, {high}), triple gap {gap}\n{"=" * 92}',
          flush=True)

    T, (POS_A, POS_B), n_tok, example_prompt = regime_positions(low, high)
    print(f'T={T}  first-number last tok @ {POS_A}  second-number last tok @ {POS_B}', flush=True)

    # -- L13 directions come from the numberreps run, not retrained here ------------------------
    npz_path = os.path.join(RESULTS_DIR, f'numberreps_{tag}.npz')
    R = dict(np.load(npz_path))
    RM = json.load(open(os.path.join(RESULTS_DIR, 'meta.json')))['regimes'][tag]['meta']
    assert (RM['pos_first'], RM['pos_second']) == (POS_A, POS_B), \
        'token layout must match the saved numberreps run'

    l13_das_np = R['das_dir'] / np.linalg.norm(R['das_dir'])            # trained @ n1
    l13_pc1b_np = R['pca3_components_b'][0]
    l13_pc1b_np = l13_pc1b_np / np.linalg.norm(l13_pc1b_np)            # PC1 of the n2 cloud
    l13_das = torch.tensor(l13_das_np.astype(np.float32), device=device)
    P_l13b = torch.tensor(l13_pc1b_np.astype(np.float32), device=device).reshape(-1, 1)
    print(f'loaded L13 directions from {npz_path}: cos(L13 DAS, L13 PC1@n2) = '
          f'{float(l13_das_np @ l13_pc1b_np):+.3f}', flush=True)

    out = {}

    # -- 1. One cloud, three read-outs ----------------------------------------------------------
    cloud_pairs = sample_cloud_pairs(N_CLOUD, SEED + 11, low, high)
    cloud_nums = np.array(cloud_pairs, dtype=np.int64)
    a_vals = cloud_nums[:, 0].astype(np.float64)
    b_vals = cloud_nums[:, 1].astype(np.float64)

    h14_cloud = np.zeros((N_CLOUD, d_model), dtype=np.float32)          # H14 output @ n2
    resid14 = np.zeros((N_CLOUD, d_model), dtype=np.float32)            # residual leaving L14 @ n2
    resid14_pre = np.zeros((N_CLOUD, d_model), dtype=np.float32)        # L14 residual pre-MLP @ n2
    l13_das_comp = np.zeros(N_CLOUD, dtype=np.float64)                  # L13 DAS component @ n1
    for i0 in range(0, N_CLOUD, CLOUD_BS):
        chunk = cloud_nums[i0:i0 + CLOUD_BS]
        ids = D.tokenizer([prompt_gen(list(p)) for p in chunk],
                          return_tensors='pt')['input_ids'].to(device)
        assert ids.shape[1] == T, 'cloud prompt length must match the regime template'
        h13, h14, h_pre, h_l14 = capture_cloud(ids, POS_A, POS_B)
        h14_cloud[i0:i0 + len(chunk)] = h14.cpu().numpy()
        resid14[i0:i0 + len(chunk)] = h_l14.cpu().numpy()
        resid14_pre[i0:i0 + len(chunk)] = h_pre.cpu().numpy()
        l13_das_comp[i0:i0 + len(chunk)] = (h13 @ l13_das).cpu().numpy()   # un-centred
        del h13, h14, h_pre, h_l14
        torch.cuda.empty_cache()
    print(f'cloud captured: h14 {h14_cloud.shape}, resid14 {resid14.shape}, '
          f'resid14_pre {resid14_pre.shape}', flush=True)

    out.update(cloud_nums=cloud_nums, a_vals=a_vals, b_vals=b_vals,
               h14_cloud=h14_cloud, resid14=resid14, resid14_pre=resid14_pre,
               l13_das_comp_n1=l13_das_comp,
               l13_das_dir=l13_das_np.astype(np.float32),
               l13_pc1b_dir=l13_pc1b_np.astype(np.float32))

    # -- 1b. H14's attention row at n2 ----------------------------------------------------------
    # The component along u is what the head *writes*; this is what it was *reading* while doing
    # so, on the same prompts. Numbers can span several tokens, so the per-number totals sum the
    # weight over each number's whole token span rather than only its last token.
    attn_row = np.zeros((N_CLOUD, T), dtype=np.float32)
    for i0 in range(0, N_CLOUD, CLOUD_BS):
        chunk = cloud_nums[i0:i0 + CLOUD_BS]
        ids = D.tokenizer([prompt_gen(list(p)) for p in chunk],
                          return_tensors='pt')['input_ids'].to(device)
        attn_row[i0:i0 + len(chunk)] = capture_attn(ids, POS_B).cpu().numpy()
        torch.cuda.empty_cache()
    span_n1 = slice(POS_A - n_tok[0] + 1, POS_A + 1)
    span_n2 = slice(POS_B - n_tok[1] + 1, POS_B + 1)
    out.update(h14_attn_row_n2=attn_row,
               h14_attn_to_n1=attn_row[:, span_n1].sum(axis=1).astype(np.float64),
               h14_attn_to_n2=attn_row[:, span_n2].sum(axis=1).astype(np.float64))
    print(f'H14 attention @n2: to n1 {out["h14_attn_to_n1"].mean():.3f}  '
          f'to n2 {out["h14_attn_to_n2"].mean():.3f}   '
          f'corr(attn->n1, A) = {np.corrcoef(a_vals, out["h14_attn_to_n1"])[0, 1]:+.3f}  '
          f'corr(attn->n1, B) = {np.corrcoef(b_vals, out["h14_attn_to_n1"])[0, 1]:+.3f}', flush=True)

    # -- 2. PCA of H14's own output cloud -------------------------------------------------------
    # The unsupervised rank-1 structure of what H14 writes, at the same rank as the DAS direction
    # trained below -- so the two differ only in how the direction was found.
    pca_h14 = PCA(n_components=3).fit(h14_cloud)
    h14_pc = pca_h14.transform(h14_cloud)
    h14_pc1_np = pca_h14.components_[0].astype(np.float32)
    h14_pc1_np = h14_pc1_np / np.linalg.norm(h14_pc1_np)
    out.update(h14_pca_coords=h14_pc.astype(np.float32),
               h14_pca_components=pca_h14.components_.astype(np.float32),
               h14_pca_evr=pca_h14.explained_variance_ratio_.astype(np.float64),
               h14_pca_evals=pca_h14.explained_variance_.astype(np.float64),
               h14_pca_mean=pca_h14.mean_.astype(np.float32),
               h14_pc1_dir=h14_pc1_np,
               h14_pc1_comp=h14_pc[:, 0].astype(np.float64),
               h14_pc1_comp_raw=(h14_cloud @ h14_pc1_np).astype(np.float64))
    out['logfit_h14_pc1_a'] = log_fit_block(a_vals, h14_pc[:, 0].astype(np.float64), fit_lo)
    out['logfit_h14_pc1_b'] = log_fit_block(b_vals, h14_pc[:, 0].astype(np.float64), fit_lo)
    out['h14_pc1_vs_l13_das_r'] = float(np.corrcoef(l13_das_comp, h14_pc[:, 0])[0, 1])
    print(f'H14 output PCA: EVR {np.round(pca_h14.explained_variance_ratio_, 4).tolist()}',
          flush=True)
    print(f'  corr(H14 PC1, A) = {np.corrcoef(a_vals, h14_pc[:, 0])[0, 1]:+.3f}   '
          f'corr(H14 PC1, B) = {np.corrcoef(b_vals, h14_pc[:, 0])[0, 1]:+.3f}   '
          f'corr(H14 PC1, L13 DAS comp @n1) = {out["h14_pc1_vs_l13_das_r"]:+.3f}', flush=True)

    # -- 3. Counterfactual batches --------------------------------------------------------------
    # Same pool and split as data_numberreps.py, so `eval_triples` is byte-identical to the eval set
    # there and every number in this file sits on the same examples as the ones already saved.
    pool = build_triples(N_EVAL + N_H14_DAS, SEED, low, high, gap)
    eval_triples, das_triples = pool[:N_EVAL], pool[N_EVAL:]
    assert not set(map(tuple, das_triples)) & set(map(tuple, eval_triples)), 'must be held out'

    bat = make_batch(eval_triples, CASE_SLOT[CASE], T)
    bat['pld_clean'], bat['pld_corr'] = pos_anchors(bat)
    tbat = make_batch(das_triples, CASE_SLOT[CASE], T)
    tbat['pld_clean'], tbat['pld_corr'] = pos_anchors(tbat)

    ctx_clean_tr = capture_ctx(tbat['ids_c'], POS_B)

    # -- 4. H14-space DAS direction -------------------------------------------------------------
    Q_h14, h14_curve = train_h14_das(tbat, ctx_clean_tr, POS_B, init_seed=SEED)
    h14_das_np = Q_h14[:, 0].cpu().numpy().astype(np.float32)
    del ctx_clean_tr
    torch.cuda.empty_cache()

    # How much of the learned direction is actually reachable through W_O^h. Anything outside that
    # column space is unconstrained by the objective, so this is the sanity check that the direction
    # means something in H14's output space rather than being an artifact of the ambient 3584 dims.
    _hs, _he = head_slice(H_HEAD)
    Q_col = torch.linalg.qr(W_O[:, _hs:_he])[0]
    reachable = float((Q_col.T @ Q_h14[:, 0]).norm())
    print(f'H14-space DAS: loss {h14_curve[0]:.4f} -> {h14_curve[-1]:.4f}   '
          f'reachable {reachable:.3f} (chance ~ {np.sqrt(head_dim / d_model):.3f})', flush=True)

    # Component of the H14 output cloud along the learned direction, and its log fits vs each number.
    h14_das_comp = ((h14_cloud - h14_cloud.mean(axis=0)) @ h14_das_np).astype(np.float64)
    out.update(h14_das_dir=h14_das_np, h14_das_loss_curve=h14_curve,
               h14_das_comp=h14_das_comp,
               h14_das_comp_raw=(h14_cloud @ h14_das_np).astype(np.float64),
               h14_das_reachable=np.array(reachable),
               h14_das_train_triples=np.array(das_triples, dtype=np.int64))
    out['logfit_h14_das_a'] = log_fit_block(a_vals, h14_das_comp, fit_lo)
    out['logfit_h14_das_b'] = log_fit_block(b_vals, h14_das_comp, fit_lo)
    out['h14_das_vs_l13_das_r'] = float(np.corrcoef(l13_das_comp, h14_das_comp)[0, 1])
    for lab, blk in (('A', out['logfit_h14_das_a']), ('B', out['logfit_h14_das_b'])):
        print(f'log fit, H14 DAS component vs {lab}: all points y={blk["p_all"]:+.3f}*log(x)'
              f'{blk["q_all"]:+.3f}  R2={blk["r2_all"]:.3f}  (pearson r={blk["pearson_r"]:+.3f})',
              flush=True)
    out['h14_das_pc1_cos'] = np.array(float(h14_das_np @ h14_pc1_np))
    print(f'corr(H14 DAS component @n2, L13 DAS component @n1) = '
          f'{out["h14_das_vs_l13_das_r"]:+.3f}   '
          f'cos(H14 DAS, H14 PC1) = {float(out["h14_das_pc1_cos"]):+.3f}', flush=True)

    # -- 5. Residual PCAs + log-magnitude probes, at both layer-14 sites -----------------------
    # Two clouds, one recipe (`pca_and_probes`): the residual leaving layer 14, and the pre-MLP
    # residual one sublayer earlier. The same train/test split serves both, so the two held-out R2
    # values are a like-for-like comparison of how linearly each number is present at each site.
    idx_tr, idx_te = train_test_split(np.arange(N_CLOUD), test_size=PROBE_TEST_FRAC,
                                      random_state=SEED)
    out['probe_train_idx'] = idx_tr.astype(np.int64)
    out['probe_test_idx'] = idx_te.astype(np.int64)
    ARROW_DIRS = {'h14_das': h14_das_np, 'h14_pc1': h14_pc1_np, 'l13_pc1b': l13_pc1b_np}

    probe_dirs = pca_and_probes(resid14, '14', '', ARROW_DIRS, idx_tr, idx_te,
                                a_vals, b_vals, out, 'L14 residual @ n2')
    probe_dirs_pre = pca_and_probes(resid14_pre, '14pre', '_pre', ARROW_DIRS, idx_tr, idx_te,
                                    a_vals, b_vals, out, 'L14 pre-MLP residual @ n2')

    # -- 6. Cosine similarities between the directions ------------------------------------------
    # All of these live in the same d_model residual-stream basis -- H14's output is *added* into
    # that stream, so its direction needs no change of basis to be compared with L13's or a probe's.
    #
    # Both sites' probes are in the matrix. u is written into the stream *before* layer 14's MLP,
    # which (per the rank-2 patching result) is also the site where an intervention in
    # span(u, v2) actually works -- so `probe log A pre` is the read-out point u should be
    # compared against, and the post-MLP probe is the one to compare it against if the claim is
    # about what leaves the layer. Keeping both lets the figure make either comparison and says
    # how much the MLP moved the probes. H14 PC1 rides along as the last row.
    cos_names = ['H14 DAS', 'L13 PC1 @n2', 'probe log A', 'probe log B',
                 'probe log A pre', 'probe log B pre', 'H14 PC1']
    W = np.stack([np.asarray(v, dtype=np.float64) / np.linalg.norm(v)
                  for v in (h14_das_np, l13_pc1b_np, probe_dirs['logA'], probe_dirs['logB'],
                            probe_dirs_pre['logA'], probe_dirs_pre['logB'], h14_pc1_np)])
    out['cos_dirs'] = W.astype(np.float32)
    out['cos_matrix'] = W @ W.T
    print(f'cosine matrix (chance ~ {1 / np.sqrt(d_model):.3f}):', flush=True)
    for nm, row in zip(cos_names, out['cos_matrix']):
        print(f'  {nm:>16}  {np.round(row, 3).tolist()}', flush=True)
    _ia, _ib = cos_names.index('probe log A'), cos_names.index('probe log B')
    _pa, _pb = cos_names.index('probe log A pre'), cos_names.index('probe log B pre')
    _deg = lambda i, j: np.degrees(np.arccos(np.clip(out['cos_matrix'][i, j], -1, 1)))
    print(f'  probe pair: post-MLP {out["cos_matrix"][_ia, _ib]:+.3f} ({_deg(_ia, _ib):.1f} deg)'
          f'   pre-MLP {out["cos_matrix"][_pa, _pb]:+.3f} ({_deg(_pa, _pb):.1f} deg)', flush=True)
    print(f'  u vs its own probe: post-MLP {out["cos_matrix"][0, _ia]:+.3f}'
          f'   pre-MLP {out["cos_matrix"][0, _pa]:+.3f}   '
          f'|  MLP moved the probes: cos(pA, pA_pre) = {out["cos_matrix"][_ia, _pa]:+.3f}, '
          f'cos(pB, pB_pre) = {out["cos_matrix"][_ib, _pb]:+.3f}', flush=True)

    # -- 7. Clean-run sources for every patching condition --------------------------------------
    ch13 = clean_hidden_at(bat['ids_c'], LAYER)
    ch14 = clean_hidden_at(bat['ids_c'], H_LAYER)
    ctx_clean = capture_ctx(bat['ids_c'], POS_B)
    ch_pre14 = capture_premlp(bat['ids_c'], POS_B)
    sources = {LAYER: ch13, H_LAYER: ch14, 'ctx': ctx_clean, 'premlp': ch_pre14}

    out['eval_triples'] = np.array(eval_triples, dtype=np.int64)
    out['pld_clean'] = bat['pld_clean'].cpu().numpy()
    out['pld_corr'] = bat['pld_corr'].cpu().numpy()

    # -- 8. The headline IIA conditions ---------------------------------------------------------
    L13_PC1 = (LAYER, 'sub', P_l13b)
    L13_FULL = (LAYER, 'full', None)
    # The plane the shared representation is claimed to live in: span(u, v2), orthonormalized by
    # QR (the two are near-orthogonal already, so the second column stays essentially v2).
    # Patching the L14 residual inside it is the rank-2 version of the `L14 full` ceiling -- 2 of
    # d_model directions -- and the rank-1 halves are its controls. `preMLP` runs the same patch
    # one sublayer earlier, where H14 has just written and layer 14's own MLP can still read it.
    P_uv = torch.linalg.qr(torch.tensor(np.stack([h14_das_np, l13_pc1b_np]).T,
                                        device=device))[0]                        # (d_model, 2)
    P_u = torch.tensor(h14_das_np, device=device).reshape(-1, 1)
    out['uv_plane_basis'] = P_uv.cpu().numpy().astype(np.float32)
    CONDITIONS = {
        'L13 full @n2':            dict(resid=[L13_FULL]),
        'L13 PC1 @n2':             dict(resid=[L13_PC1]),
        'H14 DAS':                 dict(head=(H_HEAD, Q_h14)),
        'H14 DAS + L13 PC1 @n2':   dict(resid=[L13_PC1], head=(H_HEAD, Q_h14)),
        'L14 u @n2':               dict(resid=[(H_LAYER, 'sub', P_u)]),
        'L14 v2 @n2':              dict(resid=[(H_LAYER, 'sub', P_l13b)]),
        'L14 uv plane @n2':        dict(resid=[(H_LAYER, 'sub', P_uv)]),
        'L14 full @n2':            dict(resid=[(H_LAYER, 'full', None)]),
        'L14 uv plane preMLP @n2': dict(premlp=P_uv),
        'L14 full preMLP @n2':     dict(premlp=None),
    }
    print(f'\n--- conditions ({CASE}, patched @ second-number position, n={N_EVAL})', flush=True)
    for name, spec in CONDITIONS.items():
        rec, iia = eval_spec(spec, bat, sources, POS_B)
        key = name.replace(' ', '_').replace('+', 'and').replace('@', 'at')
        out[f'cond_posrec_{key}'] = rec
        out[f'cond_iia_{key}'] = iia
        print(f'{name:>22}: position recovery {rec.mean():6.3f} +/- '
              f'{rec.std() / np.sqrt(len(rec)):.3f}   IIA {iia.mean():.3f}', flush=True)
    out['condition_names'] = np.array(list(CONDITIONS))

    # -- 8b. Dose-response sweep of the (u, v2) plane, at the pre-MLP site ----------------------
    # The rank-2 interchange above says the plane is sufficient at one counterfactual point per
    # example. This asks the same question as a surface: set the two coordinates to the values a
    # chosen pair of numbers would put there, over a grid of pairs, and see which number the model
    # names. Run pre-MLP because that is the only site the rank-2 patch works at.
    #
    # The number -> coordinate map is the cloud mean over a +/- SWEEP_WIN window of that number,
    # marginalizing the partner. Coordinates are raw (uncentred) so the patch target is directly a
    # residual-stream vector. At this site a per-number mean is a good description of each
    # coordinate (R2 ~ 0.87 for u vs y1, ~0.90 for v2 vs y2); at the end of layer 14 it is not,
    # which is the other reason the sweep lives here.
    sweep_vals = sweep_grid_values(low, high)
    cu_raw = resid14_pre @ P_uv[:, 0].cpu().numpy()
    cv_raw = resid14_pre @ P_uv[:, 1].cpu().numpy()
    coord_of = {}
    for v in sweep_vals:
        ma = np.abs(a_vals - v) <= SWEEP_WIN
        mb = np.abs(b_vals - v) <= SWEEP_WIN
        coord_of[v] = (float(cu_raw[ma].mean()), float(cv_raw[mb].mean()))
    sweep_bat = D._slice_batch(bat, 0, min(N_SWEEP, N_EVAL))
    print(f'\n--- (u, v2) plane sweep, pre-MLP @n2: {len(sweep_vals)}x{len(sweep_vals)} grid '
          f'x {sweep_bat["n"]} prompts', flush=True)
    sweep_ld, sweep_am, sweep_toks = sweep_uv_plane(sweep_bat, P_uv, coord_of, sweep_vals, POS_B)
    out.update(sweep_values=np.array(sweep_vals, dtype=np.int64),
               sweep_coords=np.array([coord_of[v] for v in sweep_vals], dtype=np.float64),
               sweep_tokens=sweep_toks, sweep_ld_slot=sweep_ld, sweep_argmax=sweep_am,
               sweep_tok_prompt=sweep_bat['tok_b'].cpu().numpy(),
               sweep_tok_prompt_nc=sweep_bat['tok_nc'].cpu().numpy())
    # Headline read-outs: does the model name the implanted larger number, and is the boundary the
    # y1* = y2* diagonal?
    _m = np.nanmean(sweep_ld, axis=2)
    _G = len(sweep_vals)
    _off = [(i, j) for i in range(_G) for j in range(_G) if i != j]
    _follow = np.mean([(sweep_am[i, j] == (sweep_toks[i, 0] if sweep_vals[i] > sweep_vals[j]
                                           else sweep_toks[j, 1])).mean() for i, j in _off])
    _slot1 = np.mean([(sweep_am[i, j] == sweep_bat['tok_nc'].cpu().numpy()[None, :]).mean()
                      for i, j in _off if sweep_vals[i] > sweep_vals[j]])
    _sign_ok = np.mean([float((_m[i, j] > 0) == (sweep_vals[i] > sweep_vals[j])) for i, j in _off])
    print(f'  answer logit difference: sign matches "y1* > y2*" on {_sign_ok:.3f} of cells; '
          f'mean {np.mean([_m[i, j] for i, j in _off if sweep_vals[i] > sweep_vals[j]]):+.2f} '
          f'above the diagonal vs '
          f'{np.mean([_m[i, j] for i, j in _off if sweep_vals[i] < sweep_vals[j]]):+.2f} below',
          flush=True)
    print(f'  (the implanted numbers\' own tokens win {_follow:.3f} of the time -- they are not '
          f'in the prompt, so this is expected to be ~0 and is not the read-out)', flush=True)
    print(f'  names the first-slot number when y1* > y2*: {_slot1:.3f}', flush=True)
    del sweep_ld, sweep_am

    # -- 9. Per-head sweep over all of layer 14's heads ------------------------------------------
    # Each head's own output contribution is fully interchanged at n2. `alone` is the head by
    # itself; the two joint arms add an L13 interchange at the same position, because H14 writes on
    # top of whatever L13 already holds there -- the joint number is what says the head's write is
    # sufficient given the right substrate.
    JOINT = {'alone': [], 'with_L13_PC1_n2': [L13_PC1], 'with_L13_full_n2': [L13_FULL]}
    print(f'\n--- per-head sweep, layer {H_LAYER}, {n_heads} heads ({CASE}, @n2)', flush=True)
    for arm, resid in JOINT.items():
        rec_m, rec_s, iia_m = np.zeros(n_heads), np.zeros(n_heads), np.zeros(n_heads)
        rec_all = np.zeros((n_heads, N_EVAL), dtype=np.float32)
        iia_all = np.zeros((n_heads, N_EVAL), dtype=np.float32)
        for h in range(n_heads):
            rec, iia = eval_spec(dict(resid=resid, head=(h, None)), bat, sources, POS_B)
            rec_all[h], iia_all[h] = rec, iia
            rec_m[h], rec_s[h], iia_m[h] = rec.mean(), rec.std() / np.sqrt(len(rec)), iia.mean()
        out[f'head_posrec_{arm}'] = rec_all
        out[f'head_iia_{arm}'] = iia_all
        top = np.argsort(-iia_m)[:5]
        print(f'{arm:>18}: best heads by IIA ' +
              '  '.join(f'H{h}={iia_m[h]:.3f}' for h in top), flush=True)
        print(f'{"":>18}  H{H_HEAD} posrec {rec_m[H_HEAD]:+.3f} +/- {rec_s[H_HEAD]:.3f}   '
              f'IIA {iia_m[H_HEAD]:.3f}', flush=True)
    out['head_arms'] = np.array(list(JOINT))

    # -- 9b. The same conditions and sweep in the other case -----------------------------------
    # Everything above is measured on `n1 larger`, the case v1 was trained in. This repeats the
    # headline conditions and the plane sweep on `ALT_CASE` prompts while REUSING Q_h14, P_l13b
    # and P_uv exactly as found -- nothing is refitted, so it asks whether the directions found in
    # one case still do anything in the other, which is the stronger claim.
    #
    # Read the result with the counterfactual design in mind. `make_batch` varies only the number
    # at the winning slot, so in `n2 larger` the FIRST number is identical in the clean and
    # corrupted prompts. H14 reads the first number, so its output at n2 is the same in both runs
    # and the v1 patch writes back a value it already had. A near-zero v1 bar here is therefore a
    # specificity control -- "v1 does nothing when the information it carries has not changed" --
    # and not evidence that v1 is inert when the second number wins. See docs/data_sharedrep.md
    # for the batch design that would test that.
    if ALT_CASE:
        print(f'\n{"=" * 92}\n=== alt case {ALT_CASE}: same directions, nothing refitted\n'
              f'{"=" * 92}', flush=True)
        abat = make_batch(eval_triples, CASE_SLOT[ALT_CASE], T)
        abat['pld_clean'], abat['pld_corr'] = pos_anchors(abat)
        a_ch13 = clean_hidden_at(abat['ids_c'], LAYER)
        a_ch14 = clean_hidden_at(abat['ids_c'], H_LAYER)
        a_ctx = capture_ctx(abat['ids_c'], POS_B)
        a_pre = capture_premlp(abat['ids_c'], POS_B)
        a_sources = {LAYER: a_ch13, H_LAYER: a_ch14, 'ctx': a_ctx, 'premlp': a_pre}
        out['alt_pld_clean'] = abat['pld_clean'].cpu().numpy()
        out['alt_pld_corr'] = abat['pld_corr'].cpu().numpy()
        for name, spec in CONDITIONS.items():
            rec, iia = eval_spec(spec, abat, a_sources, POS_B)
            key = name.replace(' ', '_').replace('+', 'and').replace('@', 'at')
            out[f'alt_cond_posrec_{key}'] = rec
            out[f'alt_cond_iia_{key}'] = iia
            print(f'{name:>26}: position recovery {rec.mean():6.3f} +/- '
                  f'{rec.std() / np.sqrt(len(rec)):.3f}   IIA {iia.mean():.3f}', flush=True)

        a_sweep_bat = D._slice_batch(abat, 0, min(N_SWEEP, N_EVAL))
        print(f'\n--- (v1, v2) plane sweep, pre-MLP @n2, {ALT_CASE}: '
              f'{len(sweep_vals)}x{len(sweep_vals)} grid x {a_sweep_bat["n"]} prompts', flush=True)
        a_ld, a_am, _ = sweep_uv_plane(a_sweep_bat, P_uv, coord_of, sweep_vals, POS_B)
        out.update(alt_sweep_ld_slot=a_ld, alt_sweep_argmax=a_am,
                   alt_sweep_tok_prompt=a_sweep_bat['tok_b'].cpu().numpy(),
                   alt_sweep_tok_prompt_nc=a_sweep_bat['tok_nc'].cpu().numpy())
        _am2 = np.nanmean(a_ld, axis=2)
        _ok2 = np.mean([float((_am2[i, j] > 0) == (sweep_vals[i] > sweep_vals[j]))
                        for i, j in _off])
        print(f'  answer logit difference: sign matches "y1* > y2*" on {_ok2:.3f} of cells',
              flush=True)
        del a_ch13, a_ch14, a_ctx, a_pre, a_sources, a_ld, a_am
        torch.cuda.empty_cache()

    del ch13, ch14, ctx_clean, ch_pre14, sources
    torch.cuda.empty_cache()

    out['meta'] = dict(tag=tag, low=low, high=high, gap=gap, fit_lo=fit_lo, T=T,
                       pos_first=int(POS_A), pos_second=int(POS_B), n_tok_per_number=n_tok,
                       example_prompt=example_prompt, case=CASE, alt_case=ALT_CASE,
                       h_layer=H_LAYER, h_head=H_HEAD, n_heads=n_heads, head_dim=head_dim,
                       l13_layer=LAYER, d_model=d_model, model=D.MODELNAME, seed=SEED,
                       n_cloud=N_CLOUD, n_eval=N_EVAL, n_h14_das=N_H14_DAS,
                       h14_das_steps=H14_DAS_STEPS, h14_das_lr=H14_DAS_LR,
                       h14_das_bs=H14_DAS_BS, h14_das_reachable=reachable,
                       cos_names=cos_names,
                       condition_names=list(CONDITIONS), head_arms=list(JOINT),
                       probe_test_frac=PROBE_TEST_FRAC,
                       l13_dirs_from=npz_path,
                       runtime_s=round(time.time() - t0, 1))
    print(f'\nregime {tag} done in {out["meta"]["runtime_s"]:.1f} s', flush=True)
    return out


# ----------------------------------------------------------------------------------------------
# Run and save
# ----------------------------------------------------------------------------------------------
def save_regime(tag, res):
    """Split the result dict: arrays -> .npz, scalars/dicts -> the returned json-able block."""
    arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    path = os.path.join(RESULTS_DIR, f'sharedrep_{tag}.npz')
    np.savez_compressed(path, **arrays)
    return path, arrays, scalars


if __name__ == '__main__':
    D.load_model()
    init_head_machinery()

    all_meta = dict(model=D.MODELNAME, l13_layer=LAYER, h_layer=H_LAYER, h_head=H_HEAD,
                    seed=SEED, d_model=d_model, regimes={})
    sizes = {}

    for tag in TAGS:
        res = run_regime(tag)
        path, arrays, scalars = save_regime(tag, res)
        all_meta['regimes'][tag] = scalars
        sizes[tag] = dict(path=path,
                          on_disk_MB=os.path.getsize(path) / 1e6,
                          in_memory_MB=sum(a.nbytes for a in arrays.values()) / 1e6,
                          arrays={k: dict(shape=list(a.shape), dtype=str(a.dtype),
                                          MB=a.nbytes / 1e6)
                                  for k, a in arrays.items()})
        del res
        gc.collect()
        torch.cuda.empty_cache()

    meta_path = os.path.join(RESULTS_DIR, 'sharedrep_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(all_meta, f, indent=2, default=float)

    print(f'\n{"=" * 92}\n=== RESULT SIZES\n{"=" * 92}', flush=True)
    total_disk = 0.0
    for tag, s in sizes.items():
        print(f'\n--- {s["path"]}  ({len(s["arrays"])} arrays)', flush=True)
        for k, a in sorted(s['arrays'].items(), key=lambda kv: -kv[1]['MB']):
            print(f'    {k:<34} {str(a["shape"]):>16} {a["dtype"]:>10}  {a["MB"]:9.3f} MB',
                  flush=True)
        print(f'    {"TOTAL uncompressed":<34} {"":>16} {"":>10}  {s["in_memory_MB"]:9.3f} MB',
              flush=True)
        print(f'    {"ON DISK (npz, compressed)":<34} {"":>16} {"":>10}  {s["on_disk_MB"]:9.3f} MB',
              flush=True)
        total_disk += s['on_disk_MB']

    meta_MB = os.path.getsize(meta_path) / 1e6
    total_disk += meta_MB
    print(f'\n{meta_path}: {meta_MB:.3f} MB', flush=True)
    print(f'\nTOTAL new results on disk: {total_disk:.3f} MB', flush=True)
    print('\ndone.', flush=True)
