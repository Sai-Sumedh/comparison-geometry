"""Collect everything the comparator analysis needs, with the (v1, v2) plane as the base patch.

Companion to `data_sharedrep.py`, and the batch version of `explore_comparatorfig.ipynb`. The one
substantive change from the notebook is the **base intervention**: every downstream measurement
here sits on top of a rank-2 interchange of the pre-MLP L14 residual inside span(v1, v2), rather
than on the pair of rank-1 patches (v1 inside H14's output, v2 in the L13 residual) the notebook
used.

Why that matters. The claim being tested is "v1 and v2 are the variables, they are summed into one
plane at n2, and the comparator reads that plane". The rank-1 pair establishes the first two legs
-- each direction moves the answer exactly when its own number changes, and the residual stream is
a sum. But it never sets the *summed* coordinate to its counterfactual value: it sets two upstream
sources, whose effects then propagate through L14 attention (which can rewrite n2, and which also
changes what H14 itself reads). The rank-2 pre-MLP patch sets the plane coordinate directly, at the
consumer's input, and reaches 0.81/0.95 IIA against a 0.82/0.99 full-rank ceiling at the same site,
where the rank-1 pair reaches 0.68/0.88. Since attribution scores are gradients evaluated *at the
patched state*, the base run should be the one that actually realises the variable in the claim.

Everything is computed for **both** counterfactual cases, and the case is the perturbation:
`make_batch` varies only the winning slot's number, so `n1 larger` perturbs y1 and `n2 larger`
perturbs y2.

What is computed:
  1. Headline conditions: the plane patch, its rank-1 halves, the rank-1 pair, and the full-rank
     ceilings, at the pre-MLP and post-MLP L14 sites.
  2. Module freezes (MLP / attention, L14-L16) under the plane patch.
  3. Per-neuron attribution over MLP14 + MLP15 (37888 neurons), base = the plane patch.
  4. Top-k freeze curve with random-k and bottom-k controls.
  5. Top 10 / 20 / 30 per case, their overlaps, and the IIA of freezing each overlap set.
  6. Clean-MLP injection (the sufficiency direction), scored on both tok_nc and tok_a.
  7. The cloud in the plane: coordinates, neuron activations, receptive fields on a dense grid.
  8. Read-in and write-out geometry: in-plane components, angles, and the SVD of the read-ins.
  9. Connectivity: L14 -> L15 weight paths and causal edges.
 10. Gate preactivation boundaries in the plane.
 11. DAS at the L15 residual, ranks 1-3, and DAS inside the MLP neuron space.

Outputs -> results/comparator_data_{tag}.npz and results/comparator_data_meta.json.
Nothing is plotted here; `makefig_comparator.ipynb` loads these and draws the figures.

See docs/data_comparator.md.
"""

import os
import json
import time

import numpy as np
import torch

import data_numberreps as D
import data_sharedrep as S
from data_numberreps import (build_triples, make_batch, regime_positions, pos_anchors,
                             posrec_and_iia, _slice_batch, CASE_SLOT, SEED, LAYER, REGIMES,
                             EVAL_BS, N_EVAL, device)

# ----------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------
RESULTS_DIR = 'results'
TAG = '2digit'
CASES = ['n1 larger', 'n2 larger']
H_LAYER = S.H_LAYER                 # 14
NEURON_LAYERS = [14, 15]            # the neuron search pool; L16 has no module-level effect
SWEEP_LAYERS = [14, 15, 16]         # layers whose whole MLP / attention are frozen

ATP_BS, ATP_SCALE = 16, 100.0       # attribution minibatch, fp16 loss scaling
K_GRID = [1, 3, 10, 30, 100, 300, 1000, 3000]
RAND_SEEDS = [0, 1]
OVERLAP_KS = [10, 20, 30]           # depths intersected between the two cases
N_PROFILE = 30                      # neurons profiled individually (receptive fields, geometry)
assert N_PROFILE in (10, 20, 30), 'N_PROFILE must be one of OVERLAP_KS'

N_CLOUD, CLOUD_BS = 1500, EVAL_BS   # cloud for plane coordinates and tuning
RF_STEP = 4                         # grid stride for receptive fields
N_ALIGN_BG = 500                    # random neurons for the alignment / SVD null
SVD_NBOOT = 4000

INJ_LAYERS = [[14], [14, 15], [14, 15, 16]]

DAS15_LAYER, DAS15_RANKS = 15, [1, 2, 3]
DAS15_STEPS, DAS15_LR, DAS15_BS, DAS15_N = 100, 0.05, 32, 128
MLPDAS_RANKS, MLPDAS_N = [1, 2], 64

os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(SEED)
torch.random.manual_seed(SEED)

ck = (lambda s: s.replace(' ', '_'))


# ----------------------------------------------------------------------------------------------
# Neuron-level machinery (the residual / head / pre-MLP machinery is imported from data_sharedrep)
# ----------------------------------------------------------------------------------------------
def mlp_of(layer):
    return D.model.model.layers[layer].mlp


def block_of(site):
    """('mlp'|'attn', layer) -> the sub-block whose output a module freeze pins."""
    kind, layer = site
    m = D.model.model.layers[layer]
    return m.mlp if kind == 'mlp' else m.self_attn


def make_out_patch_hook(target, pos):
    """Forward hook pinning a sub-block's output at `pos` to `target` (B, d_model).

    Used both ways: `target` from the corrupted run is a freeze (block it from passing the patch
    on), `target` from the clean run is an injection (hand the rest of the model that output).
    """
    def _hook(mod, inp, out, _p=pos, _t=target):
        hs = out[0] if isinstance(out, tuple) else out
        new = hs.clone()
        new[:, _p, :] = _t.to(hs.dtype)
        return ((new,) + tuple(out[1:])) if isinstance(out, tuple) else new
    return _hook


def make_neuron_freeze_hook(idx, vals, pos):
    """`down_proj` pre-hook pinning neurons `idx` at `pos` to `vals` (B, len(idx)).

    Writes in place: the `down_proj` input is `act_fn(gate(x)) * up(x)`, freshly allocated by the
    MLP's forward and consumed by nothing else, so the mutation is invisible elsewhere. Only valid
    under `no_grad`; the DAS version below clones.
    """
    def _pre(mod, inp, _i=idx, _v=vals, _p=pos):
        inp[0][:, _p, _i] = _v.to(inp[0].dtype)
    return _pre


def make_neuron_das_hook(idx, a_clean, P, pos):
    """`down_proj` pre-hook, rank-r interchange inside the activations of `idx`. P None -> full."""
    def _pre(mod, inp, _i=idx, _a=a_clean, _P=P, _p=pos):
        x = inp[0].clone()
        cur = x[:, _p, _i].float()
        x[:, _p, _i] = (_a if _P is None else cur + ((_a - cur) @ _P) @ _P.T).to(x.dtype)
        return (x,) + tuple(inp[1:])
    return _pre


def capture_neurons(ids, layers, pos, bs=EVAL_BS, idx=None, patch=None):
    """{layer: (n, d_ff or len(idx))} post-SwiGLU activations at `pos` -- the `down_proj` input."""
    store, chunks = {}, {l: [] for l in layers}

    def mk(l):
        def _pre(mod, inp):
            a = inp[0][:, pos, :].detach().float()
            store[l] = a if idx is None else a[:, idx[l]]
        return _pre

    hs = [mlp_of(l).down_proj.register_forward_pre_hook(mk(l)) for l in layers]
    try:
        for lo in range(0, ids.shape[0], bs):
            hi = min(lo + bs, ids.shape[0])
            extra = patch(lo, hi) if patch is not None else []
            try:
                with torch.no_grad():
                    _ = D.model(ids[lo:hi])
            finally:
                for h in extra:
                    h.remove()
            for l in layers:
                chunks[l].append(store.pop(l))
    finally:
        for h in hs:
            h.remove()
    return {l: torch.cat(v, dim=0) for l, v in chunks.items()}


def capture_sites(sites, ids, pos, bs=EVAL_BS):
    """{site: (n, d_model)} sub-block output at `pos`, all sites in one pass."""
    store, chunks = {}, {s: [] for s in sites}

    def mk(site):
        def _hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            store[site] = hs[:, pos, :].detach().float()
        return _hook

    hs = [block_of(s).register_forward_hook(mk(s)) for s in sites]
    try:
        for lo in range(0, ids.shape[0], bs):
            with torch.no_grad():
                _ = D.model(ids[lo:lo + bs])
            for s in sites:
                chunks[s].append(store.pop(s))
    finally:
        for h in hs:
            h.remove()
    return {s: torch.cat(v, dim=0) for s, v in chunks.items()}


def eval_cond(spec, bat, sources, pos, extra=None, bs=EVAL_BS):
    """Per-example position recovery and IIA for one condition.

    `spec` goes to `data_sharedrep.build_handles` (resid / head / premlp); `extra(lo, hi)` returns
    any further handles, which is how module freezes, neuron freezes and injections are added.
    """
    recs, iias = [], []
    for lo in range(0, bat['n'], bs):
        hi = min(lo + bs, bat['n'])
        sub = _slice_batch(bat, lo, hi)
        handles = S.build_handles(spec, sources, lo, hi, pos) if spec else []
        if extra is not None:
            handles += extra(lo, hi)
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


def train_das_resid(layer, pos, k, bat, ch, seed, steps=DAS15_STEPS, lr=DAS15_LR, bs=DAS15_BS):
    """Rank-k DAS on the residual leaving `layer` at `pos`.

    Generalises `data_numberreps.train_das_1d` (pinned to L13, rank 1) and
    `data_sharedrep.train_h14_das` (a head's output space): same QR retraction, same
    counterfactual-pair objective, gradients accumulated over minibatches.
    """
    torch.manual_seed(seed)
    das = D.DASSubspace(D.d_model, k, device)
    opt = torch.optim.Adam([das.raw], lr=lr)
    bounds = [(i, min(i + bs, bat['n'])) for i in range(0, bat['n'], bs)]
    curve = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for lo, hi in bounds:
            sub = _slice_batch(bat, lo, hi)
            hnd = D.model.model.layers[layer].register_forward_hook(
                D.make_subspace_patch_hook(das.basis(), ch[lo:hi], pos))
            try:
                with torch.enable_grad():
                    lg = D.model(sub['ids_r']).logits[:, -1, :].float()
            finally:
                hnd.remove()
            loss = D.cf_pair_loss(lg, sub) * (sub['n'] / bat['n'])
            loss.backward()
            tot += float(loss)
            del lg
        opt.step()
        curve.append(tot)
    torch.cuda.empty_cache()
    with torch.no_grad():
        return das.basis().detach(), np.array(curve, dtype=np.float32)


def train_das_neurons(layer, idx, a_clean, r, bat, pos, seed,
                      steps=DAS15_STEPS, lr=DAS15_LR, bs=DAS15_BS):
    """Rank-r DAS inside the k-dimensional activation space of neurons `idx`."""
    torch.manual_seed(seed)
    das = D.DASSubspace(len(idx), r, device)
    opt = torch.optim.Adam([das.raw], lr=lr)
    bounds = [(i, min(i + bs, bat['n'])) for i in range(0, bat['n'], bs)]
    curve = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for lo, hi in bounds:
            sub = _slice_batch(bat, lo, hi)
            hnd = mlp_of(layer).down_proj.register_forward_pre_hook(
                make_neuron_das_hook(idx, a_clean[lo:hi], das.basis(), pos))
            try:
                with torch.enable_grad():
                    lg = D.model(sub['ids_r']).logits[:, -1, :].float()
            finally:
                hnd.remove()
            loss = D.cf_pair_loss(lg, sub) * (sub['n'] / bat['n'])
            loss.backward()
            tot += float(loss)
            del lg
        opt.step()
        curve.append(tot)
    torch.cuda.empty_cache()
    with torch.no_grad():
        return das.basis().detach(), np.array(curve, dtype=np.float32)


def das15_write_alignment(bases, layer=DAS15_LAYER):
    """How much of each MLP`layer` neuron's write lies inside each saved DAS subspace.

    Neuron j adds `a_j * W_down[:, j]` to the residual, and the DAS basis Q spans a subspace of
    that same residual, so the two are directly comparable with no change of basis and no gamma
    folding (nothing normalises between `down_proj` and the residual add -- unlike the read side).

        rho_j = ||Q^T w_j|| / ||w_j||        (= |cos| for rank 1)

    Computed for **every** neuron in the layer, so the ones of interest can be placed in their own
    empirical null; the analytic sqrt(r/d) chance level is not trustworthy here because down_proj
    columns are not isotropic. `bases` is {(case, k): Q}.
    """
    Wd = mlp_of(layer).down_proj.weight.detach().float()          # (d_model, d_ff), col = write
    nrm = Wd.norm(dim=0)
    got = {}
    for (case, k), Q in bases.items():
        got[f'dasfrac_L{layer}_{ck(case)}_k{k}'] = (
            (Q.T @ Wd).norm(dim=0) / nrm).cpu().numpy().astype(np.float32)
    return got


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def run():
    t0 = time.time()
    out, meta = {}, {}
    cfg = REGIMES[TAG]
    low, high, gap = cfg['low'], cfg['high'], cfg['gap']
    T, (POS_A, POS_B), _n_tok, _p = regime_positions(low, high)

    R = dict(np.load(f'{RESULTS_DIR}/sharedrep_{TAG}.npz'))
    M = json.load(open(f'{RESULTS_DIR}/sharedrep_meta.json'))['regimes'][TAG]['meta']
    assert (POS_A, POS_B) == (M['pos_first'], M['pos_second']), 'token layout must match'
    d_model, d_ff = D.d_model, D.model.config.intermediate_size
    print(f'T={T}  n2={POS_B}  d_model={d_model}  d_ff={d_ff}', flush=True)

    v1 = R['h14_das_dir'] / np.linalg.norm(R['h14_das_dir'])        # H14 DAS, carries y1 at n2
    v2 = R['l13_pc1b_dir'] / np.linalg.norm(R['l13_pc1b_dir'])      # L13 PC1 @ n2, carries y2
    Q_v1 = torch.tensor(v1.astype(np.float32), device=device).reshape(-1, 1)
    P_v2 = torch.tensor(v2.astype(np.float32), device=device).reshape(-1, 1)
    # The plane, orthonormalised exactly as data_sharedrep does it, so the two files agree.
    P_uv = torch.linalg.qr(torch.tensor(np.stack([v1, v2]).T.astype(np.float32),
                                        device=device))[0]           # (d_model, 2)
    out['v1'], out['v2'] = v1.astype(np.float32), v2.astype(np.float32)
    out['uv_plane_basis'] = P_uv.cpu().numpy().astype(np.float32)

    # The display frame: e1 = v1, e2 = v2 orthogonalised against it (makefig_sharedreps Fig 6).
    e1 = torch.tensor(v1.astype(np.float32), device=device)
    e2 = torch.tensor(v2.astype(np.float32), device=device)
    e2 = e2 - (e2 @ e1) * e1
    e2 = e2 / e2.norm()

    BASE = dict(premlp=P_uv)                     # <- the base patch for everything downstream
    ARMS = {'v1 (H14 DAS)':        dict(head=(S.H_HEAD, Q_v1)),
            'v2 (L13 PC1 @n2)':    dict(resid=[(LAYER, 'sub', P_v2)]),
            'v1 & v2 separate':    dict(resid=[(LAYER, 'sub', P_v2)], head=(S.H_HEAD, Q_v1)),
            'uv plane preMLP':     BASE,
            'uv plane postMLP':    dict(resid=[(H_LAYER, 'sub', P_uv)]),
            'L14 full preMLP':     dict(premlp=None),
            'L14 full postMLP':    dict(resid=[(H_LAYER, 'full', None)])}

    # -- batches, sources, and the activations every later block reuses --------------------------
    pool = build_triples(N_EVAL + S.N_H14_DAS, SEED, low, high, gap)
    eval_triples, rank_triples = pool[:N_EVAL], pool[N_EVAL:]
    out['eval_triples'] = np.array(eval_triples, dtype=np.int64)
    out['rank_triples'] = np.array(rank_triples, dtype=np.int64)

    SITES = [(k, l) for l in SWEEP_LAYERS for k in ('mlp', 'attn')]
    BAT, SRC, FRZ, NEUR, TOKA = {}, {}, {}, {}, {}
    for case in CASES:
        b = make_batch(eval_triples, CASE_SLOT[case], T)
        b['pld_clean'], b['pld_corr'] = pos_anchors(b)
        BAT[case] = b
        SRC[case] = {LAYER: S.clean_hidden_at(b['ids_c'], LAYER),
                     H_LAYER: S.clean_hidden_at(b['ids_c'], H_LAYER),
                     'ctx': S.capture_ctx(b['ids_c'], POS_B),
                     'premlp': S.capture_premlp(b['ids_c'], POS_B)}
        FRZ[case] = capture_sites(SITES, b['ids_r'], POS_B)
        NEUR[case] = capture_neurons(b['ids_r'], NEURON_LAYERS, POS_B)
        # the clean run's winning number, for the injection's second metric
        strs = [D.prompt_gen(list(p)) for p in b['clean_pairs']]
        enc = D.tokenizer(strs, return_tensors='pt', return_offsets_mapping=True)
        TOKA[case] = torch.tensor(
            [int(enc['input_ids'][i, D.num_tok_span(strs[i], list(b['clean_pairs'][i]),
                                                    b['win_slot'],
                                                    enc['offset_mapping'][i].numpy())[0]])
             for i in range(b['n'])], device=device)
        out[f'tok_a_{ck(case)}'] = TOKA[case].cpu().numpy()
        print(f'{case}: batch and sources ready ({time.time() - t0:.0f} s)', flush=True)

    # -- 1. headline conditions ------------------------------------------------------------------
    print('\n--- 1. headline conditions', flush=True)
    for case in CASES:
        for name, spec in ARMS.items():
            rec, iia = eval_cond(spec, BAT[case], SRC[case], POS_B)
            out[f'cond_posrec_{ck(case)}_{ck(name)}'] = rec.astype(np.float32)
            out[f'cond_iia_{ck(case)}_{ck(name)}'] = iia.astype(np.float32)
            print(f'{case:>10} {name:>20}: posrec {rec.mean():6.3f}   IIA {iia.mean():.3f}',
                  flush=True)
    meta['arm_names'] = list(ARMS)

    # -- 2. module freezes under the plane patch --------------------------------------------------
    FREEZE_SETS = {'none': ()}
    FREEZE_SETS.update({f'MLP{l}': (('mlp', l),) for l in SWEEP_LAYERS})
    FREEZE_SETS.update({f'attn{l}': (('attn', l),) for l in SWEEP_LAYERS})
    FREEZE_SETS['MLP14+15'] = (('mlp', 14), ('mlp', 15))
    meta['freeze_sets'] = {k: [list(s) for s in v] for k, v in FREEZE_SETS.items()}
    print('\n--- 2. module freezes, base = uv plane preMLP', flush=True)
    for case in CASES:
        row = []
        for fz_name, fz in FREEZE_SETS.items():
            def extra(lo, hi, _fz=fz, _c=case):
                return [block_of(s).register_forward_hook(
                    make_out_patch_hook(FRZ[_c][s][lo:hi], POS_B)) for s in _fz]
            rec, iia = eval_cond(BASE, BAT[case], SRC[case], POS_B, extra=extra)
            out[f'mod_posrec_{ck(case)}_{fz_name}'] = rec.astype(np.float32)
            out[f'mod_iia_{ck(case)}_{fz_name}'] = iia.astype(np.float32)
            row.append(f'{fz_name} {iia.mean():.3f}')
        print(f'{case:>10}: ' + '  '.join(row), flush=True)

    # -- 3. per-neuron attribution, base = the plane patch ----------------------------------------
    # score_i ~= (a_i^corr - a_i^patched) * dPLD/da_i, evaluated on the patched run. Large negative
    # = freezing that neuron destroys the effect. Nanda 2023; Syed et al. 2023; Kramar et al. 2024.
    print('\n--- 3. attribution', flush=True)
    for case in CASES:
        rbat = make_batch(rank_triples, CASE_SLOT[case], T)
        rsrc = {LAYER: S.clean_hidden_at(rbat['ids_c'], LAYER),
                'ctx': S.capture_ctx(rbat['ids_c'], POS_B),
                'premlp': S.capture_premlp(rbat['ids_c'], POS_B)}
        a_corr = capture_neurons(rbat['ids_r'], NEURON_LAYERS, POS_B, bs=ATP_BS)
        attr = {l: np.zeros(d_ff, dtype=np.float64) for l in NEURON_LAYERS}
        hs_store, a_store = {}, {}

        def mk_atp(l):
            def _pre(mod, inp):
                x = inp[0]
                # A zero handle that requires grad: the model's parameters are frozen, so this is
                # what creates a graph from this activation down to the logits.
                h = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=torch.float32,
                                requires_grad=True)
                hs_store[l], a_store[l] = h, x[:, POS_B, :].detach().float()
                xn = x.clone()
                xn[:, POS_B, :] = xn[:, POS_B, :] + h.to(x.dtype)
                return (xn,) + tuple(inp[1:])
            return _pre

        for lo in range(0, rbat['n'], ATP_BS):
            hi = min(lo + ATP_BS, rbat['n'])
            sub = _slice_batch(rbat, lo, hi)
            handles = S.build_handles(BASE, rsrc, lo, hi, POS_B)
            handles += [mlp_of(l).down_proj.register_forward_pre_hook(mk_atp(l))
                        for l in NEURON_LAYERS]
            try:
                with torch.enable_grad():
                    lg = D.model(sub['ids_r']).logits[:, -1, :].float()
                    ix = torch.arange(sub['n'], device=device)
                    # sum, not mean: each example's PLD depends only on its own handle row
                    (ATP_SCALE * (lg[ix, sub['tok_nc']] - lg[ix, sub['tok_b']]).sum()).backward()
            finally:
                for h in handles:
                    h.remove()
            for l in NEURON_LAYERS:
                g = hs_store[l].grad / ATP_SCALE
                attr[l] += ((a_corr[l][lo:hi] - a_store[l]) * g).sum(0).double().cpu().numpy()
            hs_store.clear(); a_store.clear()
            del lg
            torch.cuda.empty_cache()
        for l in NEURON_LAYERS:
            out[f'attr_{ck(case)}_L{l}'] = (attr[l] / rbat['n']).astype(np.float32)
        print(f'{case:>10}: min {min(attr[l].min() for l in NEURON_LAYERS) / rbat["n"]:+.4f} '
              f'({time.time() - t0:.0f} s)', flush=True)
        del rsrc, a_corr
        torch.cuda.empty_cache()

    POOL_LAYER = np.concatenate([np.full(d_ff, l) for l in NEURON_LAYERS])
    POOL_WITHIN = np.concatenate([np.arange(d_ff) for _ in NEURON_LAYERS])
    SCORES = {c: np.concatenate([out[f'attr_{ck(c)}_L{l}'] for l in NEURON_LAYERS]) for c in CASES}
    ORDER = {c: np.argsort(SCORES[c]) for c in CASES}
    n_pool = len(POOL_LAYER)
    out['pool_layer'], out['pool_within'] = POOL_LAYER, POOL_WITHIN
    for c in CASES:
        out[f'order_{ck(c)}'] = ORDER[c][:1000].astype(np.int64)      # the top 1000 is plenty

    def neuron_sel(pooled):
        """pooled indices -> {layer: LongTensor of that layer's neuron indices}."""
        pooled = np.asarray(pooled)
        return {l: torch.as_tensor(POOL_WITHIN[pooled[POOL_LAYER[pooled] == l]],
                                   device=device, dtype=torch.long) for l in NEURON_LAYERS}

    def freeze_neurons(case, pooled):
        sel = neuron_sel(pooled)

        def extra(lo, hi, _sel=sel, _c=case):
            return [mlp_of(l).down_proj.register_forward_pre_hook(
                make_neuron_freeze_hook(ii, NEUR[_c][l][lo:hi][:, ii], POS_B))
                for l, ii in _sel.items() if len(ii)]
        return eval_cond(BASE, BAT[case], SRC[case], POS_B, extra=extra)

    # -- 4. top-k freeze curve --------------------------------------------------------------------
    print('\n--- 4. top-k freeze curve', flush=True)
    for case in CASES:
        for k in K_GRID:
            series = {'top': ORDER[case][:k], 'bottom': ORDER[case][-k:]}
            for s in RAND_SEEDS:
                series[f'rand{s}'] = np.random.default_rng(s).choice(n_pool, k, replace=False)
            for nm, pooled in series.items():
                rec, iia = freeze_neurons(case, pooled)
                out[f'topk_posrec_{ck(case)}_{nm}_k{k}'] = rec.astype(np.float32)
                out[f'topk_iia_{ck(case)}_{nm}_k{k}'] = iia.astype(np.float32)
        rec, iia = freeze_neurons(case, np.arange(n_pool))
        out[f'topk_posrec_{ck(case)}_all'] = rec.astype(np.float32)
        out[f'topk_iia_{ck(case)}_all'] = iia.astype(np.float32)
        print(f'{case:>10}: k=10 {out[f"topk_iia_{ck(case)}_top_k10"].mean():.3f}   '
              f'all {iia.mean():.3f}   ({time.time() - t0:.0f} s)', flush=True)
    meta['k_grid'], meta['rand_seeds'] = K_GRID, RAND_SEEDS

    # -- 5. per-case top lists, their overlaps, and freezing each overlap ---------------------------
    print('\n--- 5. overlap between the two cases', flush=True)
    for k in OVERLAP_KS:
        for c in CASES:
            out[f'top{k}_{ck(c)}'] = ORDER[c][:k].astype(np.int64)
        sh = np.array(sorted(set(ORDER[CASES[0]][:k]) & set(ORDER[CASES[1]][:k])), dtype=np.int64)
        out[f'overlap_k{k}'] = sh
        print(f'  top-{k}: {len(sh)} shared (chance {k * k / n_pool:.3f}), '
              f'L14/L15 {(POOL_LAYER[sh] == 14).sum()}/{(POOL_LAYER[sh] == 15).sum()}', flush=True)
        if not len(sh):
            continue
        for case in CASES:
            sets = {'shared': sh, 'own': ORDER[case][:len(sh)]}
            for s in RAND_SEEDS:
                sets[f'rand{s}'] = np.random.default_rng(100 + s).choice(n_pool, len(sh),
                                                                         replace=False)
            for nm, pooled in sets.items():
                rec, iia = freeze_neurons(case, pooled)
                out[f'ov{k}_posrec_{ck(case)}_{nm}'] = rec.astype(np.float32)
                out[f'ov{k}_iia_{ck(case)}_{nm}'] = iia.astype(np.float32)
            print(f'    {case:>10} shared {out[f"ov{k}_iia_{ck(case)}_shared"].mean():.3f}   '
                  f'own {out[f"ov{k}_iia_{ck(case)}_own"].mean():.3f}   '
                  f'rand {out[f"ov{k}_iia_{ck(case)}_rand0"].mean():.3f}', flush=True)
    meta['overlap_ks'] = OVERLAP_KS

    # The neurons profiled individually below: the union of both cases' top-N_PROFILE, ordered by
    # the better of their two ranks, so a neuron strong in either case is included.
    pos_in = {}
    for c in CASES:
        p = np.empty(n_pool, dtype=np.int64)
        p[ORDER[c]] = np.arange(n_pool)
        pos_in[c] = p
    union = sorted(set(ORDER[CASES[0]][:N_PROFILE]) | set(ORDER[CASES[1]][:N_PROFILE]),
                   key=lambda i: min(pos_in[CASES[0]][i], pos_in[CASES[1]][i]))
    PROF = np.array(union, dtype=np.int64)
    PL, PW = POOL_LAYER[PROF], POOL_WITHIN[PROF]
    out['prof_pooled'], out['prof_layer'], out['prof_within'] = PROF, PL, PW
    for c in CASES:
        out[f'prof_rank_{ck(c)}'] = (pos_in[c][PROF] + 1).astype(np.int64)
        out[f'prof_score_{ck(c)}'] = SCORES[c][PROF].astype(np.float32)
    out['prof_shared'] = np.isin(PROF, out[f'overlap_k{N_PROFILE}']).astype(np.int64)
    print(f'\nprofiling {len(PROF)} neurons (union of both top-{N_PROFILE}), '
          f'L14/L15 {(PL == 14).sum()}/{(PL == 15).sum()}', flush=True)
    prof_idx = {l: torch.as_tensor(PW[PL == l], device=device, dtype=torch.long)
                for l in NEURON_LAYERS}

    # -- 6. injection: the clean MLP output, no upstream patch --------------------------------------
    print('\n--- 6. clean-MLP injection', flush=True)
    for case in CASES:
        clean_out = capture_sites([('mlp', l) for l in INJ_LAYERS[-1]], BAT[case]['ids_c'], POS_B)
        for layers in [[]] + INJ_LAYERS:
            tag = 'none' if not layers else '+'.join(f'MLP{l}' for l in layers)
            recs, i_nc, i_a = [], [], []
            for lo in range(0, BAT[case]['n'], EVAL_BS):
                hi = min(lo + EVAL_BS, BAT[case]['n'])
                sub = _slice_batch(BAT[case], lo, hi)
                handles = [block_of(('mlp', l)).register_forward_hook(
                    make_out_patch_hook(clean_out[('mlp', l)][lo:hi], POS_B)) for l in layers]
                try:
                    with torch.no_grad():
                        lg = D.model(sub['ids_r']).logits[:, -1, :].float()
                finally:
                    for h in handles:
                        h.remove()
                r, i = posrec_and_iia(lg, sub)
                recs.append(r); i_nc.append(i)
                i_a.append((lg.argmax(-1) == TOKA[case][lo:hi]).float().cpu().numpy())
                del lg
            out[f'inj_posrec_{ck(case)}_{tag}'] = np.concatenate(recs).astype(np.float32)
            out[f'inj_iia_nc_{ck(case)}_{tag}'] = np.concatenate(i_nc).astype(np.float32)
            out[f'inj_iia_a_{ck(case)}_{tag}'] = np.concatenate(i_a).astype(np.float32)
        del clean_out
        torch.cuda.empty_cache()
        print(f'{case:>10}: MLP14+15 -> tok_nc '
              f'{out[f"inj_iia_nc_{ck(case)}_MLP14+MLP15"].mean():.3f}   tok_a '
              f'{out[f"inj_iia_a_{ck(case)}_MLP14+MLP15"].mean():.3f}', flush=True)
    meta['inj_tags'] = ['none'] + ['+'.join(f'MLP{l}' for l in ls) for ls in INJ_LAYERS]

    # -- 7. the cloud: plane coordinates, neuron activations, gate preactivations -------------------
    print('\n--- 7. cloud in the plane', flush=True)
    pairs = np.array(S.sample_cloud_pairs(N_CLOUD, SEED + 11, low, high), dtype=np.int64)
    store, acc = {}, {k: [] for k in ('pre14', 'pre15', 'm14', 'm15', 'gate', 'up')}
    acts = {l: [] for l in NEURON_LAYERS}
    E_raw = torch.stack([e1, e2], dim=1)              # unsigned; the sign is applied below
    xsum, nx = torch.zeros(D.d_model, device=device), 0
    Wd = {l: mlp_of(l).down_proj.weight.detach().float() for l in NEURON_LAYERS}

    def _cap_resid(l, key):
        return D.model.model.layers[l].post_attention_layernorm.register_forward_pre_hook(
            lambda m, i, _k=key: store.__setitem__(_k, i[0][:, POS_B, :].detach().float()))

    hs = [_cap_resid(14, 'pre14'), _cap_resid(15, 'pre15')]
    hs += [mlp_of(l).down_proj.register_forward_pre_hook(
        (lambda _l: (lambda m, i: store.__setitem__(f'a{_l}',
                                                    i[0][:, POS_B, :].detach().float())))(l))
        for l in NEURON_LAYERS]
    hs += [mlp_of(14).gate_proj.register_forward_hook(
        lambda m, i, o: store.__setitem__('gate', o[:, POS_B, prof_idx[14]].detach().float())),
        mlp_of(14).up_proj.register_forward_hook(
        lambda m, i, o: store.__setitem__('up', o[:, POS_B, prof_idx[14]].detach().float()))]
    try:
        for i0 in range(0, len(pairs), CLOUD_BS):
            ids = D.tokenizer([D.prompt_gen(list(p)) for p in pairs[i0:i0 + CLOUD_BS]],
                              return_tensors='pt')['input_ids'].to(device)
            assert ids.shape[1] == T, 'cloud prompt length must match the template'
            with torch.no_grad():
                _ = D.model(ids)
            for l in NEURON_LAYERS:
                acts[l].append(store[f'a{l}'][:, prof_idx[l]].cpu().numpy())
            m14 = store['a14'][:, prof_idx[14]] @ Wd[14][:, prof_idx[14]].T
            m15 = store['a15'][:, prof_idx[15]] @ Wd[15][:, prof_idx[15]].T
            for k, v in (('pre14', store['pre14']), ('pre15', store['pre15']),
                         ('m14', m14), ('m15', m15)):
                acc[k].append((v @ E_raw).cpu().numpy())
            xsum += store['pre14'].sum(0)
            nx += store['pre14'].shape[0]
            acc['gate'].append(store['gate'].cpu().numpy())
            acc['up'].append(store['up'].cpu().numpy())
            store.clear()
    finally:
        for h in hs:
            h.remove()
    out['cloud_y1'], out['cloud_y2'] = pairs[:, 0], pairs[:, 1]
    mean_x = xsum / nx                                 # full-dim mean pre-MLP residual @ n2
    raw_plane = {k: np.concatenate(acc[k]) for k in ('pre14', 'pre15', 'm14', 'm15')}
    out['gate_pre_L14'] = np.concatenate(acc['gate']).astype(np.float32)
    out['up_pre_L14'] = np.concatenate(acc['up']).astype(np.float32)
    for l in NEURON_LAYERS:
        out[f'cloud_acts_L{l}'] = np.concatenate(acts[l]).astype(np.float32)
        out[f'cloud_idx_L{l}'] = prof_idx[l].cpu().numpy()

    # Sign the display axes so each rises with its own number, and measure the difference axis:
    # the in-plane direction along which log y1 - log y2 grows fastest.
    pq = raw_plane['pre14'].astype(np.float64)
    pq = pq - pq.mean(0)
    y1f, y2f = pairs[:, 0].astype(np.float64), pairs[:, 1].astype(np.float64)
    sgn = np.sign([np.corrcoef(pq[:, 0], np.log(y1f))[0, 1],
                   np.corrcoef(pq[:, 1], np.log(y2f))[0, 1]])
    out['plane_sgn'] = sgn.astype(np.float32)
    for k, v in raw_plane.items():                     # every stage in the one signed frame
        out[f'plane_{k}'] = (v * sgn).astype(np.float32)
    pqs = pq * sgn
    out['plane_pq'] = pqs.astype(np.float32)
    w = np.linalg.lstsq(np.c_[pqs, np.ones(len(pqs))], np.log(y1f) - np.log(y2f), rcond=None)[0][:2]
    out['diff_axis'] = (w / np.linalg.norm(w)).astype(np.float32)
    print(f'  difference axis at '
          f'{np.degrees(np.arctan2(*out["diff_axis"][::-1])):+.1f} deg', flush=True)

    # -- 8. receptive fields on a dense (y1, y2) grid -----------------------------------------------
    print('\n--- 8. receptive fields', flush=True)
    grid = np.arange(low + 1, high, RF_STEP)
    gpairs = np.array([(a, b) for a in grid for b in grid], dtype=np.int64)   # y1-major
    gids = []
    for i0 in range(0, len(gpairs), CLOUD_BS):
        enc = D.tokenizer([D.prompt_gen(list(p)) for p in gpairs[i0:i0 + CLOUD_BS]],
                          return_tensors='pt')['input_ids']
        assert enc.shape[1] == T
        gids.append(enc.to(device))
    gacts = capture_neurons(torch.cat(gids, dim=0), NEURON_LAYERS, POS_B, idx=prof_idx)
    out['rf_grid'] = grid.astype(np.int64)
    for l in NEURON_LAYERS:
        out[f'rf_acts_L{l}'] = gacts[l].cpu().numpy().astype(np.float32)
    del gids, gacts
    torch.cuda.empty_cache()

    # -- 9. read-in / write-out geometry in the plane ------------------------------------------------
    # A SwiGLU neuron reads through two gamma-folded rows; it writes through its down_proj column.
    # All three are compared with the same signed plane basis, so the arrows and the cloud share a
    # frame. Chance in-plane fraction for a random unit vector is sqrt(2/d_model).
    print('\n--- 9. read-in and write-out geometry', flush=True)
    Es = torch.stack([e1 * float(sgn[0]), e2 * float(sgn[1])], dim=1)
    out['plane_basis_signed'] = Es.cpu().numpy().astype(np.float32)
    rng = np.random.default_rng(0)
    for l in NEURON_LAYERS:
        g = D.model.model.layers[l].post_attention_layernorm.weight.detach().float()
        mats = {'gate': mlp_of(l).gate_proj.weight.detach().float() * g,
                'up': mlp_of(l).up_proj.weight.detach().float() * g}
        sel = torch.as_tensor(PW[PL == l], device=device, dtype=torch.long)
        bg = torch.as_tensor(rng.choice(d_ff, N_ALIGN_BG, replace=False), device=device)
        for which, W in mats.items():
            for nm, ii in (('prof', sel), ('bg', bg)):
                Ww = W[ii]
                P = (Ww @ Es).cpu().numpy()
                out[f'read_{which}_{nm}_L{l}'] = P.astype(np.float32)
                out[f'readfrac_{which}_{nm}_L{l}'] = (
                    np.linalg.norm(P, axis=1) / Ww.norm(dim=1).cpu().numpy()).astype(np.float32)
        Wd_l = Wd[l]
        for nm, ii in (('prof', sel), ('bg', bg)):
            Pw = (Wd_l[:, ii].T @ Es).cpu().numpy()
            out[f'write_{nm}_L{l}'] = Pw.astype(np.float32)
            out[f'writefrac_{nm}_L{l}'] = (
                np.linalg.norm(Pw, axis=1) / Wd_l[:, ii].norm(dim=0).cpu().numpy()).astype(
                    np.float32)
        out[f'bg_idx_L{l}'] = bg.cpu().numpy()

    # SVD of the L14 read-ins inside the plane, with a bootstrap null over random L14 neurons.
    # Rows are normalised by the full weight norm, so a row's length is its in-plane fraction;
    # uncentred, because a zero row means the neuron does not read the plane at all.
    def _unit(P, frac):
        d = P / np.linalg.norm(P, axis=1, keepdims=True)
        return d * np.asarray(frac)[:, None]

    diff_ax = out['diff_axis'].astype(np.float64)
    for which in ('gate', 'up'):
        Mx = _unit(out[f'read_{which}_prof_L14'].astype(np.float64),
                   out[f'readfrac_{which}_prof_L14'])
        bgM = _unit(out[f'read_{which}_bg_L14'].astype(np.float64),
                    out[f'readfrac_{which}_bg_L14'])

        def _top(A):
            _u, s, vt = np.linalg.svd(A, full_matrices=False)
            v = vt[0]
            return s[0], s[1], (-v if v @ diff_ax < 0 else v)

        s1, s2, vv = _top(Mx)
        brng = np.random.default_rng(0)
        ratios = np.empty(SVD_NBOOT)
        for b in range(SVD_NBOOT):
            r1, r2, _ = _top(bgM[brng.choice(len(bgM), len(Mx), replace=False)])
            ratios[b] = r1 / r2          # both from the same draw
        out[f'svd_{which}_M'] = Mx.astype(np.float32)
        out[f'svd_{which}_v1'] = vv.astype(np.float32)
        out[f'svd_{which}_null_ratios'] = ratios.astype(np.float32)
        out[f'svd_{which}_stats'] = np.array([s1, s2, s1 / s2,
                                              np.degrees(np.arctan2(vv[1], vv[0]))],
                                             dtype=np.float32)
        print(f'  SVD {which}: s1/s2 {s1 / s2:.2f} (null median {np.median(ratios):.2f})',
              flush=True)

    # -- 10. gate boundaries in the plane -------------------------------------------------------------
    # A SwiGLU neuron is off where its gate preactivation is negative, so the boundary is the
    # hyperplane w.x = 0 (RMSNorm scales by a positive number and cannot move it). A hyperplane has
    # no exact 2D trace, so two lines are stored: the trace at the cloud's mean out-of-plane state,
    # whose normal IS the in-plane gradient, and the least-squares fit of z on (p, q).
    print('\n--- 10. gate boundaries', flush=True)
    G = out['gate_pre_L14'].astype(np.float64)
    g14 = D.model.model.layers[14].post_attention_layernorm.weight.detach().float()
    W14 = mlp_of(14).gate_proj.weight.detach()[prof_idx[14]].float() * g14
    grad = (W14 @ Es).cpu().numpy()
    c0 = (W14 @ mean_x).cpu().numpy()
    X = np.c_[pqs, np.ones(len(pqs))]
    coef = np.linalg.lstsq(X, G, rcond=None)[0]
    pred = X @ coef
    out['bnd_grad'], out['bnd_c0'] = grad.astype(np.float32), c0.astype(np.float32)
    out['bnd_coef'] = coef.astype(np.float32)
    out['bnd_r2'] = (1 - ((G - pred) ** 2).sum(0) / ((G - G.mean(0)) ** 2).sum(0)).astype(
        np.float32)
    out['bnd_acc_fit'] = (((pred > 0) == (G > 0)).mean(0)).astype(np.float32)
    out['bnd_acc_slice'] = ((((pqs @ grad.T + c0) > 0) == (G > 0)).mean(0)).astype(np.float32)
    out['bnd_on_frac'] = ((G > 0).mean(0)).astype(np.float32)

    # -- 11. connectivity, L14 -> L15 ------------------------------------------------------------------
    # Weight path: the "virtual weight" of Elhage et al. 2021 -- for an MLP -> MLP path through the
    # residual stream it is a plain inner product of the write column with the gamma-folded read row.
    # Causal edge: patch one L14 neuron to its clean value, read the shift in each L15 neuron.
    print('\n--- 11. L14 -> L15 connectivity', flush=True)
    i14 = PW[PL == 14]
    i15 = PW[PL == 15]
    t14 = torch.as_tensor(i14, device=device, dtype=torch.long)
    t15 = torch.as_tensor(i15, device=device, dtype=torch.long)
    g15 = D.model.model.layers[15].post_attention_layernorm.weight.detach().float()
    A = Wd[14][:, t14]
    An = A / A.norm(dim=0, keepdim=True)
    for which, Wr in (('gate', mlp_of(15).gate_proj.weight.detach().float() * g15),
                      ('up', mlp_of(15).up_proj.weight.detach().float() * g15)):
        Rn = Wr / Wr.norm(dim=1, keepdim=True)
        allc = (Rn @ An).cpu().numpy()
        out[f'path_{which}'] = allc[i15].T.astype(np.float32)          # (n14, n15)
        out[f'path_{which}_null'] = allc.astype(np.float32)            # (d_ff, n14)
    sd15 = out['cloud_acts_L15'].std(0)
    out['sd15'] = sd15.astype(np.float32)
    for case in CASES:
        a_cl = capture_neurons(BAT[case]['ids_c'], [14], POS_B, idx={14: t14})[14]
        base = capture_neurons(BAT[case]['ids_r'], [15], POS_B, idx={15: t15})[15].cpu().numpy()
        rows = []
        for c in range(len(i14)):
            got = capture_neurons(
                BAT[case]['ids_r'], [15], POS_B, idx={15: t15},
                patch=lambda lo, hi, _c=c, _a=a_cl: [
                    mlp_of(14).down_proj.register_forward_pre_hook(
                        make_neuron_freeze_hook(t14[_c:_c + 1], _a[lo:hi, _c:_c + 1],
                                                POS_B))])[15].cpu().numpy()
            rows.append((got - base).mean(0) / sd15)
        got = capture_neurons(
            BAT[case]['ids_r'], [15], POS_B, idx={15: t15},
            patch=lambda lo, hi, _a=a_cl: [mlp_of(14).down_proj.register_forward_pre_hook(
                make_neuron_freeze_hook(t14, _a[lo:hi], POS_B))])[15].cpu().numpy()
        out[f'edge_{ck(case)}'] = np.array(rows, dtype=np.float32)
        out[f'edge_all_{ck(case)}'] = ((got - base).mean(0) / sd15).astype(np.float32)
        del a_cl
        torch.cuda.empty_cache()
    print(f'  edges done ({time.time() - t0:.0f} s)', flush=True)

    # -- 12. DAS at the L15 residual, ranks 1-3 ---------------------------------------------------
    print('\n--- 12. DAS at the L15 residual', flush=True)
    das_tri = rank_triples[:DAS15_N]
    bases15 = {}
    for case in CASES:
        tbat = make_batch(das_tri, CASE_SLOT[case], T)
        ch = S.clean_hidden_at(tbat['ids_c'], DAS15_LAYER)
        for k in DAS15_RANKS:
            Q, curve = train_das_resid(DAS15_LAYER, POS_B, k, tbat, ch, seed=SEED + k)
            bases15[case, k] = Q
            out[f'das15_basis_{ck(case)}_k{k}'] = Q.cpu().numpy().astype(np.float32)
            out[f'das15_curve_{ck(case)}_k{k}'] = curve
            print(f'{case:>10} rank {k}: loss {curve[0]:.4f} -> {curve[-1]:.4f} '
                  f'({time.time() - t0:.0f} s)', flush=True)
        del ch
        torch.cuda.empty_cache()
    for ev in CASES:
        src15 = {DAS15_LAYER: S.clean_hidden_at(BAT[ev]['ids_c'], DAS15_LAYER)}
        for tr in CASES:
            for k in DAS15_RANKS:
                rec, iia = eval_cond(dict(resid=[(DAS15_LAYER, 'sub', bases15[tr, k])]),
                                     BAT[ev], src15, POS_B)
                out[f'das15_posrec_{ck(ev)}_from_{ck(tr)}_k{k}'] = rec.astype(np.float32)
                out[f'das15_iia_{ck(ev)}_from_{ck(tr)}_k{k}'] = iia.astype(np.float32)
        rec, iia = eval_cond(dict(resid=[(DAS15_LAYER, 'full', None)]), BAT[ev], src15, POS_B)
        out[f'das15_posrec_{ck(ev)}_full'] = rec.astype(np.float32)
        out[f'das15_iia_{ck(ev)}_full'] = iia.astype(np.float32)
        del src15
        torch.cuda.empty_cache()
    # How much of each MLP15 neuron's write lands in the DAS subspace, for every neuron so the
    # ones of interest have their own null.
    out.update(das15_write_alignment(bases15))

    # the cloud inside each DAS subspace, coloured downstream by which slot holds the max
    hstore, chunks = {}, []
    hnd = D.model.model.layers[DAS15_LAYER].register_forward_hook(
        lambda m, i, o: hstore.__setitem__('h', (o[0] if isinstance(o, tuple) else o)
                                           [:, POS_B, :].detach().float()))
    try:
        for i0 in range(0, len(pairs), CLOUD_BS):
            ids = D.tokenizer([D.prompt_gen(list(p)) for p in pairs[i0:i0 + CLOUD_BS]],
                              return_tensors='pt')['input_ids'].to(device)
            with torch.no_grad():
                _ = D.model(ids)
            chunks.append(hstore.pop('h'))
    finally:
        hnd.remove()
    H15 = torch.cat(chunks, dim=0)
    for case in CASES:
        for k in DAS15_RANKS:
            out[f'das15_proj_{ck(case)}_k{k}'] = (H15 @ bases15[case, k]).cpu().numpy().astype(
                np.float32)
    del H15, chunks
    torch.cuda.empty_cache()
    meta['das15_ranks'] = DAS15_RANKS

    # -- 13. DAS inside the MLP's own neuron space --------------------------------------------------
    # 6-ish dimensions instead of 3584, and the learned direction is a weighting over named neurons.
    print('\n--- 13. DAS inside the neuron space', flush=True)
    mlp_tri = rank_triples[:MLPDAS_N]
    bases_n = {}
    for case in CASES:
        tbat = make_batch(mlp_tri, CASE_SLOT[case], T)
        for l in NEURON_LAYERS:
            idx = prof_idx[l]
            a_tr = capture_neurons(tbat['ids_c'], [l], POS_B, idx={l: idx})[l]
            for r in MLPDAS_RANKS:
                P, curve = train_das_neurons(l, idx, a_tr, r, tbat, POS_B, seed=SEED + r)
                bases_n[case, l, r] = P
                out[f'mlpdas_basis_{ck(case)}_L{l}_r{r}'] = P.cpu().numpy().astype(np.float32)
                out[f'mlpdas_curve_{ck(case)}_L{l}_r{r}'] = curve
                print(f'{case:>10} L{l} rank {r}: loss {curve[0]:.4f} -> {curve[-1]:.4f} '
                      f'({time.time() - t0:.0f} s)', flush=True)
            del a_tr
            torch.cuda.empty_cache()
    for case in CASES:
        for l in NEURON_LAYERS:
            idx = prof_idx[l]
            a_ev = capture_neurons(BAT[case]['ids_c'], [l], POS_B, idx={l: idx})[l]
            for tag, P in [(f'r{r}', bases_n[case, l, r]) for r in MLPDAS_RANKS] + [('full', None)]:
                def extra(lo, hi, _l=l, _i=idx, _a=a_ev, _P=P):
                    return [mlp_of(_l).down_proj.register_forward_pre_hook(
                        make_neuron_das_hook(_i, _a[lo:hi], _P, POS_B))]
                rec, iia = eval_cond(None, BAT[case], SRC[case], POS_B, extra=extra)
                out[f'mlpdas_posrec_{ck(case)}_L{l}_{tag}'] = rec.astype(np.float32)
                out[f'mlpdas_iia_{ck(case)}_L{l}_{tag}'] = iia.astype(np.float32)
            del a_ev
            torch.cuda.empty_cache()
    meta['mlpdas_ranks'] = MLPDAS_RANKS

    meta.update(tag=TAG, model=D.MODELNAME, cases=CASES, seed=SEED, n_eval=N_EVAL,
                n_rank=len(rank_triples), l13_layer=LAYER, h_layer=H_LAYER, h_head=S.H_HEAD,
                pos_first=int(POS_A), pos_second=int(POS_B), T=int(T), d_model=int(d_model),
                d_ff=int(d_ff), neuron_layers=NEURON_LAYERS, sweep_layers=SWEEP_LAYERS,
                n_cloud=N_CLOUD, rf_step=RF_STEP, n_profile=N_PROFILE,
                n_align_bg=N_ALIGN_BG, atp_bs=ATP_BS, atp_scale=ATP_SCALE,
                base_patch='uv plane preMLP (rank-2 in span(v1, v2))',
                dirs_from=f'{RESULTS_DIR}/sharedrep_{TAG}.npz',
                runtime_s=round(time.time() - t0, 1))
    return out, meta


if __name__ == '__main__':
    D.load_model()
    S.init_head_machinery()
    res, meta = run()

    path = f'{RESULTS_DIR}/comparator_data_{TAG}.npz'
    arrays = {k: np.asarray(v) for k, v in res.items()}
    np.savez_compressed(path, **arrays)
    with open(f'{RESULTS_DIR}/comparator_data_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=float)

    print(f'\n{"=" * 92}\n=== SAVED\n{"=" * 92}', flush=True)
    for k, a in sorted(arrays.items(), key=lambda kv: -kv[1].nbytes)[:15]:
        print(f'    {k:<40} {str(a.shape):>16} {str(a.dtype):>9}  {a.nbytes / 1e6:8.3f} MB',
              flush=True)
    print(f'\n{path}: {os.path.getsize(path) / 1e6:.3f} MB, {len(arrays)} arrays', flush=True)
    print(f'total runtime {meta["runtime_s"]:.0f} s', flush=True)
