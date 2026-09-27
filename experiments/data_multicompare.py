"""Machinery for the three-number `max(n1, n2, n3)` version of the comparator analysis.

Unlike its siblings in this directory (`data_numberreps.py`, `data_sharedrep.py`,
`data_comparator.py`) this module has **no `run()` and no `__main__`**: it is imported by
`explore_multicompare.ipynb`, which owns the compute schedule and the caching. The split exists
because the notebook is the deliverable here -- every block writes its own `.npz` and draws its
figure immediately, so results appear as they are produced rather than at the end.

Task and prompt come from `e15_multi_compare` (`algo_multi_compare.ipynb` /
`localize_multi_compare.ipynb`): the same one-shot anchor widened to three operands, the same
`order_key` convention for value orders, and the same clean/corrupted construction (the maximum,
at its own slot, replaced in place by a value below all three, so the win moves to the runner-up).

Everything else -- hooks, interchange interventions, the position-recovery / IIA metrics, the DAS
parametrization, the neuron-level machinery -- is imported from the two-number modules rather than
reimplemented, so the three-number numbers sit on exactly the same measurement code as the
two-number ones:

    data_numberreps   : DASSubspace, cf_pair_loss, posrec_and_iia, pos_anchors,
                        make_full_patch_hook, make_subspace_patch_hook, _slice_batch
    data_sharedrep    : per-head output patching at L14, the pre-MLP L14 residual hooks
    data_comparator   : MLP neuron capture / freezing, sub-block output pinning

What this module adds is the K = 3 versions of the things that were hard-coded to two numbers:
the prompt and its token layout, quadruple sampling, the order-keyed counterfactual batch, a
`build_handles` that can patch **several** heads at once (the three-number space needs H14 and H18
together), and the 3-D dose-response sweep.

Notation used throughout, at the **third**-number position `n3` (where all three numbers are
present, because L14's copy heads have written the first two on top of what L13 already holds):

    v1 = 1D DAS inside head L14.H14's output   -- y1, copied in from n1
    v2 = 1D DAS inside head L14.H18's output   -- y2, copied in from n2
    v3 = PC1 of the L13 residual at n3         -- y3, already in place

See docs/explore_multicompare.md.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

import data_numberreps as D
import data_sharedrep as S
import data_comparator as C
from data_numberreps import (DASSubspace, cf_pair_loss, posrec_and_iia, pos_anchors,
                             make_full_patch_hook, make_subspace_patch_hook, _slice_batch,
                             tok_indices_for_chars, SEED, EVAL_BS, device)

# ----------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------
K          = 3                  # three operands
ANCHOR     = 'The maximum of '
LAYER      = 13                 # the residual stream leaving decoder layer 13 (v3 lives here)
H_LAYER    = 14                 # the copy heads' layer
HEAD_V1    = 14                 # L14.H14 moves the first number  -> n3   (e15 localize/algo)
HEAD_V2    = 18                 # L14.H18 moves the second number -> n3

# Same two digit regimes as data_numberreps; `gap` is the minimum separation between consecutive
# values of a quadruple, so the four values are comfortably apart at every rank.
REGIMES = {
    '2digit': dict(low=10,  high=100,  gap=10),
    '3digit': dict(low=100, high=1000, gap=100),
}

# ---- value orders ------------------------------------------------------------------------------
# An order is the tuple of slot indices in descending value order (`order_key`), so (0, 2, 1) reads
# s1 > s3 > s2: slot 1 holds the largest, slot 3 the runner-up, slot 2 the smallest.
#
# HEAD_GROUPS -- the three "top two" arrangements the per-head sweep compares. They are exactly
# e15 `algo_multi_compare.ipynb`'s groups A / B / C, chosen because A vs B holds the max at s1 and
# moves the runner-up, while B vs C holds the runner-up at s3 (identical patch site, identical
# token) and moves the max. Each is patched at its own runner-up position.
HEAD_GROUPS = {'s1>s2': (0, 1, 2),
               's1>s3': (0, 2, 1),
               's2>s3': (1, 2, 0)}

# CASES -- the perturbation cases the shared-representation work uses, all patched at n3. The case
# name says which number the counterfactual moves: `make_batch` only ever varies the winning slot's
# value, so the case *is* the perturbed variable, and each vj should be causal in exactly its own
# case. 'y1'/'y2' reuse HEAD_GROUPS' s1>s3 / s2>s3; 'y3' puts the max at s3 itself (the direct
# analogue of the two-number `n2 larger`, where the max also sits at the patch site) with the
# runner-up fixed at s2.
CASES = {'y1': (0, 2, 1),
         'y2': (1, 2, 0),
         'y3': (2, 1, 0)}
CASE_LIST = ['y1', 'y2', 'y3']

# Every order whose batch is needed -- the union of the two dicts, de-duplicated.
ALL_ORDERS = sorted(set(HEAD_GROUPS.values()) | set(CASES.values()))

N_EVAL     = 400                # held-out quadruples every reported number is measured on
N_TRAIN    = 128                # disjoint quadruples used to fit directions and rank neurons
N_CLOUD    = 1500               # triples in the representation cloud
CLOUD_BS   = 48

DAS_STEPS, DAS_LR, DAS_BS = 100, 0.05, 16
RIDGE_ALPHAS = np.logspace(-2, 4, 13)
PROBE_TEST_FRAC = 0.2
SWEEP_WIN = 2                   # +/- window around a grid value when averaging its coordinate

ck = (lambda s: str(s).replace(' ', '_').replace('>', 'gt'))


# ----------------------------------------------------------------------------------------------
# Prompt and token layout (K = 3)
# ----------------------------------------------------------------------------------------------
def number_list(nums):
    """'89, 67 and 34' -- comma-separated with `and` before the last operand."""
    return f"{', '.join(str(n) for n in nums[:-1])} and {nums[-1]}"


def prompt_gen(nums):
    """e15's three-number one-shot prompt. The anchor example is fixed at three operands spanning
    one, two and three digits, so example length never co-varies with anything being measured."""
    return (f'Answer in the following format with a single answer. '
            f'The maximum of 12, 437 and 5 is 437. '
            f'The maximum of {number_list(nums)} is ')


def num_char_spans(prompt, nums):
    """Char span of each number in the question clause.

    Scanned forward from past the **last** anchor, with the cursor advanced past each match, so the
    one-shot example can never be matched and a digit string repeated across slots is matched once
    per slot in order.
    """
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
    """Qwen splits numbers into digits, so a number's *first* token is its leading digit. Every
    readout here is a first token, which is only unambiguous when no two numbers share one."""
    return len({str(n)[0] for n in nums}) == len(nums)


def order_key(nums):
    """Slot indices in descending value order. (0, 2, 1) = slot 1 > slot 3 > slot 2."""
    return tuple(int(i) for i in np.argsort(-np.asarray(nums)))


def order_label(o):
    return ' > '.join(f's{i + 1}' for i in o)


def regime_positions(low, high):
    """Template token layout for a regime: (T, last-token position of each number, tokens per
    number, the reference prompt). Every prompt in a regime has the same digit count, hence the
    same layout -- asserted, never assumed, in `make_batch`."""
    ref = [low + 1, low + 2, low + 3]
    p = prompt_gen(ref)
    enc = D.tokenizer(p, return_tensors='pt', return_offsets_mapping=True)
    offs = enc['offset_mapping'][0].numpy()
    T = enc['input_ids'].shape[1]
    pos_last = [num_tok_span(p, ref, i, offs)[-1] for i in range(K)]
    n_tok = [len(num_tok_span(p, ref, i, offs)) for i in range(K)]
    return T, pos_last, n_tok, p


# ----------------------------------------------------------------------------------------------
# Counterfactual construction
# ----------------------------------------------------------------------------------------------
def build_quads(n, seed, low, high, gap):
    """`n` quadruples v0 > v1 > v2 > v3, consecutive values at least `gap` apart, all in
    [low, high), all four with distinct leading digits.

    Four values, not three: v0..v2 are the operands and v3 is the replacement that corrupts the
    winner. Requiring the replacement to be below all three operands is what makes the corrupted
    winner *always* the clean runner-up, whatever order the operands are laid out in -- so one pool
    of quadruples serves every order, and the orders are compared on the same values.
    """
    g = torch.Generator().manual_seed(seed)
    out, seen = [], set()
    while len(out) < n:
        v = sorted(torch.randint(low, high, (4,), generator=g).tolist(), reverse=True)
        if (all(v[i] > v[i + 1] + gap for i in range(3))
                and leading_digits_distinct(v) and tuple(v) not in seen):
            seen.add(tuple(v))
            out.append(tuple(v))
    return out


def make_batch(quads, order, T):
    """Tokenised clean/corrupted batch laying each quadruple out in `order`.

    clean      -- v0, v1, v2 placed so that `order` is the descending value order; the model
                  answers v0, which sits at slot `order[0]`.
    corrupted  -- v3 replaces v0 *in place* at slot `order[0]`; every other slot is untouched, so
                  the two prompts tokenize identically and the win moves to slot `order[1]`.

    Readout tokens follow e15's `make_compute_metrics` and data_numberreps' `make_batch`:
      tok_nc -- first token of v3, the replacement sitting at slot `order[0]`. A *value-independent*
                readout of "slot order[0] wins": the patch never supplies this value, so the model
                can only emit it by having been convinced that that slot holds the maximum.
      tok_b  -- first token of v1, at slot `order[1]`: the corrupted run's own answer, shared
                between the two runs.
    A successful interchange moves the answer from tok_b to tok_nc.
    """
    win, run = order[0], order[1]
    clean, corr = [], []
    for v in quads:
        cp = [None] * K
        for rank, slot in enumerate(order):
            cp[slot] = v[rank]
        rp = list(cp)
        rp[win] = v[3]
        assert order_key(cp) == tuple(order), 'clean layout must realise the order'
        assert int(np.argmax(rp)) == run, 'corruption must move the win to the runner-up slot'
        clean.append(cp)
        corr.append(rp)

    cs = [prompt_gen(p) for p in clean]
    rs = [prompt_gen(p) for p in corr]
    enc_c = D.tokenizer(cs, return_tensors='pt', return_offsets_mapping=True)
    enc_r = D.tokenizer(rs, return_tensors='pt', return_offsets_mapping=True)
    assert enc_c['input_ids'].shape[1] == T and enc_r['input_ids'].shape[1] == T, \
        'every prompt in a regime must share the token layout'
    ids_c, ids_r = enc_c['input_ids'].to(device), enc_r['input_ids'].to(device)

    slot_toks, ta = [], []
    for i in range(len(quads)):
        orr = enc_r['offset_mapping'][i].numpy()
        occ = enc_c['offset_mapping'][i].numpy()
        # First token of each of the corrupted prompt's own three numbers. tok_nc and tok_b are two
        # of these; the third is kept because the 3-D sweep scores "which slot did the model name"
        # over all three slots, not just the two the interchange moves between.
        slot_toks.append([int(ids_r[i, num_tok_span(rs[i], corr[i], s, orr)[0]]) for s in range(K)])
        ta.append(int(ids_c[i, num_tok_span(cs[i], clean[i], win, occ)[0]]))
    slot_toks = torch.tensor(slot_toks, device=device)

    return dict(ids_c=ids_c, ids_r=ids_r, n=len(quads), order=tuple(order),
                win_slot=win, run_slot=run,
                clean_nums=np.array(clean), corr_nums=np.array(corr),
                tok_slot=slot_toks,
                tok_nc=slot_toks[:, win],
                tok_b=slot_toks[:, run],
                tok_a=torch.tensor(ta, device=device))


def solved_mask(bat, bs=EVAL_BS):
    """Boolean (n,): the model answers the clean run with tok_a and the corrupted run with tok_b.

    Three-number `max` is not solved perfectly (`e15/multicompare_accuracy.ipynb`), and position
    recovery divides by `PLD_clean - PLD_corr`. On a case the model gets wrong that denominator is
    small and the ratio explodes, so the pool is filtered rather than the ratios winsorised --
    e15's `model_solves`, batched.
    """
    ok = []
    for lo in range(0, bat['n'], bs):
        hi = min(lo + bs, bat['n'])
        with torch.no_grad():
            ac = D.model(bat['ids_c'][lo:hi]).logits[:, -1, :].argmax(-1)
            ar = D.model(bat['ids_r'][lo:hi]).logits[:, -1, :].argmax(-1)
        ok.append(((ac == bat['tok_a'][lo:hi]) & (ar == bat['tok_b'][lo:hi])).cpu().numpy())
    return np.concatenate(ok)


# ----------------------------------------------------------------------------------------------
# Interventions as declarative specs
# ----------------------------------------------------------------------------------------------
def build_handles(spec, sources, lo, hi, pos):
    """Register everything one condition asks for, and return the handles for the caller to remove.

    spec['resid']  : list of (layer, 'full'|'sub', P) -- residual-stream interchange at `pos`.
    spec['premlp'] : P or None -- interchange the pre-MLP L14 residual at `pos` (None = full rank).
    spec['heads']  : list of (head, P or None) -- rewrite those heads' own output contributions.

    The only difference from `data_sharedrep.build_handles` is that `heads` is a **list**. The
    three-number space needs H14 and H18 patched together, and that composes correctly: `o_proj`'s
    output is the sum over heads, each post-hook adds its own head's delta, and PyTorch feeds each
    hook the previous hook's returned output, so two deltas accumulate rather than one clobbering
    the other.
    """
    handles = []
    for layer, kind, P in spec.get('resid', ()):
        src = sources[layer][lo:hi]
        hook = (make_full_patch_hook(src, pos) if kind == 'full'
                else make_subspace_patch_hook(P, src, pos))
        handles.append(D.model.model.layers[layer].register_forward_hook(hook))
    if 'premlp' in spec:
        handles += S.premlp_patch_handles(sources['premlp'][lo:hi], pos, spec['premlp'])
    for head, P in spec.get('heads', ()):
        handles += S.head_patch_handles(head, pos, S.head_out(sources['ctx'][lo:hi], head), P)
    return handles


def eval_spec(spec, bat, sources, pos, extra=None, bs=EVAL_BS):
    """Per-example position recovery and IIA for one condition, in forward-pass minibatches.

    `extra(lo, hi) -> [handles]` adds anything the spec language does not cover -- module freezes,
    neuron freezes, injections. Metrics are reduced per minibatch so the full (n, |vocab|) logits
    tensor is never held.
    """
    recs, iias = [], []
    for lo in range(0, bat['n'], bs):
        hi = min(lo + bs, bat['n'])
        sub = _slice_batch(bat, lo, hi)
        handles = build_handles(spec, sources, lo, hi, pos) if spec else []
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


def sources_for(bat, pos, layers=(LAYER, H_LAYER)):
    """The clean-run activations every condition patches from, captured once per (batch, pos)."""
    src = {l: S.clean_hidden_at(bat['ids_c'], l) for l in layers}
    src['ctx'] = S.capture_ctx(bat['ids_c'], pos)
    src['premlp'] = S.capture_premlp(bat['ids_c'], pos)
    return src


# ----------------------------------------------------------------------------------------------
# DAS (Geiger et al. 2023, single-causal-variable rank-1 case)
# ----------------------------------------------------------------------------------------------
def train_head_das(head, tbat, ctx_clean, pos, seed, steps=DAS_STEPS, lr=DAS_LR, bs=DAS_BS):
    """Rank-1 DAS inside one L14 head's own output contribution at `pos`.

    `data_sharedrep.train_h14_das` generalised to an arbitrary head, so H14 and H18 are fitted by
    identical code. Same QR retraction on the Stiefel manifold, same counterfactual-pair
    cross-entropy objective, gradients accumulated over minibatches so the training-set size is
    decoupled from the backward pass's activation memory.

    The reachable search space is only the `head_dim`-dimensional column space of that head's W_O
    block -- anything orthogonal to it leaves the head's output untouched -- so this is a 128-d
    search embedded in 3584 dims.
    """
    torch.manual_seed(seed)
    das = DASSubspace(D.d_model, 1, device)
    opt = torch.optim.Adam([das.raw], lr=lr)
    bounds = [(i, min(i + bs, tbat['n'])) for i in range(0, tbat['n'], bs)]
    curve = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for lo, hi in bounds:
            sub = _slice_batch(tbat, lo, hi)
            # basis() is rebuilt per minibatch: each backward() frees that minibatch's QR graph.
            handles = S.head_patch_handles(head, pos, S.head_out(ctx_clean[lo:hi], head),
                                           P=das.basis())
            try:
                with torch.enable_grad():
                    lg = D.model(sub['ids_r']).logits[:, -1, :].float()
            finally:
                for h in handles:
                    h.remove()
            loss = cf_pair_loss(lg, sub) * (sub['n'] / tbat['n'])     # mean over the full batch
            loss.backward()
            total += float(loss)
            del lg
        opt.step()
        curve.append(total)
    torch.cuda.empty_cache()
    with torch.no_grad():
        return das.basis().detach(), np.array(curve, dtype=np.float32)


def train_resid_das(layer, tbat, ch, pos, k, seed, steps=DAS_STEPS, lr=DAS_LR, bs=DAS_BS):
    """Rank-k DAS on the residual stream leaving `layer` at `pos` -- the control for v3, whose
    headline definition is PC1 rather than a fit."""
    torch.manual_seed(seed)
    das = DASSubspace(D.d_model, k, device)
    opt = torch.optim.Adam([das.raw], lr=lr)
    bounds = [(i, min(i + bs, tbat['n'])) for i in range(0, tbat['n'], bs)]
    curve = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for lo, hi in bounds:
            sub = _slice_batch(tbat, lo, hi)
            hnd = D.model.model.layers[layer].register_forward_hook(
                make_subspace_patch_hook(das.basis(), ch[lo:hi], pos))
            try:
                with torch.enable_grad():
                    lg = D.model(sub['ids_r']).logits[:, -1, :].float()
            finally:
                hnd.remove()
            loss = cf_pair_loss(lg, sub) * (sub['n'] / tbat['n'])
            loss.backward()
            total += float(loss)
            del lg
        opt.step()
        curve.append(total)
    torch.cuda.empty_cache()
    with torch.no_grad():
        return das.basis().detach(), np.array(curve, dtype=np.float32)


# ----------------------------------------------------------------------------------------------
# Representation cloud at n3
# ----------------------------------------------------------------------------------------------
def sample_cloud_triples(n, seed, low, high):
    """`n` triples of distinct-leading-digit numbers, sampled independently per slot.

    No order constraint and no gap: the cloud is what the representation looks like over the task's
    whole input distribution, not over the counterfactual design. Sampling each slot independently
    keeps y1, y2 and y3 mutually uncorrelated, which is what lets a PCA component or a probe be
    attributed to one number rather than to a mixture.
    """
    g = torch.Generator().manual_seed(seed)
    out, seen = [], set()
    while len(out) < n:
        t = tuple(torch.randint(low, high, (3,), generator=g).tolist())
        if leading_digits_distinct(t) and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def capture_cloud(triples, T, pos3, heads=(HEAD_V1, HEAD_V2), bs=CLOUD_BS):
    """One pass per minibatch -> every representation at n3 the analysis reads, row-aligned.

    Returns a dict of float32 arrays over the cloud:
      'l13'      residual leaving L13            (the number already in place at n3)
      'head{h}'  that head's own output contribution (its copy of an earlier number)
      'pre14'    the pre-MLP L14 residual        (the comparator's actual input)
      'post14'   the residual leaving L14

    All of them come from the same forward pass, so the H14 component and the L13 component of a
    row describe the same prompt -- required for the probes and for the cosines between directions
    read on different clouds.
    """
    store = {}

    def resid_hook(key):
        def _h(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            store[key] = hs[:, pos3, :].detach().float()
        return _h

    def ctx_pre(mod, inp):
        store['ctx'] = inp[0][:, pos3, :].detach().float()

    acc = {k: [] for k in ['l13', 'pre14', 'post14'] + [f'head{h}' for h in heads]}
    for i0 in range(0, len(triples), bs):
        chunk = triples[i0:i0 + bs]
        ids = D.tokenizer([prompt_gen(list(t)) for t in chunk],
                          return_tensors='pt')['input_ids'].to(device)
        assert ids.shape[1] == T, 'cloud prompt length must match the template'
        pre_handles, pre_store = S._premlp_hooks(pos3, None)
        handles = [D.model.model.layers[LAYER].register_forward_hook(resid_hook('l13')),
                   D.model.model.layers[H_LAYER].register_forward_hook(resid_hook('post14')),
                   S.attn.o_proj.register_forward_pre_hook(ctx_pre)] + pre_handles
        try:
            with torch.no_grad():
                _ = D.model(ids)
        finally:
            for h in handles:
                h.remove()
        acc['l13'].append(store['l13'].cpu().numpy())
        acc['post14'].append(store['post14'].cpu().numpy())
        acc['pre14'].append(pre_store['pre'].cpu().numpy())
        for h in heads:
            acc[f'head{h}'].append(S.head_out(store['ctx'], h).cpu().numpy())
        store.clear()
    return {k: np.concatenate(v).astype(np.float32) for k, v in acc.items()}


def capture_resid_cloud(triples, T, sites, bs=CLOUD_BS):
    """{(layer, pos): (n, d_model)} residual stream leaving `layer` at `pos`, over a cloud.

    `capture_cloud` is pinned to the sites the (v1, v2, v3) work needs. This is the generic
    version, for reading some other layer or some other position over the *same* triples --
    the L15 residual at n3, and the L13 residual at n1 / n2. Every site is read in one pass per
    minibatch, so rows stay aligned with `triples` and with anything `capture_cloud` returned for
    the same list.
    """
    layers = sorted({l for l, _ in sites})
    store, acc = {}, {s: [] for s in sites}

    def mk(l):
        def _h(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            store[l] = hs.detach().float()
        return _h

    for i0 in range(0, len(triples), bs):
        ids = D.tokenizer([prompt_gen(list(t)) for t in triples[i0:i0 + bs]],
                          return_tensors='pt')['input_ids'].to(device)
        assert ids.shape[1] == T, 'cloud prompt length must match the template'
        handles = [D.model.model.layers[l].register_forward_hook(mk(l)) for l in layers]
        try:
            with torch.no_grad():
                _ = D.model(ids)
        finally:
            for h in handles:
                h.remove()
        for (l, p) in sites:
            acc[(l, p)].append(store[l][:, p, :].cpu().numpy())
        store.clear()
    return {s: np.concatenate(v).astype(np.float32) for s, v in acc.items()}


def pc1_of(cloud):
    """Unit PC1 of a cloud, together with the 3-component PCA that the 3-D scatter plots use."""
    pca = PCA(n_components=3).fit(cloud)
    d = pca.components_[0]
    return d / np.linalg.norm(d), pca


def fit_probes(cloud, targets, seed=SEED):
    """Ridge log-magnitude probes, one per number: {name: (unit direction, held-out R2)}.

    Linear probing follows Alain & Bengio 2016; ridge because the target is continuous, with
    `RidgeCV` choosing alpha by leave-one-out CV on the training split so the regularisation is not
    an arbitrary constant. The held-out R2 is what says the quantity is linearly present at all;
    the unit weight vector is the direction the cosine heatmap compares against v1, v2, v3.
    """
    idx_tr, idx_te = train_test_split(np.arange(len(cloud)), test_size=PROBE_TEST_FRAC,
                                      random_state=seed)
    out = {}
    for name, y in targets.items():
        r = RidgeCV(alphas=RIDGE_ALPHAS).fit(cloud[idx_tr], np.log(y[idx_tr]))
        w = r.coef_ / np.linalg.norm(r.coef_)
        out[name] = (w.astype(np.float32), float(r.score(cloud[idx_te], np.log(y[idx_te]))))
    return out


# ----------------------------------------------------------------------------------------------
# Attribution patching over MLP neurons
# ----------------------------------------------------------------------------------------------
def attribution(base_spec, bat, sources, pos, layers, bs=C.ATP_BS, scale=C.ATP_SCALE):
    """Per-neuron attribution scores over `layers`, base run = `base_spec` applied to `bat`.

        s_i ~= (a_i^corr - a_i^patched) * d PLD / d a_i,   evaluated at the patched state

    PLD = logit[tok_nc] - logit[tok_b], the numerator of position recovery -- IIA is an argmax and
    has no gradient. A large **negative** score means freezing that neuron back to its corrupted
    value destroys the patch's effect, i.e. the neuron carries it.

    The gradient is taken w.r.t. a zero handle added to the `down_proj` input: the model's own
    parameters are frozen, so that handle is what creates a graph from the activation to the
    logits. Attribution patching as in Nanda 2023 / Syed et al. 2023 / Kramar et al. 2024 (AtP*).
    """
    d_ff = D.model.config.intermediate_size
    a_corr = C.capture_neurons(bat['ids_r'], layers, pos, bs=bs)
    attr = {l: np.zeros(d_ff, dtype=np.float64) for l in layers}
    hs_store, a_store = {}, {}

    def mk(l):
        def _pre(mod, inp):
            x = inp[0]
            h = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=torch.float32,
                            requires_grad=True)
            hs_store[l], a_store[l] = h, x[:, pos, :].detach().float()
            xn = x.clone()
            xn[:, pos, :] = xn[:, pos, :] + h.to(x.dtype)
            return (xn,) + tuple(inp[1:])
        return _pre

    for lo in range(0, bat['n'], bs):
        hi = min(lo + bs, bat['n'])
        sub = _slice_batch(bat, lo, hi)
        handles = build_handles(base_spec, sources, lo, hi, pos)
        handles += [C.mlp_of(l).down_proj.register_forward_pre_hook(mk(l)) for l in layers]
        try:
            with torch.enable_grad():
                lg = D.model(sub['ids_r']).logits[:, -1, :].float()
                ix = torch.arange(sub['n'], device=device)
                # sum, not mean: each example's PLD depends only on its own handle row
                (scale * (lg[ix, sub['tok_nc']] - lg[ix, sub['tok_b']]).sum()).backward()
        finally:
            for h in handles:
                h.remove()
        for l in layers:
            g = hs_store[l].grad / scale
            attr[l] += ((a_corr[l][lo:hi] - a_store[l]) * g).sum(0).double().cpu().numpy()
        hs_store.clear()
        a_store.clear()
        del lg
        torch.cuda.empty_cache()
    del a_corr
    torch.cuda.empty_cache()
    return {l: (attr[l] / bat['n']).astype(np.float32) for l in layers}


# ----------------------------------------------------------------------------------------------
# Receptive fields over a dense (y1, y2, y3) grid
# ----------------------------------------------------------------------------------------------
def receptive_fields(grid, T, pos3, layers, idx, bs=CLOUD_BS):
    """Post-SwiGLU activations of neurons `idx` on the full `grid**3` cube of prompts.

    Returned y1-major, i.e. reshapeable to (G, G, G, n_neurons) with axes (y1, y2, y3). Every cell
    is one deterministic forward pass on one prompt -- not an average over repeats -- so structure
    in the field is real per-number variation rather than sampling noise.

    The leading-digits-distinct guard is deliberately **not** applied: nothing here reads an answer
    token, only activations, so the cube is allowed to be complete.
    """
    trips = np.array([(a, b, c) for a in grid for b in grid for c in grid], dtype=np.int64)
    chunks = []
    for i0 in range(0, len(trips), bs):
        enc = D.tokenizer([prompt_gen(list(t)) for t in trips[i0:i0 + bs]],
                          return_tensors='pt')['input_ids']
        assert enc.shape[1] == T, 'grid prompt length must match the template'
        chunks.append(enc.to(device))
    acts = C.capture_neurons(torch.cat(chunks, dim=0), layers, pos3, bs=bs, idx=idx)
    del chunks
    torch.cuda.empty_cache()
    return trips, {l: v.cpu().numpy().astype(np.float32) for l, v in acts.items()}


# ----------------------------------------------------------------------------------------------
# 3-D dose-response sweep over span(v1, v2, v3)
# ----------------------------------------------------------------------------------------------
def sweep_grid_values(low, high):
    """One grid value per distinct leading digit, placed mid-decade (15, 25, ... 95 for 2-digit).

    The model names a number by emitting its leading digit, so two grid values sharing one would be
    indistinguishable in the answer logits. One value per digit is the finest grid that stays
    readable.
    """
    step = 10 ** (len(str(high - 1)) - 1)
    return [v for v in (d * step + step // 2 for d in range(1, 10)) if low <= v < high]


def first_tok_of(value, slot):
    """The token the model must emit to name `value` at `slot`.

    Read off a prompt with `value` in every slot: `num_char_spans` advances its cursor past each
    match, so slot `i` picks the i-th occurrence and the span is unambiguous. Same derivation
    `make_batch` uses for tok_nc / tok_b, so the sweep and the interchange conditions score the
    same event.
    """
    trip = [value] * K
    p = prompt_gen(trip)
    enc = D.tokenizer(p, return_tensors='pt', return_offsets_mapping=True)
    span = num_tok_span(p, trip, slot, enc['offset_mapping'][0].numpy())
    return int(enc['input_ids'][0, span[0]])


def mean_coords(cloud_coords, values_per_slot, grid, win=SWEEP_WIN):
    """{grid value: (c1, c2, c3)} -- the cloud-mean coordinate a number puts on each axis.

    Axis j's coordinate for value y is averaged over the cloud rows whose *slot j* number is within
    `win` of y. The window smooths what would otherwise be a handful of rows per exact value.
    """
    out = {}
    for y in grid:
        c = []
        for j in range(3):
            m = np.abs(values_per_slot[:, j] - y) <= win
            assert m.sum() > 0, f'no cloud rows near y{j + 1}={y}'
            c.append(cloud_coords[m, j].mean())
        out[y] = np.array(c, dtype=np.float32)
    return out


def sweep_space(bat, P, coord_of, grid, pos, bs=EVAL_BS):
    """Set the three plane coordinates to (y1*, y2*, y3*) over a cube and read off what is answered.

    The continuous counterpart of the rank-3 interchange, and the three-number version of
    `data_sharedrep.sweep_uv_plane`. Instead of copying one counterfactual example's coordinates it
    *sets* all three to the cloud-mean coordinate of a chosen number, over a grid of triples, and
    asks what comes out.

    **What is scored.** The model only ever emits a number that is in its prompt, so "did it say
    y1*?" is unanswerable -- t(y1*) never wins. What the implanted coordinates can decide is which
    **slot** the comparison picks, which is exactly the mechanism's claim: v1 carries the first
    slot's number, v2 the second's, v3 the third's, and the read-out names whichever slot won. The
    returned `argmax` is therefore scored against each slot's own token table by the caller, giving
    three probability fields over the cube.

    Cells with a repeated coordinate (y_i* == y_j*) are skipped and left as -1: they sit on a
    decision boundary with no predicted winner.

    This is the interchange-intervention logic of Geiger et al. 2021/2023 run as a dose-response
    surface rather than at a single counterfactual point.
    """
    n, G = bat['n'], len(grid)
    am = np.full((G, G, G, n), -1, dtype=np.int64)
    Pt = P if torch.is_tensor(P) else torch.tensor(P, device=device)
    for i, y1 in enumerate(grid):
        for j, y2 in enumerate(grid):
            for k, y3 in enumerate(grid):
                if len({y1, y2, y3}) < 3:
                    continue
                target = (coord_of[y1][0] * Pt[:, 0] + coord_of[y2][1] * Pt[:, 1]
                          + coord_of[y3][2] * Pt[:, 2])
                for lo in range(0, n, bs):
                    hi = min(lo + bs, n)
                    handles = S.premlp_patch_handles(
                        target.unsqueeze(0).expand(hi - lo, -1), pos, Pt)
                    try:
                        with torch.no_grad():
                            lg = D.model(bat['ids_r'][lo:hi]).logits[:, -1, :].float()
                    finally:
                        for h in handles:
                            h.remove()
                    am[i, j, k, lo:hi] = lg.argmax(dim=-1).cpu().numpy()
                    del lg
        print(f'  sweep slab y1*={y1}: done', flush=True)
    return am


# ----------------------------------------------------------------------------------------------
# Late layers, read at the last prompt token: layer-generic per-head machinery
# ----------------------------------------------------------------------------------------------
# `data_sharedrep`'s head functions close over module globals that `init_head_machinery` pins to
# H_LAYER, which is all sections 1-12 ever need. The late-layer sweep asks the same three questions
# of some other layer, so the same three operations -- read the `o_proj` input, turn it into one
# head's own residual-space contribution, rewrite that contribution at one position -- are repeated
# here with the layer as an argument. The decomposition is unchanged: `o_proj`'s output is the sum
# over heads of z_h W_O^h (Elhage et al. 2021, the residual-stream / attention-head view), so
# rewriting one head is exactly `out += target_h - current_h` at that position, and the o_proj bias
# is left alone because it belongs to no single head.
_WO_CACHE = {}


def w_o_of(layer):
    """(d_model, d_model) o_proj weight of `layer`, cached; its input is head-major."""
    if layer not in _WO_CACHE:
        _WO_CACHE[layer] = D.model.model.layers[layer].self_attn.o_proj.weight.detach().float()
    return _WO_CACHE[layer]


def head_out_at(ctx, head, layer):
    """ctx: (..., d_model) o_proj input of `layer` -> `head`'s own output contribution."""
    s, e = head * D.head_dim, (head + 1) * D.head_dim
    return ctx[..., s:e].float() @ w_o_of(layer)[:, s:e].T


def capture_ctx_at(ids, pos, layer, bs=EVAL_BS):
    """(B, d_model) `layer`'s o_proj input at `pos`. One capture serves every head of that layer,
    which is why the context is stored rather than the per-head outputs."""
    store, chunks = {}, []
    hnd = D.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(
        lambda mod, inp: store.__setitem__('ctx', inp[0][:, pos, :].detach().float()))
    try:
        for lo in range(0, ids.shape[0], bs):
            with torch.no_grad():
                _ = D.model(ids[lo:lo + bs])
            chunks.append(store.pop('ctx'))
    finally:
        hnd.remove()
    return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]


def head_patch_handles_at(layer, head, pos, target, P=None):
    """`S.head_patch_handles` with the layer as an argument. P None -> full interchange of that
    head's output at `pos`; P (d, k) -> rank-k interchange inside it. Caller removes the handles."""
    o_proj = D.model.model.layers[layer].self_attn.o_proj
    store = {}
    s, e = head * D.head_dim, (head + 1) * D.head_dim
    W_h = w_o_of(layer)[:, s:e]

    def pre(mod, inp):
        store['ctx'] = inp[0]

    def post(mod, inp, out):
        cur = store['ctx'][:, pos, s:e].float() @ W_h.T
        delta = (target - cur) if P is None else ((target - cur) @ P) @ P.T
        out_new = out.clone()
        out_new[:, pos, :] = out[:, pos, :] + delta.to(out.dtype)
        return out_new

    return [o_proj.register_forward_pre_hook(pre), o_proj.register_forward_hook(post)]
