"""n2-position machinery for the three-number `max(n1, n2, n3)` comparator.

`data_multicompare.py` reads everything at n3, where all three numbers are present. This module is
the small delta needed to run the same analyses at **n2**, and imports everything else from it --
prompt, quadruple pool, counterfactual batches, intervention specs, DAS, attribution, clouds.

Why n2 is a different object. Attention is causal, so the residual at n2 can only hold y1 and y2;
y3 has not been read yet. The space is therefore two-dimensional at most, the `y3` perturbation
case is meaningless (its counterfactual changes a token strictly to the right of the patch site),
and the receptive field of a neuron read at n2 is a surface over (y1, y2) rather than a cube.

The two cases kept are exactly `data_multicompare.CASES['y1']` and `['y2']`, so every number here
is measured on the same quadruples, in the same value orders, against the same clean/corrupted
construction as the n3 notebook -- only the patch position moves. That makes the pair
(n2 result, n3 result) a within-design comparison rather than two separate experiments.

    y1 case, order s1 > s3 > s2 : the maximum sits at n1, so n2 can only matter through a copy of
                                  y1 that some earlier head wrote there.
    y2 case, order s2 > s3 > s1 : the maximum sits at n2 itself, so n2 is the source position.

See docs/explore_multicompare2.md.
"""

import numpy as np
import torch

import data_numberreps as D
import data_sharedrep as S
import data_comparator as C
import data_multicompare as MC
from data_multicompare import (LAYER, H_LAYER, SWEEP_WIN, CLOUD_BS, prompt_gen)
from data_numberreps import EVAL_BS, device

# The n3 notebook's cases minus y3. Values are `order_key` tuples: slot indices in descending
# value order, so (0, 2, 1) reads s1 > s3 > s2.
CASES2 = {'y1': MC.CASES['y1'],
          'y2': MC.CASES['y2']}
CASE_LIST2 = ['y1', 'y2']


def mean_coords_k(cloud_coords, values_per_slot, grid, k, win=SWEEP_WIN):
    """{grid value: (c_1, ..., c_k)} -- the cloud-mean coordinate a number puts on each of the
    first `k` axes. `MC.mean_coords` fixed to three axes; the n2 plane has two."""
    out = {}
    for y in grid:
        c = []
        for j in range(k):
            m = np.abs(values_per_slot[:, j] - y) <= win
            assert m.sum() > 0, f'no cloud rows near y{j + 1}={y}'
            c.append(cloud_coords[m, j].mean())
        out[y] = np.array(c, dtype=np.float32)
    return out


def sweep_plane(bat, P, coord_of, grid, pos, bs=EVAL_BS):
    """Set the two plane coordinates at `pos` to (y1*, y2*) over a grid and read off the answer.

    The two-dimensional counterpart of `MC.sweep_space`, and the three-number counterpart of
    `data_sharedrep.sweep_uv_plane`: instead of copying one counterfactual example's coordinates
    it *sets* both to the cloud-mean coordinate of a chosen number and asks what comes out.

    Returns (ld, am), both (G, G, n):
      ld -- logit[tok_nc] - logit[tok_b], the quantity position recovery is built from, so the
            surface and the interchange conditions score one thing;
      am -- argmax token id, which the caller scores against each slot's own token table to get
            "which slot did the model name" as a probability field.

    Cells with y1* == y2* sit on the decision boundary and are skipped (nan / -1).
    """
    n, G = bat['n'], len(grid)
    ld = np.full((G, G, n), np.nan, dtype=np.float32)
    am = np.full((G, G, n), -1, dtype=np.int64)
    Pt = P if torch.is_tensor(P) else torch.tensor(P, device=device)
    for i, y1 in enumerate(grid):
        for j, y2 in enumerate(grid):
            if y1 == y2:
                continue
            target = coord_of[y1][0] * Pt[:, 0] + coord_of[y2][1] * Pt[:, 1]
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
                ix = torch.arange(hi - lo, device=device)
                ld[i, j, lo:hi] = (lg[ix, bat['tok_nc'][lo:hi]]
                                   - lg[ix, bat['tok_b'][lo:hi]]).cpu().numpy()
                am[i, j, lo:hi] = lg.argmax(dim=-1).cpu().numpy()
                del lg
        print(f'  sweep row y1*={y1}: done', flush=True)
    return ld, am


def receptive_fields_2d(grid, y3_fixed, T, pos, layers, idx, bs=CLOUD_BS):
    """Post-SwiGLU activations of neurons `idx` at `pos`, over the `grid**2` (y1, y2) plane.

    `MC.receptive_fields` sweeps a (y1, y2, y3) cube because a neuron read at n3 sees all three
    numbers. At n2 the causal mask makes y3 unreachable, so it is pinned at one value and the
    field is a surface. Returned y1-major, i.e. reshapeable to (G, G, n_neurons) with axes
    (y1, y2). Every cell is one deterministic forward pass on one prompt.
    """
    pairs = np.array([(a, b, y3_fixed) for a in grid for b in grid], dtype=np.int64)
    chunks = []
    for i0 in range(0, len(pairs), bs):
        enc = D.tokenizer([prompt_gen(list(t)) for t in pairs[i0:i0 + bs]],
                          return_tensors='pt')['input_ids']
        assert enc.shape[1] == T, 'grid prompt length must match the template'
        chunks.append(enc.to(device))
    acts = C.capture_neurons(torch.cat(chunks, dim=0), layers, pos, bs=bs, idx=idx)
    del chunks
    torch.cuda.empty_cache()
    return pairs, {l: v.cpu().numpy().astype(np.float32) for l, v in acts.items()}


def y3_is_unreachable(T, pos, low, high, tol=0.0):
    """Sanity check behind everything above: the residual at `pos` must not depend on y3.

    Two prompts differing only in their third number, read at `pos` after layer `LAYER` and after
    `H_LAYER`. Returns the largest absolute difference at each layer; both should be exactly 0
    under a causal mask.
    """
    a, b = [low + 1, low + 2, low + 3], [low + 1, low + 2, high - 1]
    ids = D.tokenizer([prompt_gen(a), prompt_gen(b)], return_tensors='pt')['input_ids'].to(device)
    assert ids.shape[1] == T, 'check prompts must match the template'
    got = {}

    def mk(l):
        def _h(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            got[l] = hs[:, pos, :].detach().float()
        return _h

    handles = [D.model.model.layers[l].register_forward_hook(mk(l)) for l in (LAYER, H_LAYER)]
    try:
        with torch.no_grad():
            _ = D.model(ids)
    finally:
        for h in handles:
            h.remove()
    return {l: float((got[l][0] - got[l][1]).abs().max()) for l in (LAYER, H_LAYER)}


# ----------------------------------------------------------------------------------------------
# Whole-task ablation: does the two-number circuit carry `max` itself?
# ----------------------------------------------------------------------------------------------
# Everything above measures a counterfactual at one token. These three functions ask the blunter
# question instead: knock the neurons out of the model entirely and see whether it can still do
# the task, over operand lists of increasing length. Zero-ablation of MLP neurons (setting the
# post-SwiGLU activation to 0, its value when the gate is closed) is the standard knockout in this
# literature -- Geva et al. 2021/2022 for the key-value view of MLP neurons, and the mean/zero
# ablation contrast discussed in Wang et al. (2022) and Conmy et al. (2023).
def make_neuron_zero_hook(idx):
    """Zero neurons `idx` of one MLP at **every** position, not just the patch site.

    Registered on `down_proj`'s input, which is the post-SwiGLU activation vector, so setting a
    column to 0 removes that neuron's `a_j * W_down[:, j]` write from the residual everywhere.
    """
    def _pre(mod, inp, _i=idx):
        x = inp[0].clone()
        x[..., _i] = 0
        return (x,) + tuple(inp[1:])
    return _pre


def sample_max_prompts(k, n, seed, low, high):
    """`n` lists of `k` distinct values in [low, high), for the `max` task at list length `k`.

    Distinct values only -- a tie has no unique answer. No leading-digit constraint, because the
    answer is scored by decoding the generated digits rather than by reading one token, so
    operands are free to share a leading digit (unavoidable once k > 9).
    """
    span = high - low
    assert k <= span, f'cannot draw {k} distinct values from [{low}, {high})'
    g = torch.Generator().manual_seed(seed)
    out, seen = [], set()
    while len(out) < n:
        t = tuple((torch.randperm(span, generator=g)[:k] + low).tolist())
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def task_accuracy(nums_list, n_digits, make_handles=None, bs=16):
    """Greedy-decode the answer for each prompt and compare it with the true maximum.

    Returns a bool array, one entry per prompt. `make_handles() -> [handles]` installs an
    ablation for the duration of the decode; None runs the intact model. Decoding is a manual
    argmax loop rather than `generate` so the ablation hooks stay attached across all steps and
    nothing depends on a generation config.

    Within one call every prompt has the same operand count and digit count, so they tokenise to
    the same length and no padding is needed -- asserted per minibatch.
    """
    ok = []
    for i0 in range(0, len(nums_list), bs):
        chunk = nums_list[i0:i0 + bs]
        ids = D.tokenizer([prompt_gen(list(t)) for t in chunk],
                          return_tensors='pt')['input_ids'].to(device)
        assert (ids.shape[1] == D.tokenizer(prompt_gen(list(chunk[0])),
                                            return_tensors='pt')['input_ids'].shape[1]), \
            'prompts in one call must share a token length'
        handles = make_handles() if make_handles is not None else []
        try:
            cur = ids
            with torch.no_grad():
                for _ in range(n_digits + 1):
                    nxt = D.model(cur).logits[:, -1:, :].argmax(-1)
                    cur = torch.cat([cur, nxt], dim=1)
        finally:
            for h in handles:
                h.remove()
        gen = D.tokenizer.batch_decode(cur[:, ids.shape[1]:])
        ok += [g.strip()[:n_digits] == str(max(t)) for g, t in zip(gen, chunk)]
        del cur
    torch.cuda.empty_cache()
    return np.array(ok, dtype=bool)


def make_neuron_zero_hook_at(idx, pos):
    """Zero neurons `idx` of one MLP at a single token position.

    The position-restricted counterpart of `make_neuron_zero_hook`. Together the two separate
    "these neurons are needed somewhere in the forward pass" from "these neurons are needed at
    the token where the comparison is read". `pos` is an absolute index into the prompt, so it
    stays valid as greedy decoding appends tokens to the right of it.
    """
    def _pre(mod, inp, _i=idx, _p=pos):
        x = inp[0].clone()
        x[:, _p, _i] = 0
        return (x,) + tuple(inp[1:])
    return _pre


def last_number_pos(nums):
    """Token index of the **last** token of the final operand -- the k-operand generalisation of
    the n3 read-out site, where every number has been seen under causal attention."""
    p = prompt_gen(list(nums))
    enc = D.tokenizer(p, return_tensors='pt', return_offsets_mapping=True)
    offs = enc['offset_mapping'][0].numpy()
    return int(MC.num_tok_span(p, list(nums), len(nums) - 1, offs)[-1])
