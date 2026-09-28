"""Appendix compute: IIA scored on the whole generated number, next to the first-digit IIA.

Batch version of patching_model_behavior.ipynb. Re-runs the interchange interventions behind the
main-text IIA bars (u, v1, v2, the (v1, v2) plane, the full-rank patches, the rank-1 L15 DAS
direction) on the 400 held-out counterfactuals, greedy-decodes the answer, and scores it two ways:
  first digit  -- argmax of the last-position logits is t(r), the metric of every IIA figure
  whole number -- the greedily decoded answer (one token more than the operands' digit count, the
                  full-number accuracy of the ablations) begins with the number r
The patch is applied on the prompt pass only; generated tokens read it through the KV cache.
Nothing is refitted: every direction is loaded from results/.

Writes results/app_whole_number.npz.
"""

import re
import time

import numpy as np
import torch

import data_numberreps as D
import data_sharedrep as S

OUT = 'results/app_whole_number.npz'
TAG = '2digit'
N_NEW = 3                                   # two digits + one token
CASES = {'y1': 'n1 larger', 'y2': 'n2 larger'}


@torch.no_grad()
def generate(ids, handles_fn, n_new=N_NEW, bs=D.EVAL_BS):
    out = []
    for lo in range(0, ids.shape[0], bs):
        hi = min(lo + bs, ids.shape[0])
        handles = handles_fn(lo, hi)
        try:
            o = D.model(ids[lo:hi], use_cache=True)
        finally:
            for h in handles:
                h.remove()
        pkv, nxt = o.past_key_values, o.logits[:, -1, :].float().argmax(-1)
        toks = [nxt]
        for _ in range(n_new - 1):
            o = D.model(nxt[:, None], past_key_values=pkv, use_cache=True)
            pkv, nxt = o.past_key_values, o.logits[:, -1, :].float().argmax(-1)
            toks.append(nxt)
        out.append(torch.stack(toks, 1).cpu())
        del o, pkv
    torch.cuda.empty_cache()
    return torch.cat(out).numpy()


if __name__ == '__main__':
    t0 = time.time()
    D.load_model()
    S.init_head_machinery()
    tok = D.tokenizer
    cfg = D.REGIMES[TAG]
    T, (pos_a, pos_b), _, _ = D.regime_positions(cfg['low'], cfg['high'])

    R = np.load(f'results/numberreps_{TAG}.npz')
    SR = np.load(f'results/sharedrep_{TAG}.npz')
    Z = np.load(f'results/comparator_data_{TAG}.npz')
    basis = lambda v: torch.tensor(np.asarray(v, np.float32).reshape(v.shape[0], -1),
                                   device=D.device)
    Q_u, Q_v1, Q_v2 = basis(R['das_dir']), basis(SR['h14_das_dir']), basis(SR['l13_pc1b_dir'])
    P_uv, Q_15 = basis(SR['uv_plane_basis']), basis(Z['das15_basis_n1_larger_k1'])

    pool = D.build_triples(D.N_EVAL + D.N_DAS, D.SEED, cfg['low'], cfg['high'], cfg['gap'])
    eval_triples = pool[:D.N_EVAL]
    for arr in (R['triples_n1_larger'], SR['eval_triples'], Z['eval_triples']):
        assert (np.array(eval_triples) == arr).all(), 'eval triples differ from the saved runs'

    # key: (spec for data_sharedrep.build_handles, patch position, cases)
    COND = {
        'L13_full_y1': (dict(resid=[(13, 'full', None)]), pos_a, ['y1']),
        'u':           (dict(resid=[(13, 'sub', Q_u)]), pos_a, ['y1']),
        'v1_premlp':   (dict(premlp=Q_v1), pos_b, ['y1']),
        'L13_full_y2': (dict(resid=[(13, 'full', None)]), pos_b, ['y1', 'y2']),
        'v2':          (dict(resid=[(13, 'sub', Q_v2)]), pos_b, ['y1', 'y2']),
        'v1':          (dict(head=(S.H_HEAD, Q_v1)), pos_b, ['y1', 'y2']),
        'v1_v2':       (dict(resid=[(13, 'sub', Q_v2)], head=(S.H_HEAD, Q_v1)), pos_b, ['y1', 'y2']),
        'plane':       (dict(premlp=P_uv), pos_b, ['y1', 'y2']),
        'L14_full':    (dict(resid=[(14, 'full', None)]), pos_b, ['y1', 'y2']),
        'L15_das':     (dict(resid=[(15, 'sub', Q_15)]), pos_b, ['y1', 'y2']),
    }
    # the first-digit IIA already saved for each run, as a consistency check
    saved = {('L13_full_y1', 'y1'): R['iia_full_n1_larger_first_number'],
             ('u', 'y1'): R['iia_das_n1_larger_first_number'],
             ('L15_das', 'y1'): Z['das15_iia_n1_larger_from_n1_larger_k1'],
             ('L15_das', 'y2'): Z['das15_iia_n2_larger_from_n1_larger_k1']}
    for k, sk in [('L13_full_y2', 'L13_full_atn2'), ('v2', 'L13_PC1_atn2'), ('v1', 'H14_DAS'),
                  ('v1_v2', 'H14_DAS_and_L13_PC1_atn2'), ('plane', 'L14_uv_plane_preMLP_atn2'),
                  ('L14_full', 'L14_full_atn2')]:
        for c in CASES:
            saved[k, c] = SR[('' if c == 'y1' else 'alt_') + 'cond_iia_' + sk]

    digit = {str(d): tok.convert_tokens_to_ids(str(d)) for d in range(10)}
    out = {'conditions': np.array(list(COND))}
    n_runs, done = sum(len(v[2]) for v in COND.values()), 0
    for c, case in CASES.items():
        b = D.make_batch(eval_triples, D.CASE_SLOT[case], T)
        r = b['corr_pairs'][:, b['win_slot']]
        t_r = np.array([digit[str(v)[0]] for v in r])
        assert (t_r == b['tok_nc'].cpu().numpy()).all()
        src = {l: S.clean_hidden_at(b['ids_c'], l) for l in (13, 14, 15)}
        src['ctx'] = S.capture_ctx(b['ids_c'], pos_b)
        src['premlp'] = S.capture_premlp(b['ids_c'], pos_b)
        out.update({f'clean_{c}': b['clean_pairs'], f'corr_{c}': b['corr_pairs'], f'r_{c}': r})
        for k, (spec, pos, cases) in COND.items():
            if c not in cases:
                continue
            gen = generate(b['ids_r'], lambda lo, hi: S.build_handles(spec, src, lo, hi, pos))
            text = [tok.decode(g) for g in gen]
            num = np.array([int(m.group(1)) if (m := re.match(r'\s*(\d+)', t)) else -1
                            for t in text])
            first = (gen[:, 0] == t_r).astype(np.float64)
            whole = (num == r).astype(np.float64)
            out.update({f'{k}_{c}_first': first, f'{k}_{c}_whole': whole,
                        f'{k}_{c}_num': num, f'{k}_{c}_text': np.array(text)})
            sv = saved.get((k, c))                  # v1_premlp has no saved counterpart
            ref = 'no saved value' if sv is None else \
                f'saved {sv.mean():.3f}, agree {(sv == first).mean():.3f}'
            done += 1
            print(f'{k:>12} {c}: first digit {first.mean():.3f} ({ref})   '
                  f'whole number {whole.mean():.3f}', flush=True)
            print(f'PROGRESS {done}/{n_runs}', flush=True)
        del src
        torch.cuda.empty_cache()
    np.savez_compressed(OUT, **out)
    print(f'saved {OUT} ({time.time() - t0:.0f} s)\ndone.', flush=True)
