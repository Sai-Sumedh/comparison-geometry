"""Appendix compute: behavioural accuracy on pairwise max, and what the model says under patching.

Writes
  results/app_behaviour.npz        -- 2-digit accuracy over every ordered pair; 3-digit accuracy on
                                      a random sample of ordered pairs
  results/app_patch_outcomes.npz   -- the number the model generates for 400 held-out
                                      counterfactuals, unpatched and under two interventions
"""

import re
import time

import numpy as np
import torch

import data_numberreps as D

OUT_BEH = 'results/app_behaviour.npz'
OUT_PATCH = 'results/app_patch_outcomes.npz'
BS = 96
N_3DIGIT = 4000
N_UNITS = -(-90 * 89 // BS) + -(-N_3DIGIT // BS) + 3 * -(-D.N_EVAL // BS)   # decode batches
_done = 0


def greedy_numbers(prompts, n_new, hook=None, bs=BS):
    """Greedy-decode `n_new` tokens per prompt (manual loop, so a patch hook stays attached for
    every step) and return the first integer in each continuation, or -1 if there is none."""
    global _done
    out, text = [], []
    for lo in range(0, len(prompts), bs):
        ids = D.tokenizer(prompts[lo:lo + bs], return_tensors='pt')['input_ids'].to(D.device)
        assert ids.shape[1] == D.tokenizer(prompts[lo], return_tensors='pt')['input_ids'].shape[1]
        hnd = D.model.model.layers[D.LAYER].register_forward_hook(hook(lo, lo + ids.shape[0])) \
            if hook is not None else None
        try:
            cur = ids
            with torch.no_grad():
                for _ in range(n_new):
                    nxt = D.model(cur).logits[:, -1:, :].argmax(-1)
                    cur = torch.cat([cur, nxt], dim=1)
        finally:
            if hnd is not None:
                hnd.remove()
        for g in D.tokenizer.batch_decode(cur[:, ids.shape[1]:]):
            m = re.search(r'\d+', g)
            out.append(int(m.group()) if m else -1)
            text.append(g)
        _done += 1
        print(f'PROGRESS {_done}/{N_UNITS}', flush=True)
    return np.array(out), text


def behaviour():
    t0 = time.time()
    lo2, hi2 = D.REGIMES['2digit']['low'], D.REGIMES['2digit']['high']
    pairs2 = [(a, b) for a in range(lo2, hi2) for b in range(lo2, hi2) if a != b]
    gen2, _ = greedy_numbers([D.prompt_gen(p) for p in pairs2], n_new=3)
    pairs2 = np.array(pairs2)
    correct2 = gen2 == pairs2.max(1)
    print(f'2-digit: {len(pairs2)} ordered pairs, accuracy {correct2.mean():.4f}', flush=True)

    lo3, hi3 = D.REGIMES['3digit']['low'], D.REGIMES['3digit']['high']
    rng = np.random.default_rng(D.SEED)
    pairs3 = []
    while len(pairs3) < N_3DIGIT:
        a, b = rng.integers(lo3, hi3, 2)
        if a != b:
            pairs3.append((int(a), int(b)))
    gen3, _ = greedy_numbers([D.prompt_gen(p) for p in pairs3], n_new=4)
    pairs3 = np.array(pairs3)
    correct3 = gen3 == pairs3.max(1)
    print(f'3-digit: {len(pairs3)} ordered pairs, accuracy {correct3.mean():.4f}  '
          f'({time.time() - t0:.0f} s)', flush=True)
    np.savez_compressed(OUT_BEH, pairs2=pairs2, gen2=gen2, correct2=correct2,
                        pairs3=pairs3, gen3=gen3, correct3=correct3)


def patch_outcomes():
    """Categorise the generated answer under interventions that move the winning operand."""
    t0 = time.time()
    cfg = D.REGIMES['2digit']
    T, (pos_a, pos_b), _, _ = D.regime_positions(cfg['low'], cfg['high'])
    eval_triples = D.build_triples(D.N_EVAL + D.N_DAS, D.SEED, cfg['low'], cfg['high'],
                                   cfg['gap'])[:D.N_EVAL]
    u = np.load('results/numberreps_2digit.npz')['das_dir']
    Qu = torch.tensor(u, device=D.device, dtype=torch.float32).reshape(-1, 1)
    D.LAYER = 13
    res = {}
    conds = [('none', 'n2_larger', None, None),
             ('full_at_y2', 'n2_larger', 'full', pos_b),
             ('u_at_y1', 'n1_larger', 'u', pos_a)]
    for name, case, kind, pos in conds:
        slot = 0 if case == 'n1_larger' else 1
        bat = D.make_batch(eval_triples, slot, T)
        hook = None
        if kind is not None:
            ch = D.clean_hidden(bat['ids_c'])
            if kind == 'full':
                hook = (lambda lo, hi, _c=ch, _p=pos: D.make_full_patch_hook(_c[lo:hi], _p))
            else:
                hook = (lambda lo, hi, _c=ch, _p=pos: D.make_subspace_patch_hook(Qu, _c[lo:hi], _p))
        prompts = [D.prompt_gen(list(p)) for p in bat['corr_pairs']]
        gen, text = greedy_numbers(prompts, n_new=3, hook=hook)
        clean_max = bat['clean_pairs'].max(1)
        r_val = bat['corr_pairs'][:, slot]            # the corrupted prompt's own value at the winner
        b_val = bat['corr_pairs'][:, 1 - slot]
        cat = np.full(len(gen), 3)                   # 0 = r, 1 = b, 2 = clean max a, 3 = other
        cat[gen == r_val], cat[gen == b_val], cat[gen == clean_max] = 0, 1, 2
        res[f'{name}_gen'] = gen
        res[f'{name}_cat'] = cat
        res[f'{name}_clean'] = bat['clean_pairs']
        res[f'{name}_corr'] = bat['corr_pairs']
        res[f'{name}_text'] = np.array(text)
        print(f'{name:>12} ({case}): r {np.mean(cat == 0):.3f}  b {np.mean(cat == 1):.3f}  '
              f'a {np.mean(cat == 2):.3f}  other {np.mean(cat == 3):.3f}', flush=True)
        torch.cuda.empty_cache()
    res['conditions'] = np.array([c[0] for c in conds])
    np.savez_compressed(OUT_PATCH, **res)
    print(f'patch outcomes done ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    D.load_model()
    behaviour()
    patch_outcomes()
    print('done.', flush=True)
