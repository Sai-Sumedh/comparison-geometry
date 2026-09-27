"""Appendix compute, three numbers: residual-stream causal trace over (layer, question position)
for the three perturbation cases of explore_multicompare.ipynb, on its 400 held-out quadruples.

Usage: python compute_app_trace_k3.py <case>   (case in y1, y2, y3; one SLURM job per case)
Writes results/app_trace_k3_<case>.npz.
"""

import sys
import time

import numpy as np
import torch

import data_numberreps as D
import data_multicompare as MC

N_UNITS = 28
_done = 0


def question_positions(T, low, high):
    ref = [low + 1, low + 2, low + 3]
    p = MC.prompt_gen(ref)
    enc = D.tokenizer(p, return_tensors='pt', return_offsets_mapping=True)
    offs = enc['offset_mapping'][0].numpy()
    spans = [MC.num_tok_span(p, ref, i, offs) for i in range(3)]
    positions = list(range(spans[0][0], T))
    labels = []
    for t in positions:
        lab = D.tokenizer.decode([int(enc['input_ids'][0, t])])
        for i, s in enumerate(spans):
            if t in s:
                lab = f'y{i + 1}' if t == s[-1] else f'y{i + 1}_first'
        labels.append(lab)
    return positions, labels


if __name__ == '__main__':
    case = sys.argv[1]
    assert case in MC.CASE_LIST
    OUT = f'results/app_trace_k3_{case}.npz'
    t0 = time.time()
    D.load_model()
    cfg = MC.REGIMES['2digit']
    T, pos3, _, _ = MC.regime_positions(cfg['low'], cfg['high'])
    pool = np.load('results/multicompare_pool_2digit.npz')
    assert int(pool['T']) == T and list(pool['pos']) == list(pos3)
    quads = [tuple(int(v) for v in q) for q in pool['eval_quads']]
    positions, labels = question_positions(T, cfg['low'], cfg['high'])
    print(f'T={T}, question positions {positions} = {labels}', flush=True)

    bat = MC.make_batch(quads, MC.CASES[case], T)
    bat['pld_clean'], bat['pld_corr'] = D.pos_anchors(bat)
    rec = np.zeros((D.n_layers, len(positions), bat['n']), dtype=np.float32)
    iia = np.zeros_like(rec)
    for l in range(D.n_layers):
        D.LAYER = l
        ch = D.clean_hidden(bat['ids_c'])
        for j, p in enumerate(positions):
            r, i = D.eval_patched(lambda c, _p=p: D.make_full_patch_hook(c, _p), bat, ch)
            rec[l, j], iia[l, j] = r, i
        del ch
        torch.cuda.empty_cache()
        _done += 1
        print(f'  {case} L{l}: max PR {rec[l].mean(1).max():.3f}   PROGRESS {_done}/{N_UNITS}',
              flush=True)
    np.savez_compressed(OUT, positions=np.array(positions), labels=np.array(labels),
                        n=np.array(len(quads)), order=np.array(MC.CASES[case]), posrec=rec, iia=iia)
    print(f'saved {OUT} ({time.time() - t0:.0f} s)\ndone.', flush=True)
