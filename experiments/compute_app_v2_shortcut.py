"""Appendix compute: why v2 is PC1 rather than a DAS fit.

Fits a rank-1 DAS direction on the L13 residual at y2's position in the y2-larger case (the case
that perturbs y2), with data_numberreps.py's data, seed and optimisation settings, then
  * scores it, PC1 of the y2 cloud and the full-rank patch on the 400 held-out counterfactuals,
  * projects the cached 2000-prompt y2 cloud (results/numberreps_2digit.npz) onto it.

Writes results/app_v2_shortcut.npz.
"""

import time

import numpy as np
import torch

import data_numberreps as D

OUT = 'results/app_v2_shortcut.npz'

if __name__ == '__main__':
    t0 = time.time()
    D.load_model()
    D.LAYER = 13
    cfg = D.REGIMES['2digit']
    T, (pos_a, pos_b), _, _ = D.regime_positions(cfg['low'], cfg['high'])
    pool = D.build_triples(D.N_EVAL + D.N_DAS, D.SEED, cfg['low'], cfg['high'], cfg['gap'])
    eval_triples, das_triples = pool[:D.N_EVAL], pool[D.N_EVAL:]

    das_bat = D.make_batch(das_triples, 1, T)
    das_bat['pld_clean'], das_bat['pld_corr'] = D.pos_anchors(das_bat)
    Q, curve = D.train_das_1d(das_bat, pos_b, init_seed=D.SEED)
    w = Q[:, 0].cpu().numpy().astype(np.float32)
    print('PROGRESS 1/2', flush=True)

    R = np.load('results/numberreps_2digit.npz')
    pc1b = R['pc1b_dir'].astype(np.float32)
    out = dict(das_dir=w, das_loss_curve=curve, das_pc1b_cos=np.array(float(w @ pc1b)))
    bat = D.make_batch(eval_triples, 1, T)
    bat['pld_clean'], bat['pld_corr'] = D.pos_anchors(bat)
    ch = D.clean_hidden(bat['ids_c'])
    for arm, fac in (('full', lambda c: D.make_full_patch_hook(c, pos_b)),
                     ('das', lambda c: D.make_subspace_patch_hook(Q, c, pos_b)),
                     ('pc1b', lambda c: D.make_subspace_patch_hook(
                         torch.tensor(pc1b, device=D.device).reshape(-1, 1), c, pos_b))):
        r, i = D.eval_patched(fac, bat, ch)
        out[f'posrec_{arm}'], out[f'iia_{arm}'] = r, i
        print(f'  {arm:>5}: IIA {i.mean():.3f}  PR {r.mean():.3f}', flush=True)

    Xb = R['resid_b'].astype(np.float64)
    Xb -= Xb.mean(0)
    out['comp_das'] = Xb @ w.astype(np.float64)
    out['comp_pc1b'] = Xb @ pc1b.astype(np.float64)
    out['a_vals'], out['b_vals'] = R['a_vals'], R['b_vals']
    np.savez_compressed(OUT, **out)
    print(f'cos(DAS@y2, PC1 of y2 cloud) = {float(w @ pc1b):+.3f}', flush=True)
    print(f'PROGRESS 2/2\nsaved {OUT} ({time.time() - t0:.0f} s)\ndone.', flush=True)
