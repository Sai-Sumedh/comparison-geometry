"""Appendix compute: data_numberreps' L13 analysis repeated at earlier layers.

For each requested layer, on the residual stream leaving that layer:
  * rank-1 DAS direction u at y1's position, fitted in the y1-larger case (same data, seed and
    optimisation settings as data_numberreps.py, seed 0 only),
  * PC1 of the y1 cloud and PC1 of the y2 cloud (the v2 convention),
  * held-out IIA / position recovery of full-rank, u and PC1 patches at y1 (y1-larger case) and of
    full-rank and PC1(y2) patches at y2 (y2-larger case),
  * the components of the 2000-prompt cloud along u and both PC1s.

Usage: python compute_app_u_layers.py <layer> [<layer> ...]
Writes results/app_u_layers_L<layer>.npz per layer.
"""

import sys
import time

import numpy as np
import torch
from sklearn.decomposition import PCA

import data_numberreps as D

LAYERS = [int(a) for a in sys.argv[1:]]
N_UNITS = len(LAYERS)


def run_layer(l, T, pos_a, pos_b, cloud_pairs, das_triples, eval_triples):
    t0 = time.time()
    D.LAYER = l
    resid = np.zeros((len(cloud_pairs), 2, D.d_model), dtype=np.float32)
    for i0 in range(0, len(cloud_pairs), D.CLOUD_BS):
        chunk = cloud_pairs[i0:i0 + D.CLOUD_BS]
        ids = D.tokenizer([D.prompt_gen(list(p)) for p in chunk], return_tensors='pt')['input_ids']
        resid[i0:i0 + len(chunk)] = D.capture_resid(ids.to(D.device), [pos_a, pos_b]).cpu().numpy()
    out = {}
    dirs = {}
    for k, name in ((0, 'a'), (1, 'b')):
        X = resid[:, k, :]
        pca = PCA(n_components=3).fit(X)
        dirs[f'pc1{name}'] = (pca.components_[0] / np.linalg.norm(pca.components_[0])).astype(np.float32)
        out[f'pca3_evr_{name}'] = pca.explained_variance_ratio_
        out[f'pca3_coords_{name}'] = pca.transform(X).astype(np.float32)

    das_bat = D.make_batch(das_triples, 0, T)
    das_bat['pld_clean'], das_bat['pld_corr'] = D.pos_anchors(das_bat)
    ch_das = D.clean_hidden(das_bat['ids_c'])
    Q, curve = D.train_das_1d(das_bat, pos_a, ch=ch_das, init_seed=D.SEED)
    del ch_das
    u = Q[:, 0].cpu().numpy().astype(np.float32)
    dirs['u'] = u
    out.update(u_dir=u, das_loss_curve=curve, pc1a_dir=dirs['pc1a'], pc1b_dir=dirs['pc1b'])

    Xa, Xb = resid[:, 0, :], resid[:, 1, :]
    out['comp_u'] = ((Xa - Xa.mean(0)) @ u).astype(np.float64)
    out['comp_pc1a'] = ((Xa - Xa.mean(0)) @ dirs['pc1a']).astype(np.float64)
    out['comp_pc1b'] = ((Xb - Xb.mean(0)) @ dirs['pc1b']).astype(np.float64)
    tot = ((Xa - Xa.mean(0)) ** 2).sum() / len(Xa)
    out['evr_u'] = np.array(np.var(out['comp_u']) / tot)

    Qt = {k: torch.tensor(v, device=D.device, dtype=torch.float32).reshape(-1, 1)
          for k, v in dirs.items()}
    for case, slot, pos, arms in (('n1_larger', 0, pos_a, ('full', 'u', 'pc1a')),
                                  ('n2_larger', 1, pos_b, ('full', 'pc1b'))):
        bat = D.make_batch(eval_triples, slot, T)
        bat['pld_clean'], bat['pld_corr'] = D.pos_anchors(bat)
        ch = D.clean_hidden(bat['ids_c'])
        for arm in arms:
            fac = ((lambda c, _p=pos: D.make_full_patch_hook(c, _p)) if arm == 'full' else
                   (lambda c, _p=pos, _Q=Qt[arm]: D.make_subspace_patch_hook(_Q, c, _p)))
            r, i = D.eval_patched(fac, bat, ch)
            out[f'posrec_{arm}_{case}'], out[f'iia_{arm}_{case}'] = r, i
            print(f'  L{l} {case} {arm:>5}: IIA {i.mean():.3f}  PR {r.mean():.3f}', flush=True)
        del ch
        torch.cuda.empty_cache()
    out['a_vals'] = cloud_pairs[:, 0].astype(np.float64)
    out['b_vals'] = cloud_pairs[:, 1].astype(np.float64)
    out['layer'] = np.array(l)
    np.savez_compressed(f'results/app_u_layers_L{l}.npz', **out)
    print(f'L{l} done in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    D.load_model()
    cfg = D.REGIMES['2digit']
    T, (pos_a, pos_b), _, _ = D.regime_positions(cfg['low'], cfg['high'])
    cloud_pairs = np.array(D.sample_cloud_pairs(D.N_CLOUD, D.SEED + 7, cfg['low'], cfg['high']))
    pool = D.build_triples(D.N_EVAL + D.N_DAS, D.SEED, cfg['low'], cfg['high'], cfg['gap'])
    eval_triples, das_triples = pool[:D.N_EVAL], pool[D.N_EVAL:]
    for n, l in enumerate(LAYERS, 1):
        run_layer(l, T, pos_a, pos_b, cloud_pairs, das_triples, eval_triples)
        print(f'PROGRESS {n}/{N_UNITS}', flush=True)
    print('done.', flush=True)
