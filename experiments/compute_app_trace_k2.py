"""Appendix compute, two numbers: residual-stream causal trace and linear probes over
(layer, question-clause position).

Writes
  results/app_trace_k2.npz   -- position recovery / IIA of restoring one clean residual vector at
                                every (layer, position), for both perturbation cases
  results/app_probes_k2.npz  -- held-out ridge-probe R^2 for log y1 and log y2, and held-out
                                accuracy of a ridge classifier for 1[y1 > y2], at every cell
"""

import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import data_numberreps as D

N_UNITS = 2 * 28 + 1 + 28      # trace layers per case, cloud capture, probe layers (for PROGRESS)
_done = 0


def tick(k=1):
    global _done
    _done += k
    print(f'PROGRESS {_done}/{N_UNITS}', flush=True)

OUT_TRACE, OUT_PROBE = 'results/app_trace_k2.npz', 'results/app_probes_k2.npz'
N_TRACE = 400
ALPHAS = np.logspace(-2, 4, 13)        # the same grid the other probes in this work use
TEST_FRAC = 0.2


def question_positions(T, low, high):
    """Token indices from the first token of y1 to the last prompt token, with a label for each."""
    ref = [low + 1, low + 2]
    p = D.prompt_gen(ref)
    enc = D.tokenizer(p, return_tensors='pt', return_offsets_mapping=True)
    offs = enc['offset_mapping'][0].numpy()
    spans = [D.num_tok_span(p, ref, i, offs) for i in range(2)]
    positions = list(range(spans[0][0], T))
    labels = []
    for t in positions:
        lab = D.tokenizer.decode([int(enc['input_ids'][0, t])])
        for i, s in enumerate(spans):
            if t in s:
                lab = f'y{i + 1}' if t == s[-1] else f'y{i + 1}_first'
        labels.append(lab)
    return positions, labels


def trace(T, positions, eval_triples):
    out = {}
    for case, slot in (('n1_larger', 0), ('n2_larger', 1)):
        bat = D.make_batch(eval_triples[:N_TRACE], slot, T)
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
            print(f'  trace {case} L{l}: max PR {rec[l].mean(1).max():.3f}', flush=True)
            tick()
        out[f'posrec_{case}'], out[f'iia_{case}'] = rec, iia
    return out


def capture_cloud(pairs, T, positions, bs=48):
    """(N, n_layers, len(positions), d_model) float16 residual streams leaving every layer."""
    H = torch.zeros((len(pairs), D.n_layers, len(positions), D.d_model), dtype=torch.float16)
    for lo in range(0, len(pairs), bs):
        ids = D.tokenizer([D.prompt_gen(list(p)) for p in pairs[lo:lo + bs]],
                          return_tensors='pt')['input_ids'].to(D.device)
        assert ids.shape[1] == T
        with torch.no_grad():
            hs = D.model(ids, output_hidden_states=True).hidden_states   # [0] is the embedding
        H[lo:lo + ids.shape[0]] = torch.stack([h[:, positions, :] for h in hs[1:]], 1).cpu()
        del hs
    return H


def ridge_loo(Xtr, Xte, ys):
    """Ridge with the penalty chosen by exact leave-one-out on the training split (the RidgeCV
    rule), in dual form on the GPU; one eigendecomposition serves every target in `ys`.
    Returns the held-out predictions for each target."""
    mu = Xtr.mean(0)
    A, B = Xtr - mu, Xte - mu
    lam, V = torch.linalg.eigh(A @ A.T)
    preds = []
    for ytr in ys:
        ym = ytr.mean()
        Vy = V.T @ (ytr - ym)
        best, best_c = None, None
        for a in ALPHAS:
            inv = 1.0 / (lam + a)
            c = V @ (inv * Vy)
            loo = float(((c / ((V ** 2) @ inv)) ** 2).mean())
            if best is None or loo < best:
                best, best_c = loo, c
        preds.append((B @ (A.T @ best_c) + ym).cpu().numpy())
    return preds


def probes(H, pairs):
    y1, y2 = pairs[:, 0].astype(np.float64), pairs[:, 1].astype(np.float64)
    tr, te = train_test_split(np.arange(len(pairs)), test_size=TEST_FRAC, random_state=D.SEED)
    targets = {'logy1': np.log(y1), 'logy2': np.log(y2), 'y1gty2': np.where(y1 > y2, 1.0, -1.0)}
    L, P = H.shape[1], H.shape[2]
    out = {f'r2_{k}': np.zeros((L, P)) for k in ('logy1', 'logy2')}
    out['acc_y1gty2'] = np.zeros((L, P))
    dev = D.device
    for l in range(L):
        for p in range(P):
            X = H[:, l, p, :].to(dev, torch.float64)
            ys = [torch.tensor(t[tr], device=dev) for t in targets.values()]
            for (k, t), pred in zip(targets.items(), ridge_loo(X[tr], X[te], ys)):
                if k == 'y1gty2':
                    out['acc_y1gty2'][l, p] = float(np.mean(np.sign(pred) == t[te]))
                else:
                    ss = ((t[te] - pred) ** 2).sum()
                    out[f'r2_{k}'][l, p] = 1.0 - ss / ((t[te] - t[te].mean()) ** 2).sum()
        print(f'  probes L{l}: R2(log y1) {out["r2_logy1"][l].max():.3f}  '
              f'R2(log y2) {out["r2_logy2"][l].max():.3f}  acc {out["acc_y1gty2"][l].max():.3f}',
              flush=True)
        tick()
    out['test_idx'] = te
    return out


if __name__ == '__main__':
    t0 = time.time()
    D.load_model()
    cfg = D.REGIMES['2digit']
    T, _, _, _ = D.regime_positions(cfg['low'], cfg['high'])
    positions, labels = question_positions(T, cfg['low'], cfg['high'])
    print(f'T={T}, question positions {positions} = {labels}', flush=True)
    eval_triples = D.build_triples(D.N_EVAL + D.N_DAS, D.SEED, cfg['low'], cfg['high'],
                                   cfg['gap'])[:D.N_EVAL]

    tr = trace(T, positions, eval_triples)
    np.savez_compressed(OUT_TRACE, positions=np.array(positions), labels=np.array(labels),
                        n=np.array(N_TRACE), **tr)
    print(f'trace saved ({time.time() - t0:.0f} s)', flush=True)

    pairs = np.array(D.sample_cloud_pairs(D.N_CLOUD, D.SEED + 7, cfg['low'], cfg['high']))
    H = capture_cloud(pairs, T, positions)
    tick()
    pr = probes(H, pairs)
    np.savez_compressed(OUT_PROBE, positions=np.array(positions), labels=np.array(labels),
                        pairs=pairs, **pr)
    print(f'probes saved ({time.time() - t0:.0f} s)\ndone.', flush=True)
