"""Add the MLP15 write-alignment arrays to an existing comparator_data_*.npz.

`data_comparator.py` now computes these inside section 12, but a run made before that block
existed will not have them. This recovers them without redoing the 14-minute run: the DAS bases
are already saved, and the alignment is pure weight algebra -- no forward passes, so the only cost
is loading the model.

    python topup_dasalign.py          # ~1 minute

Idempotent: it merges into the existing file and leaves everything else untouched. Once a full
`data_comparator.py` run has happened, this script is unnecessary.
"""
import json
import os

import numpy as np
import torch

import data_numberreps as D
import data_sharedrep as S
import data_comparator as C

if __name__ == '__main__':
    path = f'{C.RESULTS_DIR}/comparator_data_{C.TAG}.npz'
    Z = dict(np.load(path))
    meta = json.load(open(f'{C.RESULTS_DIR}/comparator_data_meta.json'))
    ranks = meta['das15_ranks']

    D.load_model()
    S.init_head_machinery()

    bases = {(case, k): torch.tensor(Z[f'das15_basis_{C.ck(case)}_k{k}'], device=D.device)
             for case in meta['cases'] for k in ranks}
    new = C.das15_write_alignment(bases)
    added = [k for k in new if k not in Z]
    Z.update(new)
    # Write to a sibling and rename: an interrupted or failed write must not be able to damage
    # the existing results, which cost a full run to produce.
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **Z)
    with np.load(tmp) as chk:
        assert len(chk.files) == len(Z), 'the rewritten file is short -- leaving the original'
    os.replace(tmp, path)
    print(f'{path}: {len(Z)} arrays ({len(added)} added)')
    for k in sorted(new):
        print(f'    {k:<40} {new[k].shape}  max {new[k].max():.4f}')
