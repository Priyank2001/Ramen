"""Sanity checks before any table: data reproduces cached predictions; CSLS with kappa 1 / beta 0 == zero-shot."""
import sys

import numpy as np

from data import DATASETS, RAMEN_FEATS, StreamData
from hubness import Csls, ZeroShot, run

for ds in DATASETS:
    for level in ('zero-shot', 'ramen'):
        if level == 'ramen' and ds not in RAMEN_FEATS:
            continue
        sd = StreamData(ds, 0, level)
        agree, n, gap = sd.sanity()
        bat = sd.batches()
        a = run(ZeroShot(sd.W), sd.Z, bat)
        b = run(Csls(sd.W, 1, 0.0), sd.Z, bat)
        print(f'{ds:17s} {level:9s} n={len(sd.Z):,} classes={sd.classes} argmax z.W == cached pred {agree:.3f}% '
              f'({n:,} differ, largest top-2 gap {gap:.3f}); zero-shot == CSLS(b=0) {np.mean(a == b) * 100:.2f}%; batch sizes '
              f'{sorted({e - s for s, e in bat})[:3]}', flush=True)
        assert gap < 0.05 and (a == b).all()
print('SANITY OK')
