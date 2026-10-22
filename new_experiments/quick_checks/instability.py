"""Is instability across stream orders a 'fall back to zero-shot' signal?"""
import os
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results') + '/'
RUNS = {'CIFAR10C': 'CIFAR10C_clip_vitbase32_mixed_nall_bs100_s{}_{}',
        'CIFAR100C': 'CIFAR100C_clip_vitbase16_mixed_nall_bs100_s{}_{}'}
for ds, pat in RUNS.items():
    preds, probs = [], []
    for s in range(5):
        d = pd.read_pickle(R + pat.format(s, 'Ramen') + '.pkl')
        o = np.argsort(d.domain_idx.values * 100000 + d.sample_idx.values)
        preds.append(d.pred.values[o]); probs.append(np.load(R + pat.format(s, 'Ramen') + '_probs.npy')[o].astype(np.float32))
        if s == 0:
            y = d.label.values[o]; dom = d.domain.values[o]
    za = pd.read_pickle(R + pat.format(0, 'NoAdapt') + '.pkl')
    zpred = za.pred.values[np.argsort(za.domain_idx.values * 100000 + za.sample_idx.values)]
    P = np.stack(preds, 1)
    unstable = ~(P == P[:, :1]).all(1)
    print(f'== {ds}: unstable (seeds disagree) {unstable.mean() * 100:.1f}% of samples')
    print(f'  on unstable samples: zero-shot acc {(zpred[unstable] == y[unstable]).mean() * 100:.1f}, '
          f'Ramen seed0 {(P[unstable, 0] == y[unstable]).mean() * 100:.1f}, '
          f'5-seed avg {(np.mean(probs, 0)[unstable].argmax(1) == y[unstable]).mean() * 100:.1f}')
    st = ~unstable
    print(f'  on stable samples:   zero-shot acc {(zpred[st] == y[st]).mean() * 100:.1f}, Ramen {(P[st, 0] == y[st]).mean() * 100:.1f}')
    # share of each seed's mistakes that sit on unstable samples
    print(f'  share of Ramen seed0 errors that are on unstable samples: {(unstable & (P[:, 0] != y)).sum() / (P[:, 0] != y).sum() * 100:.1f}%')
    # per-domain instability rate
    t = pd.Series(unstable).groupby(dom).mean().sort_values() * 100
    print('  instability by domain (%):', ', '.join(f'{k} {v:.0f}' for k, v in t.items()))
