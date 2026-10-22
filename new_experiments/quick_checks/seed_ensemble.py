"""Headroom from variance reduction: average Ramen's probabilities over seeds (stream orders) 0-4."""
import os
import itertools
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results') + '/'
RUNS = {'CIFAR10C': 'CIFAR10C_clip_vitbase32_mixed_nall_bs100_s{}_{}',
        'CIFAR100C': 'CIFAR100C_clip_vitbase16_mixed_nall_bs100_s{}_{}'}

for ds, pat in RUNS.items():
    probs, ref = [], None
    for s in range(5):
        d = pd.read_pickle(R + pat.format(s, 'Ramen') + '.pkl')
        p = np.load(R + pat.format(s, 'Ramen') + '_probs.npy').astype(np.float32)
        key = d.domain_idx.values * 100000 + d.sample_idx.values
        order = np.argsort(key)
        if ref is None:
            ref = d.iloc[order].reset_index(drop=True)
        else:
            assert (np.sort(key) == ref.domain_idx.values * 100000 + ref.sample_idx.values).all()
        probs.append(p[order])
    y = ref.label.values
    za = pd.read_pickle(R + pat.format(0, 'NoAdapt') + '.pkl')
    zp = np.load(R + pat.format(0, 'NoAdapt') + '_probs.npy').astype(np.float32)
    zp = zp[np.argsort(za.domain_idx.values * 100000 + za.sample_idx.values)]

    acc = lambda p: (p.argmax(1) == y).mean() * 100
    singles = [acc(p) for p in probs]
    print(f'== {ds}: NoAdapt {acc(zp):.2f}; single-seed Ramen {np.mean(singles):.2f} (min {min(singles):.2f}, max {max(singles):.2f})')
    for n in (2, 3, 5):
        accs = [acc(np.mean([probs[i] for i in c], 0)) for c in itertools.combinations(range(5), n)]
        print(f'  average of {n} stream orders: {np.mean(accs):.2f}  (+{np.mean(accs) - np.mean(singles):.2f})')
    votes = np.stack([p.argmax(1) for p in probs], 1)
    agree_all = (votes == votes[:, :1]).all(1)
    print(f'  all 5 seeds agree on {agree_all.mean() * 100:.1f}% of samples; acc there {(votes[agree_all, 0] == y[agree_all]).mean() * 100:.1f}%, '
          f'acc elsewhere (seed 0) {(votes[~agree_all, 0] == y[~agree_all]).mean() * 100:.1f}%')
    oracle5 = (votes == y[:, None]).any(1).mean() * 100
    print(f'  any of 5 seeds right (oracle): {oracle5:.2f}')
    # control: ensemble of one Ramen run with zero-shot (a cheap ensemble that needs no extra stream passes)
    for w in (0.25, 0.5):
        print(f'  control: (1-{w})*Ramen_s0 + {w}*zero-shot probs: {acc((1 - w) * probs[0] + w * zp):.2f}')
