"""Causal kNN smoothing of Ramen's outputs: average Ramen's probabilities on similar PAST samples (seed 0)."""
import os
import sys
import numpy as np
import pandas as pd
import torch

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results') + '/'
ds = sys.argv[1]
pat = {'CIFAR10C': 'CIFAR10C_clip_vitbase32_mixed_nall_bs100_s0_{}',
       'CIFAR100C': 'CIFAR100C_clip_vitbase16_mixed_nall_bs100_s0_{}'}[ds]
dev = 'cuda'
df = pd.read_pickle(R + pat.format('Ramen') + '.pkl')
assert (df.position.values == np.arange(len(df))).all()
y = df.label.values
N = len(y)
P = torch.tensor(np.load(R + pat.format('Ramen') + '_probs.npy'), device=dev).float()
keys = {'adapted feats': np.load(R + pat.format('Ramen') + '_feats.npy'),
        'zero-shot feats': np.load(R + pat.format('NoAdapt') + '_feats.npy')}
acc = lambda p: (p.argmax(1).cpu().numpy() == y).mean() * 100
print(f'== {ds}: Ramen {acc(P):.2f}')
batch_start = (np.arange(N) // 100) * 100
for kname, Z in keys.items():
    Z = torch.tensor(Z, device=dev).float()
    for K in (10, 50):
        Q = torch.zeros_like(P)
        for s in range(100, N, 1000):          # first batch has no past
            e = min(s + 1000, N)
            sim = Z[s:e] @ Z[:e].T                                           # (n, e)
            bs = torch.tensor(batch_start[s:e], device=dev)
            sim[torch.arange(e, device=dev)[None, :] >= bs[:, None]] = -float('inf')   # only previous batches
            v, idx = sim.topk(min(K, s), dim=1)
            w = torch.softmax(v / 0.01, dim=1)                               # sharp similarity weighting
            Q[s:e] = (w[..., None] * P[idx]).sum(1)
        Q[:100] = P[:100]
        line = ', '.join(f'w={w}: {acc((1 - w) * P + w * Q):.2f}' for w in (0.25, 0.5, 0.75))
        print(f'  key={kname}, K={K}: neighbours alone {acc(Q):.2f} | Ramen + neighbours {line}')
