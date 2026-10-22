"""Does neighbour smoothing reduce Ramen's dependence on stream order (cross-seed churn)?"""
import os
import sys
import numpy as np
import pandas as pd
import torch

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results') + '/'
dev = 'cuda'
for ds, pat in {'CIFAR10C': 'CIFAR10C_clip_vitbase32_mixed_nall_bs100_s{}_{}',
                'CIFAR100C': 'CIFAR100C_clip_vitbase16_mixed_nall_bs100_s{}_{}'}.items():
    z0 = pd.read_pickle(R + pat.format(0, 'NoAdapt') + '.pkl')
    Z0 = np.load(R + pat.format(0, 'NoAdapt') + '_feats.npy')
    key0 = pd.Series(np.arange(len(z0)), index=z0.domain_idx.values * 100000 + z0.sample_idx.values)
    raw, sm = [], []
    for s in range(5):
        df = pd.read_pickle(R + pat.format(s, 'Ramen') + '.pkl')
        P = torch.tensor(np.load(R + pat.format(s, 'Ramen') + '_probs.npy'), device=dev).float()
        Z = torch.tensor(Z0[key0.loc[df.domain_idx.values * 100000 + df.sample_idx.values].values], device=dev).float()
        N = len(df); Q = P.clone(); bstart = (np.arange(N) // 100) * 100
        for a in range(100, N, 1000):
            e = min(a + 1000, N)
            sim = Z[a:e] @ Z[:e].T
            bs = torch.tensor(bstart[a:e], device=dev)
            sim[torch.arange(e, device=dev)[None, :] >= bs[:, None]] = -float('inf')
            v, idx = sim.topk(10, dim=1)
            Q[a:e] = (torch.softmax(v / 0.01, 1)[..., None] * P[idx]).sum(1)
        S = 0.5 * P + 0.5 * Q
        o = np.argsort(df.domain_idx.values * 100000 + df.sample_idx.values)   # align samples across seeds
        raw.append(P.argmax(1).cpu().numpy()[o]); sm.append(S.argmax(1).cpu().numpy()[o])
        if s == 0:
            y = df.label.values[o]
    raw, sm = np.stack(raw, 1), np.stack(sm, 1)
    unstable = ~(raw == raw[:, :1]).all(1)
    pair = lambda M: np.mean([(M[:, i] != M[:, j]).mean() for i in range(5) for j in range(i + 1, 5)]) * 100
    print(f'== {ds}')
    print(f'  samples where all 5 stream orders agree: Ramen {100 - unstable.mean() * 100:.1f}% -> smoothed {(sm == sm[:, :1]).all(1).mean() * 100:.1f}%')
    print(f'  predictions that change between two seeds: Ramen {pair(raw):.2f}% -> smoothed {pair(sm):.2f}%')
    g_un = ((sm[unstable] == y[unstable, None]).mean() - (raw[unstable] == y[unstable, None]).mean()) * 100
    g_st = ((sm[~unstable] == y[~unstable, None]).mean() - (raw[~unstable] == y[~unstable, None]).mean()) * 100
    print(f'  accuracy gain from smoothing: on unstable samples {g_un:+.2f} pts, on stable samples {g_st:+.2f} pts; '
          f'share of total gain from unstable samples {g_un * unstable.mean() / (g_un * unstable.mean() + g_st * (~unstable).mean()) * 100:.0f}%')
