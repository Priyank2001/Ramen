"""kNN output smoothing over seeds 0-4 (zero-shot-feature keys), alone and stacked with the prior correction."""
import os
import sys
import numpy as np
import pandas as pd
import torch

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results') + '/'
ds = sys.argv[1]
pat = {'CIFAR10C': 'CIFAR10C_clip_vitbase32_mixed_nall_bs100_s{}_{}',
       'CIFAR100C': 'CIFAR100C_clip_vitbase16_mixed_nall_bs100_s{}_{}'}[ds]
dev = 'cuda'
z0 = pd.read_pickle(R + pat.format(0, 'NoAdapt') + '.pkl')
Z0 = np.load(R + pat.format(0, 'NoAdapt') + '_feats.npy')
key0 = pd.Series(np.arange(len(z0)), index=z0.domain_idx.values * 100000 + z0.sample_idx.values)


def smooth(P, Z, batch_start, K=10, temp=0.01):
    N = len(P)
    Q = P.clone()
    for s in range(100, N, 1000):
        e = min(s + 1000, N)
        sim = Z[s:e] @ Z[:e].T
        bs = torch.tensor(batch_start[s:e], device=dev)
        sim[torch.arange(e, device=dev)[None, :] >= bs[:, None]] = -float('inf')
        v, idx = sim.topk(min(K, s), dim=1)
        w = torch.softmax(v / temp, dim=1)
        Q[s:e] = (w[..., None] * P[idx]).sum(1)
    return Q


def prior_correct(p, batch, lam=2.0, m=0.99):
    """Same as marginal_correction.corrected_probs (mixed mode, soft, include_current=True)."""
    p = p.astype(np.float64); k = p.shape[1]
    target = np.full(k, 1 / k); floor = 0.01 / k
    logp = np.log(np.clip(p, 1e-30, None)); out = np.empty_like(logp)
    starts = np.flatnonzero(np.r_[True, batch[1:] != batch[:-1]]); ends = np.r_[starts[1:], len(batch)]
    pi = target.copy()
    for s, e in zip(starts, ends):
        pi = m * pi + (1 - m) * p[s:e].mean(0)
        out[s:e] = logp[s:e] - lam * np.log(np.maximum(pi, floor) / target)
    out -= out.max(1, keepdims=True); q = np.exp(out)
    return q / q.sum(1, keepdims=True)


rows = []
for s in range(5):
    df = pd.read_pickle(R + pat.format(s, 'Ramen') + '.pkl')
    assert (df.position.values == np.arange(len(df))).all()
    y = df.label.values
    P = torch.tensor(np.load(R + pat.format(s, 'Ramen') + '_probs.npy'), device=dev).float()
    Z = torch.tensor(Z0[key0.loc[df.domain_idx.values * 100000 + df.sample_idx.values].values], device=dev).float()
    bstart = (np.arange(len(df)) // 100) * 100
    acc = lambda p: ((p.argmax(1).cpu().numpy() if torch.is_tensor(p) else p.argmax(1)) == y).mean() * 100
    base = acc(P)
    Q = smooth(P, Z, bstart)
    S = (0.5 * P + 0.5 * Q).cpu().numpy()
    pc = prior_correct(P.cpu().numpy(), df.batch.values)
    both = prior_correct(S, df.batch.values)
    rows.append({'seed': s, 'Ramen': base, 'smooth': acc(S) - base, 'prior corr.': acc(pc) - base,
                 'smooth then prior corr.': acc(both) - base})
t = pd.DataFrame(rows).set_index('seed')
print(f'== {ds}: gain over Ramen (pts), smoothing = 0.5*Ramen + 0.5*neighbours (K=10, zero-shot keys)')
print(t.round(2).to_string())
print('mean ± std:', ', '.join(f'{c} {t[c].mean():+.2f} ± {t[c].std():.2f}' for c in t.columns[1:]))
