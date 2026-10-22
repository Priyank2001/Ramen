"""Is Ramen's retrieved support set a useful classifier by itself (seed 0, logged run)?"""
import os
import sys
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results') + '/'
ds = sys.argv[1]
pat = {'CIFAR10C': 'CIFAR10C_clip_vitbase32_mixed_nall_bs100_s0_{}',
       'CIFAR100C': 'CIFAR100C_clip_vitbase16_mixed_nall_bs100_s0_{}'}[ds]

log = pd.read_pickle(R + pat.format('RamenLog') + '.pkl')
ram = pd.read_pickle(R + pat.format('Ramen') + '.pkl')
zs = pd.read_pickle(R + pat.format('NoAdapt') + '.pkl')
for d in (ram, zs):
    assert (d.domain_idx.values == log.domain_idx.values).all() and (d.sample_idx.values == log.sample_idx.values).all()
y = log.label.values
N = len(y)
p_ram = np.load(R + pat.format('Ramen') + '_probs.npy').astype(np.float32)
p_zs = np.load(R + pat.format('NoAdapt') + '_probs.npy').astype(np.float32)
z = np.load(R + pat.format('NoAdapt') + '_feats.npy')  # zero-shot features = cache keys, unit norm, fp16
ids_all = np.load(R + pat.format('RamenLog') + '_support_ids.npy', mmap_mode='r')
alpha_all = np.load(R + pat.format('RamenLog') + '_support_alpha.npy', mmap_mode='r')
C = ids_all.shape[1]

vote, vote_noself, proto = (np.zeros((N, C), np.float32) for _ in range(3))
step = 200
for s in range(0, N, step):
    ids = np.asarray(ids_all[s:s + step]).astype(np.int64)
    a = np.asarray(alpha_all[s:s + step]).astype(np.float32)
    valid = ids >= 0
    a = a * valid
    rows = np.arange(s, s + len(ids))[:, None, None]
    vote[s:s + step] = a.sum(2)
    notself = ids != rows
    vote_noself[s:s + step] = (a * notself).sum(2)
    zj = z[np.where(valid, ids, 0)].astype(np.float32)                 # (n, C, k, D)
    mu = (zj * (a * notself)[..., None]).sum(2)                        # (n, C, D) weighted local class mean
    mu /= np.linalg.norm(mu, axis=-1, keepdims=True) + 1e-8
    proto[s:s + step] = np.einsum('ncd,nd->nc', mu, z[s:s + step].astype(np.float32))

acc = lambda p: (p.argmax(1) == y).mean() * 100
norm = lambda v: v / (v.sum(1, keepdims=True) + 1e-12)
print(f'== {ds}: zero-shot {acc(p_zs):.2f}, Ramen {acc(p_ram):.2f}')
print(f'  support vote (incl. self)    alone: {acc(vote):.2f}')
print(f'  support vote (excl. self)    alone: {acc(vote_noself):.2f}')
print(f'  local class-mean classifier  alone: {acc(proto):.2f}')
print(f'  ensemble with zero-shot (control): best over w in [0,1]: '
      f'{max(acc((1 - w) * p_ram + w * p_zs) for w in np.linspace(0, 1, 21)):.2f}')
for name, v in [('vote excl. self', norm(vote_noself))]:
    for w in (0.1, 0.2, 0.3, 0.5):
        print(f'  Ramen + {name}, (1-w)p + w*vote, w={w}: {acc((1 - w) * p_ram + w * v):.2f}')
for T in (50.0, 100.0):
    pp = np.exp(T * (proto - proto.max(1, keepdims=True))); pp = norm(pp)
    for w in (0.1, 0.2, 0.3, 0.5):
        print(f'  Ramen + local class-mean (temp {T:.0f}), w={w}: {acc((1 - w) * p_ram + w * pp):.2f}')
