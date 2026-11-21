"""
Stage 1b: the offline screen of PREREGISTRATION.md (no model; GPU only for distances).

    python screen.py        # needs results/ImageNetC5K_masked_niqe.npz (extract.py); -> tables/screen_*.csv, verdicts

S1/S2: 10 nearest neighbours of every image (75,000), in the whitened masked NIQE vector: share that are the same
       photo under another corruption (content leakage), share with the same corruption, share with the same label.
S3:    nss_active_set.replay_queues / simulate_selection (Ramen's exact queues, topk 1, beta 0): top-8 same-corruption
       share of the active set when each queue's entry is picked by CLIP distance + lambda * descriptor distance,
       180 snapshot batches in 3 interleaved groups (the groups give the replay noise).
Reuses the shared nss_active_set.py functions read-only.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from nss_active_set import normalise, replay_queues, simulate_selection, stream_arrays, DEVICE  # noqa: E402

DATASET = 'ImageNetC5K'
LAMBDAS = [0.03, 0.1, 0.3]
CONTROL = {'centre96': 'randbox96', 'centre136': 'randbox136', 'ring96': 'randring96', 'ring136': 'randring136'}
TABLES = os.path.join(HERE, 'tables')


def knn_profile(x, domain, photo, label, k=10, chunk=1500):
    """Share of the k nearest neighbours (all images, self excluded) that share corruption / photo / label."""
    t = torch.tensor(x, device=DEVICE)
    dom, pho, lab = (torch.tensor(a, device=DEVICE) for a in (domain, photo, label))
    hits = np.zeros(3)
    for i in range(0, len(t), chunk):
        d = torch.cdist(t[i:i + chunk], t)
        d[torch.arange(len(d)), torch.arange(i, i + len(d))] = float('inf')
        idx = d.topk(k, dim=1, largest=False).indices
        for j, a in enumerate((dom, pho, lab)):
            hits[j] += (a[idx] == a[i:i + chunk, None]).sum().item()
    return hits / (len(t) * k) * 100


def main():
    exp, s = stream_arrays(DATASET)
    z = pd.read_pickle(exp._cache_path('NoAdapt'))
    f = np.load(os.path.join(HERE, 'results', f'{DATASET}_masked_niqe.npz'))
    names = [k for k in f.files if k != 'environments']
    dom, smp = z.domain_idx.values, z.sample_idx.values
    desc = {k: normalise(f[k][dom, smp], 'white') for k in names}  # stream order, whole-stream whitening
    n_classes = int(max(s['zs_pred'].max(), s['label'].max())) + 1
    cfg = exp._load_cfg('Ramen')
    assert cfg['topk'] == 1 and cfg['beta'] == 0

    rows = []
    for k in names:
        c, p, l = knn_profile(desc[k], s['domain'], smp, s['label'])
        rows.append({'variant': k, '10-NN same corruption %': c, '10-NN same photo (leakage) %': p, '10-NN same label %': l,
                     'NaN vectors': int(np.isnan(f[k][dom, smp]).any(axis=1).sum())})
    knn = pd.DataFrame(rows).set_index('variant')
    print(f'== S1/S2: 10-NN in the whitened NIQE vector, {len(dom):,} images (chance same corruption 6.7%)')
    print(knn.round(2).to_string())
    knn.to_csv(os.path.join(TABLES, 'screen_knn.csv'))

    n_batches = int(np.ceil(len(dom) / 100))
    batches = np.linspace(n_batches // 180, n_batches - 1, 180).astype(int)
    snaps = replay_queues(s['zs_pred'], n_classes, cfg['max_capacity'], 100, batches)
    groups = [{int(t): snaps[int(t)] for t in batches[g::3]} for g in range(3)]
    rep = []
    for g, sn in enumerate(groups):
        r = simulate_selection(s, sn, desc, {k: [0.0] + LAMBDAS if k == 'full' else LAMBDAS for k in names})
        rep.append(r.assign(group=g))
        print(f'replay group {g + 1}/3 done', flush=True)
    rep = pd.concat(rep)
    col = 'same-corruption top8 %'
    rep.to_csv(os.path.join(TABLES, 'screen_replay_groups.csv'), index=False)
    piv = rep.pivot_table(index=['descriptor', 'lambda'], columns='group', values=col)
    summ = pd.DataFrame({'share %': piv.mean(axis=1), 'group range': piv.max(axis=1) - piv.min(axis=1)})
    summ['label purity %'] = rep.groupby(['descriptor', 'lambda'])['label purity %'].mean()
    print('\n== S3: replayed top-8 same-corruption share (mean of 3 groups; Ramen 43.6, NIQE vector λ 0.1 ≈ 49.4)')
    print(summ.round(2).to_string())
    summ.to_csv(os.path.join(TABLES, 'screen_replay.csv'))

    base = piv.loc[('full', 0.0)]
    print(f"\nlambda 0 (= Ramen's selection): {base.mean():.2f}%")
    verdict = []
    for x, ctrl in CONTROL.items():
        lam = max(LAMBDAS, key=lambda l: summ.loc[(x, l), 'share %'])
        vs_full, vs_ctrl = piv.loc[(x, lam)] - piv.loc[('full', lam)], piv.loc[(x, lam)] - piv.loc[(ctrl, lam)]
        noise = 2 * max(np.ptp(vs_full.values), np.ptp(vs_ctrl.values))
        leak_ok = knn.loc[x, '10-NN same photo (leakage) %'] < knn.loc['full', '10-NN same photo (leakage) %']
        passed = (vs_full.mean() > max(1.0, noise)) and (vs_ctrl.mean() > max(1.0, noise)) and leak_ok
        verdict.append({'variant': x, 'best lambda': lam, 'share %': summ.loc[(x, lam), 'share %'],
                        'vs full': vs_full.mean(), 'vs control': vs_ctrl.mean(), '2 x group range': noise,
                        'leakage lower than full': leak_ok, 'PASSES': passed})
    v = pd.DataFrame(verdict).set_index('variant')
    print('\n== screen rule (PREREGISTRATION.md): shares in points; passes needs both diffs > max(1.0, 2 x range) and lower leakage')
    print(v.round(2).to_string())
    v.to_csv(os.path.join(TABLES, 'screen_verdict.csv'))
    print('SCREEN DONE', flush=True)


if __name__ == '__main__':
    main()
