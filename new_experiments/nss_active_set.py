"""
Follow-up to quality_active_set.md: regularise Ramen's active-set selection with the NSS *feature vectors* behind
BRISQUE and NIQE (36 numbers each) instead of their scalar scores, which merge distortion types.

extract   python nss_active_set.py extract ImageNetC5K
          NSS vectors (BRISQUE's 36 features | NIQE's 36 block-mean features) and both scores for every image,
          cached as results/{dataset}_nss_features.npz, indexed [domain, sample].
simulate  python nss_active_set.py simulate ImageNetC5K
          Offline, from the cached seed-0 zero-shot run (features, predictions, entropies): (1) how well each
          descriptor identifies the corruption (10-nearest-neighbour domain purity); (2) Ramen's selection replayed
          exactly (per-class fp16 priority queues) on sampled batches, with each queue's entry picked by
          CLIP distance + lambda * R, reporting the active set's same-corruption share and label purity. Checked
          against the purity logged by the real lambda = 0 and scalar-score runs.
run       CUDA_VISIBLE_DEVICES=0 python nss_active_set.py run ImageNetC5K 0 [variant ...]
analyse   python nss_active_set.py analyse ImageNetC5K 0 [more seeds]
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from active_seeds import NUM_WORKERS, cluster_bootstrap
from quality_active_set import _acc, _load, support_domain_purity
from ramen_analysis import make_experiment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from models.iqa import ImageQuality  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_SETS = {'brisque': slice(0, 36), 'niqe': slice(36, 72), 'both': slice(0, 72)}


def experiment(dataset, seed):
    return make_experiment(dataset, tta_mode='mixed', batch_size=100, n_per_domain=None, seed=seed,
                           num_workers=NUM_WORKERS[dataset])


def _features_path(exp):
    return os.path.join(exp.results_dir, f'{exp.dataset_name}_nss_features.npz')


# ---------------------------------------------------------------------------- extract
def extract(dataset):
    exp = experiment(dataset, 0).load()
    iqa = ImageQuality(DEVICE)
    n = max(len(ds) for ds in exp.datasets.datasets)
    nss = np.full((len(exp.environments), n, 72), np.nan, dtype=np.float32)
    scores = np.full((len(exp.environments), n, 2), np.nan, dtype=np.float32)
    for d, ds in enumerate(exp.datasets.datasets):
        loader = DataLoader(ds, batch_size=250, shuffle=False, num_workers=exp.num_workers, pin_memory=True)
        i = 0
        for x, _ in tqdm(loader, desc=exp.environments[d]):
            s, f = iqa.with_features(x.to(DEVICE))
            scores[d, i:i + len(x)], nss[d, i:i + len(x)] = s.cpu().numpy(), f.cpu().numpy()
            i += len(x)
    np.savez(_features_path(exp), nss=nss, scores=scores, environments=np.array(exp.environments))
    print(f'saved {_features_path(exp)}; NaN rows: {np.isnan(nss).any(axis=2).sum()}')

    # the scores must match those RamenQuality computed inside the stream (seed-0 lambda = 0 run)
    ramen, st = _load(exp, 'Ramen')[0], _load(exp, 'quality_l0')[1]
    if st is not None:
        here = scores[ramen.domain_idx.values, ramen.sample_idx.values]
        there = np.stack([np.concatenate(st['niqe']), np.concatenate(st['brisque'])], axis=1)
        diff = np.abs(here - there)
        print(f'vs in-stream scores: max |diff| NIQE {np.nanmax(diff[:, 0]):.2e}, BRISQUE {np.nanmax(diff[:, 1]):.2e}')


# ---------------------------------------------------------------------------- simulate
def stream_arrays(dataset, seed=0):
    """Per stream position: domain, label, zero-shot prediction and entropy, CLIP feature, NSS vector, scores."""
    exp = experiment(dataset, seed)
    z = pd.read_pickle(exp._cache_path('NoAdapt'))
    f = np.load(_features_path(exp))
    s = {'domain': z.domain_idx.values, 'label': z.label.values, 'zs_pred': z.pred.values,
         'entropy': z.entropy.values.astype(np.float32),
         'feats': np.load(exp._cache_path('NoAdapt')[:-len('.pkl')] + '_feats.npy').astype(np.float32),
         'nss': f['nss'][z.domain_idx.values, z.sample_idx.values],
         'scores': f['scores'][z.domain_idx.values, z.sample_idx.values]}
    return exp, s


def normalise(x, how):
    """Whole-stream normalisation of descriptors (n, k): z-score, per-dimension rank, or PCA whitening."""
    x = np.where(np.isnan(x), np.nanmedian(x, axis=0), x).astype(np.float64)
    if how == 'z':
        return ((x - x.mean(0)) / x.std(0).clip(1e-12)).astype(np.float32)
    if how == 'rank':
        return (pd.DataFrame(x).rank(pct=True).values).astype(np.float32)
    if how == 'white':
        xc = x - x.mean(0)
        val, vec = np.linalg.eigh(np.cov(xc, rowvar=False))
        return (xc @ vec / np.sqrt(val.clip(val.max() * 1e-6))).astype(np.float32)
    raise ValueError(how)


def knn_domain_purity(x, domain, k=10, n=20000, metric='euclidean'):
    rng = np.random.default_rng(0)
    pick = rng.choice(len(x), min(n, len(x)), replace=False)
    t = torch.tensor(x[pick], device=DEVICE)
    dom = torch.tensor(domain[pick], device=DEVICE)
    hits = 0
    for i in range(0, len(t), 2000):
        d = -(t[i:i + 2000] @ t.T) if metric == 'cosine' else torch.cdist(t[i:i + 2000], t)
        d[torch.arange(len(d)), torch.arange(i, i + len(d))] = float('inf')
        idx = d.topk(k, dim=1, largest=False).indices
        hits += (dom[idx] == dom[i:i + 2000, None]).sum().item()
    return hits / (len(t) * k) * 100


def replay_queues(zs_pred, n_classes, capacity, batch, snapshot_batches):
    """
    Ramen's memory replayed exactly: per class a PriorityCache with fp16 priorities = stream counter (saturating),
    append while there is room, else replace the lowest priority if the new one is strictly higher.
    Returns {batch t: (n_classes, capacity) stream positions held after batch t's writes, -1 = empty}.
    """
    ids = np.full((n_classes, capacity), -1, dtype=np.int64)
    pri = np.full((n_classes, capacity), -np.inf, dtype=np.float16)
    size = np.zeros(n_classes, dtype=np.int64)
    snaps, want = {}, set(snapshot_batches)
    for t in range(int(np.ceil(len(zs_pred) / batch))):
        rows = np.arange(t * batch, min((t + 1) * batch, len(zs_pred)))
        with np.errstate(over='ignore'):  # positions > 65,504 become inf in fp16, as in Ramen
            fp16 = rows.astype(np.float16)
        for i, p in zip(rows, fp16):
            c = zs_pred[i]
            if size[c] < capacity:
                slot = size[c]
                size[c] += 1
            else:
                slot = np.argmin(pri[c])
                if not p > pri[c, slot]:
                    continue
            ids[c, slot], pri[c, slot] = i, p
        if t in want:
            snaps[t] = ids.copy()
    return snaps


def simulate_selection(s, snaps, descriptors, lambdas, batch=100, n_top=8):
    """
    For the queries of each snapshot batch: per class queue, the entry minimising d + lambda * R (topk = 1, beta = 0,
    as ImageNetC5K's Ramen config), R = RMS difference of the descriptors. Ramen's weights exp(-H_j).
    Returns one row per (descriptor, lambda): top-8 same-domain share (as logged by RamenQuality), same-domain share
    over the whole active set, label purity, and the mean CLIP distance of the selected entries.
    """
    feats = torch.tensor(s['feats'], device=DEVICE)
    H = torch.tensor(s['entropy'], device=DEVICE)
    dom = torch.tensor(s['domain'], device=DEVICE)
    lab = torch.tensor(s['label'], device=DEVICE)
    desc = {k: torch.tensor(v, device=DEVICE) for k, v in descriptors.items()}
    acc = {}
    for t, cand in tqdm(snaps.items(), desc='replay', leave=False):
        q = torch.arange(t * batch, min((t + 1) * batch, len(feats)), device=DEVICE)
        cand = torch.tensor(cand, device=DEVICE)
        valid = cand >= 0
        cflat = cand.clamp_min(0).view(-1)
        d = torch.sqrt((2 - 2 * feats[q] @ feats[cflat].T).clamp_min(0))  # (Q, C * cap)
        for name, x in desc.items():
            R = torch.cdist(x[q], x[cflat]) / np.sqrt(x.shape[1])
            for lam in lambdas[name]:
                sc = (d + lam * R).view(len(q), *cand.shape).masked_fill(~valid, float('inf'))
                best = sc.argmin(dim=2)  # (Q, C)
                has = valid.any(dim=1)[None, :].expand(len(q), -1)
                sel = cand[torch.arange(cand.shape[0], device=DEVICE)[None, :], best]  # (Q, C) stream positions
                w = torch.exp(-H[sel.clamp_min(0)]) * has
                same_dom = (dom[sel.clamp_min(0)] == dom[q, None]).float()
                same_lab = (lab[sel.clamp_min(0)] == lab[q, None]).float()
                wn = w / w.sum(dim=1, keepdim=True)
                top_w, top = wn.topk(n_top, dim=1)
                dsel = d.view(len(q), *cand.shape).gather(2, best[..., None]).squeeze(2)
                a = acc.setdefault((name, lam), np.zeros(6))
                a += [(top_w * same_dom.gather(1, top)).sum().item(), top_w.sum().item(),
                      (wn * same_dom).sum().item(), (wn * same_lab).sum().item(),
                      ((dsel * has).sum() / has.sum()).item() * len(q), len(q)]
    rows = [{'descriptor': name, 'lambda': lam, 'same-corruption top8 %': a[0] / a[1] * 100,
             'same-corruption all %': a[2] / a[5] * 100, 'label purity %': a[3] / a[5] * 100,
             'mean CLIP dist': a[4] / a[5]} for (name, lam), a in acc.items()]
    return pd.DataFrame(rows)


def simulate(dataset, n_snapshots=60):
    exp, s = stream_arrays(dataset)
    cfg = exp._load_cfg('Ramen')
    assert cfg['topk'] == 1 and cfg['beta'] == 0, 'the replay implements topk = 1, beta = 0 (ImageNetC5K)'
    n_classes = int(max(s['zs_pred'].max(), s['label'].max())) + 1  # queues of never-predicted classes stay empty
    nss, scores = s['nss'], s['scores']

    print(f'== {dataset}: how well each descriptor identifies the corruption '
          f'(10-NN same-domain %, chance {100 / (s["domain"].max() + 1):.1f}%)')
    rank2 = normalise(scores, 'rank')
    spaces = {'CLIP image features': (s['feats'], 'cosine'),
              'scores: NIQE (rank)': (rank2[:, :1], 'euclidean'),
              'scores: BRISQUE (rank)': (rank2[:, 1:], 'euclidean'),
              'scores: both (ranks)': (rank2, 'euclidean')}
    for fs, sl in FEATURE_SETS.items():
        for how in ('z', 'rank', 'white'):
            spaces[f'NSS {fs} {sl.stop - sl.start}-d ({how})'] = (normalise(nss[:, sl], how), 'euclidean')
    purity = {name: knn_domain_purity(x, s['domain'], metric=m) for name, (x, m) in spaces.items()}
    print(pd.Series(purity, name='10-NN same-domain %').round(1).to_string())

    n_batches = int(np.ceil(len(nss) / 100))
    snap_batches = np.linspace(n_batches // n_snapshots, n_batches - 1, n_snapshots).astype(int)
    snaps = replay_queues(s['zs_pred'], n_classes, cfg['max_capacity'], 100, snap_batches)

    descriptors = {'scores both (ranks)': rank2}
    lambdas = {'scores both (ranks)': [0.0, 0.1, 0.3, 1.0, 3.0]}
    for fs, sl in FEATURE_SETS.items():
        for how in ('z', 'white'):
            descriptors[f'NSS {fs} ({how})'] = normalise(nss[:, sl], how)
            lambdas[f'NSS {fs} ({how})'] = [0.01, 0.03, 0.1, 0.3, 1.0]
    t = simulate_selection(s, snaps, descriptors, lambdas)
    pd.set_option('display.width', 200)
    print(f'\n== Replayed selection on {n_snapshots} batches ({n_snapshots * 100} queries), seed 0')
    print('   real runs (logged): lambda = 0 top8 43.55%, label 0.48%; scalar both match '
          'lambda 0.1 / 0.3 / 1 / 3 -> top8 47.0 / 46.6 / 39.8 / 28.6%, label 0.43 / 0.37 / 0.28 / 0.22%')
    print(t.round(3).to_string(index=False))
    t.to_csv(os.path.join(exp.results_dir, f'{dataset}_nss_selection_replay.csv'), index=False)


# ---------------------------------------------------------------------------- real runs
VARIANTS = {'nss_l0': {'nss_lambda': 0.0}}


def add_variants(feature_sets, norm, lambdas):
    for fs in feature_sets:
        for lam in lambdas:
            VARIANTS[f'nss_{fs}_{norm}_l{lam:g}'] = {'nss_features': fs, 'nss_norm': norm, 'nss_lambda': lam}


add_variants(FEATURE_SETS, 'white', (0.03, 0.1, 0.3))  # the replay's best normalisation and lambda range
add_variants(FEATURE_SETS, 'z', (0.1,))


def run(dataset, seed, names):
    exp = experiment(dataset, seed).load()
    exp.run('Ramen')
    for name in names or VARIANTS:
        exp.run('RamenNSS', name=name, cfg=VARIANTS[name])
        print(f'{name} done', flush=True)


def analyse(dataset, seeds):
    tables, ramen_accs = [], []
    for seed in seeds:
        exp = experiment(dataset, seed)
        ramen, _ = _load(exp, 'Ramen')
        ramen_accs.append(_acc(ramen))
        cluster = ramen.sample_idx.values if dataset != 'DomainNet' else \
            ramen.domain_idx.values * 10**7 + ramen.sample_idx.values
        rows = {}
        for name, cfg in VARIANTS.items():
            if not os.path.exists(exp._cache_path(name)):
                continue
            v, st = _load(exp, name)
            assert (v.sample_idx.values == ramen.sample_idx.values).all(), f'{name}: stream order differs'
            lo, hi = cluster_bootstrap(v.correct.values.astype(float) - ramen.correct.values,
                                       ramen.domain_idx.values, cluster)
            rows[name] = {'acc': _acc(v), 'gain': _acc(v) - _acc(ramen), 'CI low': lo, 'CI high': hi,
                          'same pred %': (v.pred.values == ramen.pred.values).mean() * 100,
                          'label purity %': np.nanmean(np.concatenate(st['support_purity'])) * 100,
                          'same-corruption top8 %': support_domain_purity(st, ramen.domain_idx.values,
                                                                          ramen.domain_idx.values)}
        t = pd.DataFrame(rows).T.astype(float)
        print(f'\n=== {dataset} seed {seed}: Ramen {_acc(ramen):.2f} ===')
        if t.empty:
            print('no RamenNSS runs yet')
            continue
        print(t.round(2).to_string())
        t.to_csv(os.path.join(exp.results_dir, f'{dataset}_nss_active_set_s{seed}.csv'))
        tables.append(t)
    if len(tables) > 1:
        all_t = pd.concat(tables)
        floor = 2 * (max(ramen_accs) - min(ramen_accs))  # the project's rule (seed_variance.ipynb)
        g = all_t.groupby(level=0).gain.agg(['mean', 'std', 'min', 'max', 'count'])
        g['positive seeds'] = all_t.groupby(level=0).gain.apply(lambda x: (x > 0).sum())
        g['paired t'] = g['mean'] / (g['std'] / np.sqrt(g['count']))
        g['above floor'] = g['mean'] > floor
        print(f'\nGain over Ramen across seeds {seeds} (noise floor = 2 x Ramen range = {floor:.2f}):')
        print(g[g['count'] > 1].sort_values('mean', ascending=False).round(2).to_string())


if __name__ == '__main__':
    mode, dataset = sys.argv[1], sys.argv[2]
    if mode == 'extract':
        extract(dataset)
    elif mode == 'simulate':
        simulate(dataset)
    elif mode == 'run':
        run(dataset, int(sys.argv[3]), sys.argv[4:])
    elif mode == 'analyse':
        analyse(dataset, [int(s) for s in sys.argv[3:]])
    else:
        raise SystemExit(__doc__)
