"""
Image-quality regularised active set (src/methods/RamenQuality.py; NIQE and BRISQUE from blindqa.pdf).

Each class queue's active-set entries are picked by  CLIP distance + lambda * R  instead of CLIP distance alone.
Experiments, one per metric choice and regulariser:
  metric   niqe     NIQE alone                 (opinion-unaware: distance to the pristine-patch MVG)
           brisque  BRISQUE alone              (opinion-aware: SVR on LIVE human scores)
           both     NIQE and BRISQUE together  (their stream ranks combined with equal weight)
  R        match    the entry's quality is close to the query's (a distortion-consistent active set)
           prior    the entry's image looks natural (low score)
  lambda   LAMBDAS; quality_l0 is lambda = 0, which must equal Ramen exactly.

Run:      CUDA_VISIBLE_DEVICES=0 python quality_active_set.py run ImageNetC5K 0 [variant ...]   (default: all)
Analyse:  python quality_active_set.py analyse ImageNetC5K 0 [more seeds]

Ramen and the variants must see the identical stream order, which depends on num_workers (tqdm stream-order
gotcha); NUM_WORKERS matches the cached Ramen runs, so they are reused and analyse() asserts the orders match.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from active_seeds import NUM_WORKERS, cluster_bootstrap
from ramen_analysis import make_experiment

LAMBDAS = (0.1, 0.3, 1.0, 3.0)
VARIANTS = {'quality_l0': {'quality_lambda': 0.0}}
for _metric in ('niqe', 'brisque', 'both'):
    for _reg in ('match', 'prior'):
        for _lam in LAMBDAS:
            VARIANTS[f'quality_{_metric}_{_reg}_l{_lam:g}'] = {
                'quality_metric': _metric, 'quality_reg': _reg, 'quality_lambda': _lam}
# on top of the best class weighting so far (RamenActive prob T=3, active_seeds.py)
VARIANTS['quality_both_match_l0.3_probT3'] = {'quality_metric': 'both', 'quality_reg': 'match',
                                              'quality_lambda': 0.3, 'active_set': 'prob', 'temperature': 3.0}


def experiment(dataset, seed):
    return make_experiment(dataset, tta_mode='mixed', batch_size=100, n_per_domain=None, seed=seed,
                           num_workers=NUM_WORKERS[dataset])


def run(dataset, seed, names):
    exp = experiment(dataset, seed).load()
    exp.run('Ramen')
    for name in names or VARIANTS:
        exp.run('RamenQuality', name=name, cfg=VARIANTS[name])
        print(f'{name} done', flush=True)


# ---------------------------------------------------------------------------- analysis
def _load(exp, name):
    df = pd.read_pickle(exp._cache_path(name))
    stats_path = exp._cache_path(name)[:-len('.pkl')] + '_stats.pkl'
    return df, (pd.read_pickle(stats_path) if os.path.exists(stats_path) else None)


def _acc(df):
    return df.groupby('domain_idx').correct.mean().mean() * 100


def support_domain_purity(stats, domain_of_position, query_domain):
    """Weight share of the logged top-n active-set entries that come from the query's domain (corruption)."""
    ids, w = np.concatenate(stats['support_ids']), np.concatenate(stats['support_weight'])
    dom = np.where(ids >= 0, domain_of_position[np.clip(ids, 0, None)], -1)
    return float(((dom == query_domain[:, None]) * w).sum() / w.sum() * 100)


def support_sources(exp, ramen, name, n_top=2):
    """For the queries of each domain: the domains its logged top-n active-set entries come from (weight share %)."""
    st = _load(exp, name)[1]
    ids, w = np.concatenate(st['support_ids']), np.concatenate(st['support_weight'])
    envs = list(ramen.domain.cat.categories)
    src = np.where(ids >= 0, ramen.domain_idx.values[np.clip(ids, 0, None)], -1)
    rows = {}
    for d, env in enumerate(envs):
        q = ramen.domain_idx.values == d
        share = np.bincount(src[q].ravel()[src[q].ravel() >= 0], weights=w[q].ravel()[src[q].ravel() >= 0],
                            minlength=len(envs))
        share = share / share.sum() * 100
        top = np.argsort(-share)[:n_top]
        rows[env] = {'own %': f'{share[d]:.1f}', **{f'#{i + 1} source': f'{envs[j]} {share[j]:.0f}%' for i, j in enumerate(top)}}
    return pd.DataFrame(rows).T


def variant_table(dataset, seed):
    exp = experiment(dataset, seed)
    ramen, _ = _load(exp, 'Ramen')
    domain_of_position = ramen.domain_idx.values  # the stream position of a row is its index
    same_images = dataset != 'DomainNet'  # corruption datasets: every domain holds the same images
    cluster = ramen.sample_idx.values if same_images else ramen.domain_idx.values * 10**7 + ramen.sample_idx.values
    rows = {}
    for name, cfg in VARIANTS.items():
        if not os.path.exists(exp._cache_path(name)):
            continue
        v, st = _load(exp, name)
        assert (v.sample_idx.values == ramen.sample_idx.values).all(), f'{name}: stream order differs from Ramen'
        lo, hi = cluster_bootstrap(v.correct.values.astype(float) - ramen.correct.values, ramen.domain_idx.values,
                                   cluster)
        gap = np.concatenate(st['support_gap']).mean(axis=0)
        rows[name] = {
            'metric': cfg.get('quality_metric', '-'), 'R': cfg.get('quality_reg', '-'),
            'lambda': cfg['quality_lambda'], 'acc': _acc(v), 'gain': _acc(v) - _acc(ramen), 'CI low': lo,
            'CI high': hi, 'same pred %': (v.pred.values == ramen.pred.values).mean() * 100,
            'label purity %': np.nanmean(np.concatenate(st['support_purity'])) * 100,
            'domain purity top8 %': support_domain_purity(st, domain_of_position, ramen.domain_idx.values),
            'NIQE rank gap': gap[0], 'BRISQUE rank gap': gap[1]}
    t = pd.DataFrame(rows).T
    t.index.name = f'seed {seed}'
    return exp, ramen, t


def per_domain_gains(exp, ramen, names):
    def by_domain(df):
        return df.groupby('domain', observed=True).correct.mean()  # categorical: in environment order
    return pd.DataFrame({name: (by_domain(_load(exp, name)[0]) - by_domain(ramen)) * 100 for name in names})


def quality_profile(exp, ramen, stats):
    """What NIQE / BRISQUE look like on the stream (from the lambda = 0 run, which logs both for every image)."""
    niqe, brisque = (np.concatenate(stats[m]).astype(float) for m in ('niqe', 'brisque'))
    noadapt_path = exp._cache_path('NoAdapt')
    zs = pd.read_pickle(noadapt_path) if os.path.exists(noadapt_path) else None
    d = pd.DataFrame({'domain': ramen.domain.values, 'NIQE': niqe, 'BRISQUE': brisque,
                      'Ramen acc': ramen.correct.values * 100.0})
    if zs is not None and (zs.sample_idx.values == ramen.sample_idx.values).all():
        d['zero-shot acc'] = zs.correct.values * 100.0
    t = d.groupby('domain', observed=True).agg(
        {'NIQE': 'median', 'BRISQUE': 'median', 'Ramen acc': 'mean',
         **({'zero-shot acc': 'mean'} if 'zero-shot acc' in d else {})})
    if 'zero-shot acc' in t:
        t['Ramen gain'] = t['Ramen acc'] - t['zero-shot acc']
    print(f'NaN scores: NIQE {np.isnan(niqe).sum()}, BRISQUE {np.isnan(brisque).sum()} of {len(niqe)}')
    print('\nPer-domain median score (higher = worse) and accuracy:')
    print(t.round(2))
    if 'Ramen gain' in t:
        for m in ('NIQE', 'BRISQUE'):
            print(f'  Spearman over domains, median {m} vs zero-shot acc {spearmanr(t[m], t["zero-shot acc"])[0]:+.2f}, '
                  f'vs Ramen gain {spearmanr(t[m], t["Ramen gain"])[0]:+.2f}')

    # How much domain (corruption) information the scores carry: 10 nearest images in stream-rank space
    rng = np.random.default_rng(0)
    pick = rng.choice(len(d), min(20000, len(d)), replace=False)
    ranks = np.stack([pd.Series(niqe).rank(pct=True).values, pd.Series(brisque).rank(pct=True).values], axis=1)
    ranks = np.nan_to_num(ranks[pick], nan=1.0)
    dom = ramen.domain_idx.values[pick]
    print(f'\nDomain purity of the 10 nearest images in quality space (chance {100 / (dom.max() + 1):.1f}%):')
    for label, cols in [('NIQE', [0]), ('BRISQUE', [1]), ('both', [0, 1])]:
        nn = NearestNeighbors(n_neighbors=11).fit(ranks[:, cols])
        idx = nn.kneighbors(ranks[:, cols], return_distance=False)[:, 1:]
        print(f'  {label:8s} {(dom[idx] == dom[:, None]).mean() * 100:.1f}%')


def analyse(dataset, seeds):
    tables, ramen_accs = [], []
    for seed in seeds:
        exp, ramen, t = variant_table(dataset, seed)
        ramen_accs.append(_acc(ramen))
        print(f'\n=== {dataset} seed {seed}: Ramen {_acc(ramen):.2f} ===')
        if t.empty:
            print('no RamenQuality runs yet')
            continue
        print(t.drop(columns=['metric', 'R', 'lambda']).astype(float).round(2).to_string())
        if 'quality_l0' in t.index:
            print(f"lambda = 0 vs Ramen: {t.loc['quality_l0', 'same pred %']:.2f}% same predictions")
        t.to_csv(os.path.join(exp.results_dir, f'{dataset}_quality_active_set_s{seed}.csv'))
        tables.append(t.assign(seed=seed))
        if seed == seeds[0]:
            best = (t[t['lambda'] > 0].astype({'gain': float}).groupby('metric').gain.idxmax())
            print('\nPer-domain gain over Ramen of the best variant per metric:')
            print(per_domain_gains(exp, ramen, list(best.values)).round(2).to_string())
            if 'quality_l0' in t.index:
                quality_profile(exp, ramen, _load(exp, 'quality_l0')[1])
                strongest = t[(t.R == 'match') & (t.metric == 'both')].astype({'lambda': float}).sort_values('lambda')
                for name in ['quality_l0'] + list(strongest.index[-1:]):
                    print(f'\nWhere the top-{len(_load(exp, name)[1]["support_ids"][0][0])} active-set entries of '
                          f'each domain come from, {name}:')
                    print(support_sources(exp, ramen, name).to_string())
    if len(tables) > 1:
        all_t = pd.concat(tables)
        all_t['gain'] = all_t.gain.astype(float)
        floor = 2 * (max(ramen_accs) - min(ramen_accs))  # the project's rule (seed_variance.ipynb), all seeds given
        g = all_t.groupby(level=0).gain.agg(['mean', 'std', 'min', 'max', 'count'])
        g['positive seeds'] = all_t.groupby(level=0).gain.apply(lambda x: (x > 0).sum())
        g['paired t'] = g['mean'] / (g['std'] / np.sqrt(g['count']))
        g['above floor'] = g['mean'] > floor
        print(f'\nGain over Ramen across seeds {seeds} (noise floor = 2 x Ramen range = {floor:.2f}):')
        print(g[g['count'] > 1].sort_values('mean', ascending=False).round(2).to_string())
        all_t.to_csv(os.path.join(exp.results_dir, f'{dataset}_quality_active_set_seeds.csv'))


if __name__ == '__main__':
    mode, dataset = sys.argv[1], sys.argv[2]
    if mode == 'run':
        run(dataset, int(sys.argv[3]), sys.argv[4:])
    elif mode == 'analyse':
        analyse(dataset, [int(s) for s in sys.argv[3:]])
    else:
        raise SystemExit(__doc__)
