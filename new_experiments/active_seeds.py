"""
Seeds for the zero-shot-weighted active set (vqa_paper_directions.md, Direction 1).

Run:     CUDA_VISIBLE_DEVICES=0 python active_seeds.py run ImageNetC5K 1 4    # Ramen + active_prob_T3, seeds 1 and 4
Analyse: python active_seeds.py analyse ImageNetC5K 0 1 2 3 4                  # per-seed gains, bootstrap, t

Ramen and the variant must see the identical stream order, which depends on num_workers (see the tqdm stream-order
gotcha in the findings). NUM_WORKERS matches the cached Ramen runs of each dataset, so cached Ramen results are
reused; analyse() asserts the orders match. Results are cached in results/ like any exp.run().
"""
import sys

import numpy as np
import pandas as pd

from ramen_analysis import make_experiment

VARIANT = ('active_prob_T3', 'RamenActive', {'active_set': 'prob', 'temperature': 3.0})


NUM_WORKERS = {'ImageNetC5K': 16, 'DomainNet': 16, 'CIFAR100C': 4, 'CIFAR10C': 4}


def experiment(dataset, seed):
    return make_experiment(dataset, tta_mode='mixed', batch_size=100, n_per_domain=None, seed=seed,
                           num_workers=NUM_WORKERS[dataset])


def run(dataset, seeds):
    for seed in seeds:
        exp = experiment(dataset, seed).load()
        exp.run('Ramen')
        name, algo, cfg = VARIANT
        exp.run(algo, name=name, cfg=cfg)
        print(f'seed {seed} done', flush=True)


def cluster_bootstrap(diff, domain, cluster, n=2000, rng=np.random.default_rng(0)):
    """
    95% CI, in points, of the reported statistic: the mean over domains of each domain's mean paired difference.
    Resamples clusters of samples that share an image: the base image on corruption datasets (one image appears under
    every corruption), each single image on DomainNet. Poisson(1) weights per cluster (the Poisson bootstrap).
    """
    cl = pd.factorize(cluster)[0]
    n_dom = domain.max() + 1
    S = np.zeros((cl.max() + 1, n_dom))  # per cluster x domain: sum of differences and number of samples
    C = np.zeros_like(S)
    np.add.at(S, (cl, domain), diff)
    np.add.at(C, (cl, domain), 1)
    S, C = S.astype(np.float32), C.astype(np.float32)
    boot = []
    for _ in range(0, n, 100):  # 100 replicates at a time keeps the weight matrix small on DomainNet
        W = rng.poisson(1.0, size=(100, len(S))).astype(np.float32)
        boot.append(((W @ S) / (W @ C)).mean(axis=1) * 100)
    boot = np.concatenate(boot)
    return np.percentile(boot, [2.5, 97.5])


def analyse(dataset, seeds):
    name = VARIANT[0]
    rows = {}
    for seed in seeds:
        exp = experiment(dataset, seed)
        r, v = (pd.read_pickle(exp._cache_path(a)) for a in ('Ramen', name))
        assert (r.sample_idx.values == v.sample_idx.values).all() and (r.domain_idx.values == v.domain_idx.values).all()
        acc_r = r.groupby('domain_idx').correct.mean().mean() * 100
        acc_v = v.groupby('domain_idx').correct.mean().mean() * 100
        same_images = dataset != 'DomainNet'  # corruption datasets: every domain holds the same images
        cluster = r.sample_idx.values if same_images else r.domain_idx.values * 10**7 + r.sample_idx.values
        lo, hi = cluster_bootstrap(v.correct.values.astype(float) - r.correct.values, r.domain_idx.values, cluster)
        rows[seed] = {'Ramen': acc_r, name: acc_v, 'gain': acc_v - acc_r, 'CI low': lo, 'CI high': hi,
                      'same pred %': (r.pred.values == v.pred.values).mean() * 100}
    t = pd.DataFrame(rows).T
    t.index.name = 'seed'
    print(t.round(2))
    print(f'\ngain over seeds: {t.gain.mean():+.2f} ± {t.gain.std():.2f} (std), min {t.gain.min():+.2f}, '
          f'positive on {(t.gain > 0).sum()}/{len(t)} seeds')
    print(f'Ramen seed-to-seed: mean {t.Ramen.mean():.2f}, std {t.Ramen.std():.2f}, '
          f'range {t.Ramen.max() - t.Ramen.min():.2f}')
    print(f'paired t over seeds: {t.gain.mean() / (t.gain.std() / np.sqrt(len(t))):.1f}')
    floor = 2 * (t.Ramen.max() - t.Ramen.min())  # the project's rule, see seed_variance.ipynb
    print(f'noise floor (2 x Ramen range): {floor:.2f} | mean gain above it: {t.gain.mean() > floor} | '
          f'seeds above it: {(t.gain > floor).sum()}/{len(t)}')
    t.to_csv(exp.results_dir + f'/{dataset}_{name}_seeds.csv')


if __name__ == '__main__':
    mode, dataset, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
    {'run': run, 'analyse': analyse}[mode](dataset, seeds)
