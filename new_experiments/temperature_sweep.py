"""
Temperature sweep for RamenActive's `prob` active set (class weight = softmax(zero-shot logits / T)).

T=3 (picked on ImageNet-C) helps on ImageNet-C and DomainNet but loses -0.15 on CIFAR-100-C over seeds 0-4
(vqa_paper_directions.md). This sweep asks whether a different T recovers CIFAR-100-C. Large T flattens the class
weights towards Ramen's equal weights (T -> infinity is Ramen); small T concentrates them on the zero-shot top class,
whose queue holds the query itself.

Run:     CUDA_VISIBLE_DEVICES=0 python temperature_sweep.py run CIFAR100C 0 1 30     # seed 0, T = 1 and 30
Analyse: python temperature_sweep.py analyse CIFAR100C 0 1 3 5 10 30 100

Each T is cached as results/<dataset>_..._s<seed>_active_prob_T<T>.pkl, the same name active_seeds.py uses for T=3.
"""
import sys

import numpy as np
import pandas as pd

from active_seeds import cluster_bootstrap, experiment


def name(T):
    return f'active_prob_T{T:g}'


def run(dataset, seed, temps):
    exp = experiment(dataset, seed).load()
    exp.run('Ramen')  # cached for every dataset we sweep; needed for the comparison
    for T in temps:
        exp.run('RamenActive', name=name(T), cfg={'active_set': 'prob', 'temperature': T})
        print(f'T={T:g} done', flush=True)


def analyse(dataset, seed, temps):
    exp = experiment(dataset, seed)
    ramen = pd.read_pickle(exp._cache_path('Ramen'))
    acc_r = ramen.groupby('domain_idx').correct.mean().mean() * 100
    same_images = dataset != 'DomainNet'
    cluster = ramen.sample_idx.values if same_images else ramen.domain_idx.values * 10**7 + ramen.sample_idx.values
    rows = {}
    for T in temps:
        v = pd.read_pickle(exp._cache_path(name(T)))
        assert (v.sample_idx.values == ramen.sample_idx.values).all()
        acc = v.groupby('domain_idx').correct.mean().mean() * 100
        lo, hi = cluster_bootstrap(v.correct.values.astype(float) - ramen.correct.values, ramen.domain_idx.values,
                                   cluster)
        rows[T] = {'accuracy': acc, 'gain over Ramen': acc - acc_r, 'CI low': lo, 'CI high': hi,
                   'same pred as Ramen %': (v.pred.values == ramen.pred.values).mean() * 100}
    t = pd.DataFrame(rows).T
    t.index.name = 'T'
    print(f'{dataset}, seed {seed}: Ramen {acc_r:.2f}')
    print(t.round(2))

    # The project's noise floor (2 x range of Ramen over seeds 0-4, seed_variance.ipynb), if those runs are cached.
    try:
        accs = [pd.read_pickle(experiment(dataset, s)._cache_path('Ramen')).groupby('domain_idx').correct.mean().mean()
                * 100 for s in range(5)]
        print(f'noise floor (2 x Ramen range over seeds 0-4): {2 * (max(accs) - min(accs)):.2f}; '
              f'a single-seed gain must clear it, then be confirmed on seeds 0-4')
    except FileNotFoundError:
        print('noise floor unknown: Ramen seeds 0-4 are not all cached')
    t.to_csv(exp.results_dir + f'/{dataset}_s{seed}_temperature_sweep.csv')


if __name__ == '__main__':
    mode, dataset, seed, temps = sys.argv[1], sys.argv[2], int(sys.argv[3]), [float(T) for T in sys.argv[4:]]
    {'run': run, 'analyse': analyse}[mode](dataset, seed, temps)
