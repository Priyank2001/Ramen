"""
Held-out runs on ImageNetVariants (ImageNet-R + -A + -Sketch, 1,000 classes), see PREREGISTRATION.md.

    CUDA_VISIBLE_DEVICES=0 python run_heldout.py 0 Ramen          # one (seed, run); runs: NoAdapt | Ramen | active_prob_T3
    CUDA_VISIBLE_DEVICES=0 python run_heldout.py --smoke          # 200 images per domain: loader + label check
Results are cached in results/ like any exp.run(), with full probabilities (NoAdapt also with features).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ramen_analysis import make_experiment  # noqa: E402

DATASET = 'ImageNetVariants'
NUM_WORKERS = 16
RUNS = {'NoAdapt': ('NoAdapt', None),
        'Ramen': ('Ramen', None),
        'active_prob_T3': ('RamenActive', {'active_set': 'prob', 'temperature': 3.0})}  # == active_seeds.py


def experiment(seed, n_per_domain=None):
    return make_experiment(DATASET, tta_mode='mixed', batch_size=100, n_per_domain=n_per_domain, seed=seed,
                           num_workers=NUM_WORKERS).load()


def run(seed, name):
    exp = experiment(seed)
    algo, cfg = RUNS[name]
    df = exp.run(algo, name=name, cfg=cfg, capture_probs=True, capture_features=name == 'NoAdapt')
    acc = df.groupby('domain', observed=True).correct.mean() * 100
    print(f'{DATASET} seed {seed} {name}: domain-avg {acc.mean():.2f} | ' +
          ', '.join(f'{d} {a:.2f}' for d, a in acc.items()), flush=True)


def smoke():
    """Sizes, label ranges and zero-shot accuracy per domain on a 200-image subset (not cached under the real names)."""
    exp = experiment(0, n_per_domain=200)
    for env, ds in zip(exp.environments, exp.datasets.datasets):
        full = sorted(set(ds.targets))
        print(f'{env}: {len(ds)} images (subset), {len(full)} distinct labels in subset, label range {full[0]}-{full[-1]}')
    df = exp.run('NoAdapt', use_cache=False, name='NoAdapt_smoke')
    print((df.groupby('domain', observed=True).correct.mean() * 100).round(1).to_string())
    os.remove(exp._cache_path('NoAdapt_smoke'))


if __name__ == '__main__':
    if sys.argv[1] == '--smoke':
        smoke()
    else:
        run(int(sys.argv[1]), sys.argv[2])
