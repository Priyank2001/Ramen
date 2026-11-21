"""
Re-run Ramen / the zero-shot-weighted active set (T=3) with the full softmax saved, for the offline stacking analysis.

The cached runs of seeds 1-4 (and every active_prob_T3 run) only kept per-sample predictions, but prior correction
and kNN smoothing need the whole probability vector. Results go under new names (<variant>_p) so the cached runs other
notebooks use are never overwritten; comparing the two is the sanity check (predictions must be 100% identical).

    CUDA_VISIBLE_DEVICES=0 python run_probs.py DomainNet 2 active_prob_T3
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ramen_analysis import make_experiment  # noqa: E402

VARIANTS = {'Ramen': ('Ramen', None),
            'active_prob_T3': ('RamenActive', {'active_set': 'prob', 'temperature': 3.0})}  # == active_seeds.py
NUM_WORKERS = {'ImageNetC5K': 16, 'DomainNet': 16}  # must match the cached runs, or the stream order differs


def run(dataset, seed, variant):
    exp = make_experiment(dataset, tta_mode='mixed', batch_size=100, n_per_domain=None, seed=seed,
                          num_workers=NUM_WORKERS[dataset]).load()
    algo, cfg = VARIANTS[variant]
    new = exp.run(algo, name=f'{variant}_p', cfg=cfg, capture_probs=True)
    old = pd.read_pickle(exp._cache_path(variant))
    same_order = (old.domain_idx.values == new.domain_idx.values).all() and \
                 (old.sample_idx.values == new.sample_idx.values).all()
    same = (old.pred.values == new.pred.values).mean() * 100
    print(f'{dataset} seed {seed} {variant}: same stream order {same_order}, same predictions as cached run '
          f'{same:.2f}% ({(old.pred.values != new.pred.values).sum():,} differ of {len(new):,})', flush=True)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3])
