"""
Stack the whitened NIQE-vector active-set regulariser (RamenNSS, lambda 0.1) with RamenActive's zero-shot class
weighting (active_set prob, T = 3) on ImageNetC5K, mixed stream.

Both settings were fixed before this run (PREREGISTRATION.md): niqe / white / lambda 0.1 from nss_active_set.md,
T = 3 from active_seeds.py. No method code is new: RamenNSS (src/methods/RamenNSS.py) already composes with
RamenActive's `active_set` / `temperature` keys, so the variants are config overrides only.

Usage (from this folder):
    python run_stack.py smoke              # n_per_domain=200: nss l0 + prob T3 must equal RamenActive prob T3 (100%)
    python run_stack.py run 0 3            # full ImageNetC5K, seeds 0 and 3, one after the other
Results land in ../results/ImageNetC5K_clip_vitbase16_mixed_nall_bs100_s<seed>_<name>.pkl (harness cache).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from active_seeds import NUM_WORKERS  # noqa: E402  (matches the cached Ramen runs' stream order)
from ramen_analysis import make_experiment  # noqa: E402

DATASET = 'ImageNetC5K'
T3 = {'active_set': 'prob', 'temperature': 3.0}
NIQE = {'nss_features': 'niqe', 'nss_norm': 'white'}

# name -> (algo, cfg overrides). Baselines already cached for seeds 0-4: Ramen, active_prob_T3, nss_niqe_white_l0.1
VARIANTS = {
    'nss_niqe_white_l0.1_probT3': ('RamenNSS', {**NIQE, 'nss_lambda': 0.1, **T3}),
}
# Sanity check only: lambda 0 must reproduce RamenActive prob T3 exactly
SMOKE = {
    'smoke_active_prob_T3': ('RamenActive', dict(T3)),
    'smoke_nss_l0_probT3': ('RamenNSS', {**NIQE, 'nss_lambda': 0.0, **T3}),
    'smoke_nss_niqe_white_l0.1_probT3': VARIANTS['nss_niqe_white_l0.1_probT3'],
}


def experiment(seed, n_per_domain=None):
    return make_experiment(DATASET, tta_mode='mixed', batch_size=100, n_per_domain=n_per_domain, seed=seed,
                           num_workers=NUM_WORKERS[DATASET]).load()


def smoke():
    exp = experiment(0, n_per_domain=200)  # 15 x 200 = 3,000 images = 30 full batches (cdist fp16 needs > 25 rows)
    for name, (algo, cfg) in SMOKE.items():
        exp.run(algo, name=name, cfg=cfg, use_cache=False)
    a, b = exp.results['smoke_active_prob_T3'], exp.results['smoke_nss_l0_probT3']
    same = (a.pred.values == b.pred.values).mean() * 100
    print(f'lambda 0 vs RamenActive prob T3: {same:.2f}% same predictions '
          f'({(a.pred.values == b.pred.values).sum():,} of {len(a):,})', flush=True)
    s = exp.results['smoke_nss_niqe_white_l0.1_probT3']
    print(f'stack lambda 0.1 differs on {(s.pred.values != a.pred.values).sum():,} of {len(a):,}', flush=True)
    assert same == 100.0, 'lambda 0 does not reproduce RamenActive prob T3'
    print('SMOKE OK', flush=True)


def run(seeds):
    for seed in seeds:
        exp = experiment(seed)
        for name, (algo, cfg) in VARIANTS.items():
            exp.run(algo, name=name, cfg=cfg)
            acc = exp.results[name].groupby('domain_idx').correct.mean().mean() * 100
            print(f'seed {seed} {name} done: domain-avg acc {acc:.2f}', flush=True)
        del exp


if __name__ == '__main__':
    if sys.argv[1] == 'smoke':
        smoke()
    elif sys.argv[1] == 'run':
        run([int(s) for s in sys.argv[2:]])
    else:
        raise SystemExit(__doc__)
