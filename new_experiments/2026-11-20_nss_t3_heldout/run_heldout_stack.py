"""
Held-out test of the NIQE vector + T3 stack on ImageNetVariants (ImageNet-R + -A + -Sketch), see PREREGISTRATION.md.

There is no cfg/ImageNetVariants/RamenNSS.yaml (cfg/ is read-only), so every run passes the full config:
cfg/ImageNetC5K/RamenNSS.yaml + the run's overrides. Its Ramen part equals cfg/ImageNetVariants/Ramen.yaml (asserted).

    python run_heldout_stack.py smoke             # 1,000 images per domain: lambda 0 + T3 == RamenActive T3, NaN rate
    python run_heldout_stack.py run stack:0 niqe:2  # full runs, one after the other (run:seed)
Results: ../results/ImageNetVariants_clip_vitbase16_mixed_nall_bs100_s<seed>_<name>.pkl (+ _stats, _probs).
Cached baselines reused: Ramen, active_prob_T3 (../2026-10-25_heldout/).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from ramen_analysis import make_experiment  # noqa: E402

DATASET = 'ImageNetVariants'
NUM_WORKERS = 16  # == ../2026-10-25_heldout/run_heldout.py, so stream orders match the cached Ramen / T3 runs


def _yaml(path):
    with open(os.path.join(ROOT, 'cfg', path)) as f:
        return yaml.safe_load(f)


BASE = _yaml('ImageNetC5K/RamenNSS.yaml')
assert all(BASE[k] == v for k, v in _yaml('ImageNetVariants/Ramen.yaml').items()), 'Ramen settings differ'

T3 = {'active_set': 'prob', 'temperature': 3.0}
NIQE = {'nss_features': 'niqe', 'nss_norm': 'white', 'nss_lambda': 0.1}
# short key -> (cached run name, algo, full config); frozen in PREREGISTRATION.md
RUNS = {
    'stack': ('nss_niqe_white_l0.1_probT3', 'RamenNSS', {**BASE, **NIQE, **T3}),
    'niqe': ('nss_niqe_white_l0.1', 'RamenNSS', {**BASE, **NIQE}),
}


def experiment(seed, n_per_domain=None):
    return make_experiment(DATASET, tta_mode='mixed', batch_size=100, n_per_domain=n_per_domain, seed=seed,
                           num_workers=NUM_WORKERS).load()


def nan_rate(exp):
    """Share of images whose block-mean NIQE vector is undefined (every block flat), per domain. Label-free."""
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    from models.iqa import ImageQuality
    iqa = ImageQuality(torch.device('cuda'))
    for env, ds in zip(exp.environments, exp.datasets.datasets):
        loader = torch.utils.data.DataLoader(ds, batch_size=250, num_workers=NUM_WORKERS)
        bad = n = 0
        for x, _ in loader:
            _, nss = iqa.with_features(x.cuda())
            bad += int(torch.isnan(nss[:, 36:72]).any(dim=1).sum())
            n += len(x)
        print(f'{env}: undefined NIQE vector on {bad:,} of {n:,} images ({bad / n:.1%})', flush=True)


def smoke():
    exp = experiment(0, n_per_domain=1000)  # 3 x 1,000 = 3,000 images = 30 full batches
    ref = exp.run('RamenActive', name='smoke_active_prob_T3', cfg=dict(T3), use_cache=False)
    off = exp.run('RamenNSS', name='smoke_nss_l0_probT3', cfg={**BASE, **NIQE, **T3, 'nss_lambda': 0.0},
                  use_cache=False)
    on = exp.run('RamenNSS', name='smoke_stack', cfg=RUNS['stack'][2], use_cache=False)
    same = int((ref.pred.values == off.pred.values).sum())
    print(f'lambda 0 vs RamenActive prob T3: {same:,} of {len(ref):,} same predictions ({same / len(ref):.2%})')
    print(f'stack lambda 0.1 differs on {int((on.pred.values != ref.pred.values).sum()):,} of {len(ref):,}')
    nan_rate(exp)
    assert same == len(ref), 'lambda 0 does not reproduce RamenActive prob T3'
    print('SMOKE OK', flush=True)


def run(jobs):
    for job in jobs:
        key, seed = job.split(':')
        name, algo, cfg = RUNS[key]
        exp = experiment(int(seed))
        df = exp.run(algo, name=name, cfg=cfg, capture_probs=True)
        acc = df.groupby('domain', observed=True).correct.mean() * 100
        print(f'{DATASET} seed {seed} {name}: domain-avg {acc.mean():.2f} | ' +
              ', '.join(f'{d} {a:.2f}' for d, a in acc.items()), flush=True)
        del exp, df


if __name__ == '__main__':
    if sys.argv[1] == 'smoke':
        smoke()
    elif sys.argv[1] == 'run':
        run(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
