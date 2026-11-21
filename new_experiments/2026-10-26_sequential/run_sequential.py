"""
Domain-sequential runs (PREREGISTRATION.md): domains one after another, never reset (Ramen, T3), or reset at every
domain start (Ramen-reset = tta_mode 'single'). NoAdapt is reused from the mixed runs, except the sanity run.

    CUDA_VISIBLE_DEVICES=0 python run_sequential.py DomainNet 0 Ramen   # runs: Ramen | active_prob_T3 | Ramen_reset | NoAdapt | Ramen_cdistcheck
Results are cached in results/ like any exp.run(), with full probabilities, under ..._sequential_... / ..._single_...
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ramen_analysis import make_experiment  # noqa: E402

NUM_WORKERS = 16
RUNS = {'NoAdapt': ('NoAdapt', None, 'sequential'),
        'Ramen': ('Ramen', None, 'sequential'),
        'active_prob_T3': ('RamenActive', {'active_set': 'prob', 'temperature': 3.0}, 'sequential'),  # == active_seeds.py
        'Ramen_reset': ('Ramen', None, 'single'),
        'Ramen_cdistcheck': ('Ramen', None, 'mixed'),  # patched cdist must reproduce the cached mixed Ramen exactly
        'Ramen_fp16check': ('Ramen', None, 'mixed'),  # refactored priorities at the default (fp16): same check
        'Ramen_fp32': ('Ramen', {'priority_fp32': True}, 'sequential'),
        'active_prob_T3_fp32': ('RamenActive', {'active_set': 'prob', 'temperature': 3.0, 'priority_fp32': True}, 'sequential')}


def run(ds, seed, name):
    algo, cfg, mode = RUNS[name]
    exp = make_experiment(ds, tta_mode=mode, batch_size=100, seed=seed, num_workers=NUM_WORKERS).load()
    df = exp.run(algo, name=name, cfg=cfg, capture_probs=True)
    acc = df.groupby('domain', observed=True).correct.mean() * 100
    print(f'{ds} {mode} seed {seed} {name}: domain-avg {acc.mean():.2f} | ' +
          ', '.join(f'{d} {a:.2f}' for d, a in acc.items()), flush=True)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3])
