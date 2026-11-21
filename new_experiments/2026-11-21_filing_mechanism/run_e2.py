"""
E2 runner (see PREREGISTRATION.md).

    python run_e2.py smoke DomainNet               # subset: hide none == RamenTrueLabel / Ramen (100%); A-D run
    python run_e2.py run DomainNet:hide_samezs_wrong_true ImageNetC5K:hide_random_wrong_true ...   # one after another
Results: ../results/<dataset>_<model>_mixed_nall_bs100_s0_<variant>.pkl (+ _stats.pkl)
"""
import gc
import sys

import torch

from experiment import SMOKE_PAIRS, VARIANTS, make_local_experiment

SMOKE_N = {'DomainNet': 200, 'ImageNetC5K': 200, 'EuroSAT': 3000}  # full 100-image batches only


def smoke(dataset):
    exp = make_local_experiment(dataset, seed=0, n_per_domain=SMOKE_N[dataset]).load()
    ok = True
    for check, ((algo, cfg), (ref_algo, ref_cfg)) in SMOKE_PAIRS.items():
        a = exp.run(algo, name='smoke_new', cfg=cfg, use_cache=False)
        b = exp.run(ref_algo, name='smoke_ref', cfg=ref_cfg, use_cache=False)
        same = int((a.pred.values == b.pred.values).sum())
        print(f'{dataset} | {check}: {same:,} of {len(a):,} same ({same / len(a):.2%})', flush=True)
        ok &= same == len(a)
    for name, (algo, cfg) in VARIANTS.items():
        df = exp.run(algo, name=f'smoke_{name}', cfg=cfg, use_cache=False)
        st = exp.method_stats[f'smoke_{name}']
        hidden = torch.cat(st['hidden']).float()
        print(f'{dataset} | {name}: acc {df.correct.mean() * 100:.2f}, hidden per query mean {hidden.mean():.1f}, '
              f'queries with something hidden {(hidden > 0).float().mean():.1%}', flush=True)
    print('SMOKE OK' if ok else 'SMOKE FAILED', flush=True)


def run(jobs):
    for job in jobs:
        dataset, name = job.split(':')
        exp = make_local_experiment(dataset, seed=0).load()
        algo, cfg = VARIANTS[name]
        df = exp.run(algo, name=name, cfg=cfg)
        acc = df.groupby('domain_idx').correct.mean().mean() * 100
        print(f'{dataset} {name} done: domain-avg acc {acc:.2f}', flush=True)
        del exp, df
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    if sys.argv[1] == 'smoke':
        smoke(sys.argv[2])
    elif sys.argv[1] == 'run':
        run(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
