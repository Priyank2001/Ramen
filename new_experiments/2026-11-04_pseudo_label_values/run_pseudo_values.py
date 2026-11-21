"""
Run the pseudo-label value runs (pseudo_values.py) on one dataset and seed; results are cached in
new_experiments/results/ like every exp.run(), so the notebook loads them.

Smoke test on 200 images per domain (separate results folder; pv_off must equal Ramen and pv_true the label oracle):
    CUDA_VISIBLE_DEVICES=0 python run_pseudo_values.py DomainNet 0 --smoke smoke
Full runs, one process per GPU, detached:
    CUDA_VISIBLE_DEVICES=0 setsid nohup python run_pseudo_values.py DomainNet 0 --runs pv_zs,pv_adapted \
        > logs/DomainNet_s0_a.log 2>&1 < /dev/null &
"""
import sys
import time

from pseudo_values import PV_VARIANTS, SMOKE_CHECKS, SMOKE_RUNS, PseudoValueExperiment


def run(dataset, seed, names):
    exp = PseudoValueExperiment(dataset, seed=seed)
    for name in names:
        t = time.time()
        exp.run(name)
        print(f'[{time.strftime("%H:%M:%S")}] {dataset} seed {seed}: {name} done in {(time.time() - t) / 60:.1f} min',
              flush=True)


def smoke(dataset, seed, results_dir):
    exp = PseudoValueExperiment(dataset, seed=seed, n_per_domain=200, results_dir=results_dir)
    for n in SMOKE_RUNS:
        exp.run(n)
    print(exp.summary_table(list(SMOKE_RUNS)).to_string())
    print(exp.target_table([n for n in SMOKE_RUNS if n in PV_VARIANTS and n != 'pv_off']).to_string())
    if not all(exp.sanity_check(a, b) for a, b in SMOKE_CHECKS):
        sys.exit('SANITY CHECK FAILED')


if __name__ == '__main__':
    dataset, seed = sys.argv[1], int(sys.argv[2])
    if '--smoke' in sys.argv:
        smoke(dataset, seed, sys.argv[sys.argv.index('--smoke') + 1])
    else:
        names = sys.argv[sys.argv.index('--runs') + 1].split(',')
        unknown = set(names) - set(PV_VARIANTS)
        if unknown:
            sys.exit(f'unknown runs: {sorted(unknown)}')
        run(dataset, seed, names)
