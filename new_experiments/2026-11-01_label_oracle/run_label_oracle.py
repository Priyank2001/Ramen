"""
Run the label-value oracle runs (label_oracle.py) on one dataset and seed; results are cached in
new_experiments/results/ like every exp.run(), so the notebook loads them.

Smoke test on 200 images per domain (separate results folder; the 'off' runs must equal their references 100%):
    CUDA_VISIBLE_DEVICES=0 python run_label_oracle.py DomainNet 0 --smoke <dir> [--runs <name,...>]   # rerun only these
One full run per GPU, detached:
    CUDA_VISIBLE_DEVICES=0 setsid nohup python run_label_oracle.py DomainNet 0 --runs labeloracle_true_ce \
        > logs/DomainNet_s0_labeloracle_true_ce.log 2>&1 < /dev/null &
"""
import sys
import time

from label_oracle import EXCL_RUNS, SMOKE_CHECKS, SMOKE_RUNS, VARIANTS, LabelOracleExperiment, new_runs


def run(dataset, seed, names):
    exp = LabelOracleExperiment(dataset, seed=seed)
    for name in names:
        t = time.time()
        exp.run(name)
        print(f'[{time.strftime("%H:%M:%S")}] {dataset} seed {seed}: {name} done in {(time.time() - t) / 60:.1f} min',
              flush=True)


def smoke(dataset, seed, results_dir, names=None):
    """names: rerun only these smoke runs (the others, references included, load from the smoke cache)."""
    exp = LabelOracleExperiment(dataset, seed=seed, n_per_domain=200, results_dir=results_dir)
    runs = list(SMOKE_RUNS) + ['excl_numerics'] + (list(EXCL_RUNS) if dataset == 'ImageNetC5K' else [])
    for n in runs:
        exp.run(n, use_cache=names is not None and n not in names)
    print(exp.summary_table(runs).to_string())
    exp.sanity_check('excl_numerics', 'Ramen')  # fp32 own retrieval path, nothing hidden: expect just below 100%
    if not all(exp.sanity_check(a, b) for a, b in SMOKE_CHECKS):
        sys.exit('SANITY CHECK FAILED')
    for n in ('labeloracle_zs_ce', 'labeloracle_true_ce'):
        revealed = sum(exp.stats[n]['revealed'])
        print(f'{n}: label gradients stored for {revealed:,} entries')


if __name__ == '__main__':
    dataset, seed = sys.argv[1], int(sys.argv[2])
    if '--smoke' in sys.argv:
        names = sys.argv[sys.argv.index('--runs') + 1].split(',') if '--runs' in sys.argv else None
        smoke(dataset, seed, sys.argv[sys.argv.index('--smoke') + 1], names)
    else:
        names = sys.argv[sys.argv.index('--runs') + 1].split(',') if '--runs' in sys.argv else new_runs(dataset)
        unknown = set(names) - set(VARIANTS)
        if unknown:
            sys.exit(f'unknown runs: {sorted(unknown)}')
        run(dataset, seed, names)
