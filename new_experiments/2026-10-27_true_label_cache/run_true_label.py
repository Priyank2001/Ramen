"""
Run the true-label cache experiment (true_label_cache.py) on one dataset and seed. Runs are cached in
new_experiments/results/ like every exp.run(), so the notebooks load them instead of re-running.

Run detached, one process per GPU:
    cd new_experiments/2026-10-27_true_label_cache
    CUDA_VISIBLE_DEVICES=0 setsid nohup python run_true_label.py DomainNet 0 > logs/DomainNet_s0.log 2>&1 < /dev/null &
Only some runs (comma-separated names from VARIANTS):
    ... python run_true_label.py DomainNet 1 --runs Ramen,true_label ...
Smoke test on 200 images per domain (separate results folder; zero_shot_filing must equal Ramen):
    CUDA_VISIBLE_DEVICES=0 python run_true_label.py DomainNet 0 --smoke <dir>
"""
import sys
import time

from true_label_cache import DEFAULT_RUNS, VARIANTS, TrueLabelCacheExperiment


def run(dataset, seed, names):
    exp = TrueLabelCacheExperiment(dataset, seed=seed)
    for name in names:
        t = time.time()
        exp.run(name)
        print(f'[{time.strftime("%H:%M:%S")}] {dataset} seed {seed}: {name} done in {(time.time() - t) / 60:.1f} min',
              flush=True)


def smoke(dataset, seed, results_dir):
    exp = TrueLabelCacheExperiment(dataset, seed=seed, n_per_domain=200, results_dir=results_dir)
    exp.run_all(('NoAdapt', 'Ramen', 'zero_shot_filing', 'true_label'), use_cache=False)
    print(exp.summary_table().to_string())
    if not exp.sanity_check('zero_shot_filing', 'Ramen'):
        sys.exit('SANITY CHECK FAILED: zero_shot_filing differs from Ramen')


if __name__ == '__main__':
    dataset, seed = sys.argv[1], int(sys.argv[2])
    if '--smoke' in sys.argv:
        smoke(dataset, seed, sys.argv[sys.argv.index('--smoke') + 1])
    else:
        names = sys.argv[sys.argv.index('--runs') + 1].split(',') if '--runs' in sys.argv else DEFAULT_RUNS
        unknown = set(names) - set(VARIANTS)
        if unknown:
            sys.exit(f'unknown runs: {sorted(unknown)}')
        run(dataset, seed, names)
