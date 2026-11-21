"""
Record a run (experiment.py) on one dataset and seed, then check it reproduces the plain method. Runs are cached in
new_experiments/results/ like every exp.run(), so the notebooks load them instead of re-running.

Quick subset (n images per domain):
    CUDA_VISIBLE_DEVICES=0 python run_records.py DomainNet 0 --n-per-domain 500
Full dataset, detached:
    CUDA_VISIBLE_DEVICES=0 setsid nohup python run_records.py DomainNet 0 > logs/DomainNet_s0.log 2>&1 < /dev/null &
Another variant (from VARIANTS):
    ... python run_records.py DomainNet 0 --variant true_label
"""
import argparse
import sys
import time

from experiment import DATASETS, VARIANTS, RecordExperiment

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset', choices=list(DATASETS))
    p.add_argument('seed', type=int)
    p.add_argument('--n-per-domain', type=int, default=None, help='images per domain (default: all)')
    p.add_argument('--variant', default='Ramen', choices=list(VARIANTS))
    p.add_argument('--no-cache', action='store_true', help='re-run even if the recorded run is cached')
    p.add_argument('--skip-check', action='store_true', help='do not compare with the plain run')
    a = p.parse_args()

    t = time.time()
    exp = RecordExperiment(a.dataset, seed=a.seed, n_per_domain=a.n_per_domain, variant=a.variant)
    exp.run(use_cache=not a.no_cache)
    print(f'[{time.strftime("%H:%M:%S")}] recorded {len(exp.records):,} images in {(time.time() - t) / 60:.1f} min '
          f'-> {exp.prefix}_rec_*.npy', flush=True)
    print(exp.accuracy_table().to_string())
    if not a.skip_check and not exp.sanity_check():
        sys.exit('SANITY CHECK FAILED: the recorder changed the predictions')
