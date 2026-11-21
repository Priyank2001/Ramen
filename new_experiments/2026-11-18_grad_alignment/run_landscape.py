"""
Replay Ramen with the loss-landscape probe on one dataset and seed (cached in new_experiments/results/).

Smoke (200 images per domain, selection from the smoke ga_probe frame; must pass every check):
    CUDA_VISIBLE_DEVICES=0 python run_landscape.py DomainNet 0 --smoke smoke
Full run, detached:
    CUDA_VISIBLE_DEVICES=0 setsid nohup python run_landscape.py DomainNet 0 > logs/landscape_DomainNet_s0.log 2>&1 < /dev/null &
"""
import sys
import time

from landscape_experiment import LandscapeExperiment


def main():
    dataset, seed = sys.argv[1], int(sys.argv[2])
    smoke = '--smoke' in sys.argv
    kwargs = dict(n_per_domain=200, results_dir=sys.argv[sys.argv.index('--smoke') + 1]) if smoke else {}
    exp = LandscapeExperiment(dataset, seed=seed, **kwargs)
    t = time.time()
    exp.run_all()
    print(f'[{time.strftime("%H:%M:%S")}] {dataset} seed {seed}: landscape run done in {(time.time() - t) / 60:.1f} min',
          flush=True)
    if not exp.sanity_check_landscape():
        sys.exit('SANITY CHECK FAILED')
    print('all checks passed')


if __name__ == '__main__':
    main()
