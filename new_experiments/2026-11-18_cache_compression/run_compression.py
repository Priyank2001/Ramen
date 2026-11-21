"""
Run cache-compression variants on one dataset and seed (cached in new_experiments/results/ like every exp.run()).

Smoke (200 images per domain, separate folder; cc_off must equal Ramen on 100.00% of images):
    CUDA_VISIBLE_DEVICES=0 python run_compression.py ImageNetC5K 0 --smoke smoke
Full runs, one process per GPU, detached:
    CUDA_VISIBLE_DEVICES=0 setsid nohup python run_compression.py ImageNetC5K 0 --runs cc_off,cc_int8 \
        > logs/ImageNetC5K_s0_gpu0.log 2>&1 < /dev/null &
"""
import gc
import sys
import time

import torch

from cc_experiment import CompressionExperiment


def run(exp, names):
    for name in names:
        t = time.time()
        exp.run(name)
        gc.collect()
        torch.cuda.empty_cache()
        print(f'[{time.strftime("%H:%M:%S")}] {exp.dataset} seed {exp.seed}: {name} done in '
              f'{(time.time() - t) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    dataset, seed = sys.argv[1], int(sys.argv[2])
    if '--smoke' in sys.argv:
        exp = CompressionExperiment(dataset, seed=seed, n_per_domain=200,
                                    results_dir=sys.argv[sys.argv.index('--smoke') + 1])
        names = ['NoAdapt', 'Ramen', *exp.variants]
        run(exp, names)
        print(exp.table(names).round(4).to_string())
        if not exp.sanity_check('cc_off', 'Ramen'):
            sys.exit('SANITY CHECK FAILED')
    else:
        exp = CompressionExperiment(dataset, seed=seed)
        names = sys.argv[sys.argv.index('--runs') + 1].split(',')
        unknown = set(names) - set(exp.variants) - {'NoAdapt', 'Ramen'}
        if unknown:
            sys.exit(f'unknown runs: {sorted(unknown)}')
        run(exp, names)
