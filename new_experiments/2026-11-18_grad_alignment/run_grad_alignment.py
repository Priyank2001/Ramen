"""
Run Ramen with the gradient probe on one dataset and seed (cached in new_experiments/results/ like every exp.run()).

Smoke (200 images per domain, separate folder; ga_probe must equal Ramen on 100.00% of images):
    CUDA_VISIBLE_DEVICES=0 python run_grad_alignment.py DomainNet 0 --smoke smoke
Full run, detached:
    CUDA_VISIBLE_DEVICES=0 setsid nohup python run_grad_alignment.py DomainNet 0 \
        > logs/DomainNet_s0.log 2>&1 < /dev/null &
"""
import sys
import time

from ga_configs import PROBE_RUN
from ga_experiment import GradAlignmentExperiment


def main():
    dataset, seed = sys.argv[1], int(sys.argv[2])
    smoke = '--smoke' in sys.argv
    kwargs = dict(n_per_domain=200, results_dir=sys.argv[sys.argv.index('--smoke') + 1]) if smoke else {}
    exp = GradAlignmentExperiment(dataset, seed=seed, **kwargs)
    for name in ('NoAdapt', 'Ramen', PROBE_RUN):
        t = time.time()
        exp.run(name)
        print(f'[{time.strftime("%H:%M:%S")}] {dataset} seed {seed}: {name} done in {(time.time() - t) / 60:.1f} min',
              flush=True)
    if not exp.sanity_check():
        sys.exit('SANITY CHECK FAILED: the probe changed Ramen\'s predictions')
    df = exp.frame()
    print(df.groupby('outcome', observed=False)[['cos_update', 'cos_step', 'cos_self', 'cos_update_shuffled']]
          .median().round(4).to_string())


if __name__ == '__main__':
    main()
