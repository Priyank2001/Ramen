"""
Run the cache-contamination oracle variants (src/methods/RamenOracle.py) for the notebook
cache_contamination_oracles.ipynb. Runs are cached in new_experiments/results/ like every exp.run(), so the
notebook loads them instead of re-running.

Run detached, one shard per GPU (a shard takes every n-th variant):
    cd new_experiments/2026-10-23_cache_contamination
    CUDA_VISIBLE_DEVICES=0 setsid nohup python run_oracles.py ImageNetC5K 0 0 1 > logs/ImageNetC5K_s0_gpu0.log 2>&1 &
    #                                                          dataset   seed shard n_shards
Only some variants (comma-separated run names), e.g. a seed sweep of the variants that cleared the screen:
    CUDA_VISIBLE_DEVICES=1 setsid nohup python run_oracles.py DomainNet 1 0 1 --variants oracle_drop_cw,oracle_random_like_cw > logs/x.log 2>&1 &
Smoke test on a small subset (separate results folder, compares `none` with Ramen):
    CUDA_VISIBLE_DEVICES=0 python run_oracles.py ImageNetC5K 0 0 1 --smoke /tmp/oracle_smoke
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # new_experiments/, for ramen_analysis

from ramen_analysis import make_experiment  # noqa: E402

# the settings of the cached NoAdapt / Ramen runs; num_workers changes the stream order (tqdm gotcha), keep it
MODELS = {'ImageNetC5K': 'clip_vitbase16', 'DomainNet': 'clip_vitbase32'}
NUM_WORKERS = {'ImageNetC5K': 16, 'DomainNet': 16}

# run name -> RamenOracle config overrides
VARIANTS = {
    'oracle_none': {'oracle': 'none'},
    'oracle_file_true': {'oracle': 'file_true'},
    'oracle_drop_wrong': {'oracle': 'drop_wrong'},
    'oracle_drop_cw': {'oracle': 'drop_confident_wrong'},
    'oracle_random_like_wrong': {'oracle': 'random_like_wrong'},
    'oracle_random_like_cw': {'oracle': 'random_like_cw'},
}


def experiment(dataset, seed, n_per_domain=None, results_dir=None):
    kwargs = {} if results_dir is None else {'results_dir': results_dir}
    return make_experiment(dataset, model=MODELS[dataset], tta_mode='mixed', batch_size=100,
                           n_per_domain=n_per_domain, seed=seed, num_workers=NUM_WORKERS[dataset], **kwargs)


def run(dataset, seed, shard, n_shards, variants=None):
    exp = experiment(dataset, seed).load()
    selected = [(n, c) for n, c in VARIANTS.items() if variants is None or n in variants]
    for i, (name, cfg) in enumerate(selected):
        if i % n_shards == shard:
            t = time.time()
            exp.run('RamenOracle', name=name, cfg=cfg)
            print(f'[{time.strftime("%H:%M:%S")}] {dataset} seed {seed}: {name} done in {(time.time() - t) / 60:.1f} min',
                  flush=True)


def smoke(dataset, seed, results_dir):
    """Small subset: every variant runs, and `none` must give exactly Ramen's predictions."""
    exp = experiment(dataset, seed, n_per_domain=200, results_dir=results_dir).load()
    exp.run('Ramen')
    for name, cfg in VARIANTS.items():
        exp.run('RamenOracle', name=name, cfg=cfg, use_cache=False)
        st = exp.method_stats[name]
        acc = exp.results[name].correct.mean() * 100
        print(f'{name:26s} acc {acc:6.2f}  memory snapshot {st["memory"][-1] if st["memory"] else None}')
    same = (exp.results['oracle_none'].pred.values == exp.results['Ramen'].pred.values).mean()
    print(f'sanity: oracle_none == Ramen on {same:.4%} of samples')


if __name__ == '__main__':
    dataset, seed, shard, n_shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    if '--smoke' in sys.argv:
        smoke(dataset, seed, sys.argv[sys.argv.index('--smoke') + 1])
    else:
        variants = sys.argv[sys.argv.index('--variants') + 1].split(',') if '--variants' in sys.argv else None
        unknown = set(variants or []) - set(VARIANTS)
        if unknown:
            sys.exit(f'unknown variants: {sorted(unknown)}')
        run(dataset, seed, shard, n_shards, variants)
