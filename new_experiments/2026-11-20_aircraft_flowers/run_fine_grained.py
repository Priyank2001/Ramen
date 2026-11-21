"""
    CUDA_VISIBLE_DEVICES=0 python run_fine_grained.py --smoke              # both datasets, 300 images: loader, labels, zero-shot vs Ramen
    CUDA_VISIBLE_DEVICES=0 python run_fine_grained.py Aircraft 0 Ramen     # dataset, seed, run: NoAdapt | Ramen | active_prob_T3
Results are cached in new_experiments/results/ like any exp.run().
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fine_grained_experiment import EXPERIMENTS, make_fine_grained  # noqa: E402

RUNS = {'NoAdapt': ('NoAdapt', None),
        'Ramen': ('Ramen', None),
        'active_prob_T3': ('RamenActive', {'active_set': 'prob', 'temperature': 3.0})}


def run(dataset, seed, name):
    exp = make_fine_grained(dataset, seed=seed)
    algo, cfg = RUNS[name]
    df = exp.run(algo, name=name, cfg=cfg, capture_probs=True, capture_features=name == 'NoAdapt')
    print(f'{dataset} seed {seed} {name}: accuracy {df.correct.mean() * 100:.2f}', flush=True)


def smoke():
    """Full split sizes, then accuracy on a 300-image subset; cached under *_smoke names, then deleted."""
    for dataset in EXPERIMENTS:
        full = make_fine_grained(dataset, seed=0).datasets[0]
        print(f'{dataset}: full split {len(full)} images, {len(set(full.targets))} distinct labels, '
              f'classes {full.targets and min(full.targets)}-{max(full.targets)}')
        exp = make_fine_grained(dataset, seed=0, n_per_domain=300)
        print(f'  names: {exp.classes[:3]} ... {exp.classes[-2:]}')
        for algo in ('NoAdapt', 'Ramen'):
            df = exp.run(algo, use_cache=False, name=f'{algo}_smoke')
            print(f'  {algo}: accuracy {df.correct.mean() * 100:.2f}')
            os.remove(exp._cache_path(f'{algo}_smoke'))


if __name__ == '__main__':
    smoke() if sys.argv[1] == '--smoke' else run(sys.argv[1], int(sys.argv[2]), sys.argv[3])
