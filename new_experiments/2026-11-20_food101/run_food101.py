"""
Food-101 runs.

    CUDA_VISIBLE_DEVICES=0 python run_food101.py --smoke          # 500 images: loader, labels, zero-shot vs Ramen
    CUDA_VISIBLE_DEVICES=0 python run_food101.py 0 Ramen          # one (seed, run); runs: NoAdapt | Ramen | active_prob_T3
Results are cached in new_experiments/results/ like any exp.run().
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from experiment import make_food101  # noqa: E402

RUNS = {'NoAdapt': ('NoAdapt', None),
        'Ramen': ('Ramen', None),
        'active_prob_T3': ('RamenActive', {'active_set': 'prob', 'temperature': 3.0})}


def run(seed, name):
    exp = make_food101(seed=seed)
    algo, cfg = RUNS[name]
    df = exp.run(algo, name=name, cfg=cfg, capture_probs=True, capture_features=name == 'NoAdapt')
    print(f'Food101 seed {seed} {name}: accuracy {df.correct.mean() * 100:.2f}', flush=True)


def smoke():
    """Sizes, label range, class names and accuracy on a 500-image subset; cached under *_smoke names, then deleted."""
    exp = make_food101(seed=0, n_per_domain=500)
    ds = exp.datasets[0]
    labels = sorted(set(ds.targets))
    print(f'{len(ds)} images (subset), {len(labels)} distinct labels, range {labels[0]}-{labels[-1]}, '
          f'{len(exp.classes)} classes: {exp.classes[:3]} ... {exp.classes[-2:]}')
    for algo in ('NoAdapt', 'Ramen'):
        df = exp.run(algo, use_cache=False, name=f'{algo}_smoke')
        print(f'{algo}: accuracy {df.correct.mean() * 100:.2f}')
        os.remove(exp._cache_path(f'{algo}_smoke'))


if __name__ == '__main__':
    smoke() if sys.argv[1] == '--smoke' else run(int(sys.argv[1]), sys.argv[2])
