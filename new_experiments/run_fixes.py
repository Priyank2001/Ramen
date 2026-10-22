"""
Step 5: run the step-4 fixes on the full stream (cached under their names in results/).
Usage (from new_experiments/):  CUDA_VISIBLE_DEVICES=0 python run_fixes.py CIFAR100C clean_g1 nsgd ...
"""
import sys

from ramen_analysis import make_experiment

VARIANTS = {
    'clean_g1': ('RamenClean', {'gamma': 1.0}),
    'clean_g2': ('RamenClean', {'gamma': 2.0}),
    'fp32prio': ('RamenClean', {'gamma': 0.0, 'priority_dtype': 'float32'}),  # Ramen + fp32 cache priorities only
    'nsgd': ('RamenNSGD', {}),
}

if __name__ == '__main__':
    exp = make_experiment(sys.argv[1]).load()
    exp.evaluate()
    for name in sys.argv[2:]:
        algo, cfg = VARIANTS[name]
        exp.run(algo, name=name, cfg=cfg)
        print(f'{sys.argv[1]} {name}: acc {exp.results[name].correct.mean() * 100:.2f}  '
              f'(Ramen {exp.results["Ramen"].correct.mean() * 100:.2f})', flush=True)
