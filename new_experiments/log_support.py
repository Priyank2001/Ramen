"""
Step 1: log one full Ramen run with per-sample support sets (methods/RamenLog.py).
Usage (from new_experiments/):  CUDA_VISIBLE_DEVICES=1 python log_support.py CIFAR100C
Writes results/<run>_RamenLog.pkl plus _zs_probs/_support_ids/_support_alpha/_inserted.npy.
"""
import sys

from ramen_analysis import make_experiment

exp = make_experiment(sys.argv[1]).load()
exp.evaluate()  # NoAdapt + Ramen (cached)
exp.run('RamenLog')
same = (exp.results['RamenLog'].pred.values == exp.results['Ramen'].pred.values).mean()
print(f'{sys.argv[1]}: RamenLog prediction agreement with Ramen = {same:.4%}')
