"""
CLI: run every variant of one dataset / seed (offline; cached under new_experiments/results/).

    python run_cluster_shift.py ImageNetC5K 0
    python run_cluster_shift.py EuroSAT 0 --only kmeans_k20_a1.0_text
"""
import argparse

from cshift_configs import VARIANTS
from cshift_experiment import ClusterShiftExperiment

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('seed', type=int)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    e = ClusterShiftExperiment(a.dataset, a.seed)
    e.sanity_check()
    names = a.only or list(VARIANTS)
    print(e.summary(names).round(2).to_string(index=False))
