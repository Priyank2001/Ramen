"""
CLI: sanity check + every variant of one dataset / seed (offline, CPU; cached under new_experiments/results/).

    python run_gmm.py EuroSAT 0
    python run_gmm.py DomainNet 0 --only zs lda
    python run_gmm.py DomainNet 0 --rerun      # ignore cached variant results
"""
import argparse
import json
import os

import pandas as pd

from gmm_configs import VARIANTS
from gmm_experiment import GMMLabelExperiment

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('seed', type=int)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--rerun', action='store_true')
    a = ap.parse_args()
    e = GMMLabelExperiment(a.dataset, a.seed)
    check = e.sanity_check()
    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    with open(os.path.join(HERE, 'logs', f'sanity_{a.dataset}_s{a.seed}.json'), 'w') as f:
        json.dump({k: (float(v) if not isinstance(v, str) else v) for k, v in check.items()}, f, indent=1)
    names = a.only or list(VARIANTS)
    for n in names:
        e.run(n, use_cache=not a.rerun)
    cols = ['variant', 'acc_domain_avg', 'acc_pooled', 'gain_pooled', 'fixed', 'broken', 'rescue_rate', 'break_rate',
            'max_class_share', 'n_classes_used', 'verdict']
    with pd.option_context('display.width', 200):
        print(e.table(names)[cols].round(4).to_string(index=False))
