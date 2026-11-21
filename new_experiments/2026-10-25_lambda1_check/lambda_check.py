"""
Step #1 of the post-stacking plan: does prior correction at the frozen lambda = 1 keep its gain on the small-label
datasets where lambda = 2 was tuned? Pre-registration and decision rule: PREREGISTRATION.md.

    python lambda_check.py              # seeds 0-4 on CIFAR10C, CIFAR100C, EuroSAT -> results/lambda1_check/*.csv
Offline and exact: uses the cached Ramen probabilities and the stacking code of ../2026-10-25_stacking/stack.py.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '2026-10-25_stacking'))
import stack  # noqa: E402

DATASETS = ['CIFAR10C', 'CIFAR100C', 'EuroSAT']
SEEDS = range(5)
LAMS = (2.0, 1.0, 0.5)
OUT = os.path.join(stack.R, 'lambda1_check') + '/'
# earlier seed means at lambda = 2 (marginal_correction.py, quick_checks/knn_seeds.py, eurosat runs): the sanity check
REFERENCE = {'CIFAR10C': {'Ramen + pc(lam=2)': 0.57, 'Ramen + knn + pc(lam=2)': 1.25},
             'CIFAR100C': {'Ramen + pc(lam=2)': 1.50, 'Ramen + knn + pc(lam=2)': 2.07},
             'EuroSAT': {'Ramen + pc(lam=2)': 2.79, 'Ramen + knn + pc(lam=2)': 7.02}}
SHOW = ['Ramen + pc(lam=2)', 'Ramen + pc(lam=1)', 'Ramen + pc(lam=0.5)', 'Ramen + knn',
        'Ramen + knn + pc(lam=2)', 'Ramen + knn + pc(lam=1)']


def rename(v):
    """stack.py calls the lambda = 2 correction plain 'pc'; here every lambda is spelled out."""
    return v.replace('pc', 'pc(lam=2)') if v.endswith(' pc') else v


def run():
    os.makedirs(OUT, exist_ok=True)
    tabs = []
    for ds in DATASETS:
        for seed in SEEDS:
            df, preds, diag = stack.stack_seed(ds, seed, LAMS)
            t = stack.table(ds, seed, df, preds)
            t.insert(0, 'dataset', ds)
            t['variant'] = t.variant.map(rename)
            tabs.append(t)
            print(f'{ds} seed {seed} done', flush=True)
    t = pd.concat(tabs, ignore_index=True)
    t.to_csv(OUT + 'gains.csv', index=False)
    return t


def load():
    return pd.read_csv(OUT + 'gains.csv')


def summary(t):
    """Mean +- std gain over seeds per (dataset, variant), seeds positive / above the floor, mean images vs Ramen."""
    rows = {}
    for (ds, v), g in t[t.variant.isin(SHOW)].groupby(['dataset', 'variant'], sort=False):
        fl = stack.NOISE_FLOOR[ds]
        rows[(stack.LABEL[ds], v)] = {'gain mean': g.gain.mean(), 'gain std': g.gain.std(), 'gain min': g.gain.min(),
                                      'positive': f'{(g.gain > 0).sum()}/{len(g)}',
                                      'above floor': f'{(g.gain > fl).sum()}/{len(g)}',
                                      'images vs Ramen': g['images vs Ramen'].mean(),
                                      'reference (earlier)': REFERENCE[ds].get(v, np.nan)}
    s = pd.DataFrame(rows).T
    order = [(stack.LABEL[ds], v) for ds in DATASETS for v in SHOW]
    return s.loc[[o for o in order if o in s.index]]


def decision(t):
    """The pre-registered rule: lambda = 1 mean gain above the floor AND >= 50% of the lambda = 2 mean gain."""
    rows = {}
    for ds in DATASETS:
        g = t[t.dataset == ds].groupby('variant').gain.mean()
        g1, g2, fl = g['Ramen + pc(lam=1)'], g['Ramen + pc(lam=2)'], stack.NOISE_FLOOR[ds]
        rows[stack.LABEL[ds]] = {'lam=2 gain': g2, 'lam=1 gain': g1, 'noise floor': fl,
                                 'lam=1 / lam=2': g1 / g2, '(i) above floor': g1 > fl, '(ii) >= 50% of lam=2': g1 >= 0.5 * g2,
                                 'passes': bool(g1 > fl and g1 >= 0.5 * g2)}
    return pd.DataFrame(rows).T


if __name__ == '__main__':
    t = run()
    pd.set_option('display.width', 220)
    print(summary(t).round(2).to_string())
    print(decision(t).to_string())
