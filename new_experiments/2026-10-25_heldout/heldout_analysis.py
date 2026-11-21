"""
Analysis of the held-out test (PREREGISTRATION.md): T3 + pc(lam=1) vs Ramen on ImageNetVariants, seeds 0-4.

    python heldout_analysis.py          # -> results/heldout/{gains,per_domain,knn_diag}.csv, prints the verdict
Offline from the saved probabilities, with the stacking code of ../2026-10-25_stacking/stack.py.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '2026-10-25_stacking'))
import stack  # noqa: E402

DS = 'ImageNetVariants'
SEEDS = range(5)
OUT = os.path.join(stack.R, 'heldout') + '/'
PRIMARY = 'T3 + pc(lam=1)'
SECONDARY = ['T3', 'Ramen + pc(lam=1)']
SHOW = [PRIMARY] + SECONDARY + ['T3 + pc(lam=2)', 'T3 + pc(lam=0.5)', 'Ramen + knn', 'T3 + knn + pc(lam=1)']


def role(v):
    return 'primary' if v == PRIMARY else 'secondary' if v in SECONDARY else 'post-hoc'


def rename(v):
    """stack.py calls the lambda = 2 correction plain 'pc'; spell every lambda out."""
    return v.replace('pc', 'pc(lam=2)') if v.endswith(' pc') else v


def run():
    os.makedirs(OUT, exist_ok=True)
    tabs, doms, diags = [], [], []
    for seed in SEEDS:
        df, preds, diag = stack.stack_seed(DS, seed, lams=(1.0, 2.0, 0.5))
        preds = {rename(k): v for k, v in preds.items()}
        tabs.append(stack.table(DS, seed, df, preds))
        doms.append(stack.per_domain(DS, seed, df, preds))
        diags.append({'seed': seed, **diag})
        print(f'seed {seed} done', flush=True)
    t = pd.concat(tabs, ignore_index=True)
    t['role'] = t.variant.map(role)
    t.to_csv(OUT + 'gains.csv', index=False)
    pd.concat(doms, ignore_index=True).to_csv(OUT + 'per_domain.csv', index=False)
    pd.DataFrame(diags).to_csv(OUT + 'knn_diag.csv', index=False)
    return t


def load(kind='gains'):
    return pd.read_csv(OUT + f'{kind}.csv')


def noise_floor():
    return stack.floor(DS)


def summary(t, floor):
    rows = {}
    for v in SHOW:
        g = t[t.variant == v]
        rows[v] = {'role': role(v), 'gain mean': g.gain.mean(), 'gain std': g.gain.std(), 'gain min': g.gain.min(),
                   'positive': f'{(g.gain > 0).sum()}/{len(g)}', 'above floor': f'{(g.gain > floor).sum()}/{len(g)}',
                   'CI excludes 0': f'{(g["CI low"] > 0).sum()}/{len(g)}',
                   'images vs Ramen': g['images vs Ramen'].mean()}
    return pd.DataFrame(rows).T


def verdict(t, floor):
    g = t[t.variant == PRIMARY].gain
    return {'primary': PRIMARY, 'mean gain': round(g.mean(), 3), 'std': round(g.std(), 3), 'noise floor': round(floor, 3),
            'seeds above floor': f'{(g > floor).sum()}/{len(g)}', 'claim holds': bool(g.mean() > floor)}


def vs_noadapt(t):
    """Domain-averaged accuracy of NoAdapt, Ramen, T3 and the primary, mean over seeds (does Ramen help here at all?)."""
    z = pd.read_pickle(stack.path(DS, 0, 'NoAdapt') + '.pkl')
    acc0 = stack.domain_acc(z, z.pred.values)
    a = t.groupby('variant').acc.mean()
    rows = {'NoAdapt': acc0, **{v: a[v] for v in ['Ramen', 'T3', 'Ramen + pc(lam=1)', PRIMARY]}}
    return pd.DataFrame({'acc (domain-avg)': rows, 'vs NoAdapt': {k: v - acc0 for k, v in rows.items()}})


if __name__ == '__main__':
    t = run()
    fl, accs = noise_floor()
    pd.set_option('display.width', 220)
    print('Ramen per seed:', [round(a, 2) for a in accs], 'floor', round(fl, 3))
    print(summary(t, fl).to_string())
    print(vs_noadapt(t).round(2).to_string())
    print(verdict(t, fl))
