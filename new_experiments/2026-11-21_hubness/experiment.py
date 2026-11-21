"""
The hubness check (PREREGISTRATION.md): tuning on ImageNet-C seed 0, frozen held-out evaluation at the zero-shot level
(6 datasets x seeds 0-4), and the Ramen level (seed 0). Offline, GPU only for the similarity products.

    python experiment.py tune         # -> tables/tuning.csv, tables/frozen.json
    python experiment.py zero-shot    # -> tables/zero_shot_runs.csv, tables/zero_shot_summary.csv, verdicts
    python experiment.py ramen        # -> tables/ramen_runs.csv
Uses stack.py (read-only) for cache paths, the noise floors and the pc reference.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import skew

from data import DATASETS, FLOOR, RAMEN_FEATS, StreamData, stack
from hubness import Centring, Csls, InvertedSoftmax, PermutedCsls, ZeroShot, run

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, 'tables')
FROZEN = os.path.join(TABLES, 'frozen.json')
GRID = {'csls': [dict(kappa=k, beta=b) for k in (5, 20, 100) for b in (0.5, 1, 2, 4)],
        'inverted softmax': [dict(tau=t) for t in (20, 50, 100)]}
TUNE_DATASET = 'ImageNetC5K'


def build(method, W, params, seed=0):
    """name -> corrector. Parameters come from the frozen settings; seed only fixes the control's permutation."""
    if method == 'zero-shot':
        return ZeroShot(W)
    if method == 'csls':
        return Csls(W, **params)
    if method == 'permuted csls':
        return PermutedCsls(W, **params, seed=seed)
    if method == 'inverted softmax':
        return InvertedSoftmax(W, **params)
    if method == 'centring':
        return Centring(W)
    raise ValueError(method)


def hubness_profile(pred, label, ref_pred, classes):
    """Diagnostics (the TV and true-share columns use labels; they describe the result, nothing consumes them)."""
    n, true = np.bincount(pred, minlength=classes), np.bincount(label, minlength=classes)
    ref = np.bincount(ref_pred, minlength=classes)
    hubs = np.argsort(-ref)[:max(1, classes // 100)]  # the 1% most-predicted classes under zero-shot
    return {'skew N1': skew(n), 'TV to true marginal': 0.5 * np.abs(n / n.sum() - true / true.sum()).sum(),
            'share in zero-shot top-1% classes %': n[hubs].sum() / n.sum() * 100,
            'true share of those classes %': true[hubs].sum() / true.sum() * 100}


def evaluate(sd, method, params, base_pred, seed):
    pred = run(build(method, sd.W, params, seed), sd.Z, sd.batches())
    ok, base_ok = pred == sd.label, base_pred == sd.label
    return pred, {'acc': stack.domain_acc(sd.df, pred), 'fixed': int((ok & ~base_ok).sum()), 'broken': int((~ok & base_ok).sum()),
                  'changed': int((pred != base_pred).sum()), 'n': len(pred),
                  **hubness_profile(pred, sd.label, base_pred, sd.classes)}


# ---------------------------------------------------------------------------- tuning
def tune():
    sd = StreamData(TUNE_DATASET, 0)
    base_pred = run(ZeroShot(sd.W), sd.Z, sd.batches())
    base = stack.domain_acc(sd.df, base_pred)
    rows = []
    for fam, grid in GRID.items():
        for p in grid:
            _, r = evaluate(sd, fam, p, base_pred, 0)
            rows.append({'family': fam, **p, 'acc': r['acc'], 'gain': r['acc'] - base, 'fixed': r['fixed'], 'broken': r['broken']})
            print(f"{fam:17s} {p} acc {r['acc']:.2f} gain {r['acc'] - base:+.2f} (fixed {r['fixed']:,}, broken {r['broken']:,})", flush=True)
    t = pd.DataFrame(rows)
    os.makedirs(TABLES, exist_ok=True)
    t.to_csv(os.path.join(TABLES, 'tuning.csv'), index=False)
    frozen = {}
    for fam, g in t.groupby('family'):
        key = [c for c in ('beta', 'tau') if c in g and g[c].notna().any()][0]
        best = g[g.gain >= g.gain.max() - 1e-12].sort_values(key).iloc[0]  # ties -> smaller beta / tau
        frozen[fam] = {c: (int(best[c]) if c in ('kappa',) else float(best[c])) for c in ('kappa', 'beta', 'tau') if pd.notna(best[c])}
        print(f"frozen {fam}: {frozen[fam]} (ImageNet-C seed 0 zero-shot {base:.2f} -> {best.acc:.2f}, {best.gain:+.2f})")
    frozen['permuted csls'] = dict(frozen['csls'])
    frozen['centring'] = {}
    json.dump(frozen, open(FROZEN, 'w'), indent=1)
    print('TUNE DONE', flush=True)


def frozen():
    return json.load(open(FROZEN))


# ---------------------------------------------------------------------------- held-out
METHODS = ['csls', 'permuted csls', 'inverted softmax', 'centring']


def zero_shot():
    fz, rows = frozen(), []
    for ds in DATASETS:
        for seed in range(5):
            sd = StreamData(ds, seed)
            base_pred = run(ZeroShot(sd.W), sd.Z, sd.batches())
            base = {'acc': stack.domain_acc(sd.df, base_pred), **hubness_profile(base_pred, sd.label, base_pred, sd.classes)}
            rows.append({'dataset': ds, 'seed': seed, 'method': 'zero-shot', 'gain': 0.0, 'fixed': 0, 'broken': 0,
                         'changed': 0, 'n': len(base_pred), **base})
            for m in METHODS:
                _, r = evaluate(sd, m, fz[m], base_pred, seed)
                rows.append({'dataset': ds, 'seed': seed, 'method': m, 'gain': r['acc'] - base['acc'], **r})
            print(f'{ds} seed {seed}: zero-shot {base["acc"]:.2f}; ' + ', '.join(
                f"{m} {rows[-4 + i]['gain']:+.2f}" for i, m in enumerate(METHODS)), flush=True)
            del sd
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TABLES, 'zero_shot_runs.csv'), index=False)
    summarise(t)


def summarise(t):
    fl = FLOOR
    rows = []
    for (ds, m), g in t[t.method != 'zero-shot'].groupby(['dataset', 'method'], sort=False):
        rows.append({'dataset': ds, 'method': m, 'zero-shot acc': t[(t.dataset == ds) & (t.method == 'zero-shot')].acc.mean(),
                     'gain mean': g.gain.mean(), 'gain std': g.gain.std(), 'seeds positive': int((g.gain > 0).sum()),
                     'floor': fl[ds], 'above floor': g.gain.mean() > fl[ds]})
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(TABLES, 'zero_shot_summary.csv'), index=False)
    pd.set_option('display.width', 220)
    print('\n' + s.round(2).to_string(index=False))
    csls = s[s.method == 'csls'].set_index('dataset')
    ctrl = s[s.method == 'permuted csls'].set_index('dataset')
    passes = csls['above floor']
    margin = csls['gain mean'] - ctrl['gain mean']
    print(f"\nPRIMARY (frozen CSLS): above the floor on {int(passes.sum())} of {len(passes)} datasets "
          f"({', '.join(passes.index[passes]) or 'none'}); needs >= 4")
    for ds in csls.index:
        print(f"  {ds:17s} gain {csls.at[ds, 'gain mean']:+.2f} vs control {ctrl.at[ds, 'gain mean']:+.2f}: "
              f"margin {margin[ds]:+.2f} ({'> floor' if margin[ds] > fl[ds] else 'not > floor'})")
    ok = int(passes.sum()) >= 4 and all(margin[d] > fl[d] for d in passes.index[passes])
    print(f"PRIMARY CLAIM: {'PASS' if ok else 'FAIL'}")
    print('ZERO-SHOT DONE', flush=True)


# ---------------------------------------------------------------------------- Ramen level
def ramen_level():
    fz, rows = frozen(), []
    for ds in RAMEN_FEATS:
        sd = StreamData(ds, 0, level='ramen')
        base_pred = run(ZeroShot(sd.W), sd.Z, sd.batches())  # replayed Ramen prediction (fp32)
        base = stack.domain_acc(sd.df, base_pred)
        df, P = stack.load(ds, 0, 'Ramen')
        pc = stack.domain_acc(df, stack.prior_correct(P, df.batch.values, lam=1.0))
        rows.append({'dataset': ds, 'method': 'Ramen (replayed)', 'acc': base, 'gain': 0.0})
        rows.append({'dataset': ds, 'method': 'Ramen + pc (lambda 1, uniform target; reference)', 'acc': pc, 'gain': pc - base})
        for m in METHODS:
            _, r = evaluate(sd, m, fz[m], base_pred, 0)
            rows.append({'dataset': ds, 'method': f'Ramen + {m}', 'acc': r['acc'], 'gain': r['acc'] - base,
                         'fixed': r['fixed'], 'broken': r['broken'], 'floor': FLOOR[ds]})
        print(ds, [(x['method'], round(x['gain'], 2)) for x in rows[-6:]], flush=True)
        del sd
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TABLES, 'ramen_runs.csv'), index=False)
    pd.set_option('display.width', 220)
    print('\n' + t.round(2).to_string(index=False))
    print('RAMEN DONE', flush=True)


if __name__ == '__main__':
    {'tune': tune, 'zero-shot': zero_shot, 'ramen': ramen_level}[sys.argv[1]]()
