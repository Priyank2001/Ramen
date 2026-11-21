"""
Why does filing Ramen's cache by the TRUE label (ORACLE) lower accuracy? Offline test of the "containment" hypothesis.

Hypothesis (pre-registered 2026-11-18, before running this script):
  Ramen's stored value is an entropy gradient, which pushes toward the stored image's zero-shot class. Zero-shot
  filing puts every image CLIP confuses the same way (e.g. quickdraw drawings it calls 'envelope') into ONE queue,
  and Ramen averages queues equally, so such a confusion is contained to ~1/|C| of the update. True-label filing
  spreads the same images over all their true classes, so nearly every queue's top-k hits nearby images that
  share the query's confusion, and the update re-imports the query's own self-confirming (zero-shot-sharpening)
  direction.
  C1  On zero-shot-WRONG queries, the share of update weight on entries whose zero-shot class equals the query's
      zero-shot class is at least 2x larger under true-label filing than under zero-shot filing.
  C2  The effect is largest where zero-shot predictions are concentrated on few classes (effective number of
      predicted classes << number of true classes), i.e. quickdraw on DomainNet.
  C3  Per query, the oracle keeps the zero-shot prediction more often when that share is higher.

No GPU, no gradients: the replay (2026-11-01_pillow_sink/cache_replay.py) is exact up to fp16 arithmetic, since
Ramen's cache never depends on its adapted outputs. The adapted predictions come from the two recorded runs.

    CUDA_VISIBLE_DEVICES= python containment.py DomainNet 0 --per-domain 1000
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '2026-11-01_pillow_sink'))
from cache_replay import CacheReplayRun, TrueLabelFiling, ZeroShotFiling  # noqa: E402
from sink_runs import SinkRuns  # noqa: E402

SAME_ZS = "entries zero-shot-predicted as query's zero-shot class"
METRICS = {
    SAME_ZS: 'same zero-shot class as query',
    "entries truly query's class": 'truly query class',
    'entries that are zero-shot errors': 'zero-shot errors',
    "entries from query's domain": "query's domain",
    'from zero-shot-class queue': "query's zero-shot queue",
    'from true-class queue': "query's true-class queue",
    'self': 'own entry',
}


def effective_classes(labels):
    """exp(entropy) of a label distribution: how many classes it effectively spreads over."""
    p = np.bincount(labels).astype(float)
    p = p[p > 0] / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def sample_rows(table, per_domain, seed):
    rng = np.random.default_rng(seed)
    rows = [rng.choice(np.flatnonzero(table.domain_idx.to_numpy() == d), size=per_domain, replace=False)
            for d in sorted(table.domain_idx.unique())]
    return np.sort(np.concatenate(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset'), ap.add_argument('seed', type=int)
    ap.add_argument('--per-domain', type=int, default=1000)
    a = ap.parse_args()

    runs = SinkRuns(a.dataset, a.seed, device='cpu')
    t = runs.table
    out_dir = os.path.join(HERE, 'results')
    os.makedirs(out_dir, exist_ok=True)

    # data: how concentrated zero-shot predictions are, per domain (C2)
    conc = []
    for dom, g in t.groupby('domain', observed=True):
        conc.append({'domain': dom, 'images': len(g), 'zs wrong': int((g.zs_pred != g.label).sum()),
                     'eff. true classes': effective_classes(g.label.to_numpy()),
                     'eff. zs-predicted classes': effective_classes(g.zs_pred.to_numpy()),
                     'eff. zs classes of zs errors': effective_classes(g.zs_pred[g.zs_pred != g.label].to_numpy())})
    conc = pd.DataFrame(conc).set_index('domain')
    print('\n== zero-shot prediction concentration ==\n', conc.round(1).to_string(), flush=True)

    rows = sample_rows(t, a.per_domain, a.seed)
    res = CacheReplayRun(runs, [ZeroShotFiling(), TrueLabelFiling()]).run(rows)
    s = res.summary.reset_index()
    s = s.join(t[['domain', 'label', 'zs_pred', 'ramen_pred', 'oracle_pred']], on='row')
    s['zs right'] = s.zs_pred == s.label
    s['kept zs'] = np.where(s.cache == 'Ramen cache', s.ramen_pred, s.oracle_pred) == s.zs_pred
    s['correct'] = np.where(s.cache == 'Ramen cache', s.ramen_pred, s.oracle_pred) == s.label
    with open(os.path.join(out_dir, f'{a.dataset}_s{a.seed}_containment.pkl'), 'wb') as f:
        pickle.dump({'summary': s, 'concentration': conc, 'rows': rows}, f)

    cols = list(METRICS)
    pd.set_option('display.width', 250)
    print(f'\n== update-weight shares (%), mean over {len(rows):,} sampled queries ==')
    print((s.groupby(['zs right', 'cache'])[cols].mean() * 100).rename(columns=METRICS).round(1).to_string())
    print('\n== zero-shot-WRONG queries, by domain: share on same-zero-shot-class entries (%) ==')
    w = s[~s['zs right']]
    print((w.pivot_table(index='domain', columns='cache', values=SAME_ZS, aggfunc='mean', observed=True) * 100)
          .round(1).to_string())
    print('\n== outcomes on the sample: kept zero-shot prediction / correct (%) ==')
    print((s.groupby(['domain', 'cache'], observed=True)[['kept zs', 'correct']].mean() * 100).round(1).unstack()
          .to_string())
    print('\n== C3: oracle cache, zs-wrong queries: kept zs rate by quintile of same-zero-shot-class share ==')
    o = w[w.cache == 'oracle cache'].copy()
    o['quintile'] = pd.qcut(o[SAME_ZS], 5, labels=False, duplicates='drop')
    print(o.groupby('quintile').agg(n=('row', 'size'), share=(SAME_ZS, 'mean'), kept_zs=('kept zs', 'mean'))
          .assign(share=lambda d: (d.share * 100).round(1), kept_zs=lambda d: (d.kept_zs * 100).round(1)).to_string())


if __name__ == '__main__':
    main()
