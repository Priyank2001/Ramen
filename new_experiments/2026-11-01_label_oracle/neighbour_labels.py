"""
How informative are the retrieved entries' TRUE labels? With label values, each retrieved entry pushes toward its
own true class, so an update can only point the right way as often as the retrieved neighbours vote for the
query's class. Replays Ramen's cache (zero-shot filing; 2026-11-01_pillow_sink, read-only import) for random
images after the first 20% of the stream, and for each image takes the weighted vote of the retrieved entries'
true labels:

  per-queue vote   Ramen's active set (10 nearest of every class queue), weights e^(-H) e^(-beta d)
  global-10 vote   the 10 nearest entries of the whole cache (exact: each is among its own queue's 10 nearest)

and compares it with what zero-shot, Ramen and the label-value runs predicted for the same images.

    nl = NeighbourLabels('DomainNet', seed=0, n=2000)
    nl.summary()       # vote accuracy vs run accuracy, per domain
    nl.crosstab('global_ce_k10', 'global-10 vote right')
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), '2026-11-01_pillow_sink'))  # read-only import

from cache_replay import CacheReplayRun, ZeroShotFiling  # noqa: E402
from sink_runs import SinkRuns  # noqa: E402

RUNS = ('labeloracle_zs_ce', 'global_ce_k10', 'global_ce_k100')


class NeighbourLabels:

    def __init__(self, dataset='DomainNet', seed=0, n=2000, min_position=0.2, use_cache=True, runs=RUNS):
        self.runs = SinkRuns(dataset, seed, device='cpu')
        path = self.runs.exp._cache_path(f'labeloracle_neighbour_labels_n{n}')
        if use_cache and os.path.exists(path):
            self.table = pd.read_pickle(path)
            return
        t = self.runs.table
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(np.arange(int(min_position * len(t)), len(t)), n, replace=False))
        result = CacheReplayRun(self.runs, [ZeroShotFiling()]).run(rows, keep_rows=rows)
        out = []
        for r in rows:
            e = result.retrieved[(int(r), 'Ramen cache')]
            q = t.iloc[r]
            vote = e.groupby('true')['weight'].sum()
            top10 = e.nsmallest(10, 'dist')
            vote10 = top10.groupby('true')['weight'].sum()
            out.append({'row': int(r), 'domain': q.domain, 'label': int(q.label),
                        'zero-shot right': q.zs_pred == q.label, 'Ramen right': q.ramen_pred == q.label,
                        'per-queue vote right': vote.idxmax() == q.label,
                        'per-queue share on own class %': vote.get(q.label, 0.0) * 100,
                        'global-10 vote right': vote10.idxmax() == q.label,
                        'global-10 share on own class %': vote10.get(q.label, 0.0) / vote10.sum() * 100,
                        'global-10 neighbours of own class': int((top10['true'] == q.label).sum())})
        self.table = pd.DataFrame(out)
        exp = self.runs.exp
        for name in runs:
            df = pd.read_pickle(exp._cache_path(name)).sort_values('position')
            self.table[f'{name} right'] = df['correct'].to_numpy()[self.table['row'].to_numpy()]
        self.table.to_pickle(path)

    def summary(self):
        cols = [c for c in self.table.columns if c.endswith('right') or c.endswith('%') or c.endswith('class')]
        t = self.table.assign(**{c: self.table[c].astype(float) * (100 if c.endswith('right') else 1) for c in cols})
        out = t.groupby('domain', observed=True)[cols].mean()
        out.loc['all (pooled)'] = t[cols].mean()
        out.insert(0, 'images', list(t.groupby('domain', observed=True).size()) + [len(t)])
        return out.round(1)

    def crosstab(self, run, vote='global-10 vote right'):
        """How the label-value run fares when the neighbours' vote is right / wrong, split by zero-shot."""
        t = self.table
        g = t.groupby([vote, 'zero-shot right'])[f'{run} right'].agg(['size', 'mean'])
        g['mean'] = (g['mean'] * 100).round(1)
        return g.rename(columns={'size': 'images', 'mean': f'{run} right %'})
