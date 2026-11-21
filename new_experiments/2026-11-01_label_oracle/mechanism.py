"""
Why true-label filing hurts even with label values: what each filing rule retrieves for a random sample of images.

Retrieval does not depend on the values (weights are e^(-H) e^(-beta d) from zero-shot entropies and features), so
the cache replay of 2026-11-01_pillow_sink (read-only import) gives exactly the entries the label-value runs
retrieved. With label values, an entry pushes toward ITS true class, so the share of the update weight on
entries of the query's own true class is the share that pushes the right way.

    agree = LabelAgreement('DomainNet', seed=0, n=1000)
    agree.per_query      # one row per (query, filing): share on own class, top class, effective classes, ...
    agree.by_domain()    # means per domain and filing
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), '2026-11-01_pillow_sink'))  # read-only import

from cache_replay import CacheReplayRun, TrueLabelFiling, ZeroShotFiling  # noqa: E402
from sink_runs import SinkRuns  # noqa: E402

FILING_NAMES = {'Ramen cache': 'zero-shot filing', 'oracle cache': 'true-label filing'}


class LabelAgreement:

    def __init__(self, dataset='DomainNet', seed=0, n=1000, min_position=0.2, use_cache=True):
        self.runs = SinkRuns(dataset, seed, device='cpu')
        path = self.runs.exp._cache_path(f'labeloracle_agreement_n{n}')
        if use_cache and os.path.exists(path):
            self.per_query = pd.read_pickle(path)
            return
        t = self.runs.table
        rng = np.random.default_rng(seed)
        # late enough for every queue to be full, as for most of the stream
        rows = np.sort(rng.choice(np.arange(int(min_position * len(t)), len(t)), n, replace=False))
        result = CacheReplayRun(self.runs, [ZeroShotFiling(), TrueLabelFiling()]).run(rows, keep_rows=rows)
        self.per_query = pd.DataFrame([self._row(r, cache, e) for (r, cache), e in result.retrieved.items()])
        self.per_query.to_pickle(path)

    def _row(self, row, cache, e):
        t = self.runs.table.iloc[row]
        by_class = e.groupby('true')['weight'].sum()
        p = by_class.to_numpy()
        return {'row': row, 'filing': FILING_NAMES[cache], 'domain': t.domain,
                'share on own true class %': by_class.get(t.label, 0.0) * 100,
                'share on largest single class %': p.max() * 100,
                'own class is the largest': by_class.idxmax() == t.label,
                'effective number of classes': float(np.exp(-(p * np.log(p)).sum())),
                'share from own domain %': e.loc[e.domain_idx == t.domain_idx, 'weight'].sum() * 100}

    def by_domain(self):
        cols = ['share on own true class %', 'share on largest single class %', 'own class is the largest',
                'effective number of classes', 'share from own domain %']
        g = self.per_query.assign(**{'own class is the largest': self.per_query['own class is the largest'] * 100})
        out = g.groupby(['domain', 'filing'], observed=True)[cols].mean()
        allr = g.groupby('filing')[cols].mean()
        allr.index = pd.MultiIndex.from_product([['all (pooled)'], allr.index], names=['domain', 'filing'])
        out = pd.concat([out, allr]).rename(columns={'own class is the largest': 'own class is the largest %'})
        n = self.per_query.groupby(['domain', 'filing'], observed=True).size()
        out.insert(0, 'queries', n.reindex(out.index).fillna(len(self.per_query) // 2).astype(int))
        return out.round(1)
