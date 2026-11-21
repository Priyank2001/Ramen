"""
Is DomainNet's global-retrieval gain a near-copy leak? On ImageNet-C the global top-k gain came from other copies
of the same photo (other corruptions) carrying its label; DomainNet has no such groups but may hold near-identical
images. Replays Ramen's cache (zero-shot filing; 2026-11-01_pillow_sink, read-only import) for the same random
images as NeighbourLabels and, per image, records the distance to its nearest cache entry, whether that entry is
the image itself (no leak: its value is still the entropy gradient when it is predicted), and how many OTHER
entries lie within `radius` (a near-copy whose label value would leak the answer).

    nc = NearCopies('DomainNet', seed=0, n=2000)
    nc.summary()                                      # how many images have their own entry / another near-copy
    nc.gain_split('global_ce_k10', 'labeloracle_zs_ce')   # the gain with and without the near-copy images
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), '2026-11-01_pillow_sink'))  # read-only import

from cache_replay import CacheReplayRun, ZeroShotFiling  # noqa: E402
from sink_runs import SinkRuns  # noqa: E402

RUNS = ('Ramen', 'labeloracle_zs_ce', 'global_ce_k10', 'global_ce_k100', 'global_ce_k1000')


class NearCopies:

    def __init__(self, dataset='DomainNet', seed=0, n=2000, radius=0.1, min_position=0.2, use_cache=True,
                 runs=RUNS):
        self.radius = radius
        self.runs = SinkRuns(dataset, seed, device='cpu')
        path = self.runs.exp._cache_path(f'labeloracle_near_copies_n{n}_r{radius}')
        if use_cache and os.path.exists(path):
            self.table = pd.read_pickle(path)
            return
        t = self.runs.table
        rng = np.random.default_rng(seed)  # the same rows as NeighbourLabels(dataset, seed, n)
        rows = np.sort(rng.choice(np.arange(int(min_position * len(t)), len(t)), n, replace=False))
        result = CacheReplayRun(self.runs, [ZeroShotFiling()]).run(rows, keep_rows=rows)
        out = []
        for r in rows:
            e = result.retrieved[(int(r), 'Ramen cache')]
            close = e[e['dist'] < radius]
            out.append({'row': int(r), 'domain': t.iloc[r].domain,
                        'nearest distance': float(e['dist'].min()),
                        'own entry retrieved': bool((close['id'] == r).any()),
                        'other near-copies': int((close['id'] != r).sum()),
                        'near-copies of same class': int(((close['id'] != r) & (close['true'] == t.iloc[r].label)).sum())})
        self.table = pd.DataFrame(out)
        exp = self.runs.exp
        for name in runs:
            df = pd.read_pickle(exp._cache_path(name)).sort_values('position')
            self.table[f'{name} right'] = df['correct'].to_numpy()[self.table['row'].to_numpy()]
        self.table.to_pickle(path)

    def summary(self):
        t = self.table
        g = t.assign(**{'has another near-copy': t['other near-copies'] > 0})
        out = g.groupby('domain', observed=True).agg(
            images=('row', 'size'), **{'own entry retrieved': ('own entry retrieved', 'sum'),
                                       'has another near-copy': ('has another near-copy', 'sum'),
                                       'median nearest distance': ('nearest distance', 'median')})
        out.loc['all (pooled)'] = [len(g), g['own entry retrieved'].sum(), g['has another near-copy'].sum(),
                                   g['nearest distance'].median()]
        return out.astype({'images': int, 'own entry retrieved': int, 'has another near-copy': int}).round(3)

    def gain_split(self, run, reference):
        """Accuracy gain of `run` over `reference` (pooled, points) on all sampled images, and without near-copies."""
        t = self.table
        parts = {'all sampled images': t, 'with another near-copy': t[t['other near-copies'] > 0],
                 'without near-copies': t[t['other near-copies'] == 0]}
        return pd.DataFrame({part: {'images': len(s), f'{run} − {reference} (pts)':
                                    round((s[f'{run} right'].mean() - s[f'{reference} right'].mean()) * 100, 2)}
                             for part, s in parts.items()}).T.astype({'images': int})
