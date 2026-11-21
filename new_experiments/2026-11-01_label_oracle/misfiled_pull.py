"""
Does a test image get pulled toward class X by images of ITS OWN class that zero-shot misfiled under X?
(e.g. a cat query retrieving cats filed in the dog queue, whose stored entropy gradients sharpen "dog")

Replays Ramen's cache (zero-shot filing; 2026-11-01_pillow_sink, read-only import) for two groups of images
after the first 20% of the stream, DomainNet seed 0:
  broken   zero-shot right, Ramen wrong          -> X = the class Ramen moved it to
  control  zero-shot right, Ramen right          -> X = Ramen's runner-up class (the nearest competitor)
and measures, per image, shares of its update weight:
  own class misfiled under X   entries truly the query's class that zero-shot predicted X
  all entries predicted X      every entry whose stored gradient sharpens X
  own class, any misfiling     entries truly the query's class with a wrong zero-shot prediction
  self                         the query's own entry

    pull = MisfiledPull('DomainNet', seed=0, n=500)
    pull.summary()
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), '2026-11-01_pillow_sink'))  # read-only import

from cache_replay import CacheReplayRun, ZeroShotFiling  # noqa: E402
from sink_runs import SinkRuns, count_pct  # noqa: E402


class MisfiledPull:

    def __init__(self, dataset='DomainNet', seed=0, n=500, min_position=0.2, use_cache=True):
        self.runs = SinkRuns(dataset, seed, device='cpu')
        path = self.runs.exp._cache_path(f'labeloracle_misfiled_pull_n{n}')
        if use_cache and os.path.exists(path):
            self.per_query = pd.read_pickle(path)
            return
        t = self.runs.table
        late = t.position.to_numpy() >= int(min_position * len(t))
        zs_ok, ramen_ok = (t.zs_pred == t.label).to_numpy(), (t.ramen_pred == t.label).to_numpy()
        rng = np.random.default_rng(seed)
        groups = {'broken': np.flatnonzero(late & zs_ok & ~ramen_ok), 'control': np.flatnonzero(late & zs_ok & ramen_ok)}
        self.pool = {g: len(r) for g, r in groups.items()}
        picked = {g: np.sort(rng.choice(r, n, replace=False)) for g, r in groups.items()}
        rows = np.sort(np.concatenate(list(picked.values())))
        result = CacheReplayRun(self.runs, [ZeroShotFiling()]).run(rows, keep_rows=rows)
        probs = self.runs.probs('ramen', rows)
        runner_up = dict(zip(rows, np.argsort(probs, axis=1)[:, -2]))
        out = []
        for g, rs in picked.items():
            for r in rs:
                e = result.retrieved[(int(r), 'Ramen cache')]
                q = t.iloc[r]
                x = int(q.ramen_pred) if g == 'broken' else int(runner_up[r])
                own, mis = e['true'] == q.label, e['zs_pred'] != e['true']
                mean_w = e['weight'].mean()
                out.append({'group': g, 'row': int(r), 'domain': q.domain, 'X': x,
                            'own class misfiled under X %': e.loc[own & (e.zs_pred == x), 'weight'].sum() * 100,
                            'all entries predicted X %': e.loc[e.zs_pred == x, 'weight'].sum() * 100,
                            'own class, any misfiling %': e.loc[own & mis, 'weight'].sum() * 100,
                            'own class, correctly filed %': e.loc[own & ~mis, 'weight'].sum() * 100,
                            'self %': e.loc[e['self'], 'weight'].sum() * 100,
                            'weight of own misfiled entry vs average entry': (e.loc[own & mis, 'weight'].mean() / mean_w
                                                                              if (own & mis).any() else np.nan)})
        self.per_query = pd.DataFrame(out)
        self.per_query.attrs['pool'] = self.pool
        self.per_query.to_pickle(path)

    def summary(self):
        cols = [c for c in self.per_query.columns if c.endswith('%') or c.startswith('weight of')]
        g = self.per_query.groupby('group')[cols]
        return pd.concat({'mean': g.mean(), 'median': g.median()}, axis=1).T.round(2)

    def any_misfiled_under_x(self):
        """Share of images that retrieve at least one entry of their own class misfiled under X."""
        d = self.per_query
        return d.groupby('group').apply(lambda x: count_pct(int((x['own class misfiled under X %'] > 0).sum()), len(x)))
