"""
The sink case study for one dataset, seed and target class: the recorded runs, the trace of where the target's
predictions come from, the example images, and one replay of both caches over the stream (cached on disk).

    study = SinkCaseStudy('DomainNet', seed=0, target='pillow')
    study.runs.sanity_check(); study.validate_replay()
    study.trace.routes()
    study.result.summary                     # per query image and cache: where the update weight comes from
    study.card.show_caches(study.examples.index[0])

Another sink class is one argument: SinkCaseStudy('DomainNet', 0, target='teapot').
"""
import os
import pickle

import numpy as np
import pandas as pd

from cache_replay import CacheReplayRun, TrueLabelFiling, ZeroShotFiling
from example_card import ExampleCard
from sink_runs import SinkRuns
from sink_trace import SinkTrace

# (zero-shot class, Ramen outcome, oracle fate, how many): 'route k' = the k-th largest route into the target
EXAMPLES = [
    ('route 0', 'wrong target', 'still target', 2),
    ('route 0', 'wrong target', 'other wrong', 1),
    ('route 1', 'wrong target', 'still target', 1),
    ('route 1', 'wrong target', 'other wrong', 1),
    ('route 2', 'wrong target', None, 1),
    ('route 0', 'correct', None, 1),          # contrast: same zero-shot error, Ramen fixes it
    ('target', 'right target', None, 1),      # contrast: a true target image that stays right
]
N_CHECKPOINTS = 60


class SinkCaseStudy:

    def __init__(self, dataset='DomainNet', seed=0, target='pillow', examples=EXAMPLES, use_cache=True):
        self.dataset, self.seed, self.target_name = dataset, seed, target
        self.runs = SinkRuns(dataset, seed)
        self.trace = SinkTrace(self.runs, target)
        self.target = self.trace.target
        self.groups = self.trace.query_groups()
        self.examples = self._pick(examples)
        self.rules = [ZeroShotFiling(), TrueLabelFiling()]
        self.replay = CacheReplayRun(self.runs, self.rules)
        self.result = self._replay(use_cache)
        self.card = ExampleCard(self.runs, self.result, target)

    # ------------------------------------------------------------------ examples
    def _resolve(self, name):
        if name == 'target':
            return self.target_name
        if name.startswith('route '):
            return self.trace.route_classes[int(name.split()[1])]
        return name

    def _pick(self, plan):
        rows = []
        for i, (zs, ramen, oracle, n) in enumerate(plan):
            zs_class = self._resolve(zs)
            for r in self.trace.pick(zs_class, ramen=ramen, oracle=oracle, n=n, seed=self.seed + i):
                rows.append({'row': int(r), 'zero-shot class': zs_class, 'Ramen': ramen,
                             'oracle': oracle or 'any'})
        ex = pd.DataFrame(rows).set_index('row')
        t = self.runs.table.loc[ex.index]
        ex['true class'] = t.true_class.astype(str)
        ex['domain'] = t.domain.astype(str)
        ex['stream position %'] = (ex.index / len(self.runs.table) * 100).round(1)
        return ex

    # ------------------------------------------------------------------ replay
    @property
    def cache_path(self):
        return self.runs.exp._cache_path(f'pillow_sink_replay_{self.target_name}')

    def _replay(self, use_cache):
        if use_cache and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                res = pickle.load(f)
            if set(res.retrieved) >= {(int(r), c.name) for r in self.examples.index for c in self.rules}:
                return res
        queries = np.concatenate(list(self.groups.values()))
        rows = self.examples.index.to_numpy()
        t = self.runs.table
        queue_classes = {int(r): sorted({self.target, int(t.zs_pred.iat[r]), int(t.label.iat[r])}) for r in rows}
        checkpoints = np.linspace(self.runs.exp.batch_size - 1, len(t) - 1, N_CHECKPOINTS).astype(int)
        res = self.replay.run(queries, keep_rows=rows, queue_classes=queue_classes, checkpoints=checkpoints,
                              watch_classes=[self.target], target=self.target)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(res, f)
        return res

    def validate_replay(self, n=200):
        """The zero-shot-filed replay vs the logged retrieval of a RamenActive 'all' run (= Ramen) on this stream."""
        exp = self.runs.exp
        stats = pd.read_pickle(exp._cache_path('active_all')[:-len('.pkl')] + '_stats.pkl')
        logged = pd.read_pickle(exp._cache_path('active_all'))
        assert (logged.sample_idx.to_numpy() == self.runs.table.sample_idx.to_numpy()).all(), 'different streams'
        return self.replay.validate(stats, logged.pred.to_numpy(), n=n)

    # ------------------------------------------------------------------ tables
    def group_summary(self):
        """Mean share (%) of the update weight per source, for each query group and cache."""
        s = self.result.summary
        out = {}
        for g, rows in self.groups.items():
            for cache in s.index.get_level_values('cache').unique():
                out[(g, cache)] = s.xs(cache, level='cache').loc[rows].mean()
        df = pd.DataFrame(out).T
        pct = [c for c in df.columns if c not in ('non-empty queues', 'top-10 entries truly target')]
        df[pct] = df[pct] * 100
        df.columns = [c.replace('target', self.target_name) for c in df.columns]
        df.insert(0, 'images', [len(self.groups[g]) for g, _ in df.index])
        return df.round(1)

    def show_example(self, i, queue='target'):
        """Notebook view of example i: heading, logits/probabilities table and figure, cache figure and table."""
        import matplotlib.pyplot as plt
        from IPython.display import Markdown, display
        if i >= len(self.examples):
            print(f'example {i}: no image matched (only {len(self.examples)} picked)')
            return
        row = int(self.examples.index[i])
        e, r, C = self.examples.iloc[i], self.runs.table.iloc[row], self.runs.classes
        oracle = '' if e.oracle == 'any' else f', oracle cache: {e.oracle.replace("target", self.target_name)}'
        ramen = e.Ramen.replace('target', self.target_name)
        display(Markdown(f'### Example {i + 1}: zero-shot *{e["zero-shot class"]}* → Ramen {ramen}{oracle}\n'
                         f'Image #{row:,} of the stream ({e["stream position %"]}% in), domain **{r.domain}**, '
                         f'true class **{C[r.label]}**.'))
        fmt = {c: '{:.2f}' for c in self.card.prediction_table(row).columns if c.startswith('logit')}
        fmt.update({c: '{:.3f}' for c in self.card.prediction_table(row).columns if c.startswith('prob')})
        display(self.card.prediction_table(row).style.format(fmt))
        self.card.show_prediction(row)
        plt.show()
        self.card.show_caches(row, queue=queue)
        plt.show()
        display(self.card.cache_table(row))

    def domain_col(self, name):
        return f'from domain {self.runs.environments.index(name)} %'
