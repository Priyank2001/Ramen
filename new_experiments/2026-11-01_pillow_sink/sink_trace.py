"""
Where a sink class's predictions come from: for every image Ramen predicts as the target class, its true class,
its zero-shot class (before adaptation) and what the true-label oracle predicts for it instead.

    trace = SinkTrace(runs, 'pillow')
    trace.counts()               # predicts / correct / wrong, under zero-shot, Ramen and the oracle
    trace.routes()               # Ramen's wrong target predictions by zero-shot class, and the oracle's fate of each
    trace.true_classes()         # ... by true class
    trace.by_domain()
    trace.seed_table()           # the same counts for Ramen on seeds 0-4 (cached plain runs)
    trace.pick('envelope', oracle='still target', n=2)   # example images, drawn at random

No GPU, no model: everything comes from SinkRuns.table.
"""
import os

import numpy as np
import pandas as pd

from sink_runs import count_pct  # also sets up the import paths

ORACLE_FATES = ('still target', 'correct', 'other wrong')


class SinkTrace:

    def __init__(self, runs, target, n_routes=3):
        self.runs = runs
        self.target_name = target
        self.target = runs.classes.index(target)
        t = runs.table
        self.t = t.assign(
            oracle_fate=np.select([t.oracle_pred == self.target, t.oracle_pred == t.label],
                                  ['still target', 'correct'], 'other wrong'))
        wrong = self.wrong_rows()
        top = self.t.loc[wrong, 'zs_class'].value_counts().index[:n_routes]
        self.route_classes = [str(c) for c in top]   # the zero-shot classes that feed the sink most

    # ------------------------------------------------------------------ selections
    def wrong_rows(self):
        """Rows Ramen wrongly predicts as the target."""
        t = self.t
        return np.flatnonzero(((t.ramen_pred == self.target) & (t.label != self.target)).to_numpy())

    def query_groups(self, contrast_class=None):
        """
        The images whose cache retrieval the notebook summarises:
          Ramen's wrong target predictions, split by whether the oracle cache still predicts the target;
          contrast: zero-shot wrongly says `contrast_class` (default: the largest route) and Ramen fixes it;
          contrast: true target images that zero-shot and Ramen both get right.
        """
        t, c = self.t, self.target
        contrast_class = contrast_class or self.route_classes[0]
        wrong = self.wrong_rows()
        still = (t.oracle_fate.to_numpy()[wrong] == 'still target')
        name = self.target_name
        return {
            f'Ramen wrong {name}, oracle still {name}': wrong[still],
            f'Ramen wrong {name}, oracle not {name}': wrong[~still],
            f'zero-shot {contrast_class} → Ramen correct': np.flatnonzero(
                ((t.zs_class.astype(str) == contrast_class) & (t.zs_pred != t.label) &
                 (t.ramen_pred == t.label)).to_numpy()),
            f'true {name}, zero-shot and Ramen right': np.flatnonzero(
                ((t.label == c) & (t.zs_pred == c) & (t.ramen_pred == c)).to_numpy()),
        }

    def route_of(self, rows):
        zs = self.t.zs_class.to_numpy()[rows].astype(str)
        return np.where(np.isin(zs, self.route_classes), zs, 'other')

    # ------------------------------------------------------------------ tables
    def counts(self):
        """How often each model predicts the target, and how often that is right."""
        t, c = self.t, self.target
        out = {}
        for name, col in [('zero-shot', 'zs_pred'), ('Ramen', 'ramen_pred'), ('oracle cache', 'oracle_pred')]:
            pred = t[col].to_numpy() == c
            right = pred & (t.label.to_numpy() == c)
            out[name] = {f'predicts {self.target_name}': int(pred.sum()), 'correct': int(right.sum()),
                         'wrong': int(pred.sum() - right.sum()),
                         f'recall of true {self.target_name} %': right.sum() / (t.label == c).sum() * 100}
        df = pd.DataFrame(out).T
        df.attrs['true images'] = int((t.label == c).sum())
        return df.round(2).astype({k: int for k in df.columns if not k.endswith('%')})

    def routes(self):
        """Ramen's wrong target predictions grouped by zero-shot class, split by what the oracle predicts."""
        rows = self.wrong_rows()
        d = pd.DataFrame({'zero-shot class': self.route_of(rows), 'oracle': self.t.oracle_fate.to_numpy()[rows]})
        out = pd.crosstab(d['zero-shot class'], d['oracle']).reindex(columns=ORACLE_FATES, fill_value=0)
        out = out.reindex(self.route_classes + ['other'])
        out.insert(0, 'images', out.sum(1))
        out.insert(1, '% of wrong', (out['images'] / len(rows) * 100).round(1))
        out.loc['all'] = out.sum()
        out.loc['all', '% of wrong'] = 100.0
        out.index = [f'{r} → {self.target_name}' if r not in ('all', 'other') else r for r in out.index]
        return out.astype({c: int for c in out.columns if c != '% of wrong'})

    def true_classes(self, k=12):
        rows = self.wrong_rows()
        s = self.t.true_class.to_numpy()[rows].astype(str)
        v = pd.Series(s).value_counts()
        out = pd.DataFrame({'images': v.head(k)})
        out['% of wrong'] = (out['images'] / len(rows) * 100).round(1)
        out.loc[f'other ({len(v) - k} classes)'] = [int(v.iloc[k:].sum()), round(v.iloc[k:].sum() / len(rows) * 100, 1)]
        return out.astype({'images': int})

    def by_domain(self):
        """Per domain: images, zero-shot / Ramen / oracle wrong target predictions."""
        t, c = self.t, self.target
        g = t.assign(zs=(t.zs_pred == c) & (t.label != c), ramen=(t.ramen_pred == c) & (t.label != c),
                     oracle=(t.oracle_pred == c) & (t.label != c)).groupby('domain', observed=True)
        out = pd.DataFrame({'images': g.size(), f'zero-shot wrong {self.target_name}': g.zs.sum(),
                            f'Ramen wrong {self.target_name}': g.ramen.sum(),
                            f'oracle wrong {self.target_name}': g.oracle.sum()})
        out.loc['all'] = out.sum()
        return out.astype(int)

    def oracle_fate(self):
        rows = self.wrong_rows()
        v = self.t.oracle_fate.iloc[rows].value_counts().reindex(ORACLE_FATES, fill_value=0)
        return pd.Series({f: count_pct(int(n), len(rows)) for f, n in v.items()}, name='oracle prediction')

    def oracle_returns_to_zero_shot(self):
        """Of Ramen's wrong targets the oracle moves away from the target: how many go back to their zero-shot class."""
        rows = self.wrong_rows()
        t = self.t.iloc[rows]
        moved = t[t.oracle_pred != self.target]
        back = int((moved.oracle_pred == moved.zs_pred).sum())
        return pd.Series({'moved away from the target by the oracle': f'{len(moved):,} of {len(rows):,}',
                          'back to their zero-shot class': count_pct(back, len(moved))})

    def same_as_zero_shot(self):
        """Per domain: % of images where Ramen / the oracle keep the zero-shot prediction (how much each adapts)."""
        t = self.t
        g = t.assign(ramen=t.ramen_pred == t.zs_pred, oracle=t.oracle_pred == t.zs_pred).groupby('domain',
                                                                                                 observed=True)
        out = pd.DataFrame({'images': g.size(), 'Ramen keeps zero-shot %': g.ramen.mean() * 100,
                            'oracle keeps zero-shot %': g.oracle.mean() * 100})
        out.loc['all'] = [len(t), (t.ramen_pred == t.zs_pred).mean() * 100, (t.oracle_pred == t.zs_pred).mean() * 100]
        return out.round(1).astype({'images': int})

    def oracle_moves_to(self, k=8):
        """The classes the oracle predicts instead, for Ramen's wrong targets it does not leave at the target."""
        rows = self.wrong_rows()
        t = self.t.iloc[rows]
        v = t.loc[t.oracle_fate == 'other wrong', 'oracle_class'].astype(str).value_counts().head(k)
        return v.rename('images')

    def seed_table(self, seeds=(0, 1, 2, 3, 4)):
        """Ramen's target predictions on each seed (cached plain Ramen runs); a seed reorders the stream."""
        exp, c = self.runs.exp, self.target
        rows = {}
        for s in seeds:
            path = exp._cache_path('Ramen').replace(f'_s{self.runs.seed}_', f'_s{s}_')
            if not os.path.exists(path):
                continue
            d = pd.read_pickle(path)
            pred, y = d.pred.to_numpy() == c, d.label.to_numpy() == c
            rows[f'seed {s}'] = {f'Ramen predicts {self.target_name}': int(pred.sum()),
                                 'correct': int((pred & y).sum()), 'wrong': int((pred & ~y).sum())}
        out = pd.DataFrame(rows).T
        out.loc['mean ± std'] = [f'{out[k].mean():,.0f} ± {out[k].std():,.0f}' for k in out.columns]
        return out

    # ------------------------------------------------------------------ examples
    def pick(self, zs_class, ramen='wrong target', oracle=None, n=2, min_position=None, seed=0):
        """
        n images drawn at random (fixed seed, not hand-picked), late enough for the cache to be full (default:
        after 20% of the stream).
          zs_class  zero-shot class of the image
          ramen     'wrong target' (Ramen wrongly says target), 'correct' (Ramen right), 'right target'
          oracle    one of ORACLE_FATES, or None for any
        """
        t = self.t
        min_position = int(0.2 * len(t)) if min_position is None else min_position
        m = (t.zs_class.astype(str) == zs_class) & (t.position >= min_position)
        if ramen == 'wrong target':
            m &= (t.ramen_pred == self.target) & (t.label != self.target)
        elif ramen == 'right target':
            m &= (t.ramen_pred == self.target) & (t.label == self.target)
        elif ramen == 'correct':
            m &= (t.ramen_pred == t.label) & (t.zs_pred != t.label)
        else:
            raise ValueError(ramen)
        if oracle is not None:
            m &= t.oracle_fate == oracle
        pool = np.flatnonzero(m.to_numpy())
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(pool, min(n, len(pool)), replace=False)) if len(pool) else pool
