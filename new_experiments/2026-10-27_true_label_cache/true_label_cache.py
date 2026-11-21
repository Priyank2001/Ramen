"""
True-label cache experiment: Ramen with its memory filed by the TRUE label (src/methods/RamenTrueLabel.py),
compared with Ramen and the zero-shot model on any dataset of the harness (new_experiments/ramen_analysis.py).

    from true_label_cache import TrueLabelCacheExperiment
    exp = TrueLabelCacheExperiment('DomainNet', seed=0)
    exp.run_all()                # cached runs load from new_experiments/results/, missing ones run on the GPU
    exp.summary_table()          # one row per run, domain-averaged accuracy, counts vs NoAdapt and vs Ramen

Extending:
  - a new dataset: add a line to DATASETS (the model and num_workers of its cached NoAdapt / Ramen runs)
  - a new variant: add a line to VARIANTS (run name -> method + config overrides); every table and plot picks it up
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # new_experiments/, for ramen_analysis

from ramen_analysis import make_experiment  # noqa: E402

# The settings of the cached NoAdapt / Ramen runs. num_workers changes the stream order (tqdm gotcha): keep it.
DATASETS = {
    'DomainNet': dict(model='clip_vitbase32', num_workers=16),
    'ImageNetC5K': dict(model='clip_vitbase16', num_workers=16),
    'ImageNetVariants': dict(model='clip_vitbase16', num_workers=16),
    'CIFAR10C': dict(model='clip_vitbase32', num_workers=4),
    'CIFAR100C': dict(model='clip_vitbase16', num_workers=4),
    'EuroSAT': dict(model='clip_vitbase16', num_workers=16),
}

# run name -> (method in src/methods, config overrides on cfg/<dataset>/<method>.yaml)
VARIANTS = {
    'NoAdapt': ('NoAdapt', None),
    'Ramen': ('Ramen', None),
    'true_label': ('RamenTrueLabel', {'filing': 'true_label'}),    # ORACLE: uses the true labels
    'zero_shot_filing': ('RamenTrueLabel', {'filing': 'zero_shot'}),  # must equal Ramen (sanity check)
}
DEFAULT_RUNS = ('NoAdapt', 'Ramen', 'true_label')  # zero_shot_filing is only run on the smoke subset

AVG_ROW = {'domain': 'domain avg', 'class': 'all (domain avg)'}  # the headline row of per-group tables
COLORS = {'NoAdapt': '#8a8984', 'Ramen': '#2a78d6', 'true_label': '#eb6834', 'zero_shot_filing': '#1baf7a'}


def count_pct(n, total):
    """'3,790 of 29,784 (12.7%)'"""
    return f'{n:,} of {total:,} ({n / total:.1%})'


def domain_avg(df):
    """Mean of the per-domain accuracies, in % (the headline convention)."""
    return df.groupby('domain_idx')['correct'].mean().mean() * 100


class  TrueLabelCacheExperiment:
    """One dataset and one seed: the runs in VARIANTS over the same mixed stream, plus tables and plots."""

    def __init__(self, dataset, seed=0, n_per_domain=None, results_dir=None, device=None):
        if dataset not in DATASETS:
            raise ValueError(f'Unknown dataset {dataset!r}; add it to DATASETS ({list(DATASETS)})')
        self.dataset, self.seed = dataset, seed
        kwargs = dict(DATASETS[dataset], tta_mode='mixed', batch_size=100, n_per_domain=n_per_domain, seed=seed,
                      device=device)
        if results_dir is not None:
            kwargs['results_dir'] = results_dir
        self.exp = make_experiment(dataset, **kwargs)
        self.loaded = False  # the model and images are loaded only when a run is not cached
        self._classes = None

    # ------------------------------------------------------------------ running
    @property
    def results(self):
        """run name -> per-sample DataFrame (batch, domain_idx, sample_idx, label, pred, ..., correct, domain)"""
        return self.exp.results

    @property
    def stats(self):
        """run name -> per-batch log of the method (RamenTrueLabel: zs_wrong, nonempty_queues)"""
        return self.exp.method_stats

    def is_cached(self, name):
        return os.path.exists(self.exp._cache_path(name))

    def run(self, name, use_cache=True):
        """Load `name` from the cache, or run it (loads CLIP and the images the first time)."""
        method, cfg = VARIANTS[name]
        if not (use_cache and self.is_cached(name)) and not self.loaded:
            self.exp.load()
            self.loaded = True
        return self.exp.run(method, name=name, cfg=cfg, use_cache=use_cache)

    def run_all(self, names=DEFAULT_RUNS, use_cache=True):
        for name in names:
            self.run(name, use_cache=use_cache)
        return self

    # ------------------------------------------------------------------ tables
    @property
    def domains(self):
        return list(next(iter(self.results.values()))['domain'].cat.categories)

    @property
    def classes(self):
        """Class names, in label order (reads the dataset's file list, not the images or CLIP)."""
        if self._classes is None:
            self._classes = list((self.exp if self.loaded else self.exp._build_datasets(None)).classes)
        return self._classes

    def group_of(self, df, by='domain'):
        """The group of each row of `df`: its domain, or its true class (for single-domain datasets)."""
        if by == 'domain':
            return df['domain']
        if by == 'class':
            return pd.Series(pd.Categorical.from_codes(df['label'].values, self.classes), index=df.index, name='class')
        raise ValueError(f"by must be 'domain' or 'class', not {by!r}")

    def aligned(self, names=None):
        """Predictions of several runs, one row per image, matched on (domain_idx, sample_idx)."""
        names = names or list(self.results)
        base = self.results[names[0]][['domain_idx', 'sample_idx', 'label', 'domain']]
        out = base.set_index(['domain_idx', 'sample_idx']).copy()
        for n in names:
            out[n] = self.results[n].set_index(['domain_idx', 'sample_idx'])['pred']
        return out.reset_index()

    def data_table(self):
        """Images and classes per domain, and how many the zero-shot model gets right (the denominators)."""
        df = self.results['NoAdapt']
        g = df.groupby('domain', observed=True)
        t = pd.DataFrame({'images': g.size(), 'classes': g['label'].nunique(),
                          'NoAdapt correct': g['correct'].sum()})
        t.loc['all'] = [len(df), df['label'].nunique(), df['correct'].sum()]
        t = t.astype(int)
        t['NoAdapt correct'] = [count_pct(c, n) for c, n in zip(t['NoAdapt correct'], t['images'])]
        return t

    def accuracy_table(self, names=None, by='domain'):
        """Accuracy (%) per domain (or per true class) for each run; last row = domain average (the headline)."""
        names = names or list(self.results)
        acc = pd.DataFrame({n: self.results[n].groupby(self.group_of(self.results[n], by), observed=True)['correct']
                            .mean() * 100 for n in names})
        acc.loc[AVG_ROW[by]] = [domain_avg(self.results[n]) for n in names]
        return acc.round(2)

    def change_table(self, name, baseline='NoAdapt', by='domain'):
        """Per domain (or class): accuracy of both runs, images fixed / broken by `name` vs `baseline`, net."""
        a = self.aligned([name, baseline])
        right, right_base = a[name] == a['label'], a[baseline] == a['label']
        group = self.group_of(a, by)
        t = pd.DataFrame({'images': group.value_counts(sort=False),
                          'fixed': (right & ~right_base).groupby(group, observed=True).sum(),
                          'broken': (~right & right_base).groupby(group, observed=True).sum()})
        t['net (images)'] = t['fixed'] - t['broken']
        t['share of net gain'] = (t['net (images)'] / t['net (images)'].sum()).map('{:.1%}'.format)
        cols = ['images', 'fixed', 'broken', 'net (images)']
        return self.accuracy_table([baseline, name], by).join(t).astype({c: 'Int64' for c in cols})

    def summary_table(self, names=None, baseline='Ramen', reference='NoAdapt'):
        """
        One row per run: correct images, domain-averaged accuracy, change vs `baseline` (points and images),
        images fixed / broken vs the zero-shot `reference`, and agreement with `baseline`.
        """
        names = names or list(self.results)
        a = self.aligned(names)
        N = len(a)
        right = {n: a[n] == a['label'] for n in names}
        rows = {}
        for n in names:
            fixed = int((right[n] & ~right[reference]).sum())
            broken = int((~right[n] & right[reference]).sum())
            rows[n] = {
                'correct': count_pct(int(right[n].sum()), N),
                'acc (domain avg)': round(domain_avg(self.results[n]), 2),
                f'vs {baseline} (pts)': f'{domain_avg(self.results[n]) - domain_avg(self.results[baseline]):+.2f}',
                f'vs {baseline} (images)': f'{int(right[n].sum() - right[baseline].sum()):+,}',
                f'fixed vs {reference}': count_pct(fixed, N),
                f'broken vs {reference}': count_pct(broken, N),
                f'same pred as {baseline}': f'{(a[n] == a[baseline]).mean():.2%}',
            }
        return pd.DataFrame(rows).T

    def prediction_counts(self, names=None):
        """How often each class is predicted by each run, next to how often it is the true class."""
        names = names or list(self.results)
        labels = self.results[names[0]]['label']
        t = pd.DataFrame({'true': labels.value_counts(), **{n: self.results[n]['pred'].value_counts() for n in names}})
        t = t.reindex(range(len(self.classes))).fillna(0).astype(int).set_axis(self.classes)
        return t.sort_values('true', ascending=False)

    def sanity_check(self, name='zero_shot_filing', baseline='Ramen'):
        """Share of images where `name` and `baseline` predict the same class (must be 100.00% for an 'off' run)."""
        a = self.aligned([name, baseline])
        same = int((a[name] == a[baseline]).sum())
        print(f'{name} == {baseline} on {same:,} of {len(a):,} images ({same / len(a):.2%})')
        return same == len(a)

    # ------------------------------------------------------------------ plots
    def plot_accuracy(self, names=None, by='domain', ax=None):
        """Accuracy per domain (or class), one dot per run (dot plot: runs differ by a few points on a ~50% scale)."""
        names = names or list(self.results)
        acc = self.accuracy_table(names, by)
        ax = ax or plt.subplots(figsize=(7, 0.45 * len(acc) + 1))[1]
        y = np.arange(len(acc))[::-1]
        for n in names:
            ax.plot(acc[n], y, 'o', ms=8, color=COLORS.get(n, 'k'), label=n, mec='white', mew=1.5)
        ax.set_yticks(y, acc.index)
        ax.set_xlabel('accuracy (%)')
        ax.grid(axis='x', alpha=0.3)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(names), frameon=False)
        return ax

    def plot_gain(self, name, baseline='NoAdapt', by='domain', ax=None):
        """Gain of `name` over `baseline` per domain (or class), labelled with points and images."""
        a = self.aligned([name, baseline])
        a['delta'] = (a[name] == a['label']).astype(int) - (a[baseline] == a['label']).astype(int)
        g = a.groupby(self.group_of(a, by), observed=True)['delta'].agg(['mean', 'sum'])
        g.loc[AVG_ROW[by]] = [a.groupby('domain', observed=True)['delta'].mean().mean(), g['sum'].sum()]
        ax = ax or plt.subplots(figsize=(7, 0.45 * len(g) + 1))[1]
        y = np.arange(len(g))[::-1]
        ax.axvline(0, color='#8a8984', lw=1)
        ax.hlines(y, 0, g['mean'] * 100, color=COLORS.get(name, 'k'), lw=2)
        ax.plot(g['mean'] * 100, y, 'o', ms=8, color=COLORS.get(name, 'k'), mec='white', mew=1.5)
        for yi, (m, s) in zip(y, g.values):
            ax.annotate(f'{m * 100:+.2f} pts ({int(s):+,} images)', (m * 100, yi), xytext=(6, 6),
                        textcoords='offset points', fontsize=8)
        lo, hi = min(0, g['mean'].min() * 100), max(0, g['mean'].max() * 100)
        ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.45 * (hi - lo))  # room for the labels
        ax.set_yticks(y, g.index)
        ax.set_xlabel(f'{name} − {baseline} accuracy (points)')
        ax.grid(axis='x', alpha=0.3)
        return ax

    def plot_stream_gain(self, name, baseline='NoAdapt', window=10_000, ax=None):
        """Rolling gain of `name` over `baseline` along the test stream (in `name`'s arrival order), pooled."""
        keys = ['domain_idx', 'sample_idx']
        d = self.results[name][keys + ['position', 'correct']].merge(
            self.results[baseline][keys + ['correct']], on=keys, suffixes=('', '_base')).sort_values('position')
        gain = (d['correct'].astype(int) - d['correct_base'].astype(int)).rolling(window, min_periods=window // 2)
        ax = ax or plt.subplots(figsize=(9, 3))[1]
        ax.axhline(0, color='#8a8984', lw=1)
        ax.plot(d['position'], gain.mean() * 100, lw=1, color=COLORS.get(name, 'k'))
        ax.set_xlabel('images seen in the test stream')
        ax.set_ylabel(f'{name} − {baseline}\n(pooled points, window {window:,})')
        ax.grid(alpha=0.3)
        return ax


# ---------------------------------------------------------------------------- across seeds
def seed_table(dataset, names=('Ramen',), seeds=range(5)):
    """Domain-averaged accuracy of each cached run for each seed (NaN where not run); no model load needed."""
    rows = {}
    for s in seeds:
        exp = TrueLabelCacheExperiment(dataset, seed=s).exp
        rows[s] = {n: domain_avg(pd.read_pickle(exp._cache_path(n))) if os.path.exists(exp._cache_path(n))
                   else np.nan for n in names}
    return pd.DataFrame(rows).T.rename_axis('seed')


def noise_floor(acc):
    """2 x (max − min) of a run's accuracy over seeds: a gain must clear this to count."""
    return 2 * (acc.max() - acc.min())
