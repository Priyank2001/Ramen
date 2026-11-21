"""
Confusion-records experiment: run Ramen with a recorder (src/methods/RamenRecorder.py) over a dataset, or a quick
subset of it, and get per-image records (zero-shot and Ramen probability vectors, embeddings before / after
adaptation), the class confusion of both, and the images behind each confusion.

    from experiment import RecordExperiment
    exp = RecordExperiment('DomainNet', seed=0, n_per_domain=500)   # quick: 500 images per domain
    exp = RecordExperiment('DomainNet', seed=0)                     # full dataset (n_per_domain=None)
    exp.run()                    # cached under new_experiments/results/, runs on the GPU only if missing
    exp.sanity_check()           # recorded predictions == plain Ramen on the same stream (must be 100.00%)
    exp.records                  # ImageRecords
    exp.confusion('zero_shot')   # ConfusionMatrix (also 'ramen'; domain='sketch' restricts)
    exp.shift()                  # ConfusionShift zero-shot -> Ramen
    exp.gallery.show(true='dog', pred='cat')

A subset is a different, shorter stream: Ramen's memory sees fewer images, so its accuracy on a subset is not
its accuracy on those images in the full run. Use subsets to look and to debug; quote numbers from full runs.

Extending:
  - a dataset: add a line to DATASETS (model and num_workers of its cached runs: num_workers sets the stream order)
  - a Ramen variant: write `class XRecorder(RecorderMixin, X)` in src/methods/RamenRecorder.py, register it,
    and add a line to VARIANTS
"""
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # new_experiments/, for ramen_analysis (which puts src/ on the path)

from ramen_analysis import make_experiment  # noqa: E402

from confusion import ConfusionMatrix, ConfusionShift  # noqa: E402
from gallery import ConfusionGallery  # noqa: E402
from records import ImageRecords  # noqa: E402

DATASETS = {
    'DomainNet': dict(model='clip_vitbase32', num_workers=16),
    'ImageNetC5K': dict(model='clip_vitbase16', num_workers=16),
    'ImageNetVariants': dict(model='clip_vitbase16', num_workers=16),
    'CIFAR10C': dict(model='clip_vitbase32', num_workers=4),
    'CIFAR100C': dict(model='clip_vitbase16', num_workers=4),
    'EuroSAT': dict(model='clip_vitbase16', num_workers=16),
}


@dataclass(frozen=True)
class Variant:
    """A recorded method and the plain run it must reproduce."""
    recorder: str                     # method class in src/methods
    method: str                       # the plain method: its cfg/<dataset>/<method>.yaml is used
    overrides: dict = field(default_factory=dict)
    reference: str = None             # cached run name of the plain method (sanity check), e.g. 'Ramen'


VARIANTS = {
    'Ramen': Variant('RamenRecorder', 'Ramen', reference='Ramen'),
    # ORACLE (uses the true labels to file the memory); reference run from 2026-10-27_true_label_cache
    'true_label': Variant('RamenTrueLabelRecorder', 'RamenTrueLabel', {'filing': 'true_label'}, 'true_label'),
}


def count_pct(n, total):
    """'3,790 of 29,784 (12.7%)'"""
    return f'{n:,} of {total:,} ({n / total:.1%})' if total else f'{n:,} of 0'


class RecordExperiment:
    """One dataset, one seed, one variant, on the full dataset or n_per_domain images per domain."""

    def __init__(self, dataset, seed=0, n_per_domain=None, variant='Ramen', results_dir=None, device=None):
        if dataset not in DATASETS:
            raise ValueError(f'Unknown dataset {dataset!r}; add it to DATASETS ({list(DATASETS)})')
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant {variant!r}; add it to VARIANTS ({list(VARIANTS)})')
        self.dataset, self.seed, self.n_per_domain = dataset, seed, n_per_domain
        self.variant_name, self.variant = variant, VARIANTS[variant]
        kwargs = dict(DATASETS[dataset], tta_mode='mixed', batch_size=100, n_per_domain=n_per_domain, seed=seed,
                      device=device)
        if results_dir is not None:
            kwargs['results_dir'] = results_dir
        self.exp = make_experiment(dataset, **kwargs)
        self.loaded = False   # CLIP and the transformed images, needed only to run
        self._records = self._gallery = None

    # ------------------------------------------------------------------ running
    @property
    def run_name(self):
        return f'rec_{self.variant_name}'

    @property
    def prefix(self):
        return self.exp._cache_path(self.run_name)[:-len('.pkl')]

    def is_cached(self, name=None):
        from methods.RamenRecorder import RecorderMixin
        if name is not None:
            return os.path.exists(self.exp._cache_path(name))
        return os.path.exists(self.prefix + '.pkl') and RecorderMixin.has_records(self.prefix)

    def _ensure_loaded(self):
        if not self.loaded:
            self.exp.load()
            self.loaded = True

    def run(self, use_cache=True):
        """Load the recorded run, or run it on the GPU (and cache it) if it is missing."""
        if not (use_cache and self.is_cached()):
            self._ensure_loaded()
            cfg = {**self.exp._load_cfg(self.variant.method), **self.variant.overrides}
            self.exp.run(self.variant.recorder, name=self.run_name, cfg=cfg, use_cache=False)
        self._records = ImageRecords.load(self.prefix, self.classes)
        return self

    def run_reference(self, use_cache=True):
        """The plain (non-recording) run on the same stream, from the cache or run now."""
        v = self.variant
        if not (use_cache and self.is_cached(v.reference)):
            self._ensure_loaded()
        return self.exp.run(v.method, name=v.reference, cfg=v.overrides or None, use_cache=use_cache)

    def sanity_check(self, use_cache=True):
        """
        The recorder must not change the method: its predictions must equal the plain run's on 100.00% of images
        (same stream, same seed). Also reports how often the recorded zero-shot prediction equals NoAdapt's.
        """
        rec = self.records.table
        ok = True
        for name, col in [(self.variant.reference, 'pred'), ('NoAdapt', 'zs_pred')]:
            if name == 'NoAdapt':
                if not (use_cache and self.is_cached('NoAdapt')):
                    self._ensure_loaded()
                ref = self.exp.run('NoAdapt', use_cache=use_cache)
            else:
                ref = self.run_reference(use_cache)
            keys = ['domain_idx', 'sample_idx']
            m = rec[keys + [col]].merge(ref[keys + ['pred']].rename(columns={'pred': 'pred_ref'}), on=keys)
            same = int((m[col] == m['pred_ref']).sum())
            what = 'recorded Ramen prediction' if col == 'pred' else 'recorded zero-shot prediction'
            print(f'{what} == {name}: {count_pct(same, len(m))}')
            if col == 'pred':
                ok &= same == len(m) == len(rec)
        return ok

    # ------------------------------------------------------------------ results
    @property
    def records(self):
        if self._records is None:
            if not self.is_cached():
                raise RuntimeError(f'{self.run_name} is not recorded yet for this setting: call run()')
            self._records = ImageRecords.load(self.prefix, self.classes)
        return self._records

    @property
    def classes(self):
        """Class names (from the dataset's file list; neither CLIP nor the images are loaded)."""
        if not hasattr(self.exp, 'classes'):
            self.exp.classes = list(self._datasets().classes)
        return self.exp.classes

    def _datasets(self):
        """The dataset objects (for class names and raw images), built without loading CLIP if needed."""
        if not hasattr(self.exp, 'datasets'):
            self.exp.datasets = self.exp._build_datasets(None)
        return self.exp.datasets

    def image(self, domain_idx, sample_idx):
        self._datasets()
        return self.exp._image(domain_idx, sample_idx)

    @property
    def gallery(self):
        if self._gallery is None:
            self._gallery = ConfusionGallery(self.records, self.image)
        return self._gallery

    def confusion(self, source='ramen', **filters):
        return ConfusionMatrix.from_records(self.records, source, **filters)

    def shift(self, **filters):
        return ConfusionShift(self.confusion('zero_shot', **filters), self.confusion('ramen', **filters))

    # ------------------------------------------------------------------ tables
    def data_table(self):
        """Images and classes per domain: the denominators of everything else."""
        t = self.records.table
        g = t.groupby('domain', observed=True)
        out = pd.DataFrame({'images': g.size(), 'classes present': g['label'].nunique()})
        out.loc['all'] = [len(t), t['label'].nunique()]
        return out.astype(int)

    def accuracy_table(self):
        """Zero-shot and Ramen accuracy per domain, with counts; last row = domain average (headline)."""
        t = self.records.table
        g = t.groupby('domain', observed=True)
        out = pd.DataFrame({'images': g.size(), 'zero-shot correct': g['zs_correct'].sum(),
                            'Ramen correct': g['correct'].sum()})
        out['zero-shot %'] = (out['zero-shot correct'] / out['images'] * 100).round(2)
        out['Ramen %'] = (out['Ramen correct'] / out['images'] * 100).round(2)
        out['Δ (pts)'] = (out['Ramen %'] - out['zero-shot %']).round(2)
        out.loc['domain avg'] = [len(t), t['zs_correct'].sum(), t['correct'].sum(), out['zero-shot %'].mean(),
                                 out['Ramen %'].mean(), out['Δ (pts)'].mean()]
        return out.astype({'images': int, 'zero-shot correct': int, 'Ramen correct': int}).round(2)

    def outcome_table(self, by='domain'):
        """Images per zero-shot -> Ramen outcome (fixed, broken, ...), per domain (or 'true_class'), with %."""
        t = self.records.table
        counts = pd.crosstab(t[by], t['outcome'], margins=True, margins_name='all').drop(columns='all')
        return counts.astype(int)

    def sink_fate(self, cls, k=8):
        """
        The images zero-shot WRONGLY predicts as `cls`: how many Ramen fixes, leaves at `cls`, or moves to another
        wrong class, and which classes it moves them to. Answers 'does Ramen repair a sink or relocate it?'.
        """
        t = self.records.table
        c = self.records.class_index(cls)
        s = t[(t['zs_pred'] == c) & ~t['zs_correct']]
        fate = pd.Series({'zero-shot false positives': len(s),
                          'fixed by Ramen': int(s['correct'].sum()),
                          f'still {self.classes[c]}': int((s['pred'] == c).sum()),
                          'moved to another wrong class': int(((s['pred'] != c) & ~s['correct']).sum())})
        moved = s[(s['pred'] != c) & ~s['correct']]['pred_class'].value_counts().head(k)
        return fate, moved[moved > 0].rename(f'moved from {self.classes[c]} to')

    def sink_origin(self, cls, k=8):
        """Ramen's false positives of `cls`: what zero-shot predicted for them (the new sink's sources), and domains."""
        t = self.records.table
        c = self.records.class_index(cls)
        s = t[(t['pred'] == c) & ~t['correct']]
        zs = s['zs_class'].value_counts().head(k)
        dom = s['domain'].value_counts()
        return (len(s), zs[zs > 0].rename(f'zero-shot prediction of Ramen\'s wrong {self.classes[c]}'),
                dom[dom > 0].rename('domain'))

    def embedding_shift_table(self):
        """How far one adaptation step moves the embedding (1 - cos), by outcome: median and 90th percentile."""
        t = self.records.table
        g = t.groupby('outcome', observed=True)['emb_shift']
        return pd.DataFrame({'images': g.size(), 'median shift': g.median(), 'p90 shift': g.quantile(0.9),
                             'mean shift': g.mean()}).round(5)

    def embedding_shift_by_domain(self):
        """Median embedding shift per (domain, outcome): pooled medians mix domains that move by different amounts."""
        t = self.records.table
        return t.pivot_table(index='domain', columns='outcome', values='emb_shift', aggfunc='median',
                             observed=True).round(4)
