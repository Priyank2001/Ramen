"""
Label-value oracle experiment: does Ramen's cache help once its entries point toward their TRUE labels?

Two factors, each Ramen's choice or the oracle's, crossed (2 x 2):
  filing  which queue an image is written to:   zero-shot prediction (Ramen) | true label (ORACLE)
  value   what its entry holds once predicted:  own entropy gradient (Ramen)  | cross-entropy toward true label (ORACLE)
Ramen and the filing-only oracle (oracle_file_true, 2026-10-23_cache_contamination) are cached; the two
label-value runs are new (label_methods.RamenLabelValues).

    from label_oracle import LabelOracleExperiment
    exp = LabelOracleExperiment('DomainNet', seed=0)
    exp.run_all()            # cached runs load from new_experiments/results/, missing ones run on the GPU
    exp.factor_table()       # the 2 x 2: domain-averaged accuracy, change vs Ramen
    exp.summary_table()      # (from TrueLabelCacheExperiment) one row per run, counts vs NoAdapt and vs Ramen

The table and plot methods come from TrueLabelCacheExperiment (2026-10-27_true_label_cache, imported read-only);
this class changes only which methods can be built (the local registry) and the run list.
"""
import os
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # new_experiments/, for ramen_analysis
sys.path.insert(0, os.path.join(os.path.dirname(HERE), '2026-10-27_true_label_cache'))  # read-only import

from ramen_analysis import EXPERIMENTS  # noqa: E402  (also puts src/ on the path)
from registry import config_name, get_method_class  # noqa: E402
from true_label_cache import DATASETS, TrueLabelCacheExperiment, domain_avg  # noqa: E402

# run name -> (method, config overrides on cfg/<dataset>/<method>.yaml, (filing, value))
VARIANTS = {
    'NoAdapt': ('NoAdapt', None, None),
    'Ramen': ('Ramen', None, ('zero-shot', 'entropy')),
    'oracle_file_true': ('RamenOracle', {'oracle': 'file_true'}, ('true label', 'entropy')),  # ORACLE, cached
    'labeloracle_zs_ce': ('RamenLabelValues', {'filing': 'zero_shot', 'value': 'true_label_ce'},
                          ('zero-shot', 'true-label CE')),                                   # ORACLE
    'labeloracle_true_ce': ('RamenLabelValues', {'filing': 'true_label', 'value': 'true_label_ce'},
                            ('true label', 'true-label CE')),                                # ORACLE
    # written AFTER prediction (RamenWriteAfterPrediction): needed to file by the adapted prediction; the zero-shot
    # filing runs are the timing controls (same filing as Ramen / labeloracle_zs_ce, only the write moved)
    'late_zs_entropy': ('RamenWriteAfterPrediction', {'filing': 'zero_shot', 'value': 'entropy'},
                        ('zero-shot', 'entropy')),
    'late_zs_ce': ('RamenWriteAfterPrediction', {'filing': 'zero_shot', 'value': 'true_label_ce'},
                   ('zero-shot', 'true-label CE')),                                          # ORACLE
    'late_adapted_entropy': ('RamenWriteAfterPrediction', {'filing': 'adapted', 'value': 'entropy'},
                             ('adapted', 'entropy')),                                        # no labels: a method
    'late_adapted_ce': ('RamenWriteAfterPrediction', {'filing': 'adapted', 'value': 'true_label_ce'},
                        ('adapted', 'true-label CE')),                                       # ORACLE
    # global top-k retrieval (RamenGlobalRetrieval): the k nearest entries of the whole cache instead of top-k per
    # class queue; filing and write order are Ramen's
    **{f'global_ce_k{k}': ('RamenGlobalRetrieval', {'value': 'true_label_ce', 'retrieval': 'global_knn', 'global_k': k},
                           ('zero-shot', 'true-label CE')) for k in (10, 100, 1000)},              # ORACLE
    'global_entropy_k100': ('RamenGlobalRetrieval', {'value': 'entropy', 'retrieval': 'global_knn', 'global_k': 100},
                            ('zero-shot', 'entropy')),                                       # no labels
    # ImageNet-C: hide an image's OTHER corrupted copies of the same photo (its own entry stays), so label values
    # cannot leak a photo's label through its copies. 'base_image' is resolved per experiment (see run)
    'excl_zs_ce': ('RamenGlobalRetrieval', {'value': 'true_label_ce', 'retrieval': 'per_class',
                                            'exclude_groups': 'base_image'}, ('zero-shot', 'true-label CE')),  # ORACLE
    **{f'excl_global_ce_k{k}': ('RamenGlobalRetrieval', {'value': 'true_label_ce', 'retrieval': 'global_knn',
                                                         'global_k': k, 'exclude_groups': 'base_image'},
                                ('zero-shot', 'true-label CE')) for k in (10, 100, 1000)},             # ORACLE
    'excl_global_entropy_k100': ('RamenGlobalRetrieval', {'value': 'entropy', 'retrieval': 'global_knn',
                                                          'global_k': 100, 'exclude_groups': 'base_image'},
                                 ('zero-shot', 'entropy')),
    'excl_zs_entropy': ('RamenGlobalRetrieval', {'value': 'entropy', 'retrieval': 'per_class',
                                                 'exclude_groups': 'base_image'}, ('zero-shot', 'entropy')),
    # smoke only: this folder's own per-class retrieval (used when copies are hidden) with nothing hidden, vs Ramen's
    # code path; fp32 weights and sums instead of Ramen's fp16, so agreement just below 100% is expected
    'excl_numerics': ('RamenGlobalRetrieval', {'value': 'entropy', 'retrieval': 'per_class', 'exclude_groups': 'unique'},
                      None),
    # smoke subset only: must reproduce Ramen / the filing-only oracle exactly
    'global_off': ('RamenGlobalRetrieval', {'value': 'entropy', 'retrieval': 'per_class'}, None),
    'labeloracle_off': ('RamenLabelValues', {'filing': 'zero_shot', 'value': 'entropy'}, None),
    # also the filing oracle on datasets without a cached oracle_file_true (smoke: == RamenTrueLabel)
    'labeloracle_true_entropy': ('RamenLabelValues', {'filing': 'true_label', 'value': 'entropy'},
                                 ('true label', 'entropy')),                                 # ORACLE
    'RamenTrueLabel_true': ('RamenTrueLabel', {'filing': 'true_label'}, None),
}
LABEL_RUNS = ('labeloracle_zs_ce', 'labeloracle_true_ce')
LATE_RUNS = ('late_zs_entropy', 'late_zs_ce', 'late_adapted_entropy', 'late_adapted_ce')
GLOBAL_RUNS = ('global_ce_k10', 'global_ce_k100', 'global_ce_k1000', 'global_entropy_k100')
EXCL_RUNS = ('excl_zs_entropy', 'excl_zs_ce', 'excl_global_ce_k10', 'excl_global_ce_k100', 'excl_global_ce_k1000',
             'excl_global_entropy_k100')
# when an image is written to the cache: Ramen's order, or after its prediction (needed for adapted filing)
TIMING = {**{n: 'after prediction' for n in LATE_RUNS}}
# the filing-only oracle: cached from 2026-10-23_cache_contamination where it exists, else run here
FILING_ORACLE = {'DomainNet': 'oracle_file_true', 'ImageNetC5K': 'oracle_file_true'}


def full_runs(dataset):
    return ('NoAdapt', 'Ramen', FILING_ORACLE.get(dataset, 'labeloracle_true_entropy'), *LABEL_RUNS)


def new_runs(dataset):
    """The runs this folder adds for `dataset` (the label-value runs, and the filing oracle if not cached)."""
    return tuple(n for n in full_runs(dataset)[2:] if n not in FILING_ORACLE.values())


FULL_RUNS, NEW_RUNS = full_runs('DomainNet'), new_runs('DomainNet')
SMOKE_RUNS = ('NoAdapt', 'Ramen', 'labeloracle_off', 'RamenTrueLabel_true', 'labeloracle_true_entropy',
              'labeloracle_zs_ce', 'labeloracle_true_ce', *LATE_RUNS, 'global_off', *GLOBAL_RUNS)
SMOKE_CHECKS = [('labeloracle_off', 'Ramen'), ('labeloracle_true_entropy', 'RamenTrueLabel_true'),
                ('global_off', 'Ramen')]
LABELS = {'NoAdapt': 'zero-shot', 'Ramen': 'Ramen', 'oracle_file_true': 'filing oracle',
          'labeloracle_true_entropy': 'filing oracle',
          'labeloracle_zs_ce': 'label-value oracle', 'labeloracle_true_ce': 'filing + label-value oracle',
          'late_zs_entropy': 'written after prediction', 'late_zs_ce': 'written after prediction + label values',
          'late_adapted_entropy': 'adapted filing', 'late_adapted_ce': 'adapted filing + label values',
          'global_ce_k10': 'global k=10 + label values', 'global_ce_k100': 'global k=100 + label values',
          'global_ce_k1000': 'global k=1000 + label values', 'global_entropy_k100': 'global k=100 (no labels)',
          'excl_zs_entropy': 'Ramen, copies hidden', 'excl_zs_ce': 'label values, copies hidden',
          'excl_global_ce_k10': 'global k=10 + label values, copies hidden',
          'excl_global_ce_k100': 'global k=100 + label values, copies hidden',
          'excl_global_ce_k1000': 'global k=1000 + label values, copies hidden',
          'excl_global_entropy_k100': 'global k=100, copies hidden (no labels)'}


class LocalMethodsMixin:
    """Lets the harness build this folder's methods; every other behaviour is the harness's own."""

    def build_method(self, algo, cfg=None):
        # Same as RamenExperiment.build_method (new_experiments/ramen_analysis.py); only the class and config lookups
        # differ (a local method starts from its repo base's config)
        args = SimpleNamespace(config={**self._load_cfg(config_name(algo)), **(cfg or {})}, device=self.device,
                               max_batch_size=self.batch_size, use_tbn=False)
        return get_method_class(algo)(deepcopy(self.clip), self.datasets, args)


def make_local_experiment(dataset, **kwargs):
    base = EXPERIMENTS[dataset]
    return type(f'Local{base.__name__}', (LocalMethodsMixin, base), {})(**kwargs)


class LabelOracleExperiment(TrueLabelCacheExperiment):
    """One dataset and seed: the runs in VARIANTS over the same mixed stream, with the 2 x 2 factor table."""

    def __init__(self, dataset, seed=0, n_per_domain=None, results_dir=None, device=None):
        self.dataset, self.seed = dataset, seed
        kwargs = dict(DATASETS[dataset], tta_mode='mixed', batch_size=100, n_per_domain=n_per_domain, seed=seed,
                      device=device)
        if results_dir is not None:
            kwargs['results_dir'] = results_dir
        self.exp = make_local_experiment(dataset, **kwargs)
        self.full_runs, self.new_runs = full_runs(dataset), new_runs(dataset)
        self.filing_oracle = self.full_runs[2]
        self.loaded = False
        self._classes = None

    def base_image_groups(self):
        """Group id per stream position = the photo (sample_idx), from Ramen's cached run of the same stream."""
        if self.dataset != 'ImageNetC5K':
            raise ValueError('base-image groups are defined for ImageNetC5K only (same 5,000 photos per corruption)')
        df = self.results.get('Ramen')
        df = self.run('Ramen') if df is None else df
        return df.sort_values('position')['sample_idx'].to_numpy()

    def run(self, name, use_cache=True):
        method, cfg, _ = VARIANTS[name]
        if cfg and cfg.get('exclude_groups') == 'base_image':
            cfg = {**cfg, 'exclude_groups': self.base_image_groups()}
        elif cfg and cfg.get('exclude_groups') == 'unique':   # every image its own group: nothing is excluded
            cfg = {**cfg, 'exclude_groups': np.arange(len(self.run('Ramen')))}
        if not (use_cache and self.is_cached(name)) and not self.loaded:
            self.exp.load()
            self.loaded = True
        return self.exp.run(method, name=name, cfg=cfg, use_cache=use_cache)

    def run_all(self, names=None, use_cache=True):
        for name in names or self.full_runs:
            self.run(name, use_cache=use_cache)
        return self

    def factor_table(self):
        """The 2 x 2 (filing x value): domain-averaged accuracy and change vs Ramen, in points and images."""
        base = self.results['Ramen']
        rows = []
        for name in self.full_runs[1:]:
            filing, value = VARIANTS[name][2]
            df = self.results[name]
            rows.append({'filing': filing, 'value': value, 'run': name,
                         'acc (domain avg)': round(domain_avg(df), 2),
                         'vs Ramen (pts)': round(domain_avg(df) - domain_avg(base), 2),
                         'vs Ramen (images)': int(df['correct'].sum() - base['correct'].sum())})
        return pd.DataFrame(rows).set_index(['filing', 'value'])

    def filing_table(self):
        """Every run that has a (filing, value) cell, with when it writes: the 3 filings x 2 values view."""
        base = self.results['Ramen']
        rows = []
        for name in (*self.full_runs[1:], *LATE_RUNS):
            if name not in self.results:
                continue
            filing, value = VARIANTS[name][2]
            df = self.results[name]
            rows.append({'write': TIMING.get(name, 'before retrieval (Ramen)'), 'filing': filing, 'value': value,
                         'run': name, 'uses labels': name != 'Ramen' and (filing == 'true label' or value != 'entropy'),
                         'acc (domain avg)': round(domain_avg(df), 2),
                         'vs Ramen (pts)': round(domain_avg(df) - domain_avg(base), 2),
                         'vs Ramen (images)': int(df['correct'].sum() - base['correct'].sum())})
        return pd.DataFrame(rows).set_index(['write', 'filing', 'value'])

    def vs_table(self, names, baselines=('Ramen',)):
        """Domain-averaged accuracy of each run, and its change vs each baseline in points and images."""
        rows = []
        for name in names:
            df = self.results[name]
            row = {'run': name, 'label': LABELS.get(name, name), 'acc (domain avg)': round(domain_avg(df), 2)}
            for b in baselines:
                base = self.results[b]
                row[f'vs {b} (pts)'] = round(domain_avg(df) - domain_avg(base), 2)
                row[f'vs {b} (images)'] = int(df['correct'].sum() - base['correct'].sum())
            rows.append(row)
        return pd.DataFrame(rows).set_index('run')

    def sink_table(self, class_names, names=None):
        """How often each run predicts each class, and how many of those are wrong ('3,201 (2,950 wrong)')."""
        names = names or self.full_runs
        idx = {c: self.classes.index(c) for c in class_names}
        out = {}
        for n in names:
            df = self.results[n]
            out[LABELS.get(n, n)] = {c: f'{int((df.pred == i).sum()):,} ({int(((df.pred == i) & ~df.correct).sum()):,} wrong)'
                                     for c, i in idx.items()}
        t = pd.DataFrame(out)
        t.insert(0, 'truly this class', [int((self.results[names[0]].label == i).sum()) for i in idx.values()])
        return t

    def top_sinks(self, k=10, names=None):
        """The k classes Ramen predicts wrongly most often, with each run's wrong predictions of them."""
        names = names or self.full_runs
        wrong = {LABELS.get(n, n): self.results[n].loc[~self.results[n]['correct'], 'pred'].value_counts() for n in names}
        t = pd.DataFrame(wrong).fillna(0).astype(int)
        t = t.sort_values('Ramen', ascending=False).head(k)
        t.index = [self.classes[i] for i in t.index]
        t.insert(0, 'truly this class', [int((self.results[names[0]].label == self.classes.index(c)).sum())
                                         for c in t.index])
        return t.rename_axis('class (wrong predictions)')
