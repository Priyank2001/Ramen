"""
Pseudo-label values: can a LABEL-FREE target recover part of the label-value oracle's gain?

    from pseudo_values import PseudoValueExperiment
    exp = PseudoValueExperiment('DomainNet', seed=0)
    exp.run_all(exp.screen_runs)     # cached runs load from new_experiments/results/, missing ones run on the GPU
    exp.vs_table(exp.screen_runs)    # accuracy and change vs Ramen (from LabelOracleExperiment)
    exp.target_table()               # how often each run's target is right, and what it changes vs zero-shot

LabelOracleExperiment (2026-11-01_label_oracle, imported read-only) supplies the tables and the cached oracle runs;
this class only adds the runs below and builds this folder's method through pv_registry.
"""
import os
import sys
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_ORACLE_DIR = os.path.join(os.path.dirname(HERE), '2026-11-01_label_oracle')
sys.path.insert(0, LABEL_ORACLE_DIR)  # read-only import: label_methods, registry, label_oracle

import label_oracle  # noqa: E402  (puts new_experiments/ and src/ on the path)
from label_oracle import LABELS, LabelOracleExperiment  # noqa: E402
from ramen_analysis import EXPERIMENTS  # noqa: E402
from true_label_cache import DATASETS  # noqa: E402

sys.path.insert(0, HERE)  # this folder's modules before the read-only ones
from pv_registry import config_name, get_method_class  # noqa: E402

# run name -> (method, config overrides on cfg/<dataset>/Ramen.yaml, target description)
PV_VARIANTS = {
    'pv_zs': ('RamenPseudoValues', {'target': 'zero_shot'}, 'zero-shot prediction'),
    'pv_adapted': ('RamenPseudoValues', {'target': 'adapted'}, 'adapted prediction'),
    'pv_knn': ('RamenPseudoValues', {'target': 'knn', 'knn_k': 10}, 'kNN vote (10 nearest, own excluded)'),
    'pv_mix_p25': ('RamenPseudoValues', {'target': 'mix', 'mix_p': 0.25}, 'ORACLE: true label for 25% of zs errors'),
    'pv_mix_p50': ('RamenPseudoValues', {'target': 'mix', 'mix_p': 0.50}, 'ORACLE: true label for 50% of zs errors'),
    # smoke only: no target = Ramen exactly; true_label target = labeloracle_zs_ce (other code path, same values)
    'pv_off': ('RamenPseudoValues', {'target': 'none'}, None),
    'pv_true': ('RamenPseudoValues', {'target': 'true_label'}, 'ORACLE: true label'),
}
LABEL_FREE = ('pv_zs', 'pv_adapted', 'pv_knn')
DOSE = ('pv_mix_p25', 'pv_mix_p50')
SMOKE_RUNS = ('NoAdapt', 'Ramen', 'labeloracle_zs_ce', 'pv_off', 'pv_true', *LABEL_FREE, *DOSE)
SMOKE_CHECKS = [('pv_off', 'Ramen'), ('pv_true', 'labeloracle_zs_ce')]
PV_LABELS = {'pv_zs': 'CE → zero-shot', 'pv_adapted': 'CE → adapted pred', 'pv_knn': 'CE → kNN vote',
             'pv_mix_p25': 'CE → 25% errors corrected (oracle)', 'pv_mix_p50': 'CE → 50% errors corrected (oracle)',
             'pv_true': 'CE → true label (oracle)'}
LABELS.update(PV_LABELS)


def screen_runs(dataset):
    """Seed-0 screen: the label-free runs everywhere; the dose-response where the label oracle has no copy leak."""
    return (*LABEL_FREE, *DOSE) if dataset != 'ImageNetC5K' else LABEL_FREE


class LocalMethodsMixin:
    """Same as RamenExperiment.build_method; only the class and config lookups differ (pv_registry)."""

    def build_method(self, algo, cfg=None):
        args = SimpleNamespace(config={**self._load_cfg(config_name(algo)), **(cfg or {})}, device=self.device,
                               max_batch_size=self.batch_size, use_tbn=False)
        return get_method_class(algo)(deepcopy(self.clip), self.datasets, args)


class PseudoValueExperiment(LabelOracleExperiment):

    def __init__(self, dataset, seed=0, n_per_domain=None, results_dir=None, device=None):
        super().__init__(dataset, seed=seed, n_per_domain=n_per_domain, results_dir=results_dir, device=device)
        kwargs = dict(DATASETS[dataset], tta_mode='mixed', batch_size=100, n_per_domain=n_per_domain, seed=seed,
                      device=device)
        if results_dir is not None:
            kwargs['results_dir'] = results_dir
        base = EXPERIMENTS[dataset]
        self.exp = type(f'PV{base.__name__}', (LocalMethodsMixin, base), {})(**kwargs)
        self.screen_runs = screen_runs(dataset)

    def run(self, name, use_cache=True):
        if name not in PV_VARIANTS:
            return super().run(name, use_cache=use_cache)
        method, cfg, _ = PV_VARIANTS[name]
        if not (use_cache and self.is_cached(name)) and not self.loaded:
            self.exp.load()
            self.loaded = True
        return self.exp.run(method, name=name, cfg=cfg, use_cache=use_cache)

    def target_table(self, names=None):
        """Per run: how often its target is right vs zero-shot, and the errors it fixes / correct ones it breaks."""
        rows = []
        for n in names or self.screen_runs:
            s = self.stats.get(n)
            if not s or not s.get('target_right'):
                continue
            N = sum(s['batch_size'])
            rows.append({'run': n, 'target': PV_VARIANTS[n][2], 'images': N,
                         'zero-shot right %': round(100 * sum(s['zs_right']) / N, 2),
                         'target right %': round(100 * sum(s['target_right']) / N, 2),
                         'target ≠ zero-shot': sum(s['target_changed']),
                         'zs errors fixed by target': sum(s['target_fixes']),
                         'zs-correct broken by target': sum(s['target_breaks'])})
        return pd.DataFrame(rows).set_index('run')
