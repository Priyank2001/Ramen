"""
Gradient-alignment experiment: run (or load) Ramen with the gradient probe and join it with NoAdapt and Ramen.

    from ga_experiment import GradAlignmentExperiment
    e = GradAlignmentExperiment('DomainNet', seed=0)     # n_per_domain=200, results_dir=... for a smoke run
    e.run_all()                                          # NoAdapt, Ramen (cached) and ga_probe (GPU if not cached)
    e.sanity_check()                                     # ga_probe must predict exactly what Ramen predicts
    df = e.frame()                                       # one row per image: outcome + every probe field

Nothing in src/, cfg/ or the shared harness is edited: RamenGradientProbe is built through a local build_method
override (identical to RamenExperiment.build_method apart from the class and base-config lookups).
"""
import os
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # new_experiments/, for ramen_analysis (puts src/ on the path)
from ramen_analysis import EXPERIMENTS  # noqa: E402

sys.path.insert(0, HERE)
from ga_configs import DATASETS, PROBE_RUN, RUNS  # noqa: E402
from ga_registry import config_name, get_method_class  # noqa: E402

OUTCOMES = ('both right', 'fixed by Ramen', 'broken by Ramen', 'both wrong')


class LocalMethodsMixin:
    """Same as RamenExperiment.build_method; only the class and base-config lookups differ (ga_registry)."""

    def build_method(self, algo, cfg=None):
        args = SimpleNamespace(config={**self._load_cfg(config_name(algo)), **(cfg or {})}, device=self.device,
                               max_batch_size=self.batch_size, use_tbn=False)
        return get_method_class(algo)(deepcopy(self.clip), self.datasets, args)


def outcome(zs_ok, ramen_ok):
    """Zero-shot vs Ramen correctness -> one of OUTCOMES (as an ordered categorical)."""
    zs_ok, ramen_ok = np.asarray(zs_ok, bool), np.asarray(ramen_ok, bool)
    codes = np.select([zs_ok & ramen_ok, ~zs_ok & ramen_ok, zs_ok & ~ramen_ok], [0, 1, 2], default=3)
    return pd.Categorical.from_codes(codes, OUTCOMES)


class GradAlignmentExperiment:
    def __init__(self, dataset, seed=0, n_per_domain=None, results_dir=None, device=None):
        self.dataset, self.seed = dataset, seed
        d = DATASETS[dataset]
        kwargs = dict(model=d['model'], num_workers=d['num_workers'], tta_mode='mixed', batch_size=100,
                      n_per_domain=n_per_domain, seed=seed, device=device)
        if results_dir is not None:
            kwargs['results_dir'] = results_dir
        base = EXPERIMENTS[dataset]
        self.exp = type(f'GA{base.__name__}', (LocalMethodsMixin, base), {})(**kwargs)
        self.loaded = False

    # ------------------------------------------------------------------ running
    @property
    def results(self):
        return self.exp.results

    @property
    def stats(self):
        return self.exp.method_stats

    def is_cached(self, name):
        return os.path.exists(self.exp._cache_path(name))

    def run(self, name, use_cache=True):
        if use_cache and self.is_cached(name):
            # Read only the per-image table and stats; the harness would also load the large cached feature /
            # probability arrays of NoAdapt and Ramen (~3 GB on DomainNet, from the HDD), which nothing here needs
            path = self.exp._cache_path(name)
            self.exp.results[name] = pd.read_pickle(path)
            stats = path[:-len('.pkl')] + '_stats.pkl'
            if os.path.exists(stats):
                self.exp.method_stats[name] = pd.read_pickle(stats)
            return self.exp.results[name]
        if not self.loaded:
            self.exp.load()
            self.loaded = True
        if name in ('NoAdapt', 'Ramen'):
            return self.exp.run(name, use_cache=use_cache)
        algo, cfg = RUNS[name]
        return self.exp.run(algo, name=name, cfg=cfg, use_cache=use_cache)

    def run_all(self):
        for name in ('NoAdapt', 'Ramen', PROBE_RUN):
            self.run(name)
        return self

    # ------------------------------------------------------------------ checks
    def _in_stream_order(self, name):
        return self.results[name].sort_values('position').reset_index(drop=True)

    def sanity_check(self):
        """The probe must predict what Ramen predicts on every image, and see the same stream and labels."""
        ramen, probe = self._in_stream_order('Ramen'), self._in_stream_order(PROBE_RUN)
        same_stream = (ramen[['domain_idx', 'sample_idx', 'label']].to_numpy()
                       == probe[['domain_idx', 'sample_idx', 'label']].to_numpy()).all()
        same = int((ramen['pred'] == probe['pred']).sum())
        print(f'{self.dataset}: same stream as Ramen: {same_stream}; {PROBE_RUN} == Ramen on '
              f'{same:,} of {len(ramen):,} images ({same / len(ramen):.2%})')
        return bool(same_stream) and same == len(ramen)

    # ------------------------------------------------------------------ the per-image table
    def probe_table(self):
        """The probe's per-image records, in stream order (row i = position i)."""
        batches = self.stats[PROBE_RUN]['probe']
        return pd.DataFrame({k: np.concatenate([b[k] for b in batches]) for k in batches[0]})

    def frame(self):
        """One row per image: stream position, domain, labels, zero-shot / Ramen outcome, p(true), probe fields."""
        zs, ramen, probe = (self._in_stream_order(n) for n in ('NoAdapt', 'Ramen', PROBE_RUN))
        rec = self.probe_table()
        assert len(rec) == len(ramen), 'probe log and results have different lengths'
        assert (rec['label'].to_numpy() == ramen['label'].to_numpy()).all(), 'probe log is not in stream order'
        # Zero-shot = the prediction Ramen filed each image under (logged by the probe). The cached NoAdapt run
        # computes the same logits through another code path; fp16 near-ties make a few argmaxes differ
        self.zs_mismatch = int((rec['zs_pred'].to_numpy() != zs['pred'].to_numpy()).sum())
        df = ramen[['position', 'domain', 'label']].copy()
        df['zs_pred'], df['ramen_pred'] = rec['zs_pred'].to_numpy(), ramen['pred'].to_numpy()
        df['zs_ok'], df['ramen_ok'] = df['zs_pred'] == df['label'], ramen['correct'].to_numpy()
        df['outcome'] = outcome(df['zs_ok'], df['ramen_ok'])
        df['p_true_ramen'] = ramen['true_prob'].to_numpy()
        # dlogp_true: change of the true-class log-probability caused by Ramen's step (= minus the change of the CE loss)
        for k in rec.columns.drop(['label', 'zs_pred']):
            df[k] = rec[k].to_numpy()
        df['dlogp_true'] = np.log(np.clip(df['p_true_ramen'], 1e-12, None)) - np.log(np.clip(df['p_true_zs'], 1e-12, None))
        return df
