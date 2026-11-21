"""
Cache-compression experiment: Ramen with block-quantised gradient values vs Ramen with a matched smaller queue.

    from cc_experiment import CompressionExperiment
    e = CompressionExperiment('ImageNetC5K', seed=0)            # n_per_domain=200, results_dir=... for a smoke run
    e.run('cc_int8')                                             # cached runs load; others run on the GPU
    e.sanity_check('cc_off', 'Ramen')                            # must be 100.00% identical predictions
    e.table(['Ramen', 'cc_off', 'cc_int8', 'cc_cap_int8'])       # accuracy, vs Ramen, memory, verdict inputs

Nothing in src/, cfg/ or the shared harness is edited: RamenCompressed is built through a local build_method
override (identical to RamenExperiment.build_method apart from the class lookup).
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
from cc_configs import DATASETS, memory_ratio, variants  # noqa: E402
from cc_registry import config_name, get_method_class  # noqa: E402


class LocalMethodsMixin:
    """Same as RamenExperiment.build_method; only the class and base-config lookups differ (cc_registry)."""

    def build_method(self, algo, cfg=None):
        args = SimpleNamespace(config={**self._load_cfg(config_name(algo)), **(cfg or {})}, device=self.device,
                               max_batch_size=self.batch_size, use_tbn=False)
        return get_method_class(algo)(deepcopy(self.clip), self.datasets, args)


def domain_avg(correct, domain):
    return 100 * pd.Series(np.asarray(correct)).groupby(np.asarray(domain)).mean().mean()


class CompressionExperiment:
    def __init__(self, dataset, seed=0, n_per_domain=None, results_dir=None, device=None):
        self.dataset, self.seed = dataset, seed
        d = DATASETS[dataset]
        kwargs = dict(model=d['model'], num_workers=d['num_workers'], tta_mode='mixed', batch_size=100,
                      n_per_domain=n_per_domain, seed=seed, device=device)
        if results_dir is not None:
            kwargs['results_dir'] = results_dir
        base = EXPERIMENTS[dataset]
        self.exp = type(f'CC{base.__name__}', (LocalMethodsMixin, base), {})(**kwargs)
        self.variants = variants(dataset)
        self.floor = d['floor']
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
            # probability arrays of NoAdapt and Ramen (~3 GB on DomainNet, from the HDD), which no table here needs
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
        algo, cfg = self.variants[name]
        return self.exp.run(algo, name=name, cfg=cfg, use_cache=use_cache)

    # ------------------------------------------------------------------ checks and tables
    def aligned(self, names):
        """Predictions of several runs side by side, in stream order; checks the streams are the same images."""
        ref = self.results[names[0]].sort_values('position').reset_index(drop=True)
        out = ref[['position', 'domain', 'domain_idx', 'sample_idx', 'label']].copy()
        for n in names:
            df = self.results[n].sort_values('position').reset_index(drop=True)
            assert (df['sample_idx'].to_numpy() == ref['sample_idx'].to_numpy()).all(), f'{n}: different stream'
            assert (df['domain_idx'].to_numpy() == ref['domain_idx'].to_numpy()).all(), f'{n}: different stream'
            out[n] = df['pred'].to_numpy()
        return out

    def sanity_check(self, a='cc_off', b='Ramen'):
        al = self.aligned([a, b])
        same = int((al[a] == al[b]).sum())
        print(f'{self.dataset}: {a} == {b} on {same:,} of {len(al):,} images ({same / len(al):.2%})')
        return same == len(al)

    def table(self, names):
        names = list(names)
        al = self.aligned(['NoAdapt', 'Ramen', *[n for n in names if n not in ('NoAdapt', 'Ramen')]])
        ramen_ok = al['Ramen'] == al['label']
        ramen_acc = domain_avg(ramen_ok, al['domain'])
        rows = []
        for n in names:
            ok = al[n] == al['label']
            s = self.stats.get(n, {}) if n not in ('NoAdapt', 'Ramen') else {}
            codec = s.get('codec', 'fp16') if n != 'NoAdapt' else '-'
            cap = s.get('max_capacity', DATASETS[self.dataset]['capacity']) if n != 'NoAdapt' else 0
            acc = domain_avg(ok, al['domain'])
            rows.append(dict(run=n, codec=codec, capacity=cap,
                             correct=int(ok.sum()), images=len(al), acc=acc, vs_ramen=acc - ramen_acc,
                             vs_ramen_images=int(ok.sum() - ramen_ok.sum()),
                             same_pred_as_ramen=float((al[n] == al['Ramen']).mean()),
                             fixed_vs_ramen=int((ok & ~ramen_ok).sum()), broken_vs_ramen=int((~ok & ramen_ok).sum()),
                             cache_MB=s.get('cache_bytes', np.nan) / 2**20,
                             mem_ratio=(memory_ratio(codec) * cap / DATASETS[self.dataset]['capacity']
                                        if codec != '-' else np.nan),
                             peak_GB=s.get('peak_mem_bytes', np.nan) / 2**30,
                             mean_zero_frac=float(np.mean(s['zero_frac'])) if s.get('zero_frac') else np.nan))
        return pd.DataFrame(rows).set_index('run')
