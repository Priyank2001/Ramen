"""
Offline cluster-shift experiment: replay the cached zero-shot embeddings of a dataset through a StreamCorrector and
classify with the zero-shot text weights. No model forward pass, so a full DomainNet run takes minutes on a CPU.

    from cshift_experiment import ClusterShiftExperiment
    e = ClusterShiftExperiment('ImageNetC5K', seed=0)
    e.sanity_check()                 # alpha = 0 reproduces cached NoAdapt (must be ~100%)
    e.run('kmeans_k20_a1.0_text')    # cached as results/<...>_cshift_<name>.pkl
    e.summary(['none', 'kmeans_k20_a1.0_text'])   # domain-averaged accuracy vs NoAdapt and Ramen

Inputs, all cached by earlier runs (seed-s stream order): NoAdapt_feats / NoAdapt.pkl / Ramen.pkl / Ramen_feats, and
the CLIP text features (computed once on the GPU if missing, via 2026-11-01_pillow_sink `TextFeatures`).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(HERE)
for p in (NEW, os.path.join(NEW, '2026-10-28_confusion_records'), os.path.join(NEW, '2026-11-01_pillow_sink'), HERE):
    sys.path.insert(0, p)

from cluster_shift import (KMeansShift, NoShift, OracleGroupShift, RandomGroupShift,  # noqa: E402
                           StreamRunner)
from cshift_configs import DATASETS, VARIANTS  # noqa: E402

RESULTS = os.path.join(NEW, 'results')


def domain_average(df, col='correct'):
    """Headline accuracy convention: mean of per-domain accuracies, in percent."""
    return 100 * df.groupby('domain', observed=True)[col].mean().mean()


class ClusterShiftExperiment:
    def __init__(self, dataset, seed=0, device='cuda'):
        self.dataset, self.seed, self.device = dataset, seed, device
        self.model = DATASETS[dataset]['model']
        self.prefix = os.path.join(RESULTS, f'{dataset}_clip_{self.model[5:]}_mixed_nall_bs100_s{seed}')
        self.noadapt = pd.read_pickle(f'{self.prefix}_NoAdapt.pkl')
        self.ramen = pd.read_pickle(f'{self.prefix}_Ramen.pkl')
        self.Z = np.load(f'{self.prefix}_NoAdapt_feats.npy').astype(np.float32)
        self.Z_ramen = np.load(f'{self.prefix}_Ramen_feats.npy', mmap_mode='r')
        self.W = self._text_features()
        self.labels = self.noadapt['label'].to_numpy()
        self.domain_ids = self.noadapt['domain_idx'].to_numpy()
        self.outputs = {}       # variant -> per-sample DataFrame
        self.groups = {}        # variant -> group id per image

    # ------------------------------------------------------------------ inputs
    def _text_features(self):
        from sink_runs import RecordExperiment, TextFeatures
        rec = RecordExperiment(self.dataset, seed=self.seed)
        W = TextFeatures(rec.exp, rec.classes).load(self.device)
        return W / np.linalg.norm(W, axis=1, keepdims=True)

    def predict(self, Z):
        """Zero-shot classification argmax_c z . w_c (the 100x temperature does not change the argmax)."""
        out = np.empty(len(Z), np.int64)
        for s in range(0, len(Z), 50_000):
            out[s:s + 50_000] = (Z[s:s + 50_000] @ self.W.T).argmax(1)
        return out

    # ------------------------------------------------------------------ running
    def _cache(self, name):
        return f'{self.prefix}_cshift_{name}.pkl'

    def _corrector(self, v, k_means_sizes=None):
        m_ref = self.W.mean(0) if v.ref == 'text' else None
        if v.kind == 'none':
            return NoShift()
        if v.kind == 'kmeans':
            return KMeansShift(v.k, v.alpha, m_ref, seed=self.seed)
        if v.kind == 'random':
            return RandomGroupShift(k_means_sizes, v.alpha, m_ref, seed=self.seed)
        if v.kind == 'oracle':
            return OracleGroupShift(v.alpha, m_ref)
        raise ValueError(v.kind)

    def run(self, name, use_cache=True):
        path = self._cache(name)
        if use_cache and os.path.exists(path):
            self.outputs[name] = pd.read_pickle(path)
            self.groups[name] = self.outputs[name]['group'].to_numpy()
            return self.outputs[name]
        v = VARIANTS[name]
        sizes = None
        if v.kind == 'random':   # control: cluster sizes of the matching label-free run
            twin = f'kmeans_k{v.k}_a{v.alpha}_{v.ref}'
            self.run(twin)
            sizes = np.bincount(self._groups_of(twin), minlength=v.k)
        corrector = self._corrector(v, sizes)
        Zc = StreamRunner(100).apply(corrector, self.Z, self.domain_ids if v.kind == 'oracle' else None)
        df = self.noadapt[['batch', 'domain_idx', 'sample_idx', 'label', 'position', 'domain']].copy()
        df['pred'] = self.predict(Zc)
        df['correct'] = df['pred'] == df['label']
        df['group'] = corrector.groups
        df.to_pickle(path)
        self.outputs[name], self.groups[name] = df, corrector.groups
        return df

    def _groups_of(self, name):
        if name not in self.groups:
            self.groups[name] = self.run(name)['group'].to_numpy()
        return self.groups[name]

    # ------------------------------------------------------------------ checks and tables
    def sanity_check(self):
        """alpha = 0 (NoShift) vs the cached NoAdapt: fraction of identical predictions (fp16 ties may differ)."""
        df = self.run('none', use_cache=False)
        same = (df['pred'].to_numpy() == self.noadapt['pred'].to_numpy()).mean()
        acc = domain_average(df), domain_average(self.noadapt)
        print(f'{self.dataset}: identity replay == cached NoAdapt on {same:.4%} of {len(df):,} images; '
              f'domain-avg acc {acc[0]:.2f} vs cached {acc[1]:.2f}')
        return same

    def summary(self, names):
        base, ramen = domain_average(self.noadapt), domain_average(self.ramen)
        rows = [dict(variant='NoAdapt (cached)', acc=base), dict(variant='Ramen (cached)', acc=ramen)]
        for n in names:
            rows.append(dict(variant=n, acc=domain_average(self.run(n))))
        t = pd.DataFrame(rows)
        t['vs_noadapt'] = t['acc'] - base
        t['pct_of_ramen_gain'] = 100 * t['vs_noadapt'] / (ramen - base)
        return t
