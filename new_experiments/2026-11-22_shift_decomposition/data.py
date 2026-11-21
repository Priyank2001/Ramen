"""
Cached inputs of the shift decomposition (offline, read-only): CLIP image features, class text embeddings, labels,
domain ids, leave-out groups, and the cached NoAdapt / Ramen predictions, all row-aligned in seed 0's stream order.

    d = ShiftData('ImageNetC5K')
    d.Z (N, D) float32 unit-norm image features    d.W (C, D) float32 unit-norm text embeddings
    d.y, d.dom (N,) int labels / domain ids         d.group (N,) int leave-out group (photo on ImageNet-C, else image)
    d.noadapt_pred, d.ramen_correct                 cached runs (seed 0)
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(HERE)
RESULTS = os.path.join(NEW, 'results')
TEXT = os.path.join(NEW, '2026-11-21_hubness', 'results')

MODEL = {'ImageNetC5K': 'clip_vitbase16', 'DomainNet': 'clip_vitbase32', 'ImageNetVariants': 'clip_vitbase16',
         'EuroSAT': 'clip_vitbase16'}
LABEL = {'ImageNetC5K': 'ImageNet-C 5K', 'DomainNet': 'DomainNet', 'ImageNetVariants': 'ImageNet-R/A/Sketch',
         'EuroSAT': 'EuroSAT'}
FLOOR = {'ImageNetC5K': 0.54, 'DomainNet': 0.11, 'ImageNetVariants': 0.48, 'EuroSAT': 1.51}
PHOTO_GROUPS = ('ImageNetC5K',)   # the same photo appears once under every corruption


class ShiftData:
    def __init__(self, dataset, seed=0):
        self.dataset, self.seed = dataset, seed
        prefix = os.path.join(RESULTS, f'{dataset}_{MODEL[dataset]}_mixed_nall_bs100_s{seed}')
        self.noadapt = pd.read_pickle(f'{prefix}_NoAdapt.pkl')
        self.ramen = pd.read_pickle(f'{prefix}_Ramen.pkl')
        assert (self.noadapt.position.values == np.arange(len(self.noadapt))).all()
        assert (self.noadapt.sample_idx.values == self.ramen.sample_idx.values).all()
        assert (self.noadapt.domain_idx.values == self.ramen.domain_idx.values).all()
        Z = np.load(f'{prefix}_NoAdapt_feats.npy').astype(np.float32)
        self.Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        W = np.load(os.path.join(TEXT, f'{dataset}_text_features.npy')).astype(np.float32)
        self.W = W / np.linalg.norm(W, axis=1, keepdims=True)
        self.y = self.noadapt.label.values.astype(np.int64)
        self.dom = self.noadapt.domain_idx.values.astype(np.int64)
        names = self.noadapt.groupby('domain_idx', observed=True).domain.first()
        self.domain_names = [str(names[d]) for d in range(self.dom.max() + 1)]
        self.C, self.n_dom = self.W.shape[0], len(self.domain_names)
        self.group = self._groups()
        self.noadapt_pred = self.noadapt.pred.values
        self.noadapt_correct = self.noadapt.correct.values.astype(bool)
        self.ramen_correct = self.ramen.correct.values.astype(bool)

    def _groups(self):
        if self.dataset not in PHOTO_GROUPS:
            return np.arange(len(self.y))
        g = self.noadapt.sample_idx.values.astype(np.int64)
        # every photo: one label, at most one copy per domain (else the within-cell leave-out would be wrong)
        df = pd.DataFrame(dict(g=g, y=self.y, d=self.dom))
        assert (df.groupby('g').y.nunique() == 1).all()
        assert not df.duplicated(['g', 'd']).any()
        return g

    def describe(self):
        counts = np.bincount(self.dom, minlength=self.n_dom)
        cells = np.zeros((self.C, self.n_dom), np.int64)
        np.add.at(cells, (self.y, self.dom), 1)
        rows = [dict(domain=n, images=int(counts[d]), classes_present=int((cells[:, d] > 0).sum()),
                     median_per_class=float(np.median(cells[cells[:, d] > 0, d])),
                     zero_shot_errors=int((~self.noadapt_correct[self.dom == d]).sum()))
                for d, n in enumerate(self.domain_names)]
        return pd.DataFrame(rows)
