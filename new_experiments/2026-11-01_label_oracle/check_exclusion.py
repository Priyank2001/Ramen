"""
ImageNet-C: does `exclude_groups` hide an image's other corrupted copies of the same photo, and how much of the
update did those copies carry without it? Replays the first N batches of the real stream with a watching subclass.

  X1  with exclusion: retrieved entries (non-zero weight) that are another copy of the query's photo -> must be 0
  X2  without exclusion (same cache state): share of the update weight on other copies of the query's photo,
      for per-class top-1 (Ramen's ImageNet-C retrieval) and for global top-k
  X3  the traced batches predict what the cached full run predicted (once that run exists)

    CUDA_VISIBLE_DEVICES=2 python check_exclusion.py --batches 200 | tee logs/check_exclusion_ImageNetC5K.log
"""
import sys

import numpy as np
import torch

from label_oracle import VARIANTS, LabelOracleExperiment  # first: sets up the import paths
from label_methods import RamenGlobalRetrieval
from trace_active_set import frac, traced_run


class TracedExclusion(RamenGlobalRetrieval):
    """Records, per batch, copies retrieved with and without exclusion; behaviour unchanged."""

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.x1, self.x2_per_class, self.x2_global = [], [], []

    @torch.no_grad()
    def retrieve(self, feats, grads):
        update = super().retrieve(feats, grads)
        B, first = feats.shape[0], self._batch_first
        qpos = torch.arange(first, first + B, device=self.device)
        qg = self.groups[qpos]
        classes = [c for c in range(self.num_classes) if self.cache[c].size > 0]
        q = feats.detach().to(device=self.device, dtype=self.dtype)
        rows = []
        for c in classes:
            cache = self.cache[c]
            pos = self.slot_pos[c, :cache.size]
            d = torch.cdist(q, cache.keys[:cache.size], compute_mode='use_mm_for_euclid_dist').float()
            copy = (self.groups[pos][None] == qg[:, None]) & (pos[None] != qpos[:, None])
            rows.append((d, copy, cache.entropies[:cache.size].float()))
        # X1: the method's own masked distances never select a copy
        hit = 0
        for (d, copy, h), c in zip(rows, classes):
            dm = self._distances(feats, c)
            k = min(self.global_k if self.retrieval == 'global_knn' else self.cfg['topk'], dm.shape[1])
            if self.retrieval == 'per_class':
                sd, idx = dm.topk(k, dim=1, largest=False)
                hit += int((copy.gather(1, idx) & torch.isfinite(sd)).sum())
        if self.retrieval == 'global_knn':
            dm_all = torch.cat([self._distances(feats, c) for c in classes], 1)
            sd, idx = dm_all.topk(min(self.global_k, dm_all.shape[1]), dim=1, largest=False)
            hit = int((torch.cat([r[1] for r in rows], 1).gather(1, idx) & torch.isfinite(sd)).sum())
        self.x1.append(hit)
        # X2: without exclusion, how much weight would copies get
        w_pc, w_pc_copy = torch.zeros(B, device=self.device), torch.zeros(B, device=self.device)
        for d, copy, h in rows:
            sd, idx = d.topk(1, dim=1, largest=False)
            w = torch.exp(-h[idx]).squeeze(1) * torch.exp(-self.beta * sd).squeeze(1)
            w_pc += w
            w_pc_copy += w * copy.gather(1, idx).squeeze(1)
        self.x2_per_class.append(float((w_pc_copy / w_pc).mean()))
        d_all, c_all = torch.cat([r[0] for r in rows], 1), torch.cat([r[1] for r in rows], 1)
        h_all = torch.cat([r[2] for r in rows])
        sd, idx = d_all.topk(min(self.global_k, d_all.shape[1]), dim=1, largest=False)
        w = torch.exp(-h_all[idx]) * torch.exp(-self.beta * sd)
        self.x2_global.append(float(((w * c_all.gather(1, idx)).sum(1) / w.sum(1)).mean()))
        return update


def report(n_batches, names=('excl_zs_ce', 'excl_global_ce_k10')):
    exp = LabelOracleExperiment('ImageNetC5K', seed=0)
    exp.exp.load()
    exp.loaded = True
    groups = exp.base_image_groups()
    for name in names:
        cfg = {**VARIANTS[name][1], 'exclude_groups': groups}
        VARIANTS[name] = (VARIANTS[name][0], cfg, VARIANTS[name][2])   # resolved for traced_run (local copy only)
        method, preds, keys = traced_run(exp, name, n_batches, cls=TracedExclusion)
        late = slice(n_batches // 2, None)
        print(f'\n=== {name} (retrieval {method.retrieval}, k {method.global_k if method.retrieval == "global_knn" else method.cfg["topk"]}), first {n_batches} batches')
        print(f"X1  retrieved entries that are another copy of the query's photo: {sum(method.x1):,} (must be 0)")
        print(f'X2  without exclusion, mean share of the update on other copies: per-class top-1 '
              f'{np.mean(method.x2_per_class):.1%} (second half of the traced batches {np.mean(method.x2_per_class[late]):.1%}); '
              f'global top-{method.global_k} {np.mean(method.x2_global):.1%} (second half {np.mean(method.x2_global[late]):.1%})')
        if exp.is_cached(name):
            ref = exp.exp.run(name).iloc[:len(preds)]
            print(f'X3  traced prediction == cached full run: {frac(preds == ref.pred.to_numpy())} images')
        method.cache = None
        del method.model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    n = int(sys.argv[sys.argv.index('--batches') + 1]) if '--batches' in sys.argv else 200
    report(n)
