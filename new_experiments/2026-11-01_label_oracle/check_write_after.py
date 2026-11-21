"""
Evidence for RamenWriteAfterPrediction (filing by the adapted prediction): replays the first N batches of the real
stream with a watching subclass and checks

  W0  (once the full runs exist) the traced batches predict what the cached full run predicted
  W1  the first batch, with an empty cache, is not adapted: its predictions are the zero-shot ones
  W2  retrieval sees earlier batches only: no entry of the current batch is in the cache when it retrieves
  W3  each image is written to the queue of the class the method RETURNED for it (adapted filing), or of its
      zero-shot class (zero-shot filing); checked against the returned logits, not the method's own variable
  W4  each written slot holds that image's key and the value the rule says (entropy or label gradient)
  W5  adapted and zero-shot filing really differ: images filed differently, and queue sizes after N batches

    CUDA_VISIBLE_DEVICES=0 python check_write_after.py DomainNet 0 --batches 30 | tee logs/check_write_after_DomainNet.log
"""
import sys

import numpy as np
import torch

from label_oracle import LabelOracleExperiment  # first: sets up the import paths (src/)
from label_methods import RamenWriteAfterPrediction
from trace_active_set import frac, traced_run


class TracedWriteAfter(RamenWriteAfterPrediction):
    """RamenWriteAfterPrediction that records its writes and what the cache held at retrieval; behaviour unchanged."""

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.filed, self.zs, self.checks = {}, {}, {'W2': [], 'W4': []}

    def forward(self, x, labels=None):
        self._first = self.counter
        return super().forward(x, labels)

    @torch.no_grad()
    def retrieve(self, feats, grads):
        stored = sum(int(c.size) for c in self.cache)
        written_before = sum(1 for i in self.filed if i < self._first)
        self.checks['W2'].append(stored <= written_before and max(self.filed, default=-1) < self._first)
        return super().retrieve(feats, grads)

    @torch.no_grad()
    def write(self, feats, values, logits, queues):
        first = self.counter
        slots = super().write(feats, values, logits, queues)
        zs = logits.argmax(-1)
        for b, (c, slot) in enumerate(slots):
            self.filed[first + b], self.zs[first + b] = c, int(zs[b])
            if slot is not None:
                cache = self.cache[c]
                self.checks['W4'].append(torch.equal(cache.keys[slot], feats[b].to(cache.dtype)) and
                                         torch.equal(cache.values[slot], values[b].to(cache.dtype)))
        return slots


def report(dataset, seed, n_batches, names=('late_zs_entropy', 'late_adapted_entropy', 'late_adapted_ce')):
    exp = LabelOracleExperiment(dataset, seed=seed)
    exp.exp.load()
    exp.loaded = True
    runs = {}
    for name in names:
        method, preds, keys = traced_run(exp, name, n_batches, cls=TracedWriteAfter)
        ids = np.arange(len(preds))
        filed = np.array([method.filed[i] for i in ids])
        zs = np.array([method.zs[i] for i in ids])
        B = exp.exp.batch_size
        print(f'\n=== {name} (filing = {method.filing}, value = {method.stats["value"]}), first {n_batches} batches')
        if exp.is_cached(name):
            ref = exp.exp.run(name)  # cached
            ref = ref.iloc[:len(preds)]
            print(f'W0  traced prediction == cached full run: {frac(preds == ref.pred.to_numpy())} images '
                  f'(same images in order: {bool((ref[["domain_idx", "sample_idx"]].to_numpy() == keys).all())})')
        else:
            print('W0  full run not cached yet: skipped')
        print(f'W1  first batch adapted: {method.stats["adapted"][0]}; its predictions == zero-shot: '
              f'{frac(preds[:B] == zs[:B])}; later batches adapted: {frac(method.stats["adapted"][1:])}')
        print(f'W2  batches whose retrieval saw earlier batches only: {frac(method.checks["W2"])}')
        target = preds if method.filing == 'adapted' else zs
        print(f'W3  images written to the queue of their {"returned (adapted)" if method.filing == "adapted" else "zero-shot"} '
              f'class: {frac(filed == target)}')
        print(f'W4  written slots holding the image\'s key and value: {frac(method.checks["W4"])}')
        sizes = np.array([int(c.size) for c in method.cache])
        runs[name] = (filed, sizes)
        print(f'    images whose adapted prediction differs from zero-shot: {frac(preds != zs)}')
        method.cache = None
        del method.model
        torch.cuda.empty_cache()
    if {'late_zs_entropy', 'late_adapted_entropy'} <= set(runs):
        (fz, sz), (fa, sa) = runs['late_zs_entropy'], runs['late_adapted_entropy']
        print(f'\nW5  images filed in a different queue (adapted vs zero-shot filing): {frac(fz != fa)}; '
              f'queues whose size differs after {n_batches} batches: {frac(sz != sa)}')


if __name__ == '__main__':
    dataset, seed = sys.argv[1], int(sys.argv[2])
    n = int(sys.argv[sys.argv.index('--batches') + 1]) if '--batches' in sys.argv else 30
    report(dataset, seed, n)
