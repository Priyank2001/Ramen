"""
Ramen that records, for every test image, what the model saw and predicted before and after adaptation.
It changes nothing in Ramen: predictions must be 100% the same as Ramen's (sanity check).

Per image (row i <-> row i of the harness's per-sample DataFrame, i.e. stream order):
  zs_pred      argmax of the zero-shot logits (int32, exact; argmax of the float16 zs_probs can tie)
  zs_probs     softmax of the zero-shot logits (the pass before adaptation, the one that files the image in memory)
  probs        softmax of the returned logits (Ramen's prediction, after its one-step adaptation)
  emb_before   L2-normalised CLIP image feature with the pretrained norm parameters (= zero-shot feature)
  emb_after    L2-normalised CLIP image feature with the adapted norm parameters (the one Ramen classifies)
The vectors are stored as float16; save_log writes them to <run prefix>_rec_<key>.npy.

How it records: Ramen's forward calls model.featurize once before the update and once after it (inside
model(x)). RecorderMixin wraps featurize and keeps the first and last call of each batch, so the method's own
forward is never copied or edited. Any variant whose forward follows that pattern can be recorded the same way:

    class RamenActiveRecorder(RecorderMixin, RamenActive): pass
"""
import os

import numpy as np
import torch

from .Ramen import Ramen
from .RamenTrueLabel import RamenTrueLabel

RECORD_KEYS = ('zs_pred', 'zs_probs', 'probs', 'emb_before', 'emb_after')


class RecorderMixin:
    """Put before a Ramen-like method in the bases; records RECORD_KEYS for every image."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = {k: [] for k in RECORD_KEYS}
        self._feature_calls = []
        featurize = self.model.featurize

        def recording_featurize(x):
            f = featurize(x)
            self._feature_calls.append(f.detach())
            return f

        self.model.featurize = recording_featurize  # model(x) looks featurize up on the instance, so this is used

    def forward(self, x, *args, **kwargs):
        self._feature_calls.clear()
        logits = super().forward(x, *args, **kwargs)
        if len(self._feature_calls) < 2:
            raise RuntimeError(f'{type(self).__name__}: expected featurize before and after adaptation, '
                               f'got {len(self._feature_calls)} call(s)')
        before, after = self._feature_calls[0], self._feature_calls[-1]
        with torch.no_grad():
            zs_logits = self.model.classify(before)
            batch = {'zs_probs': zs_logits.float().softmax(dim=1), 'probs': logits.float().softmax(dim=1),
                     'emb_before': before, 'emb_after': after}
            for k, v in batch.items():
                self.records[k].append(v.float().cpu().numpy().astype(np.float16))
            self.records['zs_pred'].append(zs_logits.argmax(-1).cpu().numpy().astype(np.int32))
        self._feature_calls.clear()
        return logits

    def save_log(self, prefix):
        """Called by the harness after the run: one .npy per record key."""
        for k, parts in self.records.items():
            np.save(f'{prefix}_rec_{k}.npy', np.concatenate(parts))

    @staticmethod
    def record_paths(prefix):
        return {k: f'{prefix}_rec_{k}.npy' for k in RECORD_KEYS}

    @classmethod
    def has_records(cls, prefix):
        return all(os.path.exists(p) for p in cls.record_paths(prefix).values())


class RamenRecorder(RecorderMixin, Ramen):
    pass


class RamenTrueLabelRecorder(RecorderMixin, RamenTrueLabel):
    pass  # ORACLE when filing=true_label (uses the true labels; records_labels is inherited)
