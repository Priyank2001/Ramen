"""
Ramen with an NSS feature-vector regulariser on the active-set selection (follow-up to RamenQuality).

RamenQuality matched images on their scalar NIQE / BRISQUE scores, which merge distortion types (contrast, JPEG and
pixelate score alike; new_experiments/quality_active_set.md). Here an entry's selection score uses the natural scene
statistics (NSS) feature vectors behind those scores instead:

    s_bj = d_bj + lambda * R_bj,        R_bj = || w(v_b) - w(v_j) ||_2 / sqrt(k)

v = the image's NSS vector (config `nss_features`), k its length:
  brisque  BRISQUE's 36 features: GGD fit to the MSCN coefficients and AGGD fits to their 4 paired products, 2 scales
  niqe     NIQE's 36 block features, averaged over the image's 96 px blocks
  both     the two concatenated (72)
w = a causal normalisation, fitted on the test images seen so far including the current batch (`nss_norm`):
  white    PCA whitening with the stream's running covariance (default)
  z        per-feature z-score
lambda = `nss_lambda`; 0 is Ramen exactly. RamenQuality's scalar-score term (`quality_lambda`) still applies, and
defaults to 0. Selection, weights, diagnostics and the stored true labels (diagnostics only) are RamenQuality's.

Memory: NSSCache, i.e. QualityCache (Ramen's class queue + NIQE, BRISQUE and true label per entry) + the NSS vector.
"""
import math

import numpy as np
import torch

from .losses import softmax_entropy
from .RamenQuality import QualityCache, RamenQuality

NSS_FEATURES = {'brisque': slice(0, 36), 'niqe': slice(36, 72), 'both': slice(0, 72)}  # of iqa.with_features()


class NSSCache(QualityCache):
    """QualityCache that also stores each entry's 72-d NSS vector (BRISQUE's 36 | NIQE's 36 block-mean features)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nss = torch.full((self.max_capacity, 72), float('nan'), device=self.device)

    def stored_nss(self):
        return self.nss[:self.size]

    def reset(self):
        super().reset()
        self.nss.fill_(float('nan'))


class StreamNormaliser:
    """Running mean and covariance of the vectors seen so far -> whitening (or z-scoring) map. NaN -> the mean."""

    def __init__(self, dim, how, device):
        if how not in ('white', 'z'):
            raise ValueError(f"Unknown nss_norm {how!r}; choose 'white' or 'z'")
        self.how, self.n = how, 0
        self.s1 = torch.zeros(dim, dtype=torch.float64, device=device)
        self.s2 = torch.zeros((dim, dim), dtype=torch.float64, device=device)
        self.mean = torch.zeros(dim, dtype=torch.float64, device=device)
        self.map = torch.eye(dim, dtype=torch.float64, device=device)

    def update(self, x):
        x = x[~torch.isnan(x).any(dim=1)].double()
        if len(x) == 0:
            return
        self.n += len(x)
        self.s1 += x.sum(dim=0)
        self.s2 += x.T @ x
        self.mean = self.s1 / self.n
        cov = self.s2 / self.n - torch.outer(self.mean, self.mean)
        if self.how == 'z':
            self.map = torch.diag(1 / cov.diagonal().clamp_min(1e-12).sqrt())
        else:
            val, vec = torch.linalg.eigh(cov)
            self.map = vec / val.clamp_min(val.max() * 1e-6).sqrt()  # same floor as nss_active_set.normalise

    def __call__(self, x):
        x = torch.where(torch.isnan(x), self.mean.to(x.dtype).expand_as(x), x)
        return ((x.double() - self.mean) @ self.map).float()


class RamenNSS(RamenQuality):

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        features = self.cfg.get('nss_features', 'both')
        if features not in NSS_FEATURES:
            raise ValueError(f'Unknown nss_features {features!r}; choose from {list(NSS_FEATURES)}')
        self.nss_slice = NSS_FEATURES[features]
        self.nss_lam = float(self.cfg.get('nss_lambda', 0.0))
        norm = self.cfg.get('nss_norm', 'white')
        dim = self.nss_slice.stop - self.nss_slice.start
        self.normaliser = StreamNormaliser(dim, norm, self.device)
        self.cache = [NSSCache(max_capacity=self.cfg['max_capacity'], key_dim=self.feat_dim, value_dim=self.grad_dim,
                               device=self.device, dtype=self.dtype)
                      for _ in range(self.num_classes)]
        self.stats.update({'nss_features': features, 'nss_norm': norm, 'nss_lambda': self.nss_lam})
        self._nss = self._q_norm = None  # the current batch's raw and normalised NSS vectors

    def measure(self, x):
        scores, self._nss = self.iqa.with_features(x)
        v = self._nss[:, self.nss_slice]
        self.normaliser.update(v)  # causal: the stream so far, including this batch
        self._q_norm = self.normaliser(v)
        return scores

    @torch.no_grad()
    def write(self, feats, grads, zs_logits, ids, quality, labels):
        """RamenQuality.write, also storing each entry's NSS vector."""
        B = feats.shape[0]
        zs_pred = zs_logits.argmax(-1)
        priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.dtype)
        entropies = softmax_entropy(zs_logits, reduction='none')
        self.counter = self.counter + B
        for b in range(B):
            self.cache[zs_pred[b]].add(feats[b:b + 1], grads[b:b + 1], entropies[b:b + 1], priorities[b:b + 1],
                                       ids[b:b + 1], quality[b:b + 1], labels[b:b + 1], nss=self._nss[b:b + 1])
        self.stats['filed_as'].append(zs_pred.cpu().numpy().astype(np.int32))

    def class_penalty(self, cache, q_rank, e_rank):
        pen = super().class_penalty(cache, q_rank, e_rank)  # the scalar-score term, None when quality_lambda = 0
        if self.nss_lam == 0:
            return pen
        e_norm = self.normaliser(cache.stored_nss()[:, self.nss_slice])
        r = self.nss_lam * torch.cdist(self._q_norm, e_norm) / math.sqrt(e_norm.shape[1])
        return r if pen is None else pen + r
