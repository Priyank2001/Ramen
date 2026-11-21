"""
Stream data for the hubness check: CLIP features, class text embeddings, labels, batches (offline, from caches).

    sd = StreamData('ImageNetC5K', seed=0)             # zero-shot features in seed 0's stream order
    sd = StreamData('EuroSAT', seed=0, level='ramen')  # the adapted features saved by the cached Ramen run (seed 0 only)
sd.Z (n, D) float32 on GPU, sd.W (C, D) L2-normalised, sd.df (labels, domain_idx, batch, pred of the cached Ramen run),
sd.zs_pred (cached NoAdapt prediction per row), sd.logit_pred (the run being replayed: NoAdapt or Ramen).
Text embeddings: CLIP text tower with the Ramen config's templates, computed once per dataset and cached in results/.
"""
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '2026-10-25_stacking'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import stack  # noqa: E402  (read-only: cache paths, model per dataset, zero-shot feature alignment)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASETS = ['ImageNetC5K', 'DomainNet', 'ImageNetVariants', 'CIFAR10C', 'CIFAR100C', 'EuroSAT']
FLOOR = {**stack.NOISE_FLOOR, 'ImageNetVariants': 0.48}
RAMEN_FEATS = ['ImageNetC5K', 'DomainNet', 'CIFAR10C', 'CIFAR100C', 'EuroSAT']  # Ramen features saved for seed 0
KEY = 10**7


def text_features(dataset):
    """(C, D) float32 L2-normalised class embeddings, as Ramen classifies against them (cached under results/)."""
    path = os.path.join(HERE, 'results', f'{dataset}_text_features.npy')
    if not os.path.exists(path):
        from ramen_analysis import make_experiment
        from models.clip_utils import DEFAULT_TEMPLATES, encode_text
        from models.get_pretrained_model import get_pretrained_model
        exp = make_experiment(dataset, tta_mode='mixed', batch_size=100, n_per_domain=None, seed=0, num_workers=0).load()
        templates = exp._load_cfg('Ramen').get('templates') or DEFAULT_TEMPLATES
        with torch.no_grad():
            t = encode_text(exp.clip, list(exp.classes), templates)
        np.save(path, t.float().cpu().numpy())
    W = np.load(path).astype(np.float32)
    return W / np.linalg.norm(W, axis=1, keepdims=True)


class StreamData:

    def __init__(self, dataset, seed, level='zero-shot'):
        self.dataset, self.seed, self.level = dataset, seed, level
        self.df = pd.read_pickle(stack.path(dataset, seed, 'Ramen') + '.pkl')  # stream order, labels, batches
        assert (self.df.position.values == np.arange(len(self.df))).all()
        self.label = self.df.label.values
        self.domain = self.df.domain_idx.values
        self.batch = self.df.batch.values
        z0 = pd.read_pickle(stack.path(dataset, 0, 'NoAdapt') + '.pkl')
        key0 = pd.Series(z0.pred.values, index=z0.domain_idx.values * KEY + z0.sample_idx.values)
        self.zs_pred = key0.loc[self.domain * KEY + self.df.sample_idx.values].values  # cached NoAdapt prediction
        if level == 'zero-shot':
            Z = stack.zero_shot_keys(dataset, self.df)
            self.logit_pred = self.zs_pred
        elif level == 'ramen':
            assert seed == 0, 'adapted features are saved for seed 0 only'
            Z = np.load(stack.path(dataset, 0, 'Ramen') + '_feats.npy')
            self.logit_pred = self.df.pred.values
        else:
            raise ValueError(level)
        self.Z = torch.tensor(Z, dtype=torch.float32, device=DEVICE)
        self.Z = self.Z / self.Z.norm(dim=1, keepdim=True)
        self.W = torch.tensor(text_features(dataset), device=DEVICE)
        self.classes = self.W.shape[0]

    def sanity(self):
        """
        (% of rows where argmax z.W equals the cached run's prediction, number that differ, largest top-2 logit gap
        among the differing rows). The saved features are fp16, so rows with a tie-level gap (< 0.05 in logit units)
        may flip; any other difference is a bug.
        """
        pred, gap = [], []
        for i in range(0, len(self.Z), 20000):
            top = (100 * self.Z[i:i + 20000] @ self.W.T).topk(2, dim=1)
            pred.append(top.indices[:, 0]), gap.append(top.values[:, 0] - top.values[:, 1])
        pred, gap = torch.cat(pred).cpu().numpy(), torch.cat(gap).cpu().numpy()
        diff = pred != self.logit_pred
        return (~diff).mean() * 100, int(diff.sum()), float(gap[diff].max()) if diff.any() else 0.0

    def batches(self):
        starts = np.flatnonzero(np.r_[True, self.batch[1:] != self.batch[:-1]])
        return list(zip(starts, np.r_[starts[1:], len(self.batch)]))
