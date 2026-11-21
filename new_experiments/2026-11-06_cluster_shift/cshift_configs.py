"""
Datasets and variants of the cluster-shift experiment (plain data, no behaviour).

DATASETS: model of the cached seed-s runs (their stream order is the row order of the cached files).
VARIANTS: name -> Variant(kind, k, alpha, ref).
  kind  'none' (identity), 'kmeans' (label-free), 'random' (control for the 'kmeans' variant with the same k, alpha,
        ref), 'oracle' (ORACLE: true domain / corruption id as the group)
  ref   'text' (move group centres to the mean class text embedding) or 'global' (running mean of all embeddings)
"""
from dataclasses import dataclass

DATASETS = {
    'DomainNet': dict(model='clip_vitbase32'),
    'ImageNetC5K': dict(model='clip_vitbase16'),
    'EuroSAT': dict(model='clip_vitbase16'),
}


@dataclass(frozen=True)
class Variant:
    kind: str
    k: int = 1
    alpha: float = 1.0
    ref: str = 'text'


def _build():
    v = {'none': Variant('none', alpha=0.0)}
    for ref in ('text', 'global'):
        for k in (1, 5, 20, 50):
            if ref == 'global' and k == 1:
                continue                       # one group shifted to the global mean: the identity
            for a in (0.5, 1.0):
                v[f'kmeans_k{k}_a{a}_{ref}'] = Variant('kmeans', k, a, ref)
    for ref in ('text', 'global'):
        for k in (5, 20, 50):
            for a in (0.5, 1.0):
                v[f'random_k{k}_a{a}_{ref}'] = Variant('random', k, a, ref)
    for ref in ('text', 'global'):
        for a in (0.5, 1.0):
            v[f'oracle_domain_a{a}_{ref}'] = Variant('oracle', 0, a, ref)
    return v


VARIANTS = _build()
