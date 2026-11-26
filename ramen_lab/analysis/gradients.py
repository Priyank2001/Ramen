"""Pure gradient geometry: cosines between per-sample gradient matrices, whole and per parameter segment."""
from typing import Sequence

import torch
from torch import Tensor

EPS = 1e-30  # below this squared norm a gradient counts as zero (cosine undefined -> NaN)


def cosine(a: Tensor, b: Tensor) -> Tensor:
    """Row-wise cosine of two (B, D) matrices, computed in float32. NaN where either row is all zeros."""
    a, b = a.float(), b.float()
    na, nb = (a * a).sum(1), (b * b).sum(1)
    cos = (a * b).sum(1) / (na * nb).sqrt()
    return torch.where((na > EPS) & (nb > EPS), cos, torch.full_like(cos, float("nan")))


def segment_cosines(a: Tensor, b: Tensor, sizes: Sequence[int]) -> Tensor:
    """(B, S) cosines, one per consecutive column segment of widths `sizes` (sum(sizes) == D)."""
    if sum(sizes) != a.shape[1]:
        raise ValueError(f"segment sizes add up to {sum(sizes)}, but the matrices have {a.shape[1]} columns")
    return torch.stack([cosine(x, y) for x, y in zip(a.split(list(sizes), 1), b.split(list(sizes), 1))], dim=1)


def sign_agreement(a: Tensor, b: Tensor) -> Tensor:
    """(B,) fraction of coordinates where sign(a) == sign(b), among coordinates where both are non-zero."""
    sa, sb = a.float().sign(), b.float().sign()
    both = (sa != 0) & (sb != 0)
    agree = ((sa == sb) & both).float().sum(1)
    n = both.float().sum(1)
    return torch.where(n > 0, agree / n.clamp_min(1), torch.full_like(n, float("nan")))
