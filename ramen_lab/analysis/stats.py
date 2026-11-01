"""Statistics for comparing runs. Pure functions: arrays in, numbers out."""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def noise_floor(seed_accs: Sequence[float]) -> float:
    """2 x (max - min) of a method's mean accuracy over seeds (normally 0-4). Stricter than the std on purpose: a gain
    counts only if it is large compared with how far the baseline itself moves when only the stream order changes."""
    accs = np.asarray(seed_accs, dtype=float)
    if len(accs) < 2:
        raise ValueError("a noise floor needs at least two seeds")
    return float(2 * (accs.max() - accs.min()))


@dataclass(frozen=True)
class PairedGain:
    """Variant minus baseline, paired by seed."""
    gains: Tuple[float, ...]
    mean: float
    std: float
    n_positive: int
    floor: Optional[float]

    @property
    def clears_floor(self) -> Optional[bool]:
        return None if self.floor is None else self.mean >= self.floor


def paired_gain(variant: Sequence[float], baseline: Sequence[float], floor: Optional[float] = None) -> PairedGain:
    g = np.asarray(variant, dtype=float) - np.asarray(baseline, dtype=float)
    return PairedGain(tuple(g.round(4)), float(g.mean()), float(g.std(ddof=1)) if len(g) > 1 else 0.0,
                      int((g > 0).sum()), floor)


def cluster_bootstrap(diff: np.ndarray, domain: np.ndarray, cluster: np.ndarray, n: int = 2000,
                      seed: int = 0) -> Tuple[float, float]:
    """
    95% CI (points) of the headline statistic, the mean over domains of each domain's mean paired difference.
    Resamples clusters of samples that share a source image (on CIFAR-C and ImageNet-C one image appears under every
    corruption, so pass its sample index; on DomainNet each image is its own cluster) with Poisson(1) weights.
    """
    rng = np.random.default_rng(seed)
    codes = np.unique(cluster, return_inverse=True)[1]
    n_dom = int(domain.max()) + 1
    sums = np.zeros((codes.max() + 1, n_dom), dtype=np.float64)
    counts = np.zeros_like(sums)
    np.add.at(sums, (codes, domain), diff)
    np.add.at(counts, (codes, domain), 1)
    boot = []
    for _ in range(0, n, 100):  # 100 replicates at a time keeps the weight matrix small
        w = rng.poisson(1.0, size=(100, len(sums))).astype(np.float64)
        boot.append(((w @ sums) / (w @ counts)).mean(axis=1) * 100)
    low, high = np.percentile(np.concatenate(boot), [2.5, 97.5])
    return float(low), float(high)


def outcome_counts(variant_correct: np.ndarray, baseline_correct: np.ndarray) -> dict:
    """Per image, against the baseline: fixed (wrong -> right), broken (right -> wrong), both right, both wrong."""
    v, b = variant_correct.astype(bool), baseline_correct.astype(bool)
    return {"fixed": int((v & ~b).sum()), "broken": int((~v & b).sum()), "both right": int((v & b).sum()),
            "both wrong": int((~v & ~b).sum())}
