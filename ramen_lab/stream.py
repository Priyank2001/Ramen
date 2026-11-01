"""
The order in which test images arrive. A seed changes only this order: an explicit torch.Generator(seed) drives the
shuffle, so every method run with the same seed sees exactly the same stream. Upstream main.py instead shuffles from
the global RNG, whose state depends on what ran before.

Subsets (n per domain) are drawn with a fixed generator (SUBSET_SEED), independent of the run's seed, so different
seeds shuffle the same images.
"""
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from torch import Generator
from torch.utils.data import DataLoader, Dataset

SUBSET_SEED = 12345


class TaggedStream(Dataset):
    """Images of several domains as one dataset; yields (image, label, domain_idx, sample_idx), like upstream's
    TaggedMultipleDataset, but over any subset of (domain, sample) pairs."""

    def __init__(self, domains: Sequence[Dataset], index: Sequence[Tuple[int, int]]):
        self.domains = domains
        self.index = list(index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        domain_idx, sample_idx = self.index[i]
        image, label = self.domains[domain_idx][sample_idx]
        return image, label, domain_idx, sample_idx


class StreamFactory:
    """Builds the DataLoader(s) for one run."""

    def __init__(self, batch_size: int, seed: int, num_workers: int, n_per_domain: Optional[int] = None):
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.n_per_domain = n_per_domain

    def domain_index(self, domains: Sequence[Dataset], domain_idx: int) -> List[Tuple[int, int]]:
        n = len(domains[domain_idx])
        samples = np.arange(n)
        if self.n_per_domain is not None and self.n_per_domain < n:
            rng = np.random.default_rng(SUBSET_SEED + domain_idx)
            samples = np.sort(rng.choice(n, size=self.n_per_domain, replace=False))
        return [(domain_idx, int(s)) for s in samples]

    def loader(self, domains: Sequence[Dataset], index: Sequence[Tuple[int, int]]) -> DataLoader:
        return DataLoader(TaggedStream(domains, index), batch_size=self.batch_size, shuffle=True,
                          generator=Generator().manual_seed(self.seed), num_workers=self.num_workers, pin_memory=True)

    def mixed(self, datasets: Any) -> DataLoader:
        """All domains shuffled into one stream (Ramen's main setting)."""
        domains = datasets.datasets
        index = [pair for d in range(len(domains)) for pair in self.domain_index(domains, d)]
        return self.loader(domains, index)

    def single(self, datasets: Any) -> List[DataLoader]:
        """One stream per domain; the method is reset between them."""
        domains = datasets.datasets
        return [self.loader(domains, self.domain_index(domains, d)) for d in range(len(domains))]
