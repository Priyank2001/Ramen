"""
What to run: per-dataset defaults (backbone, fixed batch, RAM budget) and an immutable, validated RunConfig.

The fixed batch per dataset (D18) is the largest that fits the 11 GB GPU 1 with Ramen's memory full. It is found
once by the week-01 batch search and frozen here; `None` means "not measured yet", and a run must then pass its
batch explicitly.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

MAX_WORKERS = 4  # DataLoader worker cap (D17: each worker is a process holding its own copy of Python objects)
TTA_MODES = ("mixed", "single")
# main: the test domains (upstream default). heldout: only the extra corruptions upstream loads with extra=True
# (src/datasets/corruption/utils.py:extra_corruptions), for tuning (AGENTS.md rule 8). Corruption sets only.
DOMAIN_SETS = ("main", "heldout")
HELDOUT_DATASETS = ("CIFAR10C", "CIFAR100C", "ImageNetC5K")


@dataclass(frozen=True)
class DatasetSpec:
    """Defaults for one dataset. `short` goes into run ids."""
    name: str
    short: str
    backbone: str
    batch_size: Optional[int]  # frozen fixed batch (D18); None until measured
    ram_budget_gb: float  # process-tree RAM budget per run (D17); 1.5 x the measured peak once known


DATASETS: Dict[str, DatasetSpec] = {spec.name: spec for spec in (
    # 2026-11-01 batch search on the 2080 Ti, memory full: peak 9.18 of 10.76 GB GPU, 5.48 GB RAM (4 workers)
    DatasetSpec("CIFAR10C", "C10", "clip_vitbase32", batch_size=100, ram_budget_gb=8.2),
    # 2026-11-04 batch search, same method (candidates 100, 64, 50, 48, 32, 24; fits = peak <= 95% of the 10.76 GB):
    # CIFAR-100-C: 100/64/50/48 OOM, 32 fits (9.63 GB GPU, 6.51 GB RAM). ImageNet-C: 32 does not fit (10.49 GB),
    # 24 fits (9.81 GB, 4.00 GB RAM). DomainNet: 50 and 48 do not fit (10.62, 10.63 GB), 32 fits (10.18 GB, 4.46 GB RAM).
    # RAM budget = 2 x the search peak (the 1.5 x used for CIFAR-10-C left little margin: full run 7.3 of 8.2 GB).
    DatasetSpec("CIFAR100C", "C100", "clip_vitbase16", batch_size=32, ram_budget_gb=13.0),
    DatasetSpec("ImageNetC5K", "INC", "clip_vitbase16", batch_size=24, ram_budget_gb=8.0),
    DatasetSpec("DomainNet", "DN", "clip_vitbase32", batch_size=32, ram_budget_gb=9.0),
)}


@dataclass(frozen=True)
class RunConfig:
    """One run. `name` labels the variant (defaults to the method); `overrides` are merged over the upstream yaml."""
    dataset: str
    method: str = "Ramen"
    seed: int = 0
    name: Optional[str] = None
    batch_size: Optional[int] = None  # None: the dataset's frozen batch
    backbone: Optional[str] = None  # None: the dataset's default
    overrides: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    tta_mode: str = "mixed"
    n_per_domain: Optional[int] = None  # None: the full stream
    domains: str = "main"  # DOMAIN_SETS
    num_workers: int = MAX_WORKERS
    data_root: str = "~/data"

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"unknown dataset {self.dataset!r}; known: {', '.join(DATASETS)}")
        if self.tta_mode not in TTA_MODES:
            raise ValueError(f"tta_mode must be one of {TTA_MODES}: {self.tta_mode!r}")
        if not 0 <= self.num_workers <= MAX_WORKERS:
            raise ValueError(f"num_workers must be in [0, {MAX_WORKERS}] (RAM safety, D17): {self.num_workers}")
        if self.domains not in DOMAIN_SETS:
            raise ValueError(f"domains must be one of {DOMAIN_SETS}: {self.domains!r}")
        if self.domains == "heldout" and self.dataset not in HELDOUT_DATASETS:
            raise ValueError(f"{self.dataset} has no held-out domains; held-out sets exist for {HELDOUT_DATASETS}")
        if self.seed < 0 or (self.n_per_domain is not None and self.n_per_domain <= 0):
            raise ValueError("seed must be >= 0 and n_per_domain > 0")
        if self.resolved_batch_size is None:
            raise ValueError(f"no frozen batch for {self.dataset} yet (config.DATASETS): pass batch_size explicitly")

    @classmethod
    def make(cls, dataset: str, overrides: Optional[Dict[str, Any]] = None, **kwargs) -> "RunConfig":
        """Build from a plain dict of overrides (stored sorted, so equal configs compare equal)."""
        return cls(dataset=dataset, overrides=tuple(sorted((overrides or {}).items())), **kwargs)

    @property
    def spec(self) -> DatasetSpec:
        return DATASETS[self.dataset]

    @property
    def resolved_batch_size(self) -> Optional[int]:
        return self.batch_size or self.spec.batch_size

    @property
    def resolved_backbone(self) -> str:
        return self.backbone or self.spec.backbone

    @property
    def variant(self) -> str:
        return self.name or self.method

    def run_id(self, machine: str) -> str:
        """e.g. pc2_C10_Ramen_b64_s0, pc2_C10_Ramen_b64_s0_n200 for a subset, pc2_C10_Ramen_b64_s0_heldout."""
        subset = f"_n{self.n_per_domain}" if self.n_per_domain else ""
        heldout = "_heldout" if self.domains == "heldout" else ""
        return f"{machine}_{self.spec.short}_{self.variant}_b{self.resolved_batch_size}_s{self.seed}{subset}{heldout}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(overrides=dict(self.overrides), batch_size=self.resolved_batch_size, backbone=self.resolved_backbone)
        if self.domains == "main":
            del d["domains"]  # main runs keep the config dict of runs made before `domains` existed (Runner skip check)
        return d
