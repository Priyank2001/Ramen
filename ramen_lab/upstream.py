"""
The only module that imports upstream Ramen (src/, frozen at UPSTREAM_HEAD; docs/decisions.md D18).

Everything else in ramen_lab depends on the small functions below, never on src/ directly, so if upstream moves or
another code base is added, only this adapter changes. Upstream modules have generic top-level names (`methods`,
`models`, `datasets`), so src/ goes first on sys.path and every import is checked to resolve inside src/.
"""
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
CFG = REPO / "cfg"
UPSTREAM_HEAD = "ef0fdc7"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import datasets as _datasets  # noqa: E402  (upstream src/datasets)
import methods as _methods  # noqa: E402  (upstream src/methods)
from datasets.corruption import utils as _corruption_utils  # noqa: E402  (upstream src/datasets/corruption/utils.py)
from methods import losses as _losses  # noqa: E402  (upstream src/methods/losses.py)
from models import ModelForBySampleTTA as _by_sample  # noqa: E402  (upstream src/models/ModelForBySampleTTA.py)
from models import get_pretrained_model as _models  # noqa: E402  (upstream src/models)


def _check_is_upstream(module: ModuleType) -> None:
    origin = Path(module.__file__).resolve()
    if SRC not in origin.parents:
        raise ImportError(f"{module.__name__} resolved to {origin}, not to upstream {SRC}: a package name clash")


for _module in (_datasets, _methods, _losses, _by_sample, _models, _corruption_utils):
    _check_is_upstream(_module)

softmax_entropy = _losses.softmax_entropy  # upstream's entropy (Ramen's loss and memory weight)

UPSTREAM_METHODS = ("NoAdapt", "Tent", "Ramen")


class _TorchWithMmCdist:
    """Stands in for `torch` inside upstream's Ramen module, with one difference: `cdist` always takes the matrix-product
    path. Why: torch 2.0.1 picks a direct kernel when both inputs have <= 25 rows, and that kernel has no fp16 CUDA
    implementation, so Ramen's `PriorityCache.query` (fp16 keys) crashes for a batch of <= 25 images while a queue
    holds <= 25 entries (ImageNet-C at batch 24, or a short last batch). With either input > 25 rows torch's default
    already takes the matrix-product path, so every call that used to work returns exactly what it did before
    (tests/test_lab_upstream_patch.py)."""

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)

    @staticmethod
    def cdist(x1: torch.Tensor, x2: torch.Tensor, p: float = 2.0) -> torch.Tensor:
        return torch.cdist(x1, x2, p=p, compute_mode="use_mm_for_euclid_dist")


def _patch_small_batch_cdist() -> None:
    module = sys.modules["methods.Ramen"]  # `methods.Ramen` as an attribute is the class, so go through sys.modules
    _check_is_upstream(module)
    module.torch = _TorchWithMmCdist()


_patch_small_batch_cdist()


def method_class(name: str) -> type:
    """An upstream TTA method class (NoAdapt, Tent, Ramen)."""
    return _methods.get_method_class(name)


def by_sample_clip_model(model: torch.nn.Module, class_names: Any, cfg: Dict[str, Any], args: SimpleNamespace) -> Any:
    """Upstream's CLIPModelForBySampleTTA, built exactly as `Ramen.__init__` builds it (per-sample LayerNorm affine copies,
    SignSGD over them). Replaces the LayerNorms of `model` in place."""
    return _by_sample.CLIPModelForBySampleTTA(model, class_names, cfg, args)


@torch.no_grad()
def rearm_by_sample_grads(by_sample_model: Any, batch_size: int) -> None:
    """Make a by-sample model ready for another `set_by_sample_grad` + step on the same batch, as if a backward had just
    run. Needed because `reset_parameters()` clears `recent_B` and the optimizer's `zero_grad()` (torch 2.x default
    set_to_none) drops the grad tensors; rows >= batch_size get zero gradient, exactly what a backward leaves there."""
    for module in by_sample_model.model.modules():
        if isinstance(module, (_by_sample.BySampleLayerNorm, _by_sample.BySampleBatchNorm)):
            module.recent_B = batch_size
            for param in (module.weight_by_sample, module.bias_by_sample):
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                else:
                    param.grad.zero_()


def load_model(backbone: str, device: torch.device) -> Tuple[torch.nn.Module, Any]:
    """A fresh CLIP model and its preprocessing transform. Fresh matters: upstream methods replace the model's
    LayerNorms in place, so a model must never be shared between two method instances."""
    return _models.get_pretrained_model(SimpleNamespace(model=backbone, device=device))


def load_datasets(name: str, root: str, transform: Any, domains: str = "main") -> Any:
    """Upstream multi-domain dataset: .datasets (one per domain), .environments, .classes, .num_classes.
    domains="heldout": upstream's extra=True set (main + extra corruptions), cut down to the extra corruptions only."""
    cls = _datasets.get_dataset_class(name)
    root = str(Path(root).expanduser())
    if domains == "main":
        return cls(root=root, transform=transform)
    if domains != "heldout":
        raise ValueError(f"unknown domain set {domains!r}")
    data = cls(root=root, transform=transform, extra=True)
    keep = [i for i, env in enumerate(data.environments) if env in _corruption_utils.extra_corruptions]
    if not keep:
        raise ValueError(f"{name}: no extra corruptions found in {data.environments}")
    data.environments = [data.environments[i] for i in keep]
    data.datasets = [data.datasets[i] for i in keep]  # the main corruptions' arrays are freed with their datasets
    return data


def load_method_config(dataset: str, algo: str) -> Dict[str, Any]:
    """The yaml upstream main.py would load: cfg/<dataset>/<algo>.yaml, else cfg/default/<algo>.yaml, else {}."""
    for path in (CFG / dataset / f"{algo}.yaml", CFG / "default" / f"{algo}.yaml"):
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
    return {}


def method_args(config: Dict[str, Any], device: torch.device, batch_size: int, use_tbn: bool = False) -> SimpleNamespace:
    """The fields of upstream main.py's `args` that method constructors read. max_batch_size sizes the per-sample
    LayerNorm buffers, so it must be at least the stream's batch size."""
    return SimpleNamespace(config=dict(config), device=device, max_batch_size=batch_size, batch_size=batch_size,
                           use_tbn=use_tbn)


def seed_everything(seed: int) -> None:
    """Upstream main.setup_seed: global RNGs plus deterministic cuDNN."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
