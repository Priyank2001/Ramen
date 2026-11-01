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
from models import get_pretrained_model as _models  # noqa: E402  (upstream src/models)


def _check_is_upstream(module: ModuleType) -> None:
    origin = Path(module.__file__).resolve()
    if SRC not in origin.parents:
        raise ImportError(f"{module.__name__} resolved to {origin}, not to upstream {SRC}: a package name clash")


for _module in (_datasets, _methods, _models):
    _check_is_upstream(_module)

UPSTREAM_METHODS = ("NoAdapt", "Tent", "Ramen")


def method_class(name: str) -> type:
    """An upstream TTA method class (NoAdapt, Tent, Ramen)."""
    return _methods.get_method_class(name)


def load_model(backbone: str, device: torch.device) -> Tuple[torch.nn.Module, Any]:
    """A fresh CLIP model and its preprocessing transform. Fresh matters: upstream methods replace the model's
    LayerNorms in place, so a model must never be shared between two method instances."""
    return _models.get_pretrained_model(SimpleNamespace(model=backbone, device=device))


def load_datasets(name: str, root: str, transform: Any) -> Any:
    """Upstream multi-domain dataset: .datasets (one per domain), .environments, .classes, .num_classes."""
    return _datasets.get_dataset_class(name)(root=str(Path(root).expanduser()), transform=transform)


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
