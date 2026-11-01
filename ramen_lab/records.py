"""
Per-sample outputs of a run, and its citable record (metrics.json, AGENTS.md section 5).

RecordWriter preallocates one array per field for the whole stream (its length is known before the run), so memory
stays flat instead of growing with Python lists (D17). The base fields cost about 30 bytes per image (586,575 images on
DomainNet: about 17 MB). Optional per-class probabilities go to a disk-backed fp16 memmap, never into RAM.
"""
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ramen_lab.upstream import REPO

FIELDS = {"batch": np.int32, "domain_idx": np.int16, "sample_idx": np.int32, "label": np.int32, "pred": np.int32,
          "conf": np.float32, "true_prob": np.float32, "entropy": np.float32}


def _run_record_module():
    """scripts/run_record.py (shared with the review tooling): write_metrics, machine_tag, gpu_name."""
    spec = importlib.util.spec_from_file_location("run_record", REPO / "scripts" / "run_record.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_record = _run_record_module()
read_metrics = run_record.read_metrics


class RecordWriter:
    """Fills per-sample arrays batch by batch, in stream order; save() writes records.npz."""

    def __init__(self, n_total: int, num_classes: int, out_dir: Path, capture_probs: bool = False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.arrays = {name: np.zeros(n_total, dtype=dtype) for name, dtype in FIELDS.items()}
        self.probs = (np.lib.format.open_memmap(self.out_dir / "probs.npy", mode="w+", dtype=np.float16,
                                                shape=(n_total, num_classes)) if capture_probs else None)
        self.n = 0
        self.batches = 0

    @torch.no_grad()
    def add(self, logits: torch.Tensor, label: torch.Tensor, domain_idx: torch.Tensor, sample_idx: torch.Tensor) -> None:
        prob = logits.float().softmax(dim=1).cpu()
        conf, pred = prob.max(dim=1)
        rows = slice(self.n, self.n + len(label))
        values = {"batch": self.batches, "domain_idx": domain_idx.numpy(), "sample_idx": sample_idx.numpy(),
                  "label": label.numpy(), "pred": pred.numpy(), "conf": conf.numpy(),
                  "true_prob": prob.gather(1, label[:, None]).squeeze(1).numpy(),
                  "entropy": -(prob * prob.clamp_min(1e-12).log()).sum(dim=1).numpy()}
        for name, value in values.items():
            self.arrays[name][rows] = value
        if self.probs is not None:
            self.probs[rows] = prob.numpy().astype(np.float16)
        self.n += len(label)
        self.batches += 1

    def correct(self) -> np.ndarray:
        return self.arrays["pred"][:self.n] == self.arrays["label"][:self.n]

    def save(self) -> Path:
        path = self.out_dir / "records.npz"
        np.savez(path, **{name: a[:self.n] for name, a in self.arrays.items()})
        if self.probs is not None:
            self.probs.flush()
        return path


def accuracy_metrics(correct: np.ndarray, domain_idx: np.ndarray, domain_names: Sequence[str]) -> Dict[str, object]:
    """Per-domain accuracy (%), their mean (`acc`, the headline), the per-image `acc_micro`, and counts."""
    per_domain = {name: float(100 * correct[domain_idx == d].mean()) for d, name in enumerate(domain_names)
                  if (domain_idx == d).any()}
    return {"acc": round(float(np.mean(list(per_domain.values()))), 4),
            "acc_micro": round(float(100 * correct.mean()), 4),
            "per_domain": {k: round(v, 4) for k, v in per_domain.items()},
            "n_samples": int(len(correct)), "n_correct": int(correct.sum())}


def load_records(run_dir: Path) -> Dict[str, np.ndarray]:
    with np.load(Path(run_dir) / "records.npz") as f:
        return {name: f[name] for name in f.files}


def write_metrics(results_dir: Path, run_id: str, script: str, config: Dict, metrics: Dict, resources: Dict,
                  baseline_run_id: Optional[str] = None, raw_outputs: Sequence[Path] = (),
                  notes: Optional[str] = None, git: Optional[Tuple[str, bool]] = None) -> Path:
    """metrics.json with the AGENTS.md schema plus `resources` (peak RAM / GPU, minutes). `git`: (hash, dirty) captured
    when the run started; None reads them now."""
    git_hash, git_dirty = git if git is not None else ("auto", None)
    return run_record.write_metrics(results_dir, run_id, script, config=config, metrics=metrics,
                                    baseline_run_id=baseline_run_id, raw_outputs=list(raw_outputs), notes=notes,
                                    extra={"resources": resources}, git_hash=git_hash, git_dirty=git_dirty)


def find_runs(results_dir: Path) -> List[Path]:
    return sorted(p.parent for p in Path(results_dir).glob("*/metrics.json"))
