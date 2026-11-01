"""
One run = one RunConfig -> per-sample records + metrics.json, under the RAM guard (D17).

The Runner wires the parts together and owns nothing else: the upstream adapter loads models, data and configs, the
method registry builds the method, StreamFactory fixes the order, RecordWriter keeps outputs, MemoryGuard watches RAM.
All of them are injected, so tests can swap any one (e.g. fake datasets) without touching the rest.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from tqdm import tqdm

from ramen_lab import upstream
from ramen_lab.config import RunConfig
from ramen_lab.methods import METHODS, uses_labels
from ramen_lab.records import RecordWriter, accuracy_metrics, run_record, write_metrics
from ramen_lab.registry import Registry
from ramen_lab.resources import GB, MemoryExceeded, MemoryGuard, ResourceBudget
from ramen_lab.stream import StreamFactory


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    metrics: Dict[str, Any]
    resources: Dict[str, Any]
    skipped: bool = False  # True if a finished run was found and not re-run


class Runner:
    """Runs configs and records them under `results_dir/<run_id>/`."""

    def __init__(self, results_dir: Path, script: str, methods: Registry = METHODS, adapter: Any = upstream,
                 device: Optional[torch.device] = None, budget: Optional[ResourceBudget] = None,
                 capture_probs: bool = False, guard_factory: Callable[[ResourceBudget], MemoryGuard] = MemoryGuard):
        self.results_dir = Path(results_dir)
        self.script = script
        self.methods = methods
        self.adapter = adapter
        self.device = device or torch.device("cuda")
        self.budget = budget
        self.capture_probs = capture_probs
        self.guard_factory = guard_factory
        self.machine = run_record.machine_tag()

    # ------------------------------------------------------------------ building
    def build(self, config: RunConfig):
        """A fresh model, the datasets and the method for this config (nothing is shared between runs)."""
        self.adapter.seed_everything(config.seed)
        model, preprocess = self.adapter.load_model(config.resolved_backbone, self.device)
        datasets = self.adapter.load_datasets(config.dataset, config.data_root, preprocess)
        method_cfg = {**self.adapter.load_method_config(config.dataset, config.method), **dict(config.overrides)}
        args = self.adapter.method_args(method_cfg, self.device, config.resolved_batch_size)
        method = self.methods.get(config.method)(model, datasets, args)
        return method, datasets, method_cfg

    # ------------------------------------------------------------------ running
    def run(self, config: RunConfig, baseline_run_id: Optional[str] = None, overwrite: bool = False,
            notes: Optional[str] = None) -> RunResult:
        run_id = config.run_id(self.machine)
        run_dir = self.results_dir / run_id
        if (run_dir / "metrics.json").exists() and not overwrite:
            rec = json.loads((run_dir / "metrics.json").read_text())
            stored = {k: v for k, v in rec["config"].items() if k != "method_config"}
            if stored != json.loads(json.dumps(config.to_dict())):
                raise ValueError(f"{run_id} exists with a different config; give the variant a new `name`")
            return RunResult(run_id, run_dir, rec["metrics"], rec.get("resources", {}), skipped=True)

        method, datasets, method_cfg = self.build(config)
        streams = StreamFactory(config.resolved_batch_size, config.seed, config.num_workers, config.n_per_domain)
        loaders = [streams.mixed(datasets)] if config.tta_mode == "mixed" else streams.single(datasets)
        writer = RecordWriter(sum(len(l.dataset) for l in loaders), datasets.num_classes, run_dir, self.capture_probs)

        budget = self.budget or ResourceBudget(rss_gb=config.spec.ram_budget_gb)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.time()
        guard = self.guard_factory(budget)
        try:
            with guard:
                for loader in loaders:
                    self._adapt(method, loader, writer, guard, run_id)
                    method.reset()  # single mode: forget the domain; mixed: leave a clean method behind
        except MemoryExceeded as e:
            (run_dir / "aborted.json").write_text(json.dumps({"run_id": run_id, "reason": str(e),
                                                              "images_done": writer.n}, indent=2) + "\n")
            raise

        resources = self._resources(guard, budget, start)
        metrics = accuracy_metrics(writer.correct(), writer.arrays["domain_idx"][:writer.n], datasets.environments)
        records = writer.save()
        write_metrics(self.results_dir, run_id, self.script,
                      config={**config.to_dict(), "method_config": method_cfg}, metrics=metrics, resources=resources,
                      baseline_run_id=baseline_run_id, raw_outputs=[records], notes=notes)
        return RunResult(run_id, run_dir, metrics, resources)

    def _adapt(self, method, loader, writer: RecordWriter, guard: MemoryGuard, desc: str) -> None:
        for image, label, domain_idx, sample_idx in tqdm(loader, desc=desc, mininterval=30):
            guard.check()
            image = image.to(self.device, non_blocking=True)
            logits = method(image, labels=label) if uses_labels(method) else method(image)
            writer.add(logits, label, domain_idx, sample_idx)

    def _resources(self, guard: MemoryGuard, budget: ResourceBudget, start: float) -> Dict[str, Any]:
        gpu = {}
        if self.device.type == "cuda":
            gpu = {"peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated(self.device) / GB, 2),
                   "peak_gpu_reserved_gb": round(torch.cuda.max_memory_reserved(self.device) / GB, 2)}
        return {"peak_ram_gb": round(guard.peak_gb, 2), "ram_budget_gb": budget.rss_gb,
                "min_system_available_gb": round(guard.min_available_gb, 1), **gpu,
                "minutes": round((time.time() - start) / 60, 2)}
