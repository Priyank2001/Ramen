"""
Week 01 baseline (docs/plans/week01_baseline.md): fixed batch per dataset, then NoAdapt + Ramen seeds 0-4 and the
noise floor at that batch. Analysis lives in baseline.ipynb, which imports this file.

Commands (PC2; prefix with the RAMEN_ENV python and PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID):

    CUDA_VISIBLE_DEVICES=1 python baseline.py batch-search CIFAR10C
        Largest batch in CANDIDATE_BATCHES that fits GPU 1 (11 GB, the smaller card) with every memory queue full.
        Writes results/batch_search/<dataset>.json. Freeze the result in ramen_lab/config.py by hand (a commit).
    CUDA_VISIBLE_DEVICES=0 python baseline.py check CIFAR10C [--batch B]
        200 images per domain: NoAdapt seeds 0 and 1 must agree on every prediction (a seed changes only the order),
        and Ramen seed 0 run twice must agree on every prediction (determinism).
    python -m ramen_lab.launch --gpu 0 --log results/logs/gpu0.log -- baseline.py run CIFAR10C --seeds 0,2,4
    python -m ramen_lab.launch --gpu 1 --log results/logs/gpu1.log -- baseline.py run CIFAR10C --seeds 1,3
        Full stream at the frozen batch: NoAdapt for the first seed listed, then Ramen for every seed. Seeds are tied to
        GPUs (0, 2, 4 -> GPU 0; 1, 3 -> GPU 1) so later variants can be paired with a baseline on the same GPU model.
"""
import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the repo root, for ramen_lab
from ramen_lab.config import DATASETS, RunConfig  # noqa: E402
from ramen_lab.records import load_records, read_metrics  # noqa: E402
from ramen_lab.resources import GB, MemoryGuard, ResourceBudget  # noqa: E402
from ramen_lab.runner import Runner  # noqa: E402
from ramen_lab.stream import StreamFactory  # noqa: E402

RESULTS = HERE / "results"
CANDIDATE_BATCHES = (100, 64, 50, 48, 32, 24, 16)
GPU_HEADROOM = 0.05  # keep 5% of the card free: other kernels' workspaces and fragmentation
SEED_GPU = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}  # pre-registered in the plan
CHECK_N_PER_DOMAIN = 200


# ---------------------------------------------------------------------------- batch search
def fill_memory(method) -> int:
    """Fill every class queue of an upstream Ramen to capacity with random entries, so the batch search measures the
    peak of a full memory (top-k = k from every queue), not of the empty memory at the start of a stream.
    Reads upstream PriorityCache fields (keys, values, entropies, priorities, size); measurement only."""
    filled = 0
    for queue in method.cache:
        n = queue.max_capacity
        keys = torch.nn.functional.normalize(torch.randn(n, queue.key_dim, device=queue.device), dim=1)
        queue.keys.copy_(keys.to(queue.dtype))
        queue.values.copy_((torch.randn(n, queue.value_dim, device=queue.device) * 1e-3).to(queue.dtype))
        queue.entropies.copy_(torch.rand(n, device=queue.device).to(queue.dtype))
        queue.priorities.copy_(torch.arange(n, device=queue.device).to(queue.dtype))
        queue.size = n
        filled += n
    return filled


def measure_batch(dataset: str, batch: int, n_batches: int = 3) -> Dict[str, object]:
    """Peak GPU memory (driver view: CUDA context + PyTorch's reserved pool) and process RAM for a few batches of
    Ramen with a full memory. Out of memory is a result, not an error."""
    runner = Runner(RESULTS / "batch_search", script=__file__)
    config = RunConfig(dataset, batch_size=batch, n_per_domain=batch * n_batches)  # >= n_batches full batches
    total = torch.cuda.get_device_properties(0).total_memory
    report: Dict[str, object] = {"batch": batch, "gpu_total_gb": round(total / GB, 2)}
    method = None
    try:
        with MemoryGuard(ResourceBudget(rss_gb=float(DATASETS[dataset].ram_budget_gb))) as guard:
            method, datasets, _ = runner.build(config)
            report["memory_entries"] = fill_memory(method)
            context = torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0] - torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            loader = StreamFactory(batch, 0, config.num_workers, config.n_per_domain).mixed(datasets)
            for i, (image, *_rest) in enumerate(loader):
                guard.check()
                method(image.cuda())
                if i + 1 == n_batches:
                    break
            peak = context + torch.cuda.max_memory_reserved()
            report.update(fits=bool(peak <= (1 - GPU_HEADROOM) * total), peak_gpu_gb=round(peak / GB, 2),
                          peak_ram_gb=round(guard.peak_gb, 2))
    except torch.cuda.OutOfMemoryError:
        report.update(fits=False, peak_gpu_gb=None, oom=True)
    finally:
        del method
        gc.collect()
        torch.cuda.empty_cache()
    return report


def batch_search(dataset: str) -> Dict[str, object]:
    reports: List[Dict[str, object]] = []
    for batch in CANDIDATE_BATCHES:
        reports.append(measure_batch(dataset, batch))
        print(json.dumps(reports[-1]), flush=True)
        if reports[-1]["fits"]:
            break
    chosen = next((r for r in reports if r["fits"]), None)
    out = {"dataset": dataset, "gpu": torch.cuda.get_device_name(0), "chosen_batch": chosen and chosen["batch"],
           "suggested_ram_budget_gb": chosen and round(max(4.0, 1.5 * chosen["peak_ram_gb"]), 1), "tried": reports}
    path = RESULTS / "batch_search" / f"{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{dataset}: chosen batch {out['chosen_batch']} on {out['gpu']} -> {path}")
    return out


# ---------------------------------------------------------------------------- checks and runs
def check(dataset: str, batch: Optional[int]) -> None:
    """Seed independence of NoAdapt and determinism of Ramen, on a subset. Raises on failure."""
    def run(results: Path, **kw):
        cfg = RunConfig(dataset, batch_size=batch, n_per_domain=CHECK_N_PER_DOMAIN, **kw)
        return load_records(Runner(results, script=__file__).run(cfg, overwrite=True).run_dir)

    base = RESULTS / "check"
    na0, na1 = run(base / "a", method="NoAdapt", seed=0), run(base / "a", method="NoAdapt", seed=1)
    by_image = lambda r: r["pred"][np.lexsort((r["sample_idx"], r["domain_idx"]))]  # order-free comparison
    same = by_image(na0) == by_image(na1)
    print(f"NoAdapt seed 0 vs seed 1: {same.sum():,} of {len(same):,} predictions identical ({same.mean():.2%})")
    r1, r2 = run(base / "a", method="Ramen", seed=0), run(base / "b", method="Ramen", seed=0)
    det = r1["pred"] == r2["pred"]
    print(f"Ramen seed 0, two runs: {det.sum():,} of {len(det):,} predictions identical ({det.mean():.2%})")
    assert same.all() and det.all(), "check failed: stop (see docs/plans/week01_baseline.md, red flags)"


def run(dataset: str, seeds: List[int]) -> None:
    runner = Runner(RESULTS, script=__file__)
    first = seeds[0]
    noadapt = runner.run(RunConfig(dataset, method="NoAdapt", seed=first))
    print(f"NoAdapt s{first}: {noadapt.metrics['acc']:.2f}  {noadapt.resources}", flush=True)
    for seed in seeds:
        result = runner.run(RunConfig(dataset, method="Ramen", seed=seed), baseline_run_id=noadapt.run_id)
        print(f"Ramen s{seed}: {result.metrics['acc']:.2f}  {result.resources}", flush=True)


# ---------------------------------------------------------------------------- analysis (baseline.ipynb)
def load_runs(dataset: str) -> Dict[str, Dict[int, dict]]:
    """{'NoAdapt': {seed: metrics.json}, 'Ramen': {...}} for the full-stream runs at the dataset's frozen batch."""
    spec, runs = DATASETS[dataset], {"NoAdapt": {}, "Ramen": {}}
    for path in sorted(RESULTS.glob(f"*_{spec.short}_*_b{spec.batch_size}_s*/metrics.json")):
        rec = read_metrics(path)
        if rec["config"]["n_per_domain"] is None and rec["config"]["method"] in runs:
            runs[rec["config"]["method"]][rec["config"]["seed"]] = rec
    return runs


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("batch-search", "check", "run"))
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--batch", type=int, default=None, help="check only: batch before it is frozen")
    parser.add_argument("--seeds", default="0", help="run only: comma-separated seeds")
    args = parser.parse_args(argv)
    if args.command == "batch-search":
        batch_search(args.dataset)
    elif args.command == "check":
        check(args.dataset, args.batch)
    else:
        run(args.dataset, [int(s) for s in args.seeds.split(",")])


if __name__ == "__main__":
    main()
