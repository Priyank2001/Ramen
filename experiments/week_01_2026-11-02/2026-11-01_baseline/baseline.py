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
    PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 0 --ram-gb 8.2 --log results/logs/gpu0.log \
        -- baseline.py run CIFAR10C --seeds 0,2,4
    PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 1 --ram-gb 8.2 --log results/logs/gpu1.log \
        -- baseline.py run CIFAR10C --seeds 1,3
        Full stream at the frozen batch: NoAdapt for the first seed listed, then Ramen for every seed. Seeds are tied to
        GPUs (0, 2, 4 -> GPU 0; 1, 3 -> GPU 1) so later variants can be paired with a baseline on the same GPU model.
"""
import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# Paper, arXiv 2604.21728 (docs/papers/pdf/ramen_2604.21728.pdf), ViT-B/32 on CIFAR-10-C, batch 100, mixture of
# 15 corruptions. Table 1: per corruption and average; "Ensemble" = zero-shot CLIP with the 7-template prompt
# ensemble, which is what upstream NoAdapt computes. Table 5: Ramen's average with its s.d.
PAPER_C10 = {
    "corruptions": ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
                    "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                    "jpeg_compression"],
    "Ramen": [61.0, 63.3, 56.0, 77.2, 64.1, 77.2, 78.9, 80.9, 80.3, 78.0, 88.1, 79.8, 72.0, 65.5, 67.5],
    "Ramen_avg": 72.7, "Ramen_avg_sd": 0.1, "Ensemble_avg": 61.3,
}


def seed_table(dataset: str):
    """One row per run: method, seed, GPU, accuracy (mean over corruptions), images, minutes, peak RAM / GPU."""
    import pandas as pd
    rows = []
    for method, by_seed in load_runs(dataset).items():
        for seed, rec in sorted(by_seed.items()):
            m, r = rec["metrics"], rec["resources"]
            rows.append({"method": method, "seed": seed, "gpu": rec["gpu"], "acc": m["acc"], "acc_micro": m["acc_micro"],
                         "images": m["n_samples"], "correct": m["n_correct"], "minutes": r["minutes"],
                         "peak_ram_gb": r["peak_ram_gb"], "ram_budget_gb": r["ram_budget_gb"],
                         "peak_gpu_reserved_gb": r["peak_gpu_reserved_gb"]})
    return pd.DataFrame(rows)


def per_corruption(dataset: str):
    """Per corruption: NoAdapt (seed 0), Ramen mean / min / max over seeds, the paper's Ramen, and the gaps."""
    import pandas as pd
    runs = load_runs(dataset)
    ramen = pd.DataFrame({s: r["metrics"]["per_domain"] for s, r in runs["Ramen"].items()})
    table = pd.DataFrame({"NoAdapt": pd.Series(runs["NoAdapt"][min(runs["NoAdapt"])]["metrics"]["per_domain"]),
                          "Ramen mean": ramen.mean(axis=1), "Ramen min": ramen.min(axis=1),
                          "Ramen max": ramen.max(axis=1)})
    table["seed range"] = table["Ramen max"] - table["Ramen min"]
    if dataset == "CIFAR10C":
        table["paper Ramen"] = pd.Series(PAPER_C10["Ramen"], index=PAPER_C10["corruptions"])
        table["ours - paper"] = table["Ramen mean"] - table["paper Ramen"]
    table["Ramen - NoAdapt"] = table["Ramen mean"] - table["NoAdapt"]
    return table.loc[PAPER_C10["corruptions"]] if dataset == "CIFAR10C" else table


def paired_records(dataset: str, seed: int = 0):
    """NoAdapt and Ramen per-image records for one seed (same GPU and stream order, so rows line up)."""
    runs = load_runs(dataset)
    a = load_records(RESULTS / runs["NoAdapt"][seed]["run_id"])
    b = load_records(RESULTS / runs["Ramen"][seed]["run_id"])
    assert runs["NoAdapt"][seed]["gpu"] == runs["Ramen"][seed]["gpu"], "per-image pairs need the same GPU model"
    assert np.array_equal(a["sample_idx"], b["sample_idx"]) and np.array_equal(a["domain_idx"], b["domain_idx"])
    return a, b


def over_stream(dataset: str, seed: int = 0, n_bins: int = 15):
    """Accuracy of NoAdapt and Ramen per stream bin (cold memory at the start)."""
    import pandas as pd
    a, b = paired_records(dataset, seed)
    bins = np.array_split(np.arange(len(a["label"])), n_bins)
    acc = lambda r, idx: 100 * float((r["pred"][idx] == r["label"][idx]).mean())
    return pd.DataFrame({"start": [int(i[0]) for i in bins], "end": [int(i[-1]) + 1 for i in bins],
                         "NoAdapt": [acc(a, i) for i in bins], "Ramen": [acc(b, i) for i in bins]})


def check_agreement(method: str, dir_a: str, dir_b: str, seeds: Tuple[int, int]) -> Tuple[int, int]:
    """Subset check runs (results/check): predictions identical per image between two runs, (identical, total)."""
    spec = DATASETS["CIFAR10C"]
    by_image = lambda r: r["pred"][np.lexsort((r["sample_idx"], r["domain_idx"]))]
    run = lambda d, s: load_records(RESULTS / "check" / d / f"pc2_{spec.short}_{method}_b{spec.batch_size}_s{s}_n{CHECK_N_PER_DOMAIN}")
    a, b = by_image(run(dir_a, seeds[0])), by_image(run(dir_b, seeds[1]))
    return int((a == b).sum()), len(a)


def full_noadapt_agreement(dataset: str) -> Tuple[int, int]:
    """NoAdapt seed 0 vs seed 1 on the full stream, per image (these two ran on different GPU models)."""
    runs = load_runs(dataset)["NoAdapt"]
    by_image = lambda r: r["pred"][np.lexsort((r["sample_idx"], r["domain_idx"]))]
    a, b = (by_image(load_records(RESULTS / runs[s]["run_id"])) for s in (0, 1))
    return int((a == b).sum()), len(a)


def plot_paper_gap(table, sd: float):
    """Dot per corruption: our Ramen mean minus the paper's, with our seed range as a bar and the paper's s.d."""
    import matplotlib.pyplot as plt
    from ramen_lab.analysis.plots import color
    t = table.sort_values("ours - paper")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    y = np.arange(len(t))
    ax.hlines(y, t["Ramen min"] - t["paper Ramen"], t["Ramen max"] - t["paper Ramen"], color=color("Ramen"), alpha=0.35, lw=3)
    ax.scatter(t["ours - paper"], y, color=color("Ramen"), zorder=3, s=30)
    ax.axvline(0, color="#555555", lw=1)
    ax.axvspan(-sd, sd, color="#999999", alpha=0.12, label=f"paper's s.d. of the average ({sd})")
    ax.set_yticks(y, t.index)
    ax.set_xlabel("our Ramen minus the paper's Ramen (pts); bar = range over our seeds 0-4")
    ax.set_title("Every corruption within 0.21 pts of the paper's Table 1")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_over_stream(df):
    """Accuracy per stream bin for NoAdapt and Ramen (seed 0): does Ramen start cold?"""
    import matplotlib.pyplot as plt
    from ramen_lab.analysis.plots import color
    x = (df.start + df.end) / 2 / 1000
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for name in ("Ramen", "NoAdapt"):
        ax.plot(x, df[name], marker="o", ms=3, color=color(name), label=name)
    ax.set_xlabel("stream position (x1,000 images)")
    ax.set_ylabel("accuracy in the bin (%)")
    ax.set_title("Ramen's gain over NoAdapt is smaller in the first 10,000 images")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


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
