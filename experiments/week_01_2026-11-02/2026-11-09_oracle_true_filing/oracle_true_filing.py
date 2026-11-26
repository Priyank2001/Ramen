"""
ORACLE: what if Ramen filed every test sample under its TRUE class? (docs/plans/oracle_true_filing.md)

One run keeps three memories over the same stream (ramen_lab/methods/multi_cache.py, `RamenMultiCache`):
    copy 0  zero_shot   Ramen itself (bit-identical; its predictions are the run's records.npz / metrics.json)
    copy 1  true_label  ORACLE: queue = true class
    copy 2  random      control: queue = a random class (same spreading, no label information)
Same keys, gradients, entropies, priorities, stream, batch, seed and GPU; only the queue differs. Per-copy predictions,
active-set statistics, example active sets and memory snapshots land in results/<run_id>/multicache.npz.
Analysis: oracle_true_filing.ipynb.

Commands (prefix: PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>, RAMEN_ENV python):

    python oracle_true_filing.py size-pool CIFAR10C
        Replays the filing of all copies over the seed-0 stream (zero-shot predictions from the NoAdapt seed-0 run) with
        the real cache classes and 1-dim values: the peak number of gradient rows the shared pool must hold.
    python oracle_true_filing.py check CIFAR10C [--n 200]
        Subset: copy 0 must equal plain Ramen on every prediction (100%).
    PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 0 --ram-gb 13 --log results/logs/gpu0_cifar.log \
        -- oracle_true_filing.py run CIFAR10C CIFAR100C --seed 0 --ram-gb 13
    PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 0 --ram-gb 10 --log results/logs/gpu0_inc_dn.log \
        -- oracle_true_filing.py run ImageNetC5K --seed 0 --ram-gb 10
    CUDA_MODULE_LOADING=LAZY PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 0 --ram-gb 10 \
        --log results/logs/gpu0_dn_lazy.log -- oracle_true_filing.py run DomainNet --seed 0 --ram-gb 10
        DomainNet needs lazy CUDA kernel loading: eager loading takes ~1.1 GiB of the 12 GB card (vs ~0.46 GiB lazy)
        and the two-copy run does not fit next to it.
"""
import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the repo root, for ramen_lab
from ramen_lab import upstream  # noqa: E402
from ramen_lab.config import DATASETS, RunConfig  # noqa: E402
from ramen_lab.methods.cache_pool import FILINGS, GradientPool, PooledPriorityCache  # noqa: E402
from ramen_lab.methods.multi_cache import DEFAULT_FILINGS  # noqa: E402
from ramen_lab.methods.multi_cache_log import load_log  # noqa: E402
from ramen_lab.records import accuracy_metrics, load_records, read_metrics  # noqa: E402
from ramen_lab.resources import ResourceBudget  # noqa: E402
from ramen_lab.runner import Runner  # noqa: E402

RESULTS = HERE / "results"
BASELINE = HERE.parent / "2026-11-01_baseline" / "results"
METHOD = "RamenMultiCache"
FILING_SEED = 0
N_EXAMPLES = 48  # per category (broken / fixed), reservoir over the whole stream
SNAPSHOT_FRACTIONS = (0.25, 0.5, 0.75)  # plus the end of the stream
NUM_CLASSES = {"CIFAR10C": 10, "CIFAR100C": 100, "ImageNetC5K": 1000, "DomainNet": 345}
POOL_MARGIN = 1.05  # the replay uses NoAdapt's zero-shot predictions; Ramen's by-sample model differs on very few
# (DomainNet seed 0: 371 of 586,575 images differ in zero-shot correctness, grad_alignment run vs NoAdapt run), so
# DomainNet, where GPU memory is tight, gets a 2% margin.
POOL_MARGIN_OF: Dict[str, float] = {"DomainNet": 1.02}
# DomainNet: three copies need ~11.1 GiB allocated on the 12 GiB card (replay peak 130,801 rows): too tight, so the
# random control is dropped there.
FILINGS_OF: Dict[str, Tuple[str, ...]] = {"DomainNet": ("zero_shot", "true_label")}
# Peak pool rows from `size-pool` on the seed-0 stream (filled in from its output; None = not measured yet).
POOL_PEAK: Dict[str, Optional[int]] = {"CIFAR10C": 82_625, "CIFAR100C": 87_744, "ImageNetC5K": 73_796, "DomainNet": 126_338}


# ---------------------------------------------------------------------------- pool sizing
def filings_of(dataset: str) -> Tuple[str, ...]:
    return FILINGS_OF.get(dataset, DEFAULT_FILINGS)


def replay_pool_peak(zs_pred: np.ndarray, labels: np.ndarray, num_classes: int, capacity: int, batch: int,
                     filings: Sequence[str] = DEFAULT_FILINGS, seed: int = FILING_SEED,
                     device: torch.device = torch.device("cuda")) -> int:
    """Peak rows in use when every copy files the stream (in stream order, batch by batch) exactly as RamenMultiCache
    does: same fp16 priorities, same eviction, same batch-end freeing. Values and keys are 1-dim placeholders."""
    policies = [FILINGS.get(n)(num_classes, seed + i) for i, n in enumerate(filings)]
    pool = GradientPool(len(filings) * num_classes * capacity, 1, device)
    caches = [[PooledPriorityCache(pool, capacity, 1, device) for _ in range(num_classes)] for _ in filings]
    zero_key, zero_ent = torch.zeros(1, device=device), torch.zeros((), device=device)
    for start in range(0, len(labels), batch):
        zs = torch.as_tensor(zs_pred[start:start + batch], dtype=torch.long)
        y = torch.as_tensor(labels[start:start + batch], dtype=torch.long)
        n = len(y)
        rows = pool.put(torch.zeros(n, 1, device=device), torch.arange(start, start + n), zs, y).tolist()
        prio = torch.arange(start, start + n, device=device, dtype=torch.half)
        for policy, queues in zip(policies, caches):
            for b, q in enumerate(policy.queues(zs, y if policy.uses_labels else None).tolist()):
                queues[q].add(zero_key, rows[b], zero_ent, prio[b])
        pool.finish_batch()
    return pool.peak_in_use


def size_pool(dataset: str, seed: int = 0) -> int:
    """Replay on the NoAdapt run of the same seed (same stream order: ramen_lab streams are seeded explicitly)."""
    spec = DATASETS[dataset]
    run = BASELINE / RunConfig(dataset, method="NoAdapt", seed=seed).run_id("pc2")
    rec = load_records(run)
    cfg = upstream.load_method_config(dataset, "Ramen")
    num_classes = NUM_CLASSES[dataset]
    filings = filings_of(dataset)
    peak = replay_pool_peak(rec["pred"], rec["label"], num_classes, cfg["max_capacity"], spec.batch_size, filings)
    full = len(filings) * num_classes * cfg["max_capacity"]
    print(f"{dataset} ({', '.join(filings)}): peak pool rows {peak:,} = {peak / (num_classes * cfg['max_capacity']):.2f} x one memory "
          f"(worst case {full:,}); with margin {int(peak * POOL_MARGIN_OF.get(dataset, POOL_MARGIN)):,}", flush=True)
    return peak


# ---------------------------------------------------------------------------- runs
def multi_config(dataset: str, seed: int, results: Path, n_per_domain: Optional[int] = None,
                 n_images: int = 0) -> RunConfig:
    """Ramen's yaml + the copies, the pool size, and where the log goes."""
    probe = RunConfig(dataset, method=METHOD, seed=seed, n_per_domain=n_per_domain)
    run_dir = results / probe.run_id(Runner(results, script=__file__).machine)
    peak = POOL_PEAK.get(dataset) if n_per_domain is None else None
    yaml = upstream.load_method_config(dataset, "Ramen")
    filings = filings_of(dataset)
    worst = len(filings) * NUM_CLASSES[dataset] * yaml["max_capacity"]
    margin = POOL_MARGIN_OF.get(dataset, POOL_MARGIN)
    pool_rows = min(worst, n_images, int(peak * margin) if peak else worst)  # never more rows than images
    overrides = {**yaml, "filings": list(filings), "filing_seed": FILING_SEED, "n_examples": N_EXAMPLES,
                 "log_path": str(run_dir / "multicache.npz"), "pool_capacity": pool_rows,
                 "snapshot_at": [int(f * n_images) for f in SNAPSHOT_FRACTIONS]}
    return RunConfig.make(dataset, overrides, method=METHOD, seed=seed, n_per_domain=n_per_domain)


def stream_length(dataset: str, n_per_domain: Optional[int] = None) -> int:
    lengths = {"CIFAR10C": (15, 10_000), "CIFAR100C": (15, 10_000), "ImageNetC5K": (15, 5_000),
               "DomainNet": (6, None)}
    domains, per_domain = lengths[dataset]
    if per_domain is None:  # DomainNet: unequal domains; read the length off the Ramen baseline
        return len(load_records(BASELINE / RunConfig(dataset, seed=0).run_id("pc2"))["label"])
    return domains * (n_per_domain or per_domain)


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def check(dataset: str, n: int) -> None:
    base = RESULTS / "check"
    ramen = Runner(base / "ramen", script=__file__).run(RunConfig(dataset, method="Ramen", n_per_domain=n),
                                                        overwrite=True)
    free_gpu()
    cfg = multi_config(dataset, 0, base / "multi", n_per_domain=n, n_images=stream_length(dataset, n))
    multi = Runner(base / "multi", script=__file__).run(cfg, overwrite=True)
    r, m = load_records(ramen.run_dir), load_records(multi.run_dir)
    same = r["pred"] == m["pred"]
    print(f"Ramen copy vs plain Ramen: {same.sum():,} of {len(same):,} predictions identical ({same.mean():.2%})")
    print(f"resources: {multi.resources}")
    assert same.all(), "copy 0 is not Ramen: stop"


def run(datasets: Sequence[str], seed: int, ram_gb: Optional[float] = None) -> None:
    """ram_gb: the in-process MemoryGuard budget; None = the dataset's frozen budget (ramen_lab/config.py)."""
    for dataset in datasets:
        budget = ResourceBudget(rss_gb=ram_gb) if ram_gb else None
        runner = Runner(RESULTS, script=__file__, budget=budget)
        baseline = RunConfig(dataset, method="Ramen", seed=seed).run_id(runner.machine)
        cfg = multi_config(dataset, seed, RESULTS, n_images=stream_length(dataset))
        result = runner.run(cfg, baseline_run_id=baseline,
                            notes="ORACLE: copies zero_shot (= Ramen, these records), true_label, random; "
                                  "per-copy outputs in multicache.npz")
        print(f"{result.run_id}: Ramen copy {result.metrics['acc']:.2f}  {result.resources}", flush=True)
        write_copy_metrics(result.run_dir)
        free_gpu()


def write_copy_metrics(run_dir: Path) -> Path:
    """copy_metrics.json next to metrics.json: every copy's accuracy, computed from multicache.npz with the same
    `accuracy_metrics` that wrote metrics.json. Copy 0 must reproduce metrics.json exactly (checked here)."""
    run_dir = Path(run_dir)
    rec, log, metrics = load_records(run_dir), load_log(run_dir / "multicache.npz"), read_metrics(run_dir / "metrics.json")
    if not (np.array_equal(log["stream_id"], np.arange(len(rec["label"]))) and np.array_equal(rec["label"], log["label"])):
        raise AssertionError("multicache.npz is not in the stream order of records.npz")
    domains = list(metrics["metrics"]["per_domain"])
    out = {"run_id": metrics["run_id"], "source": "multicache.npz", "copies": {}}
    for c, name in enumerate(log["copy_names"]):
        out["copies"][str(name)] = accuracy_metrics(log[f"pred__copy{c}"] == log["label"], rec["domain_idx"], domains)
    if out["copies"]["zero_shot"] != metrics["metrics"]:
        raise AssertionError("copy 0 in multicache.npz does not reproduce metrics.json: the log is misaligned")
    out["copies"]["zero_shot_init"] = accuracy_metrics(log["zs_pred"] == log["label"], rec["domain_idx"], domains)
    path = run_dir / "copy_metrics.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("size-pool")
    s.add_argument("datasets", nargs="+")
    c = sub.add_parser("check")
    c.add_argument("dataset")
    c.add_argument("--n", type=int, default=200)
    r = sub.add_parser("run")
    r.add_argument("datasets", nargs="+")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--ram-gb", type=float, default=None, help="MemoryGuard budget (default: the dataset's frozen one)")
    args = parser.parse_args()
    if args.cmd == "size-pool":
        for d in args.datasets:
            size_pool(d)
    elif args.cmd == "check":
        check(args.dataset, args.n)
    else:
        run(args.datasets, args.seed, args.ram_gb)


if __name__ == "__main__":
    main()


# ============================================================================ analysis (oracle_true_filing.ipynb)
# Pure functions: data in, numbers / figures out. The notebook calls them; nothing here touches the GPU.
from dataclasses import dataclass  # noqa: E402
from typing import Any, List, Tuple  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ramen_lab.analysis.plots import PALETTE, color  # noqa: E402
from ramen_lab.analysis.stats import cluster_bootstrap, noise_floor, outcome_counts  # noqa: E402

COPY_LABEL = {"zero_shot": "Ramen (zero-shot filing)", "true_label": "ORACLE (true-label filing)",
              "random": "control (random filing)", "zero_shot_init": "zero-shot CLIP (no adaptation)"}
COPY_COLOR = {"zero_shot": color("Ramen"), "true_label": color("variant"), "random": color("control"),
              "zero_shot_init": color("NoAdapt")}
DISPLAY = {"CIFAR10C": "CIFAR-10-C", "CIFAR100C": "CIFAR-100-C", "ImageNetC5K": "ImageNet-C", "DomainNet": "DomainNet"}
ONLY_HERE = "#e8833a"  # border of an entry that is in this active set / memory but not in the other one
SHARED = "#bdbdbd"


@dataclass
class Run:
    """One RamenMultiCache run: its records (copy 0 = Ramen), the per-copy log and the cited metrics files."""
    dataset: str
    run_dir: Path
    metrics: Dict[str, Any]
    copy_metrics: Dict[str, Any]
    records: Dict[str, np.ndarray]
    log: Dict[str, np.ndarray]

    @property
    def copies(self) -> List[str]:
        return [str(c) for c in self.log["copy_names"]]

    @property
    def domains(self) -> List[str]:
        return list(self.metrics["metrics"]["per_domain"])

    def pred(self, copy: str) -> np.ndarray:
        return self.log["zs_pred"] if copy == "zero_shot_init" else self.log[f"pred__copy{self.copies.index(copy)}"]

    def correct(self, copy: str) -> np.ndarray:
        return self.pred(copy) == self.log["label"]

    def stat(self, name: str, copy: str) -> np.ndarray:
        return self.log[f"{name}__copy{self.copies.index(copy)}"]


def run_dir_of(dataset: str, seed: int = 0) -> Path:
    return RESULTS / RunConfig(dataset, method=METHOD, seed=seed).run_id("pc2")


def load_run(dataset: str, seed: int = 0) -> Run:
    d = run_dir_of(dataset, seed)
    return Run(dataset, d, read_metrics(d / "metrics.json"), json.loads((d / "copy_metrics.json").read_text())["copies"],
               load_records(d), load_log(d / "multicache.npz"))


def baseline_accs(dataset: str, method: str = "Ramen", seeds: Sequence[int] = range(5)) -> List[float]:
    """Headline accuracy of the week-01 baseline runs (metrics.json), seeds 0-4."""
    return [read_metrics(BASELINE / RunConfig(dataset, method=method, seed=s).run_id("pc2") / "metrics.json")
            ["metrics"]["acc"] for s in seeds if (BASELINE / RunConfig(dataset, method=method, seed=s).run_id("pc2")).exists()]


def clusters(run: Run) -> np.ndarray:
    """Bootstrap clusters: one base image under all corruptions (corruption sets); each image alone on DomainNet."""
    return np.arange(len(run.log["label"])) if run.dataset == "DomainNet" else run.records["sample_idx"]


def gain_table(run: Run, floor: float) -> List[Dict[str, Any]]:
    """Every copy against the Ramen copy: accuracy (copy_metrics.json), paired gain with a 95% cluster-bootstrap CI,
    fixed / broken counts, and whether |gain| clears the noise floor."""
    rows, base = [], run.correct("zero_shot")
    for copy in ["zero_shot_init"] + run.copies:
        acc = run.copy_metrics[copy]["acc"]
        row = {"dataset": DISPLAY[run.dataset], "copy": COPY_LABEL[copy], "key": copy, "acc": acc,
               "gain": acc - run.copy_metrics["zero_shot"]["acc"], "floor": floor}
        if copy != "zero_shot":
            diff = run.correct(copy).astype(float) - base
            row["ci_low"], row["ci_high"] = cluster_bootstrap(diff, run.records["domain_idx"], clusters(run))
            row.update(outcome_counts(run.correct(copy), base))
        rows.append(row)
    return rows


def gain_figure(rows: Sequence[Dict[str, Any]], title: str) -> plt.Figure:
    """Gain over the Ramen copy (points) per dataset: oracle and random control with 95% CI; band = +/- noise floor."""
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    keys = ("true_label", "random", "zero_shot_init")
    fig, ax = plt.subplots(figsize=(7.5, 0.9 + 0.95 * len(datasets)))
    for y, ds in enumerate(datasets):
        floor = next(r["floor"] for r in rows if r["dataset"] == ds)
        ax.add_patch(Rectangle((-floor, y - 0.4), 2 * floor, 0.8, color=color("Ramen"), alpha=0.10, lw=0))
        for k, key in enumerate(keys):
            r = next((r for r in rows if r["dataset"] == ds and r["key"] == key), None)
            if r is None:
                continue
            yy = y + (k - 1) * 0.25
            ax.errorbar(r["gain"], yy, xerr=[[r["gain"] - r["ci_low"]], [r["ci_high"] - r["gain"]]], fmt="o",
                        color=COPY_COLOR[key], ms=6, capsize=3, label=COPY_LABEL[key] if y == 0 else None)
            ax.annotate(f"{r['gain']:+.2f}", (r["ci_high"], yy), xytext=(5, -3), textcoords="offset points", fontsize=7)
    ax.axvline(0, color=color("Ramen"), lw=1.2, label="Ramen copy (0)")
    ax.add_patch(Rectangle((0, 0), 0, 0, color=color("Ramen"), alpha=0.10, label="± noise floor (Ramen seeds 0-4)"))
    ax.set_yticks(range(len(datasets)), datasets)
    ax.set_ylim(len(datasets) - 0.5, -0.5)
    ax.set_xlabel("accuracy gain over the Ramen copy (points, mean over domains; 95% CI)")
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.set_size_inches(10.5, 1.2 + 0.95 * len(datasets))
    fig.tight_layout()
    return fig


def stream_figure(runs: Sequence[Run], bins: int = 30) -> plt.Figure:
    """Accuracy along the stream (equal-size bins, in stream order) for every copy: cold memory shows up on the left."""
    fig, axes = plt.subplots(1, len(runs), figsize=(max(4.2 * len(runs), 8), 3.4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        n = len(run.log["label"])
        edges = np.linspace(0, n, bins + 1).astype(int)
        mid = (edges[:-1] + edges[1:]) / 2000
        for copy in ["zero_shot_init"] + run.copies:
            c = run.correct(copy)
            ax.plot(mid, [100 * c[i:j].mean() for i, j in zip(edges[:-1], edges[1:])], "-", lw=1.6,
                    color=COPY_COLOR[copy], label=COPY_LABEL[copy])
        ax.set(xlabel="stream position (x1,000 images)", title=DISPLAY[run.dataset])
    axes[0, 0].set_ylabel("accuracy in the bin (%)")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Accuracy along the stream, per copy (30 equal bins in stream order)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------- prediction balance
def prediction_shares(run: Run, copy: str) -> np.ndarray:
    """(C,) share of all predictions that go to each class."""
    return np.bincount(run.pred(copy), minlength=NUM_CLASSES[run.dataset]) / len(run.log["label"])


def balance_table(run: Run) -> List[Dict[str, Any]]:
    """How far each copy's predicted-class histogram is from the true one: total variation (TV, 0 = same histogram,
    100 = disjoint) and the share of all predictions taken by the 5 most-predicted classes."""
    true = np.bincount(run.log["label"], minlength=NUM_CLASSES[run.dataset]) / len(run.log["label"])
    rows = []
    for copy in ["zero_shot_init"] + run.copies:
        p = prediction_shares(run, copy)
        rows.append({"dataset": DISPLAY[run.dataset], "copy": COPY_LABEL[copy], "TV(pred, true) %": 50 * np.abs(p - true).sum(),
                     "top-5 classes' share of predictions %": 100 * np.sort(p)[-5:].sum(),
                     "true top-5 share %": 100 * np.sort(true)[-5:].sum()})
    return rows


def prediction_histogram_figure(run: Run, class_names: Sequence[str], top: int = 12) -> plt.Figure:
    """Share of predictions per class for every copy, against the true share; the `top` classes the oracle
    over-predicts most (all classes when there are few)."""
    true = np.bincount(run.log["label"], minlength=NUM_CLASSES[run.dataset]) / len(run.log["label"])
    copies = ["zero_shot_init"] + run.copies
    shares = {c: prediction_shares(run, c) for c in copies}
    order = np.argsort(-(shares["true_label"] - true))[:min(top, len(true))]
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(order) + 2), 3.6))
    width = 0.8 / len(copies)
    x = np.arange(len(order))
    for k, c in enumerate(copies):
        ax.bar(x + (k - (len(copies) - 1) / 2) * width, 100 * shares[c][order], width, color=COPY_COLOR[c],
               label=COPY_LABEL[c])
    ax.scatter(x, 100 * true[order], marker="_", s=600, color="black", zorder=3, label="true share")
    ax.set_xticks(x, [class_names[i] for i in order], rotation=30 if len(order) > 10 else 0, ha="right" if len(order) > 10 else "center")
    ax.set_ylabel("share of all predictions (%)")
    ax.set_title(f"{DISPLAY[run.dataset]}: which classes each copy predicts (sorted by the oracle's over-prediction)")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    return fig


def attractor_table(run: Run, class_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Per domain: its attractor = the class zero-shot CLIP predicts most often there, and the share of that domain's
    images each copy assigns to it (vs. the attractor's true share in the domain)."""
    rows, dom, label = [], run.records["domain_idx"], run.log["label"]
    for d, name in enumerate(run.domains):
        m = dom == d
        a = int(np.bincount(run.log["zs_pred"][m]).argmax())
        row = {"dataset": DISPLAY[run.dataset], "domain": name, "attractor": class_names[a],
               "true share %": 100 * float((label[m] == a).mean())}
        for copy in ["zero_shot_init"] + run.copies:
            row[COPY_LABEL[copy]] = 100 * float((run.pred(copy)[m] == a).mean())
        rows.append(row)
    return rows


def attractor_figure(rows: Sequence[Dict[str, Any]], copies: Sequence[str], title: str) -> plt.Figure:
    """Per domain: share of predictions going to the domain's zero-shot attractor class, per copy; tick = true share.
    Sorted by how much the ORACLE over-predicts it relative to Ramen."""
    keys = ["zero_shot_init"] + list(copies)
    rows = sorted(rows, key=lambda r: r[COPY_LABEL["true_label"]] - r[COPY_LABEL["zero_shot"]])
    fig, ax = plt.subplots(figsize=(7.5, 0.9 + 0.33 * len(rows)))
    for y, r in enumerate(rows):
        ax.plot([r[COPY_LABEL["zero_shot"]], r[COPY_LABEL["true_label"]]], [y, y], color="#cccccc", lw=2, zorder=1)
        ax.scatter(r["true share %"], y, marker="|", s=200, color="black", zorder=3, label="true share" if y == 0 else None)
        for k in keys:
            ax.scatter(r[COPY_LABEL[k]], y, s=28, color=COPY_COLOR[k], zorder=2, label=COPY_LABEL[k] if y == 0 else None)
    ax.set_yticks(range(len(rows)), [f"{r['domain']} → {r['attractor']}" for r in rows], fontsize=7.5)
    ax.set_xlabel("share of the domain's images predicted as its attractor class (%)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.set_size_inches(10, 0.9 + 0.33 * len(rows))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------- the memories
def entry_lookup(run: Run) -> Tuple[np.ndarray, np.ndarray]:
    """For stream ids: true label and zero-shot argmax (the class the stored entropy gradient pushes toward)."""
    return run.log["label"], run.log["zs_pred"]


def memory_matrix(run: Run, copy: str, by: str, snapshot: str = "final") -> np.ndarray:
    """(C, C) counts: rows = queue a memory entry sits in, columns = the entry's true label (by='label') or its
    zero-shot argmax (by='push')."""
    snap = run.log[f"snapshot__{snapshot}"][run.copies.index(copy)]
    label, push = entry_lookup(run)
    attr = label if by == "label" else push
    C = NUM_CLASSES[run.dataset]
    m = np.zeros((C, C), dtype=np.int64)
    for q in range(C):
        ids = snap[q][snap[q] >= 0]
        m[q] = np.bincount(attr[ids], minlength=C)
    return m


def memory_heatmap_figure(run: Run, class_names: Sequence[str], snapshot: str = "final",
                          classes: Optional[Sequence[int]] = None) -> plt.Figure:
    """Ramen's memory by true label (impure), the oracle's memory by push class (impure), and the random memory by push
    class. Row-normalised %, so each row shows what one queue holds. `classes`: show only these queues and columns,
    plus an 'other' column for everything else (for datasets with many classes)."""
    panels = [("zero_shot", "label", "Ramen memory (filed by zero-shot pred.)\ncolumns: TRUE label of the entry"),
              ("true_label", "push", "ORACLE memory (filed by true label)\ncolumns: class the entry's gradient PUSHES to"),
              ("random", "push", "random memory (control)\ncolumns: class the entry's gradient PUSHES to")]
    panels = [p for p in panels if p[0] in run.copies]
    keep = list(range(len(class_names))) if classes is None else list(classes)
    cols = [class_names[c][:14] for c in keep] + ([] if classes is None else ["(other)"])
    size = 0.42 * len(cols) + 1.6
    fig, axes = plt.subplots(1, len(panels), figsize=(size * len(panels) + 1.0, size))
    for ax, (copy, by, title) in zip(np.atleast_1d(axes), panels):
        m = memory_matrix(run, copy, by, snapshot).astype(float)[keep]
        sub = m[:, keep] if classes is None else np.column_stack([m[:, keep], m.sum(1) - m[:, keep].sum(1)])
        pct = 100 * sub / np.maximum(m.sum(1, keepdims=True), 1)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for r in range(pct.shape[0]):
            for c in range(pct.shape[1]):
                if pct[r, c] >= 1:
                    ax.text(c, r, f"{pct[r, c]:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if pct[r, c] > 55 else PALETTE["text"])
        ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(keep)), [f"{class_names[q][:14]} ({int(n):,})" for q, n in zip(keep, m.sum(1))], fontsize=7)
        ax.set_ylabel("queue (entries in it)")
        ax.set_title(title, fontsize=9)
        ax.grid(False)
    fig.suptitle(f"{DISPLAY[run.dataset]}: what each memory holds at the end of the stream (row %, {snapshot})",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def queue_domain_figure(run: Run, class_names: Sequence[str], queues: Sequence[int], snapshot: str = "final") -> plt.Figure:
    """For the chosen queues, the share of each memory's entries that come from each domain (stacked bars; Ramen's
    queue on top, the ORACLE's below)."""
    snap, dom = run.log[f"snapshot__{snapshot}"], run.records["domain_idx"]
    copies = [c for c in ("zero_shot", "true_label") if c in run.copies]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 0.6 + 0.62 * len(queues) * len(copies)))
    labels, y = [], 0
    for q in queues:
        for copy in copies:
            ids = snap[run.copies.index(copy), q]
            ids = ids[ids >= 0]
            share = np.bincount(dom[ids], minlength=len(run.domains)) / max(len(ids), 1)
            left = 0.0
            for d, v in enumerate(share):
                ax.barh(y, 100 * v, left=left, color=cmap(d), label=run.domains[d] if y == 0 else None)
                if v > 0.07:
                    ax.text(left + 50 * v, y, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.5, color="white")
                left += 100 * v
            labels.append(f"{class_names[q][:14]}: {'Ramen' if copy == 'zero_shot' else 'ORACLE'} ({len(ids):,})")
            y += 1
        y += 0.4
    ax.set_yticks([k + 0.4 * (k // len(copies)) for k in range(len(labels))], labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the queue's entries from each domain (%)")
    ax.set_title(f"{DISPLAY[run.dataset]}: where each queue's entries come from ({snapshot})", pad=22)
    ax.legend(fontsize=7, ncol=len(run.domains), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    return fig


def queue_purity(run: Run, snapshot: str = "final") -> Dict[str, np.ndarray]:
    """Per queue: label purity of Ramen's memory and push purity of the oracle's / random memory (share of entries whose
    true label resp. push class equals the queue). Empty queues are NaN."""
    out = {}
    for copy, by in (("zero_shot", "label"), ("true_label", "push"), ("random", "push")):
        if copy in run.copies:
            m = memory_matrix(run, copy, by, snapshot).astype(float)
            with np.errstate(invalid="ignore"):
                out[copy] = np.diag(m) / m.sum(1)
    return out


def purity_figure(runs: Sequence[Run]) -> plt.Figure:
    """Distribution over queues of how many of a queue's entries push toward the queue's own class (oracle, random)
    vs. how many are truly of that class (Ramen). Ramen's push purity and the oracle's label purity are 100% by
    construction."""
    fig, axes = plt.subplots(1, len(runs), figsize=(4.0 * len(runs), 3.2), squeeze=False)
    labels = {"zero_shot": "Ramen: true label = queue", "true_label": "ORACLE: push class = queue",
              "random": "random: push class = queue"}
    for ax, run in zip(axes[0], runs):
        for copy, purity in queue_purity(run).items():
            x = np.sort(100 * purity[~np.isnan(purity)])
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=COPY_COLOR[copy], lw=1.8,
                    label=f"{labels[copy]} (median {np.median(x):.0f}%)")
        ax.set(xlabel="share of a queue's entries (%)", title=DISPLAY[run.dataset], xlim=(0, 100))
        ax.legend(fontsize=6.5, loc="upper left")
    axes[0, 0].set_ylabel("share of queues with purity <= x")
    fig.suptitle("Ramen's queues all push one way; the oracle's queues push many ways", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------- images
class ImageSource:
    """Raw test images by stream id (row of records.npz -> (domain, sample index) -> upstream dataset, no transform)."""

    def __init__(self, run: Run, data_root: str = "~/data", size: int = 64):
        self.datasets = upstream.load_datasets(run.dataset, data_root, transform=None)
        self.domain_idx, self.sample_idx = run.records["domain_idx"], run.records["sample_idx"]
        self.class_names = [str(c) for c in self.datasets.classes]
        self.size = size

    def image(self, stream_id: int) -> np.ndarray:
        img, _ = self.datasets.datasets[int(self.domain_idx[stream_id])][int(self.sample_idx[stream_id])]
        img = img.convert("RGB")
        img.thumbnail((self.size, self.size))
        return np.asarray(img)


def _thumb(ax, img: np.ndarray, edge: str, width: float, caption: str, caption_color: str = PALETTE["text"]) -> None:
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(edge)
        spine.set_linewidth(width)
    ax.set_title(caption, fontsize=5.5, pad=1.5, color=caption_color)


def _caption(run: Run, e: int, text: str) -> str:
    """Thumbnail caption; on DomainNet the image's domain goes on a first line."""
    return f"{run.domains[run.records['domain_idx'][e]]}\n{text}" if run.dataset == "DomainNet" else text


def example_tables(run: Run, category: str) -> List[Dict[str, Any]]:
    """The reservoir examples of one category, one row each."""
    p = f"ex_{category}__"
    rows = []
    for i, sid in enumerate(run.log[p + "stream_id"]):
        ids, w = run.log[p + "active_ids"][i], run.log[p + "active_weight"][i]
        r_ids, o_ids = set(ids[0][ids[0] >= 0]), set(ids[1][ids[1] >= 0])
        rows.append({"example": i, "stream_id": int(sid), "domain": run.domains[run.records["domain_idx"][sid]],
                     "true": int(run.log[p + "label"][i]), "zero-shot": int(run.log[p + "zs_pred"][i]),
                     "Ramen pred": int(run.log[p + "preds"][i, 0]), "ORACLE pred": int(run.log[p + "preds"][i, 1]),
                     "active set size": len(r_ids), "shared entries %": 100 * len(r_ids & o_ids) / max(len(r_ids), 1)})
    return rows


def _push_share(run: Run, ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(C,) share of an active set's weight per push class (zero-shot argmax of the entry)."""
    ok = ids >= 0
    return np.bincount(run.log["zs_pred"][ids[ok]], weights=weights[ok], minlength=NUM_CLASSES[run.dataset]) / weights[ok].sum()


def example_figure(run: Run, src: ImageSource, category: str, i: int, per_queue: int = 6, n_queues: int = 10) -> plt.Figure:
    """One query: the Ramen and ORACLE active sets side by side. Rows = the queues carrying the most weight; each row
    shows that queue's top entries by weight w_ij. Orange border = the entry is NOT in the other copy's active set.
    Caption under each thumbnail: true label -> push class (zero-shot argmax of the entry); red = pushes away from the
    query's true class."""
    per_queue = max(per_queue, 4)  # the header needs >= 4 columns per side; unused slots stay blank (k = 1 on ImageNet-C)
    p = f"ex_{category}__"
    ids, queue, w = (run.log[p + k][i] for k in ("active_ids", "active_queue", "active_weight"))
    names, y_query = src.class_names, int(run.log[p + "label"][i])
    sets = [set(ids[c][ids[c] >= 0]) for c in (0, 1)]
    mass = [np.bincount(queue[c][ids[c] >= 0], weights=w[c][ids[c] >= 0], minlength=NUM_CLASSES[run.dataset]) for c in (0, 1)]
    queues = np.argsort(-(mass[0] / mass[0].sum() + mass[1] / mass[1].sum()))[:min(n_queues, NUM_CLASSES[run.dataset])]
    fig = plt.figure(figsize=(2 * per_queue * (0.95 if run.dataset == "DomainNet" else 0.78) + 1.8,
                              (1.15 if run.dataset == "DomainNet" else 0.95) * len(queues) + 3.4))
    two_lines = run.dataset == "DomainNet"
    grid = fig.add_gridspec(len(queues) + 2, 2 * per_queue + 1, height_ratios=[2.8, 0.45] + [1] * len(queues),
                            width_ratios=[1] * per_queue + [0.35] + [1] * per_queue, hspace=0.95 if two_lines else 0.6,
                            wspace=0.12 if two_lines else 0.08)
    _example_header(fig, grid, run, src, category, i, ids, w, per_queue)
    for c, offset in ((0, 0), (1, per_queue + 1)):
        for r, q in enumerate(queues):
            sel = np.flatnonzero((queue[c] == q) & (ids[c] >= 0))
            sel = sel[np.argsort(-w[c][sel])][:per_queue]
            for k in range(per_queue):
                ax = fig.add_subplot(grid[r + 2, offset + k])
                if k >= len(sel):
                    ax.axis("off")
                    continue
                e = int(ids[c][sel[k]])
                only = e not in sets[1 - c]
                y, pushed = run.log["label"][e], run.log["zs_pred"][e]
                _thumb(ax, src.image(e), ONLY_HERE if only else SHARED, 2.4 if only else 0.8,
                       _caption(run, e, f"{names[y][:7]}→{names[pushed][:7]}"),
                       PALETTE["worse"] if pushed != y_query else PALETTE["text"])
                if k == 0:
                    ax.set_ylabel(f"queue {names[q][:9]}\n{100 * mass[c][q] / mass[c].sum():.0f}% of w", fontsize=6,
                                  rotation=0, ha="right", va="center")
    return fig


def _example_header(fig, grid, run, src, category, i, ids, w, per_queue) -> None:
    """Query image, its labels and predictions, and the push-class weight shares of both active sets."""
    p, names = f"ex_{category}__", src.class_names
    sid = int(run.log[p + "stream_id"][i])
    y, zs, preds = int(run.log[p + "label"][i]), int(run.log[p + "zs_pred"][i]), run.log[p + "preds"][i]
    ax = fig.add_subplot(grid[0, :2])
    _thumb(ax, src.image(sid), "black", 1.0, f"query #{sid:,}")
    info = fig.add_subplot(grid[0, 2:4])
    info.axis("off")
    info.text(0.0, 0.5, f"domain: {run.domains[run.records['domain_idx'][sid]]}\ntrue: {names[y]}\nzero-shot: {names[zs]}\n"
              f"Ramen: {names[preds[0]]} {'✓' if preds[0] == y else '✗'}\nORACLE: {names[preds[1]]} {'✓' if preds[1] == y else '✗'}",
              fontsize=8, va="center", ha="left", linespacing=1.5)
    ax = fig.add_subplot(grid[0, 5:])
    share = [_push_share(run, ids[c], w[c]) for c in (0, 1)]
    top = np.argsort(-(share[0] + share[1]))[:8]
    x = np.arange(len(top))
    for c, (key, off) in enumerate((("zero_shot", -0.2), ("true_label", 0.2))):
        ax.bar(x + off, 100 * share[c][top], 0.4, color=COPY_COLOR[key], label=COPY_LABEL[key])
    ax.set_xticks(x, [("★" if t == y else "") + names[t][:10] for t in top], fontsize=7, rotation=20)
    ax.set_ylabel("active-set weight (%)", fontsize=7)
    ax.set_title("where each active set pushes\n(weight share by the entries' zero-shot class; ★ = true class)", fontsize=8)
    ax.legend(fontsize=6.5)
    for cols, txt in ((slice(0, per_queue), "Ramen active set (zero-shot filing)"),
                      (slice(per_queue + 1, None), "ORACLE active set (true-label filing)")):
        a = fig.add_subplot(grid[1, cols])
        a.axis("off")
        a.text(0.5, 0.1, txt, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        a.text(0.5, -0.25, "orange = not in the other set; caption true→push (red: push ≠ query's class)",
               ha="center", va="bottom", fontsize=5.5)


def memory_gallery_figure(run: Run, src: ImageSource, snapshot: str = "final", per_queue: int = 8,
                          queues: Optional[Sequence[int]] = None, seed: int = 0) -> plt.Figure:
    """Both memories as images: per queue, `per_queue` entries drawn at random (seeded). Border: orange = the other
    memory holds this entry in a DIFFERENT queue (filed differently: zero-shot class != true class); dashed grey = the
    other memory does not hold it at all (evicted there, or never kept: queues freeze at different times under fp16
    priorities); thin grey = same queue in both. Caption: true / zero-shot class, red where it disagrees with the queue."""
    snap = run.log[f"snapshot__{snapshot}"]
    names, label, push = src.class_names, run.log["label"], run.log["zs_pred"]
    queues = list(range(NUM_CLASSES[run.dataset])) if queues is None else list(queues)
    where = [{int(e): q for q in range(snap.shape[1]) for e in snap[c, q][snap[c, q] >= 0]} for c in (0, 1)]
    rng = np.random.default_rng(seed)
    wide = run.dataset == "DomainNet"  # larger images and two-line captions (domain on its own line)
    fig, axes = plt.subplots(len(queues), 2 * per_queue + 1,
                             figsize=(2 * per_queue * (0.95 if wide else 0.72) + 1.5, (1.15 if wide else 0.82) * len(queues) + 1.0),
                             gridspec_kw={"width_ratios": [1] * per_queue + [0.3] + [1] * per_queue,
                                          "wspace": 0.12 if wide else 0.08, "hspace": 0.8 if wide else 0.45}, squeeze=False)
    for r, q in enumerate(queues):
        axes[r, per_queue].axis("off")
        for c, offset in ((0, 0), (1, per_queue + 1)):
            ids = snap[c, q][snap[c, q] >= 0]
            pick = np.sort(rng.choice(ids, size=min(per_queue, len(ids)), replace=False)) if len(ids) else ids
            for k in range(per_queue):
                ax = axes[r, offset + k]
                if k >= len(pick):
                    ax.axis("off")
                    continue
                e = int(pick[k])
                other = where[1 - c].get(e)
                edge, width = ((SHARED, 0.8) if other == q else (ONLY_HERE, 2.4) if other is not None else ("#888888", 1.2))
                _thumb(ax, src.image(e), edge, width, _caption(run, e, f"{names[label[e]][:7]}/{names[push[e]][:7]}"),
                       PALETTE["worse"] if (label[e] if c == 0 else push[e]) != q else PALETTE["text"])
                if other is None:
                    for spine in ax.spines.values():
                        spine.set_linestyle((0, (2, 2)))
                if k == 0:
                    ax.set_ylabel(f"{names[q][:10]}\n({len(ids):,})", fontsize=6.5, rotation=0, ha="right", va="center")
    for col, txt in ((per_queue // 2, "Ramen memory (queue = zero-shot prediction)"),
                     (per_queue + 1 + per_queue // 2, "ORACLE memory (queue = true label)")):
        axes[0, col].text(0.5, 1.0, txt, transform=axes[0, col].transAxes, ha="center", va="bottom", fontsize=9,
                          fontweight="bold", color=PALETTE["text"]).set_y(1.45 if wide else 1.3)
    fig.suptitle(f"{DISPLAY[run.dataset]}: {per_queue} random entries per queue ({snapshot}). Caption: true / zero-shot class "
                 "(red = disagrees with the queue).\nBorder: orange = filed in a different queue in the other memory; "
                 "dashed = not in the other memory; thin grey = same queue in both.", fontsize=8.5, y=0.998)
    return fig


# ---------------------------------------------------------------------------- mechanism (active-set statistics)
STAT_LABEL = {"share_push_true": "pushes toward the TRUE class",
              "share_push_zs": "pushes toward the query's zero-shot class", "share_push_pred": "pushes toward the copy's own prediction",
              "share_push_top": "pushes toward the single most-pushed class", "share_label_true": "is truly of the query's class",
              "shared_with_ramen": "is also in Ramen's active set"}


def outcome_masks(run: Run, copy: str = "true_label") -> Dict[str, np.ndarray]:
    r, o = run.correct("zero_shot"), run.correct(copy)
    return {"all": np.ones_like(r), "broken (Ramen ✓, copy ✗)": r & ~o, "fixed (Ramen ✗, copy ✓)": ~r & o,
            "both right": r & o, "both wrong": ~r & ~o}


def mechanism_table(run: Run, stats: Sequence[str] = ("share_push_true", "share_push_zs", "share_push_pred", "share_push_top",
                                                       "share_label_true", "shared_with_ramen")) -> List[Dict[str, Any]]:
    """Mean share (%) of active-set weight with each property, per copy, on all samples and by outcome vs Ramen."""
    rows = []
    for group, mask in outcome_masks(run).items():
        for copy in run.copies:
            row = {"dataset": DISPLAY[run.dataset], "samples": group, "n": int(mask.sum()), "copy": COPY_LABEL[copy]}
            row.update({STAT_LABEL[s]: 100 * float(run.stat(s, copy)[mask].mean()) if mask.any() else np.nan for s in stats})
            rows.append(row)
    return rows


def mechanism_figure(runs: Sequence[Run], stat: str = "share_push_true") -> plt.Figure:
    """Per dataset and outcome group: mean share of active-set weight that pushes toward the query's true class."""
    groups = ["all", "broken (Ramen ✓, copy ✗)", "fixed (Ramen ✗, copy ✓)"]
    fig, axes = plt.subplots(1, len(runs), figsize=(3.9 * len(runs), 3.4), squeeze=False, sharey=True)
    for ax, run in zip(axes[0], runs):
        masks = outcome_masks(run)
        x = np.arange(len(groups))
        width = 0.8 / len(run.copies)
        for k, copy in enumerate(run.copies):
            vals = [100 * run.stat(stat, copy)[masks[g]].mean() for g in groups]
            bars = ax.bar(x + (k - (len(run.copies) - 1) / 2) * width, vals, width, color=COPY_COLOR[copy],
                          label=COPY_LABEL[copy])
            ax.bar_label(bars, fmt="%.0f", fontsize=6.5, padding=1)
        ax.set_xticks(x, [g.replace(" (", "\n(") + f"\nn={int(masks[g].sum()):,}" for g in groups], fontsize=7)
        ax.set_title(DISPLAY[run.dataset])
    axes[0, 0].set_ylabel(f"mean active-set weight that\n{STAT_LABEL[stat]} (%)")
    axes[0, 0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle(f"Active sets: weight that {STAT_LABEL[stat]}", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def broken_breakdown(run: Run, copy: str = "true_label") -> List[Dict[str, Any]]:
    """Images Ramen gets right and `copy` gets wrong, split by what zero-shot CLIP said: wrong and the copy kept that
    zero-shot answer (Ramen fixed it, the copy did not); wrong and the copy said something else; right (the copy broke
    a correct zero-shot answer). Same for the reverse (the copy fixed what Ramen got wrong)."""
    y, zs, r, o = run.log["label"], run.log["zs_pred"], run.pred("zero_shot"), run.pred(copy)
    rows = []
    for name, m in (("broken: Ramen right, copy wrong", (r == y) & (o != y)), ("fixed: Ramen wrong, copy right", (r != y) & (o == y))):
        n = int(m.sum())
        wrong_side = o if name.startswith("broken") else r
        parts = {"zero-shot wrong; the wrong one repeats it": m & (zs != y) & (wrong_side == zs),
                 "zero-shot wrong; the wrong one says another wrong class": m & (zs != y) & (wrong_side != zs),
                 "zero-shot right; the wrong one broke it": m & (zs == y)}
        row = {"dataset": DISPLAY[run.dataset], "copy": COPY_LABEL[copy], "images": name, "n": n,
               "the wrong one": "copy" if name.startswith("broken") else "Ramen"}
        row.update({k: f"{int(v.sum()):,} ({100 * v.sum() / max(n, 1):.1f}%)" for k, v in parts.items()})
        rows.append(row)
    return rows
