"""
ORACLE: what if Ramen's memory stored each image's CE gradient toward its TRUE class? (docs/plans/oracle_value.md)

One run per variant (ramen_lab/methods/stored_value.py); all share the stream, batch, seed and GPU model with the
week-01 Ramen baseline, so predictions are paired per image across runs:
    RamenNoSelf                 entropy value, zero-shot filing, own entry not retrieved      (control)
    CEpseudoNoSelf              CE toward the zero-shot argmax, zero-shot filing, no self      (loss-form control)
    ORACLE_CEtrueNoSelf         CE toward the true label, filed under the true label, no self  (ORACLE, primary)
    ORACLE_CEtrueZsFileNoSelf   CE toward the true label, zero-shot filing, no self            (ORACLE, CIFAR only)
Per-sample diagnostics land in results/<run_id>/stored_value.npz. Analysis: oracle_value.ipynb.

Commands (prefix: PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>, RAMEN_ENV python):

    python oracle_value.py check CIFAR10C [--n 200]
        Subset: the default RamenStoredValue must equal plain Ramen on every prediction (100%); then every variant
        once on the subset (smoke run: peak RAM / GPU, minutes).
    PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 0 --ram-gb 13 --log results/logs/gpu0_cifar.log \
        -- oracle_value.py run CIFAR10C CIFAR100C --seed 0
    PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 0 --ram-gb 10 --log results/logs/gpu0_inc.log \
        -- oracle_value.py run ImageNetC5K --seed 0
    CUDA_MODULE_LOADING=LAZY PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 0 --ram-gb 10 \
        --log results/logs/gpu0_dn.log -- oracle_value.py run DomainNet --seed 0
"""
import argparse
import gc
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the repo root, for ramen_lab
from ramen_lab import upstream  # noqa: E402
from ramen_lab.config import RunConfig  # noqa: E402
from ramen_lab.records import load_records, read_metrics  # noqa: E402
from ramen_lab.resources import ResourceBudget  # noqa: E402
from ramen_lab.runner import Runner  # noqa: E402

RESULTS = HERE / "results"
BASELINE = HERE.parents[1] / "week_01_2026-11-02" / "2026-11-01_baseline" / "results"
LOG_NAME = "stored_value.npz"

# variant name -> (method, config on top of Ramen's yaml)
VARIANTS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "RamenNoSelf": ("RamenStoredValue", {"exclude_self": True}),
    "CEpseudoNoSelf": ("RamenStoredValue", {"stored_value": "ce_pseudo", "exclude_self": True}),
    "ORACLE_CEtrueNoSelf": ("RamenStoredValueOracle",
                            {"stored_value": "ce_true", "filing": "true_label", "exclude_self": True}),
    "ORACLE_CEtrueZsFileNoSelf": ("RamenStoredValueOracle", {"stored_value": "ce_true", "exclude_self": True}),
}
MAIN = ("RamenNoSelf", "CEpseudoNoSelf", "ORACLE_CEtrueNoSelf")
VARIANTS_OF: Dict[str, Tuple[str, ...]] = {"CIFAR10C": tuple(VARIANTS), "CIFAR100C": tuple(VARIANTS),
                                           "ImageNetC5K": MAIN, "DomainNet": MAIN}


# ---------------------------------------------------------------------------- runs
def variant_config(dataset: str, variant: str, seed: int, results: Path, n_per_domain: Optional[int] = None) -> RunConfig:
    method, extra = VARIANTS[variant]
    probe = RunConfig(dataset, method=method, name=variant, seed=seed, n_per_domain=n_per_domain)
    log_path = results / probe.run_id(Runner(results, script=__file__).machine) / LOG_NAME
    overrides = {**upstream.load_method_config(dataset, "Ramen"), **extra, "log_path": str(log_path)}
    return RunConfig.make(dataset, overrides, method=method, name=variant, seed=seed, n_per_domain=n_per_domain)


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def check(dataset: str, n: int) -> None:
    """Identity on a subset, then a smoke run of every variant (resources)."""
    base = RESULTS / "check"
    ramen = Runner(base, script=__file__).run(RunConfig(dataset, method="Ramen", n_per_domain=n), overwrite=True)
    free_gpu()
    identity = Runner(base, script=__file__).run(
        RunConfig.make(dataset, upstream.load_method_config(dataset, "Ramen"), method="RamenStoredValue",
                       name="StoredValueDefault", n_per_domain=n), overwrite=True)
    free_gpu()
    r, m = load_records(ramen.run_dir), load_records(identity.run_dir)
    same = r["pred"] == m["pred"]
    print(f"default RamenStoredValue vs plain Ramen: {same.sum():,} of {len(same):,} identical ({same.mean():.2%})")
    assert same.all(), "the default setting is not Ramen: stop"
    print(f"Ramen {ramen.metrics['acc']:.2f}  {ramen.resources}", flush=True)
    for variant in VARIANTS_OF[dataset]:
        result = Runner(base, script=__file__).run(variant_config(dataset, variant, 0, base, n), overwrite=True)
        print(f"{variant}: {result.metrics['acc']:.2f}  {result.resources}", flush=True)
        free_gpu()


def run(datasets: Sequence[str], seed: int, variants: Optional[Sequence[str]] = None,
        ram_gb: Optional[float] = None) -> None:
    """ram_gb: the in-process MemoryGuard budget; None = the dataset's frozen budget (ramen_lab/config.py)."""
    for dataset in datasets:
        runner = Runner(RESULTS, script=__file__, budget=ResourceBudget(rss_gb=ram_gb) if ram_gb else None)
        baseline = RunConfig(dataset, method="Ramen", seed=seed).run_id(runner.machine)
        for variant in variants or VARIANTS_OF[dataset]:
            notes = "ORACLE (reads labels for the stored value / filing)" if "ORACLE" in variant else "label-free"
            result = runner.run(variant_config(dataset, variant, seed, RESULTS), baseline_run_id=baseline, notes=notes)
            print(f"{result.run_id}: {result.metrics['acc']:.2f}  {result.resources}", flush=True)
            free_gpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("dataset")
    c.add_argument("--n", type=int, default=200)
    r = sub.add_parser("run")
    r.add_argument("datasets", nargs="+")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--variants", nargs="*", default=None, choices=list(VARIANTS))
    r.add_argument("--ram-gb", type=float, default=None, help="MemoryGuard budget (default: the dataset's frozen one)")
    args = parser.parse_args()
    if args.cmd == "check":
        check(args.dataset, args.n)
    else:
        run(args.datasets, args.seed, args.variants, args.ram_gb)


if __name__ == "__main__":
    main()


# ============================================================================ analysis (oracle_value.ipynb)
# Pure functions: data in, numbers / figures out. The notebook calls them; nothing here touches the GPU.
from dataclasses import dataclass  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ramen_lab.analysis.plots import color  # noqa: E402
from ramen_lab.analysis.stats import cluster_bootstrap, noise_floor, outcome_counts  # noqa: E402

DISPLAY = {"CIFAR10C": "CIFAR-10-C", "CIFAR100C": "CIFAR-100-C", "ImageNetC5K": "ImageNet-C", "DomainNet": "DomainNet"}
LABEL = {"NoAdapt": "zero-shot CLIP (NoAdapt)", "Ramen": "Ramen",
         "RamenNoSelf": "Ramen without self (control)", "CEpseudoNoSelf": "CE to zero-shot label (control)",
         "ORACLE_CEtrueNoSelf": "ORACLE: CE to true label, filed by true label",
         "ORACLE_CEtrueZsFileNoSelf": "ORACLE: CE to true label, zero-shot filing"}
COLOR = {"NoAdapt": color("NoAdapt"), "Ramen": color("Ramen"), "RamenNoSelf": "#7fa7e8",
         "CEpseudoNoSelf": color("control"), "ORACLE_CEtrueNoSelf": color("variant"),
         "ORACLE_CEtrueZsFileNoSelf": "#c2a5ec"}
REFERENCE = "RamenNoSelf"  # the matched control every variant is compared with (plan: pre-registered reading)


@dataclass
class Run:
    """One run: its metrics.json, records.npz and (for the stored-value variants) stored_value.npz."""
    dataset: str
    variant: str
    run_dir: Path
    metrics: Dict[str, Any]
    records: Dict[str, np.ndarray]
    log: Optional[Dict[str, np.ndarray]]

    @property
    def correct(self) -> np.ndarray:
        return self.records["pred"] == self.records["label"]

    @property
    def acc(self) -> float:
        return self.metrics["metrics"]["acc"]


def run_dir_of(dataset: str, variant: str, seed: int = 0) -> Path:
    if variant in ("Ramen", "NoAdapt"):
        return BASELINE / RunConfig(dataset, method=variant, seed=seed).run_id("pc2")
    method = VARIANTS[variant][0]
    return RESULTS / RunConfig(dataset, method=method, name=variant, seed=seed).run_id("pc2")


def load_run(dataset: str, variant: str, seed: int = 0) -> Run:
    d = run_dir_of(dataset, variant, seed)
    log = None
    if (d / LOG_NAME).exists():
        with np.load(d / LOG_NAME) as f:
            log = {k: f[k] for k in f.files}
    return Run(dataset, variant, d, read_metrics(d / "metrics.json"), load_records(d), log)


def available(dataset: str, variants: Sequence[str]) -> List[str]:
    return [v for v in variants if (run_dir_of(dataset, v) / "metrics.json").exists()]


def load_dataset(dataset: str) -> Dict[str, Run]:
    """NoAdapt, Ramen and every finished variant; checks that all runs saw the same stream in the same order."""
    runs = {v: load_run(dataset, v) for v in available(dataset, ("NoAdapt", "Ramen") + tuple(VARIANTS))}
    ref = runs["Ramen"].records
    for v, r in runs.items():
        for key in ("sample_idx", "domain_idx", "label"):
            if not np.array_equal(r.records[key], ref[key]):
                raise AssertionError(f"{v}: {key} differs from the Ramen baseline: the runs are not paired")
    return runs


def floor_of(dataset: str) -> float:
    """Noise floor: 2 x range of the Ramen baseline's accuracy over seeds 0-4 (metrics.json)."""
    return noise_floor([read_metrics(run_dir_of(dataset, "Ramen", s) / "metrics.json")["metrics"]["acc"]
                        for s in range(5)])


def clusters(run: Run) -> np.ndarray:
    """Bootstrap clusters: one base image under all corruptions (corruption sets); each image alone on DomainNet."""
    return np.arange(len(run.correct)) if run.dataset == "DomainNet" else run.records["sample_idx"]


def data_table(runs: Dict[str, Run]) -> List[Dict[str, Any]]:
    """Images and errors per domain for NoAdapt and Ramen."""
    rec = runs["Ramen"].records
    domains = list(runs["Ramen"].metrics["metrics"]["per_domain"])
    rows = []
    for d, name in enumerate(domains):
        m = rec["domain_idx"] == d
        row = {"domain": name, "images": int(m.sum())}
        for v in ("NoAdapt", "Ramen"):
            err = int((~runs[v].correct[m]).sum())
            row[f"{v} errors"] = f"{err:,} ({100 * err / m.sum():.1f}%)"
        rows.append(row)
    return rows


def gain_table(runs: Dict[str, Run], floor: float, reference: str = REFERENCE) -> List[Dict[str, Any]]:
    """Every run against `reference`: accuracy (metrics.json), paired gain with a 95% cluster-bootstrap CI, fixed /
    broken counts, and whether the gain clears the noise floor."""
    ref = runs[reference]
    rows = []
    for v, r in runs.items():
        row = {"dataset": DISPLAY[r.dataset], "variant": LABEL[v], "key": v, "acc": r.acc, "gain": r.acc - ref.acc,
               "floor": floor}
        if v != reference:
            diff = r.correct.astype(float) - ref.correct
            row["ci_low"], row["ci_high"] = cluster_bootstrap(diff, r.records["domain_idx"], clusters(r))
            row.update(outcome_counts(r.correct, ref.correct))
            row["beyond floor"] = "yes" if abs(row["gain"]) > floor else "no"
        rows.append(row)
    return rows


def gain_figure(rows: Sequence[Dict[str, Any]], title: str, reference: str = REFERENCE) -> plt.Figure:
    """Gain over `reference` (points, 95% CI) per dataset and variant; band = +/- noise floor."""
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    keys = [k for k in ("ORACLE_CEtrueNoSelf", "ORACLE_CEtrueZsFileNoSelf", "CEpseudoNoSelf", "Ramen", "NoAdapt")
            if any(r["key"] == k for r in rows)]
    fig, ax = plt.subplots(figsize=(10.5, 1.3 + 1.2 * len(datasets)))
    step = 0.8 / max(len(keys), 1)
    for y, ds in enumerate(datasets):
        floor = next(r["floor"] for r in rows if r["dataset"] == ds)
        ax.add_patch(Rectangle((-floor, y - 0.45), 2 * floor, 0.9, color=color("Ramen"), alpha=0.10, lw=0))
        for k, key in enumerate(keys):
            r = next((r for r in rows if r["dataset"] == ds and r["key"] == key), None)
            if r is None:
                continue
            yy = y - 0.4 + step * (k + 0.5)
            ax.errorbar(r["gain"], yy, xerr=[[r["gain"] - r["ci_low"]], [r["ci_high"] - r["gain"]]], fmt="o",
                        color=COLOR[key], ms=6, capsize=3, label=LABEL[key] if y == 0 else None)
            ax.annotate(f"{r['gain']:+.2f}", (r["ci_high"], yy), xytext=(5, -3), textcoords="offset points", fontsize=7)
    ax.axvline(0, color=COLOR[reference], lw=1.2, label=f"{LABEL[reference]} (0)")
    ax.add_patch(Rectangle((0, 0), 0, 0, color=color("Ramen"), alpha=0.10, label="± noise floor (Ramen seeds 0-4)"))
    ax.set_yticks(range(len(datasets)), datasets)
    ax.set_ylim(len(datasets) - 0.5, -0.5)
    ax.set_xlabel(f"accuracy gain over {LABEL[reference]} (points, mean over domains; 95% CI)")
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig


def zs_correct(runs: Dict[str, Run]) -> np.ndarray:
    """Zero-shot correctness of every image, from the by-sample model's own zero-shot argmax (stored_value.npz)."""
    log = runs[REFERENCE].log
    return log["zs_pred"] == runs[REFERENCE].records["label"]


def zs_split_table(runs: Dict[str, Run]) -> List[Dict[str, Any]]:
    """Per run: accuracy (per image) on the images zero-shot gets right and on those it gets wrong."""
    zs = zs_correct(runs)
    rows = []
    for v, r in runs.items():
        row = {"dataset": DISPLAY[r.dataset], "variant": LABEL[v], "key": v}
        for name, m in (("zero-shot right", zs), ("zero-shot wrong", ~zs)):
            c = int(r.correct[m].sum())
            row[f"{name}: correct"] = f"{c:,} of {int(m.sum()):,} ({100 * c / m.sum():.2f}%)"
            row[f"{name} acc"] = 100 * c / m.sum()
        rows.append(row)
    return rows


def zs_split_figure(tables: Sequence[Sequence[Dict[str, Any]]]) -> plt.Figure:
    """Accuracy on zero-shot-wrong (left) and zero-shot-right (right) images, per dataset and run."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 1.4 + 0.9 * len(tables)), sharey=True)
    for ax, part in zip(axes, ("zero-shot wrong", "zero-shot right")):
        for y, rows in enumerate(tables):
            for r in rows:
                ax.plot(r[f"{part} acc"], y, "o", color=COLOR[r["key"]], ms=7,
                        label=LABEL[r["key"]] if (y == 0 and part == "zero-shot wrong") else None)
        ax.set_title(f"images {part} (per-image accuracy, %)")
        ax.set_xlabel("accuracy (%)")
    axes[0].set_yticks(range(len(tables)), [rows[0]["dataset"] for rows in tables])
    axes[0].set_ylim(len(tables) - 0.5, -0.5)
    fig.legend(fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def push_share_true(run: Run) -> np.ndarray:
    """Per image: share of the active set's push mass (w_ij ||v_j||) on entries pushing toward the image's TRUE class.
    Exact when the true class is among the 5 logged push classes, else 0 (a lower bound, flagged by `in_top5`)."""
    hit = run.log["push_top_cls"] == run.records["label"][:, None]
    return (run.log["push_top_share"].astype(np.float32) * hit).sum(1)


def mechanism_table(runs: Dict[str, Run]) -> List[Dict[str, Any]]:
    """Push mass on the true class and on the single largest push class, split by zero-shot outcome (medians)."""
    zs = zs_correct(runs)
    rows = []
    for v, r in runs.items():
        if r.log is None:
            continue
        true_share, top_share = push_share_true(r), r.log["push_top_share"][:, 0].astype(np.float32)
        in_top5 = (r.log["push_top_cls"] == r.records["label"][:, None]).any(1)
        for name, m in (("zero-shot right", zs), ("zero-shot wrong", ~zs)):
            rows.append({"dataset": DISPLAY[r.dataset], "variant": LABEL[v], "images": name,
                         "median push share, true class": float(np.median(true_share[m])),
                         "mean push share, true class": float(true_share[m].mean()),
                         "true class in top-5 push": f"{100 * in_top5[m].mean():.1f}%",
                         "median top-1 push share": float(np.median(top_share[m]))})
    return rows


def mechanism_figure(runs: Dict[str, Run], bins: int = 40) -> plt.Figure:
    """Distribution of the push share on the true class, zero-shot-wrong images (left) and right images (right)."""
    zs = zs_correct(runs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=False)
    edges = np.linspace(0, 1, bins + 1)
    for ax, (name, m) in zip(axes, (("zero-shot wrong", ~zs), ("zero-shot right", zs))):
        for v, r in runs.items():
            if r.log is None:
                continue
            share = push_share_true(r)[m]
            ax.hist(share, bins=edges, histtype="step", lw=1.6, color=COLOR[v], weights=np.full(len(share), 1 / len(share)),
                    label=f"{LABEL[v]} (median {np.median(share):.2f})")
        ax.set_title(f"{DISPLAY[next(iter(runs.values())).dataset]}: images {name}")
        ax.set_xlabel("share of push mass toward the true class")
        ax.set_ylabel("share of images")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def stream_figure(all_runs: Sequence[Dict[str, Run]], bins: int = 30) -> plt.Figure:
    """Accuracy along the stream (equal-size bins, stream order) per run: cold memory shows up on the left."""
    fig, axes = plt.subplots(1, len(all_runs), figsize=(max(4.2 * len(all_runs), 8), 3.4), squeeze=False)
    for ax, runs in zip(axes[0], all_runs):
        for v, r in runs.items():
            parts = np.array_split(r.correct, bins)
            centers = np.cumsum([len(p) for p in parts]) - np.array([len(p) for p in parts]) / 2
            ax.plot(centers / 1000, [100 * p.mean() for p in parts], color=COLOR[v], lw=1.4, label=LABEL[v])
        ax.set_title(DISPLAY[next(iter(runs.values())).dataset])
        ax.set_xlabel("stream position (×1,000 images)")
        ax.set_ylabel("accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    return fig


def sanity_table(runs: Dict[str, Run]) -> List[Dict[str, Any]]:
    """Per variant: weight share of the query's own entry in Ramen's top-k (what exclude_self removes), fp16 underflow
    of the stored values, and resources."""
    rows = []
    for v, r in runs.items():
        res = r.metrics.get("resources", {})
        row = {"dataset": DISPLAY[r.dataset], "variant": LABEL[v], "peak RAM (GB)": res.get("peak_ram_gb"),
               "RAM budget (GB)": res.get("ram_budget_gb"), "peak GPU alloc (GB)": res.get("peak_gpu_allocated_gb"),
               "minutes": res.get("minutes"), "git dirty": r.metrics.get("git_dirty")}
        if r.log is not None:
            row["mean self share"] = float(r.log["self_share"].mean())
            row["median value zero frac"] = float(np.median(r.log["value_zero_frac"]))
            row["all-zero values"] = int((r.log["value_zero_frac"] == 1).sum())
        rows.append(row)
    return rows
