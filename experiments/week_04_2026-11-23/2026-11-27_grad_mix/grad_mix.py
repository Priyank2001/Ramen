"""
Ramen stepping on lambda g_R + (1 - lambda) g_H: a lambda sweep on CIFAR-10-C (docs/plans/grad_mix.md).

g_R is Ramen's aggregated active-set gradient, g_H the image's own entropy gradient (ramen_lab/methods/grad_mix.py).
Variants: lambda in LAMBDAS, mixed `raw` (literal) or `unit` (each term normalised first); lambda 0 and 1 are the same
in both modes and run once. lambda* is chosen on the HELD-OUT corruptions only; the test-stream curve is diagnostic.
Per-image diagnostics land in results/<run_id>/stored_value.npz. Analysis: grad_mix.ipynb.

Commands (prefix: PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>, RAMEN_ENV python):

    python grad_mix.py check CIFAR10C [--n 200]
        Subset: GradMix100 must equal plain Ramen on every prediction (100%); then every variant once (smoke run).
    PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 1 --ram-gb 9 --log results/logs/gpu1_heldout.log \
        -- grad_mix.py run CIFAR10C --domains heldout --seed 0 --noadapt
    PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 0 --ram-gb 9 --log results/logs/gpu0_test.log \
        -- grad_mix.py run CIFAR10C --seed 0
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
METHOD = "RamenGradMix"
LAMBDAS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)  # docs/plans/grad_mix.md, Variants
MODES = ("raw", "unit")


def variant_name(lam: float, mode: str) -> str:
    """GradMix0 / GradMix100 (mode-free endpoints), else GradMixRaw25, GradMixUnit90, ..."""
    pct = int(round(100 * lam))
    return f"GradMix{pct}" if lam in (0.0, 1.0) else f"GradMix{mode.capitalize()}{pct}"


def build_variants() -> Dict[str, Tuple[float, str]]:
    out: Dict[str, Tuple[float, str]] = {}
    for mode in MODES:
        for lam in LAMBDAS:
            out.setdefault(variant_name(lam, mode), (lam, mode))
    return out


VARIANTS: Dict[str, Tuple[float, str]] = build_variants()  # name -> (lambda, mode); 12 variants


# ---------------------------------------------------------------------------- runs
def variant_config(dataset: str, variant: str, seed: int, results: Path, domains: str = "main",
                   n_per_domain: Optional[int] = None) -> RunConfig:
    lam, mode = VARIANTS[variant]
    probe = RunConfig(dataset, method=METHOD, name=variant, seed=seed, n_per_domain=n_per_domain, domains=domains)
    log_path = results / probe.run_id(Runner(results, script=__file__).machine) / LOG_NAME
    overrides = {**upstream.load_method_config(dataset, "Ramen"), "mix_lambda": lam, "mix_norm": mode,
                 "log_path": str(log_path)}
    return RunConfig.make(dataset, overrides, method=METHOD, name=variant, seed=seed, n_per_domain=n_per_domain,
                          domains=domains)


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def check(dataset: str, n: int) -> None:
    """Identity on a subset (GradMix100 vs plain Ramen, main and held-out), then a smoke run of every variant."""
    base = RESULTS / "check"
    for domains in ("main", "heldout"):
        ramen = Runner(base, script=__file__).run(
            RunConfig(dataset, method="Ramen", n_per_domain=n, domains=domains), overwrite=True)
        free_gpu()
        mix = Runner(base, script=__file__).run(variant_config(dataset, "GradMix100", 0, base, domains, n), overwrite=True)
        free_gpu()
        r, m = load_records(ramen.run_dir), load_records(mix.run_dir)
        same = r["pred"] == m["pred"]
        print(f"[{domains}] GradMix100 vs plain Ramen: {same.sum():,} of {len(same):,} identical ({same.mean():.2%})")
        assert same.all(), "lambda = 1 is not Ramen: stop"
        print(f"[{domains}] Ramen {ramen.metrics['acc']:.2f}  {ramen.resources}", flush=True)
    for variant in VARIANTS:
        result = Runner(base, script=__file__).run(variant_config(dataset, variant, 0, base, "main", n), overwrite=True)
        print(f"{variant}: {result.metrics['acc']:.2f}  {result.resources}", flush=True)
        free_gpu()


def run(dataset: str, seed: int, domains: str, variants: Optional[Sequence[str]] = None, noadapt: bool = False,
        ram_gb: Optional[float] = None) -> None:
    """One chain on one GPU: (NoAdapt,) then every variant. ram_gb: MemoryGuard budget (None = the frozen one)."""
    runner = Runner(RESULTS, script=__file__, budget=ResourceBudget(rss_gb=ram_gb) if ram_gb else None)
    baseline = RunConfig(dataset, method="Ramen", seed=seed, domains=domains).run_id(runner.machine)
    if domains == "heldout":  # no separate Ramen run there: GradMix100 is Ramen (identity-checked)
        baseline = variant_config(dataset, "GradMix100", seed, RESULTS, domains).run_id(runner.machine)
    if noadapt:
        result = runner.run(RunConfig(dataset, method="NoAdapt", seed=seed, domains=domains))
        print(f"{result.run_id}: {result.metrics['acc']:.2f}  {result.resources}", flush=True)
        free_gpu()
    for variant in variants or VARIANTS:
        result = runner.run(variant_config(dataset, variant, seed, RESULTS, domains), baseline_run_id=baseline,
                            notes=f"label-free; lambda={VARIANTS[variant][0]}, mix={VARIANTS[variant][1]}")
        print(f"{result.run_id}: {result.metrics['acc']:.2f}  {result.resources}", flush=True)
        free_gpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("dataset")
    c.add_argument("--n", type=int, default=200)
    r = sub.add_parser("run")
    r.add_argument("dataset")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--domains", default="main", choices=("main", "heldout"))
    r.add_argument("--variants", nargs="*", default=None, choices=list(VARIANTS))
    r.add_argument("--noadapt", action="store_true", help="also run NoAdapt first (zero-shot reference)")
    r.add_argument("--ram-gb", type=float, default=None, help="MemoryGuard budget (default: the dataset's frozen one)")
    args = parser.parse_args()
    if args.cmd == "check":
        check(args.dataset, args.n)
    else:
        run(args.dataset, args.seed, args.domains, args.variants, args.noadapt, args.ram_gb)


if __name__ == "__main__":
    main()


# ============================================================================ analysis (grad_mix.ipynb)
# Pure functions: data in, numbers / figures out. The notebook calls them; nothing here touches the GPU.
from dataclasses import dataclass  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from ramen_lab.analysis.plots import color  # noqa: E402
from ramen_lab.analysis.stats import cluster_bootstrap, noise_floor, outcome_counts  # noqa: E402

DATASET = "CIFAR10C"
MODE_COLOR = {"raw": color("variant"), "unit": "#2a9d8f"}
RAMEN = "GradMix100"  # lambda = 1: Ramen (identity-checked against the baseline)


@dataclass
class Run:
    """One run: metrics.json, records.npz and (for RamenGradMix) stored_value.npz."""
    name: str
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


def run_dir_of(name: str, domains: str, seed: int = 0) -> Path:
    """Ramen on the test stream is the week-01 baseline; everything else is in this experiment's results."""
    if name == "Ramen":
        return BASELINE / RunConfig(DATASET, method="Ramen", seed=seed).run_id("pc2")
    if name == "NoAdapt":
        root = BASELINE if domains == "main" else RESULTS
        return root / RunConfig(DATASET, method="NoAdapt", seed=seed, domains=domains).run_id("pc2")
    return RESULTS / RunConfig(DATASET, method=METHOD, name=name, seed=seed, domains=domains).run_id("pc2")


def load_run(name: str, domains: str, seed: int = 0) -> Run:
    d = run_dir_of(name, domains, seed)
    log = None
    if (d / LOG_NAME).exists():
        with np.load(d / LOG_NAME) as f:
            log = {k: f[k] for k in f.files}
    return Run(name, d, read_metrics(d / "metrics.json"), load_records(d), log)


def load_stream(domains: str) -> Dict[str, Run]:
    """NoAdapt, (Ramen on main,) and every finished variant; checks that all runs saw the same images in one order."""
    names = ["NoAdapt"] + (["Ramen"] if domains == "main" else []) + list(VARIANTS)
    runs = {n: load_run(n, domains) for n in names if (run_dir_of(n, domains) / "metrics.json").exists()}
    ref = runs[RAMEN].records
    for n, r in runs.items():
        for key in ("sample_idx", "domain_idx", "label"):
            if not np.array_equal(r.records[key], ref[key]):
                raise AssertionError(f"{n}: {key} differs from {RAMEN}: the runs are not paired")
    return runs


def floor_cifar10c() -> float:
    """Noise floor: 2 x range of the Ramen baseline over seeds 0-4 (metrics.json)."""
    return noise_floor([read_metrics(run_dir_of("Ramen", "main", s) / "metrics.json")["metrics"]["acc"]
                        for s in range(5)])


def zs_correct(runs: Dict[str, Run]) -> np.ndarray:
    """Zero-shot correctness per image, from the by-sample model's zero-shot argmax logged by every variant."""
    return runs[RAMEN].log["zs_pred"] == runs[RAMEN].records["label"]


def sweep_table(runs: Dict[str, Run]) -> List[Dict[str, Any]]:
    """Per variant: lambda, mix, accuracy (metrics.json), gain over lambda = 1, fixed / broken images versus it, and
    accuracy on the images zero-shot gets right / wrong."""
    ref, zs = runs[RAMEN], zs_correct(runs)
    rows = []
    for n, (lam, mode) in VARIANTS.items():
        if n not in runs:
            continue
        r = runs[n]
        row = {"variant": n, "lambda": lam, "mix": mode if lam not in (0.0, 1.0) else "-", "acc": r.acc,
               "gain vs lambda=1": r.acc - ref.acc, **outcome_counts(r.correct, ref.correct)}
        for part, m in (("zs right", zs), ("zs wrong", ~zs)):
            row[f"acc on {part} (%)"] = 100 * r.correct[m].mean()
        rows.append(row)
    return rows


def select_lambda(heldout: Dict[str, Run]) -> str:
    """Pre-registered: the variant with the highest held-out accuracy (ties: the larger lambda, closer to Ramen)."""
    names = [n for n in VARIANTS if n in heldout]
    return max(names, key=lambda n: (heldout[n].acc, VARIANTS[n][0]))


def gate_row(test: Dict[str, Run], chosen: str, floor: float) -> Dict[str, Any]:
    """lambda* against the Ramen baseline on the test stream, with a 95% CI clustered by base image."""
    v, b = test[chosen], test["Ramen"]
    diff = v.correct.astype(float) - b.correct
    low, high = cluster_bootstrap(diff, v.records["domain_idx"], v.records["sample_idx"])
    gain = v.acc - b.acc
    return {"lambda*": chosen, "test acc": v.acc, "Ramen acc (baseline seed 0)": b.acc, "gain": gain,
            "95% CI": f"[{low:+.2f}, {high:+.2f}]", "noise floor": floor, "passes gate": bool(gain > floor),
            **outcome_counts(v.correct, b.correct)}


def sweep_figure(streams: Dict[str, Dict[str, Run]], floor: float, chosen: Optional[str] = None) -> plt.Figure:
    """Accuracy against lambda per stream; one line per mix (lambda 0 and 1 shared). Band: +/- floor around lambda = 1."""
    fig, axes = plt.subplots(1, len(streams), figsize=(5.6 * len(streams), 3.8), squeeze=False)
    for ax, (title, runs) in zip(axes[0], streams.items()):
        ramen = runs[RAMEN].acc
        ax.axhspan(ramen - floor, ramen + floor, color=color("Ramen"), alpha=0.12, lw=0,
                   label=f"Ramen (lambda = 1) ± noise floor ({floor:.2f})")
        if "NoAdapt" in runs:
            ax.axhline(runs["NoAdapt"].acc, color=color("NoAdapt"), ls="--", lw=1.2, label="zero-shot (NoAdapt)")
        for mode in MODES:
            pts = sorted((lam, runs[n].acc) for n, (lam, m) in VARIANTS.items()
                         if n in runs and (m == mode or lam in (0.0, 1.0)))
            ax.plot(*zip(*pts), "o-", color=MODE_COLOR[mode], lw=1.6, ms=5, label=f"mix = {mode}")
        if chosen and chosen in runs:
            ax.plot(VARIANTS[chosen][0], runs[chosen].acc, "*", color="black", ms=13, label=f"lambda* ({chosen})")
        ax.set(xlabel="lambda (1 = Ramen's active set only, 0 = own gradient only)", ylabel="accuracy (%, mean over domains)",
               title=title)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    return fig


def zs_split_figure(rows: Sequence[Dict[str, Any]], title: str) -> plt.Figure:
    """Accuracy on zero-shot-wrong (left) and zero-shot-right (right) images against lambda, per mix."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, part in zip(axes, ("zs wrong", "zs right")):
        for mode in MODES:
            pts = sorted((r["lambda"], r[f"acc on {part} (%)"]) for r in rows if r["mix"] in (mode, "-"))
            ax.plot(*zip(*pts), "o-", color=MODE_COLOR[mode], lw=1.6, ms=5, label=f"mix = {mode}")
        ax.set(xlabel="lambda", ylabel="accuracy (%)", title=f"images zero-shot gets {part.split()[1]}")
        ax.legend(fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def mechanism_table(runs: Dict[str, Run]) -> List[Dict[str, Any]]:
    """Per variant: median ||g_R|| / ||g_H||, the median effective share of g_R in the raw mix
    (lambda r / (lambda r + 1 - lambda), r = the norm ratio; for unit it is lambda), and the median share of
    coordinates whose step sign differs from Ramen's."""
    rows = []
    for n, (lam, mode) in VARIANTS.items():
        if n not in runs or runs[n].log is None:
            continue
        log = runs[n].log
        ratio = log["norm_R"] / np.maximum(log["norm_H"], 1e-30)
        share = lam * ratio / (lam * ratio + (1 - lam)) if mode == "raw" else np.full_like(ratio, lam)
        rows.append({"variant": n, "lambda": lam, "mix": mode if lam not in (0.0, 1.0) else "-",
                     "median ||g_R||/||g_H||": float(np.median(ratio)),
                     "median effective share of g_R": float(np.median(share)),
                     "median flipped coords vs Ramen": float(np.median(log["flip_frac"])),
                     "mean flipped coords vs Ramen": float(log["flip_frac"].mean())})
    return rows


def sanity_table(runs: Dict[str, Run]) -> List[Dict[str, Any]]:
    rows = []
    for n, r in runs.items():
        res = r.metrics.get("resources", {})
        rows.append({"run": r.metrics["run_id"], "gpu": r.metrics.get("gpu"), "git dirty": r.metrics.get("git_dirty"),
                     "peak RAM (GB)": res.get("peak_ram_gb"), "RAM budget (GB)": res.get("ram_budget_gb"),
                     "peak GPU alloc (GB)": res.get("peak_gpu_allocated_gb"), "minutes": res.get("minutes")})
    return rows
