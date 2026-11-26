"""
Does Ramen's update direction agree with the cross-entropy gradient? (docs/plans/grad_alignment.md)

ORACLE diagnostic: `RamenGradAlignment` (ramen_lab/methods/grad_alignment.py) is Ramen plus label-reading probes. Its
predictions equal Ramen's, so accuracy matches the baseline run on the same seed, GPU and batch. The per-sample
cosines land in results/<run_id>/alignment.npz, in the same order as that run's records.npz. Analysis: grad_alignment.ipynb.

Commands (prefix: PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>, RAMEN_ENV python):

    python grad_alignment.py check CIFAR10C [--n 200]
        Subset run: predictions must equal plain Ramen's (100%), and the CE gradient with loss scale 1 and 1024 must
        agree where scale 1 did not underflow.
    PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 0 --ram-gb 8.2 --log results/logs/ga_c10_s0.log \
        -- grad_alignment.py run CIFAR10C --seed 0
        Full stream at the frozen batch.
"""
import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the repo root, for ramen_lab
from ramen_lab import upstream  # noqa: E402
from ramen_lab.analysis.plots import color  # noqa: E402
from ramen_lab.config import RunConfig  # noqa: E402
from ramen_lab.records import load_records  # noqa: E402
from ramen_lab.runner import Runner  # noqa: E402

RESULTS = HERE / "results"
METHOD = "RamenGradAlignment"


def probe_config(dataset: str, seed: int, results: Path, n_per_domain: Optional[int] = None,
                 loss_scale: Optional[float] = None) -> RunConfig:
    """Ramen's own yaml as overrides (upstream has no yaml for the probe) plus where to write alignment.npz."""
    run_dir = results / RunConfig(dataset, method=METHOD, seed=seed, n_per_domain=n_per_domain).run_id(
        Runner(results, script=__file__).machine)
    overrides = {**upstream.load_method_config(dataset, "Ramen"), "probe_path": str(run_dir / "alignment.npz")}
    if loss_scale is not None:
        overrides["probe_loss_scale"] = loss_scale
    return RunConfig.make(dataset, overrides, method=METHOD, seed=seed, n_per_domain=n_per_domain)


def load_alignment(run_dir: Path) -> Dict[str, np.ndarray]:
    with np.load(Path(run_dir) / "alignment.npz") as f:
        return {name: f[name] for name in f.files}


def free_gpu() -> None:
    """Each run builds a full Ramen cache on the GPU; drop the previous run's before the next one."""
    gc.collect()
    torch.cuda.empty_cache()


def check(dataset: str, n: int) -> None:
    base = RESULTS / "check"
    ramen = Runner(base / "ramen", script=__file__).run(RunConfig(dataset, method="Ramen", n_per_domain=n), overwrite=True)
    free_gpu()
    probe = Runner(base / "probe", script=__file__).run(probe_config(dataset, 0, base / "probe", n), overwrite=True)
    r, p = load_records(ramen.run_dir), load_records(probe.run_dir)
    same = r["pred"] == p["pred"]
    print(f"probe vs Ramen: {same.sum():,} of {len(same):,} predictions identical ({same.mean():.2%})")
    a = load_alignment(probe.run_dir)
    print(f"rows {len(a['cos_R_CE']):,}; undefined cos_R_CE (zero CE gradient): {np.isnan(a['cos_R_CE']).mean():.1%}")
    free_gpu()
    s1 = Runner(base / "scale1", script=__file__).run(probe_config(dataset, 0, base / "scale1", n, 1.0), overwrite=True)
    b = load_alignment(s1.run_dir)
    ok = ~np.isnan(b["cos_R_CE"]) & ~np.isnan(a["cos_R_CE"])
    diff = np.abs(a["cos_R_CE"][ok] - b["cos_R_CE"][ok])
    print(f"scale 1 vs 1024: undefined {np.isnan(b['cos_R_CE']).mean():.1%} vs {np.isnan(a['cos_R_CE']).mean():.1%}; "
          f"max |dcos| where both defined {diff.max():.4f}")
    assert same.all(), "the probe changed Ramen's predictions: stop"


def run(dataset: str, seed: int) -> None:
    runner = Runner(RESULTS, script=__file__)
    baseline = RunConfig(dataset, method="Ramen", seed=seed).run_id(runner.machine)  # e.g. pc2_C10_Ramen_b100_s0
    result = runner.run(probe_config(dataset, seed, RESULTS), baseline_run_id=baseline)
    print(f"{result.run_id}: {result.metrics['acc']:.2f}  {result.resources}", flush=True)


# ---------------------------------------------------------------------------- analysis (grad_alignment.ipynb)
STATS = {"cos_R_CE": "cos(g_R, g_CE)", "cos_signR_CE": "cos(sign g_R, g_CE)  [the step taken]",
         "cos_H_CE": "cos(g_H, g_CE)  [own entropy gradient]", "cos_R_CEnull": "null: cos(g_R, other sample's g_CE)"}
STAT_COLORS = {"cos_R_CE": "Ramen", "cos_signR_CE": "variant", "cos_H_CE": "NoAdapt", "cos_R_CEnull": "control"}


def summary_rows(a: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    """Per statistic: median, mean, count and share of samples with cosine > 0, overall and by zero-shot outcome."""
    right = a["zs_correct"] == 1
    rows = []
    for key, label in STATS.items():
        for group, mask in (("all", np.ones_like(right)), ("zero-shot right", right), ("zero-shot wrong", ~right)):
            x = a[key][mask]
            x = x[~np.isnan(x)]
            rows.append({"statistic": label, "samples": group, "n": len(x), "median": float(np.median(x)),
                         "mean": float(x.mean()), "n_positive": int((x > 0).sum()), "share_positive": float((x > 0).mean())})
    return rows


def ecdf_figure(a: Dict[str, np.ndarray], title: str) -> plt.Figure:
    """ECDF of each cosine; the curve further right is the better aligned. Dotted line marks cosine 0."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for key, label in STATS.items():
        x = np.sort(a[key][~np.isnan(a[key])])
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=color(STAT_COLORS[key]), label=label, lw=1.8)
    ax.axvline(0, color="#999999", ls=":")
    ax.set(xlabel="cosine with the cross-entropy gradient", ylabel="share of samples with cosine <= x", xlim=(-1, 1))
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def histogram_figure(a: Dict[str, np.ndarray], title: str, keys: Sequence[str] = ("cos_R_CE", "cos_H_CE", "cos_R_CEnull"),
                     bins: int = 40) -> plt.Figure:
    """Overlaid histograms (share of samples per bin) of the chosen cosines: all samples, zero-shot right, zero-shot wrong."""
    right = a["zs_correct"] == 1
    edges = np.linspace(-1, 1, bins + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, (group, mask) in zip(axes, (("all samples", np.ones_like(right)), ("zero-shot right", right),
                                        ("zero-shot wrong", ~right))):
        for key in keys:
            x = a[key][mask]
            x = x[~np.isnan(x)]
            ax.hist(x, bins=edges, weights=np.full(len(x), 100 / len(x)), histtype="stepfilled", alpha=0.3,
                    color=color(STAT_COLORS[key]))
            ax.hist(x, bins=edges, weights=np.full(len(x), 100 / len(x)), histtype="step", lw=1.5,
                    color=color(STAT_COLORS[key]), label=f"{STATS[key]}  (median {np.median(x):.2f})")
            ax.axvline(np.median(x), color=color(STAT_COLORS[key]), ls="--", lw=1)
        ax.axvline(0, color="#999999", ls=":")
        ax.set(xlabel="cosine with g_CE", title=f"{group} (n = {int(mask.sum()):,})", xlim=(-1, 1))
        ax.legend(fontsize=6.5, loc="upper left")
    axes[0].set_ylabel("share of samples per bin (%)")
    fig.suptitle(title, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def domain_figure(a: Dict[str, np.ndarray], domain_idx: np.ndarray, names: Sequence[str], title: str) -> plt.Figure:
    """Median cos(g_R, g_CE) per domain, split by whether zero-shot CLIP was right on the sample."""
    right = a["zs_correct"] == 1
    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.45 * len(names)))
    med = {g: [np.nanmedian(a["cos_R_CE"][(domain_idx == d) & m]) if ((domain_idx == d) & m).any() else np.nan
               for d in range(len(names))] for g, m in (("zero-shot right", right), ("zero-shot wrong", ~right))}
    order = np.argsort(np.nan_to_num(med["zero-shot right"]))
    y = np.arange(len(names))
    for g, c, off in (("zero-shot right", "Ramen", -0.18), ("zero-shot wrong", "worse", 0.18)):
        ax.barh(y + off, np.array(med[g])[order], height=0.34, color=color(c), label=g)
    ax.set_yticks(y, [names[i] for i in order])
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set(xlabel="median cos(g_R, g_CE)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def layer_figure(a: Dict[str, np.ndarray], title: str) -> plt.Figure:
    """Median per-layer cosine with the CE gradient, for Ramen's g_R and the own entropy gradient g_H.
    Layers left to right: ln_pre, the transformer blocks (ln_1, ln_2), ln_post, in model order; weight and bias apart."""
    names = [str(n) for n in a["layer_names"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, kind in zip(axes, ("weight", "bias")):
        idx = [i for i, n in enumerate(names) if n.endswith(kind)]
        for key, c, label in (("layer_cos_R_CE", "Ramen", "g_R"), ("layer_cos_H_CE", "NoAdapt", "g_H")):
            ax.plot(range(len(idx)), np.nanmedian(a[key][:, idx].astype(np.float32), axis=0), "o-", ms=3,
                    color=color(c), label=label)
        ax.axhline(0, color="#999999", ls=":")
        ax.set(xlabel=f"LayerNorm index in model order ({kind})", title=f"LayerNorm {kind}")
    axes[0].set_ylabel("median cosine with g_CE")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def stream_figure(a: Dict[str, np.ndarray], title: str, bins: int = 30) -> plt.Figure:
    """Mean cos(g_R, g_CE) along the stream: low at the start would mean a cold memory."""
    n = len(a["cos_R_CE"])
    edges = np.linspace(0, n, bins + 1).astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for key, c in (("cos_R_CE", "Ramen"), ("cos_R_CEnull", "control")):
        ax.plot((edges[:-1] + edges[1:]) / 2000, [np.nanmean(a[key][i:j]) for i, j in zip(edges[:-1], edges[1:])],
                "o-", ms=3, color=color(c), label=STATS[key])
    ax.set(xlabel="stream position (x1,000 images)", ylabel="mean cosine")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def ce_change_figure(a: Dict[str, np.ndarray], title: str, bins: int = 10) -> plt.Figure:
    """Realised CE change of the step (after - before, per sample) by decile of cos(g_R, g_CE). Negative = CE went down."""
    ok = ~np.isnan(a["cos_R_CE"])
    cos, d = a["cos_R_CE"][ok], (a["ce_after"] - a["ce_before"])[ok]
    edges = np.quantile(cos, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, cos, side="right") - 1, 0, bins - 1)
    mean = np.array([d[which == b].mean() for b in range(bins)])
    centre = np.array([cos[which == b].mean() for b in range(bins)])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(range(bins), mean, color=[color("Ramen") if m < 0 else color("worse") for m in mean])
    ax.set_xticks(range(bins), [f"{c:.2f}" for c in centre], fontsize=7)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set(xlabel="mean cos(g_R, g_CE) of the decile (low -> high)", ylabel="mean CE change (after - before)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("dataset")
    c.add_argument("--n", type=int, default=200)
    r = sub.add_parser("run")
    r.add_argument("dataset")
    r.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    check(args.dataset, args.n) if args.cmd == "check" else run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
