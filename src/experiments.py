#!/usr/bin/env python3
"""
Experiment sweep runner (converted from experiments.ipynb).

Runs the TTA pipeline in-process over a (dataset x model x tta_algo x flags) grid,
parses any results already saved under ../log/, and writes a LaTeX report
(experiments_report.tex + figures/) summarising every output so the results are
readable without opening the notebook.

Usage (from src/):
    python experiments.py                     # run sweep + diagnostics + t-SNE, write report
    python experiments.py --no-sweep          # skip the grid sweep, only parse ../log/
    python experiments.py --no-diagnostics --no-tsne
    python experiments.py --out-dir ../report --cpu

Typical tmux workflow:
    tmux new -s ramen
    cd src && python experiments.py 2>&1 | tee ../log/experiments_run.txt
    # detach: Ctrl-b d   reattach: tmux attach -t ramen
    # then build the PDF:  cd ../report && pdflatex experiments_report.tex
"""

import argparse
import csv
import itertools
import os
import random
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # headless: no display under tmux
import matplotlib.pyplot as plt
import pandas as pd

from models.get_pretrained_model import get_pretrained_model
from datasets import get_dataset_class
from datasets.utils import TaggedMultipleDataset
from methods import get_method_class


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
DATA_ROOT = os.path.expanduser("~/data")
CONFIG_ROOT = "../cfg"
LOG_ROOT = "../log"
NUM_WORKERS = 0
NUM_THREADS = 4
SEED = 0

# (dataset, model, batch_size) triples, same pairing as shell/run_ramen.sh
DATASET_MODEL_BATCH = [
    ("CIFAR10C", "clip_vitbase32", 100),
    ("CIFAR100C", "clip_vitbase16", 100),
    # ("ImageNetC5K", "clip_vitlarge14", 50),
    # ("DomainNet",   "clip_vitbase32", 100),
]
TTA_ALGOS = ["NoAdapt", "Tent", "Ramen"]
USE_TBN_OPTIONS = [False, True]
TTA_MODES = ["mixed"]  # add "single" to compare shared-state vs per-domain-reset adaptation

DIAGNOSTICS_CONFIGS = [
    dict(dataset="CIFAR10C", model="clip_vitbase32", batch_size=100, tta_algo="Ramen", use_tbn=False, tta_mode="mixed"),
    dict(dataset="CIFAR100C", model="clip_vitbase16", batch_size=100, tta_algo="Ramen", use_tbn=False, tta_mode="mixed"),
]
TSNE_CONFIGS = [
    dict(dataset="CIFAR10C", model="clip_vitbase32", batch_size=100),
    dict(dataset="CIFAR100C", model="clip_vitbase16", batch_size=100),
]


# --------------------------------------------------------------------------------------
# Pipeline helpers (copied from main.py / the notebook)
# --------------------------------------------------------------------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def single_domain_test(datasets, tta_model, args):
    dataset_accs = []
    for i, dataset in enumerate(datasets):
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True)
        num_correct, num_total = 0, 0
        for image, label in tqdm(dataloader):
            image, label = image.to(args.device), label.to(args.device)
            logits = tta_model(image)
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                num_correct += pred.eq(label).sum().item()
                num_total += label.size(0)
        dataset_accs.append(num_correct / num_total)
        tqdm.write(f"{datasets.environments[i]}: {dataset_accs[-1]}")
        tta_model.reset()
    return datasets.environments, dataset_accs


def mixed_domain_test(datasets, tta_model, args):
    mix_dataset = TaggedMultipleDataset(datasets)
    dataset_num_corrects = torch.zeros(len(datasets), dtype=torch.int)
    dataset_num_samples = np.array([len(dataset) for dataset in datasets])
    dataloader = DataLoader(mix_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    for image, label, domain_idx, sample_idx in tqdm(dataloader):
        image, label = image.to(args.device), label.to(args.device)
        logits = tta_model(image)
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            is_correct = pred.eq(label).cpu().int()
            dataset_num_corrects.index_add_(0, domain_idx, is_correct)
    dataset_accs = dataset_num_corrects.numpy() / dataset_num_samples
    tta_model.reset()
    return datasets.environments, dataset_accs


def load_tta_config(config_root, dataset, tta_algo):
    filepaths = [
        os.path.join(config_root, dataset, tta_algo + ".yaml"),
        os.path.join(config_root, "default", tta_algo + ".yaml"),
    ]
    for filepath in filepaths:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return yaml.safe_load(f) or {}, filepath
    return {}, None


def build_args(dataset, model, tta_algo, tta_mode="mixed", batch_size=100, use_tbn=False,
               cuda=True, seed=SEED, config_overrides=None, data_root=DATA_ROOT,
               config_root=CONFIG_ROOT, num_workers=NUM_WORKERS, num_threads=NUM_THREADS):
    config, config_path = load_tta_config(config_root, dataset, tta_algo)
    config = dict(config)
    if config_overrides:
        config.update(config_overrides)
    args = SimpleNamespace(
        dataset=dataset, data_root=data_root, tta_mode=tta_mode, model=model,
        use_tbn=use_tbn, tta_algo=tta_algo, batch_size=batch_size,
        num_workers=num_workers, seed=seed, cuda=cuda, num_threads=num_threads,
        config=config, config_path=config_path, save_to=None,
    )
    args.device = torch.device("cuda") if torch.cuda.is_available() and cuda else torch.device("cpu")
    args.max_batch_size = batch_size
    return args


def run_single_experiment(args, save_to=None, verbose=True):
    setup_seed(args.seed)
    if verbose:
        print("-" * 80)
        print(f"dataset={args.dataset} model={args.model} tta_algo={args.tta_algo} "
              f"tta_mode={args.tta_mode} use_tbn={args.use_tbn} batch_size={args.batch_size}")

    model, preprocess = get_pretrained_model(args)
    datasets = get_dataset_class(args.dataset)(root=args.data_root, transform=preprocess)
    tta_method = get_method_class(args.tta_algo)(model, datasets, args)

    if args.tta_mode == "single":
        domain_names, dataset_accs = single_domain_test(datasets, tta_method, args)
    elif args.tta_mode == "mixed":
        domain_names, dataset_accs = mixed_domain_test(datasets, tta_method, args)
    else:
        raise ValueError(f"Unknown tta mode: {args.tta_mode}")

    avg_acc = float(np.mean(dataset_accs))
    if verbose:
        print("-" * 80)
        for env, acc in zip(domain_names, dataset_accs):
            print(f"{env}: {acc * 100:.2f}%")
        print(f"total: {avg_acc * 100:.2f}%")

    record = dict(
        dataset=args.dataset, model=args.model, tta_algo=args.tta_algo, tta_mode=args.tta_mode,
        use_tbn=args.use_tbn, batch_size=int(args.batch_size), seed=int(args.seed), avg_acc=avg_acc,
        domains=list(domain_names), accs=[float(a) for a in dataset_accs], config=dict(args.config),
    )

    if save_to is not None:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, "a", newline="") as f:
            writer = csv.writer(f)
            args_dict = {k: str(v) for k, v in vars(args).items() if isinstance(v, (int, str, float, bool))}
            writer.writerow([f"{k}={v}" for k, v in args_dict.items()])
            writer.writerow([f"{k}={v}" for k, v in args.config.items()])
            writer.writerow(list(domain_names) + ["Average"])
            writer.writerow([f"{a * 100:.2f}" for a in dataset_accs] + [f"{avg_acc * 100:.2f}"])
            writer.writerow([])

    del model, tta_method
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def run_sweep(experiment_configs, save_dir=LOG_ROOT, save_csv=True):
    records = []
    for i, cfg in enumerate(experiment_configs):
        print(f"\n=== [{i + 1}/{len(experiment_configs)}] {cfg} ===")
        try:
            args = build_args(**cfg)
            save_to = None
            if save_csv:
                suffix = "_tbn1" if args.use_tbn else ""
                save_to = os.path.join(save_dir, f"{args.dataset}_{args.tta_mode}",
                                       f"{args.tta_algo}_{args.model}{suffix}.csv")
            records.append(run_single_experiment(args, save_to=save_to))
        except Exception as e:
            print(f"FAILED: {cfg} -> {e}")
            records.append(dict(cfg, error=str(e)))
    return records


def records_to_long_df(records):
    rows = []
    for r in records:
        if "error" in r:
            continue
        for domain, acc in zip(r["domains"], r["accs"]):
            rows.append(dict(
                dataset=r["dataset"], model=r["model"], tta_algo=r["tta_algo"],
                tta_mode=r["tta_mode"], use_tbn=bool(r["use_tbn"]), batch_size=r["batch_size"],
                seed=r["seed"], domain=domain, accuracy=acc * 100, avg_acc=r["avg_acc"] * 100,
            ))
    return pd.DataFrame(rows)


def parse_log_csv(path):
    with open(path, newline="") as f:
        rows = [row for row in csv.reader(f)]
    records = []
    i = 0
    while i < len(rows):
        if not rows[i]:
            i += 1
            continue
        args_row, config_row, header_row, values_row = rows[i:i + 4]
        args_dict = dict(kv.split("=", 1) for kv in args_row)
        domains = header_row[:-1]
        accs = [float(v) / 100 for v in values_row[:-1]]
        avg_acc = float(values_row[-1]) / 100
        records.append(dict(
            dataset=args_dict.get("dataset"), model=args_dict.get("model"),
            tta_algo=args_dict.get("tta_algo"), tta_mode=args_dict.get("tta_mode"),
            use_tbn=args_dict.get("use_tbn") == "True",
            batch_size=int(args_dict.get("batch_size", -1)),
            seed=int(args_dict.get("seed", -1)),
            domains=domains, accs=accs, avg_acc=avg_acc, source_file=path,
        ))
        i += 4
    return records


def load_all_logs(log_root=LOG_ROOT):
    all_records = []
    for dirpath, _, filenames in os.walk(log_root):
        for fname in sorted(filenames):
            if fname.endswith(".csv"):
                try:
                    all_records.extend(parse_log_csv(os.path.join(dirpath, fname)))
                except Exception as e:
                    print(f"skipping unparseable log {fname}: {e}")
    return all_records


# --------------------------------------------------------------------------------------
# Diagnostics: per-sample predictions + confusion matrices
# --------------------------------------------------------------------------------------
def collect_predictions(datasets, tta_model, args):
    records = []
    if args.tta_mode == "single":
        for domain_idx, dataset in enumerate(datasets):
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
            offset = 0
            for image, label in tqdm(dataloader, desc=datasets.environments[domain_idx]):
                image, label = image.to(args.device), label.to(args.device)
                logits = tta_model(image)
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                for b in range(label.size(0)):
                    records.append(dict(domain_idx=domain_idx, sample_idx=offset + b,
                                        label=int(label[b]), pred=int(pred[b])))
                offset += label.size(0)
            tta_model.reset()
    elif args.tta_mode == "mixed":
        mix_dataset = TaggedMultipleDataset(datasets)
        dataloader = DataLoader(mix_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True)
        for image, label, domain_idx, sample_idx in tqdm(dataloader):
            image, label = image.to(args.device), label.to(args.device)
            logits = tta_model(image)
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
            for b in range(label.size(0)):
                records.append(dict(domain_idx=int(domain_idx[b]), sample_idx=int(sample_idx[b]),
                                    label=int(label[b]), pred=int(pred[b])))
        tta_model.reset()
    else:
        raise ValueError(f"Unknown tta mode: {args.tta_mode}")
    return pd.DataFrame(records)


def run_for_confusion(cfg, verbose=True):
    args = build_args(**cfg)
    setup_seed(args.seed)
    model, preprocess = get_pretrained_model(args)
    datasets = get_dataset_class(args.dataset)(root=args.data_root, transform=preprocess)
    tta_method = get_method_class(args.tta_algo)(model, datasets, args)

    pred_df = collect_predictions(datasets, tta_method, args)
    pred_df["dataset"] = args.dataset
    pred_df["tta_algo"] = args.tta_algo
    pred_df["domain"] = pred_df["domain_idx"].map(lambda i: datasets.environments[i])
    pred_df["class_name"] = pred_df["label"].map(lambda i: datasets.classes[i])
    pred_df["pred_class_name"] = pred_df["pred"].map(lambda i: datasets.classes[i])
    pred_df["correct"] = pred_df["label"] == pred_df["pred"]
    if verbose:
        acc = pred_df["correct"].mean()
        print(f"{args.dataset}/{args.tta_algo}: overall acc = {acc * 100:.2f}% over {len(pred_df)} samples")

    del model, tta_method
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return datasets, pred_df


def compute_confusion_matrix(pred_df, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(cm, (pred_df["label"].values, pred_df["pred"].values), 1)
    return cm


def top_confusions(cm, class_names, k=10):
    cm = cm.copy()
    np.fill_diagonal(cm, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-cm, axis=None), cm.shape))[0][:k]
    rows = [dict(true=class_names[i], pred=class_names[j], count=int(cm[i, j]))
            for i, j in idx if cm[i, j] > 0]
    return pd.DataFrame(rows)


def plot_confusion_matrix(cm, class_names, title, path, normalize=True, annotate_max=20):
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            disp = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    else:
        disp = cm.astype(float)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=disp.max() or 1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    n = len(class_names)
    if n <= annotate_max:
        ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(class_names, fontsize=7)
        for i in range(n):
            for j in range(n):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if disp[i, j] > 0.5 else "black")
    else:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_domain_accuracy_grid(all_df, path):
    datasets = sorted(all_df["dataset"].unique())
    combos = [(d, a) for d in datasets
              for a in sorted(all_df[all_df["dataset"] == d]["tta_algo"].unique())]
    if not combos:
        return None
    ncols = 2
    nrows = int(np.ceil(len(combos) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax, (dataset, algo) in zip(axes, combos):
        sub = all_df[(all_df["dataset"] == dataset) & (all_df["tta_algo"] == algo)
                     & (~all_df["use_tbn"])].sort_values("accuracy")
        if sub.empty:
            ax.axis("off"); continue
        avg = sub["accuracy"].mean()
        colors = ["crimson" if a < avg else "steelblue" for a in sub["accuracy"]]
        ax.bar(sub["domain"], sub["accuracy"], color=colors)
        ax.axhline(avg, color="black", linestyle="--", linewidth=1, label=f"avg = {avg:.1f}%")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{dataset} | {algo} | use_tbn=False")
        ax.tick_params(axis="x", rotation=60)
        ax.legend()
    for ax in axes[len(combos):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# t-SNE
# --------------------------------------------------------------------------------------
def run_tsne_section(out_dir, args_cuda):
    import torch.nn.functional as F
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn not installed -> skipping t-SNE section")
        return []

    class PretrainedFeaturizer:
        def __init__(self, model):
            self.model = model.visual
            self.dtype = self.model.conv1.weight.dtype

        def featurize(self, x):
            return F.normalize(self.model(x.type(self.dtype)), dim=1)

    @torch.no_grad()
    def extract_features(featurizer, datasets, args, max_samples_per_domain=150, seed=SEED):
        rng = np.random.RandomState(seed)
        feat_chunks, meta_rows = [], []
        for domain_idx, dataset in enumerate(datasets):
            n = len(dataset)
            idxs = rng.choice(n, size=min(max_samples_per_domain, n), replace=False)
            subset = torch.utils.data.Subset(dataset, idxs)
            dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
            ptr = 0
            for image, label in tqdm(dataloader, desc=f"features: {datasets.environments[domain_idx]}"):
                feat = featurizer.featurize(image.to(args.device)).float().cpu().numpy()
                feat_chunks.append(feat)
                for b in range(feat.shape[0]):
                    meta_rows.append(dict(domain_idx=domain_idx, sample_idx=int(idxs[ptr + b]),
                                          label=int(label[b])))
                ptr += feat.shape[0]
        return np.concatenate(feat_chunks, axis=0), pd.DataFrame(meta_rows)

    figs = []
    for cfg in TSNE_CONFIGS:
        try:
            args = build_args(tta_algo="NoAdapt", cuda=args_cuda, **cfg)
            setup_seed(args.seed)
            base_model, preprocess = get_pretrained_model(args)
            datasets_obj = get_dataset_class(args.dataset)(root=args.data_root, transform=preprocess)
            feats, meta = extract_features(PretrainedFeaturizer(base_model), datasets_obj, args)
            print(f"{args.dataset}: t-SNE on {feats.shape[0]} points ({feats.shape[1]}-d)...")
            xy = TSNE(n_components=2, init="pca", random_state=args.seed).fit_transform(feats)

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            cmap_c = plt.get_cmap("tab20", len(datasets_obj.classes))
            axes[0].scatter(xy[:, 0], xy[:, 1], c=meta["label"].values, cmap=cmap_c, s=8, alpha=0.7)
            axes[0].set_title(f"{args.dataset}: t-SNE colored by class")
            cmap_d = plt.get_cmap("tab20", len(datasets_obj.environments))
            for d, env in enumerate(datasets_obj.environments):
                m = meta["domain_idx"].values == d
                axes[1].scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.6, color=cmap_d(d), label=env)
            axes[1].set_title(f"{args.dataset}: t-SNE colored by corruption domain")
            axes[1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
            for ax in axes:
                ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout()
            p = os.path.join(out_dir, "figures", f"tsne_{args.dataset}.png")
            fig.savefig(p, dpi=120, bbox_inches="tight")
            plt.close(fig)
            figs.append((args.dataset, os.path.relpath(p, out_dir)))

            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"t-SNE FAILED for {cfg}: {e}")
    return figs


# --------------------------------------------------------------------------------------
# LaTeX report
# --------------------------------------------------------------------------------------
def tex_escape(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def df_to_tex_table(df, caption, float_fmt="%.2f"):
    body = df.to_latex(index=False, float_format=lambda x: float_fmt % x,
                       escape=True, na_rep="--")
    return f"\\begin{{table}}[H]\n\\centering\n\\small\n{body}\n\\caption{{{caption}}}\n\\end{{table}}\n"


def write_report(out_dir, all_df, diag_results, tsne_figs, domain_fig, meta_info):
    tex = []
    tex.append(r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{longtable}
\usepackage{hyperref}
\title{Ramen TTA -- Experiment Report}
\author{Auto-generated by experiments.py}
\date{%s}
\begin{document}
\maketitle
""" % tex_escape(meta_info["date"]))

    tex.append(r"\section{Run configuration}" + "\n\\begin{itemize}\n")
    for k, v in meta_info.items():
        tex.append(f"  \\item \\textbf{{{tex_escape(k)}}}: {tex_escape(v)}\n")
    tex.append("\\end{itemize}\n")

    if all_df.empty:
        tex.append("No results available (sweep skipped and no parseable logs under ../log/).\n")
    else:
        # Flag-effect table: avg accuracy per (dataset, tta_algo, use_tbn)
        tex.append(r"\section{Average accuracy per algorithm / flag}" + "\n")
        tex.append("Each cell is the mean over all corruption domains of that run's per-domain "
                   "accuracy (\\%). Compare adaptation on/off (NoAdapt vs Tent vs Ramen) and the "
                   "\\texttt{use\\_tbn} toggle.\n\n")
        pivot = (all_df.groupby(["dataset", "tta_algo", "use_tbn"])["accuracy"].mean()
                 .reset_index())
        pivot["use_tbn"] = pivot["use_tbn"].map({True: "on", False: "off"})
        wide = pivot.pivot_table(index=["dataset", "tta_algo"], columns="use_tbn",
                                 values="accuracy").reset_index()
        tex.append(df_to_tex_table(wide, "Mean per-domain accuracy (\\%) by dataset, algorithm and use\\_tbn."))

        # Best config per dataset
        best = (all_df.groupby(["dataset", "tta_algo", "use_tbn"])["avg_acc"].first()
                .reset_index().sort_values(["dataset", "avg_acc"], ascending=[True, False]))
        tex.append(df_to_tex_table(best, "All runs ranked by average accuracy within each dataset."))

        # Per-domain accuracy table + weakest domains, per dataset
        tex.append(r"\section{Per-domain accuracy}" + "\n")
        for dataset in sorted(all_df["dataset"].unique()):
            sub = all_df[all_df["dataset"] == dataset]
            wide = sub.pivot_table(index="domain", columns="tta_algo", values="accuracy")
            wide = wide.reset_index().sort_values(wide.columns[-1] if len(wide.columns) > 1 else "domain")
            tex.append(f"\\subsection*{{{tex_escape(dataset)}}}\n")
            tex.append(df_to_tex_table(wide, f"{tex_escape(dataset)}: per-domain accuracy (\\%) by algorithm (use\\_tbn=off), sorted ascending."))
            # weakest 5 by the last algo column
            algo_cols = [c for c in wide.columns if c != "domain"]
            if algo_cols:
                worst = wide.nsmallest(5, algo_cols[-1])[["domain"] + algo_cols]
                tex.append(df_to_tex_table(worst, f"{tex_escape(dataset)}: 5 weakest corruption domains (by {tex_escape(algo_cols[-1])})."))

        if domain_fig:
            tex.append(r"\section{Per-domain accuracy (figure)}" + "\n")
            tex.append("Bars sorted ascending; dashed line = that run's own average; "
                       "red bars are domains below the average (dragging the total down).\n\n")
            tex.append(f"\\begin{{figure}}[H]\\centering\\includegraphics[width=\\textwidth]{{{domain_fig}}}"
                       r"\caption{Per-domain accuracy, per (dataset, algorithm).}\end{figure}" + "\n")

    # Diagnostics
    if diag_results:
        tex.append(r"\section{Confusion matrices \& top confused class pairs}" + "\n")
        for d in diag_results:
            tex.append(f"\\subsection*{{{tex_escape(d['dataset'])} / {tex_escape(d['tta_algo'])}}}\n")
            tex.append(f"Overall accuracy: {d['overall_acc']:.2f}\\%. "
                       f"Weakest domain: \\texttt{{{tex_escape(d['worst_domain'])}}} "
                       f"({d['worst_domain_acc']:.2f}\\%).\n\n")
            tex.append(f"\\begin{{figure}}[H]\\centering\\includegraphics[width=0.7\\textwidth]{{{d['cm_fig']}}}"
                       f"\\caption{{{tex_escape(d['dataset'])} / {tex_escape(d['tta_algo'])} row-normalized confusion matrix.}}\\end{{figure}}\n")
            tex.append(df_to_tex_table(d["top_conf"], f"{tex_escape(d['dataset'])} / {tex_escape(d['tta_algo'])}: biggest off-diagonal (true, predicted) class pairs.", float_fmt="%.0f"))

    # t-SNE
    if tsne_figs:
        tex.append(r"\section{t-SNE of frozen CLIP visual embeddings}" + "\n")
        tex.append("Left: colored by class (are classes still separable?). "
                   "Right: colored by corruption domain (does the corruption signature dominate "
                   "the embedding more than object identity?).\n\n")
        for name, fig in tsne_figs:
            tex.append(f"\\begin{{figure}}[H]\\centering\\includegraphics[width=\\textwidth]{{{fig}}}"
                       f"\\caption{{{tex_escape(name)}: t-SNE colored by class (left) and corruption domain (right).}}\\end{{figure}}\n")

    tex.append(r"\end{document}" + "\n")

    path = os.path.join(out_dir, "experiments_report.tex")
    with open(path, "w") as f:
        f.write("\n".join(tex))
    print(f"\nLaTeX report written to {path}")
    print(f"Build it with:  cd {out_dir} && pdflatex experiments_report.tex")
    return path


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="../report")
    ap.add_argument("--no-sweep", action="store_true", help="skip the grid sweep, only parse ../log/")
    ap.add_argument("--no-diagnostics", action="store_true")
    ap.add_argument("--no-tsne", action="store_true")
    ap.add_argument("--no-save-csv", action="store_true", help="do not append sweep runs to ../log/")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    cuda = not args.cpu
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    torch.set_num_threads(NUM_THREADS)
    setup_seed(args.seed)
    print(f"cuda available: {torch.cuda.is_available()}  using cuda: {cuda and torch.cuda.is_available()}")

    experiment_configs = [
        dict(dataset=dataset, model=model, batch_size=batch_size,
             tta_algo=tta_algo, use_tbn=use_tbn, tta_mode=tta_mode)
        for (dataset, model, batch_size) in DATASET_MODEL_BATCH
        for tta_algo in TTA_ALGOS
        for use_tbn in (USE_TBN_OPTIONS if tta_algo != "NoAdapt" else [False])
        for tta_mode in TTA_MODES
    ]
    # thread the cuda flag through
    for c in experiment_configs:
        c["cuda"] = cuda

    # 1. sweep
    sweep_records = []
    if not args.no_sweep:
        print(f"\n{len(experiment_configs)} configs queued")
        sweep_records = run_sweep(experiment_configs, save_csv=not args.no_save_csv)
    sweep_df = records_to_long_df(sweep_records)

    # 2. parse existing logs
    log_df = records_to_long_df(load_all_logs())

    all_df = pd.concat([sweep_df, log_df], ignore_index=True) if not sweep_df.empty else log_df.copy()
    if not all_df.empty:
        all_df["flag"] = all_df["tta_algo"] + all_df["use_tbn"].map(lambda b: "+tbn" if b else "")
        all_df.to_csv(os.path.join(out_dir, "all_results_long.csv"), index=False)

    # 3. per-domain figure
    domain_fig = None
    if not all_df.empty:
        p = plot_domain_accuracy_grid(all_df, os.path.join(out_dir, "figures", "domain_accuracy.png"))
        domain_fig = os.path.relpath(p, out_dir) if p else None

    # 4. diagnostics
    diag_results = []
    if not args.no_diagnostics:
        for cfg in DIAGNOSTICS_CONFIGS:
            cfg = dict(cfg, cuda=cuda)
            try:
                datasets_obj, pred_df = run_for_confusion(cfg)
                cm = compute_confusion_matrix(pred_df, datasets_obj.num_classes)
                cm_path = os.path.join(out_dir, "figures", f"cm_{cfg['dataset']}_{cfg['tta_algo']}.png")
                plot_confusion_matrix(cm, datasets_obj.classes,
                                      f"{cfg['dataset']} / {cfg['tta_algo']} (row-normalized)", cm_path)
                worst_domain = pred_df.groupby("domain")["correct"].mean().idxmin()
                diag_results.append(dict(
                    dataset=cfg["dataset"], tta_algo=cfg["tta_algo"],
                    overall_acc=pred_df["correct"].mean() * 100,
                    worst_domain=worst_domain,
                    worst_domain_acc=pred_df[pred_df["domain"] == worst_domain]["correct"].mean() * 100,
                    cm_fig=os.path.relpath(cm_path, out_dir),
                    top_conf=top_confusions(cm, datasets_obj.classes, k=10),
                ))
                pred_df.to_csv(os.path.join(out_dir, f"predictions_{cfg['dataset']}_{cfg['tta_algo']}.csv"), index=False)
            except Exception as e:
                print(f"diagnostics FAILED for {cfg}: {e}")

    # 5. t-SNE
    tsne_figs = []
    if not args.no_tsne:
        tsne_figs = run_tsne_section(out_dir, cuda)

    # 6. report
    meta_info = dict(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sweep_run=str(not args.no_sweep),
        diagnostics_run=str(not args.no_diagnostics),
        tsne_run=str(not args.no_tsne),
        device="cuda" if (cuda and torch.cuda.is_available()) else "cpu",
        num_result_rows=str(len(all_df)),
        datasets=", ".join(sorted(all_df["dataset"].unique())) if not all_df.empty else "none",
    )
    write_report(out_dir, all_df, diag_results, tsne_figs, domain_fig, meta_info)


if __name__ == "__main__":
    main()
