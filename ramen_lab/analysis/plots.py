"""Static matplotlib charts with one shared look. Pure: data in, Figure out (no file I/O)."""
from typing import Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"Ramen": "#2a6fdb", "NoAdapt": "#e8833a", "variant": "#7b4fc4", "control": "#8c8c8c", "worse": "#d1495b",
           "text": "#333333", "grid": "#e6e6e6"}


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": PALETTE["grid"],
        "axes.edgecolor": "#999999", "axes.labelcolor": PALETTE["text"], "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"], "legend.frameon": False,
    })


def color(name: str) -> str:
    return PALETTE.get(name, PALETTE["variant"])


def seed_accuracies(accs: Mapping[str, Sequence[float]], seeds: Sequence[int], title: str,
                    floor: Optional[float] = None) -> plt.Figure:
    """Dot per seed for each method, mean as a bar tick; the noise floor drawn as a band around the first method's mean."""
    fig, ax = plt.subplots(figsize=(6.5, 1.0 + 0.7 * len(accs)))
    for row, (name, values) in enumerate(accs.items()):
        values = np.asarray(values, dtype=float)
        ax.scatter(values, np.full(len(values), row), s=45, color=color(name), zorder=3, edgecolor="white")
        ax.plot([values.mean()] * 2, [row - 0.25, row + 0.25], color=color(name), lw=2)
        for s, v in zip(seeds, values):
            ax.annotate(f"s{s}", (v, row), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
        ax.annotate(f"mean {values.mean():.2f}", (values.max(), row), xytext=(12, -3), textcoords="offset points",
                    fontsize=8, color=color(name))
        if floor is not None and row == 0:
            ax.axvspan(values.mean() - floor / 2, values.mean() + floor / 2, color=color(name), alpha=0.08,
                       label=f"noise floor {floor:.2f} (2 x range)")
    ax.set_yticks(range(len(accs)), list(accs))
    ax.set_ylim(-0.6, len(accs) - 0.4)
    ax.set_xlabel("accuracy (%, mean over domains)")
    ax.set_title(title)
    if floor is not None:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def domain_dumbbell(before: Mapping[str, float], after: Mapping[str, float], labels: Sequence[str],
                    title: str) -> plt.Figure:
    """Per domain: `before` -> `after` accuracy, sorted by the change."""
    domains = sorted(before, key=lambda d: after[d] - before[d])
    fig, ax = plt.subplots(figsize=(6.5, 0.8 + 0.32 * len(domains)))
    for y, d in enumerate(domains):
        ax.plot([before[d], after[d]], [y, y], color="#bbbbbb", lw=2, zorder=1)
        ax.scatter(before[d], y, color=color(labels[0]), s=35, zorder=2)
        ax.scatter(after[d], y, color=color(labels[1]), s=35, zorder=2)
        ax.annotate(f"{after[d] - before[d]:+.1f}", (max(before[d], after[d]), y), xytext=(6, -3),
                    textcoords="offset points", fontsize=7)
    ax.set_yticks(range(len(domains)), domains)
    ax.scatter([], [], color=color(labels[0]), label=labels[0])
    ax.scatter([], [], color=color(labels[1]), label=labels[1])
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("accuracy (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def outcome_bars(outcomes: Mapping[str, Dict[str, int]], title: str) -> plt.Figure:
    """Stacked horizontal bars: fixed / broken / both right / both wrong per comparison, labelled with % and count."""
    parts = [("both right", "#d9d8d4"), ("fixed", PALETTE["Ramen"]), ("broken", PALETTE["worse"]),
             ("both wrong", PALETTE["control"])]
    fig, ax = plt.subplots(figsize=(7, 0.9 + 0.5 * len(outcomes)))
    for y, (name, counts) in enumerate(outcomes.items()):
        total, left = sum(counts.values()), 0.0
        for part, c in parts:
            share = 100 * counts[part] / total
            ax.barh(y, share, left=left, color=c, label=part if y == 0 else None)
            if share > 4:
                ax.text(left + share / 2, y, f"{share:.1f}%\n{counts[part]:,}", ha="center", va="center", fontsize=7)
            left += share
    ax.set_yticks(range(len(outcomes)), list(outcomes))
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of test images")
    ax.set_title(title)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.35), fontsize=8)
    fig.tight_layout()
    return fig
