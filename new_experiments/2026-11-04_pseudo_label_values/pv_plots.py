"""Plots for the pseudo-label value screen. Each function draws one finding and returns its axes."""
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'label-free': '#2a78d6', 'oracle': '#eb6834', 'control': '#8a8984'}


def plot_gain_vs_floor(table, floor, title, ax=None):
    """Dot plot: change vs Ramen (domain-averaged points) per run, with the noise floor band around 0."""
    t = table
    ax = ax or plt.subplots(figsize=(7, 0.45 * len(t) + 1.2))[1]
    y = np.arange(len(t))[::-1]
    ax.axvspan(-floor, floor, color='#e8e7e3', zorder=0, label=f'noise floor ±{floor}')
    ax.axvline(0, color='k', lw=0.8)
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.plot(r['vs Ramen (pts)'], yi, 'o', ms=8, color=COLORS[r['kind']], zorder=3)
        ax.text(r['vs Ramen (pts)'], yi + 0.28, f"{r['vs Ramen (pts)']:+.2f} ({r['vs Ramen (images)']:+,} images)",
                ha='center', fontsize=7)
    ax.set_yticks(y, t['label'])
    ax.set_xlabel('change vs Ramen (domain-averaged points)')
    ax.set_title(title, loc='left')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    return ax


def plot_dose_response(points, line_runs, title, ax=None):
    """Change vs Ramen against target accuracy (% of entries whose target is the true class), one point per run."""
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    oracle = points.loc[[r for r in line_runs if r in points.index]].sort_values("target right %")
    ax.plot(oracle['target right %'], oracle['vs Ramen (pts)'], '-', color=COLORS['oracle'], lw=1, zorder=1)
    for _, r in points.iterrows():
        ax.plot(r['target right %'], r['vs Ramen (pts)'], 'o', ms=8, color=COLORS[r['kind']], zorder=3)
        ax.annotate(r['label'], (r['target right %'], r['vs Ramen (pts)']), textcoords='offset points',
                    xytext=(6, -3), fontsize=7)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('target right (% of stream images)')
    ax.set_ylabel('change vs Ramen (pts)')
    ax.set_title(title, loc='left')
    ax.grid(alpha=0.3)
    return ax
