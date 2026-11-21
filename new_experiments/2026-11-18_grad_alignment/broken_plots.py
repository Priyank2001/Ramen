"""
Figures for the broken-image analysis (zero-shot right -> Ramen wrong); one function per figure, matplotlib, static.
Each takes {dataset: BrokenImages} (or one BrokenImages) from broken_analysis.py.

Colour = image group, the same in every figure: broken (orange), kept right at matched confidence (blue),
kept right, all (grey). In the decomposition, colour = part of the update: own entry (yellow), own queue (aqua),
other queues (violet). Palette from ga_plots.py (copied from new_experiments/ramen_analysis.py, already validated).

    fig = plot_group_dots(bis)          # medians of the cosines per group, one panel per dataset
    fig = plot_step_histograms(bis)     # distribution of cos(sign step, g_ce): broken vs kept (matched)
    fig = plot_break_grid(bis)          # break rate by confidence band x alignment quintile
    fig = plot_decomposition(bis)       # where the alignment of each group comes from
    fig = plot_stream(bi, title)        # DomainNet: break rate and own entry stored over the stream
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from broken_analysis import GROUPS, PARTS, QUANTITIES, SUM_COL
from ga_plots import (AQUA, BLUE, GRID, MUTED, ORANGE, SURFACE, TEXT, TEXT_2, VIOLET, YELLOW, _style,
                      _zero_line)

GROUP_COLOR = dict(zip(GROUPS, (ORANGE, BLUE, MUTED)))
PART_COLOR = {'part_self': YELLOW, 'part_true': AQUA, 'part_other': VIOLET}
DOT_ROWS = ('cos_self', 'cos_update', 'cos_step', 'cos_update_shuffled')
SHORT = {'cos_self': 'own entropy gradient', 'cos_update': "Ramen's update", 'cos_step': "Ramen's sign step",
         'cos_update_shuffled': "control: update vs\nanother image's g_ce"}


def plot_group_dots(bis):
    """One panel per dataset: median cosine with g_ce of each direction, for the three image groups."""
    fig, axes = plt.subplots(1, len(bis), figsize=(4.2 * len(bis), 3.6), sharey=True)
    y = np.arange(len(DOT_ROWS))[::-1]
    for ax, (ds, bi) in zip(np.atleast_1d(axes), bis.items()):
        s = bi.summary()
        for g, dy in zip(GROUPS, (0.2, 0.0, -0.2)):
            vals = [s.loc[g, QUANTITIES[c]] for c in DOT_ROWS]
            n = f'{len(bi.broken):,}' if g != 'kept (all)' else f'{len(bi.kept):,}'
            ax.scatter(vals, y + dy, s=60, color=GROUP_COLOR[g], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=f'{g} ({n}{" weighted" if g == "kept (matched)" else ""})')
        _zero_line(ax)
        ax.set_yticks(y, [SHORT[c] for c in DOT_ROWS], fontsize=8)
        ax.set_xlabel('median cosine with the true-label CE gradient g_ce', fontsize=8, color=TEXT_2)
        ax.set_title(f'{ds}: {len(bi.broken):,} broken of {len(bi.r):,} zero-shot right', fontsize=9, color=TEXT)
        ax.legend(fontsize=7, loc='lower right')
        _style(ax, grid='x')
    fig.suptitle("On images Ramen breaks, its step is less aligned with the true class than on images it keeps "
                 "at the same confidence,\nand below the shared-direction control", fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig


def plot_step_histograms(bis):
    """Distribution of cos(sign step, g_ce): broken vs kept matched to the same confidence."""
    fig, axes = plt.subplots(1, len(bis), figsize=(4.2 * len(bis), 3.2))
    for ax, (ds, bi) in zip(np.atleast_1d(axes), bis.items()):
        bins, h = bi.histogram('cos_step')
        mid = (bins[:-1] + bins[1:]) / 2
        s = bi.summary()
        for g in GROUPS[:2]:
            neg = s.loc[g, 'sign step points away from g_ce (cos < 0)']
            ax.step(mid, h[g], where='mid', lw=2, color=GROUP_COLOR[g], label=f'{g}: cos < 0 on {neg}')
        _zero_line(ax)
        ax.set_title(ds, fontsize=10, color=TEXT)
        ax.set_xlabel("cos(Ramen's sign step, g_ce)", fontsize=8, color=TEXT_2)
        ax.set_ylabel('share of images', fontsize=8, color=TEXT_2)
        ax.set_ylim(0, 1.45 * max(h[g].max() for g in GROUPS[:2]))
        ax.legend(fontsize=6.5, loc='upper right')
        _style(ax)
    fig.suptitle('Broken images sit at or below zero: the step barely moves them toward their (true) class',
                 fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig


def plot_break_grid(bis):
    """Heatmap per dataset: break rate by zero-shot confidence band (rows) x alignment quintile within the band."""
    cmap = LinearSegmentedColormap.from_list('seq', [SURFACE, ORANGE])
    fig, axes = plt.subplots(1, len(bis), figsize=(4.4 * len(bis), 3.6))
    for ax, (ds, bi) in zip(np.atleast_1d(axes), bis.items()):
        g = bi.break_rate_grid()
        bands = list(dict.fromkeys(g['band']))
        R = g.pivot(index='band', columns='quintile', values='rate').loc[bands].to_numpy()
        B = g.pivot(index='band', columns='quintile', values='broken').loc[bands].to_numpy()
        N = g.pivot(index='band', columns='quintile', values='images').loc[bands].to_numpy()
        ax.imshow(R, cmap=cmap, vmin=0, vmax=0.7, aspect='auto')
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                ax.text(j, i, f'{R[i, j]:.0%}\n{int(B[i, j]):,}/{int(N[i, j]):,}', ha='center', va='center',
                        fontsize=6.5, color=TEXT if R[i, j] < 0.45 else SURFACE)
        ax.set_xticks(range(R.shape[1]), [f'Q{q}' for q in range(1, R.shape[1] + 1)], fontsize=8)
        ax.set_yticks(range(len(bands)), [b.replace('(-0.001', '[0') for b in bands], fontsize=8)
        ax.set_xlabel("quintile of cos(sign step, g_ce) within the band (Q1 = least aligned)", fontsize=7,
                      color=TEXT_2)
        ax.set_ylabel('zero-shot p(true class)', fontsize=8, color=TEXT_2)
        ax.set_title(ds, fontsize=10, color=TEXT)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle('At every confidence level, the least-aligned fifth of images is broken most often '
                 '(cell: break rate, broken / images)', fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig


def plot_decomposition(bis):
    """Stacked horizontal bars: mean parts of cos(update, g_ce) per group, one panel per dataset."""
    fig, axes = plt.subplots(len(bis), 1, figsize=(9, 1.6 * len(bis) + 0.8), sharex=True)
    for ax, (ds, bi) in zip(np.atleast_1d(axes), bis.items()):
        t = bi.decomposition()
        y = np.arange(len(GROUPS))[::-1]
        pos, neg = np.zeros(len(GROUPS)), np.zeros(len(GROUPS))
        for c, name in PARTS.items():
            v = t.loc[list(GROUPS), name].to_numpy()
            left = np.where(v >= 0, pos, neg)
            ax.barh(y, v, left=left, height=0.6, color=PART_COLOR[c], edgecolor=SURFACE, linewidth=2,
                    label=name if ds == next(iter(bis)) else None)
            pos, neg = np.where(v >= 0, pos + v, pos), np.where(v < 0, neg + v, neg)
        for yi, g in zip(y, GROUPS):
            ax.text(max(pos[list(GROUPS).index(g)], 0) + 0.005, yi, f'{t.loc[g, SUM_COL]:+.3f}',
                    va='center', fontsize=7.5, color=TEXT_2)
        _zero_line(ax)
        ax.set_yticks(y, GROUPS, fontsize=8)
        ax.set_title(ds, fontsize=9, color=TEXT, loc='left')
        _style(ax, grid='x')
    ax.set_xlabel("mean additive share of cos(Ramen's update, g_ce)", fontsize=8, color=TEXT_2)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=7.5, frameon=False)
    fig.suptitle('What broken images lack is mostly the push from all the other queues (and, on EuroSAT, '
                 'their own queue)', fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_stream(bi, ds='DomainNet'):
    """Two panels over the stream: break rate of zero-shot-right images; share whose own entry was stored."""
    t = bi.over_stream(n_bins=20)
    x = (t['start'] + t['end']) / 2 / 1000
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True)
    axes[0].plot(x, 100 * t['rate'], color=ORANGE, lw=2, marker='o', ms=4)
    axes[0].set_ylabel('broken, % of zero-shot right', fontsize=8, color=TEXT_2)
    a, b = t.iloc[0], t.iloc[-1]
    axes[0].set_title(f'{int(a["broken"]):,} of {int(a["zs_right"]):,} ({a["rate"]:.1%}) broken in the first '
                      f'twentieth -> {int(b["broken"]):,} of {int(b["zs_right"]):,} ({b["rate"]:.1%}) in the last',
                      fontsize=8.5, color=TEXT)
    axes[1].plot(x, 100 * t['stored'], color=YELLOW, lw=2, marker='o', ms=4)
    axes[1].set_ylabel('own entry stored, % of zero-shot right', fontsize=8, color=TEXT_2)
    axes[1].set_title('own entry stops being stored (fp16 priority freeze)', fontsize=8.5, color=TEXT)
    for ax in axes:
        ax.set_xlabel('stream position (thousand images)', fontsize=8, color=TEXT_2)
        ax.set_ylim(bottom=0)
        _style(ax)
    fig.suptitle(f'{ds}: Ramen breaks more zero-shot-right images as the memory stops taking their own entry',
                 fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig
