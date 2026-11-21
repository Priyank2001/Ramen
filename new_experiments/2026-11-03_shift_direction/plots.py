"""Plots over ShiftDecomposition. Each function draws one finding and returns its axes."""
import matplotlib.pyplot as plt
import numpy as np

from analysis import OUTCOMES

OUTCOME_COLORS = {'fixed': '#1baf7a', 'broken': '#e34948',
                  'wrong -> same wrong': '#8a8984', 'wrong -> other wrong': '#eda100'}


def plot_explained_fraction(table, ax=None):
    """Distribution of explained_frac (how much of the shift sits on the image's own from->to axis), by outcome."""
    present = [o for o in OUTCOMES if (table['outcome'] == o).any()]
    data = [table.loc[table['outcome'] == o, 'explained_frac'].to_numpy() for o in present]
    ax = ax or plt.subplots(figsize=(8, 3))[1]
    bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=0.6)
    for patch, o in zip(bp['boxes'], present):
        patch.set_facecolor(OUTCOME_COLORS[o])
    ax.set_yticks(np.arange(1, len(present) + 1), [f'{o} ({len(d):,})' for o, d in zip(present, data)])
    ax.set_xlim(0, 1)
    ax.set_xlabel('explained_frac = (shift on the from→to axis)² / |shift|²')
    ax.set_title('How much of each move is targeted at its own (from → to) class pair', loc='left')
    ax.grid(axis='x', alpha=0.3)
    return ax


def plot_delta_margin(table, ax=None):
    """Distribution of delta_margin = 100 * shift projected on the from->to axis = exact change in (logit_to - logit_from)."""
    present = [o for o in OUTCOMES if (table['outcome'] == o).any()]
    data = [table.loc[table['outcome'] == o, 'delta_margin'].to_numpy() for o in present]
    ax = ax or plt.subplots(figsize=(8, 3))[1]
    bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=0.6)
    for patch, o in zip(bp['boxes'], present):
        patch.set_facecolor(OUTCOME_COLORS[o])
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(np.arange(1, len(present) + 1), [f'{o} ({len(d):,})' for o, d in zip(present, data)])
    ax.set_xlabel('Δ margin = (logit_to − logit_from) after − before  (exact, CLIP logits are linear in z)')
    ax.set_title('Does the step move the margin toward the class it ends up predicting?', loc='left')
    ax.grid(axis='x', alpha=0.3)
    return ax


def plot_coherence(coherence_table, null_by_n, ax=None):
    """
    Dot plot: destination-sharing coherence R per class, against its own bootstrap null (random images of the
    same group size that do not share a destination). null_by_n: {class -> (lo, mid, hi)} at that class's n.
    """
    t = coherence_table
    ax = ax or plt.subplots(figsize=(7, 0.4 * len(t) + 1.2))[1]
    y = np.arange(len(t))[::-1]
    for yi, (_, row) in zip(y, t.iterrows()):
        lo, mid, hi = null_by_n[row['class']]
        ax.plot([lo, hi], [yi, yi], color='#c9c8c3', lw=4, solid_capstyle='butt', zorder=1)
        ax.plot(mid, yi, '|', color='#5b5a56', ms=14, zorder=2)
    ax.scatter(t['R'], y, s=36, color='#2a78d6', zorder=3, label='R (shared destination)')
    for yi, r, n in zip(y, t['R'], t['images']):
        ax.text(r + 0.015, yi, f'n={n:,}', va='center', fontsize=7)
    ax.set_yticks(y, t['class'])
    ax.set_xlabel('R = mean resultant length of unit shift directions (bar = random-group null, 2.5–97.5 pct)')
    ax.set_title('Do images that end up at the same wrong class move in a shared direction?', loc='left')
    ax.set_xlim(0, max(0.05, t['R'].max() * 1.15))
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    return ax
