"""
Figures of the shift decomposition, one function per figure (repo palette from ramen_analysis).

    shapley_vs_ramen(['ImageNetC5K', 'DomainNet', 'ImageNetVariants', 'EuroSAT'])
    variance_shares([...])
    per_domain_shapley('DomainNet')
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramen_analysis import BLUE, CATEGORICAL, GRID, MUTED, ORANGE, SURFACE, TEXT, TEXT_2  # noqa: E402

from analysis import Summary  # noqa: E402
from data import LABEL  # noqa: E402
from experiment import ShiftDecomposition  # noqa: E402

COMP = {'A': ('appearance A', CATEGORICAL[0]), 'G': ('concept Γ', CATEGORICAL[1]), 'L': ('label shift L', CATEGORICAL[2])}


def _style(ax):
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=TEXT_2)


def shapley_vs_ramen(datasets, title=None):
    """Dot plot: oracle Shapley values (95% CI) per component and world, with Ramen's gain as a reference line."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6), sharey=True)
    rows = [('image', 'A'), ('image', 'G'), ('image', 'L'), ('text', 'A'), ('text', 'L')]
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        s = Summary(ShiftDecomposition(ds).load()).shapley().set_index(['world', 'component'])
        ramen = s.loc[('cached', 'Ramen - NoAdapt'), 'phi']
        for i, (w, k) in enumerate(rows):
            r = s.loc[('image (img)' if w == 'image' else 'text', k)]
            col = COMP[k][1]
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=2, alpha=0.6)
            ax.scatter(r.phi, i, s=70, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f'{r.phi:+.2f}', (r.phi, i), xytext=(0, 7), textcoords='offset points', ha='center',
                        fontsize=8, color=TEXT_2)
        ax.axvline(ramen, color=BLUE, lw=1.5, ls='--')
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.text(ramen, len(rows) - 0.4, f' Ramen {ramen:+.2f}', color=TEXT, fontsize=8, va='bottom')
        ax.set_title(LABEL[ds], color=TEXT, fontsize=10)
        ax.set_xlabel('accuracy points (domain-avg)', color=TEXT_2, fontsize=8)
        _style(ax)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(rows)))
    ax0.set_yticklabels([f'{w}: {COMP[k][0]}' for w, k in rows], color=TEXT)
    ax0.set_ylim(len(rows) - 0.2, -0.6)
    fig.suptitle(title or 'Oracle ceilings per shift component (Shapley, 95% CI) vs Ramen\'s gain', color=TEXT)
    fig.tight_layout()
    return fig


def variance_shares(datasets, title=None):
    """Stacked horizontal bars: debiased share of embedding variance per ANOVA term."""
    terms = [('domain offset alpha (appearance)', 'domain offset α', CATEGORICAL[0]),
             ('class x domain gamma (concept)', 'class×domain γ', CATEGORICAL[1]),
             ('class effect beta', 'class β', CATEGORICAL[6]), ('within cell', 'within cell', '#c3c2b7')]
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.55 * len(datasets)))
    for i, ds in enumerate(datasets):
        tab = {r['term']: r for r in ShiftDecomposition(ds).load().meta['anova']}
        left = 0
        for key, lab, col in terms:
            w = tab[key]['share_pct']
            ax.barh(i, w, left=left, color=col, edgecolor=SURFACE, linewidth=2, label=lab if i == 0 else None)
            if w > 4:
                ax.text(left + w / 2, i, f'{w:.0f}%', ha='center', va='center', fontsize=8,
                        color=TEXT if col == '#c3c2b7' else SURFACE)
            left += w
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([LABEL[d] for d in datasets], color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('share of embedding variance (%, debiased)', color=TEXT_2)
    ax.legend(ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02), loc='lower center', fontsize=8)
    _style(ax)
    ax.set_title(title or 'Where the embedding variance sits', color=TEXT, pad=28)
    fig.tight_layout()
    return fig


def per_domain_shapley(ds, title=None):
    """Dot plot per domain: image-world Shapley values of A, G, L and Ramen's gain."""
    t = Summary(ShiftDecomposition(ds).load()).per_domain()
    fig, ax = plt.subplots(figsize=(7.5, 0.7 + 0.45 * len(t)))
    for j, (k, col) in enumerate([('phi_A', COMP['A'][1]), ('phi_G', COMP['G'][1]), ('phi_L', COMP['L'][1])]):
        ax.scatter(t[k], np.arange(len(t)) + (j - 1) * 0.18, s=55, color=col, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label=COMP[k[-1]][0])
    ax.scatter(t.ramen_gain, np.arange(len(t)), marker='|', s=200, color=BLUE, lw=2, zorder=4, label='Ramen gain')
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f'{d} ({n:,})' for d, n in zip(t.domain, t.images)], color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel('accuracy points', color=TEXT_2)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    _style(ax)
    ax.set_title(title or f'{LABEL[ds]}: oracle Shapley values per domain (image world)', color=TEXT)
    fig.tight_layout()
    return fig
