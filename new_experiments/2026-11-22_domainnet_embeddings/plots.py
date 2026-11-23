"""
Figures for the DomainNet embedding study, one function per figure (repo palette from ramen_analysis).

    fig = scatter_by_domain(proj, d)                 # t-SNE coloured by domain
    fig = domain_panels(proj, d)                     # one panel per domain: its images, right vs wrong
    fig = class_panels(proj, d, classes)             # one panel per domain: selected classes
    fig = heatmap(matrix, title, fmt='{:.2f}')       # any domain x domain table
    fig = neighbour_bars(table)                      # stacked bars: same/other class x same/other domain
    fig = margin_distributions(frame, d)             # image-text margin per domain
    fig = image_grid(d, 'cat', per_domain=6)         # thumbnails of one class across domains
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramen_analysis import CATEGORICAL, GRID, MUTED, RED, SEQ, SURFACE, TEXT, TEXT_2  # noqa: E402

RIGHT_BLUE, BACKGROUND = '#86b6ef', '#e9e8e4'
# Fixed colour per domain (identity), used in every figure
DOMAIN_COLOR = dict(zip(['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'], CATEGORICAL))


def _clean(ax, axes_off=False):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    if axes_off:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


def _counts(n, of):
    return f'{n:,} of {of:,} ({100 * n / of:.1f}%)'


# ---------------------------------------------------------------------------- projections
def scatter_by_domain(proj, d, title=None):
    """All projected points coloured by domain; order shuffled so no domain is drawn on top of the others."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    p = proj.sample(frac=1, random_state=0)
    for name in d.domains:  # legend entries in fixed domain order
        ax.scatter([], [], s=30, color=DOMAIN_COLOR[name], label=f'{name} ({(proj.domain == name).sum():,})')
    ax.scatter(p.x, p.y, s=4, c=p.domain.map(DOMAIN_COLOR), linewidths=0, alpha=0.8, rasterized=True)
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), labelcolor=TEXT)
    ax.set_title(title or 'Zero-shot CLIP embedding coloured by domain', color=TEXT, fontsize=11, loc='left')
    _clean(ax, axes_off=True)
    fig.tight_layout()
    return fig


def domain_panels(proj, d, title=None):
    """Small multiples on the shared projection: one domain per panel, its images coloured right/wrong (zero-shot),
    every other image as a faint background."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), d.domains):
        me = proj[proj.domain == name]
        ax.scatter(proj.x, proj.y, s=2, color=BACKGROUND, linewidths=0, rasterized=True)
        ax.scatter(me.x[me.correct], me.y[me.correct], s=4, color=RIGHT_BLUE, linewidths=0, rasterized=True)
        ax.scatter(me.x[~me.correct], me.y[~me.correct], s=4, color=RED, linewidths=0, rasterized=True)
        ax.set_title(f'{name}: wrong {_counts(int((~me.correct).sum()), len(me))}', color=TEXT, fontsize=9, loc='left')
        _clean(ax, axes_off=True)
    axes[0, 0].scatter([], [], s=20, color=RIGHT_BLUE, label='zero-shot right')
    axes[0, 0].scatter([], [], s=20, color=RED, label='zero-shot wrong')
    axes[0, 0].scatter([], [], s=20, color=BACKGROUND, label='other domains')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper right', ncol=3, frameon=False, fontsize=8)
    fig.suptitle(title or 'Where each domain sits in the embedding, and where its zero-shot errors are',
                 color=TEXT, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def class_panels(proj, d, classes, title=None):
    """Small multiples, one panel per domain: images of the selected classes coloured by TRUE class
    (hollow marker = zero-shot wrong). If a class's colour lands in the same region in every panel, CLIP places
    that class consistently across domains."""
    colors = dict(zip(classes, CATEGORICAL))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), d.domains):
        ax.scatter(proj.x, proj.y, s=2, color=BACKGROUND, linewidths=0, rasterized=True)
        me = proj[proj.domain == name]
        for c in classes:
            m = me[me.true_name == c]
            ax.scatter(m.x[m.correct], m.y[m.correct], s=10, color=colors[c], linewidths=0, rasterized=True)
            ax.scatter(m.x[~m.correct], m.y[~m.correct], s=10, facecolors='none', edgecolors=colors[c],
                       linewidths=0.8, rasterized=True)
        ax.set_title(f'{name}: zero-shot right {_counts(int(me.correct.sum()), len(me))}', color=TEXT, fontsize=9,
                     loc='left')
        _clean(ax, axes_off=True)
    handles = [plt.Line2D([], [], ls='', marker='o', color=colors[c], label=c) for c in classes]
    handles.append(plt.Line2D([], [], ls='', marker='o', mfc='none', mec=TEXT_2, label='hollow = zero-shot wrong'))
    fig.legend(handles=handles, loc='upper right', ncol=min(len(handles), 5), frameon=False, fontsize=8)
    fig.suptitle(title or 'Do the same classes land in the same place in every domain?', color=TEXT, x=0.01,
                 ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ---------------------------------------------------------------------------- tables as figures
def heatmap(M, title, fmt='{:.2f}', cmap=SEQ, vmin=None, vmax=None, label=''):
    """Domain x domain (or any small) matrix with the value written in each cell."""
    fig, ax = plt.subplots(figsize=(1.0 * M.shape[1] + 2.2, 0.75 * M.shape[0] + 1.4))
    im = ax.imshow(M.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.grid(False)
    lo, hi = im.get_clim()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8,
                    color=SURFACE if (v - lo) / (hi - lo + 1e-12) > 0.6 else TEXT)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, rotation=30, ha='right', color=TEXT, fontsize=8)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(M.index, color=TEXT, fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label=label)
    ax.set_title(title, color=TEXT, fontsize=10, loc='left')
    fig.tight_layout()
    return fig


NEIGHBOUR_GROUPS = [('same class, same domain', '#2a78d6'), ('same class, other domain', '#86b6ef'),
                    ('other class, same domain', '#eb6834'), ('other class, other domain', '#f5b89c')]


def neighbour_bars(table, k=10, title=None):
    """Stacked horizontal bars: what the k nearest neighbours of a query image are, per query domain."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    y = np.arange(len(table))
    left = np.zeros(len(table))
    for col, color in NEIGHBOUR_GROUPS:
        v = table[col].values
        ax.barh(y, v, left=left, color=color, edgecolor=SURFACE, linewidth=2, label=col, height=0.7)
        for yi, (l, w) in enumerate(zip(left, v)):
            if w >= 6:
                ax.text(l + w / 2, yi, f'{w:.0f}%', ha='center', va='center', fontsize=8, color=TEXT)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f'{n} ({q:,} queries)' for n, q in zip(table.index, table.queries)], color=TEXT, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(f'% of the {k} nearest neighbours (cosine, all 596,006 images)', color=TEXT_2, fontsize=8)
    ax.legend(ncol=4, frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.0))
    ax.set_title(title or 'Neighbours share the domain more often than the class', color=TEXT, fontsize=11,
                 loc='left', pad=28)
    _clean(ax)
    fig.tight_layout()
    return fig


def margin_distributions(frame, d, title=None, bins=80):
    """Per domain: histogram of the image-text margin cos(z, w_y) - max_{c != y} cos(z, w_c); left of 0 = wrong."""
    fig, axes = plt.subplots(len(d.domains), 1, figsize=(8, 1.4 * len(d.domains) + 0.8), sharex=True)
    edges = np.linspace(np.percentile(frame.margin, 0.5), np.percentile(frame.margin, 99.5), bins)
    for ax, name in zip(axes, d.domains):
        m = frame[frame.domain == name]
        ax.hist(m.margin, bins=edges, color=DOMAIN_COLOR[name], alpha=0.85)
        ax.axvline(0, color=TEXT_2, lw=1)
        wrong = int((m.margin < 0).sum())
        ax.set_title(f'{name}: margin < 0 for {_counts(wrong, len(m))}', loc='left', fontsize=8, color=TEXT, pad=2)
        ax.set_yticks([])
        _clean(ax)
        ax.spines['left'].set_visible(False)
    axes[-1].set_xlabel('cos(z, true-class text) − cos(z, best wrong text)', color=TEXT_2, fontsize=8)
    fig.suptitle(title or 'Image-text margin per domain: quickdraw sits almost entirely left of 0', color=TEXT,
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def class_domain_heatmap(acc, counts, rows, title=None):
    """Zero-shot accuracy per (class, domain) for the given classes."""
    M = acc.loc[rows]
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(rows) + 1.6))
    im = ax.imshow(M.values, cmap=SEQ, vmin=0, vmax=100, aspect='auto')
    ax.grid(False)
    for i, c in enumerate(rows):
        for j, dom in enumerate(M.columns):
            v = M.values[i, j]
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=7, color=SURFACE if v > 60 else TEXT)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, color=TEXT, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{c} (n={counts.loc[c].sum():,})' for c in rows], color=TEXT, fontsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='zero-shot accuracy %')
    ax.set_title(title or 'Zero-shot accuracy per class and domain', color=TEXT, fontsize=10, loc='left')
    fig.tight_layout()
    return fig


def domain_bars(values, title, xlabel, fmt='{:.1f}'):
    """One horizontal bar per domain (magnitude), domain colours, value labels."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    names = list(values.index)
    ax.barh(range(len(names)), values.values, color=[DOMAIN_COLOR[n] for n in names], height=0.65)
    for i, v in enumerate(values.values):
        ax.text(v, i, ' ' + fmt.format(v), va='center', fontsize=8, color=TEXT)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=TEXT, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=8)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=10, loc='left')
    _clean(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------- images
def image_grid(d, cls, per_domain=6, seed=0, size=2.0):
    """Random images of one TRUE class, one row per domain, captioned with the zero-shot prediction."""
    c = d.classes.index(cls)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(d.n_dom, per_domain, figsize=(size * per_domain, size * 1.15 * d.n_dom))
    for k, name in enumerate(d.domains):
        pool = np.flatnonzero((d.y == c) & (d.dom == k))
        rows = rng.choice(pool, min(per_domain, len(pool)), replace=False)
        hit = int(d.correct[pool].sum())
        for j, ax in enumerate(axes[k]):
            _clean(ax, axes_off=True)
            if j >= len(rows):
                continue
            i = rows[j]
            ax.imshow(d.image(i))
            ok = d.correct[i]
            ax.set_title(('✓ ' if ok else '✗ ') + d.classes[d.pred[i]], fontsize=8,
                         color=TEXT_2 if ok else RED)
        axes[k, 0].set_ylabel(f'{name}\n{_counts(hit, len(pool))} right', color=TEXT, fontsize=8)
        axes[k, 0].yaxis.set_visible(True)
        axes[k, 0].set_yticks([])
    fig.suptitle(f'True class "{cls}" in each domain (caption = zero-shot prediction)', color=TEXT, x=0.01,
                 ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
