"""
Aggregate figures of the sink case study, one finding per figure; per-image figures are in example_card.py.

    plot_prediction_counts(trace)                     # how often each model predicts the target, right vs wrong
    plot_routes(trace)                                # Ramen's wrong targets by zero-shot class, oracle fate stacked
    plot_update_sources(summary, groups, target)      # mean update share per source, Ramen cache vs oracle cache
    plot_queue_history(history, target, n_stream)     # the target queue over the stream, both caches
"""
import matplotlib.pyplot as plt
import numpy as np

from sink_trace import ORACLE_FATES  # also sets up the import paths
from ramen_analysis import AQUA, BLUE, MUTED, RED, TEXT_2  # noqa: E402

CACHE_COLORS = {'Ramen cache': BLUE, 'oracle cache': '#4a3aa7'}
FATE_COLORS = {'still target': RED, 'correct': AQUA, 'other wrong': MUTED}
GAP = dict(edgecolor='white', linewidth=2)  # 2px surface gap between adjacent fills


def plot_prediction_counts(trace):
    c = trace.counts()
    name = trace.target_name
    fig, ax = plt.subplots(figsize=(9, 2.6))
    y = np.arange(len(c))[::-1]
    ax.barh(y, c['correct'], color=AQUA, height=0.6, label='correct', **GAP)
    ax.barh(y, c['wrong'], left=c['correct'], color=RED, height=0.6, label='wrong', **GAP)
    for yi, (_, r) in zip(y, c.iterrows()):
        total = r[f'predicts {name}']
        ax.text(total, yi, f'  {total:,} ({r["wrong"]:,} wrong)', va='center', fontsize=8.5, color=TEXT_2)
    ax.set_yticks(y, c.index)
    ax.set_xlim(0, c[f'predicts {name}'].max() * 1.3)
    ax.set_xlabel(f'images predicted "{name}" (truly {name}: {c.attrs["true images"]:,})')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right', ncol=2)
    z, r, o = c[f'predicts {name}']
    ax.set_title(f'Ramen predicts {name} {r / z:.1f}× as often as zero-shot; with the oracle cache it is still '
                 f'{o / z:.1f}×')
    return fig


def plot_routes(trace):
    t = trace.routes().drop(index='all')
    name = trace.target_name
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(t) + 1.4))
    y = np.arange(len(t))[::-1]
    left = np.zeros(len(t))
    for fate in ORACLE_FATES:
        label = {'still target': f'oracle still says {name}', 'correct': 'oracle correct',
                 'other wrong': 'oracle: another wrong class'}[fate]
        ax.barh(y, t[fate], left=left, color=FATE_COLORS[fate], height=0.62, label=label, **GAP)
        left += t[fate].to_numpy()
    for yi, n, pct, kept in zip(y, t['images'], t['% of wrong'], t['still target']):
        ax.text(n, yi, f'  {n:,} ({pct:.0f}%) · oracle keeps {kept:,} ({kept / n:.0%})',
                va='center', fontsize=8.5, color=TEXT_2)
    ax.set_yticks(y, t.index)
    ax.set_xlim(0, t['images'].max() * 1.55)
    ax.set_xlabel(f'images Ramen wrongly predicts as {name}, by their zero-shot class')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right', fontsize=8)
    allr = trace.routes().loc['all']
    kept, n, fixed = int(allr['still target']), int(allr['images']), int(allr['correct'])
    ax.set_title(f'The oracle cache keeps {kept:,} of {n:,} ({kept / n:.0%}) at {name} and fixes {fixed}; '
                 f'the rest move to another wrong class')
    return fig


def plot_update_sources(summary, groups, target, metrics=None, title=None):
    """Small multiples, one per group of queries: mean share of the update weight, Ramen cache vs oracle cache."""
    metrics = metrics or ['self', 'from target queue', 'from zero-shot-class queue', 'from true-class queue',
                          'entries zero-shot-predicted target', 'entries truly target', 'entries truly query\'s class',
                          'entries that are zero-shot errors', 'entries from query\'s domain',
                          'weight on entries both caches retrieve']
    caches = list(summary.index.get_level_values('cache').unique())
    ncol = min(2, len(groups))
    nrow = -(-len(groups) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, (0.42 * len(metrics) + 1.6) * nrow), sharey=True,
                             squeeze=False)
    axes = axes.ravel()
    y = np.arange(len(metrics))[::-1]
    for ax, (gname, rows) in zip(axes, groups.items()):
        means = {c: summary.xs(c, level='cache').loc[rows, metrics].mean() * 100 for c in caches}
        ax.hlines(y, np.minimum(*means.values()), np.maximum(*means.values()), color=MUTED, lw=2)
        for c in caches:
            ax.plot(means[c].to_numpy(), y, 'o', ms=8, color=CACHE_COLORS.get(c, MUTED), label=c,
                    markeredgecolor='white', markeredgewidth=1.5)
        for c in caches:  # label each dot on its outer side
            other = means[caches[1] if c == caches[0] else caches[0]]
            for yi, v, w in zip(y, means[c], other):
                left = v < w or (v == w and c == caches[0])
                ax.text(v + (-2 if left else 2), yi, f'{v:.0f}', ha='right' if left else 'left', va='center',
                        fontsize=7.5, color=TEXT_2)
        ax.set_title(f'{gname}\n({len(rows):,} images)', fontsize=10)
        ax.set_xlim(-3, 103)
        ax.set_xlabel('mean % of the update weight')
        ax.grid(axis='y', visible=False)
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    axes[0].set_yticks(y, [m.replace('target', target) for m in metrics], fontsize=8.5)
    axes[0].legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.suptitle(title or 'Where each image\'s update comes from, Ramen cache vs oracle cache', fontsize=12,
                 fontweight='bold', x=0.01, ha='left', y=1.03)
    return fig


def plot_queue_history(history, target, n_stream, freeze=65_520, domain_name=None, domain_col=None):
    """The target queue over the stream, both caches: how much of it is truly the class, and from one domain."""
    h = history
    panels = [('truly this class %', f'% of the "{target}" queue truly {target}'),
              ('zero-shot-predicted this class %', f'% of the "{target}" queue zero-shot calls {target}')]
    if domain_col:
        panels.append((domain_col, f'% of the "{target}" queue from {domain_name}'))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 3.6))
    for ax, (col, title) in zip(np.atleast_1d(axes), panels):
        for cache, g in h.groupby('cache'):
            color = CACHE_COLORS.get(cache, MUTED)
            ax.plot(g.position / n_stream * 100, g[col], lw=2, color=color, label=cache)
            ax.text(100.5, g[col].iloc[-1], f' {g[col].iloc[-1]:.0f}%', va='center', fontsize=8, color=TEXT_2)
            # the queue's last change: after the fp16 counter saturates, new entries (priority inf) replace the
            # finite-priority ones once, then the full queue freezes
            last = g['newest entry'].max() / n_stream * 100
            ax.plot([last], [g[col].iloc[-1]], marker='|', ms=14, mew=2, color=color)
        ax.axvline(freeze / n_stream * 100, color=MUTED, lw=1, ls='--')
        ax.text(freeze / n_stream * 100, 4, f' fp16 counter saturates ({freeze:,})\n | = queue\'s last change',
                fontsize=7.5, color=TEXT_2, va='bottom')
        ax.set_ylim(0, 105)
        ax.set_xlim(0, 108)
        ax.set_xlabel('% of the test stream seen')
        ax.set_title(title, fontsize=10)
    np.atleast_1d(axes)[0].legend(loc='lower right')
    return fig
