"""
Figures for the gradient-alignment experiment (one function per figure, matplotlib, static).

Colour = which gradient is compared with the true-label CE gradient, the same in every figure:
Ramen's update (blue), the image's own entropy gradient (orange), the shuffled control (grey).

    fig = plot_cosine_distributions(df)
    fig = plot_outcome_dumbbell(df)
    fig = plot_domain_dumbbell(df)
    fig = plot_stream(df, window=5000)
    fig = plot_decomposition(df)
    fig = plot_outcome_vs_alignment(df)

Palette copied from new_experiments/ramen_analysis.py (already validated; not imported, to avoid loading torch).
"""
import matplotlib.pyplot as plt
import numpy as np

from ga_analysis import DIRECTIONS, PARTS, binned_outcome_rates, decomposition, stream_rolling, zs_group

TEXT, TEXT_2, MUTED, GRID, SURFACE, NEUTRAL = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb', '#f0efec'
BLUE, ORANGE, AQUA, YELLOW, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7'
COLOR = {'cos_update': BLUE, 'cos_self': ORANGE, 'cos_update_shuffled': MUTED}
# the own entry IS the image's own entropy gradient, so it shares that colour
PART_COLOR = {'part_self': ORANGE, 'part_true': AQUA, 'part_zs': YELLOW, 'part_other': VIOLET}
GROUPS = ('zero-shot right', 'zero-shot wrong')


def _style(ax, grid='y'):
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=TEXT_2, labelsize=8)


def _zero_line(ax, vertical=True):
    (ax.axvline if vertical else ax.axhline)(0, color=MUTED, lw=1, zorder=1)


def _pct(n, total):
    return f'{n:,} of {total:,} ({n / total:.1%})'


def plot_cosine_distributions(df):
    """Histograms of the three cosines with g_ce, one panel per zero-shot group."""
    groups = zs_group(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    bins = np.linspace(-1, 1, 81)
    for ax, g in zip(axes, GROUPS):
        d = df[groups == g]
        for col, name in DIRECTIONS.items():
            ax.hist(d[col], bins=bins, histtype='step', lw=2, color=COLOR[col],
                    label=f'{name} (median {d[col].median():+.2f})')
        _zero_line(ax)
        pos = int((d['cos_update'] > 0).sum())
        ax.set_title(f'{g}: {len(d):,} images\nRamen update cos > 0 on {_pct(pos, len(d))}', fontsize=10)
        ax.set_xlabel('cosine with the true-label CE gradient g_ce', fontsize=8, color=TEXT_2)
        ax.set_ylabel('images', fontsize=8, color=TEXT_2)
        ax.legend(fontsize=7, loc='upper left')
        _style(ax)
    w = df[groups == 'zero-shot wrong']
    fig.suptitle(f"Where zero-shot is wrong, Ramen's update (median {w['cos_update'].median():+.2f}) points "
                 f"{'closer to' if w['cos_update'].median() > w['cos_self'].median() else 'further from'} the "
                 f"true class\nthan the image's own entropy gradient (median {w['cos_self'].median():+.2f})",
                 fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig


def _dumbbell(ax, labels, own, ramen, control, counts):
    y = np.arange(len(labels))[::-1]
    for yi, a, b in zip(y, own, ramen):
        ax.plot([a, b], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle='round')
    for vals, col, name in ((control, 'cos_update_shuffled', DIRECTIONS['cos_update_shuffled']),
                            (own, 'cos_self', DIRECTIONS['cos_self']), (ramen, 'cos_update', DIRECTIONS['cos_update'])):
        ax.scatter(vals, y, s=70 if col != 'cos_update_shuffled' else 40, color=COLOR[col], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=name, marker='o' if col != 'cos_update_shuffled' else 'D')
    ax.set_yticks(y, [f'{l}\n{c}' for l, c in zip(labels, counts)], fontsize=8)
    _zero_line(ax)
    _style(ax, grid='x')


def plot_outcome_dumbbell(df):
    """Median cosine with g_ce per outcome group: own entropy gradient -> Ramen's update (control as a diamond)."""
    groups = list(df['outcome'].cat.categories)
    g = df.groupby('outcome', observed=False)
    med = g[list(DIRECTIONS)].median().reindex(groups)
    counts = [_pct(int(n), len(df)) for n in g.size().reindex(groups)]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    _dumbbell(ax, groups, med['cos_self'], med['cos_update'], med['cos_update_shuffled'], counts)
    for yi, v in zip(np.arange(len(groups))[::-1], med['cos_update']):
        ax.annotate(f'{v:+.2f}', (v, yi), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=7,
                    color=TEXT_2)
    ax.set_xlabel('median cosine with the true-label CE gradient g_ce', fontsize=8, color=TEXT_2)
    ax.legend(fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    fixed, both_wrong = med.loc['fixed by Ramen', 'cos_update'], med.loc['both wrong', 'cos_update']
    ax.set_title(f"Median alignment of Ramen's update with the true class: {fixed:+.2f} on images it fixes,\n"
                 f"{both_wrong:+.2f} on images it leaves wrong", fontsize=10, pad=28)
    fig.tight_layout()
    return fig


def plot_domain_dumbbell(df):
    """Median cosine with g_ce per domain, own entropy -> Ramen's update, one panel per zero-shot group."""
    groups = zs_group(df)
    domains = list(df['domain'].cat.categories)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(domains) + 1.8), sharex=True)
    for ax, g in zip(axes, GROUPS):
        d = df[groups == g]
        m = d.groupby('domain', observed=False)[list(DIRECTIONS)].median().reindex(domains)
        n = d.groupby('domain', observed=False).size().reindex(domains)
        tot = df.groupby('domain', observed=False).size().reindex(domains)
        _dumbbell(ax, domains, m['cos_self'], m['cos_update'], m['cos_update_shuffled'],
                  [_pct(int(a), int(b)) for a, b in zip(n, tot)])
        ax.set_title(g, fontsize=10)
        ax.set_xlabel('median cosine with g_ce', fontsize=8, color=TEXT_2)
    axes[0].legend(fontsize=7, loc='lower center', bbox_to_anchor=(1.1, 1.06), ncol=3)
    w = df[groups == 'zero-shot wrong'].groupby('domain', observed=False)['cos_update'].median()
    fig.suptitle(f"Per domain: alignment on zero-shot-wrong images is lowest on {w.idxmin()} ({w.min():+.2f}) "
                 f"and highest on {w.idxmax()} ({w.max():+.2f})", fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig


def plot_stream(df, window=5000, freeze_at=None):
    """Rolling mean of each cosine over the stream, one panel per zero-shot group; optional memory-freeze marker."""
    roll = stream_rolling(df, window)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharex=True)
    for ax, g in zip(axes, GROUPS):
        r = roll[g]
        for col, name in DIRECTIONS.items():
            ax.plot(r['position'], r[col], color=COLOR[col], lw=2, label=name)
        _zero_line(ax, vertical=False)
        if freeze_at is not None:
            ax.axvline(freeze_at, color=MUTED, lw=1, ls='--')
            ax.text(freeze_at, ax.get_ylim()[1], ' memory mostly frozen\n (fp16 priorities)', fontsize=7,
                    color=TEXT_2, va='top')
        ax.set_title(g, fontsize=10)
        ax.set_xlabel('stream position (images seen)', fontsize=8, color=TEXT_2)
        ax.set_ylabel(f'mean cosine with g_ce\n(rolling, {window:,} images)', fontsize=8, color=TEXT_2)
        _style(ax)
    axes[0].legend(fontsize=7, loc='lower right')
    w = roll['zero-shot wrong']['cos_update'].dropna()
    fig.suptitle(f"Ramen's alignment on zero-shot-wrong images over the stream: {w.iloc[0]:+.2f} at the start, "
                 f"{w.iloc[-1]:+.2f} at the end", fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig


def plot_decomposition(df):
    """Mean additive share of cos(Ramen update, g_ce) contributed by each kind of queue, per outcome group."""
    t = decomposition(df)
    groups = list(t.index)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    h = 0.19
    y = np.arange(len(groups))[::-1]
    for k, (col, name) in enumerate(PARTS.items()):
        ax.barh(y + (1.5 - k) * h, t[col], height=h - 0.03, color=PART_COLOR[col], label=name, zorder=2)
    ax.scatter(t['cos_update'], y, marker='|', s=400, color=TEXT, zorder=3, label='total = mean cos(update, g_ce)')
    ax.set_yticks(y, [f'{g}\n{int(n):,} images' for g, n in zip(groups, t['images'])], fontsize=8)
    _zero_line(ax)
    ax.set_xlabel('mean contribution to cos(Ramen update, g_ce)', fontsize=8, color=TEXT_2)
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3)
    _style(ax, grid='x')
    fx = t.loc['fixed by Ramen']
    ax.set_title(f"Where the alignment comes from. Fixed images: {fx['cos_update']:+.2f} in total = own entry "
                 f"{fx['part_self']:+.2f},\ntrue-class queue {fx['part_true']:+.2f}, zero-shot queue "
                 f"{fx['part_zs']:+.2f}, other queues {fx['part_other']:+.2f}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_outcome_vs_alignment(df, col='cos_step', n_bins=10):
    """Fix rate (zero-shot wrong) and break rate (zero-shot right) vs alignment decile; median change of log p(y)."""
    r = binned_outcome_rates(df, col, n_bins)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    titles = {'zero-shot wrong': 'share of zero-shot errors Ramen fixes',
              'zero-shot right': 'share of zero-shot hits Ramen breaks'}
    for ax, g in zip(axes[:2], ('zero-shot wrong', 'zero-shot right')):
        s = r[r['group'] == g]
        ax.plot(s['x'], 100 * s['rate'], color=BLUE, lw=2, marker='o', ms=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5)
        for _, row in s.iloc[[0, -1]].iterrows():
            ax.annotate(f"{row['events']:,} of {row['images']:,}", (row['x'], 100 * row['rate']), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=7, color=TEXT_2)
        ax.set_title(f'{g}:\n{titles[g]}', fontsize=9)
        ax.set_ylabel('%', fontsize=8, color=TEXT_2)
        ax.margins(y=0.15)
    ax = axes[2]
    for g, ls, m in (('zero-shot right', '-', 'o'), ('zero-shot wrong', ':', 's')):
        s = r[r['group'] == g]
        ax.plot(s['x'], s['dlogp_median'], color=BLUE, lw=2, ls=ls, marker=m, ms=6, markeredgecolor=SURFACE,
                label=g)
    _zero_line(ax, vertical=False)
    ax.legend(fontsize=7)
    ax.set_title("median change of log p(true class)\nafter Ramen's step", fontsize=9)
    for ax in axes:
        ax.set_xlabel(f'cos(sign of Ramen update, g_ce), decile median', fontsize=8, color=TEXT_2)
        _style(ax)
    w = r[r['group'] == 'zero-shot wrong']
    fig.suptitle(f"Zero-shot errors fixed by Ramen: {w['rate'].iloc[0]:.1%} in the lowest alignment decile vs "
                 f"{w['rate'].iloc[-1]:.1%} in the highest",
                 fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig
