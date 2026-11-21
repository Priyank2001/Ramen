"""
Figures for the GMM label-source experiment (one function per figure, matplotlib, static).

    fig = plot_gain(tables, pass_points)          # tables: {dataset: GMMLabelExperiment.table(...)}
    fig = plot_fixed_broken(tables)
    fig = plot_iterations(iters, 'acc_pooled')    # iters: {dataset: {variant: e.iters[variant]}}
    fig = plot_alignment(iters, variant='lda')

Palette copied from new_experiments/ramen_analysis.py (already validated; not imported, to avoid loading torch).
"""
import matplotlib.pyplot as plt

TEXT, TEXT_2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
BLUE, ORANGE, AQUA, RED = '#2a78d6', '#eb6834', '#1baf7a', '#e34948'
FAMILY_COLOR = {'label-free': BLUE, 'oracle': ORANGE}
VARIANT_COLOR = {'vmf': BLUE, 'vmf_prior': '#86b6ef', 'lda': AQUA, 'lda_prior': '#8fd9bd'}


def _style(ax):
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=TEXT_2, labelsize=8)


def plot_gain(tables, pass_points):
    """Dot plot: label accuracy minus zero-shot (pooled points) per variant, one panel per dataset."""
    fig, axes = plt.subplots(1, len(tables), figsize=(4.2 * len(tables), 3.2), sharey=True)
    for ax, (ds, t) in zip(axes, tables.items()):
        t = t[t['variant'] != 'zs'].iloc[::-1].reset_index(drop=True)
        for i, r in t.iterrows():
            c = FAMILY_COLOR['oracle' if r['oracle'] else 'label-free']
            ax.scatter(r['gain_pooled'], i, s=70, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f"{r['gain_pooled']:+.1f} ({r['fixed'] - r['broken']:+,} img)", (r['gain_pooled'], i),
                        xytext=(6, 4), textcoords='offset points', fontsize=7, color=TEXT_2)
        ax.set_yticks(range(len(t)), t['variant'])
        ax.axvline(0, color=TEXT_2, lw=1)
        ax.axvline(pass_points, color=MUTED, lw=1, ls='--')
        ax.set_ylim(-1.0, len(t) - 0.5)
        ax.text(pass_points, -0.8, f' pass bar +{pass_points:g}', fontsize=7, color=TEXT_2, va='center')
        ax.set_title(ds, fontsize=10, color=TEXT)
        ax.set_xlabel('label accuracy − zero-shot (points, pooled)', fontsize=8, color=TEXT_2)
        _style(ax)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, ms=8, label=k) for k, c in FAMILY_COLOR.items()]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Label-free EM beats zero-shot only on EuroSAT (10 classes, one domain)', fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_fixed_broken(tables):
    """Diverging bars: zero-shot errors fixed (right) vs zero-shot-correct images broken (left), per variant."""
    fig, axes = plt.subplots(1, len(tables), figsize=(4.2 * len(tables), 3.2), sharey=True)
    for ax, (ds, t) in zip(axes, tables.items()):
        t = t[t['variant'] != 'zs'].iloc[::-1].reset_index(drop=True)
        ax.barh(range(len(t)), t['fixed'], color=BLUE, height=0.7, label='fixed (zs wrong → right)')
        ax.barh(range(len(t)), -t['broken'], color=RED, height=0.7, label='broken (zs right → wrong)')
        for i, r in t.iterrows():
            ax.text(r['fixed'], i, f" {r['fixed']:,}", va='center', fontsize=7, color=TEXT_2)
            ax.text(-r['broken'], i, f"{r['broken']:,} ", va='center', ha='right', fontsize=7, color=TEXT_2)
        ax.set_yticks(range(len(t)), t['variant'])
        ax.axvline(0, color=TEXT_2, lw=1)
        lim = max(t['fixed'].max(), t['broken'].max()) * 1.45
        ax.set_xlim(-lim, lim)
        ax.set_title(f"{ds} (zs wrong {t['zs_wrong'].iloc[0]:,} / right {t['zs_right'].iloc[0]:,})",
                     fontsize=9, color=TEXT)
        ax.set_xlabel('images', fontsize=8, color=TEXT_2)
        _style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Label-free variants break more correct images than they fix, except LDA on EuroSAT',
                 fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_iterations(iters, col='acc_pooled', title=None):
    """Line per label-free variant: a quantity over EM iterations (0 = zero-shot), one panel per dataset."""
    fig, axes = plt.subplots(1, len(iters), figsize=(4.2 * len(iters), 3.0))
    for ax, (ds, per_variant) in zip(axes, iters.items()):
        for v, c in VARIANT_COLOR.items():
            d = per_variant[v]
            ax.plot(d['iteration'], d[col], lw=2, color=c, marker='o', ms=4, label=v)
        ax.axhline(per_variant['vmf'][col].iloc[0], color=MUTED, lw=1, ls='--')
        ax.set_title(ds, fontsize=10, color=TEXT)
        ax.set_xlabel('EM iteration (0 = zero-shot)', fontsize=8, color=TEXT_2)
        ax.grid(axis='y', color=GRID, lw=0.8)
        _style(ax)
    axes[0].set_ylabel(col, fontsize=8, color=TEXT_2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(title or col, fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def plot_alignment(iters, variant='lda', title=None):
    """NMI of the components with the true class and with the domain, over EM iterations (multi-domain datasets)."""
    multi = {ds: v for ds, v in iters.items() if v[variant]['nmi_domain'].max() > 0}
    fig, axes = plt.subplots(1, len(multi), figsize=(4.2 * len(multi), 3.0), squeeze=False)
    for ax, (ds, per_variant) in zip(axes[0], multi.items()):
        d = per_variant[variant]
        ax.plot(d['iteration'], d['nmi_class'], lw=2, color=BLUE, marker='o', ms=4, label='NMI with class')
        ax.plot(d['iteration'], d['nmi_domain'], lw=2, color=ORANGE, marker='o', ms=4, label='NMI with domain')
        ax.set_title(f'{ds} ({variant})', fontsize=10, color=TEXT)
        ax.set_xlabel('EM iteration (0 = zero-shot)', fontsize=8, color=TEXT_2)
        ax.grid(axis='y', color=GRID, lw=0.8)
        _style(ax)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(title or 'What the components track over EM', fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig
