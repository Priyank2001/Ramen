"""
Figures for the cache-compression experiment (one function per figure, matplotlib, static).

    fig = plot_accuracy_vs_memory(tables, floors)   # tables: {dataset: CompressionExperiment.table(...)}
    fig = plot_peak_memory(tables)

Palette copied from new_experiments/ramen_analysis.py (already validated; not imported, to avoid loading torch).
"""
import matplotlib.pyplot as plt

TEXT, TEXT_2, MUTED, GRID, SURFACE, NEUTRAL = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb', '#f0efec'
BLUE, ORANGE = '#2a78d6', '#eb6834'
SERIES = {'compressed values, full queue': BLUE, 'fp16 values, shorter queue': ORANGE}
LEVELS = ('int8', 'int4', 'sign')


def _style(ax, grid='y'):
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=TEXT_2, labelsize=8)


def plot_accuracy_vs_memory(tables, floors):
    """Accuracy minus Ramen vs cache memory (fraction of Ramen's), codec vs matched shorter queue, one panel per set."""
    fig, axes = plt.subplots(1, len(tables), figsize=(4.4 * len(tables), 3.4))
    for ax, (ds, t) in zip(axes, tables.items()):
        ax.axhspan(-floors[ds], floors[ds], color=NEUTRAL, zorder=0)
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        for (label, color), prefix in zip(SERIES.items(), ('cc_', 'cc_cap_')):
            rows = t.loc[['Ramen', *[prefix + c for c in LEVELS]]]
            ax.plot(rows['mem_ratio'], rows['vs_ramen'], color=color, lw=2, marker='o', ms=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        for c in LEVELS:      # label each pair on its outside: the higher point above, the lower one below
            a, b = t.loc[f'cc_{c}'], t.loc[f'cc_cap_{c}']
            for r, other in ((a, b), (b, a)):
                up = r['vs_ramen'] >= other['vs_ramen']
                ax.annotate(f"{r['vs_ramen']:+.2f}", (r['mem_ratio'], r['vs_ramen']), xytext=(5, 5 if up else -6),
                            textcoords='offset points', fontsize=7, color=TEXT_2, va='bottom' if up else 'top')
        ax.margins(y=0.12)
        ax.set_xscale('log')
        ax.set_xticks([0.105, 0.29, 0.537, 1.0], ['10.5%\nsign', '29%\nint4', '54%\nint8', '100%\nfp16'])
        ax.minorticks_off()
        ax.set_xlim(0.08, 1.3)
        ax.text(1.25, floors[ds], f'noise floor ±{floors[ds]}', fontsize=7, color=TEXT_2, ha='right', va='bottom')
        ax.set_title(ds, fontsize=10, color=TEXT)
        ax.set_xlabel('cache memory, share of Ramen (log scale)', fontsize=8, color=TEXT_2)
        ax.set_ylabel('accuracy − Ramen (points, domain avg)', fontsize=8, color=TEXT_2)
        _style(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('At equal memory, compressed values keep Ramen\'s accuracy; a shorter queue loses it '
                 '(except EuroSAT)', fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def plot_peak_memory(tables):
    """Horizontal bars: cache size and peak GPU memory per run, one panel per dataset."""
    runs = ['cc_off', 'cc_int8', 'cc_int4', 'cc_sign']
    fig, axes = plt.subplots(1, len(tables), figsize=(4.4 * len(tables), 2.6), sharey=True)
    for ax, (ds, t) in zip(axes, tables.items()):
        r = t.loc[runs[::-1]]
        y = range(len(r))
        ax.barh(y, r['peak_GB'], color=GRID, height=0.7, label='peak GPU memory')
        ax.barh(y, r['cache_MB'] / 1024, color=BLUE, height=0.7, label='cache (keys + values)')
        for i, (_, row) in enumerate(r.iterrows()):
            ax.text(row['peak_GB'] + 0.2, i, f"{row['cache_MB'] / 1024:.2f} / {row['peak_GB']:.1f} GB",
                    fontsize=7, color=TEXT_2, va='center')
        ax.set_yticks(list(y), ['sign (1 bit)', 'int4', 'int8', 'fp16 (Ramen)'])
        ax.set_xlim(0, 17)
        ax.set_title(ds, fontsize=10, color=TEXT)
        ax.set_xlabel('GB', fontsize=8, color=TEXT_2)
        _style(ax, grid='x')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle('The 1-bit cache is 9.5x smaller and cuts peak GPU memory by 5–7 GB', fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig
