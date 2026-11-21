"""
Figures for the loss-landscape section (matplotlib, static). Take Landscapes objects from landscape_analysis.py.

Colour = direction, as in ga_plots.py: Ramen's step blue, oracle step aqua, random step grey (control).
Slices: true-label CE in greys (darker = higher loss), the region where the true class is predicted tinted aqua with
its boundary in violet; theta_0 / z0 black circle, Ramen's theta_i / z1 blue triangle, oracle step aqua square.

    fig = plot_line_profiles({'EuroSAT': L1, ...})    # CE change along the three 1D lines, per group
    fig = plot_slices(L, 'DomainNet')                 # one typical image per group: parameter and embedding slices
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter

from ga_experiment import OUTCOMES
from ga_plots import AQUA, BLUE, MUTED, SURFACE, TEXT, TEXT_2, VIOLET, _style
from landscape import GRID_A, GRID_B, LINE_T

DIR_COLOR = {'ramen': BLUE, 'oracle': AQUA, 'random': MUTED}
DIR_NAME = {'ramen': "Ramen's step", 'oracle': 'oracle step', 'random': 'random step (control)'}
CE_CMAP = LinearSegmentedColormap.from_list('ce', [SURFACE, '#3d3c39'])


def plot_line_profiles(Ls):
    """Rows = datasets, columns = outcome groups: median (IQR band) of CE(t) - CE(0) along each direction."""
    fig, axes = plt.subplots(len(Ls), len(OUTCOMES), figsize=(13, 2.7 * len(Ls)), squeeze=False)
    for i, (ds, L) in enumerate(Ls.items()):
        prof = L.line_profiles()
        for j, g in enumerate(OUTCOMES):
            ax = axes[i, j]
            if g not in prof:
                ax.set_axis_off()
                continue
            for d, (med, lo, hi, n) in prof[g].items():
                ax.fill_between(LINE_T, lo, hi, color=DIR_COLOR[d], alpha=0.15, lw=0)
                ax.plot(LINE_T, med, color=DIR_COLOR[d], lw=2, label=DIR_NAME[d])
            ax.axvline(1, color=MUTED, lw=1, ls=':')
            ax.axhline(0, color=MUTED, lw=1)
            ax.set_title(f'{ds}: {g} ({n} images)', fontsize=8.5, color=TEXT)
            if i == len(Ls) - 1:
                ax.set_xlabel('step multiple t (1 = the step Ramen takes)', fontsize=7.5, color=TEXT_2)
            if j == 0:
                ax.set_ylabel('true-label CE change', fontsize=7.5, color=TEXT_2)
            _style(ax)
    axes[0, 0].legend(fontsize=7, loc='lower left')
    fig.suptitle("True-label loss along Ramen's step, the oracle step and a random step of the same length "
                 '(median, IQR band; dotted line = Ramen\'s actual step)', fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig


def _slice(ax, ce, pred, y, a, b, vmax):
    A, B = np.meshgrid(a, b)
    cs = ax.contourf(A, B, np.clip(ce, 0, vmax), levels=np.linspace(0, vmax, 15), cmap=CE_CMAP)
    true = (pred == y).astype(float)
    if 0 < true.mean() < 1:
        ax.contourf(A, B, true, levels=[0.5, 1.5], colors=[AQUA], alpha=0.22)
        ax.contour(A, B, true, levels=[0.5], colors=[VIOLET], linewidths=1.6)
    elif true.mean() == 1:
        ax.contourf(A, B, true, levels=[0.5, 1.5], colors=[AQUA], alpha=0.22)
    return cs


def _markers(ax, oracle_xy=None):
    ax.annotate('', xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.8))
    ax.scatter([0], [0], s=55, color=TEXT, edgecolor=SURFACE, lw=1.5, zorder=4)
    ax.scatter([1], [0], s=70, marker='^', color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=4)
    if oracle_xy is not None:
        ax.scatter([oracle_xy[0]], [oracle_xy[1]], s=55, marker='s', color=AQUA, edgecolor=SURFACE, lw=1.5, zorder=4)


def plot_slices(L, ds, names=None):
    """Row 1: parameter-space slice; row 2: embedding-space slice; one column per outcome group (typical image)."""
    ex = L.examples()
    fig, axes = plt.subplots(2, len(OUTCOMES), figsize=(15, 7.8), squeeze=False, layout='constrained')
    for j, g in enumerate(OUTCOMES):
        if g not in ex:
            axes[0, j].set_axis_off()
            axes[1, j].set_axis_off()
            continue
        r = ex[g]
        emb = L.embedding_slice(r)
        ce0, ce1 = r['line_ramen_ce'][np.isclose(LINE_T, 0)][0], r['line_ramen_ce'][np.isclose(LINE_T, 1)][0]
        vmax = max(2.5 * max(ce0, ce1), 1.0)       # local scale: higher losses saturate (darkest)
        cs = _slice(axes[0, j], r['grid_ce'], r['grid_pred'], r['label'], GRID_A, GRID_B, vmax)
        _markers(axes[0, j], r['oracle_xy'])
        _slice(axes[1, j], emb['ce'], emb['pred'], r['label'], GRID_A, GRID_B, vmax)
        _markers(axes[1, j])
        cb = fig.colorbar(cs, ax=axes[:, j], location='bottom', fraction=0.035, pad=0.02)
        cb.set_label('true-label CE (clipped at 2.5x the larger of the two points)', fontsize=7, color=TEXT_2)
        cb.set_ticks(np.linspace(0, vmax, 5))
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        cb.ax.tick_params(labelsize=6.5, colors=TEXT_2)
        nm = (lambda c: names[c]) if names else str
        axes[0, j].set_title(f'{g}\ntrue: {nm(r["label"])}\nzero-shot: {nm(r["zs_pred"])} -> Ramen: {nm(r["ramen_pred"])}\n'
                             f'CE {ce0:.2f} -> {ce1:.2f}; cos(sign step, g_ce) {r["cos_step"]:+.3f}',
                             fontsize=7.5, color=TEXT)
        axes[0, j].set_xlabel("along Ramen's step (units of ||step||)", fontsize=7, color=TEXT_2)
        axes[1, j].set_xlabel('along z1 - z0 (units of ||z1 - z0||)', fontsize=7, color=TEXT_2)
        axes[1, j].set_title(f'embedding space, ||z1 - z0|| = {emb["step_norm"]:.3f}', fontsize=7.5, color=TEXT)
        for ax in axes[:, j]:
            ax.tick_params(colors=TEXT_2, labelsize=7)
            for s in ax.spines.values():
                s.set_color(MUTED)
    axes[0, 0].set_ylabel('along the oracle step, orthogonalised', fontsize=7, color=TEXT_2)
    axes[1, 0].set_ylabel('along the true-class direction of the head,\northogonalised', fontsize=7, color=TEXT_2)
    handles = [plt.Line2D([], [], marker='o', ls='', color=TEXT, label='start: theta_0 / z0 (zero-shot)'),
               plt.Line2D([], [], marker='^', ls='', color=BLUE, label="after Ramen's step: theta_i / z1"),
               plt.Line2D([], [], marker='s', ls='', color=AQUA, label='oracle SignSGD step on the true-label CE'),
               plt.Line2D([], [], color=VIOLET, lw=1.6, label='boundary of the region predicting the true class '
                                                                '(tinted)')]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, fontsize=7.5, frameon=False)
    fig.suptitle(f'{ds}: true-label loss around the start and the adapted point, one typical image per group '
                 '(darker = higher loss)', fontsize=11, color=TEXT)
    return fig
