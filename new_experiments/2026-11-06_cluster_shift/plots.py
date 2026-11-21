"""
Figures for the cluster-shift experiment (one function per figure).

    fig = plot_gain_vs_floor(table, ramen_gain, noise_floor, title)   # table from ClusterShiftExperiment.summary
"""
import matplotlib.pyplot as plt


def plot_gain_vs_floor(table, ramen_gain, noise_floor, title):
    """Dot plot: accuracy gain over NoAdapt per variant, with Ramen's gain and the +/- noise floor marked."""
    t = table[~table['variant'].str.contains('cached')].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(t) + 1.2))
    colors = ['#c0392b' if v.startswith('oracle') else '#7f8c8d' if v.startswith('random') else '#2c7fb8'
              for v in t['variant']]
    ax.scatter(t['vs_noadapt'], range(len(t)), c=colors, zorder=3)
    ax.set_yticks(range(len(t)), t['variant'], fontsize=8)
    ax.axvline(0, color='k', lw=0.8)
    ax.axvline(ramen_gain, color='#27ae60', lw=1.5, label=f'Ramen (+{ramen_gain:.2f})')
    ax.axvspan(-noise_floor, noise_floor, color='#bbb', alpha=0.3, label=f'noise floor (±{noise_floor})')
    ax.set_xlabel('accuracy gain over NoAdapt (domain-averaged, points)')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc='lower left')
    return fig
