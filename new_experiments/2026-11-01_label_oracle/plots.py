"""
Figures of the label-value oracle experiment, one finding per figure.

    plot_vs_ramen(exp, names, floor)          # domain-averaged change vs Ramen (or `baseline`) per run, with the floor
    plot_domain_change(exp, names)            # change vs Ramen per domain, one dot per run
    plot_class_predictions(exp, cls, names)   # how often each run predicts `cls`, right vs wrong
"""
import matplotlib.pyplot as plt
import numpy as np

from label_oracle import LABELS  # also sets up the import paths
from ramen_analysis import AQUA, BLUE, MUTED, ORANGE, RED, TEXT_2  # noqa: E402
from true_label_cache import domain_avg  # noqa: E402

# fixed per run (categorical order of the project palette), never by rank
RUN_COLORS = {'NoAdapt': MUTED, 'Ramen': BLUE, 'oracle_file_true': ORANGE, 'labeloracle_true_entropy': ORANGE,
              'labeloracle_zs_ce': AQUA,
              'labeloracle_true_ce': '#eda100', 'late_zs_entropy': '#e87ba4', 'late_zs_ce': '#008300',
              'late_adapted_entropy': '#4a3aa7', 'late_adapted_ce': '#e34948',
              # global retrieval: one hue, darker = smaller k; no labels in brown
              'global_ce_k10': '#2e1f78', 'global_ce_k100': '#6a5ad0', 'global_ce_k1000': '#a99ee8',
              'global_entropy_k100': '#9b7b4a'}
# a run with the other copies of each photo hidden (ImageNet-C) is the same entity as its unhidden version
RUN_COLORS.update({'excl_zs_entropy': BLUE, 'excl_zs_ce': AQUA,
                   **{f'excl_{n}': RUN_COLORS[n] for n in ('global_ce_k10', 'global_ce_k100', 'global_ce_k1000',
                                                           'global_entropy_k100')}})
GAP = dict(edgecolor='white', linewidth=2)


def _ring(color):
    return dict(ms=9, color=color, markeredgecolor='white', markeredgewidth=1.5)


def plot_vs_ramen(exp, names, floor, title=None, baseline='Ramen'):
    base = domain_avg(exp.results[baseline])
    rows = [(n, domain_avg(exp.results[n]) - base,
             int(exp.results[n]['correct'].sum() - exp.results[baseline]['correct'].sum())) for n in names]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-floor, floor, color=MUTED, alpha=0.15, lw=0)
    ax.axvline(0, color=MUTED, lw=1)
    for yi, (n, d, imgs) in zip(y, rows):
        ax.hlines(yi, 0, d, color=RUN_COLORS[n], lw=2)
        ax.plot([d], [yi], 'o', **_ring(RUN_COLORS[n]))
        ax.annotate(f'{d:+.2f} pts ({imgs:+,} images)', (d, yi), xytext=(8 if d >= 0 else -8, 7),
                    textcoords='offset points', ha='left' if d >= 0 else 'right', fontsize=8.5, color=TEXT_2)
    ds = [d for _, d, _ in rows] + [-floor, floor]
    pad = 0.25 * (max(ds) - min(ds))
    ax.set_xlim(min(ds) - 1.6 * pad, max(ds) + pad)  # room for labels left of negative dots
    ax.set_yticks(y, [LABELS.get(n, n) for n, _, _ in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(f'change in domain-averaged accuracy vs {LABELS.get(baseline, baseline)} (points); '
                  f'grey band = noise floor ±{floor:.2f}')
    ax.grid(axis='y', visible=False)
    ax.set_title(title or f'Change vs {LABELS.get(baseline, baseline)}')
    return fig


def plot_domain_change(exp, names, title=None, by='domain'):
    acc = exp.accuracy_table(['Ramen', *names], by=by)
    delta = acc[names].sub(acc['Ramen'], axis=0)
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(delta) + 1.6))
    y = np.arange(len(delta))[::-1]
    ax.axvline(0, color=MUTED, lw=1)
    offsets = np.linspace(0.22, -0.22, len(names))
    for n, off in zip(names, offsets):
        ax.plot(delta[n].to_numpy(), y + off, 'o', label=LABELS.get(n, n), **_ring(RUN_COLORS[n]))
        for yi, d in zip(y + off, delta[n]):
            ax.text(d + (0.12 if d >= 0 else -0.12), yi, f'{d:+.2f}', va='center', ha='left' if d >= 0 else 'right',
                    fontsize=7.5, color=TEXT_2)
    for yi in y[1:] + 0.5:
        ax.axhline(yi, color=MUTED, lw=0.4, alpha=0.5)
    lo, hi = min(0, delta.min().min()), max(0, delta.max().max())
    ax.set_xlim(lo - 0.15 * (hi - lo) - 0.3, hi + 0.15 * (hi - lo) + 0.3)
    ax.set_yticks(y, delta.index)
    ax.set_xlabel('accuracy change vs Ramen (points)')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(names), borderaxespad=0.3)
    ax.set_title(title or 'Change vs Ramen per domain', pad=28)
    return fig


def plot_class_predictions(exp, cls, names, title=None):
    i = exp.classes.index(cls)
    rows = []
    for n in names:
        df = exp.results[n]
        pred = df['pred'] == i
        rows.append((n, int((pred & df['correct']).sum()), int((pred & ~df['correct']).sum())))
    truly = int((exp.results[names[0]]['label'] == i).sum())
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    right = np.array([r[1] for r in rows])
    wrong = np.array([r[2] for r in rows])
    ax.barh(y, right, color=AQUA, height=0.6, label='correct', **GAP)
    ax.barh(y, wrong, left=right, color=RED, height=0.6, label='wrong', **GAP)
    for yi, r, w in zip(y, right, wrong):
        ax.text(r + w, yi, f'  {r + w:,} ({w:,} wrong)', va='center', fontsize=8.5, color=TEXT_2)
    ax.set_yticks(y, [LABELS.get(r[0], r[0]) for r in rows])
    ax.set_xlim(0, (right + wrong).max() * 1.35)
    ax.set_xlabel(f'images predicted "{cls}" (truly {cls}: {truly:,})')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right', ncol=2)
    ax.set_title(title or f'Predictions of {cls}')
    return fig


FILING_COLORS = {'zero-shot filing': RUN_COLORS['labeloracle_zs_ce'], 'true-label filing': RUN_COLORS['labeloracle_true_ce']}


def plot_agreement(table, metric='own class is the largest %', title=None):
    """Dumbbell per domain: a retrieval metric under zero-shot vs true-label filing (LabelAgreement.by_domain)."""
    t = table[metric].unstack('filing')
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(t) + 1.6))
    y = np.arange(len(t))[::-1]
    ax.hlines(y, t.min(1), t.max(1), color=MUTED, lw=2)
    for f, color in FILING_COLORS.items():
        ax.plot(t[f].to_numpy(), y, 'o', label=f, **_ring(color))
    for yi, a, b in zip(y, t['zero-shot filing'], t['true-label filing']):
        for v, other in ((a, b), (b, a)):
            ax.text(v + (1.2 if v >= other else -1.2), yi, f'{v:.0f}', va='center', ha='left' if v >= other else 'right',
                    fontsize=8, color=TEXT_2)
    ax.set_yticks(y, t.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel(metric.replace(' %', ' (% of images)'))
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.3)
    ax.set_title(title or metric, pad=28)
    return fig
