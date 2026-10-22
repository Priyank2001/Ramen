"""
Per-sample flip analysis of Ramen vs NoAdapt (zero-shot CLIP) on CIFAR-10-C, CIFAR-100-C and ImageNet-C.

Every sample falls in exactly one of four outcomes (same sample, same stream order for both methods):
    fixed       NoAdapt wrong -> Ramen right
    broken      NoAdapt right -> Ramen wrong
    kept wrong  both wrong (split into: same wrong class / a different wrong class)
    both right  both right

Reads ./results/*.pkl, writes counts (CSV), figures (PNG) and report.md into ./flip_analysis/.
Run from anywhere:  RAMEN_ENV/bin/python flip_analysis.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ramen_analysis import TEXT, TEXT_2, MUTED, GRID, SURFACE, BLUE, RED, _top_legend  # noqa: E402  (palette + rcParams)

RESULTS = os.path.join(HERE, 'results')
OUT = os.path.join(HERE, 'flip_analysis')

DATASETS = {  # label -> results file prefix
    'CIFAR-10-C': 'CIFAR10C_clip_vitbase32_mixed_nall_bs100_s0',
    'CIFAR-100-C': 'CIFAR100C_clip_vitbase16_mixed_nall_bs100_s0',
    'ImageNet-C': 'ImageNetC5K_clip_vitbase16_mixed_nall_bs100_s0',
}
MODELS = {'CIFAR-10-C': 'ViT-B/32', 'CIFAR-100-C': 'ViT-B/16', 'ImageNet-C': 'ViT-B/16'}

OUTCOMES = ['both right', 'fixed', 'broken', 'kept wrong']
OUTCOME_COLORS = {'both right': '#d9d8d4', 'fixed': BLUE, 'broken': RED, 'kept wrong': MUTED}
OUTCOME_LABELS = {'both right': 'both right', 'fixed': 'fixed (NoAdapt wrong -> Ramen right)',
                  'broken': 'broken (NoAdapt right -> Ramen wrong)', 'kept wrong': 'kept wrong (both wrong)'}


def paired(prefix):
    """Ramen rows joined with NoAdapt's prediction for the same sample, plus the flip outcome."""
    keys = ['domain_idx', 'sample_idx']
    ramen = pd.read_pickle(os.path.join(RESULTS, f'{prefix}_Ramen.pkl'))
    base = pd.read_pickle(os.path.join(RESULTS, f'{prefix}_NoAdapt.pkl'))
    p = ramen.merge(base[keys + ['pred', 'correct']], on=keys, suffixes=('', '_base'), validate='one_to_one')
    assert len(p) == len(ramen) == len(base)
    p['outcome'] = np.select([p.correct & ~p.correct_base, ~p.correct & p.correct_base, p.correct & p.correct_base],
                             ['fixed', 'broken', 'both right'], default='kept wrong')
    p['kept_same_wrong'] = (p.outcome == 'kept wrong') & (p.pred == p.pred_base)
    return p


def summarize(p):
    """One row of counts and percentages for a paired frame (or a slice of it)."""
    c = p.outcome.value_counts().reindex(OUTCOMES, fill_value=0)
    n, base_wrong, base_right = len(p), (~p.correct_base).sum(), p.correct_base.sum()
    same = int(p.kept_same_wrong.sum())
    return pd.Series({
        'n': n,
        'fixed': c['fixed'], 'broken': c['broken'], 'kept wrong': c['kept wrong'], 'both right': c['both right'],
        'kept wrong, same class': same, 'kept wrong, other class': c['kept wrong'] - same,
        'net fixed': c['fixed'] - c['broken'],
        # % of all samples
        'fixed %': c['fixed'] / n * 100, 'broken %': c['broken'] / n * 100,
        'kept wrong %': c['kept wrong'] / n * 100, 'both right %': c['both right'] / n * 100,
        'net %': (c['fixed'] - c['broken']) / n * 100,
        # conditional rates
        'fix rate % (of NoAdapt wrong)': c['fixed'] / base_wrong * 100,
        'kept-wrong rate % (of NoAdapt wrong)': c['kept wrong'] / base_wrong * 100,
        'break rate % (of NoAdapt right)': c['broken'] / base_right * 100,
        'NoAdapt acc %': base_right / n * 100, 'Ramen acc %': p.correct.mean() * 100,
    })


# ---------------------------------------------------------------------------- figures
def plot_overall(overall):
    """100% stacked bar per dataset; segments labelled with count and % of all samples."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    y = np.arange(len(overall))[::-1]
    left = np.zeros(len(overall))
    for o in OUTCOMES:
        pct = overall[f'{o} %'].values
        ax.barh(y, pct, left=left, height=0.6, color=OUTCOME_COLORS[o], label=OUTCOME_LABELS[o],
                edgecolor=SURFACE, linewidth=2)
        for yi, l, v, cnt in zip(y, left, pct, overall[o].values):
            ink = TEXT if o == 'both right' else 'white'
            if v >= 5:
                ax.text(l + v / 2, yi, f'{v:.1f}%\n{cnt:,}', ha='center', va='center', fontsize=8, color=ink)
            elif v >= 2.5:  # too narrow for the count; it is in the report tables
                ax.text(l + v / 2, yi, f'{v:.1f}', ha='center', va='center', fontsize=7, color=ink)
        left += pct
    ax.set_yticks(y, [f'{d}\n(n={n:,})' for d, n in zip(overall.index, overall.n)])
    ax.set_xlim(0, 100)
    ax.set_xlabel('% of all test samples')
    ax.grid(axis='y', visible=False)
    ax.set_title('Ramen vs NoAdapt: per-sample outcome, all corruptions pooled', pad=26)
    _top_legend(ax, ncol=4)
    fig.tight_layout()
    return fig


def plot_conditional(overall):
    """What Ramen does to the samples NoAdapt got wrong, and to those it got right."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.9), gridspec_kw={'width_ratios': [1, 1]})
    y = np.arange(len(overall))[::-1]
    # left: of NoAdapt-wrong samples -> fixed / kept wrong (same class) / kept wrong (other class)
    ax = axes[0]
    wrong = overall['fixed'] + overall['kept wrong']
    parts = [('fixed', overall['fixed'], BLUE, 'fixed'),
             ('same', overall['kept wrong, same class'], MUTED, 'kept wrong, same class'),
             ('other', overall['kept wrong, other class'], '#b9b8b3', 'kept wrong, other class')]
    left = np.zeros(len(overall))
    for _, cnt, color, label in parts:
        pct = (cnt / wrong * 100).values
        ax.barh(y, pct, left=left, height=0.6, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        for yi, l, v in zip(y, left, pct):
            if v >= 6:
                ax.text(l + v / 2, yi, f'{v:.1f}%', ha='center', va='center', fontsize=8,
                        color='white' if color != '#b9b8b3' else TEXT)
        left += pct
    ax.set_yticks(y, [f'{d}\n({w:,} wrong)' for d, w in zip(overall.index, wrong)])
    ax.set_xlim(0, 100)
    ax.set_xlabel('% of samples NoAdapt got wrong')
    ax.set_title('Samples NoAdapt got wrong: what Ramen does', pad=24)
    _top_legend(ax, ncol=3)
    ax.grid(axis='y', visible=False)
    # right: of NoAdapt-right samples -> broken
    ax = axes[1]
    right = overall['both right'] + overall['broken']
    rate = (overall['broken'] / right * 100).values
    ax.barh(y, rate, height=0.6, color=RED)
    for yi, v, cnt in zip(y, rate, overall['broken']):
        ax.text(v + 0.15, yi, f'{v:.1f}%  ({cnt:,} broken)', va='center', fontsize=8, color=TEXT_2)
    ax.set_yticks(y, [f'{d}\n({r:,} right)' for d, r in zip(overall.index, right)])
    ax.set_xlim(0, max(rate) * 1.6)
    ax.set_xlabel('% of samples NoAdapt got right')
    ax.set_title('Samples NoAdapt got right: share Ramen breaks', pad=24)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    return fig


def plot_per_corruption(per_dom):
    """Small multiples: fixed (right of 0) vs broken (left of 0) per corruption, kept-wrong % as text."""
    names = list(per_dom)
    n_rows = max(len(per_dom[d]) for d in names)
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 0.34 * n_rows + 1.8), sharex=True)
    xmax = max(per_dom[d]['fixed %'].max() for d in names)
    xmin = max(per_dom[d]['broken %'].max() for d in names)
    for ax, d in zip(axes, names):
        t = per_dom[d].sort_values('net %')
        y = np.arange(len(t))
        ax.barh(y, t['fixed %'], height=0.62, color=BLUE, label='fixed')
        ax.barh(y, -t['broken %'], height=0.62, color=RED, label='broken')
        ax.axvline(0, color=TEXT_2, lw=1)
        for yi, (_, r) in zip(y, t.iterrows()):
            ax.text(1.02, yi, f"net {r['net %']:+.1f} | kept wrong {r['kept wrong %']:.0f}%",
                    transform=ax.get_yaxis_transform(), va='center', fontsize=7.5, clip_on=False,
                    color=TEXT if r['net %'] < 0 else TEXT_2, fontweight='bold' if r['net %'] < 0 else 'normal')
        ax.set_yticks(y, t.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(-xmin * 1.15, xmax * 1.05)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('% of samples in the corruption')
        ax.set_title(d, pad=24)
        _top_legend(ax)
    fig.suptitle('Per corruption: fixed vs broken (worst net on top)', x=0.01, ha='left',
                 fontsize=12, fontweight='bold', color=TEXT)
    fig.tight_layout()
    return fig


def plot_per_corruption_stacked(per_dom):
    """Small multiples: full four-way split per corruption (100% stacked)."""
    names = list(per_dom)
    n_rows = max(len(per_dom[d]) for d in names)
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 0.34 * n_rows + 1.8), sharex=True)
    for ax, d in zip(axes, names):
        t = per_dom[d].sort_values('kept wrong %')
        y = np.arange(len(t))
        left = np.zeros(len(t))
        for o in OUTCOMES:
            pct = t[f'{o} %'].values
            ax.barh(y, pct, left=left, height=0.7, color=OUTCOME_COLORS[o], label=o, edgecolor=SURFACE, linewidth=1)
            left += pct
        ax.set_yticks(y, t.index, fontsize=8)
        ax.set_xlim(0, 100)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('% of samples in the corruption')
        ax.set_title(d, pad=24)
        _top_legend(ax, ncol=4)
    fig.suptitle('Per corruption: four-way outcome split (fewest kept-wrong at the bottom)', x=0.01, ha='left',
                 fontsize=12, fontweight='bold', color=TEXT)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------- report
def _md_table(df, int_cols, pct_cols):
    cols = int_cols + pct_cols
    head = '| ' + ' | '.join([df.index.name or ''] + cols) + ' |\n|' + '---|' * (len(cols) + 1) + '\n'
    rows = ''.join('| ' + ' | '.join([str(i)] + [f'{r[c]:,.0f}' for c in int_cols] + [f'{r[c]:.2f}' for c in pct_cols])
                   + ' |\n' for i, r in df.iterrows())
    return head + rows


def write_report(overall, per_dom, figs):
    count_cols = ['n', 'fixed', 'broken', 'net fixed', 'kept wrong', 'kept wrong, same class',
                  'kept wrong, other class', 'both right']
    pct_cols = ['fixed %', 'broken %', 'net %', 'kept wrong %', 'both right %']
    rate_cols = ['fix rate % (of NoAdapt wrong)', 'kept-wrong rate % (of NoAdapt wrong)',
                 'break rate % (of NoAdapt right)']
    acc_cols = ['NoAdapt acc %', 'Ramen acc %']
    md = ['# Ramen vs NoAdapt: fixes, breaks and kept-wrong samples (as of 2026-10-22)\n',
          'Mixed-domain stream (all corruptions shuffled together), severity 5, seed 0, batch size 100. '
          'Models: ' + ', '.join(f'{d} {m}' for d, m in MODELS.items()) + '. Both methods see the same '
          'samples in the same order, so each sample is paired with itself:\n',
          '- **fixed**: NoAdapt wrong, Ramen right\n- **broken**: NoAdapt right, Ramen wrong\n'
          '- **kept wrong**: both wrong. *same class* means Ramen made the same wrong prediction as NoAdapt, '
          '*other class* means it switched to a different wrong class\n- **both right**: both right\n',
          '## Overall counts\n', _md_table(overall, count_cols, []),
          '\n## Overall percentages (% of all samples)\n', _md_table(overall, [], pct_cols + acc_cols),
          '\n## Conditional rates\n', _md_table(overall, [], rate_cols),
          '\nThe fix rate is fixed / (fixed + kept wrong). The break rate is broken / (broken + both right).\n']
    for name, path in figs.items():
        md.append(f'\n## {name}\n\n![{name}]({os.path.basename(path)})\n')
    md.append('\n## Per-corruption breakdown\n')
    for d, t in per_dom.items():
        md.append(f'\n### {d}\n\n' + _md_table(t.sort_values('net %'), ['n', 'fixed', 'broken', 'kept wrong'],
                                               ['fixed %', 'broken %', 'net %', 'kept wrong %']))
    md.append('\nCSV files: `overall.csv`, `per_corruption.csv`.\n')
    with open(os.path.join(OUT, 'report.md'), 'w') as f:
        f.write('\n'.join(md))


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs = {}
    for d, prefix in DATASETS.items():
        if all(os.path.exists(os.path.join(RESULTS, f'{prefix}_{a}.pkl')) for a in ('NoAdapt', 'Ramen')):
            pairs[d] = paired(prefix)
        else:
            print(f'skipping {d}: no NoAdapt/Ramen results for {prefix} in {RESULTS}')
    overall = pd.DataFrame({d: summarize(p) for d, p in pairs.items()}).T
    overall.index.name = 'dataset'
    per_dom = {}
    for d, p in pairs.items():
        t = p.groupby('domain', observed=True).apply(summarize, include_groups=False)
        t.index = t.index.astype(str)
        t.index.name = 'corruption'
        per_dom[d] = t
    for c in overall.columns:
        if '%' not in c:
            overall[c] = overall[c].astype(int)
    overall.round(3).to_csv(os.path.join(OUT, 'overall.csv'))
    pd.concat(per_dom, names=['dataset']).round(3).to_csv(os.path.join(OUT, 'per_corruption.csv'))

    figs = {}
    for name, fn, arg in [('Overall outcome split', plot_overall, overall),
                          ('What happens to NoAdapt-wrong and NoAdapt-right samples', plot_conditional, overall),
                          ('Per corruption: fixed vs broken', plot_per_corruption, per_dom),
                          ('Per corruption: four-way split', plot_per_corruption_stacked, per_dom)]:
        path = os.path.join(OUT, name.lower().replace(':', '').replace(' ', '_').replace('-', '_') + '.png')
        fig = fn(arg)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        figs[name] = path
    write_report(overall, per_dom, figs)
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(overall.round(2).T.to_string())
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
