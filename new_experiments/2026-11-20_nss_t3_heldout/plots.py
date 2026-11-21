"""One function per figure of heldout.ipynb (copied from ../2026-11-20_nss_t3_stack/plots.py, per-domain labels), over analysis.py's tables. Palette and style from ramen_analysis.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ramen_analysis import AQUA, BLUE, ORANGE, SURFACE, TEXT, TEXT_2, MUTED  # noqa: E402  (also sets rcParams)

RUN_COLORS = {'T3': BLUE, 'NIQE': AQUA, 'NIQE + T3': ORANGE}  # fixed per run in every figure
ORDER = ['NIQE', 'T3', 'NIQE + T3']


def seed_gains(per_seed, floor):
    """Gain over Ramen per seed (small dots) and the mean (large dot) for each run, against the noise floor."""
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    runs = [r for r in ORDER if r in set(per_seed.run)]
    for y, run in enumerate(runs):
        g = per_seed[per_seed.run == run]['gain vs Ramen'].values
        ax.scatter(g, np.full(len(g), y) + np.linspace(-0.12, 0.12, len(g)), s=28, color=RUN_COLORS[run],
                   alpha=0.6, edgecolor=SURFACE, linewidth=1, zorder=2)
        ax.scatter(g.mean(), y, s=110, color=RUN_COLORS[run], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(g.mean(), y + 0.3, f'{g.mean():+.2f} ± {g.std(ddof=1):.2f}', ha='center', fontsize=9, color=TEXT)
    ax.axvline(floor, color=TEXT_2, lw=1, ls='--')
    ax.text(floor, len(runs) - 0.45, f' noise floor {floor:.2f}', color=TEXT_2, fontsize=9, va='center')
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(runs)), runs)
    ax.set_ylim(-0.6, len(runs) - 0.2)
    ax.set_xlabel('gain over Ramen, points (domain-averaged accuracy); small dots = seeds 0–4, large = mean')
    stack = per_seed[per_seed.run == 'NIQE + T3']['gain vs Ramen'].mean()
    ax.set_title(f"NIQE + T3: {stack:+.2f} over Ramen, {'above' if stack > floor else 'below'} the "
                 f'{floor:.2f} floor', loc='left', fontsize=11)
    fig.tight_layout()
    return fig


def stack_vs_t3(per_seed):
    """Paired gain of the stack over T3 alone per seed, with the photo-clustered 95% CI."""
    s = per_seed[per_seed.run == 'NIQE + T3'].sort_values('seed')
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    y = np.arange(len(s))
    ax.hlines(y, s['CI low vs T3'], s['CI high vs T3'], color=ORANGE, lw=2, alpha=0.5)
    ax.scatter(s['gain vs T3'], y, s=70, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for yi, (_, r) in zip(y, s.iterrows()):
        ax.text(r['CI high vs T3'], yi, f"  {r['gain vs T3']:+.2f}  ({int(r['fixed vs T3']):,} fixed, "
                f"{int(r['broken vs T3']):,} broken)", va='center', fontsize=8, color=TEXT_2)
    ax.axvline(0, color=TEXT_2, lw=1)
    ax.set_yticks(y, [f'seed {k}' for k in s.seed])
    ax.set_xlabel('NIQE + T3 minus T3 alone, points (bars: 95% CI, photos resampled)')
    m = s['gain vs T3'].mean()
    ax.set_title(f'Adding the NIQE vector to T3: {m:+.2f} on average, {(s["gain vs T3"] > 0).sum()}/{len(s)} seeds '
                 'positive', loc='left', fontsize=11)
    ax.margins(x=0.45)
    fig.tight_layout()
    return fig


def per_corruption(corr):
    """Mean gain over Ramen per corruption (seeds averaged), one dot per run."""
    m = corr[corr.run != 'Ramen'].groupby(['corruption', 'run'])['gain vs Ramen'].mean().unstack()
    m = m.sort_values('NIQE + T3')
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    y = np.arange(len(m))
    for run in [r for r in ORDER if r in m]:
        ax.scatter(m[run], y, s=46 if run == 'NIQE + T3' else 30, color=RUN_COLORS[run], label=run,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3 if run == 'NIQE + T3' else 2)
    ax.axvline(0, color=TEXT_2, lw=1)
    ax.set_yticks(y, m.index)
    ax.set_xlabel('gain over Ramen, points (mean over seeds)')
    lost = (m['NIQE + T3'] < 0).sum()
    ax.set_title(f'Per domain: the stack loses to Ramen on {lost} of {len(m)} domains', loc='left',
                 fontsize=11)
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


def same_corruption(summary):
    """Weight share of the top-8 retrieved entries that share the query's corruption, per run (seed mean)."""
    s = summary.set_index('run').loc[[r for r in ORDER if r in set(summary.run)]]
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    ax.barh(range(len(s)), s['same-corruption top8 %'], color=[RUN_COLORS[r] for r in s.index], height=0.6)
    for i, v in enumerate(s['same-corruption top8 %']):
        ax.text(v, i, f' {v:.1f}%', va='center', fontsize=9, color=TEXT)
    ax.set_yticks(range(len(s)), s.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel('weight share of top-8 retrieved entries from the query\'s domain, % (domains: 34% / 8% / 58% of images)')
    ax.set_title('Same-domain share of the retrieved support', loc='left', fontsize=11)
    fig.tight_layout()
    return fig
