"""
Analysis helpers for cache_contamination_oracles.ipynb: count tables and plots for the RamenOracle runs
(src/methods/RamenOracle.py, launched by run_oracles.py). Everything here works on CPU from cached runs.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # new_experiments/, for ramen_analysis

from ramen_analysis import BLUE, ORANGE, AQUA, RED, MUTED, TEXT, TEXT_2, GRID  # noqa: E402
from run_oracles import VARIANTS, experiment  # noqa: E402

NOISE_FLOOR = {'ImageNetC5K': 0.54, 'DomainNet': 0.11}  # 2 x (max - min) of Ramen over seeds 0-4
LABEL = {'ImageNetC5K': 'ImageNet-C (5K, severity 5)', 'DomainNet': 'DomainNet'}

# run name -> (short label for tables/plots, family); a family pairs an oracle with its random control
RUNS = {
    'oracle_none': ('none (= Ramen)', 'sanity'),
    'oracle_file_true': ('file under true class', 'filing'),
    'oracle_drop_wrong': ('hide all wrong entries', 'all wrong'),
    'oracle_random_like_wrong': ('hide random, same rate', 'all wrong'),
    'oracle_drop_cw': ('hide confident-wrong entries', 'confident wrong'),
    'oracle_random_like_cw': ('hide random, same rate', 'confident wrong'),
}
ORDER = ['oracle_none', 'oracle_file_true', 'oracle_drop_wrong', 'oracle_random_like_wrong', 'oracle_drop_cw',
         'oracle_random_like_cw']


def fmt(n, total):
    """'3,790 of 29,784 (12.7%)'"""
    return f'{int(n):,} of {int(total):,} ({n / total:.1%})'


def domain_avg(df):
    return df.groupby('domain_idx').correct.mean().mean() * 100


def load(dataset, seed=0, run_missing=False):
    """Experiment with NoAdapt, Ramen and every cached oracle run. Missing runs are skipped (they belong on a GPU,
    via run_oracles.py) unless run_missing=True."""
    exp = experiment(dataset, seed).load()
    exp.evaluate()
    for name in ORDER:
        if run_missing or os.path.exists(exp._cache_path(name)):
            exp.run('RamenOracle', name=name, cfg=VARIANTS[name])
        else:
            print(f'{name}: not run yet')
    return exp


def available(exp):
    return [n for n in ORDER if n in exp.results]


# ---------------------------------------------------------------------------- the data the code ran on
def causal_median_flags(noadapt):
    """Wrong and confident-wrong flags of every image, in stream order, as RamenOracle computes them: confident =
    entropy below the running median of the entropies seen so far (this batch included). Uses the NoAdapt run's
    entropies (fp32); RamenOracle uses Ramen's internal fp16 zero-shot pass, so a handful of borderline images can
    differ."""
    df = noadapt.sort_values('position')
    ent = df.entropy.values
    batch = df.batch.values
    med = np.empty(len(ent))
    seen = np.empty(0)
    starts = np.flatnonzero(np.r_[True, batch[1:] != batch[:-1]])
    ends = np.r_[starts[1:], len(ent)]
    for s, e in zip(starts, ends):
        seen = np.sort(np.concatenate([seen, ent[s:e]]))  # same rule as the method: sorted, lower median
        med[s:e] = seen[(len(seen) - 1) // 2]
    wrong = ~df.correct.values
    out = pd.DataFrame({'position': df.position.values, 'domain': df.domain.values, 'wrong': wrong,
                        'confident_wrong': wrong & (ent < med)})
    return out.sort_values('position').reset_index(drop=True)


def data_table(exps):
    """Images, domains, classes, and how many images zero-shot CLIP gets wrong / confidently wrong."""
    rows = {}
    for ds, exp in exps.items():
        base = exp.results['NoAdapt']
        flags = causal_median_flags(base)
        n = len(base)
        rows[LABEL[ds]] = {
            'images': f'{n:,}',
            'domains': f'{len(exp.environments)}',
            'images per domain': ', '.join(f'{v:,}' for v in base.domain.value_counts(sort=False).values)
            if base.domain.nunique() <= 6 else f'{n // base.domain.nunique():,} each',
            'classes': f'{len(exp.classes):,}',
            'model': exp.model_name,
            'zero-shot correct': fmt(base.correct.sum(), n),
            'zero-shot wrong': fmt(flags.wrong.sum(), n),
            'confident wrong': fmt(flags.confident_wrong.sum(), n),
            'Ramen correct (seed 0)': fmt(exp.results['Ramen'].correct.sum(), n),
        }
    return pd.DataFrame(rows)


def per_domain_counts(exp):
    base = exp.results['NoAdapt']
    flags = causal_median_flags(base)
    g = flags.groupby('domain', observed=True)
    t = pd.DataFrame({'images': g.size(), 'zero-shot wrong': g.wrong.sum(), 'confident wrong': g.confident_wrong.sum()})
    t['wrong %'] = t['zero-shot wrong'] / t.images * 100
    t['confident wrong %'] = t['confident wrong'] / t.images * 100
    t.loc['all'] = [t.images.sum(), t['zero-shot wrong'].sum(), t['confident wrong'].sum(),
                    t['zero-shot wrong'].sum() / t.images.sum() * 100,
                    t['confident wrong'].sum() / t.images.sum() * 100]
    t[['images', 'zero-shot wrong', 'confident wrong']] = t[['images', 'zero-shot wrong', 'confident wrong']].astype(int)
    return t.round(1)


# ---------------------------------------------------------------------------- sanity
def sanity(exp):
    a, b = exp.results['oracle_none'].pred.values, exp.results['Ramen'].pred.values
    same = int((a == b).sum())
    return f'oracle none vs cached Ramen: same prediction on {fmt(same, len(a))}'


# ---------------------------------------------------------------------------- results
def results_table(exp, dataset):
    """One row per run: correct images, accuracy, gain over Ramen (points and images), fixed / broken vs zero-shot,
    agreement with Ramen, and how contaminated the retrieved support was."""
    base, ramen = exp.results['NoAdapt'], exp.results['Ramen']
    n = len(base)
    ramen_acc, ramen_correct = domain_avg(ramen), int(ramen.correct.sum())
    rows = []
    for name in ['NoAdapt', 'Ramen', *available(exp)]:
        r = exp.results[name]
        assert (r.sample_idx.values == base.sample_idx.values).all(), f'{name}: different stream order'
        acc, correct = domain_avg(r), int(r.correct.sum())
        fixed = int((~base.correct.values & r.correct.values).sum())
        broken = int((base.correct.values & ~r.correct.values).sum())
        st = exp.method_stats.get(name, {})
        sw = np.concatenate(st['support_wrong']) if 'support_wrong' in st else None
        scw = np.concatenate(st['support_cw']) if 'support_cw' in st else None
        mem = st.get('memory')
        hidden = f'{mem[-1][4]:,} of {mem[-1][1]:,}' if mem else ''
        rows.append({
            'run': RUNS[name][0] if name in RUNS else name,
            'name': name,
            'correct images': f'{correct:,} of {n:,}',
            'accuracy %': round(acc, 2),
            'vs Ramen (pts)': round(acc - ramen_acc, 2),
            'vs Ramen (images)': f'{correct - ramen_correct:+,}',
            'clears floor': '' if name in ('NoAdapt', 'Ramen', 'oracle_none') else
            ('yes' if acc - ramen_acc > NOISE_FLOOR[dataset] else 'no'),
            'fixed vs zero-shot': fmt(fixed, n),
            'broken vs zero-shot': fmt(broken, n),
            'same pred as Ramen': f'{(r.pred.values == ramen.pred.values).mean():.1%}',
            'support weight from wrong entries %': round(sw.mean() * 100, 1) if sw is not None else np.nan,
            'support weight from confident-wrong %': round(scw.mean() * 100, 1) if scw is not None else np.nan,
            'hidden entries in memory at end': hidden,
        })
    return pd.DataFrame(rows).set_index('run')


def per_domain_gain(exp):
    ramen = exp.results['Ramen'].groupby('domain', observed=True).correct.mean() * 100
    t = pd.DataFrame({RUNS[n][0] + ('' if RUNS[n][1] in ('sanity', 'filing') else f' [{RUNS[n][1]}]'):
                      exp.results[n].groupby('domain', observed=True).correct.mean() * 100 - ramen
                      for n in available(exp) if n != 'oracle_none'})
    return t.T


# ---------------------------------------------------------------------------- plots
def plot_gains(exps):
    """Gain over Ramen per run, one panel per dataset, with the noise floor marked; each oracle sits next to its
    random control."""
    fig, axes = plt.subplots(1, len(exps), figsize=(7.4 * len(exps), 3.8), squeeze=False)
    for ax, (ds, exp) in zip(axes[0], exps.items()):
        ramen = exp.results['Ramen']
        r_acc, r_cor = domain_avg(ramen), int(ramen.correct.sum())
        names = [n for n in available(exp) if n != 'oracle_none'][::-1]
        for y, name in enumerate(names):
            d = domain_avg(exp.results[name]) - r_acc
            dimg = int(exp.results[name].correct.sum()) - r_cor
            oracle = not name.startswith('oracle_random')
            color = BLUE if oracle else MUTED
            ax.hlines(y, 0, d, color=color, lw=2, alpha=0.6)
            ax.plot(d, y, 'o', ms=8, color=color, mfc=color if oracle else 'white', mew=1.6)
            ax.text(1.02, y, f'{d:+.2f} pts  ({dimg:+,} images)', transform=ax.get_yaxis_transform(),
                    va='center', ha='left', fontsize=8.5, color=TEXT if oracle else TEXT_2)
        fl = NOISE_FLOOR[ds]
        ax.axvline(0, color=TEXT_2, lw=1)
        for x in (fl, -fl):
            ax.axvline(x, color=MUTED, lw=1, ls='--')
        ax.text(fl, len(names) - 0.4, f' noise floor {fl}', color=TEXT_2, fontsize=8.5, va='bottom')
        ax.set_yticks(range(len(names)), [f'{RUNS[n][0]}' + ('' if RUNS[n][1] == 'filing' else f'\n[{RUNS[n][1]}]')
                                          for n in names], fontsize=8.5)
        lim = max(abs(np.array([domain_avg(exp.results[n]) - r_acc for n in names])).max(), fl) * 1.9
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6, len(names) - 0.1)
        ax.set_xlabel('accuracy minus Ramen (points, domain-averaged)')
        ax.set_title(f'{LABEL[ds]}: Ramen {r_acc:.2f}%', loc='left', fontsize=10.5)
    fig.suptitle('Gain over Ramen, seed 0: oracle (blue) vs random control (hollow grey)', x=0.01, ha='left',
                 fontsize=12)
    fig.tight_layout()
    return fig


def _rolling(values, window):
    return pd.Series(values).rolling(window, min_periods=max(1, window // 5)).mean().values


def plot_support_contamination(exps, window=None):
    """Share of each query's update weight that comes from zero-shot-wrong entries, along the stream."""
    fig, axes = plt.subplots(1, len(exps), figsize=(6.2 * len(exps), 3.6), squeeze=False)
    # one colour per family, dashed = its random control
    colors = {'oracle_none': ORANGE, 'oracle_file_true': AQUA, 'oracle_drop_cw': BLUE, 'oracle_random_like_cw': BLUE,
              'oracle_drop_wrong': RED, 'oracle_random_like_wrong': RED}
    for ax, (ds, exp) in zip(axes[0], exps.items()):
        n = len(exp.results['NoAdapt'])
        w = window or max(500, n // 100)
        for name in available(exp):
            sw = np.concatenate(exp.method_stats[name]['support_wrong']) * 100
            label = RUNS[name][0] + ('' if RUNS[name][1] in ('sanity', 'filing') else f' [{RUNS[name][1]}]')
            ax.plot(np.arange(n), _rolling(sw, w), color=colors[name], lw=1.4, label=label,
                    ls='--' if 'random' in name else '-')
        ax.set_xlabel('stream position (images seen)')
        ax.set_ylabel('% of update weight from wrong entries')
        ax.set_ylim(-3, 100)
        ax.set_title(f'{LABEL[ds]} (rolling mean over {w:,} images)', loc='left', fontsize=10.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, frameon=False, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('How contaminated is the retrieved support? (dashed = random control)', x=0.01, ha='left',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def plot_memory_composition(exps, name='oracle_none'):
    """Ramen's memory over the stream: entries filed correctly, wrong but not confident, confident-wrong (counts)."""
    fig, axes = plt.subplots(1, len(exps), figsize=(6.2 * len(exps), 3.4), squeeze=False)
    for ax, (ds, exp) in zip(axes[0], exps.items()):
        m = np.array(exp.method_stats[name]['memory'], dtype=float)
        pos, size, wrong, cw = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
        ax.stackplot(pos, size - wrong, wrong - cw, cw, colors=[AQUA, '#f3b49a', RED], alpha=0.85,
                     labels=['zero-shot right', 'wrong, not confident', 'wrong and confident'])
        ax.set_xlabel('stream position (images seen)')
        ax.set_ylabel('entries in memory')
        ax.set_title(f'{LABEL[ds]}\nat the end: {fmt(wrong[-1], size[-1])} wrong, '
                     f'{fmt(cw[-1], size[-1])} confident-wrong', loc='left', fontsize=9.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    axes[0][0].legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle("What is in Ramen's memory (run 'none' = Ramen)", x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    return fig


def plot_domain_gain(exp, dataset):
    t = per_domain_gain(exp)
    fig, ax = plt.subplots(figsize=(1.1 * t.shape[1] + 3.5, 0.5 * t.shape[0] + 1.6))
    lim = np.nanmax(np.abs(t.values))
    im = ax.imshow(t.values, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(range(t.shape[1]), [str(c).replace('_', ' ') for c in t.columns], rotation=40, ha='right',
                  fontsize=8.5)
    ax.set_yticks(range(t.shape[0]), t.index, fontsize=8.5)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, f'{t.values[i, j]:+.1f}', ha='center', va='center', fontsize=7.5,
                    color='white' if abs(t.values[i, j]) > 0.6 * lim else TEXT)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='points vs Ramen')
    ax.set_title(f'{LABEL[dataset]}: accuracy minus Ramen per domain (blue = better)', loc='left', fontsize=10.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------- seeds 1-4
SWEEP = ['oracle_drop_cw', 'oracle_random_like_cw']  # the variants run on every seed (run_seed_sweep.sh)


def _cached(dataset, seed, name):
    """A cached run's per-sample results, read straight from disk (no model, no NoAdapt run), or None."""
    path = experiment(dataset, seed)._cache_path(name)
    return pd.read_pickle(path) if os.path.exists(path) else None


def seed_table(dataset, seeds=(0, 1, 2, 3, 4)):
    """One row per seed: Ramen, the confident-wrong oracle and its random control, as accuracy, change vs Ramen in
    points and in images, and the oracle's margin over the control. `aligned` checks that every run of that seed saw
    the same stream as the cached Ramen run (same image at every stream position)."""
    rows = []
    for seed in seeds:
        ramen = _cached(dataset, seed, 'Ramen')
        runs = {n: _cached(dataset, seed, n) for n in SWEEP}
        if ramen is None or any(r is None for r in runs.values()):
            print(f'{dataset} seed {seed}: not run yet')
            continue
        key = ramen.sort_values('position')[['domain_idx', 'sample_idx']].values
        row = {'seed': seed, 'Ramen %': domain_avg(ramen)}
        aligned = True
        for n, r in runs.items():
            aligned &= bool((r.sort_values('position')[['domain_idx', 'sample_idx']].values == key).all())
            row[f'{RUNS[n][0]} %'] = domain_avg(r)
            row[f'{n} pts'] = domain_avg(r) - row['Ramen %']
            row[f'{n} images'] = int(r.correct.sum() - ramen.correct.sum())
        row['oracle minus control (pts)'] = row['oracle_drop_cw pts'] - row['oracle_random_like_cw pts']
        row['aligned'] = aligned
        rows.append(row)
    return pd.DataFrame(rows).set_index('seed')


def seed_summary(tables):
    """Per dataset: mean +- std over seeds of each run's change vs Ramen, seeds positive, and the noise floor."""
    out = {}
    for ds, t in tables.items():
        n = len(t)
        for name in SWEEP:
            d = t[f'{name} pts']
            out[(LABEL[ds], RUNS[name][0] + (' (control)' if 'random' in name else ''))] = {
                'seeds': n,
                'vs Ramen, mean +- std (pts)': f'{d.mean():+.2f} +- {d.std(ddof=1) if n > 1 else 0:.2f}',
                'vs Ramen, mean (images)': f'{t[f"{name} images"].mean():+,.0f}',
                'seeds above Ramen': f'{(d > 0).sum()} of {n}',
                'noise floor': NOISE_FLOOR[ds],
                'mean clears floor': 'yes' if d.mean() > NOISE_FLOOR[ds] else 'no',
            }
        m = t['oracle minus control (pts)']
        out[(LABEL[ds], 'oracle minus control')] = {
            'seeds': n, 'vs Ramen, mean +- std (pts)': f'{m.mean():+.2f} +- {m.std(ddof=1) if n > 1 else 0:.2f}',
            'vs Ramen, mean (images)': '', 'seeds above Ramen': f'{(m > 0).sum()} of {n} (oracle > control)',
            'noise floor': '', 'mean clears floor': ''}
    return pd.DataFrame(out).T


def plot_seeds(tables):
    """Change vs Ramen on every seed: oracle (blue) and random control (hollow grey), noise floor shaded."""
    fig, axes = plt.subplots(1, len(tables), figsize=(7.4 * len(tables), 3.2), squeeze=False)
    for ax, (ds, t) in zip(axes[0], tables.items()):
        fl = NOISE_FLOOR[ds]
        ax.axvspan(-fl, fl, color=MUTED, alpha=0.18, lw=0)
        ax.axvline(0, color=TEXT_2, lw=1)
        for y, name in enumerate(SWEEP[::-1]):
            oracle = 'random' not in name
            color = BLUE if oracle else MUTED
            d = t[f'{name} pts'].values
            jitter = np.linspace(-0.12, 0.12, len(d))
            ax.plot(d, y + jitter, 'o', ms=7, color=color, mfc=color if oracle else 'white', mew=1.6, alpha=0.9)
            ax.plot([d.mean()] * 2, [y - 0.25, y + 0.25], color=color, lw=2.5)
            ax.text(1.02, y, f'mean {d.mean():+.2f} pts, {(d > 0).sum()} of {len(d)} seeds > 0',
                    transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=8.5,
                    color=TEXT if oracle else TEXT_2)
        ax.text(fl, 1.45, f' noise floor {fl}', color=TEXT_2, fontsize=8.5, va='bottom')
        ax.set_yticks(range(len(SWEEP)), [RUNS[n][0] + (' (control)' if 'random' in n else '') for n in SWEEP[::-1]],
                      fontsize=8.5)
        lim = max(abs(t[[f'{n} pts' for n in SWEEP]].values).max(), fl) * 1.6
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6, 1.8)
        ax.set_xlabel('accuracy minus Ramen (points, domain-averaged)')
        ax.set_title(f'{LABEL[ds]}: seeds {", ".join(map(str, t.index))}', loc='left', fontsize=10.5)
    fig.suptitle('Hide confident-wrong entries vs Ramen, every seed (dot = seed, bar = mean)', x=0.01, ha='left',
                 fontsize=12)
    fig.tight_layout()
    return fig
