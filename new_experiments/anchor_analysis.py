"""
Zero-shot anchor for Ramen (src/methods/RamenAnchor.py): diagnostics and variant comparison.

Works on a ramen_analysis experiment that has NoAdapt probabilities:
    exp = CIFAR10CExperiment().load(); exp.evaluate(capture_probs=True)
    flips_by_margin(exp)                       # where do harmful / helpful flips sit in zero-shot margin?
    exp.run('RamenAnchor', name='self50', cfg={'lambda_self': 50})
    compare_variants(exp, ['Ramen', 'self50'])
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ramen_analysis import BLUE, RED, CATEGORICAL, _top_legend

KEYS = ['domain_idx', 'sample_idx']
MARGIN_BINS = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _bin_labels(t):
    return [f'{max(iv.left, 0):.1f}-{iv.right:.1f}' for iv in t.index]


def zero_shot_margin(exp):
    """NoAdapt rows with the zero-shot margin m_i = p_(1) - p_(2)."""
    if 'NoAdapt' not in exp.probs:
        raise RuntimeError('Needs NoAdapt probabilities; run exp.evaluate(capture_probs=True).')
    p = np.sort(exp.probs['NoAdapt'], axis=1)
    z = exp.results['NoAdapt'][KEYS + ['label', 'pred', 'correct']]
    return z.assign(margin=p[:, -1] - p[:, -2]).rename(columns={'pred': 'pred_zs', 'correct': 'correct_zs'})


def paired_with_margin(exp, name='Ramen'):
    """Rows of one method joined with the zero-shot prediction and margin of the same sample."""
    d = exp.results[name][KEYS + ['domain', 'pred', 'correct']].merge(zero_shot_margin(exp), on=KEYS)
    d['harmful'] = d.correct_zs & ~d.correct   # zero-shot right -> adapted wrong
    d['helpful'] = ~d.correct_zs & d.correct   # zero-shot wrong -> adapted right
    return d


def flips_by_margin(exp, name='Ramen', bins=MARGIN_BINS):
    """Harmful / helpful flips per zero-shot margin bin (counts and % of the bin's samples)."""
    d = paired_with_margin(exp, name)
    g = d.groupby(pd.cut(d.margin, bins, include_lowest=True), observed=False)
    t = pd.DataFrame({
        'n': g.size(),
        'zero-shot acc %': g.correct_zs.mean() * 100,
        f'{name} acc %': g.correct.mean() * 100,
        'harmful': g.harmful.sum(),
        'helpful': g.helpful.sum(),
    })
    t['harmful % of bin'] = t.harmful / t.n * 100
    t['helpful % of bin'] = t.helpful / t.n * 100
    t['net % of bin'] = t['helpful % of bin'] - t['harmful % of bin']
    t['share of all harmful %'] = t.harmful / t.harmful.sum() * 100
    t['share of all helpful %'] = t.helpful / t.helpful.sum() * 100
    t.index.name = 'zero-shot margin'
    return t.round(2)


def plot_flips_by_margin(exp, name='Ramen', bins=MARGIN_BINS):
    """Left: flip counts per margin bin. Right: flips as % of the bin (how risky adapting is at that margin)."""
    t = flips_by_margin(exp, name, bins)
    labels = _bin_labels(t)
    x = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, (harm, help_), ylabel in zip(axes, [('harmful', 'helpful'), ('harmful % of bin', 'helpful % of bin')],
                                         ['samples', '% of samples in bin']):
        ax.bar(x - 0.2, t[help_], width=0.37, color=BLUE, label='helpful (zero-shot wrong -> right)')
        ax.bar(x + 0.2, t[harm], width=0.37, color=RED, label='harmful (zero-shot right -> wrong)')
        ax.set_xticks(x, labels, rotation=30, ha='right')
        ax.grid(axis='x', visible=False)
        ax.set_xlabel('zero-shot margin  p(1) - p(2)')
        ax.set_ylabel(ylabel)
    axes[0].set_title(f'{name}: flips per margin bin', pad=24)
    axes[1].set_title(f'{name}: flips as % of the bin', pad=24)
    _top_legend(axes[0])
    fig.tight_layout()
    return fig


def margin_gate_sweep(exp, name='Ramen', taus=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """
    Option 4 with gate_scales=[0] computed post hoc: keep the zero-shot prediction wherever m_i > tau.
    Exact, because the gate only changes the query's own prediction and the cache is written before adapting.
    (If adapted and zero-shot agree, falling back changes nothing, so "flipped and m_i > tau" == "m_i > tau".)
    """
    d = paired_with_margin(exp, name)
    rows = []
    for tau in taus:
        g = d.margin > tau
        pred = np.where(g, d.pred_zs, d.pred)
        correct = pred == d.label
        rows.append({'tau': tau, 'gated %': g.mean() * 100, 'acc %': correct.mean() * 100,
                     'harmful': (d.correct_zs & ~correct).sum(), 'helpful': (~d.correct_zs & correct).sum()})
    t = pd.DataFrame(rows).set_index('tau')
    t.attrs[f'{name} acc %'] = d.correct.mean() * 100
    return t.round(2)


def compare_variants(exp, names, high_margin=0.3):
    """One row per method: accuracy, flips against zero-shot, and gate usage for RamenAnchor runs."""
    base = exp.results['Ramen'].groupby('domain_idx').correct.mean() * 100 if 'Ramen' in exp.results else None
    rows = {}
    for name in names:
        d = paired_with_margin(exp, name)
        dom = d.groupby('domain_idx').correct.mean() * 100
        hi = d.margin > high_margin
        row = {
            'avg acc %': dom.mean(),
            'worst corruption acc %': dom.min(),
            'delta vs Ramen': dom.mean() - base.mean() if base is not None else np.nan,
            'corruptions worse than Ramen': int((dom < base - 1e-9).sum()) if base is not None else np.nan,
            'harmful': int(d.harmful.sum()),
            'helpful': int(d.helpful.sum()),
            f'harmful @ m>{high_margin}': int(d.harmful[hi].sum()),
            f'helpful @ m>{high_margin}': int(d.helpful[hi].sum()),
        }
        stats = exp.method_stats.get(name)
        if stats:
            row['self share of alpha %'] = stats['self_share'] / stats['n'] * 100
            row['step shrunk %'] = (1 - stats['gate'][1.0] / stats['n']) * 100
        rows[name] = row
    return pd.DataFrame(rows).T.round(2)


def domain_table(exp, names):
    """Per-corruption accuracy for each method (plus average)."""
    t = pd.DataFrame({n: exp.results[n].groupby('domain_idx').correct.mean() * 100 for n in names})
    t.index = [exp.environments[i] for i in t.index]
    t.loc['average'] = t.mean()
    return t.round(2)


def plot_variant_flips_by_margin(exp, names, bins=MARGIN_BINS):
    """Harmful and helpful flip rate per margin bin, one line per method. The anchor should cut harmful at high m."""
    labels = None
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for i, name in enumerate(names):
        t = flips_by_margin(exp, name, bins)
        labels = _bin_labels(t)
        x = np.arange(len(t))
        for ax, col in zip(axes, ['harmful % of bin', 'helpful % of bin']):
            ax.plot(x, t[col], lw=2, marker='o', ms=7, mec='white', mew=1.5, color=CATEGORICAL[i % 8], label=name)
    for ax, title in zip(axes, ['harmful flips (zero-shot right -> wrong)', 'helpful flips (zero-shot wrong -> right)']):
        ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha='right')
        ax.set_xlabel('zero-shot margin  p(1) - p(2)')
        ax.set_title(title)
    axes[0].set_ylabel('% of samples in bin')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(names), bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def sample_predictions(exp, names, domain_idx, sample_idx):
    """What each method predicts for one test image, with its zero-shot margin."""
    z = zero_shot_margin(exp)
    z = z[(z.domain_idx == domain_idx) & (z.sample_idx == sample_idx)].iloc[0]
    rows = {'zero-shot': {'pred': exp.classes[z.pred_zs], 'correct': bool(z.correct_zs)}}
    for name in names:
        d = exp.results[name]
        r = d[(d.domain_idx == domain_idx) & (d.sample_idx == sample_idx)].iloc[0]
        rows[name] = {'pred': exp.classes[r.pred], 'conf': r.conf, 'correct': bool(r.correct)}
    t = pd.DataFrame(rows).T
    t.attrs['margin'] = z.margin
    print(f'{exp.environments[domain_idx]} #{sample_idx}: true {exp.classes[z.label]}, zero-shot margin {z.margin:.3f}')
    return t
