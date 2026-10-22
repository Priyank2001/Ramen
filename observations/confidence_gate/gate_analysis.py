"""
Why the zero-shot confidence gate fails: reproduces every number in README.md (same folder).

Works only from the cached per-sample results in new_experiments/results/ (written by the
new_experiments/*_ramen.ipynb notebooks), so no model is run. Needs the zero-shot feature caches
(*_NoAdapt_feats.npy) for the nearest-neighbour signals.

Usage (from the repo root):
    ~/miniconda3/envs/RAMEN/bin/python observations/confidence_gate/gate_analysis.py [--device cuda:0] [--seed 0]

Writes, next to this file:
    results/<dataset>_*.csv   one CSV per table
    results/summary.md        every table, in the order README.md cites them
    figures/*.png
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_IN = os.path.join(ROOT, 'new_experiments', 'results')
OUT, FIG = os.path.join(HERE, 'results'), os.path.join(HERE, 'figures')

DATASETS = {'CIFAR10C': 'clip_vitbase32', 'CIFAR100C': 'clip_vitbase16'}
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
KNN_K = 20

# Same palette as new_experiments/ramen_analysis.py (fixed = blue, broken = orange)
TEXT, TEXT_2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT_2, 'axes.titlecolor': TEXT,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'xtick.color': TEXT_2, 'ytick.color': TEXT_2, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.frameon': False, 'legend.fontsize': 9, 'text.color': TEXT, 'lines.linewidth': 2,
})

_summary = []


def emit(title, table, name, ds, floatfmt=3):
    """Print a table, save it as CSV and queue it for results/summary.md."""
    table.to_csv(os.path.join(OUT, f'{ds}_{name}.csv'))
    _summary.append((f'{ds}: {title}', table, floatfmt))
    print(f'\n## {ds}: {title}\n{table.round(floatfmt).to_string()}')


def to_markdown(table, floatfmt):
    t = table.reset_index() if table.index.name or not isinstance(table.index, pd.RangeIndex) else table
    fmt = lambda v: f'{v:,.{floatfmt}f}' if isinstance(v, (float, np.floating)) else (
        f'{v:,}' if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else str(v))
    lines = ['| ' + ' | '.join(map(str, t.columns)) + ' |', '|' + '---|' * len(t.columns)]
    lines += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in t.itertuples(index=False)]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------- data
def class_names(ds):
    """Class list from src/datasets/corruption/<ds>.py, read without importing the dataset code."""
    src = open(os.path.join(ROOT, 'src', 'datasets', 'corruption', f'{ds}.py')).read()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == 'classes' and isinstance(node.value, ast.List)):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise RuntimeError(f'no class list in {ds}.py')


def load(ds, seed):
    prefix = os.path.join(RESULTS_IN, f'{ds}_{DATASETS[ds]}_mixed_nall_bs100_s{seed}')
    z, r = pd.read_pickle(f'{prefix}_NoAdapt.pkl'), pd.read_pickle(f'{prefix}_Ramen.pkl')
    keys = ['domain_idx', 'sample_idx', 'label', 'batch']
    assert (z[keys].values == r[keys].values).all(), 'NoAdapt and Ramen streams differ'
    p = r[keys + ['domain']].copy()
    for col in ('pred', 'conf', 'entropy', 'correct'):
        p[f'{col}_zs'], p[f'{col}_rm'] = z[col].values, r[col].values
    p['broken'] = p.correct_zs & ~p.correct_rm
    p['fixed'] = ~p.correct_zs & p.correct_rm
    p['disagree'] = p.pred_zs != p.pred_rm
    p['outcome'] = np.select([p.fixed, p.broken, p.correct_zs & p.correct_rm, p.disagree],
                             ['fixed', 'broken', 'both right', 'both wrong, disagree'], 'both wrong, agree')
    feats = np.load(f'{prefix}_NoAdapt_feats.npy')  # zero-shot image features, row i <-> p.iloc[i]
    assert len(feats) == len(p)
    return p, feats


# ---------------------------------------------------------------------------- O1: gain identity
def gate_table(p, ds):
    """Appendix A of the research-directions doc, plus a check that gain = broken saved - fixed lost."""
    n, rows = len(p), []
    for tau in TAUS:
        g = p.conf_zs >= tau
        total = np.where(g, p.correct_zs, p.correct_rm).mean() * 100
        saved, lost = int((g & p.broken).sum()), int((g & p.fixed).sum())
        assert np.isclose(total - p.correct_rm.mean() * 100, (saved - lost) / n * 100), 'identity violated'
        rows.append({'tau': tau, 'gated %': g.mean() * 100,
                     'gated samples that disagree %': p.disagree[g].mean() * 100,
                     'NoAdapt acc in gate': p.correct_zs[g].mean() * 100,
                     'Ramen acc in gate': p.correct_rm[g].mean() * 100,
                     'total acc': total, 'broken saved': saved, 'fixed lost': lost,
                     'gain (pts)': (saved - lost) / n * 100})
    emit('confidence gate sweep (gain = (broken saved - fixed lost) / N, checked)',
         pd.DataFrame(rows).set_index('tau'), 'gate_sweep', ds, 2)

    out = p.outcome.value_counts().reindex(
        ['fixed', 'broken', 'both right', 'both wrong, disagree', 'both wrong, agree'])
    t = pd.DataFrame({'samples': out, '% of stream': out / n * 100})
    t.loc['Ramen acc / ceiling'] = [np.nan, p.correct_rm.mean() * 100]
    t.loc['ceiling (zs or Ramen right)'] = [np.nan, (p.correct_zs | p.correct_rm).mean() * 100]
    t.loc['P(zs right | exactly one right)'] = [np.nan, p.broken.sum() / (p.broken | p.fixed).sum() * 100]
    emit('per-sample outcomes', t, 'outcomes', ds, 2)


# ---------------------------------------------------------------------------- O2: where samples sit
def confidence_profile(p, ds):
    med = p.groupby('outcome').conf_zs.describe(percentiles=[.25, .5, .75])[['count', '25%', '50%', '75%']]
    emit('zero-shot confidence by outcome (quartiles)', med, 'conf_by_outcome', ds)

    rows = []
    for tau in TAUS:
        g = p.conf_zs >= tau
        b, f = (g & p.broken).sum(), (g & p.fixed).sum()
        rows.append({'tau': tau, '% of all broken with conf >= tau': b / p.broken.sum() * 100,
                     '% of all fixed with conf >= tau': f / p.fixed.sum() * 100,
                     'broken / (broken + fixed) in gate %': b / max(b + f, 1) * 100})
    emit('how much of the harm a gate at tau can reach', pd.DataFrame(rows).set_index('tau'), 'reach', ds, 1)

    edges = np.linspace(0, 1, 11)
    bins = pd.cut(p.conf_zs, edges, include_lowest=True)
    t = pd.DataFrame({'broken': p.broken.groupby(bins, observed=False).sum(),
                      'fixed': p.fixed.groupby(bins, observed=False).sum(),
                      'all samples': p.broken.groupby(bins, observed=False).size()})
    t['broken / (broken + fixed) %'] = t.broken / (t.broken + t.fixed) * 100
    t['net if this bin kept zero-shot'] = t.broken - t.fixed
    t.index = t.index.astype(str)
    emit('broken and fixed per zero-shot confidence bin', t, 'conf_bins', ds, 1)
    return t


# ---------------------------------------------------------------------------- O3: signals
def running_overshoot(p, k):
    """Per sample: predicted-share / (1/k) - 1 of Ramen's predictions in all *earlier* batches (causal)."""
    counts = np.zeros((p.batch.max() + 1, k))
    np.add.at(counts, (p.batch.values, p.pred_rm.values), 1)
    before = np.vstack([np.zeros(k), np.cumsum(counts, 0)[:-1]])  # counts strictly before each batch
    tot = before.sum(1, keepdims=True)
    share = np.divide(before, tot, out=np.full_like(before, 1 / k), where=tot > 0)
    return share * k - 1  # (num_batches, k)


def knn_labels(p, feats, query, device, causal, k=KNN_K, chunk=1024):
    """
    Indices of the k nearest neighbours (cosine, zero-shot features) for each query row.
    Always excludes the query's own underlying image (same sample_idx, any corruption), so the 15 corrupted
    copies of one image can't vouch for each other. causal=True keeps only samples from earlier batches.
    """
    X = torch.nn.functional.normalize(torch.tensor(feats, dtype=torch.float32, device=device), dim=1)
    sid = torch.tensor(p.sample_idx.values, device=device)
    bat = torch.tensor(p.batch.values, device=device)
    out = []
    for i in range(0, len(query), chunk):
        q = torch.tensor(query[i:i + chunk], device=device)
        S = X[q] @ X.T
        S[sid[q][:, None] == sid[None, :]] = -np.inf
        if causal:
            S[bat[None, :] >= bat[q][:, None]] = -np.inf
        top = S.topk(k, dim=1)
        idx = top.indices.masked_fill(torch.isinf(top.values), -1)  # early batches can have < k neighbours
        out.append(idx.cpu().numpy())
    return np.concatenate(out)


def neighbour_share(nb, values, target):
    """Fraction of valid neighbours whose `values` equal `target` (per row)."""
    valid = nb >= 0
    hit = (values[np.where(valid, nb, 0)] == target[:, None]) & valid
    return hit.sum(1) / np.maximum(valid.sum(1), 1)


def best_gate(score, p):
    """Best possible 'keep zero-shot when score >= t' gate, threshold tuned on the test stream (optimistic)."""
    m = p.disagree.values  # agreeing samples are unaffected by any gate
    s, d = score[m], (p.broken.values[m].astype(int) - p.fixed.values[m].astype(int))
    order = np.argsort(-s, kind='stable')
    gain = np.cumsum(d[order])
    i = int(gain.argmax())
    if gain[i] <= 0:
        return 0.0, 0
    return gain[i] / len(p) * 100, i + 1


def signals(p, feats, k, device, ds):
    over_glob = np.bincount(p.pred_rm, minlength=k) / len(p) * k - 1   # uniform target: CIFAR-C is balanced
    over_run = running_overshoot(p, k)
    rm_target_run = over_run[p.batch.values, p.pred_rm.values]
    zs_source_run = over_run[p.batch.values, p.pred_zs.values]

    # Nearest neighbours only for the samples whose ranking matters (broken + fixed); others get score 0.
    q = np.where(p.broken | p.fixed)[0]
    nb_c = knn_labels(p, feats, q, device, causal=True)
    nb_o = knn_labels(p, feats, q, device, causal=False)
    knn_causal = np.zeros(len(p))
    knn_causal[q] = (neighbour_share(nb_c, p.pred_rm.values, p.pred_zs.values[q])
                     - neighbour_share(nb_c, p.pred_rm.values, p.pred_rm.values[q]))
    knn_oracle = np.zeros(len(p))
    sup_zs = neighbour_share(nb_o, p.label.values, p.pred_zs.values[q])
    sup_rm = neighbour_share(nb_o, p.label.values, p.pred_rm.values[q])
    knn_oracle[q] = sup_zs - sup_rm

    S = {
        'zero-shot conf': p.conf_zs.values,
        '- zero-shot entropy': -p.entropy_zs.values,
        '- Ramen conf': -p.conf_rm.values,
        'zero-shot conf - Ramen conf': (p.conf_zs - p.conf_rm).values,
        'overshoot of Ramen class (whole stream)': over_glob[p.pred_rm.values],
        'overshoot of Ramen class (running, causal)': rm_target_run,
        'overshoot Ramen class - zs class (running)': rm_target_run - zs_source_run,
        f'kNN pseudo-label vote, causal (k={KNN_K})': knn_causal,
        f'kNN TRUE-label vote (oracle, k={KNN_K})': knn_oracle,
    }
    bf = (p.broken | p.fixed).values
    rows = []
    for name, s in S.items():
        gain, n_changed = best_gate(s, p)
        rows.append({'signal (high = keep zero-shot)': name,
                     'AUROC broken vs fixed': roc_auc_score(p.broken.values[bf], s[bf]),
                     'AUROC doc def. (zs right, all disagreements)':
                         roc_auc_score(p.correct_zs.values[p.disagree.values], s[p.disagree.values])
                         if 'kNN' not in name else np.nan,  # kNN computed on broken+fixed only
                     'best gate gain (pts, tuned on test)': gain,
                     'disagreements switched': n_changed})
    t = pd.DataFrame(rows).set_index('signal (high = keep zero-shot)')
    emit('separating broken from fixed: AUROC and best achievable gate', t, 'signals', ds)

    nbt = pd.DataFrame({
        'neighbours truly of zero-shot class %': [sup_zs[p.broken.values[q]].mean() * 100,
                                                   sup_zs[p.fixed.values[q]].mean() * 100],
        "neighbours truly of Ramen's class %": [sup_rm[p.broken.values[q]].mean() * 100,
                                                 sup_rm[p.fixed.values[q]].mean() * 100],
        "% where Ramen's class has more neighbours": [(sup_rm > sup_zs)[p.broken.values[q]].mean() * 100,
                                                      (sup_rm > sup_zs)[p.fixed.values[q]].mean() * 100],
    }, index=pd.Index(['broken (zs class = true)', "fixed (Ramen's class = true)"], name='outcome'))
    emit(f'zero-shot feature neighbourhood (k={KNN_K}, true labels, own image excluded)', nbt, 'neighbourhood', ds, 1)
    return t, over_glob


# ---------------------------------------------------------------------------- O4: base-rate bound
def binormal_curve(n_broken, n_fixed, n, aurocs):
    """
    Max gain (pts) of a threshold gate whose score is N(d,1) on broken and N(0,1) on fixed samples
    (equal-variance binormal, AUROC = Phi(d / sqrt 2)). Optimal threshold: t* = d/2 + ln(n_fixed/n_broken)/d.
    """
    out = []
    for a in aurocs:
        d = np.sqrt(2) * norm.ppf(a)
        if d <= 0:
            out.append(0.0)
            continue
        t = d / 2 + np.log(n_fixed / n_broken) / d
        out.append(max(0.0, n_broken * norm.sf(t - d) - n_fixed * norm.sf(t)) / n * 100)
    return np.array(out)


def base_rate_bound(p, sig, ds):
    nb, nf, n = int(p.broken.sum()), int(p.fixed.sum()), len(p)
    grid = np.array([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    t = pd.DataFrame({'AUROC': grid, 'max gain (pts), binormal': binormal_curve(nb, nf, n, grid),
                      'max gain at balanced base rate (pts)': binormal_curve(nb, nb, n, grid)}).set_index('AUROC')
    t.loc['ceiling (all broken saved)'] = [nb / n * 100, nb / n * 100]
    emit(f'gain a gate can reach vs signal AUROC ({nb:,} broken vs {nf:,} fixed)', t, 'bound', ds, 2)
    return nb, nf, n


# ---------------------------------------------------------------------------- O5: class marginal
def class_marginal(p, over_glob, classes, ds):
    tgt = pd.DataFrame({'outcome': ['broken', 'fixed']})
    rows = []
    for name, m in (('broken', p.broken), ('fixed', p.fixed)):
        o = over_glob[p.pred_rm.values[m]]
        rows.append({'outcome': name, "Ramen's class over-predicted %": (o > 0).mean() * 100,
                     "median overshoot of Ramen's class %": np.median(o) * 100})
    emit("which class Ramen moves samples into (whole-stream marginal)", pd.DataFrame(rows).set_index('outcome'),
         'into_overpredicted', ds, 1)

    k = len(classes)
    b = np.bincount(p.pred_rm[p.broken], minlength=k)
    f = np.bincount(p.pred_rm[p.fixed], minlength=k)
    per = pd.DataFrame({'class': classes, 'Ramen overshoot %': over_glob * 100, 'broken into class': b,
                        'fixed into class': f, 'broken / (broken + fixed) %': b / np.maximum(b + f, 1) * 100})
    rho, pval = spearmanr(per['Ramen overshoot %'], per['broken / (broken + fixed) %'])
    per = per.sort_values('Ramen overshoot %', ascending=False).set_index('class')
    emit(f'per target class (Spearman rho = {rho:.2f}, p = {pval:.1e})', per.head(10), 'per_class', ds, 1)
    per.to_csv(os.path.join(OUT, f'{ds}_per_class_all.csv'))


# ---------------------------------------------------------------------------- figures
def fig_confidence(bins_by_ds):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    centers = np.arange(10) / 10 + 0.05
    for ax, (ds, t) in zip(axes, bins_by_ds.items()):
        w = 0.038
        ax.bar(centers - w / 2 - 0.001, t.fixed, w, color=BLUE, label='fixed (zs wrong, Ramen right)')
        ax.bar(centers + w / 2 + 0.001, t.broken, w, color=ORANGE, label='broken (zs right, Ramen wrong)')
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_xlim(0, 1)
        ax.set_xlabel('zero-shot confidence (bins of 0.1)')
        ax.set_ylabel('samples')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.grid(axis='x', visible=False)
        ax.set_title(f'{ds}: broken samples sit where fixed ones do', pad=42)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=1, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'confidence_by_outcome.png'), dpi=150)


def fig_bound(per_ds):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    a = np.linspace(0.5, 0.995, 200)
    for ax, (ds, (nb, nf, n, sig)) in zip(axes, per_ds.items()):
        ax.plot(a, binormal_curve(nb, nf, n, a), color=BLUE, label=f'bound at the real base rate (1 : {nf / nb:.1f})')
        ax.plot(a, binormal_curve(nb, nb, n, a), color=MUTED, label='bound if broken = fixed (1 : 1)')
        ax.axhline(nb / n * 100, color=TEXT_2, lw=1)
        ax.text(0.505, nb / n * 100, ' ceiling: every broken sample saved', va='bottom', fontsize=8, color=TEXT_2)
        real = sig[~sig.index.str.contains('oracle')]
        ax.scatter(real['AUROC broken vs fixed'], real['best gate gain (pts, tuned on test)'], s=40, color=ORANGE,
                   edgecolor=SURFACE, linewidth=2, zorder=3, label='deployable signals')
        orc = sig[sig.index.str.contains('oracle')]
        ax.scatter(orc['AUROC broken vs fixed'], orc['best gate gain (pts, tuned on test)'], s=40, color=AQUA,
                   edgecolor=SURFACE, linewidth=2, zorder=3, label='kNN true-label vote (oracle)')
        zc = sig.loc['zero-shot conf']
        ax.annotate('zero-shot conf', (zc.iloc[0], zc.iloc[2]), xytext=(8, 10), textcoords='offset points',
                    fontsize=8, color=TEXT_2)
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, nb / n * 100 * 1.12)
        ax.set_xlabel('AUROC, broken vs fixed')
        ax.set_ylabel('accuracy gain over Ramen (pts)')
        ax.set_title(f'{ds}: gate gain vs signal quality', pad=44)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'gain_vs_auroc.png'), dpi=150)


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    bins_by_ds, bound_by_ds = {}, {}
    for ds in DATASETS:
        classes = class_names(ds)
        p, feats = load(ds, args.seed)
        print(f'\n# {ds}: {len(p):,} samples, Ramen {p.correct_rm.mean() * 100:.2f}, '
              f'NoAdapt {p.correct_zs.mean() * 100:.2f}')
        gate_table(p, ds)
        bins_by_ds[ds] = confidence_profile(p, ds)
        sig, over_glob = signals(p, feats, len(classes), args.device, ds)
        nb, nf, n = base_rate_bound(p, sig, ds)
        bound_by_ds[ds] = (nb, nf, n, sig)
        class_marginal(p, over_glob, classes, ds)

    fig_confidence(bins_by_ds)
    fig_bound(bound_by_ds)
    with open(os.path.join(OUT, 'summary.md'), 'w') as f:
        f.write(f'# gate_analysis.py output (seed {args.seed})\n\nGenerated; do not edit by hand.\n')
        for title, table, ff in _summary:
            f.write(f'\n## {title}\n\n{to_markdown(table, ff)}\n')
    print(f'\nwrote {OUT}/ and {FIG}/')


if __name__ == '__main__':
    main()
