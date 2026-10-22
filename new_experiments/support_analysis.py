"""
Steps 2-3 of the support-set diagnosis, on a logged Ramen run (src/methods/RamenLog.py, log_support.py).

    exp = make_experiment('CIFAR100C').load(); exp.evaluate(); exp.run('RamenLog')
    sl = SupportLog(exp)
    transition_table(sl)            # 1. zero-shot right/wrong x adapted right/wrong
    topm_recall(sl)                 # 2. top-m recall of zero-shot CLIP
    label_purity(sl)                # 3. share of the support set whose true label == queue label
    domain_purity(sl)               # 4. share of the support set from the query's own corruption
    pick_branch(sl)                 # step 3: which fix the numbers point to
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ramen_analysis import BLUE, RED, ORANGE, MUTED, TEXT_2, _top_legend

LOG_KEYS = ('zs_probs', 'support_ids', 'support_alpha', 'inserted')
CELLS = ['right -> right', 'right -> wrong', 'wrong -> right', 'wrong -> same wrong', 'wrong -> other wrong']
FOCUS_DOMAINS = ('contrast', 'fog', 'defocus_blur', 'elastic_transform')


class SupportLog:
    """A logged run: per-sample results plus memory-mapped support sets, indexed by stream position."""

    def __init__(self, exp, name='RamenLog'):
        self.exp, self.name = exp, name
        prefix = exp._cache_path(name)[:-len('.pkl')]
        self.log = {k: np.load(f'{prefix}_{k}.npy', mmap_mode='r') for k in LOG_KEYS}
        df = exp.results[name]
        assert (df.position.values == np.arange(len(df))).all(), 'results must be in stream order'
        self.df = df
        self.label = df.label.values
        self.domain = df.domain_idx.values
        self.zs_probs = np.asarray(self.log['zs_probs'])
        self.zs_pred = self.zs_probs.argmax(1)  # == the queue each query was written to
        top2 = np.sort(self.zs_probs, axis=1)[:, -2:]
        self.margin = top2[:, 1] - top2[:, 0]
        self.classes, self.domains = exp.classes, exp.environments
        self.C = len(self.classes)

    def chunks(self, size=5000):
        """(rows, ids, alpha, valid) for consecutive blocks of queries."""
        ids_all, alpha_all = self.log['support_ids'], self.log['support_alpha']
        for s in range(0, len(self.df), size):
            ids = np.asarray(ids_all[s:s + size])
            alpha = np.asarray(alpha_all[s:s + size]).astype(np.float32)
            valid = ids >= 0
            yield np.arange(s, s + len(ids)), np.where(valid, ids, 0), alpha * valid, valid


def _domain_rows(table, sl):
    table.index = [sl.domains[i] for i in table.index]
    return table


# ---------------------------------------------------------------------------- 1. transitions
def transition_frame(sl, name=None):
    """Zero-shot (from the log) vs adapted prediction for the logged run or any other run on the same stream."""
    d = sl.df if name is None else sl.exp.results[name]
    keys = ['domain_idx', 'sample_idx']
    z = sl.df[keys].assign(zs_pred=sl.zs_pred, margin=sl.margin)
    d = d[keys + ['label', 'pred']].merge(z, on=keys)
    zs_ok, ad_ok = d.zs_pred == d.label, d.pred == d.label
    d['cell'] = np.select([zs_ok & ad_ok, zs_ok & ~ad_ok, ~zs_ok & ad_ok, ~zs_ok & (d.pred == d.zs_pred)],
                          CELLS[:4], default=CELLS[4])
    return d


def transition_table(sl, name=None, by_domain=False):
    """% of samples in each zero-shot -> adapted cell (overall, or one row per corruption)."""
    d = transition_frame(sl, name)
    if by_domain:
        t = pd.crosstab(d.domain_idx, d.cell, normalize='index') * 100
        t = _domain_rows(t.reindex(columns=CELLS, fill_value=0), sl)
        t.loc['overall'] = (d.cell.value_counts(normalize=True) * 100).reindex(CELLS).values
    else:
        t = pd.DataFrame({'%': d.cell.value_counts(normalize=True).reindex(CELLS, fill_value=0) * 100,
                          'n': d.cell.value_counts().reindex(CELLS, fill_value=0)})
    return t.round(2)


# ---------------------------------------------------------------------------- 2. top-m recall
def topm_recall(sl, ms=(1, 2, 3, 5, 10)):
    """% of samples whose true class is in zero-shot CLIP's top m (overall and per corruption)."""
    ms = [m for m in ms if m <= sl.C]
    order = np.argsort(-sl.zs_probs, axis=1)
    rank = (order == sl.label[:, None]).argmax(1)  # 0 = top-1
    t = pd.DataFrame({f'top-{m}': pd.Series(rank < m).groupby(sl.domain).mean() * 100 for m in ms})
    t = _domain_rows(t, sl)
    t.loc['overall'] = [np.mean(rank < m) * 100 for m in ms]
    return t.round(2)


def error_concentration(sl, top=3):
    """Per true class: share of its zero-shot errors that land in its `top` most frequent wrong classes."""
    err = sl.zs_pred != sl.label
    pairs = pd.Series(1, index=pd.MultiIndex.from_arrays([sl.label[err], sl.zs_pred[err]])).groupby(level=[0, 1]).sum()
    rows = []
    for c, g in pairs.groupby(level=0):
        g = g.sort_values(ascending=False)
        rows.append({'class': sl.classes[c], 'errors': int(g.sum()),
                     f'share in top-{top} wrong %': g.head(top).sum() / g.sum() * 100,
                     'top wrong classes': ', '.join(sl.classes[p] for p in g.head(top).index.get_level_values(1))})
    t = pd.DataFrame(rows).set_index('class').sort_values('errors', ascending=False)
    t.attrs['error-weighted share %'] = (t[f'share in top-{top} wrong %'] * t.errors).sum() / t.errors.sum()
    return t.round(1)


# ---------------------------------------------------------------------------- 3-4. support purity
def _purity(sl, same_fn):
    """Per query: unweighted and alpha-weighted share of support entries for which same_fn(...) holds."""
    unw, wtd = np.empty(len(sl.df)), np.empty(len(sl.df))
    for rows, ids, alpha, valid in sl.chunks():
        same = same_fn(rows, ids) & valid
        unw[rows] = same.sum((1, 2)) / np.maximum(valid.sum((1, 2)), 1)
        wtd[rows] = (alpha * same).sum((1, 2)) / np.maximum(alpha.sum((1, 2)), 1e-12)
    return unw, wtd


def label_purity(sl):
    """
    Share of S_i whose true label equals its queue label (the pseudo-label its gradient pushes towards),
    unweighted and weighted by Ramen's alpha_ij. Also for the query's own queue only.
    """
    queue = np.arange(sl.C)[None, :, None]
    unw, wtd = _purity(sl, lambda rows, ids: sl.label[ids] == queue)
    own_unw, own_wtd = _purity(sl, lambda rows, ids: (sl.label[ids] == queue) & (queue == sl.zs_pred[rows, None, None]))
    own_share_unw, own_share_wtd = _purity(sl, lambda rows, ids: np.broadcast_to(queue == sl.zs_pred[rows, None, None], ids.shape))
    d = pd.DataFrame({'purity %': unw, 'alpha-weighted purity %': wtd,
                      'own-queue purity %': own_unw / np.maximum(own_share_unw, 1e-12),
                      'own-queue alpha-weighted purity %': own_wtd / np.maximum(own_share_wtd, 1e-12)}) * 100
    t = _domain_rows(d.groupby(sl.domain).mean(), sl)
    t.loc['overall'] = d.mean()
    t['zero-shot acc %'] = list(pd.Series(sl.zs_pred == sl.label).groupby(sl.domain).mean() * 100) + \
        [np.mean(sl.zs_pred == sl.label) * 100]
    return t.round(2)


def domain_purity(sl):
    """
    Share of S_i that comes from the query's own corruption (unweighted and alpha-weighted).
    lift = purity / that corruption's share of all retrieved entries (1 = retrieval ignores the corruption).
    """
    unw, wtd = _purity(sl, lambda rows, ids: sl.domain[ids] == sl.domain[rows, None, None])
    counts = np.zeros(len(sl.domains))
    for rows, ids, alpha, valid in sl.chunks():
        counts += np.bincount(sl.domain[ids][valid], minlength=len(sl.domains))
    base = counts / counts.sum() * 100
    d = pd.DataFrame({'purity %': unw * 100, 'alpha-weighted purity %': wtd * 100})
    t = _domain_rows(d.groupby(sl.domain).mean(), sl)
    t['chance %'] = base
    t['lift'] = t['purity %'] / t['chance %']
    t['alpha-weighted lift'] = t['alpha-weighted purity %'] / t['chance %']
    return t.round(2)


# ---------------------------------------------------------------------------- cache health
def cache_health(sl, n_bins=10):
    """Along the stream: share of queries the cache accepted, and the mean age (in samples) of retrieved entries."""
    age = np.empty(len(sl.df))
    for rows, ids, alpha, valid in sl.chunks():
        a = (rows[:, None, None] - ids) * valid
        age[rows] = (a * alpha).sum((1, 2)) / np.maximum(alpha.sum((1, 2)), 1e-12)
    b = np.minimum(np.arange(len(sl.df)) * n_bins // len(sl.df), n_bins - 1)
    t = pd.DataFrame({'inserted %': pd.Series(np.asarray(sl.log['inserted'])).groupby(b).mean() * 100,
                      'alpha-weighted age of support': pd.Series(age).groupby(b).mean(),
                      'Ramen acc %': pd.Series(sl.df.correct.values).groupby(b).mean() * 100,
                      'zero-shot acc %': pd.Series(sl.zs_pred == sl.label).groupby(b).mean() * 100})
    t.index = [f'{i * 100 // n_bins}-{(i + 1) * 100 // n_bins}%' for i in t.index]
    t.index.name = 'stream position'
    t['Ramen gain'] = t['Ramen acc %'] - t['zero-shot acc %']
    return t.round(2)


def plot_cache_health(sl, n_bins=20):
    t = cache_health(sl, n_bins)
    x = np.arange(len(t))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.6))
    ax1.plot(x, t['inserted %'], lw=2, marker='o', ms=6, color=BLUE)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel('% of queries written to the cache')
    ax1.set_title(f'{sl.exp.dataset_name}: cache acceptance along the stream')
    ax2.plot(x, t['Ramen gain'], lw=2, marker='o', ms=6, color=ORANGE)
    ax2.axhline(0, color=TEXT_2, lw=1)
    ax2.set_ylabel('Ramen - zero-shot acc (pts)')
    ax2.set_title('Ramen gain along the stream')
    for ax in (ax1, ax2):
        ax.set_xticks(x[::2], t.index[::2], rotation=30, ha='right')
        ax.set_xlabel('stream position')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------- step 3
def pick_branch(sl, recall_m=None, recall_gap=15.0, purity_low=50.0, lift_ratio=0.5, harmful_share=50.0,
                high_margin=0.5):
    """
    Evaluates each row of the step-3 table and returns them with the evidence. Thresholds are explicit arguments.
      row 1: 'wrong -> same wrong' is the largest cell after 'right -> right' (which is the largest whenever
             zero-shot is decent), and top-m recall - top-1 >= recall_gap (m = 5, or 3 when there are < 50 classes).
      row 2: alpha-weighted label purity < purity_low %.
      row 3: contrast's and fog's alpha-weighted domain lift are both < lift_ratio * the median lift over corruptions.
      row 4: >= harmful_share % of harmful flips have zero-shot margin > high_margin.
      row 5: none of the above.
    """
    m = recall_m or (5 if sl.C >= 50 else 3)
    tt = transition_table(sl)['%']
    errors = tt.drop('right -> right')
    rec = topm_recall(sl, ms=(1, m)).loc['overall']
    lp = label_purity(sl).loc['overall', 'alpha-weighted purity %']
    dp = domain_purity(sl)['alpha-weighted lift']
    med = dp.median()
    d = transition_frame(sl)
    harm = d[d.cell == 'right -> wrong']
    hshare = (harm.margin > high_margin).mean() * 100

    rows = [
        {'row': 1, 'fix': f'candidate-set entropy H_m on top-{m} classes',
         'evidence': f"largest error cell = '{errors.idxmax()}' ({errors.max():.1f}%); "
                     f"top-{m} {rec[f'top-{m}']:.1f}% vs top-1 {rec['top-1']:.1f}% (gap {rec[f'top-{m}'] - rec['top-1']:.1f})",
         'fires': errors.idxmax() == 'wrong -> same wrong' and rec[f'top-{m}'] - rec['top-1'] >= recall_gap},
        {'row': 2, 'fix': 'contamination-aware weighting / retrieval',
         'evidence': f'alpha-weighted label purity {lp:.1f}% (low < {purity_low}%)',
         'fires': lp < purity_low},
        {'row': 3, 'fix': 'change retrieval embedding / distance for contrast, fog',
         'evidence': f"lift contrast {dp.get('contrast', np.nan):.2f}, fog {dp.get('fog', np.nan):.2f}, median {med:.2f}",
         'fires': bool(dp.get('contrast', np.inf) < lift_ratio * med and dp.get('fog', np.inf) < lift_ratio * med)},
        {'row': 4, 'fix': 'gated output (keep zero-shot when margin > tau and classes differ)',
         'evidence': f'{hshare:.1f}% of {len(harm)} harmful flips have margin > {high_margin}',
         'fires': hshare >= harmful_share},
    ]
    rows.append({'row': 5, 'fix': 'normalized SGD instead of SignSGD (ablation)', 'evidence': 'no other row fires',
                 'fires': not any(r['fires'] for r in rows)})
    return pd.DataFrame(rows).set_index('row')


# ---------------------------------------------------------------------------- step 5
def accuracy_by_domain(exp, names):
    """Per-corruption accuracy for each run, plus average and the change vs Ramen."""
    t = pd.DataFrame({n: exp.results[n].groupby('domain_idx').correct.mean() * 100 for n in names})
    t.index = [exp.environments[i] for i in t.index]
    t.loc['average'] = t.mean()
    for n in names:
        if n not in ('NoAdapt', 'Ramen'):
            t[f'{n} - Ramen'] = t[n] - t['Ramen']
    return t.round(2)


def transition_compare(sl, names):
    """Overall transition table (% of samples) for several runs on the same stream, side by side."""
    return pd.DataFrame({n: transition_table(sl, None if n == sl.name else n)['%'] for n in names})


def fix_stats(exp, names):
    """Method-side numbers for RamenClean runs: cache acceptance, mean agreement a_i, own-queue alpha share."""
    rows = {}
    for n in names:
        s = exp.method_stats.get(n)
        if s and 'agreement' in s:
            rows[n] = {'inserted %': s['inserted'] / s['n'] * 100, 'mean agreement a': s['agreement'] / s['n'],
                       'own-queue alpha share before %': s['own_share_before'] / s['n'] * 100,
                       'own-queue alpha share after %': s['own_share_after'] / s['n'] * 100}
    return pd.DataFrame(rows).T.round(3)
