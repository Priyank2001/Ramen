"""
Analysis of the domain-sequential runs (PREREGISTRATION.md).

    python sequential_analysis.py        # -> results/sequential/{gains,lag,per_domain}.csv, prints the verdict
Sequential runs: results/<ds>_<model>_sequential_..._{Ramen,active_prob_T3}.pkl; Ramen-reset: ..._single_..._Ramen_reset.pkl.
NoAdapt is stateless: the seed-0 mixed run, realigned by (domain, sample) id.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '2026-10-25_stacking'))
import stack  # noqa: E402
from active_seeds import cluster_bootstrap  # noqa: E402

DATASETS = ['ImageNetC5K', 'DomainNet', 'ImageNetVariants']
SEEDS = range(5)
OUT = os.path.join(stack.R, 'sequential') + '/'
MIXED_T3_GAIN = {'ImageNetC5K': 0.62, 'DomainNet': 0.24, 'ImageNetVariants': 0.59}  # seeds 0-4, earlier runs
RUNS = {'Ramen': ('sequential', 'Ramen'), 'T3': ('sequential', 'active_prob_T3'), 'Ramen-reset': ('single', 'Ramen_reset'),
        'Ramen (mixed)': ('mixed', 'Ramen'), 'T3 (mixed)': ('mixed', 'active_prob_T3'),
        'Ramen-fp32': ('sequential', 'Ramen_fp32'), 'T3-fp32': ('sequential', 'active_prob_T3_fp32')}  # amendment
LAG_BATCHES = 10  # pre-registered: first 10 batches of each domain vs the rest


def path(ds, seed, mode, name):
    return stack.R + f'{ds}_{stack.MODEL[ds]}_{mode}_nall_bs100_s{seed}_{name}.pkl'


def exists(ds, seed, run):
    return os.path.exists(path(ds, seed, *RUNS[run]))


def key(df):
    return df.domain_idx.values * 10**7 + df.sample_idx.values


def load(ds, seed, run):
    df = pd.read_pickle(path(ds, seed, *RUNS[run]))
    return df.set_index(key(df))


def noadapt(ds):
    z = pd.read_pickle(stack.path(ds, 0, 'NoAdapt') + '.pkl')
    return z.set_index(key(z))


def sanity():
    """Sequential-mode NoAdapt (ImageNet-C seed 0) vs the mixed NoAdapt: same images, labels and predictions."""
    s = pd.read_pickle(path('ImageNetC5K', 0, 'sequential', 'NoAdapt'))
    s = s.set_index(key(s))
    z = noadapt('ImageNetC5K').loc[s.index]
    order_ok = bool((np.diff(s.domain_idx.values) >= 0).all())  # domains visited one after another
    return {'images': f'{len(s):,}', 'same label': stack.fmt(int((s.label == z.label).sum()), len(s)),
            'same prediction': stack.fmt(int((s.pred == z.pred).sum()), len(s)), 'domains in order': order_ok}


def code_change_checks():
    """Mixed ImageNet-C Ramen seed 0 re-run after each code change vs the cached run: must be identical."""
    old = pd.read_pickle(path('ImageNetC5K', 0, 'mixed', 'Ramen'))
    rows = {}
    for name, change in [('Ramen_cdistcheck', 'cdist on the matmul path'), ('Ramen_fp16check', '+ priority_dtype refactor (fp16 default)')]:
        if os.path.exists(path('ImageNetC5K', 0, 'mixed', name)):
            new = pd.read_pickle(path('ImageNetC5K', 0, 'mixed', name))
            rows[change] = {'same order': bool((new.sample_idx.values == old.sample_idx.values).all()),
                            'same prediction': stack.fmt(int((new.pred.values == old.pred.values).sum()), len(old))}
    return pd.DataFrame(rows).T


def reset_sanity(datasets=DATASETS, seeds=SEEDS):
    """Ramen and Ramen-reset see the same order and differ only after the first domain: domain 0 must be identical."""
    rows = []
    for ds in datasets:
        for seed in seeds:
            if not (exists(ds, seed, 'Ramen') and exists(ds, seed, 'Ramen-reset')):
                continue
            a, b = load(ds, seed, 'Ramen'), load(ds, seed, 'Ramen-reset')
            same_order = bool((a.index == b.index).all())
            d0 = a.domain_idx.values == 0
            rows.append({'dataset': ds, 'seed': seed, 'same order': same_order,
                         'same prediction, first domain': stack.fmt(int((a.pred.values[d0] == b.pred.values[d0]).sum()), int(d0.sum())),
                         'same prediction, later domains': stack.fmt(int((a.pred.values[~d0] == b.pred.values[~d0]).sum()), int((~d0).sum()))})
    return pd.DataFrame(rows)


def within_domain_batch(df):
    """Batch index counted from the start of the image's domain in this stream (sequential / single only)."""
    first = df.groupby('domain_idx').batch.transform('min')
    return (df.batch - first).values


def seed_rows(ds, seed):
    """Domain-averaged accuracy of every run, the paired gains, and lag rows, for one seed."""
    z = noadapt(ds)
    runs = {r: load(ds, seed, r) for r in RUNS if exists(ds, seed, r)}
    base = runs['Ramen']
    idx = base.index
    dom = base.domain_idx.values
    y = base.label.values
    cluster = base.sample_idx.values if ds in stack.CORRUPTION_SETS else idx.values
    correct = {r: (d.loc[idx].pred.values == y) for r, d in runs.items()}
    correct['NoAdapt'] = z.loc[idx].pred.values == y
    acc = {r: pd.Series(c).groupby(dom).mean().mean() * 100 for r, c in correct.items()}
    rows = []
    for a, b in [('T3', 'Ramen'), ('Ramen', 'NoAdapt'), ('T3', 'NoAdapt'), ('Ramen-reset', 'NoAdapt'),
                 ('Ramen', 'Ramen-reset'), ('Ramen (mixed)', 'NoAdapt'), ('T3 (mixed)', 'Ramen (mixed)'),
                 ('T3-fp32', 'Ramen-fp32'), ('Ramen-fp32', 'Ramen'), ('Ramen-fp32', 'NoAdapt'), ('T3-fp32', 'T3')]:
        if a not in correct or b not in correct:
            continue
        diff = correct[a].astype(float) - correct[b]
        lo, hi = cluster_bootstrap(diff, dom, cluster, n=1000)
        rows.append({'dataset': ds, 'seed': seed, 'comparison': f'{a} - {b}', 'gain': acc[a] - acc[b], 'CI low': lo,
                     'CI high': hi, 'images': int(diff.sum()), 'fixed': int(((diff > 0)).sum()),
                     'broken': int((diff < 0).sum()), 'n': len(y)})
    accs = pd.DataFrame([{'dataset': ds, 'seed': seed, 'run': r, 'acc': a} for r, a in acc.items()])
    # lag: accuracy minus NoAdapt by batch position within the domain, per stream run
    lag = []
    for r in ('Ramen', 'T3', 'Ramen-reset', 'Ramen-fp32', 'T3-fp32'):
        if r not in runs:
            continue
        d = runs[r]
        w = within_domain_batch(d)
        gap = (d.pred.values == d.label.values).astype(float) - (z.loc[d.index].pred.values == d.label.values)
        lag.append(pd.DataFrame({'dataset': ds, 'seed': seed, 'run': r, 'domain_idx': d.domain_idx.values,
                                 'batch in domain': w, 'gap': gap}))
    lag = pd.concat(lag).groupby(['dataset', 'seed', 'run', 'domain_idx', 'batch in domain']).gap.agg(['sum', 'count'])
    pdm = pd.DataFrame({r: pd.Series(c).groupby(dom).mean() * 100 for r, c in correct.items()})
    pdm.index = [list(base.domain.cat.categories)[i] for i in pdm.index]
    pdm = pdm.rename_axis('domain').reset_index().assign(dataset=ds, seed=seed)
    return pd.DataFrame(rows), accs, lag.reset_index(), pdm


def run(datasets=DATASETS, seeds=SEEDS):
    os.makedirs(OUT, exist_ok=True)
    out = {k: [] for k in ('gains', 'acc', 'lag', 'per_domain')}
    for ds in datasets:
        for seed in seeds:
            if not exists(ds, seed, 'Ramen'):
                continue
            for k, v in zip(out, seed_rows(ds, seed)):
                out[k].append(v)
            print(f'{ds} seed {seed} done', flush=True)
    for k, v in out.items():
        pd.concat(v, ignore_index=True).to_csv(OUT + f'{k}.csv', index=False)


def load_csv(kind='gains'):
    return pd.read_csv(OUT + f'{kind}.csv')


def floor(acc):
    """Noise floor per dataset: 2 x (max - min) of sequential Ramen over seeds."""
    r = acc[acc.run == 'Ramen'].groupby('dataset').acc
    return (2 * (r.max() - r.min())).to_dict()


def summary(g, fl):
    rows = []
    for (ds, c), h in g.groupby(['dataset', 'comparison'], sort=False):
        rows.append({'dataset': ds, 'comparison': c, 'seeds': len(h), 'gain mean': h.gain.mean(), 'gain std': h.gain.std(),
                     'gain min': h.gain.min(), 'floor (seq.)': fl[ds], 'above floor': f'{(h.gain > fl[ds]).sum()}/{len(h)}',
                     'CI > 0': f'{(h["CI low"] > 0).sum()}/{len(h)}', 'images': h.images.mean()})
    return pd.DataFrame(rows)


def verdict(s):
    t3 = s[s.comparison == 'T3 - Ramen'].set_index('dataset')
    res = {ds: {'T3 - Ramen (seq.)': round(t3.loc[ds, 'gain mean'], 3), 'floor': round(t3.loc[ds, 'floor (seq.)'], 3),
                'seeds': int(t3.loc[ds, 'seeds']), 'above floor': t3.loc[ds, 'above floor'],
                'mixed-stream gain': MIXED_T3_GAIN[ds]} for ds in t3.index}
    res['H1 holds (mean above floor on all three)'] = bool(len(t3) == 3 and (t3['gain mean'] > t3['floor (seq.)']).all())
    f = s[s.comparison == 'T3-fp32 - Ramen-fp32'].set_index('dataset')  # amendment: same floor, same rule
    res['amendment: T3-fp32 - Ramen-fp32'] = {ds: {'gain': round(f.loc[ds, 'gain mean'], 3), 'seeds': int(f.loc[ds, 'seeds']),
                                                   'above floor': f.loc[ds, 'above floor']} for ds in f.index}
    return res


def lag_table(lag):
    """Pre-registered lag: accuracy minus NoAdapt (pts) in the first 10 batches of each domain vs the rest, mean over seeds."""
    lag = lag.assign(early=lag['batch in domain'] < LAG_BATCHES)
    t = lag.groupby(['dataset', 'run', 'seed', 'early'])[['sum', 'count']].sum()
    t = (t['sum'] / t['count'] * 100).groupby(['dataset', 'run', 'early']).mean().unstack()
    t.columns = [f'batches {LAG_BATCHES}+' if not c else f'first {LAG_BATCHES} batches' for c in t.columns]
    return t[[f'first {LAG_BATCHES} batches', f'batches {LAG_BATCHES}+']]


# ---------------------------------------------------------------------------- memory replay (no GPU)
def replay(filed, capacity, half=True, batch=100, snapshot=None, at=None):
    """
    Ramen's queue rule replayed offline: sample i (stream id i) goes into queue filed[i] with priority i, stored in fp16
    (half=True, as Ramen) or fp32; append while there is room, else replace the first lowest-priority slot if strictly
    higher. Retrieval never changes membership, so this is exact given the filing classes.
    Returns (ids: per class array of stored stream ids at the end, accepted: bool per sample, checkpoints) where
    snapshot(i_end, ids) is called after every batch (i_end = first id of the next batch), or only at the stream
    positions in `at` if given (e.g. domain ends, which are not multiples of the batch size).
    """
    n, k = len(filed), int(filed.max()) + 1
    with np.errstate(over='ignore'):  # ids >= 65,520 overflow to inf in fp16, exactly as in Ramen
        prio = np.arange(n).astype(np.float16 if half else np.float32)
    P = np.full((k, capacity), -np.inf, dtype=prio.dtype)
    ids = np.full((k, capacity), -1, dtype=np.int64)
    size = np.zeros(k, dtype=np.int64)
    accepted = np.zeros(n, dtype=bool)
    due = (lambda e: e in at) if at is not None else (lambda e: e % batch == 0 or e == n)
    for i in range(n):
        c = filed[i]
        if size[c] < capacity:
            j = size[c]; size[c] += 1
        else:
            j = int(np.argmin(P[c]))
            if not prio[i] > P[c, j]:
                if snapshot is not None and due(i + 1):
                    snapshot(i + 1, ids)
                continue
        P[c, j], ids[c, j], accepted[i] = prio[i], i, True
        if snapshot is not None and due(i + 1):
            snapshot(i + 1, ids)
    return ids, accepted


def filed_as(ds, seed, mode='sequential', name='active_prob_T3'):
    """Exact filing classes (Ramen-internal zero-shot predictions) from a RamenActive run's log, in stream order."""
    st = pd.read_pickle(path(ds, seed, mode, name)[:-4] + '_stats.pkl')
    return np.concatenate(st['filed_as'])


def validate_replay(ds='ImageNetC5K', seed=0, mode='mixed', name='active_prob_T3_p', n_check=2000):
    """Replayed queue contents vs RamenActive's logged snapshots (queue of each sample's predicted class, after its batch)."""
    st = pd.read_pickle(path(ds, seed, mode, name)[:-4] + '_stats.pkl')
    filed, snaps = np.concatenate(st['filed_as']), np.concatenate(st['queue_ids'])
    pred = pd.read_pickle(path(ds, seed, mode, name)).pred.values
    cap = snaps.shape[1]
    check = set(np.random.default_rng(0).choice(len(filed), n_check, replace=False).tolist())
    by_end = {}
    for i in check:
        by_end.setdefault((i // 100 + 1) * 100, []).append(i)
    ok = []

    def snapshot(i_end, ids):
        for i in by_end.get(min(i_end, len(filed)), []):
            got = set(ids[pred[i]][ids[pred[i]] >= 0].tolist())
            ok.append(got == set(snaps[i][snaps[i] >= 0].tolist()))
    replay(filed, cap, half=True, snapshot=snapshot)
    return {'samples checked': len(ok), 'replayed queue == logged queue': stack.fmt(int(sum(ok)), len(ok))}


def memory_composition(ds, seed, filed=None):
    """
    Domain composition of the whole memory at the end of each domain of the sequential stream, fp16 vs fp32
    priorities, plus the share of each domain's samples that entered the memory. Descriptive (replay).
    """
    df = pd.read_pickle(path(ds, seed, 'sequential', 'Ramen'))
    filed = filed_as(ds, seed) if filed is None else filed
    dom = df.domain_idx.values
    names = list(df.domain.cat.categories)
    ends = set((np.flatnonzero(np.diff(dom)) + 1).tolist()) | {len(dom)}
    cap = {'ImageNetC5K': 75, 'ImageNetVariants': 75, 'DomainNet': 300}[ds]
    rows = []
    for half in (True, False):
        comp = {}

        def snapshot(i_end, ids):
            if i_end in ends:
                stored = ids[ids >= 0]
                comp[names[dom[i_end - 1]]] = np.bincount(dom[stored], minlength=len(names)) / len(stored) * 100
        _, acc = replay(filed, cap, half=half, snapshot=snapshot, at=ends)
        for d_end, c in comp.items():
            rows.append({'dataset': ds, 'seed': seed, 'priorities': 'fp16' if half else 'fp32', 'after domain': d_end,
                         **{f'% memory from {n}': v for n, v in zip(names, c)},
                         'entered memory, this domain %': acc[dom == names.index(d_end)].mean() * 100})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- plots
def plot_primary(s):
    """T3 - Ramen per dataset in the sequential stream (dots per seed) next to the mixed-stream mean, floor dashed."""
    import matplotlib.pyplot as plt
    from ramen_analysis import BLUE, MUTED, TEXT, TEXT_2
    g = load_csv()
    g = g[g.comparison == 'T3 - Ramen']
    fig, axes = plt.subplots(1, 3, figsize=(15, 2.6))
    for ax, ds in zip(axes, DATASETS):
        h = g[g.dataset == ds]
        fl = s[(s.dataset == ds) & (s.comparison == 'T3 - Ramen')]['floor (seq.)'].iloc[0]
        ax.plot(h.gain, np.ones(len(h)), 'o', ms=6, mfc='white', mec=BLUE, mew=1.3)
        ax.plot(h.gain.mean(), 1, 'o', ms=9, color=BLUE)
        ax.plot(MIXED_T3_GAIN[ds], 0, 'o', ms=9, color=MUTED)
        ax.axvline(0, color=TEXT_2, lw=1)
        ax.axvline(fl, color=MUTED, lw=1, ls='--')
        ax.text(fl, 1.45, f' floor {fl:.2f}', fontsize=8.5, color=TEXT_2)
        ax.set_yticks([0, 1], ['mixed (earlier)', 'sequential'])
        ax.set_ylim(-0.6, 1.7)
        lo = min(0, h.gain.min(), MIXED_T3_GAIN[ds]); hi = max(fl, h.gain.max(), MIXED_T3_GAIN[ds])
        ax.set_xlim(lo - 0.1 * (hi - lo), hi + 0.25 * (hi - lo))
        ax.set_title(f'{stack.LABEL[ds]}: {h.gain.mean():+.2f} ± {h.gain.std():.2f} (n={len(h)})', loc='left',
                     fontsize=10.5, color=TEXT)
        ax.set_xlabel('T3 minus Ramen (pts, domain-avg)')
    fig.suptitle('T3 vs Ramen in the domain-sequential stream: hollow = one seed, filled = mean; grey = mixed-stream mean',
                 x=0.01, ha='left', fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_lag(lag, max_batch=60):
    """Accuracy minus NoAdapt by batch position within each domain (mean over domains and seeds)."""
    import matplotlib.pyplot as plt
    from ramen_analysis import AQUA, BLUE, ORANGE, TEXT_2
    colour = {'Ramen': ORANGE, 'T3': BLUE, 'Ramen-reset': AQUA, 'Ramen-fp32': '#b0561f', 'T3-fp32': '#1c4f8f'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        h = lag[(lag.dataset == ds) & (lag['batch in domain'] < max_batch)]
        for r, g in h.groupby('run'):
            if g.seed.nunique() < 5:  # the seed-0-only fp32 screens are too noisy for a curve
                continue
            # bins of 5 batches; per seed x domain gap, then mean (each domain counts equally)
            g = g.assign(bin=g['batch in domain'] // 5 * 5)
            c = g.groupby(['seed', 'domain_idx', 'bin'])[['sum', 'count']].sum()
            m = (c['sum'] / c['count'] * 100).groupby('bin').mean()
            ax.plot(m.index + 2, m.values, '-o' if 'fp32' not in r else '--s', ms=3.5, lw=2, color=colour[r], label=r)
        ax.axhline(0, color=TEXT_2, lw=1)
        ax.axvline(LAG_BATCHES, color='#bbb', lw=1, ls=':')
        ax.set_title(stack.LABEL[ds], loc='left', fontsize=10.5)
        ax.set_xlabel('batch within the domain (bins of 5)')
        ax.set_ylabel('accuracy minus NoAdapt (pts)')
    axes[1].legend(frameon=False, fontsize=8.5, ncol=2)  # DomainNet has every run (fp32 with 5 seeds)
    fig.suptitle('After a domain switch: gain over zero-shot by position in the domain (mean over domains and seeds)',
                 x=0.01, ha='left', fontsize=11.5)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    run()
    pd.set_option('display.width', 220)
    acc, g = load_csv('acc'), load_csv()
    fl = floor(acc)
    print(acc.groupby(['dataset', 'run']).acc.mean().unstack().round(2).to_string())
    s = summary(g, fl)
    print(s.round(2).to_string(index=False))
    print(lag_table(load_csv('lag')).round(2).to_string())
    print(verdict(s))
