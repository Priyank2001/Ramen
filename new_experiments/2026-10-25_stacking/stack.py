"""
Offline stacking of the output-level fixes on Ramen and on the zero-shot-weighted active set (T=3).

Both fixes act only on a method's output probabilities, which never feed back into Ramen's cache, so they are exact
post-hoc transforms of a saved run:
    prior correction (pc)  logit - lam * log(EMA_m(predicted marginal) / uniform), batch by batch (marginal_correction.py)
    kNN smoothing (knn)    0.5 * p + 0.5 * softmax-weighted mean of p over the K most similar images of EARLIER
                           batches, similarity = cosine of zero-shot CLIP features (quick_checks/knn_seeds.py)
Stack order = knn then pc, as on CIFAR / EuroSAT. Pre-registered settings (fixed before seeing these datasets):
lam=2, m=0.99, K=10, temp 0.01, mix 0.5, T=3. Anything else in here is post-hoc and labelled so.

    python stack.py ImageNetC5K 0 1 2 3 4          # per-seed table -> results/stacking/<ds>_stack_seeds.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from active_seeds import cluster_bootstrap  # noqa: E402

R = os.path.join(HERE, '..', 'results') + '/'
OUT = os.path.join(R, 'stacking') + '/'
MODEL = {'ImageNetC5K': 'clip_vitbase16', 'DomainNet': 'clip_vitbase32', 'CIFAR10C': 'clip_vitbase32',
         'CIFAR100C': 'clip_vitbase16', 'EuroSAT': 'clip_vitbase16',
         'ImageNetVariants': 'clip_vitbase16'}
LABEL = {'ImageNetC5K': 'ImageNet-C 5K', 'DomainNet': 'DomainNet', 'CIFAR10C': 'CIFAR-10-C', 'CIFAR100C': 'CIFAR-100-C',
         'EuroSAT': 'EuroSAT', 'ImageNetVariants': 'ImageNet-R/A/Sketch'}
CORRUPTION_SETS = ('ImageNetC5K', 'CIFAR10C', 'CIFAR100C')  # every base image appears once per corruption
# 2 x Ramen range over seeds 0-4 (floor() rounded; CIFAR / EuroSAT from seed_variance.ipynb)
NOISE_FLOOR = {'ImageNetC5K': 0.53, 'DomainNet': 0.11, 'CIFAR10C': 0.31, 'CIFAR100C': 0.21, 'EuroSAT': 1.51}
PRE = dict(lam=2.0, m=0.99, K=10, temp=0.01, mix=0.5)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def path(ds, seed, name):
    return R + f'{ds}_{MODEL[ds]}_mixed_nall_bs100_s{seed}_{name}'


def base_name(variant, ds, seed):
    """
    Cached run holding a variant's probabilities: the original run if it saved them (Ramen seed 0 on ImageNet-C /
    DomainNet, every Ramen seed on CIFAR / EuroSAT), otherwise the probability-saving re-run `<variant>_p`.
    """
    return variant if os.path.exists(path(ds, seed, variant) + '_probs.npy') else f'{variant}_p'


def available(ds, seed, variant):
    return os.path.exists(path(ds, seed, base_name(variant, ds, seed)) + '_probs.npy')


def load(ds, seed, variant):
    """(per-sample frame, probs) of one run; asserts the probs reproduce the frame's predictions."""
    n = base_name(variant, ds, seed)
    df = pd.read_pickle(path(ds, seed, n) + '.pkl')
    P = np.load(path(ds, seed, n) + '_probs.npy')
    assert (df.position.values == np.arange(len(df))).all()
    assert (P.argmax(1) == df.pred.values).all(), f'{ds} s{seed} {variant}: probs do not match predictions'
    return df, P


def zero_shot_keys(ds, df):
    """NoAdapt (zero-shot) CLIP features in df's stream order. NoAdapt is stateless, so seed 0's run covers every seed."""
    z0 = pd.read_pickle(path(ds, 0, 'NoAdapt') + '.pkl')
    Z0 = np.load(path(ds, 0, 'NoAdapt') + '_feats.npy')
    key0 = pd.Series(np.arange(len(z0)), index=z0.domain_idx.values * 10**7 + z0.sample_idx.values)
    return Z0[key0.loc[df.domain_idx.values * 10**7 + df.sample_idx.values].values]


def neighbours(Z, K=PRE['K'], temp=PRE['temp'], chunk=1000):
    """
    For each image: its K most similar images from earlier batches (index, softmax weight). Batch 0 has none.
    Z is used as stored (unit-norm up to fp16 rounding), NOT re-normalised: at temp 0.01 that rounding moves the
    weights, and the pre-registered numbers (knn_seeds.py) used the raw dot product.
    """
    Z = torch.tensor(Z, device=dev).float()
    N = len(Z)
    bstart = torch.arange(N, device=dev) // 100 * 100
    idx = torch.zeros(N, K, dtype=torch.long, device=dev)
    w = torch.zeros(N, K, device=dev)
    for s in range(100, N, chunk):
        e = min(s + chunk, N)
        sim = Z[s:e] @ Z[:e].T
        sim[torch.arange(e, device=dev)[None, :] >= bstart[s:e, None]] = -float('inf')  # only previous batches
        v, i = sim.topk(min(K, s), dim=1)
        idx[s:e, :i.shape[1]], w[s:e, :i.shape[1]] = i, torch.softmax(v / temp, dim=1)
    return idx.cpu().numpy(), w.cpu().numpy()


def smooth(P, idx, w, mix=PRE['mix']):
    """(1 - mix) * p + mix * (weighted mean of p over the image's earlier neighbours); batch 0 keeps p."""
    Pt = torch.tensor(P, device=dev).float()
    Q = Pt.clone()
    it, wt = torch.tensor(idx, device=dev), torch.tensor(w, device=dev)
    for s in range(100, len(P), 20000):
        e = min(s + 20000, len(P))
        Q[s:e] = (wt[s:e, :, None] * Pt[it[s:e]]).sum(1)
    return ((1 - mix) * Pt + mix * Q).cpu().numpy()


def prior_correct(p, batch, lam=PRE['lam'], m=PRE['m']):
    """Same as marginal_correction.corrected_probs (mixed mode, soft, include_current=True, uniform target)."""
    p = p.astype(np.float64)
    k = p.shape[1]
    target, floor = np.full(k, 1 / k), 0.01 / k
    out = np.log(np.clip(p, 1e-30, None))
    starts = np.flatnonzero(np.r_[True, batch[1:] != batch[:-1]])
    ends = np.r_[starts[1:], len(batch)]
    pi = target.copy()
    for s, e in zip(starts, ends):
        pi = m * pi + (1 - m) * p[s:e].mean(0)
        out[s:e] -= lam * np.log(np.maximum(pi, floor) / target)
    return out.argmax(1)  # only predictions are needed downstream


def domain_acc(df, pred):
    return pd.Series(pred == df.label.values).groupby(df.domain_idx.values).mean().mean() * 100


def floor(ds, seeds=range(5)):
    """Noise floor from the cached Ramen runs: 2 x (max - min) of domain-averaged accuracy over seeds 0-4."""
    acc = [pd.read_pickle(path(ds, s, 'Ramen') + '.pkl').pipe(lambda d: domain_acc(d, d.pred.values)) for s in seeds]
    return 2 * (max(acc) - min(acc)), acc


def stack_seed(ds, seed, lams=(PRE['lam'],)):
    """
    Predictions of every combination for one seed. Returns (frame of Ramen's rows, {name: pred}, kNN diagnostics).
    Names: '<base>', '<base> + pc', '<base> + knn', '<base> + knn + pc' for base in Ramen / T3; lam != 2 adds
    '... pc(lam=x)' variants, which are POST-HOC.
    """
    df, P_r = load(ds, seed, 'Ramen')
    bases = {'Ramen': P_r}
    if available(ds, seed, 'active_prob_T3'):
        dt, P_t = load(ds, seed, 'active_prob_T3')
        assert (dt.domain_idx.values == df.domain_idx.values).all() and (dt.sample_idx.values == df.sample_idx.values).all()
        bases['T3'] = P_t
    idx, w = neighbours(zero_shot_keys(ds, df))
    batch = df.batch.values
    preds, flat = {}, {}
    for b, P in bases.items():
        S = smooth(P, idx, w)
        preds[b] = P.argmax(1)
        preds[f'{b} + knn'] = S.argmax(1)
        for lam in lams:
            tag = 'pc' if lam == PRE['lam'] else f'pc(lam={lam:g})'
            preds[f'{b} + {tag}'] = prior_correct(P, batch, lam)
            preds[f'{b} + knn + {tag}'] = prior_correct(S, batch, lam)
        # why knn and pc clash: smoothing flattens p, so the same log-prior shift moves more argmaxes
        for name, X in ((b, P), (f'{b} + knn', S)):
            flat[f'{name}: median log(p1/p2)'] = log_margin(X)
            flat[f'{name}: pc flips %'] = (preds[f'{name} + pc'] != preds[name]).mean() * 100
    diag = {**knn_diagnostics(df, idx, w, corruption_set=ds in CORRUPTION_SETS), **flat}
    return df, preds, diag


def log_margin(P):
    """Median over images of log(top-1 prob / top-2 prob): how far an additive logit shift must go to flip the argmax."""
    top2 = np.partition(P, -2, axis=1)[:, -2:].astype(np.float64)
    return float(np.median(np.log(np.clip(top2[:, 1], 1e-30, None) / np.clip(top2[:, 0], 1e-30, None))))


def knn_diagnostics(df, idx, w, corruption_set):
    """Where the kNN neighbour weight goes (labels used only to describe it): same true class, same domain, same image."""
    q = slice(100, None)
    i, ww = idx[q], w[q]
    lab, dom, smp = df.label.values, df.domain_idx.values, df.sample_idx.values
    out = {'same true class': (ww * (lab[i] == lab[q, None])).sum(1).mean() * 100,
           'same domain': (ww * (dom[i] == dom[q, None])).sum(1).mean() * 100}
    if corruption_set:  # one base image appears under every corruption (DomainNet ids repeat across domains)
        out['same base image'] = (ww * (smp[i] == smp[q, None])).sum(1).mean() * 100
    return out


def table(ds, seed, df, preds):
    """One row per combination: domain-avg accuracy, gain vs Ramen (pts, images), fixed / broken vs Ramen, CI."""
    y, r = df.label.values, preds['Ramen']
    acc_r = domain_acc(df, r)
    same_images = ds in CORRUPTION_SETS
    cluster = df.sample_idx.values if same_images else df.domain_idx.values * 10**7 + df.sample_idx.values
    rows = []
    for name, p in preds.items():
        diff = (p == y).astype(float) - (r == y)
        lo, hi = cluster_bootstrap(diff, df.domain_idx.values, cluster, n=1000) if name != 'Ramen' else (0, 0)
        rows.append({'seed': seed, 'variant': name, 'acc': domain_acc(df, p), 'gain': domain_acc(df, p) - acc_r,
                     'CI low': lo, 'CI high': hi, 'correct': int((p == y).sum()),
                     'images vs Ramen': int((p == y).sum() - (r == y).sum()),
                     'fixed vs Ramen': int(((p == y) & (r != y)).sum()),
                     'broken vs Ramen': int(((p != y) & (r == y)).sum()), 'n': len(y),
                     'same pred %': (p == r).mean() * 100})
    return pd.DataFrame(rows)


POSTHOC_LAMS = (1.0, 0.5)  # lam=2 failed on DomainNet before (uniform target); these are reported, labelled post-hoc
MAIN = ['Ramen + knn', 'Ramen + pc', 'Ramen + knn + pc', 'T3', 'T3 + knn', 'T3 + pc', 'T3 + knn + pc']


def per_domain(ds, seed, df, preds):
    """Accuracy per domain and gain over Ramen, one row per (variant, domain)."""
    y, dom = df.label.values, df.domain_idx.values
    names = list(df['domain'].cat.categories)
    r = pd.Series(preds['Ramen'] == y).groupby(dom).mean() * 100
    rows = []
    for name, p in preds.items():
        a = pd.Series(p == y).groupby(dom).mean() * 100
        for d in a.index:
            rows.append({'seed': seed, 'variant': name, 'domain': names[d], 'n': int((dom == d).sum()),
                         'acc': a[d], 'gain': a[d] - r[d]})
    return pd.DataFrame(rows)


def main(ds, seeds, lams=(PRE['lam'],) + POSTHOC_LAMS):
    """Per-seed tables -> results/stacking/<ds>_{stack_seeds,per_domain,knn_diag}.csv (one file per seed, merged)."""
    os.makedirs(OUT, exist_ok=True)
    for seed in seeds:
        df, preds, diag = stack_seed(ds, seed, lams)
        t = table(ds, seed, df, preds)
        t['posthoc'] = t.variant.str.contains('lam=')
        t.to_csv(OUT + f'{ds}_s{seed}_stack.csv', index=False)
        per_domain(ds, seed, df, preds).to_csv(OUT + f'{ds}_s{seed}_per_domain.csv', index=False)
        pd.DataFrame([{'seed': seed, **diag}]).to_csv(OUT + f'{ds}_s{seed}_knn_diag.csv', index=False)
        print(f'== {ds} seed {seed}\n' + t.drop(columns=['seed', 'n']).round(2).to_string(index=False), flush=True)


def collect(ds, kind='stack'):
    """All seeds computed so far for one dataset, kind = stack | per_domain | knn_diag."""
    files = sorted(f for f in os.listdir(OUT) if f.startswith(f'{ds}_s') and f.endswith(f'_{kind}.csv'))
    return pd.concat([pd.read_csv(OUT + f) for f in files], ignore_index=True) if files else pd.DataFrame()


# ---------------------------------------------------------------------------- summaries
def fmt(n, total):
    return f'{n:,} of {total:,} ({n / total:.1%})'


def seed_summary(ds, t=None):
    """Per variant over seeds: mean +- std gain, min, seeds positive, above the noise floor, mean images vs Ramen."""
    t = collect(ds) if t is None else t
    fl = NOISE_FLOOR[ds]
    rows = {}
    for v, g in t[t.variant != 'Ramen'].groupby('variant', sort=False):
        rows[v] = {'seeds': len(g), 'gain mean': g.gain.mean(), 'gain std': g.gain.std(), 'gain min': g.gain.min(),
                   'positive': f'{(g.gain > 0).sum()}/{len(g)}', 'above floor': f'{(g.gain > fl).sum()}/{len(g)}',
                   'mean above floor': g.gain.mean() > fl, 'images vs Ramen': g['images vs Ramen'].mean(),
                   'fixed vs Ramen': g['fixed vs Ramen'].mean(), 'broken vs Ramen': g['broken vs Ramen'].mean(),
                   'post-hoc': bool(g.posthoc.iloc[0])}
    return pd.DataFrame(rows).T


def additivity(ds, t=None, base='T3'):
    """Is the stack worth the sum of its parts? Per seed: gain of the full stack vs the sum of the three single fixes."""
    t = collect(ds) if t is None else t
    g = t.pivot(index='seed', columns='variant', values='gain')
    parts = {'T3': g['T3'], 'knn': g['Ramen + knn'], 'pc': g['Ramen + pc']}
    out = pd.DataFrame(parts)
    out['sum of parts'] = out.sum(axis=1)
    out['full stack'] = g[f'{base} + knn + pc']
    out['stack - sum'] = out['full stack'] - out['sum of parts']
    return out


def data_table(datasets, seed=0):
    """What the code ran on: images, domains, classes; NoAdapt and Ramen correct counts (seed 0)."""
    rows = {}
    for ds in datasets:
        z = pd.read_pickle(path(ds, 0, 'NoAdapt') + '.pkl')
        r = pd.read_pickle(path(ds, seed, 'Ramen') + '.pkl')
        sizes = z.groupby('domain', observed=True).size()
        rows[LABEL.get(ds, ds)] = {
            'model': MODEL[ds], 'images': f'{len(z):,}', 'domains': len(sizes),
            'images per domain': (f'{sizes.iloc[0]:,} each' if sizes.nunique() == 1
                                  else f'{sizes.min():,} to {sizes.max():,}'),
            'classes': z.label.nunique(), 'batches of 100': f'{r.batch.nunique():,}',
            'NoAdapt correct': fmt(int(z.correct.sum()), len(z)),
            f'Ramen correct (seed {seed})': fmt(int(r.correct.sum()), len(r)),
            'NoAdapt acc (domain-avg)': round(domain_acc(z, z.pred.values), 2),
            f'Ramen acc (domain-avg, seed {seed})': round(domain_acc(r, r.pred.values), 2)}
    return pd.DataFrame(rows)


def sanity_table(ds, seeds=range(5)):
    """Each probability-saving re-run vs the cached run it replaces: stream order and predictions must be identical."""
    rows = []
    for seed in seeds:
        for v in ('Ramen', 'active_prob_T3'):
            n = base_name(v, ds, seed)
            if not os.path.exists(path(ds, seed, n) + '.pkl'):
                continue
            new, old = pd.read_pickle(path(ds, seed, n) + '.pkl'), pd.read_pickle(path(ds, seed, v) + '.pkl')
            P = np.load(path(ds, seed, n) + '_probs.npy', mmap_mode='r')
            argmax_ok = all((np.asarray(P[i:i + 50000]).argmax(1) == new.pred.values[i:i + 50000]).all()
                            for i in range(0, len(new), 50000))
            rows.append({'seed': seed, 'run': v, 'file': n,
                         'same order': bool((new.sample_idx.values == old.sample_idx.values).all()
                                            and (new.domain_idx.values == old.domain_idx.values).all()),
                         'same predictions': fmt(int((new.pred.values == old.pred.values).sum()), len(new)),
                         'probs argmax == pred': argmax_ok})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- plots
def plot_seed_gains(tables, variants=MAIN):
    """Gain over Ramen per variant: one hollow dot per seed, filled dot = mean, dashed line = noise floor."""
    import matplotlib.pyplot as plt
    from ramen_analysis import BLUE, MUTED, TEXT, TEXT_2
    fig, axes = plt.subplots(1, len(tables), figsize=(7.2 * len(tables), 0.5 * len(variants) + 1.8), squeeze=False)
    for ax, (ds, t) in zip(axes[0], tables.items()):
        names = [v for v in variants if v in set(t.variant)][::-1]
        lim, low = 0, 0
        for y, v in enumerate(names):
            g = t[t.variant == v]
            ax.plot(g.gain, np.full(len(g), y), 'o', ms=6, mfc='white', mec=MUTED, mew=1.3)
            ax.plot(g.gain.mean(), y, 'o', ms=9, color=BLUE)
            ax.text(1.02, y, f'{g.gain.mean():+.2f} ± {g.gain.std():.2f} pts  ({g["images vs Ramen"].mean():+,.0f} img)',
                    transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=8.5, color=TEXT)
            lim, low = max(lim, g.gain.max()), min(low, g.gain.min())
        fl = NOISE_FLOOR[ds]
        ax.axvline(0, color=TEXT_2, lw=1)
        ax.axvline(fl, color=MUTED, lw=1, ls='--')
        ax.text(fl, len(names) - 0.45, f' noise floor {fl}', color=TEXT_2, fontsize=8.5, va='bottom')
        ax.set_yticks(range(len(names)), names, fontsize=9)
        pad = 0.08 * (max(lim, fl) - low)
        ax.set_xlim(low - pad, max(lim, fl) + pad)
        ax.set_ylim(-0.6, len(names) - 0.05)
        ax.set_xlabel('accuracy minus Ramen (points, domain-averaged)')
        n = t.seed.nunique()
        ax.set_title(f'{LABEL.get(ds, ds)}: Ramen {t[t.variant == "Ramen"].acc.mean():.2f}% (mean of {n} seeds)',
                     loc='left', fontsize=10.5)
    fig.suptitle('Gain over Ramen: hollow = one seed, filled = mean over seeds', x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    return fig


def plot_domain_gain(ds, variants=('Ramen + knn', 'Ramen + pc', 'T3', 'T3 + pc', 'T3 + knn + pc'), sort_by='T3 + pc'):
    """Mean gain over Ramen per domain (rows, sorted by `sort_by`) and variant (columns), diverging colours at 0."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    pdm = collect(ds, 'per_domain')
    m = pdm[pdm.variant.isin(variants)].groupby(['domain', 'variant']).gain.mean().unstack()[list(variants)]
    m = m.loc[m[sort_by].sort_values(ascending=False).index]
    cmap = LinearSegmentedColormap.from_list('div', ['#c4442f', '#f2f2f0', '#2a78d6'])
    v = np.abs(m.values).max()
    fig, ax = plt.subplots(figsize=(1.5 * len(variants) + 2.5, 0.36 * len(m) + 1.4))
    ax.imshow(m.values, cmap=cmap, norm=TwoSlopeNorm(0, -v, v), aspect='auto')
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f'{m.values[i, j]:+.2f}', ha='center', va='center', fontsize=8.5,
                    color='white' if abs(m.values[i, j]) > 0.6 * v else '#1f1f1f')
    ax.set_xticks(range(len(variants)), variants, fontsize=9)
    ax.set_yticks(range(len(m)), m.index, fontsize=9)
    ax.grid(False)
    ax.set_title(f'{LABEL.get(ds, ds)}: gain over Ramen per domain (pts, mean over {pdm.seed.nunique()} seeds)',
                 loc='left', fontsize=10.5)
    fig.tight_layout()
    return fig


def plot_fixed_broken(tables, variants=MAIN):
    """Images fixed and broken relative to Ramen (mean over seeds): the net gain is a small difference of two flows."""
    import matplotlib.pyplot as plt
    from ramen_analysis import AQUA, RED, TEXT
    fig, axes = plt.subplots(1, len(tables), figsize=(7.2 * len(tables), 0.5 * len(variants) + 1.8), squeeze=False)
    for ax, (ds, t) in zip(axes[0], tables.items()):
        names = [v for v in variants if v in set(t.variant)][::-1]
        s = t.groupby('variant')[['fixed vs Ramen', 'broken vs Ramen', 'n']].mean().loc[names]
        y = np.arange(len(names))
        ax.barh(y, s['fixed vs Ramen'], color=AQUA, height=0.6, label='fixed (Ramen wrong, variant right)')
        ax.barh(y, -s['broken vs Ramen'], color=RED, height=0.6, label='broken (Ramen right, variant wrong)')
        for yi, (f, b, n) in enumerate(s.values):
            ax.text(1.02, yi, f'+{f:,.0f} / −{b:,.0f}  (net {f - b:+,.0f} of {n:,.0f})',
                    transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=8.5, color=TEXT)
        ax.axvline(0, color='#555', lw=1)
        ax.set_yticks(y, names, fontsize=9)
        ax.set_xlabel('images (mean over seeds); left = broken, right = fixed')
        ax.set_title(LABEL.get(ds, ds), loc='left', fontsize=10.5)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle('Every fix both fixes and breaks images; the gain is the difference', x=0.01, ha='left',
                 fontsize=12)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    main(sys.argv[1], [int(s) for s in sys.argv[2:]])
