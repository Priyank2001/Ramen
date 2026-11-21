"""
Prior correction toward an estimated class marginal (PREREGISTRATION.md), offline from saved probabilities.

    pc(target):  logit - lam * log(pi_t / q_t),  pi_t = EMA_m of the method's soft predicted marginal (current batch in)
    targets      uniform | oracle (true label marginal, USES LABELS) | zs-EM | zs-marginal | self-EM

    python prior_estimation.py ImageNetVariants 0 1 2 3 4    # -> results/prior_estimation/<ds>_s<seed>_{gains,per_domain,diag}.csv
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

OUT = os.path.join(stack.R, 'prior_estimation') + '/'
DATASETS = ['ImageNetC5K', 'DomainNet', 'ImageNetVariants', 'CIFAR10C', 'CIFAR100C', 'EuroSAT']
NOISE_FLOOR = {**stack.NOISE_FLOOR, 'ImageNetVariants': 0.48}
LAM, M = 1.0, 0.99  # frozen 2026-10-25
TARGETS = ['uniform', 'oracle', 'zs-EM', 'zs-marginal', 'self-EM']
PRIMARY = 'zs-EM'


def zero_shot_probs(ds, df):
    """NoAdapt softmax in df's stream order (seed 0's run; NoAdapt is stateless)."""
    z0 = pd.read_pickle(stack.path(ds, 0, 'NoAdapt') + '.pkl')
    P0 = np.load(stack.path(ds, 0, 'NoAdapt') + '_probs.npy')
    assert (P0.argmax(1) == z0.pred.values).all()
    key0 = pd.Series(np.arange(len(z0)), index=z0.domain_idx.values * 10**7 + z0.sample_idx.values)
    return P0[key0.loc[df.domain_idx.values * 10**7 + df.sample_idx.values].values]


def batch_bounds(batch):
    starts = np.flatnonzero(np.r_[True, batch[1:] != batch[:-1]])
    return starts, np.r_[starts[1:], len(batch)]


def ema_marginal(p, batch, m=M):
    """Per batch: EMA of the soft marginal, current batch included, starting from uniform. (n_batches, K)."""
    k = p.shape[1]
    q, out = np.full(k, 1 / k), []
    for s, e in zip(*batch_bounds(batch)):
        q = m * q + (1 - m) * p[s:e].astype(np.float64).mean(0)
        out.append(q)
    return np.array(out)


def online_em(p, batch, m=M):
    """
    Online Saerens EM: posterior under the current prior estimate, r_i ∝ p_i * q (the classifier's own prior taken as
    uniform), then q <- m q + (1 - m) mean_batch(r). Current batch included, like pi_t. (n_batches, K).
    """
    k = p.shape[1]
    q, out = np.full(k, 1 / k), []
    for s, e in zip(*batch_bounds(batch)):
        r = p[s:e].astype(np.float64) * q
        r /= r.sum(1, keepdims=True)
        q = m * q + (1 - m) * r.mean(0)
        out.append(q)
    return np.array(out)


def target_series(name, df, P, Pzs):
    k, nb = P.shape[1], df.batch.nunique()
    if name == 'uniform':
        return np.full((nb, k), 1 / k)
    if name == 'oracle':  # ORACLE: true labels
        return np.tile(np.bincount(df.label.values, minlength=k) / len(df), (nb, 1))
    if name == 'zs-EM':
        return online_em(Pzs, df.batch.values)
    if name == 'zs-marginal':
        return ema_marginal(Pzs, df.batch.values)
    if name == 'self-EM':
        return online_em(P, df.batch.values)
    raise ValueError(name)


def prior_correct(p, batch, Q, lam=LAM, m=M):
    """stack.prior_correct with a per-batch target Q (n_batches, K) instead of uniform; returns argmax."""
    k = p.shape[1]
    floor = 0.01 / k
    out = np.log(np.clip(p.astype(np.float64), 1e-30, None))
    pi = np.full(k, 1 / k)
    for b, (s, e) in enumerate(zip(*batch_bounds(batch))):
        pi = m * pi + (1 - m) * p[s:e].astype(np.float64).mean(0)
        out[s:e] -= lam * np.log(np.maximum(pi, floor) / np.maximum(Q[b], floor))
    return out.argmax(1)


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def run_seed(ds, seed):
    df, P_r = stack.load(ds, seed, 'Ramen')
    bases = {'Ramen': P_r}
    if stack.available(ds, seed, 'active_prob_T3'):
        dt, P_t = stack.load(ds, seed, 'active_prob_T3')
        assert (dt.domain_idx.values == df.domain_idx.values).all() and (dt.sample_idx.values == df.sample_idx.values).all()
        bases['T3'] = P_t
    Pzs = zero_shot_probs(ds, df)
    y, batch, k = df.label.values, df.batch.values, P_r.shape[1]
    true = np.bincount(y, minlength=k) / len(y)
    preds, diag = {}, {'seed': seed, 'TV(true, uniform)': tv(true, 1 / k),
                       'TV(true, zero-shot hard marginal)': tv(true, np.bincount(Pzs.argmax(1), minlength=k) / len(y))}
    for b, P in bases.items():
        preds[b] = P.argmax(1)
        diag[f'TV(true, {b} hard marginal)'] = tv(true, np.bincount(preds[b], minlength=k) / len(y))
        for t in TARGETS:
            Q = target_series(t, df, P, Pzs)
            name = f'{b} + pc[{t}]'
            preds[name] = prior_correct(P, batch, Q)
            if t == 'uniform':  # sanity: identical to the stacking code's pc(lam=1)
                assert (preds[name] == stack.prior_correct(P, batch, LAM)).all()
            diag[f'TV(true, final target {t}) [{b}]'] = tv(true, Q[-1])
            diag[f'TV(true, {name} hard marginal)'] = tv(true, np.bincount(preds[name], minlength=k) / len(y))
            diag[f'{name} flips %'] = (preds[name] != preds[b]).mean() * 100
    return df, preds, diag


def table(ds, seed, df, preds):
    """One row per (base, target): accuracy, gain over the same base without pc (pts and images), CI, fixed / broken."""
    y = df.label.values
    cluster = df.sample_idx.values if ds in stack.CORRUPTION_SETS else df.domain_idx.values * 10**7 + df.sample_idx.values
    rows = []
    for name, p in preds.items():
        base = name.split(' + ')[0]
        r = preds[base]
        diff = (p == y).astype(float) - (r == y)
        lo, hi = cluster_bootstrap(diff, df.domain_idx.values, cluster, n=1000) if name != base else (0, 0)
        rows.append({'dataset': ds, 'seed': seed, 'base': base, 'target': name.split('pc[')[1][:-1] if '[' in name else '-',
                     'variant': name, 'acc': stack.domain_acc(df, p),
                     'gain': stack.domain_acc(df, p) - stack.domain_acc(df, r),
                     'gain vs Ramen': stack.domain_acc(df, p) - stack.domain_acc(df, preds['Ramen']),
                     'CI low': lo, 'CI high': hi, 'images vs base': int((p == y).sum() - (r == y).sum()),
                     'fixed vs base': int(((p == y) & (r != y)).sum()), 'broken vs base': int(((p != y) & (r == y)).sum()),
                     'n': len(y)})
    return pd.DataFrame(rows)


def main(ds, seeds):
    os.makedirs(OUT, exist_ok=True)
    for seed in seeds:
        df, preds, diag = run_seed(ds, seed)
        t = table(ds, seed, df, preds)
        t.to_csv(OUT + f'{ds}_s{seed}_gains.csv', index=False)
        pdm = stack.per_domain(ds, seed, df, preds)
        pdm.insert(0, 'dataset', ds)
        pdm.to_csv(OUT + f'{ds}_s{seed}_per_domain.csv', index=False)
        pd.DataFrame([{'dataset': ds, **diag}]).to_csv(OUT + f'{ds}_s{seed}_diag.csv', index=False)
        print(f'== {ds} seed {seed}\n' + t[['variant', 'acc', 'gain', 'CI low', 'CI high']].round(2).to_string(index=False),
              flush=True)


# ---------------------------------------------------------------------------- post-hoc
def structure_target(df, k):
    """
    POST-HOC ORACLE (labels used): knows each domain's size and its set of classes, but assumes the classes within a
    domain are equally common: q_k = sum_d (n_d / N) * 1[k in C_d] / |C_d|. Separates "which classes each domain covers"
    (knowable, e.g. the 200-class R/A protocol) from the exact per-class counts the full oracle also knows.
    """
    q = np.zeros(k)
    for _, g in df.groupby('domain_idx'):
        cls = np.unique(g.label.values)
        q[cls] += len(g) / len(df) / len(cls)
    return q


def structure_oracle(ds='ImageNetVariants', seeds=range(5)):
    """Gain of base + pc[oracle-structure] over the same base, per seed and base. POST-HOC."""
    rows = []
    for seed in seeds:
        df, P_r = stack.load(ds, seed, 'Ramen')
        bases = {'Ramen': P_r, 'T3': stack.load(ds, seed, 'active_prob_T3')[1]}
        y, k = df.label.values, P_r.shape[1]
        q = structure_target(df, k)
        true = np.bincount(y, minlength=k) / len(y)
        for b, P in bases.items():
            p0, p1 = P.argmax(1), prior_correct(P, df.batch.values, np.tile(q, (df.batch.nunique(), 1)))
            rows.append({'dataset': ds, 'seed': seed, 'base': b, 'target': 'oracle-structure (post-hoc)',
                         'gain': stack.domain_acc(df, p1) - stack.domain_acc(df, p0),
                         'images vs base': int((p1 == y).sum() - (p0 == y).sum()), 'TV(true, target)': tv(true, q)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- summaries
def collect(kind='gains', datasets=DATASETS):
    files = sorted(f for f in os.listdir(OUT) if f.endswith(f'_{kind}.csv') and f.split('_s')[0] in datasets)
    return pd.concat([pd.read_csv(OUT + f) for f in files], ignore_index=True)


def summary(t):
    """Per dataset x variant over seeds: gain over the same base (mean, std, min), seeds above the floor, CIs > 0."""
    rows = []
    for (ds, v), g in t[t.target != '-'].groupby(['dataset', 'variant'], sort=False):
        fl = NOISE_FLOOR[ds]
        rows.append({'dataset': ds, 'base': g.base.iloc[0], 'target': g.target.iloc[0], 'seeds': len(g),
                     'gain mean': g.gain.mean(), 'gain std': g.gain.std(), 'gain min': g.gain.min(), 'floor': fl,
                     'above floor': f'{(g.gain > fl).sum()}/{len(g)}', 'CI > 0': f'{(g["CI low"] > 0).sum()}/{len(g)}',
                     'gain vs Ramen': g['gain vs Ramen'].mean(), 'images vs base': g['images vs base'].mean()})
    return pd.DataFrame(rows)


def verdict(s):
    """The pre-registered decisions (PREREGISTRATION.md, 'What counts')."""
    g = s.set_index(['dataset', 'base', 'target'])['gain mean']
    fl = NOISE_FLOOR
    oracle = g[('ImageNetVariants', 'T3', 'oracle')]
    t3 = {ds: g[(ds, 'T3', PRIMARY)] for ds in ('ImageNetVariants', 'ImageNetC5K', 'DomainNet')}
    ramen = {ds: g[(ds, 'Ramen', PRIMARY)] for ds in DATASETS}
    return {
        '1. oracle ceiling: T3 + pc[oracle] on R/A/Sketch': round(oracle, 3),
        '   above floor (direction alive)': bool(oracle > fl['ImageNetVariants']),
        '2. zs-EM on T3 (R/A/Sketch > floor; IN-C, DN > 0)': {k: round(v, 3) for k, v in t3.items()},
        '   zs-EM on Ramen (none below -floor)': {k: round(v, 3) for k, v in ramen.items()},
        '   primary claim holds': bool(t3['ImageNetVariants'] > fl['ImageNetVariants'] and t3['ImageNetC5K'] > 0
                                       and t3['DomainNet'] > 0 and all(v > -fl[k] for k, v in ramen.items())),
    }


def target_distance(d):
    """Mean over seeds of TV(true marginal, estimated target at the end of the stream), per dataset x target and base."""
    rows = {}
    for ds, g in d.groupby('dataset', sort=False):
        r = {'uniform': g['TV(true, uniform)'].mean()}
        for t in TARGETS[2:]:
            for b in ('Ramen', 'T3'):
                c = f'TV(true, final target {t}) [{b}]'
                if c in g and g[c].notna().any() and (t == 'self-EM' or b == 'Ramen'):
                    r[t if t != 'self-EM' else f'self-EM [{b}]'] = g[c].mean()
        r['zero-shot predictions'] = g['TV(true, zero-shot hard marginal)'].mean()
        r['Ramen predictions'] = g['TV(true, Ramen hard marginal)'].mean()
        rows[stack.LABEL[ds]] = r
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------- plots
def plot_gains(t, datasets=DATASETS):
    """Gain of base + pc[target] over the same base: hollow = one seed, filled = mean, dashed = noise floor."""
    import matplotlib.pyplot as plt
    from ramen_analysis import BLUE, MUTED, ORANGE, TEXT, TEXT_2
    colour = {'T3': BLUE, 'Ramen': ORANGE}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for ax, ds in zip(axes.flat, datasets):
        g = t[(t.dataset == ds) & (t.target != '-')]
        names = [f'{b} + pc[{x}]' for b in ('T3', 'Ramen') for x in TARGETS if f'{b} + pc[{x}]' in set(g.variant)][::-1]
        lo, hi = 0, NOISE_FLOOR[ds]
        for y, v in enumerate(names):
            h = g[g.variant == v]
            c = colour[v.split(' + ')[0]]
            ax.plot(h.gain, np.full(len(h), y), 'o', ms=5, mfc='white', mec=c, mew=1.2)
            ax.plot(h.gain.mean(), y, 'o', ms=8, color=c)
            ax.text(1.02, y, f'{h.gain.mean():+.2f} ± {h.gain.std():.2f}', transform=ax.get_yaxis_transform(),
                    va='center', fontsize=8, color=TEXT)
            lo, hi = min(lo, h.gain.min()), max(hi, h.gain.max())
        ax.axvline(0, color=TEXT_2, lw=1)
        ax.axvline(NOISE_FLOOR[ds], color=MUTED, lw=1, ls='--')
        pad = 0.06 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_yticks(range(len(names)), names, fontsize=8.5)
        ax.set_title(f'{stack.LABEL[ds]} (floor {NOISE_FLOOR[ds]})', loc='left', fontsize=10.5)
        ax.set_xlabel('accuracy minus the same base without pc (pts)', fontsize=8.5)
    fig.suptitle('The true marginal is worth a lot; no label-free estimate of it helps on every dataset\n'
                 'blue = base T3, orange = base Ramen; hollow = one seed, filled = mean of 5, dashed = noise floor',
                 x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    main(sys.argv[1], [int(s) for s in sys.argv[2:]])
