"""
Before/after confusion analysis: what does Ramen change in zero-shot CLIP's confusion matrix, and why?

Compares NoAdapt (C0) and Ramen (C1) confusion matrices on the mixed stream and splits the change into
mechanisms instead of listing top confused pairs (that is what the *_confusion_probs notebooks do):

  1. prediction marginal  - sinks (classes predicted > 2x their true share) and starved classes (< 0.5x),
                            total-variation distance between the predicted and true class distribution
  2. bias counterfactual  - a per-class logit bias on NoAdapt, fitted so its predicted marginal matches Ramen's
                            (globally and per domain). How much of C1 - C0 and of the gain is a pure prior shift?
                            Plus an oracle that matches the TRUE marginal (the ceiling of any prior correction).
  3. error structure      - asymmetry (one-way pull to a sink vs mutual confusion), concentration of error
                            destinations, semantic locality of errors (rank of the predicted class among the true
                            class's nearest classes by NoAdapt image-prototype cosine; 0 = nearest, 1 = farthest)
  4. flows / classes      - fixed, broken and wrong->other-wrong samples; whether fixes come out of sinks;
                            per-class gain vs zero-shot class accuracy (levelling vs rich-get-richer)

Offline: uses the saved NoAdapt seed-0 softmax + features and Ramen's per-sample predictions for seeds 0-4.
NoAdapt is per-sample (no state), so its seed-0 run is realigned to each Ramen seed by (domain, sample).

Usage (from new_experiments/):
    python confusion_shift.py analyse [DATASET ...]      # default: all five datasets, seeds 0-4
Outputs in results/confusion_shift/: summary.csv (dataset x metric, mean/std over seeds), per-seed metrics,
<ds>_domains.csv, <ds>_sinks.csv, <ds>_counterfactual.csv and fig/*.png.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from ramen_analysis import (ROOT, BLUE, ORANGE, AQUA, MUTED, TEXT_2, GRID, DIV, SEQ, EXPERIMENTS,
                            _top_legend)
from datasets.corruption.utils import main_corruptions, read_classnames  # noqa: E402 (src on path via ramen_analysis)
from datasets.eurosat import EUROSAT_CLASSNAMES  # noqa: E402

RESULTS = os.path.join(ROOT, 'new_experiments', 'results')
OUT = os.path.join(RESULTS, 'confusion_shift')
FIG = os.path.join(OUT, 'fig')
DATA = os.path.expanduser('~/data')

MODELS = {'CIFAR10C': 'clip_vitbase32', 'CIFAR100C': 'clip_vitbase16', 'ImageNetC5K': 'clip_vitbase16',
          'DomainNet': 'clip_vitbase32', 'EuroSAT': 'clip_vitbase16'}
DATASETS = list(MODELS)
SEEDS = (0, 1, 2, 3, 4)
SINK, STARVED = 2.0, 0.5  # predicted share / true share thresholds


# ---------------------------------------------------------------------------- loading
def class_names(ds):
    if ds in ('CIFAR10C', 'CIFAR100C'):
        from datasets.corruption.CIFAR10C import CIFAR10C
        from datasets.corruption.CIFAR100C import CIFAR100C
        cls = CIFAR10C if ds == 'CIFAR10C' else CIFAR100C
        obj = cls.__new__(cls)  # set_classes only; skips loading the image arrays
        obj.set_classes()
        return obj.classes
    if ds == 'ImageNetC5K':  # ImageFolder order = sorted wnids
        root = os.path.join(DATA, 'corruption', 'ImageNet-C')
        names = read_classnames(os.path.join(root, 'classnames.txt'))
        return [names[w] for w in sorted(os.listdir(os.path.join(root, main_corruptions[0], '5')))]
    if ds == 'DomainNet':
        return sorted(os.listdir(os.path.join(DATA, 'domainbed', 'domain_net', 'real')))
    if ds == 'EuroSAT':
        return [EUROSAT_CLASSNAMES[c] for c in sorted(os.listdir(os.path.join(DATA, 'EuroSAT', '2750')))]
    raise KeyError(ds)


def _base(ds, seed):
    return os.path.join(RESULTS, f'{ds}_{MODELS[ds]}_mixed_nall_bs100_s{seed}')


class Run:
    """NoAdapt (seed 0, with softmax + features) and the NoAdapt rows realigned to Ramen seeds 0-4."""

    def __init__(self, ds):
        self.ds = ds
        self.classes = class_names(ds)
        self.K = len(self.classes)
        r0 = pd.read_pickle(_base(ds, 0) + '_NoAdapt.pkl')
        self.envs = list(r0.domain.cat.categories)
        self.y = r0.label.values
        self.dom = r0.domain_idx.values
        self.pred0 = r0.pred.values
        self._key = pd.Index(r0.domain_idx.values.astype(np.int64) * 10**7 + r0.sample_idx.values)
        assert self._key.is_unique
        self.p0 = np.load(_base(ds, 0) + '_NoAdapt_probs.npy', mmap_mode='r')
        assert len(self.p0) == len(r0) and self.p0.shape[1] == self.K, (self.p0.shape, len(r0), self.K)
        sc = EXPERIMENTS[ds].superclasses
        self.group = None if sc is None else np.array([sc[c] for c in self.classes])
        self.S = self._prototype_similarity()

    def _prototype_similarity(self):
        """Cosine similarity of class-mean NoAdapt image features (true labels, all domains)."""
        f = np.load(_base(self.ds, 0) + '_NoAdapt_feats.npy', mmap_mode='r')
        proto = np.zeros((self.K, f.shape[1]))
        for s in range(0, len(f), 200_000):
            x = np.asarray(f[s:s + 200_000], dtype=np.float32)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            np.add.at(proto, self.y[s:s + 200_000], x)
        proto /= np.linalg.norm(proto, axis=1, keepdims=True).clip(min=1e-12)
        return proto @ proto.T

    def ramen_pred(self, seed):
        """Ramen's predictions for `seed`, reordered to the NoAdapt rows."""
        r1 = pd.read_pickle(_base(self.ds, seed) + '_Ramen.pkl')
        key = pd.Index(r1.domain_idx.values.astype(np.int64) * 10**7 + r1.sample_idx.values)
        pos = key.get_indexer(self._key)
        assert (pos >= 0).all() and (r1.label.values[pos] == self.y).all()
        return r1.pred.values[pos]

    def seeds(self):
        return [s for s in SEEDS if os.path.exists(_base(self.ds, s) + '_Ramen.pkl')]


# ---------------------------------------------------------------------------- metrics
def confusion(y, p, K):
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K)


def _entropy(c):
    q = c[c > 0] / c.sum()
    return -(q * np.log(q)).sum()


def neighbor_rank(S):
    """rank[i, j] = position of class j among class i's neighbours by S, scaled to [0, 1] (0 = nearest)."""
    K = len(S)
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)
    rank = np.empty((K, K))
    rank[np.arange(K)[:, None], order] = np.arange(K)[None, :]
    return rank / max(K - 2, 1)  # self sits at the last position K-1, others span 0..K-2


def marginal_metrics(y, pred, K, mask=None):
    """Predicted vs true class distribution: TV distance, sink/starved shares, effective number of classes."""
    if mask is not None:
        y, pred = y[mask], pred[mask]
    n = len(y)
    pi, q = np.bincount(y, minlength=K) / n, np.bincount(pred, minlength=K) / n
    present = pi > 0
    ratio = np.where(present, q / np.where(present, pi, 1), np.nan)
    return {
        'acc': (y == pred).mean() * 100,
        'tv_to_true': 0.5 * np.abs(q - pi).sum() * 100,
        'sink_pred_share': q[present & (ratio > SINK)].sum() * 100,       # % of predictions going to sinks
        'starved_classes': (ratio[present] < STARVED).mean() * 100,       # % of classes predicted < 0.5x
        'eff_pred_classes': np.exp(_entropy(q * n)) / np.exp(_entropy(pi * n)) * 100,  # % of the true eff. number
    }


def error_metrics(y, pred, K, rank, group):
    """Structure of the off-diagonal mass."""
    C = confusion(y, pred, K).astype(float)
    np.fill_diagonal(C, 0)
    up = np.triu_indices(K, 1)
    pair_tot = (C + C.T)[up]
    wrong = y != pred
    dest = np.bincount(pred[wrong], minlength=K)
    m = {
        'asymmetry': np.abs(C - C.T)[up].sum() / pair_tot.sum() * 100,   # 0 = mutual, 100 = one-way
        'eff_error_dest': np.exp(_entropy(dest)) / K * 100,              # % of classes errors effectively land in
        'top5_dest_share': np.sort(dest)[::-1][:5].sum() / dest.sum() * 100,
        'err_rank': rank[y[wrong], pred[wrong]].mean(),                  # 0 = nearest class, 0.5 = chance
        'err_nn1': (rank[y[wrong], pred[wrong]] == 0).mean() * 100,      # % of errors into the nearest class
    }
    if group is not None:
        m['err_same_group'] = (group[y[wrong]] == group[pred[wrong]]).mean() * 100
    return m


def flow_metrics(y, pred0, pred1, S):
    ok0, ok1 = pred0 == y, pred1 == y
    ww = ~ok0 & ~ok1 & (pred0 != pred1)
    return {
        'fixed': (~ok0 & ok1).mean() * 100,
        'broken': (ok0 & ~ok1).mean() * 100,
        'wrong_to_other_wrong': ww.mean() * 100,
        # wrong->other wrong: is the new wrong class closer to the true class than the old one?
        'ww_closer': (S[y[ww], pred1[ww]] > S[y[ww], pred0[ww]]).mean() * 100 if ww.any() else np.nan,
        'changed': (pred0 != pred1).mean() * 100,
    }


def sink_attribution(y, pred0, pred1, in_sink0, in_new_sink1, suffix=''):
    """in_sink0[i]: NoAdapt's prediction for sample i is a NoAdapt sink; in_new_sink1[i]: Ramen's prediction is a
    class that only Ramen turns into a sink (both per sample, so sinks can be defined per domain)."""
    ok0, ok1 = pred0 == y, pred1 == y
    fixed, broken = ~ok0 & ok1, ok0 & ~ok1
    return {
        # do fixes come out of NoAdapt's sinks more often than NoAdapt's errors do in general?
        f'noadapt_err_in_sinks{suffix}': in_sink0[~ok0].mean() * 100,
        f'fixed_from_sinks{suffix}': in_sink0[fixed].mean() * 100,
        # do Ramen's new errors land in classes that Ramen turned into sinks?
        f'broken_into_new_sinks{suffix}': in_new_sink1[broken].mean() * 100 if broken.any() else np.nan,
    }


def domain_sink_flags(y, pred0, pred1, dom, K):
    """Per-sample flags as in sink_attribution, with sinks defined within each sample's own domain."""
    f0, f1 = np.zeros(len(y), bool), np.zeros(len(y), bool)
    for d in np.unique(dom):
        mk = dom == d
        s0, _ = sink_masks(y, pred0, K, mk)
        s1, _ = sink_masks(y, pred1, K, mk)
        f0[mk], f1[mk] = s0[pred0[mk]], (s1 & ~s0)[pred1[mk]]
    return f0, f1


def class_metrics(y, pred0, pred1, K):
    cnt = np.bincount(y, minlength=K)
    present = cnt > 0
    a0 = (np.bincount(y, weights=pred0 == y, minlength=K) / cnt.clip(min=1))[present] * 100
    a1 = (np.bincount(y, weights=pred1 == y, minlength=K) / cnt.clip(min=1))[present] * 100
    k10 = max(1, int(round(0.1 * len(a0))))
    worst = np.argsort(a0)[:k10]  # NoAdapt's worst decile, tracked under Ramen
    return {'class_acc_std0': a0.std(), 'class_acc_std1': a1.std(),
            'worst_decile0': a0[worst].mean(), 'worst_decile1': a1[worst].mean(),
            'classes_up': (a1 > a0).mean() * 100, 'classes_down': (a1 < a0).mean() * 100,
            'rho_acc0_gain': spearmanr(a0, a1 - a0).statistic}, a0, a1


def sink_masks(y, pred, K, mask=None):
    if mask is not None:
        y, pred = y[mask], pred[mask]
    pi, q = np.bincount(y, minlength=K), np.bincount(pred, minlength=K)
    ratio = q / pi.clip(min=1)
    return (ratio > SINK) & (pi > 0), ratio


# ---------------------------------------------------------------------------- bias counterfactual
def fit_bias(logp, target, iters=300, lr=1.0):
    """
    Per-class bias b so that argmax(logp + b) has the class counts `target` (multiplicative-weights updates on
    log counts, step decays; keeps the b with the smallest count mismatch). logp: (n, K) torch tensor.
    """
    K = logp.shape[1]
    t = torch.as_tensor(target, dtype=torch.float64)
    b = torch.zeros(K, dtype=logp.dtype)
    best, best_b = np.inf, b.clone()
    for i in range(iters):
        c = torch.bincount((logp + b).argmax(1), minlength=K).double()
        err = (c - t).abs().sum().item()
        if err < best:
            best, best_b = err, b.clone()
        if err == 0:
            break
        b += (lr * 0.99 ** i * (torch.log(t + 0.5) - torch.log(c + 0.5))).to(b.dtype)
    return best_b, best / t.sum().item() / 2 * 100  # bias, residual TV mismatch in %


def counterfactuals(run, pred1, logp):
    """NoAdapt + per-class bias matched to Ramen's marginal (global / per domain) and to the true marginal."""
    y, K, n = run.y, run.K, len(run.y)
    C0, C1 = confusion(y, run.pred0, K), confusion(y, pred1, K)
    dC = np.abs(C1 - C0).sum()
    out = {}
    specs = {'bias: Ramen marginal (global)': ('global', pred1),
             'bias: Ramen marginal (per domain)': ('domain', pred1),
             'bias: true marginal (per domain, oracle)': ('domain', y)}
    for name, (scope, target) in specs.items():
        pb = np.empty(n, dtype=np.int64)
        resid = []
        for idx in ([np.arange(n)] if scope == 'global' else
                    [np.flatnonzero(run.dom == d) for d in np.unique(run.dom)]):
            b, r = fit_bias(logp[idx], np.bincount(target[idx], minlength=K))
            pb[idx] = (logp[idx] + b).argmax(1).numpy()
            resid.append(r)
        Cb = confusion(y, pb, K)
        changed1 = run.pred0 != pred1
        out[name] = {
            'acc': (pb == y).mean() * 100,
            'gain': ((pb == y).mean() - (run.pred0 == y).mean()) * 100,
            'agree_with_ramen': (pb == pred1).mean() * 100,
            # share of the L1 change |C1 - C0| that the bias reproduces (1 - residual / total)
            'dC_explained': (1 - np.abs(C1 - Cb).sum() / dC) * 100,
            # of the samples Ramen changed, % the bias changes to the same class
            'ramen_changes_reproduced': (pb[changed1] == pred1[changed1]).mean() * 100,
            'fit_mismatch_tv': np.mean(resid),
        }
    out['Ramen'] = {'acc': (pred1 == y).mean() * 100, 'gain': ((pred1 == y).mean() - (run.pred0 == y).mean()) * 100,
                    'agree_with_ramen': 100.0, 'dC_explained': 100.0, 'ramen_changes_reproduced': 100.0}
    return out


# ---------------------------------------------------------------------------- per dataset
def analyse_dataset(ds, seeds=None):
    os.makedirs(FIG, exist_ok=True)
    run = Run(ds)
    seeds = seeds or run.seeds()
    rank = neighbor_rank(run.S)
    torch.set_num_threads(16)
    logp = torch.from_numpy(np.log(np.asarray(run.p0, dtype=np.float32).clip(1e-12)))
    y, K = run.y, run.K
    sinks0, ratio0 = sink_masks(y, run.pred0, K)

    rows, cf_rows, keep = [], [], {}
    for s in seeds:
        pred1 = run.ramen_pred(s)
        sinks1, ratio1 = sink_masks(y, pred1, K)
        m = {'dataset': ds, 'seed': s}
        for tag, p in (('0', run.pred0), ('1', pred1)):
            m.update({f'{k}{tag}': v for k, v in marginal_metrics(y, p, K).items()})
            m.update({f'{k}{tag}': v for k, v in error_metrics(y, p, K, rank, run.group).items()})
            # per-domain prior: sinks are often domain-specific (TV to each domain's own label distribution)
            m[f'tv_to_true_dom{tag}'] = np.mean([marginal_metrics(y, p, K, run.dom == d)['tv_to_true']
                                                  for d in np.unique(run.dom)])
        m.update(flow_metrics(y, run.pred0, pred1, run.S))
        m.update(sink_attribution(y, run.pred0, pred1, sinks0[run.pred0], (sinks1 & ~sinks0)[pred1]))
        m.update(sink_attribution(y, run.pred0, pred1, *domain_sink_flags(y, run.pred0, pred1, run.dom, K),
                                  suffix='_dom'))
        cm, a0, a1 = class_metrics(y, run.pred0, pred1, K)
        m.update(cm)
        rows.append(m)
        for name, v in counterfactuals(run, pred1, logp).items():
            cf_rows.append({'dataset': ds, 'seed': s, 'variant': name, **v})
        if s == seeds[0]:
            keep = dict(pred1=pred1, ratio1=ratio1, a0=a0, a1=a1)
        print(f'{ds} seed {s}: NoAdapt {m["acc0"]:.2f} -> Ramen {m["acc1"]:.2f}', flush=True)

    per_seed, cf = pd.DataFrame(rows), pd.DataFrame(cf_rows)
    per_seed.to_csv(os.path.join(OUT, f'{ds}_per_seed.csv'), index=False)
    cf.to_csv(os.path.join(OUT, f'{ds}_counterfactual.csv'), index=False)

    # ---- seed-0 detail tables: sinks and per-domain structure
    pred1, ratio1 = keep['pred1'], keep['ratio1']
    pi = np.bincount(y, minlength=K)
    sinks = pd.DataFrame({'class': run.classes, 'true_n': pi, 'pred_n_noadapt': np.bincount(run.pred0, minlength=K),
                          'pred_n_ramen': np.bincount(pred1, minlength=K), 'ratio_noadapt': ratio0,
                          'ratio_ramen': ratio1})
    sinks['acc_noadapt'] = np.bincount(y, weights=run.pred0 == y, minlength=K) / pi.clip(min=1) * 100
    sinks['acc_ramen'] = np.bincount(y, weights=pred1 == y, minlength=K) / pi.clip(min=1) * 100
    # excess = predictions absorbed beyond the class's own count; ranks sinks by absolute pull, unlike the
    # ratio, which is noisy for classes with few samples (ImageNetC5K has classes with 15)
    sinks['excess_noadapt'] = sinks.pred_n_noadapt - sinks.true_n
    sinks['excess_ramen'] = sinks.pred_n_ramen - sinks.true_n
    sinks = sinks.sort_values('excess_noadapt', ascending=False, ignore_index=True)
    sinks.to_csv(os.path.join(OUT, f'{ds}_sinks.csv'), index=False)

    dom_rows = []
    for d in np.unique(run.dom):
        mk = run.dom == d
        r = {'domain': run.envs[d]}
        for tag, p in (('0', run.pred0), ('1', pred1)):
            mm = marginal_metrics(y, p, K, mk)
            em = error_metrics(y[mk], p[mk], K, rank, run.group)
            sk, rt = sink_masks(y, p, K, mk)
            top = int(np.argmax(np.bincount(p[mk & (p != y)], minlength=K)))
            r.update({f'acc{tag}': mm['acc'], f'tv{tag}': mm['tv_to_true'], f'sink_share{tag}': mm['sink_pred_share'],
                      f'asym{tag}': em['asymmetry'], f'err_rank{tag}': em['err_rank'],
                      f'top_dest{tag}': run.classes[top],
                      f'top_dest_share{tag}': (p[mk & (p != y)] == top).mean() * 100})
        dom_rows.append(r)
    doms = pd.DataFrame(dom_rows)
    doms.to_csv(os.path.join(OUT, f'{ds}_domains.csv'), index=False)
    return dict(run=run, per_seed=per_seed, cf=cf, sinks=sinks, doms=doms, rank=rank, **keep)


# ---------------------------------------------------------------------------- figures
def _log2_ratio(x):
    return np.log2(np.clip(x, 2 ** -6, None))


def plot_marginals(res):
    """Per-class log2(predicted / true share): NoAdapt on x, Ramen on y. Below the diagonal = pulled down."""
    fig, axes = plt.subplots(1, len(res), figsize=(3.6 * len(res), 3.9), squeeze=False)
    for ax, (ds, r) in zip(axes[0], res.items()):
        sk = r['sinks'][r['sinks'].true_n > 0]
        x, yv = _log2_ratio(sk.ratio_noadapt.values), _log2_ratio(sk.ratio_ramen.values)
        lim = (-6.3, max(x.max(), yv.max()) + 0.5)
        ax.axhspan(np.log2(SINK), lim[1], color=GRID, alpha=0.5, lw=0)
        ax.axvspan(np.log2(SINK), lim[1], color=GRID, alpha=0.5, lw=0)
        ax.plot(lim, lim, color=MUTED, lw=1, ls='--')
        ax.scatter(x, yv, s=12 if len(sk) > 50 else 40, color=BLUE, alpha=0.55 if len(sk) > 50 else 0.9,
                   edgecolor='none')
        for i in np.argsort(-np.abs(x - yv))[:4]:
            ax.annotate(sk['class'].values[i][:18], (x[i], yv[i]), fontsize=7, color=TEXT_2,
                        xytext=(3, 3), textcoords='offset points')
        ax.set_xlim(lim), ax.set_ylim(lim)
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel('NoAdapt  log2(pred / true share)')
    axes[0, 0].set_ylabel('Ramen  log2(pred / true share)')
    fig.suptitle('Class prediction marginals: shaded = sink (> 2x), floor = never predicted', x=0.01, ha='left',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'marginals.png'), dpi=150)
    return fig


def plot_counterfactual(res):
    variants = ['bias: Ramen marginal (global)', 'bias: Ramen marginal (per domain)', 'Ramen',
                'bias: true marginal (per domain, oracle)']
    labels = ['bias = Ramen marginal, global', 'bias = Ramen marginal, per domain', 'Ramen',
              'bias = true marginal, per domain (oracle)']
    colors = [ORANGE, '#eda100', BLUE, MUTED]
    t = pd.concat([r['cf'] for r in res.values()]).groupby(['dataset', 'variant'])['gain'].agg(['mean', 'std'])
    dss = list(res)
    fig, ax = plt.subplots(figsize=(1.9 * len(dss) + 2, 4.2))
    w = 0.8 / len(variants)
    for j, (v, lab, c) in enumerate(zip(variants, labels, colors)):
        vals = [t.loc[(d, v), 'mean'] for d in dss]
        errs = [t.loc[(d, v), 'std'] for d in dss]
        xs = np.arange(len(dss)) + (j - (len(variants) - 1) / 2) * w
        ax.bar(xs, vals, width=w - 0.02, color=c, label=lab, yerr=errs, error_kw=dict(lw=1, ecolor=TEXT_2))
    ax.axhline(0, color=TEXT_2, lw=0.8)
    ax.set_xticks(np.arange(len(dss)), dss)
    ax.grid(axis='x', visible=False)
    ax.set_ylabel('accuracy gain over NoAdapt (pts)')
    ax.set_title('How much of Ramen\'s gain is a class-prior shift?', pad=34)
    _top_legend(ax, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'counterfactual.png'), dpi=150)
    return fig


def plot_locality(res):
    """CDF of the error's neighbour rank (0 = predicted class is the true class's nearest prototype)."""
    fig, axes = plt.subplots(1, len(res), figsize=(3.4 * len(res), 3.4), squeeze=False)
    for ax, (ds, r) in zip(axes[0], res.items()):
        run = r['run']
        for p, c, lab in ((run.pred0, ORANGE, 'NoAdapt'), (r['pred1'], BLUE, 'Ramen')):
            w = p != run.y
            v = np.sort(r['rank'][run.y[w], p[w]])
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=2, label=lab)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls='--', label='chance')
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel('neighbour rank of predicted class')
    axes[0, 0].set_ylabel('share of errors')
    axes[0, 0].legend(loc='lower right')
    fig.suptitle('Semantic locality of errors (prototype cosine)', x=0.01, ha='left', fontsize=12,
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'locality.png'), dpi=150)
    return fig


def plot_class_acc(res):
    fig, axes = plt.subplots(1, len(res), figsize=(3.4 * len(res), 3.6), squeeze=False)
    for ax, (ds, r) in zip(axes[0], res.items()):
        a0, a1 = r['a0'], r['a1']
        ax.plot([0, 100], [0, 100], color=MUTED, lw=1, ls='--')
        ax.scatter(a0, a1, s=10 if len(a0) > 50 else 40, color=BLUE, alpha=0.5 if len(a0) > 50 else 0.9,
                   edgecolor='none')
        ax.set_xlim(-2, 102), ax.set_ylim(-2, 102)
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel('NoAdapt class accuracy (%)')
    axes[0, 0].set_ylabel('Ramen class accuracy (%)')
    fig.suptitle('Per-class accuracy before and after Ramen', x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'class_acc.png'), dpi=150)
    return fig


def plot_delta_confusion(ds, r, top=24):
    """Row-normalised C0, C1 and C1 - C0. For many classes: the `top` classes with the largest |change|."""
    run = r['run']
    K = run.K
    n = np.bincount(run.y, minlength=K).clip(min=1)[:, None]
    R0 = confusion(run.y, run.pred0, K) / n * 100
    R1 = confusion(run.y, r['pred1'], K) / n * 100
    D = R1 - R0
    if K > top:
        involve = np.abs(D).sum(0) + np.abs(D).sum(1) - 2 * np.abs(np.diag(D))  # off-diagonal involvement
        idx = np.argsort(-involve)[:top]
        idx = idx[np.argsort(-np.abs(D[idx][:, idx]).sum(1))]  # order rows by size of change
        R0, R1, D = R0[np.ix_(idx, idx)], R1[np.ix_(idx, idx)], D[np.ix_(idx, idx)]
    else:
        idx = np.arange(K)
    names = [run.classes[i][:16] for i in idx]
    m = len(idx)
    panels = [(R0, 'NoAdapt', SEQ), (R1, 'Ramen', SEQ), (D, 'Ramen - NoAdapt', DIV)]
    if K > top:
        panels = panels[2:]  # 1000x1000 rows normalised: only the change is readable
    size = (0.32 * m * len(panels) + 2.5, 0.32 * m + 2.2) if K > top else (5.4 * len(panels), 5.2)
    fig, axes = plt.subplots(1, len(panels), figsize=size, squeeze=False)
    vmax_seq = max(R0.max(), R1.max())
    dmax = np.abs(D).max()
    for ax, (M, title, cmap) in zip(axes[0], panels):
        if cmap is DIV:
            im = ax.imshow(M, cmap=cmap, vmin=-dmax, vmax=dmax)
        else:
            im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax_seq)
        ax.set_xticks(range(m), names, rotation=90, fontsize=7 if m > 12 else 8)
        ax.set_yticks(range(m), names, fontsize=7 if m > 12 else 8)
        ax.grid(False)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('predicted')
        if m <= 12:
            for i in range(m):
                for j in range(m):
                    if abs(M[i, j]) >= 1:
                        ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=6.5,
                                color='white' if (cmap is not DIV and M[i, j] > 0.6 * vmax_seq) else TEXT_2)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='% of true class')
    axes[0, 0].set_ylabel('true')
    sub = '' if K <= top else f' ({m} classes with the largest change)'
    fig.suptitle(f'{ds}: confusion before / after Ramen{sub}  (blue = more, red = less)', x=0.01, ha='left',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f'delta_confusion_{ds}.png'), dpi=150)
    return fig


# ---------------------------------------------------------------------------- summary
def summarise(res):
    per_seed = pd.concat([r['per_seed'] for r in res.values()])
    per_seed.to_csv(os.path.join(OUT, 'per_seed.csv'), index=False)
    num = per_seed.drop(columns='seed').groupby('dataset', sort=False)
    summary = num.mean().T.round(2).astype(str) + ' ± ' + num.std().T.round(2).astype(str)
    summary.to_csv(os.path.join(OUT, 'summary.csv'))
    cf = pd.concat([r['cf'] for r in res.values()])
    cf_sum = cf.drop(columns='seed').groupby(['dataset', 'variant'], sort=False).mean().round(2)
    cf_sum.to_csv(os.path.join(OUT, 'counterfactual_summary.csv'))
    with pd.option_context('display.width', 250, 'display.max_rows', 200, 'display.max_columns', 20):
        print(summary)
        print(cf_sum)
        for ds, r in res.items():
            print(f'\n== {ds} domains (seed {r["per_seed"].seed.iloc[0]})')
            print(r['doms'].round(2).to_string(index=False))
            sk = r['sinks'][r['sinks'].true_n > 0]
            print('-- top NoAdapt sinks'); print(sk.head(8).round(2).to_string(index=False))
            print('-- top Ramen sinks'); print(sk.sort_values('excess_ramen', ascending=False).head(8).round(2)
                                               .to_string(index=False))
            print('-- most starved under Ramen'); print(sk.sort_values('excess_ramen').head(5).round(2)
                                                        .to_string(index=False))


def sink_error_fate(res):
    """Seed 0: what Ramen does with NoAdapt's errors that sit in a domain sink vs its other errors."""
    rows = []
    for ds, r in res.items():
        run, p1 = r['run'], r['pred1']
        y, p0 = run.y, run.pred0
        in_sink, _ = domain_sink_flags(y, p0, p1, run.dom, run.K)
        err = p0 != y
        for name, m in (('in a domain sink', err & in_sink), ('elsewhere', err & ~in_sink)):
            rows.append({'dataset': ds, 'NoAdapt error': name, '% of errors': m.sum() / err.sum() * 100,
                         'fixed': (p1[m] == y[m]).mean() * 100,
                         'moved to other wrong': ((p1[m] != y[m]) & (p1[m] != p0[m])).mean() * 100,
                         'unchanged': (p1[m] == p0[m]).mean() * 100})
    return pd.DataFrame(rows).set_index(['dataset', 'NoAdapt error']).round(1)


def before_after(res, metrics):
    """Notebook table: one row per dataset, NoAdapt / Ramen columns per metric, 'mean ± std' over seeds.
    metrics: {label: column stem}, the stem gets '0' (NoAdapt) or '1' (Ramen) appended."""
    out = {}
    for ds, r in res.items():
        p = r['per_seed']
        out[ds] = {(label, algo): f'{p[stem + tag].mean():.2f} ± {p[stem + tag].std():.2f}'
                   for label, stem in metrics.items() for tag, algo in (('0', 'NoAdapt'), ('1', 'Ramen'))}
    return pd.DataFrame(out).T


def mean_std(res, columns):
    """Notebook table: one row per dataset, 'mean ± std' over seeds for each per-seed column."""
    return pd.DataFrame({ds: {c: f'{r["per_seed"][c].mean():.2f} ± {r["per_seed"][c].std():.2f}' for c in columns}
                         for ds, r in res.items()}).T


def analyse(datasets):
    os.makedirs(FIG, exist_ok=True)
    res = {ds: analyse_dataset(ds) for ds in datasets}
    summarise(res)
    figs = [plot_marginals(res), plot_counterfactual(res), plot_locality(res), plot_class_acc(res)]
    figs += [plot_delta_confusion(ds, r) for ds, r in res.items()]
    for fig in figs:
        plt.close(fig)


if __name__ == '__main__':
    assert sys.argv[1] == 'analyse', 'usage: python confusion_shift.py analyse [DATASET ...]'
    analyse(sys.argv[2:] or DATASETS)
