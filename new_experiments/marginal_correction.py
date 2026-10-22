"""
Direction 1 (next_directions.md): stop Ramen's class over-correction with an output-level prior correction.

Ramen files every sample under its zero-shot prediction and caches gradients taken at the base weights, so its adapted
predictions never feed back into the cache. A correction applied to its output therefore changes nothing upstream and
can be evaluated exactly from the saved per-sample probabilities, without re-running the stream.

Correction, batch by batch in stream order:
    pi_t   = m * pi_{t-1} + (1 - m) * marginal(batch t)     EMA of the method's predicted class marginal, pi_0 = target
    logit' = log p - lam * log(pi_t / target)              target = uniform
lam = 0 is the uncorrected method. momentum=None uses the whole-stream marginal instead (non-causal reference).

    exp = CIFAR10CExperiment().load(); exp.evaluate(capture_probs=True)
    sweep(exp)                          # Ramen: accuracy, flips and class-share gap over lam x momentum
    sweep(exp, name='NoAdapt')          # control: how much of Ramen's gain is debiasing alone?
    add_corrected(exp, lam=1.0, momentum=0.99)   # registers the variant in exp.results / exp.probs for other plots
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ramen_analysis import (CATEGORICAL, COLORS, NOISE_CORRUPTIONS, TEXT_2, MUTED, _top_legend,
                            expected_calibration_error)

KEYS = ['domain_idx', 'sample_idx']
LAMS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
MOMENTUMS = (0.9, 0.99, 0.999, None)


def _marginal(p, estimate):
    """Predicted class marginal of a block of softmax rows: mean probability ('soft') or argmax counts ('hard')."""
    if estimate == 'soft':
        return p.mean(0)
    return np.bincount(p.argmax(1), minlength=p.shape[1]) / len(p)


def corrected_probs(exp, name='Ramen', lam=1.0, momentum=0.99, estimate='soft', include_current=True):
    """
    Prior-corrected softmax rows for one method, row i <-> exp.results[name].iloc[i].
    include_current: the batch's own predictions enter pi before it is corrected (the batch is available at test
    time); False uses only earlier batches. In 'single' mode pi restarts with each corruption, like the method does.
    """
    if name not in exp.probs:
        raise RuntimeError(f'No stored probabilities for {name}; run exp.evaluate(capture_probs=True).')
    r = exp.results[name]
    p = exp.probs[name].astype(np.float64)
    k = p.shape[1]
    target = np.full(k, 1 / k)
    floor = 0.01 / k  # a class that disappears from the estimate gets a large but finite boost
    logp = np.log(np.clip(p, 1e-30, None))

    if lam == 0:
        return p.astype(np.float32)
    if momentum is None:
        shift = np.log(np.maximum(_marginal(p, estimate), floor) / target)
        out = logp - lam * shift
    else:
        out = np.empty_like(logp)
        batch, dom = r.batch.values, r.domain_idx.values
        starts = np.flatnonzero(np.r_[True, batch[1:] != batch[:-1]])
        ends = np.r_[starts[1:], len(batch)]
        pi = target.copy()
        for s, e in zip(starts, ends):
            if exp.tta_mode == 'single' and s > 0 and dom[s] != dom[s - 1]:
                pi = target.copy()
            cur = _marginal(p[s:e], estimate)
            if include_current:
                pi = momentum * pi + (1 - momentum) * cur
            out[s:e] = logp[s:e] - lam * np.log(np.maximum(pi, floor) / target)
            if not include_current:
                pi = momentum * pi + (1 - momentum) * cur

    out -= out.max(1, keepdims=True)
    q = np.exp(out)
    return (q / q.sum(1, keepdims=True)).astype(np.float32)


def corrected_frame(exp, q, name='Ramen'):
    """exp.results[name] with pred / conf / correct / true_prob / entropy recomputed from the softmax rows q."""
    d = exp.results[name].copy()
    pred = q.argmax(1)
    d['pred'] = pred
    d['conf'] = q[np.arange(len(q)), pred]
    d['true_prob'] = q[np.arange(len(q)), d.label.values]
    d['entropy'] = -(q * np.log(np.clip(q, 1e-12, None))).sum(1)
    d['correct'] = d.label.values == pred
    return d


def metrics(exp, d, base='Ramen'):
    """
    Scalar metrics for one (possibly corrected) results frame, as in exp.summary_metrics, plus how it differs
    from the uncorrected `base` run: delta avg and the % of samples it fixes / breaks relative to base.
    """
    acc = d.groupby('domain_idx').correct.mean() * 100
    names = [exp.environments[i] for i in acc.index]
    is_noise = np.array([n in NOISE_CORRUPTIONS for n in names])
    base_acc = exp.results[base].groupby('domain_idx').correct.mean() * 100
    m = {'avg': acc.mean(), 'delta vs ' + base: acc.mean() - base_acc.mean(),
         'noise': acc[is_noise].mean(), 'other': acc[~is_noise].mean()}

    zs = d[KEYS + ['correct']].merge(exp.results['NoAdapt'][KEYS + ['correct']], on=KEYS, suffixes=('', '_zs'))
    m['fixed % (vs NoAdapt)'] = (zs.correct & ~zs.correct_zs).mean() * 100
    m['broken % (vs NoAdapt)'] = (~zs.correct & zs.correct_zs).mean() * 100
    b = exp.results[base].correct.values  # same rows as d: corrected frames are copies of a base run
    if len(b) == len(d):
        m[f'fixed % (vs {base})'] = (d.correct.values & ~b).mean() * 100
        m[f'broken % (vs {base})'] = (~d.correct.values & b).mean() * 100

    k = len(exp.classes)
    true = np.bincount(d.label, minlength=k) / len(d)
    pred = np.bincount(d.pred, minlength=k) / len(d)
    m['TV(pred, true)'] = 0.5 * np.abs(pred - true).sum()
    for c in exp.watch_classes:
        i = exp.classes.index(c)
        m[f'{c} overshoot %'] = (pred[i] / true[i] - 1) * 100
    m['ECE'] = expected_calibration_error(d)
    return pd.Series(m)


def sweep(exp, name='Ramen', lams=LAMS, momentums=MOMENTUMS, estimate='soft', include_current=True):
    """Metrics for every (momentum, lam); momentum None = whole-stream marginal. lam = 0 rows are the plain method."""
    rows = {}
    for mom in momentums:
        for lam in lams:
            q = corrected_probs(exp, name, lam, mom, estimate, include_current)
            rows[('stream' if mom is None else mom, lam)] = metrics(exp, corrected_frame(exp, q, name), base=name)
    t = pd.DataFrame(rows).T
    t.index.names = ['momentum', 'lam']
    return t.round(3)


def add_corrected(exp, name='Ramen', lam=1.0, momentum=0.99, estimate='soft', include_current=True, new_name=None):
    """Store one corrected variant in exp.results / exp.probs so the experiment's other tables and plots can use it."""
    new_name = new_name or f'{name}+pc(lam={lam:g}, m={momentum})'
    q = corrected_probs(exp, name, lam, momentum, estimate, include_current)
    exp.results[new_name] = corrected_frame(exp, q, name)
    exp.probs[new_name] = q
    return new_name


# ---------------------------------------------------------------------------- plots
def plot_sweep(exp, table, name='Ramen'):
    """Average accuracy and class-share gap (TV) against lam, one line per momentum. Two panels, never a shared axis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    moms = list(dict.fromkeys(table.index.get_level_values('momentum')))
    for ax, col, label in [(axes[0], 'avg', 'average accuracy (%)'), (axes[1], 'TV(pred, true)', 'TV(pred, true)')]:
        for i, mom in enumerate(moms):
            t = table.loc[mom]
            ax.plot(t.index, t[col], color=CATEGORICAL[i], lw=2, marker='o', ms=6,
                    label='whole-stream marginal' if mom == 'stream' else f'EMA m={mom}')
        ax.axhline(table.loc[moms[0]][col].iloc[0], color=MUTED, lw=1, ls='--')
        ax.set_xlabel('lam (strength of the correction)')
        ax.set_ylabel(label)
        _top_legend(ax, ncol=2)
    axes[0].set_title(f'{exp.dataset_name}: {name} + prior correction, accuracy', pad=46)
    axes[1].set_title('class-share gap (lower = closer to the true marginal)', pad=46)
    axes[0].text(0.99, 0.02, f'dashed = {name} uncorrected', transform=axes[0].transAxes, ha='right',
                 fontsize=8, color=TEXT_2)
    fig.tight_layout()
    return fig


def plot_class_shares(exp, names, top_k=20):
    """
    Predicted share - true share per class for each run. For > top_k classes, the top_k by largest |bias| over any
    of the runs, so both zero-shot's and Ramen's own overshoots show. Sorted by the bias of the last run.
    """
    k = len(exp.classes)
    any_df = exp.results[names[0]]
    true = np.bincount(any_df.label, minlength=k) / len(any_df) * 100
    bias = pd.DataFrame({n: np.bincount(exp.results[n].pred, minlength=k) / len(exp.results[n]) * 100 - true
                         for n in names}, index=exp.classes)
    if k > top_k:
        bias = bias.loc[bias.abs().max(axis=1).nlargest(top_k).index]
    bias = bias.sort_values(names[-1], ascending=False)

    # NoAdapt / Ramen keep their usual colours; variants take the next unused categorical hues in order
    spare = iter(c for c in CATEGORICAL if c not in COLORS.values())
    colors = {n: COLORS[n] if n in COLORS else next(spare) for n in names}
    y = np.arange(len(bias))
    h = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(bias) * max(1, len(names) / 1.5) + 1.6))
    for i, n in enumerate(names):
        ax.barh(y + (i - (len(names) - 1) / 2) * h, bias[n], height=h * 0.92, color=colors[n], label=n)
    ax.axvline(0, color=TEXT_2, lw=1)
    ax.set_yticks(y, bias.index)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('predicted share - true share (pts)   [> 0: over-predicted]')
    shown = f', top {top_k} by largest |bias|' if k > top_k else ''
    ax.set_title(f'{exp.dataset_name}: class prediction bias{shown}', pad=24 + 14 * ((len(names) - 1) // 2))
    _top_legend(ax, ncol=2)
    fig.tight_layout()
    return fig
