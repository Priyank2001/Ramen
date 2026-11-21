"""
Plots over ImageRecords / ConfusionShift. Each function draws one finding and returns its axes.
"""
import matplotlib.pyplot as plt
import numpy as np

from records import OUTCOMES

OUTCOME_COLORS = {'right -> right': '#c9c8c3', 'fixed': '#1baf7a', 'broken': '#e34948',
                  'wrong -> same wrong': '#8a8984', 'wrong -> other wrong': '#eda100'}
SOURCE_COLORS = {'zero_shot': '#8a8984', 'ramen': '#2a78d6'}


def plot_outcomes(outcome_table, ax=None, title='Zero-shot → Ramen, per domain'):
    """Stacked horizontal bars: share of each domain's images per outcome, counts in the hover-free labels."""
    t = outcome_table[[o for o in OUTCOMES if o in outcome_table.columns]]
    share = t.div(t.sum(1), axis=0) * 100
    ax = ax or plt.subplots(figsize=(9, 0.45 * len(t) + 1.4))[1]
    y = np.arange(len(t))[::-1]
    left = np.zeros(len(t))
    for o in share.columns:
        ax.barh(y, share[o], left=left, color=OUTCOME_COLORS[o], label=o, height=0.7, edgecolor='white')
        for yi, l, w, n in zip(y, left, share[o], t[o]):
            if w >= 4 and o != 'right -> right':
                ax.text(l + w / 2, yi, f'{n:,}', ha='center', va='center', fontsize=7)
        left += share[o].to_numpy()
    ax.set_yticks(y, [f'{d} ({n:,})' for d, n in zip(t.index, t.sum(1))])
    ax.set_xlim(0, 100)
    ax.set_xlabel('% of images (labels = image counts)')
    ax.set_title(title, loc='left')
    ax.legend(ncol=5, frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.04))
    return ax


def plot_pair_changes(pair_changes, names=('zero_shot', 'ramen'), ax=None, title=None):
    """Dumbbell: count of each (true -> predicted) pair under both sources."""
    a, b = names
    t = pair_changes
    ax = ax or plt.subplots(figsize=(8, 0.35 * len(t) + 1.2))[1]
    y = np.arange(len(t))[::-1]
    ax.hlines(y, t[a], t[b], color='#c9c8c3', lw=2)
    ax.plot(t[a], y, 'o', color=SOURCE_COLORS.get(a, 'k'), label=a, ms=7, mec='white')
    ax.plot(t[b], y, 'o', color=SOURCE_COLORS.get(b, 'k'), label=b, ms=7, mec='white')
    ax.set_yticks(y, [f'{tr} → {pr}' for tr, pr in zip(t['true'], t['predicted'])], fontsize=8)
    ax.set_xlabel('images (true → predicted)')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0))
    ax.set_title(title or 'Confused pairs that change most', loc='left', pad=22)
    return ax


def plot_class_report(shift, cls, ax=None):
    """TP / FN / FP of one class under both sources, as grouped bars with counts."""
    c = shift.before._idx(cls)
    stats = ('tp', 'fn', 'fp')
    ax = ax or plt.subplots(figsize=(6, 2.8))[1]
    x = np.arange(len(stats))
    for k, (name, cm) in enumerate(zip(shift.names, (shift.before, shift.after))):
        v = [int(getattr(cm, s)[c]) for s in stats]
        bars = ax.bar(x + (k - 0.5) * 0.38, v, width=0.38, color=SOURCE_COLORS.get(name, 'k'), label=name)
        ax.bar_label(bars, [f'{n:,}' for n in v], fontsize=8)
    support = int(shift.before.counts[c].sum())
    ax.set_xticks(x, ['TP', 'FN (it → other)', 'FP (other → it)'])
    ax.set_ylabel('images')
    ax.set_title(f'{shift.before.classes[c]}: {support:,} images of the class', loc='left')
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_embedding_shift(records, ax=None):
    """Distribution of the embedding shift (1 − cos before/after) per outcome."""
    t = records.table
    ax = ax or plt.subplots(figsize=(8, 3))[1]
    present = [o for o in OUTCOMES if (t['outcome'] == o).any()]
    data = [t.loc[t['outcome'] == o, 'emb_shift'].to_numpy() for o in present]
    bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=0.6)
    for patch, o in zip(bp['boxes'], present):
        patch.set_facecolor(OUTCOME_COLORS[o])
    ax.set_yticks(np.arange(1, len(present) + 1), [f'{o} ({len(d):,})' for o, d in zip(present, data)])
    ax.set_xlabel('embedding shift, 1 − cos(before, after)')
    ax.set_title('How far one adaptation step moves the image embedding', loc='left')
    ax.grid(axis='x', alpha=0.3)
    return ax


def plot_embedding_moves(records, classes, max_points=400, seed=0, ax=None, text_features=None):
    """
    PCA of the embeddings of images whose TRUE class is in `classes`: an arrow from before to after adaptation
    for each image, coloured by outcome. PCA is fitted on the before and after embeddings together.
    """
    m = records.mask(true=classes)
    idx = np.flatnonzero(m)
    if len(idx) > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_points, replace=False))
    before = np.asarray(records.embeddings('emb_before')[idx], np.float32)
    after = np.asarray(records.embeddings('emb_after')[idx], np.float32)
    X = np.concatenate([before, after])
    mu = X.mean(0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    P = vt[:2].T
    b2, a2 = (before - mu) @ P, (after - mu) @ P
    outcome = records.table['outcome'].to_numpy()[idx]
    label = records.table['label'].to_numpy()[idx]
    ax = ax or plt.subplots(figsize=(7, 6))[1]
    shapes_ = 'osD^v<>p'
    if len(classes) > len(shapes_):
        raise ValueError(f'at most {len(shapes_)} classes (one marker shape each), got {len(classes)}')
    markers = dict(zip([records.class_index(c) for c in classes], shapes_))
    for o in OUTCOMES:
        k = outcome == o
        if not k.any():
            continue
        ax.quiver(b2[k, 0], b2[k, 1], a2[k, 0] - b2[k, 0], a2[k, 1] - b2[k, 1], angles='xy', scale_units='xy',
                  scale=1, width=0.002, color=OUTCOME_COLORS[o], alpha=0.8)
        for c, mk in markers.items():
            kc = k & (label == c)
            ax.scatter(b2[kc, 0], b2[kc, 1], s=10, marker=mk, color=OUTCOME_COLORS[o])
        ax.scatter([], [], s=10, marker='o', color=OUTCOME_COLORS[o], label=f'{o} ({int(k.sum()):,})')  # legend only
    both = np.concatenate([b2, a2])
    pad = 0.05 * (both.max(0) - both.min(0))
    ax.set_xlim(both[:, 0].min() - pad[0], both[:, 0].max() + pad[0])  # quiver does not autoscale to arrow heads
    ax.set_ylim(both[:, 1].min() - pad[1], both[:, 1].max() + pad[1])
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    shapes = ', '.join(f'{m_} {records.classes[c]}' for c, m_ in markers.items())
    ax.set_title(f'Embedding before (marker) → after (arrow head)\n{len(idx):,} images, true class: {shapes}',
                 loc='left', fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return ax
