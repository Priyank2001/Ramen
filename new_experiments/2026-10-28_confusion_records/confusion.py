"""
Class confusion of one prediction source, and how it changes between two sources (zero-shot -> Ramen).

    cm = ConfusionMatrix.from_records(records, 'ramen')        # or 'zero_shot'; domain='quickdraw' to restrict
    cm.per_class()                 # support, predicted, TP, FP, FN, TN, precision, recall, F1 per class
    cm.report('dog')               # TP / FP / FN / TN of one class, as counts of N images
    cm.predicted_as('dog')         # where the dogs go: the classes images of dog are predicted as
    cm.mistaken_for('cat')         # what gets called cat: the true classes of images predicted as cat
    cm.top_pairs(20)               # the largest off-diagonal cells (true -> predicted)

    shift = ConfusionShift(ConfusionMatrix.from_records(records, 'zero_shot'), cm)
    shift.pair_changes(20)         # pairs whose count changed most from zero-shot to Ramen
    shift.per_class()              # TP / FP / FN of each class before and after

Counts are rows = true class, columns = predicted class. For one class c, over all N images:
  TP = true c, predicted c       FN = true c, predicted something else
  FP = true other, predicted c   TN = true other, predicted other (N - TP - FN - FP)
"""
import numpy as np
import pandas as pd


class ConfusionMatrix:
    """K x K counts of (true class, predicted class) over a set of images."""

    def __init__(self, y_true, y_pred, classes):
        self.classes = list(classes)
        K = len(self.classes)
        y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
        self.counts = np.bincount(y_true * K + y_pred, minlength=K * K).reshape(K, K)
        self.n = int(self.counts.sum())

    @classmethod
    def from_records(cls, records, source='ramen', **filters):
        """
        Confusion of `source` over the images matching records.mask(**filters) (e.g. domain='sketch');
        pred= and correct= filters refer to `source`'s predictions.
        """
        m = records.mask(source=source, **filters) if filters else slice(None)
        return cls(records.table['label'].to_numpy()[m], records.predictions(source)[m], records.classes)

    def _idx(self, cls):
        return int(cls) if isinstance(cls, (int, np.integer)) else self.classes.index(cls)

    # ------------------------------------------------------------------ per class
    @property
    def tp(self):
        return np.diag(self.counts)

    @property
    def fn(self):
        return self.counts.sum(1) - self.tp

    @property
    def fp(self):
        return self.counts.sum(0) - self.tp

    @property
    def tn(self):
        return self.n - self.tp - self.fn - self.fp

    def per_class(self):
        """One row per class: support (true count), predicted count, TP, FP, FN, TN, precision, recall, F1 (%)."""
        support, predicted = self.counts.sum(1), self.counts.sum(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.where(predicted > 0, self.tp / predicted, np.nan)
            recall = np.where(support > 0, self.tp / support, np.nan)
            # F1 = 2TP / (2TP + FP + FN): 0 (not NaN) for a class that is never predicted or never right
            f1 = np.where(support > 0, 2 * self.tp / (2 * self.tp + self.fp + self.fn), np.nan)
        return pd.DataFrame({'support': support, 'predicted': predicted, 'TP': self.tp, 'FP': self.fp,
                             'FN': self.fn, 'TN': self.tn, 'precision %': precision * 100,
                             'recall %': recall * 100, 'F1 %': f1 * 100},
                            index=pd.Index(self.classes, name='class')).round(2)

    def report(self, cls):
        """TP / FP / FN / TN of one class, each as 'count of N (%)'."""
        c = self._idx(cls)
        support, predicted = int(self.counts[c].sum()), int(self.counts[:, c].sum())
        cells = {'TP (true and predicted)': (self.tp[c], support, 'of its images'),
                 'FN (true, predicted other)': (self.fn[c], support, 'of its images'),
                 'FP (other, predicted as it)': (self.fp[c], predicted, 'of predictions'),
                 'TN (other, predicted other)': (self.tn[c], self.n - support, 'of other images')}
        rows = {k: {'count': int(v), 'of': int(d), 'meaning': w, '%': round(100 * v / d, 2) if d else np.nan}
                for k, (v, d, w) in cells.items()}
        return pd.DataFrame(rows).T.rename_axis(self.classes[c])

    # ------------------------------------------------------------------ who goes where
    def predicted_as(self, cls, k=10):
        """Where images of true class `cls` go: the k most common wrong predictions (count, % of its images)."""
        c = self._idx(cls)
        return self._ranked(self.counts[c], exclude=c, total=self.counts[c].sum(), k=k,
                            name=f'images of {self.classes[c]} predicted as', pct='% of its images')

    def mistaken_for(self, cls, k=10):
        """What gets called `cls`: the k most common true classes of images wrongly predicted as it."""
        c = self._idx(cls)
        return self._ranked(self.counts[:, c], exclude=c, total=self.counts[:, c].sum(), k=k,
                            name=f'true class of images predicted as {self.classes[c]}', pct='% of its predictions')

    def _ranked(self, counts, exclude, total, k, name, pct):
        counts = counts.copy()
        counts[exclude] = 0
        idx = [i for i in np.argsort(-counts, kind='stable')[:k] if counts[i] > 0]
        return pd.DataFrame({'count': counts[idx], pct: (100 * counts[idx] / max(total, 1)).round(2)},
                            index=pd.Index([self.classes[i] for i in idx], name=name))

    def pair(self, true, pred):
        return int(self.counts[self._idx(true), self._idx(pred)])

    def top_pairs(self, k=20):
        """The k largest off-diagonal cells: true -> predicted, count, % of the true class's images."""
        off = self.counts.copy()
        np.fill_diagonal(off, 0)
        flat = np.argsort(-off, axis=None, kind='stable')[:k]
        t, p = np.unravel_index(flat, off.shape)
        keep = off[t, p] > 0
        t, p = t[keep], p[keep]
        support = self.counts.sum(1)
        return pd.DataFrame({'true': [self.classes[i] for i in t], 'predicted': [self.classes[i] for i in p],
                             'count': off[t, p], '% of true class': (100 * off[t, p] / support[t]).round(2)})

    def accuracy(self):
        return self.tp.sum() / max(self.n, 1) * 100

    def to_frame(self):
        """The full matrix as a DataFrame (readable for small label spaces only)."""
        return pd.DataFrame(self.counts, index=pd.Index(self.classes, name='true'),
                            columns=pd.Index(self.classes, name='predicted'))


class ConfusionShift:
    """How confusion changes from `before` to `after` (two sources over the same images)."""

    def __init__(self, before, after, names=('zero_shot', 'ramen')):
        if before.classes != after.classes or before.n != after.n:
            raise ValueError('the two matrices must cover the same images and classes')
        self.before, self.after, self.names = before, after, names
        self.delta = after.counts - before.counts

    def per_class(self):
        """TP / FP / FN of each class under both sources, and the change (after - before)."""
        a, b = self.names
        t = pd.DataFrame({'support': self.before.counts.sum(1)}, index=pd.Index(self.before.classes, name='class'))
        for stat in ('TP', 'FP', 'FN'):
            x, y = getattr(self.before, stat.lower()), getattr(self.after, stat.lower())
            t[f'{stat} {a}'], t[f'{stat} {b}'], t[f'Δ{stat}'] = x, y, y - x
        return t

    def pair_changes(self, k=20, direction='both'):
        """
        Off-diagonal pairs whose count changed most. direction: 'grew' (new or larger confusions),
        'shrank' (confusions the second source removes) or 'both' (largest |change|).
        """
        d = self.delta.copy()
        np.fill_diagonal(d, 0)
        score = {'grew': -d, 'shrank': d, 'both': -np.abs(d)}[direction]
        flat = np.argsort(score, axis=None, kind='stable')[:k]
        t, p = np.unravel_index(flat, d.shape)
        keep = d[t, p] != 0
        t, p = t[keep], p[keep]
        a, b = self.names
        names = self.before.classes
        return pd.DataFrame({'true': [names[i] for i in t], 'predicted': [names[i] for i in p],
                             a: self.before.counts[t, p], b: self.after.counts[t, p], 'change': d[t, p]})
