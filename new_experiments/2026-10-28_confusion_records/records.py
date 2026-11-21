"""
Per-image records of one recorded run (src/methods/RamenRecorder.py): the harness's per-sample table plus, for
every image, the zero-shot and Ramen probability vectors and the CLIP embedding before and after adaptation.

    records = ImageRecords.load(prefix, classes)   # prefix = the run's cache path without '.pkl'
    records.table                                  # one row per image (see TABLE COLUMNS below)
    records[i]                                     # ImageRecord: everything about image i, vectors included
    records.where(true='dog', pred='cat')          # rows of images of dogs that Ramen calls cats

Table columns (one row per image, in stream order):
  domain, domain_idx, sample_idx, position, label, true_class
  zs_pred, zs_class, zs_conf, zs_true_prob     zero-shot prediction, its probability, probability of the true class
  pred, pred_class, conf, true_prob            Ramen's prediction (after adaptation), same
  zs_correct, correct
  outcome      zero-shot -> Ramen: 'right -> right', 'fixed', 'broken', 'wrong -> same wrong', 'wrong -> other wrong'
  emb_shift    1 - cosine(embedding before, embedding after): how far one adaptation step moved the image

Extending: a new prediction source (e.g. a post-hoc smoothed output) is added with `with_source(name, pred, probs)`; ConfusionMatrix and the gallery take any source name.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# prediction source -> (prediction column, probability-vector array)
SOURCES = {'zero_shot': ('zs_pred', 'zs_probs'), 'ramen': ('pred', 'probs')}
EMBEDDINGS = ('emb_before', 'emb_after')
OUTCOMES = ('right -> right', 'fixed', 'broken', 'wrong -> same wrong', 'wrong -> other wrong')


@dataclass(frozen=True)
class ImageRecord:
    """Everything recorded about one test image."""
    index: int          # row in ImageRecords.table
    domain: str
    domain_idx: int
    sample_idx: int
    position: int       # arrival order in the test stream
    label: int
    true_class: str
    zs_pred: int
    zs_class: str
    pred: int
    pred_class: str
    outcome: str
    emb_shift: float
    zs_probs: np.ndarray
    probs: np.ndarray
    emb_before: np.ndarray
    emb_after: np.ndarray

    def top(self, source='ramen', k=5, classes=None):
        """The k most probable classes under `source`, as (class, probability) pairs."""
        p = np.asarray(self.zs_probs if source == 'zero_shot' else self.probs, dtype=np.float32)
        idx = np.argsort(-p)[:k]
        return [(classes[i] if classes is not None else int(i), float(p[i])) for i in idx]


class ImageRecords:
    """The per-image table and the per-image vectors of one run, kept row-aligned."""

    def __init__(self, table, arrays, classes, sources=None):
        n = len(table)
        bad = {k: len(v) for k, v in arrays.items() if len(v) != n}
        if bad:
            raise ValueError(f'arrays not aligned with the {n:,}-row table: {bad}')
        self.classes = list(classes)
        self.arrays = arrays          # name -> (n, ...) array, row i <-> table row i
        self.sources = dict(sources or SOURCES)  # prediction source -> (prediction column, probability array)
        self.table = table.reset_index(drop=True)

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, prefix, classes, mmap=True):
        """
        Read a recorded run. mmap=True leaves the vectors on disk until used (full DomainNet: ~2 GB of vectors).
        """
        from methods.RamenRecorder import RecorderMixin  # src/ is on the path once ramen_analysis is imported
        table = pd.read_pickle(f'{prefix}.pkl')
        arrays = {k: np.load(p, mmap_mode='r' if mmap else None)
                  for k, p in RecorderMixin.record_paths(prefix).items()}
        return cls(cls._derive(table, arrays, classes), arrays, classes)

    @staticmethod
    def _derive(table, arrays, classes):
        """Add the zero-shot columns, class names, the zero-shot -> Ramen outcome and the embedding shift."""
        t = table.copy()
        names = np.asarray(classes, dtype=object)
        label = t['label'].to_numpy()
        rows = np.arange(len(t))
        t['zs_pred'] = np.asarray(arrays['zs_pred'])
        zs = arrays['zs_probs']
        t['zs_conf'] = np.asarray(zs[rows, t['zs_pred'].to_numpy()], dtype=np.float32)
        t['zs_true_prob'] = np.asarray(zs[rows, label], dtype=np.float32)
        t['zs_correct'] = t['zs_pred'] == t['label']
        for col, idx in [('true_class', 'label'), ('zs_class', 'zs_pred'), ('pred_class', 'pred')]:
            t[col] = pd.Categorical(names[t[idx].to_numpy()], categories=classes)
        t['outcome'] = pd.Categorical(_outcome(t['zs_correct'], t['correct'], t['zs_pred'] == t['pred']),
                                      categories=OUTCOMES)
        t['emb_shift'] = _cosine_distance(arrays['emb_before'], arrays['emb_after'])
        return t

    def subset(self, rows):
        """A new ImageRecords on the given rows (boolean mask or positions); vectors are read into memory."""
        rows = np.flatnonzero(rows) if np.asarray(rows).dtype == bool else np.asarray(rows)
        return ImageRecords(self.table.iloc[rows], {k: np.asarray(v[rows]) for k, v in self.arrays.items()},
                            self.classes, self.sources)

    # ------------------------------------------------------------------ access
    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        r = self.table.iloc[i]
        return ImageRecord(
            index=int(i), domain=str(r['domain']), domain_idx=int(r['domain_idx']), sample_idx=int(r['sample_idx']),
            position=int(r['position']), label=int(r['label']), true_class=str(r['true_class']),
            zs_pred=int(r['zs_pred']), zs_class=str(r['zs_class']), pred=int(r['pred']),
            pred_class=str(r['pred_class']), outcome=str(r['outcome']), emb_shift=float(r['emb_shift']),
            **{k: np.asarray(self.arrays[k][i]) for k in ('zs_probs', 'probs', *EMBEDDINGS)})

    def class_index(self, cls):
        """Class name or index -> index."""
        if isinstance(cls, (int, np.integer)):
            return int(cls)
        try:
            return self.classes.index(cls)
        except ValueError:
            close = [c for c in self.classes if str(cls).lower() in c.lower()][:10]
            raise KeyError(f'no class {cls!r}' + (f'; did you mean one of {close}?' if close else '')) from None

    def predictions(self, source):
        return self.table[self.sources[source][0]].to_numpy()

    def probs(self, source):
        return self.arrays[self.sources[source][1]]

    def embeddings(self, which='emb_after'):
        return self.arrays[which]

    def mask(self, true=None, pred=None, source='ramen', correct=None, outcome=None, domain=None):
        """
        Boolean row mask. true / pred: class name or index (or a list of them), pred under `source`;
        correct: True / False keeps images `source` gets right / wrong; outcome: one of OUTCOMES (or a list);
        domain: domain name (or a list).
        """
        t = self.table
        m = np.ones(len(t), dtype=bool)
        if true is not None:
            m &= t['label'].isin([self.class_index(c) for c in _as_list(true)]).to_numpy()
        if pred is not None:
            m &= np.isin(self.predictions(source), [self.class_index(c) for c in _as_list(pred)])
        if correct is not None:
            right = self.predictions(source) == t['label'].to_numpy()
            m &= right if correct else ~right
        if outcome is not None:
            unknown = set(_as_list(outcome)) - set(OUTCOMES)
            if unknown:
                raise ValueError(f'unknown outcome {unknown}; choose from {OUTCOMES}')
            m &= t['outcome'].isin(_as_list(outcome)).to_numpy()
        if domain is not None:
            m &= t['domain'].isin(_as_list(domain)).to_numpy()
        return m

    def where(self, **filters):
        """Rows of the table matching mask(**filters)."""
        return self.table[self.mask(**filters)]

    def with_source(self, name, pred, probs):
        """A new ImageRecords with one more prediction source (e.g. a post-hoc output), usable everywhere."""
        col, arr = f'{name}_pred', f'{name}_probs'
        table = self.table.assign(**{col: np.asarray(pred)})
        return ImageRecords(table, {**self.arrays, arr: probs}, self.classes, {**self.sources, name: (col, arr)})


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple, set, np.ndarray, pd.Index)) else [x]


def _outcome(zs_right, right, same_pred):
    return np.select([zs_right & right, ~zs_right & right, zs_right & ~right, same_pred],
                     ['right -> right', 'fixed', 'broken', 'wrong -> same wrong'], 'wrong -> other wrong')


def _cosine_distance(a, b, chunk=65_536):
    """1 - cos(a_i, b_i) per row, in chunks (the arrays may be memory-mapped float16)."""
    out = np.empty(len(a), dtype=np.float32)
    for s in range(0, len(a), chunk):
        x, y = np.asarray(a[s:s + chunk], np.float32), np.asarray(b[s:s + chunk], np.float32)
        out[s:s + chunk] = 1 - (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    return out
