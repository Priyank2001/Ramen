"""
What kind of embedding shift happens when Ramen's prediction ends up wrong?

CLIP's classifier is linear in the image embedding: logit(c) = 100 * z . w_c, with w_c the L2-normalised text
feature of class c (src/models/ModelForBySampleTTA.py). One adaptation step moves the embedding from z_before
("emb_before") to z_after ("emb_after"); the change in any class's logit is therefore EXACTLY
100 * (z_after - z_before) . w_c -- no approximation, a linear identity.

For every image we pick one "from" class f and one "to" class t (see `_from_to` below) and project the raw
shift delta_z = z_after - z_before onto the unit axis u = normalise(w_t - w_f):

    proj            = delta_z . u                      (signed; + means "moved from f toward t")
    delta_margin    = 100 * proj = (logit_t - logit_f) after  -  (logit_t - logit_f) before   (exact)
    explained_frac  = proj^2 / ||delta_z||^2            (how much of the *squared* shift lies on this one axis)

explained_frac close to 1: the shift is a clean, targeted move along the specific class-pair axis. Close to 0:
the shift is large but mostly orthogonal to this axis -- something else moved the embedding, and the flip (if
any) rode on top of it.

Destination coherence: for a class c that many wrong images end up predicted as, do those images' raw shifts
point in a shared direction (a real attractor in embedding space), or does each move its own way (idiosyncratic,
driven by whichever memory entries that particular image retrieved)? Measured as the mean resultant length

    R_c = || mean_i( delta_z_i / ||delta_z_i|| ) ||    in [0, 1]

of the unit shift directions of images landing at c, compared with a bootstrap null built from random images
(same group size) that do not share a destination.

Pure numpy/pandas over already-recorded data (2026-10-28_confusion_records: ImageRecords) and the cached CLIP
text features (2026-11-01_pillow_sink: TextFeatures). No GPU, no new runs, nothing here changes Ramen.

    sd = ShiftDecomposition(records, text_features)
    sd.table                               # one row per non-trivial-outcome image (see columns below)
    sd.coherence_by_destination(min_count=50)
    sd.coherence_null(n=1000, trials=200)
"""
import numpy as np
import pandas as pd

# outcomes this analysis covers: every image whose prediction changed, or stayed wrong (excludes 'right -> right',
# which has no (from, to) pair to measure a shift against)
OUTCOMES = ('fixed', 'broken', 'wrong -> same wrong', 'wrong -> other wrong')


def _from_to(table):
    """
    The (from, to) class pair whose text-embedding axis each image's shift is measured against.

    For every outcome except 'wrong -> same wrong', zs_pred == pred is false, so the natural pair is the
    zero-shot class the image moved away from and the class Ramen ends with: (zs_pred, pred).
    'wrong -> same wrong' has zs_pred == pred (the prediction never changes), so that pair is degenerate
    (zero-length axis); there we ask instead whether the shift pushes further from the true class toward the
    (unchanged) wrong prediction: (label, pred).
    """
    same_wrong = table['outcome'].astype(str).to_numpy() == 'wrong -> same wrong'
    f = np.where(same_wrong, table['label'].to_numpy(), table['zs_pred'].to_numpy())
    t = table['pred'].to_numpy()
    return f.astype(int), t.astype(int)


class ShiftDecomposition:
    """Per-image shift, decomposed along each image's own (from, to) class-text axis."""

    def __init__(self, records, text_features):
        self.records = records
        self.W = np.asarray(text_features, dtype=np.float32)   # (K, D), L2-normalised (encode_text)
        self.table = self._build_table()

    def _build_table(self):
        r = self.records
        t = r.table
        mask = t['outcome'].isin(OUTCOMES).to_numpy()
        idx = np.flatnonzero(mask)
        sub = t.iloc[idx].reset_index(drop=True)
        f, to = _from_to(sub)

        before = np.asarray(r.embeddings('emb_before')[idx], np.float32)
        after = np.asarray(r.embeddings('emb_after')[idx], np.float32)
        dz = after - before
        shift_norm = np.linalg.norm(dz, axis=1)

        axis = self.W[to] - self.W[f]
        axis_norm = np.linalg.norm(axis, axis=1)
        axis_unit = axis / np.clip(axis_norm, 1e-12, None)[:, None]
        proj = (dz * axis_unit).sum(1)
        explained_frac = np.where(shift_norm > 0, (proj / np.clip(shift_norm, 1e-12, None)) ** 2, 0.0)

        names = np.asarray(r.classes, dtype=object)
        out = sub[['domain', 'label', 'true_class', 'zs_pred', 'zs_class', 'pred', 'pred_class',
                    'outcome', 'emb_shift']].copy()
        out['row'] = idx
        out['from_idx'], out['to_idx'] = f, to
        out['from_class'], out['to_class'] = names[f], names[to]
        out['shift_norm'] = shift_norm
        out['proj'] = proj
        out['explained_frac'] = explained_frac
        out['delta_margin'] = 100.0 * proj
        return out

    # ------------------------------------------------------------------ summaries
    def summary_by_outcome(self):
        """Median shift size, how much of it sits on the (from, to) axis, and the margin it buys, per outcome."""
        g = self.table.groupby('outcome', observed=True)
        return pd.DataFrame({
            'images': g.size(),
            'median |shift|': g['shift_norm'].median(),
            'median explained_frac': g['explained_frac'].median(),
            'median delta_margin': g['delta_margin'].median(),
            'delta_margin > 0': g['delta_margin'].apply(lambda s: (s > 0).mean() * 100),
        }).round(4)

    def coherence_by_destination(self, min_count=50, max_classes=None):
        """
        For every class many images wrongly end up at, the mean resultant length R of their raw unit shift
        directions (sharing-a-destination coherence) and the mean cosine with each image's own (from, to) axis
        (how much of that shared pull is explained by the destination's text-embedding direction).
        """
        t = self.table
        wrong = t[t['pred'] != t['label']]
        rows = []
        for c, g in wrong.groupby('to_class', observed=True):
            if len(g) < min_count:
                continue
            idx = g['row'].to_numpy()
            dz = self._dz(idx)
            unit = dz / np.clip(np.linalg.norm(dz, axis=1, keepdims=True), 1e-12, None)
            R = float(np.linalg.norm(unit.mean(0)))
            axis = self.W[g['to_idx'].to_numpy()] - self.W[g['from_idx'].to_numpy()]
            axis_unit = axis / np.clip(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12, None)
            semantic_cos = float((unit * axis_unit).sum(1).mean())
            rows.append({'class': c, 'images': len(g), 'R': R, 'semantic_cos': semantic_cos})
        out = pd.DataFrame(rows).sort_values('R', ascending=False).reset_index(drop=True)
        return out.head(max_classes) if max_classes else out

    def coherence_null(self, n, trials=200, seed=0, pool='wrong'):
        """Bootstrap null for R at group size n: random images, not sharing a destination."""
        rng = np.random.default_rng(seed)
        t = self.table
        pool_rows = (t[t['pred'] != t['label']] if pool == 'wrong' else t)['row'].to_numpy()
        out = np.empty(trials, dtype=np.float32)
        for k in range(trials):
            idx = rng.choice(pool_rows, n, replace=False)
            dz = self._dz(idx)
            unit = dz / np.clip(np.linalg.norm(dz, axis=1, keepdims=True), 1e-12, None)
            out[k] = np.linalg.norm(unit.mean(0))
        return out

    def _dz(self, idx):
        before = np.asarray(self.records.embeddings('emb_before')[idx], np.float32)
        after = np.asarray(self.records.embeddings('emb_after')[idx], np.float32)
        return after - before
