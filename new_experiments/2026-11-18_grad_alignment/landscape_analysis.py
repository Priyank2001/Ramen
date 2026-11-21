"""
Tables over the loss-landscape records of LandscapeExperiment.landscape_records() (pure numpy / pandas, no GPU).

    from landscape_analysis import Landscapes
    L = Landscapes(records, text_feature)
    L.per_image()             # one row per chosen image: CE change along Ramen / oracle / random step, boundary distances
    L.group_summary()         # medians and counts per outcome group
    L.line_profiles()         # median CE change along each 1D line, per group (for the line plot)
    L.examples()              # one typical image per group (closest to the group's median CE change)
    L.embedding_slice(rec)    # the embedding-space slice of one image (landscape.embedding_landscape)

All CE values are the true-label cross-entropy -log p(y) of the image under the given parameters (ORACLE: uses y).
"""
import numpy as np
import pandas as pd

from ga_experiment import OUTCOMES
from landscape import GRID_A, GRID_B, LINE_T, embedding_landscape

DIRECTIONS = {'ramen': "Ramen's step", 'oracle': 'oracle step (sign of g_ce)', 'random': 'random sign step'}
T1 = int(np.flatnonzero(np.isclose(LINE_T, 1.0))[0])
T0 = int(np.flatnonzero(np.isclose(LINE_T, 0.0))[0])


def count_pct(n, total):
    return f'{n:,} of {total:,} ({n / total:.1%})' if total else '0 of 0'


def first_flip(pred, start):
    """Smallest t > 0 on LINE_T where the prediction differs from `start`; inf if it never does."""
    pos = LINE_T > 0
    idx = np.flatnonzero(pos & (pred != start))
    return float(LINE_T[idx[0]]) if len(idx) else np.inf


class Landscapes:

    def __init__(self, records, text_feature):
        self.records = records
        self.W = text_feature

    def per_image(self):
        rows = []
        for r in self.records:
            row = dict(position=r['position'], outcome=r['outcome'], label=r['label'], zs_pred=r['zs_pred'],
                       ramen_pred=r['ramen_pred'], ce_zs=float(r['line_ramen_ce'][T0]),
                       oracle_along=r['oracle_xy'][0], oracle_perp=r['oracle_xy'][1], cos_step=r['cos_step'],
                       cos_oracle_ramen=r['cos_oracle_ramen'], step_norm=r['step_norm'])
            for d in DIRECTIONS:
                ce = r[f'line_{d}_ce']
                row[f'dce_{d}'] = float(ce[T1] - ce[T0])                       # CE change after one step
                row[f'flip_{d}'] = first_flip(r[f'line_{d}_pred'], r['zs_pred'])   # step multiple to change class
            z0, z1 = r['z0'], r['z1']
            p0 = np.exp(100 * z0 @ self.W.T - (100 * z0 @ self.W.T).max())
            p0 /= p0.sum()
            ascent = self.W[r['label']] - p0 @ self.W                            # true-class ascent of the head at z0
            dz = z1 - z0
            row['emb_move'] = float(np.linalg.norm(dz))
            row['emb_cos_true'] = float(dz @ ascent / (np.linalg.norm(dz) * np.linalg.norm(ascent) + 1e-12))
            rows.append(row)
        df = pd.DataFrame(rows)
        df['outcome'] = pd.Categorical(df['outcome'], OUTCOMES)
        return df

    def group_summary(self):
        df = self.per_image()

        def row(d):
            n = len(d)
            out = {'images': n}
            for k, name in DIRECTIONS.items():
                out[f'median CE change, {name}'] = d[f'dce_{k}'].median()
            for k, name in DIRECTIONS.items():
                out[f'CE lowered by {name}'] = count_pct(int((d[f'dce_{k}'] < 0).sum()), n)
            out["median cos(Ramen's step, oracle step)"] = d['cos_oracle_ramen'].median()
            out['median embedding move ||z1 - z0||'] = d['emb_move'].median()
            out['median cos(z1 - z0, true-class ascent of the head)'] = d['emb_cos_true'].median()
            return pd.Series(out)
        return df.groupby('outcome', observed=True)[list(df.columns)].apply(row)

    def line_profiles(self):
        """{outcome: {direction: (median, q25, q75) arrays over LINE_T of CE(t) - CE(0)}}."""
        out = {}
        for g in OUTCOMES:
            recs = [r for r in self.records if r['outcome'] == g]
            if not recs:
                continue
            out[g] = {}
            for d in DIRECTIONS:
                M = np.stack([r[f'line_{d}_ce'] - r[f'line_{d}_ce'][T0] for r in recs])
                out[g][d] = (np.median(M, 0), np.quantile(M, 0.25, 0), np.quantile(M, 0.75, 0), len(recs))
        return out

    def examples(self):
        """One typical image per outcome group: the one whose CE change along Ramen's step is closest to the median."""
        df = self.per_image()
        by_pos = {r['position']: r for r in self.records}
        out = {}
        for g, d in df.groupby('outcome', observed=True):
            i = (d['dce_ramen'] - d['dce_ramen'].median()).abs().idxmin()
            out[g] = by_pos[int(d.loc[i, 'position'])]
        return out

    def embedding_slice(self, rec):
        return embedding_landscape(rec['z0'], rec['z1'], self.W, rec['label'])


__all__ = ['Landscapes', 'DIRECTIONS', 'GRID_A', 'GRID_B', 'LINE_T']
