"""
E1: why true-label filing hurts even when every entry holds a TRUE-LABEL value (ORACLE runs labeloracle_zs_ce vs
labeloracle_true_ce, DomainNet seed 0: 58.50 vs 56.54, i.e. -1.96 for true-label filing).

With label values an entry no longer pushes toward its zero-shot class, so the containment explanation
(../2026-11-18_true_label_mechanism/) cannot cover this loss. Earlier replay (../2026-11-01_label_oracle/mechanism.py)
found the query's own true class gets ~2.5% of the weight under either filing. This script measures what that
replay did not: the pull toward the query's WRONG zero-shot class, and which queue the right and wrong pulls come
from.

Direction of an entry j retrieved for query i (label-value runs, RamenLabelValues):
  - j from an earlier batch: its value is the CE gradient toward its TRUE label y_j      -> pushes toward y_j
  - j from the query's own batch (incl. itself): still its entropy gradient           -> pushes toward zs_j
Retrieval does not depend on values (weights e^(-H_j) e^(-beta d_ij) from zero-shot entropy and features), so the
cache replay (../2026-11-01_pillow_sink/cache_replay.py, read-only import) gives exactly the retrieved entries.

  P_true(i)  = weight share of entries pushing toward the query's true class y_i
  P_wrong(i) = weight share of entries pushing toward the query's zero-shot class zs_i (zs-wrong queries: != y_i)
each split by source queue: the query's zero-shot-class queue, its true-class queue, all other queues.

Hypotheses (pre-registered 2026-11-21, before running this script). Zero-shot-WRONG queries:
  L1 "lookup lost"  Under zero-shot filing the query's zs-class queue holds images CLIP confused the same way, and
                    their labels point to what such images truly are, often y_i. P_true from the zs-class queue is
                    larger under zero-shot filing than under true-label filing.
  L2 "wrong pull"   Under true-label filing the zs-class queue holds true members of the wrong class, whose label
                    values pull toward zs_i. P_wrong (mainly from the zs-class queue) is larger under true-label
                    filing.
  L3                Net pull P_true - P_wrong is larger under zero-shot filing, and per query it is associated with
                    the label-value run getting the image right.
Reported for zs-right queries too (for them y_i == zs_i, so only P_true is meaningful).

    CUDA_VISIBLE_DEVICES= python label_value_pull.py DomainNet 0 --per-domain 1000     # ~5 min CPU
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), '2026-11-01_pillow_sink'))  # read-only import
from cache_replay import CacheReplayRun, TrueLabelFiling, ZeroShotFiling  # noqa: E402
from sink_runs import SinkRuns  # noqa: E402

FILING = {'Ramen cache': 'zero-shot filing', 'oracle cache': 'true-label filing'}
LABEL_RUNS = {'zero-shot filing': 'labeloracle_zs_ce', 'true-label filing': 'labeloracle_true_ce'}  # ORACLE values
SOURCES = ['zs-class queue', 'true-class queue', 'other queues']


def sample_rows(table, per_domain, seed):
    """Same sample as ../2026-11-18_true_label_mechanism/containment.py (1,000 per domain, rng seed = run seed)."""
    rng = np.random.default_rng(seed)
    rows = [rng.choice(np.flatnonzero(table.domain_idx.to_numpy() == d), size=per_domain, replace=False)
            for d in sorted(table.domain_idx.unique())]
    return np.sort(np.concatenate(rows))


class LabelValuePull:
    """Per (query, filing): right / wrong pull of a label-value update, by source queue."""

    def __init__(self, runs, batch_size):
        self.runs, self.B = runs, batch_size
        t = runs.table
        self.label, self.zs = t.label.to_numpy(), t.zs_pred.to_numpy()

    def entry_direction(self, row, e):
        """Class each retrieved entry pushes toward: true label if revealed (earlier batch), else zero-shot class."""
        revealed = (e['id'].to_numpy() // self.B) < (row // self.B)
        return np.where(revealed, e['true'].to_numpy(), e['zs_pred'].to_numpy())

    def measure(self, row, cache, e):
        y, zs = self.label[row], self.zs[row]
        d = self.entry_direction(row, e)
        w, q = e['weight'].to_numpy(), e['queue'].to_numpy()
        src = np.where(q == zs, SOURCES[0], np.where(q == y, SOURCES[1], SOURCES[2]))
        out = {'row': row, 'filing': FILING[cache], 'zs right': y == zs,
               'P_true': w[d == y].sum(), 'P_wrong': w[d == zs].sum() if y != zs else np.nan,
               'zs-queue weight': w[q == zs].sum(),
               'self weight': w[e['self'].to_numpy()].sum(),
               # the query's own entry (entropy value, pushes to zs_i) sits in the zs queue or the true-class queue
               # depending on the filing; without it, the comparison is about OTHER images only
               'P_wrong excl. self': w[(d == zs) & ~e['self'].to_numpy()].sum() if y != zs else np.nan,
               # inside the zs-class queue: what its retrieved entries are, by true label
               'zs-queue entries truly y': w[(q == zs) & (e['true'].to_numpy() == y)].sum(),
               'zs-queue entries truly zs': w[(q == zs) & (e['true'].to_numpy() == zs)].sum()}
        for s in SOURCES:
            out[f'P_true from {s}'] = w[(d == y) & (src == s)].sum()
            out[f'P_wrong from {s}'] = w[(d == zs) & (src == s)].sum() if y != zs else np.nan
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset'), ap.add_argument('seed', type=int)
    ap.add_argument('--per-domain', type=int, default=1000)
    a = ap.parse_args()
    runs = SinkRuns(a.dataset, a.seed, device='cpu')
    t = runs.table
    rows = sample_rows(t, a.per_domain, a.seed)

    # label-value run outcomes, aligned on the stream (asserted)
    for filing, name in LABEL_RUNS.items():
        df = pd.read_pickle(runs.exp._cache_path(name))
        assert (df[['domain_idx', 'sample_idx']].to_numpy() == t[['domain_idx', 'sample_idx']].to_numpy()).all(), name
        t[f'pred {filing}'] = df.pred.to_numpy()

    pull = LabelValuePull(runs, runs.exp.batch_size)
    res = CacheReplayRun(runs, [ZeroShotFiling(), TrueLabelFiling()]).run(rows, keep_rows=rows)
    per = pd.DataFrame([pull.measure(r, c, e) for (r, c), e in res.retrieved.items()])
    per = per.join(t[['domain', 'label']], on='row')
    per['pred'] = [t.at[r, f'pred {f}'] for r, f in zip(per.row, per.filing)]
    per['correct'] = per.pred == per.label
    per['net'] = per.P_true - per.P_wrong.fillna(0)
    out = os.path.join(HERE, 'results')
    os.makedirs(out, exist_ok=True)
    per.to_pickle(os.path.join(out, f'{a.dataset}_s{a.seed}_label_value_pull.pkl'))

    pd.set_option('display.width', 250)
    pct = lambda x: (x * 100).round(2)  # noqa: E731
    wrong = per[~per['zs right']]
    n_w = wrong.row.nunique()
    print(f'{len(rows):,} sampled queries, {n_w:,} zero-shot wrong ({n_w / len(rows):.1%})')

    print('\n== L1/L2: zs-WRONG queries, mean weight share (%) by source queue ==')
    cols = [f'P_true from {s}' for s in SOURCES] + ['P_true'] + [f'P_wrong from {s}' for s in SOURCES] + [
        'P_wrong', 'self weight', 'P_wrong excl. self', 'net']
    print(pct(wrong.groupby('filing')[cols].mean()).T.to_string())

    print('\n== inside the zs-class queue (zs-wrong queries): weight on entries truly y_i vs truly zs_i (%) ==')
    print(pct(wrong.groupby('filing')[['zs-queue weight', 'zs-queue entries truly y', 'zs-queue entries truly zs']]
              .mean()).to_string())

    print('\n== by domain, zs-wrong: P_true / P_wrong / net (%) ==')
    print(pct(wrong.pivot_table(index='domain', columns='filing', values=['P_true', 'P_wrong', 'net'],
                                aggfunc='mean', observed=True)).to_string())

    print('\n== zs-RIGHT queries: P_true (%) by source ==')
    right = per[per['zs right']]
    print(pct(right.groupby('filing')[[f'P_true from {s}' for s in SOURCES] + ['P_true']].mean()).T.to_string())

    print('\n== L3: zs-wrong queries, label-value run correct (%) by quintile of net pull, per filing ==')
    for f, g in wrong.groupby('filing'):
        g = g.assign(q=pd.qcut(g.net, 5, labels=False, duplicates='drop'))
        print(f'-- {f}')
        print(g.groupby('q').agg(n=('row', 'size'), net=('net', 'mean'), correct=('correct', 'mean'))
              .assign(net=lambda d: pct(d.net), correct=lambda d: pct(d.correct)).to_string())

    print('\n== outcomes on the sample: correct (%) by zs right/wrong and filing ==')
    print(pct(per.groupby(['zs right', 'filing']).correct.mean()).unstack().to_string())

    print('\n== paired, zs-wrong: change in net (true - zs filing) vs who gets it right ==')
    p = wrong.pivot_table(index='row', columns='filing', values=['net', 'correct'], aggfunc='first')
    d = p['net']['true-label filing'] - p['net']['zero-shot filing']
    p = pd.DataFrame({'d_net': d, 'zs_ok': p['correct']['zero-shot filing'].astype(bool),
                      'true_ok': p['correct']['true-label filing'].astype(bool)})
    p['q'] = pd.qcut(p.d_net, 5, labels=False, duplicates='drop')
    print(p.groupby('q').agg(n=('d_net', 'size'), d_net=('d_net', 'mean'), zs_filing_ok=('zs_ok', 'mean'),
                             true_filing_ok=('true_ok', 'mean'))
          .assign(d_net=lambda x: pct(x.d_net), zs_filing_ok=lambda x: pct(x.zs_filing_ok),
                  true_filing_ok=lambda x: pct(x.true_filing_ok)).to_string())


if __name__ == '__main__':
    main()
