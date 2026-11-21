"""
E3 analysis (PREREGISTRATION_E3.md): Ramen, the true-label oracle, E2's A/B (hide same zero-shot class) and the top-m
runs with matched random controls, seed 0, DomainNet and ImageNetC5K. Reads the harness caches only.

    python analysis_e3.py        # -> tables/e3_*.csv, prints the verdict per prediction
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from analysis_e2 import DatasetRuns, FLOOR, OUT  # noqa: E402
from experiment import ORACLE_RUN  # noqa: E402

DATASETS = ('DomainNet', 'ImageNetC5K')
RUNS = {'Ramen': 'Ramen', 'oracle': None, 'A': 'hide_samezs_wrong_true', 'B': 'hide_random_wrong_true',
        'top3': 'hide_top3_wrong_true', 'rand3': 'hide_rand3_wrong_true',
        'top10': 'hide_top10_wrong_true', 'rand10': 'hide_rand10_wrong_true'}
LABEL = {'Ramen': 'Ramen', 'oracle': 'true-label oracle', 'A': 'A: hide same zs class (m=1)',
         'B': 'B: random, matched to A', 'top3': 'top-3 zs classes', 'rand3': 'random, matched to top-3',
         'top10': 'top-10 zs classes', 'rand10': 'random, matched to top-10'}


def hidden_true_share(r, run):
    """Share of the hidden entries (zs-wrong queries) that were zero-shot-predicted as the query's TRUE class."""
    st = pd.read_pickle(r.exp._cache_path(r.names[run])[:-len('.pkl')] + '_stats.pkl')
    hid = torch.cat(st['hidden']).numpy()
    w = r.zs_wrong
    if 'hidden_true_class' not in st:  # E2 runs predate this diagnostic
        return hid[w].mean(), float('nan')
    ht = torch.cat(st['hidden_true_class']).numpy()
    return hid[w].mean(), (ht[w].sum() / max(hid[w].sum(), 1)) * 100


def main(load=None, out=OUT):
    os.makedirs(out, exist_ok=True)
    rows, doms, split, verdicts = [], [], [], []
    for ds in DATASETS:
        names = {k: (v if v else ORACLE_RUN[ds]) for k, v in RUNS.items()}
        names = {**names, 'hide_samezs_wrong_true': names['A']}  # DatasetRuns reads the zs-wrong gate from A's stats
        try:
            r = load(ds, names) if load else DatasetRuns(ds, names=names)
        except FileNotFoundError as e:
            print(f'{ds}: missing {os.path.basename(str(e))}, skipped', flush=True)
            continue
        ramen, oracle = r.acc('Ramen'), r.acc('oracle')
        loss = ramen - oracle
        for k in RUNS:
            row = {'dataset': ds, 'run': LABEL[k], 'acc': r.acc(k), 'vs Ramen': r.acc(k) - ramen,
                   'recovered %': (r.acc(k) - oracle) / loss * 100}
            if k in ('top3', 'top10', 'rand3', 'rand10', 'A', 'B'):
                row['hidden per zs-wrong query'], row['hidden entries zs-predicted as true class %'] = \
                    hidden_true_share(r, k)
            rows.append(row)
            for nm, m in (('zs wrong', r.zs_wrong), ('zs right', ~r.zs_wrong)):
                split.append({'dataset': ds, 'run': LABEL[k], 'queries': nm, 'n': int(m.sum()),
                              'correct %': r.acc(k, m), 'kept zero-shot %': r.kept_zs(k, m)})
            if ds == 'DomainNet':
                for dom, g in r.df[k].groupby('domain', observed=True):
                    doms.append({'run': LABEL[k], 'domain': dom, 'acc': g.correct.mean() * 100})
        print(f'\n=== {ds}: Ramen {ramen:.2f}, oracle {oracle:.2f} (loss {loss:.2f}), floor {FLOOR[ds]} ===')
        for m in (3, 10):
            t, c = f'top{m}', f'rand{m}'
            d_ctrl, d_a = r.acc(t) - r.acc(c), r.acc(t) - r.acc('A')
            lo, hi = r.paired_ci(t, c)
            lo2, hi2 = r.paired_ci(t, 'A')
            print(f'm={m}: top - random {d_ctrl:+.2f} [{lo:+.2f}, {hi:+.2f}]; top - A {d_a:+.2f} [{lo2:+.2f}, {hi2:+.2f}]; '
                  f'recovers {(r.acc(t) - oracle) / loss * 100:.0f}% (A {(r.acc("A") - oracle) / loss * 100:.0f}%)')
            verdicts.append({'dataset': ds, 'm': m, 'top - random': d_ctrl, 'top - A': d_a})
        if ds == 'DomainNet':
            q = {k: doms_q for k, doms_q in ((k, [d['acc'] for d in doms if d['run'] == LABEL[k] and d['domain'] == 'quickdraw'][0])
                                             for k in RUNS)}
            print(f"quickdraw: Ramen {q['Ramen']:.2f}, oracle {q['oracle']:.2f}, A {q['A']:.2f}, "
                  f"top3 {q['top3']:.2f}, top10 {q['top10']:.2f}  (controls {q['rand3']:.2f} / {q['rand10']:.2f})")
            v = pd.DataFrame(verdicts)
            v = v[v.dataset == ds]
            p1 = bool(((v['top - A'] > FLOOR[ds]) & (v['top - random'] > FLOOR[ds])).any())
            best = max(q['top3'], q['top10'])
            print(f"P1 (some m: top - A > {FLOOR[ds]} and top - random > {FLOOR[ds]}): {'PASS' if p1 else 'FAIL'}")
            print(f"P2 (best top-m quickdraw >= 14.3): {best:.2f} -> {'PASS' if best >= 14.3 else 'FAIL'}")
            print(f"P3 (quickdraw A <= top3 <= top10): {q['A']:.2f} / {q['top3']:.2f} / {q['top10']:.2f} -> "
                  f"{'PASS' if q['A'] <= q['top3'] <= q['top10'] else 'FAIL'}")
            within = bool((v['top - A'].abs() <= FLOOR[ds]).all()) and best <= 13.3
            print(f"Falsifier (all top - A within the floor and quickdraw <= 13.3): {'TRIGGERED' if within else 'not triggered'}")
        else:
            v = pd.DataFrame(verdicts)
            v = v[v.dataset == ds]
            print(f"P4 (top-m - random > 0 for every m): {'PASS' if (v['top - random'] > 0).all() else 'FAIL'}")
    t, sp = pd.DataFrame(rows), pd.DataFrame(split)
    t.to_csv(os.path.join(out, 'e3_summary.csv'), index=False)
    sp.to_csv(os.path.join(out, 'e3_split.csv'), index=False)
    pd.set_option('display.width', 230)
    print('\n' + t.round(2).to_string(index=False))
    if doms:
        dm = pd.DataFrame(doms).pivot(index='domain', columns='run', values='acc')[[LABEL[k] for k in RUNS]]
        dm.to_csv(os.path.join(out, 'e3_domains_DomainNet.csv'))
        print('\nDomainNet per domain:\n' + dm.round(2).to_string())
    print('\n' + sp[sp.queries == 'zs wrong'].round(1).to_string(index=False))
    print('E3 ANALYSIS DONE', flush=True)


if __name__ == '__main__':
    main()
