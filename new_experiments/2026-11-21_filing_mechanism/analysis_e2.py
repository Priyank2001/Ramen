"""
E2 analysis (PREREGISTRATION.md): Ramen, the true-label oracle, and runs A-D, seed 0, on DomainNet / ImageNetC5K /
EuroSAT. Reads the harness caches only (no GPU).

    python analysis_e2.py        # -> tables/e2_{summary,split,domains}.csv, prints the verdict per prediction
Accuracy is domain-averaged (mean of per-domain accuracies). zs right/wrong uses the zero-shot prediction logged
inside run A (Ramen-internal, the one that decides filing and hiding).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from active_seeds import cluster_bootstrap  # noqa: E402
from experiment import DATASETS, E2_VARIANTS, ORACLE_RUN, make_local_experiment  # noqa: E402

FLOOR = {'DomainNet': 0.11, 'ImageNetC5K': 0.53, 'EuroSAT': 1.51}  # 2 x Ramen range over seeds 0-4
LABELS = {'Ramen': 'Ramen', 'oracle': 'true-label oracle', 'hide_samezs_wrong_true': 'A: oracle, hide same-zs (zs-wrong)',
          'hide_random_wrong_true': 'B: oracle, hide random (zs-wrong)', 'hide_samezs_all_true': 'C: oracle, hide same-zs (all)',
          'hide_samezs_all_zs': 'D: Ramen, hide same-zs (all)'}
OUT = os.path.join(HERE, 'tables')


class DatasetRuns:
    """The six runs of one dataset (seed 0), aligned on the stream (asserted)."""

    def __init__(self, dataset, n_per_domain=None, names=None):
        """names: key -> cached run name (default: the real runs; tests pass smoke-subset runs)."""
        self.dataset = dataset
        self.exp = make_local_experiment(dataset, seed=0, n_per_domain=n_per_domain)
        self.names = None  # set below
        names = names or {'Ramen': 'Ramen', 'oracle': ORACLE_RUN[dataset], **{v: v for v in E2_VARIANTS}}
        self.names = names
        self.df = {}
        for key, name in names.items():
            path = self.exp._cache_path(name)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            self.df[key] = pd.read_pickle(path)
        ref = self.df['Ramen']
        for key, d in self.df.items():
            assert (d[['domain_idx', 'sample_idx']].to_numpy() == ref[['domain_idx', 'sample_idx']].to_numpy()).all(), key
        st = pd.read_pickle(self.exp._cache_path(names['hide_samezs_wrong_true'])[:-len('.pkl')] + '_stats.pkl')
        self.zs = torch.cat(st['zs_pred']).numpy()
        self.label = ref.label.to_numpy()
        self.zs_wrong = self.zs != self.label
        self.domain = ref.domain_idx.to_numpy()
        # bootstrap clusters: base photo on ImageNet-C (one photo under 15 corruptions), single image otherwise
        self.cluster = ref.sample_idx.to_numpy() if dataset == 'ImageNetC5K' else self.domain * 10**7 + ref.sample_idx.to_numpy()

    def acc(self, key, mask=None):
        d = self.df[key]
        c = d.correct.to_numpy().astype(float)
        if mask is None:
            return pd.Series(c).groupby(self.domain).mean().mean() * 100
        return c[mask].mean() * 100

    def paired_ci(self, a, b):
        diff = self.df[a].correct.to_numpy().astype(float) - self.df[b].correct.to_numpy()
        return cluster_bootstrap(diff, self.domain, self.cluster)

    def kept_zs(self, key, mask):
        return (self.df[key].pred.to_numpy()[mask] == self.zs[mask]).mean() * 100


def main(load=DatasetRuns, out=OUT):
    os.makedirs(out, exist_ok=True)
    summ, split, doms = [], [], []
    for ds in DATASETS:
        try:
            r = load(ds)
        except FileNotFoundError as e:
            print(f'{ds}: missing {os.path.basename(str(e))}, skipped', flush=True)
            continue
        ramen, oracle = r.acc('Ramen'), r.acc('oracle')
        for key in LABELS:
            row = {'dataset': ds, 'run': LABELS[key], 'acc': r.acc(key), 'vs Ramen': r.acc(key) - ramen,
                   'vs oracle': r.acc(key) - oracle}
            if key not in ('Ramen', 'oracle'):
                lo, hi = r.paired_ci(key, 'oracle')
                row.update({'vs oracle CI low': lo, 'vs oracle CI high': hi})
            summ.append(row)
            for name, m in (('zs wrong', r.zs_wrong), ('zs right', ~r.zs_wrong)):
                split.append({'dataset': ds, 'run': LABELS[key], 'queries': name, 'n': int(m.sum()),
                              'correct %': r.acc(key, m), 'kept zero-shot %': r.kept_zs(key, m)})
            if ds == 'DomainNet':
                d = r.df[key]
                for dom, g in d.groupby('domain', observed=True):
                    doms.append({'run': LABELS[key], 'domain': dom, 'acc': g.correct.mean() * 100})
        a_b = r.acc('hide_samezs_wrong_true') - r.acc('hide_random_wrong_true')
        lo, hi = r.paired_ci('hide_samezs_wrong_true', 'hide_random_wrong_true')
        loss = ramen - oracle
        recovered = (r.acc('hide_samezs_wrong_true') - oracle) / loss * 100 if loss > 0 else np.nan
        summ.append({'dataset': ds, 'run': 'A - B (causal contrast)', 'vs Ramen': np.nan, 'acc': a_b,
                     'vs oracle CI low': lo, 'vs oracle CI high': hi})
        print(f'\n=== {ds}: Ramen {ramen:.2f}, oracle {oracle:.2f} (loss {loss:.2f}), floor {FLOOR[ds]} ===')
        print(f'A recovers {recovered:.0f}% of the oracle loss; A - B = {a_b:+.2f} [{lo:+.2f}, {hi:+.2f}]')
        if ds == 'DomainNet':
            p1 = r.acc('hide_samezs_wrong_true') >= ramen - 0.75 and a_b > FLOOR[ds]
            print(f"P1 (A >= Ramen - 0.75 and A - B > {FLOOR[ds]}): {'PASS' if p1 else 'FAIL'}")
        else:
            print(f"P1 direction (A - B > 0): {'PASS' if a_b > 0 else 'FAIL'}")
        m = r.zs_wrong
        kz = {k: r.kept_zs(k, m) for k in ('Ramen', 'oracle', 'hide_samezs_wrong_true')}
        p2 = abs(kz['hide_samezs_wrong_true'] - kz['Ramen']) < abs(kz['hide_samezs_wrong_true'] - kz['oracle'])
        print(f"P2 kept zs on zs-wrong: Ramen {kz['Ramen']:.1f}, oracle {kz['oracle']:.1f}, A "
              f"{kz['hide_samezs_wrong_true']:.1f} -> {'PASS' if p2 else 'FAIL'}")
        d_ramen = r.acc('hide_samezs_all_zs') - ramen
        print(f"P3 D vs Ramen {d_ramen:+.2f} ({'within' if abs(d_ramen) <= FLOOR[ds] else 'outside'} the floor)")
    s, sp = pd.DataFrame(summ), pd.DataFrame(split)
    s.to_csv(os.path.join(out, 'e2_summary.csv'), index=False)
    sp.to_csv(os.path.join(out, 'e2_split.csv'), index=False)
    pd.set_option('display.width', 220)
    print('\n' + s.round(2).to_string(index=False))
    print('\n' + sp.round(1).to_string(index=False))
    if doms:
        dm = pd.DataFrame(doms).pivot(index='domain', columns='run', values='acc')
        dm.to_csv(os.path.join(out, 'e2_domains_DomainNet.csv'))
        print('\nDomainNet per domain:\n' + dm[[LABELS[k] for k in LABELS]].round(2).to_string())
    print('E2 ANALYSIS DONE', flush=True)


if __name__ == '__main__':
    main()
