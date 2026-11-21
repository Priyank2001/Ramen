"""
Analysis of the stack NIQE vector (lambda 0.1) + RamenActive prob T3 on ImageNetC5K, seeds 0-4 (PREREGISTRATION.md).

    python analysis.py            # -> tables/{per_seed,summary,per_corruption}.csv, prints the verdict
Reads the harness caches in ../results/ only (no GPU). Seeds whose stack run is missing are skipped.

Every accuracy is domain-averaged (mean of the 15 per-corruption accuracies), the project's headline convention.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from active_seeds import NUM_WORKERS, cluster_bootstrap  # noqa: E402
from quality_active_set import _acc, _load, support_domain_purity  # noqa: E402
from ramen_analysis import make_experiment  # noqa: E402

DATASET = 'ImageNetC5K'
SEEDS = range(5)
OUT = os.path.join(HERE, 'tables')
STACK = 'nss_niqe_white_l0.1_probT3'
# display name -> cached run name; Ramen is the reference for every gain
RUNS = {'Ramen': 'Ramen', 'T3': 'active_prob_T3', 'NIQE': 'nss_niqe_white_l0.1', 'NIQE + T3': STACK}


class SeedRuns:
    """The four cached runs of one seed, aligned on the stream (asserted), with their per-sample frames and stats."""

    def __init__(self, seed):
        self.seed = seed
        self.exp = make_experiment(DATASET, tta_mode='mixed', batch_size=100, n_per_domain=None, seed=seed,
                                   num_workers=NUM_WORKERS[DATASET])
        self.df, self.stats = {}, {}
        for label, name in RUNS.items():
            if not os.path.exists(self.exp._cache_path(name)):
                raise FileNotFoundError(name)
            self.df[label], self.stats[label] = _load(self.exp, name)
        ref = self.df['Ramen']
        for label, d in self.df.items():
            assert (d.sample_idx.values == ref.sample_idx.values).all(), f'seed {seed} {label}: stream order differs'
        self.domain = ref.domain_idx.values
        self.cluster = ref.sample_idx.values  # one base photo appears under all 15 corruptions
        self.corruptions = list(ref.domain.cat.categories)

    def acc(self, label):
        return _acc(self.df[label])

    def paired(self, a, b):
        """Gain of run a over run b (points), its photo-clustered 95% CI, and fixed / broken counts."""
        ca, cb = self.df[a].correct.values.astype(bool), self.df[b].correct.values.astype(bool)
        lo, hi = cluster_bootstrap(ca.astype(float) - cb, self.domain, self.cluster)
        return {'gain': self.acc(a) - self.acc(b), 'CI low': lo, 'CI high': hi,
                'fixed': int((ca & ~cb).sum()), 'broken': int((~ca & cb).sum()), 'n': len(ca)}

    def same_corruption_share(self, label):
        """Weight share (%) of the logged top-8 active-set entries that come from the query's corruption."""
        return support_domain_purity(self.stats[label], self.domain, self.domain)

    def per_corruption(self, label):
        return self.df[label].groupby('domain_idx').correct.mean().values * 100


def per_seed_table():
    rows, corr_rows, ramen_accs = [], [], []
    for seed in SEEDS:
        try:
            s = SeedRuns(seed)
        except FileNotFoundError as e:
            print(f'seed {seed}: {e} not cached yet, skipped', flush=True)
            continue
        ramen_accs.append(s.acc('Ramen'))
        for label in RUNS:
            row = {'seed': seed, 'run': label, 'acc': s.acc(label),
                   'same-corruption top8 %': s.same_corruption_share(label) if label != 'Ramen' else np.nan}
            if label != 'Ramen':
                row.update({f'{k} vs Ramen': v for k, v in s.paired(label, 'Ramen').items()})
            if label == 'NIQE + T3':
                row.update({f'{k} vs T3': v for k, v in s.paired(label, 'T3').items()})
            rows.append(row)
            for c, a, r in zip(s.corruptions, s.per_corruption(label), s.per_corruption('Ramen')):
                corr_rows.append({'seed': seed, 'run': label, 'corruption': c, 'acc': a, 'gain vs Ramen': a - r})
        print(f'seed {seed} analysed', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(corr_rows)


def noise_floor():
    """2 x (max - min) of Ramen's accuracy over seeds 0-4 (project rule), from all five cached Ramen runs."""
    accs = [_acc(_load(make_experiment(DATASET, tta_mode='mixed', batch_size=100, n_per_domain=None, seed=s,
                                       num_workers=NUM_WORKERS[DATASET]), 'Ramen')[0]) for s in SEEDS]
    return 2 * (max(accs) - min(accs)), accs


def summary(t, floor):
    out = []
    for label in [r for r in RUNS if r != 'Ramen']:
        g = t[t.run == label]['gain vs Ramen']
        row = {'run': label, 'seeds': len(g), 'gain vs Ramen mean': g.mean(), 'std': g.std(),
               'min': g.min(), 'positive seeds': int((g > 0).sum()),
               'paired t': g.mean() / (g.std() / np.sqrt(len(g))) if len(g) > 1 else np.nan,
               'above floor': g.mean() > floor,
               'same-corruption top8 %': t[t.run == label]['same-corruption top8 %'].mean()}
        if label == 'NIQE + T3':
            h = t[t.run == label]['gain vs T3']
            row.update({'gain vs T3 mean': h.mean(), 'gain vs T3 std': h.std(),
                        'seeds above T3': int((h > 0).sum())})
        out.append(row)
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    t, corr = per_seed_table()
    floor, ramen_accs = noise_floor()
    s = summary(t, floor)
    t.to_csv(os.path.join(OUT, 'per_seed.csv'), index=False)
    s.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    corr.to_csv(os.path.join(OUT, 'per_corruption.csv'), index=False)
    pd.set_option('display.width', 200)
    print(f'\nRamen over seeds 0-4: {np.round(ramen_accs, 2)}; noise floor = 2 x range = {floor:.2f}')
    print(t[['seed', 'run', 'acc', 'gain vs Ramen', 'gain vs T3', 'fixed vs T3', 'broken vs T3',
             'same-corruption top8 %']].round(2).to_string(index=False))
    print('\n' + s.round(2).to_string(index=False))
    pc = corr[corr.run != 'Ramen'].groupby(['corruption', 'run'])['gain vs Ramen'].mean().unstack()
    print('\nMean gain vs Ramen per corruption over the analysed seeds:\n' + pc.round(2).to_string())
    stack = s[s.run == 'NIQE + T3']
    if len(stack) and stack.seeds.iloc[0] == 5:
        st = stack.iloc[0]
        print(f"\nPrimary (gain vs Ramen > floor {floor:.2f}): {st['gain vs Ramen mean']:+.2f} -> "
              f"{'PASS' if st['above floor'] else 'FAIL'}")
        print(f"Secondary (beats T3 on the mean and >= 4/5 seeds): {st['gain vs T3 mean']:+.2f}, "
              f"{int(st['seeds above T3'])}/5 -> "
              f"{'PASS' if st['gain vs T3 mean'] > 0 and st['seeds above T3'] >= 4 else 'FAIL'}")


if __name__ == '__main__':
    main()
