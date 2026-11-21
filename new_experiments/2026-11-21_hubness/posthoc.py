"""
POST-HOC checks run after the pre-registered tables were seen (labelled as such in the README).

  1. pc lambda 1 and 2 (uniform target, stack.prior_correct) as the reference the hubness corrections are compared with,
     at the zero-shot level (6 datasets x seeds 0-4, from the saved NoAdapt probabilities) and at the Ramen level (seed 0).
  2. Does CSLS's hub score h_c (end of stream) follow how often a class is *predicted* or how often it *occurs*?
     Spearman of h_c with zero-shot predicted counts and with true counts (labels used for this description only).
  3. Hubness profile table from tables/zero_shot_runs.csv (seed 0): skew, TV and top-1% share before / after.

    python posthoc.py       # -> tables/posthoc_pc.csv, posthoc_hub_vs_frequency.csv, posthoc_profile.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data import DATASETS, RAMEN_FEATS, StreamData, stack
from experiment import TABLES, frozen
from hubness import Csls, ZeroShot, run

KEY = 10**7


def zero_shot_probs(ds, df):
    z0 = pd.read_pickle(stack.path(ds, 0, 'NoAdapt') + '.pkl')
    P0 = np.load(stack.path(ds, 0, 'NoAdapt') + '_probs.npy')
    key0 = pd.Series(np.arange(len(z0)), index=z0.domain_idx.values * KEY + z0.sample_idx.values)
    return P0[key0.loc[df.domain_idx.values * KEY + df.sample_idx.values].values]


def pc_rows():
    rows = []
    for ds in DATASETS:
        for seed in range(5):
            df = pd.read_pickle(stack.path(ds, seed, 'Ramen') + '.pkl')
            P = zero_shot_probs(ds, df)
            base = stack.domain_acc(df, P.argmax(1))
            for lam in (1.0, 2.0):
                acc = stack.domain_acc(df, stack.prior_correct(P, df.batch.values, lam=lam))
                rows.append({'level': 'zero-shot', 'dataset': ds, 'seed': seed, 'method': f'pc lambda {lam:g}', 'gain': acc - base})
        print('pc zero-shot', ds, flush=True)
    for ds in RAMEN_FEATS:
        df, P = stack.load(ds, 0, 'Ramen')
        base = stack.domain_acc(df, P.argmax(1))
        for lam in (1.0, 2.0):
            acc = stack.domain_acc(df, stack.prior_correct(P, df.batch.values, lam=lam, m=0.99))
            rows.append({'level': 'Ramen', 'dataset': ds, 'seed': 0, 'method': f'pc lambda {lam:g}', 'gain': acc - base})
    return pd.DataFrame(rows)


def hub_vs_frequency():
    p = frozen()['csls']
    rows = []
    for ds in DATASETS:
        sd = StreamData(ds, 0)
        c = Csls(sd.W, **p)
        pred = run(c, sd.Z, sd.batches())
        h = c.hub().cpu().numpy()
        zs = run(ZeroShot(sd.W), sd.Z, sd.batches())
        n_pred, n_true = np.bincount(zs, minlength=sd.classes), np.bincount(sd.label, minlength=sd.classes)
        rows.append({'dataset': ds, 'classes': sd.classes,
                     'rho(h, zero-shot predicted count)': spearmanr(h, n_pred)[0],
                     'rho(h, true count)': spearmanr(h, n_true)[0],
                     'rho(predicted, true count)': spearmanr(n_pred, n_true)[0],
                     'cv of true counts': n_true.std() / n_true.mean()})
        print('hub', ds, flush=True)
        del sd
    return pd.DataFrame(rows)


def main():
    pc = pc_rows()
    pc.to_csv(os.path.join(TABLES, 'posthoc_pc.csv'), index=False)
    pd.set_option('display.width', 220)
    s = pc.groupby(['level', 'dataset', 'method'], sort=False).gain.agg(['mean', 'std']).round(2)
    print('\n== pc reference (mean gain over zero-shot / over Ramen, seeds 0-4 at zero-shot level, seed 0 at Ramen level)')
    print(s.to_string())
    hv = hub_vs_frequency()
    hv.to_csv(os.path.join(TABLES, 'posthoc_hub_vs_frequency.csv'), index=False)
    print('\n== CSLS hub score vs class frequency (Spearman)')
    print(hv.round(2).to_string(index=False))
    t = pd.read_csv(os.path.join(TABLES, 'zero_shot_runs.csv'))
    prof = t[t.seed == 0][['dataset', 'method', 'acc', 'skew N1', 'TV to true marginal',
                           'share in zero-shot top-1% classes %', 'true share of those classes %']]
    prof.to_csv(os.path.join(TABLES, 'posthoc_profile.csv'), index=False)
    print('\n== hubness profile, seed 0')
    print(prof.round(2).to_string(index=False))
    print('POSTHOC DONE', flush=True)


if __name__ == '__main__':
    main()
