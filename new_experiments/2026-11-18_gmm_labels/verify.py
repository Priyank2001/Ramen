"""
Correctness evidence on the real cached data (complements test_gmm.py, which uses synthetic data).
Run after run_gmm.py has cached every variant:

    ~/miniconda3/envs/RAMEN/bin/python verify.py > logs/verify.log

Checks, per dataset:
  1. plumbing    EM iteration 0 == cached NoAdapt predictions on 100.00% of images.
  2. E-step      our softmax(100 z.W) vs the model's cached probabilities; where the argmax disagrees, the cached
                 top-2 margin is tiny (fp16 features re-multiplied in fp32/fp64, not a different formula).
  3. EM          log-likelihood never decreases for the variants where EM guarantees it (vMF, fixed kappa);
                 LDA uses shrinkage 0.1, which is not a pure ML M-step, so its log-likelihood is reported, not asserted.
  4. LDA oracle  our one-hot supervised fit with shrinkage 0 == sklearn LinearDiscriminantAnalysis on the real
                 embeddings (EuroSAT, all 27,000 images), after removing sklearn's empirical class prior.
  5. scoring     headline accuracies recomputed from the saved prediction arrays with plain numpy == the table.
  6. read-only   no file under src/, cfg/ or the shared new_experiments/*.py is newer than this folder's first file.
"""
import os
import subprocess

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from gmm import EM, SharedCovGaussian, UniformPrior, softmax_rows
from gmm_configs import DATASETS, VARIANTS
from gmm_experiment import GMMLabelExperiment

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LABEL_FREE = [n for n, v in VARIANTS.items() if not v.is_oracle and n != 'zs']


def check_estep(e):
    P0 = e.P0
    disagree, margins_all = [], []
    for s in range(0, len(e.Z), 100_000):
        P, _ = softmax_rows(100.0 * e.Z[s:s + 100_000].astype(np.float64) @ e.W.T)
        top2 = np.sort(P0[s:s + 100_000], axis=1)[:, -2:]
        m = top2[:, 1] - top2[:, 0]
        margins_all.append(m)
        disagree.append(m[P.argmax(1) != e.zs_pred[s:s + 100_000]])
    m_all, m_dis = np.concatenate(margins_all), np.concatenate(disagree)
    print(f'  [2] E-step: {len(m_dis):,} of {len(m_all):,} ({len(m_dis) / len(m_all):.2%}) argmax disagreements; '
          f'cached top-2 prob margin there: median {np.median(m_dis):.4f}, 99th pct {np.quantile(m_dis, .99):.4f} '
          f'(all images: median {np.median(m_all):.4f}); share of all images with margin <= the 99th pct: '
          f'{(m_all <= np.quantile(m_dis, .99)).mean():.2%}')


def check_monotone(e):
    for n in LABEL_FREE:
        e.run(n)
        ll = e.iters[n]['log_likelihood'].to_numpy()[1:]
        d = np.diff(ll)
        worst = d.min() / np.abs(ll[:-1]).max()
        guaranteed = VARIANTS[n].family == 'vmf'
        status = ('OK' if worst >= -1e-9 else 'DECREASES') if guaranteed else 'reported only (shrinkage)'
        print(f'  [3] {n:10s} log-lik {ll[0]:.6g} -> {ll[-1]:.6g}; min relative step {worst:+.2e}  {status}')
        if guaranteed:
            assert worst >= -1e-9, n


def check_lda_vs_sklearn(e):
    R0 = np.eye(e.P0.shape[1], dtype=np.float32)[e.labels]
    ours = EM(SharedCovGaussian(shrinkage=0.0), UniformPrior(), n_iter=1).fit(e.Z, R0).final_probs
    X = e.Z.astype(np.float64)
    sk = LinearDiscriminantAnalysis(solver='lsqr').fit(X, e.labels)
    ref, _ = softmax_rows(sk.decision_function(X) - np.log(sk.priors_)[None, :])   # uniform prior, as ours
    agree = (ours.argmax(1) == ref.argmax(1)).mean()
    print(f'  [4] one-hot LDA (shrinkage 0) vs sklearn on all {len(X):,} images: max |p - p_sklearn| = '
          f'{np.abs(ours - ref).max():.2e}, argmax agreement {agree:.4%}')


def check_scoring(e):
    for n in VARIANTS:
        prefix = f'{e.prefix}_gmm_{n}'
        final = np.load(prefix + '_preds.npy')[-1].astype(np.int64)
        acc_np = 100 * (final == e.labels).mean()
        acc_tab = e.table([n])['acc_pooled'].iloc[0]
        assert abs(acc_np - acc_tab) < 1e-9, (n, acc_np, acc_tab)
    print(f'  [5] pooled accuracy recomputed from saved predictions == table for all {len(VARIANTS)} variants')


def check_read_only():
    first = os.path.join(HERE, 'gmm.py')
    cmd = ['find', 'src', 'cfg', '-newer', first, '-type', 'f']
    newer = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.split()
    shared = [f for f in os.listdir(os.path.join(REPO, 'new_experiments')) if f.endswith('.py')]
    newer += [f for f in shared if os.path.getmtime(os.path.join(REPO, 'new_experiments', f)) > os.path.getmtime(first)]
    print(f'[6] files under src/, cfg/ or shared new_experiments/*.py modified after this experiment started: '
          f'{newer or "none"}')


if __name__ == '__main__':
    for ds in DATASETS:
        e = GMMLabelExperiment(ds, seed=0)
        print(f'== {ds} ({len(e.Z):,} images, {e.P0.shape[1]} classes)')
        same = (np.load(f'{e.prefix}_gmm_zs_preds.npy')[0] == e.zs_pred).mean()
        print(f'  [1] iteration 0 == cached NoAdapt on {same:.4%} of images')
        check_estep(e)
        check_monotone(e)
        if ds == 'EuroSAT':
            check_lda_vs_sklearn(e)
        check_scoring(e)
    check_read_only()
