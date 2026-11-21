"""
Offline GMM label-source experiment: fit a class-conditional mixture by EM to the cached zero-shot embeddings of a
dataset (all images at once, no model forward pass) and score its labels against zero-shot.

    from gmm_experiment import GMMLabelExperiment
    e = GMMLabelExperiment('EuroSAT', seed=0)
    e.sanity_check()          # iteration 0 == cached NoAdapt; our softmax(100 z.W) == cached probs
    e.run('lda')              # cached as results/<prefix>_gmm_lda.pkl (+ _gmm_lda_iters.pkl)
    e.table(['zs', 'lda'])    # one row per variant, counts and rates (analysis.LabelScore)

Inputs, all cached by earlier runs: <prefix>_NoAdapt.pkl / _NoAdapt_feats.npy / _NoAdapt_probs.npy and the CLIP text
features <prefix>_pillow_sink_text_features.npy (2026-11-01_pillow_sink). Read-only; nothing in the repo is edited.
The EM fit is order-free and deterministic, so the seed only selects which cached files (row order) are read.
"""
import os
import time

import numpy as np
import pandas as pd

from analysis import LabelScore, component_alignment
from gmm import EM, LearnedPrior, SharedCovGaussian, SphericalVMF, UniformPrior, softmax_rows
from gmm_configs import DATASETS, MAX_BREAK_RATE, PASS_POINTS, VARIANTS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), 'results')
CLIP_LOGIT_SCALE = 100.0


class GMMLabelExperiment:
    def __init__(self, dataset, seed=0, results_dir=RESULTS):
        self.dataset, self.seed = dataset, seed
        self.cfg = DATASETS[dataset]
        self.prefix = os.path.join(results_dir, f'{dataset}_clip_{self.cfg["model"][5:]}_mixed_nall_bs100_s{seed}')
        self.noadapt = pd.read_pickle(f'{self.prefix}_NoAdapt.pkl')
        self.Z = np.load(f'{self.prefix}_NoAdapt_feats.npy').astype(np.float32)
        self.P0 = np.load(f'{self.prefix}_NoAdapt_probs.npy').astype(np.float32)
        W = np.load(f'{self.prefix}_pillow_sink_text_features.npy').astype(np.float32)
        self.W = W / np.linalg.norm(W, axis=1, keepdims=True)
        self.labels = self.noadapt['label'].to_numpy()
        self.domains = self.noadapt['domain'].astype(str).to_numpy()
        self.zs_pred = self.noadapt['pred'].to_numpy()
        self.outputs, self.iters = {}, {}

    # ------------------------------------------------------------------ building a variant
    def _model(self, v):
        if v.family == 'vmf':
            m = SphericalVMF(kappa=v.kappa)
            m.init_means(self.W)        # only used as the fallback mean of a class whose mass vanishes
            return m
        if v.family == 'lda':
            return SharedCovGaussian(shrinkage=v.shrinkage)
        raise ValueError(v.family)

    def _start(self, v):
        if v.start == 'zero_shot':
            return self.P0
        if v.start == 'true_label':     # ORACLE
            R = np.zeros_like(self.P0)
            R[np.arange(len(R)), self.labels] = 1.0
            return R
        raise ValueError(v.start)

    # ------------------------------------------------------------------ running
    def _cache(self, name):
        return f'{self.prefix}_gmm_{name}.pkl', f'{self.prefix}_gmm_{name}_iters.pkl'

    def run(self, name, use_cache=True, verbose=True):
        path, iters_path = self._cache(name)
        if use_cache and os.path.exists(path) and os.path.exists(iters_path):
            self.outputs[name], self.iters[name] = pd.read_pickle(path), pd.read_pickle(iters_path)
            return self.outputs[name]
        v = VARIANTS[name]
        em = EM(self._model(v), LearnedPrior() if v.prior == 'learned' else UniformPrior(), n_iter=v.n_iter)
        t0 = time.time()
        log = (lambda t, tr: print(f'  [{self.dataset} {name}] iter {t}: '
                                   f'acc {100 * (tr.preds[-1] == self.labels).mean():.2f} '
                                   f'll {tr.log_likelihood[-1]:.6g} ({time.time() - t0:.0f}s)', flush=True)
               ) if verbose else None
        trace = em.fit(self.Z, self._start(v), callback=log)
        rows = []
        for t, p in enumerate(trace.preds):
            if v.is_oracle and t == 0:
                continue                # iteration 0 of an oracle is the true labels themselves
            row = LabelScore.of(p, self.zs_pred, self.labels, self.domains).as_row()
            row.update(iteration=t, log_likelihood=trace.log_likelihood[t],
                       prior_max=float(np.exp(trace.log_prior[t]).max()) if t else np.nan,
                       changed_vs_prev=int((p != trace.preds[t - 1]).sum()) if t else 0,
                       **component_alignment(p, self.labels, self.domains))
            rows.append(row)
        iters = pd.DataFrame(rows)
        np.save(path[:-len('.pkl')] + '_preds.npy', np.stack(trace.preds).astype(np.int16))
        np.save(path[:-len('.pkl')] + '_means.npy', em.model.means if em.model.means is not None else self.W)
        final = trace.final_probs
        df = self.noadapt[['batch', 'domain_idx', 'sample_idx', 'label', 'position', 'domain']].copy()
        df['zs_pred'] = self.zs_pred
        df['pred'] = trace.preds[-1]
        df['conf'] = final.max(1)
        df['correct'] = df['pred'] == df['label']
        df.to_pickle(path)
        iters.to_pickle(iters_path)
        self.outputs[name], self.iters[name] = df, iters
        return df

    # ------------------------------------------------------------------ checks and tables
    def sanity_check(self):
        """(1) EM iteration 0 == cached NoAdapt predictions (must be 100.00%).
        (2) our vMF E-step with text means, softmax(100 z.W), against the cached probabilities: argmax agreement
            and max |difference| (cached features are fp16, the model computed in fp16, so tiny gaps are expected)."""
        df = self.run('zs', use_cache=False, verbose=False)
        same = (df['pred'].to_numpy() == self.zs_pred).mean()
        agree, max_diff, n = 0, 0.0, len(self.Z)
        for s in range(0, n, 100_000):
            P, _ = softmax_rows(CLIP_LOGIT_SCALE * (self.Z[s:s + 100_000].astype(np.float64) @ self.W.T))
            agree += (P.argmax(1) == self.zs_pred[s:s + 100_000]).sum()
            max_diff = max(max_diff, float(np.abs(P - self.P0[s:s + 100_000]).max()))
        out = dict(dataset=self.dataset, n=n, iter0_equal=same, recomputed_argmax_agree=agree / n,
                   recomputed_n_disagree=int(n - agree), max_abs_prob_diff=max_diff)
        print(f'{self.dataset}: EM iteration 0 == cached NoAdapt on {same:.4%} of {n:,}; recomputed softmax(100 z.W) '
              f'argmax agrees on {agree:,} of {n:,} ({agree / n:.4%}), max |p - p_cached| = {max_diff:.2e}')
        return out

    def score(self, name):
        df = self.run(name)
        return LabelScore.of(df['pred'], self.zs_pred, self.labels, self.domains)

    def table(self, names):
        rows = []
        for n in names:
            s = self.score(n)
            rows.append(dict(dataset=self.dataset, variant=n, oracle=VARIANTS[n].is_oracle, **s.as_row(),
                             verdict=('oracle' if VARIANTS[n].is_oracle else
                                      s.verdict(self.cfg['floor'], PASS_POINTS, MAX_BREAK_RATE))))
        return pd.DataFrame(rows)
