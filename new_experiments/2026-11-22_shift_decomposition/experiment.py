"""
The shift-decomposition experiment on one dataset: ANOVA (Part A), all oracle cells (Part B), controls, cross-fit,
sanity checks and bootstrap; results cached in results/<dataset>_s<seed>.npz + .pkl. ALL cells are ORACLES.

    e = ShiftDecomposition('DomainNet').run()        # or .load() if cached
    e.correct['img_AG'], e.meta['anova'], e.meta['kappa'], e.boot (B, K)

Cell names: img_<S> for S in {'', A, G, L, AG, AL, GL, AGL} (img_ = P0, pooled image prototypes);
txt_<S> for S in {'', A, L, AL} (txt_ = zero-shot); txt_Ad / txt_AdL = A from image-weighted domain means
(needs domains, not labels); img_G_n0 / img_AG_n0 = unshrunk gamma; rnd_<S> = random-domain control;
cf_<S> = 2-fold cross-fit of img_<S>.
"""
import os
import pickle
import time

import numpy as np

from anova import AdditiveFit, split_half_reliability
from attribution import ClusterBootstrap
from data import ShiftData
from oracles import Evaluator, LabelPrior, PrototypeOracle, TextOracle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')
IMG_SETS = ['', 'A', 'G', 'AG']          # prototype configurations; L is added on top of each


class ShiftDecomposition:
    def __init__(self, dataset, seed=0, B=1000, log=print):
        self.dataset, self.seed, self.B, self.log = dataset, seed, B, log
        self.path = os.path.join(OUT, f'{dataset}_s{seed}')
        self.correct, self.pred, self.meta = {}, {}, {'kappa': {}}

    # ------------------------------------------------------------------ cells
    def _add(self, name, pred):
        self.pred[name] = pred.astype(np.int32)
        self.correct[name] = pred == self.d.y

    def _image_cells(self, fit, ev, prior, prefix, sets=IMG_SETS, n0=None, suffix=''):
        for s in sets:
            t = time.time()
            clf = PrototypeOracle(fit, use_A='A' in s, use_G='G' in s, n0=n0)
            p0, pL, k = ev.predict(clf, prior)
            self._add(f'{prefix}_{s}{suffix}', p0)
            if prior is not None:
                self._add(f'{prefix}_{s}L{suffix}', pL)
                self.meta['kappa'][f'{prefix}_{s}{suffix}'] = k
            self.meta.setdefault('loo_fallbacks', {})[f'{prefix}_{s}{suffix}'] = clf.loo_fallbacks
            self.log(f'  {prefix}_{s}{suffix}: {time.time() - t:.1f}s')

    def _text_cells(self, ev, prior):
        d = self.d
        grand = d.Z.mean(0)
        alpha_dom = np.stack([d.Z[d.dom == k].mean(0) - grand for k in range(d.n_dom)])
        for name, alpha in (('', None), ('A', self.fit.alpha), ('Ad', alpha_dom)):
            p0, pL, k = ev.predict(TextOracle(d.W, alpha), prior)
            self._add(f'txt_{name}', p0)
            self._add(f'txt_{name}L', pL)
            self.meta['kappa'][f'txt_{name}'] = k

    def _random_domain_control(self):
        d = self.d
        dom_r = np.random.default_rng(self.seed).permutation(d.dom)        # same domain sizes, random membership
        fit_r = AdditiveFit.from_data(d.Z, d.y, dom_r, d.C, d.n_dom)
        ev_r = Evaluator(d.Z, d.y, dom_r, d.group)
        self._image_cells(fit_r, ev_r, None, 'rnd', sets=IMG_SETS)
        self.meta['rnd_n0'] = fit_r.n0

    def _cross_fit(self):
        d = self.d
        ug = np.unique(d.group)
        rng = np.random.default_rng(self.seed + 1)
        fold = np.isin(d.group, ug[rng.permutation(len(ug))[:len(ug) // 2]]).astype(int)
        preds = {}
        for k in (0, 1):
            train, test = fold != k, fold == k
            fit = AdditiveFit.from_data(d.Z[train], d.y[train], d.dom[train], d.C, d.n_dom)
            prior = LabelPrior.from_counts(d.y[train], d.dom[train], d.C, d.n_dom)
            ev = Evaluator(d.Z, d.y, d.dom, d.group, eval_mask=test, loo=False)
            for s in IMG_SETS:
                p0, pL, _ = ev.predict(PrototypeOracle(fit, 'A' in s, 'G' in s), prior)
                for name, p in ((s, p0), (s + 'L', pL)):
                    preds.setdefault(name, np.full(len(d.y), -1))[test] = p[test]
        for name, p in preds.items():
            assert (p >= 0).all()
            self._add(f'cf_{name}', p)

    # ------------------------------------------------------------------ checks
    def sanity(self):
        d, s = self.d, {}
        s['txt_vs_cached_noadapt_pct'] = 100 * (self.pred['txt_'] == d.noadapt_pred).mean()
        s['txt_vs_cached_noadapt_n_diff'] = int((self.pred['txt_'] != d.noadapt_pred).sum())
        if d.n_dom == 1:   # alpha = gamma = 0 by construction: A and G cells must equal P0
            for c in ('img_A', 'img_G', 'img_AG', 'img_AL', 'img_GL', 'img_AGL'):
                ref = 'img_L' if c.endswith('L') else 'img_'
                s[f'{c}_eq_{ref}_pct'] = 100 * (self.pred[c] == self.pred[ref]).mean()
        self.meta['sanity'] = s
        return s

    # ------------------------------------------------------------------ run / load
    def run(self, use_cache=True, crossfit=True, control=True):
        if use_cache and os.path.exists(self.path + '.pkl'):
            return self.load()
        t0 = time.time()
        self.d = d = ShiftData(self.dataset, self.seed)
        self.log(f'{self.dataset}: {len(d.y):,} images, {d.C} classes, {d.n_dom} domains')
        self.fit = AdditiveFit.from_data(d.Z, d.y, d.dom, d.C, d.n_dom)
        self.meta.update(anova=self.fit.ss_table(), n0=self.fit.n0, sigma2=self.fit.sigma2, tau2=self.fit.tau2,
                         backfit_iterations=self.fit.iterations, describe=d.describe(),
                         reliability=split_half_reliability(d.Z, d.y, d.dom, d.group, d.C, d.n_dom, self.seed),
                         domain_names=d.domain_names)
        self.log(f'  ANOVA done ({time.time() - t0:.0f}s), n0 = {self.fit.n0:.3g}, '
                 f'reliability {self.meta["reliability"]}')
        ev = Evaluator(d.Z, d.y, d.dom, d.group, seed=self.seed)
        prior = LabelPrior.from_counts(d.y, d.dom, d.C, d.n_dom)
        self._image_cells(self.fit, ev, prior, 'img')
        self._text_cells(ev, prior)
        if d.n_dom > 1:
            self._image_cells(self.fit, ev, None, 'img', sets=['G', 'AG'], n0=0.0, suffix='_n0')
            if control:
                self._random_domain_control()
        if crossfit:
            self._cross_fit()
        self.log(f'  sanity: {self.sanity()}')
        self.correct['noadapt'], self.correct['ramen'] = d.noadapt_correct, d.ramen_correct
        self.names = list(self.correct)
        self.boot = ClusterBootstrap(d.dom, d.group, B=self.B, seed=self.seed).replicates(
            np.stack([self.correct[n] for n in self.names], 1))
        self.meta['seconds'] = time.time() - t0
        self.save()
        self.log(f'  done in {self.meta["seconds"]:.0f}s')
        return self

    def save(self):
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(self.path + '.npz', dom=self.d.dom, y=self.d.y,
                            **{f'pred__{k}': v for k, v in self.pred.items()},
                            noadapt=self.correct['noadapt'], ramen=self.correct['ramen'])
        with open(self.path + '.pkl', 'wb') as f:
            pickle.dump(dict(meta=self.meta, names=self.names, boot=self.boot), f)

    def load(self):
        with open(self.path + '.pkl', 'rb') as f:
            blob = pickle.load(f)
        self.meta, self.names, self.boot = blob['meta'], blob['names'], blob['boot']
        z = np.load(self.path + '.npz')
        self.dom, self.y = z['dom'], z['y']
        self.pred = {k[len('pred__'):]: z[k] for k in z.files if k.startswith('pred__')}
        self.correct = {k: p == self.y for k, p in self.pred.items()}
        self.correct['noadapt'], self.correct['ramen'] = z['noadapt'], z['ramen']
        return self
