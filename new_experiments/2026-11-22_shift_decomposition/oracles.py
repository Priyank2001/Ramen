"""
Oracle classifiers of the shift decomposition (Part B of PREREGISTRATION.md). ALL use true labels and/or true domains.

    fit = AdditiveFit.from_data(...)
    clf = PrototypeOracle(fit, use_A=True, use_G=False)                 # image world, nearest class mean
    clf = TextOracle(W, alpha=fit.alpha)                                # text world, zero-shot on shifted features
    prior = LabelPrior.from_counts(y, dom, C, n_dom)                    # ORACLE per-domain class frequencies
    pred_noL, pred_L, kappa = Evaluator(Z, y, dom, group).predict(clf, prior)

Image world score:  kappa * ( z~^T m_{c,d} - ||m_{c,d}||^2 / 2 ),  z~ = z - 1[A] alpha_d,  m_{c,d} = mu + beta_c + 1[G] gamma~_{c,d}
Text world score:   tau * z'^T w_c,  z' = norm(z - alpha_d) (or z).
L adds log(C * pi_{c|d}). Leave-one-out: the true-class prototype and prior count of image i exclude i's whole
leave-out group (its photo's corrupted copies on ImageNet-C); mu and alpha are not recomputed (influence <= 1/n_d).
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

CHUNK = 20_000


class GroupStats:
    """Per-image sums over its leave-out group: whole group (for beta) and the group's members in its own domain (cell)."""

    def __init__(self, Z, dom, group, alpha):
        _, g = np.unique(group, return_inverse=True)
        G = g.max() + 1
        self.sum = np.zeros((G, Z.shape[1]), np.float64)
        np.add.at(self.sum, g, Z)
        self.alpha_sum = np.zeros((G, Z.shape[1]), np.float64)
        np.add.at(self.alpha_sum, g, alpha[dom])
        self.size = np.bincount(g, minlength=G).astype(np.float64)
        n_dom = alpha.shape[0]
        _, gd = np.unique(g * n_dom + dom, return_inverse=True)
        self.cell_sum = np.zeros((gd.max() + 1, Z.shape[1]), np.float64)
        np.add.at(self.cell_sum, gd, Z)
        self.cell_size = np.bincount(gd).astype(np.float64)
        self.g, self.gd = g, gd


class PrototypeOracle:
    """Image-world nearest-class-mean classifier built from an AdditiveFit (optionally leave-one-out)."""

    world = 'image'

    def __init__(self, fit, use_A, use_G, n0=None):
        self.fit, self.use_A, self.use_G = fit, use_A, use_G
        self.n0 = fit.n0 if n0 is None else n0
        f = fit
        self.shrink = self._shrink(f.n)
        gamma = f.gamma * self.shrink[..., None] if use_G else np.zeros_like(f.gamma)
        self.M = (f.mu + f.beta[:, None, :] + gamma).astype(np.float32)        # (C, n_dom, D)
        self.half_sq = 0.5 * (self.M.astype(np.float64) ** 2).sum(2).astype(np.float32)  # (C, n_dom)
        self.alpha = f.alpha.astype(np.float32)
        # class sums and the additive part's class term n_c mu + sum_d n_cd alpha_d (for leave-one-out beta)
        self.S_c = f.S.sum(1)
        self.A_c = f.n_c[:, None] * f.mu + f.n @ f.alpha
        self.loo_fallbacks = 0

    def _shrink(self, n):
        n = np.asarray(n, dtype=np.float64)
        if np.isinf(self.n0):
            return np.zeros_like(n)
        return np.divide(n, n + self.n0, out=np.zeros_like(n), where=n > 0)   # empty cell: no interaction term

    def features(self, Z, dom):
        return Z - self.alpha[dom] if self.use_A else Z

    def scores(self, Zt, d):
        """(n, C) scores of features Zt (already shifted) that all lie in domain d, without temperature."""
        return Zt @ self.M[:, d, :].T - self.half_sq[:, d]

    def loo_true_scores(self, Zt, idx, y, dom, gs):
        """Score of each image's true class with its leave-out group removed from that class's prototype."""
        f, yi, di = self.fit, y[idx], dom[idx]
        g, gd = gs.g[idx], gs.gd[idx]
        n_rest = f.n_c[yi] - gs.size[g]
        beta = (self.S_c[yi] - gs.sum[g] - self.A_c[yi] + gs.size[g][:, None] * f.mu + gs.alpha_sum[g])
        ok = n_rest > 0
        beta[ok] /= n_rest[ok, None]
        beta[~ok] = 0.0                       # class has no image outside the group: prototype falls back to mu
        self.loo_fallbacks += int((~ok).sum())
        m = f.mu + beta
        if self.use_G:
            n_cell = f.n[yi, di] - gs.cell_size[gd]
            S_cell = f.S[yi, di] - gs.cell_sum[gd]
            has = n_cell > 0
            gam = np.zeros_like(m)
            gam[has] = S_cell[has] / n_cell[has, None] - f.mu - f.alpha[di[has]] - beta[has]
            m = m + gam * self._shrink(n_cell)[:, None]
        return (Zt.astype(np.float64) * m).sum(1) - 0.5 * (m ** 2).sum(1)


class TextOracle:
    """Text-world zero-shot classifier on optionally domain-shifted features (alpha: (n_dom, D) or None)."""

    world = 'text'

    def __init__(self, W, alpha=None):
        self.W, self.alpha = W.astype(np.float32), None if alpha is None else alpha.astype(np.float32)

    def features(self, Z, dom):
        if self.alpha is None:
            return Z
        Zs = Z - self.alpha[dom]
        return Zs / np.linalg.norm(Zs, axis=1, keepdims=True)

    def scores(self, Zt, d):
        return Zt @ self.W.T


class LabelPrior:
    """ORACLE per-domain class frequencies, add-one smoothed; leave-one-out for the image's own label."""

    def __init__(self, counts):
        self.counts = counts.astype(np.float64)              # (C, n_dom)
        self.C = counts.shape[0]
        self.n_d = self.counts.sum(0)

    @classmethod
    def from_counts(cls, y, dom, C, n_dom):
        n = np.zeros((C, n_dom), np.int64)
        np.add.at(n, (y, dom), 1)
        return cls(n)

    def log_term(self, d, loo):
        """(C,) log(C pi_c|d) with add-one smoothing; with loo the denominator drops one image (same for all classes)."""
        return np.log(self.C * (self.counts[:, d] + 1) / (self.n_d[d] - loo + self.C))

    def loo_true_log(self, yi, d):
        """Own-label count without the image itself: (n_yd - 1) + 1."""
        return np.log(self.C * self.counts[yi, d] / (self.n_d[d] - 1 + self.C))


class Evaluator:
    """Runs a classifier over the stream (by domain, in chunks), with or without the label prior."""

    def __init__(self, Z, y, dom, group, eval_mask=None, loo=True, n_temp=20_000, seed=0):
        self.Z, self.y, self.dom, self.group, self.loo = Z, y, dom, group, loo
        self.eval_idx = np.flatnonzero(np.ones(len(y), bool) if eval_mask is None else eval_mask)
        rng = np.random.default_rng(seed)
        self.temp_idx = np.sort(rng.choice(self.eval_idx, min(n_temp, len(self.eval_idx)), replace=False))
        self._gs = {}

    def group_stats(self, clf):
        key = id(clf.fit)
        if key not in self._gs:
            self._gs[key] = GroupStats(self.Z, self.dom, self.group, clf.fit.alpha)
        return self._gs[key]

    def _base_scores(self, clf, idx):
        """Yield (rows, scores) per domain chunk; true-class column replaced by its leave-one-out value."""
        loo_proto = self.loo and clf.world == 'image'
        gs = self.group_stats(clf) if loo_proto else None
        for d in np.unique(self.dom[idx]):
            rows_d = idx[self.dom[idx] == d]
            for s in range(0, len(rows_d), CHUNK):
                rows = rows_d[s:s + CHUNK]
                Zt = clf.features(self.Z[rows], self.dom[rows])
                B = clf.scores(Zt, d).astype(np.float64)
                if loo_proto:
                    B[np.arange(len(rows)), self.y[rows]] = clf.loo_true_scores(Zt, rows, self.y, self.dom, gs)
                yield d, rows, B

    def fit_temperature(self, clf):
        """NLL-optimal temperature on the subsample (temperature scaling), before the prior is added."""
        Bs, ys = [], []
        for _, rows, B in self._base_scores(clf, self.temp_idx):
            Bs.append(B), ys.append(self.y[rows])
        B, yy = np.concatenate(Bs), np.concatenate(ys)
        Bt = B[np.arange(len(yy)), yy]

        def nll(logk):
            k = np.exp(logk)
            return float(np.mean(logsumexp(k * B, axis=1) - k * Bt))
        res = minimize_scalar(nll, bounds=(-2, 16), method='bounded', options=dict(xatol=1e-3))
        return float(np.exp(res.x))

    def predict(self, clf, prior=None):
        """(pred without prior, pred with prior or None, temperature) for every image in eval_idx (full-length arrays)."""
        N = len(self.y)
        p0, pL = np.full(N, -1), (np.full(N, -1) if prior is not None else None)
        kappa = self.fit_temperature(clf) if prior is not None else np.nan
        for d, rows, B in self._base_scores(clf, self.eval_idx):
            p0[rows] = B.argmax(1)
            if prior is not None:
                L = np.broadcast_to(prior.log_term(d, self.loo), B.shape).copy()
                if self.loo:
                    L[np.arange(len(rows)), self.y[rows]] = prior.loo_true_log(self.y[rows], d)
                pL[rows] = (kappa * B + L).argmax(1)
        return p0, pL, kappa
