"""
Two-way (class x domain) decomposition of embedding means, fitted on cell sums (Part A of PREREGISTRATION.md).

    fit = AdditiveFit.from_data(Z, y, dom, C, n_dom)
    fit.mu, fit.alpha (n_dom, D), fit.beta (C, D), fit.gamma (C, n_dom, D)     # z ~ mu + alpha_d + beta_c + gamma_cd
    fit.ss_table()        # SS, df, F, debiased share per term
    fit.n0                # interaction shrinkage pseudo-count sigma^2 / tau^2 (inf = no detectable interaction)

The additive part (mu, alpha, beta) is the weighted least-squares fit without interaction (backfitting, weights =
cell counts), i.e. OLS of the per-image model; gamma is each cell mean minus that fit (0 for empty cells).
"""
import numpy as np


class AdditiveFit:
    def __init__(self, S, n, sumsq, tol=1e-7, max_iter=500):
        """S: (C, n_dom, D) cell sums, n: (C, n_dom) cell counts, sumsq: sum_i ||z_i||^2."""
        self.S, self.n = S.astype(np.float64), n.astype(np.float64)
        self.C, self.n_dom, self.D = S.shape
        self.N = self.n.sum()
        self.n_c, self.n_d = self.n.sum(1), self.n.sum(0)
        self.mu = self.S.sum((0, 1)) / self.N
        self._backfit(tol, max_iter)
        M = np.divide(self.S, self.n[..., None], out=np.zeros_like(self.S), where=self.n[..., None] > 0)
        fitted = self.mu + self.alpha[None] + self.beta[:, None]
        self.gamma = np.where(self.n[..., None] > 0, M - fitted, 0.0)
        self.ss_w = sumsq - (np.einsum('cdk,cdk->cd', self.S, self.S)[self.n > 0] / self.n[self.n > 0]).sum()
        self.n_cells = int((self.n > 0).sum())
        self.C_present = int((self.n_c > 0).sum())
        self.df = dict(A=self.n_dom - 1, B=self.C_present - 1, G=self.n_cells - self.C_present - self.n_dom + 1,
                       W=int(self.N) - self.n_cells)
        self.sigma2 = self.ss_w / (self.D * self.df['W'])           # per-dimension within-cell variance
        ss_g = self.ss('G')
        self.tau2 = (ss_g / self.D - self.df['G'] * self.sigma2) / self.N  # per-dimension interaction variance
        self.n0 = self.sigma2 / self.tau2 if self.tau2 > 0 else np.inf

    @classmethod
    def from_data(cls, Z, y, dom, C, n_dom, **kw):
        S = np.zeros((C, n_dom, Z.shape[1]), np.float64)
        np.add.at(S, (y, dom), Z.astype(np.float64))
        n = np.zeros((C, n_dom), np.int64)
        np.add.at(n, (y, dom), 1)
        return cls(S, n, float((Z.astype(np.float64) ** 2).sum()), **kw)

    def _backfit(self, tol, max_iter):
        R = self.S - self.n[..., None] * self.mu                     # residual sums after mu
        alpha, beta = np.zeros((self.n_dom, self.D)), np.zeros((self.C, self.D))
        with np.errstate(invalid='ignore', divide='ignore'):
            for _ in range(max_iter):
                b = (R.sum(1) - np.einsum('cd,dk->ck', self.n, alpha)) / self.n_c[:, None]
                b = np.nan_to_num(b)
                a = (R.sum(0) - np.einsum('cd,ck->dk', self.n, b)) / self.n_d[:, None]
                change = max(np.abs(a - alpha).max(), np.abs(b - beta).max())
                alpha, beta = a, b
                if change < tol:
                    break
        # identifiability: weighted means of alpha and beta are zero, absorbed into mu
        ma, mb = (self.n_d[:, None] * alpha).sum(0) / self.N, (self.n_c[:, None] * beta).sum(0) / self.N
        self.alpha, self.beta, self.mu = alpha - ma, beta - mb, self.mu + ma + mb
        self.iterations = _ + 1

    def ss(self, term):
        if term == 'A':
            return float((self.n_d * (self.alpha ** 2).sum(1)).sum())
        if term == 'B':
            return float((self.n_c * (self.beta ** 2).sum(1)).sum())
        if term == 'G':
            return float((self.n * (self.gamma ** 2).sum(2)).sum())
        if term == 'W':
            return float(self.ss_w)
        raise ValueError(term)

    def ss_table(self):
        rows, debiased = [], {}
        for t, name in (('A', 'domain offset alpha (appearance)'), ('B', 'class effect beta'),
                        ('G', 'class x domain gamma (concept)')):
            ss, df = self.ss(t), self.df[t]
            F = ss / (self.D * df * self.sigma2) if df > 0 else np.nan
            debiased[t] = max(ss - df * self.D * self.sigma2, 0.0) if df > 0 else 0.0
            rows.append(dict(term=name, SS=ss, df=df, F=F, SS_debiased=debiased[t]))
        total = sum(debiased.values()) + self.ss_w
        for r, t in zip(rows, ('A', 'B', 'G')):
            r['share_pct'] = 100 * debiased[t] / total
        rows.append(dict(term='within cell', SS=self.ss_w, df=self.df['W'], F=np.nan, SS_debiased=self.ss_w,
                         share_pct=100 * self.ss_w / total))
        return rows


def split_half_reliability(Z, y, dom, group, C, n_dom, seed=0, min_n=2):
    """
    Correlation of gamma estimated on two random halves of the leave-out groups, over cells with >= min_n images in
    both halves, and its Spearman-Brown full-sample value 2r/(1+r). Near 0 = gamma not estimable at this size.
    """
    rng = np.random.default_rng(seed)
    ug = np.unique(group)
    half = np.isin(group, ug[rng.permutation(len(ug))[:len(ug) // 2]])
    fits = [AdditiveFit.from_data(Z[m], y[m], dom[m], C, n_dom) for m in (half, ~half)]
    ok = (fits[0].n >= min_n) & (fits[1].n >= min_n)
    if ok.sum() < 2 or n_dom < 2:
        return dict(cells=int(ok.sum()), r=np.nan, r_full=np.nan)
    a, b = fits[0].gamma[ok].ravel(), fits[1].gamma[ok].ravel()
    r = float(np.corrcoef(a, b)[0, 1])
    return dict(cells=int(ok.sum()), r=r, r_full=2 * r / (1 + r) if r > -1 else np.nan)
