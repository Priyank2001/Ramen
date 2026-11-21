"""
Tables over cached shift-decomposition results (no recomputation).

    r = Summary(ShiftDecomposition('DomainNet').load())
    r.cells()          # domain-averaged accuracy of every cell, gain vs zero-shot (txt_) and 95% CI
    r.shapley()        # image world {A, G, L} and text world {A, L} Shapley values with 95% CIs, text-prototype gap
    r.per_domain()     # per-domain Shapley values (image world, point estimates)
    r.decision()       # pre-registered decision rules
    overview(['ImageNetC5K', ...])   # one row per dataset
"""
import numpy as np
import pandas as pd

from attribution import ci, domain_avg, per_domain, shapley, subsets
from data import FLOOR, LABEL
from experiment import ShiftDecomposition

IMG = 'AGL'
TXT = 'AL'


def _img_name(S, prefix='img'):
    return f'{prefix}_' + ''.join(p for p in IMG if p in S)


def _txt_name(S):
    return 'txt_' + ''.join(p for p in TXT if p in S)


class Summary:
    def __init__(self, exp):
        self.e, self.ds = exp, exp.dataset
        self.col = {n: i for i, n in enumerate(exp.names)}
        self.acc = {n: domain_avg(exp.correct[n], exp.dom) for n in exp.names}

    def _boot(self, n):
        return self.e.boot[:, self.col[n]]

    def cells(self):
        base = self.acc['txt_']
        rows = []
        for n in self.e.names:
            lo, hi = ci(self._boot(n) - self._boot('txt_'))
            rows.append(dict(cell=n, acc=self.acc[n], vs_zero_shot=self.acc[n] - base, ci_lo=lo, ci_hi=hi))
        return pd.DataFrame(rows)

    def _shapley_boot(self, players, name_of):
        sets = subsets(players)
        point = shapley({S: self.acc[name_of(S)] for S in sets}, players)
        B = self.e.boot
        vals = {S: B[:, self.col[name_of(S)]] for S in sets}
        boot = shapley(vals, players)              # linear in the values, so it works column-wise
        return point, boot

    def shapley(self, prefix='img'):
        rows = []
        p, b = self._shapley_boot(IMG, lambda S: _img_name(S, prefix))
        total = sum(p.values())
        for k in IMG:
            lo, hi = ci(b[k])
            rows.append(dict(world=f'image ({prefix})', component=k, phi=p[k], ci_lo=lo, ci_hi=hi,
                             share_pct=100 * p[k] / total if total else np.nan))
        if prefix == 'img':
            p, b = self._shapley_boot(TXT, _txt_name)
            total = sum(p.values())
            for k in TXT:
                lo, hi = ci(b[k])
                rows.append(dict(world='text', component=k, phi=p[k], ci_lo=lo, ci_hi=hi,
                                 share_pct=100 * p[k] / total if total else np.nan))
            g = self._boot('img_') - self._boot('txt_')
            rows.append(dict(world='text -> image', component='prototype gap P0 - T',
                             phi=self.acc['img_'] - self.acc['txt_'], ci_lo=ci(g)[0], ci_hi=ci(g)[1], share_pct=np.nan))
            r = self._boot('ramen') - self._boot('noadapt')
            rows.append(dict(world='cached', component='Ramen - NoAdapt', phi=self.acc['ramen'] - self.acc['noadapt'],
                             ci_lo=ci(r)[0], ci_hi=ci(r)[1], share_pct=np.nan))
        return pd.DataFrame(rows)

    def per_domain(self):
        names = self.e.meta['domain_names']
        acc = {n: per_domain(self.e.correct[n], self.e.dom) for n in self.e.names if n.startswith(('img_', 'txt_'))}
        rows = []
        for d, dn in enumerate(names):
            v = {S: acc[_img_name(S)][d] for S in subsets(IMG)}
            p = shapley(v, IMG)
            vt = shapley({S: acc[_txt_name(S)][d] for S in subsets(TXT)}, TXT)
            n_img = int((self.e.dom == d).sum())
            rows.append(dict(domain=dn, images=n_img, zero_shot=acc['txt_'][d], P0=acc['img_'][d],
                             phi_A=p['A'], phi_G=p['G'], phi_L=p['L'], txt_phi_A=vt['A'], txt_phi_L=vt['L'],
                             ramen_gain=100 * (self.e.correct['ramen'][self.e.dom == d].mean()
                                               - self.e.correct['noadapt'][self.e.dom == d].mean())))
        return pd.DataFrame(rows)

    def decision(self):
        s = self.shapley().set_index(['world', 'component'])
        img = s.loc['image (img)']
        total = img.phi.sum()
        ramen = s.loc[('cached', 'Ramen - NoAdapt'), 'phi']
        dominant = [k for k in IMG if total > 0 and img.loc[k, 'phi'] >= 0.5 * total and img.loc[k, 'ci_lo'] > 0]
        domain_part = img.loc['A', 'phi'] + img.loc['G', 'phi']
        return dict(dataset=self.ds, dominant=dominant or ['none'], phi_A_plus_G=domain_part, ramen_gain=ramen,
                    domain_aware_alive=bool(domain_part > ramen))


def overview(datasets):
    rows = []
    for ds in datasets:
        r = Summary(ShiftDecomposition(ds).load())
        s = r.shapley().set_index(['world', 'component']).phi
        m = r.e.meta
        rows.append(dict(dataset=LABEL[ds], zero_shot=r.acc['txt_'], ramen_gain=s[('cached', 'Ramen - NoAdapt')],
                         floor=FLOOR[ds], P0=r.acc['img_'], full_oracle=r.acc['img_AGL'],
                         img_phi_A=s[('image (img)', 'A')], img_phi_G=s[('image (img)', 'G')],
                         img_phi_L=s[('image (img)', 'L')], txt_phi_A=s[('text', 'A')], txt_phi_L=s[('text', 'L')],
                         gamma_reliability=m['reliability']['r_full'], n0=m['n0'],
                         dominant=', '.join(r.decision()['dominant'])))
    return pd.DataFrame(rows)
