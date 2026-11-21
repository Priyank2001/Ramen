"""
Scoring a label source against zero-shot on the same images (pure functions, no model, no I/O).

    from analysis import LabelScore
    s = LabelScore.of(new_pred, zs_pred, labels, domains)
    s.as_row()      # dict: acc (domain-avg, pooled), fixed / broken counts and rates, changed, verdict inputs

Definitions (every count is over the same N images):
  fixed   = zero-shot wrong, new label right         (rescue rate = fixed / zero-shot wrong)
  broken  = zero-shot right, new label wrong         (break rate  = broken / zero-shot right)
  changed = new label != zero-shot label
  fixed - broken = (new correct) - (zero-shot correct)  (checked in LabelScore.of)
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def component_alignment(pred, labels, domains):
    """What the mixture's components track: normalised mutual information of the predicted component with the true
    class and with the domain (0 = independent, 1 = one determines the other). One-domain datasets give nmi_domain 0."""
    from sklearn.metrics import normalized_mutual_info_score as nmi
    return dict(nmi_class=nmi(labels, pred), nmi_domain=nmi(np.asarray(domains), pred))


def domain_average(correct, domains):
    """Headline convention: mean over domains of per-domain accuracy, in percent."""
    return 100 * pd.Series(correct).groupby(np.asarray(domains)).mean().mean()


@dataclass
class LabelScore:
    n: int
    acc_domain_avg: float
    acc_pooled: float
    zs_acc_domain_avg: float
    zs_acc_pooled: float
    zs_wrong: int
    zs_right: int
    fixed: int
    broken: int
    changed: int
    max_class_share: float      # share of images given the single most frequent label (sink indicator)
    n_classes_used: int

    @classmethod
    def of(cls, pred, zs_pred, labels, domains):
        pred, zs_pred, labels = (np.asarray(a) for a in (pred, zs_pred, labels))
        ok, zs_ok = pred == labels, zs_pred == labels
        fixed, broken = int((~zs_ok & ok).sum()), int((zs_ok & ~ok).sum())
        assert fixed - broken == int(ok.sum()) - int(zs_ok.sum())
        counts = np.bincount(pred)
        return cls(n=len(pred),
                   acc_domain_avg=domain_average(ok, domains), acc_pooled=100 * ok.mean(),
                   zs_acc_domain_avg=domain_average(zs_ok, domains), zs_acc_pooled=100 * zs_ok.mean(),
                   zs_wrong=int((~zs_ok).sum()), zs_right=int(zs_ok.sum()),
                   fixed=fixed, broken=broken, changed=int((pred != zs_pred).sum()),
                   max_class_share=counts.max() / len(pred), n_classes_used=int((counts > 0).sum()))

    @property
    def gain_pooled(self):
        return self.acc_pooled - self.zs_acc_pooled

    @property
    def gain_domain_avg(self):
        return self.acc_domain_avg - self.zs_acc_domain_avg

    @property
    def rescue_rate(self):
        return self.fixed / max(self.zs_wrong, 1)

    @property
    def break_rate(self):
        return self.broken / max(self.zs_right, 1)

    def verdict(self, floor, pass_points, max_break_rate):
        """Pre-registered rule (gmm_configs.py): pass / weak / null / hurts, on pooled accuracy gain."""
        g = self.gain_pooled
        if g >= pass_points and self.break_rate < max_break_rate:
            return 'pass'
        if g > floor:
            return 'weak'
        if g >= -floor:
            return 'null'
        return 'hurts'

    def as_row(self):
        return {**asdict(self), 'gain_pooled': self.gain_pooled, 'gain_domain_avg': self.gain_domain_avg,
                'rescue_rate': self.rescue_rate, 'break_rate': self.break_rate}
