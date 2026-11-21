"""
Datasets and variants of the GMM label-source experiment (plain data, no behaviour).

DATASETS: model of the cached seed-s runs and the Ramen noise floor (used here only as a minimum effect size: the
          EM label is computed on the whole dataset at once, so it does not depend on stream order or seed).
VARIANTS: name -> Variant. Pre-registered on 2026-11-18 before any result was seen; n_iter = 10 is the headline,
          earlier iterations are logged and reported as descriptive only.

  family  'vmf' (SphericalVMF, kappa = 100 = CLIP's logit scale) or 'lda' (SharedCovGaussian, shrinkage 0.1)
  prior   'uniform' (fixed 1/C, as zero-shot) or 'learned' (pi_c = N_c / N; expected to collapse onto zero-shot's
          sink classes, see 2026-10-26_prior_estimation)
  start   'zero_shot' (R0 = cached zero-shot posterior = E-step of a vMF with the text embeddings as means) or
          'true_label' (ORACLE: R0 = one-hot true labels, one M-step + one E-step = supervised fit on the test set)
"""
from dataclasses import dataclass

DATASETS = {
    'DomainNet': dict(model='clip_vitbase32', floor=0.11),
    'ImageNetC5K': dict(model='clip_vitbase16', floor=0.54),
    'EuroSAT': dict(model='clip_vitbase16', floor=1.51),
}

# Pre-registered decision rule (pooled label accuracy of the n_iter = 10 label vs zero-shot, same images):
PASS_POINTS = 10.0        # >= +10 points: comparable to the oracle mix that gave +0.75 on DomainNet (+11.8 points)
MAX_BREAK_RATE = 0.05     # ... and breaks < 5% of the images zero-shot gets right


@dataclass(frozen=True)
class Variant:
    family: str
    prior: str = 'uniform'
    start: str = 'zero_shot'
    n_iter: int = 10
    kappa: float = 100.0
    shrinkage: float = 0.1

    @property
    def is_oracle(self):
        return self.start == 'true_label'


VARIANTS = {
    'zs': Variant('vmf', n_iter=0),                                     # identity: must equal cached NoAdapt
    'vmf': Variant('vmf'),
    'vmf_prior': Variant('vmf', prior='learned'),
    'lda': Variant('lda'),
    'lda_prior': Variant('lda', prior='learned'),
    'vmf_oracle': Variant('vmf', start='true_label', n_iter=1),         # ORACLE ceiling of the vMF family
    'lda_oracle': Variant('lda', start='true_label', n_iter=1),         # ORACLE ceiling of the LDA family
}
