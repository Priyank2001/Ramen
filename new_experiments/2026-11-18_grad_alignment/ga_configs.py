"""
Datasets, the probe run, and the pre-registered reading of the gradient-alignment experiment (plain data).

Pre-registered 2026-11-18, before any full run. The probe changes no prediction, so there is no accuracy gain and no
noise-floor test; the questions are about directions:
  H1  Ramen's update carries image-specific information about the true class: the median of
      cos(g_hat, g_ce) is above the median of the shuffled control cos(g_hat_i, g_ce_j), j another image.
  H2  Memory helps where zero-shot is wrong: on zero-shot-WRONG images the median cos(g_hat, g_ce) is above the
      median cos(g_self, g_ce) (own entropy gradient; pilot on 100 DomainNet images: mean -0.31 on 38 wrong ones).
  H3  Alignment explains outcomes: among zero-shot-wrong images, Ramen's fix rate rises with cos(sign g_hat, g_ce),
      and among zero-shot-right images the images Ramen breaks have a median cos(sign g_hat, g_ce) below 0.

    from ga_configs import DATASETS, PROBE_RUN
"""

# model / num_workers of the cached NoAdapt and Ramen runs (num_workers changes the stream order)
DATASETS = {
    'DomainNet': dict(model='clip_vitbase32', num_workers=16, notebook='01_grad_alignment_domainnet.ipynb'),
    'ImageNetC5K': dict(model='clip_vitbase16', num_workers=16, notebook='02_grad_alignment_imagenetc.ipynb'),
    'EuroSAT': dict(model='clip_vitbase16', num_workers=16, notebook='03_grad_alignment_eurosat.ipynb'),
}

# cache name -> (method, cfg overrides on cfg/<dataset>/Ramen.yaml). Prefixed with the topic so no cache collides.
PROBE_RUN = 'ga_probe'
RUNS = {PROBE_RUN: ('RamenGradientProbe', {})}
