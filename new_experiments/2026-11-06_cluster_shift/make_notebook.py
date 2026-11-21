"""
Write 01_cluster_shift.ipynb (not executed).

    python make_notebook.py
    ~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace 01_cluster_shift.ipynb
"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

HEADER = r"""# Is Ramen's gain mostly a domain-level shift?

**Question.** The shift-direction analysis (2026-11-03) found that Ramen's embedding move is not aimed at the
image's own class pair, and that images ending at the same wrong class move together because they share a
*domain*. If most of Ramen's gain is such a domain-level correction, a plain label-free cluster-mean shift of the
zero-shot embedding should recover most of it.

**Hypothesis (pre-registered).** A streaming per-cluster mean shift recovers at least 70% of Ramen's gain over
NoAdapt, and beats a random-group control. Refuted if it recovers under 30%; in between = partial.

**Variants.** `none` (identity), `kmeans_k{K}` (online k-means clusters, label-free), `random_k{K}` (same cluster
sizes, random membership: control), `oracle_domain` (**ORACLE**: true domain / corruption id as the group).
`_a0.5 / _a1.0` is the shift strength $\alpha$; `_global / _text` the reference point (below).

**Data.** Seed 0, full datasets, mixed stream, batch 100: ImageNet-C 5K (75,000 images, 15 corruptions, CLIP
ViT-B/16), DomainNet (596,006 images, 6 domains, ViT-B/32), EuroSAT (27,000 images, 1 domain, ViT-B/16). A seed
sets the stream order. Only seed 0 was run; gains are compared with the noise floors of Ramen over seeds 0-4
(0.54, 0.11, 1.51).

**How produced.** Offline replay of cached `NoAdapt_feats` (no GPU forward pass):
`python run_cluster_shift.py <dataset> 0`. Identity replay matches the cached NoAdapt on 99.4-99.7% of images (the
cached features are fp16-rounded), not 100.00%.
"""

MATH = r"""## Method

Baseline (zero-shot): $z_i = f_{\theta_0}(x_i)/\lVert f_{\theta_0}(x_i)\rVert$, $\hat y_i = \arg\max_c z_i^\top w_c$.
Ramen replaces $z_i$ by $\tilde z_i$ computed with its one-step-adapted norm parameters.

Cluster shift replaces $z_i$ by
$$z_i' = \frac{z_i - \alpha\,(\mu_{g(i)} - m)}{\lVert z_i - \alpha\,(\mu_{g(i)} - m)\rVert}, \qquad \hat y_i = \arg\max_c z_i'^\top w_c$$
- $g(i)$: group of image $i$. k-means: nearest of $K$ online mini-batch centroids (updated with the current batch
  first, as Ramen writes a batch before retrieving). Random: drawn with the cluster proportions of the matching
  k-means run. **Oracle: true domain id.**
- $\mu_g$: running mean of the zero-shot embeddings assigned to $g$ so far (current batch included).
- $m$: reference. `global`: running mean of all embeddings, so only between-group differences are removed.
  `text`: $\bar w = \frac1C\sum_c w_c$, a fixed vector (adds the class-dependent term $\alpha \bar w^\top w_c$).
- $\alpha \in \{0.5, 1\}$ (`alpha`); $K \in \{1, 5, 20, 50\}$ (`k`). Code: `cluster_shift.py` (`KMeansShift`,
  `RandomGroupShift`, `OracleGroupShift`), `cshift_configs.py`.

Diagnostic: with $\Delta_i = \tilde z_i - z_i$ (Ramen minus NoAdapt features),
$$\text{explained}_{\text{beyond mean}}(g) = 1 - \frac{\sum_i \lVert \Delta_i - \bar\Delta_{g(i)} \rVert^2}{\sum_i \lVert \Delta_i - \bar\Delta \rVert^2}$$
"""

SETUP = """import sys
sys.path.insert(0, '.')
import pandas as pd
from cshift_experiment import ClusterShiftExperiment, domain_average
from analysis import ShiftExplained
from plots import plot_gain_vs_floor
from cshift_configs import VARIANTS
FLOOR = {'ImageNetC5K': 0.54, 'DomainNet': 0.11, 'EuroSAT': 1.51}
exps = {d: ClusterShiftExperiment(d, 0) for d in FLOOR}
for d, e in exps.items():
    n = len(e.noadapt)
    print(f'{d}: {n:,} images, {e.noadapt.domain.nunique()} domain(s), {len(e.W)} classes, '
          f'NoAdapt wrong on {(~e.noadapt.correct).sum():,} of {n:,} ({(~e.noadapt.correct).mean():.1%})')
"""

RESULTS = """tables = {}
for d, e in exps.items():
    names = [n for n in VARIANTS if not (d == 'EuroSAT' and n.startswith('oracle'))]
    tables[d] = e.summary(names)
    ramen_gain = tables[d].loc[1, 'vs_noadapt']
    fig = plot_gain_vs_floor(tables[d], ramen_gain, FLOOR[d], f'{d}: no variant reaches Ramen')
    display(tables[d].round(2).query("variant.str.contains('global') or variant.str.contains('cached') or variant=='none'").reset_index(drop=True))
"""

EXPLAINED = """for d, e in exps.items():
    names = ['kmeans_k5_a0.5_global', 'kmeans_k20_a0.5_global', 'random_k20_a0.5_global'] + \\
            (['oracle_domain_a1.0_global'] if d != 'EuroSAT' else [])
    print(d); display(ShiftExplained(e).table(names).round(3))
"""

INTERP = r"""## Interpretation

- **The main hypothesis is refuted.** No label-free variant gets near 70% of Ramen's gain: best `global`-reference
  variants recover 11% (ImageNet-C, +0.64 vs floor 0.54), 30% (DomainNet, +0.43 vs floor 0.11) and none on EuroSAT
  (negative). The best-of-grid pick is itself optimistic (chosen on the test stream).
- **A domain-level component is real but partial.** Using the true domain id (ORACLE) recovers 27% (ImageNet-C,
  +1.58) and 60% (DomainNet, +0.86 of Ramen's small +1.43). Random groups give ~0 (the control works).
- **Ramen's shift is only partly group-constant.** Beyond the all-image mean, group means explain 14-22% of its squared
  shift on ImageNet-C and 13-16% on DomainNet (random: ~0). On EuroSAT k-means clusters explain 28-43%, but there
  clusters track classes, and cluster-mean subtraction removes class signal (every EuroSAT variant loses accuracy).
- **`text`-reference variants are a design error, not evidence.** Adding the fixed vector $\bar w$ biases every
  prediction toward the classes aligned with it (-2 to -15 points everywhere, even for random groups).
- **Not tested:** per-domain breakdown, seeds 1-4, $\alpha > 1$, ViT-L. Only seed 0; identity replay is 99.4-99.7%
  identical to NoAdapt, not 100%.
"""

GLOSSARY = r"""## Glossary
- **NoAdapt**: zero-shot CLIP. **Ramen**: the baseline TTA method.
- **Noise floor**: 2 x (max - min) of Ramen's accuracy over seeds 0-4.
- **ORACLE**: uses the dataset's true domain id; not available to a real method.
- **Explained beyond mean**: share of Ramen's shift (after removing the single all-image mean shift) a group-constant shift reproduces.
"""

nb = nbf.v4.new_notebook()
nb.cells = [md(HEADER), md(MATH), code(SETUP), code(RESULTS), code(EXPLAINED), md(INTERP), md(GLOSSARY)]
nbf.write(nb, '01_cluster_shift.ipynb')
