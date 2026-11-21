"""
Write 01_pseudo_label_values.ipynb (not executed). The interpretation cell is INTERPRETATION below; fill it once
the runs are in.

    python make_notebook.py
    CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace \
        01_pseudo_label_values.ipynb
"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

HEADER = r"""# Pseudo-label values: can a label-free target recover the label-value oracle's gain?

**Question.** Ramen stores, for every test image, its own entropy-minimisation gradient. When zero-shot CLIP is wrong,
that gradient pushes toward the wrong class: its cosine with the true-label gradient is −0.31, against +0.97 when
zero-shot is right (`2026-11-01_label_oracle`). Swapping each stored value for the cross-entropy gradient toward
the **true** label gives +1.70 on DomainNet, +1.51 on ImageNet-C and +26.70 on EuroSAT over Ramen (seed 0). That is an
oracle. This notebook asks how much of it a **label-free** target can recover.

**Hypothesis (pre-registered, seed 0).** At least one label-free target (zero-shot prediction, Ramen's adapted
prediction, or a kNN vote over the memory) beats Ramen by more than the noise floor on each of DomainNet (0.11),
ImageNet-C (0.53) and EuroSAT (1.51). Expectation stated before the runs: `pv_zs` is a control and should land near
Ramen, because its target is Ramen's own. The gain should track how often the target is right.

**Runs** (all start from `cfg/<dataset>/Ramen.yaml`; only the stored value changes):

| run | target of the stored CE gradient | uses labels | datasets |
|---|---|---|---|
| `Ramen` | — (own entropy gradient) | no | all (cached) |
| `pv_zs` | zero-shot prediction (control: CE form, Ramen's target) | no | all |
| `pv_adapted` | Ramen's adapted prediction of the image | no | all |
| `pv_knn` | exp(−H)-weighted vote of the 10 nearest memory entries (own entry excluded), labelled by queue | no | all |
| `pv_mix_p25` / `pv_mix_p50` | true label for a random 25% / 50% of zero-shot errors, else zero-shot | **ORACLE** | DomainNet, EuroSAT |
| `labeloracle_zs_ce` | true label (cached from `2026-11-01_label_oracle`) | **ORACLE** | all (cached) |

The dose-response runs (`pv_mix_*`) fix how good a target is and measure the gain. They show how accurate a
label-free target must be to pay off. They are not run on ImageNet-C. There, the cached label oracle is inflated by
other corrupted copies of the same photo carrying its label (+1.51 per-class, +0.92 with copies hidden), so an
oracle curve would mix two effects.

**Data and model.** DomainNet: 6 domains, 345 classes, 596,006 images, CLIP ViT-B/32. ImageNet-C 5K: 15
corruptions at severity 5 × 5,000 images, 1,000 classes, ViT-B/16. EuroSAT: 27,000 images, 10 classes, ViT-B/16.
One mixed, shuffled stream per dataset, batch size 100. **Seed 0 only.** The seed sets the stream order and the
random draws of `pv_mix_*`. NoAdapt is the same for every seed. Ramen configs: DomainNet capacity 300 / top-k 10 /
β 5; ImageNet-C 75 / 1 / 0; lr 0.01, signSGD. The fp16 priority counter is kept, as in the cached Ramen, so every
comparison is like for like. On DomainNet this freezes the memory after about 28% of the stream.

**How the results were produced.**
```
cd new_experiments/2026-11-04_pseudo_label_values
CUDA_VISIBLE_DEVICES=0 python run_pseudo_values.py DomainNet 0 --smoke smoke     # also ImageNetC5K, EuroSAT
CUDA_VISIBLE_DEVICES=0 python run_pseudo_values.py DomainNet 0 --runs pv_zs,pv_adapted   # etc., one per GPU
```
Code: `pseudo_methods.py` (`RamenPseudoValues`, subclass of `RamenLabelValues`), `pseudo_values.py` (runs,
`PseudoValueExperiment`), `pv_registry.py`, `pv_plots.py`. Nothing in `src/`, `cfg/` or earlier folders is changed.

**What counts as a result.** Change vs Ramen in domain-averaged points, against the noise floor (2 × (max − min) of
Ramen over seeds 0–4). Seed 0 is a screen: a run that clears the floor here earns seeds 1–4, nothing more."""

MATH = r"""## Method

**Ramen (baseline).** Batch $t$ holds $x_1,\dots,x_B$. $\theta_0$ = the affine parameters of the image encoder's norm
layers; each sample gets its own copy $\theta_i$. $w_c$ = normalised text embedding of class $c$.

$$z_i = \frac{f_{\theta_0}(x_i)}{\lVert f_{\theta_0}(x_i)\rVert}, \quad p_i = \operatorname{softmax}(100\, z_i^\top W), \quad
\hat y_i = \arg\max_c p_{ic}, \quad H_i = -\textstyle\sum_c p_{ic}\log p_{ic}, \quad g_i = \nabla_\theta H_i\big|_{\theta_0}$$

1. **Write** $(z_i, g_i, H_i, \pi_i)$ into queue $\mathcal M_{\hat y_i}$ (before retrieval; a full queue keeps its $K$ newest).
2. **Retrieve** from every non-empty queue $c$ the $k$ entries nearest to $z_i$: $\mathcal N_k^c(z_i)$, $d_{ij} = \lVert z_i - z_j\rVert$.
3. **Aggregate** $\hat g_i = \frac{1}{|C_i|}\sum_{c\in C_i}\sum_{j\in\mathcal N_k^c(z_i)} e^{-H_j} e^{-\beta d_{ij}}\, v_j$, where $v_j$ is entry $j$'s stored value ($v_j = g_j$ in Ramen).
4. **Update** $\theta_i = \theta_0 - \eta\,\operatorname{sign}(\hat g_i)$. 5. **Predict** $\hat y_i^{R} = \arg\max_c \operatorname{softmax}(100\,\tilde z_i^\top W)_c$ under $\theta_i$; reset $\theta_i \leftarrow \theta_0$.

**Variant (the only change).** After step 5, the value of image $i$'s entry is replaced:
$$v_i \leftarrow \nabla_\theta\, \mathrm{CE}\big(100\, z_i(\theta)^\top W,\ \tau_i\big)\Big|_{\theta_0}$$
computed by a second zero-shot pass at $\theta_0$. While batch $t$ retrieves, its own entries still hold $g_i$ (as in
Ramen and the oracle), so no image ever uses its own target. The target $\tau_i$:

| run | $\tau_i$ |
|---|---|
| `pv_zs` | $\hat y_i$ |
| `pv_adapted` | $\hat y_i^{R}$ |
| `pv_knn` | $\arg\max_c \sum_{j \in \mathcal N_{10}(z_i)\setminus\{i\}} e^{-H_j}\,\mathbb 1[q_j = c]$, with $\mathcal N_{10}$ = the 10 nearest entries of the **whole** memory and $q_j$ = the queue entry $j$ is filed in |
| `pv_mix_p` (**ORACLE**) | $y_i$ if $\hat y_i \ne y_i$ and $u_i < p$, else $\hat y_i$; $u_i \sim U(0,1)$ from a fixed generator (seed 0) |
| `labeloracle_zs_ce` (**ORACLE**) | $y_i$ |

| symbol | code | config key | value here |
|---|---|---|---|
| $\tau_i$ rule | `TargetRule` subclasses in `pseudo_methods.py` | `target` | zero_shot / adapted / knn / mix |
| 10 (kNN size) | `RamenPseudoValues.knn_vote` | `knn_k` | 10 |
| $p$ | `MixedOracleTarget` | `mix_p` | 0.25 / 0.50 |
| CE value | `RamenPseudoValues.ce_values` | — | per-sample gradient, `reduction='sum'` |
| replace value | `RamenLabelValues.reveal` (2026-11-01) | — | same slot the image was written to |

The true labels reach `forward` for every run (the harness passes them). Label-free runs use them **only** in
`log_targets` (how often the target is right), never in the update."""

INTERPRETATION = """## Interpretation

**Pre-registered hypothesis not supported (seed 0).** No label-free target beats Ramen beyond the noise floor on any dataset (floors: DomainNet 0.11, ImageNet-C 0.53, EuroSAT 1.51).

| run | DomainNet | ImageNet-C | EuroSAT |
|---|---|---|---|
| CE → zero-shot (control) | +0.07 (+1,147 images) | +0.19 (+141) | −1.13 (−306) |
| CE → adapted pred | −0.65 (−4,595) | −5.28 (−3,957) | −31.17 (−8,416) |
| CE → kNN vote | −1.50 (−7,336) | −15.84 (−11,883) | −10.05 (−2,714) |
| CE → 25% errors corrected (oracle) | +0.75 (+6,158) | not run | +17.55 (+4,739) |
| CE → 50% errors corrected (oracle) | +1.29 (+9,727) | not run | +24.75 (+6,682) |

1. **The CE form alone does nothing.** The zero-shot control is within the floor everywhere, so the oracle's gain comes from the target, not from replacing entropy by cross-entropy.
2. **Targets must be much more accurate than zero-shot.** Adapted-prediction target DomainNet: right 53.75% vs 52.82%, fixes 21,649 / breaks 16,118, still −0.65. kNN vote EuroSAT: right 49.91% vs 48.09% (fixes 1,873 / breaks 1,381), still −10.05. Oracle at 74.06% target accuracy (EuroSAT) gives +24.75.
3. **Targets that move many images hurt.** kNN on ImageNet-C is right only 15.72% (vs 27.28%): breaks 11,303, fixes 2,631. The adapted target on EuroSAT is right 31.46% (vs 48.09%), a feedback loop through Ramen's own adaptation.
4. **Caveats.** Seed 0 only; the dose-response is oracle-based and excluded from ImageNet-C (same-photo copy leak). The fp16 priority freeze on DomainNet is kept for like-for-like comparison.

**Conclusion:** a label-free value target needs to be far better than zero-shot without overturning correct images; none of the three tried is. Seeds 1-4 are not warranted for these runs."""

GLOSSARY = """## Glossary

- **Value** (of a memory entry): the gradient vector it contributes to a retrieving image's update.
- **Target**: the class whose cross-entropy gradient becomes an entry's value in these runs.
- **Label-free**: uses no ground-truth labels in the update (labels may be logged).
- **ORACLE**: uses ground-truth labels in the update; an upper bound or a measuring device, not a method.
- **Dose-response** (`pv_mix_p`): targets made right for a known share of zero-shot errors, to map target
  accuracy to gain.
- **Target right %**: share of stream images whose target equals their true class.
- **Noise floor**: 2 × (max − min) of Ramen's accuracy over seeds 0–4; a gain counts only above it.
- **Domain average**: mean of per-domain accuracies (the headline convention).
- **fp16 freeze**: Ramen's priority counter saturates in fp16, after which full queues stop updating."""

cells = [
    md(HEADER),
    md(MATH),
    code("""import sys
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from pseudo_values import DOSE, LABEL_FREE, PV_VARIANTS, SMOKE_CHECKS, PseudoValueExperiment
import pv_plots

pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})

FLOORS = {'DomainNet': 0.11, 'ImageNetC5K': 0.53, 'EuroSAT': 1.51}
ORACLE = 'labeloracle_zs_ce'
exps = {}
for ds in FLOORS:
    e = PseudoValueExperiment(ds, seed=0)
    e.run_all(('NoAdapt', 'Ramen', ORACLE, *e.screen_runs))   # all cached: no GPU
    exps[ds] = e"""),
    md("""## Sanity checks

On 200 images per domain, the variant with no target must reproduce Ramen exactly. The variant with the true-label
target, which goes through this folder's second zero-shot pass, must reproduce the cached label oracle exactly."""),
    code("""for ds in FLOORS:
    s = PseudoValueExperiment(ds, seed=0, n_per_domain=200, results_dir='smoke')
    s.run_all(['Ramen', ORACLE, 'pv_off', 'pv_true'])
    for a, b in SMOKE_CHECKS:
        assert s.sanity_check(a, b), f'{ds}: {a} != {b}'"""),
    md("""## The data

Images per domain and zero-shot / Ramen accuracy: the denominators for everything below."""),
    code("""for ds, e in exps.items():
    print(ds)
    display(e.accuracy_table(['NoAdapt', 'Ramen']).round(2))"""),
    md("""## 1. Change vs Ramen

Every run against Ramen, domain-averaged, with the noise floor shaded. Blue = label-free, orange = oracle."""),
    code("""gains = {}
for ds, e in exps.items():
    names = [*e.screen_runs, ORACLE]
    t = e.vs_table(names)
    t['kind'] = ['label-free' if n in LABEL_FREE else 'oracle' for n in t.index]
    t['clears floor'] = t['vs Ramen (pts)'] > FLOORS[ds]
    gains[ds] = t
    display(t.drop(columns='kind'))
    best = t.loc[list(LABEL_FREE), 'vs Ramen (pts)']
    pv_plots.plot_gain_vs_floor(t, FLOORS[ds], f'{ds}: best label-free {best.idxmax()} {best.max():+.2f} '
                                f'(floor {FLOORS[ds]})')
    plt.show()"""),
    md("""## 2. How good is each target?

How often the target is the true class, compared with zero-shot. Also: zero-shot errors the target corrects, and
zero-shot-correct images it turns wrong (logged with the labels; the update never sees them)."""),
    code("""for ds, e in exps.items():
    print(ds); display(e.target_table())"""),
    md("""## 3. Dose-response: gain vs target accuracy

The orange line joins `pv_zs` (no errors corrected), the two partial oracles and the full label oracle. A
label-free run (blue) above the line does better than its target accuracy predicts; one below does worse.
ImageNet-C has no partial oracles, and its full-oracle point carries the copy leak (see the top of the notebook)."""),
    code("""fig, axes = plt.subplots(1, 3, figsize=(17, 4.3))
for ax, (ds, e) in zip(axes, exps.items()):
    tt, g = e.target_table(), gains[ds]
    pts = tt[['target right %']].join(g[['vs Ramen (pts)', 'label', 'kind']], how='inner')
    pts.loc[ORACLE] = [100.0, g.loc[ORACLE, 'vs Ramen (pts)'], 'true label', 'oracle']
    pv_plots.plot_dose_response(pts, ['pv_zs', *DOSE, ORACLE], ds, ax=ax)
plt.tight_layout(); plt.show()"""),
    md("""## 4. DomainNet per domain

Where the change happens. Quickdraw carried most of the label oracle's gain (+4.02)."""),
    code("""e = exps['DomainNet']
display(e.accuracy_table(['Ramen', *e.screen_runs, ORACLE]).round(2))"""),
    md(INTERPRETATION),
    md(GLOSSARY),
]

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nbf.write(nb, '01_pseudo_label_values.ipynb')
print('wrote 01_pseudo_label_values.ipynb')
