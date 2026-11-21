"""
Write one gradient-alignment notebook per dataset from shared cells (not executed).

    python make_notebook.py DomainNet            # -> 01_grad_alignment_domainnet.ipynb
    CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace \
        01_grad_alignment_domainnet.ipynb

Adding a dataset: add an entry to SPECS (and to ga_configs.DATASETS), run the probe, write its interpretation in
interpretation.py, then run this script with the dataset name.
"""
import sys

import nbformat as nbf

from ga_configs import DATASETS
from interpretation import INTERPRETATION

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SPECS = {
    'DomainNet': dict(
        data='DomainNet, 596,006 images, 6 domains (clipart, infograph, painting, quickdraw, real, sketch), 345 '
             'classes; CLIP ViT-B/32; `cfg/DomainNet/Ramen.yaml`: queue capacity $K = 300$, $k = 10$ neighbours per '
             'queue, $\\beta = 5$, SignSGD $\\eta = 0.01$.',
        K=300, k=10, beta=5, window=10_000),
    'ImageNetC5K': dict(
        data='ImageNet-C 5K, 75,000 images (the same 5,000 photos under 15 corruptions, severity 5), 1,000 classes; '
             'CLIP ViT-B/16; `cfg/ImageNetC5K/Ramen.yaml`: $K = 75$, $k = 1$, $\\beta = 0$, $\\eta = 0.01$.',
        K=75, k=1, beta=0, window=2_000),
    'EuroSAT': dict(
        data='EuroSAT, 27,000 satellite images, 1 domain, 10 classes; CLIP ViT-B/16; `cfg/EuroSAT/Ramen.yaml`: '
             '$K = 7{,}500$, $k = 50$, $\\beta = 5$, $\\eta = 0.01$.',
        K=7500, k=50, beta=5, window=1_000, one_domain=True),
}


def header(ds, s):
    return rf"""# Does Ramen's update point toward the true class? Cosine with the oracle cross-entropy gradient ({ds})

**Question.** Ramen adapts each test image with one SignSGD step along $\hat g_i$, a weighted average of *entropy*
gradients retrieved from its memory. Entropy minimisation sharpens whatever the model already predicts, right or
wrong. How close is $\hat g_i$ to the direction we would really want, the gradient of the **cross-entropy toward the
true label** $g^{{CE}}_i$? We measure the cosine similarity between the two for every test image.

**This is an oracle measurement, not a method.** $g^{{CE}}_i$ uses the ground-truth label, which a real test-time
method never has. The probe only *observes*: Ramen's predictions are unchanged (checked in section 3).

**Hypotheses (pre-registered in `ga_configs.py` before the full run).**
- **H1** Ramen's update carries image-specific information about the true class: median $\cos(\hat g_i, g^{{CE}}_i)$ is
  above the shuffled control $\cos(\hat g_i, g^{{CE}}_j)$, $j$ another image of the same batch.
- **H2** Memory helps where zero-shot is wrong: on zero-shot-wrong images, median $\cos(\hat g_i, g^{{CE}}_i)$ is above
  median $\cos(g_i, g^{{CE}}_i)$, with $g_i$ the image's own entropy gradient (pilot on 100 DomainNet images: mean
  $-0.31$ on the 38 wrong ones).
- **H3** Alignment explains outcomes: (a) among zero-shot-wrong images, the share Ramen fixes rises with
  $\cos(\operatorname{{sign}} \hat g_i, g^{{CE}}_i)$; (b) images Ramen breaks have a median
  $\cos(\operatorname{{sign}} \hat g_i, g^{{CE}}_i) < 0$.

**Why this test.** It asks directly whether Ramen's direction is "correct", using the cheapest possible oracle:
one extra backward pass per batch on a run that is otherwise Ramen itself.

**Runs.**

| Run | What it does | Uses true labels? |
|---|---|---|
| `NoAdapt` | zero-shot CLIP (cached) | no |
| `Ramen` | Ramen (cached) | no |
| `ga_probe` | Ramen + per-image gradient probe (`RamenGradientProbe`); predictions = Ramen's | **yes, only to measure** |

**Data.** {s['data']} Mixed stream (all domains shuffled together), batch size 100.

**Seeds.** Seed 0 only (a seed sets the stream order). No accuracy changes here, so the noise floor does not apply;
all numbers are descriptive, from one stream.

**How the runs were produced.**
`CUDA_VISIBLE_DEVICES=0 python run_grad_alignment.py {ds} 0` (after `--smoke smoke` on 200 images per domain, and
`check_gradients.py`). Results: `new_experiments/results/{ds}_*_s0_ga_probe.pkl` and `_stats.pkl` (per-image probe
log).

**What counts as a result.** The four pre-registered tests (section 6), plus the descriptive figures.

**Sections.** 1 Setup · 2 Data · 3 Sanity and correctness · 4 Results · 5 Mechanism · 6 Hypotheses ·
7 Interpretation · 8 Glossary
"""


def math(s):
    return rf"""## Method: what is measured

**Ramen (unchanged; notation shared by all Ramen notebooks).** Batch $x_1, \dots, x_B$ ($B = 100$); $\theta_0$ = the
affine weights and biases of every LayerNorm in the CLIP image encoder ($D = 39{{,}}936$ numbers), one copy per image.
$w_c$ = normalised text embedding of class $c$.
$$z_i = \frac{{f_{{\theta_0}}(x_i)}}{{\lVert f_{{\theta_0}}(x_i) \rVert}}, \quad
p_i = \operatorname{{softmax}}(100\, z_i^\top W), \quad \hat y_i = \arg\max_c p_{{ic}}, \quad
H_i = -\textstyle\sum_c p_{{ic}} \log p_{{ic}}, \quad g_i = \nabla_\theta H_i \big|_{{\theta_0}}$$
1. *Write* $(z_i, g_i, H_i)$ into queue $\mathcal M_{{\hat y_i}}$ (capacity $K$), **before** retrieval.
2. *Retrieve* from every non-empty queue $c \in C$ the $k$ entries nearest to $z_i$, $\mathcal N_k^c(z_i)$.
3. *Aggregate* $\displaystyle \hat g_i = \frac{{1}}{{|C|}} \sum_{{c \in C}} \sum_{{j \in \mathcal N_k^c(z_i)}}
   e^{{-H_j}} e^{{-\beta d_{{ij}}}}\, g_j$, with $d_{{ij}} = \lVert z_i - z_j \rVert$.
4. *Update* $\theta_i = \theta_0 - \eta\, \operatorname{{sign}}(\hat g_i)$, predict with $\theta_i$, reset to $\theta_0$.

**The oracle direction** (uses the true label $y_i$ — ground truth, available only in this experiment):
$$g^{{CE}}_i = \nabla_\theta \big[-\log p_{{i y_i}}\big]\Big|_{{\theta_0}}$$
the gradient of the cross-entropy toward the true class at the same unadapted parameters. Stepping along
$-g^{{CE}}_i$ raises $p_{{i y_i}}$ fastest.

**Measured per image.**
- $\cos(\hat g_i, g^{{CE}}_i)$: alignment of Ramen's update. **Positive = moving against $\hat g_i$ also lowers the
  true-label loss** (both are descent directions, so the sign convention matches).
- $\cos(\operatorname{{sign}} \hat g_i, g^{{CE}}_i)$: alignment of the step Ramen actually takes. To first order,
  $$\mathrm{{CE}}(\theta_i) - \mathrm{{CE}}(\theta_0) \approx -\eta\, \operatorname{{sign}}(\hat g_i)^\top g^{{CE}}_i
  = -\eta \sqrt D\, \lVert g^{{CE}}_i \rVert\, \cos(\operatorname{{sign}} \hat g_i, g^{{CE}}_i),$$
  so a positive value means the step lowers the true-label loss.
- Sign agreement $A_i = \sum_t |g^{{CE}}_{{it}}|\, \mathbb 1[\operatorname{{sign}} \hat g_{{it}} = \operatorname{{sign}} g^{{CE}}_{{it}}]
  / \sum_t |g^{{CE}}_{{it}}|$: the share of the oracle gradient's mass on coordinates the step moves the right way
  (0.5 = random signs).
- **Reference** $\cos(g_i, g^{{CE}}_i)$: the image's own entropy gradient, i.e. per-image entropy minimisation with no
  memory.
- **Control** $\cos(\hat g_i, g^{{CE}}_{{\pi(i)}})$, $\pi(i) = i - 1 \bmod B$ (another image of the same batch): how much
  alignment any update gets just from directions shared by all images.

**Where the alignment comes from.** $\hat g_i$ is a sum of per-queue terms, so it splits exactly into four parts,
$\hat g_i = s_i + t_i + u_i + o_i$:
$s_i$ = the image's **own entry** (written before retrieval, found at distance $\approx 0$, so
$s_i \approx e^{{-H_i}} g_i / |C|$; zero if it was not stored), $t_i$ = the other entries of the **true-class queue**
$\mathcal M_{{y_i}}$, $u_i$ = the other entries of the **zero-shot-class queue** $\mathcal M_{{\hat y_i}}$ when
$\hat y_i \ne y_i$ (else 0), $o_i$ = **all other queues**. Then
$$\cos(\hat g_i, g^{{CE}}_i) = \sum_{{v \in \{{s, t, u, o\}}}} \frac{{\langle v_i, g^{{CE}}_i \rangle}}
{{\lVert \hat g_i \rVert\, \lVert g^{{CE}}_i \rVert}}.$$

| Symbol | Meaning | Code | Value here |
|---|---|---|---|
| $g^{{CE}}_i$ | true-label CE gradient (**oracle**) | `RamenGradientProbe.oracle_ce_gradient` | — |
| $\hat g_i$ | Ramen's update | `RamenTrueLabel.retrieve` (= Ramen's) | — |
| $g_i$ | own entropy gradient | `get_by_sample_grad()` after Ramen's entropy backward | — |
| $s, t, u, o$ | parts of $\hat g_i$ | `RamenGradientProbe.decompose` / `queue_part` | — |
| cosines, $A_i$ | per-image metrics | `alignment.py`; logged as `cos_update`, `cos_step`, `agree_step`, `cos_self`, `cos_update_shuffled`, `part_*` | — |
| $K, k, \beta, \eta$ | Ramen's settings | `max_capacity`, `topk`, `beta`, `lr` | {s['K']}, {s['k']}, {s['beta']}, 0.01 |
"""


SETUP = """## 1. Setup"""
SETUP_CODE = """import os
import warnings
warnings.filterwarnings('ignore', message='IProgress not found')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')   # analysis only: everything is loaded from cached results
import pandas as pd
from IPython.display import display

from ga_experiment import GradAlignmentExperiment
from ga_analysis import (data_table, outcome_table, direction_summary, decomposition, binned_outcome_rates,
                         hypotheses, memory_freeze_position)
from ga_plots import (plot_cosine_distributions, plot_outcome_dumbbell, plot_domain_dumbbell, plot_stream,
                      plot_decomposition, plot_outcome_vs_alignment)

pd.set_option('display.width', 200, 'display.max_columns', 30)
DATASET = {ds!r}
e = GradAlignmentExperiment(DATASET, seed=0).run_all()   # cached NoAdapt, Ramen and ga_probe
df = e.frame()                                            # one row per image
print(f'{{len(df):,}} images loaded')"""

DATA_MD = """## 2. The data the code ran on

Counts are images; "fixed" = zero-shot wrong and Ramen right, "broken" = zero-shot right and Ramen wrong.
Accuracy here is **pooled over images** (not domain-averaged), because the cosines are per image. Zero-shot = the
prediction Ramen filed each image under (logged by the probe)."""
DATA_CODE = """display(data_table(df))
display(outcome_table(df))"""

CHECK_MD = """## 3. Sanity and correctness

1. The probe must not change Ramen: identical predictions on every image of the stream.
2. Zero-shot predictions inside Ramen vs the cached NoAdapt run (another code path; fp16 near-ties can differ).
3. `check_gradients.py` (log below): G1 the batched per-image CE gradient equals the gradient of the image computed
   alone; G2 a step along $-\\operatorname{sign}(g^{CE})$ raises p(true class); G3 the four parts sum back to Ramen's
   update; G4 the own-entry part is non-zero exactly when the entry was stored."""
CHECK_CODE = """assert e.sanity_check(), 'the probe changed Ramen\\'s predictions'
print(f'zero-shot prediction differs from the cached NoAdapt run on {e.zs_mismatch:,} of {len(df):,} images '
      f'({e.zs_mismatch / len(df):.3%})')
log = f'logs/check_gradients_{DATASET}.log'
print(open(log).read() if os.path.exists(log) else f'{log} missing: run check_gradients.py {DATASET}')"""

R1_MD = """## 4. Results

### 4.1 How aligned is Ramen's update with the true-label gradient?

Each panel is one group of images; each curve is one direction compared with $g^{CE}$. Blue = Ramen's update,
orange = the image's own entropy gradient (no memory), grey = control (another image's $g^{CE}$)."""
R1_CODE = """plot_cosine_distributions(df);
display(direction_summary(df, df['zs_ok'].map({True: 'zero-shot right', False: 'zero-shot wrong'})).round(3))"""

R2_MD = """### 4.2 By outcome: images Ramen fixes, breaks, or leaves

Dumbbell per outcome group: orange = median cosine of the image's own entropy gradient, blue = of Ramen's update,
grey diamond = control. The line between them is what the memory changes."""
R2_CODE = """plot_outcome_dumbbell(df);
display(direction_summary(df, 'outcome').round(3))"""

R3_MD = """### 4.3 By domain"""
R3_CODE = """plot_domain_dumbbell(df);"""

R4_MD = """### 4.4 Over the stream

Rolling mean over stream position. The dashed line marks where fewer than half of the images still get their own
entry stored (Ramen's fp16 arrival counter saturates after 65,520 inserts and full queues stop updating)."""
R4_CODE = """freeze = memory_freeze_position(df)
print(f'own entry stored for {int((df.part_self != 0).sum()):,} of {len(df):,} images; '
      f'mostly frozen from position {freeze:,}' if freeze else 'memory never freezes')
plot_stream(df, window=WINDOW, freeze_at=freeze);"""

M1_MD = """## 5. Mechanism

### 5.1 Which memory entries make the update aligned?

The alignment $\\cos(\\hat g_i, g^{CE}_i)$ split exactly into the four parts of the update (section "Method"):
the image's own entry, other entries of the true-class queue, other entries of the wrong zero-shot-class queue, and
all other queues. Bars are means over images; the black tick is the total."""
M1_CODE = """display(decomposition(df).round(3))
plot_decomposition(df);"""

M2_MD = """### 5.2 Does alignment predict whether Ramen fixes or breaks an image?

Images are split into deciles of $\\cos(\\operatorname{sign} \\hat g_i, g^{CE}_i)$ (the step Ramen takes) within each
zero-shot group. Left: share of zero-shot errors Ramen fixes; middle: share of zero-shot hits it breaks; right:
median change of $\\log p(y_i)$ caused by the step. Counts in the end labels."""
M2_CODE = """plot_outcome_vs_alignment(df);
display(binned_outcome_rates(df).round(3))"""

H_MD = """## 6. Pre-registered hypotheses"""
H_CODE = """display(hypotheses(df))"""

GLOSSARY = r"""## 8. Glossary

| Term | Meaning |
|---|---|
| Ramen | Test-time adaptation for CLIP: per-class memory of past images' entropy gradients; each image takes one SignSGD step along a weighted average of retrieved gradients, predicts, and resets |
| zero-shot | CLIP's prediction without adaptation ($\hat y_i$); Ramen files each image in this class's queue |
| $g^{CE}$ (oracle gradient) | gradient of the cross-entropy toward the **true** label; uses ground truth, so it is a measuring stick, not something a method can use |
| own entropy gradient $g_i$ | gradient of the image's own prediction entropy; minimising it sharpens the current prediction, right or wrong |
| Ramen's update $\hat g_i$ | the retrieved, weighted average of stored entropy gradients that Ramen steps along |
| sign step | Ramen moves by $-\eta\,\mathrm{sign}(\hat g_i)$, so only the sign of each coordinate matters for the step |
| cosine similarity | $\langle a, b\rangle / (\lVert a\rVert \lVert b\rVert)$, from −1 (opposite) through 0 (unrelated) to +1 (same direction) |
| sign agreement | share of $\lvert g^{CE}\rvert$ mass on coordinates where the sign step moves the same way as the oracle; 0.5 = chance |
| shuffled control | cosine of an image's update with *another* image's $g^{CE}$; alignment shared by all images, not specific to the true class |
| own entry | the image's own memory entry, written before retrieval and retrieved at distance ≈ 0 |
| fixed / broken | zero-shot wrong → Ramen right / zero-shot right → Ramen wrong |
| fp16 freeze | Ramen's arrival counter is stored in fp16 and saturates after 65,520 images; full queues then stop accepting new entries |
| pooled accuracy | correct images / all images (here), as opposed to the domain-averaged accuracy used for headline numbers |
"""


def by_domain(s):
    """Section 4.3; a one-domain dataset has nothing to split, so it gets a note instead of a repeat of 4.1."""
    if s.get('one_domain'):
        return [md(R3_MD + '\n\nOne domain only, so this is the same as section 4.1; skipped.')]
    return [md(R3_MD), code(R3_CODE)]


def build(ds):
    s = SPECS[ds]
    cells = [md(header(ds, s)), md(math(s)),
             md(SETUP), code(SETUP_CODE.format(ds=ds)),
             md(DATA_MD), code(DATA_CODE),
             md(CHECK_MD), code(CHECK_CODE),
             md(R1_MD), code(R1_CODE), md(R2_MD), code(R2_CODE), *by_domain(s),
             md(R4_MD), code(R4_CODE.replace("WINDOW", f"{s['window']:_}")),
             md(M1_MD), code(M1_CODE), md(M2_MD), code(M2_CODE),
             md(H_MD), code(H_CODE),
             md(INTERPRETATION.get(ds, '## 7. Interpretation\n\n*To be written once the results are in.*')),
             md(GLOSSARY)]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nbf.write(nb, DATASETS[ds]['notebook'])
    print('wrote', DATASETS[ds]['notebook'])


if __name__ == '__main__':
    for ds in sys.argv[1:] or ['DomainNet']:
        build(ds)
