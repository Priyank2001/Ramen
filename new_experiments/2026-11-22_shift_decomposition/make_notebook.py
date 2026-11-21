"""
Write 01_shift_decomposition.ipynb (not executed).

    python make_notebook.py
    ~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace 01_shift_decomposition.ipynb
"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

HEADER = r"""# Shift decomposition: how much is appearance, concept, and label shift worth?

**Question.** Test-time shifts are not only noise: there are domain/style shifts and label shifts too. On each dataset, how much
accuracy is available from fixing **(A)** a class-independent domain offset in CLIP's embedding ("appearance"), **(Γ)** a
class-specific domain offset ("concept"), and **(L)** the per-domain class frequencies ("label shift")? And is any of these larger than Ramen's own gain?

**Hypothesis / predictions (pre-registered in `PREREGISTRATION.md`, written before any number).** ImageNet-C: A dominates.
DomainNet: Γ dominates (largest in sketch / quickdraw). ImageNet-R/A/Sketch: L dominates. EuroSAT: φ_A = φ_Γ = 0 exactly (bug check).

**Why this test.** Oracle ceilings on cached features: if a fix that is *given* the true labels/domains cannot beat Ramen, no label-free
version of it can, so the direction can be dropped before building a method. Offline, CPU only, minutes per dataset.

**Every cell below is an ORACLE** (uses true labels and/or true domain ids). None is a method.

| cell | prototypes | features | prior | needs |
|---|---|---|---|---|
| `txt_` | text $w_c$ (zero-shot) | $z$ | uniform | nothing |
| `txt_A` | text | $\mathrm{norm}(z-\hat\alpha_d)$ | uniform | labels + domains (α from the class-adjusted fit) |
| `txt_Ad` | text | $\mathrm{norm}(z-(\bar z_d-\bar z))$ | uniform | domains only |
| `txt_L`, `txt_AL`, `txt_AdL` | as above | as above | true $\pi_{c\mid d}$ | + labels |
| `img_` (P0) | pooled image class means $\hat\mu+\hat\beta_c$ | $z$ | uniform | labels |
| `img_A`, `img_G`, `img_AG` | $+\,\tilde\gamma_{c,d}$ if Γ | $z-\hat\alpha_d$ if A | uniform | labels + domains |
| `img_*L` | as above | as above | true $\pi_{c\mid d}$ | labels + domains |
| `img_G_n0`, `img_AG_n0` | unshrunk $\hat\gamma$ | | | robustness |
| `rnd_A`, `rnd_G`, `rnd_AG` | fitted with **random** domain labels | | | control (should ≈ P0) |
| `cf_*` | fitted on the other half of the photos/images (2-fold cross-fit) | | | estimation-noise check |

**Data.** Seed 0 cached CLIP features (`results/<ds>_..._s0_NoAdapt_feats.npy`, fp16), labels, domains; Ramen's cached seed-0 predictions.
ImageNet-C 5K (75,000 images, 15 corruptions, ViT-B/16), DomainNet (596,006, 6 domains, ViT-B/32), ImageNet-R/A/Sketch
(88,389, 3 domains, 1,000-class head, ViT-B/16), EuroSAT (27,000, 1 domain, ViT-B/16).
**Seeds.** A seed only changes the stream order; these oracles do not depend on order, so one run covers every seed. Uncertainty comes from a
cluster bootstrap over images (cluster = photo on ImageNet-C), B = 1000, estimates held fixed; estimation noise is checked by the cross-fit.

**How produced.** `python run_shift_decomposition.py <dataset>` (CPU, 16 threads), cached in `results/<dataset>_s0.{npz,pkl}`.
**What counts as a result.** Pre-registered rules: a component *dominates* if its Shapley value is ≥ 50% of the total (image world) with a CI
excluding 0; the domain-aware direction is *alive* on a dataset iff φ_A + φ_Γ (image world) > Ramen's gain over NoAdapt.
All accuracies are **domain-averaged** (mean of per-domain accuracies).

**Sections.** 1 Setup · 2 Data · 3 Sanity checks · 4 Geometry (ANOVA) · 5 Oracle cells · 6 Shapley attribution · 7 Per domain ·
8 Robustness & controls · 9 Decision · 10 Interpretation · 11 Glossary
"""

MATH = r"""## Method (mathematical formulation)

**Notation.** Image $i$ has frozen CLIP feature $z_i\in\mathbb S^{D-1}$ (the NoAdapt feature of Ramen's notation, $\theta_0$), true label
$y_i$ (**oracle**), domain $d_i$ (**oracle**), leave-out group $g(i)$ (the photo on ImageNet-C, else the image). $w_c$ = normalised text
embedding. Zero-shot (= NoAdapt): $\hat y_i=\arg\max_c z_i^\top w_c$. Ramen classifies with its adapted feature $\tilde z_i$ (`references/ramen.md`).

**Embedding model** (unbalanced two-way MANOVA, weighted least squares on cell sums; `anova.AdditiveFit`):
$$z_i=\mu+\alpha_{d_i}+\beta_{y_i}+\gamma_{y_i,d_i}+\varepsilon_i,\qquad \textstyle\sum_d n_d\alpha_d=0,\ \sum_c n_c\beta_c=0 .$$
$(\mu,\alpha,\beta)$ = least-squares additive fit (backfitting), $\gamma_{c,d}$ = cell mean − additive fit. $\alpha_d$: appearance, $\gamma_{c,d}$: concept, $\pi_{c\mid d}$: label shift.

**Part A (geometry).** $SS_A=\sum_d n_d\lVert\hat\alpha_d\rVert^2$, $SS_B=\sum_c n_c\lVert\hat\beta_c\rVert^2$, $SS_\Gamma=\sum_{c,d}n_{c,d}\lVert\hat\gamma_{c,d}\rVert^2$,
$SS_W=\sum_i\lVert z_i-\hat M_{y_i,d_i}\rVert^2$; per-dimension $\sigma^2=SS_W/(D(N-\#\text{cells}))$; debiased $SS_X^*=SS_X-\mathrm{df}_X D\sigma^2$ with
$\mathrm{df}_A=|\mathcal D|-1$, $\mathrm{df}_B=C-1$, $\mathrm{df}_\Gamma=\#\text{cells}-C-|\mathcal D|+1$; $F_X=SS_X/(D\,\mathrm{df}_X\sigma^2)$.

**Part B (oracle classifiers; `oracles.py`).** Image world, $S\subseteq\{A,\Gamma,L\}$:
$$\text{score}_c(z_i)=\kappa_S\Big(\tilde z_i^\top m_{c,d_i}-\tfrac12\lVert m_{c,d_i}\rVert^2\Big)+\mathbb 1[L\in S]\log\big(C\,\hat\pi_{c\mid d_i}\big),\quad
\tilde z_i=z_i-\mathbb 1[A\in S]\hat\alpha_{d_i},\quad m_{c,d}=\hat\mu+\hat\beta_c+\mathbb 1[\Gamma\in S]\tilde\gamma_{c,d}.$$
(the isotropic-Gaussian Bayes classifier = nearest class mean). With both A and Γ, $\lVert\tilde z_i-m_{c,d}\rVert=\lVert z_i-\hat M_{c,d}\rVert$ exactly.
Text world: $\text{score}_c=\tau\,z_i'^\top w_c+\mathbb 1[L]\log(C\hat\pi_{c\mid d_i})$, $z_i'=z_i$ or $\mathrm{norm}(z_i-\hat\alpha_{d_i})$.

**Honesty rules.** (1) Leave-one-out: image $i$'s true-class prototype $\hat\beta_{y_i}$, $\hat\gamma_{y_i,d_i}$ is recomputed without all of
$g(i)$ (closed-form downdate, $\hat\mu,\hat\alpha$ fixed); prior $\hat\pi^{(-i)}_{c\mid d}=(n_{c,d}-\mathbb 1[y_i=c]+1)/(n_d-1+C)$.
(2) Shrinkage $\tilde\gamma_{c,d}=\frac{n_{c,d}}{n_{c,d}+n_0}\hat\gamma_{c,d}$, $n_0=\sigma^2/\hat\tau^2$, $\hat\tau^2=(SS_\Gamma/D-\mathrm{df}_\Gamma\sigma^2)/N$ (method of moments).
(3) $\kappa_S,\tau$ by NLL (temperature scaling) on 20,000 images per prototype configuration, before adding $L$.

**Attribution.** Shapley value over the $2^3$ cells, $\phi_k=\sum_{S\subseteq N\setminus k}\frac{|S|!(n-|S|-1)!}{n!}[\mathrm{Acc}(S\cup k)-\mathrm{Acc}(S)]$,
$\sum_k\phi_k=\mathrm{Acc}(N)-\mathrm{Acc}(\varnothing)$. Text world: over $\{A,L\}$. Text-prototype gap $=\mathrm{Acc}(P0)-\mathrm{Acc}(T)$.

| symbol | meaning | code | value |
|---|---|---|---|
| $\hat\alpha,\hat\beta,\hat\gamma$ | effects | `AdditiveFit.alpha/beta/gamma` | fitted |
| $n_0$ | shrinkage pseudo-count | `AdditiveFit.n0` | per dataset (table in §4) |
| $\kappa_S,\tau$ | temperatures | `Evaluator.fit_temperature` | `meta['kappa']` |
| $g(i)$ | leave-out group | `ShiftData.group` | photo (ImageNet-C), image |
| B | bootstrap replicates | `ClusterBootstrap` | 1000 |
"""

SETUP = """import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np, pandas as pd
pd.set_option('display.precision', 2); pd.set_option('display.width', 200)
from experiment import ShiftDecomposition
from analysis import Summary, overview
from data import LABEL
import plots
DATASETS = ['ImageNetC5K', 'DomainNet', 'ImageNetVariants', 'EuroSAT']
S = {ds: Summary(ShiftDecomposition(ds).load()) for ds in DATASETS}"""

DATA = """rows = []
for ds, s in S.items():
    t = s.e.meta['describe'].copy(); t.insert(0, 'dataset', LABEL[ds])
    t['zero_shot_errors'] = [f'{e:,} of {n:,} ({e / n:.1%})' for e, n in zip(t.zero_shot_errors, t.images)]
    rows.append(t)
pd.concat(rows).reset_index(drop=True)"""

SANITY = """pd.DataFrame({LABEL[ds]: s.e.meta['sanity'] for ds, s in S.items()}).T"""

ANOVA = """rows = []
for ds, s in S.items():
    for r in s.e.meta['anova']:
        rows.append(dict(dataset=LABEL[ds], **r))
    rel = s.e.meta['reliability']
    print(f"{LABEL[ds]:22s} n0 = {s.e.meta['n0']:.3g}   gamma split-half r = {rel['r']:.3f}, Spearman-Brown {rel['r_full']:.3f} "
          f"({rel['cells']:,} cells)")
pd.DataFrame(rows)"""

CELLS = """for ds, s in S.items():
    print('==', LABEL[ds]); display(s.cells().round(2))"""

SHAP = """for ds, s in S.items():
    print('==', LABEL[ds]); display(s.shapley().round(2))"""

PERDOM = """for ds in ['ImageNetC5K', 'DomainNet', 'ImageNetVariants']:
    print('==', LABEL[ds]); display(S[ds].per_domain().round(2))"""

ROBUST = """rows = []
for ds, s in S.items():
    if s.e.meta['domain_names'] and len(s.e.meta['domain_names']) > 1:
        a = s.acc
        rows.append(dict(dataset=LABEL[ds], P0=a['img_'], A=a['img_A'] - a['img_'], G=a['img_G'] - a['img_'],
                         AG=a['img_AG'] - a['img_'], rnd_P0=a['rnd_'], rnd_A=a['rnd_A'] - a['rnd_'],
                         rnd_G=a['rnd_G'] - a['rnd_'], rnd_AG=a['rnd_AG'] - a['rnd_'],
                         G_unshrunk=a['img_G_n0'] - a['img_'],
                         cf_P0=a['cf_'], cf_AG=a['cf_AG'] - a['cf_'], cf_AGL=a['cf_AGL'] - a['cf_']))
pd.DataFrame(rows).round(2)"""

CF_SHAP = """for ds, s in S.items():
    print('==', LABEL[ds], '(cross-fit)'); display(s.shapley(prefix='cf').round(2))"""


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(HEADER), md(MATH),
        md('## 1. Setup'), code(SETUP),
        md('## 2. The data the code ran on\nImages, classes present and median images per class in each domain; zero-shot errors as counts and %.'),
        code(DATA),
        md('## 3. Sanity checks\n`txt_` must reproduce the cached NoAdapt predictions on ≥ 99.4% of images (fp16 features); on EuroSAT '
           '(one domain) every A / Γ cell must equal P0 on 100.00%.'),
        code(SANITY),
        md('## 4. Geometry: where the embedding variance sits (Part A)'), code(ANOVA),
        code("plots.variance_shares(DATASETS);"),
        md('## 5. Oracle cells\nDomain-averaged accuracy of every cell; `vs_zero_shot` and its 95% CI are relative to `txt_` (zero-shot).'),
        code(CELLS),
        md('## 6. Shapley attribution\nImage world over {A, G, L}, text world over {A, L}, text-prototype gap P0 − T, and Ramen − NoAdapt (cached).'),
        code(SHAP), code("plots.shapley_vs_ramen(DATASETS);"),
        md('## 7. Per domain\nImage-world Shapley values per domain (point estimates) next to Ramen\'s per-domain gain.'),
        code(PERDOM), code("plots.per_domain_shapley('DomainNet');"), code("plots.per_domain_shapley('ImageNetC5K');"),
        md('## 8. Robustness and controls\nGains over P0 (points): true domains vs **random domains** (control; relative to the random-domain fit P0 `rnd_`), unshrunk γ, and the 2-fold cross-fit.'),
        code(ROBUST), code(CF_SHAP),
        md('## 9. Pre-registered decision'), code("pd.DataFrame([s.decision() for s in S.values()])"),
        code("overview(DATASETS).round(2)"),
        md(INTERP),
        md(GLOSSARY),
    ]
    nbf.write(nb, '01_shift_decomposition.ipynb')


GLOSSARY = r"""## 11. Glossary

| term | meaning |
|---|---|
| oracle | uses true labels and/or domains; an upper bound, not a method |
| appearance (A) | $\alpha_d$: the part of the domain change shared by all classes (a translation of the whole domain in embedding space) |
| concept (Γ) | $\gamma_{c,d}$: the class-specific part of the domain change (class $c$ moves differently in domain $d$) |
| label shift (L) | different class frequencies $\pi_{c\mid d}$ per domain |
| P0 | pooled image prototypes (class means over all domains): the image-world baseline |
| text-prototype gap | P0 − zero-shot: how much better image class means are than CLIP's text embeddings, with no shift fixed |
| Shapley value | average gain from adding one fix, over all orders of adding the fixes; the values sum to (all fixes − none) |
| leave-one-out / group | image $i$'s own class statistics exclude $i$ (and its corrupted copies on ImageNet-C) so it cannot vote for itself |
| shrinkage $n_0$ | γ is multiplied by $n/(n+n_0)$: cells with few images are pulled toward "no interaction" |
| cross-fit | statistics estimated on one half of the images, accuracy measured on the other |
| random-domain control | same fits with domain labels shuffled; any gain there is not a domain effect |
| domain-averaged accuracy | mean over domains of each domain's accuracy (the project's headline convention) |
| noise floor | 2 × (max − min) of Ramen's accuracy over seeds 0–4 |
"""

INTERP = r"""## 10. Interpretation

**Predictions (pre-registered).** ImageNet-C → A: **confirmed** (φ_A +1.94, CI [1.78, 2.09]; Γ +0.19; L −1.24, so the
"share" numbers are meaningless there and A is simply the only sizeable positive component). DomainNet → Γ: **confirmed**
(φ_Γ +5.25 of +8.62, 61%; quickdraw alone +20.5). ImageNet-R/A/Sketch → L: **narrowly fails** (L is 49.2% < 50%, Γ 34%,
so no single component dominates by the rule). EuroSAT A = Γ = 0: **confirmed** (identical predictions, 100.00%).

**Decision rule (φ_A + φ_Γ vs Ramen's gain).** DomainNet 8.12 vs 1.43 and R/A/Sketch 5.14 vs 1.50: **domain-aware direction
alive**. ImageNet-C 2.13 vs 5.79 and EuroSAT 0 vs 14.56: **not alive**.

**Read against zero-shot, not only P0.** Image-world gains start from pooled image prototypes, which are *below* zero-shot on
ImageNet-C (−7.90: ~4 other photos per class after leaving the photo out) and DomainNet (−4.86). Per-domain class means
(`img_AG`) vs zero-shot: DomainNet +2.76 (+3.75 with L), R/A/Sketch +9.81 (+14.25), ImageNet-C −5.64. So on DomainNet,
concept-aware prototypes are worth about 2× Ramen's gain over zero-shot; on ImageNet-C they are worse than the text prompt.

**What the controls say.** Random domain labels give +0.00 to +0.02 over their own P0 on all three multi-domain datasets, so
A and Γ gains are domain effects. The 2-fold cross-fit reproduces the Shapley values (DomainNet Γ 5.08, A 2.89; R/A/Sketch
Γ 2.77); γ split-half reliability is high on DomainNet (0.94) and R/A/Sketch (0.81), moderate on ImageNet-C (0.46).

**Mechanistic reading.** On corruptions, the domain change is mostly a common offset (α 26% of embedding variance, γ 2%, F 1.17),
and Ramen's +5.79 is far above every mean-based ceiling: its gain must come from something a mean shift of the frozen feature
cannot do (re-encoding the image with adapted norm layers; consistent with, not proven by, this result). On DomainNet the
class-specific part is real (γ 9% of variance, F 58) and concentrated in quickdraw, where Ramen's sink problems also live.

**Caveats.** Oracle ceilings (true labels and domains); seed-independent; one feature extractor per dataset; means only
(Γ is a lower bound on concept shift); L in the text world assumes CLIP's implicit prior is uniform, and it hurts on
ImageNet-C / EuroSAT.

**Next.** Step 2 screen on DomainNet and R/A/Sketch: can label-free domain inference (style statistics) plus label-free
per-domain prototypes recover a useful fraction of `img_AG − txt_` (+2.76 DomainNet)? Zero-shot pseudo-labels already failed as
a CE target, so the per-domain prototype estimate is the hard part, not the domain id (text-world A with domain means
`txt_Ad` ≈ the class-adjusted oracle `txt_A`).
"""


if __name__ == '__main__':
    build()
