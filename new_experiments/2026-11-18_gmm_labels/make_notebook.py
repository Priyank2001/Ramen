"""
Write 01_gmm_labels.ipynb (not executed).

    python make_notebook.py
    ~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace 01_gmm_labels.ipynb
"""
import nbformat as nbf

from interpretation import INTERPRETATION

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

HEADER = r"""# Can a label-free Gaussian mixture over CLIP embeddings give better labels than zero-shot?

**Question.** Ramen files every cache entry under its zero-shot label. The label-value oracle (2026-11-01) showed the
cache can use better labels, but every label-free target tried so far (zero-shot, adapted prediction, kNN vote;
2026-11-04) failed, because none was much more accurate than zero-shot. Here a class-conditional mixture is fitted
by EM to the unlabelled image embeddings themselves, starting from the text prototypes. If the class structure is
in the image density, EM should move the labels towards it.

**Hypothesis (pre-registered 2026-11-18, before any result).** After 10 EM iterations with a uniform class prior,
the mixture's label is at least **10 points** more accurate than zero-shot (pooled, same images) on at least one of
DomainNet / ImageNet-C, and breaks fewer than **5%** of the images zero-shot gets right. That is the scale of the
oracle target that gave Ramen +0.75 on DomainNet (+11.8 points of target accuracy, 0 broken). Verdict per variant:
pass (both conditions) / weak (gain above the noise floor) / null (within ± floor) / hurts.

**Why this test.** It is the cheapest decisive one: offline, CPU only, on cached embeddings, and *transductive* (EM
sees every test image at once). This is the most favourable setting for a mixture; an online, causal version inside
Ramen can only do worse. If the label is not much better here, plugging it into Ramen's CE value is not worth a GPU run.

**Variants.**

| Run | Model | Class prior | Start | Uses true labels? |
|---|---|---|---|---|
| `zs` | – (0 iterations) | uniform | zero-shot posterior | no (identity check) |
| `vmf` | von Mises-Fisher, $\kappa=100$ fixed | uniform | zero-shot posterior | no |
| `vmf_prior` | von Mises-Fisher | learned | zero-shot posterior | no |
| `lda` | Gaussian, shared covariance, shrinkage 0.1 | uniform | zero-shot posterior | no |
| `lda_prior` | Gaussian, shared covariance | learned | zero-shot posterior | no |
| `vmf_oracle` | von Mises-Fisher | uniform | one-hot true labels, 1 iteration | **yes (ceiling)** |
| `lda_oracle` | Gaussian, shared covariance | uniform | one-hot true labels, 1 iteration | **yes (ceiling)** |

**Data.** Seed-0 cached zero-shot runs (mixed stream, batch 100): DomainNet (596,006 images, 345 classes, 6 domains,
CLIP ViT-B/32), ImageNet-C 5K (75,000 images, 1,000 classes, 15 corruptions at severity 5, ViT-B/16), EuroSAT (27,000
images, 10 classes, 1 domain, ViT-B/16). Inputs: `NoAdapt_feats` (fp16, unit norm), `NoAdapt_probs`, `NoAdapt.pkl`,
and the CLIP text features cached by 2026-11-01_pillow_sink.

**Seeds.** A seed sets Ramen's stream order. The EM fit uses all images at once and is deterministic, so it does
not depend on the seed (checked by a row-permutation test). One run per dataset; the Ramen noise floors
(0.11 DomainNet, 0.54 ImageNet-C, 1.51 EuroSAT) are used only as a minimum effect size.

**How produced.** `python run_gmm.py <dataset> 0` (CPU, minutes), then `python verify.py` (real-data checks) and
`python -m unittest test_gmm` (synthetic checks). Results: `new_experiments/results/<prefix>_gmm_<variant>.pkl`,
`_iters.pkl`, `_preds.npy`, `_means.npy`.

**What counts as a result.** The pre-registered rule above, at iteration 10. Other iterations are descriptive and
post-hoc.

**Sections.** 1 Setup · 2 Data · 3 Sanity and correctness · 4 Results · 5 Mechanism · 6 Interpretation · 7 Glossary
"""

MATH = r"""## Method

**Baseline (zero-shot, as in Ramen).** $z_i = f_{\theta_0}(x_i)/\lVert f_{\theta_0}(x_i)\rVert$, $w_c$ = normalised
text embedding of class $c \in \{1..C\}$,
$$p_{ic} = \operatorname{softmax}_c\big(100\, z_i^\top w_c\big), \qquad \hat y_i = \arg\max_c p_{ic}.$$
Ramen files $(z_i, g_i, H_i)$ in queue $\mathcal M_{\hat y_i}$; the label never enters its loss (notation as in the
skill's `ramen.md`). The question here is only whether a better $\hat y_i$ exists without labels.

**Zero-shot is already a mixture posterior.** For a von Mises-Fisher mixture with means $\mu_c = w_c$, one
concentration $\kappa = 100$ and uniform prior $\pi_c = 1/C$,
$$r_{ic} = \frac{\pi_c\, C_d(\kappa)\, e^{\kappa \mu_c^\top z_i}}{\sum_{c'} \pi_{c'}\, C_d(\kappa)\, e^{\kappa \mu_{c'}^\top z_i}} = p_{ic}.$$
So "initialise EM at the text prototypes" means $R^{(0)} = P$ (the cached zero-shot probabilities). It is soft, not
the hard zero-shot filing; the only new information EM adds is the shape of the unlabelled image density.

**EM** (iterations $t = 1..T$, $T = 10$; $N_c = \sum_i r^{(t-1)}_{ic}$, $S_c = \sum_i r^{(t-1)}_{ic} z_i$):

- *M-step, vMF:* $\mu_c = S_c / \lVert S_c \rVert$ ($\kappa$ fixed at 100).
- *M-step, LDA:* $\mu_c = S_c / N_c$,
  $\;S = \frac1N\big(\sum_i z_i z_i^\top - \sum_c S_c S_c^\top / N_c\big)$,
  $\;\Sigma = (1-s)\,S + s\,\frac{\operatorname{tr} S}{d} I$ with $s = 0.1$.
- *Prior:* uniform $\pi_c = 1/C$, or learned $\pi_c = N_c / N$.
- *E-step:* $r^{(t)}_{ic} \propto \pi_c\, p(z_i \mid c)$ with $\log p(z \mid c) = \kappa \mu_c^\top z$ (vMF, up to a
  shared constant) or $\log \mathcal N(z; \mu_c, \Sigma)$ (LDA). Label: $\hat y^{(t)}_i = \arg\max_c r^{(t)}_{ic}$.

A class whose mass $N_c$ falls below $10^{-8}$ keeps its previous mean. With $s = 0$ each iteration cannot lower the
log-likelihood $\sum_i \log \sum_c \pi_c p(z_i \mid c)$ (EM guarantee, tested); with $s = 0.1$ the M-step is shrunk,
so for LDA this is reported, not guaranteed.

**Oracle ceilings (use true labels $y_i$).** $R^{(0)}_{ic} = \mathbb 1[c = y_i]$, one M-step and one E-step: the
best this model family can label these images when its class models are fitted *on the test images with their true
labels*. An upper bound, not a method.

| Symbol | Meaning | Code | Config | Value |
|---|---|---|---|---|
| $R^{(0)}$ | starting responsibilities | `GMMLabelExperiment._start` | `start` | cached `NoAdapt_probs` / one-hot $y$ (oracle) |
| $\kappa$ | vMF concentration | `SphericalVMF.kappa` | `kappa` | 100 |
| $s$ | covariance shrinkage | `SharedCovGaussian.shrinkage` | `shrinkage` | 0.1 |
| $\pi$ | class prior | `UniformPrior` / `LearnedPrior` | `prior` | uniform / learned |
| $T$ | EM iterations | `EM.n_iter` | `n_iter` | 10 (oracle 1) |
"""

SETUP = r"""import os, sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
HERE = os.path.abspath('')
sys.path.insert(0, HERE)
from gmm_configs import DATASETS, VARIANTS, PASS_POINTS, MAX_BREAK_RATE
from gmm_experiment import GMMLabelExperiment
from plots import plot_gain, plot_fixed_broken, plot_iterations, plot_alignment
pd.set_option('display.width', 200)
exps = {ds: GMMLabelExperiment(ds, seed=0) for ds in DATASETS}
names = list(VARIANTS)
tables = {ds: e.table(names) for ds, e in exps.items()}
iters = {ds: {n: e.iters[n] for n in names} for ds, e in exps.items()}
def cnt(n, N): return f'{n:,} of {N:,} ({n / N:.1%})'
"""

DATA = r"""rows = []
for ds, e in exps.items():
    n = len(e.labels); right = int((e.zs_pred == e.labels).sum())
    rows.append(dict(dataset=ds, images=f'{n:,}', classes=e.P0.shape[1], domains=e.noadapt['domain'].nunique(),
                     zero_shot_right=cnt(right, n), zero_shot_wrong=cnt(n - right, n),
                     images_per_class=f'{n / e.P0.shape[1]:,.0f}'))
display(pd.DataFrame(rows))
for ds, e in exps.items():
    if e.noadapt['domain'].nunique() > 1:
        g = e.noadapt.groupby('domain', observed=True)['correct'].agg(['size', 'sum'])
        print(ds, ' | '.join(f'{d}: {int(r["sum"]):,} of {int(r["size"]):,} right' for d, r in g.iterrows()))
"""

SANITY_MD = r"""## 3. Sanity and correctness

Three layers of evidence, all re-run below:
1. **Real-data plumbing.** EM iteration 0 must reproduce the cached NoAdapt predictions on 100.00% of images, and our
   E-step $\operatorname{softmax}(100\,z^\top W)$ must match the model's cached probabilities except on near-ties
   (the cached features are fp16).
2. **Real-data algorithm checks** (`verify.py`): log-likelihood never decreases where EM guarantees it; the one-hot LDA
   fit equals sklearn's `LinearDiscriminantAnalysis` on all 27,000 EuroSAT images; the table's accuracies equal a
   plain-numpy recount from the saved prediction arrays; no shared repo file was modified.
3. **Synthetic unit tests** (`test_gmm.py`, 22 tests): every formula against an independent reference (scipy
   densities and softmax, sklearn LDA, brute-force loops), EM recovers a planted mixture from a poor start, and is
   invariant to chunking and row order.
"""

SANITY = r"""for ds in DATASETS:
    print(json.load(open(f'logs/sanity_{ds}_s0.json')))
print(open('logs/verify.log').read())
print(open('logs/unittest.log').read()[-2500:])
"""

RESULTS = r"""cols = ['variant', 'oracle', 'acc_domain_avg', 'acc_pooled', 'gain_pooled', 'fixed', 'broken', 'changed',
        'rescue_rate', 'break_rate', 'max_class_share', 'verdict']
for ds, t in tables.items():
    print(f'== {ds}  (zero-shot wrong {t.zs_wrong.iloc[0]:,}, right {t.zs_right.iloc[0]:,}; '
          f'floor {DATASETS[ds]["floor"]}; pass = +{PASS_POINTS:g} points and break rate < {MAX_BREAK_RATE:.0%})')
    d = t[cols].copy()
    d['fixed'] = [cnt(f, w) for f, w in zip(t.fixed, t.zs_wrong)]
    d['broken'] = [cnt(b, r) for b, r in zip(t.broken, t.zs_right)]
    display(d.round(2))
"""

MECH_MD = r"""## 5. Mechanism

Two questions. (a) Does accuracy improve or degrade as EM runs, and does the log-likelihood keep rising while it
does? (b) What do the components track: the true class, or the domain? Normalised mutual information (NMI) of the
predicted component with the class and with the domain, per iteration (multi-domain datasets only).
"""

MECH = r"""fig = plot_iterations(iters, 'acc_pooled', 'Label accuracy over EM iterations (dashed = zero-shot)'); plt.show()
fig = plot_iterations(iters, 'log_likelihood', 'Log-likelihood over EM iterations'); plt.show()
fig = plot_alignment(iters, 'lda', 'LDA components: NMI with class vs with domain'); plt.show()
fig = plot_alignment(iters, 'vmf', 'vMF components: NMI with class vs with domain'); plt.show()
rows = []
for ds, per in iters.items():
    for v in ('vmf', 'lda'):
        d = per[v].set_index('iteration')
        best = d['acc_pooled'].idxmax()
        rows.append(dict(dataset=ds, variant=v, acc_iter0=d.loc[0, 'acc_pooled'], acc_iter1=d.loc[1, 'acc_pooled'],
                         best_iter=best, acc_best=d.loc[best, 'acc_pooled'], acc_iter10=d.loc[10, 'acc_pooled'],
                         nmi_class_0=d.loc[0, 'nmi_class'], nmi_class_10=d.loc[10, 'nmi_class'],
                         nmi_domain_0=d.loc[0, 'nmi_domain'], nmi_domain_10=d.loc[10, 'nmi_domain']))
display(pd.DataFrame(rows).round(3))
"""

GLOSSARY = r"""## 7. Glossary

| Term | Meaning |
|---|---|
| zero-shot | CLIP's prediction $\arg\max_c z^\top w_c$ with no adaptation (cached NoAdapt run) |
| filing | the class queue Ramen writes an image into: its zero-shot prediction |
| EM | expectation-maximisation: alternately fit class models to soft assignments (M) and recompute the assignments (E) |
| vMF | von Mises-Fisher distribution on the unit sphere, $p(z\mid c)\propto e^{\kappa\mu_c^\top z}$; zero-shot is its posterior with $\mu_c = w_c$, $\kappa=100$ |
| LDA | Gaussian classes with one shared covariance (linear discriminant), here fitted without labels by EM |
| shrinkage | blending the covariance with a scaled identity to keep it well conditioned |
| learned prior | class frequencies $\pi_c = N_c/N$ re-estimated by EM (earlier work: drifts onto zero-shot's sink classes) |
| oracle | uses true labels; an upper bound, not a method |
| fixed / broken | zero-shot wrong → new label right / zero-shot right → new label wrong |
| rescue rate / break rate | fixed ÷ zero-shot wrong / broken ÷ zero-shot right |
| pooled / domain-averaged | accuracy over all images / mean of per-domain accuracies |
| NMI | normalised mutual information between two labelings (0 independent, 1 identical up to renaming) |
| noise floor | 2 × (max − min) of Ramen's accuracy over seeds 0–4; here a minimum effect size only |
| transductive | fitted on all test images at once (not causal / streaming) |
| label switching | EM finds a cluster but names it after the wrong class, because the start put that name nearest it |
"""

nb = nbf.v4.new_notebook()
nb.cells = [
    md(HEADER), md(MATH),
    md('## 1. Setup'), code(SETUP),
    md('## 2. The data the code ran on'), code(DATA),
    md(SANITY_MD), code(SANITY),
    md('## 4. Results\n\nOne row per variant at the final iteration. Counts are images; rates are fixed ÷ zero-shot '
       'wrong and broken ÷ zero-shot right. Gains are pooled points over zero-shot on the same images.'),
    code(RESULTS),
    code("fig = plot_gain(tables, PASS_POINTS); plt.show()\nfig = plot_fixed_broken(tables); plt.show()"),
    md(MECH_MD), code(MECH),
    md(INTERPRETATION), md(GLOSSARY),
]
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, '01_gmm_labels.ipynb')
