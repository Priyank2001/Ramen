"""Writes and executes prior_estimation.ipynb: python build_notebook.py [interpretation.md]. Needs prior_estimation.py's CSVs."""
import os
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

HEADER = r"""# Prior correction toward an estimated class marginal: is the right target worth it, and can we estimate it?

**Question.** Prior correction (pc) failed the held-out test on ImageNet-R/A/Sketch because its uniform target is wrong there: 314 of 1,000 classes carry 60.5% of the images. (a) How much would pc gain with the *true* class marginal as its target? (b) Can a label-free estimate of that marginal get some of it back?

**Hypothesis (pre-registered, `PREREGISTRATION.md`, written before any estimator was computed).** Online EM on the zero-shot probabilities (zs-EM) gives a target with which T3 + pc beats T3 by more than the floor on ImageNet-R/A/Sketch, beats T3 on ImageNet-C and DomainNet, and never costs Ramen more than the floor on any of six datasets.

**Why this test.** Exact and offline: pc acts only on output probabilities, which never feed back into Ramen's cache, so every variant is a post-hoc transform of the saved runs (no GPU). The oracle target gives a ceiling before an estimator is judged.

**Variants.** pc(target): logit − λ·log(π_t / q_t), λ = 1, π_t = EMA (m = 0.99) of the method's soft predicted marginal (frozen 2026-10-25). Only the target q_t changes:

| Target | What q_t is | Uses true labels? |
|---|---|---|
| uniform | 1/K (the existing pc) | no |
| oracle | true label marginal of the whole stream | **yes (oracle, ceiling only)** |
| zs-EM (**primary**) | online Saerens EM on the zero-shot probabilities: r ∝ p_zs ⊙ q, q ← 0.99 q + 0.01 mean_batch(r) | no |
| zs-marginal | EMA (0.99) of the zero-shot soft marginal (undo only the drift adaptation adds) | no |
| self-EM | the same online EM on the method's own probabilities | no |

Bases: **T3** (RamenActive, zero-shot-weighted active set, temperature 3) on ImageNet-C 5K, DomainNet and ImageNet-R/A/Sketch, where its probabilities were saved; **Ramen** on all six datasets.

**Data.** Six mixed streams, batches of 100 (table in section 2). Zero-shot probabilities come from the seed-0 NoAdapt run, realigned to each seed's stream order.

**Seeds.** 0–4: stream order and Ramen's selection randomness; images identical. Gains are paired: base + pc vs the same base run.

**How the results were produced.** `python prior_estimation.py <dataset> 0 1 2 3 4` (CPU, ~1–15 min per dataset) → `results/prior_estimation/<ds>_s<seed>_{gains,per_domain,diag}.csv`.

**What counts as a result.** Gain over the same base, domain-averaged, mean over 5 seeds, against the noise floor (2 × Ramen's range over seeds). Decision 1: if even the oracle target does not beat T3 by the floor on ImageNet-R/A/Sketch, the direction is closed. Decision 2: the primary claim above. None of these datasets is held out any more, so a pass would still need a fresh dataset.

**Sections.** 1 Setup · 2 Data · 3 Sanity checks · 4 Pre-registered decisions · 5 All gains · 6 Why the estimators fail: how far each target is from the truth · 7 Per domain · 8 Interpretation · 9 Glossary"""

GLOSSARY = r"""## 9. Glossary

| Term | Meaning |
|---|---|
| Ramen | CLIP zero-shot + per-class memory of (feature, entropy gradient); one signSGD step per batch from retrieved entries, predict, reset |
| T3 | Ramen with a zero-shot-weighted active set: each class queue's vote weighted by the query's zero-shot probability at temperature 3 |
| pc (prior correction) | subtract λ·log(π_t / q_t) from the logits: push down classes predicted more often than the target says they occur |
| π_t | EMA of the method's own soft predicted class marginal up to batch t |
| target q_t | the class marginal pc pulls predictions toward |
| oracle | uses true labels; an upper bound, not a method |
| Saerens EM | label-shift estimation: reweight posteriors by the current prior estimate, re-estimate the prior as their mean, repeat; here one step per batch with an EMA |
| TV distance | total variation, ½ Σ\|a − b\|: 0 = identical distributions, 1 = disjoint |
| noise floor | 2 × (max − min) of Ramen's domain-averaged accuracy over seeds 0–4 |
| domain-averaged accuracy | mean of per-domain accuracies |
| fixed / broken | images the base gets wrong and the variant right / and the reverse |
| cluster bootstrap CI | 95% interval from resampling images (Poisson weights; base image on corruption sets), per seed |"""

CELLS = [
    md(HEADER),
    md('## 1. Setup'),
    code("""import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '..'); sys.path.insert(0, '../2026-10-25_stacking')
import numpy as np, pandas as pd
import stack, prior_estimation as pe
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
t = pe.collect(); d = pe.collect('diag')
t.groupby('dataset').seed.nunique()"""),
    md('## 2. The data the code ran on'),
    code("stack.data_table(pe.DATASETS)"),
    code("""rows = {}
for ds in pe.DATASETS:
    z = pd.read_pickle(stack.path(ds, 0, 'NoAdapt') + '.pkl')
    c = np.bincount(z.label, minlength=z.label.max() + 1)
    k = len(c); top = np.sort(c)[::-1]
    rows[stack.LABEL[ds]] = {'classes': k, 'images per class (min / median / max)': f'{c.min():,} / {int(np.median(c)):,} / {c.max():,}',
                             'share of images in the 10% largest classes': stack.fmt(int(top[:max(1, k // 10)].sum()), len(z)),
                             'TV(true marginal, uniform)': round(0.5 * np.abs(c / len(z) - 1 / k).sum(), 3)}
pd.DataFrame(rows).T  # how far each dataset is from the uniform target pc assumes"""),
    md("""## 3. Sanity checks

- `prior_estimation.run_seed` asserts that pc with the uniform target gives **identical** predictions to the stacking code's pc(λ = 1) (`stack.prior_correct`) on every seed and base. It ran to completion, so they match.
- `stack.load` asserts that every saved softmax reproduces its run's predictions; `zero_shot_probs` asserts the same for NoAdapt.
- The table below reproduces the published pc(uniform, λ = 1) numbers: T3 + pc on ImageNet-R/A/Sketch vs Ramen −1.15 (2026-10-25)."""),
    code("""u = t[t.target == 'uniform'].groupby(['dataset', 'base'])['gain vs Ramen'].mean().unstack().round(2)
u.index = u.index.map(stack.LABEL); u"""),
    md('## 4. Pre-registered decisions'),
    code("s = pe.summary(t); pe.verdict(s)"),
    md('## 5. All gains (base + pc[target] minus the same base without pc)'),
    code("fig = pe.plot_gains(t)"),
    code("""show = s.copy(); show['dataset'] = show.dataset.map(stack.LABEL)
show.set_index(['dataset', 'base', 'target'])[['gain mean', 'gain std', 'gain min', 'floor', 'above floor', 'CI > 0', 'gain vs Ramen', 'images vs base']].astype(
    {'gain mean': float, 'gain std': float, 'gain min': float, 'gain vs Ramen': float, 'images vs base': float}).round(2)"""),
    md("""## 6. Why the label-free estimators fail: how far each target ends up from the truth

Total-variation distance between the true class marginal and each target at the end of the stream (mean over seeds; labels used only to measure). A target only helps if it is closer to the truth than the method's own predicted marginal. **Lower is better.**"""),
    code("pe.target_distance(d).round(3)"),
    md("""How often each correction changes the base's prediction, and how far the corrected predictions' marginal is from the truth (mean over seeds)."""),
    code("""cols = {}
for ds, g in d.groupby('dataset', sort=False):
    for b in ('T3', 'Ramen'):
        if f'TV(true, {b} hard marginal)' not in g or g[f'TV(true, {b} hard marginal)'].isna().all():
            continue
        r = {'no pc': g[f'TV(true, {b} hard marginal)'].mean()}
        r.update({x: g[f'TV(true, {b} + pc[{x}] hard marginal)'].mean() for x in pe.TARGETS})
        r.update({f'flips % [{x}]': g[f'{b} + pc[{x}] flips %'].mean() for x in pe.TARGETS})
        cols[(stack.LABEL[ds], b)] = r
pd.DataFrame(cols).T.round(3)  # first block: TV(true, predicted hard marginal); second: % of predictions changed"""),
    md("""### ImageNet-R/A/Sketch: share of predictions on the 314 R/A classes (true share 60.5%)"""),
    code("""ds = 'ImageNetVariants'
df, P = stack.load(ds, 0, 'active_prob_T3'); Pzs = pe.zero_shot_probs(ds, df)
y, b = df.label.values, df.batch.values
ra = np.zeros(1000, bool); ra[np.unique(y[df.domain.astype(str).values != 'sketch'])] = True
rows = {'true labels': ra[y].mean() * 100, 'zero-shot predictions': ra[Pzs.argmax(1)].mean() * 100, 'T3 predictions': ra[P.argmax(1)].mean() * 100}
for x in pe.TARGETS:
    Q = pe.target_series(x, df, P, Pzs)
    rows[f'target {x} (end of stream)'] = Q[-1][ra].sum() * 100
    rows[f'T3 + pc[{x}] predictions'] = ra[pe.prior_correct(P, b, Q)].mean() * 100
pd.Series(rows, name='% on R/A classes (seed 0)').round(1).to_frame()"""),
    md("""### Post-hoc: how much of the oracle is attainable knowledge, and how much is exact counts?

The full oracle knows the exact count of every class. On ImageNet-C 5K the 5,000 images are a random draw, so per-class counts run from 0 to 12 although the population marginal is uniform; there the oracle's gain is **entirely** count knowledge no estimator could have. On ImageNet-R/A/Sketch most of the skew is structure. The **structure oracle** (post-hoc, uses labels) knows only each domain's size and class set and assumes classes are equally common within a domain."""),
    code("""z = pd.read_pickle(stack.path('ImageNetC5K', 0, 'NoAdapt') + '.pkl')
c = np.bincount(z[z.domain_idx == 0].label, minlength=1000)
print(f'ImageNet-C 5K, images per class within one corruption: min {c.min()}, max {c.max()}, classes absent {(c == 0).sum()}')
so = pd.read_csv(pe.OUT + 'ImageNetVariants_structure_oracle.csv')
full = t[(t.dataset == 'ImageNetVariants') & (t.target == 'oracle')]
pd.DataFrame({'full oracle gain': full.groupby('base').gain.mean(), 'structure oracle gain (post-hoc)': so.groupby('base').gain.mean(),
              'structure oracle std': so.groupby('base').gain.std(), 'TV(true, structure target)': so.groupby('base')['TV(true, target)'].mean()}).round(3)"""),
    md('## 7. Per domain\n\nGain over the same base per domain (pts, mean over seeds), T3 base where available.'),
    code("""pdm = pe.collect('per_domain')
pdm['base'] = pdm.variant.str.split(' + ', regex=False).str[0]
pdm = pdm[pdm.variant.str.contains('pc')].merge(pdm[~pdm.variant.str.contains('pc')][['dataset', 'seed', 'domain', 'base', 'acc']],
                                               on=['dataset', 'seed', 'domain', 'base'], suffixes=('', '_base'))
pdm['gain vs base'] = pdm.acc - pdm.acc_base
pdm['target'] = pdm.variant.str.extract(r'\\[(.*)\\]')[0]
for ds in ['ImageNetVariants', 'DomainNet']:
    print(stack.LABEL[ds])
    display(pdm[(pdm.dataset == ds) & (pdm.base == 'T3')].groupby(['domain', 'target'])['gain vs base'].mean().unstack()[pe.TARGETS].round(2))"""),
    md('## 8. Interpretation\n\nINTERPRETATION'),
    md(GLOSSARY),
]


def build(interpretation):
    nb = nbformat.v4.new_notebook()
    nb.cells = CELLS
    nb.cells[-2] = md('## 8. Interpretation\n\n' + interpretation)
    nb.metadata['kernelspec'] = {'name': 'ramen-env-kernel', 'display_name': 'Python 3 (RAMEN)', 'language': 'python'}
    ExecutePreprocessor(timeout=1800, kernel_name='ramen-env-kernel').preprocess(nb, {'metadata': {'path': HERE}})
    out = os.path.join(HERE, 'prior_estimation.ipynb')
    nbformat.write(nb, out)
    return out


if __name__ == '__main__':
    print(build(open(sys.argv[1]).read() if len(sys.argv) > 1 else '(written after the results)'))
