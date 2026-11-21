"""Writes and executes heldout.ipynb: python build_notebook.py [interpretation.md]. Needs heldout_analysis.py's CSVs."""
import os
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

HEADER = r"""# Held-out test: does T3 + prior correction (λ = 1) beat Ramen on ImageNet-R / -A / -Sketch?

**Question.** T3 + pc(λ = 1) was assembled on ImageNet-C and DomainNet (λ = 1 picked post-hoc there), then frozen and checked not to break CIFAR / EuroSAT. Does it beat Ramen on data none of its settings were tuned on?

**Hypothesis (pre-registered, `PREREGISTRATION.md`, written before any model saw these images).** Mean gain of T3 + pc(λ = 1) over Ramen, domain-averaged, seeds 0–4, is above the noise floor of this stream.

**Variants** (true labels are used only for scoring):

| Name | Role | What it is |
|---|---|---|
| NoAdapt | reference | CLIP ViT-B/16 zero-shot, default 7-prompt ensemble |
| Ramen | baseline | `cfg/ImageNetVariants/Ramen.yaml` = ImageNet-C config (capacity 75, topk 1, β 0, signSGD lr 0.01) |
| T3 + pc(lam=1) | **primary** | RamenActive `active_set: prob`, `temperature: 3`, then logit − 1·log(π_t / uniform), π_t = EMA (m = 0.99) of the predicted class marginal |
| T3, Ramen + pc(lam=1) | secondary | the two halves of the primary |
| T3 + pc(lam=2), pc(lam=0.5), knn variants | **post-hoc** | for context only |

**Data.** One shuffled mixed stream, batches of 100, over **all 1,000 ImageNet classes**: ImageNet-R (30,000 renditions, 200 classes present), ImageNet-A (7,500 natural adversarial photos, 200 classes), ImageNet-Sketch (50,889 sketches, 1,000 classes). Labels are remapped from WordNet ids to the ImageNet-1k index. Because of the 1,000-way head, R / A accuracies are **not comparable** to the usual 200-class numbers in the literature. Downloads were verified by size, by SHA-256 (Sketch), and by image counts.

**Seeds.** 0–4: stream order and Ramen's selection randomness; images identical. NoAdapt is stateless (seed 0 only).

**How the runs were produced.** `bash run_queue.sh` (detached, one worker per GPU) runs `run_heldout.py <seed> <run>` with probabilities saved. Then `python heldout_analysis.py` → `results/heldout/*.csv`.

**What counts as a result.** Primary mean gain > noise floor = 2 × (max − min) of Ramen's domain-averaged accuracy over seeds 0–4 on this stream (computed below).

**Sections.** 1 Setup · 2 Data · 3 Sanity checks · 4 Primary result · 5 Secondary and post-hoc · 6 Per domain · 7 Interpretation · 8 Glossary"""

GLOSSARY = r"""## 8. Glossary

| Term | Meaning |
|---|---|
| Ramen | CLIP zero-shot + per-class memory of (feature, entropy gradient); one signSGD step per batch from retrieved entries, predict, reset |
| T3 | zero-shot-weighted active set: each class queue's vote weighted by the query's zero-shot probability at temperature 3 |
| pc (prior correction) | subtract λ·log(estimated predicted-class share / uniform) from the logits |
| knn | 50/50 mix of a prediction with the predictions on the 10 most similar earlier images (post-hoc here) |
| held-out | a dataset on which no setting was chosen |
| pre-registered | the setting, metric and decision rule were written down before the data was run |
| noise floor | 2 × (max − min) of Ramen's accuracy over seeds 0–4 |
| domain-averaged accuracy | mean of the three per-domain accuracies (ImageNet-A, the smallest domain, counts as much as Sketch) |
| fixed / broken vs Ramen | images Ramen gets wrong that the variant gets right / and the reverse |
| cluster bootstrap CI | 95% interval from resampling images (Poisson weights), per seed |"""

CELLS = [
    md(HEADER),
    md('## 1. Setup'),
    code("""import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '..'); sys.path.insert(0, '../2026-10-25_stacking')
import pandas as pd
import stack, heldout_analysis as ha
pd.set_option('display.width', 200)
t = ha.load(); FLOOR, ACCS = ha.noise_floor()
stack.NOISE_FLOOR[ha.DS] = round(FLOOR, 2)
sorted(t.seed.unique())"""),
    md('## 2. The data the code ran on'),
    code("stack.data_table([ha.DS])"),
    code("""z = pd.read_pickle(stack.path(ha.DS, 0, 'NoAdapt') + '.pkl')
pd.DataFrame({'images': z.groupby('domain', observed=True).size(),
              'classes present': z.groupby('domain', observed=True).label.nunique(),
              'NoAdapt correct': z.groupby('domain', observed=True).correct.sum(),
              'NoAdapt acc %': (z.groupby('domain', observed=True).correct.mean() * 100).round(2)})"""),
    code("pd.Series({**{f'Ramen seed {s}': round(a, 3) for s, a in enumerate(ACCS)}, 'range': round(max(ACCS) - min(ACCS), 3), 'noise floor (2 x range)': round(FLOOR, 3)})"),
    md("""## 3. Sanity checks

- **Labels.** Zero-shot accuracy on Sketch (1,000 classes, the standard setting) should be close to the published ~46–48% for CLIP ViT-B/16 (table above). Wrong label remapping would show as near-chance accuracy.
- **Saved probabilities.** Every loaded run asserts that the argmax of its saved softmax equals its saved predictions (`stack.load`).
- **Stream alignment.** Ramen and T3 of the same seed see the identical stream order (asserted in `stack.stack_seed`)."""),
    md('## 4. Primary result (pre-registered)'),
    code("ha.verdict(t, FLOOR)"),
    code("""s = ha.summary(t, FLOOR)
s.astype({'gain mean': float, 'gain std': float, 'gain min': float, 'images vs Ramen': float}).round(2)"""),
    code("fig = stack.plot_seed_gains({ha.DS: t}, variants=ha.SHOW)"),
    md('## 5. Secondary: every method vs zero-shot, and where the gain comes from'),
    code("ha.vs_noadapt(t).round(2)"),
    code("fig = stack.plot_fixed_broken({ha.DS: t}, variants=ha.SHOW)"),
    md('## 6. Per domain\n\nGain over Ramen per domain (points, mean over seeds).'),
    code("""pdm = ha.load('per_domain')
pdm[pdm.variant.isin(ha.SHOW)].groupby(['domain', 'variant']).gain.mean().unstack()[ha.SHOW].round(2)"""),
    md("""### Why prior correction fails here

Labels are used only to describe what happened. The 314 classes of ImageNet-R ∪ ImageNet-A carry most images, so the true class marginal is far from uniform. Prior correction pulls predictions toward uniform, away from those classes."""),
    code("""import numpy as np
df, P = stack.load(ha.DS, 0, 'Ramen'); _, T = stack.load(ha.DS, 0, 'active_prob_T3')
y = df.label.values; N = len(y)
ra = np.zeros(1000, bool); ra[np.unique(y[df.domain.astype(str).values != 'sketch'])] = True
true = np.bincount(y, minlength=1000) / N
z5 = pd.read_pickle(stack.path('ImageNetC5K', 0, 'NoAdapt') + '.pkl'); t5 = np.bincount(z5.label, minlength=1000) / len(z5)
print(f'R/A classes: {ra.sum()}; images with an R/A label: {stack.fmt(int(ra[y].sum()), N)}')
print(f'TV(true marginal, uniform): {0.5 * np.abs(true - 1e-3).sum():.3f} here vs {0.5 * np.abs(t5 - 1e-3).sum():.3f} on ImageNet-C 5K')
rows = {}
for name, X in [('Ramen', P), ('T3', T)]:
    for lam in (0, 1.0):
        pred = X.argmax(1) if lam == 0 else stack.prior_correct(X, df.batch.values, lam)
        rows[name + ('' if lam == 0 else ' + pc(lam=1)')] = {
            'predictions on R/A classes': stack.fmt(int(ra[pred].sum()), N),
            'wrong predictions into Sketch-only classes': f'{int(((pred != y) & ~ra[pred]).sum()):,}',
            'pooled acc %': round((pred == y).mean() * 100, 2)}
pd.DataFrame(rows).T  # seed 0; true share of R/A labels is 60.5%"""),
    md('## 7. Interpretation\n\nINTERPRETATION'),
    md(GLOSSARY),
]


def build(interpretation):
    nb = nbformat.v4.new_notebook()
    nb.cells = CELLS
    nb.cells[-2] = md('## 7. Interpretation\n\n' + interpretation)
    nb.metadata['kernelspec'] = {'name': 'ramen-env-kernel', 'display_name': 'Python 3 (RAMEN)', 'language': 'python'}
    ExecutePreprocessor(timeout=1800, kernel_name='ramen-env-kernel').preprocess(nb, {'metadata': {'path': HERE}})
    out = os.path.join(HERE, 'heldout.ipynb')
    nbformat.write(nb, out)
    return out


if __name__ == '__main__':
    print(build(open(sys.argv[1]).read() if len(sys.argv) > 1 else '(written after the results)'))
