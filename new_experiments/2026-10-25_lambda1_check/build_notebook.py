"""Writes and executes lambda1_check.ipynb: python build_notebook.py [interpretation.md]. Needs lambda_check.py's CSV."""
import os
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

HEADER = r"""# Does prior correction at λ = 1 survive on the small-label datasets?

**Question.** On ImageNet-C and DomainNet, prior correction (pc) only worked at λ = 1, chosen post-hoc (`../2026-10-25_stacking/`). λ = 2 was originally tuned on CIFAR. Before λ = 1 goes to a held-out dataset, does it still help on CIFAR-10-C, CIFAR-100-C and EuroSAT, or does it give up the gains λ = 2 had there?

**Pre-registration** (`PREREGISTRATION.md`, written before any λ = 1 number was computed on these datasets). λ = 1 passes on a dataset if its mean gain over Ramen is (i) above the noise floor **and** (ii) at least 50% of the λ = 2 gain. Pass on all three → λ = 1 is the single setting for the held-out test.

**Variants** (offline transforms of saved Ramen outputs; no labels used except for scoring):

| Name | What it is |
|---|---|
| Ramen + pc(lam=λ) | logit − λ·log(π_t / uniform), π_t = EMA (m = 0.99) of the predicted class marginal, batch by batch, current batch included |
| Ramen + knn | 0.5·p + 0.5·(weighted mean of p over the 10 most similar images of earlier batches, zero-shot CLIP features, softmax(sim / 0.01)) |
| Ramen + knn + pc(lam=λ) | knn, then pc |

T3 is left out here: it does not help with ≤ 100 classes (CIFAR-100-C −0.15), so the frozen setting applies pc to Ramen on small label spaces.

**Data.** CIFAR-10-C (ViT-B/32) and CIFAR-100-C (ViT-B/16): 15 corruptions × 10,000 images, severity 5. EuroSAT (ViT-B/16): 27,000 satellite images, 10 classes, one domain. One shuffled mixed stream each, batches of 100.

**Seeds.** 0–4. A seed sets the stream order; the images are the same. NoAdapt features from seed 0 are realigned to every seed.

**How the numbers were produced.** `python lambda_check.py` (seconds per seed on one GPU). It reads the cached Ramen probabilities in `results/` and the stacking code in `../2026-10-25_stacking/stack.py`, and writes `results/lambda1_check/gains.csv`.

**What counts as a result.** The pre-registered rule above. Noise floor = 2 × (max − min) of Ramen's domain-averaged accuracy over seeds 0–4: CIFAR-10-C 0.31, CIFAR-100-C 0.21, EuroSAT 1.51.

**Sections.** 1 Setup · 2 Data · 3 Sanity check · 4 Results · 5 Decision · 6 Interpretation · 7 Glossary"""

GLOSSARY = r"""## 7. Glossary

| Term | Meaning |
|---|---|
| Ramen | CLIP zero-shot + per-class memory of (feature, entropy gradient); one signSGD step per batch from retrieved entries, predict, reset |
| pc (prior correction) | subtract λ·log(estimated predicted-class share / uniform) from the logits, pushing over-predicted classes down |
| λ | strength of the prior correction; λ = 0 is plain Ramen |
| knn (kNN smoothing) | mix a prediction 50/50 with the predictions on the 10 most similar earlier images |
| T3 | zero-shot-weighted active set at temperature 3 (RamenActive); helps only with large label spaces |
| noise floor | 2 × (max − min) of Ramen's accuracy over seeds 0–4; a gain must exceed it to count |
| pre-registered | the setting and the decision rule were written down before the numbers were computed |
| domain-averaged accuracy | mean of per-domain accuracies (EuroSAT has a single domain) |"""

CELLS = [
    md(HEADER),
    md('## 1. Setup'),
    code("""import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '..'); sys.path.insert(0, '../2026-10-25_stacking')
import pandas as pd
import stack, lambda_check as lc
pd.set_option('display.width', 200)
t = lc.load()
t.groupby('dataset').seed.unique()"""),
    md('## 2. The data the code ran on'),
    code("stack.data_table(lc.DATASETS)"),
    code("""pd.DataFrame({stack.LABEL[ds]: {**{f'Ramen seed {s}': round(a, 2) for s, a in enumerate(accs)}, 'noise floor': round(f, 2)}
              for ds in lc.DATASETS for f, accs in [stack.floor(ds)]})"""),
    md("""## 3. Sanity check

At λ = 2 the code must reproduce the earlier seed means (`marginal_correction.py`, `quick_checks/knn_seeds.py`, EuroSAT held-out runs). Five of the six match to 0.01. EuroSAT knn + pc comes out 0.03 lower (6.99 vs 7.02, about 8 of 27,000 images; seed std 0.27). The script behind the earlier EuroSAT number is not in the repo, so the cause is not identified. Its components match (pc 2.79 = 2.79, knn 2.07 vs 2.06), and the decision below compares λ = 1 with λ = 2 computed by the same code."""),
    code("""s = lc.summary(t)
chk = s.dropna(subset=['reference (earlier)'])[['gain mean', 'reference (earlier)']].astype(float)
chk['difference'] = chk['gain mean'] - chk['reference (earlier)']
chk.round(2)"""),
    md('## 4. Results\n\nGain over Ramen (points), mean over seeds 0–4.'),
    code("s.drop(columns='reference (earlier)').astype({'gain mean': float, 'gain std': float, 'gain min': float, 'images vs Ramen': float}).round(2)"),
    code("""tables = {ds: t[t.dataset == ds].drop(columns='dataset').assign(posthoc=False) for ds in lc.DATASETS}
fig = stack.plot_seed_gains(tables, variants=lc.SHOW)"""),
    md('## 5. Decision (pre-registered rule)'),
    code("lc.decision(t)"),
    md('## 6. Interpretation\n\nINTERPRETATION'),
    md(GLOSSARY),
]


def build(interpretation):
    nb = nbformat.v4.new_notebook()
    nb.cells = CELLS
    nb.cells[-2] = md('## 6. Interpretation\n\n' + interpretation)
    nb.metadata['kernelspec'] = {'name': 'ramen-env-kernel', 'display_name': 'Python 3 (RAMEN)', 'language': 'python'}
    ExecutePreprocessor(timeout=900, kernel_name='ramen-env-kernel').preprocess(nb, {'metadata': {'path': HERE}})
    out = os.path.join(HERE, 'lambda1_check.ipynb')
    nbformat.write(nb, out)
    return out


if __name__ == '__main__':
    print(build(open(sys.argv[1]).read() if len(sys.argv) > 1 else '(written after the results)'))
