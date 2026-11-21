"""Writes stacking.ipynb (cells below) and executes it: python build_notebook.py. Needs results/stacking/*.csv (stack.py)."""
import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

HEADER = r"""# Do the confirmed fixes stack on ImageNet-C and DomainNet?

**Question.** Two kinds of fix beat Ramen separately. (a) Output fixes: *prior correction* (pc) and causal *kNN smoothing* (knn), +2.07 on CIFAR-100-C and +7.02 on EuroSAT together, never tried with kNN on the large-label datasets. (b) The *zero-shot-weighted active set* at T=3 (T3), +0.62 on ImageNet-C and +0.24 on DomainNet, which helps only with many classes. Do they add up on the two large-label datasets?

**Hypothesis.** T3 + knn + pc beats Ramen by more than the noise floor on both datasets: > 0.53 pts on ImageNet-C 5K, > 0.11 on DomainNet (domain-averaged accuracy, mean over seeds 0–4).

**Why this test.** pc and knn transform a run's output probabilities only. Ramen files each image under its *zero-shot* prediction and caches gradients taken at the base weights, so its outputs never feed back into its memory. The stack is therefore an exact offline transform of a saved run, with no new method code. The only GPU work is re-running Ramen and T3 with the full softmax saved (the earlier runs kept only the top prediction).

**Variants** (none uses true labels; labels are used only to score):

| Name | What it is |
|---|---|
| Ramen | baseline, `cfg/<ds>/Ramen.yaml` |
| T3 | RamenActive, `active_set: prob`, `temperature: 3`: each class queue's vote weighted by the query's zero-shot probability^(1/3) |
| + pc | logit − λ·log(π_t / uniform), π_t = EMA (m = 0.99) of the predicted class marginal, updated batch by batch including the current batch; λ = 2 |
| + knn | 0.5·p + 0.5·(weighted mean of p over the K = 10 most similar images from *earlier* batches; similarity = zero-shot CLIP feature dot product, weights softmax(sim / 0.01)) |
| + knn + pc | knn first, then pc (the order used on CIFAR / EuroSAT) |
| + pc(λ=1), pc(λ=0.5) | **post-hoc**: λ = 2 with a uniform target hurt DomainNet before (−0.53), so smaller λ are reported, labelled post-hoc |

All settings (λ = 2, m = 0.99, K = 10, temperature 0.01, mix 0.5, T = 3) were fixed on CIFAR / ImageNet-C before this experiment: **pre-registered**. T = 3 was tuned on ImageNet-C seed 0, so T3 on ImageNet-C is not held-out; on DomainNet it is.

**Data.** ImageNet-C 5K (15 corruptions × 5,000 images, severity 5, 997 classes, CLIP ViT-B/16) and DomainNet (6 domains, 596,006 images, 345 classes, CLIP ViT-B/32), one shuffled mixed stream each, batches of 100.

**Seeds.** 0–4. A seed sets the stream order (and Ramen's selection randomness); the images are the same. NoAdapt (zero-shot) is stateless, so its features from seed 0 are realigned to every seed by (domain, image id).

**How the runs were produced.** `bash run_queue.sh` (detached, one worker per GPU) runs `run_probs.py <ds> <seed> <Ramen|active_prob_T3>`. It saves `results/<ds>_..._s<seed>_<variant>_p.{pkl,_probs.npy}` and never overwrites the cached runs. Seed-0 Ramen probabilities were already cached. Then `python stack.py <ds> 0 1 2 3 4` writes `results/stacking/<ds>_s<seed>_{stack,per_domain,knn_diag}.csv`.

**What counts as a result.** Mean gain over Ramen above the noise floor, 2 × (max − min) of Ramen's accuracy over seeds 0–4: **0.53** (ImageNet-C), **0.11** (DomainNet). Also reported: per-seed gains, seeds positive, and per-seed cluster-bootstrap 95% CIs (clusters = base image on ImageNet-C, image on DomainNet).

**How to read this notebook.** 1 Setup · 2 Data · 3 Sanity checks · 4 Results (pre-registered) · 5 Does the stack add up? · 6 Per domain · 7 Post-hoc λ · 8 Mechanism: what kNN averages over · 9 Interpretation · 10 Glossary"""

GLOSSARY = r"""## 10. Glossary

| Term | Meaning |
|---|---|
| Ramen | CLIP zero-shot + a per-class memory of (feature, entropy gradient); each batch takes one signSGD step built from retrieved memory entries, predicts, and resets |
| NoAdapt / zero-shot | plain CLIP with class-name prompts, no adaptation |
| T3 | zero-shot-weighted active set: each class queue's contribution weighted by the query's zero-shot probability at temperature 3 (RamenActive) |
| pc (prior correction) | subtract λ·log(estimated predicted-class share / uniform) from the logits, so over-predicted classes are pushed down |
| knn (kNN smoothing) | mix a prediction 50/50 with the predictions on the 10 most similar earlier images |
| domain-averaged accuracy | mean of the per-domain accuracies (headline metric; differs from pooled accuracy on DomainNet, whose domains have unequal sizes) |
| noise floor | 2 × (max − min) of Ramen's domain-averaged accuracy over seeds 0–4: deliberately strict; a gain must be large compared with how much the baseline moves with the stream order |
| fixed / broken vs Ramen | images Ramen gets wrong that the variant gets right / and the reverse |
| pre-registered / post-hoc | settings fixed before seeing these datasets / chosen after |
| cluster bootstrap CI | 95% interval from resampling whole clusters of correlated images (one base image appears under all 15 corruptions) |
| same base image | on ImageNet-C a neighbour that is the same original photo under a different corruption |"""

CELLS = [
    md(HEADER),
    md('## 1. Setup'),
    code("""import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import stack
from stack import collect, seed_summary, additivity, NOISE_FLOOR, LABEL
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
DATASETS = os.environ.get('STACK_DATASETS', 'ImageNetC5K,DomainNet').split(',')
tables = {ds: collect(ds) for ds in DATASETS}
{ds: sorted(t.seed.unique()) for ds, t in tables.items()}"""),
    md('## 2. The data the code ran on\n\nDenominators for everything below. Accuracies are domain-averaged unless stated; counts are pooled.'),
    code("stack.data_table(DATASETS)"),
    code("""fl = {ds: stack.floor(ds) for ds in DATASETS}
pd.DataFrame({LABEL[ds]: {**{f'Ramen seed {s}': round(a, 3) for s, a in enumerate(accs)}, 'range': round(max(accs) - min(accs), 3),
              'noise floor (2 x range)': round(f, 2)} for ds, (f, accs) in fl.items()})"""),
    md("""## 3. Sanity checks

1. Each probability-saving re-run must reproduce the cached run it stands in for: identical stream order and **identical predictions** (100.00%), and the saved probabilities' argmax must equal the saved predictions.
2. `stack.py` re-implements the earlier CIFAR scripts; on CIFAR-100-C seed 0 it gives knn +0.64, pc +1.45, knn + pc +2.11, the same numbers as `quick_checks/knn_seeds.py` (checked when the module was written)."""),
    code("pd.concat({LABEL[ds]: stack.sanity_table(ds) for ds in DATASETS})"),
    md("""## 4. Results (pre-registered settings)

Gain over Ramen per seed (points, domain-averaged), then the summary over seeds."""),
    code("""for ds, t in tables.items():
    t0 = t[~t.posthoc]
    print(f'== {LABEL[ds]}: gain over Ramen (pts) per seed')
    print(t0.pivot(index='variant', columns='seed', values='gain').loc[[v for v in ['Ramen'] + stack.MAIN if v in set(t0.variant)]].round(2).to_string(), '\\n')"""),
    code("""pd.concat({LABEL[ds]: seed_summary(ds, t[~t.posthoc]).astype({'gain mean': float, 'gain std': float, 'gain min': float,
           'images vs Ramen': float, 'fixed vs Ramen': float, 'broken vs Ramen': float}).round(2) for ds, t in tables.items()})"""),
    code("fig = stack.plot_seed_gains({ds: t[~t.posthoc] for ds, t in tables.items()})"),
    code("""ci = pd.concat({LABEL[ds]: t[~t.posthoc & (t.variant != 'Ramen')].set_index(['variant', 'seed'])[['gain', 'CI low', 'CI high']]
                for ds, t in tables.items()})
ci['CI excludes 0'] = ci['CI low'] > 0
ci.round(2).unstack('seed')['CI excludes 0']"""),
    code("fig = stack.plot_fixed_broken({ds: t[~t.posthoc] for ds, t in tables.items()})"),
    md("""## 5. Does the stack add up?

Per seed: the full stack's gain vs the sum of the three single fixes (T3 alone, knn on Ramen, pc on Ramen). Stack − sum < 0 means the fixes overlap: partly correct the same images."""),
    code("pd.concat({LABEL[ds]: additivity(ds, t[~t.posthoc]) for ds, t in tables.items()}).round(2)"),
    md('## 6. Per domain'),
    code("for ds in DATASETS:\n    fig = stack.plot_domain_gain(ds)"),
    md("""## 7. Post-hoc: weaker prior correction

**Post-hoc** (λ chosen after the earlier DomainNet failure, not pre-registered). The uniform target is wrong on both datasets: ImageNet-C 5K has 0–180 images per class, and DomainNet's domains are strongly imbalanced. A smaller λ trusts the uniform target less."""),
    code("""rows = {}
for ds, t in tables.items():
    for v, g in t[t.variant.str.contains('pc')].groupby('variant'):
        rows[(LABEL[ds], v)] = {'gain mean': g.gain.mean(), 'gain std': g.gain.std(), 'positive': f'{(g.gain > 0).sum()}/{len(g)}',
                                'above floor': f'{(g.gain > NOISE_FLOOR[ds]).sum()}/{len(g)}', 'post-hoc': bool(g.posthoc.iloc[0])}
pd.DataFrame(rows).T.astype({'gain mean': float, 'gain std': float}).round(2)"""),
    md("""## 8. Mechanism: what does kNN average over?

Share of the neighbour weight (mean over images after the first batch) that falls on images with the same **true** class, the same domain, and, on ImageNet-C, the **same base photo under another corruption**. Labels are used only to describe the neighbours, never by the method. A large same-base-image share means kNN smoothing partly exploits a quirk of corruption benchmarks, where every photo appears 15 times."""),
    code("""diag = pd.concat({LABEL[ds]: collect(ds, 'knn_diag').set_index('seed') for ds in DATASETS})
diag[[c for c in diag.columns if c.startswith('same')]].round(1)"""),
    md("""### Why kNN and λ = 2 prior correction clash

pc adds the same log-prior shift to every image's logits. Whether that flips the prediction depends on the image's top-1 vs top-2 log-margin. kNN smoothing averages each prediction with its neighbours', which shrinks that margin, so the same shift flips more predictions, and fixes and breaks nearly cancel (section 4). Below: the median log(p1/p2) and the share of predictions pc (λ = 2) changes, without and with smoothing (mean over seeds)."""),
    code("""m = diag[[c for c in diag.columns if 'log(p1/p2)' in c or 'pc flips' in c]].groupby(level=0).mean().T
m.index = pd.MultiIndex.from_tuples([tuple(i.split(': ')) for i in m.index], names=['run', 'quantity'])
m.unstack('quantity').round(2)"""),
    md('## 9. Interpretation\n\nINTERPRETATION'),
    md(GLOSSARY),
]


def build(interpretation='(written after the results)', execute=True):
    nb = nbformat.v4.new_notebook()
    nb.cells = CELLS
    nb.cells[-2] = md('## 9. Interpretation\n\n' + interpretation)
    nb.metadata['kernelspec'] = {'name': 'ramen-env-kernel', 'display_name': 'Python 3 (RAMEN)', 'language': 'python'}
    if execute:
        ExecutePreprocessor(timeout=1800, kernel_name='ramen-env-kernel').preprocess(nb, {'metadata': {'path': HERE}})
    out = os.path.join(HERE, 'stacking.ipynb')
    nbformat.write(nb, out)
    return out


if __name__ == '__main__':
    import sys
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else '(written after the results)'
    print(build(text))
