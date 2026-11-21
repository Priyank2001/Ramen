"""
Write the label-value oracle notebook (not executed).

    python make_notebook.py
Execute with (CPU is enough once the runs are cached):
    CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace \
        01_label_oracle_domainnet.ipynb
"""
import nbformat as nbf

from make_dataset_notebooks import adapted_filing_cells, global_retrieval_cells

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

FILE = '01_label_oracle_domainnet.ipynb'

HEADER = """# Does Ramen's cache help once its entries point toward their true labels?

**Question.** Filing Ramen's cache by the true label (every queue pure) made DomainNet accuracy *worse* than
Ramen: 55.29% vs 56.80%, with quickdraw at 11.05% vs 17.54% (`2026-10-23_cache_contamination`,
`2026-11-01_pillow_sink`). The proposed reason: a queue's label never reaches the update. Every entry stores the
image's own **entropy** gradient, which sharpens its *zero-shot* prediction, right or wrong. If so, giving the
entries a gradient toward their **true** label should turn the oracle into a large gain. If it does not, the
explanation (or the oracle pipeline) is wrong.

**Design: two factors, crossed.** Ramen makes two choices about the cache; each is replaced by an oracle choice.

| | value = own entropy gradient (Ramen) | value = cross-entropy gradient toward the true label (ORACLE) |
|---|---|---|
| **filed by zero-shot prediction (Ramen)** | `Ramen` (cached) | `labeloracle_zs_ce` (new) |
| **filed by true label (ORACLE)** | `oracle_file_true` (cached, 2026-10-23) | `labeloracle_true_ce` (new) |

All four runs use true labels except Ramen: **the three oracles are upper bounds, not methods.**

**No image uses its own label.** Ramen writes each image into the cache *before* retrieval, so an image always
retrieves its own entry (distance 0). Here an entry is written with the entropy gradient, as in Ramen; its label
is revealed only *after* the image is predicted, and only then is its value replaced by the label gradient. So
each prediction uses the labels of earlier images only (an online stream with label feedback after each batch).
Retrieval weights, retrieval, the signSGD step and the reset are Ramen's.

**Hypotheses (written before the full runs).**
- **H1.** `labeloracle_true_ce` beats Ramen by more than the noise floor (0.11 points), by several points.
- **H2.** Label values matter more than filing: `labeloracle_zs_ce` also beats Ramen by more than the floor, and
  the filing factor changes accuracy far less than the value factor.
- **H3.** The *pillow* sink shrinks: label-value runs predict *pillow* wrongly far less often than Ramen's 6,484.

**Data.** DomainNet, 6 domains, 345 classes, 596,006 images; CLIP ViT-B/32; one mixed (shuffled) stream, batch
size 100. Config `cfg/DomainNet/Ramen.yaml` (capacity 300, top-k 10, β 5, signSGD lr 0.01, fp16 priorities),
unchanged for every run.

**Seeds.** Seed 0 only (screening). A seed sets the order of the stream; all runs here share seed 0's stream.
Noise floor = 2 × (max − min) of Ramen over seeds 0–4.

**How the runs were produced.** New code in this folder only (`label_methods.py`: `RamenLabelValues`, a subclass
of `RamenTrueLabel`, built through a local registry; nothing in `src/` or `cfg/` is changed):
```
python run_label_oracle.py DomainNet 0 --smoke smoke                                   # sanity check, 1,200 images
CUDA_VISIBLE_DEVICES=0 python run_label_oracle.py DomainNet 0 --runs labeloracle_true_ce
CUDA_VISIBLE_DEVICES=1 python run_label_oracle.py DomainNet 0 --runs labeloracle_zs_ce
```
Results are cached in `new_experiments/results/` and loaded below.

**What counts as a result.** A change vs Ramen larger than the noise floor, in domain-averaged accuracy.

**Sections.** 1 Setup · 2 Data · 3 Sanity checks · 4 Results (2 × 2, per domain, adapted filing, global retrieval) ·
5 Mechanism: what each filing retrieves, the neighbours' labels, near-copies · 6 The *pillow* sink · 7 Interpretation ·
8 Glossary
"""

INTERPRETATION = """## 7. Interpretation

**What the numbers say (seed 0; noise floor 0.11 points).**

| | entropy values (Ramen) | true-label CE values (ORACLE) | effect of label values |
|---|---|---|---|
| **zero-shot filing (Ramen)** | 56.80 (Ramen) | **58.50 (+1.70, +12,207 images)** | +1.70 |
| **true-label filing (ORACLE)** | 55.29 (−1.50) | 56.54 (−0.25, −1,007 images) | +1.25 |
| **effect of true-label filing** | −1.50 | −1.96 | |

- **H1 is rejected.** Filing *and* values both from the true label do **not** beat Ramen: −0.25 points
  (−1,007 images). Quickdraw is 17.34% against Ramen's 17.54%.
- **H2 is half right.** Label values help under either filing: +1.70 (zero-shot filing) and +1.25 (true-label
  filing). Filing by true label hurts just as much: −1.50 and −1.96. With zero-shot filing and label values,
  accuracy rises in every domain: infograph +0.58 up to quickdraw +4.02.
- **H3 holds.** The *pillow* sink disappears once values carry labels: 377 wrong *pillow* predictions
  (zero-shot filing) and 750 (true-label filing), against Ramen's 6,484 and zero-shot's 1,148.

**What this says about the filing oracle.** Its loss has two causes. The first, values that sharpen zero-shot
mistakes, is real: label values recover 1.25 points. The second is that **true-label filing makes retrieval
worse, whatever the values**. Section 5 shows why. Ramen's update is a vote of the 10 nearest entries from each
of about 345 queues. It is spread over about 235 classes, and the image's own class gets only about 2.5% of the
weight under either filing.

- **With true-label filing, every queue holds its class from every domain.** A quickdraw image therefore gets
  ten quickdraw sketches from each of 345 classes: 100% of its update comes from its own domain, and the weight
  is spread over even more classes (276).
- **Its own class leads the vote less often.** That happens for 39% of images, against 46% with zero-shot
  filing; on quickdraw, 11% against 20%.
- **Zero-shot filing crowds look-alike images into a few queues,** and the top-10 cut limits how much they can
  outvote the rest.

**Filing by the adapted prediction (section 4.4) does not help either.** Without labels it scores −0.27
points vs Ramen (−2,639 images) and −0.21 vs its timing control, almost all of it on quickdraw (15.94% vs
17.51%). Moving the write after prediction costs only −0.05 by itself, so neither self-retrieval nor the
same-batch entries matter much. With label values, adapted filing gives +1.26 vs +1.86 for zero-shot filing with
the same timing. Like true-label filing, a "better" filing label lowers accuracy; the stored values, not the
queue labels, carry the effect.

**What it does not show.**
- One seed.
- **These are oracles, not methods.**
- The vote-share measure counts weight, not gradient direction.
- **Only the first 168,502 images (28%) were ever stored.** Ramen's fp16 counter freezes the full queues, so
  every run here works with a fixed memory after that point. The label-value gain is what labels on those
  images buy.

**Ceiling, and what limits it (sections 4.5, 5.2, 5.3).** With the true labels of earlier images, Ramen's
retrieval yields +1.70 points. Retrieving the k nearest entries of the whole cache instead of 10 per queue raises
it:

| retrieval (label values) | vs Ramen |
|---|---|
| 10 per queue (Ramen) | +1.70 (+12,207 images) |
| 10 nearest overall | **+2.86 (+22,366)** |
| 100 nearest overall | +2.54 (+18,947) |
| 1,000 nearest overall | +1.84 (+12,264) |
| 100 nearest overall, **no labels** | +0.04 (−1,068) |

- **The queue-averaged vote is part of the limit on DomainNet:** global top-10 adds +1.16 points over per-queue
  retrieval, ten times the floor, and the gain shrinks as k grows toward Ramen's spread-out vote. **Most of it is
  quickdraw:** 21.56% → 27.04% (+5.48) over per-queue label values; the other five domains move by −0.17 to +0.45.
- **It is not a near-copy leak** (section 5.3; on ImageNet-C the same pattern was one). Of 2,000 sampled images,
  only 19 (0.9%) have another cache entry within distance 0.1. Without them the sample's gain over per-queue
  retrieval moves from +2.70 to +2.62 points.
- **The neighbours themselves are the next limit** (section 5.2). On average only 4.5 of an image's 10 nearest
  entries share its class (quickdraw 2.1), and their label vote is right for 57.6% of the sampled images against
  51.6% for zero-shot. That puts the reachable accuracy for this retrieval well below 100%.
- **Without labels, global retrieval does nothing** (+0.04): it only helps once the values carry labels.

**Next step.** The open question is no longer the vote but where correct labels could come from without
supervision. Global retrieval is only worth using if a label-free value can be made to point toward the right
class more often than the entropy gradient does.
"""

MECHANISM = """## 5. Mechanism: what each filing rule retrieves

### 5.1 How the update weight splits over classes

Retrieval never depends on the stored values: an entry's weight is e^(−H)·e^(−β·d) from its zero-shot entropy and
feature. So the offline cache replay of `2026-11-01_pillow_sink` (exact, CPU) gives the entries the label-value
runs retrieved. With label values, each entry pushes toward *its* true class, so for 1,000 random images (after
the first 20% of the stream) we measure, per image, how the update weight splits over the entries' true classes:

- **share on own true class**: the weight that pushes toward the right class;
- **own class is the largest**: the image's true class gets more weight than any other single class (a
  plurality proxy for whether the step's direction favours the right class);
- **effective number of classes**: exp(entropy) of the weight split over classes (345 = perfectly spread);
- **share from own domain**.

Caveat: this counts weight, not gradient direction. Gradients toward similar classes partly agree, so it is a
proxy for the step's direction, not the step itself.
"""

GLOSSARY = """## 8. Glossary

| term | meaning |
|---|---|
| zero-shot | CLIP classifies by cosine similarity between the image embedding and each class's text embedding. No adaptation. |
| Ramen | test-time adaptation. Each image is written into a class queue with its own gradient; the gradients of the nearest stored entries are retrieved, one signSGD step is taken on the image encoder's LayerNorm parameters, the image is predicted, and the parameters are reset. |
| cache / queue / entry | Ramen's memory: one queue of up to 300 entries per class. An entry is (zero-shot feature, value = a gradient, zero-shot entropy, arrival counter). |
| filing | which queue an image is written to: its zero-shot prediction (Ramen) or its true label (oracle). |
| value | the gradient an entry stores. Ramen: the image's own entropy gradient (sharpens the zero-shot prediction). Label oracle: the cross-entropy gradient toward the true label, stored once the image has been predicted. |
| filing oracle | `oracle_file_true`: filed by true label, Ramen's entropy values. |
| label-value oracle | `labeloracle_zs_ce`: filed by zero-shot prediction, label values. |
| filing + label-value oracle | `labeloracle_true_ce`: both. |
| oracle | a run that uses ground-truth labels: an upper bound on what a method could reach, not a method. |
| retrieval weight | e^(−H)·e^(−β·d) for an entry with zero-shot entropy H at feature distance d; the top 10 per non-empty queue, queues averaged. |
| noise floor | 2 × (max − min) of Ramen's domain-averaged accuracy over seeds 0–4; a change must exceed it to count. |
| fixed / broken | images a run gets right that the baseline got wrong / wrong that the baseline got right. |
| sink | a class a model predicts for many images of other classes (here *pillow*). |
| domain average | the mean of the 6 per-domain accuracies (the headline convention). |
| global top-k retrieval | the k nearest entries of the whole cache, whatever their queue (`RamenGlobalRetrieval`, `retrieval: global_knn`), instead of 10 per queue. |
| label vote | the retrieval-weighted vote of the retrieved entries' true labels: the class a label-value update pushes toward most. |
| near-copy | another cache entry within feature distance 0.1 of the image (cosine similarity above 0.995); with label values it would leak the image's label. |
"""

CELLS = [
    md(HEADER),
    md('## 1. Setup'),
    code("""import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # everything below reads cached runs

import pandas as pd
from IPython.display import display

from label_oracle import FULL_RUNS, LABELS, NEW_RUNS, SMOKE_CHECKS, SMOKE_RUNS, LabelOracleExperiment
from true_label_cache import count_pct, noise_floor, seed_table
import plots

pd.set_option('display.width', 220, 'display.max_columns', 20)
exp = LabelOracleExperiment('DomainNet', seed=0).run_all(FULL_RUNS)
ORACLES = list(FULL_RUNS[2:])  # the three oracle runs, compared with Ramen"""),
    md('## 2. The data the code ran on\n\nImages and classes per domain, and how many zero-shot CLIP gets right: '
       'the denominators of everything below.'),
    code('exp.data_table()'),
    md('## 3. Sanity checks\n\n### 3.1 The "off" settings reproduce their references exactly\n\n'
       'On 200 images per domain (1,200), `RamenLabelValues` with Ramen\'s value (entropy) must give the same '
       'predictions as Ramen (zero-shot filing) and as `RamenTrueLabel` (true-label filing): 100.00%.'),
    code("""smoke = LabelOracleExperiment('DomainNet', seed=0, n_per_domain=200, results_dir='smoke').run_all(SMOKE_RUNS)
assert all(smoke.sanity_check(a, b) for a, b in SMOKE_CHECKS), 'off settings differ from their references'
smoke.summary_table(list(SMOKE_RUNS))"""),
    md('### 3.2 The label value points toward the true label\n\nOne signSGD step on each image\'s *own* stored '
       'value, on one batch of 100 images (`check_label_gradient.py`). The CE value should raise the '
       'probability of the true class; Ramen\'s entropy value raises the probability of the zero-shot class.'),
    code("print(open('logs/check_label_gradient.log').read())"),
    md('### 3.3 The full runs\n\nSame stream as Ramen; the labels the method saw agree with the scored labels '
       '(its count of zero-shot errors equals the zero-shot model\'s); and every written entry got its label '
       'value.'),
    code("""keys = ['domain_idx', 'sample_idx']
ram = exp.results['Ramen'][keys].to_numpy()
zs_wrong = int((~exp.results['NoAdapt']['correct']).sum())
rows = []
for n in NEW_RUNS:
    st = exp.stats[n]
    rows.append({'run': n,
                 'same stream as Ramen': bool((exp.results[n][keys].to_numpy() == ram).all()),
                 'zero-shot errors the method counted': f"{sum(st['zs_wrong']):,}",
                 'zero-shot errors (NoAdapt)': f'{zs_wrong:,}',
                 'entries given a label value': count_pct(sum(st['revealed']), len(ram))})
pd.DataFrame(rows).set_index('run')"""),
    md('The method counts 5 more zero-shot errors than NoAdapt (281,166 vs 281,161): fp16 rounding between the two '
       'forward passes on near-ties. Only 28% of entries ever get a label value because only 28% of images are '
       'ever stored: after Ramen\'s fp16 arrival counter saturates (65,520), new entries replace each full queue\'s '
       'older ones once and then the queues stop changing (see `2026-11-01_pillow_sink`). This holds for every '
       'run here, Ramen included.'),
    md('### 3.4 Noise floor'),
    code("""seeds = seed_table('DomainNet', ('Ramen',))
floor = noise_floor(seeds['Ramen'])
print(f'Ramen over seeds 0-4: {seeds.Ramen.round(2).tolist()} -> noise floor {floor:.2f} points')"""),
    md('## 4. Results\n\n### 4.1 One row per run\n\nDomain-averaged accuracy; counts are pooled over all '
       '596,006 images. Fixed / broken are relative to zero-shot.'),
    code('exp.summary_table(list(FULL_RUNS))'),
    md('### 4.2 The 2 × 2: filing × value'),
    code("""ft = exp.factor_table()
display(ft)
d = ft['vs Ramen (pts)']
value_effect = (d[('zero-shot', 'true-label CE')] + d[('true label', 'true-label CE')]
                - d[('zero-shot', 'entropy')] - d[('true label', 'entropy')]) / 2
filing_effect = (d[('true label', 'entropy')] + d[('true label', 'true-label CE')]
                 - d[('zero-shot', 'entropy')] - d[('zero-shot', 'true-label CE')]) / 2
print(f'main effect of label values: {value_effect:+.2f} pts; main effect of true-label filing: {filing_effect:+.2f} pts')"""),
    code("""by_run = ft.reset_index().set_index('run')['vs Ramen (pts)']
best = by_run[ORACLES].idxmax()
title = f"{LABELS[best]}: {by_run[best]:+.2f} pts vs Ramen; filing alone: {by_run['oracle_file_true']:+.2f}"
plots.plot_vs_ramen(exp, ORACLES, floor, title=title);"""),
    md('### 4.3 Per domain'),
    code("exp.accuracy_table(list(FULL_RUNS))"),
    code("""delta = exp.accuracy_table(ORACLES + ['Ramen'])
qd = delta.loc['quickdraw']
plots.plot_domain_change(exp, ORACLES, title=f"Quickdraw vs Ramen: label values {qd['labeloracle_zs_ce'] - qd['Ramen']:+.2f}, "
                         f"filing + label values {qd['labeloracle_true_ce'] - qd['Ramen']:+.2f}, "
                         f"filing alone {qd['oracle_file_true'] - qd['Ramen']:+.2f} pts");"""),
    md('Fixed and broken images vs Ramen, per domain, for the two label-value runs:'),
    code("""for n in NEW_RUNS:
    print(LABELS[n])
    display(exp.change_table(n, baseline='Ramen'))"""),
    *adapted_filing_cells('domain', 'domain'),
    *global_retrieval_cells('domain', 'domain'),
    md(MECHANISM),
    code("""from mechanism import LabelAgreement
agree = LabelAgreement('DomainNet', seed=0, n=1000)
mech = agree.by_domain()
mech"""),
    code("""m = mech['own class is the largest %'].unstack('filing')
plots.plot_agreement(mech, title=f"Own class leads the update: {m.loc['all (pooled)', 'zero-shot filing']:.0f}% of images "
                     f"(zero-shot filing) vs {m.loc['all (pooled)', 'true-label filing']:.0f}% (true-label filing)");"""),
    md("""### 5.2 How often the nearest entries carry the right label

With label values, each retrieved entry pushes toward its own true class, so an update can point the right way
only as often as the retrieved entries vote for the image's class. `NeighbourLabels` replays Ramen's cache for
2,000 random images (after the first 20% of the stream) and takes the retrieval-weighted vote of the entries' true
labels: over Ramen's active set (10 per queue) and over the 10 nearest entries overall."""),
    code("""from neighbour_labels import NeighbourLabels
nl = NeighbourLabels('DomainNet', seed=0, n=2000)
nls = nl.summary()
nls"""),
    md("When the global-10 vote is right or wrong, split by zero-shot: how often the global top-10 run gets the image right."),
    code("nl.crosstab('global_ce_k10')"),
    md("""### 5.3 Is the global gain a near-copy leak?

On ImageNet-C the global gain came from other copies of the same photo (section 4.6 of that notebook). DomainNet
has no such groups, but it may hold near-identical images. `NearCopies` replays the same 2,000 images and counts,
per image, its own entry among the retrieved (no leak: it still holds the entropy value at prediction time) and
other entries within distance 0.1. Note: the sample is pooled, so quickdraw (590 of 2,000) weighs more than in the
domain average."""),
    code("""from near_copies import NearCopies
nc = NearCopies('DomainNet', seed=0, n=2000)
display(nc.summary())
for n in ('global_ce_k10', 'global_ce_k100', 'global_ce_k1000'):
    display(nc.gain_split(n, 'labeloracle_zs_ce'))"""),
    md('## 6. The *pillow* sink\n\nHow often each run predicts *pillow* and the two classes most of Ramen\'s '
       'wrong *pillow*s came from (zero-shot *envelope* and *squiggle*), with how many of those are wrong.'),
    code("exp.sink_table(['pillow', 'envelope', 'squiggle'])"),
    code("""wrong = {n: int(((exp.results[n].pred == exp.classes.index('pillow')) & ~exp.results[n].correct).sum())
         for n in FULL_RUNS}
plots.plot_class_predictions(exp, 'pillow', list(FULL_RUNS),
                             title=f"Wrong pillow predictions: Ramen {wrong['Ramen']:,}, filing + label values "
                                   f"{wrong['labeloracle_true_ce']:,}, label values {wrong['labeloracle_zs_ce']:,}");"""),
    md(INTERPRETATION),
    md(GLOSSARY),
]


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nbf.write(nb, FILE)
    print('wrote', FILE)


if __name__ == '__main__':
    build()
