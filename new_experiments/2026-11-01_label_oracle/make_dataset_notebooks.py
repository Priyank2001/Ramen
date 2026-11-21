"""
Write the label-value oracle notebooks for the datasets after DomainNet (not executed). DomainNet's notebook, with
its cache replay and pillow case, is written by make_notebook.py.

    python make_dataset_notebooks.py                 # all of SPECS
    python make_dataset_notebooks.py EuroSAT
Execute with:
    CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace <file>

Another dataset: add an entry to SPECS (and to DATASETS in 2026-10-27_true_label_cache if it is new there).
"""
import sys

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

IMNET = '''**What the numbers say (seed 0; noise floor 0.53 points).**

| | entropy values (Ramen) | true-label CE values (ORACLE) |
|---|---|---|
| **zero-shot filing (Ramen)** | 33.08 | **34.58 (+1.51, +1,129 images)** |
| **true-label filing (ORACLE)** | 32.37 (−0.71) | 32.17 (−0.91, −684 images) |
| **adapted filing, written after prediction** | 33.20 (+0.12) | 34.45 (+1.37) |
| **zero-shot filing, written after prediction (timing control)** | 32.91 (−0.17) | 34.72 (+1.64) |

- **H1 holds.** Label values with Ramen's filing beat Ramen by +1.51 points, about 3 × the noise floor, and
  raise all 15 corruptions (from +0.12 on defocus blur to +3.04 on frost). Part of this is a leak (other corrupted copies
  of the same photo carry its label, section 4.6): with those hidden the gain is **+0.92**, still above the floor.
  The other label-value runs in this table carry the same leak.
- **H2 holds.** True-label filing lowers accuracy under both values: −0.71 with entropy values, and −0.91
  against +1.51 with label values. With labels in the values, true-label filing costs 2.42 points.
- **Adapted filing is not a gain.** +0.12 vs Ramen and +0.29 vs its timing control, both below the floor. It
  helps some corruptions (shot noise +1.60, gaussian noise +1.18) and hurts others (elastic −1.06, impulse
  −0.78).
- **The same pattern as DomainNet, with a flatter vote.** ImageNet-C's config takes the single nearest entry
  from each of 1,000 queues (top-k 1, β 0), so an update is an entropy-weighted average over up to 1,000
  entries, one per class.

**Global retrieval (sections 4.5–4.6): its gain here was a leak.**

| run | copies visible, vs Ramen | copies hidden, vs Ramen with copies hidden |
|---|---|---|
| per-queue label values | +1.51 | **+0.92 (+693 images)** |
| global k=10 + label values | +3.97 | −1.00 (−750) |
| global k=100 + label values | +3.82 | +0.18 (+136) |
| global k=1000 + label values | +2.80 | +0.95 (+709) |
| global k=100, no labels | −0.86 | −1.57 (−1,181) |

- With the copies visible, global top-10 looked like +3.97. The stream holds 15 corrupted copies of each photo,
  and with label values a copy carries the query's own label. Copies are only 4.0% of a global top-10 update
  (`check_exclusion.py`), but they all point the same, right way, while the rest of the update is spread over
  many classes.
- With the copies hidden, label values still help per queue (+0.92, above the 0.53 floor), but **global retrieval
  adds nothing**: k=1000 is level with per-queue (+0.95), and k=10 is worse than Ramen (−1.00). Hiding the copies
  costs Ramen itself nothing (−0.05).
- Global retrieval without labels hurts (−0.86, or −1.57 with copies hidden).

**What it does not show.** One seed; the oracles are upper bounds, not methods. No cache replay was run here
(it needs recorded zero-shot features), so the vote-share mechanism is DomainNet's, not measured on ImageNet-C.
'''

EURO = '''**What the numbers say (seed 0; noise floor 1.51 points).**

| | entropy values (Ramen) | true-label CE values (ORACLE) |
|---|---|---|
| **zero-shot filing (Ramen)** | 62.63 | **89.33 (+26.70, +7,210 images)** |
| **true-label filing (ORACLE)** | 56.13 (−6.50) | 70.72 (+8.09, +2,184 images) |
| **adapted filing, written after prediction** | 56.59 (−6.04) | 88.53 (+25.90) |
| **zero-shot filing, written after prediction (timing control)** | 63.16 (+0.53) | 89.11 (+26.49) |

- **H1 holds, by far.** Label values with Ramen's filing reach 89.33%: +26.70 points, 18 × the floor.
  *Herbaceous Vegetation Land* goes from 0.30% to 88.40%; 8 of 10 classes improve (Residential Buildings −2.96 and
  Highway or Road −0.16 are the exceptions). Ramen's most over-predicted class, *Highway or Road* (3,375 wrong
  predictions), drops to 337.
- **H2 holds.** True-label filing lowers accuracy under both values: −6.50 with entropy values, and +8.09 instead
  of +26.70 with label values (it costs 18.6 points).
- **Adapted filing hurts badly:** −6.04 vs Ramen and −6.57 vs its timing control. *Sea or Lake* falls from 54.63% to
  26.30% and *Forest* from 73.13% to 51.83%. A likely reason (not tested): filing by Ramen's own output feeds its
  over-predictions back into the queues, which with only 10 classes and top-50 retrieval shifts the vote.
- **Timing alone does nothing measurable.** Zero-shot filing with the late write is +0.53 vs Ramen, below the floor.
- EuroSAT's memory never evicts (10 × 7,500 > 27,000): every image is stored, so the label-value runs carry the
  true labels of all earlier images.

**Global retrieval (section 4.5) adds nothing measurable here.** With label values, the 100 nearest entries
overall give +27.59 against +26.70 per queue: a +0.89 difference, below the 1.51 floor. 10 nearest is worse
(+23.11) and 1,000 nearest in between (+25.91). Without labels, global top-100 hurts: −7.51. With 10 classes and
top-50 per queue, Ramen's active set is already 500 entries, so its vote is not spread over many classes as on
DomainNet. EuroSAT has no repeated photos, so the ImageNet-C leak does not apply.

**What it does not show.** One seed; the oracles are upper bounds, not methods. The feedback explanation for
adapted filing is a hypothesis; checking it needs the queue composition over the stream (a cache replay).
'''

SPECS = {
    'ImageNetC5K': dict(
        file='02_label_oracle_imagenetc.ipynb', name='ImageNet-C', by='domain', group='corruption',
        data='ImageNet-C (`ImageNetC5K`): 15 corruptions at severity 5, the same 5,000 ImageNet validation images '
             'under each (75,000 images), 1,000 classes; CLIP ViT-B/16; one mixed stream over all corruptions, '
             'batch size 100. Config `cfg/ImageNetC5K/Ramen.yaml`: queue capacity 75, top-k **1**, β **0** '
             '(each queue returns its single nearest entry, weighted e^(−H)), signSGD lr 0.01, fp16 priorities.',
        known='On ImageNet-C the filing-only oracle (`oracle_file_true`, 2026-10-23) was −0.71 points vs Ramen.',
        smoke='200 images per corruption (3,000)',
        global_note='\n**On ImageNet-C these numbers carry a label leak** (other corrupted copies of the same photo); '
                    'section 4.6 removes it.',
        sections=', global retrieval, copies hidden',
        interpretation=IMNET),
    'EuroSAT': dict(
        file='03_label_oracle_eurosat.ipynb', name='EuroSAT', by='class', group='class',
        data='EuroSAT: 27,000 satellite images, 10 land-use classes, one domain (so "domain average" = pooled '
             'accuracy); CLIP ViT-B/16 with the prompt "a centered satellite photo of {}."; one shuffled stream, '
             'batch size 100. Config `cfg/EuroSAT/Ramen.yaml`: queue capacity 7,500, top-k 50, β 5, signSGD lr '
             '0.01. The memory never evicts (10 × 7,500 > 27,000), so every image is stored.',
        known='EuroSAT has no earlier filing-only oracle run; it is run here as `labeloracle_true_entropy` '
              '(`RamenLabelValues` with true-label filing and entropy values, which reproduces `RamenTrueLabel` '
              'exactly on the smoke subset).',
        smoke='200 images',
        sections=', global retrieval',
        interpretation=EURO),
}


def header(s):
    return f"""# Label-value oracle on {s['name']}: does Ramen's cache help once its entries point toward their true labels?

**Question.** On DomainNet (`01_label_oracle_domainnet.ipynb`), giving each cache entry a cross-entropy
gradient toward its **true** label, instead of its own entropy gradient, raised accuracy by +1.70 points, while
filing entries by their true label *lowered* it under either value (−1.50 with entropy values, −0.25 with label
values). Does the same pattern hold on {s['name']}? {s['known']}

**Design: two factors, crossed** (all three oracles use true labels: **upper bounds, not methods**).

| | value = own entropy gradient (Ramen) | value = CE gradient toward the true label (ORACLE) |
|---|---|---|
| **filed by zero-shot prediction (Ramen)** | `Ramen` | `labeloracle_zs_ce` |
| **filed by true label (ORACLE)** | filing oracle | `labeloracle_true_ce` |

No image uses its own label: its entry holds its entropy gradient while its batch is adapted (as in Ramen), and
the label gradient replaces it only after the image has been predicted.

**Hypotheses (written before the runs, from DomainNet).**
- **H1.** Label values with Ramen's zero-shot filing beat Ramen by more than the noise floor.
- **H2.** True-label filing lowers accuracy under both values (filing oracle < Ramen, and
  `labeloracle_true_ce` < `labeloracle_zs_ce`).

**Data.** {s['data']}

**Seeds.** Seed 0 only (screening); a seed sets the order of the stream. Noise floor = 2 × (max − min) of Ramen
over seeds 0–4.

**How the runs were produced.** Code in this folder only (`RamenLabelValues`, local registry; `src/` and `cfg/`
unchanged):
```
python run_label_oracle.py {s['key']} 0 --smoke smoke     # sanity check on {s['smoke']}
python run_label_oracle.py {s['key']} 0 [--runs <name>]   # the new runs, one GPU each
```

**What counts as a result.** A change vs Ramen larger than the noise floor, in domain-averaged accuracy.

**Sections.** 1 Setup · 2 Data · 3 Sanity checks · 4 Results (2 × 2, per {s['group']}, adapted-prediction
filing{s['sections']}) · 5 Over-predicted classes · 6 Interpretation · 7 Glossary
"""


GLOSSARY = """## 7. Glossary

| term | meaning |
|---|---|
| zero-shot | CLIP classifies by cosine similarity between the image embedding and each class's text embedding. No adaptation. |
| Ramen | test-time adaptation. Each image is written into a class queue with its own gradient; the gradients of the nearest stored entries are retrieved (top-k per non-empty queue, weighted e^(−H)·e^(−β·d), queues averaged), one signSGD step is taken on the image encoder's LayerNorm parameters, the image is predicted, and the parameters are reset. |
| filing | which queue an image is written to: its zero-shot prediction (Ramen, as in the paper and its code), its true label (oracle), or its adapted prediction (written after prediction). |
| timing control | zero-shot filing with the write moved after prediction: isolates the timing change that adapted filing needs. |
| value | the gradient an entry stores: the image's own entropy gradient (Ramen; sharpens the zero-shot prediction) or the cross-entropy gradient toward its true label (label oracle), stored once the image has been predicted. |
| filing oracle | filed by true label, Ramen's entropy values (`oracle_file_true` or `labeloracle_true_entropy`). |
| label-value oracle | `labeloracle_zs_ce`: filed by zero-shot prediction, label values. |
| filing + label-value oracle | `labeloracle_true_ce`: both. |
| oracle | a run that uses ground-truth labels: an upper bound, not a method. |
| main effect | the change from switching one factor, averaged over the two settings of the other factor. |
| noise floor | 2 × (max − min) of Ramen's domain-averaged accuracy over seeds 0–4; a change must exceed it to count. |
| fixed / broken | images a run gets right that the baseline got wrong / wrong that the baseline got right. |
| domain average | the mean of the per-domain (here per-corruption) accuracies; pooled accuracy for a one-domain dataset. |
| global top-k retrieval | the k nearest entries of the whole cache, whatever their queue (`RamenGlobalRetrieval`, `retrieval: global_knn`), instead of Ramen's top-k of every queue. |
| copies hidden | ImageNet-C only: entries made from another corruption of the query's photo are left out of retrieval (`exclude_groups`); with label values they would carry the query's label. |
"""


def adapted_filing_cells(by, group):
    """Section 4.4, shared by every notebook: filing by the adapted prediction (written after prediction)."""
    return [
        md(f"""### 4.4 A third filing rule: the adapted prediction

The Ramen paper (Sec. 3.2) and its code file each image under its **zero-shot** prediction. Filing under the
**adapted** prediction instead (Ramen's own output, more often right) needs the write moved *after* prediction,
because the adapted prediction exists only after retrieval and the step (`RamenWriteAfterPrediction`). That also
changes two things besides filing: an image no longer retrieves its own entry or later images of its batch, and
the first batch (empty cache) is predicted zero-shot. So each adapted run has a **timing control**: zero-shot
filing with the same late write.

| run | write | filing | value | uses labels? |
|---|---|---|---|---|
| `late_zs_entropy` | after prediction | zero-shot | entropy | no (timing control for Ramen) |
| `late_zs_ce` | after prediction | zero-shot | true-label CE | oracle (timing control) |
| `late_adapted_entropy` | after prediction | **adapted** | entropy | **no: a method** |
| `late_adapted_ce` | after prediction | **adapted** | true-label CE | oracle |

Checks (`check_write_after.py`, first 3,000 DomainNet images): the first batch is not adapted and equals zero-shot;
retrieval sees earlier batches only; every image is written to the queue of the class the method returned
(adapted filing) or its zero-shot class; 485 of 3,000 images are filed differently by the two rules."""),
        code("""from label_oracle import LATE_RUNS
exp.run_all(LATE_RUNS)
ft3 = exp.filing_table()
ft3"""),
        code("""v = ft3.reset_index().set_index('run')['vs Ramen (pts)']
plots.plot_vs_ramen(exp, list(LATE_RUNS), floor,
                    title=f"Adapted filing: {v['late_adapted_entropy']:+.2f} pts vs Ramen, "
                          f"{v['late_adapted_entropy'] - v['late_zs_entropy']:+.2f} vs its timing control");"""),
        md(f'Per {group}: Ramen, its timing control, and adapted filing (both with entropy values).'),
        code(f"""late = exp.accuracy_table(['Ramen', 'late_zs_entropy', 'late_adapted_entropy'], by={by!r})
late['adapted − timing control'] = late['late_adapted_entropy'] - late['late_zs_entropy']
late.round(2)"""),
    ]


def global_retrieval_cells(by, group, note=''):
    """Section 4.5, shared by every notebook: label values with the k nearest entries of the whole cache."""
    return [
        md(f"""### 4.5 Global top-k retrieval: is the queue-averaged vote the bottleneck?

Ramen takes the top-k entries of **every** non-empty class queue and averages them, so most of an update comes
from classes the image is not. `RamenGlobalRetrieval` with `retrieval: global_knn` keeps everything else (filing,
values, timing, weights e^(−H)·e^(−β·d)) and instead takes the **k nearest entries of the whole cache**, normalised
by their total weight.

| run | retrieval | value | uses labels? |
|---|---|---|---|
| `labeloracle_zs_ce` | top-k per queue (Ramen) | true-label CE | oracle |
| `global_ce_k10`, `global_ce_k100`, `global_ce_k1000` | k nearest overall | true-label CE | oracle |
| `global_entropy_k100` | 100 nearest overall | entropy | **no: a method** |

Check: with `retrieval: per_class` (`global_off`) the class reproduces Ramen on every smoke image (section 3.1).
{note}"""),
        code("""from label_oracle import GLOBAL_RUNS
exp.run_all(GLOBAL_RUNS)
gt = exp.vs_table(['labeloracle_zs_ce', *GLOBAL_RUNS], baselines=('Ramen', 'labeloracle_zs_ce'))
gt"""),
        code("""g = gt['vs Ramen (pts)']
best = g[list(GLOBAL_RUNS)].idxmax()
plots.plot_vs_ramen(exp, ['labeloracle_zs_ce', *GLOBAL_RUNS], floor,
                    title=f"{LABELS[best]}: {g[best]:+.2f} pts vs Ramen, per-queue label values "
                          f"{g['labeloracle_zs_ce']:+.2f}");"""),
        md(f'Per {group}:'),
        code(f"""gacc = exp.accuracy_table(['Ramen', 'labeloracle_zs_ce', *GLOBAL_RUNS], by={by!r})
gacc.round(2)"""),
    ]


def exclusion_cells():
    """Section 4.6, ImageNet-C only: hide the other corrupted copies of each photo from retrieval."""
    return [
        md("""### 4.6 Hiding the other copies of each photo (ImageNet-C only)

`ImageNetC5K` uses the **same 5,000 photos** under all 15 corruptions, so the stream holds 15 copies of every
photo. With label values, an entry made from another corruption of the query's photo carries the query's own
label: a leak that per-class retrieval barely touches (one entry per queue) but global retrieval can (the nearest
entries are often other copies). The `excl_` runs hide from retrieval every entry whose photo is the query's,
except the query's own entry (whose value is still its entropy gradient when it is predicted). `excl_zs_entropy`
is Ramen with copies hidden, the baseline for this section.

Checks (`check_exclusion.py`, first 200 batches): no retrieved entry is another copy (X1 = 0); without exclusion,
other copies carry 0.0% of the update with per-class top-1 but 4.0% with global top-10 (X2). On the smoke subset
the exclusion code path with nothing hidden matches Ramen on 2,988 of 3,000 images (its distances are fp32)."""),
        code("print(''.join(l for l in open('logs/check_exclusion_ImageNetC5K.log') if l.startswith(('===', 'X'))))"),
        code("""from label_oracle import EXCL_RUNS
exp.run_all(EXCL_RUNS)
pairs = {'labeloracle_zs_ce': 'excl_zs_ce', **{n: f'excl_{n}' for n in GLOBAL_RUNS}}
v = exp.vs_table([*pairs, *pairs.values(), 'excl_zs_entropy'], baselines=('Ramen', 'excl_zs_entropy'))
leak = pd.DataFrame({'copies visible: vs Ramen (pts)': v.loc[list(pairs), 'vs Ramen (pts)'].to_numpy(),
                     'copies hidden: vs Ramen with copies hidden (pts)':
                         v.loc[list(pairs.values()), 'vs excl_zs_entropy (pts)'].to_numpy(),
                     'copies hidden (images)': v.loc[list(pairs.values()), 'vs excl_zs_entropy (images)'].to_numpy()},
                    index=[LABELS[n] for n in pairs])
print(f"Ramen with copies hidden vs Ramen: {v.loc['excl_zs_entropy', 'vs Ramen (pts)']:+.2f} pts")
leak"""),
        code("""h = v['vs excl_zs_entropy (pts)']
plots.plot_vs_ramen(exp, list(pairs.values()), floor, baseline='excl_zs_entropy',
                    title=f"Copies hidden: global k=10 {h['excl_global_ce_k10']:+.2f}, per-queue label values "
                          f"{h['excl_zs_ce']:+.2f} pts");"""),
    ]


def cells(key, s):
    by, group = s['by'], s['group']
    plural = 'classes' if group == 'class' else f'{group}s'
    return [
        md(header({**s, 'key': key})),
        md('## 1. Setup'),
        code(f"""import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # everything below reads cached runs

import pandas as pd
from IPython.display import display

from label_oracle import LABELS, SMOKE_CHECKS, SMOKE_RUNS, LabelOracleExperiment
from true_label_cache import count_pct, noise_floor, seed_table
import plots

pd.set_option('display.width', 220, 'display.max_columns', 20)
DATASET, BY = {key!r}, {by!r}
exp = LabelOracleExperiment(DATASET, seed=0).run_all()
RUNS, ORACLES, FILING = list(exp.full_runs), list(exp.full_runs[2:]), exp.filing_oracle"""),
        md('## 2. The data the code ran on\n\nImages, classes and zero-shot correct counts: the denominators of '
           'everything below.'),
        code('exp.data_table()'),
        md(f'## 3. Sanity checks\n\n### 3.1 The "off" settings reproduce their references exactly\n\n'
           f'On {s["smoke"]}, `RamenLabelValues` with Ramen\'s value (entropy) must give the same predictions as '
           f'Ramen (zero-shot filing) and as `RamenTrueLabel` (true-label filing): 100.00%.'),
        code("""smoke = LabelOracleExperiment(DATASET, seed=0, n_per_domain=200, results_dir='smoke').run_all(SMOKE_RUNS)
assert all(smoke.sanity_check(a, b) for a, b in SMOKE_CHECKS), 'off settings differ from their references'
smoke.summary_table(list(SMOKE_RUNS))"""),
        md('### 3.2 The full runs\n\nSame stream as Ramen; the labels the method saw agree with the scored labels '
           '(its count of zero-shot errors matches the zero-shot model\'s, up to fp16 near-ties); how many entries '
           'were ever stored and given a label value.'),
        code("""keys = ['domain_idx', 'sample_idx']
ram = exp.results['Ramen'][keys].to_numpy()
zs_wrong = int((~exp.results['NoAdapt']['correct']).sum())
rows = []
for n in exp.new_runs:
    st = exp.stats[n]
    row = {'run': n, 'same stream as Ramen': bool((exp.results[n][keys].to_numpy() == ram).all()),
           'zero-shot errors the method counted': f"{sum(st['zs_wrong']):,}",
           'zero-shot errors (NoAdapt)': f'{zs_wrong:,}'}
    if st['revealed']:
        row['entries given a label value'] = count_pct(sum(st['revealed']), len(ram))
    rows.append(row)
pd.DataFrame(rows).set_index('run')"""),
        md('### 3.3 Noise floor'),
        code("""seeds = seed_table(DATASET, ('Ramen',))
floor = noise_floor(seeds['Ramen'])
print(f'Ramen over seeds 0-4: {seeds.Ramen.round(2).tolist()} -> noise floor {floor:.2f} points')"""),
        md('## 4. Results\n\n### 4.1 One row per run\n\nDomain-averaged accuracy; counts are pooled over all '
           'images. Fixed / broken are relative to zero-shot.'),
        code('exp.summary_table(RUNS)'),
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
title = f"{LABELS[best]}: {by_run[best]:+.2f} pts vs Ramen; filing alone: {by_run[FILING]:+.2f}"
plots.plot_vs_ramen(exp, ORACLES, floor, title=title);"""),
        md(f'### 4.3 Per {group}'),
        code("exp.accuracy_table(RUNS, by=BY)"),
        code(f"""delta = exp.accuracy_table(ORACLES + ['Ramen'], by=BY).iloc[:-1]  # without the average row
gain = delta['labeloracle_zs_ce'] - delta['Ramen']
plots.plot_domain_change(exp, ORACLES, by=BY,
                         title=f"Label values vs Ramen: {{(gain > 0).sum()}} of {{len(gain)}} {plural} up "
                               f"({{gain.min():+.2f}} to {{gain.max():+.2f}} points)");"""),
        md('Fixed and broken images vs Ramen for the two label-value runs:'),
        code("""for n in ('labeloracle_zs_ce', 'labeloracle_true_ce'):
    print(LABELS[n])
    display(exp.change_table(n, baseline='Ramen', by=BY))"""),
        *adapted_filing_cells(by, group),
        *global_retrieval_cells(by, group, s.get('global_note', '')),
        *(exclusion_cells() if key == 'ImageNetC5K' else []),
        md('## 5. Over-predicted classes\n\nThe classes Ramen predicts wrongly most often, and how often each run '
           'predicts them wrongly (a sink is a class predicted for many images of other classes).'),
        code("exp.top_sinks(10)"),
        code("""sink = exp.top_sinks(1).index[0]
wrong = exp.top_sinks(1).iloc[0]
plots.plot_class_predictions(exp, sink, RUNS,
                             title=f"Wrong {sink} predictions: Ramen {wrong['Ramen']:,}, label values "
                                   f"{wrong[LABELS['labeloracle_zs_ce']]:,}, filing + label values "
                                   f"{wrong[LABELS['labeloracle_true_ce']]:,}");"""),
        md('## 6. Interpretation\n\n' + s['interpretation']),
        md(GLOSSARY),
    ]


def build(key):
    s = SPECS[key]
    nb = nbf.v4.new_notebook()
    nb.cells = cells(key, s)
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nbf.write(nb, s['file'])
    print('wrote', s['file'])


if __name__ == '__main__':
    for key in sys.argv[1:] or SPECS:
        build(key)
