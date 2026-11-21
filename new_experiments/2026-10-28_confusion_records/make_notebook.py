"""
Write the confusion-records notebooks (not executed). Every notebook has the same cells; what differs per dataset
(description, focus classes, plot titles, interpretation) is one entry of NOTEBOOKS.

    python make_notebook.py                  # all notebooks
    python make_notebook.py 02_eurosat       # one (key of NOTEBOOKS)
Execute with:
    ~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace <file>.ipynb

A new notebook = a new entry in NOTEBOOKS (and its interpretation in INTERPRETATIONS once the run is in).
"""
import sys

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

RECORDS_TEXT = """**What this notebook is.** This is a tool for looking at errors, not a test of a new method. Nothing here
changes Ramen. Every image is run through `RamenRecorder` (`src/methods/RamenRecorder.py`). This is Ramen with
one addition: it records four things for each image:

| record | what it is |
|---|---|
| `zs_probs` | zero-shot softmax over the {K} classes, from the pass before adaptation (this pass also decides which memory queue the image is filed in) |
| `probs` | Ramen's softmax, after its one signSGD step on the norm parameters |
| `emb_before` | L2-normalised CLIP image embedding with the pretrained norm parameters (= the zero-shot embedding) |
| `emb_after` | the same embedding with the adapted norm parameters (the one Ramen classifies) |

The `zero_shot → ramen` **outcome** of each image is one of: right → right, *fixed*, *broken*,
wrong → same wrong, wrong → other wrong. The **embedding shift** is 1 − cos(before, after).

The class confusion of each source is a {K} × {K} count matrix: rows are the true class, columns the predicted
class (`confusion.py`). For one class *c* over the N images:
- TP: true *c*, predicted *c*
- FN: true *c*, predicted something else
- FP: true something else, predicted *c*
- TN: everything else"""

PRODUCED_TEXT = """**How the results were produced.**
```
cd new_experiments/2026-10-28_confusion_records
{cmd}
```
The run is cached under `new_experiments/results/{cache}*`.
Code: `experiment.py` (RecordExperiment), `records.py` (ImageRecords), `confusion.py` (ConfusionMatrix,
ConfusionShift), `gallery.py` (ConfusionGallery), `plots.py`.

**Sanity check (what counts as valid).** The recorder must not change Ramen. Its predictions must equal those of
plain Ramen on the same stream for 100.00% of images (cell 2). The recorded zero-shot prediction is compared with
the separate NoAdapt run for information only. The two use different model wrappers, so fp16 near-ties can flip a
handful of images."""

GLOSSARY = """## Glossary

- **Zero-shot (NoAdapt)**: CLIP classifies by cosine similarity between the image embedding and the text embedding
  of each class name. No adaptation.
- **Ramen**: test-time adaptation. Each image is written into the memory queue of its zero-shot class, together
  with its own entropy gradient. Gradients of the nearest stored images are retrieved, one signSGD step is taken
  on the LayerNorm parameters, the image is predicted, and the parameters are reset.
- **Embedding before / after**: the L2-normalised CLIP image feature with the pretrained / adapted norm parameters.
- **Embedding shift**: 1 − cosine(before, after). 0 means the step did not move the image.
- **TP / FP / FN / TN** (for class *c*): true and predicted *c* / predicted *c* but true other / true *c* but
  predicted other / neither.
- **Precision / recall / F1** (for class *c*): TP / predicted *c*; TP / true *c*; 2TP / (2TP + FP + FN), which is 0
  when TP is 0.
- **Sink**: a class predicted far more often than it occurs (large FP).
- **Sink fate**: what Ramen does with the images zero-shot wrongly sends to a sink. They can be fixed, stay at the
  sink, or move to another wrong class.
- **Sink origin**: for Ramen's false positives of a class, what zero-shot had predicted for them.
- **Outcome**: zero-shot → Ramen correctness of one image. *fixed* = wrong → right; *broken* = right → wrong.
- **Subset (`n_per_domain`)**: a random sample of images per domain, which makes a shorter stream. Ramen's memory
  differs from the full run, so its accuracy is not comparable to the full run.
- **Domain average**: the mean of the per-domain accuracies (the headline convention in this project)."""

NOTEBOOKS = {
    '01_domainnet_subset': dict(
        file='01_confusion_records_domainnet.ipynb',
        title='Which images does Ramen confuse, and with what? Per-image records on DomainNet',
        dataset='DomainNet', n_per_domain=2000, K=345,
        focus_class='dog', focus_pair=('snake', 'squiggle'), old_sink='squiggle', new_sink='pillow',
        outcome_by='domain', example_domain='quickdraw',
        data="""**Data and model.** DomainNet has 6 domains and 345 classes. This run uses a quick subset: **2,000 images per
domain, 12,000 in total**, drawn at random within each domain with the seed. Model: CLIP ViT-B/32. Ramen config:
`cfg/DomainNet/Ramen.yaml` (capacity 300 per class, top-k 10, β 5, lr 0.01, signSGD). One mixed, shuffled stream,
batch size 100. **Seed 0**: the seed sets the subset and the stream order.

**Important caveat about subsets.** A subset is a shorter stream, so Ramen's memory holds fewer images. Ramen's
accuracy here is therefore *not* its accuracy on these same images in the full 596,006-image run. On the full
run, Ramen gains +1.43 points over zero-shot (domain average, seed 0); on this subset it gains +1.01. Use the
subset to look at examples. Quote numbers from a full run (notebook 04).""",
        cmd='CUDA_VISIBLE_DEVICES=1 python run_records.py DomainNet 0 --n-per-domain 2000     # ~1 min after loading',
        cache='DomainNet_clip_vitbase32_mixed_n2000_bs100_s0_rec_Ramen',
        data_note="""With 2,000 images per domain, a class has on average about 35 images in total (about 6 per domain),
so per-class counts here are small.""",
        outcomes_title='Most zero-shot errors stay errors; fixed and broken are a thin slice',
        pairs_title='Ramen removes squiggle / envelope confusions and creates new ones',
    ),
    '02_eurosat': dict(
        file='02_confusion_records_eurosat.ipynb',
        title='Which images does Ramen confuse, and with what? Per-image records on EuroSAT',
        dataset='EuroSAT', n_per_domain=None, K=10,
        focus_class='Herbaceous Vegetation Land', focus_pair=('Sea or Lake', 'Highway or Road'),
        old_sink='Highway or Road', new_sink='Permanent Crop Land',
        outcome_by='true_class', example_domain=None,
        data="""**Data and model.** EuroSAT RGB is a **full run** on all **27,000** Sentinel-2 satellite images (64 × 64) in
**10 land-cover classes**, with one domain. The shift here is from natural photos to satellite images. Model: CLIP
ViT-B/16, prompt "a centered satellite photo of {}.". Ramen config: `cfg/EuroSAT/Ramen.yaml` (capacity 7,500 per
class, top-k 50, β 5, lr 0.01, signSGD). The capacity is larger than the stream, so the memory never evicts.
One shuffled stream, batch size 100. **Seed 0** sets the stream order. Known baseline, seed 0: NoAdapt 48.07 →
Ramen 62.63. Seeds 0–4 give a noise floor of 1.51.""",
        cmd='CUDA_VISIBLE_DEVICES=0 nice -n 10 python run_records.py EuroSAT 0     # ~2 min',
        cache='EuroSAT_clip_vitbase16_mixed_nall_bs100_s0_rec_Ramen',
        data_note="""There is one domain, so the outcome table below is split by true class instead of by domain.
With only 10 classes the full confusion matrix is readable, and section 1 shows it.""",
        outcomes_title='Herbaceous Vegetation Land is almost never right under Ramen; Sea or Lake and Forest gain most',
        pairs_title='Pairs whose count changes most from zero-shot to Ramen',
    ),
    '03_cifar100c': dict(
        file='03_confusion_records_cifar100c.ipynb',
        title='Which images does Ramen confuse, and with what? Per-image records on CIFAR-100-C',
        dataset='CIFAR100C', n_per_domain=None, K=100,
        focus_class='keyboard', focus_pair=('maple_tree', 'oak_tree'), old_sink='worm', new_sink='crab',
        outcome_by='domain', example_domain='gaussian_noise',
        data="""**Data and model.** CIFAR-100-C, severity 5, is a **full run** on **15 corruptions × 10,000 images =
150,000**, in **100 classes**. Every corruption applies to the same 10,000 test images. Model: CLIP ViT-B/16
(images upsampled from 32 × 32). Ramen config: `cfg/CIFAR100C/Ramen.yaml` (capacity 750 per class, top-k 5, β 5,
lr 0.01, signSGD). One mixed, shuffled stream over all corruptions, batch size 100. **Seed 0** sets the stream
order. Known baseline, seed 0: NoAdapt 37.10 → Ramen 46.10. Seeds 0–4 give a noise floor of 0.21.""",
        cmd='CUDA_VISIBLE_DEVICES=0 nice -n 10 python run_records.py CIFAR100C 0',
        cache='CIFAR100C_clip_vitbase16_mixed_nall_bs100_s0_rec_Ramen',
        data_note="""Each class has 100 images per corruption, so 1,500 in total: per-class counts are large enough to read.""",
        outcomes_title='On every corruption, more errors switch class than get fixed',
        pairs_title='Ramen removes worm / keyboard confusions and creates crab / tractor / lion ones',
    ),
    '04_domainnet_full': dict(
        file='04_confusion_records_domainnet_full.ipynb',
        title='Which images does Ramen confuse, and with what? Per-image records on the full DomainNet',
        dataset='DomainNet', n_per_domain=None, K=345,
        focus_class='dog', focus_pair=('snake', 'squiggle'), old_sink='squiggle', new_sink='pillow',
        outcome_by='domain', example_domain='quickdraw',
        data="""**Data and model.** DomainNet is a **full run** on all **596,006 images** in 6 domains (clipart, infograph,
painting, quickdraw, real, sketch) and **345 classes**. Model: CLIP ViT-B/32. Ramen config:
`cfg/DomainNet/Ramen.yaml` (capacity 300 per class, top-k 10, β 5, lr 0.01, signSGD, fp16 priorities as in Ramen).
One mixed, shuffled stream, batch size 100. **Seed 0** sets the stream order. Known baseline, seed 0: NoAdapt
55.37 → Ramen 56.80 (domain average). Seeds 0–4 give a noise floor of 0.11. This is the same stream as the cached
Ramen run, so these are the full-run numbers; notebook 01 is the 12,000-image subset.""",
        cmd='CUDA_VISIBLE_DEVICES=0 nice -n 10 python run_records.py DomainNet 0     # ~40 min, ~2 GB of vectors',
        cache='DomainNet_clip_vitbase32_mixed_nall_bs100_s0_rec_Ramen',
        data_note="""The vectors are memory-mapped from the HDD and read only when needed.""",
        outcomes_title='Quickdraw: 75,958 images switch to another wrong class, 10,521 are fixed',
        pairs_title='Ramen removes squiggle / envelope confusions and sends rectangles to pillow',
    ),
}


def cells_for(s, interpretation):
    small = s['K'] <= 20
    example = (f"records.where(outcome='fixed', domain='{s['example_domain']}')" if s['example_domain']
               else "records.where(outcome='fixed')")
    return [
        md(f"""# {s['title']}

**Question.** When the zero-shot model or Ramen gets an image wrong, which class does it pick instead? What do
those images look like? And how does one adaptation step change the image's embedding and prediction?

{RECORDS_TEXT.format(K=s['K'])}

{s['data']}

{PRODUCED_TEXT.format(cmd=s['cmd'], cache=s['cache'])}"""),
        code(f"""import sys
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from experiment import RecordExperiment, count_pct
import plots

pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
plt.rcParams.update({{'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False}})

# ---- settings: change these to look at something else -------------------------------------------------
DATASET, SEED = {s['dataset']!r}, 0
N_PER_DOMAIN = {s['n_per_domain']!r}                      # None = the whole dataset; an int = that many images per domain
FOCUS_CLASS = {s['focus_class']!r}                        # one class to inspect in detail
FOCUS_PAIR = {s['focus_pair']!r}                          # one (true, predicted) confusion to inspect
OLD_SINK = {s['old_sink']!r}                              # a class zero-shot over-predicts
NEW_SINK = {s['new_sink']!r}                              # a class Ramen over-predicts more than zero-shot does"""),
        code("""exp = RecordExperiment(DATASET, seed=SEED, n_per_domain=N_PER_DOMAIN).run()   # cached -> no GPU needed
assert exp.sanity_check(), 'the recorder changed Ramen'
records = exp.records
zs, ra = exp.confusion('zero_shot'), exp.confusion('ramen')
shift = exp.shift()
print(f'{len(records):,} images recorded; vectors: ' +
      ', '.join(f'{k} {tuple(v.shape)} {v.dtype}' for k, v in records.arrays.items()))"""),
        md(f"""## The data

Images and classes present per domain. These are the denominators for everything below. {s['data_note']}"""),
        code("""display(exp.data_table())
support = records.table['label'].value_counts()
print(f'images per class: median {support.median():.0f}, min {support.min()}, max {support.max()}')"""),
        md("""## Headline: zero-shot vs Ramen

Accuracy per domain, with counts. Below that, each image's outcome from zero-shot to Ramen. The net gain is
*fixed − broken*."""),
        code(f"""display(exp.accuracy_table())
ot = exp.outcome_table(by={s['outcome_by']!r})
display(ot)
n = len(records)
for o in ['fixed', 'broken', 'wrong -> same wrong', 'wrong -> other wrong']:
    print(f'{{o:22s}} {{count_pct(int(ot.loc["all", o]), n)}}')
plots.plot_outcomes(ot.drop('all'), title={s['outcomes_title']!r})
plt.show()"""),
        md("""## 1. Per-class confusion: TP, FP, FN, TN

`ConfusionMatrix.per_class()` lists every class. The interesting classes are **sinks**: classes predicted far more
often than they occur, which shows up as a large FP. The table below compares zero-shot and Ramen on the classes
whose FP changes most."""),
        *([code("""print('Full confusion matrices (rows = true class, columns = predicted class):')
for name, cm in [('zero-shot', zs), ('Ramen', ra)]:
    print(name)
    display(cm.to_frame().style.background_gradient(cmap='Blues', axis=None))""")] if small else []),
        code(f"""pc = shift.per_class()
pc['predicted zero_shot'] = zs.counts.sum(0); pc['predicted ramen'] = ra.counts.sum(0)
cols = ['support', 'predicted zero_shot', 'predicted ramen', 'TP zero_shot', 'TP ramen', 'FP zero_shot', 'FP ramen', 'ΔFP',
        'FN zero_shot', 'FN ramen', 'ΔFN']
print('Classes whose false positives Ramen REMOVES most (zero-shot sinks):')
display(pc.sort_values('ΔFP')[cols].head({5 if small else 8}))
print('Classes whose false positives Ramen ADDS most (new sinks):')
display(pc.sort_values('ΔFP', ascending=False)[cols].head({5 if small else 8}))"""),
        code(f"""print('Ramen, per class, sorted by FP (TN is large for every class: N minus the rest):')
display(ra.per_class().sort_values('FP', ascending=False).head({10}))"""),
        md("""### Sink fate and sink origin

**Sink fate.** Take the images zero-shot wrongly sends to `OLD_SINK`. Does Ramen fix them, leave them there, or
move them to another wrong class?

**Sink origin.** Take Ramen's false positives of `NEW_SINK`. What did zero-shot predict for them?"""),
        code("""fate, moved = exp.sink_fate(OLD_SINK)
display(fate.to_frame(f'{OLD_SINK}: images'))
display(moved.to_frame('images'))
n_fp, origin, domains = exp.sink_origin(NEW_SINK)
print(f'Ramen predicts {NEW_SINK} wrongly for {n_fp:,} images (zero-shot: {int(zs.fp[zs._idx(NEW_SINK)]):,})')
display(origin.to_frame('images')); display(domains.head(6).to_frame('images'))"""),
        md("""## 2. One class in detail

Set `FOCUS_CLASS` in the settings cell. Three views:
1. its TP / FN / FP / TN under both sources;
2. where its images go (`predicted_as`, the row of the matrix);
3. which classes get called by its name (`mistaken_for`, the column).

Then the images themselves. Each caption shows the true class, then the zero-shot prediction with its probability
and the probability of the true class, then the same for Ramen, then the domain and the embedding shift."""),
        code("""display(pd.concat({'zero-shot': zs.report(FOCUS_CLASS)[['count', 'of', '%']],
                   'Ramen': ra.report(FOCUS_CLASS)[['count', 'of', '%']]}, axis=1))
plots.plot_class_report(shift, FOCUS_CLASS); plt.show()"""),
        code("""display(pd.concat({'zero-shot': zs.predicted_as(FOCUS_CLASS, 8).reset_index(),
                   'Ramen': ra.predicted_as(FOCUS_CLASS, 8).reset_index()}, axis=1))
display(pd.concat({'zero-shot': zs.mistaken_for(FOCUS_CLASS, 8).reset_index(),
                   'Ramen': ra.mistaken_for(FOCUS_CLASS, 8).reset_index()}, axis=1))"""),
        code("""exp.gallery.show(true=FOCUS_CLASS, correct=False, n=16, title=f'Images of {FOCUS_CLASS} that Ramen gets wrong '
                 f'(most confident first)'); plt.show()
exp.gallery.show(pred=FOCUS_CLASS, correct=False, n=8, title=f'Images Ramen wrongly calls {FOCUS_CLASS}'); plt.show()"""),
        md("""## 3. Confused pairs

These are the largest off-diagonal cells under each source, then the pairs that change most from zero-shot to Ramen."""),
        code("""display(pd.concat({'zero-shot': zs.top_pairs(12), 'Ramen': ra.top_pairs(12)}, axis=1))"""),
        code(f"""changes = pd.concat([shift.pair_changes(10, 'shrank'), shift.pair_changes(10, 'grew')]).drop_duplicates()
plots.plot_pair_changes(changes.iloc[::-1].reset_index(drop=True), title={s['pairs_title']!r}); plt.show()"""),
        code("""true, pred = FOCUS_PAIR
n_zs, n_ra = zs.pair(true, pred), ra.pair(true, pred)
print(f'{true} -> {pred}: zero-shot {n_zs:,}, Ramen {n_ra:,} (of {int(zs.counts[zs._idx(true)].sum()):,} images of {true})')
exp.gallery.show(true=true, pred=pred, source='zero_shot', n=16, order='confident',
                 title=f'Zero-shot calls these {true} images {pred}; the caption shows what Ramen does with them'); plt.show()"""),
        code("""exp.gallery.show(pred=NEW_SINK, correct=False, outcome=['broken', 'wrong -> other wrong'], n=16,
                 title=f'Images Ramen newly sends to {NEW_SINK} (zero-shot predicted something else)'); plt.show()"""),
        md("""## 4. Embeddings before and after adaptation

One step on the norm parameters moves each image's embedding a little. The boxes show how far images move, grouped
by outcome. The PCA plot draws an arrow from each image's embedding before adaptation to its embedding after, for
the images of the focus pair's two classes."""),
        code("""display(exp.embedding_shift_table())
if records.table['domain'].nunique() > 1:
    print('Median shift per domain and outcome (domains move by different amounts, so compare within a row):')
    display(exp.embedding_shift_by_domain())
plots.plot_embedding_shift(records); plt.show()
plots.plot_embedding_moves(records, list(dict.fromkeys(FOCUS_PAIR))); plt.show()"""),
        md("""## 5. One image

`gallery.show_image(i)` shows the image with its top classes under zero-shot and under Ramen. The ★ marks the true
class. `records[i]` returns the raw `ImageRecord`, which holds both probability vectors and both embeddings."""),
        code(f"""i = int({example}.index[0])
exp.gallery.show_image(i); plt.show()
j = int(records.where(outcome='broken').sort_values('zs_conf', ascending=False).index[0])
exp.gallery.show_image(j); plt.show()
r = records[i]
print(r.true_class, r.zs_class, '->', r.pred_class, '| emb_before', r.emb_before.shape, '| top-3 Ramen', r.top('ramen', 3, records.classes))"""),
        md(interpretation),
        md(GLOSSARY),
    ]


def write(key):
    s = NOTEBOOKS[key]
    interp = INTERPRETATIONS.get(key, '## Interpretation\n\n_(filled in after the run)_')
    nb = nbf.v4.new_notebook(cells=cells_for(s, interp),
                             metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}})
    nbf.write(nb, s['file'])
    print('wrote', s['file'])


INTERPRETATIONS = {
    '04_domainnet_full': """## Interpretation (seed 0, full DomainNet, 596,006 images)

**The subset's picture holds at full scale: Ramen mostly relocates errors.** Zero-shot gets 314,840 of 596,006
right and Ramen gets 324,966: 55.37 → 56.80 domain-averaged, +1.43, far above the 0.11 noise floor. Ramen fixes
20,428 images (3.4%) and breaks 10,302 (1.7%). Another 99,028 (16.6%) switch from one wrong class to another, and
75,958 of those are quickdraw. In quickdraw, 7 images switch wrong class for every 1 that gets fixed (75,958 vs
10,521).

**Zero-shot's squiggle sink is relocated, not repaired.** Zero-shot wrongly calls 44,623 images *squiggle*. Ramen
fixes 3,613 of them (8.1%), leaves 11,331 at squiggle, and moves 29,679 (66.5%) to other wrong classes: boomerang
1,848, pillow 1,248, teapot 1,005, rake 990, pear 934, hockey_stick 923. *envelope* is worse: of its 14,586 false
positives, Ramen fixes 800 (5.5%) and moves 9,634, with 3,541 going to *pillow*. Squiggle also loses 254 of its own
939 correct images.

**Pillow is the main new sink, and it is fed by envelope.** Ramen wrongly predicts pillow 6,484 times, against
1,147 for zero-shot. Of those 6,484, 3,570 were zero-shot *envelope* predictions and 1,248 *squiggle*; 6,129 are
quickdraw. The pairs that grow most all end in pillow: square, door, map, laptop, keyboard and picture_frame
(+151 to +243 images each). These are rectangle drawings. Teapot (+3,302 false positives), helmet (+1,972) and
toaster (+1,942) grow the same way, mainly from squiggle and pear. Ramen's median confidence on its new false
positives is 0.30–0.49, against 0.97 for its correct predictions.

**dog, at full scale.** 2,609 images. Zero-shot gets 1,836 right and Ramen 1,776 (−60). Ramen also cuts dog's
false positives from 742 to 511. Squiggle and envelope dominate the error picture, so a single animal class
changes little.

**Embedding shift: the pooled difference is mostly domain mix.** Pooled over all images, the median shift is 0.050
for fixed images and 0.030 for images that stay right. But quickdraw moves the most (median 0.056 even for
right → right) and supplies most of the fixed and switched images. Within each domain, fixed images move only
0.001–0.008 more than images that stay right (table above). On the 12,000-image subset, images that stay right
move *most* in every domain. That disagreement is not explained. One guess, untested: a short stream fills the
memory differently, so the retrieved gradients differ. Either way, the size of the shift does not separate helpful
steps from harmful ones on DomainNet.

**Caveats.** Seed 0 only. These are counts from one stream order. The headline gain is the cached Ramen run's.""",
    '03_cifar100c': """## Interpretation (seed 0, full CIFAR-100-C, 150,000 images)

**More errors change wrong class than get fixed.** Zero-shot gets 55,635 of 150,000 right (37.09%) and Ramen gets
69,144 (46.10%). Ramen fixes 20,301 images (13.5%) and breaks 6,792 (4.5%), a net gain of 13,509. Another 43,764
images (29.2%) move from one wrong class to a different one. That is twice the number fixed, and it holds on every
corruption.

**Zero-shot's sinks are mostly relocated, as on DomainNet.** Zero-shot wrongly calls 4,264 images *worm*. Ramen
fixes 1,095 of them (26%), leaves 675 at worm, and moves 2,494 (58%) to other wrong classes (crab 139, snake 129,
trout 128, ...). *mouse* behaves the same way: 4,536 false positives, 825 fixed, 2,866 moved.

**De-sinking can cost a class its own correct predictions.** *keyboard* drops from 2,567 false positives to 256,
but its true positives also fall from 933 to 625 of 1,500. The images it loses go to television, mountain and
road.

**New sinks appear: crab, tractor, lion.** Crab false positives rise from 396 to 1,628, tractor from 543 to 1,524,
and lion from 459 to 1,355. Most of these new errors are unsure: median Ramen confidence is 0.29 for the 1,415
images newly sent to crab and 0.46 for tractor, against 0.87 for Ramen's correct predictions. The most confident
ones are not unsure, though. The crab gallery shows blurred animals with legs radiating from a body (spiders,
turtles, lizards, lobsters) called crab at 0.94–0.99. glass_blur alone gives 429 of the crab errors. tractor is
one of the classes the earlier analyses flagged as a Ramen bias (`watch_classes` in the harness).

**Fine-grained confusions grow.** maple_tree → oak_tree rises from 476 to 509 images and whale → dolphin from 344
to 418. worm → snake rises from 98 to 268 while snake → worm falls from 459 to 220, so Ramen reverses the direction
of that confusion rather than resolving it.

**The size of the embedding step carries no signal here.** Median shift is 0.147–0.161 for every outcome, so fixed
and broken images move the same distance (0.161 vs 0.151). This matches the DomainNet subset, not EuroSAT.

**Caveats.** Seed 0 only. The gain (+9.01) is far above the 0.21 noise floor, but the class and pair counts come
from one stream order.""",
    '02_eurosat': """## Interpretation (seed 0, full EuroSAT, 27,000 images)

**On EuroSAT, Ramen really does fix most of what it changes.** Zero-shot gets 12,984 of 27,000 images right (48.09%)
and Ramen gets 16,910 (62.63%). It fixes 5,008 images (18.5%) and breaks 1,082 (4.0%), a net gain of 3,926 images.
Another 2,871 (10.6%) move from one wrong class to another.

**Ramen repairs the main zero-shot sink instead of relocating it, which is the opposite of DomainNet.** Zero-shot
wrongly calls 5,353 images *Highway or Road*. Ramen fixes 2,322 of them (43%), leaves 2,287 at Highway, and moves
744 elsewhere. Its Highway false positives fall to 3,375. On the 12,000-image DomainNet subset, Ramen fixed only
32 of squiggle's 552 false positives.

**What remains is images that do contain a road-like line.** Sea or Lake → Highway falls from 2,024 to 1,183 of
3,000 lake images. The lake images Ramen still calls Highway show a thin linear structure (a boat wake, a trail,
a shoreline), and Ramen raises its confidence in Highway from about 0.6 to 0.9–1.0. Herbaceous Vegetation Land
images with visible dirt tracks go the same way, to Highway at confidence 1.00.

**Herbaceous Vegetation Land was already lost under zero-shot. Ramen removes the last of it.** Zero-shot gets 200
of 3,000 right (6.67%); Ramen gets 9 (0.30%) and predicts the class only 21 times in total. Its images go to Forest
(1,497; zero-shot 1,476) and Highway (710; zero-shot 590). Ramen files every image in the memory queue of its
zero-shot class, so a class with few correct zero-shot predictions has a small, impure queue. Every confident step
then pulls its images toward the classes that dominate.

**The new sink sits inside the crop and pasture group.** Ramen wrongly predicts *Permanent Crop Land* for 608 images,
against 237 for zero-shot. Of those 608, 211 were zero-shot *Annual Crop Land* predictions and 187 were *Pasture
Land*. These are confusions between neighbouring agricultural classes, not unrelated mistakes. They are also
low-confidence. For the 546 images that zero-shot did not call Permanent Crop Land, Ramen's median confidence in
it is 0.37 (90th percentile 0.48), against 0.89 for Ramen's correct predictions. Ramen's remaining Highway false
positives are confident (median 0.63 over 3,375 images). A confidence threshold could catch the crop flips but
not the road ones. Earlier gates failed on Ramen as a whole, so treat this as specific to EuroSAT.

**On EuroSAT, the embedding step is larger and its size does carry signal.** The median shift is 0.225 for fixed
images and 0.143 for images that stay right, against 0.04–0.05 for every outcome on DomainNet. Here, images that
move further are more often fixed. That fits a single-domain stream: the memory never evicts (capacity 7,500 per
class is more than the 27,000-image stream), and every retrieved gradient comes from the same satellite domain.

**Caveats.** Seed 0 only. The gain (+14.54) is far above the 1.51 noise floor, but the per-class numbers are counts
from one stream order.""",
    '01_domainnet_subset': """## Interpretation (seed 0, 12,000-image subset)

**Ramen mostly moves zero-shot's errors to other wrong classes. It rarely corrects them.** Ramen fixes 268 of
12,000 images (2.2%) and breaks 147 (1.2%). Another 1,143 (9.5%) go from one wrong class to a different wrong
class, and 664 of those are quickdraw images. Net gain: +1.01 points domain-averaged (quickdraw +3.10, real −0.15).

**De-sinking is mostly relocation.** Zero-shot wrongly calls 552 images *squiggle*, and Ramen cuts that to 250.
But of those 552 images, Ramen fixes only 32. It sends 271 to another wrong class, mainly boomerang 26, pear 20,
pillow 15, circle 10, smiley_face 10. Envelope behaves the same way: of its 184 false positives, Ramen fixes 8
and sends 84 elsewhere, 35 of them to *pillow*.

**New sinks come from the old ones.** Ramen wrongly predicts *pillow* for 73 images (zero-shot: 10). Of those 73,
36 were zero-shot *envelope* errors and 15 were *squiggle* errors, and 65 are quickdraw images. The gallery shows
why: they are rectangle-like sketches (washing machine, dishwasher, stereo) that zero-shot already got wrong. This
repeats on single images what the earlier confusion-shift analysis found in aggregate: Ramen de-sinks without
fixing the sink's errors, and it creates new sinks.

**Single classes are too small to read here.** *dog* has 50 images in the subset, and zero-shot and Ramen agree
on its TP / FN (38 / 12). Its errors are scattered: no wrong class takes more than 2 of them. Pairs with 10 or
more images (goatee → beard, bus → school_bus, postcard → envelope, octagon → hexagon) are genuine fine-grained
confusions, and Ramen leaves them almost unchanged.

**Embeddings.** One step moves embeddings by a median 1 − cos of 0.04–0.05 for every outcome. Fixed and broken
images move slightly *less* than images that stay right (median 0.044 / 0.042 vs 0.051). So the size of the move
does not tell the step helped. Whether it did depends on direction (toward which class), not magnitude.
On the full run (notebook 04) the pooled ordering reverses. That reversal comes mostly from the domain mix, and
the subset's ordering is not explained; see notebook 04.

**Caveats.** This is one seed on a subset, so Ramen's memory is not the full run's. The counts above describe
this stream and are not population estimates. No noise floor applies, because nothing here is a claimed gain.

**Next.** Run the full dataset (`python run_records.py DomainNet 0`, about 40 min, about 2 GB) to get class-level
counts large enough to read. Then measure the direction of the embedding shift: does `emb_after − emb_before`
point toward the text embedding of the class Ramen moves the image to (for example pillow)?""",
}

if __name__ == '__main__':
    for key in (sys.argv[1:] or NOTEBOOKS):
        write(key)
