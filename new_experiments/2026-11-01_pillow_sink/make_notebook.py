"""
Write the sink case-study notebook (not executed).

    python make_notebook.py
Execute with:
    ~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace 01_pillow_sink_domainnet.ipynb

Another sink class: copy the NOTEBOOK entry with another target (the cells only use TARGET).
"""
import nbformat as nbf

import sink_runs  # noqa: F401  sets up the import paths

from case_study import EXAMPLES  # noqa: E402

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

NOTEBOOK = dict(file='01_pillow_sink_domainnet.ipynb', dataset='DomainNet', seed=0, target='pillow')

HEADER = """# Why does Ramen send DomainNet images to *pillow*, and does a true-label cache stop it?

**Question.** After Ramen's adaptation, DomainNet images are predicted *pillow* far more often than before.
For those images: what was the class before adaptation, what is it after, and what was in Ramen's cache when
each image arrived? And what changes if the cache is filled using the true labels instead (an oracle)?

**What was already known** (`2026-10-28_confusion_records`, notebook 04): Ramen's wrong *pillow* predictions
grow from 1,147 (zero-shot) to 6,484. They are mostly quickdraw images that zero-shot called *envelope* or
*squiggle*. The sink is moved to another wrong class, not repaired.

**Hypotheses** (written before the cache replay was run):
- **H1, contamination.** Ramen files each image under its *zero-shot* prediction, so the *envelope*, *squiggle*
  and *pillow* queues are full of quickdraw rectangles of other classes. If those mis-filed entries cause the
  sink, the oracle cache (every queue pure) should keep fewer than 20% of Ramen's wrong *pillow* predictions
  at *pillow*.
- **H2, neighbours.** If instead the update is set by *who the image's neighbours are*, whatever queue they are
  filed in, the oracle cache retrieves largely the same entries (high "weight on entries both caches
  retrieve"). The sink then survives.

**Why this test.** Ramen's cache never depends on its adapted outputs. Each image is written, before retrieval,
with its zero-shot feature, its zero-shot entropy and its own gradient, so both caches can be **replayed
offline** (no GPU) from the recorded features, and replayed exactly. Only the oracle's adapted predictions need
a GPU run.

**What is compared.**

| Source | What it is | Uses true labels? |
|---|---|---|
| zero-shot | CLIP before adaptation (the pass that files each image in the cache) | no |
| Ramen | Ramen as published: each image is filed in the queue of its **zero-shot** prediction | no |
| oracle cache | `RamenTrueLabel`, `filing: true_label`: each image is filed in the queue of its **true** label; everything else is Ramen's | **yes: an upper bound, not a method** |

**Data.** DomainNet, 6 domains, 345 classes, 596,006 images, CLIP ViT-B/32, one mixed (shuffled) stream, batch
size 100. Config `cfg/DomainNet/Ramen.yaml`: queue capacity 300, top-k 10, β 5, signSGD lr 0.01. The oracle
uses the same values. Both runs use the default fp16 priorities, so full queues stop changing after 65,520
images.

**Seeds.** Seed 0 only. A seed sets the order of the stream. The *pillow* counts for seeds 1–4 come from the
cached plain Ramen runs (section 4.4). This is a descriptive case study, not an accuracy claim, so the noise floor
(0.11 pts on DomainNet) is not the test. The counts are.

**How the results were produced.**
```
cd new_experiments/2026-10-28_confusion_records
CUDA_VISIBLE_DEVICES=0 python run_records.py DomainNet 0                                     # Ramen, recorded (2026-10-28)
CUDA_VISIBLE_DEVICES=0 python run_records.py DomainNet 0 --variant true_label --skip-check   # oracle, recorded (2026-11-01, 36 min)
```
`--skip-check` skips the built-in check against an uncached plain `true_label` run. Cell 3.1 checks the recording
against the cached `oracle_file_true` run (same oracle, earlier code path) instead. Code in this folder:
`sink_runs.py`, `sink_trace.py`, `cache_replay.py`, `example_card.py`, `sink_plots.py`, `case_study.py`. The
cache replay is stored as `results/DomainNet_..._s0_pillow_sink_replay_pillow.pkl`.

**How to read this notebook.**
1. Setup
2. The data
3. Sanity checks
4. Where *pillow* predictions come from
5. What the two caches retrieve, over all images
6. Examples, image by image
7. Interpretation
8. Glossary

Counts are given with percentages. Accuracies are domain-averaged unless stated."""

SETUP = """import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # no GPU needed: the text features and the replay are cached
sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
import ramen_analysis  # palette; puts src/ on the path

from case_study import SinkCaseStudy
from sink_plots import plot_prediction_counts, plot_routes, plot_update_sources, plot_queue_history

pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
DATASET, SEED, TARGET = {dataset!r}, {seed}, {target!r}
study = SinkCaseStudy(DATASET, seed=SEED, target=TARGET)
runs, trace, card = study.runs, study.trace, study.card"""

CELLS = [
    md(HEADER),
    md('## 1. Setup'),
    code(SETUP.format(**NOTEBOOK)),
    md('## 2. The data\nImages and classes per domain. These are the denominators for everything below.'),
    code('runs.data_table()'),
    md('Accuracy of the three sources per domain (correct images and %). The last row is the domain average, the '
       'headline convention.'),
    code('runs.accuracy_table()'),
    md('## 3. Sanity checks\n### 3.1 The recorded runs\nEach recording must predict exactly what its plain cached '
       'run predicts (100.00%). Both recordings must see the same stream, starting from the same zero-shot '
       'embeddings. The logits used below are recomputed as 100 · embedding · text featureᵀ: their argmax must '
       'give the recorded zero-shot class, up to fp16 near-ties.'),
    code('runs.sanity_check()'),
    md('### 3.2 The cache replay\nThe zero-shot-filed replay is compared with what a `RamenActive` *all* run '
       '(identical to Ramen on this stream) logged for 200 random images: the contents of the predicted class\'s '
       'queue, and the 8 highest-weight retrieved entries. The oracle replay runs the same code with a different '
       'filing rule, so it has no log of its own to check against.'),
    code('study.validate_replay()'),
    md('## 4. Where *pillow* predictions come from\n### 4.1 How often each model predicts *pillow*'),
    code('c = trace.counts(); display(c); print(f"truly {TARGET}: {c.attrs[\'true images\']:,} images")\n'
         'plot_prediction_counts(trace); plt.show()'),
    md('### 4.2 Routes into *pillow*: the zero-shot class of Ramen\'s wrong *pillow* predictions\n'
       'Each route is split by what the oracle cache predicts for the same image: still *pillow*, correct, or '
       'another wrong class.'),
    code('display(trace.routes())\nplot_routes(trace); plt.show()\nprint(trace.oracle_fate().to_string())'),
    md('What those images truly are, where they are, and where the oracle sends the ones it moves away from '
       '*pillow*:'),
    code('display(trace.true_classes())\ndisplay(trace.by_domain())\ndisplay(trace.oracle_moves_to())'),
    md('Where the oracle sends them is mostly **back to their zero-shot class**. On quickdraw the oracle cache '
       'keeps the zero-shot prediction much more often than Ramen does. It adapts less there; it does not repair.'),
    code('display(trace.oracle_returns_to_zero_shot().to_frame(\'\'))\ndisplay(trace.same_as_zero_shot())'),
    md('### 4.3 Is it the same on other seeds?\nPlain Ramen, seeds 0–4. A seed reorders the stream.'),
    code('trace.seed_table()'),
    md('## 5. What the two caches retrieve, over all images\n### 5.1 The *pillow* queue over the stream\n'
       'How much of the *pillow* queue is truly *pillow*, and how much comes from quickdraw, under each cache. '
       'After the dashed line the fp16 counter has saturated: new entries replace the older ones once, and the queue '
       'freezes at the tick mark.'),
    code('plot_queue_history(study.result.history, TARGET, len(runs.table), domain_name="quickdraw",\n'
         '                   domain_col=study.domain_col("quickdraw")); plt.show()'),
    md('### 5.2 Where each image\'s update comes from\nFor each test image, Ramen sums the weighted gradients of '
       'the retrieved entries: the top 10 of every non-empty queue, weighted by e^(−H)·e^(−5d). The table gives '
       'the mean share of that weight, in %, coming from each source, for four groups of images. *Entries '
       'zero-shot-predicted pillow* matters because an entry\'s stored gradient sharpens **its own** zero-shot '
       'prediction, whichever queue it is filed in.'),
    code('g = study.group_summary(); display(g)\n'
         'w = g.loc[[k for k in g.index if k[0].startswith("Ramen wrong")]]\n'
         'ov = w.xs("oracle cache", level=1)["weight on entries both caches retrieve"]\n'
         'dom = w["entries from query\'s domain"]\n'
         'title = (f"Wrong {TARGET}s: the oracle cache retrieves different entries (only {ov.min():.0f}–{ov.max():.0f}% '
         'shared), "\n'
         '         f"{dom.xs(\'oracle cache\', level=1).min():.0f}–{dom.xs(\'oracle cache\', level=1).max():.0f}% '
         'from the image\'s own domain "\n'
         '         f"(Ramen: {dom.xs(\'Ramen cache\', level=1).min():.0f}–{dom.xs(\'Ramen cache\', level=1).max():.0f}%)")\n'
         'plot_update_sources(study.result.summary, study.groups, TARGET, title=title); plt.show()'),
    md('## 6. Examples, image by image\nImages drawn at random (fixed seed, not hand-picked) after the first 20% '
       'of the stream, so the cache is full. Each example shows:\n'
       '- the logits and probabilities of the classes that matter, under the three sources (table, then figure);\n'
       '- for each cache: the share of the update per queue, the 10 highest-weight retrieved entries, and the '
       'newest entries of the *pillow* queue at that moment;\n'
       '- the summary of where the update weight came from.'),
    code('study.examples'),
]
n_examples = sum(n for *_, n in EXAMPLES)
CELLS += [code(f'study.show_example({i})') for i in range(n_examples)]
CELLS += [
    md('''## 7. Interpretation

**What the numbers show (seed 0, 596,006 images).**
- **The sink.** Zero-shot predicts *pillow* 2,101 times (1,147 wrong); Ramen predicts it 7,605 times (6,484
  wrong). 1,736 images are truly *pillow*. 6,129 of the 6,484 wrong ones (94.5%) are quickdraw sketches of
  rectangular objects (square, laptop, picture_frame, door, map, ...). Their zero-shot class was mostly *envelope*
  (3,570, 55.1%) or *squiggle* (1,248, 19.2%). The sink appears on every seed: 6,142–6,484 wrong *pillow*
  predictions over seeds 0–4.
- **H1 (contamination) is rejected.** With every queue filed by true label, 2,166 of the 6,484 (33.4%) stay at
  *pillow*, above the 20% the hypothesis allowed, and only 42 (0.6%) become correct. The oracle does remove
  two-thirds of the sink, but 3,499 of the 4,318 images it moves (81%) go **back to their zero-shot class**
  (mostly *envelope* and *squiggle*). On quickdraw the oracle keeps the zero-shot prediction for 66.8% of images,
  Ramen for 48.0%. Its quickdraw accuracy (11.05%) is below zero-shot's (13.33%) and Ramen's (17.54%). A pure
  cache makes Ramen adapt less on quickdraw; it does not fix the rectangles.
- **H2 as written is also rejected.** The two caches do *not* retrieve the same entries: for the wrong *pillow*s
  only 12–23% of the update weight falls on entries both retrieve.
- **What the caches do show.** Every image's update is a vote of all 345 queues with about 1% each. The *pillow*
  queue gives only about 1%, and entries that are truly *pillow* about 1%. So the sink does not come from
  retrieving *pillow* entries. What differs is where the entries come from:
  - **Oracle cache:** 90–99% of the weight comes from the image's own domain, and 78–89% sits on entries that are
    zero-shot errors. Each class queue now holds that class's own quickdraw sketches, and for a rectangle query
    the nearest ones are rectangle sketches (postcard, passport, calculator) that zero-shot calls *envelope* or
    *squiggle*. Their stored gradients sharpen *those* predictions, which fits the images returning to *envelope*
    and *squiggle*.
  - **Ramen cache:** 62–67% of the weight comes from quickdraw and 42–47% from zero-shot errors. The top-weighted
    entries are confident, correctly classified sketches of unrelated classes (palm tree, stop sign, frog),
    because quickdraw rectangles have been filed away under *envelope*, *squiggle* and *pillow*.
- **Controls.** For true *pillow* images that both models get right (mostly real photos), the two caches look alike:
  about 56–61% shared weight and 57–60% from the image's own domain. Filing only matters where zero-shot is badly
  wrong, i.e. quickdraw. For zero-shot *envelope* errors that Ramen fixes (800), the Ramen-cache statistics look
  like those of the images it sends to *pillow* (60% same domain, 42% zero-shot errors). These retrieval
  statistics do not separate a repair from a sink, as earlier replays also found.
- **The *pillow* queue itself.** In Ramen's cache it is 44% truly *pillow* and 41% quickdraw, and it stops changing
  after image 147,057 (24.7% of the stream; fp16 counter). So the late examples see a cache built from the first
  quarter of the stream.

**What this does not show.** The replay knows which entries are retrieved and with what weight, but not their
gradients. So it cannot say *which* entries turn the summed step toward *pillow*. The leading candidate is the
mix of confident non-rectangle quickdraw entries and other-domain entries, which pure queues remove. That is a
hypothesis, not a result. The mechanism numbers are from seed 0 only.

**Next step.** Step 5 of the plan (GPU, about 200 images): recompute the retrieved entries' gradients for wrong
*pillow* images and remove one group at a time from Ramen's retrieval: other-domain entries, confident correct
entries, the *envelope* / *squiggle* / *pillow* queues. The prediction is that removing other-domain entries
sends them back to *envelope* / *squiggle* as the oracle does. If none of these groups changes the prediction, the
step direction is a property of the whole 345-queue average, and a fix has to act on the update (e.g. its size
per domain) rather than on what is retrieved.'''),
    md("""## 8. Glossary

| term | meaning |
|---|---|
| zero-shot | CLIP classifies by cosine similarity between the image embedding and each class's text embedding (7 prompt templates averaged). No adaptation. |
| Ramen | test-time adaptation. Each image is written into a class queue with its own entropy gradient. The gradients of the nearest stored entries are retrieved, one signSGD step is taken on the image encoder's LayerNorm parameters, the image is predicted, and the parameters are reset. |
| cache / queue | Ramen's memory: one queue of up to 300 entries per class. An entry is (zero-shot feature, own gradient, zero-shot entropy, arrival counter). |
| filing | which queue an image is written to. Ramen uses its zero-shot prediction; the oracle uses its true label. |
| oracle cache | the Ramen run whose cache is filed by true label (`RamenTrueLabel`). It uses ground truth, so it is an upper bound on what a perfectly filed cache could do, not a method. |
| retrieved entries / update weight | for each test image, the top 10 nearest entries of every non-empty queue, weighted e^(−H)·e^(−β·d). Queues are averaged. "Share of the update" = an entry's weight / total weight. |
| self | the test image's own entry: it is written before retrieval, at distance 0. |
| sink | a class that a model predicts for many images of other classes. |
| route | the zero-shot class an image had before Ramen moved it to the sink. |
| fp16 priority freeze | Ramen counts arrivals in fp16, which becomes inf after 65,520 images. After that, new entries (priority inf) replace a full queue's older finite-priority entries once, and then the queue stops changing (the *pillow* queue last changes at image 147,057). |
| zero-shot error (entry) | a cache entry whose zero-shot prediction is not its true class; its stored gradient sharpens that wrong class. |
| logit | 100 · cos(image embedding, class text embedding). The probability is the softmax over the 345 logits. |
| domain average | the mean of the 6 per-domain accuracies (the headline convention). |
"""),
]


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nbf.write(nb, NOTEBOOK['file'])
    print('wrote', NOTEBOOK['file'])


if __name__ == '__main__':
    build()
