"""
Write the Ramen-baseline notebook of one dataset (same sections for every dataset; the dataset-specific text is in
NOTEBOOKS). Execute it afterwards with the RAMEN kernel:

    python make_baseline_notebook.py ImageNetC5K
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=ramen-env-kernel 02_ramen_imagenetc.ipynb

Add a dataset: one entry in NOTEBOOKS (its numbers in `interpretation` come from the executed notebook, so write that
entry after a first execution).
"""
import sys

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

NOTEBOOKS = {
    'DomainNet': dict(
        file='01_ramen_domainnet.ipynb', title='DomainNet', by='domain', window=10_000,
        data="""DomainNet, 6 domains (clipart, infograph, painting, quickdraw, real, sketch), 345 classes, 596,006 images,
mixed into one shuffled stream with batch size 100. Model: CLIP ViT-B/32, default 7-template prompt ensemble.""",
        config='cfg/DomainNet/Ramen.yaml: capacity 300 per class, top-k 10, β 5, signSGD lr 0.01',
        data_note='Domains differ a lot in size: quickdraw and real are 29% each of the stream, clipart 8%.',
        smoke="""200 images per domain = 1,200 images, seed 0:

- `zero_shot_filing` == `Ramen` on 1,200 of 1,200 images (100.00%)
- `true_label` == `RamenOracle` with `oracle: file_true` (the earlier implementation, 2026-10-23_cache_contamination)
  on 1,200 of 1,200 images (100.00%)

So the new class is Ramen when switched off, and it reproduces the earlier true-label oracle when switched on.""",
        pooled_note="""Pooled accuracy (all images weighted equally) is lower than the domain average because quickdraw, the
hardest domain, is the largest.""",
        titles=('Ramen is above zero-shot on every domain', 'Most of the gain comes from quickdraw',
                'The gain over zero-shot is flat along the stream'),
        interpretation="""- **Ramen beats zero-shot by +1.43 points** (55.37 → 56.80, domain-averaged, seed 0). That is +10,121 images
  (pooled 52.83% → 54.52%). Ramen changes 21.8% of predictions to reach this: 20,419 images fixed against 10,298 broken.
  Most of what Ramen changes is wrong → wrong, and about one change in three that matters is a break.
- **The gain is concentrated in quickdraw**: +4.21 points, net +7,264 images, 72% of the total net gain. The other five
  domains gain +0.15 (real) to +1.12 (painting). Real is almost a wash (2,560 fixed, 2,298 broken).
- **Seed-to-seed variation is tiny**: 56.79–56.85 over seeds 0–4, so the **noise floor is 0.11 points**. A true-label
  filing gain smaller than 0.11 on DomainNet is not distinguishable from changing the stream order.
- **What to expect next.** `RamenOracle file_true` (the same oracle, 2026-10-23) gave **−1.50** vs Ramen on seed 0
  (quickdraw −6.49). The next step reruns it with `RamenTrueLabel` and looks at why pure queues hurt.
  The class keeps per-batch logs for that (`zs_wrong`, `nonempty_queues`). One candidate is the stored gradient: it still
  sharpens the zero-shot prediction, so a pure queue of class c holds gradients that pull toward *other* classes
  whenever zero-shot was wrong.
- Caveat: per-domain numbers are seed 0 only.""",
    ),
    'ImageNetC5K': dict(
        file='02_ramen_imagenetc.ipynb', title='ImageNet-C', by='domain', window=2_000,
        data="""ImageNet-C, severity 5, the 5,000-image subset (`src/datasets/imagenet_test_image_ids.txt`) under each of
15 corruptions = 75,000 images. The same 5,000 images appear once per corruption. Labels cover 997 of the 1,000 classes.
All corruptions are mixed into one shuffled stream with batch size 100. Model: CLIP ViT-B/16, default 7-template prompt
ensemble. Here a *domain* is a corruption.""",
        config='cfg/ImageNetC5K/Ramen.yaml: capacity 75 per class, top-k 1, β 0, signSGD lr 0.01',
        data_note="""Every corruption has 5,000 images, so the domain average and the pooled accuracy are the same number.
Zero-shot accuracy ranges from 14.7% (elastic transform) to 57.3% (brightness).""",
        smoke="""200 images per corruption = 3,000 images, seed 0:

- `zero_shot_filing` == `Ramen` on 3,000 of 3,000 images (100.00%)

(The match of `true_label` with the earlier `RamenOracle file_true` was checked on DomainNet; the code path is the same.)""",
        pooled_note='All corruptions have 5,000 images, so pooled = domain-averaged here.',
        titles=('Ramen is above zero-shot on every corruption', 'The gain is spread over all corruptions',
                'The gain grows as the memory fills'),
        interpretation="""- **Ramen beats zero-shot by +5.79 points** (27.29 → 33.08, seed 0), +4,341 of 75,000 images. Ramen changes
  52.5% of predictions: 7,205 images fixed and 2,864 broken. Most changes are wrong → wrong, as on DomainNet, but the
  fixed:broken ratio is better (2.5:1 against 2.0:1).
- **Unlike DomainNet, the gain is spread out**: every corruption gains, from +0.88 (brightness, the easiest, 57.3% zero-shot)
  to +10.72 (contrast). No corruption carries more than 12.3% of the net gain (contrast, net +536). The corruptions zero-shot
  finds hardest (elastic, the three noises, glass blur, all ≤ 17% zero-shot) are among the largest gains, next to contrast.
- **Seeds**: 32.91–33.17 over seeds 0–4, so the **noise floor is 0.53 points**, about 5× DomainNet's. The stream is 8×
  shorter and a 1,000-class memory with capacity 75 is much sparser, so the order matters more. A true-label filing
  change must move ImageNet-C by more than 0.53 to count.
- **The gain grows along the stream** (pooled gain over zero-shot by thirds of the stream, mean over seeds 0–4):
  +5.0 in the first 25,000 images, +6.2 in the last 25,000, higher at the end on 5 of 5 seeds. With 1,000 class
  queues of 75 entries, the memory takes much of the stream to fill. DomainNet's gain is flat by thirds (1.7 points throughout).
- **What to expect next.** `RamenOracle file_true` gave **−0.71** vs Ramen on ImageNet-C seed 0 (2026-10-23), below the
  floor. The smoke subset here points the same way (−0.50 on 3,000 images). The full `true_label` run is the next step.
- Caveat: per-corruption numbers are seed 0 only.""",
    ),
    'EuroSAT': dict(
        file='03_ramen_eurosat.ipynb', title='EuroSAT', by='class', window=1_000,
        data="""EuroSAT RGB (`~/data/EuroSAT/2750`), 27,000 Sentinel-2 satellite images of 64×64 pixels, 10 land-cover
classes (2,000–3,000 images each), **one domain**: the shift is natural photos → satellite images. One shuffled stream
with batch size 100. Model: CLIP ViT-B/16 with the single prompt "a centered satellite photo of {}." (CoOp class names).
With one domain, the per-domain sections are **per true class** here.""",
        config='cfg/EuroSAT/Ramen.yaml: capacity 7,500 per class, top-k 50, β 5, signSGD lr 0.01 (cache settings taken from CIFAR-10-C, not tuned on EuroSAT)',
        data_note='One domain, so the domain average is the pooled accuracy.',
        smoke="""200 images, seed 0:

- `zero_shot_filing` == `Ramen` on 200 of 200 images (100.00%)

(Only two batches. The match of `true_label` with `RamenOracle file_true` was checked on DomainNet, 1,200 images.)""",
        pooled_note='One domain: pooled = domain-averaged.',
        titles=('Ramen gains on 8 of 10 classes and loses on 2', 'Two classes carry 60% of the net gain',
                'The gain shrinks along the stream'),
        extra=[md("""### 4b'. Which classes each run predicts

How often each class is predicted, next to how often it is the true class. A class predicted much less often than it
occurs is being absorbed by others."""),
               code("exp.prediction_counts(['NoAdapt', 'Ramen'])")],
        interpretation="""- **Ramen beats zero-shot by +14.56 points** (48.07 → 62.63, seed 0), +3,932 of 27,000 images: 5,007 fixed,
  1,075 broken (4.7:1, the best ratio of the three datasets). Ramen changes 33.2% of predictions.
- **Two classes carry most of the gain**: Sea or Lake (+41.3 points, net +1,239, 31.5%) and Forest (+37.3, net +1,120,
  28.5%). Annual Crop Land and Highway or Road add another 31%.
- **Two classes get worse**: Herbaceous Vegetation Land (6.67% → 0.30%, net −191) and Pasture Land (53.6% → 46.5%,
  net −143). Zero-shot predicts Herbaceous Vegetation Land only 531 times for 3,000 true images, so its queue ends with
  531 entries, of which 200 really are Herbaceous (37.7%). After adapting, Ramen predicts it only 21 times. True-label
  filing would give this queue all 3,000 Herbaceous images, so this is where the oracle could change most on this dataset.
- **Seeds**: 61.93–62.68 over seeds 0–4, so the **noise floor is 1.51 points**, the largest of the three. With 10 classes
  and capacity 7,500 per class the memory never evicts (10 × 7,500 > 27,000 images), so it is the stream order
  alone that changes what each image retrieves. A true-label change
  must move EuroSAT by more than 1.51 to count.
- **The gain shrinks along the stream** (pooled gain over zero-shot by thirds of the stream, mean over seeds 0–4):
  +15.2 in the first 9,000 images, +13.7 in the last 9,000, lower at the end on 5 of 5 seeds. The memory never evicts,
  so late images retrieve from a memory full of earlier images filed under their zero-shot prediction. Whether the
  growing queues of absorbing classes (Highway or Road, Annual Crop Land) cause the decline is not tested here.
- No earlier true-label result exists on EuroSAT. In the 200-image smoke test `true_label` was −0.50 (1 image), which says nothing.
- Caveat: per-class numbers are seed 0 only.""",
    ),
}


def header(d, ds):
    group = 'class' if d['by'] == 'class' else 'domain'
    return f"""# Ramen on {d['title']}: the baseline for the true-label cache experiment

**Question.** How well does Ramen do on {d['title']}, per {group} and across seeds? This notebook fixes the baseline and
the noise floor that the true-label cache oracle is compared against.

**The experiment this folder builds up to.** Ramen files every test image into the memory queue of its *zero-shot
prediction*, so every zero-shot error lands in the wrong class queue. The oracle `true_label` files each image under
its **true label** instead (it uses ground truth, so it is an upper bound, not a method). Everything else (stored
gradient, retrieval, weighting, step, reset) stays Ramen's. This notebook only covers the baseline.

**Runs in this notebook.**

| Run name | What it is | Uses true labels? |
|---|---|---|
| `NoAdapt` | CLIP zero-shot, no adaptation | no |
| `Ramen` | Ramen ({d['config']}) | no |

**Data.** {d['data']}

**Seeds.** A seed sets the stream order (which images arrive together and in what order). NoAdapt is identical for every seed.
Seed 0 is used for the per-{group} analysis; seeds 0–4 of Ramen give the noise floor.

**How the runs were produced.** `exp.run(...)` in `new_experiments/ramen_analysis.py` through
`true_label_cache.TrueLabelCacheExperiment` (this folder), with the settings in its `DATASETS` (num_workers changes the stream order, so keep it).
Results are cached in `new_experiments/results/{ds}_<model>_mixed_nall_bs100_s<seed>_<run>.pkl`.
A missing run is computed on the GPU. Use `python run_true_label.py {ds} <seed>` for a detached run.

**What counts as a result.** A later variant's gain over Ramen counts only if it is above the **noise floor** = 2 × (max − min)
of Ramen's domain-averaged accuracy over seeds 0–4 (computed in section 4).

**How to read this notebook.** 1 Setup · 2 The data · 3 Sanity check of the new method · 4 Ramen results (per {group}, over the stream, across seeds) ·
5 Interpretation · 6 Glossary. Accuracy is **domain-averaged** (mean of the per-domain accuracies) unless a table says *pooled*."""


GLOSSARY = """## 6. Glossary

| Term | Meaning |
|---|---|
| NoAdapt | CLIP zero-shot classifier, no test-time adaptation |
| Ramen | Test-time adaptation with a per-class memory of (feature, own entropy gradient, entropy). Each image takes one signSGD step on gradients retrieved from the memory, predicts, and the weights are reset |
| queue / filing | Ramen's memory is one fixed-size queue per class; *filing* = choosing which queue a new image is written to (zero-shot prediction in Ramen) |
| true-label filing (`true_label`) | Oracle: file each image under its true class. Uses ground truth, so it is an upper bound, not a method |
| `zero_shot_filing` | `RamenTrueLabel` with filing by zero-shot prediction, which must equal Ramen (sanity check) |
| domain | A part of the stream with one kind of shift: a DomainNet domain, an ImageNet-C corruption; EuroSAT has one |
| domain-averaged accuracy | Mean of the per-domain accuracies (headline convention) |
| pooled accuracy | Correct images / all images |
| fixed / broken | Image wrong under NoAdapt and right under the run / right under NoAdapt and wrong under the run |
| seed | Sets the stream order; NoAdapt does not depend on it |
| noise floor | 2 × (max − min) of the baseline's accuracy over seeds 0–4; a gain must exceed it to count |
| mixed stream | All domains shuffled together into one test stream |"""


def build(ds):
    d = NOTEBOOKS[ds]
    by, group = d['by'], ('class' if d['by'] == 'class' else 'domain')
    t_acc, t_gain, t_stream = d['titles']
    cells = [
        md(header(d, ds)),
        md('## 1. Setup'),
        code(f"""import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # only used if a run is not cached

import matplotlib.pyplot as plt
import pandas as pd
from true_label_cache import TrueLabelCacheExperiment, seed_table, noise_floor, domain_avg

pd.set_option('display.width', 200)
plt.rcParams.update({{'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False}})

DATASET, SEED, BY = {ds!r}, 0, {by!r}
exp = TrueLabelCacheExperiment(DATASET, seed=SEED).run_all(('NoAdapt', 'Ramen'))"""),
        md(f"""## 2. The data the code ran on

Images and classes per domain, and how many the zero-shot model already gets right. These counts are the denominators
for everything below. {d['data_note']}"""),
        code('exp.data_table()'),
        md(f"""## 3. Sanity check of the new method

`RamenTrueLabel` (src/methods/RamenTrueLabel.py) has a switch `filing`. With `filing: zero_shot` it must be Ramen
exactly, or any later comparison would measure a bug. The smoke test (`python run_true_label.py {ds} 0 --smoke <dir>`,
{d['smoke']}"""),
        md('## 4. Ramen results\n\n### 4a. Seed 0, overall'),
        code("exp.summary_table(baseline='NoAdapt')"),
        md(f"""*fixed* = NoAdapt wrong and Ramen right; *broken* = NoAdapt right and Ramen wrong. Net gain = fixed − broken.
{d['pooled_note']}"""),
        code("""print(f"pooled accuracy: NoAdapt {exp.results['NoAdapt'].correct.mean():.2%}, Ramen {exp.results['Ramen'].correct.mean():.2%}")"""),
        md(f'### 4b. Seed 0, per {group}'),
        code("exp.change_table('Ramen', 'NoAdapt', by=BY)"),
        code(f"""fig, axes = plt.subplots(1, 2, figsize=(13, 0.32 * len(exp.change_table('Ramen', by=BY)) + 2.2))
exp.plot_accuracy(['NoAdapt', 'Ramen'], by=BY, ax=axes[0])
axes[0].set_title({t_acc!r}, loc='left', pad=28)
exp.plot_gain('Ramen', 'NoAdapt', by=BY, ax=axes[1])
axes[1].set_title({t_gain!r}, loc='left')
fig.tight_layout()"""),
        *d.get('extra', []),
        md(f"### 4c. Seed 0, over the stream\n\nThe gain over zero-shot along the stream (rolling window of {d['window']:,} images, pooled over domains)."),
        code(f"""ax = exp.plot_stream_gain('Ramen', 'NoAdapt', window={d['window']})
ax.set_title({t_stream!r}, loc='left');"""),
        md("""### 4d. Seeds 0–4: the noise floor

Ramen's domain-averaged accuracy for each seed (the stream order changes, the data and model do not)."""),
        code("""seeds = seed_table(DATASET, names=('Ramen',))
floor = noise_floor(seeds['Ramen'])
print(f"Ramen over seeds 0-4: {seeds['Ramen'].mean():.2f} ± {seeds['Ramen'].std():.2f}, "
      f"range {seeds['Ramen'].max() - seeds['Ramen'].min():.3f} pts -> noise floor = 2 x range = {floor:.2f} pts")
noadapt = domain_avg(exp.results['NoAdapt'])
seeds.assign(**{'vs NoAdapt (pts)': (seeds['Ramen'] - noadapt).round(2)}).round(2)"""),
        code("""fig, ax = plt.subplots(figsize=(7, 2.2))
ax.plot(seeds['Ramen'], [0] * len(seeds), 'o', ms=9, color='#2a78d6', mec='white', mew=1.5)
for s, v in seeds['Ramen'].items():
    ax.annotate(f'seed {s}', (v, 0), xytext=(0, 9 if s % 2 else -16), textcoords='offset points', ha='center', fontsize=8)
ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xlabel('Ramen accuracy, domain avg (%)')
ax.set_title(f'Seeds move Ramen by {seeds.Ramen.max() - seeds.Ramen.min():.2f} pts: a gain must clear {floor:.2f}', loc='left');"""),
        md('## 5. Interpretation\n\n' + d['interpretation']),
        md(GLOSSARY),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'display_name': 'Python 3 (RAMEN)', 'language': 'python', 'name': 'ramen-env-kernel'}
    nbf.write(nb, d['file'])
    return d['file']


if __name__ == '__main__':
    for ds in sys.argv[1:] or NOTEBOOKS:
        print(build(ds))
