"""Writes and executes sequential.ipynb: python build_notebook.py [interpretation.md]. Needs sequential_analysis.py's CSVs."""
import os
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

HEADER = r"""# Domain-sequential streams: does T3's gain survive when domains arrive one after another?

**Question.** Every result so far comes from mixed streams (all domains shuffled together). In deployment, shifts arrive domain by domain, and Ramen's per-class queues carry one domain's entries into the next. Does the zero-shot-weighted active set (T3) still beat Ramen there, and what does the stream order do to Ramen itself?

**Hypothesis (pre-registered, `PREREGISTRATION.md`, written before any sequential run).** T3 − Ramen, domain-averaged, mean over seeds 0–4, is above the noise floor of the sequential stream on **all three** of ImageNet-C 5K, DomainNet and ImageNet-R/A/Sketch. Mixed-stream gains for reference: +0.62 / +0.24 / +0.59.

**Why this test.** It is the main untested deployment condition. Same images, configs and seeds as the mixed runs; only the order changes.

**Runs.**

| Name | Stream | What it is | Knows domain boundaries? |
|---|---|---|---|
| NoAdapt | — | CLIP zero-shot (stateless; the seed-0 mixed run, realigned) | — |
| Ramen | sequential | `cfg/<ds>/Ramen.yaml`, never reset | no |
| T3 | sequential | RamenActive `active_set: prob`, `temperature: 3`, never reset | no |
| Ramen-reset | single | Ramen reset at every domain start; same within-domain order | **yes (reference)** |
| Ramen (mixed), T3 (mixed) | mixed | the earlier runs, for comparison | no |

**Data and order.** ImageNet-C 5K: 15 corruptions × 5,000, standard order gaussian_noise → jpeg_compression. DomainNet: clipart → infograph → painting → quickdraw → real → sketch (596,006). ImageNet-R/A/Sketch: rendition → adversarial → sketch (88,389; 1,000-class head). One order per dataset (not varied).

**Seeds.** 0–4: the within-domain order and Ramen's selection randomness; the images are identical.

**How the runs were produced.** `bash run_queue.sh` (detached, one worker per GPU) runs `run_sequential.py <dataset> <seed> <run>`; the harness's new `tta_mode='sequential'` visits domains in order without resetting. Then `python sequential_analysis.py` → `results/sequential/*.csv`.

**What counts as a result.** Noise floor = 2 × (max − min) of sequential Ramen over seeds 0–4, per dataset (computed below). Secondary, with no pass/fail: the order effect on Ramen (sequential vs mixed, both minus NoAdapt), sequential Ramen vs Ramen-reset, and the adaptation lag after each domain switch (first 10 batches of a domain vs the rest).

**Amendment (before any sequential result).** Ramen's fp16 priorities freeze full queues after 65,520 samples, which in a sequential stream locks the memory on early domains. Ramen-fp32 and T3-fp32 (`priority_fp32: true`, eviction only) were added: DomainNet seeds 0–4, the other two seed 0 as a screen. On DomainNet, T3-fp32 − Ramen-fp32 is read with the same floor and rule as H1.

**Sections.** 1 Setup · 2 Data · 3 Sanity checks · 4 Primary result · 5 What the order does to Ramen · 6 Adaptation lag, memory composition, fp32 · 7 Per domain · 8 Interpretation · 9 Glossary"""

GLOSSARY = r"""## 9. Glossary

| Term | Meaning |
|---|---|
| Ramen | CLIP zero-shot + per-class memory of (feature, entropy gradient); one signSGD step per batch from retrieved entries, predict, reset the weights (the memory persists) |
| T3 | Ramen with a zero-shot-weighted active set: each class queue's vote weighted by the query's zero-shot probability at temperature 3 |
| mixed stream | all domains shuffled into one stream |
| sequential stream | domains one after another, each shuffled within itself; the method is never told where a domain starts |
| Ramen-reset | Ramen whose memory is emptied at every domain start (needs the boundaries; a reference, not a fair competitor) |
| adaptation lag | how long after a domain switch the gain over zero-shot takes to recover |
| fp16 / fp32 priorities | Ramen evicts the lowest arrival counter from a full queue; in fp16 the counter becomes inf after 65,520 samples and full queues stop changing (Ramen as published); fp32 keeps them updating |
| noise floor | 2 × (max − min) of sequential Ramen's domain-averaged accuracy over seeds 0–4 |
| domain-averaged accuracy | mean of per-domain accuracies |
| fixed / broken | images the reference gets wrong and the variant right / and the reverse |
| cluster bootstrap CI | 95% interval from resampling images (Poisson weights; base image on ImageNet-C), per seed |"""

CELLS = [
    md(HEADER),
    md('## 1. Setup'),
    code("""import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '..'); sys.path.insert(0, '../2026-10-25_stacking')
import numpy as np, pandas as pd
import stack, sequential_analysis as sa
pd.set_option('display.width', 220)
acc, g, lag, pdm = sa.load_csv('acc'), sa.load_csv(), sa.load_csv('lag'), sa.load_csv('per_domain')
FL = sa.floor(acc)
acc.groupby(['dataset', 'run']).seed.nunique().unstack()"""),
    md('## 2. The data the code ran on'),
    code("stack.data_table(sa.DATASETS)"),
    code("""pd.DataFrame({stack.LABEL[ds]: {'domain order': ' → '.join(pd.read_pickle(sa.path(ds, 0, 'sequential', 'Ramen')).domain.cat.categories),
                                 'Ramen (seq.) per seed': ', '.join(f'{a:.2f}' for a in acc[(acc.dataset == ds) & (acc.run == 'Ramen')].acc),
                                 'noise floor (2 x range)': round(FL[ds], 3)} for ds in sa.DATASETS}).T"""),
    md("""## 3. Sanity checks

- **Sequential loader.** A NoAdapt run in sequential mode (ImageNet-C, seed 0) must give the same images, labels and predictions as the mixed NoAdapt run, with domains visited in order.
- **Reset reference.** Ramen and Ramen-reset see the same order; they must agree exactly on the first domain and may differ only afterwards.
- **Code changes for this experiment** (both default to the old behaviour): `torch.cdist` forced onto its matmul path (the default crashed on DomainNet's 1-image last batch of infograph; bitwise identical wherever the default worked), and a `priority_fp32` switch. Re-running the mixed ImageNet-C Ramen seed 0 after each change must reproduce the cached run on 100% of images.
- **Memory replay.** The offline replay of Ramen's queue rule (used in section 6b) must reproduce RamenActive's logged queue contents."""),
    code("sa.sanity()"),
    code("sa.reset_sanity()"),
    code("sa.code_change_checks()"),
    code("sa.validate_replay()"),
    md('## 4. Primary result (pre-registered)'),
    code("s = sa.summary(g, FL); sa.verdict(s)"),
    code("fig = sa.plot_primary(s)"),
    code("""show = s.copy(); show['dataset'] = show.dataset.map(stack.LABEL)
show.set_index(['dataset', 'comparison']).astype({'gain mean': float, 'gain std': float, 'gain min': float, 'images': float}).round(2)"""),
    md('## 5. What the order does to Ramen\n\nDomain-averaged accuracy (mean over seeds). Same images in every column.'),
    code("""a = acc.groupby(['dataset', 'run']).acc.mean().unstack()[['NoAdapt', 'Ramen (mixed)', 'Ramen', 'Ramen-reset', 'T3 (mixed)', 'T3']]
a.columns = ['NoAdapt', 'Ramen mixed', 'Ramen sequential', 'Ramen-reset (seq.)', 'T3 mixed', 'T3 sequential']
a['Ramen gain: mixed'] = a['Ramen mixed'] - a.NoAdapt
a['Ramen gain: sequential'] = a['Ramen sequential'] - a.NoAdapt
a['Ramen gain: reset'] = a['Ramen-reset (seq.)'] - a.NoAdapt
a.index = a.index.map(stack.LABEL); a.round(2)"""),
    md('## 6. Adaptation lag after a domain switch'),
    code("fig = sa.plot_lag(lag)"),
    code("t = sa.lag_table(lag); t.index = t.index.set_levels(t.index.levels[0].map(stack.LABEL), level=0); t.round(2)  # pts vs NoAdapt"),
    md("""### 6b. What the memory holds: fp16 vs fp32 priorities (amendment; descriptive)

Ramen's arrival-counter priorities are fp16. Every insert after sample 65,520 gets priority inf, and a full queue stops updating once all its slots are inf. Below, the whole memory's domain composition at the end of each domain, replayed offline (exact, validated in section 3) from the filing classes logged by the sequential T3 run, seed 0. Also shown: the share of each domain's images that entered the memory."""),
    code("""mc = pd.concat([sa.memory_composition(ds, 0) for ds in sa.DATASETS])
mc.assign(dataset=mc.dataset.map(stack.LABEL)).set_index(['dataset', 'priorities', 'after domain']).drop(columns='seed').round(1).dropna(axis=1, how='all')"""),
    md("""### 6c. fp32 priorities (amendment)

T3-fp32 − Ramen-fp32 uses the same floor and rule as H1 (DomainNet seeds 0–4; ImageNet-C and R/A/Sketch seed 0 only, as a screen). Ramen-fp32 − Ramen shows what the frozen memory costs or gains."""),
    code("""f = s[s.comparison.isin(['T3-fp32 - Ramen-fp32', 'Ramen-fp32 - Ramen', 'T3-fp32 - T3', 'Ramen-fp32 - NoAdapt', 'Ramen - NoAdapt', 'T3 - Ramen'])]
f.assign(dataset=f.dataset.map(stack.LABEL)).set_index(['dataset', 'comparison'])[['seeds', 'gain mean', 'gain std', 'gain min', 'floor (seq.)', 'above floor', 'CI > 0', 'images']].astype(
    {'gain mean': float, 'gain std': float, 'gain min': float, 'images': float}).round(2)"""),
    md('## 7. Per domain\n\nAccuracy minus NoAdapt per domain (pts, mean over seeds), domains in stream order. The first domain starts from an empty memory in every stream run.'),
    code("""for ds in sa.DATASETS:
    p = pdm[pdm.dataset == ds]
    order = list(pd.read_pickle(sa.path(ds, 0, 'sequential', 'Ramen')).domain.cat.categories)
    cols = [c for c in ['Ramen (mixed)', 'Ramen', 'Ramen-reset', 'T3', 'Ramen-fp32', 'T3-fp32'] if c in p and p[c].notna().all()]
    m = p.groupby('domain')[['NoAdapt'] + cols].mean().loc[order]
    out = m[cols].sub(m.NoAdapt, axis=0)
    out.columns = [{'Ramen (mixed)': 'Ramen mixed', 'Ramen': 'Ramen seq.', 'T3': 'T3 seq.'}.get(c, c) for c in cols]
    out['T3 - Ramen (seq.)'] = out['T3 seq.'] - out['Ramen seq.']
    print(stack.LABEL[ds]); display(out.round(2))"""),
    md('## 8. Interpretation\n\nINTERPRETATION'),
    md(GLOSSARY),
]


def build(interpretation):
    nb = nbformat.v4.new_notebook()
    nb.cells = CELLS
    nb.cells[-2] = md('## 8. Interpretation\n\n' + interpretation)
    nb.metadata['kernelspec'] = {'name': 'ramen-env-kernel', 'display_name': 'Python 3 (RAMEN)', 'language': 'python'}
    ExecutePreprocessor(timeout=1800, kernel_name='ramen-env-kernel').preprocess(nb, {'metadata': {'path': HERE}})
    out = os.path.join(HERE, 'sequential.ipynb')
    nbformat.write(nb, out)
    return out


if __name__ == '__main__':
    print(build(open(sys.argv[1]).read() if len(sys.argv) > 1 else '(written after the results)'))
