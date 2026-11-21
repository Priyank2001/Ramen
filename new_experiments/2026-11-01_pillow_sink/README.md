# Pillow sink on DomainNet (2026-11-01)

**Question.** After adaptation, Ramen predicts *pillow* for many more DomainNet images than zero-shot CLIP does.
For those images, the study asks four things:
- What was the class before adaptation, and what is it after?
- What did Ramen's cache hold and retrieve when each image arrived?
- What changes when the cache is filed by the true label instead? This is an **oracle**: it uses ground truth.

**How it works.** Ramen's cache never depends on its adapted outputs. Each image is written before retrieval, and
an entry holds its zero-shot feature, zero-shot entropy, own gradient and arrival counter. So both caches are
**replayed offline**: Ramen's (filed by zero-shot prediction) and the oracle's (filed by true label). No GPU is
needed. The adapted predictions and embeddings come from two recorded runs: Ramen and the true-label oracle
(`src/methods/RamenRecorder.py`). Logits are recomputed exactly as `100 · embedding · text_featureᵀ`.

Nothing in `src/`, `cfg/` or earlier experiment folders is changed. The recordings are produced by
`2026-10-28_confusion_records/run_records.py`, which this folder only imports and runs.

| file | job |
|---|---|
| `sink_runs.py` | `SinkRuns`: the two recorded runs, row-aligned, with logits recomputed from the embeddings (`TextFeatures`), the per-domain data and accuracy tables, and `sanity_check` |
| `sink_trace.py` | `SinkTrace`: where the target class's predictions come from (counts, routes by zero-shot class, true classes, domains, oracle fate, seeds 0–4), query groups, random example picks |
| `cache_replay.py` | filing rules (`ZeroShotFiling`, `TrueLabelFiling` = ORACLE); `ReplayedCache` (Ramen's `PriorityCache` with stream ids, fp16 priorities); `CacheReplayRun`: one pass, both caches, per-query retrieval summary, kept entries/queues, queue history, `validate` against a logged `RamenActive` run |
| `example_card.py` | `ExampleCard`: one image's logits/probabilities (table + figure), and both caches' update shares, retrieved entries and target queue (figure + table) |
| `sink_plots.py` | aggregate figures: prediction counts, routes × oracle fate, update sources (dumbbells), target queue over the stream |
| `case_study.py` | `SinkCaseStudy`: builds everything for one dataset / seed / target, picks the examples (`EXAMPLES`), runs and caches the replay, `show_example(i)` |
| `make_notebook.py` | writes `01_pillow_sink_domainnet.ipynb` |
| `01_pillow_sink_domainnet.ipynb` | the presentation notebook (executed) |

## How to run

```bash
# 1. recordings (GPU, ~36 min each; Ramen's was made on 2026-10-28)
cd new_experiments/2026-10-28_confusion_records
CUDA_VISIBLE_DEVICES=0 setsid nohup python run_records.py DomainNet 0 --variant true_label --skip-check \
    > logs/DomainNet_s0_true_label.log 2>&1 < /dev/null &
# 2. notebook (CPU; the first run replays both caches, ~3 min, cached afterwards)
cd ../2026-11-01_pillow_sink
python make_notebook.py
~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace 01_pillow_sink_domainnet.ipynb
```
`--skip-check`: the built-in check needs a plain `true_label` run that is not cached. `SinkRuns.sanity_check`
compares the recording with the cached `oracle_file_true` run instead (the same oracle). Use the RAMEN conda env.

Another sink class (e.g. *teapot*): `SinkCaseStudy('DomainNet', 0, target='teapot')`. The replay is cached per
target.

## Results (seed 0, full DomainNet, 596,006 images)

Checks: both recordings reproduce their cached runs on 100.00% of images, and both start from identical zero-shot
embeddings. The replay matches a logged `RamenActive` run: the queue of the predicted class 100%, top-1 retrieved
entry 98.5% (fp16 distances).

| | predicts *pillow* | wrong | domain-avg accuracy | quickdraw accuracy |
|---|---|---|---|---|
| zero-shot | 2,101 | 1,147 | 55.37 | 13.33 |
| Ramen | 7,605 | 6,484 (6,129 quickdraw) | 56.80 | 17.54 |
| oracle cache (true-label filing) | 4,710 | 3,679 | 55.29 | 11.05 |

- **Routes into the sink:** of Ramen's 6,484 wrong *pillow*s, zero-shot said *envelope* for 3,570 (55.1%),
  *squiggle* for 1,248 (19.2%) and *pillow* for 822 (12.7%). Seeds 0–4: 6,142–6,484 wrong *pillow*s.
- **Contamination is not the cause (H1 rejected):** the oracle cache keeps 2,166 (33.4%) at *pillow* and fixes
  42 (0.6%). 3,499 of the 4,318 it moves (81%) return to their zero-shot class. On quickdraw it keeps the zero-shot
  prediction on 66.8% of images, against 48.0% for Ramen.
- **The caches retrieve different entries (H2 as stated rejected):** only 12–23% of the update weight is shared.
  The oracle's update is 90–99% from the query's own domain and 78–89% from zero-shot errors (rectangle sketches
  filed under their true class). Ramen's is 62–67% same-domain and 42–47% zero-shot errors. The *pillow* queue
  gives only about 1% of any update.
- **Open:** which retrieved entries turn Ramen's step toward *pillow*. This needs the entries' gradients: plan
  step 5, GPU leave-one-group-out.
