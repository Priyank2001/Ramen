# Label-value oracle on DomainNet, ImageNet-C and EuroSAT (2026-11-01)

**Question.** Filing Ramen's cache by the true label (`oracle_file_true`) made DomainNet *worse* than Ramen
(55.29 vs 56.80, quickdraw 11.05 vs 17.54). The proposed reason is that a queue's label never reaches the update:
every entry stores the image's own entropy gradient, which sharpens its zero-shot prediction, right or wrong. This
experiment gives the entries a gradient toward their **true** label instead, crossing two factors:

| | value = own entropy gradient (Ramen) | value = CE gradient toward the true label (ORACLE) |
|---|---|---|
| filed by zero-shot prediction (Ramen) | `Ramen` (cached) | `labeloracle_zs_ce` (new) |
| filed by true label (ORACLE) | `oracle_file_true` (cached, 2026-10-23) | `labeloracle_true_ce` (new) |

No image uses its own label: an entry is written with the entropy gradient (as in Ramen, before retrieval), and
its value is replaced by the label gradient only after the image has been predicted. Everything else is Ramen's.

Nothing in `src/` or `cfg/` is changed. `RamenLabelValues` subclasses `RamenTrueLabel` and is built through a
local registry; it starts from `cfg/DomainNet/Ramen.yaml`.

| file | job |
|---|---|
| `label_methods.py` | `RamenLabelValues` (ORACLE): value rules `OwnEntropyGradient` / `TrueLabelGradient`; writes keep each entry's slot so its value can be replaced after prediction. `RamenWriteAfterPrediction`: writes after prediction, which allows filing by the adapted prediction (`filing: adapted`). `RamenGlobalRetrieval`: `retrieval: global_knn` takes the `global_k` nearest entries of the whole cache; `exclude_groups` hides entries of the query's group (ImageNet-C: other corruptions of the same photo) |
| `registry.py` | local method lookup before the repo's, and the repo config each local method starts from |
| `label_oracle.py` | `VARIANTS`, `LabelOracleExperiment` (harness with the local registry; tables from `TrueLabelCacheExperiment`, 2026-10-27), `factor_table` (2 × 2), `sink_table` |
| `run_label_oracle.py` | CLI: `--smoke <dir>` (1,200 images, off settings must match 100%) or full runs |
| `check_label_gradient.py` | one step on each image's own value: the CE value must raise p(true class) |
| `trace_active_set.py` | evidence trace of the active set: replays the first N batches of the real stream with an instrumented `RamenLabelValues` and checks filing, self-retrieval, the captured entries, a float64 recomputation, no label leak, and that label values do not change selection (C0–C6); output in `logs/trace_active_set_DomainNet.log` |
| `check_exclusion.py` | evidence for `exclude_groups` on ImageNet-C (X1–X2): no retrieved entry is another copy of the query's photo; without exclusion, copies carry 0.0% (per-class top-1) vs 4.0% (global top-10) of the update; output in `logs/check_exclusion_ImageNetC5K.log` |
| `check_write_after.py` | evidence for `RamenWriteAfterPrediction` (W0–W5): first batch zero-shot, retrieval sees earlier batches only, each image filed under its returned class, adapted vs zero-shot filing differ; output in `logs/check_write_after_DomainNet.log` |
| `misfiled_pull.py` | `MisfiledPull`: replays Ramen's cache for 500 images Ramen breaks vs 500 it keeps right; update weight from own-class entries misfiled under the wrong class |
| `neighbour_labels.py` | `NeighbourLabels`: replays Ramen's cache for 2,000 random DomainNet images; the retrieval-weighted vote of the retrieved entries' true labels (per queue and global top-10) vs what each run predicted |
| `near_copies.py` | `NearCopies`: same 2,000 images; own entry retrieved vs other entries within distance 0.1 (near-copies), and the global gain with and without them |
| `mechanism.py` | `LabelAgreement`: replays both caches (2026-11-01_pillow_sink, read-only import) for 1,000 random images; how each update's weight splits over the entries' true classes |
| `plots.py` | change vs Ramen (with noise floor), change per domain, predictions of one class, vote share by filing |
| `make_notebook.py` | writes `01_label_oracle_domainnet.ipynb` (with the cache replay and the pillow sink) |
| `make_dataset_notebooks.py` | writes `02_label_oracle_imagenetc.ipynb` and `03_label_oracle_eurosat.ipynb` (`SPECS`); `adapted_filing_cells` (4.4) and `global_retrieval_cells` (4.5) are shared by all three, `exclusion_cells` (4.6) is ImageNet-C's |
| `01_label_oracle_domainnet.ipynb`, `02_label_oracle_imagenetc.ipynb`, `03_label_oracle_eurosat.ipynb` | the notebooks (executed) |
| `methods/` | the methods as equations, one markdown file each (label values, write after prediction, global retrieval and copy exclusion, replay analyses), with code maps and results |
| `smoke/` | smoke-test results the notebooks' sanity checks load (200 images per domain) |
| `logs/` | run logs, smoke log, gradient check output |

## How to run

```bash
cd new_experiments/2026-11-01_label_oracle
CUDA_VISIBLE_DEVICES=2 python run_label_oracle.py DomainNet 0 --smoke smoke > logs/smoke_DomainNet_s0.log 2>&1
CUDA_VISIBLE_DEVICES=2 python check_label_gradient.py > logs/check_label_gradient.log
for pair in "0 labeloracle_true_ce" "1 labeloracle_zs_ce"; do set -- $pair
  CUDA_VISIBLE_DEVICES=$1 setsid nohup python run_label_oracle.py DomainNet 0 --runs $2 \
      > logs/DomainNet_s0_$2.log 2>&1 < /dev/null &
done   # ~40 min each
python make_notebook.py
CUDA_VISIBLE_DEVICES= jupyter nbconvert --to notebook --execute --inplace 01_label_oracle_domainnet.ipynb
```
Use the RAMEN conda env (`~/miniconda3/envs/RAMEN/bin/python`).

## Checks

- Smoke (1,200 images): `value: entropy` reproduces Ramen (zero-shot filing) and `RamenTrueLabel` (true-label
  filing) on 1,200 of 1,200 images each.
- One step on an image's own CE value raises its true-class probability on 97 of 100 images (correct 62 → 97);
  its own entropy value raises the zero-shot class's on 90 of 100 (correct 62 → 64).

## Results (seed 0; change vs Ramen in domain-averaged points)

| run | write | filing | value | DomainNet (floor 0.11) | ImageNet-C (floor 0.53) | EuroSAT (floor 1.51) |
|---|---|---|---|---|---|---|
| Ramen | before retrieval | zero-shot | entropy | 56.80 | 33.08 | 62.63 |
| filing oracle | before retrieval | true label | entropy | −1.50 | −0.71 | −6.50 |
| `labeloracle_zs_ce` | before retrieval | zero-shot | true-label CE | **+1.70** | **+1.51** | **+26.70** |
| `labeloracle_true_ce` | before retrieval | true label | true-label CE | −0.25 | −0.91 | +8.09 |
| `late_zs_entropy` (timing control) | after prediction | zero-shot | entropy | −0.05 | −0.17 | +0.53 |
| `late_zs_ce` | after prediction | zero-shot | true-label CE | +1.86 | +1.64 | +26.49 |
| `late_adapted_entropy` (**no labels**) | after prediction | adapted | entropy | −0.27 | +0.12 | −6.04 |
| `late_adapted_ce` | after prediction | adapted | true-label CE | +1.26 | +1.37 | +25.90 |

Global top-k retrieval (zero-shot filing, Ramen's timing; ImageNet-C in two columns: other corrupted copies of the
photo visible, vs Ramen / hidden, vs Ramen with copies hidden):

| run | retrieval | value | DomainNet | ImageNet-C visible | ImageNet-C hidden | EuroSAT |
|---|---|---|---|---|---|---|
| `labeloracle_zs_ce` | 10 per queue (1 on ImageNet-C, 50 on EuroSAT) | true-label CE | +1.70 | +1.51 | **+0.92** | +26.70 |
| `global_ce_k10` | 10 nearest overall | true-label CE | **+2.86** | +3.97 | −1.00 | +23.11 |
| `global_ce_k100` | 100 nearest overall | true-label CE | +2.54 | +3.82 | +0.18 | **+27.59** |
| `global_ce_k1000` | 1,000 nearest overall | true-label CE | +1.84 | +2.80 | +0.95 | +25.91 |
| `global_entropy_k100` (**no labels**) | 100 nearest overall | entropy | +0.04 | −0.86 | −1.57 | −7.51 |

(Ramen with copies hidden, `excl_zs_entropy`, is −0.05 vs Ramen.)

- Label values help on all three datasets, by far the most on EuroSAT (*Herbaceous Vegetation* 0.30% → 88.40%).
  On DomainNet the *pillow* sink disappears (6,484 wrong → 377 / 750).
- True-label filing hurts under either value on all three. Filing by the adapted prediction (the paper and its
  code file by the **zero-shot** prediction, Sec. 3.2) is no better: −0.21 / +0.29 / −6.57 vs its timing control.
  Moving the write after prediction is within noise on all three.
- DomainNet replay (1,000 images): an update spreads over ~235 classes; the image's own class gets ~2.5% of the
  weight, and leads the vote for 46% of images (zero-shot filing) vs 39% (true-label filing).
- Images Ramen breaks (zero-shot right, Ramen wrong; 10,298 on DomainNet) are not mainly pulled by own-class
  entries misfiled under the wrong class (0.14% of the update vs 0.09% for kept images). They have less support
  from correctly filed own-class entries (0.98% vs 3.58%). Hiding confident-wrong entries recovers 1,022 of them
  (random control 783); label values recover 6,025.
- On DomainNet only 168,502 images (28%) are ever stored (fp16 counter freezes full queues), so after that point an
  image has no entry of its own in the cache.
- **Global retrieval.** On DomainNet the queue-averaged vote is part of the limit: global top-10 adds +1.16 over
  per-queue label values, ten times the floor, mostly on quickdraw (+5.48; the other domains −0.17 to +0.45). It is not a near-copy leak: only 19 of 2,000 sampled images (0.9%)
  have another entry within distance 0.1, and without them the sample's gain moves from +2.70 to +2.62. On
  ImageNet-C the apparent global gain (+3.97) was a leak from the other 14 corrupted copies of each photo. With
  those hidden, global retrieval adds nothing (k=1000 +0.95 vs per-queue +0.92; k=10 −1.00). On EuroSAT it is
  within the floor (+0.89). Without labels it never helps (+0.04 / −0.86 / −7.51).
- The next limit on DomainNet is the neighbours themselves. Only 4.5 of an image's 10 nearest entries share its
  class (quickdraw 2.1), and their label vote is right for 57.6% of 2,000 sampled images (zero-shot 51.6%).
- Next: a label-free value that points toward the right class more often than the entropy gradient (the label
  gradient and entropy gradient agree, cosine +0.97, when zero-shot is right, and mostly oppose, −0.31, when it is
  wrong); GPU leave-one-group-out on the broken images.
