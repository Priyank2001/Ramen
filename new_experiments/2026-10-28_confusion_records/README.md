# Confusion records (2026-10-28)

This folder records, for each test image, what the zero-shot model and Ramen predicted (full softmax vectors) and
the CLIP embedding before and after Ramen's adaptation step. It then lets you inspect, class by class and pair by
pair, which images get predicted as what.

| file | responsibility |
|---|---|
| `../../src/methods/RamenRecorder.py` | `RecorderMixin` wraps `featurize` and records the zero-shot / Ramen probabilities and the before / after embeddings; `RamenRecorder`, `RamenTrueLabelRecorder` |
| `records.py` | `ImageRecords`: the per-image table (outcome, embedding shift, ...) plus the vectors, with filtering (`mask` / `where`), `subset`, `records[i]` → `ImageRecord` |
| `confusion.py` | `ConfusionMatrix` (TP/FP/FN/TN per class, `predicted_as`, `mistaken_for`, `top_pairs`), `ConfusionShift` (zero-shot → Ramen) |
| `gallery.py` | `ConfusionGallery`: image grids for any filter, single-image probability view |
| `plots.py` | outcome bars, pair-change dumbbells, class report, embedding shift / PCA moves |
| `experiment.py` | `RecordExperiment`: dataset / seed / subset / variant, run or load, sanity check, tables |
| `run_records.py` | CLI: `python run_records.py DomainNet 0 [--n-per-domain 2000] [--variant true_label]` |
| `make_notebook.py` | writes `01_confusion_records_domainnet.ipynb` |

Quick runs use `--n-per-domain N` (a shorter stream, so Ramen differs from the full run). Leave it out to run the
whole dataset. Use the RAMEN conda env.

## Notebooks (seed 0)

| notebook | data | zero-shot → Ramen |
|---|---|---|
| `01_confusion_records_domainnet.ipynb` | DomainNet subset, 2,000 per domain (12,000) | 55.28 → 56.29 (subset stream) |
| `02_confusion_records_eurosat.ipynb` | EuroSAT, full (27,000) | 48.09 → 62.63 |
| `03_confusion_records_cifar100c.ipynb` | CIFAR-100-C sev 5, full (150,000) | 37.09 → 46.10 |
| `04_confusion_records_domainnet_full.ipynb` | DomainNet, full (596,006) | 55.37 → 56.80 (cached Ramen) |

All are written by `make_notebook.py`: the cells are shared, and each dataset is one entry of `NOTEBOOKS`
(description, focus classes, plot titles) plus its interpretation in `INTERPRETATIONS`.
