# True-label cache (2026-10-27)

Oracle: Ramen with its memory filed by each test image's **true label** instead of its zero-shot prediction
(`src/methods/RamenTrueLabel.py`, config key `filing: zero_shot | true_label`, `cfg/<Dataset>/RamenTrueLabel.yaml`).
It uses ground truth, so it is an upper bound and not a method.

| File | What it is |
|---|---|
| `true_label_cache.py` | `TrueLabelCacheExperiment(dataset, seed)`: runs / loads the runs in `VARIANTS`, tables (data, accuracy, summary vs Ramen and NoAdapt, sanity check) and plots; `seed_table`, `noise_floor` |
| `run_true_label.py` | detached runner and `--smoke` sanity test |
| `make_baseline_notebook.py` | writes the baseline notebooks (same sections for every dataset; dataset text in `NOTEBOOKS`) |
| `01_ramen_domainnet.ipynb` | Ramen baseline on DomainNet: per domain, over the stream, seeds 0–4, noise floor 0.11 |
| `02_ramen_imagenetc.ipynb` | the same on ImageNet-C 5K (per corruption), noise floor 0.53 |
| `03_ramen_eurosat.ipynb` | the same on EuroSAT (one domain, so per class, plus prediction counts), noise floor 1.51 |

Add a dataset: one line in `DATASETS`. Add a variant: one line in `VARIANTS`.

```bash
cd new_experiments/2026-10-27_true_label_cache
CUDA_VISIBLE_DEVICES=0 python run_true_label.py DomainNet 0 --smoke <scratch dir>   # zero_shot_filing must == Ramen
CUDA_VISIBLE_DEVICES=0 setsid nohup python run_true_label.py DomainNet 0 > logs/DomainNet_s0.log 2>&1 < /dev/null &
```
Use the RAMEN conda env (`~/miniconda3/envs/RAMEN/bin/python`, kernel `ramen-env-kernel`).

Smoke tests (200 images per domain, seed 0): `zero_shot_filing` == Ramen on DomainNet 1,200 / 1,200, ImageNet-C 3,000 / 3,000,
EuroSAT 200 / 200 images; on DomainNet `true_label` == `RamenOracle file_true` on 1,200 / 1,200 images.
