# 2026-10-25 — Held-out test of T3 + prior correction (λ = 1) on ImageNet-R / -A / -Sketch

Pre-registered (`PREREGISTRATION.md`, written before any model saw the images): T3 + pc(λ = 1) beats Ramen by more than the noise floor on one mixed stream of ImageNet-R (30,000), ImageNet-A (7,500) and ImageNet-Sketch (50,889) over all 1,000 ImageNet classes. CLIP ViT-B/16, ImageNet-C configs unchanged, seeds 0–4.

**Result: the claim fails; T3 alone is confirmed.**

| Gain over Ramen (pts, domain-avg, seeds 0–4; floor 0.48) | Mean ± std | Seeds above floor |
|---|---|---|
| T3 + pc(λ = 1), the claim | −1.15 ± 0.09 | 0/5 |
| T3 | +0.59 ± 0.04 | 5/5 |
| Ramen + pc(λ = 1) | −2.58 ± 0.10 | 0/5 |

Why: 314 of the 1,000 classes (R ∪ A) carry 60.5% of the images, so the true marginal is far from uniform (TV 0.31 vs 0.17 on ImageNet-C). Ramen already under-predicts those classes, and correcting toward uniform makes it worse.

| File | What it is |
|---|---|
| `PREREGISTRATION.md` | Data, label space, methods and success criterion, frozen before the runs |
| `download.sh` | Downloads + verification (size; SHA-256 for Sketch; image counts), safe extraction to `~/data/imagenet_variants/` |
| `run_heldout.py`, `run_queue.sh` | Runs (NoAdapt seed 0; Ramen and T3 seeds 0–4, probabilities saved) and the `--smoke` label check |
| `heldout_analysis.py` | Pre-registered analysis → `results/heldout/{gains,per_domain,knn_diag}.csv` |
| `heldout.ipynb`, `build_notebook.py`, `interpretation.md` | Write-up |

Dataset code: `src/datasets/imagenet_variants.py` (labels remapped from WordNet ids to the ImageNet-1k index), `ImageNetVariantsExperiment` in `ramen_analysis.py`, configs `cfg/ImageNetVariants/`. Smoke test: zero-shot Sketch 45.0% on 200 images (published ~46–48%); the full run gives 48.43%.
