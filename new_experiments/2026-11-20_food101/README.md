# Food-101 loader and config

Food-101 on the Ramen harness: CLIP ViT-B/16, one shuffled stream of the 25,250 test images (101 classes, ~250 each), batches of 100. Nothing under `src/` or `cfg/` is edited. No result has been run yet except the 500-image smoke test.

Data: `/data/priyank1/data/food-101/` (HDD; official ETH archive, md5 verified).

| File | Job |
|---|---|
| `food101.py` | `Food101` dataset: `ImageFolder` restricted to `meta/test.txt`, class names = folder names with spaces |
| `experiment.py` | `Food101Experiment` (subclass of the harness's `DomainNetExperiment`): uses `Food101`, reads configs from `cfg/Food101/` first then the repo's `cfg/`, data root on the HDD; `make_food101(seed=...)` |
| `cfg/Food101/Ramen.yaml` | Ramen: CIFAR100C cache settings (capacity 750, topk 5, beta 5, signSGD lr 0.01), prompt `a photo of {}, a type of food.` |
| `cfg/Food101/RamenActive.yaml` | same, plus the `active_set` options (T3 = `prob`, temperature 3) |
| `cfg/Food101/NoAdapt.yaml` | zero-shot with the same prompt (the repo default has no `templates`) |
| `run_food101.py` | `--smoke`, or `<seed> <run>` with run in NoAdapt, Ramen, active_prob_T3 |

Run (conda env `RAMEN`):
`CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/RAMEN/bin/python run_food101.py 0 Ramen`; for long runs use `setsid nohup ... > logs/x.log 2>&1 &`.

## Choices fixed before any full run
- **Cache settings are not tuned on Food-101.** Copied from CIFAR100C, the closest setup (100 classes, same backbone). `max_capacity` is per class queue and the stream has only ~250 images per class, so queues never fill (CIFAR100C: 1,500 per class).
- **Prompt:** the single Food-101 template used in CLIP / CoOp / TPT, applied to NoAdapt and Ramen alike.
- **Single domain**, like EuroSAT: the shift is web food photos vs CLIP's pretraining, not a domain mix.
- The cache file names start `Food101_clip_vitbase16_mixed_...`, in the shared `new_experiments/results/` (HDD).

## Smoke test (500 images, seed 0, 5 batches; far too few to compare methods)
Loader: 500 images, 100 distinct labels, range 0-100, 101 class names. Zero-shot 89.4% (matches CLIP ViT-B/16's ~89% reported on Food-101), Ramen 89.0%.
