# FGVC-Aircraft and Flowers102 loaders and configs

Both on the Ramen harness: CLIP ViT-B/16, one shuffled stream of the test split, batches of 100. Nothing under `src/` or `cfg/` is edited. Only the 300-image smoke test has been run, no full run.

| Dataset | Test split | Classes | Data |
|---|---|---|---|
| Aircraft | 3,333 images (`images_variant_test.txt`) | 100 variants (`variants.txt`) | `/data/priyank1/data/fgvc-aircraft-2013b/data/` |
| Flowers102 | 6,149 images (`setid.mat` `tstid`) | 102 | `/data/priyank1/data/flowers-102/` |

| File | Job |
|---|---|
| `fine_grained_loaders.py` | `Aircraft` and `Flowers102` (a shared `ListedImages` dataset with `.samples`/`.targets`, as the harness expects) |
| `fine_grained_experiment.py` | `AircraftExperiment`, `Flowers102Experiment` (subclasses of the harness's `DomainNetExperiment`); configs read from `cfg/<dataset>/` first, then the repo's `cfg/`; `make_fine_grained(name, seed=...)` |
| `flowers102_classnames.json` | the 102 class names in label order 1..102 (see provenance) |
| `cfg/{Aircraft,Flowers102}/{NoAdapt,Ramen,RamenActive}.yaml` | cache settings and prompt |
| `run_fine_grained.py` | `--smoke`, or `<dataset> <seed> <run>` with run in NoAdapt, Ramen, active_prob_T3 |

Run (conda env `RAMEN`): `CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/RAMEN/bin/python run_fine_grained.py Aircraft 0 Ramen`; use `setsid nohup ... > logs/x.log 2>&1 &` for long runs. Cache files are named `<Dataset>_clip_vitbase16_mixed_...` in `new_experiments/results/`.

## Choices fixed before any full run
- **Cache settings are not tuned.** Copied from CIFAR100C, as for Food-101 (`../2026-11-20_food101/`). The test splits hold about 33 (Aircraft) and 60 (Flowers) images per class, so the per-class queues (capacity 750) never fill.
- **Prompts:** `a photo of a {}, a type of aircraft.` and `a photo of a {}, a type of flower.` (CLIP / CoOp / TPT), the same for NoAdapt and Ramen.
- **Aircraft labels are the 100 variants** (not the 70 families or 30 manufacturers). Variant names are used as the class names.
- **Single domain** per dataset, like EuroSAT and Food-101.

## Provenance
`flowers102_classnames.json` comes from the `cat_to_name.json` of github.com/udacity/aipnd-project (102 names, the same lowercase names CoOp uses), re-ordered by label 1..102; sha256 `31cbd8f3...6f8e37`. I did not diff it against CoOp's copy.

## Smoke test (300 images, seed 0, 3 batches: too few to compare methods)
Aircraft: zero-shot 22.0, Ramen 23.7. Flowers102: zero-shot 68.3, Ramen 69.3. Literature zero-shot for CLIP ViT-B/16 with these prompts: about 23.2 (Aircraft) and 67.4 (Flowers102), so the loaders and names look right.
