# NIQE and BRISQUE as a regulariser on Ramen's active-set selection

*2026-10-16. Source: R. Soundararajan, "Blind Image Quality Assessment" (IISc lecture slides,
[blindqa.pdf](../blindqa.pdf)): BRISQUE (Mittal, Moorthy, Bovik, TIP 2012) and NIQE (Mittal, Soundararajan, Bovik,
SPL 2013). This adds to [vqa_paper_directions.md](vqa_paper_directions.md), which found that NSS quality features do
not predict *which queries* Ramen helps (AUROC about 0.5). Here the scores pick *which memory entries* adapt a query.*

## The short version

- **Built:**
  - a GPU port of the original MATLAB NIQE and BRISQUE, validated against pyiqa;
  - `QualityCache`: Ramen's per-class memory queue that also stores each image's NIQE, BRISQUE and true label;
  - `RamenQuality`: picks each class queue's active-set entries by *CLIP distance + λ · quality term*.
  λ = 0 reproduces Ramen exactly: 100% identical predictions on the 75k-image ImageNet-C stream.
- **Two regularisers, each with NIQE alone, BRISQUE alone and both together:**
  - *match*: prefer entries whose image quality is close to the query's;
  - *prior*: prefer entries whose images look natural.
- **Result on ImageNet-C, seed 0** (19 runs):
  - *Match* with a light touch (λ = 0.1–0.3) gains +0.1 to +0.2 points. BRISQUE and both-together do best: +0.19 / +0.21.
  - Stronger matching loses steadily, down to −0.9 at λ = 3.
  - *Prior* loses at every λ (−0.35 to −2.1).
- **Seeds 0–4 at fixed settings: a real but tiny gain.** Both-together at λ = 0.1 gives **+0.11 ± 0.07** and is
  positive on all 5 seeds. BRISQUE alone gives +0.10 and NIQE alone +0.07. All of these are below the project's
  ImageNet-C noise floor of 0.53, so this is not a result to build on as it stands.
- **It stacks with RamenActive prob T = 3** (seed 0 only): +0.66 over Ramen, against +0.51 for prob T = 3 alone.
- **Why stronger matching hurts:** one quality number cannot tell distortion *types* apart. Contrast, JPEG and
  pixelate images get similar scores, and so do elastic, frost and brightness. Strong matching therefore fills a
  contrast query's active set with JPEG and pixelate images. The share of active-set entries from the query's own
  corruption drops from 44% to 27%. A light touch only breaks ties among entries that CLIP already found, and raises
  that share to 47%.
- **Together beats separately for identifying the corruption:**
  - NIQE and BRISQUE carry different information. Among an image's 10 nearest neighbours in quality space, 25% share
    its corruption with NIQE alone, 20% with BRISQUE alone and 38% with both (chance 6.7%).
  - In accuracy the difference is smaller. At seed 0 both-together and BRISQUE alone tie at λ = 0.1, and NIQE alone
    is weakest.

## What was built

| piece | file | what it does |
|---|---|---|
| IQA port | [src/models/iqa.py](../src/models/iqa.py) | batched float64 GPU port of `brisque_feature.m` + LIVE SVR, and `computequality.m` + `modelparameters.mat`; 40 ms per 100 images |
| model files | [src/models/iqa_params/](../src/models/iqa_params/) | the LIVE releases (NIQE pristine MVG; BRISQUE SVR: 774 support vectors, γ = 0.05, ρ = −153.591) |
| memory | `QualityCache` in [src/methods/RamenQuality.py](../src/methods/RamenQuality.py) | Ramen's class queue (same insert, replace and query rules) plus per-entry NIQE, BRISQUE and true label |
| method | `RamenQuality`, same file | quality-regularised selection; composes with RamenActive's `active_set` class weighting |
| configs | `cfg/{ImageNetC5K,CIFAR10C,CIFAR100C,DomainNet}/RamenQuality.yaml` | each dataset's Ramen settings + `quality_metric`, `quality_reg`, `quality_lambda` |
| experiments | [quality_active_set.py](quality_active_set.py) | `run` / `analyse` (bootstrap CIs, purity diagnostics, quality profile, multi-seed summary) |
| port check | [quality_checks.py](quality_checks.py) | `validate` against pyiqa |

True labels are passed in by the harness as `forward(x, labels=...)`, only to methods that set
`records_labels = True`. This is wired into `ramen_analysis.run` and `src/main.py`, but not `src/experiments.py`,
where labels are stored as −1. The labels are recorded for diagnostics only: the label purity of the active set.
Nothing that selects, weights or predicts reads them.

## The regulariser

Ramen takes, from each class queue c, the top-k entries nearest the query in CLIP space. On ImageNet-C, k = 1 and
β = 0. RamenQuality changes the score that ranks the entries:

    s_bj = d_bj + λ · R_bj        (Ramen's weight exp(−H_j)·exp(−β d_bj) becomes exp(−H_j)·exp(−β s_bj))

- **match:** R_bj = sqrt(mean_m (r_m(b) − r_m(j))²)
- **prior:** R_bj = mean_m r_m(j)

Here m runs over the chosen metrics: NIQE, BRISQUE or both. r_m is the rank of an image's score among all scores
seen so far in the stream, scaled to [0, 1]. The ranks are causal and unlabelled, and they include the current
batch, which Ramen writes into memory before retrieving. Ranks put NIQE and BRISQUE on one scale, so λ is in units
of CLIP feature distance. For scale: a random pair of images differs by about 0.33 in rank.

The scores are computed on CLIP's 224 × 224 preprocessed view, which is what the model sees. At that size NIQE
uses 2 × 2 blocks of 96 × 96 pixels. Ramen's fp16 priority counter is kept, as in Ramen, so that λ = 0 is
bit-identical. It saturates after 65,504 samples (see [findings_and_directions.md](findings_and_directions.md)).

## Checks of the IQA port

- **Against pyiqa 0.1.16** (300 ImageNet-C images, same `rgb2gray` input). Images without flat regions: largest
  difference 0.003 NIQE and 0.05 BRISQUE (Pearson 0.9999). The per-corruption means agree.
- **Flat regions are ill-defined in every implementation, MATLAB included.** In saturated or blocky areas (snow,
  JPEG, contrast) the true MSCN coefficient is exactly 0. Filter rounding leaves ±1e-14 of noise instead, and the
  AGGD fit counts samples by sign, so that noise moves BRISQUE by up to 12–16 points. Even a 0.1% share of such
  coefficients moves it by 3. The port therefore sets |I − μ| < 1e-8 to 0 (`iqa.FLAT_TOL`), which is a deliberate,
  documented deviation. Real structure in 8-bit images is at least about 1e-5.
- **Model files.**
  - OpenCV's `brisque_model_live.yml` is a retrained model (ρ = −149.6, different feature ranges), not the LIVE
    release, so it is not used.
  - scikit-video's `niqe_image_params.mat` is also its own retrain, and its diagonal features contain a
    `bl3, bl3` typo, so it is not used either.
- **Speed.** A batched `torch.linalg.pinv` on the 36 × 36 covariance took 840 ms per batch. It is replaced by
  `solve()`, which gives the same result because the pristine covariance is positive definite. The total is now
  40 ms per 100 images. A RamenQuality run takes about 15 min on ImageNet-C, against about 7 for Ramen; the extra
  time is per-class rank lookups.

## Results: ImageNet-C (ImageNetC5K, ViT-B/16, mixed stream, seed 0, Ramen 33.08)

Gain over Ramen in points. Brackets: 95% bootstrap CI over base images. "Same-corruption" is the weight share of
the top-8 active-set entries that come from the query's corruption; Ramen's is 43.6%.

| regulariser | λ | NIQE | BRISQUE | both | same-corruption (both) |
|---|---|---|---|---|---|
| match | 0.1 | +0.11 [−0.01, +0.23] | **+0.19** [+0.06, +0.32] | **+0.19** [+0.08, +0.32] | 47.0% |
| match | 0.3 | −0.04 | +0.03 | **+0.21** [+0.07, +0.35] | 46.6% |
| match | 1 | −0.37 | −0.27 | −0.23 | 39.8% |
| match | 3 | −0.88 | −0.88 | −0.85 | 28.6% |
| prior | 0.3 | −0.47 | −0.53 | −0.35 | 14.7% |
| prior | 1 | −2.14 | −1.95 | −2.03 | 8.3% |

- **Prior** pulls in entries from easy corruptions: brightness and frost score as most natural. This amplifies the
  entropy scale bias that Ramen already has (vqa_paper_directions.md), and same-corruption support collapses to 8%.
- **Label purity of the active set** falls with λ in every variant: 0.48% for Ramen, 0.43% at λ = 0.1, 0.22% at
  λ = 3. Under topk = 1 the active set holds one entry from each of the 1000 classes, so purity is small by
  construction.
- **Where the light-touch gains come from.** Per corruption for both, λ = 0.3: frost +1.5, glass blur +1.1,
  brightness +0.7, elastic +0.6. The losses are on Gaussian and shot noise (−0.4 each) and contrast (−0.5).

## Results: seeds 0–4 at fixed settings

These are λ = 0.1 for each metric, plus λ = 0.3 for both together, all chosen from the seed-0 grid. Ramen's
cached runs at seeds 1–4 are reused, with the same stream order (asserted). Gain over Ramen in points:

| match variant | s0 | s1 | s2 | s3 | s4 | mean ± std | positive | paired t |
|---|---|---|---|---|---|---|---|---|
| both, λ = 0.1 | +0.19 | +0.07 | +0.14 | +0.14 | +0.03 | **+0.11 ± 0.07** | 5/5 | 3.8 |
| BRISQUE, λ = 0.1 | +0.19 | +0.06 | +0.09 | +0.10 | +0.03 | +0.10 ± 0.06 | 5/5 | 3.6 |
| both, λ = 0.3 | +0.21 | +0.08 | +0.14 | +0.03 | −0.00 | +0.09 ± 0.08 | 4/5 | 2.4 |
| NIQE, λ = 0.1 | +0.11 | +0.05 | +0.05 | +0.11 | +0.03 | +0.07 ± 0.04 | 5/5 | 4.3 |

- **Below the noise floor.** Ramen's range over seeds 0–4 is 0.26, so the floor (2 × range, from
  seed_variance.ipynb) is 0.53. No variant comes close. The gain is consistent in sign but about a fifth of the floor.
- **Seed 0 is optimistic.** It was the selection seed, and every variant gains the most there. The other four seeds
  average +0.06 to +0.10.
- **Alone versus together.** The order is the same as on seed 0: both ≥ BRISQUE > NIQE. The gaps between them
  (0.01–0.04) are smaller than the seed-to-seed spread, so "together is better" holds for identifying the corruption
  (38% vs 20–25% nearest-neighbour purity) but is not established for accuracy.
- **The same ordering shows in the active set.** At λ = 0.1, on each of seeds 1–4, the share of top-8 entries from the
  query's own corruption is highest for both-together, then BRISQUE, then NIQE. For example, seed 1: 45.2% / 44.7% /
  44.0%; seed 4: 50.0% / 49.6% / 48.6%. Ramen's own share is logged only for seed 0 (43.6%, via the λ = 0 run).
- **Stacking with RamenActive prob T = 3** (`quality_both_match_l0.3_probT3`, seed 0 only): 33.74, which is +0.66
  over Ramen [+0.44, +0.87]. prob T = 3 alone on this seed is +0.51, so the regulariser adds about +0.15, in line
  with its standalone effect. Same-corruption support reaches 61.7%. Untested on seeds 1–4.

## Why strong matching hurts: a quality score is not a distortion type

Median score per corruption on the stream (higher = worse):

| | NIQE | BRISQUE |
|---|---|---|
| Gaussian / shot noise | 33.0 / 29.0 | 77.9 / 76.8 |
| defocus blur | 12.7 | 75.1 |
| contrast / pixelate / JPEG | 11.2 / 11.3 / 9.7 | 57.8 / 46.3 / 61.8 |
| elastic / frost / brightness | 3.9 / 3.7 / 4.3 | 16.4 / 10.4 / 16.6 |

- **BRISQUE scores defocus blur like noise.** NIQE scores contrast like pixelate and JPEG.
- **Elastic, frost and brightness look alike on both.**
- **What happens to the active set at both / match / λ = 3:**
  - contrast queries draw 7% of their support from contrast (Ramen: 47%);
  - elastic queries draw 4% from elastic (Ramen: 22%), with brightness now supplying 50%;
  - zoom-blur queries draw 10% from zoom blur (Ramen: 27%).
- **Only Gaussian noise gains same-corruption support** (26% → 45%), and its accuracy still falls.
- **Contrast is where Ramen gains the most** over zero-shot (+10.7), and it is among the largest losers under strong
  matching.

CLIP features already sort entries by corruption much better than a scalar quality score does. The regulariser
helps only when it is weak enough to reorder near-ties within what CLIP retrieved.

## Caveats

- **Seed 0 used for selection.** One dataset, and λ was chosen on seed 0. Seeds 1–4 repeat those settings, so they
  check stream-order noise but are not a held-out dataset.
- **CLIP's view, not the original image.** The scores are measured on the 224 px preprocessed image. On CIFAR this
  view is upsampled from 32 px, so its NIQE would mostly measure the upsampling.
- **No held-out dataset yet.** The DomainNet, CIFAR-10-C and CIFAR-100-C configs exist but have not been run.

## Next steps

1. **Match on the 36-dimensional NSS feature vector instead of the score.** The diagnostic says the failure is that
   one number merges distortion types. The BRISQUE feature vector, which `iqa.brisque_features` already returns,
   keeps them apart, since GGD shape and paired-product asymmetry differ between blur, noise and blocking. This is
   the most promising follow-up.
2. **Do not pursue the scalar-score regulariser further on its own.** Its gain is consistent but about a fifth of
   the noise floor. If it is kept, it is as a cheap add-on to RamenActive prob T = 3: check the stacked variant on
   seeds 1–4 first (4 runs, about 1 h on 3 GPUs). Only then go held-out on DomainNet with λ fixed from ImageNet-C.
3. **Drop the prior regulariser.**

## Files

- **Code:** src/models/iqa.py, src/models/iqa_params/, src/methods/RamenQuality.py,
  cfg/*/RamenQuality.yaml, new_experiments/quality_active_set.py, new_experiments/quality_checks.py.
- **Results:** `new_experiments/results/ImageNetC5K_*_quality_*.pkl` (per-sample), `*_stats.pkl` (NIQE and BRISQUE
  per image, active-set diagnostics), `ImageNetC5K_quality_active_set_s{0..4}.csv` and `_seeds.csv` (tables).
- **Logs:** `results/logs/`. The full analysis printout is in `results/logs/quality_ImageNetC5K_analysis.txt`
  (`python quality_active_set.py analyse ImageNetC5K 0 1 2 3 4`).
