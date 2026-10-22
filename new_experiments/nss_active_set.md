# NSS feature vectors as a regulariser on Ramen's active-set selection

*2026-10-18. Follow-up #1 of [quality_active_set.md](quality_active_set.md). That write-up found that the scalar
NIQE and BRISQUE scores give a consistent but tiny gain (+0.11 ± 0.07 over seeds 0–4, against a noise floor of
0.53), and that strong matching on them hurts because one number merges distortion types. This file replaces the
scores with the natural scene statistics (NSS) feature vectors behind them. It does not repeat the earlier
material.*

## The short version

- **The scores' problem is fixed.** NSS vectors identify the corruption about twice as well as the scores (80% vs
  38% 10-NN purity), and better than CLIP features (71%).
- **Settings chosen offline.** An exact offline replay of Ramen's memory, validated against the real runs to about
  1 point, picked whitened vectors at λ = 0.1–0.3 before any GPU run.
- **Seeds 0–4 on ImageNet-C: every variant positive on every seed, and 2–3× the scalar scores' gain.**
  - NIQE vector, λ = 0.1: +0.26 ± 0.04 (paired t = 14).
  - BRISQUE vector, λ = 0.3: +0.29 ± 0.10. This setting was picked on seed 0.
  - Scalar scores, best: +0.11 ± 0.07.
- **Still below the noise floor of 0.53**, so this is not yet a result. It is the strongest selection-side signal so
  far after RamenActive prob T = 3 (+0.62).
- **Alone vs together:** each vector alone does as well as or better than both concatenated (+0.22).

## What was built

| piece | file | what it does |
|---|---|---|
| features | `ImageQuality.with_features` in [src/models/iqa.py](../src/models/iqa.py) | scores and the 72-d NSS vector (BRISQUE's 36 features, then NIQE's 36 block features averaged over the image) in one pass; scores bit-identical to before |
| memory | `NSSCache` in [src/methods/RamenNSS.py](../src/methods/RamenNSS.py) | QualityCache (Ramen's class queue + NIQE, BRISQUE, true label) + the NSS vector per entry |
| method | `RamenNSS`, same file | selection score = CLIP distance + λ · RMS distance of the normalised NSS vectors |
| normaliser | `StreamNormaliser`, same file | causal running mean and covariance → PCA whitening (`nss_norm: white`) or z-score (`z`) |
| configs | `cfg/{ImageNetC5K,CIFAR10C,CIFAR100C,DomainNet}/RamenNSS.yaml` | each dataset's Ramen settings + `nss_features`, `nss_norm`, `nss_lambda` |
| experiments | [nss_active_set.py](nss_active_set.py) | `extract` (NSS vectors of every image), `simulate` (offline replay), `run`, `analyse` |

- **Hooks added to RamenQuality**, so RamenNSS can reuse its retrieval, weights, diagnostics and label logging:
  `measure()`, `class_penalty()`, and `**extra` fields in `QualityCache.add`. They change no behaviour: RamenQuality
  reproduces its pre-change subset run with 100% identical predictions.
- **λ = 0 checks.** RamenNSS at λ = 0 matches Ramen 100% on the subset. The NSS scores recomputed offline are
  bit-identical to the ones RamenQuality computed inside the stream.

## Offline analysis first: is the vector worth running?

All of this uses the cached seed-0 zero-shot run (CLIP features, predictions, entropies) and the NSS vectors of all
75,000 ImageNet-C images. No adaptation runs are involved.

### 1. How well each descriptor identifies the corruption

For each image: the share of its 10 nearest images (from a random 20,000) that come from the same corruption.
Chance is 6.7%.

| descriptor | 10-NN same corruption |
|---|---|
| scalar scores: NIQE / BRISQUE / both (ranks) | 25.6% / 20.3% / 38.1% |
| CLIP image features (what Ramen retrieves with) | 71.1% |
| NSS BRISQUE 36-d (z / rank / whitened) | 80.1% / 80.4% / 81.1% |
| NSS NIQE 36-d (z / rank / whitened) | 79.2% / 79.2% / 78.4% |
| NSS both 72-d (z / rank / whitened) | 80.5% / 80.7% / 79.7% |

The vectors identify the corruption about twice as well as the scores, and better than CLIP features do. This is
the property the scalar scores lacked. Either vector alone is about as good as both together, so concatenating them
adds little.

### 2. Ramen's selection, replayed

`simulate` replays Ramen's memory exactly: per-class priority queues with fp16 priorities, including the saturation
after 65,504 samples. For 60 evenly spaced batches (6,000 queries), it then picks each class queue's entry by
CLIP distance + λ · R and weights it by exp(−H), as Ramen does with topk = 1 and β = 0.

**Checked against the real runs' logs:**

| | replay | real run |
|---|---|---|
| λ = 0: same-corruption top-8 / label purity | 44.8% / 0.47% | 43.6% / 0.48% |
| scalar both, λ = 0.1 / 0.3 / 1 / 3: same-corruption top-8 | 47.9 / 47.4 / 40.4 / 28.7% | 47.0 / 46.6 / 39.8 / 28.6% |
| scalar both, λ = 0.1 / 0.3 / 1 / 3: label purity | 0.42 / 0.36 / 0.28 / 0.22% | 0.43 / 0.37 / 0.28 / 0.22% |

The replay tracks reality to about 1 point, so it can choose settings before spending GPU time.

**Replayed selection.** "Same-corruption" is the weight share of the top-8 active-set entries from the query's
corruption. Label purity is the weight share of the whole active set with the query's label.

| descriptor | λ | same-corruption top-8 | label purity |
|---|---|---|---|
| scalar both (ranks), best | 0.1 | 47.9% | 0.42% |
| NSS BRISQUE, whitened | 0.1 / 0.3 | **51.8%** / 53.5% | 0.43% / 0.35% |
| NSS NIQE, whitened | 0.1 / 0.3 | 50.3% / 50.9% | 0.43% / 0.35% |
| NSS both, whitened | 0.1 / 0.3 | 48.6% / 49.2% | 0.45% / 0.39% |
| NSS BRISQUE / NIQE / both, z-scored | 0.1 | 48.5% / 48.2% / 48.5% | 0.37% |
| NSS any, any normalisation | 1 | 38–46% | 0.22–0.30% |

- **Whitened vectors beat the best scalar setting on both proxies at λ = 0.1.** They give 3–4 more points of
  same-corruption support at the same or better label purity.
- **Z-scoring loses label purity for little gain.**
- **λ = 1 is too strong for every descriptor:** it trades away CLIP's semantic match and both proxies fall.
- **Concatenating both vectors is worse than either alone** in the replay. A likely reason, not tested: the NIQE half
  repeats much of BRISQUE (both are AGGD fits to MSCN products), and whitening the 72-d covariance spends weight on
  that redundancy.

So the real runs use whitened vectors at λ ∈ {0.1, 0.3}, for each feature set.

## Results: ImageNet-C, seeds 0–4

Gain over Ramen in points, ViT-B/16, mixed stream. Ramen's cached runs are reused, with the stream order asserted
identical. λ = 0.1 was fixed for all three feature sets before the seed runs. BRISQUE at λ = 0.3 was the seed-0
best and was added afterwards, so its mean is slightly optimistic.

| whitened NSS vector | s0 | s1 | s2 | s3 | s4 | mean ± std | paired t | same-corruption top-8 | label purity |
|---|---|---|---|---|---|---|---|---|---|
| NIQE 36-d, λ = 0.1 | +0.27 | +0.27 | +0.27 | +0.31 | +0.20 | **+0.26 ± 0.04** | 14.3 | 46.8–51.5% | 0.43% |
| BRISQUE 36-d, λ = 0.1 | +0.20 | +0.28 | +0.26 | +0.33 | +0.12 | +0.24 ± 0.08 | 6.4 | 47.6–52.6% | 0.43% |
| both 72-d, λ = 0.1 | +0.25 | +0.17 | +0.19 | +0.31 | +0.18 | +0.22 ± 0.06 | 8.4 | 45.2–49.7% | 0.45% |
| BRISQUE 36-d, λ = 0.3 (seed-0 pick) | +0.32 | +0.23 | +0.38 | +0.37 | +0.16 | +0.29 ± 0.10 | 6.9 | 49.7–54.5% | 0.35% |
| *scalar scores, both, λ = 0.1 (earlier)* | +0.19 | +0.07 | +0.14 | +0.14 | +0.03 | +0.11 ± 0.07 | 3.8 | | 0.43% |

Seed 0 only, λ = 0.3: NIQE +0.24, both +0.25.

- **Every variant is positive on all 5 seeds,** and each roughly doubles the scalar-score gain at the same λ and
  label purity. Moving from score to vector buys about +0.1 to +0.15.
- **Still below the noise floor.** Ramen's range over seeds 0–4 is 0.26, so the floor is 0.53, and no variant
  clears it. The project rule says to treat this as promising, not established. For comparison, RamenActive prob
  T = 3 cleared it on the mean (+0.62).
- **Same-corruption support tracks the gain.** Ramen's share is 43.6%; the vectors raise it to about 47–54%.
  BRISQUE at λ = 0.3 buys the most corruption match but loses label purity (0.35%), and ends up about where
  λ = 0.1 does.
- **Alone vs together.** NIQE's vector is the most consistent (std 0.04). Concatenating the two vectors is the
  weakest of the three, as the replay predicted.

## Caveats

- **ImageNet-C only.** The settings came from seed 0 and the replay. DomainNet, CIFAR-10-C and CIFAR-100-C configs
  exist but have not been run.
- **The replay assumes topk = 1 and β = 0.** Datasets with β > 0 (CIFAR, DomainNet) need the weight term added
  first.
- **Speed.** About 15 min per run vs 7 for Ramen, as for RamenQuality: the per-class rank lookups are inherited
  even though the scalar term is off here.

## Next steps

1. **Stack with RamenActive prob T = 3.** Scalar scores stacked at seed 0 (+0.66 vs +0.51). NIQE vector λ = 0.1 +
   prob T = 3 on seeds 0–4 is 5 runs. If the gains add, it should clear the floor.
2. **Held-out test.** DomainNet with NIQE vector, λ = 0.1, fixed in advance. This needs the replay extended to β = 5
   first, to check that λ transfers.
3. **Drop the per-class scalar-rank work when `quality_lambda` = 0.** That should bring the run time close to
   Ramen's.

## Files

- **Code:** src/methods/RamenNSS.py, `ImageQuality.with_features` in src/models/iqa.py, cfg/*/RamenNSS.yaml,
  new_experiments/nss_active_set.py.
- **Data and results:** `results/ImageNetC5K_nss_features.npz` (NSS vectors of all 75k images),
  `results/ImageNetC5K_nss_selection_replay.csv`, `results/ImageNetC5K_nss_active_set_s{0..4}.csv`, and the
  per-sample `*_nss_*.pkl` files.
- **Logs:** the full analysis printout is in `results/logs/nss_ImageNetC5K_analysis.txt`.
