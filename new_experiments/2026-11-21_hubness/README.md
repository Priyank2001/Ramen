# Hubness correction, offline check (2026-11-21)

Question (see [PREREGISTRATION.md](PREREGISTRATION.md)): CLIP zero-shot piles predictions onto a few "hub" classes. Can a
label-free, causal correction from retrieval (CSLS, inverted softmax, centring), which measures hubness from the
similarity *geometry* instead of from the predicted marginal, fix this where prior correction (pc) fails?
**Verdict: the pre-registered primary claim fails. Hubness is real and the geometric hub score tracks it, but removing it
does not buy accuracy on the large label spaces, and no method beats pc consistently.**

## Files

| File | Job |
|---|---|
| `PREREGISTRATION.md` | methods, tuning grid, claims, sanity checks (one addendum about fp16 ties, before any result) |
| `data.py` | `StreamData`: features, class text embeddings, labels, batches in a seed's stream order; zero-shot or Ramen level |
| `hubness.py` | one class per method: `ZeroShot`, `Csls`, `PermutedCsls` (control), `InvertedSoftmax`, `Centring` |
| `sanity.py` | cached predictions reproduced (fp16 ties only); CSLS(β = 0) == zero-shot → `logs/sanity.log` |
| `experiment.py` | `tune` (ImageNet-C seed 0), `zero-shot` (6 datasets × seeds 0–4), `ramen` (seed 0) → `tables/` |
| `posthoc.py` | POST-HOC: pc λ = 1, 2 reference; hub score vs class frequency; hubness profile table |
| `tables/`, `logs/`, `results/` | outputs; cached text embeddings |

Everything is offline from cached features (CPU/GPU for the similarity products only). No label is used by any method.

## Math

Notation: $z_i$ = L2-normalised CLIP image feature (saved NoAdapt feature; at the Ramen level the saved adapted feature
$\tilde z_i$), $w_c$ = normalised text embedding, $\cos_{ic} = z_i^\top w_c$, prediction $\hat y_i = \arg\max_c s_{ic}$.
Baseline (zero-shot) $s_{ic} = \cos_{ic}$ (Ramen's logits are $100\,\cos$; the scale does not change the argmax). Batches $t$ of
100 in the stream order; $\mathcal{R}_t$ = all images up to and including batch $t$ (the current batch is in the reference,
as Ramen writes before retrieving).

| Method | score | symbols | config (frozen) |
|---|---|---|---|
| CSLS | $s_{ic} = \cos_{ic} - \beta\, h_c,\; h_c = \frac{1}{\kappa}\sum_{j \in \mathrm{top}_\kappa(\cos_{\cdot c}\,;\,\mathcal R_t)} \cos_{jc}$ | κ nearest queries, β strength | κ = 100, β = 0.5 |
| inverted softmax | $s_{ic} = \tau \cos_{ic} - \log\sum_{j\in\mathcal R_t} e^{\tau \cos_{jc}}$ | τ temperature | τ = 50 |
| centring | $s_{ic} = \hat z_i^\top \hat w_c$ with $\hat z_i = \frac{z_i - \bar z_t}{\lVert z_i - \bar z_t\rVert}$, $\hat w_c = \frac{w_c - \bar w}{\lVert w_c - \bar w \rVert}$ | $\bar z_t$ mean of $\mathcal R_t$ | none |
| permuted CSLS (control) | CSLS with $h_{\pi(c)}$ for a fixed random permutation $\pi$ | same κ, β | κ = 100, β = 0.5 |

Tuning was on ImageNet-C seed 0 (zero-shot 27.27): CSLS best κ = 100, β = 0.5 (+0.56), inverted softmax best τ = 50 (+0.50). **Both
sit at a grid edge or near it: CSLS's best is the largest κ (capped at the batch size 100) and the smallest β on the grid,
so a milder β may be better; this was not explored.** Floors are 2 × Ramen's range over seeds 0–4 (DomainNet 0.11, ImageNet-C
0.53, ImageNet-R/A/Sketch 0.48, CIFAR-10-C 0.31, CIFAR-100-C 0.21, EuroSAT 1.51).

## Results

**Zero-shot level, mean gain over zero-shot, seeds 0–4** (std ≤ 0.11 everywhere except the control: ±0.85 CIFAR-10-C, ±3.5 EuroSAT).
`*` = in-sample (ImageNet-C tuned CSLS and inverted softmax). Bold = above the floor.

| dataset (zero-shot acc) | floor | CSLS | permuted CSLS | inverted softmax | centring | pc λ=1 | pc λ=2 |
|---|---|---|---|---|---|---|---|
| ImageNet-C 5K (27.27) | 0.53 | **+0.54\*** | −1.59 | +0.49\* | −1.29 | +0.92 | −0.01 |
| DomainNet (55.37) | 0.11 | −0.55 | −2.04 | −0.70 | −0.28 | +0.61 | −0.12 |
| ImageNet-R/A/Sketch (46.34) | 0.48 | −1.32 | −2.95 | −1.51 | **+0.65** | −2.62 | −8.14 |
| CIFAR-10-C (61.39) | 0.31 | −1.70 | −1.91 | **+0.60** | **+1.19** | +1.11 | +1.07 |
| CIFAR-100-C (37.09) | 0.21 | −0.25 | −4.43 | **+1.28** | **+0.63** | +1.61 | +0.75 |
| EuroSAT (48.09) | 1.51 | **+10.82** | −7.25 | **+7.63** | **+7.60** | +6.51 | +9.44 |
| datasets above floor | | 2 / 6 | 0 / 6 | 3 / 6 | 4 / 6 | | |

The pc columns are POST-HOC references (`posthoc.py`; uniform target, the project's pc).

**Ramen level (seed 0 only; Ramen's adapted features; gain over Ramen).**

| dataset (floor) | CSLS | inverted softmax | centring | pc λ=1 | pc λ=2 |
|---|---|---|---|---|---|
| ImageNet-C (0.53) | +0.65 | +0.52 | −1.74 | +0.75 | +0.23 |
| DomainNet (0.11) | −0.48 | −0.68 | −0.34 | +0.20 | −0.53 |
| CIFAR-10-C (0.31) | −0.90 | +0.20 | +0.46 | +0.33 | +0.49 |
| CIFAR-100-C (0.21) | +0.31 | +1.46 | +0.57 | +1.12 | +1.45 |
| EuroSAT (1.51) | +4.77 | **+7.64** | +5.72 | +1.87 | +2.49 |

**Hubness itself (seed 0, ImageNet-C / DomainNet / EuroSAT).** The 1% most-predicted zero-shot classes (10 / 3 / 1 classes)
receive 12.6% / 11.8% / 26.4% of the predictions, against a true share of 0.98% / 0.93% / 9.26%. Inverted softmax cuts that to
2.0% / 1.7% / 10.3% (skew of the predicted counts 16.6 → 2.0 on ImageNet-C, 14.2 → 1.5 on DomainNet), CSLS only to 9.0% /
8.1% / 12.5%. CSLS's hub score follows how often a class is *predicted* (Spearman 0.34 / 0.21 / 0.71) and not how often it
*occurs* (0.11 / −0.00 / −0.10; labels used for this description only), as the premise needs.

Pre-registered verdicts:
- **Primary FAIL:** frozen CSLS clears the floor on 2 of 6 datasets (ImageNet-C, in-sample, and EuroSAT; needs ≥ 4) and loses on
  DomainNet, R/A/Sketch and CIFAR-10-C. It beats its permuted control on every dataset it passes (+2.13, +18.07).
- **Secondary:** inverted softmax FAIL (3 / 6; ImageNet-C +0.49 is 0.04 under its floor). Centring passes the written rule
  (4 / 6, every seed positive there) but see below.
- **Sanity:** passed (every disagreement with the cached predictions is an fp16 tie, top-2 gap ≤ 0.045 logit units).

## Why it does or does not make sense

- **Why it should help:** hubness is real here (above), the geometric score separates "sink" from "frequent" (Spearman
  above), and unlike pc it does not count queries, so it should survive a skewed marginal.
- **What happened:** it does not pay on the large label spaces. Removing the hubs works as a *hubness* fix but not as an
  *accuracy* fix: inverted softmax takes the ImageNet-C top-1% share from 12.6% to 2.0% and gains only +0.49 (2,863 fixed vs 2,490
  broken of 75,000 on seed 0); on DomainNet it loses (−0.70). The predictions it moves off the hubs go to other wrong classes
  about as often as to the right one, which is the "de-sinking without fixing sink errors" pattern of
  `../2026-10-18` and matches pc's λ-sensitivity (pc λ=2: −0.01 / −0.12 there).
- **Same split as pc:** all methods gain on the balanced 10/100-class datasets (CIFAR-C, EuroSAT) and lose or tie on the large
  or non-uniform ones. The pc reference (λ=1) beats every hubness correction on ImageNet-C (+0.92) and DomainNet (+0.61), and at
  the zero-shot level on CIFAR-C is within 0.1–1.0 of them. So hubness correction is not a better pc there.
- **Where it is different:** (i) centring is the only method that gains on ImageNet-R/A/Sketch (+0.65 vs pc −2.62), because it
  uses no label marginal at all; but it loses on ImageNet-C (−1.29, 5/5 seeds, the tuning set) and DomainNet (−0.28 vs floor 0.11),
  so it is not a general method. (ii) On EuroSAT at the Ramen level, inverted softmax (+7.64), centring (+5.72) and CSLS (+4.77) are
  well above pc (+1.87 / +2.49): seed 0 only, a balanced 10-class set, so I treat this as a lead, not a result.
- **Controls:** the permuted-hub control hurts everywhere (−1.6 to −7.3), so the corrections are large enough that *which* classes
  get shifted matters, and beating it is a low bar (it is not a matched control for centring, which has none).
- **Not shown:** that a milder CSLS (β < 0.5) helps (grid edge); that these stack with T3/knn; seeds 1–4 at the Ramen level
  (adapted features saved for seed 0 only); any effect on Ramen's *filing* (the zero-shot prediction that decides the queue),
  which would need a GPU replay: a hub-corrected filing would only matter if it changes which queue the sink images go to.

**Verdicts:** CSLS hurts / null (fails the primary claim). Inverted softmax null (3 / 6, below floor on ImageNet-C). Centring
mixed: helps where the marginal is skewed (R/A/Sketch), hurts on ImageNet-C and DomainNet. Lead worth one follow-up: EuroSAT
(Ramen-level gain +4.8 to +7.6 over Ramen, above pc), seeds 1–4 need Ramen re-runs that save adapted features.
