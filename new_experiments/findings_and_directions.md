# Ramen test-time adaptation: findings so far and next directions

*2026-10-08. Supersedes the status sections of [next_directions.md](next_directions.md), which keeps the detailed CIFAR tables.*

## Summary

1. **Ramen helps on all four benchmarks, but much less on DomainNet:** +11.3 pts on CIFAR-10-C, +9.0 on CIFAR-100-C, +5.8 on ImageNet-C, +1.4 on DomainNet. The gain is largest where zero-shot CLIP is worst.
2. **On the large label spaces, Ramen's main effect is removing zero-shot "attractor" classes.** Zero-shot CLIP dumps many errors onto a few classes (`CRT monitor` on ImageNet-C; `squiggle`, `envelope` on DomainNet), and Ramen drains them. On CIFAR-10-C it does the opposite: it moves the over-prediction to other classes (bird → deer and truck).
3. **Our best idea, an output prior correction, does not transfer.** It was solid on CIFAR (+0.57 ± 0.05 and +1.50 ± 0.03 pts over seeds 0–4, above the noise floor). With its hyperparameter fixed on CIFAR, it gives only **+0.23 on ImageNet-C and −0.53 on DomainNet**. The assumption of a uniform label distribution and the fixed strength λ are the likely causes.
4. **Four other fixes did not beat Ramen:** a zero-shot anchor, contamination-aware cache weighting, normalized SGD, and gating on zero-shot confidence. The gate gives at most +0.01 pts on all four datasets, even though the per-sample oracle ceiling is +1.7 to +5.7 pts.
5. **There is a bug in the released Ramen code with no effect on the standard benchmarks:** its cache priorities are fp16 and overflow after 65,520 samples. On the shuffled streams used for evaluation, even the 596k-sample DomainNet stream, the gain over zero-shot stays flat. It should matter on non-shuffled (domain-ordered) streams, which we have not tested yet.

---

## 1. Setup

- **Method:** Ramen (retrieval-based per-sample adaptation of CLIP's LayerNorm parameters). Each test sample retrieves gradients of similar past samples from a class-wise cache and takes one SignSGD step (lr 0.01), and the weights are reset afterwards. The hyperparameters are the repo's per-dataset configs in `cfg/`.
- **Baseline:** NoAdapt (zero-shot CLIP), evaluated on the identical sample stream, so every comparison is paired per sample.
- **Stream:** "mixed" mode, where all corruptions or domains are shuffled into one stream, with batch size 100 and seed 0 unless stated.

| dataset | model | stream | classes |
|---|---|---|---|
| CIFAR-10-C (sev. 5) | ViT-B/32 | 15 × 10k = 150k | 10 |
| CIFAR-100-C (sev. 5) | ViT-B/16 | 15 × 10k = 150k | 100 |
| ImageNet-C (sev. 5, the repo's 5k-image subset) | ViT-B/16 | 15 × 5k = 75k | 1000 |
| DomainNet (6 domains) | ViT-B/32 | 596k | 345 |

**ImageNet-C uses ViT-B/16, not the paper's ViT-L/14.** Ramen with L/14 does not fit on our 24 GB GPUs. Our ImageNet-C numbers are therefore comparable to our CIFAR-100-C runs, not to the paper's ImageNet-C table.

**Noise floor:** a gain counts only if it is above the seed-to-seed noise of the paired difference. That is +0.31 pts on CIFAR-10-C and +0.21 on CIFAR-100-C, from seeds 0–4. ImageNet-C and DomainNet so far have seed 0 only, so they have no noise floor yet.

## 2. Ramen across the four benchmarks

| dataset | NoAdapt | Ramen | gain | per-sample oracle ceiling* |
|---|---|---|---|---|
| CIFAR-10-C | 61.3 | 72.61 | +11.3 | +5.7 |
| CIFAR-100-C | 37.1 | 46.10 | +9.0 | +4.5 |
| ImageNet-C | 27.29 | 33.08 | +5.8 | +3.8 |
| DomainNet | 55.37 | 56.80 | +1.4 | +1.7 (per sample: 54.52 → 56.25) |

\*The accuracy gained if, for each sample, we picked whichever of Ramen and NoAdapt is right. It bounds any method that selects between the two.

- **ImageNet-C:** Ramen improves all 15 corruptions. The biggest gains are where zero-shot is weakest: contrast +10.7, elastic +9.0, impulse noise +8.3. The smallest are where it is strongest: brightness +0.9, frost +2.0.
- **DomainNet:** quickdraw gains +4.2 (zero-shot only gets 13.3% there). Clipart, infograph, painting and sketch gain about +1, and real gains only +0.15. DomainNet averages are over the 6 domains. Weighted per image, Ramen gets 54.52%, because quickdraw is both the largest and the hardest domain.
- **Churn under the net gain:** Ramen fixes 2–6% of each DomainNet domain and breaks 1–2%. On CIFAR-10-C, fewer than half of the frog → cat errors are shared between the two methods, even though the error rate barely changes.
- **No degradation along the stream on any dataset.** The gain over zero-shot stays flat or grows over the stream (DomainNet: +1.5 to +2 pts throughout 596k samples; ImageNet-C: +3 to +9 pts throughout).

## 3. What Ramen actually does to the predictions

Class prediction bias here means predicted share minus true share, in percentage points.

| dataset | class | NoAdapt | Ramen |
|---|---|---|---|
| ImageNet-C | CRT monitor | +4.2 | +0.3 |
| ImageNet-C | window screen | +1.2 | +0.6 |
| DomainNet | squiggle | +7.3 | +1.7 |
| DomainNet | envelope | +2.4 | +0.8 |
| CIFAR-10-C | bird (share, true 10%) | 18.5% | 9.8% |
| CIFAR-10-C | deer (share, true 10%) | 6.9% | 12.8% |
| CIFAR-10-C | truck (share, true 10%) | 6.1% | 11.0% |

- **On DomainNet, Ramen spreads errors away from zero-shot's typical mistakes.** For each class we took the 5 classes zero-shot most often confuses it with. 48% of NoAdapt's errors land in one of those 5, against 36% for Ramen. Errors into the attractors fall sharply, for example calculator → envelope 32.6% → 12.9% and camouflage → squiggle 36.5% → 23.8%.
- **On ImageNet-C, Ramen keeps the same confusers (39% vs 39%) and simply puts more probability on the true class** (17.6% → 25.4% on average).
- **Fine-grained near-synonyms get worse with adaptation.** On ImageNet-C: barbell → dumbbell 47% → 73% of barbell images, bow tie → bridegroom, balance beam → horizontal bar, and dog-breed pairs. On DomainNet: coffee cup ↔ mug, bus → school bus, goatee → beard.
- **On CIFAR-10-C the over-prediction moves rather than disappears.** frog → deer goes from 4.1% to 15.9%, and automobile → truck from 7.1% to 14.3%. The truck/automobile boundary moves; it does not get sharper.

## 4. Ideas tested and ruled out (CIFAR-10-C / CIFAR-100-C)

| idea | result | why it fails |
|---|---|---|
| Zero-shot anchor: pull the adapted weights or predictions back toward zero-shot (4 variants) | no variant beats Ramen | harmful and helpful flips have the same zero-shot margin profile, so no anchor strength separates them |
| Keep zero-shot when it is confident | ≤ +0.01 pts on all four datasets | zero-shot confidence predicts poorly whether adaptation will hurt (AUROC 0.61–0.67) |
| Contamination-aware weighting of cached neighbours | −0.7 / −2.9 pts | support label purity rises 48.8 → 76.6%, but the effective support shrinks 158 → 12 samples |
| Normalized SGD instead of SignSGD | −3.8 / −5.8 pts | the fixed-size sign step matters |
| Noise corruptions contaminate the shared cache | single-domain vs mixed within ±0.5 pts | — |

## 5. The output prior correction and its held-out test

**Idea.** Ramen over- and under-predicts some classes. We keep an EMA π̂ of Ramen's predicted class marginal and correct each output as `logit' = log p − λ · log(π̂ / uniform)`. Ramen's cache only stores zero-shot features and gradients at the base weights, so its outputs never feed back into the stream. A correction to the output can therefore be evaluated exactly offline from one logged run. Code: [marginal_correction.py](marginal_correction.py).

**CIFAR result (λ = 2, m = 0.99, chosen on seed 0), paired gain over Ramen:**

| | CIFAR-10-C | CIFAR-100-C |
|---|---|---|
| seeds 0–4 | +0.57 ± 0.05 | +1.50 ± 0.03 |
| above noise floor on every seed | yes | yes |
| class-share gap TV(pred, true) | 0.054 → 0.029 | 0.170 → 0.086 |
| ECE | 8.35 → 8.02 | 9.90 → 7.85 |

The same correction applied to zero-shot gives +1.2 / +1.7, against Ramen's +11.3 / +9.0. So Ramen's gain is not mainly debiasing; the correction stacks on top of it.

**Held-out test (2026-10-08): the same setting, nothing re-tuned, seed 0.** Table: `results/marginal_correction_<dataset>_s0_heldout.csv`.

| λ (m = 0.99) | ImageNet-C gain | DomainNet gain |
|---|---|---|
| 0.5 | +0.51 | +0.19 |
| 1 | +0.75 | +0.20 |
| **2 (pre-registered)** | **+0.23** | **−0.53** |
| 3 | −1.64 | −3.00 |
| control: zero-shot + correction (λ = 2) | +0.01 | −0.12 |

- **The pre-registered λ = 2 does not transfer, so this is a negative result for the method as specified.** λ = 1 would help on both datasets, but choosing it now would mean tuning on the test sets. ImageNet-C and DomainNet have no noise floor yet either.
- **On DomainNet it helps only quickdraw** (+1.7), the attractor-heavy domain, and hurts every other domain (infograph −1.9, painting −1.5). A single global π̂ cannot fit a bias that differs by domain.
- **The uniform-target assumption is wrong on both new datasets.** DomainNet classes have 878–2,799 images (TV from uniform 0.08). The ImageNet-C 5k subset is also imbalanced: 0–180 images per class, 3 classes absent, TV 0.17. Pushing predictions toward uniform therefore partly pushes them away from the true label distribution.
- **On these datasets, debiasing does not help zero-shot either** (the +0.01 / −0.12 control). That fits Section 3: on the large label spaces Ramen has already removed most of the attractor bias, so there is less left to correct.

## 6. Bug in the released code

- **What happens:** Ramen's priority cache stores priorities in fp16. They overflow to `inf` after 65,520 samples, after which full class queues stop accepting new entries. In the last 10% of the CIFAR stream, only 30–47% of samples get inserted.
- **Effect on accuracy:** fixing it (fp32) changes accuracy by about 0 on shuffled streams. On DomainNet, where the stream is about 9× past the overflow point, the gain over zero-shot also shows no late decline.
- **Where it should matter:** a domain-ordered stream, where the frozen cache would hold only early domains.

## 7. Directions, in priority order

**1. Fix the prior correction's two assumptions, and pre-register before testing.**
- **Label distribution:** replace the uniform target with an estimated one. For example, EM prior re-estimation (Saerens et al. 2002), started from zero-shot predictions; it needs no labels.
- **Strength:** make λ scale-free in the number of classes, or set it from a label-free criterion such as the drop in entropy of π̂.
- **Per domain:** use a per-domain π̂ by clustering zero-shot features, since domains are unknown at test time.
- **Protocol:** develop on CIFAR, fix every choice, then test once on ImageNet-C and DomainNet.
- **Cost:** all of this is offline from the saved probabilities, so it is cheap.

**2. Seeds 1–4 on ImageNet-C and DomainNet.** We need noise floors there before claiming any gain or loss below about 1 pt. Cost is about 2 × 9 min per ImageNet-C seed and 2 × 36 min per DomainNet seed, and two GPUs can run in parallel.

**3. Why does Ramen give so little on DomainNet?** Ramen helps quickdraw but does almost nothing on real (+0.15), where zero-shot is already at 82%. Questions to answer:
- (a) Is it the configuration (capacity 300, top-10, β = 5) or the method? Run a small sweep on one domain.
- (b) Does single-domain mode help, given that DomainNet domains differ in style, not just low-level corruption?
- (c) Does the oracle ceiling (+1.7 per sample) mean there is little left to gain on DomainNet at all?

**4. A better signal for when to adapt.** The oracle ceilings are +1.7 to +5.7 pts, and zero-shot confidence captures none of that. Score candidate signals offline by AUROC on the samples where Ramen and NoAdapt disagree:
- label agreement in the retrieved neighbours;
- neighbour distance;
- size of the logit shift;
- whether adaptation moves a sample *into* an attractor or near-synonym class (Section 3).

**5. Per-sample step size.** Every adapted parameter moves exactly ±0.01. One logged run with the aggregated retrieved gradient per sample gives, exactly offline:
- accuracy at other global step sizes;
- the oracle per-sample step size.

A small oracle gap means we drop this direction.

**6. Domain-ordered (non-shuffled) streams.** This is the realistic deployment setting, and the one where the fp16 cache bug (Section 6) should show. Compare Ramen, Ramen with fp32 priorities, and NoAdapt on DomainNet with domains presented in sequence.

## 8. Where things are

- **Notebooks** (in `new_experiments/`):
  - Ramen vs NoAdapt per dataset: `cifar10c_ramen.ipynb`, `cifar100c_ramen.ipynb`, `imagenetc_ramen.ipynb`, `domainnet_ramen.ipynb`.
  - Probability and confusion analysis: the `*_confusion_probs.ipynb` notebook for each dataset.
- **Analysis code:** `ramen_analysis.py`, with one experiment class per dataset, and `marginal_correction.py`.
- **Cached per-sample results** (predictions, probabilities, features): `new_experiments/results/`.
- **Data:** `~/data/corruption/{CIFAR-10-C, CIFAR-100-C, ImageNet-C}` (severity 5 only) and `~/data/domainbed/domain_net`.
