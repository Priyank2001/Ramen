# Pre-registration: prior correction toward an estimated class marginal

Written 2026-10-26, before any of the estimators below have been computed on any dataset. Everything runs offline from saved probabilities (the correction never feeds back into Ramen's cache), so no new GPU runs are needed.

## Why
Prior correction (pc) subtracts λ·log(π_t / target) from the logits, where π_t is the EMA of the method's own predicted marginal. With a **uniform** target it helped on near-uniform label spaces (CIFAR, EuroSAT, roughly ImageNet-C) but failed the held-out test on ImageNet-R/A/Sketch (T3 + pc(λ = 1): −1.15 ± 0.09 vs Ramen), where 314 of 1,000 classes carry 60.5% of the images. The question is whether the idea survives with the *right* target, and whether that target can be estimated without labels.

Caveat stated up front: from one unlabelled stream, a true label shift and a classifier bias toward some classes look the same. An estimator can only separate them by trusting some reference. The ones below trust the **zero-shot** classifier, which Ramen's adaptation has not touched.

## Targets (fixed)
All variants use λ = 1, m = 0.99 for π_t (frozen on 2026-10-25), the soft marginal, the current batch included, and a floor of 0.01/K on both π_t and the target, as in `stack.prior_correct`.

| Name | Target q_t | Uses labels? | Role |
|---|---|---|---|
| uniform | 1/K | no | reference (the existing pc(λ = 1)) |
| **oracle** | true label marginal of the whole stream | **yes (oracle)** | ceiling: is the right target worth anything at all? |
| **zs-EM** | online EM (Saerens et al. 2002) on the zero-shot probabilities: per batch r_i ∝ p_zs(x_i) ⊙ q_{t−1}, then q_t = m·q_{t−1} + (1 − m)·mean_batch(r), q_0 uniform, m = 0.99 | no | **primary** label-free estimator |
| zs-marginal | EMA (m = 0.99) of the zero-shot soft marginal: "undo only the drift that adaptation adds" | no | secondary |
| self-EM | the same online EM on the method's own probabilities | no | secondary; expected to be near-useless because it mostly re-estimates the method's own biased marginal |

Zero-shot probabilities: the seed-0 NoAdapt run (stateless), realigned to each seed's stream order by (domain, sample) id.

## Bases and data (fixed)
- Base = **T3** where its probabilities exist (ImageNet-C 5K, DomainNet, ImageNet-R/A/Sketch), and **Ramen** on all six datasets (adding CIFAR-10-C, CIFAR-100-C and EuroSAT). Seeds 0–4. No kNN.
- Noise floors (2 × Ramen's range over seeds 0–4): ImageNet-C 0.53, DomainNet 0.11, CIFAR-10-C 0.31, CIFAR-100-C 0.21, EuroSAT 1.51, ImageNet-R/A/Sketch 0.48.

## What counts
Metric: domain-averaged accuracy, gain of **base + pc(target)** over **the same base without pc**, mean over seeds 0–4, plus per-seed cluster-bootstrap 95% CIs.

1. **Oracle ceiling (decides whether the direction is alive).** If T3 + pc(oracle) does not beat T3 by more than the floor on ImageNet-R/A/Sketch, correcting toward even the true marginal does not pay, and the direction is closed.
2. **Primary claim.** zs-EM "works" if, with base T3, its mean gain is above the floor on ImageNet-R/A/Sketch **and** above zero on ImageNet-C and DomainNet, **and**, with base Ramen, its mean gain is not below −floor on any of the six datasets.
3. Secondary, reported but not part of the claim: zs-marginal, self-EM, uniform; how close each estimated target ends up to the true marginal (total-variation distance); predicted vs true share of the R/A classes.

## Status of the datasets
None of these is held out any more: ImageNet-R/A/Sketch was used for the 2026-10-25 test and the motivation comes from its failure. λ and m are frozen and the estimators are fixed here, so nothing is tuned, but **a pass needs confirmation on a new dataset** before it goes into the method. Candidates: ImageNet-V2 (near-uniform, 1,000 classes), or ImageNet-R/A each with its own 200-class head.
