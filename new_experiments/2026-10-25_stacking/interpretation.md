**The pre-registered hypothesis is rejected.** The full stack (T3 + knn + pc at λ = 2) does not beat Ramen on either dataset: ImageNet-C −0.02 ± 0.11 pts (positive on 3/5 seeds, floor 0.53), DomainNet −1.13 ± 0.04 (0/5, floor 0.11). It is worse than T3 alone on both.

**What each piece does on the large-label datasets (pre-registered settings, mean ± std over seeds 0–4):**

| | ImageNet-C 5K (floor 0.53) | DomainNet (floor 0.11) |
|---|---|---|
| T3 | +0.62 ± 0.11 (3/5 above floor) | +0.24 ± 0.01 (5/5) |
| knn on Ramen | −0.11 ± 0.03 (0/5 positive) | +0.20 ± 0.02 (5/5 above floor, held-out) |
| pc (λ = 2) on Ramen | +0.16 ± 0.09 | −0.52 ± 0.02 |
| T3 + knn | +0.42 ± 0.08 | +0.22 ± 0.02 |
| T3 + pc | **+0.72 ± 0.10 (5/5 above floor)** | −0.19 ± 0.03 |
| T3 + knn + pc | −0.02 ± 0.11 | −1.13 ± 0.04 |

- **No pre-registered combination wins on both datasets.** T3 + pc clears the ImageNet-C floor on every seed, but T3 was tuned on ImageNet-C seed 0, so that number is not held-out, and the same combination loses on DomainNet. T3 alone is still the best pre-registered result that holds on both.
- **kNN smoothing helps only where neighbours share the true class.** On DomainNet 54% of the neighbour weight falls on the query's true class and knn gives +0.20. On ImageNet-C only 25% does (19% is the same photo under another corruption), and knn hurts on all 5 seeds. With 1,000 classes and severity-5 corruptions, zero-shot features are too degraded for a feature-space vote. On DomainNet, T3 and knn do not add up: +0.22 together vs +0.24 and +0.20 alone, so they fix largely the same images.
- **kNN and pc at λ = 2 clash on both datasets** (stack − sum of parts: −0.56 to −0.82 on ImageNet-C, about −1.0 on DomainNet). Smoothing shrinks each image's top-1/top-2 log-margin (median 0.86 → 0.69 on ImageNet-C, 2.27 → 1.61 on DomainNet), so the same log-prior shift flips more predictions: 31 → 38% and 20 → 24% of all images. Most of the extra flips are breaks. λ = 2 is too strong once outputs are smoothed.
- **λ = 2 with a uniform target is the weak link.** Neither dataset has uniform classes (ImageNet-C 5K: 0–180 images per class; DomainNet: strongly imbalanced). On DomainNet, T3 + pc gains +1.30 on quickdraw but loses −1.42 on infograph and −0.95 on painting. It adds +470 images pooled but loses −0.19 domain-averaged.

**Post-hoc (λ chosen after seeing these results; not evidence yet):** T3 + pc with **λ = 1** clears both floors on all 5 seeds: ImageNet-C +1.13 ± 0.10 (+846 images), DomainNet +0.41 ± 0.01 (+2,818 images). λ = 0.5 is close (+0.96 / +0.39), so the result is not a knife-edge in λ. Adding kNN on top adds nothing on ImageNet-C (+1.15) and costs 0.17 on DomainNet (+0.24).

**Caveats.** Seeds change only the stream order of the same images, so the five seeds rule out stream-order luck, not overfitting to these two datasets. T3 on ImageNet-C is not held-out. Every λ = 1 / 0.5 number is post-hoc.

**Next step.** Pre-register **T3 + pc(λ = 1)** (no kNN) and test it on a dataset none of these settings was tuned on. It needs a large label space, because T3 only helps there. ImageNet-R / ImageNet-Sketch / ImageNet-V2 would need a download. The cheap fallback is re-scoring CIFAR-100-C and EuroSAT with λ = 1 to check it doesn't break the small-label datasets where λ = 2 was tuned. If λ = 1 holds on a fresh dataset, the next lever is the uniform target: estimate the class prior instead of assuming it, possibly per cluster (direction 2 of the earlier plan).
