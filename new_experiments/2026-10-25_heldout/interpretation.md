**The pre-registered claim fails.** T3 + pc(λ = 1) is **−1.15 ± 0.09** pts against Ramen (below Ramen on 5/5 seeds, every per-seed CI below zero; floor 0.48). The prior correction breaks it: Ramen + pc(λ = 1) alone is −2.58 ± 0.10.

**T3 alone is confirmed on held-out data:** **+0.59 ± 0.04**, above the floor on 5/5 seeds (min +0.56), every per-seed CI excluding zero. This is the first time a change carried over to a dataset it was never tuned on and cleared the floor on every seed. T3 = 48.44% vs Ramen 47.84% vs NoAdapt 46.36% (domain-averaged).

**Why prior correction fails here: the uniform target points the wrong way.**
- 314 of the 1,000 classes (those of ImageNet-R ∪ ImageNet-A) carry 53,453 of 88,389 images (60.5%). The true class marginal is far from uniform: its total-variation distance from uniform is 0.31, against 0.17 on ImageNet-C 5K.
- Ramen already under-predicts those classes: 49.0% of its predictions vs 60.5% true (seed 0). Correcting toward uniform pushes it further the wrong way, to 44.3%, and sends 3,437 more wrong predictions into Sketch-only classes (29,686 → 33,123).
- So pc helps only when the label marginal is close to uniform. That held on CIFAR, EuroSAT and roughly on ImageNet-C 5K, but not here or on DomainNet at λ = 2. Its gains on the earlier datasets partly reflected their near-uniform marginals, not a general property.

**Post-hoc (context only):** every λ hurts here (λ = 0.5: −0.13, λ = 2: −3.90). kNN smoothing on Ramen gives +0.43 ± 0.12 (5/5 positive, 2/5 above the floor), but stacking it with pc(λ = 1) on T3 is −2.56.

**Caveats.** The 1,000-way label space is a design choice (fixed in advance). Under each dataset's own 200-class head the marginal would be closer to uniform and pc might behave differently. That is untested, and testing it now would be post-hoc.

**What this changes.**
1. T3 is the one held-out-confirmed improvement: ImageNet-C +0.62, DomainNet +0.24, ImageNet-R/A/Sketch +0.59, each on 5/5 seeds.
2. Prior correction with a uniform target should not be part of the method. The open question is whether an *estimated* target fixes it: EM label-shift estimation, compared against an oracle true-marginal ceiling. This stream, with its strongly non-uniform marginal, is now the natural stress test. Any result on it would be post-hoc for this dataset.
