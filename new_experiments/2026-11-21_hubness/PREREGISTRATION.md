# Pre-registration: hubness correction, offline check (2026-11-21, before any number is computed)

**Question.** CLIP zero-shot predictions pile up on a few "hub" classes (the sinks: pillow, envelope, tractor, crab).
Prior correction toward a uniform or an estimated marginal fails on skewed label marginals, and label-free marginal
estimation is a dead end because it estimates the prior from the same predictions the hubs distort. Hubness
corrections come from retrieval (CSLS, inverted softmax / QB-Norm, centring) and measure a class's hubness from the
*geometry*: how similar the class vector is to its nearest queries, not how often it wins. Does that separate "sink"
from "frequent" better than the predicted marginal, and so give a label-free output correction that works?

**Methods** (all causal and label-free; reference set = every image of the stream so far, the current batch included;
batches of 100 in the stream's order; cos = cosine of CLIP image feature and class text embedding):

| name | score for class c of query i (argmax = prediction) | parameters |
|---|---|---|
| zero-shot | cos_ic | none |
| CSLS | cos_ic − β · h_c, h_c = mean of the κ largest cos_jc over the reference queries | κ, β |
| inverted softmax | τ · cos_ic − log Σ_j exp(τ cos_jc) over the reference queries | τ |
| centring | cos of (z_i − μ_z) and (w_c − μ_w), renormalised; μ_z = reference mean | none |
| CONTROL: permuted CSLS | CSLS with the h_c values permuted across classes (fixed per run) | same κ, β |

**Tuning (only here):** ImageNetC5K, seed 0, zero-shot level. Grid: CSLS κ ∈ {5, 20, 100} × β ∈ {0.5, 1, 2, 4};
inverted softmax τ ∈ {20, 50, 100}. Per family, the setting with the largest domain-averaged gain over zero-shot is
frozen (ties → smaller β / τ). Nothing is re-tuned afterwards.

**Held-out evaluation (frozen settings):** zero-shot level on ImageNetC5K, DomainNet, ImageNetVariants (R/A/Sketch),
CIFAR10C, CIFAR100C, EuroSAT, seeds 0–4 (a seed = stream order; per-image features are seed-independent). ImageNet-C is
the tuning set, so its numbers are in-sample.

**Primary claim.** Frozen CSLS gains over zero-shot more than the noise floor (DomainNet 0.11, ImageNet-C 0.53,
ImageNet-R/A/Sketch 0.48, CIFAR-10-C 0.31, CIFAR-100-C 0.21, EuroSAT 1.51; the floors are 2 × Ramen's range over seeds
0–4 and are reused for zero-shot) on at least 4 of the 6 datasets (mean of 5 seeds), **and** beats its permuted-hub
control by more than the floor on each dataset it passes. If the permuted control gains as much, "which classes are
hubs" is not what is being used; the correction is just a random per-class shift.

**Secondary.** Same test for inverted softmax and for centring. **Diagnostic (not a claim):** hubness before/after —
skewness of the predicted-class counts, TV distance of the predicted marginal to the true marginal (uses labels), and
the share of predictions that land in the 1% most-predicted zero-shot classes versus their true share.

**Ramen level (seed 0, CIFAR-10-C, CIFAR-100-C, DomainNet, EuroSAT, ImageNet-C — the datasets with saved Ramen features).**
The same frozen settings on Ramen's adapted logits `100 z̃_i · W` (z̃ = the adapted feature saved by the cached Ramen run;
hub scores from the z̃ of the stream so far). Reported against Ramen and, as a reference, against Ramen + pc (λ = 1: +0.35 /
+1.15 / +2.13 on CIFAR-10-C / CIFAR-100-C / EuroSAT). Informational: seed 0 only, no claim.

**Sanity checks (before the tables).** argmax of z·W reproduces the cached NoAdapt predictions on ≥ 99.9% of images
(fp16 ties); for the Ramen level, argmax of z̃·W reproduces Ramen's cached predictions the same way; κ = 1, β = 0 CSLS ==
zero-shot exactly.

**Written in advance:** a gain on ImageNet-C alone does not count (tuning set); a gain only at the Ramen level with no
zero-shot-level gain would point to Ramen's adapted features having their own hubs, not to CLIP's.

**Addendum (2026-11-21, after the first sanity run, before any result table).** The 99.9% agreement threshold above was
too tight: on ImageNet-C (1,000 classes) argmax of the fp32-normalised saved fp16 features reproduces 99.43% of the cached
predictions (426 of 75,000). The text embeddings equal the cached ones exactly (max |diff| 0). All 426 differing rows have a
top-2 logit gap below 0.04 (median 0.006; median over all rows 0.61), i.e. they are fp16 rounding ties. The sanity check is
therefore: every row where argmax z·W differs from the cached prediction has a top-2 logit gap (×100) below 0.05. Nothing
else changes; accuracies are computed from the replayed predictions (fp32), and the zero-shot reference in every gain is
the replayed zero-shot (same arithmetic), not the cached one.
