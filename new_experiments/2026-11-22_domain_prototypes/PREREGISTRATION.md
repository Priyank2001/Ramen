# Pre-registration: Step 2, label-free domain-keyed prototypes (2026-11-22)

Written before any code or number of this experiment. Builds on `2026-11-22_shift_decomposition` (Step 1).

## Why this experiment

Step 1 (oracle) found a large class-specific domain component ("concept", γ) on DomainNet (Shapley +5.25, quickdraw +20.5)
and ImageNet-R/A/Sketch (+3.46), but not on ImageNet-C (+0.19). Per-domain class means with TRUE labels beat zero-shot by
**+2.76** on DomainNet (`img_AG − txt_`), about 2× Ramen's gain (+1.43). Centring by plain domain means (no labels) already
gives +0.85, so the domain id is not the bottleneck; label-free per-domain *class* prototypes are.

**Question.** Can a fully label-free, causal (streaming) classifier that keeps class prototypes **per inferred domain**
recover a useful part of that +2.76, and does keying by domain add anything over the same prototypes kept globally?

**What is already known not to work** (so this design avoids it): zero-shot pseudo-labels as CE targets (2026-11-04), EM /
GMM from text prototypes (drifts to domains, DomainNet −11.1, 2026-11-18), label-free prior estimation (2026-10-26). Here
pseudo-labels only *select* confident images into bounded per-class caches (no EM iteration, no gradient), and the image
prototype is fused with the text classifier, as in Tip-Adapter / TDA / DMN [1–3]. The new part is the domain key.

## Setup

Seed-0 stream order and batches of 100 (cached `Ramen.pkl` row order), cached frozen CLIP features `z_t` (fp16, unit norm),
text embeddings `w_c`. Offline, CPU only (16 threads). Every quantity used for batch t is computed from batches ≤ t
(the current batch is written before it is classified, as Ramen does).

### Domain inference (label-free)
Assignment g(i) ∈ {1..K} by online mini-batch k-means [4] on a domain feature u_i:
- `u = z` (final embedding), or
- `u = r = (I − P)z / ‖(I − P)z‖`, P = projector onto the top-k principal directions of {w_c} (k = min(256, C)),
  i.e. the part of the embedding orthogonal to the main class directions (a style proxy from the frozen feature).

### Classifier (variant V3, the proposal)
1. **Centre** by the cluster's running mean: z'_i = norm(z_i − (μ̄_{g(i)} − μ̄)), μ̄_g running mean of cluster g, μ̄ of all.
2. **Pseudo-label** with zero-shot on z': ŷ_i = argmax_c z'_iᵀw_c, confidence p_i = max softmax(100 z'_iᵀW).
3. **Cache** z'_i in queue Q_{ŷ_i, g(i)} (capacity M); when full, evict the lowest-confidence entry if p_i is higher.
4. **Prototype** with shrinkage toward the pooled (all-cluster) prototype of the class:
   m_{c,g} = norm( Σ_{j∈Q_{c,g}} z'_j + n₀·m̄_c ),  m̄_c = norm( Σ_g Σ_{j∈Q_{c,g}} z'_j ) (m̄_c = w_c if no entries).
5. **Predict** with text + prototype fusion (both cosine, same 100× scale):
   ŷ_i^{V3} = argmax_c [ z'_iᵀw_c + λ·z'_iᵀm_{c,g(i)} ].

## Variants (all label-free unless marked ORACLE)

| name | domain key | centring | prototypes | role |
|---|---|---|---|---|
| V0 | — | no | none (zero-shot) | baseline |
| V1 | inferred | yes | none | appearance only (label-free) |
| V2 | **one global cluster (K = 1)** | no | global caches, same M, n₀, λ | **TDA-like control: is the domain key needed?** |
| V3 | inferred | yes | per (class, cluster) caches | proposal |
| V3-rand | random clusters, same sizes as V3 | yes | per (class, cluster) | control: any split vs a domain split |
| V3-dom | **ORACLE true domain** | yes | per (class, domain) | domain-inference ceiling |
| V3-clean | inferred | yes | caches admit only correct pseudo-labels (**ORACLE labels**) | label-noise cost |
| Step-1 ceiling | ORACLE domain | yes | true-label class means, non-causal | +2.76 DomainNet |
| Ramen | — | — | — | cached reference (+1.43 DomainNet) |

## Tuning (DomainNet only) and freezing

DomainNet is the development set. Grid (16 configs): domain feature {z, r} × K {6, 20} × M {8, 32} × λ {0.5, 1}; n₀ = 4
fixed. The best V3 config by DomainNet domain-averaged accuracy is picked once; V2 / V3-rand / V3-dom / V3-clean use the same
M, n₀, λ (V2 gets its own best λ, M from the same grid, so the control is not handicapped). That config is then **frozen** for
ImageNet-R/A/Sketch (held-out), ImageNet-C and EuroSAT. All DomainNet numbers are post-hoc best-of-grid; held-out numbers are not.

## Hypotheses and pass criteria (domain-averaged accuracy, seed 0 screen)

- **H1 (DomainNet, primary).** V3 − V2 > 0.11 (noise floor) **and** V3 − V0 > 0.11: domain keying adds accuracy label-free.
- **H2 (DomainNet, size).** Recovery R = (V3 − V0) / 2.76. R ≥ 30% (≥ +0.83) = promising; R < 10% (< +0.28) = dead; in
  between = weak.
- **H3 (held-out, frozen config).** ImageNet-R/A/Sketch: V3 − V2 > 0.48 and V3 − V0 > 0.48. ImageNet-C: V3 ≥ V0 − 0.54
  (no harm where Step 1 found no concept component). EuroSAT (one domain): reported only; V3 should ≈ V2 there.
- **Diagnostics (no pass/fail).** Domain recovery of the clusters vs true domains (NMI [5], purity) for u = z and u = r;
  V3-dom − V3 (cost of inferring the domain); V3-clean − V3 (cost of pseudo-label noise); per-domain gains (quickdraw);
  cache composition (pseudo-label accuracy of cached entries, counts and %).

## Decisions

- H1 and H2 (≥ 30%) pass on DomainNet **and** H3 passes on R/A/Sketch → run seeds 1–4 on DomainNet and R/A/Sketch, then test
  stacking on Ramen's adapted features.
- V3-dom ≫ V3 (gap > half of V3-dom − V0) → domain inference is the bottleneck: Step 2b extracts intermediate-layer style
  statistics on the GPU (token mean/std) and re-runs the domain feature only.
- V3-clean ≫ V3 with V3-dom ≈ V3 → pseudo-label noise is the bottleneck: label-free prototypes are dead in this form;
  report, do not tune further.
- V3 ≈ V2 (H1 fails) while V2 − V0 is large → the gain is TDA-like global caching, not domain keying; the domain story
  is not supported label-free (report as such; V2 itself is prior art [2, 3], not our contribution).
- V3-rand ≈ V3 → splitting helps for a reason other than domain; H1 counts as failed.

## Sanity checks (before results are read)

- V0 reproduces cached NoAdapt predictions on ≥ 99.4% of images (fp16 features, as in Step 1).
- λ = 0 with no centring reproduces V0 exactly (100.00%); λ = 0 with centring and true domains reproduces a causal
  version of Step 1's `txt_Ad` (reported next to it).
- Causality: shuffling the order of future batches leaves every prediction of batch t unchanged (checked on a 2,000-image prefix).

## Reporting

Counts with percentages; domain-averaged accuracy; dot plot of each variant's gain over V0 with the floor and Ramen as
reference lines; per-domain table; cache composition over the stream. Notebook + README per `ramen-experiment`.

## Cost

CPU only, ~25 runs on DomainNet (grid + controls) at a few minutes each, plus 3 frozen-config runs × 6 variants on the
other datasets; estimated 1–2 h wall-clock at 16 threads. No GPU unless the Step 2b branch triggers.

## Known limits

Seed 0 only at the screen (order matters for causal caches); DomainNet tuning is post-hoc; the domain feature uses only the
frozen final embedding; cosine fusion with fixed 100× scale is one choice among several; "domain" for the oracle = dataset
labels.

## References

1. Zhang et al., *Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification*, ECCV 2022.
2. Karmanov et al., *Efficient Test-Time Adaptation of Vision-Language Models* (TDA), CVPR 2024.
3. Zhang et al., *Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models* (DMN), CVPR 2024.
4. Sculley, *Web-Scale K-Means Clustering*, WWW 2010 (mini-batch k-means).
5. Strehl & Ghosh, *Cluster Ensembles — A Knowledge Reuse Framework for Combining Multiple Partitions*, JMLR 2002 (NMI).
