# Pre-registration: shift decomposition with oracles (2026-11-22)

Written before any number of this experiment was computed.

## Question

On each dataset, how much accuracy is available from fixing (A) a class-independent domain offset ("appearance"),
(Γ) a class-specific domain offset ("concept"), and (L) the per-domain class frequencies ("label shift")? Which of
these is larger than Ramen's own gain over zero-shot? The answer picks the design of the next method (Step 2).

All three fixes use TRUE labels and/or TRUE domain ids: every number here is an **ORACLE ceiling**, not a method.

## Data (seed 0 caches, offline, CPU only)

Frozen CLIP image features `results/<ds>_..._s0_NoAdapt_feats.npy` (fp16, 512-d), labels, domain ids, Ramen's
cached predictions, and the class text embeddings used by Ramen (`2026-11-21_hubness/results/<ds>_text_features.npy`).

| dataset | images | classes | domains | Ramen gain over NoAdapt (domain-avg) | floor |
|---|---|---|---|---|---|
| ImageNetC5K | 75,000 | 1,000 (997 present) | 15 corruptions | +5.79 | 0.54 |
| DomainNet | 596,006 | 345 | 6 | +1.43 | 0.11 |
| ImageNetVariants (R/A/Sketch) | 88,389 | 1,000-class head | 3 | +1.50 | 0.48 |
| EuroSAT | 27,000 | 10 | 1 | +14.56 | 1.51 |

## Model

Per-image embedding model (unbalanced two-way MANOVA, fitted by weighted least squares on cell sums):
z_i = μ + α_{d_i} + β_{y_i} + γ_{y_i,d_i} + ε_i, with Σ_d n_d α_d = 0, Σ_c n_c β_c = 0; the additive part (μ, α, β) is
the least-squares fit without interaction; γ = cell mean − additive fit.

## Part A: geometry

SS_A = Σ_d n_d‖α_d‖², SS_B = Σ_c n_c‖β_c‖², SS_Γ = Σ_{c,d} n_{c,d}‖γ_{c,d}‖², SS_W = Σ_i‖z_i − M̂_{y_i,d_i}‖².
Per-dimension σ² = SS_W / (D·(N − #cells)). Debiased SS_X* = SS_X − df_X·D·σ² with df_A = |D|−1, df_B = C−1,
df_Γ = #cells − C − |D| + 1. Report F_X = SS_X/(D·df_X·σ²) and shares SS_X*/(ΣSS*+SS_W).
Split-half reliability of γ (groups split at random, Spearman–Brown corrected).

## Part B: oracle classifiers

Image world (isotropic Gaussian Bayes = nearest class mean), S ⊆ {A, Γ, L}, 8 cells:
score_c(z_i) = κ_S·( z̃_iᵀ m_{c,d_i} − ‖m_{c,d_i}‖²/2 ) + 1[L∈S]·log(C·π̂_{c|d_i}),
z̃_i = z_i − 1[A∈S]·α̂_{d_i},  m_{c,d} = μ̂ + β̂_c + 1[Γ∈S]·γ̃_{c,d}.
Text world: T (zero-shot, τ·zᵀw_c), T+A (z' = norm(z − α̂_d)), T+L, T+A+L.

Honesty rules (fixed in advance):
1. Leave-one-out: the true-class prototype of image i is recomputed without i; on ImageNet-C without all 15 corrupted
   copies of i's photo (group = `sample_idx`). μ̂ and α̂ are not recomputed (influence ≤ 1/n_d; documented).
   Priors: π̂^{(−i)}_{c|d} = (n_{c,d} − 1[y_i=c] + 1)/(n_d − 1 + C).
2. Interaction shrinkage γ̃ = n/(n+n₀)·γ̂ with n₀ = σ²/τ̂², τ̂² = (SS_Γ/D − df_Γ σ²)/N (method of moments, no test-label
   tuning). n₀ = ∞ (Γ off) if τ̂² ≤ 0.
3. Temperatures κ_S, τ fitted by NLL (temperature scaling) on a 20,000-image subsample, per prototype configuration,
   before adding L.
4. Accuracy = domain-averaged (mean of per-domain accuracies).
5. CIs: cluster bootstrap (B = 1000, Poisson weights; cluster = photo on ImageNet-C, image elsewhere) with estimates
   held fixed. Estimation noise is checked separately by a 2-fold cross-fit (prototypes from the other half of the groups).

Attribution: Shapley values over {A, Γ, L} (image world) and over {A, L} (text world); text-prototype gap
= Acc(P0) − Acc(T).

Controls: random domain labels (same domain sizes) replace the true ones for A and Γ; they must give ≈ 0 over P0.
Unshrunk Γ (n₀ = 0) and label-free-given-domain A (α = image-weighted domain mean − grand mean) reported as extra rows.

## Sanity checks (must pass before results are read)

- T reproduces cached NoAdapt predictions on ≥ 99.4% of images (fp16 features).
- EuroSAT (one domain): A and Γ cells give predictions identical to P0 (α = γ = 0 by construction).

## Decision rules

- Component k "dominates" a dataset if φ_k ≥ 0.5·Σφ (image world) and its 95% CI excludes 0.
- Domain-aware direction (Step 2) is alive on a dataset iff φ_A + φ_Γ (image world) > Ramen's gain over NoAdapt.
  Then: Γ dominates → per-domain prototypes; A dominates → per-domain centring; L dominates → prior side information.

## Predictions

- ImageNetC5K: A dominates.
- DomainNet: Γ dominates (largest in sketch / quickdraw).
- ImageNetVariants: L dominates (R and A hold 200-class subsets of the 1,000-class head).
- EuroSAT: φ_A = φ_Γ = 0 exactly (bug check).

## Known limits

Means only (covariance / multimodality changes are missed, so Γ is a lower bound on concept shift); "domain" = the
dataset's domain labels; Euclidean geometry on unit vectors; one feature extractor per dataset (ViT-B/16 or B/32).
