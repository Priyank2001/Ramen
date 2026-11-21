# Pre-registration: NIQE vector (λ = 0.1) + RamenActive prob T = 3 on ImageNet-C, seeds 0–4

Written 2026-11-20, before any full run of the stack.

- **Variant:** RamenNSS, `nss_features: niqe`, `nss_norm: white`, `nss_lambda: 0.1`, `active_set: prob`,
  `temperature: 3.0`; everything else = cfg/ImageNetC5K/RamenNSS.yaml (= Ramen's ImageNet-C settings).
  λ from nss_active_set.md (fixed before its seed runs), T from active_seeds.py. Neither is re-tuned here.
- **Data:** ImageNetC5K, ViT-B/16, mixed stream, batch 100, seeds 0–4 (seed = stream order).
- **Primary claim:** mean gain over Ramen (domain-averaged) > noise floor = 2 × (max − min) of Ramen over seeds 0–4
  (0.53 per nss_active_set.md; recomputed in the analysis).
- **Secondary claim:** the stack beats T3 alone (+0.62 ± 0.11) on the mean and on ≥ 4 of 5 seeds paired.
- **Additivity reference:** T3 + NIQE alone = +0.62 + 0.26 = +0.88 if the gains add.
- Seed 0 of the scalar-score stack (+0.66 vs T3 +0.51) is the only prior evidence; it was a different regulariser.
