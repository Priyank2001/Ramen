# Pre-registration: held-out test of NIQE vector + T3 on ImageNet-R / -A / -Sketch

Written 2026-11-20, before any NIQE-based run or NIQE feature extraction on this dataset.

## Frozen method (no tuning on this dataset)
RamenNSS with `nss_features: niqe`, `nss_norm: white`, `nss_lambda: 0.1`, `active_set: prob`, `temperature: 3.0`.
Every other key is `cfg/ImageNetC5K/RamenNSS.yaml`; its Ramen part (capacity 75, topk 1, beta 0, signsgd, lr 0.01)
is identical to `cfg/ImageNetVariants/Ramen.yaml`. λ and T were chosen on ImageNet-C only
(`../2026-11-20_nss_t3_stack/`: +0.87 ± 0.10 there, every seed above the floor).

## Data
`ImageNetVariants`: ImageNet-R 30,000 + ImageNet-A 7,500 + ImageNet-Sketch 50,889 = 88,389 images, one 1,000-class
head, CLIP ViT-B/16, mixed stream, batch 100, seeds 0–4 (seed = stream order). This stream was already used for the
T3 held-out test (`../2026-10-25_heldout/`), so it is held-out for λ and for the NIQE stack, not for T = 3.

## Claims
- **Primary:** mean gain of NIQE + T3 over Ramen (domain-averaged over the 3 domains, seeds 0–4) > noise floor =
  2 × (max − min) of Ramen over seeds 0–4 (0.48 from the earlier held-out test; recomputed).
- **Secondary:** NIQE + T3 beats T3 alone (+0.59 ± 0.04 here) on the mean and on ≥ 4 of 5 seeds.
- **Decomposition (descriptive, no pass/fail):** NIQE alone (λ = 0.1, no T3), seeds 0–4.

## Predictions written down in advance
- If NIQE helps only because ImageNet-C's domains are pixel-level corruptions, NIQE alone ≈ 0 here and the stack ≈ T3.
- Sketch images (white backgrounds) may get undefined NIQE vectors (all blocks flat). RamenNSS replaces those by the
  stream mean. The share of such images per domain is a label-free diagnostic, reported but not used to change anything.

## What would not change the settings
Any diagnostic, smoke result or seed-0 result. The full 5-seed set is run and reported whatever seed 0 shows.
