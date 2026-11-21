# Pre-registration: masked NIQE vector (2026-11-21, before any extraction or run)

**Question.** The whitened NIQE vector (λ = 0.1) adds +0.26 ± 0.04 alone and +0.25 ± 0.05 on top of T3 on ImageNet-C, but
not on R/A/Sketch, so it identifies the *corruption*, not quality. Does computing its statistics only over part of
the image make it a cleaner corruption descriptor (less image content leaking in) and so a better selection term?

**Method.** `masked_iqa.py`: MSCN on the whole image (no fill, no blur); every (A)GGD fit uses only kept pixels, and a
product of neighbours needs both pixels kept. Blocks/scales as NIQE (2 × 2 blocks of 96 px on the top-left 192 × 192
of CLIP's 224 view; a block with < 10% kept pixels is ignored). Check passed: the all-ones mask reproduces
`iqa.NIQE.features` exactly (max |diff| 0).

Variants (kept share of the 192 × 192 crop in brackets; "centre" = central box of the 224 view = the object, by the
ImageNet centring prior; "ring" = everything else = background):

| name | kept | role |
|---|---|---|
| `full` | 100% | baseline (= cached NIQE vector) |
| `centre96` / `centre136` | 25% / 50% | object only |
| `ring96` / `ring136` | 75% / 50% | background only |
| `randbox96` / `randbox136` | 25% / 50% | CONTROL for centre: same-size box at a random position per image |
| `randring96` / `randring136` | 75% / 50% | CONTROL for ring: complement of a random box, per image |

The controls keep exactly the same pixel count as their real mask, so "less content" cannot be confused with "fewer
pixels, noisier fits".

**Stage 1: offline screen** (no model; ImageNetC5K, seed-0 stream, 75,000 images; whole-stream PCA whitening as in
`nss_active_set.py simulate`).
- S1 content leakage: 10-NN (in the whitened vector) that are the *same photo under another corruption*, % (lower = cleaner).
- S2 10-NN with the same corruption, % (higher = better; full image ≈ 80%).
- S3 replayed selection (Ramen's exact queues, 180 snapshot batches in 3 groups of 60; topk 1, β 0, λ ∈ {0.03, 0.1, 0.3}):
  top-8 same-corruption weight share, % (Ramen 43.6, NIQE vector λ 0.1 ≈ 49.4).
  The 3 groups give the replay noise.

**Screen rule (a variant X "passes").** At X's best λ in the grid: share(X) − share(full) > 1.0 point **and**
share(X) − share(control of X) > 1.0 point, **both** larger than 2 × the range of the 3-group differences, **and**
S1(X) < S1(full). One pass = go to stage 2 for that variant. No pass = stop; the idea is a dead end on this benchmark
and I write that down. S3's share is a *correlate* of accuracy (NIQE adds +3.5 share points on top of T3 and keeps
its full gain), so it only gates stage 2; it is not the result.

**Stage 2: GPU runs** (only for passing variants). RamenNSS with the masked vector, `nss_features: niqe`, white,
`nss_lambda` = the screen's best λ, **plus T = 3**, ImageNetC5K, seeds 0–4, together with the same stack on the variant's
random control. Reference: the cached stack `nss_niqe_white_l0.1_probT3` (+0.87 ± 0.10 over Ramen).
- Primary claim: masked stack − full stack, paired over seeds, mean > +0.10 **and** positive on ≥ 4 of 5 seeds.
- Control claim: masked stack − control stack > 0 on the mean (otherwise the gain is not about *which* pixels).
- Also reported: gain over Ramen against the 0.53 noise floor (informational: the full stack already clears it).
- Sanity (smoke, 3,000 images): masked vector with the all-ones mask + T3 == `nss_niqe_white_l0.1_probT3`, 100% same predictions.

**Caveats written in advance.** Geometric "object" = centre box (no labels, no model); ImageNet objects often fill most
of the crop, so the ring can be thin. MSCN windows (7 × 7) bleed ~3 px across the mask edge. Seed 0 stream only in
stage 1; ImageNet-C only (NIQE did not generalise to R/A/Sketch; a masked version is not expected to change that and
is not tested there).
