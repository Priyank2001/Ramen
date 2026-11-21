# Masked NIQE vector (2026-11-21): does computing NIQE statistics over part of the image help?

Follow-up to `../2026-11-20_nss_t3_stack/` (+0.25 ± 0.05 from the NIQE vector on top of T3, ImageNet-C only) and
`../2026-11-20_nss_t3_heldout/` (zero on R/A/Sketch). Pre-registered in [PREREGISTRATION.md](PREREGISTRATION.md).
**Verdict: null. The offline screen failed, so the GPU stage was not run.**

## Files

| File | Job |
|---|---|
| `PREREGISTRATION.md` | variants, screen rule and stage-2 claims, fixed before any extraction |
| `masked_iqa.py` | `PixelMask` classes (centre box, ring, random-box controls), `MaskedNIQEFeatures`, `MaskedImageQuality` |
| `test_masked_iqa.py` | checks: all-ones mask == `iqa.NIQE.features` (max diff 0); controls keep equal pixel counts |
| `extract.py` | masked NIQE vector for all 75,000 ImageNetC5K images × 9 variants → `results/ImageNetC5K_masked_niqe.npz` |
| `screen.py` | S1/S2 nearest-neighbour profile, S3 exact queue replay, the screen rule → `tables/screen_*.csv` |
| `logs/` | `extract.log`, `screen.log` |

Stage 2 (RamenNSS with a masked vector + T3, seeds 0–4) was specified but is not built: nothing passed the screen.

## Math

Baseline vector (`src/models/iqa.py`): for the 4 blocks $b$ of 96 px (2 × 2, top-left 192 × 192 of CLIP's 224 view) and
the 2 scales, $\phi_b \in \mathbb{R}^{18}$ per scale from AGGD fits to the MSCN coefficients $s$ and to their products
with 4 shifts; $v = \frac{1}{4}\sum_b [\phi_b^{(1)}; \phi_b^{(2)}] \in \mathbb{R}^{36}$. Selection (RamenNSS):
$s_{ij} = d_{ij} + \lambda\, \lVert A_t(v_i - v_j)\rVert_2/\sqrt{36}$, $A_t$ = causal PCA whitening, $\lambda = 0.1$.

Masked variant. Let $m \in \{0,1\}^{224 \times 224}$ be the keep-mask. MSCN $s$ is computed on the whole image. Every sample
moment in the fits runs over kept pixels only, e.g. for a block with kept set $\Omega_b = \{p \in b : m_p = 1\}$

$$\hat\rho_b = \frac{\big(\tfrac{1}{|\Omega_b|}\sum_{p\in\Omega_b}|s_p|\big)^2}{\tfrac{1}{|\Omega_b|}\sum_{p\in\Omega_b}s_p^2},$$

and for a shifted product $s_p s_{p-\delta}$ the kept set is $\{p : m_p = m_{p-\delta} = 1\}$. A block with fewer than 10% of its
pixels kept (at either scale) is dropped from the block mean. `full` ($m \equiv 1$) is the baseline exactly.

| Variant | $m$ | kept (of the crop) | code |
|---|---|---|---|
| object | central box, side 96 / 136 | 25% / 50% | `CentreBox` |
| background | complement of that box | 75% / 50% | `Complement(CentreBox)` |
| controls | same-size box / its complement at a random position per image | 25–75% (identical pixel count) | `RandomBox`, `Complement(RandomBox)` |

No term uses labels (the centre box is a geometric prior, not an object detector).

## Results (ImageNetC5K seed-0 stream, 75,000 images, 15 corruptions × 5,000)

**S1/S2: 10 nearest neighbours in the whitened vector** (chance same-corruption 6.7%):

| variant | kept | same corruption % | same photo, other corruption % | same label % |
|---|---|---|---|---|
| full | 100% | **80.65** | 0.44 | 0.71 |
| centre136 / randbox136 | 50% | 77.85 / 77.51 | 0.41 / 0.32 | 0.65 / 0.56 |
| ring136 / randring136 | 50% | 76.01 / 76.40 | 0.37 / 0.26 | 0.57 / 0.46 |
| centre96 / randbox96 | 25% | 73.47 / 71.90 | 0.39 / 0.16 | 0.59 / 0.34 |
| ring96 / randring96 | 75% | 79.20 / 78.92 | 0.41 / 0.31 | 0.64 / 0.55 |

**S3: replayed top-8 same-corruption share** (Ramen's exact queues, 180 batches = 18,000 queries in 3 groups; group
range about 1 point; λ = 0 gives 44.08%). Best λ per variant, difference in points:

| variant | best λ | share % | vs full | vs control | 2 × group range | passes |
|---|---|---|---|---|---|---|
| centre96 | 0.1 | 49.48 | −0.24 | +0.53 | 1.02 | no |
| centre136 | 0.3 | 50.76 | +0.48 | +0.83 | 1.57 | no |
| ring96 | 0.1 | 48.90 | −0.83 | −0.29 | 0.36 | no |
| ring136 | 0.1 | 47.67 | −2.05 | −0.58 | 1.03 | no |

Full image: 49.72% at λ = 0.1, 50.28% at λ = 0.3.

## Why it does not work, and what is not shown

- **Why it should help:** if image content leaked into the vector (a textured object looks like noise), dropping the
  object would give a cleaner corruption descriptor and so more same-corruption neighbours.
- **What happened:** the leakage was not there to remove. In the full vector only 0.44% of neighbours (about 1 in
  230) are the same photo under another corruption, so the ceiling for "less content" is tiny. Removing pixels
  *lowers* corruption identification (80.65 → 79.2 / 76.0 / 77.9 / 73.5% with 75 / 50 / 50 / 25% of pixels kept), and
  it does so about equally for the matched random-position controls (78.9 / 76.4 / 77.5 / 71.9%). The replay
  share follows the same order: background-only is 0.8 and 2.1 points *below* the full image, object-only is
  within the replay noise of it (−0.24, +0.48).
- **What this says:** what matters for the descriptor is how many pixels the fits see, not which. A masked vector
  is a noisier version of the same corruption signature.
- **Not shown:** that a real object mask (CLIP patch-text similarity, the plan's step 4) behaves differently. It
  was not run, because the plan only called for it if the geometric masks showed something; they showed nothing,
  and the leakage ceiling above argues against it. Also not shown: any accuracy effect (the screen's share is a
  correlate, and no GPU run was made); the NIQE vector's lack of generalisation to R/A/Sketch is untouched.

**Verdict: null (screen failed; masking removes signal, not content).** Next decision: NIQE on ImageNet-C stays as it
was (+0.25 over T3, corruption-specific).
