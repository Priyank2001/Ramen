**What the numbers say**

- **Both pre-registered claims pass.** NIQE + T3 gains **+0.87 ± 0.10** over Ramen (seeds 0–4, all 5 positive, paired t 20.1). The mean clears the 0.53 noise floor, and so does every single seed (smallest +0.76). Each seed's photo-clustered 95% CI excludes 0 (lowest lower bound +0.56). This is the first ImageNet-C variant whose every seed is above the floor; T3 alone (+0.62) cleared it only on the mean.
- **The stack beats T3 alone on 5 of 5 seeds**, by **+0.25 ± 0.05**. Every seed's CI vs T3 excludes 0 (+0.04 to +0.42 at the extremes). In images: about 1,300 fixed vs about 1,130 broken relative to T3 per seed. The net gain of ~190 images out of 75,000 is small compared with the churn.
- **The gains add almost exactly.** T3 alone +0.62 plus NIQE alone +0.26 predicts +0.88; the stack gives +0.87. This fits the design: T3 reweights *which class queues* count, NIQE changes *which entry within a queue* is picked, and the two do not appear to interfere.

**Where the extra gain comes from (per corruption, mean over seeds)**

- NIQE fills in where T3 hurts. T3 loses to Ramen on elastic (−0.52), jpeg (−0.48) and contrast (−0.09), and gains little on fog (+0.02) and glass blur (+0.24). The stack gives −0.03, −0.08, +0.45, +0.47 and +0.94 on these. These are the structured, non-additive distortions where NIQE alone helps most (glass blur +0.90, elastic +0.44, contrast +0.42).
- On the noise corruptions, where T3's gain is already large, the stack adds little (shot +2.10 → +2.14, impulse +0.88 → +0.89).
- The stack now loses to Ramen on only 2 of 15 corruptions, both within 0.1 points (elastic −0.03, jpeg −0.08).

**Mechanism: partly measured**

- Measured: the weight share of the top-8 retrieved entries that share the query's corruption rises from 59.0% (T3) to 62.5% (stack). NIQE alone gives 49.4%; Ramen was 43.6% in `nss_active_set.md`.
- Not established: that this share *causes* the gain. NIQE adds only +3.5 points of same-corruption share on top of T3 (vs +5.8 on top of Ramen), yet its full accuracy gain carries over. Something beyond the corruption share may matter, for example *which* same-corruption entry gets picked. Treat the share as a correlate, not the mechanism.

**Caveats**

- **Not held-out.** Both settings were picked on ImageNet-C: λ = 0.1 from the seed-0 replay and real runs, T = 3 from the seed-0 sweep. This run confirms the stack across seeds on the dataset that set its settings, so it is not yet a generalisation claim.
- **Cost.** About 17–20 min per run vs ~7 for Ramen, mostly from RamenQuality's per-class rank work, which is unused when `quality_lambda = 0`, plus NIQE feature extraction.

**Next step**

Pre-registered held-out test with both settings frozen. ImageNet-R/A/Sketch is the natural first target: T3 already holds there (+0.59 ± 0.04), and the loader and cached Ramen/T3 runs exist. Unknown: whether NIQE features mean anything on renditions and sketches; flat sketch backgrounds hit the same MSCN degeneracy as quickdraw. DomainNet needs the offline replay extended to β = 5 first. A failure on R/A/Sketch would say the NIQE part only helps when the domain shift is a pixel-level corruption.
