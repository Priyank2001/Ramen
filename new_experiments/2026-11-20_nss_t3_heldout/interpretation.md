**What the numbers say**

- **The primary claim passes, but only because of T3.** NIQE + T3 gains **+0.61 ± 0.05** over Ramen (5/5 seeds positive, every seed's CI excludes 0, smallest gain +0.53 > floor 0.48). T3 alone gives +0.59 ± 0.04 on the same streams.
- **The secondary claim fails.** The stack minus T3 is **+0.01 ± 0.06**, positive on 3 of 5 seeds, and every seed's 95% CI includes 0 (−0.20 to +0.20 at the extremes). Per seed, about 990 images are fixed and about 990 broken relative to T3, out of 88,389: the NIQE term moves predictions but with no net effect.
- **NIQE alone does nothing here:** **+0.03 ± 0.09**, 2 of 5 seeds positive, every CI includes 0 (seed 4 touches it at 0.00).
- This is the outcome written down in advance as the alternative: NIQE alone ≈ 0 and the stack ≈ T3. On ImageNet-C the same settings gave NIQE alone +0.26 ± 0.04 and stack vs T3 +0.25 ± 0.05.

**Per domain (mean over seeds)**

- Adversarial: T3 +1.37, stack +1.41, NIQE alone +0.12. Rendition: +0.06 / +0.08 / −0.01. Sketch: +0.36 / +0.33 / −0.01.
- No domain shows a NIQE contribution beyond seed noise.

**Why, as far as the logs go**

- **Measured:** the same-domain share of the top-8 retrieved weight is already 78.6% under T3 and stays 78.5% with NIQE added. NIQE alone reaches 58.5%. Unlike on ImageNet-C (T3 59.0% → stack 62.5%), NIQE does not change *where* the support comes from on top of T3.
- **Hypothesis, not measured:** these domains differ in content and style (renditions, sketches, hard natural photos), not in pixel-level distortion, so NIQE's scene statistics carry little domain information beyond CLIP's own features. Also, T3 alone already concentrates the support in the query's domain here. Ramen's own same-domain share on this stream was not logged, so how much room NIQE alone had is not known.
- The flat-sketch concern is ruled out as the cause: the NIQE vector is undefined on only 0.4% of sketch images (smoke subset).

**What this means for the method**

- The ImageNet-C result (+0.87, every seed above the floor) stands, but **NIQE matching is a corruption-specific add-on, not a general improvement**. The general, held-out component is T3: +0.62 ImageNet-C, +0.59 R/A/Sketch, +0.24 DomainNet.
- Claim to write: "NIQE-vector support matching adds +0.25 on top of T3 when the test shift is a low-level image corruption (ImageNet-C), and nothing on rendition, sketch or adversarial shift."

**Caveats**

- One held-out dataset. DomainNet (style domains, β = 5) was not run. These results predict a null there, but the replay would need extending to β > 0 first.
- T = 3 was already tested on this stream, so only λ and the stack are held-out here.

**Next step**

- The natural place where NIQE *should* help is another corruption benchmark. CIFAR-100-C is the candidate, testing NIQE alone, because T3 hurts there (−0.15). But NIQE on 32 px images upsampled to 224 mostly measures the upsampling, so a null there would be uninformative.
- Otherwise, stop investing in NIQE as a domain descriptor and report it as corruption-specific.
