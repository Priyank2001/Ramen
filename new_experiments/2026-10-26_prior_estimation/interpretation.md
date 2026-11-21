**Decision 1: the right target is worth a lot (direction alive).** With the true class marginal as its target, T3 + pc beats T3 by **+2.75** on ImageNet-R/A/Sketch (floor 0.48, 5/5 seeds) and by +0.40 on DomainNet (floor 0.11). The post-hoc structure oracle, which knows only each domain's size and class set (not per-class counts), still gives **+2.32 ± 0.10** on R/A/Sketch. So most of that ceiling is "which classes occur at all", not exact counts.
- On ImageNet-C 5K the oracle's +1.96 is **not attainable**. The 5,000 images are a random draw (0 to 12 per class, 3 classes absent), while the population marginal is uniform, so that gain comes only from knowing the counts. On ImageNet-C and the CIFARs, uniform is already the right population target.

**Decision 2: the primary claim fails.** pc toward the zero-shot EM estimate (zs-EM) is negative on **all 9** dataset × base combinations: T3 on R/A/Sketch −0.40, on ImageNet-C −3.31, on DomainNet −1.86. On Ramen it ranges from −0.64 (R/A/Sketch) to **−5.29** (CIFAR-100-C), far below −floor on every dataset.

**Why the estimators fail.** No label-free target ends up closer to the truth than the method's own predictions already are.
- zs-EM's final target is the **farthest** of all from the true marginal: total-variation distance 0.66 on ImageNet-C vs 0.17 for uniform, and 0.51 vs 0.00 on CIFAR-100-C.
- EM on miscalibrated zero-shot CLIP is rich-get-richer. Each step reweights posteriors by the current prior, so mass piles onto the classes zero-shot already over-predicts (its sinks). pc then sees those classes as *under*-predicted relative to the target and boosts them, which is the opposite of what pc is for.
- On ImageNet-R/A/Sketch zs-EM gets the coarse shift right but the per-class detail wrong. Its target puts 52.0% of its mass on the 314 R/A classes (true 60.5%, uniform 31.4%; seed 0), and T3 + pc[zs-EM] gains on R and A (+0.30, +0.31). But it loses −1.82 on Sketch, because within the classes the mass sits on zero-shot's favourites.
- zs-marginal (undo only the drift adaptation adds) is ≈ 0 everywhere (−0.55 to +0.17). Zero-shot shares most of Ramen's bias, so correcting relative to it removes little.
- self-EM helps only on R/A/Sketch (+0.26 on T3; 5/5 positive but below the floor) and hurts everywhere else (−0.18 to −1.97).

**What this means.**
1. The failure of uniform-target pc on R/A/Sketch was a target problem, not a pc problem: with a good target pc is worth +2 to +3 points there.
2. That target cannot be recovered from CLIP's own outputs on one unlabelled stream. A label shift and a classifier bias look the same, and every estimator here inherits zero-shot's bias. Label-free prior estimation from model outputs is a **dead end** in this form.
3. The attainable part is **side information about the label set**: knowing which classes can occur gives most of the ceiling. That is exactly what the standard ImageNet-R/A protocol does by using a 200-class head. A method could use a known candidate class set, or a per-domain class set if domains were known, but that is a change of problem setting, not a TTA fix.
4. pc with the uniform target remains the right choice when the population is balanced (CIFAR, EuroSAT, ImageNet-C). It should be an option for known-balanced label spaces, not part of the general method. T3 alone stays the one general, held-out-confirmed change.

**Caveats.** Single λ = 1 and m = 0.99 throughout (frozen). EM with temperature calibration was not tried: it would need labelled data to fit the temperature. The structure oracle and the ImageNet-C count analysis are post-hoc. None of these datasets is held out.
