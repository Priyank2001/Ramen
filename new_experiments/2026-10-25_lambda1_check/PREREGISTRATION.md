# Pre-registration: prior correction at λ = 1 (written 2026-10-25 22:09, before computing any λ = 1 number on these datasets)

**Context.** On ImageNet-C 5K and DomainNet (seeds 0–4, `../2026-10-25_stacking/`), prior correction at the pre-registered λ = 2 failed on DomainNet. λ = 1, chosen post-hoc, made T3 + pc clear both noise floors (+1.13 / +0.41). This file fixes the setting **before** (a) checking it on the small-label datasets where λ = 2 was tuned, and (b) any further tuning on ImageNet-C / DomainNet.

## The setting (frozen)
Prior correction exactly as in `marginal_correction.corrected_probs`: soft marginal, EMA momentum m = 0.99, the current batch included, uniform target, **λ = 1**. No kNN smoothing.
- Large label spaces (≥ 300 classes): applied to the zero-shot-weighted active set, RamenActive `active_set: prob`, `temperature: 3` (T3 + pc(λ=1)).
- Small label spaces (≤ 100 classes): applied to Ramen (T3 is neutral-to-harmful there: CIFAR-100-C −0.15).

## This check (step #1)
Datasets: CIFAR-10-C, CIFAR-100-C, EuroSAT. Seeds 0–4, using cached Ramen probabilities (offline, exact). Metric: domain-averaged accuracy, gain over Ramen.
Reported for each dataset: Ramen + pc at λ ∈ {2, 1, 0.5}, and knn + pc at λ ∈ {2, 1} for context (knn + pc(λ=2) was the earlier best on these datasets).
Sanity check: λ = 2 must reproduce the earlier seed means: pc +0.57 / +1.50 / +2.79, knn + pc +1.25 / +2.07 / +7.02 (C10 / C100 / EuroSAT).

## Decision rule (fixed now)
λ = 1 **passes** on a dataset if its mean gain over Ramen is (i) above that dataset's noise floor (C10 0.31, C100 0.21, EuroSAT 1.51) **and** (ii) at least 50% of the λ = 2 mean gain.
- Passes on all three → λ = 1 is the single setting for every dataset; go to the held-out test (#3) with it unchanged.
- Fails on any → a single λ does not transfer. The held-out test then needs a rule for λ that is fixed before it runs (e.g. a function of the number of classes), and any such rule is post-hoc with respect to these five datasets.
