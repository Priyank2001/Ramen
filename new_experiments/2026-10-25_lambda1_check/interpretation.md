**λ = 1 passes the pre-registered rule on all three datasets.** It becomes the single prior-correction setting for every dataset and goes to the held-out test unchanged.

| | λ = 2 (tuned here) | λ = 1 (frozen) | kept | floor |
|---|---|---|---|---|
| CIFAR-10-C | +0.57 ± 0.05 | +0.35 ± 0.04 (4/5 seeds above floor) | 62% | 0.31 |
| CIFAR-100-C | +1.50 ± 0.03 | +1.15 ± 0.03 (5/5) | 76% | 0.21 |
| EuroSAT | +2.79 ± 0.28 | +2.13 ± 0.16 (5/5) | 76% | 1.51 |

- **λ = 1 costs something on the small datasets, but much less than λ = 2 costs on the large ones.** It gives up 0.2–0.7 pts here, while λ = 2 instead of 1 loses 0.4 on ImageNet-C (T3 + pc +0.72 vs +1.13) and 0.6 on DomainNet (−0.19 vs +0.41). One setting across five datasets is worth that trade.
- **CIFAR-10-C is the thin margin.** +0.35 clears the 0.31 floor on the mean, but seed min +0.30 does not. It passes the rule as written, but just barely.
- **With kNN, λ = 1 holds up on CIFAR:** knn + pc(λ=1) +1.15 (C10) and +2.24 (C100, higher than λ = 2's +2.07). EuroSAT drops from +6.99 to +5.40. kNN is still not part of the frozen setting, because it hurts on ImageNet-C.
- **λ = 0.5 is too weak on the small datasets:** it falls below the floor on CIFAR-10-C (+0.19) and EuroSAT (+1.20). So the useful range of a single λ is about 1, not "the smaller the better".

**Caveats.** λ = 1 was chosen on ImageNet-C / DomainNet results, and this check only shows it does not break the datasets λ = 2 was tuned on. It is not a held-out confirmation: all five datasets have now been looked at. The 0.03 EuroSAT sanity gap is unexplained (section 3) and does not affect the comparison.

**Next step.** The held-out test (#3): T3 + pc(λ = 1), no kNN, on a dataset none of this was tuned on (ImageNet-R / -A / -Sketch). Step #2, estimating the class prior instead of assuming it uniform, tunes on ImageNet-C / DomainNet again. Run it after the held-out data and settings are frozen, or keep its winner out of the held-out claim.
