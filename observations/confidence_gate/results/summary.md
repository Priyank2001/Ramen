# gate_analysis.py output (seed 0)

Generated; do not edit by hand.

## CIFAR10C: confidence gate sweep (gain = (broken saved - fixed lost) / N, checked)

| tau | gated % | gated samples that disagree % | NoAdapt acc in gate | Ramen acc in gate | total acc | broken saved | fixed lost | gain (pts) |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 55.04 | 8.75 | 83.23 | 86.01 | 71.08 | 1,986 | 4,283 | -1.53 |
| 0.60 | 46.41 | 5.20 | 87.50 | 88.83 | 71.99 | 1,111 | 2,041 | -0.62 |
| 0.70 | 38.99 | 2.98 | 90.98 | 91.48 | 72.42 | 625 | 914 | -0.19 |
| 0.80 | 31.64 | 1.71 | 93.99 | 94.00 | 72.61 | 359 | 366 | -0.00 |
| 0.90 | 22.10 | 0.83 | 97.15 | 96.89 | 72.67 | 170 | 83 | 0.06 |
| 0.95 | 14.25 | 0.52 | 98.73 | 98.43 | 72.65 | 83 | 19 | 0.04 |
| 0.99 | 2.50 | 0.21 | 99.57 | 99.39 | 72.62 | 7 | 0 | 0.00 |

## CIFAR10C: per-sample outcomes

| outcome | samples | % of stream |
|---|---|---|
| fixed | 25,381.00 | 16.92 |
| broken | 8,498.00 | 5.67 |
| both right | 83,535.00 | 55.69 |
| both wrong, disagree | 14,586.00 | 9.72 |
| both wrong, agree | 18,000.00 | 12.00 |
| Ramen acc / ceiling | nan | 72.61 |
| ceiling (zs or Ramen right) | nan | 78.28 |
| P(zs right | exactly one right) | nan | 25.08 |

## CIFAR10C: zero-shot confidence by outcome (quartiles)

| outcome | count | 25% | 50% | 75% |
|---|---|---|---|---|
| both right | 83,535.000 | 0.559 | 0.825 | 0.951 |
| both wrong, agree | 18,000.000 | 0.331 | 0.484 | 0.683 |
| both wrong, disagree | 14,586.000 | 0.228 | 0.275 | 0.349 |
| broken | 8,498.000 | 0.269 | 0.350 | 0.488 |
| fixed | 25,381.000 | 0.247 | 0.314 | 0.435 |

## CIFAR10C: how much of the harm a gate at tau can reach

| tau | % of all broken with conf >= tau | % of all fixed with conf >= tau | broken / (broken + fixed) in gate % |
|---|---|---|---|
| 0.5 | 23.4 | 16.9 | 31.7 |
| 0.6 | 13.1 | 8.0 | 35.2 |
| 0.7 | 7.4 | 3.6 | 40.6 |
| 0.8 | 4.2 | 1.4 | 49.5 |
| 0.9 | 2.0 | 0.3 | 67.2 |
| 0.9 | 1.0 | 0.1 | 81.4 |
| 1.0 | 0.1 | 0.0 | 100.0 |

## CIFAR10C: broken and fixed per zero-shot confidence bin

| conf_zs | broken | fixed | all samples | broken / (broken + fixed) % | net if this bin kept zero-shot |
|---|---|---|---|---|---|
| (-0.001, 0.1] | 0 | 0 | 0 | nan | 0 |
| (0.1, 0.2] | 365 | 1,823 | 4,435 | 16.7 | -1,458 |
| (0.2, 0.3] | 2,649 | 9,734 | 26,715 | 21.4 | -7,085 |
| (0.3, 0.4] | 2,176 | 6,081 | 21,007 | 26.4 | -3,905 |
| (0.4, 0.5] | 1,322 | 3,460 | 15,282 | 27.6 | -2,138 |
| (0.5, 0.6] | 875 | 2,242 | 12,943 | 28.1 | -1,367 |
| (0.6, 0.7] | 486 | 1,127 | 11,128 | 30.1 | -641 |
| (0.7, 0.8] | 266 | 548 | 11,025 | 32.7 | -282 |
| (0.8, 0.9] | 189 | 283 | 14,320 | 40.0 | -94 |
| (0.9, 1.0] | 170 | 83 | 33,145 | 67.2 | 87 |

## CIFAR10C: separating broken from fixed: AUROC and best achievable gate

| signal (high = keep zero-shot) | AUROC broken vs fixed | AUROC doc def. (zs right, all disagreements) | best gate gain (pts, tuned on test) | disagreements switched |
|---|---|---|---|---|
| zero-shot conf | 0.571 | 0.609 | 0.059 | 265 |
| - zero-shot entropy | 0.570 | 0.609 | 0.057 | 216 |
| - Ramen conf | 0.705 | 0.605 | 0.039 | 2,025 |
| zero-shot conf - Ramen conf | 0.720 | 0.659 | 0.103 | 2,274 |
| overshoot of Ramen class (whole stream) | 0.568 | 0.550 | 0.000 | 0 |
| overshoot of Ramen class (running, causal) | 0.573 | 0.556 | 0.004 | 41 |
| overshoot Ramen class - zs class (running) | 0.599 | 0.586 | 0.001 | 1 |
| kNN pseudo-label vote, causal (k=20) | 0.636 | nan | 0.129 | 521 |
| kNN TRUE-label vote (oracle, k=20) | 0.794 | nan | 1.059 | 4,775 |

## CIFAR10C: zero-shot feature neighbourhood (k=20, true labels, own image excluded)

| outcome | neighbours truly of zero-shot class % | neighbours truly of Ramen's class % | % where Ramen's class has more neighbours |
|---|---|---|---|
| broken (zs class = true) | 38.9 | 20.1 | 27.7 |
| fixed (Ramen's class = true) | 17.5 | 37.8 | 71.8 |

## CIFAR10C: gain a gate can reach vs signal AUROC (8,498 broken vs 25,381 fixed)

| AUROC | max gain (pts), binormal | max gain at balanced base rate (pts) |
|---|---|---|
| 0.60 | 0.00 | 0.81 |
| 0.65 | 0.04 | 1.22 |
| 0.70 | 0.21 | 1.64 |
| 0.75 | 0.54 | 2.08 |
| 0.80 | 1.01 | 2.54 |
| 0.85 | 1.63 | 3.04 |
| 0.90 | 2.40 | 3.60 |
| 0.95 | 3.42 | 4.28 |
| 0.99 | 4.72 | 5.10 |
| ceiling (all broken saved) | 5.67 | 5.67 |

## CIFAR10C: which class Ramen moves samples into (whole-stream marginal)

| outcome | Ramen's class over-predicted % | median overshoot of Ramen's class % |
|---|---|---|
| broken | 59.1 | 9.8 |
| fixed | 54.4 | 1.2 |

## CIFAR10C: per target class (Spearman rho = 0.60, p = 6.7e-02)

| class | Ramen overshoot % | broken into class | fixed into class | broken / (broken + fixed) % |
|---|---|---|---|---|
| deer | 27.7 | 2,437 | 4,654 | 34.4 |
| horse | 15.2 | 936 | 2,140 | 30.4 |
| truck | 9.8 | 1,210 | 4,992 | 19.5 |
| automobile | 1.2 | 438 | 2,027 | 17.8 |
| dog | -0.5 | 908 | 2,252 | 28.7 |
| bird | -2.3 | 627 | 1,225 | 33.9 |
| ship | -3.5 | 412 | 2,165 | 16.0 |
| cat | -7.0 | 854 | 2,402 | 26.2 |
| airplane | -10.6 | 290 | 1,849 | 13.6 |
| frog | -30.1 | 386 | 1,675 | 18.7 |

## CIFAR100C: confidence gate sweep (gain = (broken saved - fixed lost) / N, checked)

| tau | gated % | gated samples that disagree % | NoAdapt acc in gate | Ramen acc in gate | total acc | broken saved | fixed lost | gain (pts) |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 23.49 | 3.97 | 78.96 | 79.97 | 45.86 | 352 | 708 | -0.24 |
| 0.60 | 18.32 | 1.80 | 83.52 | 83.89 | 46.03 | 137 | 240 | -0.07 |
| 0.70 | 14.17 | 0.84 | 87.37 | 87.42 | 46.09 | 62 | 73 | -0.01 |
| 0.80 | 10.25 | 0.47 | 91.08 | 91.02 | 46.10 | 34 | 25 | 0.01 |
| 0.90 | 6.08 | 0.27 | 94.75 | 94.67 | 46.10 | 14 | 7 | 0.00 |
| 0.95 | 3.42 | 0.06 | 96.94 | 96.88 | 46.10 | 3 | 0 | 0.00 |
| 0.99 | 0.53 | 0.00 | 98.48 | 98.48 | 46.10 | 0 | 0 | 0.00 |

## CIFAR100C: per-sample outcomes

| outcome | samples | % of stream |
|---|---|---|
| fixed | 20,300.00 | 13.53 |
| broken | 6,800.00 | 4.53 |
| both right | 48,844.00 | 32.56 |
| both wrong, disagree | 43,760.00 | 29.17 |
| both wrong, agree | 30,296.00 | 20.20 |
| Ramen acc / ceiling | nan | 46.10 |
| ceiling (zs or Ramen right) | nan | 50.63 |
| P(zs right | exactly one right) | nan | 25.09 |

## CIFAR100C: zero-shot confidence by outcome (quartiles)

| outcome | count | 25% | 50% | 75% |
|---|---|---|---|---|
| both right | 48,844.000 | 0.312 | 0.564 | 0.836 |
| both wrong, agree | 30,296.000 | 0.133 | 0.257 | 0.456 |
| both wrong, disagree | 43,760.000 | 0.067 | 0.099 | 0.157 |
| broken | 6,800.000 | 0.108 | 0.179 | 0.293 |
| fixed | 20,300.000 | 0.087 | 0.142 | 0.246 |

## CIFAR100C: how much of the harm a gate at tau can reach

| tau | % of all broken with conf >= tau | % of all fixed with conf >= tau | broken / (broken + fixed) in gate % |
|---|---|---|---|
| 0.5 | 5.2 | 3.5 | 33.2 |
| 0.6 | 2.0 | 1.2 | 36.3 |
| 0.7 | 0.9 | 0.4 | 45.9 |
| 0.8 | 0.5 | 0.1 | 57.6 |
| 0.9 | 0.2 | 0.0 | 66.7 |
| 0.9 | 0.0 | 0.0 | 100.0 |
| 1.0 | 0.0 | 0.0 | 0.0 |

## CIFAR100C: broken and fixed per zero-shot confidence bin

| conf_zs | broken | fixed | all samples | broken / (broken + fixed) % | net if this bin kept zero-shot |
|---|---|---|---|---|---|
| (-0.001, 0.1] | 1,464 | 6,482 | 36,433 | 18.4 | -5,018 |
| (0.1, 0.2] | 2,344 | 6,961 | 36,256 | 25.2 | -4,617 |
| (0.2, 0.3] | 1,355 | 3,296 | 19,029 | 29.1 | -1,941 |
| (0.3, 0.4] | 834 | 1,868 | 13,172 | 30.9 | -1,034 |
| (0.4, 0.5] | 451 | 985 | 9,877 | 31.4 | -534 |
| (0.5, 0.6] | 215 | 468 | 7,757 | 31.5 | -253 |
| (0.6, 0.7] | 75 | 167 | 6,221 | 31.0 | -92 |
| (0.7, 0.8] | 28 | 48 | 5,879 | 36.8 | -20 |
| (0.8, 0.9] | 20 | 18 | 6,254 | 52.6 | 2 |
| (0.9, 1.0] | 14 | 7 | 9,122 | 66.7 | 7 |

## CIFAR100C: separating broken from fixed: AUROC and best achievable gate

| signal (high = keep zero-shot) | AUROC broken vs fixed | AUROC doc def. (zs right, all disagreements) | best gate gain (pts, tuned on test) | disagreements switched |
|---|---|---|---|---|
| zero-shot conf | 0.579 | 0.673 | 0.009 | 39 |
| - zero-shot entropy | 0.551 | 0.645 | 0.005 | 13 |
| - Ramen conf | 0.669 | 0.463 | 0.049 | 2,482 |
| zero-shot conf - Ramen conf | 0.711 | 0.558 | 0.093 | 4,547 |
| overshoot of Ramen class (whole stream) | 0.599 | 0.504 | 0.001 | 5 |
| overshoot of Ramen class (running, causal) | 0.601 | 0.509 | 0.004 | 244 |
| overshoot Ramen class - zs class (running) | 0.647 | 0.586 | 0.025 | 1,029 |
| kNN pseudo-label vote, causal (k=20) | 0.589 | nan | 0.045 | 363 |
| kNN TRUE-label vote (oracle, k=20) | 0.800 | nan | 0.976 | 4,316 |

## CIFAR100C: zero-shot feature neighbourhood (k=20, true labels, own image excluded)

| outcome | neighbours truly of zero-shot class % | neighbours truly of Ramen's class % | % where Ramen's class has more neighbours |
|---|---|---|---|
| broken (zs class = true) | 23.3 | 8.5 | 18.8 |
| fixed (Ramen's class = true) | 7.5 | 18.2 | 61.6 |

## CIFAR100C: gain a gate can reach vs signal AUROC (6,800 broken vs 20,300 fixed)

| AUROC | max gain (pts), binormal | max gain at balanced base rate (pts) |
|---|---|---|
| 0.60 | 0.00 | 0.64 |
| 0.65 | 0.03 | 0.97 |
| 0.70 | 0.17 | 1.31 |
| 0.75 | 0.43 | 1.66 |
| 0.80 | 0.81 | 2.03 |
| 0.85 | 1.30 | 2.43 |
| 0.90 | 1.92 | 2.88 |
| 0.95 | 2.73 | 3.42 |
| 0.99 | 3.78 | 4.08 |
| ceiling (all broken saved) | 4.53 | 4.53 |

## CIFAR100C: which class Ramen moves samples into (whole-stream marginal)

| outcome | Ramen's class over-predicted % | median overshoot of Ramen's class % |
|---|---|---|
| broken | 67.8 | 19.9 |
| fixed | 51.5 | 0.5 |

## CIFAR100C: per target class (Spearman rho = 0.62, p = 5.1e-12)

| class | Ramen overshoot % | broken into class | fixed into class | broken / (broken + fixed) % |
|---|---|---|---|---|
| fox | 109.5 | 113 | 213 | 34.7 |
| oak_tree | 104.5 | 168 | 196 | 46.2 |
| hamster | 81.0 | 89 | 271 | 24.7 |
| mountain | 78.3 | 179 | 192 | 48.2 |
| tractor | 70.5 | 202 | 378 | 34.8 |
| possum | 68.7 | 119 | 296 | 28.7 |
| skunk | 66.3 | 75 | 312 | 19.4 |
| beaver | 66.1 | 134 | 280 | 32.4 |
| pine_tree | 63.2 | 162 | 182 | 47.1 |
| crab | 59.3 | 88 | 383 | 18.7 |
