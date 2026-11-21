# 2026-10-26 — Prior correction toward an estimated class marginal

Pre-registered (`PREREGISTRATION.md`): replace pc's uniform target with (a) the true class marginal (oracle ceiling) and (b) label-free estimates: online EM on the zero-shot probabilities (**zs-EM, primary**), the zero-shot soft marginal, and EM on the method's own probabilities. λ = 1, m = 0.99 (frozen). Bases T3 (ImageNet-C 5K, DomainNet, ImageNet-R/A/Sketch) and Ramen (also CIFAR-10-C, CIFAR-100-C, EuroSAT), seeds 0–4, offline from saved probabilities (CPU only).

**Result: the right target is worth +2 to +3 points where the marginal is skewed, but no label-free estimate recovers it.**

| Gain of base + pc[target] over the same base (pts, mean of 5 seeds) | uniform | oracle | **zs-EM** | zs-marginal | self-EM | floor |
|---|---|---|---|---|---|---|
| T3, ImageNet-R/A/Sketch | −1.74 | **+2.75** | −0.40 | −0.55 | +0.26 | 0.48 |
| T3, ImageNet-C 5K | +0.51 | +1.96* | −3.31 | −0.01 | −0.97 | 0.53 |
| T3, DomainNet | +0.17 | +0.40 | −1.86 | −0.02 | −0.41 | 0.11 |
| Ramen, CIFAR-10-C / CIFAR-100-C / EuroSAT | +0.35 / +1.15 / +2.13 | = uniform (balanced) | −1.12 / −5.29 / −1.61 | ≈ 0 | negative | |

\* not attainable: the 5K subset's per-class counts (0–12) are sampling noise around a uniform population.

- Oracle ceiling alive (decision 1). The post-hoc structure oracle (domain sizes + class sets only) keeps +2.32 of the +2.75 on R/A/Sketch.
- Primary claim fails (decision 2): zs-EM is negative on all 9 dataset × base pairs. Its target ends farthest from the truth (TV 0.66 vs uniform 0.17 on ImageNet-C), because EM on miscalibrated zero-shot CLIP piles prior mass onto zero-shot's sink classes.

| File | What it is |
|---|---|
| `PREREGISTRATION.md` | Targets, bases, decision rules, frozen before computing |
| `prior_estimation.py` | Targets, pc with a per-batch target, per-seed tables, summaries, post-hoc structure oracle, plot |
| `prior_estimation.ipynb`, `build_notebook.py`, `interpretation.md` | Write-up |
| `logs/` | Run logs |

Run: `python prior_estimation.py <dataset> 0 1 2 3 4` → `results/prior_estimation/<ds>_s<seed>_{gains,per_domain,diag}.csv`. DomainNet takes ~25 min per seed on CPU (run seeds as parallel processes); the others take 1–5 min.
