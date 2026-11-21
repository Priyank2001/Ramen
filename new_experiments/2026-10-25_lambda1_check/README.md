# 2026-10-25 — Prior correction at λ = 1 on the small-label datasets

Step #1 after the stacking experiment (`../2026-10-25_stacking/`): freeze λ = 1 (pre-registered in `PREREGISTRATION.md` before computing it), then check it doesn't give up the gains λ = 2 had on CIFAR-10-C, CIFAR-100-C and EuroSAT.

**Result: passes on all three** (mean gain > floor and ≥ 50% of λ = 2's gain): +0.35 / +1.15 / +2.13 vs λ = 2's +0.57 / +1.50 / +2.79, seeds 0–4. CIFAR-10-C is the thin one (seed min +0.30, floor 0.31). Frozen setting for the held-out test: T3 + pc(λ = 1), no kNN, on large label spaces; Ramen + pc(λ = 1) on small ones.

| File | What it is |
|---|---|
| `PREREGISTRATION.md` | Frozen setting and decision rule, written before the numbers |
| `lambda_check.py` | Offline re-scoring from cached Ramen probabilities (`python lambda_check.py`), summary and decision |
| `lambda1_check.ipynb` | Write-up: data, sanity check vs earlier λ = 2 numbers, results, decision, interpretation, glossary |
| `build_notebook.py`, `interpretation.md` | Notebook builder and its interpretation text |

Output: `results/lambda1_check/gains.csv`. Sanity: λ = 2 reproduces the earlier means to 0.01, except EuroSAT knn + pc 6.99 vs 7.02. The original script isn't in the repo; the components match.
