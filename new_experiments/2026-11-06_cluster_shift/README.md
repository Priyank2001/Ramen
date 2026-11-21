# Cluster shift (2026-11-06)

**Question.** Is Ramen's gain mostly a label-free, domain-level correction? Test: replace Ramen by a streaming
per-cluster mean shift of the zero-shot embedding (no memory, no gradients) and see how much of Ramen's gain it recovers.
Pre-registered: ≥70% of Ramen's gain and beating a random-group control = confirmed; <30% = refuted.

| file | job |
|---|---|
| `cluster_shift.py` | the correctors: `KMeansShift` (label-free), `RandomGroupShift` (control), `OracleGroupShift` (ORACLE, true domain id), `NoShift`, `StreamRunner` |
| `cshift_configs.py` | `DATASETS`, `VARIANTS` (kind, K, alpha, reference) |
| `cshift_experiment.py` | `ClusterShiftExperiment`: loads cached features, replays a variant, sanity check, summary table |
| `analysis.py` | `ShiftExplained`: share of Ramen's own embedding shift a group-constant shift reproduces |
| `plots.py` | dot plot of gain vs noise floor |
| `run_cluster_shift.py` | CLI, one dataset / seed (offline) |
| `make_notebook.py`, `01_cluster_shift.ipynb` | notebook (executed) |
| `logs/` | run outputs (`shift_explained_s0.log` holds the diagnostic) |

Run: `python run_cluster_shift.py ImageNetC5K 0` (no GPU forward pass; GPU only encodes text features once if missing).
Nothing in `src/`, `cfg/` or the shared harness was edited; results are cached as `results/<run>_cshift_<variant>.pkl`.

## Math

Baseline: $z_i = f(x_i)/\lVert f(x_i)\rVert$, $\hat y_i = \arg\max_c z_i^\top w_c$; Ramen classifies with $\tilde z_i$
(one-step adapted norm parameters; full notation in the skill's `ramen.md`).

Cluster shift: $z_i' = \mathrm{norm}\big(z_i - \alpha(\mu_{g(i)} - m)\big)$, $\hat y_i = \arg\max_c z_i'^\top w_c$.
- $g(i)$: online mini-batch k-means cluster (`k`), or random with matching cluster sizes (control), or **the true domain id (ORACLE)**.
- $\mu_g$: running mean of embeddings in $g$ so far, current batch included. $\alpha$ = `alpha` (0.5, 1).
- $m$ = `ref`: `global` (running mean of all embeddings) or `text` ($\bar w=\frac1C\sum_c w_c$).

## Results (seed 0, full datasets; gain over NoAdapt, domain-averaged points)

| dataset | Ramen | noise floor | best label-free (`global`) | % of Ramen | ORACLE domain | % of Ramen |
|---|---|---|---|---|---|---|
| ImageNet-C 5K (75,000 img) | +5.79 | 0.54 | +0.64 (k20, α0.5) | 11% | +1.58 (α1) | 27% |
| DomainNet (596,006 img) | +1.43 | 0.11 | +0.43 (k20, α0.5) | 30% | +0.86 (α1) | 60% |
| EuroSAT (27,000 img, 1 domain) | +14.56 | 1.51 | −0.37 (k5, α0.5) | <0% | n/a | n/a |

Random-group controls (`global`): −0.08 to +0.09 everywhere. `text`-reference variants: −0.4 to −37 on every dataset.

Share of Ramen's squared shift reproduced by group means beyond the all-image mean: ImageNet-C 14.7–19.4% (k-means), 21.8% (oracle domain);
DomainNet 12.7–16.1%, 13.7%; EuroSAT 28.0–42.7% (clusters ≈ classes there); random control ≈ 0.0%.

## Why it should help / what happened

- *Should help:* if Ramen mostly removes a domain offset, a group mean captures it. *Happened:* the oracle domain gets 27–60%
  of the gain, label-free clusters 11–30%, none near 70%. Group means explain only 13–22% of Ramen's own shift on the two
  multi-domain datasets, so most of its movement is not group-constant (a measured fact; what the rest is, is untested).
- `text` reference: adds the class-dependent term $\alpha\bar w^\top w_c$ to every logit. This biases predictions toward some
  classes and loses accuracy even for random groups: a design error, not evidence about domains.
- EuroSAT: one domain, clusters track classes, so subtracting a cluster mean removes class signal: all variants lose.

## Verdicts (seed 0 only; best-of-grid label-free picks are post-hoc)

- k-means shift, `global` ref: **ImageNet-C: null-to-small** (+0.64, floor 0.54). **DomainNet: small, above floor** (+0.43, floor 0.11), at the 30% refutation boundary. **EuroSAT: hurts.**
- Oracle domain shift: **partial** (27% / 60%).
- `text` ref: **hurts**. Random control: null (valid control).
- Hypothesis "mostly a domain-level shift": **refuted as the main explanation; a real partial component on DomainNet.**

## Caveats

Identity replay equals cached NoAdapt on 99.43–99.70% of images (cached fp16 features), not 100.00%. One seed. Untested: seeds 1–4,
α > 1, per-domain breakdown, stacking the shift under Ramen.
