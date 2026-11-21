# Shift decomposition with oracles (2026-11-22)

**Question.** On each dataset, how much accuracy is available from fixing (A) a class-independent domain offset of the CLIP
embedding ("appearance"), (Γ) a class-specific domain offset ("concept"), and (L) the per-domain class frequencies ("label
shift")? Is any of them larger than Ramen's gain? Pre-registered in `PREREGISTRATION.md` before any number was computed.
**Every cell is an ORACLE** (true labels and/or true domain ids).

Offline on seed-0 cached CLIP features; CPU only (16 threads); 11 s (EuroSAT) to 393 s (DomainNet) per dataset.
Nothing in `src/`, `cfg/` or the shared harness was edited.

| file | job |
|---|---|
| `PREREGISTRATION.md` | question, model, cells, honesty rules, decision rules, predictions (written first) |
| `data.py` | `ShiftData`: cached features, text embeddings, labels, domains, leave-out groups, cached NoAdapt/Ramen |
| `anova.py` | `AdditiveFit` (two-way MANOVA on cell sums, SS table, shrinkage n₀), `split_half_reliability` |
| `oracles.py` | `PrototypeOracle` (image world, leave-one-out), `TextOracle`, `LabelPrior`, `Evaluator` (temperature fit, prediction) |
| `attribution.py` | domain-averaged accuracy, Shapley values, Poisson cluster bootstrap |
| `experiment.py` | `ShiftDecomposition`: all cells, controls, cross-fit, sanity checks, bootstrap, cache in `results/` |
| `analysis.py` | `Summary` (cells, Shapley with CIs, per-domain, decision), `overview` |
| `plots.py` | variance shares, Shapley vs Ramen, per-domain Shapley |
| `run_shift_decomposition.py` | CLI (`python run_shift_decomposition.py <dataset> [--smoke]`) |
| `make_notebook.py`, `01_shift_decomposition.ipynb` | notebook (executed) |
| `results/`, `logs/` | cached cells (`<ds>_s0.npz/.pkl`), run logs |

Run: `~/miniconda3/envs/RAMEN/bin/python run_shift_decomposition.py DomainNet` (threads capped inside the script).

## Math

Baseline (zero-shot = NoAdapt): $z_i=f_{\theta_0}(x_i)/\lVert f_{\theta_0}(x_i)\rVert$, $\hat y_i=\arg\max_c z_i^\top w_c$
(Ramen's notation, `references/ramen.md`; Ramen classifies with its adapted $\tilde z_i$).

Embedding model (weighted least squares on cell sums, backfitting): $z_i=\mu+\alpha_{d_i}+\beta_{y_i}+\gamma_{y_i,d_i}+\varepsilon_i$,
$\sum_d n_d\alpha_d=\sum_c n_c\beta_c=0$; $\gamma$ = cell mean − additive fit. $y_i$, $d_i$ are **oracle** inputs.

Image-world cells, $S\subseteq\{A,\Gamma,L\}$ (`img_<S>`; `img_` = P0):
$$\text{score}_c=\kappa_S\big(\tilde z_i^\top m_{c,d_i}-\tfrac12\lVert m_{c,d_i}\rVert^2\big)+\mathbb 1[L]\log(C\hat\pi_{c\mid d_i}),\quad
\tilde z_i=z_i-\mathbb 1[A]\hat\alpha_{d_i},\quad m_{c,d}=\hat\mu+\hat\beta_c+\mathbb 1[\Gamma]\tfrac{n_{c,d}}{n_{c,d}+n_0}\hat\gamma_{c,d}$$
Text-world cells (`txt_<S>`): $\tau z_i'^\top w_c+\mathbb 1[L]\log(C\hat\pi_{c\mid d_i})$, $z_i'=\mathrm{norm}(z_i-\hat\alpha_{d_i})$ if A;
`txt_Ad`: $\hat\alpha_d$ replaced by the image-weighted domain mean minus the grand mean (needs domains, not labels).

Honesty: the true-class prototype and prior count of image $i$ exclude its leave-out group (all 15 corrupted copies of the photo
on ImageNet-C; the image elsewhere); $n_0=\sigma^2/\hat\tau^2$ by method of moments; $\kappa_S,\tau$ by NLL on 20,000 images.
Shapley values over $\{A,\Gamma,L\}$ (image) and $\{A,L\}$ (text). CIs: Poisson cluster bootstrap, B = 1000.
Controls: `rnd_*` = same fits with shuffled domain labels (compared with their own `rnd_` P0); `cf_*` = 2-fold cross-fit;
`img_G_n0`, `img_AG_n0` = unshrunk γ.

## Data and sanity

| dataset | images | classes | domains | zero-shot | Ramen gain | floor | `txt_` == cached NoAdapt |
|---|---|---|---|---|---|---|---|
| ImageNet-C 5K | 75,000 | 1,000 (997 present) | 15 | 27.27 | +5.79 | 0.54 | 99.43% (426 differ, fp16) |
| DomainNet | 596,006 | 345 | 6 | 55.37 | +1.43 | 0.11 | 99.70% (1,811) |
| ImageNet-R/A/Sketch | 88,389 | 1,000 head (R, A: 200 each) | 3 | 46.34 | +1.50 | 0.48 | 99.68% (282) |
| EuroSAT | 27,000 | 10 | 1 | 48.09 | +14.56 | 1.51 | 99.58% (113) |

EuroSAT: every A / Γ cell equals P0 on 100.00% of images (α = γ = 0 by construction). Random-domain controls: +0.00 to +0.02
over their own P0 on all three multi-domain datasets.

## Results (domain-averaged points)

Geometry (debiased share of embedding variance; F of the term):

| dataset | α appearance | γ concept | β class | within | γ reliability (Spearman–Brown) | n₀ |
|---|---|---|---|---|---|---|
| ImageNet-C 5K | 25.5% (F 2,221) | 2.0% (F 1.17) | 23.3% | 49.3% | 0.46 | 31.4 |
| DomainNet | 15.6% (F 32,502) | 9.4% (F 57.7) | 18.0% | 57.1% | 0.94 | 6.1 |
| ImageNet-R/A/Sketch | 4.0% (F 2,729) | 1.9% (F 7.5) | 30.9% | 63.2% | 0.81 | 34.0 |
| EuroSAT | — | — | 39.6% | 60.4% | — | — |

Shapley values (95% CI), image world and text world; text-prototype gap = P0 − zero-shot:

| dataset | img φ_A | img φ_Γ | img φ_L | txt φ_A | txt φ_L | P0 − zero-shot | `img_AG` − zero-shot | Ramen |
|---|---|---|---|---|---|---|---|---|
| ImageNet-C 5K | **+1.94** [1.78, 2.09] | +0.19 [0.11, 0.28] | −1.24 | +1.66 | −0.79 | −7.90 | −5.64 | +5.79 |
| DomainNet | +2.87 [2.82, 2.93] | **+5.25** [5.15, 5.34] | +0.50 | +0.92 | +0.81 | −4.86 | +2.76 | +1.43 |
| ImageNet-R/A/Sketch | +1.68 [1.51, 1.86] | +3.46 [3.19, 3.72] | **+4.99** [4.81, 5.17] | −0.44 | +13.01 | +4.12 | +9.81 | +1.50 |
| EuroSAT | 0.00 | 0.00 | +0.07 | 0.00 | −0.77 | +29.40 | +29.40 | +14.56 |

Per domain (image world): DomainNet quickdraw φ_Γ **+20.50** (A +4.12), clipart Γ +5.08, infograph A +5.66 (Γ −0.24),
real Γ +2.29, sketch Γ +1.76; R/A/Sketch adversarial Γ +8.92, sketch Γ +2.06, rendition L +12.14 (Γ −0.61);
ImageNet-C: A positive on all 15 corruptions (+0.98 jpeg to +3.72 brightness), Γ ≈ 0 except brightness (+2.76).

Cross-fit Shapley (estimates from the other half): DomainNet A +2.89 / Γ +5.08 / L +0.40; R/A/Sketch +1.67 / +2.77 / +4.54;
ImageNet-C +1.51 / 0.00 (n₀ = ∞ in the halves) / −1.46. Unshrunk γ is worse than shrunk everywhere (e.g. ImageNet-C `img_G` 19.68 → 16.69).

## Why it makes sense or not

- **Appearance (A).** *Should help* wherever a domain moves the whole embedding cloud. *Happened:* the only sizeable positive
  component on ImageNet-C (+1.94; α is 26% of embedding variance), and +2.87 on DomainNet. In the text world it helps on
  ImageNet-C (+1.66) and DomainNet (+0.92) but hurts on R/A/Sketch (−0.44): moving the image off its own cone toward the text
  prototypes is not neutral there (hypothesis: modality gap; not measured). Domain means alone (`txt_Ad`) give the same as the
  class-adjusted oracle, so the domain id, not the labels, carries A.
- **Concept (Γ).** *Should help* where classes change differently per domain. *Happened:* dominant on DomainNet (+5.25; γ 9% of
  variance, F 58, reliability 0.94), +20.5 on quickdraw, the domain where Ramen's sinks live; +8.92 on ImageNet-A. On
  ImageNet-C γ is barely detectable (2%, F 1.17), so corruptions are close to a pure offset.
- **Label shift (L).** *Should help* where class frequencies differ by domain. *Happened:* large only where the class set itself
  changes (R/A use 200 of 1,000 classes: text world +13.01, rendition +12.14). With leave-one-out counts it hurts on ImageNet-C
  (−1.24) and the text world of EuroSAT (−0.77): the ImageNet-C "label shift" is subset count noise (as in the 2026-10-26 memo).
- **Ramen vs the ceilings.** On ImageNet-C Ramen's +5.79 is far above every mean-based ceiling (A + Γ = 2.13), so most of its
  gain is not a mean shift of the frozen feature (consistent with re-encoding through adapted norm layers; not measured here).
  On DomainNet and R/A/Sketch the domain ceilings are 3–6× Ramen's gain.
- **Image vs text world.** Image prototypes start below zero-shot on ImageNet-C (−7.90, ~4 other photos per class) and DomainNet
  (−4.86), so image-world Shapley values measure the value of the fixes inside a nearest-class-mean classifier. The column
  `img_AG − zero-shot` is the comparison that matters for a method: +2.76 DomainNet, +9.81 R/A/Sketch, −5.64 ImageNet-C.

## Verdicts (pre-registered rules; oracle ceilings, seed-independent)

- ImageNet-C: **A dominates** (prediction confirmed); domain-aware direction **not alive** (2.13 < 5.79).
- DomainNet: **Γ dominates** (prediction confirmed, 61%); domain-aware direction **alive** (8.12 > 1.43).
- ImageNet-R/A/Sketch: **no single dominant component** (L 49.2%, prediction narrowly fails); domain-aware direction **alive** (5.14 > 1.50).
- EuroSAT: **A = Γ = 0 exactly** (bug check passes); not applicable (one domain).
- Random-domain control: **null** (valid). Cross-fit: **reproduces** the Shapley values.

## Deviations from the plan discussed in chat

- ANOVA fitted on the full unbalanced design by weighted least squares, not on the complete sub-grid (R/A/Sketch has no complete grid beyond 200 classes).
- No refit bootstrap on ImageNet-C: estimation noise is checked by the 2-fold cross-fit instead; CIs are bootstrap with fixed estimates.
- The random-domain control is compared with its own P0 (`rnd_`), added after the first run showed that on unbalanced designs
  P0 already uses domain-adjusted class effects (all datasets re-run).
- Leave-one-out fallback: 555 ImageNet-C images (0.74%) belong to classes with a single photo; their true-class prototype falls back to μ.

## Next

Step 2 screen on DomainNet and R/A/Sketch: label-free domain inference + label-free per-domain prototypes, measured against
`img_AG − txt_` (+2.76 DomainNet). The domain id is not the bottleneck (`txt_Ad` ≈ `txt_A`); estimating class-specific
per-domain prototypes without labels is (zero-shot pseudo-labels already failed as CE targets).
