# Held-out: NIQE vector + T3 on ImageNet-R / -A / -Sketch (2026-11-20)

**Question.** The stack NIQE vector (λ = 0.1) + RamenActive prob T = 3 gave +0.87 ± 0.10 over Ramen on ImageNet-C
(`../2026-11-20_nss_t3_stack/`), with both settings chosen there. Does it hold, frozen, on a 1,000-class stream of
renditions, adversarial photos and sketches? Pre-registered in [PREREGISTRATION.md](PREREGISTRATION.md): primary =
mean gain over Ramen > noise floor (seeds 0–4); secondary = beats T3 alone on the mean and ≥ 4/5 seeds; NIQE alone
reported as a decomposition.

## Files

| File | Job |
|---|---|
| `PREREGISTRATION.md` | frozen settings, claims and predictions, written before any NIQE run on this data |
| `run_heldout_stack.py` | runner: `smoke` (λ = 0 + T3 == RamenActive T3 on 3,000 images; undefined-NIQE rate per domain), `run stack:<s> niqe:<s> ...` |
| `analysis.py` | per-seed / summary / per-domain tables → `tables/` (adapted from the ImageNet-C analysis: dataset, image-level bootstrap, optional NIQE-alone) |
| `plots.py` | notebook figures (copied from the ImageNet-C folder, per-domain labels) |
| `build_notebook.py` | writes and executes `heldout.ipynb`; reuses the ImageNet-C notebook's math cell |
| `after_runs.sh` | waits for all 10 result files, then runs the analysis and builds the notebook |
| `interpretation.md` | the notebook's interpretation cell |
| `heldout.ipynb` | the report |
| `tables/`, `logs/` | outputs (`logs/smoke.log`, per-GPU run logs, `analysis.txt`) |

No new method code and no new config file: every run passes `cfg/ImageNetC5K/RamenNSS.yaml` + overrides in full
(its Ramen part equals `cfg/ImageNetVariants/Ramen.yaml`, asserted in the runner).

## Math

Identical to `../2026-11-20_nss_t3_stack/README.md`: T3 replaces the equal queue weight $1/|C_i|$ by
$q_{ic} = \mathrm{softmax}(100\, z_i^\top W / 3)_c$; NIQE replaces the selection score $d_{ij}$ by
$d_{ij} + 0.1\,\lVert A_t(v_i - v_j)\rVert_2 / 6$ with $v$ the 36-d NIQE block-mean features and $A_t$ the causal
whitening. No term uses true labels.

## Results (ImageNetVariants, ViT-B/16, mixed stream, domain-averaged accuracy)

Ramen over seeds 0–4: 47.86 / 47.93 / 47.88 / 47.69 / 47.86 → noise floor 2 × 0.24 = **0.48**.
Smoke: λ = 0 + T3 == RamenActive T3 on 3,000 of 3,000; undefined NIQE vector 0 / 0 / 4 of 1,000 (R / A / Sketch).

| seed | Ramen | T3 | NIQE | NIQE + T3 (95% CI vs Ramen) | stack vs T3 (95% CI) | fixed / broken vs T3 |
|---|---|---|---|---|---|---|
| 0 | 47.86 | +0.61 | −0.05 | **+0.67** [+0.46, +0.90] | +0.06 [−0.07, +0.20] | 984 / 940 |
| 1 | 47.93 | +0.66 | −0.01 | **+0.60** [+0.38, +0.84] | −0.06 [−0.20, +0.08] | 997 / 1,052 |
| 2 | 47.88 | +0.56 | −0.02 | **+0.53** [+0.31, +0.76] | −0.03 [−0.17, +0.10] | 981 / 1,002 |
| 3 | 47.69 | +0.56 | +0.07 | **+0.61** [+0.39, +0.84] | +0.05 [−0.09, +0.19] | 989 / 1,015 |
| 4 | 47.86 | +0.58 | +0.16 | **+0.63** [+0.39, +0.86] | +0.05 [−0.08, +0.18] | 1,010 / 957 |
| mean ± std | | +0.59 ± 0.04 | +0.03 ± 0.09 | **+0.61 ± 0.05** | **+0.01 ± 0.06**, 3/5 | |

Per domain (mean gain vs Ramen over seeds; T3 / NIQE / stack): adversarial +1.37 / +0.12 / +1.41,
rendition +0.06 / −0.01 / +0.08, sketch +0.36 / −0.01 / +0.33.
Same-domain share of the top-8 retrieved weight: NIQE 58.5%, T3 78.6%, stack 78.5% (Ramen not logged).

- **Primary: PASS, carried by T3.** +0.61 > 0.48, but T3 alone gives +0.59.
- **Secondary: FAIL.** Stack vs T3 +0.01 ± 0.06, 3/5 seeds, every CI includes 0.
- **Decomposition:** NIQE alone +0.03 ± 0.09, which is the pre-registered alternative (NIQE ≈ 0, stack ≈ T3).

## Why it makes sense (and what is not shown)

- **Why it should help:** on ImageNet-C, NIQE added same-corruption support on top of T3 (59.0 → 62.5%) and +0.25.
- **What happened:** here NIQE does not change the support's domain composition on top of T3 (78.6 → 78.5%), and
  accuracy does not move. Hypothesis, not measured: R/A/Sketch differ in content and style rather than pixel-level
  distortion, so NIQE statistics carry no domain signal beyond CLIP's.
- **Ruled out:** undefined NIQE vectors on sketches (0.4%).

**Verdict: the NIQE part does not generalise.** The stack passes the floor only through T3. NIQE matching is a
corruption-specific add-on (ImageNet-C +0.25 on top of T3), not a general improvement.
