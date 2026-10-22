# Research Directions: Beating Ramen on Mixed-Domain TTA

> **Superseded** by [ramen-research-directions.md](ramen-research-directions.md) (2026-10-05). This version was
> written from 200-image-per-corruption subsets; several of its conclusions changed on full data.

Written 2026-10-05. Read after [ramen-tta-thesis-brief.md](ramen-tta-thesis-brief.md) (goal, how Ramen works) and
[ramen-pc2-handoff.md](ramen-pc2-handoff.md) (ablation results, pending sweeps, candidates A–C). This file turns
those, plus the failure analysis in `new_experiments/`, into a prioritised plan for the paper.

**Evidence labels used below.** *Full data* = 150k-sample runs logged in `log/` and `report/all_results_long.csv`.
*Subset* = 200 images per corruption (3,000 per dataset), seed 0, from the `new_experiments/` notebooks. Subset
numbers are signals to confirm, not results. Several subset findings have already changed on full data (see 1.3).

---

## 1. Where things stand

### 1.1 Baselines (full data, mixed mode, seed 0, RTX 3090)

| dataset | backbone | NoAdapt | Tent | Ramen |
|---|---|---|---|---|
| CIFAR-10-C | ViT-B/32 | 61.36 | 66.71 | 72.63 |
| CIFAR-100-C | ViT-B/16 | 37.10 | 41.14 | 46.12 |

### 1.2 Ablations already run (full data, from the PC2 handoff)

- **Single vs mixed mode:** within ±0.5 pts on average, no consistent sign. Sharing one memory across domains does
  not measurably hurt. The "noise domains pollute the shared memory" hypothesis is not supported.
- **β = 5 vs 0:** +2.57 (CIFAR-10-C) and +1.03 (CIFAR-100-C) pts. Gains are largest on the noise corruptions, so
  retrieval and weighting do carry signal there. Angles 1–3 of the brief are still open.

### 1.3 Ramen's gain per corruption (full data)

Ramen − NoAdapt, in pts. Smallest gains first.

| CIFAR-10-C | gain | vs Tent | | CIFAR-100-C | gain | vs Tent |
|---|---|---|---|---|---|---|
| defocus_blur | +4.5 | **−1.0** | | frost | +4.2 | +3.7 |
| brightness | +4.5 | +1.3 | | jpeg_compression | +5.6 | +1.9 |
| frost | +4.7 | +0.6 | | snow | +6.3 | +2.6 |
| snow | +7.0 | +1.1 | | zoom_blur | +7.0 | +1.7 |
| … | | | | … | | |
| shot_noise | +20.8 | +21.8 | | pixelate | +14.1 | +8.9 |
| gaussian_noise | +22.2 | +22.2 | | contrast | +15.9 | +7.3 |

Pattern: on CIFAR-10-C, Ramen gains least where zero-shot is already strong (defocus blur, brightness, frost all
start above 72%). On defocus blur, Tent beats Ramen. Correction to the subset analysis: the subset showed Ramen
losing on pixelate, but on full data pixelate gains +13.3. That was sampling noise.

### 1.4 Failure analysis (subset; confirm on full data)

| signal | CIFAR-10-C | CIFAR-100-C |
|---|---|---|
| samples where Ramen and zero-shot disagree | 28.6% | 40.9% |
| …of those: Ramen right / zero-shot right / neither | 419 / 134 / 305 | 305 / 129 / 794 |
| "broken" (zero-shot right → Ramen wrong), share of all samples | ~4.5% | ~4.3% |
| ceiling for any per-sample zero-shot/Ramen selector ("either is right") | 72.8 vs 68.3 | 47.9 vs 43.6 |
| calibration error (ECE), NoAdapt vs Ramen | 2.3% vs 12.0% | (see notebook) |
| "deer" predictions, NoAdapt vs Ramen (true deer = 200) | 197 vs 392 | – |

- **Class collapse.** Ramen predicts "deer" about twice as often as zero-shot. The extra comes mostly from frog
  (44 vs 7), bird (41 vs 17) and cat (31 vs 8). The "broken" image grid is dominated by "→ deer".
- **Overconfidence.** Zero-shot CLIP is well calibrated. Ramen is not: its accuracy sits below its confidence
  in every bin.
- **Confident zero-shot predictions are never harmed.** See Section 3. Above 0.9 zero-shot confidence, Ramen agrees
  with zero-shot on every sample. The broken samples have *low* zero-shot confidence (median 0.34 on CIFAR-10-C,
  0.17 on CIFAR-100-C), almost the same as the samples Ramen fixes (0.31 and 0.14).

---

## 2. The proposed paper story

> **Ramen balances which samples it adapts *from*, but not what it *predicts*.** Pseudo-labels from the zero-shot
> model route samples into class queues; contaminated queues pull predictions towards a few attractor classes,
> make the model overconfident, and break a steady fraction of correct zero-shot predictions.

Each direction below tests one piece of that claim. If Direction 1 confirms the mechanism and a fix beats the noise
floor, that is the core contribution. Direction 2 is the "do no harm" extension. Directions 3–4 are conditional.

---

## 3. Directions, in priority order

### Direction 0 — Noise floor and full-data diagnostics (prerequisite)

**Why.** Every effect measured so far is 0.3–3 pts and there is no seed variance yet. Nothing below is
publishable without it. The subset findings in 1.4 also need to hold on full data before code is written for them.

**Do.**
1. Seeds 0/1/2 of the baseline on one GPU (PC2 job list, or GPU 1 on the main PC).
2. Run `new_experiments/cifar10c_ramen.ipynb` and `cifar100c_ramen.ipynb` on full data.
3. Run `new_experiments/tsne_features.ipynb` on full data. This captures features, so expect one extra run per
   method.
4. `exp.gate_sweep()` on the full-data results (free once step 2 is done).

**Success criterion.** A noise floor (max − min across seeds) per dataset, for `avg`, `noise`, `other`. Rule from
the PC2 handoff: an effect counts only if it is above about 2× that range.

**Cost.** About 10–20 min per CIFAR-10-C method run on a free 3090; CIFAR-100-C somewhat longer.

---

### Direction 1 — Class collapse, attacked through the memory (main direction)

**Evidence.** The deer over-prediction and the frog/bird/cat absorption (1.4, subset).

**Hypothesised mechanism.** In `Ramen.forward` ([methods/Ramen.py](methods/Ramen.py)) every sample is written into
the queue of its *zero-shot* prediction (`init_preds`), which is about 61% accurate on CIFAR-10-C. A frog
mispredicted as deer then sits in the deer queue. For every later query, the deer queue's top-k contributes
gradients that point towards "deer", including from the mislabelled entries. Balancing retrieval across queues
cannot help if the queues themselves are contaminated. The entropy weight `exp(-H)` gives the most weight to
confident entries, including confidently wrong ones.

**Step 1a — Measure it: memory diagnostics (logging only).** This is Phase 1 of the PC2 handoff. Pass the true
label and domain into the TTA model behind a `--diag` flag, and log per batch:
- per-queue purity: fraction of entries whose pseudo-label is correct, *broken down by queue*. The prediction is
  that attractor-class queues, like deer, are the least pure;
- pseudo-label accuracy of the retrieved support set;
- share of the aggregated gradient that comes from the true class's queue.

**Step 1b — Decide between backbone and adaptation with the t-SNE notebook.**
`exp.plot_tsne('deer', select='wrong', color_by='label')`:
- If the absorbed frogs already sit next to deer in the **zero-shot** panel, the confusion is in the backbone or
  text classifier. No adaptation change can fix it; the fix is on the text side (prompt ensembles, class-name
  descriptions).
- If they only move next to deer in the **Ramen** panel, adaptation is causing it, and steps 1c–1d apply.

**Step 1c — Candidate A: write entries with the adapted label.** Retrieve → adapt → predict → *then* write the
sample into the queue of its adapted prediction (about 73% accurate instead of 61%). The cached gradient stays
valid because weights are reset to base before every batch. Pitfalls and the A0/A1 ablation split are in the PC2
handoff, Section 6: empty memory on the first batch, and loss of self-retrieval.

**Step 1d — Prediction-prior correction (new; not in the brief's four angles).** Keep a running estimate π̂ of the
adapted predictions' class marginal (EMA over batches) and correct logits as `logits − λ·log π̂`, pulling
over-predicted classes back. Notes:
- A batch-level diversity loss (as in SHOT's information maximisation) does **not** fit Ramen's cache. Its
  per-sample gradient depends on the batch marginal, so it doesn't decompose into cached per-sample gradients.
  Prior correction works at the output instead, so it is compatible.
- To evaluate it post hoc, the results need full per-sample probabilities, not just the top-1 prediction. Add a
  `save_probs` option to `RamenExperiment.run` (CIFAR-100-C: 150k × 100 × fp16 ≈ 30 MB).
- λ must be fixed across datasets, like Ramen's own hyperparameters. Otherwise it is test-set tuning.

**Success criterion.** Lower per-class prediction bias (`exp.plot_prediction_bias()`), fewer broken samples, and a
higher average accuracy than the baseline by more than the noise floor. Report on CIFAR-10-C *and* CIFAR-100-C.

**Risk.** If 1b shows the confusion is in the backbone, Direction 1's payoff is limited to the text side.

---

### Direction 2 — "Do no harm": per-sample selective adaptation

**Evidence.** About 4.5% of samples are broken. Ramen is badly calibrated (ECE 12% vs 2.3%). Gains are smallest on
corruptions where zero-shot is already good.

**What we tested and ruled out (subset).** The obvious gate, "skip adaptation when zero-shot confidence ≥ τ",
does **not** help. Ramen already leaves confident predictions alone, and harm happens at low confidence. Details
and the implementation are in [Appendix A](#appendix-a--gating-adaptation-on-zero-shot-confidence).

**What is still open.** Among disagreements, the cheap confidence signals barely separate "zero-shot was right"
from "Ramen was right":

| signal (AUROC for "keep zero-shot") | CIFAR-10-C | CIFAR-100-C |
|---|---|---|
| zero-shot confidence | 0.59 | 0.65 |
| −Ramen confidence | 0.58 | 0.46 |
| zero-shot conf − Ramen conf | 0.63 | 0.54 |

When they disagree, Ramen is right 2.4–3× as often as zero-shot (305 vs 129, 419 vs 134), so a selector needs a much stronger
signal before it gains anything. Better candidate signals need richer logging (re-run with `--diag`):
- **Support-set agreement:** fraction of the retrieved top-k whose pseudo-label equals the query's zero-shot
  prediction. Low agreement suggests the sample sits in contested territory.
- **Isolation:** mean distance to the retrieved neighbours. Far-away samples get a support set that doesn't
  represent their domain.
- **Logit shift:** how far adaptation moved the logits (‖z_adapted − z_zs‖). Large moves on a sample are where
  harm can occur.
- **Augmentation consistency:** prediction stability under a couple of flips/crops (MEMO-style). This costs extra
  forward passes, so report compute.

**Ceiling.** Even a perfect selector between zero-shot and Ramen reaches only 72.8 / 47.9 (subset) vs Ramen's
68.3 / 43.6. That is about +4.5 pts. Realistic selectors will recover a fraction of that.

**A cheaper win, separate from accuracy:** soft logit interpolation `z = (1−γ)·z_zs + γ·z_adapted` likely fixes much
of the calibration error even where it barely moves accuracy. If the Ramen paper does not report calibration (check
it), "same accuracy, far better calibration" is a legitimate secondary contribution.

**Success criterion.** Fewer broken samples with no fewer fixed ones, net accuracy gain above the noise floor, and
lower ECE.

---

### Direction 3 — Adaptive retrieval hyperparameters (conditional on the PC2 sweeps)

- **Adaptive β (Candidate C):** replace the fixed β with a per-query scale, e.g. `exp(−d / median(d_topk))`.
  Pursue only if the β sweep (0, 2.5, 5, 10, 20) shows accuracy still rising past 5, especially on the noise
  corruptions.
- **Adaptive k / K:** pursue only if the topk (10/25/50/100) or capacity (2500/5000/7500) sweeps show differences
  above the noise floor. The paper claims robustness here; flat curves would mean this angle is closed.

---

### Direction 4 — Learned retrieval metric / learned aggregation (deferred)

Angles 1–2 of the brief. They are heavier to build, and they need training data, which is awkward in test-time
adaptation without leaking test labels. Pursue only if the Direction 1 diagnostics show that retrieval (low
same-domain rate, low support-set pseudo-label accuracy) is the failing component and the memory is clean.

---

### Direction 5 — Full benchmark protocol (required, but last)

ImageNet-C (15-corruption mixture), DomainNet (6-domain mixture), ViT-L/14, and the ResNet-101 check, all on the
Ramen paper's protocol and with one fixed hyperparameter set. Start only after a CIFAR result clearly beats the
noise floor on both CIFAR datasets.

---

## 4. Order of work and decision points

```
D0 noise floor + full-data notebooks + gate_sweep
 │
 ├─ D1a memory diagnostics ──┬─ attractor queues impure ──> D1c Candidate A (A0/A1), D1d prior correction
 │                           └─ queues clean ─────────────> retrieval is the problem -> D3 / D4
 ├─ D1b t-SNE: backbone or adaptation? ── backbone ──> text-side fix; deprioritise D1c/d
 │
 ├─ D2 selective adaptation: log richer signals with --diag, measure AUROC offline before building anything
 │
 └─ PC2 sweeps ── β still rising past 5 ──> D3 adaptive β;  k/K effects > noise floor ──> D3 adaptive k/K

Any fix > noise floor on both CIFAR datasets ──> D5 full protocol
```

The rule throughout: **measure offline before building.** The results cache in `new_experiments/results/` makes
most selection and correction rules evaluable post hoc without re-running the model.

---

## 5. Experiments and figures the paper will need

| claim | experiment | figure / table |
|---|---|---|
| Ramen collapses onto attractor classes | full-data prediction bias, per dataset | `plot_prediction_bias`, `pred_composition` |
| Collapse comes from contaminated queues | D1a per-queue purity vs class bias | scatter: queue purity vs over-prediction |
| Collapse is adaptation-induced (or not) | D1b t-SNE, zero-shot vs adapted | `plot_tsne(..., select='wrong')` |
| The fix reduces collapse and improves accuracy | D1c/d vs baseline, seeds 0/1/2 | main results table with ± over seeds |
| Ramen is overconfident; fix improves calibration | ECE for all methods | `plot_calibration` |
| Naive confidence gating doesn't help | `gate_sweep` on full data | small table (negative result, motivates D2) |
| Generalises | D5 benchmarks and backbones | full results table |

---

## 6. Open questions and risks

- **Subset vs full data.** Every number in 1.4 comes from 3,000 samples per dataset. The pixelate reversal already
  showed how far subset numbers can move.
- **GPU-to-GPU differences.** The same seed gives different numbers on different GPU models. Compare only within one
  GPU model (PC2 handoff rule 1).
- **Hyperparameter discipline.** Ramen uses one hyperparameter set across settings. Every new knob (λ, τ, γ) must be
  fixed across datasets too, or the comparison is unfair.
- **Compute.** Any extra signal that costs forward passes (augmentation consistency) has to be reported against
  Ramen's runtime. Its 490× speedup is part of its contribution.

---

## Appendix A — Gating adaptation on zero-shot confidence

### A.1 Implementation (if it is ever needed)

Two facts from the code make this easy:
- In `Ramen.forward`, the first `logits` are zero-shot, because weights are reset to base after every batch.
- The optimizer is SignSGD ([models/optimizer.py](models/optimizer.py)): the update is `−lr · sign(grad)`.
  **Scaling** a sample's gradient changes nothing. A **zero** gradient row gives `sign(0) = 0`, so that sample's
  weights stay exactly at base. This holds while the config has no weight decay or momentum, which is true for all
  current configs.

Hard gate, inside `Ramen.forward` after aggregation:

```python
retrieved_grads = retrieved_grads_sum / num_class_sum

tau = self.cfg.get('gate_tau')                     # off by default -> baseline reproduces exactly
if tau is not None:
    zs_conf = logits.detach().float().softmax(-1).max(-1).values   # zero-shot confidence
    retrieved_grads[zs_conf >= tau] = 0            # SignSGD: no update -> prediction == zero-shot

self.model.set_by_sample_grad(retrieved_grads)
```

The memory write happens earlier in `forward`, so gated samples still enter the memory. That is intended:
confident samples are the most reliable support for others.

Soft gate (per-sample step size): because scaling gradients is a no-op under SignSGD, scale the parameter change
instead, after `step_and_zero_grad()` and before inference:

```python
# gamma: (B,) in [0, 1], e.g. gamma = 1 - zs_conf
for m in self.model.model.modules():
    if isinstance(m, (BySampleLayerNorm, BySampleBatchNorm)):
        for p, base in ((m.weight_by_sample, m.weight), (m.bias_by_sample, m.bias)):
            p.data[:B] = base + gamma[:, None] * (p.data[:B] - base)
```

This equals a per-sample learning rate `γ_i · lr`. The cheaper approximation is logit interpolation
`(1−γ)·z_zs + γ·z_adapted`, which needs no model surgery.

### A.2 Why it doesn't help (subset)

Because the gate leaves the memory unchanged, its accuracy can be computed exactly from cached per-sample results.
That is `exp.gate_sweep()` in `new_experiments/ramen_analysis.py`.

| τ | CIFAR-10-C gated % | total acc (Ramen = 68.30) | broken saved / fixed lost | CIFAR-100-C total acc (Ramen = 43.63) |
|---|---|---|---|---|
| 0.5 | 54.8 | 67.20 | 25 / 58 | 43.57 |
| 0.7 | 39.1 | 68.10 | 8 / 14 | 43.63 |
| 0.8 | 31.4 | 68.23 | 4 / 6 | 43.63 |
| 0.9 | 21.3 | 68.30 | 0 / 0 | 43.63 |

Above τ ≈ 0.9 the gate changes nothing, because Ramen and zero-shot already agree. Below that it loses more fixes
than it saves. The harm Ramen does is concentrated in low-confidence samples, where zero-shot confidence can't
tell a broken sample from a fixed one. Re-check with `gate_sweep()` on full data before dropping the idea for good.
