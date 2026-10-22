# Research Directions: Beating Ramen on Mixed-Domain TTA

Written 2026-10-05; updated the same day with the seed-variance results (Section 1.7, Direction 0). Replaces
[ramen-research-old_directions.md](ramen-research-old_directions.md), which was based on 200-image-per-corruption
subsets. **Every number here is from full data:** 15 corruptions × 10,000 images = 150,000 test samples per dataset,
mixed mode, RTX 3090. Sections 1.1–1.6 use seed 0; Section 1.7 uses seeds 0–4.

Read after [ramen-tta-thesis-brief.md](ramen-tta-thesis-brief.md) (goal, how Ramen works) and
[ramen-pc2-handoff.md](ramen-pc2-handoff.md) (earlier ablations, pending sweeps, candidates A–C).

**Sources.** Per-sample results in `new_experiments/results/`, produced and plotted by the notebooks in
`new_experiments/`:

| notebook | content |
|---|---|
| `cifar10c_ramen.ipynb`, `cifar100c_ramen.ipynb` | accuracy, flips, class heatmaps, prediction bias, confusions, calibration, failure grids, gate sweep |
| `tsne_features.ipynb` (deer), `tsne_features_cifar100c.ipynb` (tractor) | t-SNE of zero-shot vs adapted features |
| `seed_variance.ipynb` | seeds 0–4 on both datasets: noise floor, per-corruption spread, over-correction stability, prediction churn |

Per-corruption gains in 1.3 come from `report/all_results_long.csv` (the `log/` runs), which differ from the
notebook runs by ≤ 0.02 pts on average.

### What changed from the old version

| old claim (subset) | full data |
|---|---|
| Ramen collapses onto a few attractor classes | **No.** Ramen's predictions are *more* balanced than zero-shot's. It **over-corrects** zero-shot's class bias instead (Section 1.5). |
| Ramen loses to zero-shot on pixelate | No: +13.3 pts. It was sampling noise. |
| Calibration error 2.3% → 12.0% | 2.86% → 8.31% (CIFAR-10-C), 5.09% → 9.96% (CIFAR-100-C). Overconfident, but less than the subset suggested. |
| Confidence gate doesn't help | Confirmed on both datasets. Its best gain (+0.06 pts) is also below the noise floor. |
| 2-seed subset test: average moves 1.2 pts between seeds | **No.** Over 5 full-data seeds the average moves only 0.16 pts (CIFAR-10-C) and 0.11 pts (CIFAR-100-C) (Section 1.7). |

---

## 1. Findings

### 1.1 Baselines

| dataset | backbone | NoAdapt | Tent | Ramen |
|---|---|---|---|---|
| CIFAR-10-C | ViT-B/32 | 61.36 | 66.71 | 72.61 (log: 72.63) |
| CIFAR-100-C | ViT-B/16 | 37.10 | 41.14 | 46.10 (log: 46.12) |

The notebook pipeline reproduces the `main.py` logs, and capturing features doesn't change any prediction (re-runs
gave identical accuracy).

### 1.2 Already ruled out

| hypothesis | evidence | verdict |
|---|---|---|
| Noise-domain samples contaminate the shared memory | single vs mixed mode within ±0.5 pts on average (PC2 handoff) | not supported |
| Embedding has no signal for noise domains, so retrieval can't help | β=5 vs β=0 gains most on noise corruptions (PC2 handoff) | not supported |
| Ramen degrades over the stream (stale memory, error accumulation) | gain over NoAdapt by stream third: 11.09 / 11.27 / 11.41 (CIFAR-10-C), 8.51 / 9.15 / 9.34 (CIFAR-100-C) | not supported; gain slightly *grows* |
| Skipping adaptation when zero-shot is confident helps | gate sweep, Section 1.6 / Appendix A | ruled out |

### 1.3 Gain per corruption

Ramen − NoAdapt, in pts, from the `log/` runs. Smallest gains first.

| CIFAR-10-C | gain | vs Tent | | CIFAR-100-C | gain | vs Tent |
|---|---|---|---|---|---|---|
| defocus_blur | +4.5 | **−1.0** | | frost | +4.2 | +3.7 |
| brightness | +4.5 | +1.3 | | jpeg_compression | +5.6 | +1.9 |
| frost | +4.7 | +0.6 | | snow | +6.3 | +2.6 |
| snow | +7.0 | +1.1 | | zoom_blur | +7.0 | +1.7 |
| … | | | | … | | |
| shot_noise | +20.8 | +21.8 | | pixelate | +14.1 | +8.9 |
| gaussian_noise | +22.2 | +22.2 | | contrast | +15.9 | +7.3 |

On CIFAR-10-C, gains are smallest where zero-shot already starts high (defocus blur 72.6, brightness 83.6, frost
75.7). On defocus blur, Tent beats Ramen by 1.0 pt, well above Ramen's seed range on that corruption (0.31 pts,
Section 1.7). Tent's own seed variance is not measured.

### 1.4 Per-sample flips

Each test sample, compared between NoAdapt (zero-shot) and Ramen on the identical stream:

| | CIFAR-10-C | CIFAR-100-C |
|---|---|---|
| fixed (zero-shot wrong → Ramen right) | 25,381 (16.9%) | 20,300 (13.5%) |
| **broken** (zero-shot right → Ramen wrong) | **8,498 (5.7%)** | **6,800 (4.5%)** |
| both right / both wrong | 83,535 / 32,586 | 48,844 / 74,056 |
| disagree (different predictions) | 32.3% | 47.2% |
| ceiling: pick whichever is right per sample | 78.28 (vs 72.61) | 50.63 (vs 46.10) |

- Broken rate by corruption, CIFAR-10-C: highest on impulse / gaussian / shot noise (8.2–8.8%), but those also have
  the highest fixed rates (22–30%). Next come jpeg compression (6.5%) and defocus blur (5.9%, vs only 10.4% fixed).
- CIFAR-100-C: highest broken rates on frost (5.8%), jpeg compression (5.6%), snow and elastic transform (5.1–5.2%).
  These are mostly the corruptions where Ramen gains least.

### 1.5 Class-level behaviour: over-correction, not collapse

**Ramen reduces zero-shot's class bias overall.** Total variation distance between the predicted and true class
distributions:

| | NoAdapt | Ramen |
|---|---|---|
| CIFAR-10-C | 0.099 | 0.054 |
| CIFAR-100-C | 0.264 | 0.171 |

**But it overshoots on specific classes.** Predictions per class, CIFAR-10-C (15,000 true samples each):

| class | NoAdapt | Ramen | what happened |
|---|---|---|---|
| bird | 27,786 | 14,661 | zero-shot's big over-prediction fixed |
| deer | 10,327 | **19,154** | under-predicted → over-predicted (−31% → +28%) |
| truck | 9,120 | 16,470 | under-predicted → over-predicted (−39% → +10%) |
| horse | 13,793 | 17,280 | slightly under → over (+15%) |
| frog | 14,146 | **10,481** | roughly right → drained (−30%) |

The over-correction is consistent with the prediction-balance mechanism. Retrieval draws equally from every class
queue, which pushes towards classes zero-shot neglects, but nothing stops at the true share.

What the extra deer predictions consist of (`pred_composition('deer')`):

| true class | NoAdapt → "deer" | Ramen → "deer" |
|---|---|---|
| deer | 6,996 (recall 46.6%) | 11,390 (recall 75.9%) |
| frog / bird / cat | 618 / 609 / 469 | 2,379 / 1,905 / 1,217 |
| all non-deer | 3,331 | 7,764 |

CIFAR-100-C (1,500 true samples per class):
- The most over-predicted classes are mostly inherited from zero-shot: fox 3,063 → 3,143, hamster 3,214 → 2,715.
- The clearest **Ramen-induced** overshoots are tractor (1,206 → 2,557), oak tree (2,569 → 3,068) and possum
  (1,818 → 2,530).
- False tractors are other vehicles: pickup truck 105 → 232, lawn mower 92 → 215, train 71 → 173, bus 47 → 141.

**The confusions it adds are between semantic neighbours.** The share of errors that stay within the true class's
superclass rises from 79.7% to 83.9% on CIFAR-10-C (animal/vehicle) and from 25.5% to 30.1% on CIFAR-100-C (its 20
standard superclasses).

**Classes where Ramen is worse than zero-shot:**
- CIFAR-10-C (2 classes): bird 69.9 → 64.5, frog 55.3 → 54.1. Bird's drop is largely the removal of zero-shot's
  over-prediction, which had inflated its recall.
- CIFAR-100-C (12 classes): keyboard 62.1 → 41.7, worm 68.0 → 56.7, willow tree 44.9 → 37.5, woman 57.7 → 51.7.

**Feature space (t-SNE, both datasets).** The non-deer samples that Ramen calls deer, and the vehicles it calls
tractor, already sit among the true deer/tractors in **zero-shot** feature space. The zero-shot classifier simply
didn't assign them there. So the confusion exists before adaptation, and Ramen moves the *decision*, not the
features. The wrongly-absorbed samples also cluster by corruption (noise, fog, frost, jpeg on CIFAR-10-C; elastic,
glass blur, pixelate, snow on CIFAR-100-C). t-SNE preserves neighbourhoods only, so treat this as a hint for the
diagnostics in Direction 1, not proof.

### 1.6 Confidence and calibration

| | CIFAR-10-C | CIFAR-100-C |
|---|---|---|
| ECE NoAdapt → Ramen | 2.86% → **8.31%** | 5.09% → **9.96%** |
| median zero-shot confidence, broken samples | 0.35 | 0.18 |
| median zero-shot confidence, fixed samples | 0.31 | 0.14 |
| AUROC "keep zero-shot" among disagreements: zero-shot conf | 0.609 | 0.673 |
| … −Ramen conf | 0.605 | 0.463 |
| … zero-shot conf − Ramen conf | 0.659 | 0.558 |

- **Zero-shot CLIP is well calibrated; Ramen is overconfident.**
- **Broken samples have low zero-shot confidence**, almost the same as fixed samples, so confidence can't separate
  them. When the two methods disagree, Ramen is right about 3× as often as zero-shot on both datasets
  (25,381 vs 8,498; 20,300 vs 6,800).
- **The confidence gate fails** (Appendix A). Best result: +0.06 pts on CIFAR-10-C and +0.00 on CIFAR-100-C.

### 1.7 Seed variance: the noise floor

Seeds 0–4, both datasets, same settings, all on RTX 3090s (`seed_variance.ipynb`).

**What a seed changes.** Only the order of the test stream. Nothing in `methods/` or `models/` draws random numbers,
and re-running a seed reproduces it exactly. NoAdapt is identical across all 5 seeds (0.0000 pts difference), so
all variance below is the effect of stream order on Ramen.

**Noise floor.** "A fix must beat" = 2 × (max − min) across seeds, the rule from the PC2 handoff.

| metric | CIFAR-10-C mean | std | range | **a fix must beat** | CIFAR-100-C mean | std | range | **a fix must beat** |
|---|---|---|---|---|---|---|---|---|
| average accuracy | 72.62 | 0.06 | 0.16 | **+0.31** | 46.10 | 0.04 | 0.11 | **+0.21** |
| noise corruptions | 59.97 | 0.13 | 0.32 | +0.65 | 35.26 | 0.15 | 0.39 | +0.78 |
| other corruptions | 75.79 | 0.06 | 0.14 | +0.28 | 48.81 | 0.04 | 0.08 | +0.17 |
| fixed % | 16.88 | 0.04 | 0.10 | +0.19 | 13.55 | 0.03 | 0.08 | +0.16 |
| broken % | 5.61 | 0.04 | 0.10 | −0.20 | 4.54 | 0.03 | 0.08 | −0.16 |
| class-share gap (TV) | 0.054 | 0.001 | 0.001 | −0.003 | 0.171 | 0.001 | 0.002 | −0.005 |
| ECE | 8.35 | 0.07 | 0.17 | −0.34 | 9.90 | 0.09 | 0.24 | −0.48 |

**Per corruption.** Single corruptions move much more than the average:
- CIFAR-10-C: 0.19 (motion blur) to 0.74 pts (impulse noise).
- CIFAR-100-C: 0.38 (snow, pixelate) to 1.19 pts (shot noise).

So a per-corruption claim needs roughly 0.4–2.4 pts depending on the corruption. The ranges are plotted in the
notebook.

**The over-correction is stable.** Mean over seeds, with range:

| class | overshoot (predicted / true share − 1) | range |
|---|---|---|
| deer (CIFAR-10-C) | +28.3% | 1.1 |
| truck (CIFAR-10-C) | +9.5% | 0.9 |
| frog (CIFAR-10-C) | −30.2% | 0.4 |
| tractor (CIFAR-100-C) | +73.3% | 6.6 |
| possum (CIFAR-100-C) | +69.0% | 5.1 |
| oak tree (CIFAR-100-C) | +101.1% | 6.6 |

Every range is small compared with the effect, so Section 1.5 describes Ramen, not seed 0. On CIFAR-100-C the
multi-seed bias plot adds a nuance:
- Ramen pushes some classes far past their true share (tractor, possum, beaver, oak tree).
- Many classes zero-shot badly under-predicts stay badly under-predicted (ray, lobster, table, wardrobe: −60 to
  −75%).
- The class-share gap is still 0.171, against 0.054 on CIFAR-10-C. That leaves more room for a
  distribution-anchoring fix on CIFAR-100-C.

**Prediction churn.** Between any two seeds, 6.65–6.86% (CIFAR-10-C) and 17.98–18.39% (CIFAR-100-C) of Ramen's
individual predictions change, even though the averages barely move.
- Per-sample analyses (which samples break, failure grids, t-SNE selections) are seed-specific in detail, though
  their aggregates are stable.
- Any per-sample signal for Direction 2 has to hold up under this churn.

---

## 2. The proposed paper story

> **Ramen's class balancing over-corrects.** Zero-shot CLIP has a strong class bias. Ramen's prediction-balanced
> retrieval removes much of it, but nothing anchors predictions to the true class distribution, so neglected
> classes overshoot (deer, truck, tractor), neighbouring classes get drained or absorbed (frog; pickup truck, lawn
> mower), and about 5% of correct zero-shot predictions break, in a way confidence can't detect. The adapted model
> is also overconfident.

Testable pieces:
1. The overshoot comes from the balancing and/or from queue contamination (Direction 1 diagnostics).
2. Anchoring the prediction marginal reduces overshoot and broken samples without losing the fixes (Direction 1
   fixes).
3. A signal stronger than confidence can pick per sample between zero-shot and adapted predictions (Direction 2).
   Ceiling: +5.7 pts (CIFAR-10-C), +4.5 pts (CIFAR-100-C).

---

## 3. Directions, in priority order

### Direction 0 — Noise floor (done)

**Status.** Done (Section 1.7). Seeds 0–4 of the baseline are cached for both datasets in `new_experiments/results/`.

**How to use it for every candidate fix:**
- Run the fix at the **same seeds 0–4**, on an **RTX 3090**. Numbers from other GPU models aren't comparable.
- Compare **seed by seed** against the cached baseline. A paired difference is less noisy than comparing two means.
- Score it with `exp.summary_metrics()`, which computes exactly the metrics in the Section 1.7 table.
- A gain counts only above that metric's "a fix must beat" value, **on both datasets**. Headline thresholds:
  **+0.31 pts** (CIFAR-10-C) and **+0.21 pts** (CIFAR-100-C) average accuracy.
- For quick iteration, seed 0 alone is enough to discard clearly bad ideas. Run all 5 seeds before claiming a gain.

---

### Direction 1 — Over-correction of the class distribution (main direction)

**Evidence.** Section 1.5: deer −31% → +28%, truck −39% → +10%, frog drained by 30%, tractor more than doubled;
12 classes worse on CIFAR-100-C; errors becoming more fine-grained. Section 1.7: all of this is stable across 5
seeds. On CIFAR-100-C, many classes also stay deeply under-predicted, so the class-share gap there is 3× larger.

**Two candidate mechanisms** (not mutually exclusive):
- **(M1) Balancing has no stopping point.** In `Ramen.forward` ([methods/Ramen.py](methods/Ramen.py), lines
  136–154), every non-empty class queue contributes its top-k weighted gradients, and the sum is divided by the
  number of classes. Under-predicted classes get the same pull as everything else regardless of how often they are
  already predicted. With SignSGD only the sign of the summed gradient matters, so a consistent push towards one
  class wins easily.
- **(M2) Contaminated queues.** Samples are filed by their zero-shot prediction (`init_preds`, 61.4% accurate on
  CIFAR-10-C). Queues of over-predicted classes hold many wrong entries, whose gradients then pull further samples
  into that class.

**Step 1a — Memory diagnostics (logging only; decides between M1 and M2).** Phase 1 of the PC2 handoff, behind a
`--diag` flag. Log per batch:
- **per-queue purity** (fraction of entries with a correct pseudo-label), per class;
- the **gradient share per queue** in each sample's aggregated gradient;
- the **running predicted marginal** of the adapted model vs the true marginal.

M2 predicts that the deer/truck/tractor queues are the least pure. M1 predicts that overshoot appears even with
pure queues: compare against an oracle run that files samples under their *true* label (diagnostic only).

**Step 1b — Fix candidates**, each behind a config key, off by default:
- **Candidate A: file samples under the adapted prediction** (targets M2). It is 72.6% accurate instead of 61.4%.
  The cached gradient stays valid because weights reset to base every batch. Pitfalls (empty memory on the first
  batch, losing self-retrieval) and the A0/A1 ablation split are in the PC2 handoff, Section 6.
- **Marginal anchoring by prior correction** (targets M1). Keep an EMA of the adapted model's predicted class
  marginal π̂ and correct `logits − λ·log(π̂ / π_target)`, with π_target uniform. This uses *Ramen's own* running
  marginal, so it pulls deer/truck down and frog back up instead of re-applying zero-shot's correction.
  - It works at the output, so it is compatible with the gradient cache. A batch-level diversity loss (SHOT-style)
    is not, because its per-sample gradient depends on the batch marginal and doesn't decompose into cached
    per-sample gradients.
  - Assumes a roughly balanced label distribution, which holds for CIFAR-C and ImageNet-C. Check DomainNet's class
    balance before relying on it there.
  - Evaluable post hoc if per-sample probabilities are saved: add `save_probs` to `RamenExperiment.run`
    (CIFAR-100-C: 150k × 100 × fp16 ≈ 30 MB).
- **Balance-aware aggregation** (targets M1). Down-weight a class queue's contribution in proportion to how
  over-predicted that class currently is (from π̂), instead of the flat `/ num_class_sum`.

**Success criterion**, at seeds 0–4, on both datasets, against the Section 1.7 thresholds:
- class-share gap (TV) down by more than 0.003 (CIFAR-10-C) / 0.005 (CIFAR-100-C), with smaller per-class
  overshoot (`plot_prediction_bias`);
- broken % down by more than 0.20 / 0.16, and fixed % not down by more than 0.19 / 0.16;
- average accuracy up by more than **0.31 / 0.21 pts**.

Also watch the CIFAR-100-C classes Ramen currently hurts (keyboard, worm, willow tree, woman).

**Hyperparameter discipline.** λ and any new knob must be fixed across datasets, as Ramen does with its own.

---

### Direction 2 — Per-sample selective adaptation ("do no harm")

**Evidence.** 5.7% / 4.5% of samples broken (1.4); headroom to the per-sample ceiling of +5.7 / +4.5 pts; ECE
roughly 3× zero-shot's on CIFAR-10-C (1.6).

**Ruled out.** Gating on zero-shot confidence (Appendix A).

**Next.** Find a signal with much higher AUROC than 0.61–0.67. Since Ramen is right 3× as often as zero-shot when
they disagree, a selector needs strong separation to gain anything. Candidates, which need richer logging (re-run
with `--diag`):
- **Support-set agreement:** fraction of the retrieved top-k whose pseudo-label matches the query's zero-shot
  prediction.
- **Isolation:** mean distance to the retrieved neighbours.
- **Logit shift:** ‖z_adapted − z_zeroshot‖ per sample.
- **Augmentation consistency:** prediction stability under flips/crops (MEMO-style). It costs extra forward
  passes, so report runtime against Ramen's.
- **Class-marginal signal:** whether the adapted prediction moves the sample *into* a currently over-predicted
  class. This links Direction 2 to Direction 1, since many broken samples may be exactly the overshoot.

Measure each signal's AUROC offline on the disagreement set before building a selector. Check it at more than one
seed: 7% (CIFAR-10-C) and 18% (CIFAR-100-C) of individual predictions change between seeds (Section 1.7). A signal
tuned on one seed's disagreement set may not carry over.

**Secondary contribution: calibration.** Logit interpolation `(1−γ)·z_zs + γ·z_adapted` will likely cut ECE back
towards zero-shot's at little accuracy cost. Check whether the Ramen paper reports calibration. If it doesn't, "same
or better accuracy, much better calibration" is a legitimate claim.

---

### Direction 3 — Adaptive retrieval hyperparameters (conditional on the PC2 sweeps)

- **Adaptive β (Candidate C):** per-query scale such as `exp(−d / median(d_topk))`. Pursue only if the β sweep
  (0 / 2.5 / 5 / 10 / 20) shows accuracy still rising past 5.
- **Adaptive k / K:** pursue only if the topk or capacity sweeps show effects above the noise floor.

---

### Direction 4 — Learned retrieval metric / learned aggregation (deferred)

Angles 1–2 of the brief. Heavy, and they need training data without leaking test labels. Pursue only if Direction 1
diagnostics point at retrieval (low same-domain rate or low support-set purity *with* clean queues).

---

### Direction 5 — Full benchmark protocol (required, last)

ImageNet-C, DomainNet, ViT-L/14, ResNet-101, all on the Ramen paper's protocol with one fixed hyperparameter set.
Start once a fix beats the noise floor on both CIFAR datasets.

---

## 4. Order of work

```
D0 seeds 0-4 (noise floor)              <- DONE: thresholds +0.31 / +0.21 pts average accuracy
 │
D1a memory diagnostics + oracle-label run
 ├─ attractor queues impure (M2) ──> Candidate A (A0/A1)
 ├─ overshoot even with clean queues (M1) ──> marginal anchoring / balance-aware aggregation
 │     (add save_probs first: prior correction can then be tuned post hoc)
 │
D2 log candidate signals with --diag -> offline AUROC -> build a selector only if AUROC clearly > 0.67
 │
PC2 sweeps ──> D3 if β or k/K matter
 │
fix > noise floor on both CIFAR datasets ──> D5
```

Rule: **measure offline before building.** The per-sample cache makes most selection and correction rules evaluable
without re-running the model.

---

## 5. Experiments and figures for the paper

| claim | experiment | figure / table |
|---|---|---|
| Ramen over-corrects zero-shot's class bias | predicted vs true marginal, both datasets | `plot_prediction_bias`; table of TV distance |
| The overshoot absorbs semantic neighbours | composition of over-predicted classes; within-superclass error share | `pred_composition`, `error_granularity` |
| Confusion pre-exists in zero-shot features | t-SNE zero-shot vs adapted | `plot_tsne(..., select='pred')` |
| Mechanism (M1 / M2) | per-queue purity vs overshoot; oracle-label run | scatter: queue purity vs over-prediction |
| Over-correction is not a seed artifact | seeds 0–4 | multi-seed bias plot (`seed_variance.ipynb`) |
| Fix reduces overshoot and improves accuracy | Direction 1 fixes, seeds 0–4, paired with the cached baseline | main table with mean ± std over seeds |
| Ramen is overconfident; fix improves calibration | ECE for all methods | `plot_calibration` |
| Confidence gating can't capture the headroom | gate sweep | Appendix-style table (negative result) |
| Generalises | Direction 5 | full results table |

---

## 6. Risks and open questions

- **Baselines' seed variance.** Measured for Ramen and NoAdapt (which is seed-independent), not for Tent. Run Tent at
  seeds 0–4 before making any claim against it.
- **Per-sample churn.** 7% / 18% of predictions change between seeds (Section 1.7). Per-sample findings (specific
  broken samples, failure grids) shouldn't be cited from a single seed.
- **GPU-to-GPU differences.** Compare only within one GPU model (PC2 handoff rule 1).
- **Balanced-label assumption.** Marginal anchoring assumes balanced classes. Under label shift or temporally
  correlated streams it could hurt. If time allows, test on a class-imbalanced or class-ordered stream.
- **Compute.** Any signal that needs extra forward passes has to be reported against Ramen's runtime, since its
  efficiency is part of its contribution.

---

## Appendix A — The confidence gate

### A.1 What it is and how it would be implemented

"Skip adaptation for a sample when zero-shot confidence ≥ τ." Two code facts:
- In `Ramen.forward` the first `logits` are zero-shot, because weights reset to base after every batch.
- The optimizer is SignSGD ([models/optimizer.py](models/optimizer.py)): update = `−lr · sign(grad)`. Scaling a
  gradient does nothing; a **zero** gradient row means no update, so the prediction is exactly zero-shot. This holds
  while weight decay and momentum are 0, as in all current configs.

```python
retrieved_grads = retrieved_grads_sum / num_class_sum
tau = self.cfg.get('gate_tau')                       # None = baseline, unchanged
if tau is not None:
    zs_conf = logits.detach().float().softmax(-1).max(-1).values
    retrieved_grads[zs_conf >= tau] = 0              # these samples stay zero-shot
self.model.set_by_sample_grad(retrieved_grads)
```

Memory writes happen earlier in `forward`, so gated samples still enter the memory and every other sample is
unaffected. That is why `RamenExperiment.gate_sweep()` can compute this gate exactly from cached per-sample results.

A soft version (per-sample step size) has to scale the parameter change after `step_and_zero_grad()`:
`w_i ← w_0 + γ_i (w_i − w_0)` for every `BySampleLayerNorm` / `BySampleBatchNorm`.

### A.2 Results (full data)

CIFAR-10-C (Ramen 72.61, ceiling 78.28):

| τ | gated | NoAdapt acc in gate | Ramen acc in gate | total acc | broken saved / fixed lost |
|---|---|---|---|---|---|
| 0.5 | 55.0% | 83.23 | 86.01 | 71.08 | 1,986 / 4,283 |
| 0.7 | 39.0% | 90.98 | 91.48 | 72.42 | 625 / 914 |
| 0.8 | 31.6% | 93.99 | 94.00 | 72.61 | 359 / 366 |
| 0.9 | 22.1% | 97.15 | 96.89 | 72.67 | 170 / 83 |
| 0.95 | 14.3% | 98.73 | 98.43 | 72.65 | 83 / 19 |

CIFAR-100-C (Ramen 46.10, ceiling 50.63):

| τ | gated | total acc | broken saved / fixed lost |
|---|---|---|---|
| 0.5 | 23.5% | 45.86 | 352 / 708 |
| 0.7 | 14.2% | 46.09 | 62 / 73 |
| 0.8 | 10.3% | 46.10 | 34 / 25 |
| 0.9 | 6.1% | 46.10 | 14 / 7 |

At high τ, Ramen and zero-shot already agree, so the gate changes almost nothing. At low τ it gives up more fixes
than it saves. The harm sits at low zero-shot confidence, where broken and fixed samples look alike.
