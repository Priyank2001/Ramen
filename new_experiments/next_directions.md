# Next directions for Ramen (from the 2026-10-06 confusion analysis)

This builds on [src/ramen-research-directions.md](../src/ramen-research-directions.md). That file has the full
findings, the noise floor and the older plan. This one records what the confusion-pair and step-size analysis added,
and the directions that follow, in the order to try them.

Setup throughout: mixed stream, CIFAR-10-C (ViT-B/32) and CIFAR-100-C (ViT-B/16), seed 0 unless stated.
A gain counts only above the noise floor on **both** datasets: **+0.31 pts** (CIFAR-10-C) and **+0.21 pts**
(CIFAR-100-C) average accuracy, at seeds 0-4, compared seed by seed against the cached baseline.

---

## 0. Already ruled out (don't repeat)

| idea | result |
|---|---|
| Zero-shot anchor (RamenAnchor options 1-4), incl. "keep zero-shot when confident" | nothing beats Ramen; harmful and helpful flips have the same margin profile |
| Contamination-aware weighting (RamenClean) | support purity 48.8 → 76.6% but accuracy down (−0.7 / −2.9 on CIFAR-100-C): support shrinks |
| Normalized SGD instead of SignSGD (RamenNSGD) | −3.8 / −5.8 pts; the sign step matters |
| Noise corruptions contaminate the shared cache | single vs mixed within ±0.5 pts |
| Ramen degrades along the stream | gain over NoAdapt slightly grows by stream third |

---

## 1. New evidence

### 1.1 Ramen moves zero-shot's class bias instead of removing it

CIFAR-10-C, share of all predictions (true share is 10% for every class):

| class | NoAdapt | Ramen |
|---|---|---|
| bird | 18.5% | 9.8% |
| deer | 6.9% | 12.8% |
| truck | 6.1% | 11.0% |

The confusions Ramen increases most all end in the newly over-predicted classes (% of the true class):

| pair (true → predicted) | NoAdapt | Ramen | change |
|---|---|---|---|
| frog → deer | 4.1 | 15.9 | +11.7 |
| bird → deer | 4.1 | 12.7 | +8.6 |
| automobile → truck | 7.1 | 14.3 | +7.2 |
| cat → deer | 3.1 | 8.1 | +5.0 |
| dog → deer | 2.7 | 6.0 | +3.3 |

truck ↔ automobile flips direction: truck → automobile drops 19.7 → 7.4% while automobile → truck rises 7.1 → 14.3%.
Ramen moves the boundary between the two classes rather than separating them.

### 1.2 Unchanged error rates hide large churn

Per pair: errors made by both methods, errors Ramen removes, errors Ramen adds (CIFAR-10-C, image counts).

| pair | NoAdapt | Ramen | both | removed | added |
|---|---|---|---|---|---|
| frog → cat | 1625 | 1621 | 786 | 839 | 835 |
| cat → dog | 2533 | 2048 | 1083 | 1450 | 965 |
| dog → cat | 2138 | 1958 | 1085 | 1053 | 873 |
| frog → deer | 618 | 2379 | 397 | 221 | 1982 |
| automobile → truck | 1064 | 2150 | 856 | 208 | 1294 |

frog → cat looks untouched, yet fewer than half its errors are shared. Every pair is both helped and hurt; the
`confused_pairs` table shows only the net.

### 1.3 The step is a fixed ±0.01 per parameter, and it never reaches the cache

- SignSGD with `lr: 0.01` on all four datasets, no momentum or weight decay
  ([models/optimizer.py](../src/models/optimizer.py)). Every adapted LayerNorm parameter moves by exactly ±0.01,
  however strong or weak the retrieved evidence is.
- One step per sample, then `reset_parameters()`. Each sample is at most 0.01 from zero-shot in every parameter.

**Ramen's own predictions never feed back into the stream.** Cache keys are zero-shot features, values are gradients
at the base weights, and entries are filed under the zero-shot prediction
([methods/Ramen.py:113-131](../src/methods/Ramen.py#L113-L131)). Neither the step size nor anything applied to
Ramen's output changes what later samples retrieve. So **any output-level correction, and any change of step size, can
be evaluated exactly offline**, from one logged run, without re-running the stream.

---

## 2. Directions, in order

### Direction 1 — Stop the overshoot at the output (prior correction) — **in progress**

**Idea.** Keep a running (EMA) estimate π̂ of the method's predicted class marginal and correct
`logit' = log p − λ · log(π̂ / π_target)`, with π_target uniform. Classes Ramen currently over-predicts (deer, truck)
are pushed down, drained ones (frog) pulled up.

**Why.** 1.1: the over-prediction moves from bird to deer and truck, and nothing stops it at the true share. This
targets mechanism M1 ("balancing has no stopping point") from the older plan.

**How.** Offline and exact (1.3), from the saved per-sample probabilities
(`results/*_Ramen_probs.npy`, both datasets). Code: [marginal_correction.py](marginal_correction.py).
- Sweep λ × EMA momentum; a whole-stream marginal (non-causal) as reference.
- **Control:** the same correction on NoAdapt. If zero-shot + correction gets close to Ramen, most of Ramen's gain
  is debiasing, which changes the paper story.
- Caveats: assumes a roughly balanced label distribution (true for CIFAR-C / ImageNet-C; check DomainNet).
  π̂ is global, so a bias that differs by corruption is only corrected on average.

**Success.** Above the noise floor on both datasets with one fixed (λ, momentum); class-share gap (TV) down; broken %
down without losing fixed %.

**Results, seed 0 (2026-10-06).** Tables: `results/marginal_correction_<dataset>_s0_*.csv`.

Gain over Ramen in average accuracy (pts), soft marginal, causal EMA unless noted:

| λ | C10 m=0.9 | C10 m=0.99 | C10 whole stream | C100 m=0.9 | C100 m=0.99 | C100 whole stream |
|---|---|---|---|---|---|---|
| 1 | +0.33 | +0.30 | +0.33 | +1.13 | +1.11 | +1.16 |
| **2** | +0.47 | **+0.49** | +0.53 | +1.02 | **+1.45** | +1.48 |
| 3 | +0.48 | +0.60 | +0.55 | −0.77 | +0.79 | +0.78 |
| 4 | +0.29 | +0.48 | +0.45 | −4.78 | −1.06 | −1.29 |
| 6 | −0.62 | −0.16 | −0.24 | −17.6 | −8.36 | −9.54 |

**Best shared setting: λ = 2, m = 0.99 (soft): +0.49 (CIFAR-10-C) and +1.45 (CIFAR-100-C),** both above the noise
floor (+0.31 / +0.21). The causal EMA is within 0.04 pts of the whole-stream marginal, so estimating π̂ online costs
almost nothing. The 'hard' (argmax-count) estimate is no better and less stable at m=0.9.

What changes at λ = 2, m = 0.99:

| | CIFAR-10-C Ramen → corrected | CIFAR-100-C Ramen → corrected |
|---|---|---|
| avg acc | 72.61 → 73.10 | 46.10 → 47.54 |
| TV(pred, true) | 0.054 → 0.030 | 0.171 → 0.087 |
| fixed % vs NoAdapt | 16.92 → 16.94 | 13.53 → 15.24 |
| broken % vs NoAdapt | 5.67 → 5.19 | 4.53 → 4.79 |
| ECE | 8.31 → 8.05 | 9.96 → 7.97 |
| overshoot | deer +27.7 → +11.2%, truck +9.8 → +5.1%, frog −30.1 → −6.6% | tractor +70 → +18%, oak +105 → −8%, possum +69 → −18% |
| samples changed vs Ramen | +1.76% fixed, −1.26% broken | +4.54% fixed, −3.09% broken |

- **Control (NoAdapt + the same correction):** best +1.2 (CIFAR-10-C) / +1.7 pts (CIFAR-100-C), against Ramen's
  +11.3 / +9.0 over NoAdapt. Ramen's gain is **not** mostly debiasing, and the correction adds on top of it.
- **Per corruption (λ=2):** CIFAR-100-C improves on 14 of 15 (motion blur 0.00). CIFAR-10-C loses a little on
  impulse noise (−0.41), glass blur (−0.36) and jpeg (−0.22), and gains most on fog (+1.51), zoom / defocus blur
  (+1.1). A single global π̂ can't fit a bias that differs by corruption.
- **Over-shoot the other way:** on CIFAR-100-C a few under-predicted classes are pushed past their true share
  (leopard, lobster). On CIFAR-10-C, frog flips from drained to over-predicted at λ ≥ 3. That is why accuracy turns
  down past λ = 2–3.

**Seeds 0–4 (2026-10-07), fixed λ = 2, m = 0.99, soft.** Paired gain over Ramen on the same seed.
Table: `results/marginal_correction_seeds0-4.csv`. The re-runs that saved probabilities for seeds 1–4 reproduce the
cached Ramen predictions exactly.

| | CIFAR-10-C | CIFAR-100-C |
|---|---|---|
| gain per seed (0–4) | +0.49 / +0.57 / +0.59 / +0.55 / +0.63 | +1.45 / +1.52 / +1.53 / +1.48 / +1.52 |
| mean ± std | **+0.57 ± 0.05** | **+1.50 ± 0.03** |
| noise floor | +0.31 | +0.21 |
| every seed above floor | yes | yes |
| noise corruptions gain | +0.14 to +0.48 (floor for this subset +0.65) | +1.49 to +1.78 |
| other corruptions gain | +0.58 to +0.69 | +1.37 to +1.53 |
| TV(pred, true) | 0.054 → 0.029 | 0.170 → 0.086 |
| broken % vs NoAdapt | 5.61 → 5.12 | 4.54 → 4.78 |
| fixed % vs NoAdapt | 16.88 → 16.95 | 13.55 → 15.28 |
| ECE | 8.35 → 8.02 | 9.90 → 7.85 |

λ = 3 is slightly better on CIFAR-10-C (+0.63) but much worse on CIFAR-100-C (+0.80), so λ = 2 stays the one
setting for both. On CIFAR-100-C the correction breaks a few more zero-shot-correct samples but fixes many more.

**Caveats still open.**
1. λ was picked on seed 0. Seeds 1–4 are new stream orders of the **same** test images, so this rules out
   stream-order luck, not overfitting to CIFAR. The real held-out test is ImageNet-C / DomainNet with λ = 2 fixed.
2. On CIFAR-10-C noise corruptions the gain is below that subset's own floor; the gain there is not established.
3. The correction assumes a uniform label marginal. Check DomainNet's class balance first.

**Open follow-ups.** (a) Per-domain or per-cluster π̂ (domains unknown at test time; cluster the zero-shot features)
for the corruptions where the global correction hurts. (b) Balance-aware aggregation, below.

**Next step if it helps:** balance-aware aggregation (down-weight each class queue's contribution by how
over-predicted the class is). That changes the update, so it needs full re-runs.

### Direction 2 — Per-sample step size

**Idea.** Let the step size vary per sample instead of a fixed ±0.01.

**Why.** 1.2: half the errors in a stable-looking pair are new. One fixed step helps some samples and pushes others
across the boundary.

**How.** Exact offline (1.3). One logged run that saves each sample's aggregated retrieved gradient; then evaluate:
1. accuracy at global step sizes 0, 0.0025, 0.005, 0.01, 0.02, 0.04 (is 0.01 the best single value?);
2. the **oracle**: accuracy if each sample used its best step size from that grid.

A small oracle gap means drop this. A large one means look for a predictor of the right step: gradient-sign agreement
among the retrieved neighbours, neighbour distance, total retrieval weight.

### Direction 3 — A better signal for when to adapt

Zero-shot confidence failed (AUROC 0.61–0.67). Ceiling: +5.7 / +4.5 pts. Candidate signals, scored by AUROC offline
on the disagreement set before building any selector, at more than one seed:
- label agreement in the retrieved support set;
- distance to the retrieved neighbours;
- logit shift ‖z_adapted − z_zeroshot‖;
- **whether adaptation moves the sample into a currently over-predicted class** (links to Direction 1: many broken
  samples may be the deer / truck overshoot).

### Direction 4 — File cache entries under the adapted prediction (Candidate A)

Entries are filed under the zero-shot label (61% accurate); Ramen's is 73%. RamenClean's result makes label noise in
the cache a less likely cause, so this is lower priority, but it is a different fix and untested. Ablation split
(A0 delayed write / A1 adapted label) and pitfalls are in [src/ramen-pc2-handoff.md](../src/ramen-pc2-handoff.md),
Section 6.

### Direction 5 — Calibration (secondary result)

Ramen's ECE is about 3× zero-shot's on CIFAR-10-C. Blending zero-shot and adapted logits will likely recover most of
it at little accuracy cost. Worth claiming if the Ramen paper doesn't report calibration. Also exact offline.

---

## 3. Status

| direction | status |
|---|---|
| 1 prior correction | confirmed on seeds 0–4 (2026-10-07): +0.57 ± 0.05 / +1.50 ± 0.03 at λ=2, m=0.99; next: per-cluster π̂, other datasets |
| 2 per-sample step size | seed 0 done (2026-10-09, `step_size_sweep.py`): 0.01 is the best global step on both datasets; best step per corruption gains +0.14 / +0.00; per-sample oracle 84.25 / 57.99 vs Ramen 72.61 / 46.10 (zero-shot-or-Ramen ceiling 78.28 / 50.62). Needs a per-sample signal |
| 3 selection signal | not started |
| 4 adapted-label filing | not started |
| 5 calibration | not started |
