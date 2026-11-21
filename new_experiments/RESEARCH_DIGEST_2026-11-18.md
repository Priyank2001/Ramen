# Ramen test-time adaptation: research digest (as of 2026-11-18)

## 0. What this document is and what we want from you

This is a complete digest of about six weeks of experiments on **Ramen**, a test-time adaptation (TTA) method for
CLIP zero-shot classification. It covers what the method does, the evaluation protocol, every confirmed result,
the mechanism analyses, the oracle ceilings and every dead end, with the reason each one failed.

**Your task:** read all of it, then propose **new research directions** that are consistent with the observations.
Please:

1. Do not re-propose a dead end from Section 7 unless you name the specific new information source or mechanism that
   gets around the stated reason it failed.
2. For each direction, state: the hypothesis, which observation(s) in this document motivate it (cite section
   numbers), the cheapest decisive experiment (prefer offline replay on cached outputs, Section 9), the oracle or
   ceiling that bounds it, and a pre-registered pass/fail criterion in terms of the noise floors in Section 2.4.
3. Rank directions by (expected gain × probability of success) / cost, and say which ones are *analysis* (explaining
   Ramen) versus *method* (beating Ramen).
4. Flag anything in this document that looks internally inconsistent or suspicious; we would rather know.

---

## 1. The method: Ramen

Ramen adapts a frozen CLIP image encoder **per sample** with a single update step that uses gradients retrieved from a
memory of earlier test samples. The model is reset after every sample, so nothing accumulates in the weights; all
state lives in the memory.

For each incoming batch (processed per sample):

1. **Featurize:** compute the CLIP image embedding `z` (512-d for ViT-B/16 and B/32) and zero-shot logits
   `100 · z · Wᵀ` against the class text embeddings `W`. The zero-shot prediction is `ŷ₀ = argmax`.
2. **Per-sample gradient:** compute the gradient of the softmax entropy of the zero-shot logits with respect to all
   LayerNorm affine parameters (52 LayerNorms × (768 weight + 768 bias) = **39,936-d**).
3. **Write (before retrieval):** store `(key = z, value = entropy gradient, entropy H, priority = arrival counter)`
   in a **per-class queue indexed by the zero-shot prediction ŷ₀** (one queue per class, capacity K). When a queue
   is full, the oldest entry is evicted (FIFO by arrival counter). The query itself is therefore in its own queue.
4. **Retrieve:** for **every** non-empty class queue, take the top-k nearest entries to `z` (Euclidean distance on
   embeddings), weight each by `exp(−H) · exp(−β · dist)`, and sum within the queue. Then **average the per-queue
   sums over all non-empty queues** (each class queue gets an equal vote regardless of relevance).
5. **Update:** set the LayerNorm gradient to this retrieved aggregate, take **one SignSGD step** (lr 0.01).
6. **Predict:** run the adapted model on the same image and output its logits. Then reset the parameters.

Per-dataset hyperparameters (paper defaults, unchanged by us):

| dataset | backbone | K (queue capacity) | top-k | β |
|---|---|---|---|---|
| ImageNet-C (5K subset) | ViT-B/16 | 75 | 1 | 0 |
| DomainNet (full, 345 classes) | ViT-B/32 | 300 | 10 | 5 |
| EuroSAT | ViT-B/16 | 7,500 | (dataset cfg) | (dataset cfg) |
| CIFAR-10-C | ViT-B/32 | (dataset cfg) | | |
| CIFAR-100-C | ViT-B/16 | (dataset cfg) | | |

Key structural facts that matter for interpretation:

- **The memory is independent of the update.** What is written depends only on zero-shot outputs, never on Ramen's
  own adapted outputs. So output post-processing is exactly reproducible offline, and so are step-size changes
  (one pass over cached outputs).
- **Equal per-queue voting** means that with 345 or 1,000 classes, each update is a near-uniform average over
  hundreds of queues. The true-class queue typically gets 1–4% of the update weight.
- **fp16 priority bug (in the original code):** priorities are stored as fp16 and become `inf` after 65,520 inserts.
  After that, full queues accept one more round of replacements and then **freeze**. On DomainNet only 28% of the
  596k-image stream ever enters the memory. A `priority_fp32` flag fixes it. It changes accuracy by ~0 on shuffled
  (mixed) streams and by +0.22 on DomainNet sequential streams.

---

## 2. Experimental setup and protocol

### 2.1 Datasets and streams

| dataset | classes | domains | images | notes |
|---|---|---|---|---|
| CIFAR-10-C | 10 | 15 corruptions, sev 5 | | first experiments |
| CIFAR-100-C | 100 | 15 corruptions, sev 5 | | |
| ImageNet-C 5K (ImageNetC5K) | 1,000 | 15 corruptions, sev 5 | 75,000 (the same 5,000 photos × 15 corruptions) | per-class counts 0–12; 997 of 1,000 classes present |
| DomainNet (full) | 345 | 6 (clipart, infograph, painting, quickdraw, real, sketch) | 596,006 | strongly non-uniform class marginal (878–2,799 per class) |
| EuroSAT | 10 | 1 (satellite RGB, 64×64) | 27,000 | prompt "a centered satellite photo of {}." |
| ImageNet-R/A/Sketch ("ImageNetVariants") | 1,000-way head | 3 | | **held-out set**; every domain mapped to the 1,000-class head, so R/A numbers are not comparable to the usual 200-class protocol. 314 classes hold 60.5% of images (TV to uniform 0.31) |

- **Mixed stream** (default): all domains shuffled together, seeded.
- **Sequential stream:** domains in order, each shuffled, memory never reset.
- **Reset stream:** sequential, but the memory is reset at each domain boundary (an oracle on boundaries).

### 2.2 Baselines (mixed stream, domain-averaged accuracy, seed 0 unless noted)

| dataset | zero-shot (NoAdapt) | Ramen | gain |
|---|---|---|---|
| CIFAR-10-C (B/32) | — | 72.61 | — |
| CIFAR-100-C (B/16) | — | 46.10 | — |
| ImageNet-C 5K (B/16) | 27.29 | 33.08 | +5.79 (all 15 corruptions up; contrast +10.7, brightness +0.9) |
| DomainNet (B/32) | 55.37 | 56.80 | +1.43 (quickdraw +4.2 and 71.8% of the net gain; real +0.15) |
| EuroSAT (B/16), seeds 0–4 | 48.07 | 62.63 | +14.36 ± 0.30 |
| ImageNet-R/A/Sketch (B/16) | 46.36 | 47.84 | +1.48 |

Ramen changes 21.8% of predictions on DomainNet. Rank correlation between zero-shot accuracy and Ramen's gain across
domains is strongly negative (Spearman: C10 −0.94, ImageNet-C −0.71, DomainNet −0.71, C100 −0.32). It helps most
where zero-shot is worst.

### 2.3 Compute

- 3 × RTX 3090 (24 GB). ViT-L/14 does not fit, so all results use ViT-B.
- One Ramen run takes ~7 min (ImageNet-C), ~10 min (R/A/Sketch) or ~37–40 min (full DomainNet).
- Cached per-sample outputs exist for every run: predictions, zero-shot probabilities, Ramen probabilities and
  features. Some runs also record embeddings before and after adaptation and queue membership logs.

### 2.4 Statistical protocol

- **Noise floor** = 2 × (max − min) of Ramen's domain-averaged accuracy over seeds 0–4 (the range, not the std):
  C10 0.31, C100 0.21, ImageNet-C 0.53–0.54, DomainNet 0.11, EuroSAT 1.51, R/A/Sketch 0.48. Sequential-stream
  floors are separate (ImageNet-C 0.37, DomainNet 0.13, R/A/Sketch 0.40).
- A claim needs **seeds 0–4**, with the mean gain above the floor. The pass/fail rule is pre-registered
  (PREREGISTRATION.md files) before the runs.
- **Held-out discipline:** settings tuned on ImageNet-C or DomainNet are then tested on datasets they were not tuned
  on. ImageNet-R/A/Sketch was the clean held-out set, but it has now been used, so anything new on it is post-hoc.
- Random-rate controls accompany every oracle (for example, hide random entries at the same rate as the oracle).
- Leakage control: ImageNetC5K reuses each photo across the 15 corruptions, so any cache carrying label
  information must exclude other copies of the query's photo (`exclude_groups: base_image`).

---

## 3. Confirmed positive results (things that beat Ramen)

### 3.1 Zero-shot-weighted active set, temperature T = 3 ("T3") — the only general improvement

Instead of an equal vote per class queue, each queue's contribution is scaled by the query's zero-shot probability
for that class, `p_c^(1/T)`, with T = 3.

| setting | T3 − Ramen (seeds 0–4) | floor | verdict |
|---|---|---|---|
| ImageNet-C mixed (tuned here) | +0.62 ± 0.11, every seed positive, paired t 12.9 | 0.54 | confirmed |
| DomainNet mixed (held-out) | +0.24 ± 0.01 (5/6 domains; quickdraw −0.05) | 0.11 | confirmed |
| ImageNet-R/A/Sketch (held-out) | +0.59 ± 0.04, 5/5 above floor | 0.48 | confirmed |
| CIFAR-100-C | −0.15 ± 0.08 | 0.21 | no effect |
| ImageNet-C sequential | +0.52 ± 0.06 | 0.37 | holds |
| R/A/Sketch sequential | +0.66 ± 0.11 | 0.40 | holds |
| **DomainNet sequential** | **−0.07 ± 0.02** (fp32 priorities: −0.12; negative in all 6 domains, including the first) | 0.13 | **fails** |

- The temperature sweep on CIFAR-100-C is monotone: sharper hurts, and flat tends to Ramen. **No T helps on
  ≤ 100 classes.** It is a large-label-space effect.
- Hard top-m queue selection hurts (support shrinks; at m = 2 the self-weight reaches 0.54).
- On ImageNet-C, T3 wins 12/15 corruptions (shot noise +2.1, gaussian +1.6) and loses elastic and jpeg (−0.5 on all
  5 seeds).
- Untested hypothesis for the DomainNet sequential failure: T3 helps only when the memory mixes domains.

### 3.2 Output prior correction ("pc"): only on known-balanced label spaces

`logit − λ · log(EMA_marginal / uniform)`, with momentum m = 0.99. It is exact offline.

- Small, balanced label spaces (seeds 0–4): λ = 2 gives C10 +0.57, C100 +1.50, EuroSAT +2.79. λ = 1 gives C10 +0.35
  (thin), C100 +1.15, EuroSAT +2.13.
- Stacked with causal kNN output smoothing (10 nearest earlier images, 50/50 mix): EuroSAT +7.02, C100 +2.07.
- Large label spaces: T3 + pc(λ = 1) gives ImageNet-C +1.13, DomainNet +0.41 (post-hoc). **The pre-registered
  held-out test on R/A/Sketch failed: −1.15 ± 0.09 (0/5).** Ramen + pc(λ = 1) there is −2.58, and every λ hurts.
- Cause: the uniform target is wrong on non-uniform marginals. Ramen already under-predicts the 314 big classes
  (49.0% predicted vs 60.5% true), and pc toward uniform drops that to 44.3%.

### 3.3 Small or within-noise positives (consistent but below the floor)

- NSS-vector quality matching in the active set (36-d BRISQUE/NIQE features, causally whitened, λ = 0.1),
  ImageNet-C seeds 0–4: +0.22 to +0.29, 5/5 positive, but below the 0.54 floor. It works by raising the
  same-corruption share of retrieved entries. Scalar quality scores reach only +0.10 because they merge distortion
  types. Stacked with T3 it adds about +0.15 (seed 0 only).
- kNN output smoothing on top of Ramen on R/A/Sketch: +0.43 ± 0.12 (post-hoc).
- Averaging output probabilities over steps {0.005 … 0.02}: +0.08 / +0.34 on C10 / C100 (single seed, 3× compute).

---

## 4. Oracle ceilings (true labels or side information used)

| oracle | DomainNet | ImageNet-C | EuroSAT | other |
|---|---|---|---|---|
| Hide **confident-wrong** memory entries (wrong and entropy below the running median), seeds 0–4 | +0.17 ± 0.02 (hides 8.3% of memory) | +0.24 ± 0.07 (below floor) | | controls −0.03 / −0.24 |
| Hide **all wrong** entries | +0.15 | −0.38 (hides 68% of memory) | | |
| **File by true label** (100% pure queues), entropy values | **−1.50** (quickdraw −6.49) | −0.71 | −6.50 | |
| **Label values**: store the CE-to-true-label gradient as the value (written after the image is predicted, so there is no self-leak), file by zero-shot | **+1.70** (every domain up) | +1.51 (all 15 up) | **+26.70** (89.33%) | |
| True-label filing + label values | −0.25 | −0.91 | +8.09 | |
| Label values + **global top-k retrieval** instead of the per-queue vote | k10 **+2.86**, k100 +2.54 | no gain once photo copies are excluded (k10 −1.00, k1000 +0.95) | k100 +27.59 | entropy values + global k100: +0.04 DN, −1.57 IN-C, −7.51 ES |
| Prior correction toward the **true** marginal (on T3) | +0.40 | +1.96 (not attainable: sampling noise) | | R/A/Sketch **+2.75**; "structure oracle" (domain sizes + class sets, uniform within a domain) +2.32 |
| Per-domain true-marginal bias on zero-shot | +3.84 (vs Ramen +1.73) | below Ramen | | |
| Per-sample best step size from {0 … 0.04} | | | | C10 84.25 / C100 57.99 vs Ramen 72.61 / 46.10 (zero-shot-or-Ramen oracle 78.28 / 50.62) |
| Reset memory at domain boundaries (sequential stream) | +0.27 (≈ +0.04 with fp32 priorities) | **+0.82 ± 0.05** | | R/A/Sketch +0.16 |
| Domain-mean shift of embeddings with the true domain label (no Ramen) | +0.86 (60% of Ramen's gain) | +1.58 (27%) | | |
| Class GMM/LDA fitted on test embeddings with true labels | lda +2.99 over zero-shot | +35.76 | +46.92 | |

Label-value dose response (DomainNet / EuroSAT): mixing true-label values for 25% of entries gives +0.75 / +17.55;
50% gives +1.29 / +24.75.

**Takeaways from the ceilings:**

- Information about labels helps **only through the values**, never through filing. Pure filing always hurts.
- The per-queue equal vote caps even a perfect-label cache on DomainNet (global retrieval adds +1.16 on top, mostly
  on quickdraw: 21.56 → 27.04).
- Removing contamination is worth at most ~+0.2.
- The largest available headroom is the **per-sample step / per-sample decision** (C10 +11.6, C100 +11.9 pts),
  which no label-free signal has reached.

---

## 5. Mechanism findings: what Ramen actually does

### 5.1 Updates are diffuse, domain-dominated votes

- ImageNet-C: the active set spans ~1,000 queues; the true-class queue holds <1% of the weight; ~44% of the weight
  comes from the same corruption (which is 7% of the memory). It is effectively a **corruption-level step**.
- DomainNet (β = 5) is more local: the true-class queue holds 1–4%, and 52–61% comes from the same domain. Each update
  spreads over ~235 classes, and the image's own class gets ~2.5% of the weight.
- Under Ramen, the top-weighted retrieved entries are often unrelated classes (maze, banana, iPod at w ≈ 0.02). With
  T3 they are semantically related (for example, related insects).
- Entropy weighting has a scale bias across domains: mean `exp(−H)` is 0.31 for brightness vs 0.04 for contrast.
  Brightness supplies 23% of the top-weighted entries (chance 6.7%), and T3 halves that.
- The cosine between the entropy gradient and the true-label CE gradient is +0.97 when zero-shot is right and −0.31
  when it is wrong (33 of 38 sampled DomainNet images negative).

### 5.2 Fixed vs broken samples look alike

- Retrieval statistics, memory composition and the size of the embedding shift do **not** separate fixed from broken
  images (DomainNet medians 0.044 vs 0.042 vs 0.051 for right→right; within-domain differences 0.001–0.008).
- The exception is EuroSAT: fixed images move more (0.225 vs 0.143).
- Broken images are typically low-margin zero-shot hits tipping to a fine-grained neighbour.
- Broken images lack correctly filed own-class support (0.98% vs 3.58% of the update weight); they are not pulled by
  mis-filed entries (0.14% vs 0.09%).
- Harmful and helpful flips have the same zero-shot margin profile.

### 5.3 De-sinking without repair: Ramen relocates errors on large label spaces

- Zero-shot has domain-specific "attractor" classes (sinks): squiggle and envelope on DomainNet quickdraw, CRT monitor
  on ImageNet-C, worm on CIFAR-100-C, bird on CIFAR-10-C, Highway on EuroSAT. Ramen dissolves them (per-domain
  TV(pred, true) falls on all 5 datasets; ImageNet-C top-5 destination share 13.3% → 3.1%).
- But **sink errors are fixed at the same rate as other errors** (ImageNet-C 12.7% vs 13.8%; DomainNet 7.1% vs 7.4%).
  They are instead moved to another wrong class 1.4–2.1× as often. EuroSAT is the exception (43% vs 31% fixed).
- Sink fate (seed 0, full runs):

  | dataset | main sink | false positives (FPs) under zero-shot | fixed by Ramen | moved to another wrong class |
  |---|---|---|---|---|
  | EuroSAT | Highway | 5,353 | 43% | 744 images |
  | CIFAR-100-C | worm | 4,264 | 26% | 58% |
  | DomainNet | squiggle | 44,623 | 8% | 67% |

- **Wrong → other-wrong churn is the largest flow:** CIFAR-100-C 29%, ImageNet-C 39%, DomainNet 17% of images, vs
  13.6 / 9.6 / 3.5% fixed. On CIFAR-100-C, ImageNet-C and EuroSAT, 56–61% of those moves get closer to the true class
  (prototype cosine); on DomainNet only 42%.
- **Ramen creates new sinks:** DomainNet pillow (1,147 → 6,484 wrong; 6,129 of them quickdraw; zero-shot classes
  envelope 3,570, squiggle 1,248; quickdraw rectangles go to pillow at confidence 1.00), plus teapot, helmet and
  toaster. CIFAR-10-C deer (top error destination in 11/15 corruptions). CIFAR-100-C crab +1,232, tractor +981.
  EuroSAT Herbaceous Vegetation collapses (0.08% predicted vs 11.1% true; recall 6.67% → 0.30%). New sinks usually
  receive images at low confidence (median 0.29–0.46), with exceptions at 0.95+.
- **Not a prior shift:** a per-class logit bias matched to Ramen's own marginal recovers only 3–33% of the gain
  (ImageNet-C 3%) and reproduces 13–35% of its changed predictions.
- Small label spaces "level up" (Spearman ρ(acc₀, gain) ≈ −0.4); ImageNet-C and DomainNet "get richer" (+0.11 /
  +0.21).

### 5.4 The pillow case study (DomainNet): the contamination hypothesis is rejected

- The true-label (pure) cache keeps 33.4% of the images at pillow and fixes only 42. 81% of the images it moves away
  go back to their **zero-shot** class.
- On quickdraw, the pure cache keeps the zero-shot prediction on 66.8% of images vs 48.0% for Ramen, and its
  quickdraw accuracy is 11.05 (zero-shot 13.33, Ramen 17.54). So **a pure cache means less adaptation, not repair.**
- The two caches share only 12–23% of the update weight. The oracle cache's update is 90–99% same-domain and 78–89%
  from zero-shot errors. Ramen's is 62–67% same-domain, with top entries that are confident, correct sketches of
  unrelated classes.
- Ramen's pillow queue is 44% truly pillow and freezes at image 147,057 (fp16).
- Still open: which retrieved entries steer the step toward pillow. This needs leave-one-group-out gradient runs
  on the GPU (~200 images).

### 5.5 Geometry of the embedding shift (DomainNet, 291k non-right→right images)

- The shift is large but almost never along the specific class-pair axis that flipped: median squared share on that
  axis is 0.003–0.016. Yet the small projection is reliably signed correctly (median Δmargin +1.06 to +4.21 logits,
  positive for 73–100% of images).
- Images landing at the same wrong class move coherently (194 of 260 classes above a bootstrap null; resultant length
  up to 0.74 for pillow). This is a **domain effect, not class semantics:** cosine with the class-pair text axis is
  only 0.06–0.21, the most coherent destinations are 89–95% quickdraw, and quickdraw's wrong images have R = 0.63 vs
  0.39–0.46 in other domains.

### 5.6 How much is "domain-level shift"? Partly, not mainly

- Label-free streaming per-cluster mean shift `z' = norm(z − a(μ_g − m))`, using the zero-shot classifier and no
  Ramen: ImageNet-C +0.64 (11% of Ramen's gain), DomainNet +0.43 (30%), EuroSAT negative (clusters ≈ classes).
- With true domains: 27% / 60%. Random groups: ~0.
- Group means explain only 13–22% of Ramen's own squared shift beyond the global mean.
- Verdict: a real partial domain component on DomainNet, but not the main explanation.

### 5.7 Density structure of CLIP test embeddings is domain, not class

A label-free class GMM/EM initialised from text prototypes increases the log-likelihood every iteration while
accuracy falls: DomainNet −11.1, ImageNet-C −4.0, EuroSAT +15.3 but breaks 10% of zero-shot-correct images. The
normalized mutual information (NMI) between components and domains rises (DomainNet vMF 0.10 → 0.28, ImageNet-C
0.11 → 0.42). One mode per class cannot represent a 6-style DomainNet class. Even the true-label LDA oracle gives only
+2.99 on DomainNet.

### 5.8 Memory staleness and capacity

- Stream trend by thirds (seeds 0–4): the ImageNet-C gain grows (+5.0 → +6.2), EuroSAT shrinks (+15.2 → +13.7; the
  memory never evicts there because 10 × 7,500 > 27k), and DomainNet stays flat.
- Sequential ImageNet-C: capacity 75 × 1,000 exceeds 5,000 per corruption, so nothing is evicted and only ~1/n of the
  memory is the current corruption. A boundary reset gains +0.82.
- **New (2026-11-18, seed 0, cache compression):** values can be quantized heavily at almost no cost. int8 and int4
  values are within ±0.05 of Ramen on all three datasets. 1-bit sign values give −0.09 on ImageNet-C and +0.17 on
  EuroSAT (DomainNet still running). Matched-memory controls that instead **shrink the queue capacity**:
  - DomainNet: −0.08 (K 161), −0.22 (K 87), −0.52 (K 32).
  - ImageNet-C: −0.08 (K 40), −0.25 (K 22), −0.73 (K 8).
  - **EuroSAT improves:** +0.13 (K 4,029), +0.42 (K 2,178), **+1.64 (K 789, just above the 1.51 floor; single
    seed).**

  So on EuroSAT a smaller (fresher) memory helps, which is consistent with the stale-memory reading.

---

## 6. Summary of what we believe

1. Ramen's gain comes from a **diffuse, mostly domain-level, sign-quantized step**. It works because the aggregate of
   many confident entropy gradients from same-domain images points in a direction that is, on average, slightly
   margin-improving. It is not class-specific evidence.
2. **Label purity of the memory never buys accuracy** (contamination weighting, write-only-if-agree, true-label
   filing, hide-all-wrong all fail). Support size and self-retrieval matter more.
3. **What the value carries matters a lot**, but only if the target is far more accurate than zero-shot. A
   slightly-better pseudo-label target loses because it overturns correct images.
4. On large label spaces Ramen mostly **relocates** errors (de-sinks without repairing) and creates new sinks.
   Wrong → other-wrong churn is the largest flow and the clearest unexploited target.
5. The equal per-queue vote is a real limit on DomainNet (with label values), but not on ImageNet-C.
6. Label-free information sources tried so far (zero-shot outputs, adapted outputs, kNN votes, EM priors, cluster
   structure, image-quality features) **inherit zero-shot's bias** or encode domain rather than class.
7. The biggest headroom is per-sample: choosing the step size, or choosing between zero-shot and adapted output, per
   sample (≈ +12 pts oracle on CIFAR). No label-free per-sample signal found so far reaches it.

---

## 7. Dead ends, with the reason each failed

| idea | result | why it failed |
|---|---|---|
| Zero-shot anchor / gate (keep zero-shot when the margin or confidence is high) | never beats Ramen, all datasets including EuroSAT | harmful and helpful flips have identical margin profiles |
| Contamination weighting of entries (α·a_j^γ) | purity 48.8 → 76.6%, accuracy −0.7 / −2.9 (C100) | effective support shrinks 158 → 35 → 12 |
| Normalized SGD instead of SignSGD | −3.8 / −5.8 | the sign step matters |
| Global step size or per-corruption step | best global = paper's 0.01; per-corruption +0.14 / +0.00 | the headroom is per-sample only |
| Label-free per-sample step rules (entropy-argmin over steps, majority vote over steps) | −3.57 / −3.18, −0.76 / −0.86 | entropy is not a correctness signal across steps |
| No-reference IQA (MSCN/BRISQUE) as an adapt gate or step signal | AUROC ~0.5, gates −0.2 to −0.4 | quality does not predict correctness |
| Scalar quality score in the active set; quality "prior" mode | +0.10 tiny; prior mode −0.35 to −2.0 | scalar scores merge distortion types |
| Prompt-template selection, support-set classifier, instability → zero-shot fallback, near-synonym override | dead on all 4 datasets | |
| Write after prediction (loses self-retrieval) | −0.60 (C10) | self-retrieval matters |
| Write only if zero-shot == Ramen | −1.04 (C10), purity 61 → 84% | purity is not the bottleneck |
| File under the adapted prediction | C10 −0.24; DN −0.21, IN-C +0.29, ES −6.57 vs timing control | also, the paper files by the zero-shot prediction |
| True-label filing (oracle) | −1.50 DN, −0.71 IN-C, −6.50 ES | a pure cache is less adaptation, not repair |
| More entries per class / global top-k with entropy values | per-class n = 3: −0.23, n = 75: −2.49; global k = 1: −3.00, k = 100: −0.86 | the queue structure regularizes |
| Uniform-target prior correction on non-uniform label spaces | R/A/Sketch −1.15 (pre-registered) | wrong target |
| Label-free marginal estimation (zero-shot EM, zero-shot marginal, self-EM) | zs-EM −0.40 / −3.31 / −1.86; negative on all 6 with Ramen | EM piles mass on zero-shot's sinks; label shift and classifier bias are not separable from one unlabelled stream |
| Stacking T3 + kNN + pc (λ = 2) | −0.02 IN-C / −1.13 DN | kNN shrinks margins, so pc flips more predictions; kNN needs same-class neighbours (54% DN vs 25% IN-C) |
| Label-free CE targets as values (zero-shot, adapted, kNN-vote pseudo-labels) | zs ≈ 0; adapted −0.65 / −5.28 / −31.17; kNN −1.50 / −15.84 / −10.05 | the target must be far better than zero-shot |
| Label-free cluster mean shift as an explanation or replacement | 11% / 30% of Ramen's gain | the domain component is partial |
| Label-free class GMM/EM on embeddings | DN −11.1, IN-C −4.0 | components drift to domains |
| Averaging over the step grid | +0.08 / +0.34, 3× compute | too small |

---

## 8. Implementation facts that affect interpretation

- The fp16 priority freeze (Section 1) means late-stream DomainNet queries have no self entry (median self-share 0).
  Any non-stationary claim must use fp32 priorities.
- ImageNetC5K shares photos across corruptions: same-photo leaks inflated one label-value result from −1.00 to
  +3.97. Always exclude photo copies when the cache carries labels.
- Ramen-internal zero-shot vs the cached NoAdapt zero-shot differ on 0.2–0.6% of images (fp16 LayerNorm numerics).
- Stream order depends on `num_workers` and the tqdm version (tqdm consumes the seeded generator), so per-sample
  alignment requires the same harness settings.
- DomainNet accuracy is reported as a **domain average** (55.37 → 56.80). The sample-weighted figure is 52.83 → 54.55.
- Ramen's outputs never feed back into the memory, so any output post-processing is exact offline.

---

## 9. Tools available for cheap experiments

- **Offline cache replay (CPU, ~3 min):** rebuilds Ramen's per-class queues and active set at any stream position
  from logs plus cached NoAdapt features and probabilities. It matches the GPU logs 98–100% for the top-1 retrieved
  entry and 100% for queue membership. Any change to the **read side** (weighting, active-set selection, which
  entries count) can be scored this way without a GPU, except for the final step, which needs the gradient values.
- **Per-image records:** zero-shot and Ramen probability vectors, embeddings before and after adaptation, for full
  EuroSAT, CIFAR-100-C and DomainNet (seed 0), plus a true-label-filing record run on DomainNet.
- **Offline output post-processing:** prior correction, kNN smoothing and gating on cached probabilities, seeds 0–4,
  for all datasets.
- **Step sweep cache:** probabilities for step sizes 0 to 0.04 on C10 and C100 (seed 0), so per-sample step policies
  can be scored offline.
- **Interactive confusion explorer** over all ~300 cached runs.
- **NSS (36-d BRISQUE/NIQE) feature vectors** cached for ImageNetC5K.

---

## 10. Open questions and untested follow-ups we already know about

1. Which retrieved entries steer the step toward a new sink (pillow)? Plan: GPU leave-one-group-out over ~200
   images, removing other-domain entries, confident-correct entries, or the envelope/squiggle/pillow queues.
2. Is the shared quickdraw shift direction caused by the composition of retrieved neighbours? Does the coherence
   effect replicate on ImageNet-C corruption types?
3. A boundary-free fix for stale memory (age-based eviction, capacity matched to stream rate, recency weighting).
   Ceilings: ImageNet-C +0.82, DomainNet +0.27 (sequential); EuroSAT capacity-shrink +1.64 (mixed, seed 0).
4. Why T3 fails on DomainNet sequential (test: T3-reset vs Ramen-reset).
5. Stack NSS-vector matching with T3 on seeds 0–4.
6. A label-free contamination detector scored by AUROC against the confident-wrong oracle flags (ceiling ~+0.2).
7. Side information about the label set (known candidate classes, for example the 200-class R/A protocol) for prior
   correction. The structure oracle is worth +2.32 on R/A/Sketch.
8. Domain-centred LDA oracle ceiling; EuroSAT-only LDA labels as CE targets.
9. The cache-compression DomainNet sign run is in progress; the capacity-shrink effect on EuroSAT needs seeds 1–4.

---

## 11. Constraints on proposals

- **Label-free at test time.** Only unlabelled test images arrive online. Oracles are fine as ceilings, not as methods.
- **Causal / online.** A prediction may use only past and current samples (transductive offline analyses are fine for
  diagnosis).
- **Compute:** ViT-B/16 and B/32 on 24 GB GPUs. A DomainNet run takes ~40 min. Prefer ideas testable offline first.
- **No hyperparameter retuning on held-out sets.** Propose fixed settings or a pre-registered selection rule.
- **Generality:** a method change must not hurt small label spaces (C10, C100, EuroSAT) by more than their floors,
  and must hold on ImageNet-C, DomainNet and R/A/Sketch, ideally on both mixed and sequential streams.
- The original Ramen code is not modified. New methods are subclasses or local copies, so proposals should be
  expressible as changes to Ramen's write rule, read rule (active set and weights), value, step or output.
