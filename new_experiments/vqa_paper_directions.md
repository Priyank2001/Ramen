# Directions from the LIVE video quality paper

*2026-10-10. Source: K. Seshadrinathan, R. Soundararajan, A. C. Bovik, L. K. Cormack, "Study of Subjective and
Objective Quality Assessment of Video", IEEE TIP 19(6), 2010
([research_papers/](../research_papers/Study_of_Subjective_and_Objective_Quality_Assessment_of_Video.pdf)).
This file adds to [new_directions.md](new_directions.md), [next_directions.md](next_directions.md) and
[findings_and_directions.md](findings_and_directions.md) and does not repeat them.*

## The short version

- The paper is not about adaptation. It builds the LIVE video quality database (150 distorted videos, 38 viewers)
  and tests which quality metrics agree with people. Its useful parts for us are its **methods**:
  - it z-scores each viewer's ratings to remove that viewer's scale bias;
  - it scores each video by its difference from a hidden reference;
  - it rejects outlier viewers;
  - it chooses distortion levels so quality is spread evenly;
  - it tests significance against a "null model" set by how much people disagree with each other.
- I checked four of these ideas offline on the cached ImageNet-C runs (seed 0, no new Ramen runs). Script:
  [quick_checks/vqa_checks.py](quick_checks/vqa_checks.py).
- **New evidence: Ramen's memory weights have a scale bias by corruption.** Ramen weights each memory entry by
  exp(−zero-shot entropy). Mild corruptions have low entropy everywhere, so their entries outweigh everything else:
  - 23% of the highest-weight retrieved entries are *brightness* images, where chance is 6.7%;
  - *contrast*, where Ramen gains the most, supplies 2.6%.

  This is the paper's "viewer who rates everything high" problem, and z-scoring is its fix (Direction 2).
- **The `prob` T=3 active-set gain (+0.51) holds up as far as one seed can show.** Bootstrapping over images gives
  95% CI [+0.30, +0.72], and McNemar gives p = 7e−8. Stream-order noise is still untested and needs seeds 1–4
  (Direction 1).
- **Dead end: a no-reference image-quality signal does not predict where Ramen helps.** These are natural-scene
  statistics from the paper's own lab (BRISQUE-style MSCN features). Their AUROC is 0.49–0.51, and gating on them
  loses accuracy.

## How the paper's ideas map onto Ramen

| paper | Ramen analogue |
|---|---|
| a viewer rates videos | a memory entry votes on the update (SignSGD makes the update a weighted majority vote) |
| viewers use the rating scale differently, so the paper z-scores per viewer and session | entries from different corruptions have different entropy levels, so exp(−H) is not comparable across them |
| difference score against a hidden reference: DMOS = score(reference) − score(test) | score an entry by how much adaptation improved it, not by its absolute entropy |
| ITU-R BT.500 outlier-viewer rejection | reject memory entries whose gradient disagrees with the consensus |
| F-test against a "null model" (how much viewers disagree with each other) | compare a variant against Ramen's own seed-to-seed variation |
| distortion levels chosen so perceived quality spans the range evenly | our results are dominated by the hard corruptions, so report results by difficulty band |
| transient packet-loss distortions; temporal modelling (Temporal MOVIE) wins | bursty streams, and using an entry's age |

## What I checked

Seed 0, ImageNet-C severity 5 (75k images, ViT-B/16) unless marked.

| check | result | verdict |
|---|---|---|
| 1. `active_prob_T3` vs Ramen, paired: cluster bootstrap over the 5,000 base images, plus McNemar | +0.51, 95% CI [+0.30, +0.72]; T3-only right 2,660 vs Ramen-only right 2,280, p = 7e−8 | real with respect to image sampling; stream order untested |
| 2. Mean zero-shot entropy per corruption | brightness 1.75 … contrast 4.73; mean exp(−H) 0.31 vs 0.04 (7.6×) | **scale bias exists** |
| 2b. Which corruption the top-weighted retrieved entry comes from (Ramen) | brightness 23.0%, snow 10.5%, motion blur 11.2% … elastic 1.1%, contrast 2.6% (chance 6.7%) | **brightness dominates the vote** |
| 2c. Same for `prob` T=3 | brightness 11.7%; top entry from the query's own corruption 58.9% (Ramen 45.2%) | the zero-shot weighting halves the bias; this may be part of why it helps |
| 3. SROCC across domains between zero-shot accuracy and Ramen's gain | CIFAR-10-C −0.94 (p < 0.001), ImageNet-C −0.71 (p = 0.003), DomainNet −0.71 (6 domains, p = 0.11), CIFAR-100-C −0.32 (p = 0.24) | Ramen's gain is mostly a gain on hard domains |
| 3b. Same for `prob` T=3's gain over Ramen | −0.27 (p = 0.32) | its gain is spread across difficulty levels |
| 4. MSCN features (variance, kurtosis, local contrast, neighbour correlation): fixed vs broken samples, AUROC within each corruption | 0.487–0.514 (zero-shot entropy: 0.595) | **no signal** |
| 4b. Best "zero-shot or Ramen" gate on each feature, tuned on the test set | −0.20 to −0.40 (entropy +0.003, confidence −0.004) | **dead**, same result as the earlier gates |
| 4c. Per-corruption mean MSCN features vs Ramen's gain, SROCC | −0.34 to −0.02 | how damaged an image looks does not predict how much Ramen helps |

> **Update 2026-10-11 — Direction 1 done.** `prob` T=3 over Ramen on ImageNet-C, seeds 0–4: **+0.62 ± 0.11**,
> positive on every seed (min +0.51; each seed's paired bootstrap CI excludes 0), paired t = 12.9. Ramen's own
> accuracy ranges 32.91–33.17 over seeds, so the project's noise floor (2 × range, seed_variance.ipynb) is **0.54**:
> the mean gain clears it, but seeds 0 and 1 (+0.51) individually do not. Real by the paired test, small by the
> 2 × range rule. Gains on 12/15 corruptions (shot noise +2.1, Gaussian noise +1.6); loses elastic transform −0.5
> and JPEG −0.5 on all 5 seeds. Script: [active_seeds.py](active_seeds.py); table:
> `results/ImageNetC5K_active_prob_T3_seeds.csv`. Held-out test with T=3 fixed (seeds 0–4, 2026-10-11):
> **DomainNet +0.24 ± 0.01** (every seed, noise floor 0.11, t = 38.9; 5 of 6 domains, sketch +0.52, quickdraw −0.05);
> **CIFAR-100-C −0.15 ± 0.08** (negative on every seed, within the 0.21 noise floor). Helps on the large label spaces
> (1000 / 345 classes), not on CIFAR-100-C (100 classes, top-5 retrieval with β = 5). Tables:
> `results/{ImageNetC5K,DomainNet,CIFAR100C}_active_prob_T3_seeds.csv`.
> CIFAR-100-C temperature sweep, seed 0 ([temperature_sweep.py](temperature_sweep.py)): T = 1 / 3 / 5 / 10 / 30 / 100
> gives −1.69 / −0.23 / +0.03 / +0.05 / +0.03 / −0.01 over Ramen. Sharper weighting hurts, and flatter weighting only
> returns to Ramen: no T clears the 0.21 noise floor. On 100 classes, zero-shot class weighting has nothing to add.

## Direction 1: Significance testing as a habit, starting with `prob` T=3 at seeds 1–4

The paper never claims a metric is better without an F-test. It also compares each metric with a null model, the
spread between individual viewers, because no metric can beat human disagreement. Our ImageNet-C and DomainNet
results are still single-seed, with no noise floor.

- **Now:** run `active_prob_T3` and Ramen at seeds 1–4 on ImageNet-C. That is 8 runs of about 8 min each:
  `make_experiment('ImageNetC5K', seed=s, num_workers=16)`, then `exp.run('Ramen')` and
  `exp.run('RamenActive', name='active_prob_T3', cfg={'active_set': 'prob', 'temperature': 3.0})`.
  The two Ramen-vs-Ramen seed differences give ImageNet-C its first noise floor. That plays the role of the
  paper's null model.
- **Always:** report the cluster bootstrap CI and McNemar from `vqa_checks.py` (check 1) next to every single-seed
  number. Cluster by base image: one image appears under all 15 corruptions, so the samples are not independent.
  This is the same point the paper makes when it checks its F-test assumptions with a bootstrap.
- Also worth one run: `prob` with T = 5 and 10, to see whether flatter weighting keeps gaining.

**Success:** the `prob` T=3 gain stays above the seed noise floor on ImageNet-C. Then test it on DomainNet, and
on CIFAR-100-C, where the noise floor is known (+0.21).

## Direction 2: Remove the per-corruption scale bias in Ramen's entry weights (z-scoring)

The paper z-scores each viewer's scores within each session, so that a lenient viewer and a strict viewer count
equally. Ramen's entry weight exp(−H) has the same problem across corruptions (check 2). An entry from a brightness
image outweighs one from a contrast image by about 8× before relevance is considered. So brightness images supply
23% of the top votes, even for queries from other corruptions.

Steps, cheapest first. Each is a config option on `RamenActive` and one 8-min run:

1. **Flatten the weight:** use exp(−H/τ) with τ ∈ {2, 4, ∞}, where τ = ∞ means uniform entry weights. This is one
   line in `RamenActive.aggregate`, and it shows whether the entropy weight helps at all on ImageNet-C.
2. **Ceiling, using labels once:** z-score H within each corruption with a causal running mean and std. This uses
   the corruption label, so it is not a method. It measures how much removing the bias is worth, in the same
   spirit as "cheat once" in new_directions.md, Direction 3.
3. **Without labels:** z-score each entry's entropy against the entropies of its nearest earlier samples in feature
   space. Store z_j at write time, from a ring buffer of recent features and entropies. Neighbours are mostly the
   same corruption (top entry from the query's own corruption 45%, chance 6.7%), so this approximates step 2.

**Check:** brightness's share of the top entries should fall toward 6.7%, and accuracy should beat Ramen above the
noise floor. Then check whether it stacks with `prob` T=3, which already halves the bias.

**Risk:** the earlier purity-weighting result (RamenClean) lost accuracy because the effective support shrank.
This change doesn't remove entries, it only re-ranks them, but watch `self weight` and how spread out the
support weights are.

## Direction 3: Difference scores: weight an entry by how much adaptation helped it

DMOS scores each video against a hidden reference, which removes a viewer's taste for particular content. Ramen's
exp(−H) is an absolute score: it mixes "this content is hard" with "this sample's gradient is reliable". The
analogue is the entry's entropy drop after its own adapted prediction, H_zero-shot − H_adapted.

- Keep writing **before** retrieval, because writing after prediction cost −0.6 on CIFAR-10-C (cache_strategies). After the batch's
  prediction, overwrite the stored weight of each entry just written with its difference score. `IdCache` already
  keeps each entry's stream id, so finding the slot is one lookup.
- Variants: the raw difference; the difference z-scored per neighbourhood (combining with Direction 2); and the
  0/1 "entropy went down".
- Free diagnostic first: Ramen's adapted entropy per sample is already in `results/…_Ramen.pkl` (`entropy`) and the
  zero-shot entropy in `…_NoAdapt.pkl`. Check whether the difference separates correctly-filed from mis-filed
  entries better than absolute H does (AUROC, using the logged `filed_as`). If it doesn't, skip the run.

## Direction 4: Reject outlier voters (ITU-R BT.500 style)

The paper drops viewers whose scores fall outside ±2σ of the mean more than 5% of the time (9 of 38 were dropped).
With SignSGD, Ramen's update is a weighted majority vote over about 1000 retrieved entries, one per class queue.
The cache viewer shows that many of these voters have nothing to do with the query: for a butterfly, the top
entries were a maze, a banana and an iPod.

- Score each retrieved entry by the share of coordinates where its gradient sign disagrees with the weighted vote.
  Drop entries more than 2σ above the mean disagreement, then re-vote.
- This is **entry-level** rejection. new_directions.md Direction 2 is **coordinate-level** ("only move the knobs the
  neighbours agree on"). Run the two separately before combining them.
- Cap rejection at about 5% of entries, as the paper's rule does. The purity results say a much smaller support
  hurts.

## Direction 5: Transient-shift streams and entry age

Half of the LIVE videos have short, local, transient damage from IP and wireless packet loss. The paper also finds
that the metric which models time (Temporal MOVIE) wins. All our streams are shuffled, so they are stationary.

- Extends new_directions.md Direction 6 (realistic streams) with a concrete protocol taken from the paper. The base
  stream is one domain (say ImageNet-C brightness, or DomainNet real). Inject bursts of a second corruption at the
  paper's loss rates of 3%, 5%, 10% and 20% of the stream, with burst lengths of 1, 5 and 20 batches.
- On such streams Ramen's memory lags. Add an age term to the entry weight, exp(−age/T). Entry ids are stream
  positions, so the age is free in `IdCache`. The fp16 priority bug will also matter here, so use fp32 priorities.
- Measure accuracy inside bursts and just after them, and how quickly the memory recovers. `CacheViewer.plot_onset`
  already shows when confusions start.

## Direction 6: Report results by difficulty band (cheap reporting change)

The paper chose distortion strengths so its scores span the quality range evenly, and it criticises the VQEG set
for "poor perceptual separation". Our benchmarks have the opposite issue: Ramen's gain is strongly tied to domain
difficulty (SROCC −0.71 to −0.94), so the average mostly measures the hard corruptions.

- For every new method, add a table of gain over Ramen per zero-shot-confidence quintile (per image) next to the
  corruption average. This is a few lines on `exp.paired()`.
- `prob` T=3's gain is not tied to difficulty (check 3b). A per-band table would show whether it helps mild
  corruptions, where Ramen barely helps.
- ImageNet-C severities 1–4 were deleted. Re-downloading them (hours, Zenodo record 2235448) is only worth it if
  the per-band tables disagree with the severity-5 averages.

## Checked and ruled out

- **No-reference image quality (natural-scene statistics) as the "when to adapt" or "how far to step" signal.**
  It doesn't separate fixed from broken samples (AUROC about 0.5, within each corruption and pooled). Every
  test-tuned gate loses 0.2–0.4 points. Per corruption, how damaged images look doesn't track Ramen's gain either.
  Together with the earlier confidence and entropy gates, this suggests the per-sample headroom does not come from
  the image's damage level.
- **Not transferable:** the subjective-study machinery (display calibration, session design) and the logistic
  mapping before computing correlation. We measure accuracy directly, and calibration is already in
  next_directions.md Direction 5.

## Suggested order

1. Direction 1, seeds 1–4 (about 70 min of GPU, unattended). Every later comparison on ImageNet-C needs that noise
   floor.
2. Direction 2, step 1 (flatten the weight, 3 runs), then step 2 (the ceiling). Go to step 3 only if the ceiling is
   above the noise floor.
3. Direction 3's free diagnostic, then its run if the diagnostic is positive.
4. Directions 4 and 5 after that; Direction 6 alongside everything.

## Files

- [quick_checks/vqa_checks.py](quick_checks/vqa_checks.py): all checks above (about 15 s; caches the MSCN features
  in `results/ImageNetC5K_nss_features.npy`, row i = results row i).
- [../src/methods/RamenActive.py](../src/methods/RamenActive.py): the zero-shot-weighted active set. Directions 2–5
  are new config options here.
- [cache_viewer.py](cache_viewer.py) and [imagenetc_active_set.ipynb](imagenetc_active_set.ipynb): the active-set
  experiment and the memory viewer.
