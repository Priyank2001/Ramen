# Pre-registration: domain-sequential streams

Written 2026-10-26, before any sequential run exists. Every result so far comes from **mixed** streams (all domains shuffled together). In deployment, shifts usually arrive domain by domain. Ramen's per-class queues then fill with one domain's entries and keep them, with no reset, when the next domain begins.

## Stream (fixed)
- `tta_mode='sequential'`: the domains are visited one after another in the dataset's default order, each shuffled within itself (seeded), batches of 100. **The method is never reset between domains** (it does not know where the boundaries are). One stream per run.
- Orders (not varied; a caveat):
  - ImageNet-C 5K: the standard continual-TTA order, gaussian_noise → … → jpeg_compression (15 × 5,000).
  - DomainNet: clipart → infograph → painting → quickdraw → real → sketch (596,006).
  - ImageNet-R/A/Sketch: rendition → adversarial → sketch (88,389; 1,000-class head).
- Seeds 0–4 change the within-domain order and Ramen's selection randomness. The images are identical to the mixed runs.

## Methods (fixed, configs unchanged from the mixed runs)
- **Ramen**: `cfg/<ds>/Ramen.yaml`.
- **T3**: RamenActive `active_set: prob`, `temperature: 3`.
- **Ramen-reset** (secondary, a reference that *knows* the boundaries): `tta_mode='single'`, i.e. Ramen reset at every domain start, same within-domain order.
- NoAdapt: stateless, so the seed-0 mixed run is reused, realigned by (domain, sample) id. A seed-0 NoAdapt run in sequential mode on ImageNet-C serves as the sanity check that the sequential loader yields the same images and labels: predictions must match the mixed NoAdapt on 100% of images.

## What counts
- Metric: domain-averaged accuracy. Noise floor = 2 × (max − min) of **sequential** Ramen over seeds 0–4, per dataset.
- **Primary (H1):** T3's gain over Ramen survives the sequential stream: mean gain over seeds 0–4 is above the floor on **all three** datasets. Mixed-stream gains for reference: ImageNet-C +0.62, DomainNet +0.24, R/A/Sketch +0.59.
- **Secondary (reported, no pass/fail):**
  1. Order effect on Ramen: (Ramen − NoAdapt) in the sequential vs the mixed stream, per dataset.
  2. Does carrying memory across domains help? Sequential Ramen vs Ramen-reset.
  3. Adaptation lag: accuracy relative to NoAdapt as a function of the batch position within each domain (first 10 batches vs the rest), for Ramen, T3 and Ramen-reset.
- Anything else (e.g. output post-processing on these streams, other orders) is post-hoc and will be labelled so.

## Amendment (2026-10-26 ~02:15, before any sequential result exists; only the sanity run had finished)
Ramen stores its arrival-counter priorities in **fp16**: every insert after sample 65,520 gets priority inf, and a full queue stops updating once all its slots are inf (known since 2026-10-05; ~0 effect on shuffled streams). In a sequential stream this freezes the memory on whichever domains were streaming at that point. On DomainNet (capacity 300, 345 classes) the queues should freeze at roughly sample 65k + 300 × 345 ≈ 170k, i.e. during painting, so quickdraw, real and sketch (~70% of the stream) would never enter the memory. ImageNet-C (75k) and ImageNet-R/A/Sketch (88k) barely pass 65k.

Added runs (config switch `priority_fp32: true`, which affects eviction only; the default reproduces Ramen exactly):
- **Ramen-fp32, T3-fp32** (sequential): DomainNet seeds 0–4; ImageNet-C and ImageNet-R/A/Sketch seed 0 only as a screen (expected ≈ fp16 there; more seeds only if seed 0 differs from fp16 by more than the floor).
- Check: a mixed ImageNet-C Ramen seed 0 run with the refactored code at the default setting must reproduce the cached run on 100% of images.

How H1 is read: H1 is evaluated as pre-registered (fp16, as-is Ramen and T3). On DomainNet, **T3-fp32 − Ramen-fp32** is reported next to it with the same floor and the same rule. If the two disagree, the fp32 result is the one about the method, and the fp16 result is about the bug. Also reported: the domain composition of the memory over the stream for fp16 vs fp32, replayed offline from the zero-shot filing classes, labelled descriptive.
