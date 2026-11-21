# Pre-registration: held-out test of T3 + prior correction (λ = 1)

Written 2026-10-25 ~22:50, while the data was still downloading, before any model has seen these images. The method setting was frozen earlier in `../2026-10-25_lambda1_check/PREREGISTRATION.md` and passed that check.

## Claim under test
On a dataset none of the settings were tuned on, **T3 + pc(λ = 1)** beats Ramen by more than the noise floor. Metric: domain-averaged accuracy, mean gain over seeds 0–4.

## Data (fixed)
One shuffled **mixed stream** of three ImageNet distribution shifts, batches of 100:
| Domain | Images | Classes present |
|---|---|---|
| ImageNet-R (renditions: art, cartoons, sculptures, …) | 30,000 | 200 |
| ImageNet-A (natural adversarial photos) | 7,500 | 200 (a different subset) |
| ImageNet-Sketch | 50,889 | 1,000 |

- **Label space: all 1,000 ImageNet classes for every image**, with labels remapped from each folder's WordNet id to the ImageNet-1k index. Class-name prompts are CLIP's curated ImageNet names (`~/data/corruption/ImageNet-C/classnames.txt`), with the default 7-template ensemble, exactly as for ImageNet-C.
- Why 1,000 classes rather than each dataset's own 200: the frozen rule applies T3 only to large label spaces (≥ 300 classes), and a mixed stream needs one classifier. Consequence: R and A accuracies are **not comparable** to the usual 200-class numbers in the CLIP TTA literature. The true class marginal is far from uniform (800 classes occur only in Sketch), so pc's uniform target is knowingly wrong here. That is part of the test, not a reason to change it.

## Model and methods (fixed)
- CLIP ViT-B/16 (as ImageNet-C). Configs copied unchanged from `cfg/ImageNetC5K/`: Ramen (capacity 75, topk 1, β 0, signSGD lr 0.01); RamenActive with `active_set: prob`, `temperature: 3` (T3).
- pc: logit − 1 · log(π_t / uniform), π_t = EMA (m = 0.99) of the soft predicted marginal, batch by batch, current batch included. No kNN.
- Runs: NoAdapt (with features and probs), Ramen and T3 (with probs), seeds 0–4.

## What counts
- **Primary:** mean gain of T3 + pc(λ = 1) over Ramen > noise floor = 2 × (max − min) of Ramen's domain-averaged accuracy over seeds 0–4 on this stream. Also reported: seeds positive / above the floor, and per-seed cluster-bootstrap 95% CIs (cluster = image).
- **Secondary (reported, not part of the claim):** T3 alone, Ramen + pc(λ = 1), every method vs NoAdapt, per-domain gains, pooled accuracy.
- Anything not listed here (other λ, kNN, prior estimation, other temperatures) is post-hoc and will be labelled so.
