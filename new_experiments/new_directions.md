# New research directions for beating Ramen (simple version)

*2026-10-09. A plain-language list of new ideas. It builds on [findings_and_directions.md](findings_and_directions.md)
and does not repeat the directions there (better λ for the prior correction, seeds 1–4 on the big datasets, the
DomainNet sweep, per-sample step size, calibration).*

## The short version

- I looked for ideas that are not in the older direction docs, and tested the cheap ones offline on the saved
  results in `results/`. No new Ramen runs were needed.
- **One new idea works on CIFAR: "ask the neighbours".** After Ramen predicts an image, mix its answer 50/50 with
  Ramen's answers on the 10 most similar earlier images. That gives **+0.50** (CIFAR-10-C) and **+0.59**
  (CIFAR-100-C) over seeds 0–4, above the noise floor on every seed. **With your prior correction on top it is
  +1.25 and +2.07.** It has not been tested on ImageNet-C or DomainNet yet. That test comes first (Direction 1).
- Seven more directions follow (2–8). Each needs a small code change and a few runs.
- Four ideas I checked today don't work. They are listed at the end so you can skip them.

## What I checked today

All offline, from the saved probabilities, features and support-set logs. Gains are over Ramen, in accuracy points,
CIFAR-10-C / CIFAR-100-C unless marked otherwise. Noise floor: +0.31 / +0.21.

| idea | result | verdict |
|---|---|---|
| Ask the neighbours (Direction 1) | +0.50 ± 0.03 / +0.59 ± 0.03, seeds 0–4 | **works on CIFAR** |
| Ask the neighbours, then your prior correction (λ = 2, m = 0.99) | +1.25 ± 0.04 / +2.07 ± 0.05, seeds 0–4 | **works on CIFAR; the two add up** |
| Average Ramen over 5 stream orders. A measurement, not a method: it needs 5 passes over the stream | +0.35 / +0.73 | part of each per-image answer is luck (which images happen to be in memory) |
| Use Ramen's retrieved support set as a classifier, mixed with Ramen | at most +0.35 / +0.25, even after tuning on the test set | skip |
| Fall back to zero-shot on images where Ramen's answer depends on stream order | zero-shot is worse on those images (30.2% vs 36.7% / 14.1% vs 20.5% correct) | skip |
| Better prompt templates: CLIP's 80 templates, chosen per domain or per image without labels | −0.8 to +0.3 on all 4 datasets. Even the best template per domain, picked using labels, gains at most +0.5 | skip |
| Keep zero-shot when Ramen and zero-shot disagree between two near-synonym classes | −0.1 to −9 on all 4 datasets | skip |

The scripts are in [quick_checks/](quick_checks/) (see the last section).

---

## Direction 1 — Ask the neighbours (start here)

**The idea in one line.** If Ramen called similar-looking images "deer", this image is probably a deer too.

**What it does.**
1. Ramen predicts a batch exactly as now.
2. For each image, find the 10 most similar images from earlier batches. Similarity is the cosine similarity of the
   zero-shot CLIP features that Ramen already computes for its memory.
3. Average Ramen's predicted probabilities for those 10 images, giving more weight to the more similar ones. Mix
   that 50/50 with the image's own prediction.

**Why it is cheap.** No extra forward or backward passes. For each past image you only keep its feature (512 numbers)
and Ramen's output probabilities. Ramen's outputs never feed back into its memory, so, like the prior correction,
every variant can be tested offline from one logged run.

**Results, CIFAR, seeds 0–4** (gain over Ramen on the same seed):

| | CIFAR-10-C | CIFAR-100-C |
|---|---|---|
| ask the neighbours | +0.50 ± 0.03 | +0.59 ± 0.03 |
| prior correction alone (your number, reproduced) | +0.57 ± 0.05 | +1.50 ± 0.03 |
| both | **+1.25 ± 0.04** | **+2.07 ± 0.05** |

**Where the gain comes from.** About 13.5% (CIFAR-10-C) and 33% (CIFAR-100-C) of images get a different Ramen answer
depending on the order the stream arrives in. These are borderline images: Ramen gets only 36.7% / 20.5% of them right.
90–96% of the smoothing gain comes from these images. Smoothing does **not** make Ramen's answers more stable across
stream orders (images that change between two seeds: 6.8% → 8.3% on CIFAR-10-C). It works by pooling evidence on
borderline images, not by removing randomness.

**Caveat.** I picked the setting on CIFAR seed 0 from a small grid: 10 or 50 neighbours, mix 25/50/75%, zero-shot or
adapted features. Seeds 1–4 show the same test images in a different order. They rule out a lucky stream order, not
a setting that only suits CIFAR.

**Next steps.**
1. **Held-out test, fixed before running.** The setting: 10 neighbours, 50/50 mix, similarity weights
   `softmax(cos / 0.01)`, zero-shot features as keys, earlier batches only. Run it once on ImageNet-C and DomainNet
   (seed 0) from the saved probabilities and features. That takes minutes. Report "ask the neighbours" alone as the
   main result and the combination second: λ = 2 already failed on those datasets, so the combination may inherit
   that.
2. **Only after that, improve it on CIFAR.** Options: use Ramen's adapted features as keys (seed 0: +0.59 / +0.66
   instead of +0.50 / +0.64); also use images from the same batch; cap the memory at a window, since DomainNet
   would otherwise need about 1 GB.
3. **If it holds up, feed it back.** Use the neighbours' answer as the target of each image's own adaptation step,
   instead of plain entropy. That changes the memory, so it needs full re-runs.
4. **Related work.** See the citations in the last section.

**Stop if** the held-out gain is below about +0.3 on both datasets.

---

## Direction 2 — A quieter step: only move the knobs the neighbours agree on

**What Ramen does now.** It adds up the neighbours' gradients, then moves every adapted LayerNorm number (about 40,000
for ViT-B) by exactly +0.01 or −0.01, following the sign. When the neighbours disagree about a number, its sign is
close to a coin flip, but the number still moves the full 0.01.

**The idea.** Keep Ramen's direction, but scale each number's step by how much the neighbours agree on it:

    agreement = |sum of the weighted pushes| / (sum of the sizes of the weighted pushes)     (between 0 and 1)
    step      = 0.01 × agreement × sign

If all neighbours push the same way, it takes the full 0.01 step. If they cancel out, the number barely moves.
Variant: a full step where agreement is above a threshold, no step below it.

**Why it might work.** The borderline images from Direction 1 get answers that depend on which images happen to be
in memory, and averaging 5 stream orders gains +0.35 / +0.73. A step that skips the coin-flip numbers might get some
of that inside a single run.

**How it differs from what already failed.** Normalized SGD (−3.8 / −5.8) scaled the step by the gradient's *size*.
This scales by *agreement on the direction*, and never moves a number further than Ramen does.

**How to test.** In `Ramen.forward`, the retrieved gradients are already an (images × k × numbers) tensor just before
`.sum(dim=1)`. Add one more sum of `weights × |grad|` and compute the agreement. Then set the gradient to
`agreement × sign` and switch the config to plain `sgd` with lr 0.01. Two runs (CIFAR-10-C, CIFAR-100-C) at seed 0,
then seeds 0–4 if it looks good. Look at "fixed" and "broken" counts as well as accuracy.

**Stop if** it is below the noise floor on both datasets.

---

## Direction 3 — Cheat once, to find the ceiling for "better memory" ideas

**The idea.** Before building a better memory or retrieval, measure the best case by cheating on purpose. This is a
diagnostic only, never a result.
- **(a) Oracle domain:** each image retrieves only from images of its own true corruption.
- **(b) Oracle labels:** each image is filed in memory under its true class, not its zero-shot guess. The older docs
  proposed this; it hasn't been run.

**Why.** Several ideas only pay off if one of these ceilings is high: a learned retrieval metric, a cleaner memory,
and Direction 4. Purity fixes have already failed (RamenClean), so get a quick yes or no before spending more time.

**How to read it.**
- (a) gains less than about +0.3: drop every retrieval-metric idea, including Direction 4.
- (b) gains less than about +0.3: drop memory-cleaning ideas for good.

**Cost.** Pass the true domain and label into Ramen (the `--diag` plumbing in the PC2 handoff, Section 6), then mask
the retrieval or change the filing. Four runs on CIFAR.

---

## Direction 4 — Find neighbours by type of damage, not by object (only if 3a says yes)

**What Ramen does now.** It finds neighbours using the final CLIP feature, which mostly describes *what object* is in
the image. For each class queue it takes the entries that look most like the query.

**The idea.** Find neighbours using a "style" feature that describes *how the image is damaged* (noise, blur, fog,
sketch style). For example, use the mean and spread of an early layer's token features, which capture texture and
noise rather than the object. Keep filing images under their zero-shot class as now.

**Why.** Ramen's theory (Theorem 4.1) wants a support set where the domain stays fixed and the class varies. A
damage-type key gives "same damage, typical members of each class". The current key gives "same damage, plus
members of each class that look like the query". Domain purity is already 6–12× above chance, so the gain may be
small. That is why 3a comes first.

**Cost.** One forward hook (the early-layer statistics come from the same forward pass), a different retrieval key,
and two runs.

---

## Direction 5 — Make ViT-L/14 fit on our 24 GB GPUs

**Why it matters.** The paper's ImageNet-C table uses ViT-L/14. Our ImageNet-C numbers use ViT-B/16, so they can't be
compared with the paper. That is a hole in the thesis whichever method wins.

**Where the memory goes.** With ViT-L/14, each cached gradient has 102,400 numbers. ImageNet-C keeps 1000 classes × 75
entries, about 15 GB in fp16, which is half of the 30.6 GB the README reports. Most of the rest is probably the
backward pass over 100 images at once.

**Two fixes.**
1. **Exact (results unchanged):** compute the per-image gradients of each batch of 100 in chunks of 25. Each image's
   gradient doesn't depend on its batch-mates, and the memory update, retrieval and step still happen once per 100
   images, so the method is the same. Check that predictions match the current code on CIFAR-10-C (up to rounding).
2. **Approximate:** store cached gradients in int8 instead of fp16, which halves them. Check that predictions match
   on CIFAR-10-C before trusting it.

Fix the fp16 priority bug while you are in this code. It is probably enough to do fix 1 alone; fix 2 leaves room to
spare.

---

## Direction 6 — Test on realistic streams

**The problem.** Every number so far is on a perfectly shuffled stream. Real streams aren't shuffled:
- **(a) Label runs:** the same class arrives in bursts, as in a video. The standard protocol samples the class order
  from a Dirichlet distribution (RoTTA, NOTE).
- **(b) Domain order:** one corruption after another.

**Why it matters for us.**
- The prior correction assumes the class mix stays steady. During a burst of cats, it will push "cat" down, which is
  exactly wrong.
- Ask the neighbours (Direction 1) may change too, because the neighbours will come from the same burst.
- Ramen's fp16 cache bug should finally show up under domain order.
- "It still works on realistic streams" is a stronger thesis claim than one more point on shuffled CIFAR.

**Cost.** Ramen's outputs never feed back, so one Ramen run per stream type gives the prior correction and Direction 1
for free, offline. About four runs on CIFAR.

---

## Direction 7 — Teach the prior correction to tell "model bias" from "real label shift"

**The problem.** The prior correction can't tell "Ramen over-predicts deer" (bias: correct it) from "deer really
appears more often right now" (label shift: keep it). On shuffled CIFAR the two are the same thing, which is why it
works there. On ImageNet-C and DomainNet the label mix isn't uniform. Under label runs (Direction 6), it changes over
time.

**Ideas**, all offline on the Direction 6 runs:
- Estimate the bias with a much slower average (a long memory) and correct only that, leaving short-term changes
  alone.
- Estimate the real label mix without labels (EM, as in Direction 1 of `findings_and_directions.md`) and correct
  towards it instead of towards uniform.

Same discipline as before: develop on CIFAR, write the setting down, test once on the held-out datasets.

---

## Direction 8 — Which layers should move?

Ramen moves every LayerNorm (26 in ViT-B) by the same 0.01. Try moving only the early, the middle or the late layers.
With SignSGD a zero gradient means no move, so freezing a layer is one line. That is 3–4 runs on CIFAR-10-C.
- If the late layers alone match Ramen, the memory shrinks by about 2× (helps Direction 5).
- If early layers carry the corruption fix and late layers carry something else, that suggests a separate step size
  per layer.

---

## Ideas checked today that don't work

- **Support set as a classifier.** It has to take the same number of images from every class, including classes far
  from the query. On its own it is only as good as zero-shot on CIFAR-10-C (about 61%) and worse on CIFAR-100-C
  (19–31% vs 37%). Mixing it into Ramen adds at most +0.35 / +0.25, after tuning on the test set.
- **"Unstable, so fall back to zero-shot."** The images whose Ramen answer depends on stream order are simply hard
  images. Zero-shot is even worse on them.
- **Prompt templates.** The 7 templates Ramen uses are about as good as any label-free choice from CLIP's 80, even on
  DomainNet's sketch and quickdraw. The text side is not where the gain is.
- **Near-synonym rule.** Near-synonym mistakes (barbell → dumbbell) are 1.7–2× more common among the images Ramen
  breaks than among zero-shot's mistakes: 26–36% vs 13–21% on CIFAR-100-C, ImageNet-C and DomainNet. But Ramen fixes
  about twice as many near-synonym mistakes as it causes, so overriding it with zero-shot loses accuracy on every
  dataset.

## Suggested order

1. **Direction 1 held-out test.** Minutes, offline.
2. **Directions 3 and 2.** About a day of runs on CIFAR.
3. **Direction 5.** Needed for the paper's ImageNet-C protocol whichever method wins.
4. **Directions 6 and 7 together.**
5. **Directions 4 and 8,** only if 3 and 5 point to them.

## Files

Scripts for today's checks, in [quick_checks/](quick_checks/). Run them with the `RAMEN` env from any folder, e.g.
`python new_experiments/quick_checks/knn_seeds.py CIFAR10C`.

| script | what it checks |
|---|---|
| `knn_seeds.py` | ask the neighbours, alone and with the prior correction, seeds 0–4 (zero-shot keys) |
| `knn_smooth.py` | ask the neighbours at seed 0: neighbour count, mix weight, zero-shot vs adapted keys |
| `knn_churn.py` | which images the gain comes from, and stability across stream orders |
| `seed_ensemble.py` | averaging Ramen over 2, 3 and 5 stream orders |
| `instability.py` | how zero-shot and Ramen do on images whose answer depends on stream order |
| `support_classifier.py` | the retrieved support set as a classifier (uses the `RamenLog` logs) |
| `text_side.py` | prompt templates and near-synonym flips. Needs a GPU for the CLIP text encoder, e.g. `CUDA_VISIBLE_DEVICES=1 python new_experiments/quick_checks/text_side.py DomainNet` |

## Related work to read

*(to be filled in)*
