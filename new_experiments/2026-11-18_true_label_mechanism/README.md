# Why does a true-label-filed cache lower accuracy? (2026-11-18)

**Question.** Filing each test image into Ramen's cache by its TRUE label (an ORACLE, `RamenTrueLabel`
`filing: true_label`) makes every queue 100% pure, yet costs −1.50 on DomainNet (quickdraw −6.49), −0.71 on
ImageNet-C and −6.50 on EuroSAT (seed 0). Why?

**Method.** Offline replay of both caches (no GPU, no gradients), reusing `2026-11-01_pillow_sink/cache_replay.py`
unchanged. The replay is exact up to fp16 arithmetic, because Ramen's cache never depends on adapted outputs. The
adapted predictions come from the two recorded runs (`2026-10-28_confusion_records`, Ramen and true_label, both
equal to their cached runs on 100% of images).

| file | job |
|---|---|
| `containment.py` | pre-registered hypotheses C1–C3; the zero-shot concentration table; the replay of 1,000 random images per domain under both filings; the update-weight shares, outcomes, quintile table |
| `results/DomainNet_s0_containment.pkl` | per (query, cache) summary + concentration table + sampled rows |
| `logs/` | run log |

```bash
CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/RAMEN/bin/python containment.py DomainNet 0 --per-domain 1000
```
This takes ~5 min on the CPU. It needs both recordings and the cached text features of the pillow study.

## Math

Ramen (notation of `.claude/skills/ramen-experiment/references/ramen.md`): image $i$ is written to queue
$\mathcal M_{q(i)}$ with key $z_i$, entropy $H_i$ and value $g_i = \nabla_\theta H_i$. The value sharpens $p_i$
toward its zero-shot class $\hat y_i$, right or wrong. The update is
$\hat g_i = \frac{1}{|C|}\sum_{c}\sum_{j \in \mathcal N_k^c(z_i)} w_{ij} g_j$ with
$w_{ij} = e^{-H_j} e^{-\beta d_{ij}}$.
- Ramen: $q(i) = \hat y_i$. Oracle: $q(i) = y_i$ (**uses ground truth**). Nothing else differs.
- Measured per query $i$: $S_i = \sum_{j:\ \hat y_j = \hat y_i} w_{ij} / \sum_j w_{ij}$. This is the share of the
  update weight on entries whose value pushes toward the query's own zero-shot class (the "same-confusion share").

## Hypothesis (pre-registered in `containment.py` before the run)

**Containment.** Zero-shot filing puts all images that CLIP confuses the same way into ONE queue. Equal queue
averaging then caps a confusion's influence at about $1/|C|$. True-label filing spreads those same images over
all their true classes, so most queues' top-k contain nearby images that share the query's confusion, and the
update re-imports the self-confirming direction.
- C1: on zero-shot-wrong queries, $S$ is ≥ 2× larger under true-label filing.
- C2: the effect is largest where zero-shot predictions concentrate on few classes (quickdraw).
- C3: the oracle keeps the zero-shot prediction more often where $S$ is higher.

## Results (DomainNet seed 0, 6,000 sampled queries, 1,000 per domain)

**Zero-shot concentration.** Effective number of classes is exp(entropy) of the label distribution.

| domain | images | zs wrong | eff. true classes | eff. zs-predicted classes |
|---|---|---|---|---|
| quickdraw | 172,500 | 149,509 (86.7%) | 345.0 | **55.8** (errors only: 42.3) |
| real | 175,327 | 31,344 (17.9%) | 322.5 | 318.5 |
| other 4 | — | — | 248–295 | 215–293 |

**C1 holds.** Mean share of the update weight, zero-shot-wrong queries (2,686 of 6,000):

| | same zs class as query $S$ | zero-shot-error entries | query's domain | true-class queue |
|---|---|---|---|---|
| Ramen cache | 1.4% | 28.3% | 51.0% | 0.6% |
| oracle cache | **8.7%** (6.2×) | 49.6% | 59.9% | 1.2% |

**C2 holds.** On zs-wrong queries, $S$ goes from 1.0% to **20.2%** on quickdraw. On the other domains it goes from
1.4–2.1% to 2.5–4.0%.

**Accuracy on the sample follows.** quickdraw: Ramen 17.3% vs oracle 9.9%. The oracle keeps the zero-shot
prediction on 64.4% of quickdraw images vs 48.6% for Ramen. The other domains differ by ≤ 1.9 pts.

**C3 holds, within each domain and paired.** Oracle, zs-wrong queries, by quintile of $S$: quickdraw kept zs
29.1% → 99.4%; other domains 57.0% → 89.8%. Paired, same query under both caches, by quintile of the *increase*
in $S$ (oracle − Ramen):
- quickdraw (871 queries): $S$ +0.4 pts → Ramen keeps zs 48.6%, oracle 33.7%; $S$ +56.4 pts → Ramen 28.7%,
  oracle **99.4%**.
- other domains (1,815): $S$ −0.2 pts → oracle − Ramen kept zs −3.9 pts; $S$ +5.7 pts → +6.3 pts.

Where true-label filing does not raise $S$, the oracle adapts at least as much as Ramen. The loss is concentrated
where it does.

## Why it makes sense

*Why purity should help:* the true-class queue would hold only correct examples of the query's class.
*What actually happens:* that queue is 1 of 345 equally averaged queues (0.6% → 1.2% of the weight). What
matters is how filing changes the other 344 queues. CLIP's errors on quickdraw collapse onto ~42 effective
classes (envelope, squiggle, …). Ramen's filing keeps each collapse inside its zero-shot queue, so most queues
offer the query cross-domain entries with low entropy and high weight. These dilute the self-confirming direction,
which matches the grad-alignment probe: own −0.36 → update −0.035. True-label filing scatters the collapsed images
over every class, so for a quickdraw query nearly every queue's nearest entries are quickdraw images confused the
same way. Their entropy gradients push toward the query's wrong zero-shot class, and the step re-sharpens it
(kept zs 99.4% in the top quintile). In one line: **Ramen's "noisy" filing works as a de-duplication of
confusions. Clean filing undoes it.**

**Not tested here (hypotheses):**
- That entries with the same zero-shot class point the gradient toward that class. This is a weight share, not a
  gradient cosine. Next GPU check: the grad-alignment probe under true-label filing, predicting
  cos(ĝ, g_self) ↑ and cos(ĝ, g_ce) ↓ on zs-wrong.
- ImageNet-C (−0.71) and EuroSAT (−6.50): there are no true_label recordings, so no replay yet. EuroSAT has
  10 classes, so containment caps a confusion at 10%, not 0.3%. Its large loss would fit if its confusions are
  concentrated (Herbaceous queue 531 entries).
- True-label filing also hurts with label values (−1.96 vs zero-shot filing). There the values no longer push
  toward the zero-shot class, so containment as stated does not cover it.

**Verdict.** Containment explains the DomainNet loss (C1–C3 hold, one seed, a sample of 6,000). Purifying queues
removes Ramen's implicit cap on how much any one confusion can steer the update.
