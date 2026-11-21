# Label values (`RamenLabelValues`) — ORACLE

**Uses true labels.** This is an upper bound, not a method. Notation and the Ramen baseline are in [README.md](README.md).

## Why

In Ramen every entry stores the image's own entropy gradient $g_j$, which sharpens that image's zero-shot
prediction, right or wrong. The filing oracle (`RamenTrueLabel`, `filing: true_label`) puts images in their true
class queue but keeps that value, so the label never reaches the update. This method changes what an entry
**holds**, which tests whether label information in the values is what is missing.

## What changes

Two config keys, one per factor. The defaults (`zero_shot`, `entropy`) reproduce Ramen exactly.

**Filing** (`filing`, inherited from `RamenTrueLabel`): the queue that image $i$ is written to,

$$q_i = \begin{cases} \hat y_i & \texttt{zero\_shot}\ \text{(Ramen)} \\ y_i & \texttt{true\_label}\ \text{(oracle)} \end{cases}$$

**Value** (`value`): the per-sample cross-entropy gradient toward the true label, computed at $\theta_0$ on the
zero-shot logits:

$$u_i = \nabla_\theta\, \big[-\log p_{i,y_i}\big] \,\big|_{\theta_0} \;\in \mathbb{R}^D$$

The value is swapped only after the image has been predicted, so no image ever uses its own label:

1. **Write** (unchanged timing, before retrieval): add $(z_i,\ v_i = g_i,\ H_i,\ \pi_i)$ to $\mathcal{M}_{q_i}$.
2. **Retrieve, aggregate, update, predict, reset**: exactly as in Ramen. Image $i$'s own entry and those of its
   batch-mates still hold entropy gradients at this point.
3. **Reveal** (new, after prediction), for `value: true_label_ce`: every entry written in step 1 has its value
   replaced,

$$v_i \leftarrow u_i .$$

   An image that was not written (full queue, priority not higher) gets nothing. If two images of one batch land in
   the same slot, the later one's value stays.

So when image $i$ retrieves, an entry $j$ from an **earlier** batch holds $u_j$ and pushes toward $y_j$, while
entries from its own batch hold $g_j$. The stored entropy $H_j$ (the weight $e^{-H_j}$) is the zero-shot entropy in
both cases.

## Variants run

| run | filing $q_i$ | value | labels? |
|---|---|---|---|
| `labeloracle_off` (smoke) | $\hat y_i$ | $g_i$ | no; must equal Ramen |
| `labeloracle_zs_ce` | $\hat y_i$ | $u_i$ | yes |
| `labeloracle_true_ce` | $y_i$ | $u_i$ | yes |
| `oracle_file_true` (cached, 2026-10-23) | $y_i$ | $g_i$ | yes |

## Code map

| symbol / step | code |
|---|---|
| $q_i$ | `RamenTrueLabel.queue_of(zs_pred, labels)` |
| $g_i$ | `self.loss_fn(logits).backward(retain_graph=...)`, then `get_by_sample_grad()` |
| $u_i$ | `TrueLabelGradient.revealed`: `F.cross_entropy(logits, labels, reduction='sum').backward()` |
| write with slot | `RamenLabelValues.add` / `write` (same slot rule as `PriorityCache.add`, returns the slot) |
| reveal | `RamenLabelValues.reveal(slots, values)`; count logged in `stats['revealed']` |

## Checks

- Smoke (1,200 images): `value: entropy` gives the same predictions as Ramen (zero-shot filing) and as
  `RamenTrueLabel` (true-label filing) on 1,200 of 1,200 images each.
- One SignSGD step on an image's own $u_i$ raises $p_{i,y_i}$ on 97 of 100 images (correct 62 → 97;
  `check_label_gradient.py`).
- `trace_active_set.py` (C0–C6): no image's own label reaches its own update.

## Results (seed 0, change vs Ramen in domain-averaged points)

| run | DomainNet (floor 0.11) | ImageNet-C (floor 0.53) | EuroSAT (floor 1.51) |
|---|---|---|---|
| filing oracle ($y_i$, $g_i$) | −1.50 | −0.71 | −6.50 |
| `labeloracle_zs_ce` ($\hat y_i$, $u_i$) | **+1.70** | **+1.51** | **+26.70** |
| `labeloracle_true_ce` ($y_i$, $u_i$) | −0.25 | −0.91 | +8.09 |

The ImageNet-C +1.51 has a caveat: the same photo appears in all 15 corruptions, so with label values the other
copies carry the query's label. With those copies hidden the gain is +0.92 (see
[global_retrieval.md](global_retrieval.md)).
