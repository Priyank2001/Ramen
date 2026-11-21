# Global top-k retrieval and copy exclusion (`RamenGlobalRetrieval`)

Subclass of `RamenLabelValues`, so filing, write timing and the reveal of label values are as in
[label_values.md](label_values.md). Notation and the Ramen baseline are in [README.md](README.md).
**Uses true labels only** with `value: true_label_ce`. `global_entropy_k100` uses none.

## Why

Ramen takes $k$ neighbours from **every** non-empty class queue and averages the queues with equal weight. Every
class gets a vote, however far away its entries are. Even with perfect label values, the image's own class received
only ~2.5% of the update weight on DomainNet. Global retrieval asks whether that queue-balanced vote, rather than
the values, is the limit.

## What changes: retrieval (`retrieval`)

**`per_class`** (default): Ramen's, $\mathcal{N}_k^c(z_i)$ per non-empty queue. Without exclusion this calls
Ramen's own code path.

**`global_knn`**: the $k_g$ nearest entries of the whole cache, whatever queue they sit in:

$$\mathcal{G}_{k_g}(z_i) = \text{the } k_g \text{ entries of } \textstyle\bigcup_c \mathcal{M}_c \text{ with the smallest } d_{ij}$$

$$\hat g_i = \frac{\sum_{j\in\mathcal{G}_{k_g}(z_i)} \omega_{ij}\, v_j}{\sum_{j\in\mathcal{G}_{k_g}(z_i)} \omega_{ij}},
\qquad \omega_{ij} = e^{-H_j}\, e^{-\beta d_{ij}}$$

The weights are Ramen's. Normalising by $\sum_j \omega_{ij}$ instead of by $|C_i|$ does not change the SignSGD
step (README, "Baseline"). What does change: the prediction balance of one top-$k$ per class is gone, so classes
with many nearby entries get more of the vote. A neighbour from batch $t$ itself (including the image's own entry)
still holds its entropy gradient $g_j$. Neighbours from earlier batches hold $u_j$.

## What changes: copy exclusion (`exclude_groups`), ImageNet-C only

`ImageNetC5K` uses the same 5,000 photos in all 15 corruptions. With label values, another corrupted copy of the
query's photo carries the query's own label, which leaks the answer. Each stream position gets a group id
$\gamma(\cdot)$, the base photo (`sample_idx`, resolved from Ramen's cached run by `base_image_groups`). Retrieval
then ignores the image's **other** copies, but keeps its own entry:

$$d_{ij} \leftarrow \infty \quad \text{if } \gamma(j) = \gamma(i) \text{ and } j \neq i,
\qquad \omega_{ij} = 0 \text{ when } d_{ij} = \infty .$$

With exclusion, `per_class` runs this class's own top-$k$ with the same normalised sum as above, in fp32.
`exclude_groups: 'unique'` (every image its own group, so nothing is hidden) checks that this code path matches
Ramen's up to fp16 vs fp32 numerics.

## Variants run

| run | retrieval | $k$ | value | copies hidden | labels? |
|---|---|---|---|---|---|
| `global_off` (smoke) | per_class | `topk` | $g$ | no | no; must equal Ramen |
| `global_ce_k10` / `_k100` / `_k1000` | global_knn | 10 / 100 / 1,000 | $u$ | no | yes |
| `global_entropy_k100` | global_knn | 100 | $g$ | no | **no** |
| `excl_zs_entropy` | per_class | `topk` | $g$ | yes | no (Ramen with copies hidden) |
| `excl_zs_ce` | per_class | `topk` | $u$ | yes | yes |
| `excl_global_ce_k10` / `_k100` / `_k1000` | global_knn | 10 / 100 / 1,000 | $u$ | yes | yes |
| `excl_global_entropy_k100` | global_knn | 100 | $g$ | yes | no |
| `excl_numerics` (smoke) | per_class | `topk` | $g$ | `unique` | no |

## Code map

| symbol / step | code | config key |
|---|---|---|
| choice of $\mathcal{N}$ vs $\mathcal{G}$ | `RamenGlobalRetrieval.retrieve` | `retrieval` |
| $k_g$ | `self.global_k` | `global_k` (default 100) |
| $d_{ij}$, exclusion | `_distances(feats, c)`: `torch.cdist` then `masked_fill(same, inf)` | `exclude_groups` |
| stream position of each slot | `self.slot_pos[c, slot]`, set in `write` | — |
| $\gamma$ | `LabelOracleExperiment.base_image_groups()` | `exclude_groups: base_image` |
| classes in the retrieved set | `stats['classes_in_set']` | — |

`check_exclusion.py` (X1–X2) shows that no retrieved entry is another copy of the query's photo. Without exclusion,
copies carry 0.0% of the update with per-class top-1 vs 4.0% with global top-10.

## Results (seed 0, change in domain-averaged points)

| run | DomainNet (floor 0.11) | ImageNet-C, copies visible (floor 0.53) | ImageNet-C, copies hidden | EuroSAT (floor 1.51) |
|---|---|---|---|---|
| per-queue label values (`labeloracle_zs_ce` / `excl_zs_ce`) | +1.70 | +1.51 | **+0.92** | +26.70 |
| `global_ce_k10` | **+2.86** | +3.97 | −1.00 | +23.11 |
| `global_ce_k100` | +2.54 | +3.82 | +0.18 | **+27.59** |
| `global_ce_k1000` | +1.84 | +2.80 | +0.95 | +25.91 |
| `global_entropy_k100` (no labels) | +0.04 | −0.86 | −1.57 | −7.51 |

The "copies hidden" column is measured against Ramen with copies hidden (`excl_zs_entropy`, −0.05 vs Ramen). The
other columns are vs Ramen.

- **DomainNet**: global top-10 adds +1.16 over per-queue label values, mostly on quickdraw (21.56 → 27.04). This
  is not a near-copy leak (see [analyses.md](analyses.md)).
- **ImageNet-C**: the apparent +3.97 was the copy leak. With copies hidden, global retrieval adds nothing.
- **EuroSAT**: +0.89 over per-queue, within the floor.
- Without labels, global retrieval never helps.
