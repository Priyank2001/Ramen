# Replay measurements (no GPU)

Three diagnostics run on 2026-11-02 to explain the global-retrieval results. All three use the **true labels** of
cache entries, as measurements, not as methods. The first two replay Ramen's cache (zero-shot filing, entropy
values) with `CacheReplayRun` from `2026-11-01_pillow_sink` (a read-only import), on the same sample: 2,000 random
DomainNet images (seed 0) from after the first 20% of the stream. Notation is in [README.md](README.md).

## Neighbour-label vote (`neighbour_labels.py`, `NeighbourLabels`)

With label values, a retrieved entry $j$ pushes toward its own true class $y_j$. So an update can point the right way
only as often as the retrieved entries' labels vote for the query's class. For a retrieved set $S$:

$$\mathrm{vote}_i(c) = \sum_{j\in S,\ y_j = c} \omega_{ij}, \qquad
\text{vote right} \iff \arg\max_c \mathrm{vote}_i(c) = y_i, \qquad
\text{own-class share} = \frac{\mathrm{vote}_i(y_i)}{\sum_c \mathrm{vote}_i(c)}$$

computed for two sets:

- **per-queue**: $S = \bigcup_c \mathcal{N}_{10}^c(z_i)$, Ramen's active set;
- **global-10**: $S = \mathcal{G}_{10}(z_i)$, the 10 nearest of the whole cache. This is exact from the replay,
  because each of them is among its own queue's 10 nearest.

It also counts $|\{j \in \mathcal{G}_{10}(z_i) : y_j = y_i\}|$, the neighbours of the query's own class, and
cross-tabulates vote right/wrong against what zero-shot, Ramen and the label-value runs predicted.

**Result**: on average only 4.5 of the 10 nearest share the query's class (quickdraw 2.1). The global-10 vote is right
for 57.6% of images, against 51.6% for zero-shot.

## Near copies (`near_copies.py`, `NearCopies`)

Checks whether DomainNet's global gain is a leak through near-identical images, the way ImageNet-C's was through
copies of the same photo. For each sampled image $i$, with radius $r = 0.1$:

$$\text{own entry retrieved} = [\, i \in S,\ d_{ii} < r \,], \qquad
\text{near copies}_i = |\{ j \in S : j \neq i,\ d_{ij} < r \}|$$

and the same count restricted to $y_j = y_i$. Its own entry is not a leak, because it still holds $g_i$ when $i$ is
predicted. The gain of a run over a reference is then split between images with at least one near copy and images
without one.

**Result**: 19 of 2,000 images (0.9%) have another entry within 0.1. The gain of `global_ce_k10` over per-queue label
values (`labeloracle_zs_ce`) on the sample is +2.70 points. Without those 19 images it is +2.62. This accuracy is
pooled over the sample, not domain-averaged, so quickdraw (590 of 2,000) counts for more. Not a leak.

## Copy exclusion check (`check_exclusion.py`), ImageNet-C

Tests the exclusion rule in [global_retrieval.md](global_retrieval.md).

- **X1**: with `exclude_groups`, no retrieved entry $j \neq i$ has $\gamma(j) = \gamma(i)$.
- **X2**: without exclusion, the share of the update carried by copies is
  $\sum_{j:\,\gamma(j)=\gamma(i),\,j\neq i} \omega_{ij} \big/ \sum_j \omega_{ij}$.

**Result**: copies carry 0.0% of the update under per-class top-1 and 4.0% under global top-10. That 4.0% was enough
to inflate global top-10 from −1.00 to +3.97. Output: `logs/check_exclusion_ImageNetC5K.log`.
