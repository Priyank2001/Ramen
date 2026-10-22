# Thesis Handoff Brief: Beating Ramen on Mixed-Domain Test-Time Adaptation

## Context
MTech thesis at IISc (advisor: Prof. Soma Biswas). Prior direction was OCR/scene-text TTA;
pivoted to: **beat the results of "Ramen: Robust Test-Time Adaptation of Vision-Language
Models with Active Sample Selection" (arXiv:2604.21728), or match/exceed papers at
equivalent accuracy on the same benchmarks.**

Target benchmarks (from the paper): CIFAR-10-C, CIFAR-100-C, ImageNet-C (15-corruption
mixture), and DomainNet (6-domain mixture), evaluated under *mixed-domain* TTA — test
samples from multiple domains fully interleaved, domain identity unknown at test time.
Backbones: ViT-B/32, ViT-B/16, ViT-L/14 (CLIP), plus a ResNet-101 generalization check.

## How Ramen Works (mechanism summary)

**Core reframing:** instead of one shared adapted model for the whole test stream, give
every test sample $x_i$ its own adapted weights $w_i$, computed on the fly from a custom
support set $\mathbb{S}_i$ of previously-seen samples judged "relevant."

**Active sample selection — two criteria, one retrieval rule:**
- *Domain consistency*: retrieve samples with high image-embedding similarity to $x_i$
  (proxy for same-domain, since true domain labels are unknown). Validated empirically:
  same-domain rate drops monotonically from ~40% (top similarity decile) to ~4.5%
  (bottom decile).
- *Prediction balance*: retrieve equally from each pseudo-class, via a class-split FIFO
  memory $\mathcal{M}_1,\dots,\mathcal{M}_C$ (capacity $K$ per class). For each new sample,
  pull top-$k$ most similar entries from *each* class queue → support set is automatically
  class-balanced and domain-consistent simultaneously.

**Embedding-gradient cache (the efficiency trick):** entropy loss over a batch decomposes
linearly into a weighted sum of per-sample losses, so gradients also decompose linearly:
$$\nabla_w H(\mathbb{S}_i) = \sum_{j\in\mathbb{S}_i} \alpha_{ij}\, g_j$$
This means you never need to recompute gradients on the retrieved support set — cache each
sample's embedding $z_j$ AND its own gradient $g_j$ once, on arrival, then aggregation at
inference time is just a weighted sum of cached vectors. This is what makes per-sample
adaptation computationally viable (490x speedup vs. naive recomputation; ~14 min vs ~116
hours on CIFAR-100-C).

**Aggregation weights:** $\alpha_{ij} = \exp(-H(x_j)) \cdot \exp(-\beta\|z_i-z_j\|_2)$ —
entropy weighting (trust confident samples) × similarity weighting (trust close samples).
Both hand-designed, fixed functional form.

**Per-sample algorithm:**
1. Compute $x_i$'s embedding, pseudo-label, and gradient.
2. Push (embedding, gradient) into the pseudo-labeled class queue.
3. Retrieve support set (top-$k$ per class, by similarity).
4. Aggregate cached gradients → one gradient step → temporary adapted weights $w_i$.
5. Infer on $x_i$ with $w_i$, then reset to base weights before the next sample.

**Theoretical result (Theorem 4.1):** for a single normalization layer in binary
classification, one entropy-minimization gradient step amplifies feature $h$ in proportion
to $(t_{1h}-t_{0h})^2 \cdot M_{hh}$, where $M_{hh}$ is the feature's variance within the
support set. The problem: entropy minimization can't distinguish class-relevant variance
from domain-relevant variance — under mixed domains, domain-specific features can get
spuriously amplified. Domain consistency suppresses $M_{hh}$ for domain-relevant features
(they're now near-constant within a same-domain support set); prediction balance keeps
$M_{hh}$ high for class-relevant features (predictions still vary across a balanced set).

**Key ablation findings:** both criteria (DC, PB) independently help; removing both
approximates random-subset adaptation and is the worst variant. Both components of
$\alpha_{ij}$ (entropy weighting, similarity weighting) also independently help.
Hyperparameters ($K$, $k$, $\beta$) are robust across wide ranges — Ramen uses the *same*
hyperparameter set for single-domain and mixed-domain settings, unlike baselines which
need separate tuning per setting.

## Open Angles for Improvement (candidates to investigate)

1. **Learned retrieval metric.** Domain-consistency proxy is raw CLIP embedding similarity
   — fixed, not learned or adapted to the task. A learned/contrastive retrieval metric
   trained to predict "same domain" more directly could outperform raw cosine similarity.
2. **Learned aggregation weights.** $\alpha_{ij}$ is a fixed hand-designed function
   (entropy × similarity, both exponential kernels). Could be replaced by a small learned
   weighting function, or made adaptive per-batch/per-stream.
3. **Adaptive hyperparameters.** $K$, $k$, $\beta$ are fixed for the whole stream. Making
   these adaptive to the *local* degree of domain mixing (e.g., higher $k$ when the stream
   looks more homogeneous) is unexplored.
4. **Pseudo-label noise in memory.** Class-split memory is populated via zero-shot
   pseudo-labels, which are noisiest early in the stream / under heavy shift. A
   confidence-gated or delayed-write memory policy could reduce contamination of the
   support-set pool.
5. Possible cross-check: does the same mechanism transfer cleanly to non-CLIP backbones,
   or other TTA objectives beyond entropy minimization (worth checking that the
   embedding-gradient cache's linearity assumption still holds)?

## What's needed from here
- Bring in your existing notebook + experiment results so Claude Code has the current
  starting point (what's implemented, what's been tried, current numbers vs. Ramen's
  reported baselines).
- Decide which of the four angles (or a combination) to prioritize before scaling up
  compute — cheap ablations on CIFAR-100-C first, matching Ramen's own experimental
  protocol for a fair comparison.
