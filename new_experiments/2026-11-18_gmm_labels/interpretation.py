"""Interpretation cell of 01_gmm_labels.ipynb (kept here so README and notebook quote the same numbers)."""

INTERPRETATION = r"""## 6. Interpretation

**Verdict: hypothesis not supported.** No label-free variant passes the pre-registered bar on any dataset.

| Dataset | zero-shot (pooled) | best label-free at iteration 10 | fixed / broken | verdict | LDA oracle (true labels) |
|---|---|---|---|---|---|
| DomainNet | 52.83 | `lda` 41.73 (−11.10) | 17,614 / 83,742 | hurts | 55.82 (+2.99) |
| ImageNet-C | 27.29 | `lda` 23.29 (−4.00) | 4,807 / 7,804 | hurts | 63.05 (+35.76) |
| EuroSAT | 48.07 | `lda` 63.39 (+15.33) | 5,438 / 1,300 (10.0% of zs-right) | weak (break rate > 5%) | 94.99 (+46.92) |

vMF variants are worse everywhere except EuroSAT (+4.73): DomainNet 27.61 (−25.22), ImageNet-C 6.97 (−20.32).
A learned prior never helps (it lowers accuracy or leaves it unchanged in every pair).

**What the numbers say.**
- *EM finds structure, but not classes.* The log-likelihood rises at every iteration on every dataset, while
  accuracy falls on DomainNet and ImageNet-C. The components' NMI with the **domain** rises (DomainNet vMF
  0.10 → 0.28, LDA 0.10 → 0.19; ImageNet-C vMF 0.11 → 0.42, LDA 0.11 → 0.20). Their NMI with the **class**
  falls for vMF (DomainNet 0.57 → 0.45, ImageNet-C 0.56 → 0.44) and for LDA on DomainNet (0.57 → 0.55). The
  densest structure in CLIP image space on these mixed streams is domain or corruption, so maximising likelihood
  pulls the class components towards it. The same domain dominance appeared in 2026-11-03 (quickdraw coherence)
  and 2026-11-06 (cluster shift).
- *LDA on ImageNet-C is a second failure mode.* Class NMI stays flat (0.56 → 0.57) while accuracy falls
  30.19 → 23.29. NMI ignores names, so the components keep class-related groupings but attach them to the wrong
  class names. This is consistent with label switching (shown on synthetic data in `test_gmm.py`), but it was not
  measured directly here.
- *EuroSAT is the exception for the same reason.* It has one domain, so the clusters are the classes (as in the
  cluster-shift run) and EM moves labels the right way: +15.33 points, 5,438 fixed. But 1,300 of 12,978
  zero-shot-right images (10.0%) break. On EuroSAT the kNN target broke 1,381 and cost Ramen 10 points (2026-11-04).
- *The model family limits DomainNet even with labels.* Fitted on the test images with their true labels, one
  Gaussian per class reaches only 55.82% pooled (+2.99), and the vMF oracle is below zero-shot (−5.08).
  Hypothesis (not tested): a DomainNet class spans six styles (real photo to quickdraw), so one mode per class cannot
  represent it, while the text embedding is style-neutral.
- *Iteration 1 of `lda` on ImageNet-C is +2.90 (30.19%)*, then it declines. That is a post-hoc pick, below the bar,
  and it does not hold on DomainNet (iteration 1 is −2.34).

**What this does not show.** The fit is transductive with one $\kappa$ (100) and one shrinkage (0.1); other values
were not tried. Mixtures with several components per class, or fitted after removing a domain or cluster offset,
were not tried. Seed does not matter for this fit (order-free), so a single run per dataset is complete, but it is
one model per dataset.

**Next step.** A class mixture over raw CLIP embeddings is a dead end on multi-domain data. The one variant worth
an oracle check is *domain-factored*: subtract the domain (or k-means cluster) mean first, then fit the class
mixture. Run `lda_oracle` on domain-centred embeddings first. If even that ceiling is not far above +2.99 on
DomainNet, drop the direction. On EuroSAT alone, the `lda` label could be tried as Ramen's CE target. It is +15.3
points at 10% breaks; the oracle mix at that accuracy (61%) gave +17.55 with zero breaks. This is the only
GPU-worthy follow-up, and a small-label-space result.
"""
