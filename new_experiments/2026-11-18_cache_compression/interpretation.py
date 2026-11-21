"""Interpretation text for 01_cache_compression.ipynb (kept apart so README.md and the notebook say the same thing)."""

INTERPRETATION = r"""## 6. Interpretation

**Verdicts (pre-registered rule, seed 0, domain-averaged).**

| Dataset (floor) | int8 (54% memory) | int4 (29%) | sign, 1 bit (10.5%) |
|---|---|---|---|
| ImageNet-C (0.54) | lossless; beats queue by +0.12: **no** | lossless; +0.28: **no** | lossless; +0.64: **yes** |
| DomainNet (0.11) | lossless; +0.09: **no** | lossless; +0.22: **yes** | lossless; +0.49: **yes** |
| EuroSAT (1.51) | lossless; −0.15: **no** | lossless; −0.46: **no** | lossless; −1.47: **no** |

*Lossless-enough* held for all nine compressed runs: the largest change vs Ramen is 0.17 points (EuroSAT sign),
and every one is inside its noise floor. *Beats shrinking* held for three of nine: 1-bit on ImageNet-C and DomainNet,
4-bit on DomainNet. At 54% memory, a queue of half the length was also within the floor on all three datasets, so
int8 buys nothing that a shorter queue does not. On EuroSAT the shorter queue was the better use of memory at every
level.

**Why compressing the values costs almost nothing.** Ramen never uses a stored gradient on its own. It takes a
weighted sum of the retrieved gradients, $\hat g = \sum_j w_j g_j$, and the step is $-\eta\,\operatorname{sign}(\hat g)$.
Only the *sign of each coordinate of the sum* matters. Rounding error in each $g_j$ is small and unrelated across
entries, so it mostly cancels in the sum, and it can only flip coordinates where $\hat g$ is already near zero; those
coordinates are the ones whose update direction is uncertain anyway. Even the 1-bit codec keeps each entry's sign
and the size of each 64-value block, which is close to what a sign-of-sum step needs (the same reason 1-bit and
sign compressors work in distributed SGD). Evidence: the compressed runs predict the same class as Ramen on
91–99.7% of images, and the changes are almost symmetric (DomainNet sign: 2,081 Ramen errors fixed, 2,209 Ramen-right
images broken).

**Why a shorter queue costs more on ImageNet-C and DomainNet.** Each class queue holds the most recent
high-priority images filed under that class. With fewer slots, retrieval picks from fewer and less varied
neighbours, so $\hat g$ is built from less evidence. The loss grows steadily as the queue shrinks (DomainNet
−0.08 → −0.22 → −0.52 points; ImageNet-C −0.08 → −0.25 → −0.73), while the compressed runs stay flat.

**EuroSAT is the exception, and it says more about Ramen's EuroSAT setting than about compression.** A queue of 789
instead of 7,500 entries gave +1.64 points (444 more images right: 710 fixed, 266 broken), just above the 1.51 floor.
The 7,500 capacity was copied from the CIFAR-10-C config and never tuned for EuroSAT. This is post-hoc, single-seed
and barely past the floor; treat it as a lead for a capacity sweep with seeds 0–4, not a finding.

**Costs we did not plan for.**
- *Speed.* Decoding runs on every retrieval, in plain PyTorch with fp32 intermediates. DomainNet runs took 69–94
  min compressed vs 39 min uncompressed (1.8–2.4×); ImageNet-C took 12–13 vs 8.7 min. Only the retrieved top-k rows
  are decoded already; a fused decode kernel would be the next step. Speed was not optimised here.
- *Peak GPU memory falls less than the cache does.* The 1-bit cache is 9.5× smaller (5.65 → 0.59 GB on ImageNet-C),
  but peak memory drops from 13.5 to 8.4 GB, because per-sample gradients and activations for a batch of 100 make up
  the rest. On DomainNet (7.80 → 0.82 GB cache) the peak drops from 10.2 to 3.3 GB.

**Limits.** One seed. The noise floors come from Ramen's seed spread, not from these variants'. The block size (64)
and the codecs' rounding (nearest, not stochastic) were fixed in advance and not tuned. Queue capacity and codec
were not combined (for example a shorter int8 queue on EuroSAT).

**What this changes.** Memory is not a reason to keep Ramen's cache small: on the two large-label datasets a 1-bit
value store keeps accuracy at 10.5% of the memory, and a shorter queue at the same memory loses 0.5–0.7 points. The
next steps, if memory matters: (1) seeds 1–4 for `cc_sign` and `cc_cap_sign` on ImageNet-C and DomainNet, the two
pre-registered passes with the largest margins; (2) a EuroSAT capacity sweep (K = 7,500 / 2,000 / 800 / 200) with
seeds, since that is the one result here that could change accuracy rather than memory; (3) a fused decode, if the
slowdown matters.
"""
