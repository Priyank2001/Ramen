# Plan: grad_alignment
*2026-11-06. Status: done, seed 0 on CIFAR-10-C, CIFAR-100-C, DomainNet, ImageNet-C; notebook executed. ORACLE diagnostic: reads labels, changes no prediction.*

## Question and hypothesis
- **Question.** How often does the direction Ramen steps in agree with the cross-entropy (CE) gradient toward the true label?
- **Hypothesis.** On CIFAR-10-C seed 0, for most samples the cosine between Ramen's aggregated retrieved gradient g_R and the CE gradient g_CE is above zero, and above a shuffled null (g_R against another sample's g_CE). It is lower on samples the zero-shot model gets wrong.

## Derivation
Per sample i, with theta_i = its own copy of the LayerNorm affine parameters (`BySampleLayerNorm`):
- g_H(i) = grad_theta_i H(p_i) (own entropy gradient, `Ramen.forward` backward; the value written to memory)
- g_R(i) = (1/n_cls) sum_c sum_{k in topk_c(i)} exp(-H_k - beta d_ik) g_H(k) (`Ramen.forward`, retrieval)
- applied step: delta_i = -lr * sign(g_R(i)) (SignSGD, `src/models/optimizer.py`)
- g_CE(i) = grad_theta_i CE(p_i, y_i) (ORACLE, never used by the update)

Measured: cos(g_R, g_CE), cos(sign g_R, g_CE) (the step actually taken), cos(g_H, g_CE) (Tent-like reference), cos(g_R, g_H), per-layer cosines, sign agreement, first-order CE change -lr<sign g_R, g_CE>, and the realised CE before/after. A positive cosine means the descent step on g_R also lowers CE to first order.
Can fail because: (1) fp16 CE gradients of confident samples underflow (mitigated by loss scale 1024; check compares scale 1 vs 1024); (2) a large cosine with the shuffled null means a generic, sample-agnostic direction, not sample-specific agreement; (3) sign() changes the direction, so the two cosines are reported separately.

## Variants
| name | class | what it changes | labels |
|---|---|---|---|
| RamenGradAlignment | `ramen_lab/methods/grad_alignment.py` | nothing in the update; probes only | ORACLE (reads labels for the probe) |
Control: the shuffled null (`cos_R_CEnull`) and the entropy reference (`cos_H_CE`).

## Files
`ramen_lab/analysis/gradients.py`, `ramen_lab/methods/grad_alignment.py`, `tests/test_lab_grad_alignment.py`, `experiments/week_01_2026-11-02/2026-11-06_grad_alignment/{grad_alignment.py,grad_alignment.ipynb}`.

## Tests
Predictions identical to Ramen over a stream; no probe without labels; predictions do not depend on the labels read; CE gradient equals a single-sample backward; the cosine helpers; plus the shared invariants (determinism, reset, frozen weights).

## Runs and resources
Subset check (200 per domain, identity 100%) -> CIFAR-10-C seed 0 (GPU 0, batch 100) and CIFAR-100-C seed 0 (GPU 1, batch 32) at the frozen batch. Accuracy must equal the matching Ramen baseline run on the same seed and GPU model.

## Pre-registered reading
Diagnostic only; there is no accuracy gate. Report the distributions (median, fraction above 0), against the null and the g_H reference, split by zero-shot correct/wrong and by domain. No claim about "alignment causes gains" without a further test.

## Risks
Single seed; corruption-domain heterogeneity; cosine in ~40k dimensions is dominated by a few large coordinates (report sign agreement too).
