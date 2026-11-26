# Plan: step_size (per-image step size for Ramen's sign step)
*2026-11-27. Status: draft, not started (waits for GPU 0 after grad_mix). Probe (label-free run) + ORACLE analysis (labels used offline only).*

## Question and hypothesis
- **Question.** Ramen moves every LayerNorm coordinate of every image by exactly eta = 0.01. Would a per-image step size eta_i, chosen from a label-free signal, predict better than the constant step?
- **Two-stage hypothesis.**
  1. *Ceiling (ORACLE).* On the CIFAR-10-C test stream (seed 0, batch 100, RTX 3080 Ti), picking each image's best eta from the grid, with its label, beats Ramen by at least 1.0 point. That is about 5 x the noise floor, leaving room for a label-free rule to recover only part of it.
  2. *Rule.* A rule eta_i = f(s_i) on one label-free signal s, fitted on the held-out corruptions, beats Ramen on the test stream by more than the CIFAR-10-C noise floor (0.184 points, `experiments/week_01_2026-11-02/2026-11-01_baseline/baseline.ipynb`). It also beats a control that gives images the same step sizes shuffled among them.
- **Closed neighbours.** `docs/research_state.md`: a global eta is closed ("defaults are already optimal on the dev stream"). The per-coordinate agreement-scaled step is closed because it mostly acted as a smaller eta. That second result is why the shuffled-step control below is mandatory.

## Derivation
Ramen, per batch (`src/methods/Ramen.py:111-170`; `ramen_lab/methods/stored_value.py` names the steps):
1. z_i = f_{theta_0}(x_i), p_i = softmax(W^T z_i), yhat_i = argmax p_i, H_i = entropy(p_i)
2. g_H(i) = grad_theta_i H_i at theta_0
3. write (z_i, g_H(i), H_i, t_i) into queue M_{yhat_i}
4. G_i = (1/|C+|) sum_c sum_{j in topk_c(i)} w_ij g_H(j), w_ij = exp(-H_j) exp(-beta d_ij)
5. theta_i = theta_0 - **eta** sign(G_i), predict, reset to theta_0

**The changed term (step 5):** eta becomes eta_i, one scalar per image. The direction sign(G_i) is unchanged, so every coordinate of image i still moves by the same amount, now eta_i instead of 0.01.

**Why one probe run answers everything, offline.** Steps 1-4 happen at theta_0: features, entropies, the stored gradients, the queue and the priorities never see the adapted weights. Step 5 is undone before the next batch. So the step size of image i changes only image i's own prediction, never the memory or any later image. The probe can therefore evaluate every eta in a grid for every image in one run, and any per-image rule that picks its eta_i from that grid can then be scored **exactly** offline, with no further GPU run.

Grid: E = {0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08}. eta = 0 is the zero-shot prediction; 0.01 is Ramen.

Signals (label-free, all at theta_0, logged per image):
- H_i, entropy of the zero-shot prediction
- margin m_i = p_(1) - p_(2) between the top two zero-shot probabilities
- ||G_i||, Ramen's aggregate gradient norm
- ||g_H(i)||, the image's own gradient norm
- cos(G_i, g_H(i)), agreement between the memory and the image itself
- M_i = sum of the retrieved weights w_ij, the active-set mass (how close and confident the retrieved entries are)
- self share, the weight of the image's own entry in its active set (logged by RamenStoredValue)

Predicted effect [hypothesis]: grad_alignment (Finding 2) found the step direction is useful on zero-shot-right images and at the null on zero-shot-wrong ones. A signal that marks "direction trustworthy" (high cos(G, g_H), high M, low H) should want a larger eta, and the rest a smaller one, down to 0.

How it can fail or mislead:
1. **The oracle is optimistic.** The best of 7 predictions per image, picked by label, gains even if eta carries no information. Report it next to a random-eta control (each image a uniformly random grid eta, seeded, 20 draws) and the best single global eta (the closed hyperparameter, as a reference).
2. **"Smaller on average" masquerades as adaptivity** (the agreement-step trap). The shuffled control keeps the rule's exact multiset of step sizes and permutes them across images (20 permutations). The rule must beat it, not just Ramen.
3. **Transfer.** The rule is fitted on 40,000 held-out images whose memory never fills (10 x 7,500 capacity). Bin edges are quantiles of the held-out signal, applied unchanged to the test signal.
4. **fp16.** The LayerNorm copies are fp16. The fp16 spacing is about 0.001 for values in [1, 2) and 0.002 in [2, 4), so a 0.0025 step is rounded noticeably on larger weights, and smaller steps could vanish on very large ones. Log, per eta, the share of coordinates whose value actually changed.

## Variants
| name | class | config (on top of `cfg/CIFAR10C/Ramen.yaml`) | what it does | labels |
|---|---|---|---|---|
| StepProbe | `RamenStepProbe` | `probe_etas: [0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]` | Ramen; after computing G_i it also predicts at every eta in the grid and logs those predictions plus the signals. Returns Ramen's eta = 0.01 prediction | no |
| (offline) ORACLE best eta | analysis | - | per image, the best grid eta by label | **ORACLE** |
| (offline) random eta | analysis | - | per image, a random grid eta (control for 1) | no |
| (offline) best global eta | analysis | - | one eta for all images (reference) | test labels, reference only |
| (offline) rule f(s) | analysis | - | quintile bins of s on held-out; per bin, the grid eta with the best held-out accuracy | held-out labels only |
| (offline) shuffled rule | analysis | - | the rule's eta values permuted across test images (control for 2) | no |
Identity: StepProbe's returned predictions must equal `pc2_C10_Ramen_b100_s0` on every image, and its eta = 0.01 column must equal them too.

## Files
- `ramen_lab/methods/step_probe.py`: `RamenStepProbe(RamenStoredValue)`. It overrides `adapt_and_predict`: first Ramen's 0.01 pass (the returned logits), then the other etas, each via `upstream.rearm_by_sample_grads` + `set_by_sample_grad(G)` + an optimizer lr set to eta + step + predict + `reset_parameters`. The lr is restored to 0.01 after the batch. It also logs the signals.
- `tests/test_lab_step_probe.py`.
- `experiments/week_04_2026-11-23/<start date>_step_size/{step_size.py, step_size.ipynb}`: the CLI (check / run) and the offline rule evaluation (pure functions).

## Tests
- Returned logits bit-identical to upstream Ramen (filled, evicting queues).
- The eta = 0.01 column equals the returned argmax, and the eta = 0 column equals the zero-shot argmax.
- The optimizer lr is 0.01 again after every batch, and weights are back at theta_0.
- Predictions at an eta do not depend on which other etas are in the grid (no state leaks between passes).
- The shared invariants via `test_lab_invariants.py` (determinism, reset, frozen weights, label independence).
- Offline rule code: quantile bins fitted on one array are applied unchanged to another; the shuffled control keeps the multiset of etas.

## Runs and resources
1. Check: 200 per domain, StepProbe vs Ramen 100.00% identical; smoke run for RAM, GPU and minutes.
2. CIFAR-10-C held-out (4 extra corruptions), seed 0, GPU 1.
3. CIFAR-10-C test stream, seed 0, GPU 0 (the GPU model of `pc2_C10_Ramen_b100_s0`, so per-image comparison is valid).
Batch 100, RAM budget 10 GB (grad_mix measured 7.6 GB peak on the full test stream). Seven forward passes per batch instead of one, so the expected time is about 2-3 x a Ramen run (Ramen: about 3.6 min on the 3080 Ti).

## Pre-registered reading
- **Stage 1 (ceiling), test stream.** The ORACLE best-eta accuracy minus Ramen. **Stop** if it is below 1.0 point. Reported but not gated: random-eta control, best global eta, and fixed / broken counts against Ramen split by zero-shot right / wrong.
- **Stage 2 (rule).** For each signal: quintile edges on held-out, best grid eta per bin by held-out accuracy, applied to the test stream. The signal is chosen by held-out accuracy only. **Gate:** chosen rule minus Ramen > 0.184 **and** chosen rule minus the mean of the shuffled control > 0.184. Pass: implement it as an online method (it must equal the offline score on seed 0, which is a correctness check), then seeds 0-4 on the GPU model of each baseline seed, then ImageNet-C and DomainNet with the bins and etas fixed. Fail: stop, and record in `docs/research_state.md`.
- **Diagnostic only.** Accuracy against eta per signal bin; which signal separates "wants a smaller step" from "wants a larger step" among images whose prediction depends on eta.

## Risks
- Held-out stream never fills the memory (derivation, failure 3).
- Single seed for both stages. The gate is a screen, not a claim.
- Quintile bins x 7 etas = 35 choices per signal, 7 signals: some held-out overfitting is expected. The shuffled control and the held-out/test split bound it.
- If grad_mix passes its gate, its direction would be the better base. This plan keeps Ramen's direction so the two ideas are tested one at a time.
