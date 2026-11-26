# Plan: grad_mix (mix Ramen's active-set gradient with the image's own entropy gradient)
*2026-11-27. Status: draft. Method variant (label-free).*

## Question and hypothesis
- **Question.** If Ramen steps on a mix of its aggregated active-set gradient and the current image's own entropy gradient, G_i = lambda g_R(i) + (1 - lambda) g_H(i), is there a lambda < 1 that predicts better than Ramen (lambda = 1)?
- **Hypothesis.** The lambda chosen on the CIFAR-10-C held-out corruptions (seed 0) beats Ramen on the CIFAR-10-C test stream (seed 0, batch 100, RTX 3080 Ti) by more than the CIFAR-10-C noise floor (0.184 points, `experiments/week_01_2026-11-02/2026-11-01_baseline/baseline.ipynb`).
- **Prior evidence, both ways.** `grad_alignment.ipynb` (Finding 2): g_H is almost the true-label CE gradient on zero-shot-right images and points away from it on zero-shot-wrong ones, while g_R is above the null on right images and at the null on wrong ones. A mix gives right images a better step and wrong images a worse one. Which effect wins on accuracy is open.

## Derivation
Ramen, per batch (`src/methods/Ramen.py:111-170`; `ramen_lab/methods/stored_value.py` re-implements it as named steps):
1. z_i = f(x_i), p_i = softmax(W^T z_i), yhat_i = argmax p_i, H_i = entropy(p_i)
2. g_H(i) = grad_theta_i H_i (per-sample LayerNorm affine copy; `compute_values`)
3. write (z_i, g_H(i), H_i, t_i) into queue M_{yhat_i}
4. g_R(i) = (1/|C+|) sum_c sum_{j in topk_c(i)} exp(-H_j) exp(-beta d_ij) g_H(j)   (`retrieve`)
5. theta_i = theta_0 - lr sign(G_i), with **G_i = g_R(i)** in Ramen; predict; reset

**The changed term (step 5):**
- `raw` (the literal mix): G_i = lambda g_R(i) + (1 - lambda) g_H(i)
- `unit` (scale-free mix): G_i = lambda g_R(i)/||g_R(i)|| + (1 - lambda) g_H(i)/||g_H(i)||

lambda = 1 is Ramen (bit-identical, tested). lambda = 0 is a one-step, per-image, episodic entropy step from theta_0 (Tent-like, no memory).

Why two modes: SignSGD uses only sign(G_i), so in `raw` what decides each coordinate is lambda ||g_R|| versus (1 - lambda) ||g_H||, not lambda. g_R sums up to topk entries per queue (topk 50 on CIFAR-10-C) with weights exp(-H_j) e^{-beta d}, so its norm need not match g_H's, and the raw sweep could be flat until lambda is close to 0. `unit` makes lambda the actual share. Norms are logged per image so the effective share of the raw mix is known.

Note: Ramen writes before it retrieves, so g_R(i) already contains g_H(i) once (its own entry, distance 0, weight exp(-H_i)/|C+|). The mix adds more of it; it does not add something g_R lacks.

Predicted effect [hypothesis]: lowering lambda helps zero-shot-right images (their g_H is near the CE gradient) and hurts zero-shot-wrong ones (their g_H points to the wrong class, i.e. more confirmation of the zero-shot error). Accuracy changes only through flips, so the result depends on how many wrong images Ramen currently fixes (their step would get worse) versus how many right images it currently breaks (their step would get better).

How it can fail or mislead:
1. **Held-out stream is small.** 4 extra corruptions x 10,000 = 40,000 images; Ramen's memory holds 10 x 7,500 = 75,000, so it never fills and never evicts there. lambda chosen there may not transfer to the 150,000-image test stream.
2. **fp16.** sign(G_i) is passed to the optimizer directly (exact +-1 in fp16), so a tiny mixed coordinate cannot underflow to 0 and lose its sign. Requires SignSGD without momentum (checked in the constructor).
3. **Single seed** for the screen.

## Variants
| name | class | config (on top of `cfg/CIFAR10C/Ramen.yaml`) | what it changes | labels |
|---|---|---|---|---|
| GradMix100 | `RamenGradMix` | `mix_lambda: 1.0` | nothing: identity with Ramen | no |
| GradMixRaw{10,25,50,75,90} | `RamenGradMix` | `mix_lambda: 0.1 ... 0.9`, `mix_norm: raw` | literal mix | no |
| GradMixUnit{10,25,50,75,90} | `RamenGradMix` | same, `mix_norm: unit` | scale-free mix | no |
| GradMix0 | `RamenGradMix` | `mix_lambda: 0.0` | own entropy gradient only | no |
| NoAdapt | upstream | - | zero-shot reference | no |
lambda = 0 and 1 are the same in both modes (sign is scale-invariant), so each runs once. 12 runs per stream.

## Files
- `ramen_lab/config.py`: `RunConfig.domains` = `main` (default; unchanged run ids and metrics.json config) or `heldout` (only the 4 extra corruptions, run id suffix `_heldout`).
- `ramen_lab/upstream.py`: `load_datasets(..., domains)`, loads `extra=True` and keeps only the extra corruptions.
- `ramen_lab/runner.py`: passes `config.domains`.
- `ramen_lab/methods/grad_mix.py`: `RamenGradMix(RamenStoredValue)`, mixes in `retrieve`; mix strategies `raw` / `unit` in a registry.
- `tests/test_lab_grad_mix.py`, additions to `tests/test_lab_units.py` (held-out config).
- `experiments/week_04_2026-11-23/2026-11-27_grad_mix/{grad_mix.py, grad_mix.ipynb}`.

## Tests
lambda = 1 bit-identical to upstream Ramen (filled, evicting queues); lambda = 0 equals a step on sign(g_H) alone; the `unit` mix at lambda = 1 equals Ramen; constructor refuses a non-SignSGD optimizer, momentum, lambda outside [0, 1], and an unknown mode; the shared invariants (determinism, reset, frozen weights, label independence) via `test_lab_invariants.py`; held-out config: run id suffix, default config dict unchanged, DomainNet refused.

## Runs and resources
1. Check: 200 per domain, `GradMix100` vs Ramen 100.00% identical, every variant once (smoke: RAM, GPU, minutes).
2. Held-out sweep: CIFAR-10-C `heldout`, seed 0, 12 variants + NoAdapt, GPU 1 (2080 Ti). All in one chain, so they are paired on one GPU model.
3. Test-stream sweep: CIFAR-10-C main, seed 0, 12 variants, GPU 0 (3080 Ti, the GPU of the seed-0 Ramen baseline). `GradMix100` must equal `pc2_C10_Ramen_b100_s0` on every prediction.
Batch 100 (frozen), RAM budget 8.2 GB (frozen). Ramen CIFAR-10-C took about 5 minutes per full run on the 3080 Ti, so about an hour per chain.

## Pre-registered reading
- **Selection (held-out only).** lambda* = the variant with the highest held-out seed-0 accuracy among the 12. If lambda* is GradMix100 (Ramen), the direction stops here.
- **Gate (test stream).** lambda* minus Ramen seed 0 (`pc2_C10_Ramen_b100_s0`) > 0.184 points. Pass: seeds 1-4 on the GPU model of each baseline seed, then ImageNet-C and DomainNet with lambda* unchanged (DomainNet has no extra domains; it is not re-tuned). Fail: stop, record in `docs/research_state.md`.
- **Diagnostic only.** The test-stream curve over all lambda is reported, but no lambda other than lambda* is ever compared against Ramen as a result. Split of accuracy by zero-shot right / wrong, fixed / broken counts versus Ramen, and the logged norm ratio and sign-flip share are explanations, not gates.

## Risks
- Held-out stream never fills memory (derivation, failure 1).
- Corruption sets: one base image appears under every corruption; bootstrap by base image.
- A mixed step that is right on CIFAR-10-C may not be on ImageNet-C, where zero-shot is wrong on most images.
