# Does Ramen's update point toward the true class? (2026-11-18)

**Question.** Ramen adapts each test image with one SignSGD step along $\hat g_i$, a weighted average of *entropy*
gradients retrieved from its per-class memory. How close is that direction to the gradient of the **cross-entropy
toward the true label**, $g^{CE}_i$? We log, for every test image, the cosine similarity between the two (and a few
related directions), on an otherwise unchanged Ramen run.

**Oracle measurement, not a method.** $g^{CE}_i$ uses the ground-truth label. It is computed only to measure; it
never reaches the update, and the probe run predicts exactly what Ramen predicts.

Nothing in `src/` or `cfg/` is changed. `RamenGradientProbe` subclasses `RamenTrueLabel` (zero-shot filing = Ramen)
and is built through a local registry; it starts from `cfg/<dataset>/Ramen.yaml`.

## Files

| file | job |
|---|---|
| `alignment.py` | pure per-image metrics on (B, D) tensors: `cosine`, `projection_share` (a part's additive share of a cosine), `sign_agreement` |
| `ga_methods.py` | `RamenGradientProbe` (ORACLE measurement): Ramen's forward + the true-label CE gradient on a retained graph, the update split into own entry / true-class queue / zero-shot queue / other queues, one dict of per-image arrays per batch in `stats['probe']` |
| `ga_registry.py` | local method lookup before the repo's, and the repo config each local method starts from |
| `ga_configs.py` | datasets (model, num_workers, notebook name), the run `ga_probe`, and the pre-registered hypotheses H1–H3 |
| `ga_experiment.py` | `GradAlignmentExperiment`: harness with the local registry; runs/loads NoAdapt, Ramen, ga_probe; `sanity_check()`; `frame()` = one row per image (outcome + every probe field) |
| `ga_analysis.py` | pure tables: data per domain, outcome counts, direction medians, rolling stream means, decile fix/break rates, decomposition, `hypotheses()` |
| `ga_plots.py` | one function per figure (colour = which gradient: Ramen update blue, own entropy orange, control grey) |
| `run_grad_alignment.py` | CLI: `<dataset> <seed> [--smoke <dir>]`; runs the three, checks probe == Ramen, prints medians |
| `check_gradients.py` | correctness checks G1–G4 (per-image gradient vs batch-of-1, sign convention, decomposition sums to the update, own-entry part) |
| `make_notebook.py` | writes `0N_grad_alignment_<dataset>.ipynb` from shared cells (`SPECS` per dataset) |
| `interpretation.py` | section 7 text per dataset (shared by the notebook and this README) |
| `01_grad_alignment_domainnet.ipynb` | DomainNet notebook (executed) |
| `02_grad_alignment_imagenetc.ipynb` | ImageNet-C 5K notebook (executed) |
| `03_grad_alignment_eurosat.ipynb` | EuroSAT notebook (executed; one domain, so section 4.3 is skipped) |
| `broken_analysis.py` | `BrokenImages`: zero-shot-right images split into broken / kept, kept reweighted to the broken images' confidence; summary, AUROC, break-rate grid, decomposition, stream, own-entry effect, destinations |
| `broken_plots.py` | one function per figure of notebook 04 (colour = image group) |
| `make_broken_notebook.py` | writes `04_broken_images.ipynb` (header, math, interpretation and glossary text live here) |
| `04_broken_images.ipynb` | images Ramen breaks, all three datasets, plus section 6 loss landscape (executed; reads cached runs) |
| `landscape.py` | geometry of the slices: parameter plane (Ramen's step x oracle step), grids, offline embedding-space slice |
| `landscape_methods.py` | `RamenLandscapeProbe` (ORACLE measurement): Ramen + loss / entropy / prediction on the slices for chosen stream positions, z0 / z1, anchor checks |
| `landscape_experiment.py` | `LandscapeExperiment`: choose 50 images per outcome group from the probe frame (saved JSON), run / load `ga_landscape`, `sanity_check_landscape()` |
| `landscape_analysis.py` | `Landscapes`: per-image and per-group tables, 1D line profiles, typical examples, embedding slices |
| `landscape_plots.py` | line profiles and 2D slices |
| `run_landscape.py` | CLI: `<dataset> <seed> [--smoke <dir>]` |
| `landscape_selection_<dataset>_s0[_smoke].json` | the chosen stream positions, per outcome group (fixed before the run) |
| `smoke/` | smoke results (200 images per domain) |
| `logs/` | run, smoke and check logs |

## How to run

```bash
cd new_experiments/2026-11-18_grad_alignment
PY=~/miniconda3/envs/RAMEN/bin/python
CUDA_VISIBLE_DEVICES=0 $PY run_grad_alignment.py DomainNet 0 --smoke smoke > logs/smoke_DomainNet_s0.log 2>&1
CUDA_VISIBLE_DEVICES=1 $PY check_gradients.py DomainNet > logs/check_gradients_DomainNet.log 2>&1
CUDA_VISIBLE_DEVICES=0 setsid nohup $PY run_grad_alignment.py DomainNet 0 > logs/DomainNet_s0.log 2>&1 < /dev/null &
$PY make_notebook.py DomainNet
CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace \
    01_grad_alignment_domainnet.ipynb
```
Same commands with `ImageNetC5K` (notebook 02) or `EuroSAT` (notebook 03). One RTX 3090: DomainNet ~48 min,
ImageNet-C ~12 min, EuroSAT ~3 min.

## Checks

- Smoke (1,200 DomainNet images): `ga_probe` == Ramen on 1,200 of 1,200 images.
- `check_gradients.py` (DomainNet): G1 per-image CE gradient from the batch vs the image alone: cos ≥ 0.99993
  (fp16 model, relative error ≤ 1.3%); G2 a step along −sign(g_ce) raises p(true) on 97 of 100 images; G3 the four
  parts rebuild Ramen's update (max relative error 2.3e-4); G4 the own-entry part is non-zero exactly for the 100 of
  100 images whose entry was stored, and parallel to their own entropy gradient (cos 1.000000).
- ImageNet-C and EuroSAT: smoke `ga_probe` == Ramen on 3,000 of 3,000 and 200 of 200 images; full streams 75,000 of
  75,000 and 27,000 of 27,000. G1–G4 pass on both (G1 cos ≥ 0.99985; G2 99 and 98 of 100; G3 relative error
  3.5e-7 and 2.6e-4; G4 100 of 100, cos 1.000000). Logs in `logs/`.
- Zero-shot predictions: the prediction Ramen filed each image under differs from the cached NoAdapt run on 1 of
  1,200 smoke images (an fp16 near-tie at confidence 0.05). The analysis uses Ramen's own zero-shot prediction.

## Method

Ramen's notation (`.claude/skills/ramen-experiment/references/ramen.md`): $z_i$ normalised CLIP feature,
$p_i = \operatorname{softmax}(100 z_i^\top W)$, $\hat y_i$ zero-shot prediction, $H_i$ entropy,
$g_i = \nabla_\theta H_i|_{\theta_0}$ over the $D = 39{,}936$ LayerNorm affine parameters. Each image is written to
queue $\mathcal M_{\hat y_i}$ before retrieval;
$\hat g_i = \frac{1}{|C|}\sum_{c \in C}\sum_{j \in \mathcal N_k^c(z_i)} e^{-H_j} e^{-\beta d_{ij}} g_j$;
$\theta_i = \theta_0 - \eta \operatorname{sign}(\hat g_i)$; predict; reset.

Measured (ORACLE, uses $y_i$): $g^{CE}_i = \nabla_\theta[-\log p_{i y_i}]|_{\theta_0}$, and per image
- `cos_update` $=\cos(\hat g_i, g^{CE}_i)$; `cos_step` $=\cos(\operatorname{sign}\hat g_i, g^{CE}_i)$, which sets the
  first-order change of the true-label loss, $-\eta\sqrt D\lVert g^{CE}_i\rVert \cos(\operatorname{sign}\hat g_i, g^{CE}_i)$;
- `agree_step` = $|g^{CE}|$-weighted share of coordinates where the sign step agrees with $g^{CE}$ (0.5 = chance);
- reference `cos_self` $=\cos(g_i, g^{CE}_i)$ (own entropy gradient, no memory);
- control `cos_update_shuffled` $=\cos(\hat g_i, g^{CE}_{i-1})$ (another image of the batch);
- decomposition $\hat g_i = s_i + t_i + u_i + o_i$: own entry, other entries of the true-class queue, other entries
  of the zero-shot-class queue (if $\hat y_i \ne y_i$), all other queues; `part_*` are the additive shares
  $\langle v_i, g^{CE}_i\rangle / (\lVert\hat g_i\rVert\lVert g^{CE}_i\rVert)$, summing to `cos_update`.

## Results (DomainNet, seed 0, 596,006 images; full-stream check: ga_probe == Ramen on 596,006 / 596,006)

Median cosine with the true-label CE gradient $g^{CE}$:

| images | own entropy gradient | Ramen's update | Ramen's sign step | control (other image) | update cos > 0 |
|---|---|---|---|---|---|
| zero-shot right (314,840, 52.8%) | +0.991 | +0.227 | +0.050 | +0.070 | 269,951 (85.7%) |
| zero-shot wrong (281,166, 47.2%) | −0.363 | −0.035 | −0.002 | +0.067 | 122,924 (43.7%) |
| fixed by Ramen (20,428, 3.4%) | +0.147 | +0.123 | +0.040 | +0.067 | 15,354 (75.2%) |
| broken by Ramen (10,302, 1.7%) | +0.712 | +0.013 | +0.003 | +0.065 | 5,524 (53.6%) |
| both wrong (260,738, 43.7%) | −0.390 | −0.048 | −0.005 | +0.067 | 107,570 (41.3%) |

Pre-registered: H1 holds (+0.106 vs control +0.068). H2 holds (−0.035 vs −0.363 on zero-shot-wrong images), but
only in a weak sense: Ramen is still below the control there. H3a holds: the fix rate of zero-shot errors goes from
0.0% (3 of 28,117) in the lowest decile of cos(sign step, g_ce) to 32.6% (9,154 of 28,117) in the highest.
H3b fails: broken images are at ≈ 0 (+0.003), not negative.

Where the alignment of fixed images comes from (mean, total +0.130): other queues +0.116, true-class queue +0.013,
own entry +0.002, zero-shot queue −0.002. Own entry stored for 168,423 of 596,006 images (28.3%; fp16 freeze from
~164k).

**Mechanism.** *Why it should help:* averaging other images' entropy gradients could replace a wrong image's
self-confirming direction with its class's direction. *What happened:* the average cancels the self-confirming
direction (−0.36 → −0.035), but it adds no image-specific pull toward the true class. Sign agreement is 49.6%
(chance), and the update is below the shuffled control. The true-class queue is 1 of ~345 equally weighted queues,
so its part is tiny (+0.013). Broken images are ones whose good own direction (+0.71) the average washes out.
Fixes happen where the average happens to tilt positive, and that tilt predicts fixes almost monotonically.

**Verdict.** Descriptive oracle measurement, one seed. Ramen *neutralises* the wrong direction; it does not
*redirect* toward the true class. Full interpretation: `interpretation.py` / notebook section 7.

## Results across datasets (seed 0; probe == Ramen on every image of every stream)

Median cosine with $g^{CE}$, zero-shot-wrong images (own entropy gradient → Ramen's update; control = another
image's $g^{CE}$), and the mean share of the true-class queue in the alignment of fixed images:

| dataset (classes) | zero-shot wrong | own → update | control | sign agreement | fixed images: true-class queue / total | fix rate, lowest → highest decile |
|---|---|---|---|---|---|---|
| EuroSAT (10) | 14,016 of 27,000 (51.9%) | −0.244 → +0.100 | +0.148 | 55.6% | +0.181 / +0.257 | 3.5% → 76.8% |
| DomainNet (345) | 281,166 of 596,006 (47.2%) | −0.363 → −0.035 | +0.067 | 49.6% | +0.013 / +0.130 | 0.0% → 32.6% |
| ImageNet-C 5K (1,000) | 54,540 of 75,000 (72.7%) | −0.108 → +0.038 | +0.043 | 52.9% | +0.001 / +0.164 | 0.3% → 35.5% |

Hypotheses, all three datasets: H1 holds (EuroSAT +0.310 vs +0.152, DomainNet +0.106 vs +0.068, ImageNet-C +0.082
vs +0.044). H2 holds. H3a holds. H3b fails: broken images have a sign-step cosine of +0.068 / +0.003 / +0.023, near
zero or positive, never negative.

**Mechanism across datasets.** *Why it should help:* filing images under their zero-shot class should make each
queue mostly that class, and retrieval from the true-class queue should then pull a wrong image toward its class.
*What happened:* this works only when the true-class queue carries real weight. Ramen averages queues equally, so
that queue gets 1/|C| of the update: EuroSAT 1/10, DomainNet 1/345, ImageNet-C ~1/974. Its part of the alignment on
fixed images falls accordingly (+0.181 → +0.013 → +0.001). On EuroSAT, fixed images had a wrong own direction
(−0.21) that Ramen reversed (+0.27), which is true redirection. On DomainNet and ImageNet-C the alignment comes from
the bulk of other queues and is no better than the control on zero-shot-wrong images. On every dataset broken
images are ones whose good own direction gets diluted, not reversed. On every dataset the fix rate rises steeply
with alignment, so a direction with higher $\cos(\operatorname{sign}\hat g, g^{CE})$ would turn into fixes.
ImageNet-C's zero-shot-wrong alignment rises over the stream (0.00 → +0.05). A same-photo effect (15 corruptions of
each photo) is a **hypothesis**: this run did not log retrieved photo ids.

**Verdict (all datasets).** Ramen's memory neutralises the self-confirming entropy direction everywhere. It redirects
toward the true class only where there are few classes (EuroSAT). On large label spaces the true-class queue is
drowned out by the equal 1/|C| average. Whether giving it more weight would help there was not tested; finding
that queue without labels is the hard part.

## Follow-up (2026-11-19): images Ramen breaks (zero-shot right → Ramen wrong)

Notebook `04_broken_images.ipynb`; offline analysis of the same seed-0 probe runs (no new GPU run; probe == Ramen on
every image, re-checked in the notebook). Exploratory, not pre-registered: the broken-image medians were already
seen above.

**Math.** Same Ramen equations and probe as above. Groups among zero-shot-right images ($\hat y_i = y_i$):
broken $= \{\hat y^R_i \ne y_i\}$, kept $= \{\hat y^R_i = y_i\}$. *Kept (matched)* reweights kept images by
$w_i = n_{\text{broken}}(b_i)/n_{\text{kept}}(b_i)$, $b_i$ = quantile bin (20) of zero-shot $p_{iy_i}$. For these
images the zero-shot queue is the true-class queue, so $\hat g_i = s_i + t_i + o_i$ (own entry, other entries of
$\mathcal M_{y_i}$, all other queues), and `part_zs` = 0. All cosines except $\cos(\hat g_i, g_i)$ use the
**oracle** $g^{CE}_i$.

| | EuroSAT | ImageNet-C 5K | DomainNet |
|---|---|---|---|
| broken (of zero-shot right) | 1,082 of 12,984 (8.3%) | 2,860 of 20,460 (14.0%) | 10,302 of 314,840 (3.3%) |
| median zero-shot p(true): broken / kept | 0.32 / 0.62 | 0.25 / 0.60 | 0.29 / 0.91 |
| median cos(sign step, g_ce): broken / kept matched / control on broken | +0.068 / +0.154 / +0.159 | +0.023 / +0.052 / +0.036 | +0.003 / +0.036 / +0.065 |
| step points away (cos < 0): broken / kept matched | 196 of 1,082 (18.1%) / 0.8% | 739 of 2,860 (25.8%) / 3.7% | 4,642 of 10,302 (45.1%) / 8.1% |
| median cos(own entropy, g_ce): broken / kept matched | 0.75 / 0.92 | 0.82 / 0.90 | 0.71 / 0.88 |
| AUROC for broken: confidence → + step cosine (5-fold CV) | 0.836 → 0.907 | 0.819 → 0.862 | 0.929 → 0.951 |
| AUROC of step cosine within confidence bins | 0.81 | 0.72 | 0.78 |
| break rate, p ≤ 0.3: least- vs most-aligned fifth | 69% vs 7% | 65% vs 21% | 57% vs 11% |
| mean part from other queues: broken / kept matched | +0.045 / +0.161 | +0.099 / +0.187 | +0.009 / +0.111 |
| mean part from own queue: broken / kept matched | +0.119 / +0.222 | 0.000 / 0.000 | +0.009 / +0.011 |
| label-free cos(update, own entropy) AUROC | 0.79 | 0.54 | 0.61 |

DomainNet over the stream: the break rate rises from 2.1% to 3.7% of zero-shot-right images (first vs last twentieth)
as the share with their own entry stored falls ~100% → ~0% (fp16 priority freeze). Inside the freeze window
(positions 140k–250k), matched on position and confidence: 2.39% broken with the own entry stored vs 3.72% without
(220 cells). Destinations: DomainNet spreads over 345 classes (largest teapot 261, 2.5%, mostly from coffee cup);
ImageNet-C over 815 (≤ 19 each); EuroSAT concentrates on Highway (313, 28.9%, mostly from River) and AnnualCrop (299).

**Why it should / what happened.** *Expectation from notebooks 01–03:* broken images keep a good own direction that
the 1/|C| average washes out, so they might just be low-margin images flipped by a direction-free step.
*What happened:* confidence is not the whole story. At the same confidence, broken images get a step about half
as aligned (EuroSAT, ImageNet-C) or roughly a tenth as aligned (DomainNet) as kept images, below the shared-direction control. Up to
45% of them get a step pointing away from the true class. What is missing is mostly the generic pull that the
bulk of other queues gives kept images, plus the own queue on EuroSAT. Their own entropy direction is also less
CE-like (0.71–0.82 vs 0.88–0.92). That at low confidence entropy partly sharpens the runner-up is a **hypothesis**,
not measured. Part of the alignment–break link is first-order by construction (a negative cosine lowers p(true)).
The stored/not-stored gap on DomainNet is an association (storage also depends on whether a class's queue is full).
The causal test, `priority_fp32: true` on the mixed DomainNet stream, has not been run. Prior against a big effect: on mixed CIFAR-10-C / CIFAR-100-C seed 0, fp32 priorities change accuracy by +0.05 / −0.00 (cached `fp32prio` runs).

**Verdict.** Descriptive, one seed. Breaks = low-confidence images whose step has lost the generic alignment the
other queues give everyone else; on DomainNet the fp16 freeze is a candidate cause (association only). A label-free
fix would need a proxy for the step's alignment; the one logged here, cos(update, own entropy), is weak on large
label spaces (AUROC 0.54 / 0.61).

## Follow-up (2026-11-19): loss landscape around θ₀ and Ramen's θᵢ (notebook 04, section 6)

**What was run.** `run_landscape.py <dataset> 0` replays Ramen on the same stream with `RamenLandscapeProbe`
(run `ga_landscape`, ORACLE measurement, predictions unchanged). For 50 images per outcome group, drawn at random
(numpy seed 0) from the `ga_probe` frame before the run, it evaluates the true-label loss
$L_i(\theta) = -\log p_{y_i}(x_i;\theta)$, the entropy and the prediction:
- on the plane $\theta_0 + a\,u_i + b\,e_2$, where $u_i = \theta_i - \theta_0 = -\eta\operatorname{sign}\hat g_i$ is
  Ramen's actual step and $e_2$ is the oracle step $o_i = -\eta\operatorname{sign} g^{CE}_i$ made orthogonal to
  $u_i$ and rescaled to $\lVert u_i\rVert$; $a \in [-1, 2.5]$, $b \in [-1, 1.75]$, step 0.125;
- on the lines $\theta_0 + t v$, $v \in \{u_i, o_i, r_i\}$, $r_i = -\eta\operatorname{sign}\xi$ a random sign step
  (control), $t \in [-1, 3]$, step 0.1. All three steps have length $\eta\sqrt D = 2.0$.

It also saves $z_0$ and $z_1$ (features before and after the step). The embedding-space slice
$z(a,b) \propto z_0 + a(z_1 - z_0) + b\,t_2$ ($t_2$ = the head's true-class ascent at $z_0$, orthogonalised,
same length) is computed offline with the frozen text head.

**Checks.** Smoke (200 images per domain) and full runs: `ga_landscape` == Ramen on 27,000 / 75,000 / 596,006
images. On 200 of 200 chosen images per dataset, label, zero-shot and Ramen predictions match the probe frame, and
cos(sign step, g_ce) matches the probe (|diff| < 1e-3). Slice points (0, 0) and (1, 0) reproduce the zero-shot and
Ramen logits exactly (max difference 0.0000). Run times: EuroSAT 4.8 min, ImageNet-C 11.8 min, DomainNet 37.8 min.

**Results (median true-label loss change after one step; 50 images per group; EuroSAT / ImageNet-C / DomainNet).**

| group | Ramen's step | oracle step | random step | Ramen lowers the loss | flip along Ramen's step (median t) |
|---|---|---|---|---|---|
| both right | −0.39 / −0.12 / −0.04 | −0.47 / −0.55 / −0.08 | −0.005 / −0.000 / +0.000 | 50 / 38 / 46 of 50 | never |
| fixed | −1.54 / −1.57 / −1.23 | −1.77 / −2.24 / −2.17 | −0.005 / +0.029 / −0.053 | 50 / 50 / 50 of 50 | 0.20 / 0.30 / 0.30 |
| broken | +0.57 / +0.66 / +0.55 | −0.91 / −1.13 / −0.94 | +0.017 / +0.040 / +0.009 | 11 / 7 / 6 of 50 | 0.50 / 0.55 / 0.45 |
| both wrong | +1.05 / +0.002 / +0.06 | −1.84 / −3.71 / −3.33 | −0.019 / +0.002 / −0.021 | 15 / 25 / 23 of 50 | 2.2 / 0.4 / 1.6 (to another wrong class) |

The oracle step lowers the loss on 48–50 of 50 images in every group. On zero-shot-wrong images it reaches the true
class at t = 0.10–0.20. Median cos(Ramen's step, oracle step): 0.00–0.34. Embedding move $\lVert z_1 - z_0\rVert$
(unit-norm features): 0.47–0.68 / 0.42–0.52 / 0.25–0.32. Its cosine with the head's true-class direction:
broken −0.001 / +0.000 / −0.028 vs both-right and fixed +0.06 to +0.18.

**Why it makes sense.** *Expectation:* if Ramen's step were "right but too small" or "too big", the oracle step of
the same length would behave like it, and random steps would show how rough the surface is. *What happened:* the
same-length oracle step fixes almost everything, so the step length is not what limits Ramen. Random steps do
nothing, so the surface is flat in generic directions. Ramen's step does move the loss by O(1), in the direction
the zero-shot prediction already points: down when zero-shot is right, flat or up when it is wrong. It is almost
orthogonal to the oracle step and runs along the true-class boundary rather than across it. Broken images sit
close to that boundary (they flip at half a step), and a sideways move of that size takes them over.
In embedding space the same picture holds: large moves, orthogonal to the true-class direction on broken images.
This is the landscape view of section 4's cosines.

**Verdict.** Descriptive, one seed, 50 images per group. It confirms section 4 with a different measurement (loss
values rather than cosines). Ramen's update has enough magnitude but the wrong direction for images it gets wrong.
A same-length step in the right direction would fix most errors, so any gain must come from information about the
direction (which needs labels or a label proxy), not from step size.
