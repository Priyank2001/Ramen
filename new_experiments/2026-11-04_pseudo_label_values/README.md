# 2026-11-04 Pseudo-label values

Question: can a label-free target for the CE-gradient memory value recover part of the label-value oracle's gain (+1.70 DomainNet, +1.51 ImageNet-C, +26.70 EuroSAT)?

Seed 0, change vs Ramen in points (floors 0.11 / 0.53 / 1.51):

| run | DomainNet | ImageNet-C | EuroSAT |
|---|---|---|---|
| pv_zs (control) | +0.07 | +0.19 | −1.13 |
| pv_adapted | −0.65 | −5.28 | −31.17 |
| pv_knn | −1.50 | −15.84 | −10.05 |
| pv_mix_p25 (oracle) | +0.75 | – | +17.55 |
| pv_mix_p50 (oracle) | +1.29 | – | +24.75 |

Result: hypothesis not supported. Targets need to be far more accurate than zero-shot without overturning correct images.

Files: `pseudo_methods.py` (RamenPseudoValues, target rules), `pv_registry.py`, `pseudo_values.py` (variants, experiment), `run_pseudo_values.py` (CLI; `--smoke` or `--runs`), `pv_plots.py`, `make_notebook.py` -> `01_pseudo_label_values.ipynb`. Smoke checks: pv_off == Ramen, pv_true == labeloracle_zs_ce. Shared code is imported read-only.

## Math of each method

Notation as in the repo's Ramen formulation. Batch $x_1..x_B$ ($B=100$); $\theta_0$ = LayerNorm affine params; $z_i=f_{\theta_0}(x_i)/\lVert f_{\theta_0}(x_i)\rVert$, $p_i=\mathrm{softmax}(100\,z_i^\top W)$, $\hat y_i=\arg\max_c p_{ic}$ (zero-shot prediction), $H_i=-\sum_c p_{ic}\log p_{ic}$.

**Ramen (baseline).** Entry $j$ is filed in queue $\mathcal M_{\hat y_j}$ as $(z_j, g_j, H_j)$ with value $g_j=\nabla_\theta H_j|_{\theta_0}$. Retrieval: $\hat g_i=\frac1{|C_i|}\sum_{c\in C_i}\sum_{j\in\mathcal N_k^c(z_i)}e^{-H_j}e^{-\beta d_{ij}}g_j$, update $\theta_i=\theta_0-\eta\,\mathrm{sign}(\hat g_i)$, predict, reset.

**All runs here** keep Ramen's order (write, retrieve, step, predict, reset) and change only the value stored *after* the image has been predicted:
$$g_j \leftarrow \nabla_\theta\, \mathrm{CE}\big(100\,z_j^\top W,\; t_j\big)\Big|_{\theta_0}=\nabla_\theta\big(-\log p_{j t_j}\big),$$
with a target class $t_j$ chosen by a rule. Filing, priorities, retrieval weights ($e^{-H_j}$ keeps the entropy) and the step are unchanged, so the image never uses its own target. Entropy gradient pushes towards the *currently predicted* class sharpened; CE pushes towards $t_j$.

| run (config `target`) | target $t_j$ | oracle? |
|---|---|---|
| `pv_zs` (`zero_shot`) | $\hat y_j$ — so $\nabla\mathrm{CE}$ with the zero-shot class | no |
| `pv_adapted` (`adapted`) | $\arg\max_c\,\mathrm{Ramen}(x_j)_c$, the prediction after the one step | no |
| `pv_knn` (`knn`, $k=10$) | $\arg\max_c\sum_{m\in\mathcal N_{10}(z_j)} \mathbb 1[\mathrm{queue}(m)=c]\,e^{-H_m}$ over the whole cache, own entry excluded | no |
| `pv_mix_p` (`mix`, $p\in\{.25,.5\}$) | $y_j$ if $\hat y_j\ne y_j$ and $u_j<p$ ($u_j\sim U(0,1)$, fixed seed), else $\hat y_j$ | **yes** (true label) |
| `pv_true` / `pv_off` | $y_j$ / none (= Ramen exactly); smoke checks only | yes / no |

Labels are passed to the harness but used only for logging in label-free runs.

## Why each method does or does not make sense

Reference numbers (seed 0, change vs Ramen in points; floors 0.11 DN, 0.53 INC, 1.51 EuroSAT) are in the table above.

**Key mechanism.** Writing $g_j=\nabla_\theta\mathrm{CE}(\cdot,t_j)$ makes a retrieving image $i$ move $\theta$ so that *class $t_j$ becomes more likely for images like $z_j$*. If $t_j$ is the true class, a cache neighbour teaches the right class even when its zero-shot filing was wrong; if $t_j$ is wrong, it actively teaches the wrong class. The expected benefit is therefore roughly
$$\Delta\approx(\#\text{errors fixed by }t)\cdot a-(\#\text{correct predictions broken by }t)\cdot b,$$
and it is positive only when the target is both more accurate than $\hat y$ and changes few correct images. That is what the numbers show.

- **`pv_zs` (CE toward zero-shot).** *Should:* nothing — it is the control that isolates the CE form from the target. *Happened:* +0.07 / +0.19 / −1.13, all within floors, with target = zero-shot on 596,006 / 75,000 / 27,000 images (0 changed). So the oracle's gain is not caused by swapping entropy for CE. **Null, as designed.**
- **`pv_adapted` (CE toward the adapted prediction).** *Should help:* Ramen's adapted prediction is on average better than zero-shot (DomainNet 53.75% vs 52.82%). *Happened:* on DomainNet it fixes 21,649 errors but breaks 16,118 correct ones and ends −0.65 (−4,595 images). On ImageNet-C it differs from zero-shot on 44,416 of 75,000 (59%), fixes 5,358 / breaks 4,966, final −5.28. On EuroSAT the target is *worse* than zero-shot (31.46% vs 48.09%; fixes 3,598 / breaks 8,088) and the run falls 31 points. The target comes from the cache the value is written into, so mistakes are stored and retrieved again: a feedback loop (hypothesis supported by the EuroSAT collapse; not separately isolated). **Hurts.**
- **`pv_knn` (CE toward the neighbour vote).** *Should help:* neighbours in embedding space often share the true class, so a vote should denoise. *Happened:* the vote is only as good as neighbour class purity, and neighbours are filed by zero-shot class. Offline on DomainNet, 53% of neighbours share the class and the vote is 51.8% vs 53.5% zero-shot. Full runs: DomainNet target 48.06% right (fixes 24,363 / breaks 52,764), ImageNet-C 15.72% vs 27.28% (breaks 11,303, fixes 2,631), EuroSAT 49.91% vs 48.09% (fixes 1,873 / breaks 1,381). Even where the vote is slightly better (EuroSAT), the run loses 10.05 points: small net accuracy gain in the target, but its changes (6,055 images) are concentrated on the confusing classes, where wrong CE gradients do more harm than right ones help (hypothesis; consistent with the oracle below, which only changes errors). With 1,000 / 345 classes the neighbour vote is spread thin and is worse than zero-shot. **Hurts; worse the larger the label space.**
- **`pv_mix_p25/p50` (oracle dose-response).** *Should:* if the gain is label-driven, gain should grow with the share of corrected errors, and since only errors are changed nothing correct is broken (0 breaks). *Happened:* DomainNet +0.75 → +1.29 (target accuracy 64.62% → 76.38%); EuroSAT +17.55 → +24.75 (61.05% → 74.06%). Monotone: on DomainNet, +11.8 and +23.6 points of target accuracy over zero-shot buy only +0.75 and +1.29. This is an upper bound, not a method. **Confirms the mechanism and sets the bar** for any label-free target: improve accuracy by tens of points without breaking correct images.

**Overall.** The cache-and-retrieve machinery is able to use better labels (oracle dose-response), but no label-free source tried here is better than the zero-shot filing it already has: the adapted prediction is derived from the same cache, and the kNN vote is derived from zero-shot-filed neighbours. The next label-free candidate has to bring *independent* information (e.g. a different model or text prompt ensemble) rather than re-reading the cache. Caveats: seed 0 only; ImageNet-C has no dose-response (same-photo copy leak); DomainNet keeps the fp16 priority freeze for a like-for-like comparison.
