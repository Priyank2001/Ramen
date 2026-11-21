# Write after prediction (`RamenWriteAfterPrediction`)

Subclass of `RamenLabelValues`. Notation and the Ramen baseline are in [README.md](README.md).
**Uses true labels only** with `filing: true_label` or `value: true_label_ce`. With `filing: adapted` and
`value: entropy` it is a real method (no labels).

## Why

The question was whether filing by the **adapted** prediction beats filing by the zero-shot one. (The paper,
Sec. 3.2, and its official code, `init_preds`, file by the zero-shot prediction.) The adapted prediction
$\hat y^{\,\mathrm{ad}}_i$ exists only after retrieval and the step, so filing by it means moving the write to after
prediction. Moving the write changes two other things too: an image no longer retrieves its own entry, and it no
longer sees its batch-mates. The zero-shot filing runs are therefore the **timing controls**.

## What changes

The order of the steps. Everything else (keys, the stored entropy $H_i$, retrieval, weights, the SignSGD step, the
reset) is Ramen's.

1. **Retrieve** from the cache as it stood after batch $t-1$. There is no self entry and no batch-mate entries.
2. **Update and predict** as in Ramen. If every queue is empty (the first batch), there is no step and the batch
   is predicted zero-shot: $\theta_i = \theta_0$.
3. **Write**: add $(z_i,\ v_i,\ H_i,\ \pi_i)$ to $\mathcal{M}_{q_i}$, with

$$q_i = \begin{cases} \hat y_i & \texttt{zero\_shot} \\ y_i & \texttt{true\_label}\ \text{(oracle)} \\ \hat y^{\,\mathrm{ad}}_i & \texttt{adapted} \end{cases}
\qquad
v_i = \begin{cases} g_i & \texttt{entropy} \\ u_i & \texttt{true\_label\_ce}\ \text{(oracle)} \end{cases}$$

where $u_i = \nabla_\theta[-\log p_{i,y_i}]\,|_{\theta_0}$ as in [label_values.md](label_values.md). Since the write
comes after prediction, $u_i$ is stored directly, with no reveal step, and still never reaches image $i$'s own update.

$H_i$ and $g_i$ are always the **zero-shot** entropy and gradient, also under adapted filing. Only the queue changes.

## Variants run

| run | filing $q_i$ | value | labels? |
|---|---|---|---|
| `late_zs_entropy` | $\hat y_i$ | $g_i$ | no (timing control for Ramen) |
| `late_zs_ce` | $\hat y_i$ | $u_i$ | yes (timing control for `labeloracle_zs_ce`) |
| `late_adapted_entropy` | $\hat y^{\,\mathrm{ad}}_i$ | $g_i$ | **no: a method** |
| `late_adapted_ce` | $\hat y^{\,\mathrm{ad}}_i$ | $u_i$ | yes |

## Code map

| symbol / step | code |
|---|---|
| $q_i$ | `RamenWriteAfterPrediction.queue_after(zs_pred, labels, pred)` |
| $\hat y^{\,\mathrm{ad}}_i$ | `out.argmax(-1)` after `step_and_zero_grad` |
| no step on an empty cache | `adapted = any(cache.size > 0 ...)`; logged in `stats['adapted']` |
| write | `RamenLabelValues.write` with `values = grads` or `label_values` |

Checks W0–W5 in `check_write_after.py`: the first batch is zero-shot, retrieval sees earlier batches only, each image
is filed under its returned class, and adapted and zero-shot filing differ.

## Results (seed 0, change vs Ramen in domain-averaged points)

| run | DomainNet (floor 0.11) | ImageNet-C (floor 0.53) | EuroSAT (floor 1.51) |
|---|---|---|---|
| `late_zs_entropy` (timing control) | −0.05 | −0.17 | +0.53 |
| `late_zs_ce` | +1.86 | +1.64 | +26.49 |
| `late_adapted_entropy` (no labels) | −0.27 | +0.12 | −6.04 |
| `late_adapted_ce` | +1.26 | +1.37 | +25.90 |

Adapted filing compared with its own timing control (`late_adapted_entropy` − `late_zs_entropy`): −0.21 / +0.29 /
−6.57. Moving the write is within noise on all three datasets. Adapted filing is a dead end.
