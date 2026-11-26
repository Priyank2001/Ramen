# Plan: oracle_true_filing
*2026-11-09. Status: done (seed 0): true-label filing is below the Ramen copy beyond the floor on CIFAR-10-C, CIFAR-100-C, ImageNet-C and DomainNet, and below random filing on both CIFAR sets. DomainNet (Ramen + ORACLE copies only; random control does not fit in GPU memory; run with CUDA_MODULE_LOADING=LAZY, Ramen copy identical to the baseline on all 586,575 predictions): ORACLE also below Ramen beyond the floor, slightly below zero-shot. Hypothesis supported; direction stays closed. Notebook: `experiments/week_01_2026-11-02/2026-11-09_oracle_true_filing/oracle_true_filing.ipynb`. ORACLE diagnostic (reads labels to file memory; the Ramen copy never sees them).*

## Question and hypothesis
- **Question.** If every test sample is filed into Ramen's class-split memory under its TRUE class instead of its zero-shot prediction, with the same write time, the same stored gradient and the same retrieval, does accuracy go up?
- **Hypothesis.** On CIFAR-10-C seed 0 (batch 100), true-label filing is BELOW the Ramen copy by more than the CIFAR-10-C noise floor, and also below random filing. Reason (paper §3.2 "prediction balance", Remark 4.3, App. B): the support set is balanced over pseudo-labels only under zero-shot filing; true-label filing makes it balanced over true labels, so the entropy gradients (each pushing toward its own zero-shot argmax) no longer cancel in the bias term and the step drifts toward attractor classes. Prior: the legacy DomainNet result (write-after-prediction, batch 100, `sec_server_dual_agent`) found true filing below its control.

## Derivation
Ramen (`src/methods/Ramen.py:111-170`), per batch:
1. z_i = f(x_i), p_i = softmax(W^T z_i), yhat_i = argmax p_i, H_i = entropy(p_i)
2. g_i = grad_theta H_i (per-sample LayerNorm affine copy)
3. write (z_i, g_i, H_i, t_i) into queue M_{yhat_i}  (`Ramen.py:128-133`), FIFO by fp16 priority t
4. S_i = union_c topk_c(z_i) (Eq. 2); w_ij = exp(-H_j) exp(-beta ||z_i - z_j||); G_i = (1/|C+|) sum_c sum_{j in topk_c} w_ij g_j
5. theta_i = theta_0 - lr sign(G_i); predict; reset

The ONLY change, per copy: step 3 writes into queue q_i with
- `zero_shot` (Ramen): q_i = yhat_i
- `true_label` (ORACLE): q_i = y_i
- `random` (control): q_i ~ Uniform{1..C}, seeded generator
Stored value g_i, key z_i, weight terms H_j, t_i, top-k, beta, lr: unchanged. y_i never enters any copy's G_i except through WHICH queue an entry sits in.

Predicted effect: with true filing the per-queue top-k still gives k entries per true class, but their gradients push toward their own zero-shot argmax yhat_j. On domains where zero-shot collapses onto a few attractor classes, most of the k entries of every queue push toward the attractor, so G_i points toward the attractor for every query. Under zero-shot filing each queue's entries all push toward that queue's class, which balances the push (App. B: bias gradient ~ 0 when class-balanced).
Can fail / be misread because: (1) the effect could be dominated by "spreading" (any filing that is not by yhat), which the random control isolates; (2) cold memory early in the stream (report accuracy along the stream); (3) fp16 priorities freeze queues on long streams (`lab.md` §7), identically in all copies.

## Variants (one run, three caches over the same stream; ORACLE run)
| name | filing | what it changes | labels |
|---|---|---|---|
| Ramen copy | `zero_shot` | nothing; must equal `pc2_<DS>_Ramen_b<B>_s0` bit for bit | no |
| Oracle copy | `true_label` | queue = true class | ORACLE |
| Random control | `random` | queue = random class (same spreading, no label information) | no |
All three share z_i, g_i, H_i, t_i, the stream, batch, seed and GPU. Every sample gets one prediction per copy, so comparisons are paired per image.

## Files
- `ramen_lab/upstream.py`: expose the by-sample CLIP model, `softmax_entropy`, and a helper that re-arms the per-sample grad buffers for a second step (no upstream edit).
- `ramen_lab/methods/cache_pool.py`: `GradientPool` (shared, ref-counted fp16 gradient rows), `PooledPriorityCache` (slot-for-slot mirror of upstream `PriorityCache`), filing policies (registry).
- `ramen_lab/methods/multi_cache.py`: `RamenMultiCache` (ORACLE method; returns the Ramen copy's logits).
- `ramen_lab/methods/multi_cache_log.py`: per-sample, per-copy predictions and active-set statistics; reservoir of example active sets; cache snapshots -> `multicache.npz`.
- `tests/test_lab_multi_cache.py`; `tests/test_lab_invariants.py` gets the test config.
- `experiments/week_01_2026-11-02/2026-11-09_oracle_true_filing/{oracle_true_filing.py,.ipynb}`.

## Tests
Ramen copy bit-identical to upstream `Ramen` over a stream with a full, evicting cache (torch.equal on logits); pooled cache slot-for-slot equal to `PriorityCache` incl. fp16 saturation; pool never loses a referenced row and frees unreferenced ones; oracle queues contain only their true class; Ramen copy independent of labels while the oracle copy changes with them (non-vacuous); random filing deterministic and reset by reset(); shared invariants (determinism, reset, frozen weights).

## Runs and resources
check (200 per domain, CIFAR-10-C): Ramen copy == plain Ramen on the subset, 100.00%. Then seed 0, frozen batch, GPU 0 (3080 Ti, same model as the baselines), chained: CIFAR-10-C (b100), CIFAR-100-C (b32), ImageNet-C (b24), DomainNet (b32). Pool capacity per dataset from an offline simulation with the NoAdapt seed-0 zero-shot predictions + margin; overflow raises. RAM budget: the dataset's frozen budget.

## Pre-registered reading
- Primary (CIFAR-10-C seed 0): Oracle copy minus Ramen copy, against the CIFAR-10-C noise floor (2 x range of Ramen seeds 0-4, `2026-11-01_baseline`). Gain > +floor: filing purity matters -> reopen "cleaner filing" with a label-free proxy. Gain < -floor: hypothesis supported; the direction stays closed on pc2_lab.
- Oracle vs random: if oracle <= random + floor, the label information in the filing is not used by the update.
- Other datasets: same reading, seed 0 only, each against its own floor; report, do not tune.
- Diagnostic only: active-set statistics (push share toward the true class, overlap between copies) explain, they do not decide.

## Risks
Single seed. GPU memory (shared pool sized by simulation; fails loudly, no host risk). Extra retrieval makes runs about 2-3x slower than Ramen.
