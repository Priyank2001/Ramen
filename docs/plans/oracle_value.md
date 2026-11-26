# Plan: oracle_value (store the true-label CE gradient instead of the entropy gradient)
*2026-11-11. Status: draft. ORACLE ceiling (reads true labels to compute the stored value and to file it). Direction 1 of the 2026-11-11 direction review.*

## Question and hypothesis
- **Question.** `oracle_true_filing` showed that cleaner *labels* in Ramen's memory have a negative ceiling, because each entry's stored value still pushes toward its own zero-shot class (notebook: "What the numbers do not say", 3rd point). That leaves the **value** open. If every entry stored the cross-entropy gradient toward its TRUE class, filed under that class (so the push balance over queues is kept), does Ramen predict better?
- **Hypothesis.** On CIFAR-10-C seed 0 (batch 100, RTX 3080 Ti), the ORACLE value is above the matched control (Ramen without self-retrieval) by more than the CIFAR-10-C noise floor (0.18, `2026-11-01_baseline/baseline.ipynb`).
- **Prior evidence it could work.** `grad_alignment.ipynb` Finding 2: on zero-shot-wrong samples, Ramen's aggregate is at the shuffled null, so it does not steer them toward the true class. Correct values in the memory are the most direct way to change that.

## Derivation
Ramen, per batch (`src/methods/Ramen.py:111-170`; `ramen_lab/methods/stored_value.py` re-implements it as named steps):
1. z_i = f(x_i), p_i = softmax(W^T z_i), yhat_i = argmax p_i, H_i = entropy(p_i)
2. **v_i = grad_theta L_i**, the value written to memory. Ramen: **L_i = H(p_i)** (entropy, pushes toward yhat_i)
3. write (z_i, v_i, H_i, t_i) into queue M_{q_i}, with Ramen's q_i = yhat_i
4. S_i = union_c topk_c(z_i); w_ij = exp(-H_j) exp(-beta ||z_i - z_j||); G_i = (1/|C+|) sum_c sum_{j in topk_c(i)} w_ij v_j
5. theta_i = theta_0 - lr sign(G_i); predict; reset

What changes, one term at a time:
- **Stored value (step 2).** `entropy`: L_i = H(p_i) (Ramen). `ce_true` (ORACLE): L_i = CE(p_i, y_i). `ce_pseudo` (control): L_i = CE(p_i, yhat_i), the same loss form with the zero-shot label, so it carries no label information.
- **Filing (step 3).** q_i = y_i together with `ce_true`, so every queue's entries push toward that queue's class. That is the prediction balance of the paper (§3.2, App. B), the property `oracle_true_filing` found to matter. A value-only variant (`ce_true`, filed by yhat) is also run on CIFAR, to separate the two.
- **Self-exclusion (step 4).** Ramen writes before it retrieves, so query i retrieves its own entry at distance 0 (`lab.md` §7). With `ce_true`, that entry is grad CE(p_i, y_i): the query's own label would feed its own step, which is leakage, not a memory effect. So every compared variant drops the query's own entry from each queue's top-k and takes the next one instead. **"Ramen without self" (entropy, zero-shot filing, self excluded) is the control that the ORACLE is compared with.** Batch-mates stay retrievable: they are other images, which Ramen's memory also holds.
- w_ij keeps the zero-shot entropy H_j in every variant. The 1/|C+| factor and any positive scale of the values do not change sign(G_i).

Predicted effect [hypothesis]: with `ce_true`, entries of queue c push toward c, with magnitude ~(1 - p_j(c)). Entries that zero-shot gets wrong carry large values toward their true class. On zero-shot-wrong queries G_i should therefore move toward the true class instead of the null.

How it can fail or mislead:
1. **The stored gradient is computed at theta_0 for a different image.** A correct direction for j need not be one for i, because neighbours share the domain, not the class (`knn_heldout`). The ceiling could then be small.
2. **Near-duplicates on CIFAR-C and ImageNet-C.** The same base image appears under 15 corruptions, so the memory can hold the query's own content, labelled, under another corruption. That inflates the ceiling there. DomainNet has no such duplicates, so it is the clean check.
3. **fp16 values.** Stored values are fp16 and unscaled (Ramen's regime). The grad-alignment probe with a ×1024 loss scale overflowed on 1-7 samples per dataset, so there is no scaling here. Non-finite values raise an error. The fraction of zero coordinates (underflow) is logged per sample.

## Variants
| name (run id part) | class | config | what it changes | labels |
|---|---|---|---|---|
| identity (check only) | `RamenStoredValue` | defaults | nothing; must equal upstream `Ramen` bit for bit | no |
| `RamenNoSelf` | `RamenStoredValue` | `exclude_self: true` | own entry not retrieved (**control**) | no |
| `CEpseudoNoSelf` | `RamenStoredValue` | `stored_value: ce_pseudo`, `exclude_self: true` | value = CE toward yhat (loss-form control) | no |
| `ORACLE_CEtrueNoSelf` | `RamenStoredValueOracle` | `stored_value: ce_true`, `filing: true_label`, `exclude_self: true` | value = CE toward y, filed under y | **ORACLE** |
| `ORACLE_CEtrueZsFileNoSelf` (CIFAR only) | `RamenStoredValueOracle` | `stored_value: ce_true`, `exclude_self: true` | value = CE toward y, filed under yhat | **ORACLE** |
Plain Ramen is the week-01 baseline `pc2_<DS>_Ramen_b<B>_s0`, on the same GPU model. All runs share the stream, batch, seed and GPU 0. A seed changes only the order, and runs are deterministic, so comparisons are paired per image across runs.

## Files
- `ramen_lab/methods/stored_value.py` (new): `StoredValue` strategies (registry `VALUES`: `entropy`, `ce_pseudo`, `ce_true`), `drop_self` (pure), `RamenStoredValue`, `RamenStoredValueOracle`. Reuses `GradientPool`, `PooledPriorityCache` and `FILINGS` from `cache_pool.py`.
- `ramen_lab/methods/__init__.py`: one import line. `tests/test_lab_invariants.py`: a test config for the ORACLE class.
- `tests/test_lab_stored_value.py` (new).
- `experiments/week_02_2026-11-09/2026-11-11_oracle_value/{oracle_value.py, oracle_value.ipynb}`.

## Tests
- Default config bit-identical to upstream `Ramen` (`torch.equal` on logits, a stream that fills and evicts).
- `drop_self`: removes the own entry when present, otherwise the (k+1)-th; leaves queues with ≤ k entries whole.
- With `exclude_self`, no query's active set contains its own entry (checked through the logged self share = 0).
- `ce_true` value equals a single-sample CE backward; `ce_pseudo` equals CE toward the argmax.
- The label-free class rejects label-using settings. The ORACLE class needs labels for them, and its predictions change with the labels (non-vacuous).
- Label-free variants do not depend on labels; the shared invariants (determinism, reset, frozen weights) run on both classes.

## Runs and resources
1. `check CIFAR10C --n 200`: the identity config equals plain Ramen 100%. The same command smoke-runs every variant on the subset and reports peak RAM and GPU.
2. Chain on GPU 0 (3080 Ti, the baselines' GPU model): CIFAR-10-C (b100, 4 variants), then CIFAR-100-C (b32, 4 variants). Ramen alone takes about 4 and 12 minutes. RAM budget: each dataset's frozen budget.
3. After reading CIFAR-10-C: ImageNet-C (b24) and DomainNet (b32), with the 3 main variants and the setting unchanged (DomainNet with `CUDA_MODULE_LOADING=LAZY`, as before).

## Pre-registered reading
- **Primary** (CIFAR-10-C seed 0): Δ = acc(ORACLE_CEtrueNoSelf) − acc(RamenNoSelf), against floor 0.18.
  - **Δ ≤ +0.18:** better stored values have no ceiling. With filing already closed, the memory's contents are not the lever. Close "memory contents" on pc2_lab and move to directions 3-5 of the review.
  - **Δ > +0.18:** the value is a lever. Then the label-free part, acc(CEpseudoNoSelf) − acc(RamenNoSelf), against the floor, says how much the loss form alone gives. The gap between ORACLE and CEpseudo is the target for a label-free value (separate plan).
- Other datasets: same reading, seed 0, against each dataset's floor (2 × range of Ramen seeds 0-4, computed in the notebook from `2026-11-01_baseline`). Report them; don't tune.
- **Diagnostic only:**
  - RamenNoSelf − Ramen (what self-retrieval is worth);
  - the value-only variant (filing vs. value);
  - fixed/broken counts split by zero-shot right/wrong;
  - self share; value underflow.
- Single seed. Seeds 1-4 only if Δ is within 2 × floor on CIFAR-10-C.

## Risks
- Single seed.
- Near-duplicate inflation on the corruption sets (see the derivation, failure mode 2).
- GPU memory on DomainNet: one memory plus k+1 retrieval, about Ramen's own peak. Fails loudly on OOM; no host risk.

## Amendment (2026-11-11, after the CIFAR results; recorded before any held-out result was read)
- **Seed-0 results.**
  - Primary ORACLE: below RamenNoSelf on CIFAR-10-C and CIFAR-100-C, far beyond the floor. The pre-registered gate fails.
  - Value-only ORACLE (`ORACLE_CEtrueZsFileNoSelf`, diagnostic): below on CIFAR-10-C, above on CIFAR-100-C beyond the floor.
- **Concern.** On the corruption sets, that gain may come from near-duplicates (derivation, failure mode 2).
- **Change.** The value-only ORACLE is added to ImageNet-C and DomainNet: same setting, run after the main held-out chain on GPU 0. DomainNet has no near-duplicates, so it is the check.
- **Status of the run.** This is still a diagnostic, not a gate. Any method built on it needs its own plan.
