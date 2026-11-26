# Plan: oracle_fill (file by true label instead of prediction)
*2026-11-09. Status: draft, check and full runs not started. ORACLE experiment (reads true labels).*

## Question and hypothesis
Ramen files each sample under its zero-shot prediction. If filing by the TRUE label (oracle) gives the memory the right
neighbours, CIFAR-10-C accuracy (seed 0, batch 100, RTX 3080 Ti) rises above Ramen's by more than the noise floor.
The experiment is not a gate for a method: it asks whether the filing choice is where Ramen loses accuracy at all.
Prior: `docs/research_state.md` records that oracle true-label filing made Ramen worse on DomainNet. This tests whether
that holds on CIFAR-10-C. [unverified until the run]

## Derivation
Ramen's update for sample i (src/methods/Ramen.py, `forward`):

    write:     queue(j) = argmax f(x_j)            for every stream sample j (the changed term)
    retrieve:  g_R(i) = (1 / |C_ne|) * sum_{c in C_ne} sum_{j in topk_c(i)} w_ij * g_j
               w_ij = exp(-H_j) * exp(-beta * d_ij),  g_j = grad of H_j wrt LayerNorm affine params of sample j
    step:      theta_i <- theta_i - lr * sign(g_R(i));  prediction of i uses theta_i

The changed term is queue(j). Filing by label sets queue(j) = y_j. Two things change, and they are kept apart:
1. **Membership**: which entries are retrieved for query i (queue c holds the samples of class c, not the ones predicted as c).
2. **Gradient target**: g_j still points toward the zero-shot argmax of j, not toward y_j (lab.md, Gotchas). Filing never
   changes the direction of an entry's value, only where it sits.

Predicted effect [hypothesis]: for queries whose true class is c, label filing averages entries from queue c, which
hold the class-c samples. If zero-shot CLIP is mostly right on class c, the gradients agree with y_i and accuracy rises.
If zero-shot is wrong on many class-c samples, their gradients point away from c and accuracy drops. How it could fail:
the retrieved mass is dominated by the query's own write (distance 0, see lab.md) in both settings, so the membership
change may move little.

## Variants
| name | class | config | what it changes | uses labels? |
|---|---|---|---|---|
| Ramen (baseline) | upstream `Ramen` | yaml | none (`pc2_C10_Ramen_b100_s0`, existing, 3080 Ti) | no |
| FileByPred (identity) | `RamenFileByLabel` | `filing: prediction` | nothing; must equal Ramen to the bit | no (never read) |
| FileByLabel (ORACLE) | `RamenFileByLabel` | `filing: label` | queue = true label | **yes** |

Both new variants log `cache_snapshot.npz` (end-of-stream queues) and `retrieval.npz` (per-query mass per queue and per
gradient target, top-5 retrieved entries) into their run folder. Labels are read in the label variant only, for filing.

## Files
- `ramen_lab/methods/oracle_fill.py` (new): `TrackedPriorityCache` (upstream cache + slot positions and targets, same
  eviction), `RamenFileByLabel` (upstream Ramen's forward as named steps, the queue choice is the only difference).
- `ramen_lab/methods/__init__.py`: one import line to register the method.
- `experiments/week_02_2026-11-09/2026-11-09_oracle_fill/oracle_fill.py` (new): `check` and `run` CLI, analysis helpers.
- `tests/test_lab_oracle_fill.py` (new).
- Never `src/`, `cfg/`, `shell/`.

## Tests
- Prediction filing equals upstream Ramen (`torch.equal` on logits over a stream with a filled cache).
- Prediction filing does not depend on labels (two label seeds, identical logits).
- Label filing refuses to run without labels (ValueError).
- Label filing: every cache entry's true label equals its queue.
- Prediction filing: every entry sits in the queue of its prediction.
- Retrieval log: mass per queue and per target agree per query; the log is flushed on reset; sizes within capacity.
- The invariant suite (`tests/test_lab_invariants.py`) runs on the new method with its default (prediction) config.

## Runs and resources
1. `check CIFAR10C --n 200`: FileByPred vs upstream Ramen must match 100%. (Subset, GPU 0.)
2. Full CIFAR10C, seed 0, batch 100 (frozen), GPU 0 (RTX 3080 Ti, same model as the Ramen baseline), RAM budget 8.2 GB:
   FileByPred, then FileByLabel, chained in one detached job.
3. Compare FileByPred with the baseline: identical predictions expected (100%). If not, the comparison is void.
4. Noise floor for CIFAR10C: 2 × (max − min) of Ramen's mean accuracy over seeds 0–4 at batch 100, from
   `experiments/week_01_2026-11-02/2026-11-01_baseline/results/pc2_C10_Ramen_b100_s{0..4}/metrics.json`.
   The floor is used only if the seed-0 gap is near it, and then seeds 1–4 are run for FileByLabel.

## Pre-registered reading
- **Primary**: acc(FileByLabel) − acc(FileByPred), seed 0, same GPU, batch, stream.
- **Gate**: a gain counts as real only if it exceeds the CIFAR10C noise floor. A decrease is reported as found. The
  notebook then shows examples where FileByPred is right and FileByLabel is wrong.
- **Diagnostic only**: cache composition (queue × true label), retrieval mass by query class and gradient target,
  active-set images, examples. These explain a result and are not evidence of one by themselves.
- **Stop rule**: if FileByPred is not identical to Ramen on the full run, stop and fix the method before reading anything else.
