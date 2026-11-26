# oracle_fill: file each sample by its true label instead of its prediction

**Status:** code and tests written (the background test suite exited 0; its output was not read). The subset `check` and the full runs have not started. Notebook pending. Plan (pre-registered): [docs/plans/oracle_fill.md](../../../docs/plans/oracle_fill.md).

## Question
Regardless of what the model predicts, fill Ramen's cache by the **true label** (oracle) and compare accuracy with Ramen's own filing by prediction. This is an ORACLE: it reads labels, and it is a diagnostic, not a method.

## What the notebook will show
1. **Accuracy** of FileByLabel vs FileByPred (= Ramen), per domain and overall, on the same stream, seed, GPU and batch, with the CIFAR-10-C noise floor (2 × range of Ramen over seeds 0–4).
2. **If accuracy decreases:** examples where FileByPred is right and FileByLabel is wrong, with the query image, the two predictions, and the top retrieved entries each run used.
3. **Caches, two copies:** queue × true-label heatmaps for the cache filed by prediction (Ramen) and the cache filed by true label, both at the end of the stream.
4. **Active set:** the images held in each cache, one row per queue, with frames marking images held by only one of the two caches.
5. **Mechanism:** retrieval mass per query class and per gradient target, for both runs.

## Variants
| name | config | queue of sample j | labels read? |
|---|---|---|---|
| FileByPred | `filing: prediction` | argmax of zero-shot logits (Ramen) | no |
| FileByLabel | `filing: label` | true label y_j | **yes** (filing only) |

FileByPred must give the same predictions as upstream Ramen to the bit; `check` tests it on a subset, and the full-run
comparison against the existing baseline `pc2_C10_Ramen_b100_s0` repeats it.

## What gets written
- `results/<run_id>/metrics.json`: accuracy, per domain, resources (the citable record).
- `results/<run_id>/records.npz`: per-sample predictions, stream order.
- `results/<run_id>/cache_snapshot.npz`: every entry of every queue at the end of the stream: stream position, priority, gradient target.
- `results/<run_id>/retrieval.npz`: per query, the retrieved weight mass per queue and per gradient target, and the top-5 retrieved entries.

## Commands
```
python oracle_fill.py check CIFAR10C --n 200
python oracle_fill.py run CIFAR10C --filing prediction --seed 0   # launched detached, see ramen_lab/launch.py
python oracle_fill.py run CIFAR10C --filing label --seed 0
```

## Files
- Method: [ramen_lab/methods/oracle_fill.py](../../../ramen_lab/methods/oracle_fill.py), tests: [tests/test_lab_oracle_fill.py](../../../tests/test_lab_oracle_fill.py).
- Analysis and CLI: [oracle_fill.py](oracle_fill.py). Notebook: `oracle_fill.ipynb` (to be written and executed).
