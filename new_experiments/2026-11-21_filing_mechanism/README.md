# Why does filing Ramen's cache by the true label hurt? E1 + E2 (2026-11-21)

Follow-up to `../2026-11-18_true_label_mechanism/` (containment, DomainNet replay) and
`../2026-11-01_label_oracle/` (true-label filing also hurts with label values).

## Plain-language summary of the question

Ramen's memory has one queue ("page") per class. Each test image is written onto one page as a gradient that
makes the model more confident in CLIP's current guess for that image (right or wrong); the label is not in it.
A new image takes the nearest entries from every page, pages weighted equally. The ORACLE writes each image onto
the page of its true label instead of CLIP's guess. That is purer, yet accuracy drops. Why?

## Files

| File | Job |
|---|---|
| `label_value_pull.py` | **E1** (CPU replay): with label values, the pull toward the query's true vs wrong class, by source queue |
| `local_methods.py` | **E2** method `RamenHideConfusion` (RamenTrueLabel + per-query hidden entries; tags kept outside the repo cache) |
| `registry.py`, `experiment.py` | local method lookup + harness hook (no edits to `src/` or `cfg/`); `VARIANTS`, `DATASETS`, smoke pairs |
| `run_e2.py` | `smoke <dataset>` (hide none == RamenTrueLabel / Ramen, 100%) and `run <dataset>:<variant> ...` |
| `analysis_e2.py` | E2 tables and the pre-registered verdicts → `tables/` |
| `after_runs.sh` | waits for the 12 E2 result files, then runs `analysis_e2.py` |
| `PREREGISTRATION.md` | E2 predictions, written before any E2 run (E1's are in its docstring) |
| `results/`, `tables/`, `logs/` | outputs |

## E1 result (DomainNet seed 0, 6,000 sampled queries, 2,686 zero-shot wrong)

Label-value runs (ORACLE values): every entry from an earlier batch pushes toward its TRUE class; entries from the
query's own batch still hold entropy gradients. Mean weight share on zero-shot-wrong queries:

| | zero-shot filing | true-label filing |
|---|---|---|
| pull toward true class, total | 0.90% | 0.78% |
| ... from queues other than the true-class queue | 0.40% | 0.00% |
| pull toward the wrong zero-shot class, other images only | 0.82% | **1.33%** |
| ... from the zero-shot-class queue (other images) | 0.60% | **1.32%** |
| net (true − wrong, incl. own entry) | −0.31% | −0.99% |
| correct on these queries (label-value runs) | 10.76% | 8.49% |

- Under true filing the query's zero-shot-class queue ("envelope") holds only real members of that wrong class,
  and with label values they all pull toward it: the wrong pull from that queue doubles. Under zero-shot filing
  that queue is diluted by misfiled images that point elsewhere.
- Pull toward the true class from other queues (images CLIP misfiled there) exists only under zero-shot filing,
  but it is small (0.40%).
- Net pull predicts the outcome: top fifth of net pull 27.4% correct vs bottom fifth 4.1% (zero-shot filing).
- Pre-registered L1 ("lookup": true-class pull from the zero-shot queue) is weak: 0.10% vs 0.00%. L2 (wrong pull
  rises) and L3 (net pull lower, predicts correctness) hold.

## E2 result (seed 0; accuracy domain-averaged; floors 0.11 / 0.53 / 1.51)

Intervention: true-label filing, but each query ignores the entries that share its zero-shot class (own entry kept).
Gap to the true-label oracle: Ramen − oracle = 1.50 (DomainNet), 0.71 (ImageNet-C), 6.50 (EuroSAT).

| run | DomainNet | ImageNet-C | EuroSAT |
|---|---|---|---|
| Ramen | 56.80 | 33.08 | 62.63 |
| true-label oracle | 55.29 (−1.50) | 32.37 (−0.71) | 56.13 (−6.50) |
| **A** oracle + hide same-zs, zs-wrong queries (ORACLE gate) | 56.03 (−0.77) | 32.68 (−0.40) | 66.99 (+4.36) |
| **B** oracle + hide same number of random entries, same queries | 55.29 (−1.50) | 32.36 (−0.72) | 56.33 (−6.30) |
| C oracle + hide same-zs, all queries | 55.15 (−1.65) | 32.60 (−0.48) | 54.64 (−7.99) |
| D Ramen + hide same-zs, all queries (no labels) | 56.63 (−0.17) | 33.08 (0.00) | 61.07 (−1.56) |
| **A − B** (95% CI, clustered) | **+0.74** [+0.71, +0.76] | **+0.32** [+0.26, +0.39] | **+10.66** [+10.23, +11.08] |

- **Causal contrast holds on all three datasets.** Hiding the same number of random entries (B) changes nothing
  (DomainNet −1.50 vs −1.50), hiding the entries that share the query's zero-shot class (A) recovers 49% / 44% of
  the oracle's loss on DomainNet / ImageNet-C and more than all of it on EuroSAT.
- **It is partial where the loss was measured.** DomainNet's remaining loss after A (0.77 of 1.50) is almost entirely
  quickdraw: quickdraw Ramen 17.54, oracle 11.05, A 12.74 (26% recovered). The other five domains are at or above
  Ramen after A (infograph 45.02 vs 44.98, painting 66.16 vs 65.70, real 82.73 vs 82.27, clipart 65% recovered,
  sketch 56%).
- **The label-free version does not work (C, D).** Hiding same-zero-shot-class entries on ALL queries hurts: on
  zero-shot-right queries those entries are the helpful same-class neighbours (DomainNet zs-right accuracy
  95.6 → 93.7 in C; EuroSAT 90.3 → 64.6). A only works because its gate (is the zero-shot prediction wrong?)
  uses the true label. D is −0.17 (outside the 0.11 floor), 0.00, −1.56 (outside 1.51).

Pre-registered predictions:
- **P1 FAIL on DomainNet as written:** A recovers 49%, A = −0.77 vs the threshold −0.75 (missed by 0.02); the
  second part (A − B > 0.11) holds with room (+0.74). ImageNet-C and EuroSAT direction: PASS.
- **P2 PASS** on all three (zero-shot-wrong queries keep the zero-shot prediction 49.8% under A vs Ramen 57.5%,
  oracle 71.0%, on DomainNet). Note A goes below Ramen, not between: on EuroSAT A keeps it 18.0% (Ramen 43.8%,
  oracle 55.7%).
- **P3 FAIL on 2 of 3** (D vs Ramen: DomainNet −0.17, EuroSAT −1.56 outside the floors; ImageNet-C 0.00).

## Why it makes sense (and what is not shown)

- **Why hiding should help:** if true-label filing lowers accuracy by piling same-confusion entries into the update,
  removing exactly those entries should help, and removing the same number of random entries should not.
- **What happened:** exactly that, on three datasets with very different class counts (10 / 345 / 1,000), with
  random hiding identical to the oracle. So re-imported same-confusion entries are a cause of the loss.
- **Not the whole cause on DomainNet:** about half of it, and only a quarter on quickdraw, where containment
  looked strongest in the replay (same-confusion weight share 20.2%). Hypothesis, not tested: entries whose
  zero-shot class is a different member of the same confusion cluster (envelope / squiggle / pillow) are not
  caught by "same zero-shot class" but pull the same way. Next test: hide entries whose zero-shot class is one of
  the query's few nearest confusion classes.
- **A is an ORACLE rule, not a method.** It needs to know which queries are wrong, and the label-free D shows the
  rule is harmful without that. Seed 0 only.

## E3 result: hiding the query's top-m zero-shot classes (seed 0; `PREREGISTRATION_E3.md`, `analysis_e3.py`)

Intervention: as A (true-label filing, zs-wrong queries only, ORACLE gate), but hide every entry whose zero-shot class
is among the query's top-m zero-shot classes (m = 3, 10), against random hiding of the same number of entries.
EuroSAT skipped (10 classes, top-10 = everything). Runs: `hide_top{3,10}_wrong_true`, `hide_rand{3,10}_wrong_true`.

| | DomainNet | | ImageNet-C | |
|---|---|---|---|---|
| run | acc | recovered | acc | recovered |
| Ramen / oracle | 56.80 / 55.29 | 100% / 0% | 33.08 / 32.37 | 100% / 0% |
| A (m = 1) | 56.03 | 49% | 32.68 | 44% |
| top-3 | 55.65 | 24% | 32.74 | 52% |
| top-10 | 55.70 | 27% | 32.83 | 65% |
| random, matched to top-3 / top-10 | 55.29 / 55.29 | 0% / 0% | 32.33 / 32.30 | <0% |

top − random: DomainNet +0.36 (m=3), +0.41 (m=10); ImageNet-C +0.40, +0.53 (all CIs exclude 0).
top − A: DomainNet −0.38, −0.33 (worse than A); ImageNet-C +0.05 [−0.01, +0.12], +0.14 [+0.05, +0.23] (below the 0.53 floor).

Quickdraw (the domain A left unrepaired): Ramen 17.54, oracle 11.05, A 12.74, top-3 12.86, top-10 13.37, controls 11.06.
Other DomainNet domains get worse than under A (clipart 69.08 → 68.64, painting 66.16 → 65.59, real 82.73 → 82.21).

Pre-registered verdicts:
- **P1 FAIL** (DomainNet: no m beats A by > 0.11; both are 0.33–0.38 below A).
- **P2 FAIL** (best top-m quickdraw 13.37 < 14.3).
- **P3 PASS, barely** (quickdraw 12.74 / 12.86 / 13.37; the first step is +0.12).
- **P4 PASS** (ImageNet-C: top-m − random > 0 for both m).
- **Falsifier not triggered, by 0.07** (quickdraw 13.37 vs the 13.3 line), but only because of that: top-m is clearly not better than A on DomainNet.

What it says:
- **Same-confusion hiding is a real, specific effect again.** Random hiding of 2-4x as many entries (3,324 / 7,218 per
  zs-wrong query on DomainNet vs 1,627 for A) does nothing; targeted hiding helps on both datasets.
- **The "confusion-neighbour classes" idea does not close the quickdraw gap.** Widening m moves quickdraw by +0.12
  then +0.63 points of a 6.49 gap, while losing on the other domains.
- **The cost is helpful entries.** Hidden entries that were zero-shot-predicted as the query's TRUE class: 2.27% (top-3),
  1.85% (top-10) of hidden vs 0.28% for random (about 8x enrichment): neighbours in the zs ranking are also where the
  true class sits, so a wider net hides the helpful entries along with the harmful ones. A's rate was not logged (E2
  predates the diagnostic).
- **ImageNet-C moves the other way** (1,000 classes, m=10 recovers 65% vs A's 44%) but the gain over A is under the floor.
- **Still an oracle rule** (true labels for filing and gate); seed 0 only. Remaining quickdraw loss is not explained
  by same-class or near-class re-import; the quickdraw explanation stays open.
