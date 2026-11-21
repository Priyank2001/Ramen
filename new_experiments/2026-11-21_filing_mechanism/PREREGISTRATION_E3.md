# Pre-registration: E3, do the query's confusion neighbours close the quickdraw gap? (2026-11-21, before any E3 run)

**Background (E2).** True-label filing (ORACLE) loses 1.50 on DomainNet. Hiding the entries that share the query's
zero-shot class (run A, ORACLE gate on zs-wrong queries) recovers 49% of it, but only 26% on quickdraw
(Ramen 17.54, oracle 11.05, A 12.74), where the replay found the strongest containment. The other five domains are
already at or above Ramen after A.

**Hypothesis (untested guess from the E2 write-up).** Entries whose zero-shot class is a *different member of the
query's confusion cluster* (envelope / squiggle / pillow for a rectangle doodle) pull the same way, and "same
zero-shot class" misses them.

**Intervention.** As A (true-label filing, hide on zs-wrong queries, own entry kept) but the hidden set is widened to
entries whose zero-shot class is among the query's top-m zero-shot classes (its own logits; label-free), m = 3 and
10. Because the query's own zero-shot class is always in its top-m, this is a strict superset of A. Control: hide
the same NUMBER of random entries (per query). Seed 0, DomainNet (floor 0.11) and ImageNet-C (0.53).
EuroSAT is skipped (10 classes: top-3 / top-10 is 30% / 100% of all classes).

| run | m | hidden set | cached? |
|---|---|---|---|
| `hide_top3_wrong_true` / `hide_rand3_wrong_true` | 3 | top-3 zs classes / matched random | new |
| `hide_top10_wrong_true` / `hide_rand10_wrong_true` | 10 | top-10 zs classes / matched random | new |
| A `hide_samezs_wrong_true` (m = 1), B `hide_random_wrong_true` | 1 | same zs class / matched random | E2 |

Uses the true label for filing and for the gate (ORACLE), like A. Not a method.

**Predictions.**
- P1 (main). On DomainNet, for at least one m, top-m beats A by > 0.11 (domain-avg) and beats its own matched
  random control by > 0.11.
- P2. DomainNet quickdraw: the better top-m reaches at least 14.3 (≥ 50% of the 6.49 loss; A is 12.74).
- P3. On quickdraw, accuracy is ordered A ≤ top-3 ≤ top-10.
- P4. ImageNet-C: top-m − matched random > 0 (direction only; the loss is 0.71 against a 0.53 floor).
- **Falsifier.** If top-m is within 0.11 of A on DomainNet and quickdraw stays ≤ 13.3, the confusion-neighbour
  hypothesis is rejected and the remaining quickdraw loss comes from something else.
- **Risk, stated now.** For a zs-wrong query the true class is often within the top-m, so hiding entries
  zero-shot-predicted as the query's true class may remove helpful ones; P3 may fail for that reason. Diagnostic
  (descriptive, not scored): the share of hidden entries that were zero-shot-predicted as the query's true class.
