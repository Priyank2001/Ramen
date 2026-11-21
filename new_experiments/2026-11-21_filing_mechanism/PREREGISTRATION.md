# Pre-registration: E2, causal test of containment (2026-11-21, before any E2 run)

**Claim under test.** True-label filing (ORACLE) lowers accuracy (seed 0: DomainNet −1.50, ImageNet-C −0.71,
EuroSAT −6.50 vs Ramen) because it spreads entries that share a query's wrong zero-shot class over many class
queues, so the query's update averages many entries whose entropy gradient re-sharpens that wrong class
(DomainNet replay: weight share 1.4% → 8.7% on zs-wrong queries, quickdraw 1.0% → 20.2%).

**Intervention.** `RamenHideConfusion` (local_methods.py): true-label filing, and for each query hide the
cache entries whose zero-shot class equals the query's (own entry kept), before the per-queue top-k.
Seed 0, DomainNet / ImageNetC5K / EuroSAT, each dataset's Ramen config.

| run | filing | hidden | applied to | uses labels |
|---|---|---|---|---|
| A `hide_samezs_wrong_true` | true label | same zero-shot class as the query | zs-wrong queries | yes (filing, gate) |
| B `hide_random_wrong_true` | true label | same NUMBER of random entries | zs-wrong queries | yes (filing, gate) |
| C `hide_samezs_all_true` | true label | same zero-shot class | all queries | yes (filing) |
| D `hide_samezs_all_zs` | zero-shot (Ramen) | same zero-shot class | all queries | no |

References (cached): Ramen; the true-label oracle (`oracle_file_true` DN / IN-C, `labeloracle_true_entropy` EuroSAT).
Floors (2 × Ramen range over seeds 0–4): DomainNet 0.11, ImageNet-C 0.53, EuroSAT 1.51.

**Predictions.**
- P1 (main). On DomainNet, A recovers at least half of the oracle's loss (A ≥ Ramen − 0.75), and A − B > 0.11.
  ImageNet-C and EuroSAT: A − B > 0 (their floors are large relative to the oracle loss, so direction only).
- P2. On zs-wrong queries, the share that keep the zero-shot prediction under A is closer to Ramen's than to the
  oracle's. A and the oracle are identical on zs-right queries by construction except through the shared cache
  (the cache contents are the same; only retrieval differs), so the gain must come from zs-wrong queries.
- P3 (no strong prediction). D vs Ramen within the floor: Ramen's same-confusion share is already ~1.4%.
- If A ≈ B (both recover, or neither): containment is not the cause, or not the main one, and the loss is about
  removing entries in general.

**Smoke checks (must pass before the full runs).** On a subset: hide none + true filing == RamenTrueLabel
true_label, and hide none + zero-shot filing == Ramen, both 100% of predictions.
