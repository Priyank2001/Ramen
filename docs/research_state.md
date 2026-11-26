# Research state: what's closed, what's open

The goal: beat Ramen (mixed-domain TTA, CLIP). This file holds **conclusions and pointers to evidence only, no numbers** (AGENTS.md §4.1). Read numbers from the cited notebook or `metrics.json`.
It's input to `scripts/critique_direction.sh`, so Gemini doesn't re-propose closed directions.
Update it when a direction closes or opens (owner: the user; agents may propose edits).

**On `pc2_lab`:** the evidence paths below are on branch `sec_server_dual_agent` (old harness, batch 100). Conclusions carry over as priors; numbers do not (different batch, D18).

Seeded 2026-10-24 from Claude's project memory. Items marked † have their evidence in `../Ramen_untracked_backup_2026-10-22/` (`plan.md`, `ramen-progress-brief.md`), outside the repo.

## Closed: don't re-propose without new evidence

| Direction | Why closed | Evidence |
|---|---|---|
| Memory admission policy / gating | no gain over Ramen's FIFO-priority cache | † plan.md phases 0–5 |
| Relabelling memory entries | breaks Ramen's prediction balance | † plan.md |
| Domain-identity retrieval keys | domain consistency of retrieval is already saturated (oracle) | † plan.md |
| Spatial / SAM-based keys | no gain | † plan.md |
| Cache-classifier logits (Tip-Adapter style) | no gain | † plan.md |
| Hyperparameter tuning of Ramen | defaults are already optimal on the dev stream | † plan.md |
| ZCA-whitened retrieval keys | helps CIFAR-10-C, hurts CIFAR-100-C. Within-class / shrunk / source-statistics ZCA don't fix it; not the backbone | † ramen-progress-brief.md (E0–E9) |
| Larger cache (K beyond 7500) | flat | † ramen-progress-brief.md |
| kNN output smoothing ("ask the neighbours") | below the pre-registered stop rule on held-out IN-C/DN. Zero-shot neighbours are same-domain, not same-class, so smoothing feeds the domain attractors | `new_experiments/experiments_till_2026-10-24/knn_heldout.ipynb` |
| Agreement-scaled sign step (step_d = lr · agreement_d · sign S_d) | seed-0 gate failed on both CIFAR sets: far below Ramen, above the matched uniform-shrink control. Mean agreement is low, so it acts mostly as a smaller lr. Agreement does pick the useful coordinates (beats ctrl), untested at Ramen's step size | `new_experiments/experiments_till_2026-10-26/agreement_step.ipynb` |
| Cleaner memory filing (any "file entries under a better label" idea, incl. purity fixes) | ORACLE ceiling is negative: filing every sample under its TRUE class (after prediction) is far below the matched write-after control on DN, below zero-shot, driven by quickdraw collapsing onto a few attractor classes. Likely why [hypothesis, untested]: the cached value is the entry's entropy gradient toward its own zero-shot argmax, whatever queue it sits in; true filing spreads quickdraw entries into every queue | `new_experiments/experiments_till_2026-10-27/results/pc2_DN_vitb32_{all_after,oracle_after}_s0/metrics.json`, `docs/plans/oracle_filing.md` (seed 0; notebook not written yet) |
| Mixing the own entropy gradient into Ramen's step (G = lambda g_R + (1-lambda) g_H, global lambda, raw or unit-norm mix) | seed-0 gate failed on CIFAR-10-C: the held-out pick (lambda 0.9, unit) is within the noise floor of Ramen on the test stream; every lower lambda is below Ramen. The own gradient helps zero-shot-right images and hurts zero-shot-wrong ones more (it confirms the zero-shot error). An image-dependent lambda would need a label-free signal of zero-shot correctness [untested] | `experiments/week_04_2026-11-23/2026-11-27_grad_mix/grad_mix.ipynb`, `docs/plans/grad_mix.md` |
| prob T=3 active set | **done**, not closed: positive on all DN seeds and on IN-C. Don't re-run it; build on it | `experiments_till_2026-10-23/results/pc2_DN_vitb32_probT3_s*/metrics.json`, `seed_sweep.ipynb` |

## Known facts that any new direction must explain

- **Attractor classes.** Zero-shot CLIP has corruption- or domain-specific attractor classes. Ramen shrinks old attractors but creates new ones, and relabels whole domain clusters as a block. (`experiments_till_2026-10-22/attractor_classes.ipynb`)
- **Cold memory.** Cache-based variants lose early in the stream and gain once memory is full (prob T=3, kNN). (`seed_sweep.ipynb`, `knn_heldout.ipynb`)
- **Neighbours share the domain, not the class.** Zero-shot feature neighbours have low class purity on IN-C, and quickdraw neighbours are all quickdraw. (`knn_heldout.ipynb`)
- **Non-stationary streams cost Ramen little.** († ramen-progress-brief.md)
- **Hardware noise.** Predictions differ across GPU models (fp16 kernels), so per-sample comparisons need the same GPU. (`docs/machines/pc2.md`)

## Open

- **Hubness correction (CSLS / QB-Norm) on Ramen's retrieval**: in progress, offline Phase A (`experiments_till_2026-10-24/hubness.py`, `sweep.py`).
- Remaining items in `new_experiments/new_directions.md` on `sec_server_dual_agent` (directions 4–8; directions 2 and 3 closed).

## Protocol reminders

- Tune only on held-out domains (`--domains extra`). IN-C and DN are held out.
- Pre-register a stop rule before running the decisive test.
- Compare against the backfilled baselines (`docs/GUIDE.md` §3) on the same GPU model.
