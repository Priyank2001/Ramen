# 2026-10-26 — Domain-sequential streams

Pre-registered (`PREREGISTRATION.md`, plus an amendment written before any result): domains arrive one after another, each shuffled within itself, and the method is never reset. ImageNet-C 5K (15 corruptions, standard order), DomainNet (clipart → … → sketch), ImageNet-R/A/Sketch. Ramen and T3 seeds 0–4, Ramen-reset (knows the boundaries) seeds 0–4, Ramen-fp32 / T3-fp32 (amendment) DomainNet seeds 0–4 + seed-0 screens.

**Result: H1 fails on DomainNet.**

| T3 − Ramen, sequential (pts, seeds 0–4) | ImageNet-C | DomainNet | R/A/Sketch |
|---|---|---|---|
| fp16 (as pre-registered) | **+0.52 ± 0.06** (5/5) | **−0.07 ± 0.02** (0/5) | **+0.66 ± 0.11** (5/5) |
| fp32 priorities (amendment) | +0.58 (seed 0) | −0.12 ± 0.01 (0/5) | +0.58 (seed 0) |
| mixed stream (earlier) | +0.62 | +0.24 | +0.59 |
| noise floor (sequential Ramen) | 0.37 | 0.13 | 0.40 |

Two findings about Ramen itself:
- **The fp16 priority freeze** locks DomainNet's memory during painting: only 0.3% of sketch images ever enter it. With fp32, Ramen gains +0.22 ± 0.01 and T3 +0.17. There is no effect on the two shorter streams.
- **Stale memory across domains:** Ramen-reset beats sequential Ramen by +0.82 ± 0.05 on ImageNet-C, where capacity exceeds the per-corruption volume, so nothing is evicted and only ~1/n of the memory is the current corruption. The gap is +0.27 on DomainNet (≈ +0.04 vs fp32) and +0.16 on R/A/Sketch.

| File | What it is |
|---|---|
| `PREREGISTRATION.md` | Hypothesis, runs, decision rule; amendment on fp16 priorities |
| `run_sequential.py`, `run_queue.sh` | GPU runs (detached queue, one worker per GPU, longest first) |
| `sequential_analysis.py` | Gains, floors, CIs, lag, offline memory replay (validated vs logged queues), plots → `results/sequential/*.csv` |
| `sequential.ipynb`, `build_notebook.py`, `interpretation.md` | Write-up |
| `logs/failed_fp16_cdist/` | The first DomainNet attempt, which crashed on `cdist` in fp16 (fixed, see below) |

Code changes, both defaulting to the old behaviour and each verified by re-running the mixed ImageNet-C Ramen seed 0 (75,000 of 75,000 predictions identical):
- `harness tta_mode='sequential'` in `ramen_analysis.py`.
- `torch.cdist(..., compute_mode='use_mm_for_euclid_dist')` in `Ramen.py` / `RamenActive.py`. The default crashed on a ≤ 25 × ≤ 25 fp16 query, e.g. infograph's 1-image last batch; it is bitwise identical wherever the default worked.
- `priority_fp32` config switch (eviction only).
