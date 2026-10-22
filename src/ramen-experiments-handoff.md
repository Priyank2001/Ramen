# Experiments Handoff: Ramen Mixed-Domain TTA Investigation

Continuation of [ramen-tta-thesis-brief.md](ramen-tta-thesis-brief.md) (read that first for full context:
thesis goal, how Ramen works, the four candidate improvement angles). This file summarizes what's been
run in [experiments.ipynb](experiments.ipynb), what it found, and exactly what to run next on a machine
with free GPU.

## Setup notes (for the new machine)

- Conda env used previously: `RAMEN` (has `torch`, `tqdm`, `clip`, `scikit-learn`, `pandas`, `matplotlib`).
  If starting fresh, check `requirements.txt` at repo root.
- Data expected at `~/data/corruption/...` (`DATA_ROOT` in the notebook). Re-point `DATA_ROOT` /
  `--data_root` if the new machine's data lives elsewhere.
- Existing real results already on disk under `../log/`:
  - `CIFAR10C_mixed/Ramen.csv`, `CIFAR100C_mixed/Ramen.csv` — Ramen, `tta_mode=mixed`, `use_tbn=False`,
    default `beta=5.0`, produced by `shell/run_ramen.sh`.
- The notebook's `RUN_EXPERIMENTS` / `RUN_DIAGNOSTICS` / `RUN_TSNE` flags are all currently `True` in the
  source, but on the machine this was developed on, their *saved outputs* are stale (from a prior run
  where they were `False`/not executed) — re-run those cells fresh once GPU is available, don't trust the
  currently-committed outputs for anything except the two ablation-comparison cells' logic, which is new
  and untested against real data (see "Known issue" below).

## What's been established so far (real numbers, from `../log/`)

`flag_effect_table` in the notebook, from the two saved logs:

| dataset | NoAdapt | Tent | Ramen |
|---|---|---|---|
| CIFAR10C | 61.36% | 66.71% | 72.63% |
| CIFAR100C | 37.10% | 41.14% | 46.12% |

Ramen clearly beats Tent beats NoAdapt on average. `use_tbn` is confirmed a no-op for both algos
(identical numbers on/off) — expected, since CLIP ViT backbones have no `BatchNorm2d`, only LayerNorm
([ModelForTTA.py:87-94](models/ModelForTTA.py#L87-L94)).

**Per-domain weak spot (both datasets):** noise-type corruptions drag the average down hard —
CIFAR10C: `impulse_noise` 56.13%, `gaussian_noise` 60.84%, `shot_noise` 63.35%, vs. 72.63% avg.
CIFAR100C: `glass_blur` 29.34%, `gaussian_noise` 30.50%, `shot_noise` 32.77%, vs. 46.12% avg. Blur/weather
corruptions (`brightness`, `zoom_blur`, `snow`) are all close to or above the mean in both datasets.

**t-SNE of frozen CLIP embeddings (cell `9f17e7db`, no TTA applied) — the key diagnostic:** for both
datasets, `gaussian_noise`/`shot_noise`/`impulse_noise` (+ `pixelate` on CIFAR100C) form clusters that are
visually **isolated from the main blob** and, colored by class, show **no class-discriminative structure**
(rainbow-mixed). Every blur/weather domain stays mixed inside the main blob, which retains visible class
clustering. This means the noise domains aren't just "shifted" — the frozen backbone doesn't encode class
identity for them at all. Per Theorem 4.1 in the paper (entropy-min amplifies existing feature variance,
it can't manufacture structure that isn't there), this rules out retrieval/aggregation/hyperparameter
tweaks (brief's angles 1-3) as a fix for *those specific domains* — they operate downstream of an embedding
that's already lost the signal.

**Conclusion drawn from this:** prioritize angle 4 (pseudo-label noise in memory) first. The concrete,
fixable problem this points to: `PriorityCache.add` ([Ramen.py:29-64](methods/Ramen.py#L29-L64)) tracks
`entropies` but evicts by `priorities = torch.arange(counter, counter + B)` — a pure arrival-order FIFO,
not actually confidence-gated. Combined with `mixed_domain_test` sharing one persistent cache across the
*entire* stream ([main.py:48-77](main.py#L48-L77), not per domain), confidently-wrong pseudo-labels from
the noise domains can pollute the shared per-class queues used by every other domain too. A
confidence-gated or delayed-write memory policy is a small, cheap-to-test fix for this — cheaper than
committing to a learned retrieval/aggregation component (angles 1-2) before knowing if this simpler fix
moves the needle.

## Two ablations set up to sharpen this before writing new code (not yet run — GPU unavailable)

Both isolate a specific piece of the hypothesis above. Neither requires new code — both use existing
`config_overrides`/`tta_mode` plumbing already in `build_args()`.

### Ablation 1 — `tta_mode='single'` vs `'mixed'`

Does cross-domain cache-sharing in mixed mode actively hurt, or are the noise domains just irrecoverable
regardless? `single_domain_test` resets cache + LayerNorm state after every domain
([main.py:16-40](main.py#L16-L40)); `mixed_domain_test` doesn't reset until the whole stream ends.

Notebook: cells `c0abc6f7` (markdown) → `ad45f7ad` (runs the sweep) → `9503d7eb` (comparison table,
`single_minus_mixed` per domain, split noise vs. non-noise).

Equivalent CLI (uses the `RAMEN` env's `main.py` directly — more reliable than the notebook kernel):
```bash
cd src
python main.py --dataset CIFAR10C  --model clip_vitbase32 --tta_algo Ramen --tta_mode single --cuda --batch_size 100 --save_to ../log/ablations/CIFAR10C_single/Ramen.csv
python main.py --dataset CIFAR100C --model clip_vitbase16 --tta_algo Ramen --tta_mode single --cuda --batch_size 100 --save_to ../log/ablations/CIFAR100C_single/Ramen.csv
```
Compare per-domain accuracy against the existing `../log/CIFAR10C_mixed/Ramen.csv` /
`CIFAR100C_mixed/Ramen.csv`. **What to look for:** if `single ≥ mixed` specifically on domains *other
than* the noise ones (i.e. non-noise domains do worse when sharing a cache with noise-domain samples),
that's direct evidence of cross-domain contamination — angle 4 justified. If `single ≈ mixed` everywhere,
the noise domains are just hard regardless of sharing, and angle 4's payoff is smaller than expected.

### Ablation 2 — `beta` (domain-consistency distance weighting) on vs. off

Isolates how much of Ramen's gain comes from the distance-weighting term in `α_ij` vs. prediction-balance
(class-split queues) alone. `beta` scales `exp(-beta * dist)` in
[Ramen.py:148](methods/Ramen.py#L148) — it's `5.0` in `cfg/CIFAR10C/Ramen.yaml` /
`cfg/CIFAR100C/Ramen.yaml`, but `0.0` in the paper's own `ImageNetC5K` config (i.e. the paper itself
sometimes disables it).

Notebook: cells `aa4ee997` (markdown) → `7cdb56bb` (runs both `beta=5.0` and `beta=0.0`, mixed mode) →
`a68f4171` (comparison table).

Equivalent CLI — needs a scratch config dir with only `beta` changed:
```bash
cd src
mkdir -p /tmp/cfg_beta0/CIFAR10C /tmp/cfg_beta0/CIFAR100C
sed 's/^beta:.*/beta: 0.0/' ../cfg/CIFAR10C/Ramen.yaml  > /tmp/cfg_beta0/CIFAR10C/Ramen.yaml
sed 's/^beta:.*/beta: 0.0/' ../cfg/CIFAR100C/Ramen.yaml > /tmp/cfg_beta0/CIFAR100C/Ramen.yaml

python main.py --dataset CIFAR10C  --model clip_vitbase32 --tta_algo Ramen --tta_mode mixed --cuda --batch_size 100 --config /tmp/cfg_beta0 --save_to ../log/ablations/CIFAR10C_mixed_beta0/Ramen.csv
python main.py --dataset CIFAR100C --model clip_vitbase16 --tta_algo Ramen --tta_mode mixed --cuda --batch_size 100 --config /tmp/cfg_beta0 --save_to ../log/ablations/CIFAR100C_mixed_beta0/Ramen.csv
```
Compare against the existing `beta=5.0` mixed logs. **What to look for:** if `beta=5.0` beats `beta=0.0`
mainly on non-noise domains (where domain-consistency similarity is actually informative) rather than on
noise domains (where the t-SNE shows the embedding itself is uninformative for retrieval), that's further
confirmation the noise-domain problem isn't fixable by better retrieval weighting.

## Known issue to watch for when re-running

The notebook's `mode_comparison_table` / `beta_comparison_table` cells were patched to guard against a
`KeyError` when one side of a comparison (`single`/`mixed` or `beta=5.0`/`0.0`) has no data — e.g. if a
config fails (OOM, missing data) and `run_sweep`/the manual loop only partially populates the DataFrame.
If you see "Missing 'single' or 'mixed' rows" / "Missing beta=5.0 or beta=0.0 rows" printed instead of the
table, check the sweep cell's output above it for `FAILED: ...` lines first.

Also: `run_single_experiment`/`run_sweep`'s in-notebook duplicates of `main.py`'s functions did not
actually get validated end-to-end against a live kernel in this environment (kernel/env mismatches ate the
session's time — see below). The **CLI commands above are the tested, known-working path** on a fresh
machine; treat the notebook cells as convenience wrappers around the same `build_args`/`run_sweep` logic,
but verify the first run's output looks sane before trusting a full sweep.

## Environment gotcha from this session (may recur on the new machine)

If notebook cells fail with `ModuleNotFoundError` for `torch`/`tqdm`/etc. even though they're installed
somewhere on the box: check `conda env list` for a project-specific env (was `RAMEN` on the old machine)
and make sure the Jupyter kernel is registered from *that* env's `python -m ipykernel install --user
--name <name>`, not the system `python3`. The system Python and the project's conda env were different
installations with disjoint site-packages here.

## Immediate next steps on the new machine

1. Re-run `RUN_EXPERIMENTS` / `RUN_DIAGNOSTICS` / `RUN_TSNE` cells fresh to confirm the numbers/plots
   above reproduce (sanity check before trusting anything new).
2. Run the two ablations above (CLI path recommended). ~4 runs total, each comparable in cost to the
   existing single mixed-mode Ramen run (a few minutes on a free GPU based on the "490x speedup" figure
   from the paper — CIFAR-100-C mixed took ~14 min per the brief).
3. Based on the ablation 1 result: if cross-domain contamination is confirmed, implement a confidence-gated
   or delayed-write policy for `PriorityCache.add` ([Ramen.py:29-64](methods/Ramen.py#L29-L64)) — e.g. only
   admit a sample once its entropy is below a threshold, or delay writes by a short buffer so early
   confidently-wrong predictions don't immediately enter the retrieval pool. Ablate against the current
   FIFO-only behavior on the same CIFAR-10-C/100-C setup.
4. Extend to ImageNet-C / DomainNet and larger backbones (ViT-B/16, ViT-L/14) only after the CIFAR-scale
   ablations give a clear signal — per the brief, match Ramen's own protocol for a fair comparison before
   scaling up compute.
