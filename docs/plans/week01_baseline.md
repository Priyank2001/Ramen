# Plan: week 01: `ramen_lab` core + Ramen baseline and noise floor at a fixed batch

*2026-11-01. Status: draft, waiting for the user's go-ahead. Core-code task (new package `ramen_lab/`, tests), full loop of AGENTS.md §6. No upstream file changes (D18).*

## Question

What do NoAdapt and Ramen score on `pc2_lab`, at a batch that fits our GPUs, and how much does Ramen move across seeds 0–4? Every later idea is judged against these numbers.

**Not a hypothesis test.** The deliverables are a working, tested `ramen_lab`, a frozen batch per dataset, the CIFAR-10-C baseline (seeds 0–4) and its noise floor.

## Derivation

Nothing changes in the method. The notebook writes out Ramen's update (`references/notebook.md` §2 of the `ramen-experiment` skill) and adds the one thing that differs from the paper: the batch size $B$.
- $B$ does not enter $G_i$ directly; $G_i$ uses only the memory and $z_i$.
- $B$ does change two things:
  1. **How fresh the memory is:** with Ramen's write-before-retrieve, a query sees its $B-1$ batch-mates.
  2. **How often priorities advance:** the fp16 counter increases by $B$ per batch, so the order in which fp16 becomes coarse is the same per sample. [to verify in code]

So a smaller $B$ is expected to change accuracy a little, not the mechanism. The notebook reports the gap to the paper's number and doesn't explain it away.

## `ramen_lab` design (SOLID; see the skill's `references/code_style.md`)

| module | responsibility | key types |
|---|---|---|
| `upstream.py` | The only importer of `src/`. Puts `src` on `sys.path` once, and adapts model loading, datasets and upstream methods. | `UpstreamAdapter.load_model(name, device)`, `.load_datasets(name, root, transform)`, `.method_class(name)` |
| `config.py` | Run configuration, validated, immutable. Per-dataset defaults (backbone, fixed batch, RAM budget). | `RunConfig` (frozen dataclass), `DATASET_DEFAULTS` |
| `registry.py` | Name → factory. Upstream `NoAdapt`/`Tent`/`Ramen` are pre-registered; variants use `@METHODS.register`. Unknown names list the options. | `Registry[T]`, `METHODS` |
| `stream.py` | Mixed or single stream, explicit `torch.Generator(seed)`, optional n-per-domain subset (deterministic, seeded). | `StreamFactory`, `Stream` |
| `resources.py` | RAM safety (D17): a background sampler of process-tree RSS and system available memory; stop flag; hard exit at 1.2 × budget; peak RSS and GPU. | `ResourceBudget`, `MemoryGuard` (context manager) |
| `records.py` | Per-sample arrays written in chunks to `records.npz` (label, pred, conf, true_prob, entropy, domain_idx, sample_idx, batch). `metrics.json` via `scripts/run_record.write_metrics`. | `RecordWriter`, `RunRecord` |
| `runner.py` | Builds model, data and method from a `RunConfig` through injected factories; iterates the stream; passes labels only to `LabelAware` methods; checks the guard each batch; resets at the end. | `Runner.run(config) -> RunResult` |
| `launch.py` | CLI: refuses a busy GPU or low RAM; caps workers; `systemd-run` memory cap if linger is on, else `setsid nohup`; logs to `results/logs/`. | `Launcher` |
| `analysis/stats.py` | Noise floor, paired per-seed gains, cluster bootstrap by base image, McNemar. Pure functions. | — |
| `analysis/plots.py` | `use_style()` and the chart functions in the notebook template. Pure: data in, figure out. | — |

Experiment code: `experiments/week_01_2026-11-02/2026-11-02_baseline/{baseline.py, baseline.ipynb, results/}`. `baseline.py` defines the runs (configs), `check`, `run` and the notebook's analysis helpers.

## Tests (`tests/`)

1. `test_upstream_untouched.py` (exists).
2. `test_registry.py`, `test_config.py`: registration; unknown names; frozen and validated configs; defaults per dataset.
3. `test_stream.py`: same seed gives the same order; different seeds give different orders; the subset is deterministic and balanced per domain; every sample appears once.
4. `test_resources.py`:
   - the guard trips when the budget is exceeded (it allocates in a child process with a small budget, no GPU);
   - the runner stops after the current batch;
   - peak RSS is recorded.
5. `test_records.py`: chunked writes give the same arrays as one write; `metrics.json` has the schema in AGENTS.md §5, plus `resources`.
6. `test_runner.py` (GPU, random images, 5 fake classes, B = 32):
   - upstream `Ramen` driven by `Runner` gives the same logits as calling `Ramen` directly in a loop (the runner adds nothing);
   - `reset()` gives the same output as a fresh model;
   - determinism;
   - weights restored after each batch;
   - frozen parameters unchanged;
   - labels never reach a non-`LabelAware` method.

## Runs and resources

1. **Fixed batch per dataset** (smoke, `check` mode, 200 images per domain, GPU 1 = 11 GB):
   - try B ∈ {100, 64, 50, 48, 32, 24} from the top;
   - pick the largest with peak GPU ≤ 10.5 GB *after* the cache is full. Fill all queues with random entries first, so the measurement doesn't miss the full-cache peak.
   - Freeze B in `DATASET_DEFAULTS` (one commit per change, with the measured peak in the commit message).
   - Also measure peak RSS with 4 workers, and set each dataset's RAM budget to 1.5 × that (≤ 24 GB).
2. **CIFAR-10-C baseline:**
   - NoAdapt seeds 0 and 1: seed-independence check, expected 0.00 difference.
   - Ramen seeds 0–4.
   - GPU per seed is fixed now: seeds 0, 2, 4 → GPU 0 (3080 Ti), seeds 1, 3 → GPU 1 (2080 Ti). Every later variant runs each seed on the same GPU as its baseline, so per-seed pairs share a GPU model.
   - One run per GPU, chained. Time per run TBD from the smoke (batch-100 runs took about 7 min on the 3080 Ti).
3. **Determinism check:** Ramen seed 0 on the CIFAR-10-C subset, run twice on the same GPU, must give 100.00% identical predictions.
4. Other datasets (CIFAR-100-C, ImageNet-C, DomainNet): only step 1 (fixed batch and RAM budget) this week. Their seed-0–4 baselines are queued after CIFAR-10-C, because they are needed only when an idea passes the CIFAR-10-C screen.

## Pre-registered reading

- **Done when:**
  - all tests pass;
  - the 4 batches are frozen;
  - the CIFAR-10-C NoAdapt and Ramen `metrics.json` files exist for the seeds above;
  - the notebook reports, for CIFAR-10-C:
    - Ramen's mean ± std over seeds 0–4;
    - the noise floor, 2 × (max − min) of the five means;
    - Ramen − NoAdapt;
    - the gap to the paper's reported number, with its section or table cited, or `[unverified]` until `docs/papers/ramen.md` exists.
- **Red flags that stop the week:**
  - NoAdapt differs between seeds;
  - the determinism check fails;
  - Ramen ≤ NoAdapt on CIFAR-10-C;
  - any run trips the RAM guard at the budget derived from the smoke.

  Each one means a bug or an unsafe setting, not a result.

## Risks

- **Pairing across GPU models:** the seed → GPU map makes the noise floor include GPU-model variation (conservative). Per-sample analyses compare only runs on the same GPU.
- **An unrepresentative peak GPU measurement:** the full-cache peak may not be reached on a 200-per-domain subset. That's why the queues are pre-filled with random entries before measuring (step 1).
- **Linger is off:** the hard cgroup cap isn't used for detached runs until the user enables linger. The in-process guard is then the only protection.
