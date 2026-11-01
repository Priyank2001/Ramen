---
name: ramen-experiment
description: Plan, implement, run and report experiments in the Ramen test-time-adaptation (TTA) thesis repo on branch pc2_lab - frozen upstream Ramen, our code in the ramen_lab package (clean OOP/SOLID), weekly experiment folders, RAM-safe launches, notebooks with full math derivations and static matplotlib figures. Use whenever the user wants to try an idea, run or launch an experiment, compare a variant against Ramen, do a seed sweep or ablation, check whether a gain is real, measure a baseline or noise floor, analyse runs, or write up results in a Jupyter notebook - e.g. "test this cache idea on CIFAR-10-C", "does this beat Ramen?", "launch seeds 0-4", "make the notebook for these runs", "is +0.3 real?". Also use it for TTA, CLIP zero-shot, memory/cache, CIFAR-C, ImageNet-C or DomainNet work even when the user doesn't say "experiment".
---

# Running experiments on pc2_lab

An experiment is only useful if someone else can trust the number, read the code, and follow the notebook without you in the room. Every rule below exists because a shortcut once cost a GPU-day, froze the server, or produced a wrong claim.

Read before starting:
- `references/lab.md` covers upstream Ramen's API, the `ramen_lab` package, datasets, configs, memory safety and gotchas.
- `references/code_style.md` is the OOP/SOLID standard all new code must meet.
- `references/notebook.md` is the notebook template, including the mandatory derivation section.
- `docs/research_state.md` lists closed directions. `references/legacy_batch100.md` holds old dead ends, usable as priors only.

## 0. Four hard rules

1. **Upstream Ramen is frozen.** Never edit or add files under `src/`, `cfg/` or `shell/`, or touch `README.md` or `.gitignore`. All new code goes in `ramen_lab/`, or in the experiment's own `.py` file. The guard denies such edits, and `tests/test_upstream_untouched.py` must pass before every commit. If a change seems to need an upstream edit, wrap, subclass or inject instead (`references/code_style.md`, "Extending a frozen upstream").
2. **Never exhaust the server's RAM.** PC2 has 125 GB of RAM and only 2 GB of swap, so an exhausted host freezes for everyone.
   - Every run goes through `ramen_lab`'s `MemoryGuard`, with a per-run RAM budget and a system reserve.
   - At most 4 DataLoader workers, and one run per GPU.
   - No pinned host-RAM gradient caches.
   - Per-sample outputs are streamed to disk, not accumulated in Python lists over a whole stream.
   - Detached runs get the hard cgroup cap when linger is on (`lab.md` §5).
3. **Fixed batch per dataset.** Use the frozen value in `ramen_lab` config (the largest that fits the 11 GB GPU 1). A variant and its baseline share batch, seed, GPU model and stream.
4. **Numbers come only from `metrics.json` files or executed notebook outputs.** Never quote them from memory.

## 1. Frame the question (before any code)

- Write the hypothesis as one sentence with a number: "X raises CIFAR-10-C accuracy above Ramen by more than the noise floor".
- Check `docs/research_state.md` and the legacy dead ends. Re-running a closed direction needs new evidence, and you say what it is.
- Write the **derivation first**: Ramen's update, then exactly which term the idea changes, and what that change should do to the update and the prediction (template in `references/notebook.md` §2). If you can't write the term that changes, the idea isn't ready.
- Choose the cheapest decisive test:
  - An **oracle or upper bound** (cheating with labels) tells you whether the direction can pay off at all.
  - **Offline replay** from saved records beats a GPU run whenever the quantity doesn't feed back into the model.
- Choose the **control**. Anything that removes, reweights or reorders data needs a matched control that does the same amount at random, or that makes only the side change. Example: an oracle that writes after prediction gets a control that also writes after prediction.
- **Pre-register** the gate in `docs/plans/<task>.md` before running: success threshold, stop rule, the datasets that are held out. Use the template in `lab.md` §8.

## 2. Where it goes

```
experiments/
  week_NN_<monday>/                       week_01_2026-11-02 = the week starting Monday 2026-11-02
    <YYYY-MM-DD>_<experiment>/            start date + snake_case name, e.g. 2026-11-04_oracle_filing
      <experiment>.py                     experiment-specific logic + a thin CLI (run / check / analyse)
      <experiment>.ipynb                  derivation, data, sanity checks, results, interpretation, glossary
      results/<run_id>/metrics.json       the citable record (committed)
      results/<run_id>/records.npz        per-sample outputs (git-ignored)
      results/logs/*.log                  run logs (git-ignored)
      figures/*.png                       optional exported figures (small, committable)
```

- The week number counts from `week_01_2026-11-02`: `week_NN = 1 + (monday − 2026-11-02) / 7 days`. An experiment lives in the week it **started**, even if it runs on.
- Reusable code (a new method variant, a new statistic, a new plot type) goes in `ramen_lab/`, with tests. Code that only this experiment needs stays in its `.py`. If a second experiment wants it, move it into `ramen_lab/` then, not before.
- Plans go in `docs/plans/<task>.md`, reviews in `docs/reviews/<date>_<task>.md`.

## 3. Implement as a variant of Ramen, and prove it is one

- Follow `references/code_style.md`. In short:
  - one responsibility per class;
  - new behaviour behind an interface plus a registry, never an `if method == ...` chain;
  - type hints and short docstrings that say *why*;
  - no copy-paste of upstream logic except where `code_style.md` allows it, and then with an identity test.
- Every new behaviour has a config key whose **default reproduces Ramen**.
- **Identity check first.** The "off" config must give predictions identical to upstream `Ramen` on a subset: 100.00%, not 99%. It is a test in `tests/`, and the runner's `check` step repeats it on real data. If it fails, the comparison would measure your bug.
- Oracles (anything that reads labels) must say **ORACLE** in the class docstring, the variant name and the notebook. Only classes that declare `uses_labels = True` receive labels.
- **Log what you'll need to explain the result** (what was retrieved, written or changed, with counts), streamed to the run's records. Re-running to get a missing log costs more than logging up front.
- Tests in `tests/` for every `ramen_lab` change:
  - identity with Ramen in the off setting;
  - reset clears state;
  - weights restored after each batch;
  - frozen parameters unchanged;
  - no label access (or a non-vacuous label test for oracles);
  - same seed gives the same output.

## 4. Run safely

- **Smoke run first.** Run `check` on a small subset (e.g. 200 per domain). It must pass the identity check, measure peak GPU and RAM, and print the time per batch. Set the run's RAM budget to about 1.5 × the measured peak RSS, and never more than `lab.md` §5's per-run cap.
- **Launch detached** through the launcher (`lab.md` §5). It:
  - pins one GPU;
  - caps DataLoader workers;
  - wraps the run in `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0` when linger is on;
  - writes the log to `results/logs/`.

  Never start a second run on a GPU that is already busy. Check `nvidia-smi` and `free -g` first. If the system's available RAM is below the reserve plus the run's budget, wait.
- Seeds:
  1. **Seed 0 on CIFAR-10-C** is the screen, against the pre-registered gate.
  2. Only ideas that pass get **seeds 0–4**.
  3. Held-out datasets (ImageNet-C, DomainNet) come last, with the setting unchanged.
- A gain is real only if it clears the **noise floor**: 2 × (max − min) of Ramen's mean accuracy over seeds 0–4, at the fixed batch, on that dataset. Also report the paired per-seed gains: mean ± std, and how many of 5 are positive. Until a dataset's floor is measured, say "floor unknown", and don't call anything real.
- While jobs run, write the notebook against the records API so it executes the moment results land. Don't poll logs in a loop.

## 5. Report: the notebook

Follow `references/notebook.md`. The parts that can't be skipped:

1. **Top cell:**
   - question and hypothesis with a number;
   - each variant and what it changes;
   - datasets (sizes), backbone, batch, seeds and what a seed changes;
   - the exact command;
   - what counts as a result.
2. **Mathematical derivation:**
   - Ramen's update written out;
   - the modified term, highlighted;
   - what the change predicts, and how it could fail.

   Every symbol maps to a `file:line` in `src/` or `ramen_lab/`.
3. **Data description:** images per domain, classes, and NoAdapt and Ramen error counts, before any result.
4. **Sanity checks:** the identity check, that the oracle or control did what it claims, and memory and time used.
5. **Results:** one finding per section, finding-style title, table plus one static matplotlib chart. Give counts and percentages together. Show the noise floor next to every gain.
6. **Interpretation** in plain words:
   - what the numbers say, and what they don't;
   - caveats (single seed, post-hoc choices);
   - the mechanism, as evidence or labelled `[hypothesis]`;
   - the next step and its stop rule.
7. **Glossary** for every non-obvious term and symbol.

Execute it top to bottom (`jupyter execute --inplace`) before handing it over. A notebook with stale outputs isn't finished.

## 6. After: review, record, hand over

- Code in `ramen_lab/` or `experiments/**/*.py` goes through the `gemini-review` skill. So does a notebook that reaches a conclusion: closing a direction, calling a gain real, setting a stop rule.
- Commit code, notebook and `metrics.json`. Never commit `records.npz`, logs, `.pkl`, `.npy` or `.pt` files.
- Update `docs/research_state.md` when a direction closes or opens, in a `shared:` commit, with conclusions and pointers but no numbers. Update the plan's status line.
- Summarize for the user:
  - headline numbers next to the noise floor;
  - what is ruled out;
  - resources used (peak RAM, peak GPU, minutes);
  - the next decision.
