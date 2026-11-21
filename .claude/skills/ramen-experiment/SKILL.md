---
name: ramen-experiment
description: Plan, run, and report machine-learning experiments the careful way, with the conventions of the Ramen test-time-adaptation (TTA) project as the worked case. Use this skill whenever the user wants to try an idea, run or launch an experiment, compare a variant against a baseline, do a seed sweep or ablation, check whether a gain is real, analyse cached runs, or write up results in a Jupyter notebook — e.g. "run the oracle variants on ImageNet-C", "does this beat Ramen?", "make a notebook for these runs", "launch seeds 1-4 on the GPUs", "is +0.3 significant?". Also use it for TTA, CLIP zero-shot, cache/memory-bank, corruption-benchmark (CIFAR-C, ImageNet-C), DomainNet or EuroSAT experiments even when the user doesn't say "experiment".
---

# Running experiments (Ramen conventions)

An experiment is only useful if someone else can trust the number and understand the notebook without you in the room. Most of what follows exists because a shortcut once cost hours or produced a wrong claim. The general workflow is in this file; everything specific to the Ramen repo (harness API, datasets, noise floors, known dead ends, gotchas) is in `references/ramen.md` — read it before touching that codebase. `references/code_design.md` gives the experiment-folder layout and how to extend the repo without editing it. `references/notebook.md` is the notebook template.

## 1. Before writing code: frame the question

- Write the hypothesis as one sentence with a number attached: "removing confidently wrong cache entries raises ImageNet-C accuracy above Ramen by more than the noise floor (0.54)".
- Check what has already been tried. In the Ramen repo, read the project memory / findings notes and `references/ramen.md` "Dead ends" first; re-running a known dead end wastes a GPU day.
- Prefer the cheapest decisive test. An **oracle / upper-bound** run (cheat with true labels) tells you whether a direction can pay off at all before you build the real method. **Offline replay** from cached logits/features (no GPU) beats a new GPU run whenever the quantity doesn't feed back into the model.
- Decide the **control**. A variant that removes or reweights data needs a matched random control (same amount removed, at random), or you can't tell "removing the bad ones" from "removing any".

## 2. Implement as a variant of the baseline, and prove it is one

**Shared code is read-only.** Never edit the repository's working code (in Ramen: `src/`, `cfg/`, `new_experiments/ramen_analysis.py` and the other shared helpers in `new_experiments/`). Other runs, cached results and older notebooks depend on it behaving exactly as it does now, and a small edit there can silently change results that are already reported. Everything new goes in the experiment's own dated folder:
- **Extend first**: subclass the method or harness class and override only what changes. Make new methods available through a registry in the experiment folder, not by editing the repo's registry (`src/methods/__init__.py`).
- **Copy only when subclassing can't do it** (e.g. the change sits in the middle of a long method with no hook): copy the file into the experiment folder under a new name, add a header comment giving the source path, the git commit it was copied from and what changed, and import the copy. The original stays untouched.
- If a shared file has a real bug, tell the user and propose the fix. Don't apply it yourself.

**Object-oriented, SOLID, easy to read.** One class per job (single responsibility). Add variants by adding classes or config entries, not `if` branches in existing code (open/closed). A variant must work anywhere the baseline works (Liskov). Keep interfaces small: a mixin for one feature, not a do-everything base. Depend on what you're given (pass the model, dataset and config in; don't build them inside the class). Each folder has a README table that gives every file and its job. Each module's docstring shows how to use it. Name things by what they mean (`ZeroShotWeightedSelector`, not `Sel2`). The concrete layout and the registry/harness pattern are in `references/code_design.md`.

- Subclass the baseline method; add behaviour behind a config switch whose default reproduces the baseline.
- **Sanity check first**: the "off" setting must give *identical* predictions to the cached baseline (100.00% same, not 99%). If it doesn't, the comparison measures your bug. Run this on a small subset before any full run.
- Anything that uses ground-truth labels must say so loudly (docstring, config comment, notebook) — it is an oracle, not a method.
- Log the per-sample or per-batch internals you'll need to *explain* the result (what was retrieved, its composition, counts), not only accuracy. Re-running to get a missing log costs more than logging up front.

## 3. Run: screening seed, then seeds

- **Seed 0 first** on every dataset as a screen. Only variants that look promising get seeds 1–4.
- A gain is real only if it clears the **noise floor**: 2 × (max − min) of the baseline's accuracy over seeds 0–4 (not the std; deliberately strict). Report the paired per-seed gains too (mean ± std, how many of 5 seeds positive). A t-test can call tiny gains "significant" because variant and baseline move together across seeds; the floor asks whether the gain is large compared with how much the baseline itself moves.
- Pre-register held-out settings: tune on one dataset, apply unchanged to the others, and say which numbers are post-hoc.
- Long GPU jobs: run **detached** (`setsid nohup … > logs/<name>.log 2>&1 &`) — jobs tied to the session die when it ends. One process per GPU unless you've measured memory; shard variants across GPUs. Chain sequential jobs inside one `bash -c "a; b"`. Wait on result files or log lines, not `pgrep -f` (it matches the waiter's own command line).
- While jobs run, write the notebook and analysis code against the cached-results API so it runs the moment results land.

## 4. Report: the notebook

Follow `references/notebook.md`. The non-negotiables, and why:

1. **Top cell explains everything**: question, hypothesis, what each variant does, datasets (sizes), models, configs, **seeds and what a seed changes**, how results were produced (script + command), what counts as a result. A reader opening the notebook cold must know what they're looking at.
2. **Mathematical formulation of the method**, in its own markdown cell right after the header, written in LaTeX. Start from the baseline's equations (for Ramen, copy the notation in `references/ramen.md` "Baseline formulation" so every notebook uses the same symbols), then give the variant as the exact equations that change, with every symbol defined. Mark any term that uses true labels as the oracle term. Map each symbol to its code (class, method, config key and the value used in these runs). A sentence like "drop confident-wrong entries" can mean several things; the equations say which one the code does, and they are what a reader checks the code against. Write this cell before the code, and update it if the code changes.
3. **Counts and percentages together**: "3,790 of 29,784 (12.7%)", never a bare percentage. Percentages hide the denominator; counts make small groups and shrinking supports visible.
4. **Describe the data the code ran on**: images per dataset/domain, classes, base error counts — before any result.
5. **Visualise with the right form**: effect vs noise floor → dot plot with a reference line; parts of a whole (fixed / broken / changed) → stacked horizontal bars; a quantity over the stream → line; before/after per item → dumbbell. One finding per chart, the finding in the title, counts in labels or hover.
6. **Interpretation cell** after the results, in plain words: what the numbers say, what they don't, single-seed caveats, next step.
7. **Glossary at the end** for every non-obvious term used.

## 5. After: record and hand over

- **Experiment README** (in the experiment folder, written once results are in) must contain, besides the file table and results table:
  1. **Math of each method**: the baseline equations (copy `references/ramen.md` notation), then one block per variant/run giving exactly what changes (target, value, filing, retrieval), every symbol defined, oracle terms marked, symbols mapped to config keys.
  2. **Why it makes sense or not, per method**: a short mechanism argument tied to the measured numbers (counts and percentages, target accuracy, fixed/broken), saying what the result confirms, what it contradicts, and what remains untested. Use a "why it should help / what actually happened" structure; never explain a result with a mechanism the logs did not measure, label it a hypothesis instead.
  3. A one-line verdict per method (works / null / hurts) against the noise floor.

- Update the findings notes (project memory) with the result, the numbers, and *why* it worked or didn't — especially dead ends.
- For an advisor/collaborator update, summarise: headline numbers vs noise floor, what's ruled out, next decision.

## Working in claude.ai (no repo, no GPU)

The same protocol applies; you just can't run things. Produce the variant code, the runner command, the sanity check, and the notebook skeleton for the user to run, and review their results against sections 3–4 (noise floor, counts with percentages, controls, seeds).
