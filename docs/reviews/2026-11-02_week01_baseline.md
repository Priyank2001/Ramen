# Review: week01_baseline (2026-11-02)

- Diff: `ef0fdc7...5c19267`

## Gemini findings

ID | severity | file:line | claim | evidence | how to check
---|---|---|---|---|---
1 | bug | ramen_lab/runner.py:72 | Changes to `method_config` (the yaml defaults) are ignored when skipping already finished runs. | `stored` excludes `method_config` before comparing with `config.to_dict()`. | Modify a value in `cfg/CIFAR10C/Ramen.yaml` and run the same `RunConfig`; the runner will incorrectly skip the run.
2 | bug | tests/test_lab_runner.py:84 | Unit test asserts that Ramen weights are unchanged after adaptation batches. | `assert all(torch.equal(before[k], v) ...)` after `ramen(image)`. | If Ramen is a TTA method, weights MUST change during adaptation. If this test passes, no adaptation is actually occurring.
3 | risk | experiments/.../baseline.py:55 | `fill_memory` is tightly coupled to upstream internal state. | Directly reads/writes `.cache`, `.keys`, `.values`, `.entropies`, `.priorities`, `.size`. | If upstream changes its `PriorityCache` implementation, this script will crash or measure peak memory incorrectly.
4 | nit | ramen_lab/runner.py:72 | `stored` config comparison may fail due to type mismatches (JSON vs Python). | `stored != json.loads(json.dumps(config.to_dict()))`. | `config.to_dict()` includes tuples (for `overrides`), but JSON will have lists.
5 | nit | ramen_lab/records.py:40 | `probs.npy` (memmap) might not be flushed if a run is aborted by the `MemoryGuard`. | `save()` flushes, but the `MemoryExceeded` catch-block in `Runner.run` does not. | Check the content of `probs.npy` after an aborted run.
6 | nit | ramen_lab/resources.py:91 | `MemoryGuard` hard-stop might prevent clean-up or logging. | `on_hard_limit` calls `os._exit(75)`, which terminates the process immediately. | Ensure the hard limit is not hit; if it is, `aborted.json` will not be written.
7 | nit | ramen_lab/runner.py:112 | `RecordWriter.add` uses the current `batches` count which might drift if `_adapt` is called multiple times. | `self.batches` is incremented in `add`. | Ensure `RecordWriter` is fresh for every `_adapt` call if global batch indexing is expected. Currently, it's fresh per `Runner.run`.

## Responses (Claude)

For each finding ID: **Fix**: what changed (commit), or **Rebut**: evidence (test output, metrics.json, paper section, file:line).

- **1 (skip ignores `method_config`): Fix, `fd79ee7`.** `Runner.run` now rebuilds the expected config, including the
  merged upstream yaml (`Runner.method_config`), and refuses to skip on any difference. Test:
  `tests/test_lab_runner.py::test_finished_run_is_skipped_and_changed_config_refused` now also changes the yaml
  (PASS). Upstream yaml is frozen (`tests/test_upstream_untouched.py`), so this was a latent risk, not a wrong result.
- **2 (Ramen weights unchanged after adaptation): Rebut; test strengthened, `4e61128`.** Ramen is episodic per
  batch. `src/methods/Ramen.py:157-168`: `set_by_sample_grad`, then `step_and_zero_grad`, then predict with the
  adapted weights, then `reset_parameters`. So the weights *must* equal the base after every `forward`, and that is
  what the test checks. Gemini's concern (the check could pass with no adaptation at all) was fair, so the test now
  also asserts that every batch's output differs from the zero-shot output of the same weights (PASS). On real data:
  `results/pc2_C10_Ramen_b100_s0/metrics.json` acc 72.61 vs `pc2_C10_NoAdapt_b100_s0` 61.36.
- **3 (`fill_memory` coupled to upstream internals): Rebut.** Upstream is frozen at `ef0fdc7`. `tests/test_upstream_untouched.py`
  fails on any change to `src/`, so `PriorityCache`'s fields (`src/methods/Ramen.py:14-27`) can't change under this
  code. `fill_memory` is measurement-only (batch search), documented as reading upstream fields, and is not used by
  any run.
- **4 (tuple vs list in the skip comparison): Rebut (fixed in passing).** `config.to_dict()` turns `overrides` into
  a dict, and the comparison was already done after a JSON round trip (`json.loads(json.dumps(...))`, now
  `_as_json` in `ramen_lab/runner.py`), so tuples become lists on both sides. Evidence: the skip test passes, and
  `again.skipped` is True for an identical config.
- **5 (`probs.npy` not flushed on abort): Rebut.** An aborted run writes `aborted.json` and no `metrics.json`
  (`ramen_lab/runner.py`, the `MemoryExceeded` branch). Nothing reads its records:
  `ramen_lab.records.find_runs` and `baseline.load_runs` select only directories with `metrics.json`.
  `capture_probs` is also off by default.
- **6 (hard stop skips cleanup): Rebut, by design (D17).** The hard stop exists for when the main thread can't
  react in time. The host's responsiveness matters more than this run's cleanup. The reason is not lost: `_exit_now`
  (`ramen_lab/resources.py`) prints it to stderr, which the launcher redirects to `results/logs/<gpu>.log`, and the
  process exits with code 75. The clean path (budget exceeded, below 1.2 ×) writes `aborted.json`; tested in
  `test_guard_trip_stops_cleanly` (PASS).
- **7 (`batch` index drifts across `_adapt` calls): Rebut, intended.** A `RecordWriter` is created per run
  (`Runner.run`), so `batch` is the batch index within the whole run. In single mode it keeps counting across
  domains, so batches stay unique. Gemini's own note confirms this ("fresh per Runner.run").

Not raised by Gemini, found while responding: `metrics.json` recorded the git HEAD at the *end* of a run. HEAD moved
during the first baseline runs (launched at `ee5a729`, recorded `5c19267`). `git diff ee5a729 5c19267` touches only
the usage docstring in `baseline.py`, `git_dirty`'s definition and `experiments/.gitignore`, so no computation. Fixed
in `fd79ee7`/`4e61128`: provenance is captured at run start.

---

# Round 2

- Diff: `reviewed/week01_baseline...3888551`

## Gemini findings (round 2)

Building on the research into the `Ramen` implementation and the `Runner` logic, here is the review of the `week01_baseline` task. The review focuses on the TTA checklist (AGENTS.md §7) and the integrity of the new experiment harness.

### Review Findings

| ID | severity | file:line or cell | claim | evidence | how to check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | **risk** | `src/methods/Ramen.py:171` | Optimizer state (e.g. momentum) is not reset between batches, potentially violating episodic independence if `momentum > 0`. | `Ramen.py` calls `self.model.reset_parameters()` which restores weights but not optimizer buffers; `TTABase`'s `load_model_and_optimizer` is unused. | Add a test in `tests/test_lab_runner.py` that uses an optimizer with momentum and checks if `ramen.model.optimizer.state` is non-empty/non-zero after a batch. |
| **C1** | verified | `ramen_lab/runner.py:84` | Git state is captured at the start of a run, ensuring `metrics.json` reflects the code state even if HEAD moves during long runs. | `run_record.git_state()` is called before `self.build` and the adaptation loop. | [verified] |
| **C2** | verified | `tests/test_lab_runner.py:101-103` | Tests now explicitly verify that adaptation occurred by comparing against zero-shot output, preventing vacuous success of the restore check. | New `adapted.append(...)` and `assert all(adapted)` logic in `test_ramen_restores_weights_and_reset_empties_memory`. | [verified] |
| **C3** | verified | `ramen_lab/runner.py:79-82` | Run config integrity is enforced; changing the underlying YAML or overrides for an existing `run_id` triggers a refusal. | `rec["config"] != expected` check in `Runner.run` and the new test case in `tests/test_lab_runner.py:149-152`. | [verified] |
| **N1** | nit | `scripts/run_record.py:61-62` | `git_state()` is called twice per `write_metrics` call if `git_hash="auto"`. | In `write_metrics`, both `git_hash` and `git_dirty` call `git_state()` independently if using the "auto" default. | Consolidate to a single call if efficiency becomes an issue. |

### Technical Summary
The changes successfully implement the requirement for trustworthy, reproducible experiments by:
1. **Capturing Code State**: Moving git state capture to the start of the run (`runner.py:84`) ensures that if a user edits code while an experiment is running, the `metrics.json` remains a faithful record of the *starting* code.
2. **Strict Config Matching**: Including the full `method_config` (resolved from YAML + overrides) in the skip-check ensures that "invisible" changes to configuration files are detected, preventing accidental reuse of stale results.
3. **Improved Test Rigor**: The new check in `test_lab_runner.py` ensures that the "weight restoration" test is meaningful by first proving that the weights *did* deviate from their zero-shot initialization.

**Note on R1**: While the current `Ramen.yaml` uses `signsgd` with no momentum (stateless), the risk remains that switching to a stateful optimizer (like Adam or SGD with momentum) would leak information across batches, violating the episodic nature of the method. Using the `TTABase` state restoration tools would be a safer long-term approach.

## Responses (Claude) (round 2)

For each finding ID: **Fix**: what changed (commit), or **Rebut**: evidence (test output, metrics.json, paper section, file:line).

- **R1: Rebut + Fix (test only).** The premise doesn't hold for Ramen's config: `cfg/CIFAR10C/Ramen.yaml` sets `optimizer: signsgd`, `lr: 0.01` and no momentum, and `SignSGD.__init__` defaults `momentum=0.0` (`src/models/optimizer.py:30`), so its `step` creates no buffers. Upstream also states the requirement: `ModelForBySampleTTA.py:240` ("The optimizer has to be a state-free one"). Nothing leaks across batches in any run we have. Gemini's suggested test is a useful guard for a future variant that overrides `optimizer`, so `test_ramen_restores_weights_and_reset_empties_memory` now asserts `optimizer.state` is empty after the batches and `momentum == 0.0` (`tests/test_lab_runner.py`, 24 tests pass).
- **C1, C2, C3: verified** by Gemini; no action.
- **N1: Rebut.** `git_state()` is in `scripts/run_record.py`, called once per `write_metrics`; the double call only happens with `git_hash="auto"`, which `ramen_lab` never passes (it captures the state at run start, `runner.py:84`). It costs two `git` subprocess calls per run, so nothing to change.
