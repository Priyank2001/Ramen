# AGENTS.md: Claude Code + Gemini CLI

Shared rules for both agents. Read this at the start of every session. `CLAUDE.md` and `GEMINI.md` only add role-specific bits on top.

## 1. Context

Experiments for my **test-time adaptation (TTA) thesis**: beat **Ramen** (mixed-domain TTA with CLIP). Nothing here is production code; the goal is fast, trustworthy experiments.

- Same rules on every machine. Work branch on PC2: **`pc2_lab`**, started 2026-11-01 from pure upstream Ramen `ef0fdc7` (D18). Earlier work stays on `sec_server_dual_agent` (old harness, batch 100) and `ramen_clean` (cache framework). Machine details (env, GPUs, launching) are in `docs/machines/<machine>.md`; use the one matching `hostname`.
- **Never touch `master`.** Don't merge into it, don't push to it, and never push anything without being asked.
- **Upstream Ramen is frozen.** `src/`, `cfg/`, `shell/`, `README.md` and `.gitignore` stay byte-identical to `ef0fdc7`, and nothing is added inside them. New code goes in `ramen_lab/`, which imports upstream read-only. Enforced by `tests/test_upstream_untouched.py` and the guard (§9).
- Baseline paper: *Ramen: Robust Test-Time Adaptation of Vision-Language Models with Active Sample Selection*, arXiv 2604.21728 → `docs/papers/`.
- Closed and open directions: `docs/research_state.md`. Their evidence lives on `sec_server_dual_agent`.

## 2. Roles

| | Claude Code | Gemini CLI |
|---|---|---|
| Job | Plan, implement, run tests/experiments, write notebooks | Read-only reviewer; bulk reader (whole papers, big diffs, long logs) |
| Context | Task-scoped: only the files and paper sections needed | Whole paper markdown + the diff |
| Writes code? | Yes | **No.** Reports findings only |

Gemini sees only the diff and the paper, never Claude's reasoning. Two models agreeing is weak evidence. **Disagreements are settled by running code or reading the paper text, not by asking another model.**

## 3. Project layout (where things go)

| Path | Purpose |
|---|---|
| `src/`, `cfg/`, `shell/` | Upstream Ramen, **frozen** (read-only, §1) |
| `ramen_lab/` | Our code ("core code"): runner, method variants, memory-safety guard, run records, plotting. OOP/SOLID, see the `ramen-experiment` skill |
| `experiments/week_NN_<monday>/` | One folder per week; `week_01_2026-11-02` is the week starting Monday 2026-11-02 |
| `  <YYYY-MM-DD>_<experiment>/` | One folder per experiment, named by its start date |
| `    <experiment>.py` + `<experiment>.ipynb` | Experiment-specific logic and a thin CLI in the `.py`; the executed notebook imports it and holds the derivation, tables and figures |
| `    results/<run_id>/metrics.json` | **The citable record of a run** (schema in §5) |
| `    results/<run_id>/records/`, `*.log`, `*.npz` | Raw outputs. Not read directly, not committed |
| `tests/` | Tests for `ramen_lab/` (§4.5) and for the repo rules |
| `docs/papers/*.md` | Papers extracted with `marker` (git-ignored, third-party) |
| `docs/paper_to_code.md` | Paper section/equation → `file:function` |
| `docs/plans/<task>.md` | Plan per core-code task |
| `docs/reviews/<date>_<task>.md` | Gemini findings + Claude's fix/rebut per finding |
| `docs/GUIDE.md` | How to use this setup day to day (start here) |
| `docs/research_state.md` | Closed/open directions + facts any idea must explain (input to direction critiques) |
| `docs/decisions.md` | Why the setup is the way it is (D1, D2, …). Supersede, don't edit |
| `docs/machines/<machine>.md` | Per-machine env/GPU/launch notes |
| `docs/claims.md` | Claim → evidence (`metrics.json` / notebook) → status. **Maintained by the user** |
| `scripts/summarize_run.py` | Compresses a run/log to about 30 lines |
| `scripts/critique_direction.sh` | Asks Gemini whether a research direction is worth testing (no web) |
| `scripts/review_with_gemini.sh` | Sends a scoped diff + paper to Gemini with a fixed prompt |
| `scripts/review_state.py`, `scripts/guards/review_gate.py` | What code is still unreviewed; the hooks that trigger the automatic review (D16) |

If a file listed here doesn't exist yet, say so and propose a minimal version. Don't invent its contents.

## 4. Rules (anti-hallucination)

1. **Numbers come only from `metrics.json` files or executed notebook outputs** in `experiments/`. Never quote accuracy, deltas or hyperparameters from memory (including agent memory files, which only point to files). If no file backs a number, say it is unknown.
2. **Paper claims need a citation** (section/equation in `docs/papers/*.md`). Otherwise label them `[unverified]`.
3. **Don't guess APIs.** Check the installed version in `RAMEN_ENV` (`pip show`, `help()`, source) first.
4. **Reproduce the baseline before building on it** and record the gap to the reported number.
5. **Changes to `ramen_lab/` need tests** in `tests/`, for example:
   - weights / optimizer / cache / BN stats restored after reset
   - no label access inside the adaptation step
   - frozen parameters unchanged
   - same seed → same result
6. **Say when you're unsure.** "I don't know, here is how to check" beats a confident guess.
7. **Keep changes small and scoped.** No drive-by refactors.
8. **Tune only on held-out domains** (`--domains extra`), never on the test stream.
9. **Never exhaust the server's RAM** (D17). PC2 has 125 GB RAM and 2 GB swap, and an exhausted host freezes for everyone. Every run goes through `ramen_lab`'s memory guard, which has a per-run RAM budget, a system reserve, and a cap on DataLoader workers. At most one run per GPU. No pinned host-RAM gradient caches: choose a batch that fits the GPU instead.
10. **Batch size is fixed per dataset** (D18): the largest that fits the 11 GB GPU 1, frozen in `ramen_lab` config. A variant and its baseline always use the same batch, seed and GPU model. Old batch-100 numbers from `sec_server_dual_agent` are not comparable.

## 5. Run record: `metrics.json`

Every run writes one to `<experiment folder>/results/<run_id>/metrics.json` (`ramen_lab` does this; don't write it by hand):

```json
{
  "run_id": "pc2_DN_vitb32_ramen_probT3_s0",
  "date": "2026-10-24", "machine": "pc2", "gpu": "RTX 3080 Ti",
  "git_hash": "abc1234", "git_dirty": false, "backfilled": false,
  "script": "experiments/week_01_2026-11-02/2026-11-02_baseline/baseline.py",
  "config": {"dataset": "DomainNet", "backbone": "ViT-B/32", "batch_size": 24, "seed": 0, "...": "..."},
  "metrics": {"acc": 0.0, "per_domain": {"...": 0.0}},
  "resources": {"peak_rss_gb": 0.0, "peak_gpu_gb": 0.0, "minutes": 0.0},
  "baseline_run_id": "pc2_DN_vitb32_ramen_s0",
  "raw_outputs": ["results/....pkl"]
}
```

Notebooks read these files; they don't recompute numbers they cite. Run ids start with the machine prefix. On `pc2_lab` there are no backfilled baselines: each dataset's Ramen and NoAdapt baselines, seeds 0–4 at the fixed batch, are run first. Only compare per-sample predictions between runs on the same GPU model.

## 6. The loop

**Full loop for changes to `ramen_lab/`.** **Code review (steps 3-4) also covers experiment scripts** (`experiments/**/*.py`) and is enforced automatically: a Stop hook blocks Claude once when such code is unreviewed, and Claude then runs the `gemini-review` skill (D16). Off switch: `RAMEN_NO_AUTO_REVIEW=1` or a file `.claude/no_auto_review`. Analysis notebooks need the `metrics.json` + notebook. **Notebooks that reach a conclusion** (drop/close a direction, call a gain real, set a stop rule) also go through step 3, with the notebook's `.py`, text outputs, and cited `metrics.json` files as input.

1. **Plan**: `docs/plans/<task>.md`: goal, files to change, tests, pre-registered success/stop rule. The user may have Gemini critique it first.
2. **Implement**: code + tests → run tests → small smoke run.
3. **Review**: commit on the work branch (untracked files don't show in diffs), then Claude runs
   `scripts/review_with_gemini.sh <task> [base]` (base defaults to the tag `reviewed/<task>`, created after each review). The script uses a **fixed prompt template**: diff of `ramen_lab experiments tests scripts` + relevant paper. **No added reasoning or justification.** Output goes to `docs/reviews/<date>_<task>.md`.
4. **Respond**: for each finding in that file, either **Fix** (say what changed) or **Rebut with evidence** (test output, `metrics.json`, paper section, `file:line`). Don't accept a finding just because it was raised, and don't dismiss it just because you disagree.
5. **Log**: `metrics.json` + notebook. The user updates `docs/claims.md`.

## 7. TTA review checklist

- Label leakage: in sample selection, the cache/memory, hyperparameter tuning, or the update step
- State not reset between domains/episodes/seeds: weights, optimizer, BN stats, **cache / memory bank**
- Train/eval mode and BN momentum
- Which parameters are actually updated
- Batch size and stream-order effects (mixed vs. sequential domains, cold memory early in the stream)
- Online vs. episodic mismatch
- Hyperparameters tuned on the test stream instead of held-out domains
- Fair comparison: same data, seeds, backbone, and budget as the baseline run
- Seed noise: report the per-seed spread and compare against the noise floor for that dataset

## 8. Token discipline

- Don't read raw logs or `.pkl` files. Use `scripts/summarize_run.py` or `metrics.json`.
- Don't load whole papers. Use `docs/paper_to_code.md` plus the one relevant section. Gemini reads whole papers.
- Use subagents for codebase exploration.
- Suggest `/clear` between unrelated tasks.
- Keep this file short; put detail in `docs/`.

Changes to shared files (this file, `CLAUDE.md`, `GEMINI.md`, `docs/decisions.md`, `docs/GUIDE.md`, `docs/research_state.md`, `docs/machines/`, `scripts/`, `tests/`, `.claude/settings.json`, `.claude/skills/gemini-review/`, `.gemini/`) go in their own commit, message `shared: …`, so they can be cherry-picked onto the other machine's branch.

## 9. Filesystem boundary (every machine)

The **repo folder is the only place agents may change**. Everything outside it is read-only for agents.

| Where | Agents may |
|---|---|
| Repo root (`.../Ramen`, wherever it is checked out) | read, write; deletions limited as below |
| `/tmp`, `~/.cache` (model downloads), Claude's own memory dir | write |
| Python envs (`RAMEN_ENV`, `MARKER_ENV`), datasets, `../Ramen_untracked_backup_*`, other projects, `~` | **read only** |
| System (`/etc`, `sudo`, other users) | nothing |

- **Ask the user first** before: installing or removing packages in any env; writing anything outside the repo; `git push`, `reset --hard`, `clean`, `stash drop`, `branch -D`; deleting `results/`, `log/`, `*.pkl`, `*.npy`, `*.pt`, `*.ipynb` (untracked results can't be recovered from git); deleting a top-level repo folder; killing processes.
- **Never**: `sudo`; delete, move or chmod anything outside the repo; force-push; push to `master`.
- **Enforcement**:
  - Claude Code: `scripts/guards/claude_guard.py` runs before every Bash/Write/Edit (wired in `.claude/settings.json`; tests in `tests/test_claude_guard.py`). It blocks the "never" list and asks the user for the "ask first" list.
  - Gemini: `.gemini/settings.json` removes its write and shell tools.
  - The guard is a seatbelt, not a sandbox. These rules still apply where it can't parse a command.

## 10. Must not

- Touch, merge into, or push to `master`; push anything without being asked.
- Commit large artifacts (`.pkl`, `.npy`, `.pt`, logs). Only `metrics.json`, code, notebooks, and docs.
- Edit `AGENTS.md`, `CLAUDE.md`, or `GEMINI.md` unless the user asks.
- Claim an experiment "works" without a `metrics.json` (or executed notebook) backing it.

## 11. Status

- Strategy started: 2026-10-24. `pc2_lab` started 2026-11-01 (D17, D18).
- `pc2_lab` items:
  - [x] Branch from upstream `ef0fdc7`; tooling ported; review scope = `ramen_lab/` + `experiments/**/*.py`
  - [x] Upstream frozen: guard denies edits, and `tests/test_upstream_untouched.py` passes
  - [x] `ramen_lab` core (runner, memory guard, run records, launcher); tests `tests/test_lab_*.py`
  - [ ] Week 01: fixed batch per dataset, then Ramen + NoAdapt seeds 0–4 and the noise floor (CIFAR-10-C first)
  - [ ] `loginctl enable-linger` (user's call): enables the hard cgroup RAM cap for detached jobs (D17)
- Setup items carried over from `sec_server_dual_agent`:
  - [x] Gemini training-data use turned off (user, 2026-10-24)
  - [x] Ramen PDF in `docs/papers/pdf/`
  - [x] Gemini CLI 0.65.0 in `~/priyank/Tools/node/bin` (user). Verified it can't write files, run shell, use the web, or invoke agents. `tools.exclude` is deprecated in favour of the Policy Engine before Gemini 1.0, so migrate before upgrading.
  - [ ] `marker` in `../MARKER_ENV` (separate venv); extract Ramen → `docs/papers/ramen.md`
  - [ ] `docs/paper_to_code.md` (after extraction)
  - [x] `scripts/review_with_gemini.sh` (fixed prompt; `--dry-run` to inspect the input)
  - [x] `scripts/summarize_run.py` (logs, run dirs, metrics.json)
  - [x] `scripts/run_record.py`: shared `write_metrics()` / `domain_metrics()` / `find_all()`
  - [n/a on pc2_lab] `scripts/backfill_metrics.py` and the batch-100 backfilled baselines (they stay on `sec_server_dual_agent`)
  - [x] Filesystem guard + tests (`scripts/guards/`, `tests/test_claude_guard.py`)
  - [x] Automatic review gate + `gemini-review` skill (D16; `tests/test_review_gate.py`)
  - [ ] Invariant tests (reset / label-independence / frozen weights / determinism) for `ramen_lab` methods: part of the week 01 plan
  - [ ] Main server: branch `main_server_dual_agent`, fill `docs/machines/main_server.md`, add its hostname to `scripts/run_record.py:MACHINES`
