# How to use the Claude + Gemini setup

A practical guide to the setup on branch `sec_server_dual_agent` (built 2026-10-24).
For the rules, see `AGENTS.md`; for why each choice was made, see `docs/decisions.md`.

---

## 1. The idea in one picture

```
        you
         │  ask for an experiment / change
         ▼
   Claude Code ── plans, writes code + tests, runs experiments, writes notebooks
         │  commits, then runs scripts/review_with_gemini.sh
         ▼
   Gemini CLI ── read-only: sees only the diff + paper (+ notebook), never Claude's reasoning
         │  findings → docs/reviews/<date>_<task>.md
         ▼
   Claude Code ── answers each finding: Fix (commit) or Rebut (with evidence)
         │
         ▼
   you ── read the review file, update docs/claims.md
```

The three things that keep it honest:

1. **Every number has a file.** Each run writes `results/<run_id>/metrics.json`, and only those files (or executed notebook outputs) may be quoted.
2. **The review is independent.** Gemini gets a fixed prompt and the raw diff. Disagreements are settled by running code or reading the paper, never by asking another model.
3. **Agents can't break things outside the repo.** A guard blocks or asks before risky commands (§7).

---

## 2. Starting a session

**Claude Code**: start it inside the repo as usual. It loads `CLAUDE.md`, which pulls in `AGENTS.md`, so it already knows the rules. Useful openers:

- "Read `AGENTS.md` §11 and tell me what's still open."
- "Run the hubness check on the cached DN runs, following the analysis workflow."

**Gemini CLI**: also start it inside the repo, so it picks up `GEMINI.md` and `.gemini/settings.json`:

```bash
cd ~/priyank/Projects/Ramen
gemini                       # interactive, read-only tools only
```

If `gemini` isn't found, open a new terminal (it's on PATH via `~/.bashrc`) or run `export PATH="$HOME/priyank/Tools/node/bin:$PATH"`.

---

## 3. Workflow A: an analysis / offline check (most days)

This is for questions answered from cached runs, like hubness, kNN or attractor analyses. **No plan file and no review are needed, unless the notebook reaches a conclusion** (see step 4).

1. Ask Claude, e.g. *"Test CSLS hubness correction offline on the DN seed-0 runs; pre-register a +0.3 stop rule."*
2. Claude creates `experiments/week_NN_<monday>/<today>_<name>/<name>.py` with the experiment logic (built on `ramen_lab`) and `<name>.ipynb`, which imports it and is executed in place.
3. Every run it computes gets a record via the shared helper:
   ```python
   import sys; sys.path.insert(0, str(REPO / 'scripts'))
   from run_record import write_metrics, domain_metrics, machine_tag
   m = domain_metrics(df)   # df: one row per sample, with `correct` and `domain`
   write_metrics(HERE / 'results', f'{machine_tag()}_DN_vitb32_csls_s0', __file__, config, m,
                 baseline_run_id='pc2_DN_vitb32_Ramen_s0', raw_outputs=[pkl_path])
   ```
4. **If the notebook drops a direction, calls a gain real, or sets a stop rule**, ask Claude to review it:
   *"Commit and run the Gemini review on `<name>.ipynb`."* (The command is in §5.)

The baseline records are already backfilled. Compare against these:

| run_id | what |
|---|---|
| `pc2_DN_vitb32_NoAdapt_s0` | DomainNet zero-shot |
| `pc2_DN_vitb32_Ramen_s0` … `_s4` | DomainNet Ramen, seeds 0–4 |
| `pc2_DN_vitb32_probT3_s0` … `_s4` | DomainNet prob T=3, seeds 0–4 |
| `pc2_INC_vitb16_NoAdapt_s0`, `pc2_INC_vitb16_Ramen_s0` | ImageNet-C seed 0 (seeds 1–4 live on the main server) |

---

## 4. Workflow B: changing core code (`ramen_lab/`; upstream `src/`, `cfg/` are frozen)

For a new method, a change to Ramen's update step, cache logic, reset, and so on.

1. **Plan.** Claude writes `docs/plans/<task>.md`: goal, files to change, tests, and the success/stop rule. Read it before saying go. Optionally have Gemini critique the plan against the paper:
   ```bash
   gemini --approval-mode plan -p "Critique this plan against the Ramen paper. Point out anything the paper contradicts." \
     < <(cat docs/plans/<task>.md docs/papers/ramen.md)
   ```
2. **Implement + test.** Claude writes the code and adds the new method to `METHODS` in `tests/test_ramen_invariants.py`, then runs:
   ```bash
   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
     ~/priyank/Projects/RAMEN_ENV/bin/python tests/test_ramen_invariants.py
   ```
   All tests must pass:
   - no label access
   - labels don't change predictions
   - reset clears the stream
   - same seed gives the same result
   - weights are restored after each batch

   Then a small smoke run.
3. **Commit, then review** (§5).
4. **Respond.** For every finding in `docs/reviews/<date>_<task>.md`, Claude writes **Fix** (what changed + commit) or **Rebut** (test output, `metrics.json`, paper section, `file:line`).
5. **Full run** → `metrics.json` → notebook.
6. **You** add the claim to `docs/claims.md`.

---

## 5. Running a Gemini review

```bash
scripts/review_with_gemini.sh <task>                         # code diff since the last review of <task>
scripts/review_with_gemini.sh <task> --notebook path/x.ipynb  # also review a conclusion notebook
scripts/review_with_gemini.sh <task> --base <commit>          # diff from a specific commit
scripts/review_with_gemini.sh <task> --dry-run                # build the input only; inspect it, no Gemini call
```

- **What Gemini gets:**
  - the diff of `ramen_lab experiments tests scripts` (notebooks and logs excluded);
  - each `--notebook`'s code and text outputs (no images);
  - every `docs/papers/*.md`.
- **What it doesn't get:** Claude's reasoning. The prompt is fixed inside the script.
- **Output:** `docs/reviews/<date>_<task>.md` (findings table + an empty "Responses" section). HEAD is tagged `reviewed/<task>`, so the next review of the same task only covers new changes; a second round on the same day is appended to the same file as "Round N".
- **It warns** when uncommitted or untracked code in scope would be missed. Commit first.

You can run it yourself or ask Claude to. Either way, read the review file. It's the only place where the two models disagree in writing.

### 5a. The automatic review (you usually don't run §5 by hand)

When Claude finishes a turn after changing `ramen_lab/` or an experiment `.py` under `experiments/`, a hook stops it once and has it run the **`gemini-review` skill**:

1. Run the tests if `ramen_lab/` changed, then commit the change on the work branch.
2. Call `review_with_gemini.sh` with only a task name and the base.
3. Write **Fix** or **Rebut** for each finding. Clear bugs are fixed and re-reviewed, for at most two rounds.
4. Report back to you. Anything that would change the method, a hyperparameter or a result is **asked**, not done.

- **Check what is unreviewed:** `python3 scripts/review_state.py`.
- **Turn it off** while debugging something throwaway: `touch .claude/no_auto_review` (delete the file to turn it back on), or start Claude with `RAMEN_NO_AUTO_REVIEW=1 claude`.
- Unfinished work: Claude says it's unreviewed in one line and stops. It is asked again at a later stop.
- Edits Claude makes through shell commands only count once they're committed.
- You can also say "review this" at any time to run the skill.

### 5b. Asking Gemini whether a direction is worth testing

Use this before spending GPU time on a new idea, or when you're unsure whether it repeats a closed direction.

```bash
scripts/critique_direction.sh <name> <proposal.md>                                  # a plan or idea file
scripts/critique_direction.sh <name> docs/directions.md#"Direction 4"               # one "## " section of a file
scripts/critique_direction.sh <name> <proposal> --notebook <evidence.ipynb>          # add evidence notebooks
scripts/critique_direction.sh <name> <proposal> --dry-run                            # inspect the input only
```

- **What Gemini gets:**
  - the proposal;
  - `docs/research_state.md` (closed directions + facts any idea must explain);
  - the thesis brief and the direction notes;
  - a table of every `metrics.json` (`python3 scripts/run_record.py`);
  - the evidence notebooks and the paper.

  It gets no web access (by your choice, D15), and not Claude's opinion of the idea.
- **What it answers:** mechanism, overlap with closed directions, what the known facts predict, the cheapest decisive test with a stop rule, confounds, related work (`[unverified]`), and a verdict (TEST / TEST AFTER CHANGES / SKIP).
- **Output:** `docs/reviews/<date>_direction_<name>.md`. Claude then checks its concrete claims against the input and answers each point with Agree or Rebut.
- **The verdict is advisory.** Two models liking an idea is weak evidence; the pre-registered test decides.
- **Keep `docs/research_state.md` current.** When a direction closes, add a row with the evidence file. Otherwise Gemini can't know it's closed.

---

## 6. Checking a number

```bash
python3 scripts/summarize_run.py <dated folder>/results/<run_id>/     # one run record, ~15 lines
python3 scripts/summarize_run.py <dated folder>/results/run_gpu0.log   # a 60k-line tqdm log → ~20 lines
```

Every run on this machine:

```python
import sys; sys.path.insert(0, 'scripts')
from run_record import find_all
runs = find_all()            # {run_id: record}
runs['pc2_DN_vitb32_probT3_s3']['metrics']['acc']
```

`acc` is the headline number: the **mean over domains** of per-domain accuracy, the same as every notebook. `acc_micro` (over samples) is stored next to it.

If Claude quotes a number without naming the file it came from, ask *"which metrics.json?"*. That is rule §4.1.

---

## 7. The safety guard (what gets blocked)

Before every shell command and file edit, Claude Code runs `scripts/guards/claude_guard.py`.

| Result | Examples |
|---|---|
| **Blocked** | `sudo`; `rm`/`mv`/`chmod` on anything outside the repo; `find … -delete` outside; force-push; push to `master`; editing files outside the repo |
| **Asks you first** | `git push`; `git reset --hard`, `git clean`, `git stash drop`, `git branch -D`; `pip/conda/npm install`; writing outside the repo with `cp`/`>`/`tee`; deleting `*.pkl`, `*.npy`, `*.ipynb`, `results/`, `log/`, or a top-level repo folder; `kill` |
| **Allowed** | everything else inside the repo, plus `/tmp`, `~/.cache`, and Claude's memory folder; reading anywhere |

- **When it asks:** read the reason in the prompt and approve only if you meant it.
- **To change what it allows:** edit the guard, add a case to `tests/test_claude_guard.py`, run `python3 tests/test_claude_guard.py`, and commit as `shared:`.
- **To switch it off temporarily:** remove the `hooks` block from `.claude/settings.json`, or run `/hooks` in Claude Code.

Gemini is read-only in two ways: `--approval-mode plan` on every review call, and `.gemini/settings.json` removes its write, shell, web and agent tools. In an interactive `gemini` session it can still read any file in the repo.

---

## 8. Two machines

| | PC2 (this machine, `test-Auk`) | Main server (`priyank1@10.64.18.51`) |
|---|---|---|
| Branch | `sec_server_dual_agent` | `main_server_dual_agent` (not created yet) |
| Run ids | `pc2_…` | `main_…` |
| Notes | `docs/machines/pc2.md` | `docs/machines/main_server.md` (fill in) |

**Shared files** must be identical on both branches: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `docs/decisions.md`, `docs/GUIDE.md`, `docs/machines/`, `scripts/`, `tests/`, `.claude/settings.json` and `.gemini/`. Changes to them go in their own commit starting with `shared:`, with no experiment files in it.

To bring shared changes across (after pushing, which needs your OK):

```bash
# on the other machine
git fetch personal
git log --oneline personal/sec_server_dual_agent --grep '^shared:'   # list shared commits
git cherry-pick <hash> [<hash> …]
```

**Setting up the main server** (one time):

1. `git fetch personal && git switch -c main_server_dual_agent personal/main_server`
2. Cherry-pick all `shared:` commits from `sec_server_dual_agent`.
3. Fill in `docs/machines/main_server.md`, add its hostname to `MACHINES` in `scripts/run_record.py`, and commit as `shared:`.
4. Install Gemini CLI there, check its auth (§9), then `python3 tests/test_claude_guard.py`.
5. Backfill that machine's baselines (for example IN-C seeds 1–4): add their pickles to `RUNS` in `scripts/backfill_metrics.py` and run it.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `gemini: command not found` | New terminal, or `export PATH="$HOME/priyank/Tools/node/bin:$PATH"` |
| Gemini warns "tools.exclude is deprecated" | Harmless on 0.65. Migrate to the Policy Engine before upgrading to Gemini 1.0 |
| Gemini auth/privacy | It currently uses an **API key**. Free-tier API prompts may be used by Google. Check billing in AI Studio, or switch to your Google login with `/auth` |
| A pickle fails with `numpy._core` | It was written with numpy 2. Load it with the system `python3` (numpy 2 in `~/.local`), not `RAMEN_ENV` |
| `cdist_cuda not implemented for 'Half'` | Batch and cache both ≤25 rows. Use batch > 25 (real runs use 100) |
| A test needs `brisque_svm_weights.pth` | `*.pth` is git-ignored; copy it from the other machine |
| Claude keeps getting stopped by the review gate | `python3 scripts/review_state.py` shows what's unreviewed. Review it, or `touch .claude/no_auto_review` for throwaway work |
| Review hooks don't seem to run | Open `/hooks` once, or restart Claude, so it reloads `.claude/settings.json` |
| The guard blocks something legitimate | Approve if it asks. If it's a hard block, ask Claude to explain it, then change the guard as in §7 |

---

## 10. Cheat sheet

```bash
# review
scripts/review_with_gemini.sh <task> [--notebook nb.ipynb] [--dry-run]
python3 scripts/review_state.py                 # what's still unreviewed (auto review: §5a)
touch .claude/no_auto_review                    # pause the auto review (rm to resume)
# is this direction worth testing?
scripts/critique_direction.sh <name> <proposal.md | file.md#Section> [--notebook nb.ipynb]
# numbers
python3 scripts/summarize_run.py <run dir | log | metrics.json>
# tests
python3 tests/test_claude_guard.py
python3 tests/test_review_gate.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 ~/priyank/Projects/RAMEN_ENV/bin/python tests/test_ramen_invariants.py
# ask Gemini something about a long file (read-only)
gemini --approval-mode plan -p "question" < some_long_file
```

| File | Who edits it |
|---|---|
| `AGENTS.md`, `CLAUDE.md`, `GEMINI.md` | only on your request |
| `docs/decisions.md` | add new D-entries; don't edit old ones |
| `docs/claims.md` | **you** (claim → `metrics.json`/notebook → status) |
| `docs/research_state.md` | you (agents may propose edits): closed/open directions, the input for direction critiques |
| `docs/plans/`, `docs/reviews/` | Claude (Gemini's findings are pasted in by the script) |
