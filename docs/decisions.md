# Decisions log: dual-agent setup

One entry per decision. Change a decision by adding a new entry that supersedes the old one; don't edit old entries.
Format: **ID: decision.** Why. *How to undo/change.*

## 2026-10-24

**D1: Work branch on PC2 is `sec_server_dual_agent`**, branched from `secondary_server`. `master` is never touched.
Why: keeps the experiment tooling off `master` and leaves the old `secondary_server` history unchanged.
*Change: create a new branch from it; don't rename it, because the remote already uses this name.*

**D2: Run records are `results/<run_id>/metrics.json` inside the dated `new_experiments/experiments_till_<date>/` folder.** There is no top-level `runs/`.
Why: matches the existing convention (results next to the notebook that analyses them).
*Change: a script can collect them later with `find new_experiments -name metrics.json`.*

**D3: Rule files are wired as `AGENTS.md` (shared rules) + thin `CLAUDE.md` / `GEMINI.md`** that import it.
Why: one source of truth; the role files stay small.

**D4: The full loop (plan → tests → Gemini review) is only for core-code changes** (`src/`, `cfg/`).
Why: most work here is offline analysis on cached runs, and a full loop per notebook costs more than it catches.

**D5: Claude triggers Gemini reviews only through `scripts/review_with_gemini.sh`**, which uses a fixed prompt template. Claude never adds its own reasoning to the prompt.
Why: keeps the review independent.

**D6: Papers are extracted with `marker`** into `docs/papers/*.md`. PDFs go in `docs/papers/pdf/`, and the whole folder is git-ignored (third-party content).
Baseline paper: *Ramen: Robust Test-Time Adaptation of Vision-Language Models with Active Sample Selection* (Bao, Zhao, Yang, He), arXiv [2604.21728](https://arxiv.org/abs/2604.21728).
`marker` gets its **own venv**, not `RAMEN_ENV`, so its torch requirement can't break the experiment environment.

**D7: Agent memory holds conclusions and file pointers, never result numbers.**
Why: rule §4.1 says numbers come only from files.

**D8: Gemini data use for training is turned off** in Gemini settings (done by the user, 2026-10-24).

**D9: The same setup runs on every machine, on a parallel branch `<machine>_dual_agent`.**
- PC2 (secondary server): `sec_server_dual_agent`. Main server: `main_server_dual_agent`, branched from `main_server`.
- **Shared files are identical on all branches:** `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `docs/decisions.md`, `docs/machines/*`, `scripts/`, `tests/`, `docs/paper_to_code.md`.
- **Machine-specific details** (env path, GPUs, how to launch jobs) live in `docs/machines/<machine>.md`. The agent picks the file that matches `hostname`.
- **Syncing:** a change to shared files goes in its own commit, with no experiment files and a message starting `shared:`. On the other machine: `git fetch personal && git cherry-pick <hash>`.
- **Run ids start with the machine** (`pc2_…`, `main_…`), and `metrics.json` records `machine` and `gpu`.

Why: one rulebook means no drift. The branches differ only in their experiment folders, so cherry-picks never conflict. Run ids never collide when results are compared across machines. The GPU field matters because PC2 and the main server's 3090 don't give identical predictions (fp16 kernels).
*Change: if cherry-picking gets tedious, move the shared files to a small `tooling` branch and merge it into both machine branches.*

**D10: Backfill `metrics.json` for the baseline runs only**, with `scripts/backfill_metrics.py`. It covers stock Ramen, NoAdapt, and prob T=3 on DomainNet and ImageNet-C, seeds 0–4, wherever the cached pickles exist.
- Backfilled records set `"backfilled": true`, `"git_hash": null` (the original commit is unknown) and `"source"` (the pickle path). The number is recomputed from the pickle, not copied from a notebook.
- Each machine backfills only its own pickles, because `.pkl` files aren't in git. Older one-off analyses are not backfilled; their notebooks remain the evidence.

Why: new runs need a citable `baseline_run_id`, and backfilling everything would cost a lot for little benefit.

**D11: Each reviewed commit is tagged `reviewed/<task>`.** The next review of that task diffs from that tag.
Why: each review then covers only the new changes. Tags are local unless pushed (push only when asked).

**D12: Gemini also reviews analysis notebooks that reach a conclusion.** This covers dropping or closing a direction, claiming a gain is real or above the noise floor, and choosing a pre-registered stop rule. The script sends the notebook's `.py` and cell outputs (no images) plus the cited `metrics.json` files. Exploratory notebooks with no conclusion are skipped.
Why: the costly mistakes in this project have been wrong conclusions (leakage, seed noise, tuning on the test set), not only bugs in core code.

**D13: Agents can only change files inside the repo; everything outside is read-only** (requested by the user, 2026-10-24). Enforced by a PreToolUse guard (`scripts/guards/claude_guard.py`, set to `onFailure: block`) and by Gemini's tool exclusions. Rules are in AGENTS.md §9.
Why not Claude Code's built-in Bash sandbox: PC2 has no `socat`, so the sandbox can't start. bubblewrap would also likely kill the detached `setsid` experiment jobs when the command returns.
*Change: once `socat` is installed and background jobs are confirmed to survive, `sandbox.enabled` could be layered on top. Turn the guard off by removing the hook from `.claude/settings.json`.*

**D14: Gemini runs read-only**, in two layers: `--approval-mode plan` on every review call, and `.gemini/settings.json` excluding `write_file`, `replace`, `run_shell_command`, `web_fetch`, `google_web_search` and `invoke_agent`. Tested 2026-10-24 on 0.65.0: asked to write a file and run `touch`, it had neither tool and nothing changed. Web tools are excluded so that review input (unpublished code) never ends up in search queries, and the review uses only the given input.

**D15: Gemini also critiques research directions** (`scripts/critique_direction.sh`), with **web search off** (the user's choice, 2026-10-24).
- **Input:** the proposal, `docs/research_state.md`, the thesis brief, the direction notes, every `metrics.json`, evidence notebooks and the paper. Not Claude's opinion of the idea.
- **Output:** seven fixed questions ending in a verdict. Claude checks its concrete claims against the input and answers each point.
- The verdict is advisory; a pre-registered test decides.

Why: the closed-directions file stops Gemini from re-proposing dead ends. With web search off, unpublished ideas never reach search queries, so related-work claims stay `[unverified]`.
First run (hubness): all concrete claims were grounded in the input.

**D16: The Gemini code review starts automatically** (requested by the user, 2026-10-24).
- **Trigger:** a Stop hook (`scripts/guards/review_gate.py stop`). When Claude ends a turn and there is unreviewed code in scope, the hook blocks once and tells Claude to run the `gemini-review` skill. Unreviewed means either commits since the newest `reviewed/*` tag that is an ancestor of HEAD (else since the machine branch), or files this session edited but did not commit (logged by a PostToolUse hook).
- **Scope:** `src/`, `cfg/`, and `new_experiments/**/*.py`. This widens §6, where review was only required for `src/` and `cfg/`. Notebooks, logs, docs, `scripts/` and `tests/` don't trigger it.
- **Findings:** Claude writes Fix or Rebut for each one. It fixes clear bugs itself; anything that changes the method, a hyperparameter, a result or a claim goes to the user. At most two review rounds per task.
- **Off switch:** `RAMEN_NO_AUTO_REVIEW=1`, or a git-ignored file `.claude/no_auto_review`. Unfinished or throwaway work: Claude says so and stops; the gate fires only once per stop.

Why the end of the turn and not every edit: Gemini sees only commits, and reviewing half-finished code wastes calls. A skill alone can't fire on an event; only a hook can.
Known gap: edits made through Bash (`sed`, heredocs) aren't logged, so they only trigger the gate once committed.

**D17: No run may exhaust the server's host RAM** (requested by the user, 2026-11-01). PC2 has 125 GB RAM and only 2 GB swap, so an exhausted host stops responding. Three layers:
- **Config:** no pinned host-RAM gradient caches (choose a batch that fits the GPU, D18); DataLoader workers capped (default 4); one run per GPU.
- **In-process memory guard** (`ramen_lab`): samples the RSS of the run and its DataLoader workers and the system's available memory every couple of seconds. Over the run's budget, or with the system below its reserve, it stops the run cleanly with a clear message. Peak RSS goes into `metrics.json`.
- **Hard cap:** detached runs are launched as `systemd-run --user --scope -p MemoryMax=<budget> -p MemorySwapMax=0`. The kernel then kills only that run, never the host (verified 2026-11-01: a 400 MB allocation under a 200 MB cap was killed, exit 137). With `Linger=no`, user scopes die when the last login session ends, so detached runs use this layer only after the user runs `loginctl enable-linger`; until then the in-process guard is the protection.

**D18: `pc2_lab` starts from pure upstream Ramen, and upstream stays frozen** (requested by the user, 2026-11-01).
- **Branch:** `pc2_lab` from `ef0fdc7`. The cache framework was saved on `ramen_clean` (`0a32e63`); the old harness and results stay on `sec_server_dual_agent`.
- **Frozen:** `src/`, `cfg/`, `shell/`, `README.md` and `.gitignore` stay byte-identical, with no new files inside them. The guard denies such edits, and `tests/test_upstream_untouched.py` checks it. Ignore rules live in nested `.gitignore` files (`experiments/`, `docs/`, `.claude/`). Local leftovers go in `.git/info/exclude`.
- **Our code:** a top-level package `ramen_lab/` with its own method registry and runner. It imports upstream modules read-only, so upstream's `get_method_class` and `main.py` are never edited.
- **Layout:** `experiments/week_NN_<monday>/<date>_<experiment>/{<experiment>.py, <experiment>.ipynb, results/}`. Week 01 starts Monday 2026-11-02.
- **Protocol:**
  - Batch size is fixed per dataset: the largest that fits the 11 GB GPU 1. It is not the paper's 100, so a baseline at that batch (Ramen and NoAdapt, seeds 0–4) is measured first, and it sets the noise floor.
  - Ideas are screened on CIFAR-10-C, seed 0, against a pre-registered gate. Only ideas that pass get seeds 0–4.
  - Notebooks carry the full derivation of what the experiment changes, and static matplotlib figures.
- **Review scope (D16):** now `ramen_lab/` and `experiments/**/*.py`. The fallback review base is `ef0fdc7`.
Why: experiments built on a modified upstream made it hard to tell our changes from Ramen's. A frozen upstream plus a separate package makes "is this still Ramen?" a test, not a question.

