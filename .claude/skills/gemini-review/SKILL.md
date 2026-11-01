---
name: gemini-review
description: Run the independent Gemini code review on unreviewed experiment code (src/, cfg/, new_experiments/**/*.py) and answer every finding with Fix or Rebut. Use when the Stop-hook review gate says code is unreviewed, after finishing a code change in those paths, or when the user says "review this", "send it to Gemini", "get a second opinion on the code".
---

# Gemini review (decision D16, AGENTS.md §6 steps 3-4)

Gemini is the read-only reviewer. You are the implementer. The review is only useful if Gemini sees the code
**without your reasoning**, so pass only the task name, base and paths. Never add explanations to its input.

## 1. Decide whether to review now

```bash
python3 scripts/review_state.py        # review base + unreviewed committed files in scope
```

- Work unfinished, or a throwaway debug edit: **don't review**. Tell the user in one line what is unreviewed and
  stop. The gate will ask again at a later stop.
- The gate also fires for commits you did not make (e.g. another session's work). Review them the same way, but
  for findings in code you didn't write, report them; don't fix them.

## 2. Make the code visible

1. If `src/` changed, run the invariant tests (AGENTS.md §11 / `docs/machines/<machine>.md` has the command).
   A failing test is fixed first; Gemini doesn't review broken code.
2. Commit the in-scope files on the work branch (a per-machine commit, not `shared:`). Stage only the files
   that belong to this change; never `git add -A` (another session may have untracked work here).

## 3. Run the review

```bash
scripts/review_with_gemini.sh <task> --base "$(python3 scripts/review_state.py base)" [--notebook <nb>]...
```

- `<task>`: the plan's slug if there is a `docs/plans/<task>.md`, else a short kebab-case name of the change
  (e.g. `hubness-csls`). Reuse the same name for later rounds of the same work.
- `--notebook`: add it only if this change produced a conclusion notebook (D12).
- If the script fails (quota, network), don't retry in a loop. Tell the user, and leave the code unreviewed.

## 4. Respond to every finding

Open `docs/reviews/<date>_<task>.md`. Under **Responses (Claude)** write one entry per finding ID:

- **Fix**: what changed + commit hash. Fix it yourself only if it's a clear bug with no effect on the method or on
  a reported number (wrong reset, typo'd key, missing seed, missing test).
- **Rebut**: concrete evidence: test output, `metrics.json`, paper section/equation, `file:line`. Disagreeing is
  not evidence.
- **Ask user**: anything that changes the method, a hyperparameter, a result or a claim, or that you can't settle
  with evidence. Don't act on it; list it in the report.

Check claims against the code before agreeing: Gemini can be confidently wrong, and `[unverified]` items need
your own check.

If you made fixes, run step 3 once more (same `<task>`, it diffs only the fixes). **Two rounds at most**; anything
still open after round 2 goes to the user.

## 5. Finish

1. Commit the review file (and fix commits) on the work branch.
2. Report to the user in a few lines: task name, review file link, counts (bugs / risks / nits), what you fixed
   (commits), what you rebutted (one line each), and the **Ask user** items as questions.
