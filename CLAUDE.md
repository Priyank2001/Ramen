@AGENTS.md

## Claude Code specifics

- You are the **implementer**. For env, GPUs, and how to launch jobs, read `docs/machines/<machine>.md` (match `hostname`).
- Use the `ramen-experiment` skill for experiments. For core-code tasks, its plan goes in `docs/plans/<task>.md`.
- When calling `scripts/review_with_gemini.sh`, pass only the task name and base. Never add your reasoning to the prompt.
- When the Stop-hook review gate fires, or you finish a change to `ramen_lab/` or `experiments/**/*.py`, use the `gemini-review` skill. If the work is unfinished or throwaway, say so in one line instead.
- Never edit upstream Ramen (`src/`, `cfg/`, `shell/`, `README.md`, `.gitignore`); build in `ramen_lab/`.
- After a review, answer every finding in `docs/reviews/<date>_<task>.md` with **Fix** or **Rebut + evidence**.
