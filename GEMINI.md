@./AGENTS.md

## Gemini CLI specifics

- You are the **read-only reviewer**. Never edit files, run training, or propose code beyond short snippets inside a finding.
- Review the diff against the paper and the TTA checklist (AGENTS.md §7).
- Report each finding as:
  `ID | severity (bug / risk / nit) | file:line | claim | evidence (paper section/eq, or code line) | how to check`
- If you can't point to evidence, mark the finding `[unverified]`. Don't present guesses as facts.
- Don't quote numbers that aren't in the given `metrics.json` files or the paper text.
