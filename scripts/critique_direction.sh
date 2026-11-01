#!/usr/bin/env bash
# Ask Gemini whether a research direction is worth testing (D15). Read-only, web search off.
# The prompt is FIXED here: the caller passes only names and paths, never Claude's opinion of the idea.
#
# Usage:
#   scripts/critique_direction.sh <name> <proposal file> [--notebook <evidence.ipynb>]... [--dry-run]
#
#   <proposal file>  the idea in your words: a plan, a section of docs/directions.md copied to a file, or a .py
#                    whose docstring states the hypothesis. A "## Section" of a .md can be given as file.md#Section
#   --notebook       evidence notebook(s) the proposal leans on (code + text outputs, no images)
#
# Gemini also gets: docs/research_state.md (closed directions + known facts), the thesis brief, the direction
# files, a table of every metrics.json, and the paper(s).
# Output: docs/reviews/<date>_direction_<name>.md, with a Responses section for Claude to fill in.
set -euo pipefail

NAME=""; PROPOSAL=""; NOTEBOOKS=(); DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --notebook) NOTEBOOKS+=("$2"); shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) if [[ -z "$NAME" ]]; then NAME="$1"; else PROPOSAL="$1"; fi; shift ;;
  esac
done
[[ -n "$NAME" && -n "$PROPOSAL" ]] || { echo "usage: $0 <name> <proposal file[#Section]> [--notebook nb]... [--dry-run]" >&2; exit 2; }

ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
DATE="$(date +%F)"
OUT="docs/reviews/${DATE}_direction_${NAME}.md"
mkdir -p docs/reviews
INPUT="$(mktemp)"; trap 'rm -f "$INPUT"' EXIT

section() {  # print one "## Heading" section of a markdown file (up to the next "## ")
  awk -v h="$2" 'index($0, "## ")==1 { on = (index($0, h) > 0) } on' "$1"
}

{
  echo "# Direction critique input: $NAME"
  echo
  echo "## A. The proposal ($PROPOSAL)"
  if [[ "$PROPOSAL" == *"#"* ]]; then section "${PROPOSAL%%#*}" "${PROPOSAL#*#}"; else cat "$PROPOSAL"; fi
  echo
  echo "## B. Research state: closed directions and known facts (docs/research_state.md)"
  cat docs/research_state.md
  echo
  echo "## C. Thesis brief (docs/thesis_brief.md)"
  cat docs/thesis_brief.md
  for f in docs/directions.md docs/findings_batch100.md; do
    [[ -e "$f" ]] || continue
    echo; echo "## D. Direction notes: $f"; cat "$f"
  done
  echo
  echo "## E. Every recorded run (metrics.json; acc = mean over domains, %)"
  python3 scripts/run_record.py
  for nb in "${NOTEBOOKS[@]}"; do
    echo; echo "## F. Evidence notebook: $nb"; python3 scripts/nb_text.py "$nb"
  done
  for p in docs/papers/*.md; do
    [[ -e "$p" ]] || continue
    echo; echo "## G. Paper: $p"; cat "$p"
  done
} > "$INPUT"

echo "input: $(wc -c < "$INPUT") bytes, proposal=$PROPOSAL, notebooks=${#NOTEBOOKS[@]}"
[[ $DRY -eq 1 ]] && { cp "$INPUT" "docs/reviews/${DATE}_direction_${NAME}.input.txt"; echo "dry run: input saved to docs/reviews/${DATE}_direction_${NAME}.input.txt"; exit 0; }

PROMPT='You are a skeptical senior researcher critiquing a test-time-adaptation research direction (GEMINI.md, AGENTS.md).
Section A of the input is the proposal; B-G are the project state, prior results and the baseline paper.
Use ONLY the input. You have no web access: anything about related work comes from your own memory and must be marked [unverified].
Never quote a number that is not in the input. Answer these, each under its own heading:
1. Mechanism: what exactly is supposed to produce the gain? Is it plausible given the paper (cite section/equation) and the results in E/F?
2. Overlap with closed directions (section B): which, and would it fail for the same reason? Name the row.
3. Known facts (section B, "must explain"): what do the attractor, cold-memory and domain-not-class-neighbour findings predict for this idea?
4. Cheapest decisive test: preferably offline on cached runs. Name the metric, the comparison run_id(s) from E, the held-out data, and a pre-registered stop rule.
5. Failure modes and confounds: label leakage, tuning on the test stream, seed noise vs the spread across seeds in E, GPU differences, cold start.
6. Related work that may already do this [unverified].
7. Verdict: TEST AS IS / TEST AFTER CHANGES (list them) / SKIP, with the main reason.
Be concrete and brief. Disagree with the proposal where the evidence does.'

command -v gemini >/dev/null || export PATH="$HOME/priyank/Tools/node/bin:$PATH"
FINDINGS="$(gemini --approval-mode plan -p "$PROMPT" < "$INPUT")"

{
  echo "# Direction critique: $NAME ($DATE)"
  echo
  echo "- Proposal: \`$PROPOSAL\`"
  [[ ${#NOTEBOOKS[@]} -gt 0 ]] && echo "- Evidence notebooks: ${NOTEBOOKS[*]}"
  echo "- Gemini had no web access; related-work claims are [unverified] until checked."
  echo
  echo "## Gemini critique"
  echo
  echo "$FINDINGS"
  echo
  echo "## Responses (Claude)"
  echo
  echo "Per numbered point: **Agree**: what changes in the plan, or **Rebut**: evidence (metrics.json, notebook cell, paper section, file:line)."
  echo "The verdict is advisory; the pre-registered test decides."
} > "$OUT"
echo "critique written to $OUT"
