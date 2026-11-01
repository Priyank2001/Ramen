#!/usr/bin/env bash
# Send a scoped diff (+ papers, + conclusion notebooks) to Gemini for an independent review.
# The prompt is FIXED here on purpose: the caller passes only names and paths, never reasoning (D5).
#
# Usage:
#   scripts/review_with_gemini.sh <task> [--base <ref>] [--notebook <nb.ipynb>]... [--dry-run]
#
#   --base      diff base; default: tag reviewed/<task>, else scripts/review_state.py base
#   --notebook  conclusion notebook to review (D12): its code and text outputs (no images) are sent
#   --dry-run   build the input file and print its size, don't call Gemini or tag
#
# Output: docs/reviews/<date>_<task>.md (Gemini findings + an empty Responses section for Claude). A second
# run of the same task on the same day appends a "Round N" section instead of overwriting.
# After a successful review, HEAD is tagged reviewed/<task> (local only; D11).
set -euo pipefail

TASK=""; BASE=""; NOTEBOOKS=(); DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --notebook) NOTEBOOKS+=("$2"); shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) TASK="$1"; shift ;;
  esac
done
[[ -n "$TASK" ]] || { echo "usage: $0 <task> [--base ref] [--notebook nb.ipynb]... [--dry-run]" >&2; exit 2; }

ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
SCOPE=(ramen_lab experiments tests scripts)

if [[ -z "$BASE" ]]; then
  if git rev-parse -q --verify "refs/tags/reviewed/$TASK" >/dev/null; then BASE="reviewed/$TASK"
  else BASE="$(python3 scripts/review_state.py base)"; fi  # newest reviewed/* ancestor, else machine branch
fi

# Untracked or uncommitted files in scope would be silently missing from the diff.
DIRTY="$(git status --porcelain -- "${SCOPE[@]}" | grep -v '^??' || true)"
UNTRACKED="$(git ls-files --others --exclude-standard -- "${SCOPE[@]}" | grep -E '\.(py|sh|yaml|yml|json)$' || true)"
if [[ -n "$DIRTY" ]]; then
  echo "WARNING: uncommitted changes in scope are NOT reviewed:"; echo "$DIRTY" | head -20
fi
if [[ -n "$UNTRACKED" ]]; then
  echo "WARNING: untracked code files in scope are NOT reviewed:"; echo "$UNTRACKED" | head -20
fi

DATE="$(date +%F)"
OUT="docs/reviews/${DATE}_${TASK}.md"
mkdir -p docs/reviews
INPUT="$(mktemp)"; trap 'rm -f "$INPUT"' EXIT

{
  echo "# Review input for task: $TASK"
  echo "Diff: $BASE...HEAD ($(git rev-parse --short HEAD)), scope: ${SCOPE[*]}"
  echo
  echo "## Diff (code; notebooks excluded)"
  echo '```diff'
  git diff "$BASE"...HEAD -- "${SCOPE[@]}" ':(exclude)*.ipynb' ':(exclude)*.log'
  echo '```'
  for nb in "${NOTEBOOKS[@]}"; do
    echo
    echo "## Conclusion notebook: $nb (code + text outputs)"
    python3 scripts/nb_text.py "$nb"
  done
  for p in docs/papers/*.md; do
    [[ -e "$p" ]] || continue
    echo; echo "## Paper: $p"; cat "$p"
  done
} > "$INPUT"

echo "input: $(wc -c < "$INPUT") bytes, base=$BASE, notebooks=${#NOTEBOOKS[@]}"
[[ $DRY -eq 1 ]] && { cp "$INPUT" "docs/reviews/${DATE}_${TASK}.input.txt"; echo "dry run: input saved to docs/reviews/${DATE}_${TASK}.input.txt"; exit 0; }

PROMPT='You are the read-only reviewer described in GEMINI.md and AGENTS.md (section 7 is the TTA checklist).
The input (stdin) contains a git diff, optionally analysis notebooks with their text outputs, and the baseline paper(s).
Review ONLY what is in the input. Do not modify files or run commands.
1. For code: look for bugs, label leakage, state not reset between domains/seeds (weights, optimizer, BN stats, cache/memory bank), wrong parameters updated, train/eval mode errors, mismatches with the paper (cite section/equation), and missing tests for adaptation logic.
2. For notebooks: check whether every conclusion follows from the shown outputs: seed spread vs noise floor, tuning on the test stream, fair baseline comparison (same data/seeds/backbone/GPU), post-hoc vs pre-registered claims.
3. Output a markdown table: ID | severity (bug / risk / nit) | file:line or cell | claim | evidence | how to check.
Mark anything without concrete evidence as [unverified]. Do not quote numbers that are not in the input. If you find nothing, say so explicitly.'

command -v gemini >/dev/null || export PATH="$HOME/priyank/Tools/node/bin:$PATH"
FINDINGS="$(gemini --approval-mode plan -p "$PROMPT" < "$INPUT")"

# A later round of the same task on the same day is appended, so earlier findings and responses are kept.
ROUND=1
[[ -f "$OUT" ]] && ROUND=$(( $(grep -c '^## Gemini findings' "$OUT") + 1 ))
SUFFIX=""; [[ $ROUND -gt 1 ]] && SUFFIX=" (round $ROUND)"

{
  if [[ $ROUND -eq 1 ]]; then
    echo "# Review: $TASK ($DATE)"
  else
    echo; echo "---"; echo
    echo "# Round $ROUND"
  fi
  echo
  echo "- Diff: \`$BASE...$(git rev-parse --short HEAD)\`"
  [[ ${#NOTEBOOKS[@]} -gt 0 ]] && echo "- Notebooks: ${NOTEBOOKS[*]}"
  echo
  echo "## Gemini findings$SUFFIX"
  echo
  echo "$FINDINGS"
  echo
  echo "## Responses (Claude)$SUFFIX"
  echo
  echo "For each finding ID: **Fix**: what changed (commit), or **Rebut**: evidence (test output, metrics.json, paper section, file:line)."
} >> "$OUT"

git tag -f "reviewed/$TASK" HEAD >/dev/null
echo "review written to $OUT; tagged reviewed/$TASK at $(git rev-parse --short HEAD)"
