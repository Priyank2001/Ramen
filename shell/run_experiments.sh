#!/usr/bin/env bash
#
# One-shot driver for src/experiments.py: activates the conda env, runs the
# experiment sweep + diagnostics + t-SNE inside a detached tmux session, logs
# everything, and builds the LaTeX report into a PDF when it finishes.
#
# Usage:
#   ./run_experiments.sh                 # full run in tmux session "ramen-exp"
#   ./run_experiments.sh --no-sweep      # any flags are passed through to experiments.py
#   ./run_experiments.sh --fg            # run in the foreground (no tmux)
#   ./run_experiments.sh attach          # attach to the running session
#   ./run_experiments.sh logs            # tail -f the run log
#
# Detach from tmux:  Ctrl-b then d       Kill session:  tmux kill-session -t ramen-exp

set -euo pipefail

SESSION="ramen-exp"
CONDA_ENV="RAMEN"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$REPO_DIR/src"
REPORT_DIR="$REPO_DIR/report"
LOG_DIR="$REPO_DIR/log"
RUN_LOG="$LOG_DIR/experiments_run_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$LOG_DIR" "$REPORT_DIR"

# --- sub-commands -------------------------------------------------------------
case "${1:-}" in
  attach) exec tmux attach -t "$SESSION" ;;
  logs)   exec tail -f "$(ls -t "$LOG_DIR"/experiments_run_*.txt | head -1)" ;;
esac

FOREGROUND=0
PYARGS=()
for arg in "$@"; do
  case "$arg" in
    --fg) FOREGROUND=1 ;;
    *)    PYARGS+=("$arg") ;;
  esac
done

# --- locate conda ------------------------------------------------------------
for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
  if [ -f "$c/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$c/etc/profile.d/conda.sh"
    break
  fi
done
command -v conda >/dev/null || { echo "conda not found" >&2; exit 1; }

# --- the actual work -------------------------------------------------------
run() {
  set -x
  conda activate "$CONDA_ENV"
  cd "$SRC_DIR"
  python experiments.py --out-dir "$REPORT_DIR" "${PYARGS[@]}"
  cd "$REPORT_DIR"
  if command -v pdflatex >/dev/null; then
    pdflatex -interaction=nonstopmode experiments_report.tex >build.log 2>&1 || {
      echo "pdflatex failed; see $REPORT_DIR/build.log"; exit 1; }
    echo "PDF: $REPORT_DIR/experiments_report.pdf"
  else
    echo "pdflatex not installed; report left as $REPORT_DIR/experiments_report.tex"
  fi
}

if [ "$FOREGROUND" = 1 ]; then
  run 2>&1 | tee "$RUN_LOG"
  exit 0
fi

command -v tmux >/dev/null || { echo "tmux not found (use --fg to run without it)" >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already running -> ./run_experiments.sh attach" >&2
  exit 1
fi

# re-invoke this script in --fg mode inside tmux so arg handling stays in one place
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
CMD="'$SELF' --fg"
for a in "${PYARGS[@]}"; do CMD="$CMD '$a'"; done
tmux new-session -d -s "$SESSION" \
  "bash -lc \"$CMD; echo; echo DONE. Press enter to close.; read\""

echo "Started in tmux session '$SESSION'."
echo "  watch:   ./run_experiments.sh attach     (detach: Ctrl-b d)"
echo "  logs:    ./run_experiments.sh logs"
echo "  log file: $RUN_LOG"
