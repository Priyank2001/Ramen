#!/usr/bin/env bash
# Serve the confusion-matrix explorer on localhost (no live reload, unlike VS Code's Live Server, which reloads
# every time a running experiment writes its log).
#
#     new_experiments/confusion_explorer/serve.sh            # port 8765
#     new_experiments/confusion_explorer/serve.sh 8800       # another port
#     new_experiments/confusion_explorer/serve.sh --build    # rebuild data/ from new .pkl results first
#
# Then forward the port (VS Code "Ports" tab, or `ssh -L 8765:localhost:8765 <server>` from the laptop) and open
# http://localhost:8765. Ctrl+C stops it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/miniconda3/envs/RAMEN/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

PORT=8765
BUILD=0
for arg in "$@"; do
  case "$arg" in
    --build) BUILD=1 ;;
    ''|*[!0-9]*) echo "usage: $0 [PORT] [--build]" >&2; exit 1 ;;
    *) PORT="$arg" ;;
  esac
done

if (( BUILD )); then
  "$PY" "$HERE/build_data.py"
fi

if [[ ! -f "$HERE/data/manifest.js" ]]; then
  echo "No data/manifest.js -- run with --build first." >&2
  exit 1
fi

if ss -ltn 2>/dev/null | grep -q ":$PORT\b"; then
  echo "Port $PORT is already in use; pass another one, e.g. $0 $((PORT + 1))" >&2
  exit 1
fi

echo "Serving $HERE at http://localhost:$PORT  (Ctrl+C to stop)"
cd "$HERE"
exec "$PY" -m http.server "$PORT" --bind 127.0.0.1
