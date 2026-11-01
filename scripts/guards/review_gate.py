"""Claude Code hooks for the automatic Gemini review (decision D16).

    review_gate.py log    PostToolUse on Write|Edit|NotebookEdit|MultiEdit: remember files this session edited
    review_gate.py stop   Stop: if experiment code is unreviewed, block once and tell Claude to run the
                          gemini-review skill

Off switch: RAMEN_NO_AUTO_REVIEW=1 in the environment, or a file .claude/no_auto_review (git-ignored);
RAMEN_NO_AUTO_REVIEW=0 forces it on even when the file exists.
The stop hook blocks at most once per stop (stop_hook_active), so a failed Gemini call can't loop.
Any internal error lets Claude stop normally: this hook must never wedge a session.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATE = REPO / ".claude" / "state"
sys.path.insert(0, str(REPO / "scripts"))


def session_log(session_id):
    return STATE / f"edited_{session_id or 'unknown'}.txt"


def disabled():
    """RAMEN_NO_AUTO_REVIEW=1/0 decides when set (so tests can force the gate on); else the off-switch file."""
    env = os.environ.get("RAMEN_NO_AUTO_REVIEW")
    if env is not None:
        return env == "1"
    return (REPO / ".claude" / "no_auto_review").exists()


def log_edit(data):
    from review_state import in_scope
    path = (data.get("tool_input") or {}).get("file_path") or (data.get("tool_input") or {}).get("notebook_path")
    if not path:
        return
    try:
        rel = Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return
    if in_scope(rel):
        STATE.mkdir(parents=True, exist_ok=True)
        with session_log(data.get("session_id")).open("a") as f:
            f.write(rel + "\n")


def stop(data):
    if disabled() or data.get("stop_hook_active"):
        return
    from review_state import state
    log = session_log(data.get("session_id"))
    edited = sorted(set(log.read_text().split())) if log.exists() else []
    s = state(edited)
    if not s["committed"] and not s["uncommitted"]:
        return
    lines = ["Automatic review gate (D16): experiment code has not been reviewed by Gemini."]
    if s["uncommitted"]:
        lines.append("Edited this session but not committed (Gemini only sees commits): " + ", ".join(s["uncommitted"]))
    if s["committed"]:
        lines.append(f"Committed since {s['base']}: " + ", ".join(s["committed"]))
    lines.append("Run the gemini-review skill now. If the work is unfinished or throwaway, don't review: "
                 "say so to the user in one line and stop (this gate fires once per stop).")
    print(json.dumps({"decision": "block", "reason": "\n".join(lines)}))


if __name__ == "__main__":
    try:
        data = json.load(sys.stdin)
        {"log": log_edit, "stop": stop}[sys.argv[1]](data)
    except Exception as e:  # never wedge the session
        print(f"review_gate: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(0)
