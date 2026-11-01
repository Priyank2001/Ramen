"""Tests for the auto-review gate (scripts/guards/review_gate.py, D16). Run: python3 tests/test_review_gate.py

Uses a throwaway untracked file under experiments/ and a fake session id; both are removed afterwards.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "guards" / "review_gate.py"
sys.path.insert(0, str(REPO / "scripts"))
from review_state import in_scope  # noqa: E402

SID = "test_review_gate"
TMP = REPO / "experiments" / "_review_gate_test_tmp.py"
LOG = REPO / ".claude" / "state" / f"edited_{SID}.txt"


def run(mode, payload, **env):
    r = subprocess.run([sys.executable, str(GATE), mode], input=json.dumps(payload), capture_output=True,
                       text=True, env={**os.environ, **env})
    assert r.returncode == 0, r.stderr
    assert not r.stderr, r.stderr
    return r.stdout.strip()


def test_scope():
    assert in_scope("ramen_lab/methods/oracle.py") and in_scope("experiments/week_01_2026-11-02/2026-11-02_x/x.py")
    assert not in_scope("experiments/week_01_2026-11-02/2026-11-02_x/x.ipynb") and not in_scope("docs/GUIDE.md")
    assert not in_scope("src/methods/Ramen.py") and not in_scope("cfg/x.yaml")  # upstream: frozen, never edited
    assert not in_scope("scripts/run_record.py") and not in_scope("tests/test_ramen_invariants.py")


def test_gate():
    env = {"RAMEN_NO_AUTO_REVIEW": "0"}
    try:
        TMP.write_text("x = 1\n")
        # out-of-scope edit is not logged; in-scope edit is
        run("log", {"session_id": SID, "tool_input": {"file_path": str(REPO / "docs" / "GUIDE.md")}}, **env)
        assert not LOG.exists()
        run("log", {"session_id": SID, "tool_input": {"file_path": str(TMP)}}, **env)
        assert LOG.read_text().split() == [TMP.relative_to(REPO).as_posix()]

        out = json.loads(run("stop", {"session_id": SID, "stop_hook_active": False}, **env))
        assert out["decision"] == "block" and TMP.name in out["reason"], out
        # blocks at most once per stop, and the off switch works
        assert run("stop", {"session_id": SID, "stop_hook_active": True}, **env) == ""
        assert run("stop", {"session_id": SID, "stop_hook_active": False}, RAMEN_NO_AUTO_REVIEW="1") == ""
        # another session's edit log doesn't trigger this session
        assert "_review_gate_test_tmp" not in run("stop", {"session_id": "other", "stop_hook_active": False},
                                                  **env)
        # once the file is gone (committed or discarded) this session no longer blocks on it
        TMP.unlink()
        assert "_review_gate_test_tmp" not in run("stop", {"session_id": SID, "stop_hook_active": False}, **env)
        # malformed input never breaks the session
        r = subprocess.run([sys.executable, str(GATE), "stop"], input="not json", capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout == ""
    finally:
        TMP.unlink(missing_ok=True)
        LOG.unlink(missing_ok=True)


if __name__ == "__main__":
    for t in (test_scope, test_gate):
        t()
        print(f"PASS {t.__name__}")
