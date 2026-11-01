"""Tests for scripts/guards/claude_guard.py. Run: python -m pytest tests/test_claude_guard.py
(or plain `python tests/test_claude_guard.py` if pytest is not installed)."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / "scripts" / "guards" / "claude_guard.py"


def decide(tool, tool_input, cwd=REPO):
    env = dict(os.environ, CLAUDE_PROJECT_DIR=str(REPO))
    out = subprocess.run([sys.executable, str(GUARD)], input=json.dumps(
        {"tool_name": tool, "tool_input": tool_input, "cwd": str(cwd)}),
        capture_output=True, text=True, env=env, check=True).stdout.strip()
    return json.loads(out)["hookSpecificOutput"]["permissionDecision"] if out else "pass"


def bash(cmd, cwd=REPO):
    return decide("Bash", {"command": cmd}, cwd)


CASES = [
    # outside the repo: blocked
    ("rm -rf ../RAMEN_ENV", "deny"),
    ("rm -rf ~/priyank/Projects/Ramen_untracked_backup_2026-10-22", "deny"),
    ("cd /tmp && rm -rf /home/narendra/priyank", "deny"),
    ("sudo apt install socat", "deny"),
    ("mv ../RAMEN_ENV ../old_env", "deny"),
    ("chmod -R 777 /home/narendra", "deny"),
    ("find /home/narendra -name '*.pkl' -delete", "deny"),
    ("git push --force personal sec_server_dual_agent", "deny"),
    ("git push personal HEAD:master", "deny"),
    # needs the user's go-ahead
    ("git push personal sec_server_dual_agent", "ask"),
    ("echo hi > ~/.bashrc", "ask"),
    ("cp scripts/summarize_run.py /home/narendra/bin/", "ask"),
    ("../RAMEN_ENV/bin/pip install pytest", "ask"),
    ("~/priyank/Projects/RAMEN_ENV/bin/python -m pip install numpy", "ask"),
    ("rm new_experiments/experiments_till_2026-10-24/results/check/x.pkl", "ask"),
    ("rm -rf new_experiments", "ask"),
    ("git clean -fd", "ask"),
    ("git reset --hard HEAD~1", "ask"),
    ("rm $FOO/bar", "ask"),
    # upstream Ramen code is frozen (D18)
    ("echo x >> src/methods/Ramen.py", "deny"),
    ("sed -i 's/a/b/' cfg/CIFAR10C/Ramen.yaml", "deny"),
    ("cp ramen_lab/x.py src/methods/", "deny"),
    ("rm shell/run_ramen.sh", "deny"),
    ("mv README.md README.old", "deny"),
    ("sed -i 's/a/b/' ramen_lab/runner.py", "pass"),
    ("cat src/methods/Ramen.py | head", "pass"),
    # normal work inside the repo: untouched
    ("rm docs/reviews/tmp.txt", "pass"),
    ("PYTHONNOUSERSITE=1 setsid nohup python run.py > new_experiments/x.log 2>&1 &", "pass"),
    ("ls ../RAMEN_ENV && cat ../Ramen_untracked_backup_2026-10-22/plan.md", "pass"),
    ("echo x > /tmp/claude-1001/scratch.txt", "pass"),
    ("git commit -m 'shared: x'", "pass"),
    ("git status && git diff | head", "pass"),
    ("cp ../Ramen_untracked_backup_2026-10-22/plan.md docs/", "pass"),
]


def test_bash_cases():
    bad = [(c, want, got) for c, want in CASES if (got := bash(c)) != want]
    assert not bad, "\n".join(f"{c!r}: want {w}, got {g}" for c, w, g in bad)


def test_file_tools():
    assert decide("Write", {"file_path": str(REPO / "docs" / "x.md")}) == "pass"
    assert decide("Edit", {"file_path": "/home/narendra/.bashrc"}) == "deny"
    assert decide("Write", {"file_path": str(REPO.parent / "RAMEN_ENV" / "x.py")}) == "deny"
    assert decide("NotebookEdit", {"notebook_path": str(REPO / "a.ipynb")}) == "pass"
    assert decide("Edit", {"file_path": str(REPO / "src" / "methods" / "Ramen.py")}) == "deny"
    assert decide("Write", {"file_path": str(REPO / "src" / "methods" / "New.py")}) == "deny"
    assert decide("Write", {"file_path": str(REPO / ".gitignore")}) == "deny"
    assert decide("Write", {"file_path": str(REPO / "ramen_lab" / "runner.py")}) == "pass"
    mem = Path.home() / ".claude/projects/-home-narendra-priyank-Projects-Ramen/memory/x.md"
    if str(REPO) == "/home/narendra/priyank/Projects/Ramen":
        assert decide("Write", {"file_path": str(mem)}) == "pass"


if __name__ == "__main__":
    test_bash_cases()
    test_file_tools()
    print("ok")
