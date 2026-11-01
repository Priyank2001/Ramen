"""Upstream Ramen code is frozen on the lab branches (docs/decisions.md D18): every file of upstream commit
UPSTREAM_HEAD must be byte-identical in the working tree, and nothing new may be added under its code folders.
New code lives in ramen_lab/. Run: python3 tests/test_upstream_untouched.py (no GPU)."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM_HEAD = "ef0fdc7"
FROZEN = ["src", "cfg", "shell", "README.md", ".gitignore"]


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def test_upstream_files_unchanged():
    changed = git("diff", "--name-only", UPSTREAM_HEAD, "--", *FROZEN).split()
    assert not changed, f"upstream Ramen files differ from {UPSTREAM_HEAD}: {changed}"


def test_nothing_added_to_upstream_folders():
    added = git("ls-files", "--others", "--exclude-standard", "--", *FROZEN).split()
    assert not added, f"new files inside upstream folders (put them in ramen_lab/): {added}"


if __name__ == "__main__":
    test_upstream_files_unchanged()
    test_nothing_added_to_upstream_folders()
    print("ok")
