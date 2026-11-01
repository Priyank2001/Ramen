"""What code has not been reviewed by Gemini yet (decision D16, AGENTS.md §6).

    python3 scripts/review_state.py          # human summary
    python3 scripts/review_state.py base     # print the diff base for the next review
    python3 scripts/review_state.py --json   # machine-readable (used by the hooks)

"Reviewed" = reachable from a `reviewed/<task>` tag. The base for the next review is the newest such tag that
is an ancestor of HEAD, else the merge-base with this machine's branch (MACHINE_BRANCH below).
Trigger scope (D16, D18): anything under ramen_lab/, plus *.py under experiments/.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Upstream Ramen head the lab branches start from (D18); the review base when no reviewed/* tag is an ancestor.
UPSTREAM_HEAD = "ef0fdc7"


def git(*args, check=True):
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def in_scope(path):
    p = path.replace("\\", "/")
    return p.startswith("ramen_lab/") or (p.startswith("experiments/") and p.endswith(".py"))


def review_base():
    head = git("rev-parse", "HEAD")
    best, best_n = None, None
    for tag in git("tag", "-l", "reviewed/*").split():
        if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", tag, head]).returncode != 0:
            continue
        n = int(git("rev-list", "--count", f"{tag}..{head}"))  # fewer commits since = newer
        if best_n is None or n < best_n:
            best, best_n = tag, n
    if best:
        return best
    return git("rev-parse", "--short", UPSTREAM_HEAD)


def unreviewed_commits_files(base):
    return [f for f in git("diff", "--name-only", f"{base}..HEAD").splitlines() if in_scope(f)]


def uncommitted_files(paths):
    """Of `paths` (repo-relative), those that differ from HEAD or are untracked."""
    paths = [p for p in paths if in_scope(p)]
    if not paths:
        return []
    out = git("status", "--porcelain", "--untracked-files=all", "--", *paths, check=False)
    return [line[3:] for line in out.splitlines() if line]


def state(session_files=()):
    base = review_base()
    return {"base": base, "committed": unreviewed_commits_files(base),
            "uncommitted": uncommitted_files(list(session_files))}


if __name__ == "__main__":
    s = state()
    if "base" in sys.argv[1:]:
        print(s["base"])
    elif "--json" in sys.argv[1:]:
        print(json.dumps(s))
    else:
        print(f"review base: {s['base']}")
        print("unreviewed committed files in scope:" if s["committed"] else "no unreviewed commits in scope")
        for f in s["committed"]:
            print(f"  {f}")
