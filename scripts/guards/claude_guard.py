"""PreToolUse guard for Claude Code: keep the agent from breaking anything outside the repo
(and from destroying irreplaceable, untracked results inside it). See AGENTS.md "Filesystem boundary".

Wired in .claude/settings.json for Bash, Write, Edit, NotebookEdit. Reads the hook JSON on stdin and
prints a permissionDecision ("deny" or "ask") or nothing (= normal permission flow).

Best-effort for Bash: it parses common commands (rm, mv, cp, redirects, git, pip, ...). It is a
seatbelt, not a sandbox; the rules in AGENTS.md still apply.
"""
import json
import os
import re
import shlex
import sys
from pathlib import Path

HOME = Path.home()


def writable_roots(project):
    sanitized = re.sub(r"[^A-Za-z0-9]", "-", str(project))
    return [
        project,
        Path("/tmp"),
        HOME / ".cache",
        HOME / ".claude" / "projects" / sanitized / "memory",
    ]


def inside(path, roots):
    return any(path == r or r in path.parents for r in roots)


def resolve(token, cwd):
    p = Path(os.path.expanduser(token))
    if not p.is_absolute():
        p = Path(cwd) / p
    return Path(os.path.normpath(p))


# Upstream Ramen (ef0fdc7) is frozen on the lab branches (D18): new code goes in ramen_lab/, never in these.
UPSTREAM_DIRS = ("src", "cfg", "shell")
UPSTREAM_FILES = ("README.md", ".gitignore")


def upstream(path, project):
    """True if `path` is an upstream Ramen file or directory that agents must not change."""
    try:
        rel = path.relative_to(project)
    except ValueError:
        return False
    parts = rel.parts
    return bool(parts) and (parts[0] in UPSTREAM_DIRS or (len(parts) == 1 and parts[0] in UPSTREAM_FILES))


# Untracked, expensive-to-regenerate outputs inside the repo.
PRECIOUS = re.compile(r"(\.(pkl|npy|pt|pth|ipynb)$|/results(/|$)|/log(/|$))")

DELETERS = {"rm", "rmdir", "shred", "truncate", "unlink"}
MODIFIERS = {"chmod", "chown", "chgrp"}
COPIERS = {"cp", "rsync", "install", "ln"}
WRAPPERS = {"nohup", "setsid", "time", "env", "nice", "ionice", "timeout", "stdbuf", "xargs"}
SEPARATORS = {";", "&&", "||", "|", "&", "(", ")", "|&"}


def split_commands(cmd):
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=";&|()<>")
    lex.whitespace_split = True
    lex.commenters = ""
    tokens = list(lex)
    cmds, cur = [], []
    for t in tokens:
        if t in SEPARATORS:
            if cur:
                cmds.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        cmds.append(cur)
    return cmds


def strip_prefix(words):
    i = 0
    while i < len(words):
        w = words[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w) or w in WRAPPERS:
            i += 1
        elif words[i - 1:i] == ["timeout"] or (i > 0 and words[i - 1] in {"nice", "ionice"} and w.startswith("-")):
            i += 1
        else:
            break
    return words[i:]


def path_args(args):
    return [a for a in args if not a.startswith("-") and a not in {">", ">>", "<"}]


def check_bash(cmd, cwd, roots, project):
    """Return (decision, reason) or None."""
    verdicts = []

    def flag(decision, reason):
        verdicts.append((decision, reason))

    try:
        cmds = split_commands(cmd)
    except ValueError:
        return ("ask", "guard could not parse this command")

    for words in cmds:
        # output redirections anywhere in the simple command
        for i, w in enumerate(words[:-1]):
            if w in {">", ">>"} or re.fullmatch(r"\d?>>?", w):
                tgt = words[i + 1]
                if tgt.startswith("&") or tgt.startswith("/dev/"):
                    continue
                if upstream(resolve(tgt, cwd), project):
                    flag("deny", f"redirect writes into upstream Ramen code: {tgt}")
                elif not inside(resolve(tgt, cwd), roots):
                    flag("ask", f"redirect writes outside the repo: {tgt}")
        words = [w for w in words if w not in {">", ">>", "<"}]
        words = strip_prefix(words)
        if not words:
            continue
        prog, args = os.path.basename(words[0]), words[1:]

        if prog in {"sudo", "su", "doas", "pkexec"}:
            flag("deny", "sudo/su is not allowed for agents")
            continue

        if prog in DELETERS or prog in MODIFIERS or prog == "mv":
            paths = path_args(args)
            if prog == "chmod" or prog in {"chown", "chgrp"}:
                paths = paths[1:]  # first arg is the mode/owner
            for p in paths:
                if any(c in p for c in "$`*?["):
                    if prog in DELETERS or prog == "mv":
                        flag("ask", f"{prog} with an unexpanded path/glob: {p}")
                    continue
                rp = resolve(p, cwd)
                if upstream(rp, project):
                    flag("deny", f"{prog} on upstream Ramen code: {rp}")
                elif not inside(rp, roots):
                    flag("deny", f"{prog} on a path outside the repo: {rp}")
                elif prog in DELETERS | {"mv"} and inside(rp, [project]):
                    if rp == project or rp.parent == project or PRECIOUS.search(str(rp)):
                        flag("ask", f"{prog} on repo-level or untracked results path: {rp}")

        elif prog in COPIERS or prog in {"tee", "dd"}:
            if prog == "dd":
                dests = [a[3:] for a in args if a.startswith("of=")]
            elif prog == "tee":
                dests = path_args(args)
            elif "-t" in args:
                dests = [args[args.index("-t") + 1]] if args.index("-t") + 1 < len(args) else []
            else:
                pa = path_args(args)
                dests = pa[-1:] if len(pa) >= 2 else []
            for d in dests:
                if upstream(resolve(d, cwd), project):
                    flag("deny", f"{prog} writes into upstream Ramen code: {d}")
                elif not inside(resolve(d, cwd), roots):
                    flag("ask", f"{prog} writes outside the repo: {d}")

        elif prog in {"sed", "perl"} and any(a == "-i" or a.startswith("-i") or a == "--in-place" for a in args):
            for a in path_args(args)[1:]:
                if upstream(resolve(a, cwd), project):
                    flag("deny", f"{prog} -i on upstream Ramen code: {a}")

        elif prog == "find" and ("-delete" in args or "-exec" in args and "rm" in args):
            roots_arg = [a for a in args if not a.startswith("-")][:1] or ["."]
            rp = resolve(roots_arg[0], cwd)
            flag("deny" if not inside(rp, roots) else "ask", f"find with delete under {rp}")

        elif prog == "git" and args:
            sub = args[0]
            rest = " ".join(args[1:])
            if sub == "push":
                if re.search(r"(^|\s)(-f|--force|--force-with-lease)\b", rest) or re.search(r"(^|[\s:+])(master|main)\b", rest):
                    flag("deny", "force-push or push to master/main is not allowed")
                else:
                    flag("ask", "git push needs the user's go-ahead")
            elif sub == "reset" and "--hard" in args:
                flag("ask", "git reset --hard discards work")
            elif sub == "clean":
                flag("ask", "git clean deletes untracked files (results are untracked)")
            elif sub in {"checkout", "restore"} and ("--" in args or "." in args) and "-b" not in args:
                flag("ask", f"git {sub} may discard uncommitted changes")
            elif sub == "stash" and len(args) > 1 and args[1] in {"drop", "clear"}:
                flag("ask", "git stash drop/clear deletes stashed work")
            elif sub == "branch" and "-D" in args:
                flag("ask", "git branch -D deletes a branch")
            elif sub in {"merge", "rebase"} and re.search(r"\b(master|main)\b", rest):
                flag("ask", f"git {sub} involving master/main")

        elif prog in {"pip", "pip3", "conda", "mamba", "apt", "apt-get", "npm", "snap"} or re.search(r"/(pip3?|python3?(\.\d+)?)$", words[0]) and args[:2] == ["-m", "pip"]:
            verbs = {"install", "uninstall", "remove", "upgrade", "update", "create"}
            if verbs & set(args):
                flag("ask", f"package/env change ({prog}) affects files outside the repo")

        elif prog in {"mkfs", "fdisk", "parted", "shutdown", "reboot", "kill", "pkill", "killall"}:
            if prog in {"kill", "pkill", "killall"}:
                flag("ask", f"{prog}: make sure these are our own experiment processes")
            else:
                flag("deny", f"{prog} is not allowed for agents")

    if not verdicts:
        return None
    for d, r in verdicts:
        if d == "deny":
            return d, r
    return verdicts[0]


def main():
    data = json.load(sys.stdin)
    tool = data.get("tool_name", "")
    tin = data.get("tool_input", {}) or {}
    cwd = data.get("cwd") or os.getcwd()
    project = Path(os.environ.get("CLAUDE_PROJECT_DIR") or cwd).resolve()
    roots = writable_roots(project)

    result = None
    if tool == "Bash":
        result = check_bash(tin.get("command", ""), cwd, roots, project)
    elif tool in {"Write", "Edit", "NotebookEdit", "MultiEdit"}:
        fp = tin.get("file_path") or tin.get("notebook_path") or ""
        if fp and upstream(resolve(fp, cwd), project):
            result = ("deny", f"{tool} on upstream Ramen code is not allowed (put new code in ramen_lab/): {fp}")
        elif fp and not inside(resolve(fp, cwd), roots):
            result = ("deny", f"{tool} outside the repo is not allowed: {fp}")

    if result:
        decision, reason = result
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": f"[claude_guard] {reason} (AGENTS.md: Filesystem boundary)",
        }}))


if __name__ == "__main__":
    main()
