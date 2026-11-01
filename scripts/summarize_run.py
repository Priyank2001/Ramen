"""Compress a run into ~30 lines so agents never read raw logs.

Usage:
    python scripts/summarize_run.py <run.log | results/<run_id>/ | metrics.json> [--lines 30]

Logs: tqdm progress bars (carriage-return spam) are collapsed to the final bar per prefix;
the header, the last traceback, and the tail of real output lines are kept.
"""
import argparse
import json
import re
from pathlib import Path

TQDM = re.compile(r"\d+%\|.*\|\s*\d+/\d+")


def flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from flatten(v, key + ".")
        else:
            yield key, v


def summarize_metrics(path, max_lines):
    m = json.loads(Path(path).read_text())
    out = [f"run_id: {m.get('run_id')}  machine: {m.get('machine')}  gpu: {m.get('gpu')}",
           f"git: {m.get('git_hash')} dirty={m.get('git_dirty')} backfilled={m.get('backfilled')}  "
           f"date: {m.get('date')}",
           f"script: {m.get('script')}  baseline: {m.get('baseline_run_id')}",
           "config: " + ", ".join(f"{k}={v}" for k, v in flatten(m.get("config", {})))]
    out += [f"  {k}: {v}" for k, v in flatten(m.get("metrics", {}))]
    return out[:max_lines]


def summarize_log(path, max_lines):
    text = Path(path).read_text(errors="replace")
    real, bars = [], {}
    for line in re.split(r"[\r\n]+", text):
        line = line.rstrip()
        if not line:
            continue
        if TQDM.search(line):
            bars[line.split(":")[0]] = line  # keep the last bar per prefix
        else:
            real.append(line)

    tb = []
    for i, line in enumerate(real):
        if line.startswith("Traceback"):
            tb = real[i:i + 15]
    errors = [l for l in real if re.search(r"(Error|Exception|Killed|OOM|nan)\b", l)]

    out = [f"log: {path}  ({len(text.splitlines())} raw lines, {len(real)} non-progress lines)"]
    out += ["--- head"] + real[:4]
    out += ["--- progress (last bar)"] + [b[-160:] for b in bars.values()]
    if tb:
        out += ["--- last traceback"] + tb
    elif errors:
        out += ["--- error-like lines"] + errors[-5:]
    budget = max(max_lines - len(out) - 1, 3)
    tail = [l for l in real[4:] if l not in tb][-budget:]
    out += ["--- tail"] + tail
    return [l[:300] for l in out[:max_lines]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--lines", type=int, default=30)
    args = ap.parse_args()
    p = Path(args.path)
    if p.is_dir():
        p = p / "metrics.json"
    lines = summarize_metrics(p, args.lines) if p.suffix == ".json" else summarize_log(p, args.lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
