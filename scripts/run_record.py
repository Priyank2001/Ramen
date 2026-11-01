"""Shared run record: write/read `results/<run_id>/metrics.json` (AGENTS.md §5, decisions D2/D9/D10).

    import sys; sys.path.insert(0, '<repo>/scripts')
    from run_record import write_metrics, domain_metrics, machine_tag

    m = domain_metrics(df)                       # df: one row per sample with `correct` + `domain`
    write_metrics(HERE / 'results', f'{machine_tag()}_DN_vitb32_ramen_s0', __file__, config, m,
                  baseline_run_id=..., raw_outputs=[pkl_path])

The headline accuracy is `acc` = mean over domains of per-domain accuracy (macro). That's the number every
analysis notebook in this repo reports. `acc_micro` (over all samples) is stored next to it.
"""
import json
import os
import socket
import subprocess
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# hostname -> machine tag (run-id prefix). Add the main server's hostname here (docs/machines/).
MACHINES = {"test-Auk": "pc2"}


def machine_tag():
    return os.environ.get("RAMEN_MACHINE") or MACHINES.get(socket.gethostname(), socket.gethostname())


def gpu_name():
    """Name of the first visible GPU (respects CUDA_VISIBLE_DEVICES with PCI_BUS_ID order), or None."""
    idx = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", idx],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out.replace("NVIDIA GeForce ", "") or None
    except Exception:
        return None


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def domain_metrics(df, domain_col="domain"):
    """Per-domain accuracy (%), their mean (`acc`, the headline), and sample-level `acc_micro`."""
    per = df.groupby(domain_col, observed=True)["correct"].mean() * 100
    return {
        "acc": round(float(per.mean()), 4),
        "acc_micro": round(float(df["correct"].mean() * 100), 4),
        "per_domain": {str(k): round(float(v), 4) for k, v in per.items()},
        "n_samples": int(len(df)),
    }


def git_state():
    """(short HEAD hash, True if tracked files differ from HEAD). Untracked results don't make code dirty."""
    return _git("rev-parse", "--short", "HEAD"), bool(_git("status", "--porcelain", "--untracked-files=no"))


def write_metrics(results_dir, run_id, script, config, metrics, baseline_run_id=None, raw_outputs=(),
                  backfilled=False, gpu="auto", git_hash="auto", notes=None, extra=None, git_dirty=None):
    """Write <results_dir>/<run_id>/metrics.json and return its path. `extra`: more top-level sections (e.g.
    "resources"); they may not overwrite the schema's own keys. Pass git_hash and git_dirty explicitly to record the
    code a run STARTED from (HEAD can move while a long run is going); "auto" reads them now."""
    tag = machine_tag()
    if not run_id.startswith(tag + "_"):
        raise ValueError(f"run_id must start with the machine tag '{tag}_' (D9): {run_id}")
    out = Path(results_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    rec = {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "machine": tag,
        "gpu": gpu_name() if gpu == "auto" else gpu,
        "git_hash": git_state()[0] if git_hash == "auto" else git_hash,
        "git_dirty": git_state()[1] if git_hash == "auto" else git_dirty,
        "backfilled": backfilled,
        "script": os.path.relpath(os.path.abspath(script), REPO),
        "config": config,
        "metrics": metrics,
        "baseline_run_id": baseline_run_id,
        "raw_outputs": [os.path.relpath(os.path.abspath(p), REPO) for p in raw_outputs],
    }
    if notes:
        rec["notes"] = notes
    for key, value in (extra or {}).items():
        if key in rec:
            raise ValueError(f"extra key {key!r} would overwrite a metrics.json field")
        rec[key] = value
    path = out / "metrics.json"
    path.write_text(json.dumps(rec, indent=2, default=float) + "\n")
    return path


def read_metrics(path_or_dir):
    p = Path(path_or_dir)
    return json.loads((p / "metrics.json" if p.is_dir() else p).read_text())


def find_all(root=REPO / "experiments"):
    """Every metrics.json under root, keyed by run_id."""
    return {(r := read_metrics(p))["run_id"]: r for p in sorted(Path(root).rglob("metrics.json"))}


def table(root=REPO / "experiments"):
    """One line per recorded run: run_id, headline acc, per-domain accs (for agents' input)."""
    lines = ["run_id | acc | per_domain | baseline | machine/gpu"]
    for rid, r in find_all(root).items():
        m = r.get("metrics", {})
        per = ", ".join(f"{k}={v:.2f}" for k, v in (m.get("per_domain") or {}).items()
                        if isinstance(v, (int, float)))
        acc = m.get("acc")
        lines.append(f"{rid} | {acc if acc is None else round(acc, 3)} | {per} | {r.get('baseline_run_id')} | "
                     f"{r.get('machine')}/{r.get('gpu')}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
