"""Print a notebook's markdown, code and text outputs (no images), for agents that shouldn't read raw .ipynb JSON.

Usage: python3 scripts/nb_text.py <nb.ipynb> [--max-output 4000]
"""
import argparse
import json


def notebook_text(path, max_output=4000):
    nb = json.load(open(path))
    parts = []
    for i, c in enumerate(nb["cells"]):
        parts.append(f"\n### cell {i} [{c['cell_type']}]\n{''.join(c.get('source', ''))}")
        for o in c.get("outputs", []):
            txt = o.get("text") or o.get("data", {}).get("text/plain")
            if txt:
                parts.append("--- output\n" + "".join(txt)[:max_output])
    return "\n".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--max-output", type=int, default=4000)
    a = ap.parse_args()
    print(notebook_text(a.notebook, a.max_output))
