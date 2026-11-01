"""
ramen_lab: our experiment code on top of a frozen upstream Ramen (docs/decisions.md D18).

Dependency direction: experiments/* -> ramen_lab -> ramen_lab.upstream -> src/ (read-only).
Modules: config (what to run), registry + methods (which method), stream (data order), resources (RAM safety, D17),
records (per-sample outputs + metrics.json), runner (one run), launch (detached, RAM-capped), analysis (stats, plots).
"""
