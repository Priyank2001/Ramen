# 2026-10-23 — Cache contamination oracles

Does contamination of Ramen's memory (images filed under a wrong, often confidently wrong, zero-shot class) limit its accuracy? An oracle with access to the true labels hides contaminated entries, which gives an upper bound for any real contamination detector. Each oracle is paired with a random control that hides the same number of entries.

| File | What it is |
|---|---|
| `cache_contamination_oracles.ipynb` | The write-up: question, data counts, sanity check, results, mechanism plots, interpretation, glossary |
| `run_oracles.py` | Launches the runs (detached, one shard per GPU) and the `--smoke` subset check |
| `run_seed_sweep.sh` | Seeds 1–4 of the confident-wrong oracle and its control: a job queue with one worker per GPU |
| `oracle_analysis.py` | Count tables and plots used by the notebook (CPU, from cached runs) |
| `logs/` | Run logs |
| `skill_package/ramen-experiment.skill` | The experiment-protocol skill, packaged for upload to claude.ai |

Method: `src/methods/RamenOracle.py`; configs: `cfg/{ImageNetC5K,DomainNet}/RamenOracle.yaml`.
Per-sample results are cached in `new_experiments/results/` as `*_s0_oracle_<variant>.pkl` (+ `_stats.pkl`).
Seed 0 (screening, all variants) on ImageNet-C 5K and DomainNet; seeds 1–4 for the confident-wrong oracle and its control (`run_seed_sweep.sh`).
