# LEGACY (sec_server_dual_agent, old harness, batch 100): priors only. Paths, APIs and numbers here do NOT apply on pc2_lab; use lab.md.

Facts as of 2026-10-23. Numbers here come from `new_experiments/*.md` and `seed_variance.ipynb`. If a newer findings note disagrees, trust the note and update this file.

## Contents
1. Machines and environment
2. Repo map
3. The harness (`ramen_analysis.py`)
4. Adding a variant
5. Datasets and baselines
6. Seeds and noise floors
7. Offline replay and statistics
8. Dead ends (don't re-run)
9. Open / promising leads
10. Gotchas
11. Where findings live

---

## 1. Machines and environment

| | secondary server (PC2, this machine) | main server |
|---|---|---|
| GPUs | 0 = RTX 3080 Ti 12 GB, 1 = RTX 2080 Ti 11 GB | RTX 3090s, 24 GB |
| python | `~/priyank/Projects/RAMEN_ENV/bin/python` (venv, torch 2.0.1+cu117, numpy 1.x); no `python` on PATH | `~/miniconda3/envs/RAMEN/bin/python` |
| git branch | `secondary_server` on remote `personal` | `main_server` |

- On PC2, always set `PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID` and pick a GPU with `CUDA_VISIBLE_DEVICES`.
- tmux and screen are not installed, so use `setsid nohup …`.
- Accuracy from different GPU models is not comparable at the noise-floor level. Compare a variant with a baseline run on the same GPU model.
- Data is under `~/data`: `corruption/{CIFAR-10-C, CIFAR-100-C, ImageNet-C}` (severity 5 only), `domainbed/domain_net`, `EuroSAT`.
- Run times on PC2:
  - CIFAR-10-C: about 6.5 min (3080 Ti);
  - CIFAR-100-C: about 14 min (3080 Ti) or 20 min (2080 Ti), with the host-RAM cache (see Gotchas).

  On the 3090s: ImageNet-C about 8–9 min per run, DomainNet about 36 min.

## 2. Repo map

- `src/methods/`: TTA methods. `Ramen.py` is the trimmed upstream version. The variants are `RamenActive`, `RamenQuality`, `RamenNSS`, `RamenAnchor`, `RamenLog`, `RamenFix` (`RamenClean`, `RamenNSGD`), `RamenCache` and `RamenPolicy`. All are registered in `src/methods/__init__.py` and looked up by name via `get_method_class`.
- `cfg/<Dataset>/<Algo>.yaml`: per-dataset configs, with `cfg/default/` as the fallback. Dataset names are `CIFAR10C`, `CIFAR100C`, `ImageNetC5K`, `DomainNet`, `EuroSAT`.
- `new_experiments/`:
  - `ramen_analysis.py` is the harness;
  - the `*.ipynb` files are analysis notebooks;
  - the `*.md` files are findings notes;
  - `results/` holds the cached per-sample runs (git-ignored);
  - `quick_checks/` holds offline one-off scripts;
  - dated subfolders such as `experiments_till_2026-10-22/` hold each new batch of work, with its own `results/`.

## 3. The harness (`new_experiments/ramen_analysis.py`)

```python
from ramen_analysis import make_experiment
exp = make_experiment('ImageNetC5K', tta_mode='mixed', seed=0, num_workers=8,
                      results_dir=...)        # default: new_experiments/results
exp.load()                                     # CLIP + datasets; prints sizes
exp.evaluate()                                 # runs/loads NoAdapt + Ramen, returns per-domain accuracy table
exp.run('RamenActive', name='active_prob_T3', cfg={'active_set': 'prob', 'temperature': 3.0})
```

- **`run(algo, name=None, cfg=None, capture_features=False, capture_probs=False, use_cache=True)`** returns one row per test sample. The columns are `batch, domain_idx, sample_idx, label, pred, conf, true_prob, entropy, position, correct, domain`. `cfg` is merged over the yaml. The result is cached under `name`.
- **The cache file** is `{dataset}_{model}_{mode}_n{n|all}_bs{bs}_s{seed}_{name}.pkl`, with `_feats.npy`, `_probs.npy` and `_stats.pkl` beside it. A cached file is loaded instead of re-running, so **change `name` whenever the code or config changes**, or pass `use_cache=False`.
- **What gets saved with a run:**
  - `method.stats`, if the method has it, goes to `exp.method_stats[name]` and `_stats.pkl`. It holds one array per batch, and its rows line up with the results frame.
  - `method.save_log(prefix)`, if defined, is called after the run.
  - Methods with `records_labels = True` get `forward(x, labels=...)`. That is how an oracle or diagnostic receives ground truth.
- **Helpers:** `exp.paired()` (Ramen joined with NoAdapt per sample, with the outcome fixed / broken / both right / both wrong), `summary_metrics(algo)` (the same metrics as the noise-floor table: avg, noise, other, fixed %, broken %, TV(pred, true), ECE), `accuracy_table()`, `plot_*` methods, `show_failures`, `pick_failure`.
- **Palette and rcParams** live in the harness: `BLUE` (Ramen), `ORANGE` (NoAdapt), `RED` (worse), `AQUA`, `MUTED`, `CATEGORICAL`, `_top_legend`. Import them so notebooks look consistent.
- **The stream:** every method gets a DataLoader with `shuffle=True` and generator seed = `seed`, so every method sees the identical order and comparisons are paired per sample. In `mixed` mode all corruptions are shuffled into one stream. In `single` mode there is one stream per corruption, with `reset()` in between.

## 4. Adding a variant

1. Subclass the closest method in `src/methods/NewThing.py`, register it in `src/methods/__init__.py`, and add `cfg/<Dataset>/NewThing.yaml`, copied from `Ramen.yaml` plus the new keys. The new key's default must be the value that reproduces Ramen (e.g. `quality_lambda: 0`, `active_set: all`).
2. Write the variant's math, the meaning of each config key and the contents of `self.stats` in the module docstring. `RamenActive.py` and `RamenQuality.py` are good examples.
3. **The identity check.** Run the "off" config on a subset and compare it with Ramen on the same subset:
   ```python
   exp = make_experiment('CIFAR10C', seed=0, n_per_domain=200, results_dir=SCRATCH).load()
   a = exp.run('Ramen'); b = exp.run('NewThing', name='newthing_off', cfg={'new_key': 0})
   same = (a.pred.values == b.pred.values).mean(); assert same == 1.0, same
   ```
   Then repeat on the full stream for the first dataset. RamenQuality was checked this way at λ = 0: 100% identical on the 75k ImageNet-C stream.
4. **Oracles:** set `records_labels = True`, take `labels` in `forward`, and put "ORACLE: uses true labels" in the class docstring and the run name (e.g. `oracle_drop_wrong`).

## 5. Datasets and baselines (mixed stream, batch 100, seed 0)

| dataset (`make_experiment` key) | model | stream | classes | NoAdapt | Ramen |
|---|---|---|---|---|---|
| CIFAR-10-C (`CIFAR10C`), sev. 5 | ViT-B/32 | 15 × 10,000 = 150,000 | 10 | 61.3 | 72.61 |
| CIFAR-100-C (`CIFAR100C`), sev. 5 | ViT-B/16 | 15 × 10,000 = 150,000 | 100 | 37.1 | 46.10 |
| ImageNet-C 5k subset (`ImageNetC5K`), sev. 5 | ViT-B/16 | 15 × 5,000 = 75,000 | 1000 | 27.29 | 33.08 |
| DomainNet (`DomainNet`) | ViT-B/32 | 6 domains, 596k total | 345 | 55.37 | 56.80 |
| EuroSAT (`EuroSAT`) | see cfg | — | 10 | see `eurosat_ramen.ipynb` | |

- ImageNet-C uses **ViT-B/16, not the paper's ViT-L/14**, because L/14 doesn't fit. Don't compare these numbers with the paper's ImageNet-C table.
- The ImageNet-C 5k subset is **class-imbalanced**: 0–180 images per class, 3 classes absent, TV from uniform 0.17. DomainNet classes have 878–2,799 images. Methods that assume uniform labels break here.
- DomainNet "accuracy" is the mean over its 6 domains. Weighted per image it is lower (Ramen 54.52%), because quickdraw is both the largest and the hardest domain. Say which one you report.
- Each base image appears under all 15 corruptions, so samples are **not independent**. Bootstrap by base image.

## 6. Seeds and noise floors

- **What a seed changes:** only the stream order (the DataLoader shuffle). Nothing in `methods/` or `models/` draws random numbers, and re-running a seed reproduces it exactly. NoAdapt is seed-independent; this was verified at 0.0000 pts.
- Even so, individual predictions churn between seeds: Ramen changes 6.8% of predictions on CIFAR-10-C and 18.2% on CIFAR-100-C. Per-sample analyses (flips, failure grids) are therefore partly seed-specific.
- **Noise floor = 2 × (max − min) of Ramen's average accuracy over seeds 0–4**, measured on the 3090s:

| dataset | Ramen range, seeds 0–4 | noise floor |
|---|---|---|
| CIFAR-10-C | 72.53–72.68 | **0.31** |
| CIFAR-100-C | 46.06–46.17 | **0.21** |
| ImageNet-C | 32.91–33.17 | **0.54** (some notes round it to 0.53) |
| DomainNet | — | **0.11** |
| EuroSAT | not measured | — |

- Per-corruption claims need that corruption's own range across seeds, which is much larger than the average's (see `seed_variance.ipynb`).
- Earlier rule from next_directions.md: a general fix must clear the floor on **both** CIFAR datasets.

## 7. Offline replay and statistics

- **Exactly replayable from one logged run:**
  - Ramen's cache stores *zero-shot* features and gradients at the base weights, and the weights reset after every sample, so Ramen's outputs never feed back into the stream. Any change applied to the *output* (prior correction, neighbour smoothing, gating between Ramen and NoAdapt) can therefore be evaluated exactly from cached `_probs.npy` and `_feats.npy`.
  - A change to the global step size can be replayed from a log of the aggregated gradient per sample (see `step_size_sweep.py`).
- **Approximately replayable:** active-set selection, as in the NSS replay, which matched real runs to about 1 pt but assumes topk = 1 and β = 0. Validate any replay against a real run before trusting it.
- **Single-seed significance:** use the paired cluster bootstrap over base images plus McNemar from `quick_checks/vqa_checks.py` (check 1). Report it next to every single-seed number. It does not replace the noise floor.
- **Upper bounds worth computing first:** the per-sample oracle that picks the better of Ramen and NoAdapt gives +5.7 (C10), +4.5 (C100), +3.8 (IN-C) and +1.7 (DN). No selection method can beat it.

## 8. Dead ends (don't re-run without new evidence)

| idea | result | why |
|---|---|---|
| Zero-shot anchor (RamenAnchor 1–4), keep zero-shot when confident | no variant beats Ramen; gate ≤ +0.01 on all 4 datasets | helpful and harmful flips have the same zero-shot margin profile; confidence AUROC 0.61–0.67 |
| Contamination-aware weighting (RamenClean) | −0.7 / −2.9 | support purity 48.8 → 76.6%, but effective support shrinks from 158 to 12 |
| Normalized SGD instead of SignSGD (RamenNSGD) | −3.8 / −5.8 | the fixed-size sign step matters |
| Output prior correction, λ = 2 fixed on CIFAR | +0.57 / +1.50 on CIFAR, but +0.23 IN-C, −0.53 DN held out | uniform-label assumption and fixed λ; a single global prior can't fit per-domain bias |
| Support set as a classifier | ≈ zero-shot on C10, worse on C100; ≤ +0.35 mixed in | must take equal numbers from far classes |
| "Unstable, so fall back to zero-shot" | loses | unstable images are just hard; zero-shot is worse on them |
| Prompt templates | no gain | Ramen's 7 templates are as good as any label-free choice |
| Near-synonym override | loses everywhere | Ramen fixes about 2× more near-synonym errors than it causes |
| MSCN / NSS quality as a gate on queries | AUROC 0.49–0.51 | image damage doesn't predict where Ramen helps |
| Quality "prior" regulariser (prefer natural-looking entries) | −0.35 to −2.1 | — |
| Strong quality matching (λ ≥ 1) | down to −0.9 at λ = 3 | one scalar merges distortion types |
| Separate memory for noise corruptions | single vs mixed within ±0.5 | — |
| Memory policy/gating, relabelling entries, domain-identity keys, spatial/SAM keys, cache-classifier logits, hyperparameter tuning | closed on PC2 (plan.md phases 0–5) | defaults are optimal on the dev stream; relabelling breaks prediction balance |
| Within-class or shrunk ZCA keys for C100 | doesn't fix C100 | E3 falsified the between-class-flattening explanation |
| `prob` temperature on CIFAR-100-C | no T clears 0.21 | zero-shot class weighting has nothing to add on 100 classes |
| Ask the neighbours (kNN output smoothing), held out | −0.09 IN-C, +0.19 DN at seed 0; below the +0.3 stop rule (2026-10-24) | zero-shot neighbours are same-domain, not same-class (~10% class purity on IN-C), so smoothing feeds the domain attractors |

## 9. Open / promising leads (as of 2026-10-23)

- **RamenActive `prob` T = 3:**
  - ImageNet-C: +0.62 ± 0.11, seeds 0–4, positive on every seed; the mean clears 0.54, but seeds 0 and 1 individually don't.
  - DomainNet: +0.24 ± 0.01, held out, clears 0.11.
  - DomainNet re-run on PC2 (2026-10-23, `experiments_till_2026-10-23/seed_sweep.ipynb`): +0.25 ± 0.03, every seed above the PC2 floor of 0.12. top-20 and top-100 give +0.15 each, with seed 0 below the floor: dropped. prob T=3 loses on the cold early stream (−0.25 in the first ~49k images) and on quickdraw (−0.08, from the squiggle attractor).
  - CIFAR-100-C: −0.15 ± 0.08.
- **NSS vectors on active-set selection:** +0.26 to +0.29 on ImageNet-C, positive on every seed, but **below 0.54**.
- **ZCA-whitened retrieval keys:** +0.67 on C10 (3 seeds), −0.18 on C100. Source-statistics ZCA gives +0.33 on C10.
- **Entry-weight scale bias:** exp(−H) lets brightness entries take 23% of the top votes (chance is 6.7%). z-scoring is untested (vqa_paper_directions.md, Direction 2).
- **Domain-ordered streams:** untested. This is where the fp16 priority bug should matter.

## 10. Gotchas

- **The fp16 priority overflow:** Ramen's cache priorities are fp16 and saturate after 65,520 samples, after which full queues stop accepting new entries. This is harmless on shuffled streams. Keep it in variants (fp16), or λ = 0 won't reproduce Ramen.
- **Ramen writes a batch into memory before retrieving.** The query is therefore in its own zero-shot class queue at distance 0. With topk = 1, that class always contributes the query's own gradient. Account for this in any "retrieved composition" statistic, and exclude self-matches when measuring purity.
- **CIFAR-100-C OOM on PC2:** the cache is 100 × 750 × 39,936 fp16 = 5.9 GB, which OOMs on the 12 GB card at the first batch. Use `experiments_till_2026-10-22/run_cifar100c.py`, whose `HostValueCache` keeps the values in pinned host RAM. It was verified bit-identical to stock and takes about 30 min.
- **`import methods.Ramen` gives the class**, because the package re-exports it. To patch the module, use `sys.modules['methods.Ramen']`. `Ramen.__init__` looks up `PriorityCache` in its module at build time, so patching it there works.
- **numpy pickles:** pickles from the main server use numpy 2 (`numpy._core`). The harness shims this for numpy 1.x on PC2, so import `ramen_analysis` before `pd.read_pickle` on raw results.
- **Cache-name collisions:** `run()` silently loads an old pickle with the same name. Bump the name when the code changes.
- **`*.pkl` and `*.npy` are git-ignored,** so results stay on the machine that made them. To move them between servers, use scp. Put the notebook and the `.py` in git, never the results.
- **GPU memory:** the README lists stock Ramen at 11.6 GB (B/32 C10), 14.5 GB (B/16 C100) and 11.6 GB (B/32 DomainNet). Variants that keep extra per-entry tensors (quality scores, labels, stats) add to this. Measure before putting two jobs on one card.
- **Long logs:** `log_queue` in RamenActive is large on big queues. Turn it off unless needed.
- **Execute notebooks** with `jupyter execute --inplace`, using RAMEN_ENV/bin on PATH, from the notebook's folder.

## 11. Where findings live

- `new_experiments/findings_and_directions.md` (the 4-benchmark summary)
- `next_directions.md`, `new_directions.md`
- `vqa_paper_directions.md`, `quality_active_set.md`, `nss_active_set.md`
- `src/ramen-*.md` (handoffs, older directions)
- The project memory files (`ramen-research-state`, `pc2-run-environment`)

After an experiment, add the result and the *why* to the relevant note or a new dated note, and update section 8 or 9 of this file if a lead closes or opens.
