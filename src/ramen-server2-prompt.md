# Prompt for Claude on the second server: replicate the RamenExperiment harness

Paste everything below the line into Claude Code, started from the repo root on the new server.
Fill in the `<EXPERIMENT>` block at the end first.

---

You're continuing a thesis project on mixed-domain test-time adaptation (TTA) for CLIP. The goal is to beat
**Ramen** (per-sample LayerNorm adaptation driven by a retrieval memory). The main GPU server is full today,
so experiments move to this machine. Your first job is to set up the **same experiment harness** we use on
the main server, so that numbers from here can be compared directly with numbers from there. Only then run
the experiment described at the end.

Background docs in the repo (read the first two before you write any code):
`src/ramen-tta-thesis-brief.md`, `src/ramen-pc2-handoff.md`, `new_experiments/findings_and_directions.md`.

## 1. What the harness is

`new_experiments/ramen_analysis.py` defines `RamenExperiment` plus one subclass per dataset, and
`make_experiment(dataset, **kw)`. Every notebook and script in `new_experiments/` uses it like this:

```python
import sys; sys.path.insert(0, 'new_experiments')
from ramen_analysis import make_experiment
exp = make_experiment('CIFAR10C', tta_mode='mixed', seed=0).load()
exp.evaluate()                       # runs NoAdapt + Ramen, cached
exp.run('RamenActive', name='active_prob_T3', cfg={'active_set': 'prob', 'temperature': 3})
exp.accuracy_table(); exp.summary_metrics('Ramen'); exp.paired(); exp.gate_sweep()
```

**Step 0: find out what's already on this machine.** Run `git status`, then check whether
`new_experiments/ramen_analysis.py`, `src/methods/Ramen{Active,Anchor,Cache,Fix,Log,NSS,Policy,Quality}.py`,
`src/datasets/eurosat.py`, `src/models/iqa.py` and the `cfg/<Dataset>/Ramen*.yaml` files exist. Much of this is
**untracked in git** on the main server (branch `experiments`), so a plain clone won't have it.
- If the files were copied over (rsync/scp), **use them as they are**. Don't rewrite them. Go straight to Step 2.
- If they're missing, **stop and ask me to copy them** (`rsync -av main:~/Projects/Ramen/{new_experiments/*.py,src,cfg} .`).
  Rebuild from the spec in Section 1a only if I tell you copying isn't possible.

### 1a. Spec, for a rebuild (and to check a copied version against)

**Constructor** `RamenExperiment(model=None, tta_mode='mixed', batch_size=100, n_per_domain=None,
algos=('NoAdapt','Ramen'), seed=0, num_workers=4, severity=5, data_root='~/data', cfg_dir=<repo>/cfg,
results_dir=<repo>/new_experiments/results, device=None)`. It holds `results` (algo → per-sample DataFrame),
`features`, `probs` and `method_stats`, all as dicts keyed by run name.

**Class attributes per subclass:** `dataset_name`, `default_model`, `superclasses`, `watch_classes`,
`heatmap_by_superclass`, `env_label`.

| subclass | dataset_name | default_model | notes |
|---|---|---|---|
| CIFAR10CExperiment | CIFAR10C | clip_vitbase32 | superclasses vehicle/animal; watch deer, truck, frog |
| CIFAR100CExperiment | CIFAR100C | clip_vitbase16 | the 20 CIFAR-100 superclasses; heatmap by superclass |
| ImageNetC5KExperiment | ImageNetC5K | clip_vitbase16 | images on disk; subset via `.samples` |
| DomainNetExperiment | DomainNet | clip_vitbase32 | env_label='domain'; independent subset per domain |
| EuroSATExperiment | EuroSAT | clip_vitbase16 | one domain; full 10×10 confusion matrix |

**`load()`** seeds everything, calls `get_pretrained_model(SimpleNamespace(model=..., device=...))`, then builds
the datasets with `get_dataset_class(name)(root, severity, transform)`. With `n_per_domain`, the corruption
datasets keep **the same sorted random index subset in every corruption** (`np.random.default_rng(seed)`).
DomainNet draws its subset independently per domain.

**`build_method(algo, cfg)`**: config = `cfg/<dataset>/<algo>.yaml`, falling back to `cfg/default/<algo>.yaml`,
then updated with the `cfg` overrides. `args = SimpleNamespace(config=..., device, max_batch_size=batch_size,
use_tbn=False)`. The method gets a **`deepcopy` of the CLIP model**, because Ramen swaps its norm layers in place.

**`run(algo, use_cache=True, capture_features=False, capture_probs=False, name=None, cfg=None)`**, the core:
- Cache file: `results_dir/{dataset}_{model}_{tta_mode}_n{n_per_domain or 'all'}_bs{bs}_s{seed}_{name}.pkl`,
  with sidecar files `_feats.npy` (fp16), `_probs.npy` (fp32) and `_stats.pkl`. On a cache hit, load and return.
- Re-seed, then build the method. Wrap every domain dataset so it yields `(x, y, domain_idx, sample_idx)`.
  - `mixed` = one `ConcatDataset` stream.
  - `single` = one stream per domain, with `method.reset()` between streams.
- `DataLoader(shuffle=True, num_workers=self.num_workers, pin_memory=True,
  generator=torch.Generator().manual_seed(seed))`, **wrapped in `tqdm`**. This is what makes every method see
  the identical sample order.
- If `method.records_labels` is set, call `method(x, labels=y)`; otherwise call `method(x)`.
- One row per sample, with columns `batch, domain_idx, sample_idx, label, pred, conf, true_prob, entropy`, then
  `position, correct, domain` (categorical).
- To capture features, monkey-patch `method.model.featurize` and keep the **last** call in each batch.
- After the run: save `method.stats` if it exists, and call `method.save_log(prefix)` if that exists. Then
  `del method; gc.collect(); torch.cuda.empty_cache()`. The `gc.collect()` is required, because the method
  sits in a reference cycle that would otherwise keep Ramen's ~8 GB cache alive.

**Analysis methods:**
- `evaluate()` runs `self.algos` and returns `accuracy_table()`. The table has one row per domain plus an
  `average` row, ×100, and a `delta` column (Ramen − NoAdapt).
- `paired()` merges Ramen with NoAdapt on `(domain_idx, sample_idx)` and labels each sample fixed, broken,
  both right or both wrong.
- `gate_sweep` and `entropy_gate_sweep` are post-hoc "use zero-shot when confident" tables. Their attrs hold
  Ramen's accuracy and the oracle accuracy.
- `summary_metrics(algo)` returns avg, noise, other, fixed %, broken %, TV(pred, true), overshoot per watch
  class, ECE, and per-domain accuracy.
- The plot helpers (`plot_*`, `show_*`, `tsne`) are optional for a rebuild. Port them only if the experiment
  needs them.

**`src/` changes the harness depends on (also untracked or uncommitted on the main server):**
- `methods/__init__.py` imports every Ramen variant, so `get_method_class` can find it by name.
- `datasets/__init__.py` maps `'EuroSAT'` to `datasets/eurosat.py`.
- `models/clip_utils.py` has `DEFAULT_TEMPLATES`. Both model wrappers use `cfg.get('templates') or DEFAULT_TEMPLATES`.
- `main.py` passes `labels=` when `records_labels` is set.

## 2. Verify the harness before running anything new

1. Environment. The main server uses conda env `RAMEN` with torch 2.4.1+cu121, **tqdm 4.70.0**, numpy 2.2.6,
   pandas 2.3.3 and OpenAI `clip`. Match tqdm exactly. tqdm 4.70 calls `iter()` on the DataLoader when it is
   constructed, which consumes the seeded generator. A different tqdm version or a different `num_workers`
   gives a **different stream order**, and then per-sample results won't line up with the main server's.
   Keep `num_workers=4`.
2. Data lives under `~/data`. See the per-dataset paths in `src/ramen-pc2-handoff.md`. If they differ here,
   pass `data_root=` and don't edit the dataset code.
3. Smoke test: `n_per_domain=200` on CIFAR10C, NoAdapt + Ramen, in a scratch `results_dir`. Use
   `n_per_domain` values that give full batches of 100: torch `cdist` has no fp16 CUDA kernel for ≤25 queries.
4. Reproduce the seed-0, full-data, mixed baselines below. They should match to within ±0.02. If they don't,
   check the stream order (tqdm, num_workers) **before** you trust any new result.

   | dataset | model | NoAdapt | Ramen |
   |---|---|---|---|
   | CIFAR10C | ViT-B/32 | 61.36 | 72.61 |
   | CIFAR100C | ViT-B/16 | 37.10 | 46.10 |
   | ImageNetC5K | ViT-B/16 | 27.29 | 33.08 |
   | DomainNet | ViT-B/32 | 55.37 | 56.80 |
   | EuroSAT | ViT-B/16 | 48.07 | 62.63 (mean of seeds 0–4) |

   Reproducing one small dataset (CIFAR10C) is enough to validate the harness. Run the large ones only if the
   experiment needs them.

## 3. Rules for clean experiments (same as on the main server)

- **Baseline:** always the unmodified `Ramen` on the **same dataset, seed and stream**. Never edit `Ramen.py`
  or its cfg, because that would break comparability with every cached result. This includes the known fp16
  `PriorityCache` overflow after ~65k samples: fix it only in a variant class, never in the baseline.
- **New ideas are new method classes** (`src/methods/RamenX.py`, subclassing `Ramen`) plus a
  `cfg/<Dataset>/RamenX.yaml` that copies that dataset's Ramen settings. First check that the variant with its
  feature turned off reproduces Ramen's predictions **100% exactly**. Hyperparameter variants go through
  `exp.run('RamenX', name='x_<setting>', cfg={...})`, so each one gets its own cache file.
- **Seeds and the noise floor:** a result counts only if it holds over seeds 0–4. Noise floor = 2 × (max − min)
  of Ramen's average accuracy over seeds 0–4. That is 0.31 (C10), 0.21 (C100), 0.54 (ImageNetC5K) and
  0.11 (DomainNet). It is **not** the standard deviation. Report mean ± std of (variant − Ramen), how many
  seeds were positive, and a paired cluster bootstrap CI (clusters = base image).
- **Pre-register** hyperparameters on one dataset (ImageNetC5K or C10), then test them unchanged on the
  held-out datasets. Label any post-hoc tuning as post-hoc.
- **Dead ends:** skip ideas already tested on the main server. `findings_and_directions.md` and
  `ramen-pc2-handoff.md` list them: anchor, memory purity, NSGD, template selection, global step size, and
  confidence or entropy gates.
- **GPU:** ViT-L/14 doesn't fit on 24 GB. Run long jobs detached:
  `setsid nohup python script.py > new_experiments/results/<name>.log 2>&1 &`. Never `pkill -f` with a
  pattern that appears in your own command line.
- **Files:** never overwrite or delete existing result files without asking me. If a file must be re-run,
  move the old one to `results/backup_<reason>/` first.

## 4. What to send back

The new `*.pkl` (and `_probs.npy`, if captured) files, plus one markdown file in `new_experiments/`. That file
should include the per-seed table (variant vs Ramen, Δ, noise floor), per-domain wins and losses, the exact
cfg used, the git commit, and the package versions. Then I can rsync it all back and the cache names will
line up.

## 5. The experiment to run

<EXPERIMENT>
What: ...
Datasets / seeds: ...
Hypothesis and pre-registered settings: ...
</EXPERIMENT>
