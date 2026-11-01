# pc2_lab: upstream API, ramen_lab, data, memory safety, gotchas

Facts as of 2026-11-01. A section marked **TBD** is filled in by the week-01 baseline (`docs/plans/week01_baseline.md`). Until then it is unknown, and no number may be quoted from it.

## Contents
1. Branch and frozen upstream
2. Upstream API (what `ramen_lab` wraps)
3. `ramen_lab` architecture
4. Datasets, backbones, configs
5. Memory safety and launching
6. Seeds, streams and noise floors
7. Gotchas
8. Plan template (pre-registration)

---

## 1. Branch and frozen upstream

- Branch `pc2_lab` starts at upstream Ramen `ef0fdc7` (github.com/baowenxuan/Ramen). Earlier work is on `sec_server_dual_agent` (old harness, batch 100) and `ramen_clean` (cache framework). Decision D18 in `docs/decisions.md`.
- **Frozen:** `src/`, `cfg/`, `shell/`, `README.md`, `.gitignore`. Two checks enforce it: `tests/test_upstream_untouched.py` (no diff against `ef0fdc7`, no new files inside) and the PreToolUse guard (denies Write/Edit/redirect/`sed -i`/cp/mv/rm there).
- Ignore rules live in nested `.gitignore` files (`experiments/`, `docs/`, `.claude/`). Local leftovers from other branches are excluded in `.git/info/exclude`.

## 2. Upstream API (read-only; check `src/` before relying on a detail)

- **Model:** `models.get_pretrained_model.get_pretrained_model(args) -> (clip_model, preprocess)`. It reads `args.model` ∈ {`clip_vitbase16`, `clip_vitbase32`, `clip_vitlarge14`, `clip_rn50`, `clip_rn101`} and `args.device`.
- **Datasets:** `datasets.get_dataset_class(name)(root=..., transform=preprocess)`.
  - Names: `CIFAR10C`, `CIFAR100C`, `ImageNetC5K`, `DomainNet` (+ other DomainBed sets).
  - The object has `.datasets` (one per domain), `.environments`, `.classes` and `.num_classes`, and is iterable over domains.
  - Corruption sets also take `extra=False` (`extra=True` gives the held-out corruptions used for tuning) and `severity=5`.
  - Root is `~/data`: `corruption/CIFAR-10-C`, …, and `domainbed/domain_net`.
- **Mixed stream:** `datasets.utils.TaggedMultipleDataset(datasets)` yields `(image, label, domain_idx, sample_idx)`.
- **Methods:** `methods.get_method_class(name)` knows only `NoAdapt`, `Tent` and `Ramen`. That is why `ramen_lab` has its own registry: registering there must never require editing `src/methods/__init__.py`.
  - Constructor: `Method(model, datasets, args)`, where `args` needs `.config` (dict from the yaml), `.device`, `.max_batch_size` (= batch size; sizes the by-sample LayerNorm buffers) and `.use_tbn`.
  - `forward(x) -> logits`; `reset()`.
- **Ramen internals** (`src/methods/Ramen.py`):
  - `PriorityCache` is one per class, with fp16 keys, values, priorities and entropies.
  - `Ramen.__init__` builds `self.model = CLIPModelForBySampleTTA(...)` and `self.cache`.
  - `Ramen.forward` takes these steps:
    1. featurize, classify, entropy;
    2. backward to per-sample gradients;
    3. write the batch into the zero-shot-class queues;
    4. retrieve the top-k from every non-empty queue and weight them;
    5. one SignSGD step per sample;
    6. predict;
    7. reset the weights.
- **By-sample model** (`src/models/ModelForBySampleTTA.py`):
  - `featurize`, `classify`, `get_by_sample_grad`, `set_by_sample_grad`, `step_and_zero_grad`, `reset_parameters`, `feat_dim`, `grad_dim`.
  - Only the LayerNorm affine parameters adapt; everything else has `requires_grad_(False)`.
- **Upstream `main.py`:** `mixed_domain_test` uses `DataLoader(shuffle=True)` **without an explicit generator**, so its order depends on the global RNG state. `ramen_lab` uses an explicit seeded generator, so every method sees the same order. Its identity check compares against upstream classes driven by `ramen_lab`'s own stream, not against `main.py`.

## 3. `ramen_lab` architecture

The package is planned in `docs/plans/week01_baseline.md`. Keep this section in sync with the code once it exists.

```
ramen_lab/
  upstream.py     the ONLY module that touches src/: puts src/ on sys.path, adapts get_pretrained_model,
                  get_dataset_class and the upstream method classes to ramen_lab's interfaces
  config.py       RunConfig (frozen dataclass) + DATASET_DEFAULTS (backbone, fixed batch, RAM budget per dataset)
  registry.py     MethodRegistry: name -> factory; upstream methods pre-registered; variants via @register
  methods/        our variants (subclass / compose upstream; ORACLE variants declare uses_labels = True)
  stream.py       StreamFactory: mixed / single, explicit seeded generator, optional per-domain subset
  runner.py       Runner: builds model + data + method, iterates the stream, feeds the recorder, checks the guard
  records.py      RecordWriter (streams per-sample arrays to records.npz) + RunRecord (metrics.json via scripts/run_record.py)
  resources.py    MemoryGuard (psutil; process tree RSS + system available), ResourceBudget, GPU peak
  analysis/       stats.py (noise floor, paired gains, cluster bootstrap), plots.py (matplotlib style + chart functions)
  launch.py       CLI launcher: free-GPU check, worker cap, systemd-run memory cap when linger is on, setsid detach
```

The dependency direction is `experiments/*` → `ramen_lab` → `ramen_lab.upstream` → `src/`. Nothing outside `upstream.py` imports from `src/` directly. If upstream changes, or another backbone or code base is added, only the adapter changes (dependency inversion).

## 4. Datasets, backbones, configs

| key | backbone (default) | stream | classes | upstream cfg (`cfg/<key>/Ramen.yaml`) | fixed batch |
|---|---|---|---|---|---|
| `CIFAR10C` (sev. 5) | `clip_vitbase32` | 15 × 10,000 | 10 | max_capacity 7500, topk 50, beta 5.0, signsgd lr 0.01 | **100** (= paper; 9.18 GB peak on the 2080 Ti; RAM budget 8.2 GB) |
| `CIFAR100C` (sev. 5) | `clip_vitbase16` | 15 × 10,000 | 100 | 750, 5, 5.0, signsgd 0.01 | **TBD** |
| `ImageNetC5K` (sev. 5) | `clip_vitbase16` (paper: L/14, doesn't fit) | 15 × 5,000 | 1000 | 75, 1, 0.0, signsgd 0.01 | **TBD** |
| `DomainNet` | `clip_vitbase32` | 6 domains, 586,575 | 345 | 300, 10, 5.0, signsgd 0.01 | **TBD** |

- Upstream's `shell/run_ramen.sh` uses batch 100 (ImageNet-C L/14: 50). Upstream README reports, at batch 100, 11.6 GB for B/32 CIFAR-10-C and DomainNet, and 14.5 GB for B/16 CIFAR-100-C. Our GPUs are 12 GB (GPU 0, 3080 Ti) and 11 GB (GPU 1, 2080 Ti), so the fixed batch is smaller. Find it once by smoke runs, freeze it in `DATASET_DEFAULTS`, and never change it silently.
- The cache size doesn't depend on the batch: classes × capacity × grad_dim × 2 bytes. For DomainNet that is 345 × 300 × ~40k × 2 B ≈ 8.3 GB on the GPU. The batch only has to fit next to it.
- DomainNet "accuracy" = the mean over the 6 domains (per image it is lower, because quickdraw is large and hard). Say which one you report.
- CIFAR and ImageNet-C: one base image appears under all 15 corruptions, so samples aren't independent. Bootstrap by base image.

## 5. Memory safety and launching (D17)

PC2 has 125 GB RAM, **2 GB swap** and 32 cores, and it is shared. An exhausted host stops responding (the user hit this). Rules:

- **Budgets (defaults until measured):**
  - per run, 24 GB RSS (process tree, including DataLoader workers);
  - system reserve, 16 GB available;
  - at most 2 concurrent runs (one per GPU).

  After the week-01 smoke runs, set each dataset's budget to about 1.5 × its measured peak and record it in `DATASET_DEFAULTS`.
- **In-process `MemoryGuard`** (always on):
  - a background thread samples RSS (`psutil`, process + children) and `virtual_memory().available` every 2 s;
  - over budget, or below the reserve, it sets a flag, and the runner stops after the current batch, writes a partial-run note and exits non-zero;
  - at 1.2 × budget it hard-exits (`os._exit`) in case the main thread is stuck;
  - peak RSS goes into `metrics.json["resources"]`.
- **Config limits:**
  - `num_workers` ≤ 4;
  - `pin_memory` only for the image batch (as upstream);
  - no pinned host-RAM gradient caches (the batch is chosen to fit the GPU instead);
  - per-sample records are written in chunks to disk (np.memmap / incremental npz), never held as whole-stream Python lists of tensors;
  - features and probabilities are captured in fp16, and only when asked.
- **Hard cap (kernel):** `systemd-run --user --scope -p MemoryMax=<budget>G -p MemorySwapMax=0 <cmd>`.
  - Verified 2026-11-01: an allocation over the cap is killed (exit 137), and the host is unaffected.
  - `Linger=no` on PC2, so user scopes die when the last login session ends. `launch.py` uses the cap only when `loginctl show-user $USER -p Linger` says `yes` (the user decides whether to enable linger). Otherwise it uses plain `setsid nohup` and relies on the in-process guard.
- **Before launching:** check `nvidia-smi` (target GPU idle) and `free -g`; available must be ≥ reserve + budget. Use one run per GPU. Chain sequential runs in one detached job.
- **Environment:** `~/priyank/Projects/RAMEN_ENV/bin/python` (torch 2.0.1+cu117, numpy 1.x, psutil 7.2.2), with `PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>`. There is no `python` on PATH.

## 6. Seeds, streams and noise floors

- A seed changes only the stream order (the seeded DataLoader generator). Upstream `methods/` draw no random numbers. Verify this in the week-01 determinism test; don't assume it.
- **Noise floor = 2 × (max − min) of Ramen's mean accuracy over seeds 0–4**, at the fixed batch, per dataset and GPU model. **TBD** for every dataset on `pc2_lab`. The batch-100 floors in `legacy_batch100.md` don't apply.
- Per-domain claims need that domain's own spread across seeds, which is much larger than the mean's.
- Per-sample comparisons only between runs on the same GPU model (fp16 kernels differ between the 3080 Ti and the 2080 Ti).

## 7. Gotchas (carried over; re-verify on upstream code)

- **fp16 priorities:** `PriorityCache` priorities are fp16 arrival counters. fp16 is exact only up to 2048 and saturates at 65,504. Long streams therefore get coarse, then frozen, FIFO order. Keep this in variants (or the identity check fails), and call it out where it matters (DomainNet, 586k samples).
- **Write before retrieve:** Ramen writes the batch into memory before retrieving, so each query retrieves itself (distance 0) and its batch-mates. Any "retrieved composition" statistic must handle self-matches. Any write-after-prediction variant needs a write-after control.
- **The cached value is the entry's own entropy gradient** toward its zero-shot argmax. Changing *where* an entry is filed doesn't change which way it pushes. This is why oracle true-label filing made Ramen worse on DomainNet (legacy result, `docs/research_state.md`).
- **`torch.cdist` fp16:** with fewer than 25 query rows, cdist takes a direct kernel with no fp16 CUDA implementation in torch 2.0.1. Tests use batch ≥ 32.
- **`.pkl`, `.npy`, `.npz` and logs are git-ignored**, so results stay on the machine that made them. Move them with scp. Commit only `metrics.json`, code, notebooks and small figures.
- **Execute notebooks** with `PATH=~/priyank/Projects/RAMEN_ENV/bin:$PATH jupyter execute --inplace <nb>`, from the notebook's folder.

## 8. Plan template (`docs/plans/<task>.md`)

```markdown
# Plan: <task>
*<date>. Status: <draft / running / done: result in one line>. <ORACLE / method / diagnostic>.*
## Question and hypothesis   (one sentence each, the hypothesis with a number)
## Derivation                (Ramen's update; the changed term; predicted effect; how it could fail)
## Variants                  (table: name | class | config | what it changes | uses labels?) incl. control + identity
## Files                     (new files in ramen_lab/ or the experiment folder; never src/ cfg/ shell/)
## Tests                     (identity, reset, frozen, labels, determinism, + variant-specific)
## Runs and resources        (subset check -> seed 0 screen -> seeds 0-4 -> held out; GPU, batch, RAM budget, minutes)
## Pre-registered reading    (gate, stop rule, noise floor source, what is diagnostic only)
## Risks
```
