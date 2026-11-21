# Ramen repo reference

Facts about the Ramen TTA codebase (`~/Projects/Ramen`) as of 2026-10-23. Verify a path or flag still exists before relying on it.

## Contents
1. What Ramen does (the parts experiments touch), and its baseline formulation (shared notebook notation)
2. Harness API
3. Datasets, models, noise floors, runtimes
4. Running on the GPUs
5. Gotchas
6. Dead ends (don't re-run without a new reason)
7. What has worked

## 1. What Ramen does

CLIP zero-shot classifier + per-batch test-time adaptation (`src/methods/Ramen.py`):
- Memory = one fixed-size **queue per class**. Each test image is written, *before* retrieval, into the queue of its **zero-shot prediction** with: CLIP feature (key), its own **entropy-minimisation gradient** (value), its entropy H, and an arrival counter (priority; fp16).
- Retrieval: from every non-empty queue, top-k entries nearest to the query; weight exp(−H)·exp(−β·d); queues averaged with equal weight. SignSGD step (lr 0.01), predict, reset the parameters.
- The filed class label is never used in the loss — it only decides queue membership and eviction (newest replaces oldest when full).
- Configs: `cfg/<Dataset>/<Method>.yaml`. ImageNet-C: capacity 75, topk 1, β 0. DomainNet: capacity 300, topk 10, β 5.
- Variants in `src/methods/`: RamenActive (zero-shot-weighted active set, `IdCache` with stream ids, `filed_as` / `support_ids` logs), RamenPolicy, RamenQuality / RamenNSS (image-quality selection), RamenOracle (true-label contamination oracles), RamenCache, RamenFix (RamenClean, RamenNSGD), RamenAnchor, RamenLog. These were added before the read-only rule (2026-11-01). New methods do **not** go in `src/methods/` and are not registered in `src/methods/__init__.py`: they live in the experiment folder and are registered there (`references/code_design.md`). The extension point is `RamenExperiment.build_method(algo, cfg)` in `ramen_analysis.py`, which can be overridden in a subclass.

### Baseline formulation (copy into notebooks; write variants as changes to these lines)

Checked against `src/methods/Ramen.py` and `src/models/ModelForBySampleTTA.py` on 2026-11-02.

Notation. Batch $t$ holds $x_1,\dots,x_B$ ($B = 100$). $\theta_0$ = the affine weights and biases of every LayerNorm
(BatchNorm for ResNets) in the CLIP image encoder; nothing else is adapted, and each sample gets its own copy
$\theta_i$. $w_c$ = normalised text embedding of class $c \in \{1,\dots,C\}$.

$$z_i = \frac{f_{\theta_0}(x_i)}{\lVert f_{\theta_0}(x_i) \rVert_2}, \qquad
p_i = \operatorname{softmax}\big(100\, z_i^\top W\big), \qquad
\hat y_i = \arg\max_c p_{ic}, \qquad
H_i = -\sum_c p_{ic} \log p_{ic}, \qquad
g_i = \nabla_{\theta} H_i \big|_{\theta_0} \in \mathbb{R}^D$$

1. **Write** (before retrieval, so the batch can retrieve itself): add $(z_i, g_i, H_i, \pi_i)$ to the queue
   $\mathcal{M}_{\hat y_i}$, with priority $\pi_i$ = global arrival index (fp16). A full queue (capacity $K$) replaces its
   lowest-priority entry, i.e. keeps the $K$ newest.
2. **Retrieve**: for every non-empty queue $c$, $\mathcal{N}_k^c(z_i)$ = the $k$ entries of $\mathcal{M}_c$ nearest to $z_i$,
   with $d_{ij} = \lVert z_i - z_j \rVert_2$.
3. **Aggregate** (weights are not normalised; queues count equally):
$$\hat g_i = \frac{1}{|C_i|} \sum_{c \in C_i} \sum_{j \in \mathcal{N}_k^c(z_i)} e^{-H_j}\, e^{-\beta d_{ij}}\, g_j,
\qquad C_i = \{c : \mathcal{M}_c \neq \emptyset\}$$
4. **Update** (SignSGD, one step): $\theta_i = \theta_0 - \eta \operatorname{sign}(\hat g_i)$.
5. **Predict**: $\hat y_i^{\text{Ramen}} = \arg\max_c \operatorname{softmax}\big(100\, \tilde z_i^\top W\big)_c$ with
   $\tilde z_i$ the normalised feature of $x_i$ under $\theta_i$. Then reset $\theta_i \leftarrow \theta_0$
   (the model never drifts; only the memory carries state).

| Symbol | Meaning | Code | Config key | ImageNet-C / DomainNet |
|---|---|---|---|---|
| $K$ | queue capacity per class | `PriorityCache.max_capacity` | `max_capacity` | 75 / 300 |
| $k$ | neighbours per queue | `PriorityCache.query(topk)` | `topk` | 1 / 10 |
| $\beta$ | distance temperature | `Ramen.beta` | `beta` | 0 / 5 |
| $\eta$ | step size | `SignSGD` | `lr` | 0.01 |
| $\pi_i$ | priority (arrival index) | `Ramen.counter` | `priority_fp32` | fp16 (default; saturates after 65,520 inserts, see Gotchas) |
| $D$ | gradient dimension | `grad_dim` | — | all norm-layer affine params |

Note: with $k=1$, $\beta=0$, sample $i$ retrieves its own entry from $\mathcal{M}_{\hat y_i}$ (distance 0), so its own
gradient enters $\hat g_i$ with weight $e^{-H_i}/|C_i|$.

## 2. Harness API (`new_experiments/ramen_analysis.py`)

```python
from ramen_analysis import make_experiment
exp = make_experiment('ImageNetC5K', model='clip_vitbase16', tta_mode='mixed', batch_size=100,
                      n_per_domain=None, seed=0, num_workers=16).load()
exp.evaluate()                                   # NoAdapt + Ramen, loaded from cache
exp.run('RamenOracle', name='oracle_drop_wrong', cfg={'oracle': 'drop_wrong'})   # cached by name
exp.results[name]       # per-sample DataFrame: batch, domain_idx, sample_idx, label, pred, conf, true_prob, entropy, position, correct, domain
exp.method_stats[name]  # method.stats (per-batch logs), if the method keeps them
```
- Cache file: `new_experiments/results/<ds>_<model>_mixed_n<all|N>_bs100_s<seed>_<name>.pkl` (+ `_stats.pkl`, `_feats.npy`, `_probs.npy`). `use_cache=False` forces a re-run.
- Methods with `records_labels = True` get `forward(x, labels=...)` from `exp.run` (not from `src/main.py`'s other paths unless patched).
- Accuracy convention: **domain-averaged** (mean of per-domain accuracies) for headline numbers; sample-weighted only where stated (they differ on DomainNet).
- Offline tools: `ramen_examples.py` (CacheReplay: rebuild memory/active set at any stream position, no GPU), `cache_viewer.py`, `confusion_shift.py`, `active_set_policies.py`.
- Notebooks in a dated subfolder (e.g. `new_experiments/2026-10-23_cache_contamination/`) need `sys.path.insert(0, '..')` to import the harness.

## 3. Datasets, models, noise floors, runtimes

| Dataset (harness name) | Model | Classes | Images | NoAdapt → Ramen (seed 0) | Noise floor | num_workers | Ramen run time |
|---|---|---|---|---|---|---|---|
| CIFAR10C | clip_vitbase32 | 10 | 150,000 (15 corr × 10k, sev 5) | 61.36 → 72.61 | 0.31 | 4 | — |
| CIFAR100C | clip_vitbase16 | 100 | 150,000 | 37.10 → 46.10 | 0.21 | 4 | — |
| ImageNetC5K | clip_vitbase16 | 1000 | 75,000 (15 corr × 5k, sev 5) | 27.29 → 33.08 | 0.54 | 16 | ~7–8 min |
| DomainNet | clip_vitbase32 | 345 | 596,006 (6 domains) | 55.37 → 56.80 (domain-avg) | 0.11 | 16 | ~40 min |
| EuroSAT | clip_vitbase16 | 10 | 27,000 (1 domain) | 48.07 → 62.63 | 1.51 | — | — |
| ImageNetVariants (R 30,000 + A 7,500 + Sketch 50,889; 1,000-class head) | clip_vitbase16 | 1000 | 88,389 (3 domains) | 46.36 → 47.86 | 0.48 | 16 | ~10 min |

- Data under `~/data` (DomainNet in `domainbed/domain_net`, ImageNet-C severity 5 only, EuroSAT in `EuroSAT/2750`). ViT-L/14 does not fit on a 24 GB 3090.
- A **seed** sets the stream order (shuffled mixed stream) and the selection randomness. NoAdapt is identical across seeds.

## 4. Running on the GPUs

- 3 × RTX 3090 (24 GB). Ramen on DomainNet ≈ 8–11 GB, RamenQuality on ImageNet-C ≈ 13.5 GB → one run per GPU.
- Detached runner pattern (see `new_experiments/2026-10-23_cache_contamination/run_oracles.py`): a script with `VARIANTS = {name: cfg}`, `run(dataset, seed, shard, n_shards)`, and a `--smoke` mode on `n_per_domain=200` that checks the "off" variant == Ramen.
```bash
CUDA_VISIBLE_DEVICES=1 setsid nohup python run_x.py DomainNet 0 1 3 > logs/DomainNet_s0_shard1.log 2>&1 < /dev/null &
```

## 5. Gotchas

- **Stream order depends on `num_workers` and tqdm version** (tqdm calls `iter()` on the DataLoader at construction). Use `exp.run()` with the dataset's `num_workers` above, or predictions won't align with cached runs.
- **fp16 priorities** in Ramen's cache saturate after ~65k samples; full queues then stop updating. Default kept so variants match Ramen; `priority_fp32: true` (Ramen, RamenActive) fixes it. On sequential DomainNet the freeze costs 0.22 (memory locked during painting); use fp32 for any non-stationary claim.
- `torch.cdist` has no fp16 CUDA kernel when both sides have ≤ 25 rows. Fixed 2026-10-26 in Ramen.py / RamenActive.py with `compute_mode='use_mm_for_euclid_dist'` (bitwise identical otherwise); other variants (RamenOracle, RamenLog, RamenQuality, RamenFix, RamenPolicy) still crash on small batches.
- `pkill -f <pattern>` / `pgrep -f` inside a Bash command match that command's own shell: use a pattern like `run_sequentia[l].py`.
- Free GPU memory between runs: `gc.collect()` before `torch.cuda.empty_cache()` (method objects sit in reference cycles). Free the parent's cache before allocating a subclass's.
- `pkill -f "<pattern>"` kills its own shell if the pattern is in the command line.
- Zenodo downloads throttle and return HTML error bodies; verify checksums.

## 6. Dead ends (seed-0 unless noted)

- Zero-shot anchors / confidence or entropy gates (helpful and harmful flips have the same margin profile).
- Support label purity: RamenClean (purity 48.8 → 76.6%, accuracy −2.9 on CIFAR-100-C, effective support 158 → 12); write-only-if-agree (−1.04); more entries per class (monotone loss). **Purity never bought accuracy; support size matters.**
- NSGD (−3.8 / −5.8), global / per-domain step-size tuning, prompt-template selection, no-reference IQA as an adapt gate (AUROC ~0.5), quality *prior* in selection.
- Hard top-m active sets (support shrinks, self-gradient dominates).
- Prior correction with a uniform target as a general method: the pre-registered held-out test T3 + pc(λ = 1) on ImageNet-R/A/Sketch gives −1.15 ± 0.09 (Ramen + pc −2.58); it only helps when the label marginal is near uniform.
- Label-free class-prior estimation for pc (online EM on zero-shot probs, zero-shot marginal, EM on own probs): negative on all 6 datasets (zs-EM down to −5.29), although the true-marginal oracle gives T3 +2.75 on ImageNet-R/A/Sketch (structure oracle +2.32). EM on miscalibrated CLIP concentrates the prior on zero-shot's sinks (`new_experiments/2026-10-26_prior_estimation/`). The ImageNet-C 5K oracle gain (+1.96) is subset count noise, not attainable.
- On large label spaces: kNN output smoothing on ImageNet-C (−0.11, neighbours only 25% same class) and prior correction at λ = 2 with a uniform target (DomainNet −0.52). The CIFAR stack T3 + knn + pc(λ = 2) gives −0.02 / −1.13, seeds 0–4 (`new_experiments/2026-10-25_stacking/`).
- Label-free targets for the CE label-value (`new_experiments/2026-11-04_pseudo_label_values/`, seed 0): zero-shot target = null (+0.07 DN / +0.19 INC / −1.13 EuroSAT); adapted-prediction target −0.65 / −5.28 / −31.17; kNN-vote target −1.50 / −15.84 / −10.05. The oracle dose-response (50% of zero-shot errors corrected: +1.29 DN / +24.75 EuroSAT) shows the gain needs a target far more accurate than zero-shot; slightly better targets still lose because they overturn correct images.

- Label-free class mixture (EM from the text prototypes, vMF or shared-covariance LDA, transductive) as a label source (`new_experiments/2026-11-18_gmm_labels/`): DomainNet −11.10 / ImageNet-C −4.00 pooled vs zero-shot (vMF −25 / −20); EuroSAT +15.33 but breaks 10% of zero-shot-right images. Likelihood rises while the components drift to domain / corruption clusters. Even the true-label LDA fit is only +2.99 on DomainNet.

## 7. What has worked

- Output prior correction + causal kNN smoothing: +1.25 (C10), +2.07 (C100), +7.02 (EuroSAT) over Ramen, seeds 0–4.
- Zero-shot-weighted active set T=3: ImageNet-C +0.62 ± 0.11 (5/5 seeds), DomainNet +0.24 ± 0.01, CIFAR-100-C −0.15.
- NSS-vector quality matching: +0.26 ± 0.04 (consistent, below floor).
- Domain-sequential streams (`tta_mode='sequential'`, `new_experiments/2026-10-26_sequential/`): T3 holds on ImageNet-C (+0.52) and R/A/Sketch (+0.66) but fails on DomainNet (−0.07; fp32 −0.12). Ramen-reset beats sequential Ramen by +0.82 on ImageNet-C (stale memory: capacity > per-corruption volume).
- Zero-shot-weighted active set T3 confirmed held-out on ImageNet-R/A/Sketch (1,000-class mixed stream, `ImageNetVariants`): +0.59 ± 0.04, 5/5 seeds above the 0.48 floor (`new_experiments/2026-10-25_heldout/`).
- Cache value compression (`new_experiments/2026-11-18_cache_compression/`, seed 0): block-64 int8/int4/1-bit gradient values are within the floor of Ramen on ImageNet-C, DomainNet and EuroSAT (9/9). At equal memory, 1-bit beats an fp16 shorter queue by +0.64 (IN-C) and +0.49 (DN), and int4 by +0.22 (DN). The 1-bit cache is 9.5x smaller (peak GPU 13.5→8.4 GB IN-C, 10.2→3.3 GB DN). Unoptimised decode is 1.8–2.4x slower on DN. On EuroSAT a shorter queue helps (+1.64 at K=789, one seed): K=7,500 is untuned.
