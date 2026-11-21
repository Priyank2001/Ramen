# NIQE vector + T3 stack on ImageNet-C (2026-11-20)

**Question.** Do the two label-free active-set changes that help on ImageNet-C add up? T3 (zero-shot class
weighting, +0.62 ± 0.11) chooses *which class queues* count; the whitened NIQE vector (λ = 0.1, +0.26 ± 0.04) chooses
*which entry inside each queue*. Pre-registered in [PREREGISTRATION.md](PREREGISTRATION.md): primary = mean gain over
Ramen > noise floor 0.53 (seeds 0–4); secondary = beats T3 alone on the mean and ≥ 4/5 seeds.

## Files

| File | Job |
|---|---|
| `PREREGISTRATION.md` | settings and claims, fixed before the full runs |
| `run_stack.py` | runner: `smoke` (λ = 0 == RamenActive prob T3 on 3,000 images) and `run <seeds>` (full ImageNetC5K) |
| `analysis.py` | per-seed and summary tables from the harness caches → `tables/` (no GPU) |
| `plots.py` | one function per notebook figure |
| `build_notebook.py` | writes and executes `stack.ipynb` (`python build_notebook.py interpretation.md`) |
| `after_runs.sh` | waits for all 5 result files, then runs `analysis.py` and `build_notebook.py` |
| `interpretation.md` | the notebook's interpretation cell |
| `stack.ipynb` | the report |
| `tables/` | `per_seed.csv`, `summary.csv`, `per_corruption.csv` |
| `logs/` | `smoke.log`, per-GPU run logs, `analysis.txt` |

No new method code: RamenNSS (`src/methods/RamenNSS.py`) subclasses RamenQuality → RamenActive, so the stack is the
config override `{nss_features: niqe, nss_norm: white, nss_lambda: 0.1, active_set: prob, temperature: 3.0}` on
`cfg/ImageNetC5K/RamenNSS.yaml`. Cached as `nss_niqe_white_l0.1_probT3` in `../results/`.

## How it was run

```bash
python run_stack.py smoke                                   # 100.00% same preds (3,000 of 3,000)
tmux new-session -d -s nss_t3_stack ...                     # windows gpu0: seeds 0 3, gpu1: 1 4, gpu2: 2, analysis
CUDA_VISIBLE_DEVICES=0 python run_stack.py run 0 3          # ~17-20 min per seed, ~22 GB reserved per GPU
bash after_runs.sh                                          # -> tables/, stack.ipynb
```
(Python: `~/miniconda3/envs/RAMEN/bin/python`.)

## Math

Ramen (notation of the project's baseline formulation): every image writes $(z_i, g_i, H_i)$ to the queue of its
zero-shot class; retrieval takes, from each non-empty queue $c$, the $k$ entries with the smallest score $s_{ij}$;
the update is $\theta_i = \theta_0 - \eta\,\mathrm{sign}(\hat g_i)$ with

$$\hat g_i = \sum_{c \in C_i} q_{ic} \sum_{j \in \mathcal N_k^c(z_i)} e^{-H_j} e^{-\beta d_{ij}} g_j .$$

| Run | queue weight $q_{ic}$ | selection score $s_{ij}$ |
|---|---|---|
| Ramen | $1/\lvert C_i\rvert$ | $d_{ij} = \lVert z_i - z_j \rVert_2$ |
| T3 (`active_set: prob`, `temperature: 3`) | $\mathrm{softmax}(100\, z_i^\top W / 3)_c$ | $d_{ij}$ |
| NIQE (`nss_lambda: 0.1`) | $1/\lvert C_i\rvert$ | $d_{ij} + 0.1\,\lVert A_t(v_i - v_j)\rVert_2 / 6$ |
| NIQE + T3 | T3's | NIQE's |

$v$ = the 36 NIQE natural-scene-statistics features (block mean), $A_t$ = causal PCA whitening of the stream so far.
No term uses true labels. ImageNet-C settings: $K = 75$, $k = 1$, $\beta = 0$, $\eta = 0.01$.

## Results (ImageNetC5K, ViT-B/16, mixed stream, domain-averaged accuracy)

Ramen over seeds 0–4: 33.08 / 33.17 / 33.02 / 32.91 / 32.96 → noise floor 2 × 0.26 = **0.53**.

| seed | Ramen | T3 | NIQE | NIQE + T3 | stack vs Ramen (95% CI) | stack vs T3 (95% CI) | fixed / broken vs T3 |
|---|---|---|---|---|---|---|---|
| 0 | 33.08 | +0.51 | +0.27 | **+0.76** | [+0.56, +0.98] | +0.26 [+0.13, +0.38] | 1,338 / 1,146 |
| 1 | 33.17 | +0.51 | +0.27 | **+0.80** | [+0.60, +1.00] | +0.29 [+0.16, +0.42] | 1,315 / 1,100 |
| 2 | 33.02 | +0.73 | +0.27 | **+0.90** | [+0.69, +1.12] | +0.17 [+0.04, +0.30] | 1,296 / 1,167 |
| 3 | 32.91 | +0.72 | +0.31 | **+1.01** | [+0.80, +1.22] | +0.29 [+0.15, +0.42] | 1,329 / 1,110 |
| 4 | 32.96 | +0.63 | +0.20 | **+0.87** | [+0.65, +1.07] | +0.23 [+0.11, +0.37] | 1,297 / 1,122 |
| mean ± std | | +0.62 ± 0.11 | +0.26 ± 0.04 | **+0.87 ± 0.10** | t = 20.1, 5/5 | **+0.25 ± 0.05**, 5/5 | |

CIs: photo-clustered Poisson bootstrap (one base photo under 15 corruptions). Fixed/broken counts are out of 75,000.

- **Primary: PASS.** +0.87 > 0.53; every seed is individually above the floor (min +0.76).
- **Secondary: PASS.** The stack beats T3 on 5/5 seeds, and each seed's CI excludes 0.
- **The gains add:** +0.62 + 0.26 = +0.88 predicted, +0.87 observed.
- Same-corruption share of the top-8 retrieved weight: Ramen 43.6% (earlier), NIQE 49.4%, T3 59.0%, stack 62.5%.
- Per corruption, NIQE repairs T3's losses: elastic −0.52 → −0.03, jpeg −0.48 → −0.08, contrast −0.09 → +0.45,
  fog +0.02 → +0.47, glass blur +0.24 → +0.94. Noise corruptions are unchanged (already T3's best).

## Why it makes sense (and what is not shown)

- **Why it should help:** T3 and NIQE change different terms ($q_{ic}$ vs $s_{ij}$), so they can add if neither
  undoes the other. NIQE's help was largest on the structured distortions where T3 hurts.
- **What happened:** they add almost exactly, and the per-corruption pattern matches that complementarity.
- **Not shown:** that the same-corruption share causes the gain. NIQE adds only +3.5 share points on top of T3
  (+5.8 on top of Ramen), yet its full accuracy gain carries over. This is a hypothesis to test, not a mechanism.
- **Not shown:** generalisation. λ and T were both chosen on ImageNet-C; this is a seed confirmation, not held-out.

**Verdict: works.** +0.87 ± 0.10 over Ramen, above the floor on every seed, on ImageNet-C only. Next: a pre-registered
held-out test on ImageNet-R/A/Sketch with both settings frozen.

**Update (same day): the held-out test failed for the NIQE part** (`../2026-11-20_nss_t3_heldout/`). On
R/A/Sketch the stack minus T3 is +0.01 ± 0.06 and NIQE alone is +0.03 ± 0.09, so the +0.25 here is specific to
corruption shift.
