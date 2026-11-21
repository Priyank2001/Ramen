"""Writes and executes stack.ipynb: python build_notebook.py [interpretation.md]. Needs analysis.py's tables/."""
import os
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

HEADER = r"""# Does the NIQE vector add on top of T3 on ImageNet-C?

**Question.** Two label-free changes to Ramen's active set each help on ImageNet-C: weighting class queues by the query's zero-shot probabilities (**T3**, +0.62 ± 0.11, seeds 0–4) and adding a whitened NIQE-feature distance to the entry selection (**NIQE**, +0.26 ± 0.04). They act at different levels: T3 picks *which class queues* count, NIQE picks *which entry inside each queue*. Do they add up?

**Hypothesis (pre-registered, `PREREGISTRATION.md`).** Primary: the stack's mean gain over Ramen exceeds the noise floor (2 × Ramen's range over seeds 0–4 = 0.53). Secondary: it beats T3 alone on the mean and on at least 4 of 5 seeds. If the gains added exactly, the stack would give +0.88.

**Why this test.** Both settings were fixed in earlier experiments (λ = 0.1 from `nss_active_set.md`, T = 3 from `active_seeds.py`), so this is a confirmation run with nothing to tune. The single earlier data point, the *scalar*-score regulariser stacked with T3 on seed 0, gave +0.66 vs T3's +0.51.

**Runs** (true labels are used only for scoring and diagnostics):

| Name here | Cached run | What it changes vs Ramen | Run in this experiment? |
|---|---|---|---|
| Ramen | `Ramen` | — (baseline) | no, cached |
| T3 | `active_prob_T3` | class queues weighted by softmax(zero-shot logits / 3) | no, cached (`active_seeds.py`) |
| NIQE | `nss_niqe_white_l0.1` | selection score + 0.1 × whitened NIQE-vector distance | no, cached (`nss_active_set.py`) |
| NIQE + T3 | `nss_niqe_white_l0.1_probT3` | both | **yes** |

**Data.** ImageNet-C 5K (`ImageNetC5K`): 15 corruptions × 5,000 images = 75,000, severity 5, 1,000 classes; each base photo appears under all 15 corruptions. CLIP ViT-B/16, `cfg/ImageNetC5K/RamenNSS.yaml` (capacity 75, top-k 1, β 0, SignSGD lr 0.01), mixed (shuffled) stream, batch 100.

**Seeds.** 0–4. A seed sets the stream order; zero-shot (NoAdapt) is the same for every seed. Every run of a seed sees the identical stream (asserted).

**How the runs were produced.** `python run_stack.py run <seeds>` on 3 GPUs in tmux session `nss_t3_stack` (logs in `logs/`), after `python run_stack.py smoke`. Tables: `python analysis.py` → `tables/`. Harness caches: `../results/ImageNetC5K_clip_vitbase16_mixed_nall_bs100_s<seed>_<run>.pkl`.

**What counts as a result.** Mean gain over Ramen > 0.53 (primary). Accuracy is domain-averaged (mean of the 15 per-corruption accuracies) throughout.

**Sections.** 1 Setup · 2 Data · 3 Sanity check · 4 Results · 5 Mechanism · 6 Interpretation · 7 Glossary."""

MATH = r"""## Method: mathematical formulation

**Baseline (Ramen).** Batch $t$ holds $x_1,\dots,x_B$ ($B=100$); $\theta_0$ = the LayerNorm affine parameters of the CLIP image encoder; $w_c$ = normalised text embedding of class $c$.

$$z_i = \frac{f_{\theta_0}(x_i)}{\lVert f_{\theta_0}(x_i)\rVert_2},\quad p_i = \operatorname{softmax}(100\, z_i^\top W),\quad \hat y_i = \arg\max_c p_{ic},\quad H_i = -\textstyle\sum_c p_{ic}\log p_{ic},\quad g_i = \nabla_\theta H_i\big|_{\theta_0}$$

1. **Write** $(z_i, g_i, H_i, \pi_i)$ into the queue $\mathcal{M}_{\hat y_i}$ (capacity $K$, newest kept), before retrieval.
2. **Retrieve** from each non-empty queue $c$ the $k$ entries with the smallest score $s_{ij} = d_{ij} = \lVert z_i - z_j\rVert_2$: $\mathcal{N}_k^c(z_i)$.
3. **Aggregate** $\displaystyle \hat g_i = \frac{1}{|C_i|}\sum_{c\in C_i}\sum_{j\in\mathcal{N}_k^c(z_i)} e^{-H_j} e^{-\beta d_{ij}} g_j$.
4. **Update** $\theta_i = \theta_0 - \eta\,\operatorname{sign}(\hat g_i)$, **predict** with $\theta_i$, reset to $\theta_0$.

**T3 (RamenActive, `active_set: prob`) changes step 3** — each queue is weighted by the query's zero-shot probability at temperature $T$:
$$\hat g_i = \sum_{c\in C_i} q_{ic} \sum_{j\in\mathcal{N}_k^c(z_i)} e^{-H_j} e^{-\beta d_{ij}} g_j,\qquad q_i = \operatorname{softmax}\!\big(100\, z_i^\top W / T\big)$$
(Ramen: $q_{ic} = 1/|C_i|$).

**NIQE (RamenNSS) changes step 2** — the selection score gains a distortion-matching term:
$$s_{ij} = d_{ij} + \lambda\,\frac{\lVert A_t (v_i - \mu_t) - A_t (v_j - \mu_t)\rVert_2}{\sqrt{36}}$$
$v_i \in \mathbb{R}^{36}$ = NIQE's natural-scene-statistics features of image $i$ (AGGD fits to the MSCN coefficients and their paired products, 2 scales), averaged over the image's four 96 px blocks, computed on CLIP's 224 px grey view. $\mu_t, A_t$ = running mean and PCA-whitening map of all $v$ seen up to and including batch $t$ (causal, no labels). Ramen: $\lambda = 0$. The weight in step 3 still uses $d_{ij}$, not $s_{ij}$.

**NIQE + T3** applies both changes; steps 1 and 4 are Ramen's. No term uses the true label $y_i$.

| Symbol | Meaning | Code | Config key | Value |
|---|---|---|---|---|
| $K$ | queue capacity | `PriorityCache.max_capacity` | `max_capacity` | 75 |
| $k$ | entries per queue | `QualityCache.query` | `topk` | 1 |
| $\beta$ | distance temperature | `Ramen.beta` | `beta` | 0 |
| $\eta$ | step size | SignSGD | `lr` | 0.01 |
| $T$ | zero-shot temperature for queue weights | `RamenActive` | `active_set`, `temperature` | prob, 3.0 |
| $\lambda$ | weight of the NIQE term | `RamenNSS.class_penalty` | `nss_lambda` | 0.1 |
| $v$ | NIQE feature vector | `ImageQuality.with_features` (`src/models/iqa.py`) | `nss_features` | niqe |
| $A_t, \mu_t$ | causal whitening | `StreamNormaliser` | `nss_norm` | white |"""

GLOSSARY = """## Glossary

| Term | Meaning |
|---|---|
| Ramen | The baseline test-time adaptation method: per-class memory of entropy gradients, one SignSGD step per image from retrieved entries, then reset. |
| Active set | The memory entries a query retrieves and combines into its update. |
| T3 | RamenActive with `active_set: prob`, temperature 3: class queues weighted by the query's zero-shot probabilities instead of equally. |
| NIQE vector | The 36 natural-scene-statistics features behind the NIQE quality score; here a descriptor of *which distortion* an image has, not of its quality. |
| Whitening | Linear map that decorrelates the features and scales them to unit variance, fitted on the stream seen so far. |
| Same-corruption share | Weight share of the 8 highest-weighted retrieved entries whose corruption equals the query's (chance 1/15 = 6.7%). |
| Noise floor | 2 × (max − min) of Ramen's accuracy over seeds 0–4. A mean gain must exceed it to count as established. |
| Fixed / broken vs T3 | Images T3 gets wrong and the stack gets right / T3 right and the stack wrong. |
| Domain-averaged accuracy | Mean of the 15 per-corruption accuracies (all corruptions have 5,000 images here, so it equals pooled accuracy). |
| Photo-clustered CI | Bootstrap that resamples base photos (each appears under 15 corruptions), so correlated copies are not counted as independent. |"""

CELLS = [
    md(HEADER),
    md(MATH),
    md('## 1. Setup'),
    code("""import pandas as pd
import analysis as A
import plots
pd.set_option('display.width', 200)
per_seed = pd.read_csv('tables/per_seed.csv')
summary = pd.read_csv('tables/summary.csv')
corr = pd.read_csv('tables/per_corruption.csv')
FLOOR, RAMEN_ACCS = A.noise_floor()
print(f'seeds analysed: {sorted(per_seed.seed.unique())}; noise floor {FLOOR:.2f}')"""),
    md('## 2. The data the code ran on'),
    code("""s0 = A.SeedRuns(0)
z = s0.df['Ramen']
n = len(z)
print(f'{n:,} images, {z.label.nunique():,} classes present, {len(s0.corruptions)} corruptions x {n // len(s0.corruptions):,}')
pd.DataFrame({'images': z.groupby('domain', observed=True).size(),
              'Ramen correct (seed 0)': z.groupby('domain', observed=True).correct.sum(),
              'Ramen acc % (seed 0)': (z.groupby('domain', observed=True).correct.mean() * 100).round(2)})"""),
    code("""pd.Series({**{f'Ramen seed {s}': round(a, 2) for s, a in enumerate(RAMEN_ACCS)},
           'range': round(max(RAMEN_ACCS) - min(RAMEN_ACCS), 2), 'noise floor (2 x range)': round(FLOOR, 2)})"""),
    md("""## 3. Sanity check

`run_stack.py smoke` on a 3,000-image subset (15 × 200, seed 0): RamenNSS with λ = 0 and T3 settings made the same prediction as RamenActive prob T3 on **3,000 of 3,000 images (100.00%)**, so the stack differs from T3 only through the NIQE term. At λ = 0.1 the stack changed 377 of 3,000 predictions. Every run of a seed is asserted to see the same stream order (`analysis.SeedRuns`)."""),
    code("print(open('logs/smoke.log').read().split('mixed')[-1].strip())"),
    md('## 4. Results\n\n### Gain over Ramen, seeds 0–4'),
    code("""t = per_seed.copy()
t['correct'] = (t.acc / 100 * 75000).round().astype(int)
t['images vs Ramen'] = (t['gain vs Ramen'] / 100 * 75000).round()
t[['seed', 'run', 'acc', 'gain vs Ramen', 'CI low vs Ramen', 'CI high vs Ramen', 'images vs Ramen',
   'gain vs T3', 'fixed vs T3', 'broken vs T3']].round(2)"""),
    code("""summary.round(2)"""),
    code("fig = plots.seed_gains(per_seed, FLOOR)"),
    md('### Does NIQE add on top of T3? (paired, same stream)'),
    code("fig = plots.stack_vs_t3(per_seed)"),
    md('### Per corruption'),
    code("fig = plots.per_corruption(corr)"),
    code("""corr[corr.run != 'Ramen'].groupby(['corruption', 'run'])['gain vs Ramen'].mean().unstack()[plots.ORDER].round(2)"""),
    md("""## 5. Mechanism: does the stack retrieve more same-corruption support?

Both components were earlier found to work by changing *where* the update comes from; NIQE in particular raised the same-corruption share of the retrieved entries. Logged: the stream ids and weights of each query's 8 highest-weighted retrieved entries."""),
    code("fig = plots.same_corruption(summary)"),
    md('## 6. Interpretation\n\nINTERPRETATION'),
    md(GLOSSARY),
]


def build(interpretation):
    nb = nbformat.v4.new_notebook()
    nb.cells = CELLS
    nb.cells[-2] = md('## 6. Interpretation\n\n' + interpretation)
    nb.metadata['kernelspec'] = {'name': 'ramen-env-kernel', 'display_name': 'Python 3 (RAMEN)', 'language': 'python'}
    ExecutePreprocessor(timeout=1800, kernel_name='ramen-env-kernel').preprocess(nb, {'metadata': {'path': HERE}})
    out = os.path.join(HERE, 'stack.ipynb')
    nbformat.write(nb, out)
    return out


if __name__ == '__main__':
    print(build(open(sys.argv[1]).read() if len(sys.argv) > 1 else '(written after the results)'))
