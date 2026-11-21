"""Writes and executes heldout.ipynb: python build_notebook.py [interpretation.md]. Needs analysis.py's tables/."""
import importlib.util
import os
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

# The method is the same as in the ImageNet-C confirmation run, so its formulation cell is reused verbatim.
_spec = importlib.util.spec_from_file_location('stack_nb', os.path.join(HERE, '..', '2026-11-20_nss_t3_stack',
                                                                        'build_notebook.py'))
_stack_nb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stack_nb)
MATH = _stack_nb.MATH

HEADER = r"""# Held-out: does NIQE vector + T3 beat Ramen on ImageNet-R / -A / -Sketch?

**Question.** On ImageNet-C, stacking the whitened NIQE-vector selection term (λ = 0.1) on RamenActive's zero-shot class weighting (T = 3) gave +0.87 ± 0.10 over Ramen, every seed above the floor (`../2026-11-20_nss_t3_stack/`). Both settings were chosen on ImageNet-C. Does the stack hold on a stream whose domains are renditions, adversarially filtered photos and sketches rather than pixel-level corruptions?

**Hypothesis (pre-registered, `PREREGISTRATION.md`, written before any NIQE run on this data).** Primary: mean gain of NIQE + T3 over Ramen > the noise floor. Secondary: NIQE + T3 beats T3 alone on the mean and on ≥ 4 of 5 seeds. Written-down alternative: if NIQE only helps because ImageNet-C's domains are corruptions, NIQE alone ≈ 0 here and the stack ≈ T3.

**Runs** (true labels used only for scoring):

| Name here | Cached run | What it changes vs Ramen | Run in this experiment? |
|---|---|---|---|
| Ramen | `Ramen` | — (baseline) | no, cached (`../2026-10-25_heldout/`) |
| T3 | `active_prob_T3` | class queues weighted by softmax(zero-shot logits / 3) | no, cached (`../2026-10-25_heldout/`) |
| NIQE | `nss_niqe_white_l0.1` | selection score + 0.1 × whitened NIQE-vector distance | **yes** (decomposition) |
| NIQE + T3 | `nss_niqe_white_l0.1_probT3` | both | **yes** (primary) |

**Data.** `ImageNetVariants`: ImageNet-R 30,000 + ImageNet-A 7,500 + ImageNet-Sketch 50,889 = 88,389 images, all mapped to the 1,000 ImageNet classes (one head, so R/A numbers are not comparable to the usual 200-class protocol). CLIP ViT-B/16, mixed (shuffled) stream, batch 100. Config: `cfg/ImageNetC5K/RamenNSS.yaml` + overrides, passed in full (its Ramen part equals `cfg/ImageNetVariants/Ramen.yaml`, asserted).

**Seeds.** 0–4; a seed sets the stream order. Every run of a seed sees the identical stream (asserted).

**How the runs were produced.** `python run_heldout_stack.py smoke`, then `python run_heldout_stack.py run stack:<s> niqe:<s> ...` on 3 GPUs in tmux session `nss_heldout` (`logs/`). Tables: `python analysis.py` → `tables/`.

**What counts as a result.** Mean gain over Ramen > 2 × (max − min) of Ramen over seeds 0–4. Accuracy is domain-averaged (mean of the 3 per-domain accuracies). In the tables, the column `same-corruption top8 %` means *same domain* here (the code is shared with the ImageNet-C analysis).

**Sections.** 1 Setup · 2 Data · 3 Sanity check · 4 Results · 5 Mechanism · 6 Interpretation · 7 Glossary."""

GLOSSARY = """## Glossary

| Term | Meaning |
|---|---|
| Ramen | Baseline test-time adaptation: per-class memory of entropy gradients, one SignSGD step per image from retrieved entries, then reset. |
| T3 | RamenActive with `active_set: prob`, temperature 3: class queues weighted by the query's zero-shot probabilities instead of equally. |
| NIQE vector | The 36 natural-scene-statistics features behind the NIQE quality score (block mean); used as a descriptor of the image's low-level statistics. |
| Undefined NIQE vector | Every 96 px block of the image is flat, so a feature cannot be fitted; RamenNSS then uses the stream mean. |
| Held-out | The settings (λ = 0.1, T = 3) were fixed before this data was used for them. T = 3 itself was already tested here. |
| Same-domain share | Weight share of the 8 highest-weighted retrieved entries whose domain equals the query's. Domains are 34% / 8% / 58% of the stream. |
| Noise floor | 2 × (max − min) of Ramen's accuracy over seeds 0–4. |
| Fixed / broken vs T3 | Images T3 gets wrong and the stack gets right / T3 right and the stack wrong. |
| Image-level CI | Poisson bootstrap over images (R/A/Sketch share no photos). |"""

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
corr = pd.read_csv('tables/per_corruption.csv').rename(columns={'corruption': 'domain'})
FLOOR, RAMEN_ACCS = A.noise_floor()
print(f'seeds analysed: {sorted(per_seed.seed.unique())}; noise floor {FLOOR:.2f}')"""),
    md('## 2. The data the code ran on'),
    code("""s0 = A.SeedRuns(0)
z = s0.df['Ramen']
print(f'{len(z):,} images, {z.label.nunique():,} classes present')
pd.DataFrame({'images': z.groupby('domain', observed=True).size(),
              'classes present': z.groupby('domain', observed=True).label.nunique(),
              'Ramen correct (seed 0)': z.groupby('domain', observed=True).correct.sum(),
              'Ramen acc % (seed 0)': (z.groupby('domain', observed=True).correct.mean() * 100).round(2)})"""),
    code("""pd.Series({**{f'Ramen seed {s}': round(a, 2) for s, a in enumerate(RAMEN_ACCS)},
           'range': round(max(RAMEN_ACCS) - min(RAMEN_ACCS), 2), 'noise floor (2 x range)': round(FLOOR, 2)})"""),
    md("""## 3. Sanity check

`run_heldout_stack.py smoke` on a 3,000-image subset (1,000 per domain, seed 0): RamenNSS with λ = 0 and T3 settings made the same prediction as RamenActive prob T3 on **3,000 of 3,000 images (100.00%)**; at λ = 0.1 the stack changed 109 of 3,000 (377 of 3,000 on the ImageNet-C subset). The NIQE vector is undefined on 0 of 1,000 rendition, 0 of 1,000 adversarial and 4 of 1,000 sketch images (0.4%), so the flat-background concern does not bite."""),
    code("print(open('logs/smoke.log').read().split('mode=mixed')[-1].strip())"),
    md('## 4. Results\n\n### Gain over Ramen, seeds 0–4'),
    code("""t = per_seed.copy()
t['images vs Ramen'] = (t['gain vs Ramen'] / 100 * 88389).round()  # approximate: accuracy is domain-averaged
t[['seed', 'run', 'acc', 'gain vs Ramen', 'CI low vs Ramen', 'CI high vs Ramen', 'images vs Ramen',
   'gain vs T3', 'CI low vs T3', 'CI high vs T3', 'fixed vs T3', 'broken vs T3']].round(2)"""),
    code("summary.round(2)"),
    code("fig = plots.seed_gains(per_seed, FLOOR)"),
    md('### Does NIQE add on top of T3? (paired, same stream)'),
    code("fig = plots.stack_vs_t3(per_seed)"),
    md('### Per domain'),
    code("""corr[corr.run != 'Ramen'].groupby(['domain', 'run'])['gain vs Ramen'].mean().unstack().round(2)"""),
    code("fig = plots.per_corruption(corr.rename(columns={'domain': 'corruption'}))"),
    md('## 5. Mechanism: same-domain share of the retrieved support'),
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
    out = os.path.join(HERE, 'heldout.ipynb')
    nbformat.write(nb, out)
    return out


if __name__ == '__main__':
    print(build(open(sys.argv[1]).read() if len(sys.argv) > 1 else '(written after the results)'))
