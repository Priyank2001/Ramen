"""
Write 01_cache_compression.ipynb (not executed).

    python make_notebook.py
    ~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace 01_cache_compression.ipynb
"""
import nbformat as nbf

from interpretation import INTERPRETATION

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

HEADER = r"""# Can Ramen's cache store its gradients in 8, 4 or 1 bit without losing accuracy?

**Question.** Ramen keeps, for every class, a queue of past images: a 512-d CLIP key and the image's entropy
gradient over the LayerNorm parameters (39,936 numbers, fp16). The gradients are 99% of the cache. On ImageNet-C
(1,000 classes × 75 slots) the cache is 5.65 GB; on DomainNet (345 × 300) 7.80 GB. Can the gradient values be stored
in fewer bits, and is that a better use of memory than simply keeping fewer images per class?

**Hypothesis and decision rule (pre-registered 2026-11-18, in `cc_configs.py`, before any run).**
- A codec is **lossless-enough** if |accuracy − Ramen| ≤ the noise floor (domain-averaged, seed 0).
- It **beats shrinking** if its accuracy exceeds its *matched-memory control* (fp16 Ramen whose queue is cut to the
  same cache bytes) by more than the noise floor.
- Compression is worth using only if both hold.

**Why this test.** Compression only matters if it beats the obvious alternative, a shorter queue, at the same
memory. Comparing against Ramen alone would reward any codec that loses nothing, even when halving the queue would
also lose nothing.

**Variants** (all `RamenCompressed`, built on `cfg/<dataset>/Ramen.yaml`; block = 64 values per scale).

| Run | Values stored as | Queue capacity | Cache memory vs Ramen |
|---|---|---|---|
| `cc_off` | fp16 (Ramen's own) | Ramen's K | 100% (must equal Ramen exactly) |
| `cc_int8` / `cc_cap_int8` | int8 + fp32 block scale / fp16 | K / 0.537 K | 53.7% |
| `cc_int4` / `cc_cap_int4` | int4 (2 per byte) + scale / fp16 | K / 0.290 K | 29.0% |
| `cc_sign` / `cc_cap_sign` | 1 bit + block mean \|g\| / fp16 | K / 0.105 K | 10.5% |

K = 75 (ImageNet-C), 300 (DomainNet), 7,500 (EuroSAT); matched capacities 40 / 22 / 8, 161 / 87 / 32 and
4,029 / 2,178 / 789.

**Data.** Full streams, seed 0, mixed domains, batch 100: ImageNet-C 5K (75,000 images, 1,000 classes, 15 corruptions
at severity 5, CLIP ViT-B/16), DomainNet (596,006 images, 345 classes, 6 domains, ViT-B/32), EuroSAT (27,000 images,
10 classes, ViT-B/16). NoAdapt and Ramen are the cached runs of the shared harness.

**Seeds.** One seed. The noise floors (2 × Ramen's max − min accuracy over seeds 0–4: DomainNet 0.11, ImageNet-C 0.54,
EuroSAT 1.51) set the minimum effect size.

**How produced.** `run_compression.py <dataset> 0 --smoke smoke` (200 images per domain) then `--runs ...` on three
RTX 3090s; `verify.py` and `python -m unittest test_compression` for correctness. Results:
`new_experiments/results/<prefix>_cc_*.pkl` and `_stats.pkl`.

**What counts as a result.** The two pre-registered conditions above. Everything else (speed, peak memory, the EuroSAT
capacity effect) is descriptive.

**Sections.** 1 Setup · 2 Data · 3 Correctness · 4 Results · 5 Mechanism and costs · 6 Interpretation · 7 Glossary
"""

MATH = r"""## Method

**Ramen (unchanged).** For each test image $x_i$: key $z_i$ (unit CLIP embedding), zero-shot label $\hat y_i$, entropy
$H_i$, and per-sample gradient $g_i = \nabla_\theta H_i \in \mathbb R^D$ over the LayerNorm weights and biases
(26 LayerNorms × (768 weights + 768 biases), $D = 39{,}936$). $(z_i, g_i, H_i)$ is written into queue $\mathcal M_{\hat y_i}$ (capacity $K$,
lowest priority evicted). To adapt for a query $z$, Ramen retrieves the top-$k$ entries of every queue by key
similarity and forms
$$\hat g(z) = \sum_j w_j\, g_j, \qquad w_j \propto e^{-H_j}\, e^{-\beta d(z, z_j)},$$
then takes one SignSGD step $\theta' = \theta_0 - \eta\, \operatorname{sign}(\hat g)$, predicts, and resets to $\theta_0$.

**What changes.** Only how $g_j$ is stored. Split $g$ into blocks $b$ of 64 consecutive values (every LayerNorm
tensor has $768 = 12 \times 64$ values, so no block spans two tensors). Each block gets one fp32 scale.

- *int8 / int4* (levels $L = 127$ / $7$): $s_b = \max_{t\in b} |g_t| / L$, store $q_t = \operatorname{round}(g_t / s_b)
  \in [-L, L]$, read $\tilde g_t = s_b\, q_t$. Error $|\tilde g_t - g_t| \le s_b / 2$.
- *sign* (1 bit): store $\mathbb 1[g_t \ge 0]$ and $m_b = \operatorname{mean}_{t \in b} |g_t|$, read
  $\tilde g_t = \pm m_b$.

Keys, priorities and entropies stay fp16, so retrieval and the weights $w_j$ are bit-identical to Ramen; only
$\hat g = \sum_j w_j \tilde g_j$ differs.

**Memory per entry** (bytes): key $1{,}024$ + priority 2 + entropy 2 + value: fp16 $79{,}872$; int8 $39{,}936 + 2{,}496$;
int4 $19{,}968 + 2{,}496$; sign $4{,}992 + 2{,}496$. Ratios to fp16: 0.537, 0.290, 0.105.

| Symbol | Meaning | Code | Config | Value |
|---|---|---|---|---|
| block | values sharing one scale | `GradientCodec.block` | `block` | 64 |
| $L$ | quantisation levels each side of 0 | `Int8Codec.levels` / `Int4Codec.levels` | `codec` | 127 / 7 |
| $K$ | queue capacity per class | `PriorityCache.max_capacity` | `max_capacity` | 75 / 300 / 7,500 (matched: see table) |
| codec | value storage | `CODECS[...]` in `grad_codecs.py` | `codec` | fp16, int8, int4, sign |
"""

SETUP = r"""import os, sys, re, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
HERE = os.path.abspath('')
sys.path.insert(0, HERE)
from cc_experiment import CompressionExperiment          # first: puts src/ on the path
from cc_configs import DATASETS, memory_ratio, matched_capacity
from plots import plot_accuracy_vs_memory, plot_peak_memory
pd.set_option('display.width', 220)
NAMES = ['NoAdapt', 'Ramen', 'cc_off', 'cc_int8', 'cc_cap_int8', 'cc_int4', 'cc_cap_int4', 'cc_sign', 'cc_cap_sign']
exps = {ds: CompressionExperiment(ds, seed=0) for ds in DATASETS}
for e in exps.values():
    for n in NAMES:
        e.run(n)                      # all cached: reads the per-image tables and stats only
tables = {ds: e.table(NAMES) for ds, e in exps.items()}
floors = {ds: d['floor'] for ds, d in DATASETS.items()}
def cnt(n, N): return f'{n:,} of {N:,} ({n / N:.1%})'
"""

DATA = r"""rows = []
for ds, e in exps.items():
    r = e.results['NoAdapt']; t = tables[ds]
    rows.append(dict(dataset=ds, images=f'{len(r):,}', domains=r['domain'].nunique(), capacity_K=DATASETS[ds]['capacity'],
                     zero_shot_right=cnt(int(t.loc['NoAdapt', 'correct']), len(r)),
                     ramen_right=cnt(int(t.loc['Ramen', 'correct']), len(r)),
                     ramen_acc_domain_avg=round(t.loc['Ramen', 'acc'], 3), noise_floor=floors[ds]))
display(pd.DataFrame(rows))
display(pd.DataFrame({c: {ds: matched_capacity(ds, c) for ds in DATASETS} for c in ('int8', 'int4', 'sign')})
        .rename(columns=lambda c: f'matched K ({c}, {memory_ratio(c):.3f})'))
"""

SANITY_MD = r"""## 3. Correctness

Three layers of evidence, all re-run below:
1. **Identity on the full data.** `cc_off` runs the full new code path (codec store, new cache class, the method
   subclass) with the fp16 codec. It must predict exactly what Ramen predicts on every image of every stream.
2. **Real-data accounting** (`verify.py`, 86 checks): every run saw the same stream; the cache bytes each run logged
   from its allocated tensors equal classes × capacity × bytes-per-entry worked out from the format by hand; each
   matched control is within one queue slot of its codec's bytes; the codec ran on every batch (one logged update per
   100 images); table accuracies equal a plain recount; no file under `src/` or `cfg/` changed.
3. **Unit tests** (`test_compression.py`, 16 tests, GPU): fp16 round trip is bitwise; int8/int4 error ≤ half a step
   (+ fp16 rounding) and no non-zero block is rounded to all zeros; 4-bit and 1-bit packing is exact; sign decodes to
   sign × block mean; any index shape reads correctly; allocated bytes match the formula; the fp16 compressed cache
   equals Ramen's `PriorityCache` through 40 adds and evictions; the int8 cache retrieves the same entries as Ramen's
   and returns exactly decode(encode(value)); blocks align with LayerNorm tensors.
"""

SANITY = r"""for e in exps.values():
    assert e.sanity_check('cc_off', 'Ramen')
print(open('logs/verify.log').read())
print(open('logs/unittest.log').read())
"""

RESULTS = r"""cols = ['codec', 'capacity', 'correct', 'acc', 'vs_ramen', 'vs_ramen_images', 'same_pred_as_ramen',
        'fixed_vs_ramen', 'broken_vs_ramen', 'mem_ratio']
for ds, t in tables.items():
    print(f'== {ds}  (floor {floors[ds]}; acc = domain-averaged %)')
    display(t[cols].round(3))
"""

VERDICT = r"""rows = []
for ds, t in tables.items():
    for c in ('int8', 'int4', 'sign'):
        d_ramen, d_cap = t.loc[f'cc_{c}', 'vs_ramen'], t.loc[f'cc_{c}', 'acc'] - t.loc[f'cc_cap_{c}', 'acc']
        lossless, beats = abs(d_ramen) <= floors[ds], d_cap > floors[ds]
        rows.append(dict(dataset=ds, codec=c, memory=f'{memory_ratio(c):.1%}', vs_ramen=round(d_ramen, 3),
                         vs_matched_queue=round(d_cap, 3), floor=floors[ds], lossless_enough=lossless,
                         beats_shrinking=beats, worth_using=lossless and beats))
display(pd.DataFrame(rows))
"""

MECH_MD = r"""## 5. Mechanism and costs

How different are the compressed runs' predictions from Ramen's (same prediction, fixed, broken), how often the
aggregated update has exact zeros (SignSGD leaves those parameters unchanged), and what the compression costs in
time and saves in memory.
"""

MECH = r"""rows = []
for ds, t in tables.items():
    for n in NAMES[2:]:
        r = t.loc[n]
        rows.append(dict(dataset=ds, run=n, same_pred_as_ramen=f"{r['same_pred_as_ramen']:.3%}",
                         fixed=int(r['fixed_vs_ramen']), broken=int(r['broken_vs_ramen']),
                         net_images=int(r['vs_ramen_images']), mean_zero_frac=round(r['mean_zero_frac'], 4),
                         cache_GB=round(r['cache_MB'] / 1024, 2), peak_GB=round(r['peak_GB'], 2)))
display(pd.DataFrame(rows))
times = {}
for f in glob.glob('logs/gpu*.log'):
    for m in re.finditer(r'\] (\w+) seed 0: (\w+) done in ([\d.]+) min', open(f).read()):
        times[(m[1], m[2])] = float(m[3])
display(pd.Series(times).unstack(0).loc[NAMES[2:]].rename_axis('run').round(1).add_suffix(' (min)'))
fig = plot_peak_memory(tables); plt.show()
"""

GLOSSARY = r"""## 7. Glossary

| Term | Meaning |
|---|---|
| cache / queue | Ramen's per-class store of past images: key, gradient value, entropy, priority |
| value | an entry's per-sample entropy gradient over the LayerNorm parameters (39,936 numbers) |
| codec | how a value is stored and read back (fp16, int8, int4, sign) |
| block, scale | 64 consecutive values that share one fp32 number setting their size |
| int8 / int4 | round each value to one of 255 / 15 evenly spaced levels between −max and +max of its block |
| sign (1 bit) | keep only whether each value is ≥ 0; read back as ± the block's mean absolute value |
| matched-memory control | fp16 Ramen whose queue is shortened until its cache takes the same bytes as the codec's |
| lossless-enough | accuracy within the noise floor of Ramen's |
| beats shrinking | accuracy above the matched control's by more than the noise floor |
| same pred as Ramen | share of images where the run predicts the same class as Ramen |
| fixed / broken vs Ramen | Ramen wrong → run right / Ramen right → run wrong |
| zero_frac | share of coordinates of the aggregated gradient that are exactly 0 (no SignSGD step for them) |
| peak GPU memory | `torch.cuda.max_memory_allocated` over the run |
| noise floor | 2 × (max − min) of Ramen's accuracy over seeds 0–4 |
| domain-averaged | mean of per-domain accuracies (the headline metric) |
"""

nb = nbf.v4.new_notebook()
nb.cells = [
    md(HEADER), md(MATH),
    md('## 1. Setup'), code(SETUP),
    md('## 2. The data the code ran on'), code(DATA),
    md(SANITY_MD), code(SANITY),
    md('## 4. Results\n\nOne row per run; counts are images. `vs_ramen` is domain-averaged points, `vs_ramen_images` '
       'the difference in images right.'),
    code(RESULTS),
    md('**Pre-registered verdict.** Lossless-enough: |vs Ramen| ≤ floor. Beats shrinking: codec − matched queue > floor.'),
    code(VERDICT),
    code('fig = plot_accuracy_vs_memory(tables, floors); plt.show()'),
    md(MECH_MD), code(MECH),
    md(INTERPRETATION), md(GLOSSARY),
]
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, '01_cache_compression.ipynb')
