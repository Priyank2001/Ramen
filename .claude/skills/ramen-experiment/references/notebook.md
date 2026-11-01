# Notebook template

A results notebook is the deliverable, not a scratchpad. Someone who opens it cold should be able to say four things without opening any other file:
- what was tested;
- why it should work, in math;
- what it ran on;
- whether it worked.

That reader might be the advisor, a reviewer, or you in a month.

## Where things go

- The notebook lives at `experiments/week_NN_<monday>/<YYYY-MM-DD>_<experiment>/<experiment>.ipynb`, next to `<experiment>.py`.
- Logic goes in the `.py` (and in `ramen_lab/` if it's reusable), and the notebook imports it. Cells stay short: a call and its output. Runners, statistics and plotting functions belong in code that is reviewed in a diff and tested.
- The notebook reads results through `ramen_lab`'s records API: `metrics.json` for headline numbers, `records.npz` for per-sample analysis. It never recomputes a number it cites from `metrics.json`.
- Execute it top to bottom before handing it over:
  ```
  cd <experiment folder> && PATH=~/priyank/Projects/RAMEN_ENV/bin:$PATH jupyter execute --inplace <experiment>.ipynb
  ```
  A notebook with stale or missing outputs is not finished.

## Cell layout

### 1. Title and overview (markdown)

```markdown
# <Finding-style title: what was tested, on what>

*<date>. Week NN. Plan: docs/plans/<task>.md. Builds on <notebook>; does not repeat it.*

**Question.** <one sentence>
**Hypothesis.** <one sentence with a number: "... above Ramen by more than the noise floor (<value>, seeds 0-4, batch <b>)">

**Variants**
| name | class (ramen_lab) | config | what it changes | uses labels? |
|---|---|---|---|---|
| Ramen | upstream `Ramen` | cfg/CIFAR10C/Ramen.yaml | baseline | no |
| <variant> | `ramen_lab.methods.X` | `{...}` | the term in eq. (4) | no |
| <control> | ... | ... | same side change, without the idea | no |

**Data.** <dataset, images per domain × domains = total, classes>. **Backbone.** <...>. **Batch.** <fixed batch> (not the paper's 100; see D18).
**Stream.** mixed (all domains shuffled into one stream), severity 5.
**Seeds.** <0 (screen) / 0-4>. A seed changes only the stream order.
**How produced.** `python <experiment>.py run ...` on <GPU model>; peak RAM <x> GB, peak GPU <y> GB, <z> min per run.
**What counts.** <the pre-registered gate, copied from the plan>.
```

### 2. Mathematical derivation (markdown, LaTeX); mandatory

Write Ramen's update in full, then the modification. Tie each symbol to code. The template below matches upstream `src/methods/Ramen.py` at `ef0fdc7`; re-check the line numbers if you cite a different file.

```markdown
## How Ramen adapts, and what this experiment changes

For a batch $\{x_i\}_{i=1}^{B}$ with zero-shot CLIP (LayerNorm affine parameters $\theta_0$):

**1. Embedding, prediction, entropy** (`Ramen.py:114-116`, `losses.py:4`)
$$z_i = f_{\theta_0}(x_i), \qquad p_i = \mathrm{softmax}(W^\top z_i), \qquad \hat y_i = \arg\max_c p_{ic}, \qquad
H_i = -\sum_c p_{ic}\log p_{ic}$$

**2. Per-sample gradient** of the entropy w.r.t. the LayerNorm parameters (`Ramen.py:118-120`, by-sample LayerNorm):
$$g_i = \nabla_{\theta} H_i \big|_{\theta_0} \in \mathbb{R}^{D}$$

**3. Memory write**, before retrieval (`Ramen.py:124-133`): $(z_i, g_i, H_i, t_i)$ goes into queue $\mathcal{M}_{\hat y_i}$.
Capacity $K$ per class; when a queue is full, the entry with the lowest priority $t$ (arrival counter, fp16) is replaced.

**4. Retrieval and aggregation** (`Ramen.py:136-154`): from each non-empty queue $c$, the top-$k$ entries nearest to $z_i$:
$$w_{ij} = e^{-H_j}\, e^{-\beta \lVert z_i - z_j \rVert_2}, \qquad
G_i = \frac{1}{|\mathcal{C}^{+}|} \sum_{c \in \mathcal{C}^{+}} \sum_{j \in \mathrm{top}k_c(z_i)} w_{ij}\, g_j$$
where $\mathcal{C}^{+}$ is the set of non-empty queues.

**5. One SignSGD step per sample, predict, reset** (`Ramen.py:157-168`, `optimizer.py:9`):
$$\theta_i = \theta_0 - \eta\,\mathrm{sign}(G_i), \qquad \text{prediction } \arg\max_c\, \mathrm{softmax}(W^\top f_{\theta_i}(x_i))_c, \qquad \theta \leftarrow \theta_0$$

Hyperparameters ($K, k, \beta, \eta$) come from `cfg/<dataset>/Ramen.yaml`.

### The change
<Restate the one equation that changes, with the new term in \color{...} or boxed. Example: "write under $y_i$ instead of $\hat y_i$ in step 3".>
<Map every new symbol to a ramen_lab name: $\tilde G_i$ ↔ `RamenX.aggregate()`.>

### What it should do, and how it could fail
<Derive the expected effect: which samples' $G_i$ change, in which direction, and how much (bounds, special cases such as $\beta=0$ or empty queues).>
<List the ways it can fail, and the log or plot that would show each one. Example: "if mis-filed entries still push toward $\hat y_j$, purity alone cannot help; check the composition of $\mathrm{top}k$".>
<If the idea uses labels, state where $y_i$ enters and prove it cannot affect $x_i$'s own prediction, or say that it does.>
```

Keep the derivation honest:
- Mark anything taken from the paper with its section or equation (`docs/papers/*.md`), and anything you didn't verify with `[unverified]`.
- If a symbol has no code counterpart, the derivation doesn't describe the experiment.

### 3. Setup (code)

`%load_ext autoreload`, `%autoreload 2`, imports from the `.py` and `ramen_lab`, and `ramen_lab.analysis.plots.use_style()` so every figure shares one look. Notebooks don't touch the GPU. If one ever must, set `CUDA_VISIBLE_DEVICES` before importing torch.

### 4. The data (markdown + code)

Before any result:
- a table of images per domain and classes;
- NoAdapt and Ramen error counts, with counts and percentages together;
- the size of any subset the method touches (written entries, retrieved supports, flipped samples).

### 5. Sanity checks (markdown + code)

- **Identity:** the off config gives `identical predictions: 3,000 of 3,000 (100.00%)`, both from the test and from the `check` run.
- **Mechanism did what it says:**
  - an oracle shows its label use, e.g. "filed under the true class: 586,575 of 586,575 (100.00%)";
  - a control shows it matched the variant's amount.
- **Resources:** peak RSS against the budget, peak GPU memory and minutes per run, from `metrics.json["resources"]`.

### 6. Results (one section per finding)

Each section has three parts:
1. a markdown header that states the finding ("X gains +0.42 over Ramen; the control gains +0.05; the floor is 0.31");
2. a table (accuracy, gain over the baseline with a CI, helped/hurt counts);
3. **one static matplotlib chart**, made by a function in `ramen_lab.analysis.plots` or the experiment `.py`. No plotly: notebooks must render on GitHub and in PDF.

| what you're showing | form |
|---|---|
| gain per variant or per seed vs the noise floor | dot plot with CI, dashed lines at ±floor and at 0 |
| parts of a whole (fixed / broken / both right / kept wrong) | stacked horizontal bars, labelled with % and count |
| a quantity along the stream (accuracy, memory purity, queue fill) | line chart, x = stream position (thousands of images) |
| before/after per item (per domain, per class) | dumbbell, sorted by change |
| a per-sample score by outcome | overlaid histograms or ECDF |
| a derivation quantity (e.g. sign agreement, weight mass per queue) | histogram or line, with the analytical expectation overlaid |

Chart rules:
- One finding per chart; put the finding in the title.
- Use the shared palette: Ramen blue, NoAdapt orange, variant purple, control grey, worse red.
- Put counts in labels or annotations, and sort categories by value.
- Label axes with units: "gain over Ramen (pts)", "stream position (×1,000 images)".
- Save headline figures to `figures/` as PNG (dpi 150) only if they go into a report.

For seed sweeps, add a per-seed table:

| seed | Ramen | variant | gain |
|---|---|---|---|

Under it, give mean ± std of the gain, how many of 5 seeds are positive, the noise floor, and whether the mean clears it.

### 7. Interpretation (markdown)

Write plain sentences that cover:
- what the numbers say;
- what they don't say (single seed, post-hoc choices, what was tuned where);
- the mechanism: connect it back to the derivation and say which predicted effect showed up and which didn't; label anything unproven `[hypothesis]`;
- the next step and its stop rule.

### 8. Glossary (markdown)

Define every non-obvious term and every symbol used: $z_i, g_i, G_i, H_i, w_{ij}, K, k, \beta, \eta$, noise floor, paired gain, fixed / broken, oracle, control, mixed stream, fp16 priority, TV(pred, true), ECE, and so on.

## Number formatting

- Use counts and percentages together: `3,790 of 29,784 (12.7%)`, with thousands separators in tables.
- Give accuracy in % to 2 decimals, and gains in points with a sign to 2 decimals: `+0.62`.
- Put the noise floor next to every gain. Where the floor is unknown, say "floor unknown".
