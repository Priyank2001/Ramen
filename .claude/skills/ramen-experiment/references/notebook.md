# Experiment notebook template

One notebook per question. Sections in this order; each heading is a markdown cell. Keep code cells short and push reusable logic into a `.py` module beside the notebook.

## 0. Header (markdown, first cell)

```markdown
# <Question as a title>

**Question.** One or two sentences.
**Hypothesis.** One sentence with the number that would confirm it (vs the noise floor).
**Why this test.** Why this is the cheapest decisive experiment (oracle / replay / control).

**Variants.**
| Run name | What it changes | Uses true labels? |
|---|---|---|

**Data.** Dataset(s), number of images per domain, classes, model, config file.
**Seeds.** Which seeds, and what a seed changes (stream order + selection randomness; NoAdapt is seed-independent).
Seed 0 = screening run; seeds 1–4 only for variants that clear the screen.
**How the runs were produced.** Script + exact command; where cached results live.
**What counts as a result.** Gain over the baseline > noise floor (2 × baseline range over seeds 0–4) = <value>.
**How to read this notebook.** Section list.
```

## 0b. Method: mathematical formulation (markdown, second cell)

The method as equations, so a reader can check the code against it. Use LaTeX (`$...$` inline, `$$...$$` display; Jupyter renders both). Write it before the code and keep it in sync with what the code does.

1. **Notation.** Define every symbol once: stream, batch, features, logits, memory, parameters. For Ramen variants, reuse the symbols in `ramen.md` "Baseline formulation" instead of inventing new ones.
2. **Baseline.** The equations of the method being modified, short (insert, retrieve, aggregate, update, predict for Ramen).
3. **The variant.** Only the equations that change, each next to the baseline line it replaces. Spell out the rule as a formula, not a sentence: which set is filtered, by what condition, with what threshold, and what happens to the removed mass (renormalised or not).
4. **Oracle terms.** Write the true label as $y_i$ and say in words that any term containing it uses ground truth, so the variant is an upper bound, not a method.
5. **Controls.** The matched random control as a formula too (e.g. drop a uniform random subset $R$ with $|R| = |S_{\text{wrong}}|$).
6. **Symbol → code.** A table: symbol | meaning | class / method | config key | value in these runs.

Example of a variant block:

```markdown
**Variant `drop_wrong`** (uses true labels). Filing and eviction are as in Ramen. Each entry $j$ gets a
visibility flag fixed when it is written, $v_j = \mathbb{1}[\hat y_j = y_j]$. Retrieval searches only visible entries:
$$\mathcal{N}_k^c(z_i) = \operatorname*{arg\,min}_{S \subseteq \mathcal{V}_c,\ |S| = k} \sum_{j \in S} \lVert z_i - z_j \rVert_2,
\qquad \mathcal{V}_c = \{\, j \in \mathcal{M}_c : v_j = 1 \,\}$$
(baseline: $\mathcal{V}_c = \mathcal{M}_c$). A queue with $\mathcal{V}_c = \emptyset$ is skipped, so $C_i$ counts
queues with a visible entry. Aggregation and the update are unchanged.
Control `random_like_wrong`: $v_j \sim \text{Bernoulli}(1 - e_{t_j})$, where $e_{t_j}$ is the running zero-shot error
rate at the time $t_j$ the entry was written, so about as many entries are hidden, at random.

| Symbol | Meaning | Code | Config | Value |
|---|---|---|---|---|
| $y_j$ | true label (oracle) | `OracleCache` entry field | — | — |
| $v_j$ | entry visible to retrieval | `RamenOracle` write step | `oracle` | `drop_wrong` |
| $k$ | neighbours per queue | `PriorityCache.query` | `topk` | 1 (ImageNet-C) |
```

## 1. Setup
Imports, `sys.path`, plotting style, `CUDA_VISIBLE_DEVICES` (set before importing torch).

## 2. The data the code ran on
Table per dataset: images (total, per domain), classes, baseline correct/wrong **counts and %**, any subgroup the experiment acts on (e.g. confident-wrong images: count and %). This is the denominator for everything after.

## 3. Sanity check
"Off" variant vs cached baseline: identical predictions on N of N samples (100.00%). Stop if not.

## 4. Results
- Main table, one row per run: correct images (count), accuracy %, vs baseline in points **and** in images, fixed / broken vs the zero-shot model (count, %), same prediction as baseline (%).
- Chart: gain vs noise floor (dot plot, reference line at the floor, label each dot with points and images).
- Per-domain breakdown (heatmap or small multiples) when there are domains.

## 5. Mechanism
Whatever the method's logs show about *why* (e.g. contamination of the retrieved set over the stream, memory composition over time). Lines for quantities over the stream; stacked bars for compositions.

## 6. Interpretation (markdown)
Plain-language bullets: what the numbers show, what they rule out, caveats (single seed, oracle not a method), the next step and what result would change it.

## 7. Glossary (markdown, last cell)
Table: term | meaning, for every term a newcomer would trip on (method names, sink, noise floor, fixed/broken, oracle, …).

## Number formatting
- `f'{n:,} of {N:,} ({n / N:.1%})'` → "3,790 of 29,784 (12.7%)".
- Accuracy deltas in points with sign: `f'{d:+.2f}'`; image deltas with sign and thousands separators.
- Say which accuracy (domain-averaged vs pooled) every table uses.
