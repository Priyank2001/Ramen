# Methods in the label-value oracle experiment (2026-11-01 / 02)

The methods used in this folder's runs, written as equations and checked against the code. Each one is a change to
a few lines of Ramen. The baseline is first, so each method file can give only the lines it changes.

| file | method | class (`label_methods.py`) | uses true labels? |
|---|---|---|---|
| [label_values.md](label_values.md) | entries store a gradient toward their true label, revealed after prediction | `RamenLabelValues` | yes (oracle) |
| [write_after_prediction.md](write_after_prediction.md) | write after prediction; file by zero-shot, true or **adapted** prediction | `RamenWriteAfterPrediction` | only for `true_label` filing or `true_label_ce` values |
| [global_retrieval.md](global_retrieval.md) | k nearest entries of the whole cache instead of k per class queue; hide a photo's other copies | `RamenGlobalRetrieval` | only with `true_label_ce` values |
| [analyses.md](analyses.md) | replay measurements: neighbour-label vote, near copies, copy exclusion check | `NeighbourLabels`, `NearCopies`, `check_exclusion.py` | yes (diagnostics) |

All three methods are built through the local registry (`registry.py`). They start from `cfg/<dataset>/Ramen.yaml`
and add config keys from `label_oracle.py: VARIANTS`. Nothing in `src/` or `cfg/` is changed.

## Notation

Batch $t$ holds images $x_1,\dots,x_B$ ($B = 100$), with true labels $y_i$ (used only where a file says so).
$\theta_0$ = the affine weights and biases of every LayerNorm in the CLIP image encoder (BatchNorm for ResNets).
Nothing else is adapted, and each image gets its own copy $\theta_i$. $W = [w_1,\dots,w_C]$ = the normalised
text embeddings of the $C$ classes.

$$z_i = \frac{f_{\theta_0}(x_i)}{\lVert f_{\theta_0}(x_i)\rVert_2}, \qquad
p_i = \mathrm{softmax}\big(100\, W^\top z_i\big), \qquad
\hat y_i = \arg\max_c p_{ic}, \qquad
H_i = -\sum_c p_{ic}\log p_{ic}$$

$$g_i = \nabla_\theta H_i \,\big|_{\theta_0} \;\in \mathbb{R}^D \qquad \text{(own entropy gradient)}$$

$\mathcal{M}_c$ = the queue of class $c$, capacity $K$. Entry $j$ holds key $z_j$, value $v_j$, entropy $H_j$ and
priority $\pi_j$ (global arrival index, fp16). $d_{ij} = \lVert z_i - z_j\rVert_2$.

## Baseline: Ramen

1. **Write** (before retrieval): add $(z_i,\ v_i = g_i,\ H_i,\ \pi_i)$ to $\mathcal{M}_{\hat y_i}$. A full queue
   replaces its lowest-priority entry, so it keeps the $K$ newest.
2. **Retrieve**: for each non-empty queue $c$, $\mathcal{N}_k^c(z_i)$ = the $k$ entries of $\mathcal{M}_c$ nearest
   to $z_i$.
3. **Aggregate**, with weight $\omega_{ij} = e^{-H_j}\,e^{-\beta d_{ij}}$:

$$\hat g_i = \frac{1}{|C_i|}\sum_{c\in C_i}\ \sum_{j\in\mathcal{N}_k^c(z_i)} \omega_{ij}\, v_j,
\qquad C_i = \{c : \mathcal{M}_c \neq \emptyset\}$$

4. **Update** (SignSGD, one step): $\theta_i = \theta_0 - \eta\,\mathrm{sign}(\hat g_i)$.
5. **Predict** $\hat y^{\,\mathrm{ad}}_i = \arg\max_c \mathrm{softmax}(100\, W^\top \tilde z_i)_c$, where
   $\tilde z_i$ is the normalised feature of $x_i$ under $\theta_i$. Then reset $\theta_i \leftarrow \theta_0$.

Since the write comes first, image $i$'s own entry is in $\mathcal{M}_{\hat y_i}$ at distance 0 when it retrieves.
After the fp16 priority freeze (about 28% of the DomainNet stream), new images are no longer written, so they have
no entry of their own.

Under SignSGD only the sign of $\hat g_i$ matters, so multiplying it by any positive number gives the same step.
That is why dividing by $|C_i|$ (Ramen) and dividing by $\sum_j \omega_{ij}$ (used in `global_retrieval.md`) give
the same update.

| symbol | config key | DomainNet | ImageNet-C | EuroSAT |
|---|---|---|---|---|
| $K$ | `max_capacity` | 300 | 75 | 7,500 |
| $k$ | `topk` | 10 | 1 | 50 |
| $\beta$ | `beta` | 5 | 0 | 5 |
| $\eta$ | `lr` (optimizer `signsgd`) | 0.01 | 0.01 | 0.01 |
