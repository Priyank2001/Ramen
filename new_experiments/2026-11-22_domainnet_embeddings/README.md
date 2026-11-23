# DomainNet: how zero-shot CLIP embeds the six domains (2026-11-22)

**Question.** Does the frozen CLIP ViT-B/32 image encoder organise DomainNet images by class or by domain, where are the zero-shot
errors, and why is quickdraw at 13% while real is at 82%? Descriptive visualisation of the cached seed-0 NoAdapt features
(all 596,006 images, 345 classes, 6 domains). Offline, CPU only, ~3 min. No method, no variant, nothing in `src/`, `cfg/` or the
shared harness edited.

| file | job |
|---|---|
| `data.py` | `DomainNetEmbeddings`: cached unit-norm features `Z`, text embeddings `W`, labels, domains, zero-shot predictions (row-aligned with the seed-0 NoAdapt run), `describe()`, `sanity()`, `image(i)` (file from the HDD) |
| `analysis.py` | pure numbers: `stratified_sample`, `Projection` (PCA→t-SNE, cached), `Geometry` (variance shares, concentration, centroid similarity, cross-domain class agreement, centroid zero-shot accuracy, class×domain accuracy), `domain_probe`, `Neighbours` (10-NN composition, domain mixing), `TextAlignment` (margins, modality gap), `top_confusions` |
| `plots.py` | one function per figure (repo palette; fixed colour per domain) |
| `make_notebook.py` | writes `01_domainnet_embeddings.ipynb` |
| `01_domainnet_embeddings.ipynb` | executed notebook: header, definitions, data, maps, geometry, neighbours, alignment, images, interpretation, glossary |
| `results/` | cached 2-D projections (`proj_<method>_c<centred>_p<perplexity>_s<seed>_<hash>.pkl`) |
| `logs/` | execution logs |

Run:
```bash
cd new_experiments/2026-11-22_domainnet_embeddings
~/miniconda3/envs/RAMEN/bin/python make_notebook.py
~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace 01_domainnet_embeddings.ipynb
```
Change `CLASSES`, `GRID_CLASSES`, `PER_DOMAIN` in the notebook's setup cell to look at other classes. Use from Python:
```python
from data import DomainNetEmbeddings; from analysis import *; import plots
d = DomainNetEmbeddings()
proj = Projection(d, centre_domains=True).fit(stratified_sample(d, classes=['cat', 'dog'], per_cell=200))
plots.class_panels(proj, d, ['cat', 'dog'])
```

## Quantities

$z_i$ = unit-norm zero-shot image feature, $w_c$ = unit-norm text embedding, $\hat y_i=\arg\max_c z_i^\top w_c$ (Ramen's notation, $\theta_0$).
$\mu_d$ = domain mean, $m_{c,d}$ = (class, domain) cell mean, $\bar\mu$ = domain-balanced grand mean.
- variance share $\eta^2_g=\sum_k n_k\lVert\bar z_k-\bar z\rVert^2/\sum_i\lVert z_i-\bar z\rVert^2$ for groups = domains, classes, cells
- class agreement $A_{ab}=\mathrm{mean}_c\cos(m_{c,a}-\mu_a,\,m_{c,b}-\mu_b)$ (cells with ≥ 20 images)
- centroid zero-shot accuracy: share of classes with $\arg\max_k m_{c,d}^\top w_k=c$
- neighbour composition: the 10 nearest images by cosine among all 596,006 (self excluded), split by same/other class × same/other domain;
  kNN vote with neighbours' **true labels** is an **oracle** measurement
- margin $\delta_i=z_i^\top w_{y_i}-\max_{c\ne y_i}z_i^\top w_c$

## Results (all 596,006 images; neighbours on 6,000 balanced queries)

Sanity: argmax$(z\cdot W)$ equals the cached NoAdapt prediction on 594,195 of 596,006 (99.70%; fp16 cache).

| domain | images | zero-shot acc | centroid acc (classes) | 10-NN same domain | 10-NN same class | kNN vote (oracle) | true-text cos | best-wrong cos |
|---|---|---|---|---|---|---|---|---|
| clipart | 48,833 | 68.25 | 320 / 338 (94.7%) | 62.2% | 60.8% | 69.8 | 0.276 | 0.261 |
| infograph | 53,201 | 43.90 | 292 / 335 (87.2%) | 77.7% | 33.1% | 45.7 | 0.223 | 0.230 |
| painting | 75,759 | 64.58 | 300 / 316 (94.9%) | 73.1% | 58.9% | 70.4 | 0.264 | 0.252 |
| quickdraw | 172,500 | 13.33 | 58 / 345 (16.8%) | 100.0% | 33.5% | 47.5 | 0.264 | 0.296 |
| real | 175,327 | 82.12 | 341 / 345 (98.8%) | 88.8% | 77.0% | 83.8 | 0.283 | 0.254 |
| sketch | 70,386 | 60.05 | 325 / 342 (95.0%) | 74.8% | 52.0% | 63.0 | 0.273 | 0.266 |

Variance of $z$ explained: domain 15.8%, class 18.6%, class × domain cell 43.2%. Linear domain probe: 86.9% (chance 16.7%; quickdraw
2,000 / 2,000). Cross-domain class agreement 0.63–0.79 among the five non-quickdraw domains, 0.29–0.43 with quickdraw.
Quickdraw sinks: "squiggle" predicted for 43,917 of 172,500 quickdraw images (25.5%), "envelope" for 13,833 (8.0%).

## What it means

- **Domain is as large an axis as class** in the frozen embedding (15.8% vs 18.6% of variance, 86.9% linearly decodable), and nearest
  neighbours share the domain far above chance (62–100% vs 8–29%). Retrieval by $z$ (Ramen's memory) therefore mostly returns
  same-domain images, often of other classes. Much of the domain signal is a per-domain offset: after subtracting each domain's mean
  the domains mostly mix in t-SNE and classes land in the same regions across domains (class agreement 0.63–0.79 outside quickdraw).
- **Outside quickdraw, errors are per-image scatter around roughly correct class directions**: 87–99% of class means are zero-shot
  correct although only 44–82% of images are.
- **Quickdraw is shifted and compressed**: an isolated island (100% same-domain neighbours, mean resultant length 0.92) whose class
  means are wrong for 287 of 345 classes. After centring, its classes sit roughly where they do elsewhere but squeezed toward the centre
  (class agreement 0.29–0.43). Images of a class still cluster (oracle 10-NN vote 47.5% vs 12.9% zero-shot), so a large part of the loss
  is in matching those clusters to the text embeddings ("squiggle" absorbs 25.5% of quickdraw). This matches the shift-decomposition finding
  (Γ is the large DomainNet component, mostly quickdraw).
- Caveats: zero-shot features only; t-SNE maps are qualitative; variance shares depend on domain sizes.

Next: the same figures on Ramen's adapted features (`DomainNet_..._s0_Ramen_feats.npy`, cached); per-domain prompts as a direct test
of the text-side mismatch in quickdraw.
