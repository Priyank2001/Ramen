# GMM labels (2026-11-18)

**Question.** Can a class-conditional mixture fitted by EM to the *unlabelled* CLIP image embeddings, starting from
the text prototypes, give labels much better than zero-shot? If so, it would be a label-free source for Ramen's
filing / CE value (the label-value oracle showed better labels help; every label-free target so far failed).

**Pre-registered (before results):** after 10 EM iterations with a uniform prior, label accuracy ≥ zero-shot + 10
points (pooled) on DomainNet or ImageNet-C, with < 5% of zero-shot-right images broken. Otherwise weak (gain > floor) / null / hurts.

Offline, CPU only, transductive (EM sees all images at once). That is the most favourable setting for a mixture, so a
failure here rules out the online version.

| file | job |
|---|---|
| `gmm.py` | the models and EM: `SphericalVMF`, `SharedCovGaussian`, `UniformPrior`, `LearnedPrior`, `EM`, `EMTrace` |
| `gmm_configs.py` | `DATASETS` (model, noise floor), `VARIANTS`, the pre-registered decision constants |
| `analysis.py` | `LabelScore` (fixed / broken / rescue / break rates, verdict), `component_alignment` (NMI with class / domain) |
| `gmm_experiment.py` | `GMMLabelExperiment`: loads cached zero-shot inputs, runs and caches a variant, sanity check, tables |
| `run_gmm.py` | CLI: sanity check + all variants for one dataset / seed |
| `test_gmm.py` | 22 synthetic unit tests (stdlib `unittest`; pytest is not in the env) |
| `verify.py` | correctness checks on the real cached data |
| `plots.py` | figures (gain dot plot, fixed/broken bars, accuracy / log-likelihood / NMI over iterations) |
| `interpretation.py` | the interpretation text shared by the notebook |
| `make_notebook.py`, `01_gmm_labels.ipynb` | notebook (executed) |
| `logs/` | run logs, `sanity_*.json`, `verify.log`, `unittest.log` |

Run: `python run_gmm.py <DomainNet|ImageNetC5K|EuroSAT> 0`, then `python verify.py > logs/verify.log` and
`python -m unittest -v test_gmm`. Outputs: `results/<prefix>_gmm_<variant>.pkl` (per image), `_iters.pkl`
(per iteration), `_preds.npy` (labels at every iteration), `_means.npy` (final means).
Nothing in `src/`, `cfg/` or the shared harness was edited (`verify.py` check 6).

## Math

Zero-shot: $z_i$ unit-norm image embedding, $w_c$ unit-norm text embedding, $p_{ic}=\mathrm{softmax}_c(100\,z_i^\top w_c)$,
$\hat y_i=\arg\max_c p_{ic}$. This is exactly the posterior of a von Mises-Fisher mixture with means $w_c$, $\kappa=100$
and a uniform prior. So EM "initialised at the text prototypes" starts from $R^{(0)}=P$ (cached `NoAdapt_probs`).

EM, $t=1..10$, with $N_c=\sum_i r_{ic}$, $S_c=\sum_i r_{ic}z_i$ from the previous responsibilities:
- `vmf`: $\mu_c=S_c/\lVert S_c\rVert$, $\kappa=100$ fixed; $\log p(z\mid c)=\kappa\mu_c^\top z$ + const.
- `lda`: $\mu_c=S_c/N_c$, $S=\frac1N(\sum_i z_iz_i^\top-\sum_c S_cS_c^\top/N_c)$, $\Sigma=0.9\,S+0.1\,\frac{\mathrm{tr}S}{d}I$; $\log p(z\mid c)=\log\mathcal N(z;\mu_c,\Sigma)$.
- prior: uniform $1/C$, or `_prior`: $\pi_c=N_c/N$.
- E-step: $r_{ic}\propto\pi_c\,p(z_i\mid c)$; label $\arg\max_c r_{ic}$.
- **ORACLE** `vmf_oracle` / `lda_oracle`: $R^{(0)}=$ one-hot **true labels**, one M + E step. This is the family's ceiling when fitted on the test set with labels.

## Results (seed 0; EM is order-free, so seed-independent; pooled accuracy, same images)

| dataset | zero-shot | `vmf` | `vmf_prior` | `lda` | `lda_prior` | `vmf_oracle` | `lda_oracle` |
|---|---|---|---|---|---|---|---|
| DomainNet (596,006; 345 cl.; 6 dom.) | 52.83 | 27.61 (−25.22) | 26.86 | 41.73 (−11.10) | 41.70 | 47.75 (−5.08) | 55.82 (+2.99) |
| ImageNet-C 5K (75,000; 1,000 cl.; 15 corr.) | 27.29 | 6.97 (−20.32) | 6.50 | 23.29 (−4.00) | 23.26 | 43.77 (+16.48) | 63.05 (+35.76) |
| EuroSAT (27,000; 10 cl.; 1 dom.) | 48.07 | 52.80 (+4.73) | 42.55 | **63.39 (+15.33)** | 61.17 | 77.51 | 94.99 |

Domain-averaged (headline convention): DomainNet zs 55.37 → `lda` 43.01, `lda_oracle` 57.96; ImageNet-C and
EuroSAT are balanced over domains, so domain-averaged = pooled.

Fixed / broken vs zero-shot for `lda`: DomainNet 17,614 of 281,161 zs-wrong fixed (6.3%) / 83,742 of 314,845
zs-right broken (26.6%); ImageNet-C 4,807 of 54,532 (8.8%) / 7,804 of 20,468 (38.1%); EuroSAT 5,438 of 14,022
(38.8%) / 1,300 of 12,978 (10.0%).

## Why it should help / what happened

- **Should help:** if images of one class clump in embedding space, EM moves each class model onto its clump and
  corrects zero-shot's errors without labels. **Happened (DomainNet, ImageNet-C):** log-likelihood rises at every
  iteration (`verify.log`) while accuracy falls. NMI of the components with the domain rises (DomainNet vMF
  0.10 → 0.28, LDA → 0.19; ImageNet-C vMF 0.11 → 0.42, LDA → 0.20). The densest structure in these mixed streams
  is domain or corruption, so EM fits that.
  For LDA on ImageNet-C, class NMI stays flat (0.56 → 0.57) while accuracy falls. The groupings survive but get the
  wrong class names, consistent with label switching (synthetic test), not measured directly here.
- **EuroSAT (one domain):** clusters are classes, so EM helps (+15.33, 5,438 fixed), but it breaks 1,300 correct
  images (10.0% > the 5% bar).
- **Oracles:** one Gaussian per class fitted *with true labels* reaches only +2.99 on DomainNet (vMF −5.08).
  Hypothesis, untested: a class spans six styles, so a single mode per class cannot represent it, while the text
  prototype is style-neutral. On ImageNet-C the ceiling is high (+35.76), so the failure there is EM's, not the family's.
- **Learned prior:** never helps (as in 2026-10-26 prior estimation).

## Verdicts (pre-registered rule, iteration 10)

- `vmf`, `vmf_prior`, `lda`, `lda_prior` on DomainNet and ImageNet-C: **hurts**.
- EuroSAT `lda` +15.33, `lda_prior` +13.11: **weak** (break rate 10.0% / 11.6% > 5%). `vmf` +4.73: weak. `vmf_prior`: hurts.
- Hypothesis "a label-free class GMM gives labels ≥ 10 points better than zero-shot": **not supported**. Raw-embedding
  class mixtures are a dead end on multi-domain data.

## Correctness evidence

- `test_gmm.py` (22 tests, all pass): softmax, densities and the LDA posterior against scipy / sklearn, M-step
  against brute force, EM monotone, planted-mixture recovery (and its label-switching failure mode when the start
  misnames clusters), chunking and row-order invariance, scoring arithmetic.
- `verify.py` on real data: iteration 0 == cached NoAdapt on 100.00% of images (all datasets). Our
  $\mathrm{softmax}(100\,z^\top W)$ disagrees with the cached argmax on 0.30% / 0.57% / 0.42% of images, all
  near-ties (median cached top-2 margin 0.0000). vMF log-likelihood never decreases. One-hot LDA equals sklearn on all
  27,000 EuroSAT images (max |Δp| 3e-8, 100% argmax). Table accuracies equal a numpy recount. No shared file modified.

## Caveats and next step

Transductive; one κ (100), one shrinkage (0.1), one component per class. Untested: domain-factored mixtures (subtract
a domain or cluster mean, then fit classes), several components per class. Next, if anything: `lda_oracle` on
domain-centred embeddings as a ceiling check. On EuroSAT alone, `lda` labels as Ramen's CE target (a small-label-space
result only).
