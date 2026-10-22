"""
Offline checks behind new_experiments/vqa_paper_directions.md (ideas taken from the LIVE video quality paper,
Seshadrinathan et al., TIP 2010). Uses only cached ImageNet-C seed-0 results; no GPU.

  1. Is active_prob_T3's +0.51 over Ramen more than image-sampling noise?  (paper Sec. IV-B: significance testing)
     Paired cluster bootstrap over the 5,000 base images (each appears under all 15 corruptions) + McNemar.
  2. Does Ramen's entry weight exp(-H) have a per-corruption scale bias?  (paper Sec. III: per-subject z-scores)
     Mean zero-shot entropy per corruption, and which corruption the highest-weight retrieved entry comes from.
  3. Is Ramen's gain monotone in how hard the domain is?  (paper Sec. IV-A: SROCC)
     Spearman correlation, across domains, between zero-shot accuracy and Ramen's gain, on all 4 datasets.
  4. Does a no-reference image-quality signal predict where adaptation helps?  (paper's lab: natural scene statistics)
     MSCN statistics per image; AUROC for "fixed" vs "broken" samples, within each corruption, against zero-shot
     confidence; and the best post-hoc "zero-shot or Ramen" gate for each signal (tuned on the test set: an upper bound).

Usage (from the repo root): ~/miniconda3/envs/RAMEN/bin/python new_experiments/quick_checks/vqa_checks.py
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import binomtest, kurtosis, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from datasets import get_dataset_class  # noqa: E402

R = os.path.join(ROOT, 'new_experiments', 'results') + '/'
IMC = 'ImageNetC5K_clip_vitbase16_mixed_nall_bs100_s0_{}'
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)

base = pd.read_pickle(R + IMC.format('NoAdapt') + '.pkl')
ramen = pd.read_pickle(R + IMC.format('Ramen') + '.pkl')
t3 = pd.read_pickle(R + IMC.format('active_prob_T3') + '.pkl')
for d in (ramen, t3):
    assert (d.sample_idx.values == base.sample_idx.values).all() and (d.domain_idx.values == base.domain_idx.values).all()
domains = list(base.domain.cat.categories)

# ---------------------------------------------------------------------------- 1. significance
print('== 1. active_prob_T3 vs Ramen, paired over the same stream')
diff = t3.correct.values.astype(float) - ramen.correct.values
cluster = pd.Series(diff).groupby(base.sample_idx.values).sum().values  # one cluster per base image
rng = np.random.default_rng(0)
boot = np.array([cluster[rng.integers(0, len(cluster), len(cluster))].sum() for _ in range(5000)]) / len(diff) * 100
b = int((t3.correct.values & ~ramen.correct.values).sum())
c = int((~t3.correct.values & ramen.correct.values).sum())
print(f'gain {diff.mean() * 100:+.2f} pts | cluster bootstrap 95% CI [{np.percentile(boot, 2.5):+.2f}, '
      f'{np.percentile(boot, 97.5):+.2f}] | McNemar: T3 only right {b}, Ramen only right {c}, '
      f'p = {binomtest(b, b + c).pvalue:.1e}')
print('   (covers which images were drawn, not stream order: seeds 1-4 are still needed for that)')

# ---------------------------------------------------------------------------- 2. scale bias of exp(-H)
print('\n== 2. zero-shot entropy per corruption (Ramen weights each memory entry by exp(-H))')
H = base.groupby('domain', observed=True).entropy
scale = pd.DataFrame({'mean H': H.mean(), 'mean exp(-H)': base.assign(w=np.exp(-base.entropy)).groupby(
    'domain', observed=True).w.mean()})
for name in ('active_all', 'active_prob_T3'):
    ids = np.concatenate(pd.read_pickle(R + IMC.format(name) + '_stats.pkl')['support_ids'])
    top = ids[:, 0]
    ok = top >= 0
    src = pd.Series(base.domain.values[top[ok]]).value_counts(normalize=True) * 100
    scale[f'{name}: share of top entries %'] = src.reindex(domains).values
    print(f'{name}: top-weighted entry from the query\'s own corruption: '
          f'{(base.domain_idx.values[top[ok]] == base.domain_idx.values[ok]).mean() * 100:.1f}% (chance 6.7%)')
print(scale.round(2).sort_values('mean H'))

# ---------------------------------------------------------------------------- 3. SROCC of gain vs difficulty
print('\n== 3. Spearman correlation across domains: zero-shot accuracy vs gain')
pats = {'CIFAR10C': 'CIFAR10C_clip_vitbase32_mixed_nall_bs100_s0_{}',
        'CIFAR100C': 'CIFAR100C_clip_vitbase16_mixed_nall_bs100_s0_{}',
        'ImageNetC5K': IMC, 'DomainNet': 'DomainNet_clip_vitbase32_mixed_nall_bs100_s0_{}'}
for ds, pat in pats.items():
    acc = {a: pd.read_pickle(R + pat.format(a) + '.pkl').groupby('domain_idx').correct.mean() * 100
           for a in ('NoAdapt', 'Ramen')}
    rho, p = spearmanr(acc['NoAdapt'], acc['Ramen'] - acc['NoAdapt'])
    print(f'{ds:12s} {len(acc["NoAdapt"]):2d} domains: SROCC(zero-shot acc, Ramen gain) = {rho:+.2f} (p = {p:.3f})')
acc_t3 = t3.groupby('domain_idx').correct.mean() * 100
acc_r = ramen.groupby('domain_idx').correct.mean() * 100
acc_z = base.groupby('domain_idx').correct.mean() * 100
rho, p = spearmanr(acc_z, acc_t3 - acc_r)
print(f'ImageNetC5K: SROCC(zero-shot acc, prob_T3 gain over Ramen) = {rho:+.2f} (p = {p:.3f})')


# ---------------------------------------------------------------------------- 4. natural scene statistics
def nss_features(path):
    """MSCN statistics (BRISQUE-style, Mittal/Moorthy/Bovik 2012) of the grayscale image."""
    with Image.open(path) as img:
        x = np.asarray(img.convert('L'), dtype=np.float64)
    mu = gaussian_filter(x, 7 / 6)
    sigma = np.sqrt(np.abs(gaussian_filter(x * x, 7 / 6) - mu * mu))
    mscn = (x - mu) / (sigma + 1)
    return [mscn.var(), kurtosis(mscn, axis=None), sigma.mean(), (mscn[:, :-1] * mscn[:, 1:]).mean()]


feat_path = R + 'ImageNetC5K_nss_features.npy'  # row i <-> results row i (stream order of seed 0)
if not os.path.exists(feat_path):
    ds = get_dataset_class('ImageNetC5K')(root=os.path.expanduser('~/data'), severity=5, transform=None)
    paths = [ds[d].samples[s][0] for d, s in zip(base.domain_idx.values, base.sample_idx.values)]
    with Pool(48) as pool:
        np.save(feat_path, np.array(pool.map(nss_features, paths, chunksize=200), dtype=np.float32))
F = pd.DataFrame(np.load(feat_path), columns=['MSCN variance', 'MSCN kurtosis', 'local contrast', 'neighbour corr.'])
signals = {**{c: F[c].values for c in F}, 'zero-shot confidence': base.conf.values,
           'zero-shot entropy': base.entropy.values}

print('\n== 4. which signal tells "Ramen fixes it" from "Ramen breaks it"?')
fixed = ~base.correct.values & ramen.correct.values
broken = base.correct.values & ~ramen.correct.values
rows = {}
for name, s in signals.items():
    aucs = []
    for d in range(len(domains)):
        m = (base.domain_idx.values == d) & (fixed | broken)
        aucs.append(roc_auc_score(fixed[m], s[m]))
    m = fixed | broken
    # Gate: keep zero-shot on the q-quantile of samples at one end of the signal, Ramen elsewhere (both directions).
    best = -np.inf
    for q in np.linspace(0.05, 0.95, 19):
        for sign in (1, -1):
            keep_zs = sign * s >= np.quantile(sign * s, 1 - q)
            correct = np.where(keep_zs, base.correct.values, ramen.correct.values)
            best = max(best, correct.mean() * 100)
    rows[name] = {'AUROC within corruption (mean)': np.mean(aucs), 'AUROC pooled': roc_auc_score(fixed[m], s[m]),
                  'best gate vs Ramen (pts, test-tuned)': best - ramen.correct.mean() * 100}
print(f'{fixed.sum()} fixed, {broken.sum()} broken; AUROC 0.5 = no signal, far from 0.5 either way = signal')
print(pd.DataFrame(rows).T.round(3))
print('\nmean NSS features per corruption (sorted by Ramen gain):')
g = F.assign(domain=base.domain.values, gain=ramen.correct.values.astype(float) - base.correct.values)
print(g.groupby('domain', observed=True).mean().sort_values('gain').round(3))
for c in F:
    per = g.groupby('domain', observed=True).mean()
    print(f'SROCC across corruptions ({c}, Ramen gain) = {spearmanr(per[c], per.gain)[0]:+.2f}')
