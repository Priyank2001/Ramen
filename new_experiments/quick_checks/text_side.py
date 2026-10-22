"""Text-side headroom (prompt templates) and near-synonym flips, from saved zero-shot / Ramen-adapted features."""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from ramen_analysis import make_experiment  # noqa: E402
from models.clip_utils import encode_text, encode_text_single  # noqa: E402

RAMEN7 = ["itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.",
          "a {} in a video game.", "art of the {}.", "a photo of the small {}."]
CLIP80 = ['a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
          'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
          'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
          'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
          'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
          'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.',
          'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.',
          'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.',
          'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.',
          'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.',
          'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.',
          'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.',
          'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
          'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.',
          'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.',
          'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.',
          'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.',
          'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.',
          'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.',
          'a photo of a small {}.', 'a tattoo of the {}.']
assert len(CLIP80) == 80
ds = sys.argv[1]
dev = 'cuda'
exp = make_experiment(ds, device=dev).load()
R = exp.results_dir + '/'
pre = exp._cache_path('X')[:-len('X.pkl')]
df = pd.read_pickle(pre + 'Ramen.pkl')
zdf = pd.read_pickle(pre + 'NoAdapt.pkl')
assert (df.domain_idx.values == zdf.domain_idx.values).all() and (df.sample_idx.values == zdf.sample_idx.values).all()
y = torch.tensor(df.label.values, device=dev)
dom = df.domain_idx.values
D = len(exp.environments)
with torch.no_grad():
    T7 = encode_text(exp.clip, exp.classes, RAMEN7).float()
    T80 = torch.stack([encode_text_single(exp.clip, exp.classes, t).float() for t in CLIP80])  # (80, C, D)
    T80avg = F.normalize(T80.mean(0), dim=-1)
feats = {'zero-shot': np.load(pre + 'NoAdapt_feats.npy'), 'Ramen': np.load(pre + 'Ramen_feats.npy')}


def dom_avg(correct):
    c = correct.cpu().numpy() if torch.is_tensor(correct) else correct
    return np.mean([c[dom == d].mean() for d in range(D)]) * 100


def preds_with(Z, T, chunk=20000):
    out = []
    for s in range(0, len(Z), chunk):
        z = torch.tensor(Z[s:s + chunk], device=dev).float()
        out.append((z @ T.T).argmax(1))
    return torch.cat(out)


print(f'===== {ds} ({len(df)} samples, {len(exp.classes)} classes)')
res = {}
for name, Z in feats.items():
    p7 = preds_with(Z, T7)
    saved = torch.tensor((df if name == 'Ramen' else zdf).pred.values, device=dev)
    print(f'[{name}] reproduces saved preds on {(p7 == saved).float().mean() * 100:.2f}% of samples; '
          f'7 templates {dom_avg(p7 == y):.2f}; 80-template average {dom_avg(preds_with(Z, T80avg) == y):.2f}')
    # per-template stats per domain: accuracy (needs labels) and mean confidence (label-free)
    acc_td = np.zeros((80, D)); conf_td = np.zeros((80, D))
    domt = torch.tensor(dom, device=dev)
    for t in range(80):
        corr, conf = [], []
        for s in range(0, len(Z), 20000):
            z = torch.tensor(Z[s:s + 20000], device=dev).float()
            pr = (100 * z @ T80[t].T).softmax(1)
            corr.append(pr.argmax(1) == y[s:s + 20000]); conf.append(pr.max(1).values)
        corr, conf = torch.cat(corr).float(), torch.cat(conf)
        for d in range(D):
            m = domt == d
            acc_td[t, d] = corr[m].mean().item() * 100; conf_td[t, d] = conf[m].mean().item()
    print(f'  oracle best single template per {exp.env_label} (uses labels): {acc_td.max(0).mean():.2f}')
    for m in (5, 10, 20):
        accs = []
        for d in range(D):
            top = np.argsort(-conf_td[:, d])[:m]
            Td = F.normalize(T80[top].mean(0), dim=-1)
            pd_ = preds_with(Z[dom == d], Td)
            accs.append((pd_ == y[torch.tensor(dom == d, device=dev)]).float().mean().item() * 100)
        print(f'  label-free: per-{exp.env_label} top-{m} templates by mean confidence (true {exp.env_label} id as grouping): {np.mean(accs):.2f}')
    # per-sample: keep the m most confident templates for this sample, average their probabilities
    for m in (10, 20):
        out = []
        for s in range(0, len(Z), 4000):
            z = torch.tensor(Z[s:s + 4000], device=dev).float()
            pr = (100 * torch.einsum('nd,tcd->ntc', z, T80)).softmax(-1)          # (n, 80, C)
            top = pr.max(-1).values.topk(m, dim=1).indices                       # (n, m)
            sel = torch.gather(pr, 1, top[..., None].expand(-1, -1, pr.shape[-1]))
            out.append(sel.mean(1).argmax(1))
        print(f'  label-free: per-sample top-{m} most confident templates: {dom_avg(torch.cat(out) == y):.2f}')
    best_tpl = [CLIP80[i] for i in acc_td.argmax(0)]
    print('  oracle best template per domain:', dict(zip(exp.environments, best_tpl)))
    res[name] = p7

# ---- near-synonym flips: is Ramen's wrong answer the true class's nearest neighbour in text space?
S = (T7 @ T7.T).cpu().numpy(); np.fill_diagonal(S, -np.inf)
rank = np.argsort(-S, axis=1)                    # rank[c, :] = classes ordered by text similarity to c
yy, pz, pr_ = df.label.values, res['zero-shot'].cpu().numpy(), res['Ramen'].cpu().numpy()
near3 = lambda a, b: (rank[a, :3] == b[:, None]).any(1)
broken = (pz == yy) & (pr_ != yy); fixed = (pz != yy) & (pr_ == yy)
zs_err = pz != yy; r_err = pr_ != yy
print(f'  near-synonym share (wrong class among the true class\'s 3 nearest text neighbours):')
print(f'    zero-shot errors {near3(yy[zs_err], pz[zs_err]).mean() * 100:.1f}% | Ramen errors {near3(yy[r_err], pr_[r_err]).mean() * 100:.1f}% '
      f'| Ramen-broken samples {near3(yy[broken], pr_[broken]).mean() * 100:.1f}% (n={broken.sum()}) '
      f'| zero-shot errors that Ramen fixed {near3(yy[fixed], pz[fixed]).mean() * 100:.1f}% (n={fixed.sum()})')
# two-stage decode: when the two methods disagree only between near-synonyms, trust zero-shot
dis = pz != pr_
for k in (1, 3, 5):
    nearzs = (rank[pr_, :k] == pz[:, None]).any(1) & dis
    final = np.where(nearzs, pz, pr_)
    print(f'    rule "if Ramen and zero-shot disagree between {k}-nearest text neighbours, keep zero-shot": '
          f'{dom_avg(final == yy):.2f} vs Ramen {dom_avg(pr_ == yy):.2f} (applies to {nearzs.mean() * 100:.1f}% of samples)')
