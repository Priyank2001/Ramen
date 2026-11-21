"""
Stage 1a: masked NIQE vectors (36-d block mean) for every ImageNetC5K image, for every mask variant in one pass
(GPU, ~minutes; needs the dataset). Saved as results/ImageNetC5K_masked_niqe.npz: one (n_domains, n, 36) array per
variant, indexed [domain, sample] like results/ImageNetC5K_nss_features.npz in the shared folder.

    CUDA_VISIBLE_DEVICES=0 python extract.py            # writes the npz, then checks 'full' against the shared cache
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from active_seeds import NUM_WORKERS  # noqa: E402
from masked_iqa import MaskedNIQEFeatures, make_masks, iqa  # noqa: E402
from ramen_analysis import make_experiment  # noqa: E402

DATASET = 'ImageNetC5K'
OUT = os.path.join(HERE, 'results', f'{DATASET}_masked_niqe.npz')
DEVICE = torch.device('cuda')


def main():
    exp = make_experiment(DATASET, tta_mode='mixed', batch_size=100, n_per_domain=None, seed=0,
                          num_workers=NUM_WORKERS[DATASET]).load()
    masks = make_masks()
    feat = MaskedNIQEFeatures()
    n = max(len(ds) for ds in exp.datasets.datasets)
    out = {k: np.full((len(exp.environments), n, 36), np.nan, dtype=np.float32) for k in masks}
    for d, ds in enumerate(exp.datasets.datasets):
        loader = DataLoader(ds, batch_size=250, shuffle=False, num_workers=exp.num_workers, pin_memory=True)
        i = 0
        for x, _ in tqdm(loader, desc=exp.environments[d]):
            gray = iqa.clip_to_gray(x.to(DEVICE))
            for k, m in masks.items():
                f = feat(gray, m(len(x), DEVICE))
                out[k][d, i:i + len(x)] = torch.nanmean(f, dim=1).float().cpu().numpy()
            i += len(x)
    np.savez(OUT, environments=np.array(exp.environments), **out)
    print('saved', OUT)
    for k, v in out.items():
        print(f'{k:12s} NaN rows: {np.isnan(v).any(axis=2).sum():,} of {v.shape[0] * v.shape[1]:,}')
    shared = np.load(os.path.join(exp.results_dir, f'{DATASET}_nss_features.npz'))['nss'][:, :, 36:72]
    diff = np.nanmax(np.abs(out['full'] - shared))
    print(f"'full' vs shared cache of the NIQE vector: max |diff| {diff:.2e}")
    assert diff < 1e-4, 'full-mask vector does not reproduce the cached NIQE vector'


if __name__ == '__main__':
    main()
