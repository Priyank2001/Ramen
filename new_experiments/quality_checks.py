"""
Check of the NIQE / BRISQUE port in src/models/iqa.py.

    python quality_checks.py validate PYIQA_SRC [n_per_domain]

Compares against pyiqa 0.1.16's functional NIQE / BRISQUE (themselves checked against the MATLAB releases) on
ImageNet-C images, all on the same rgb2gray input. PYIQA_SRC = the unpacked wheel (folder holding pyiqa/), e.g.
from `pip download --no-deps pyiqa==0.1.16` + unzip; only its functional code is imported.
What the scores look like on a whole stream: quality_active_set.py analyse (from the lambda = 0 run's log).
"""
import importlib
import os
import sys
import types

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import models.iqa as iqa  # noqa: E402
from models.iqa import BRISQUE, NIQE, clip_to_gray  # noqa: E402
from ramen_analysis import make_experiment  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pyiqa(src):
    """Import pyiqa's niqe / brisque functions without running pyiqa's package __init__ files (heavy deps)."""
    src = os.path.join(src, 'pyiqa')
    for name, sub in [('pyiqa', ''), ('pyiqa.utils', 'utils'), ('pyiqa.archs', 'archs')]:
        mod = types.ModuleType(name)
        mod.__path__ = [os.path.join(src, sub)]
        sys.modules[name] = mod
    stubs = {
        'pyiqa.utils.download_util': {'load_file_from_url': None},
        'pyiqa.utils.registry': {'ARCH_REGISTRY': types.SimpleNamespace(register=lambda *a, **k: (lambda f: f))},
        'pyiqa.archs.arch_util': {'get_url_from_name': lambda *a, **k: ''},
        'pyiqa.archs.fsim_arch': {'_construct_filters': None},
    }
    for name, attrs in stubs.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
    return importlib.import_module('pyiqa.archs.niqe_arch'), importlib.import_module('pyiqa.archs.brisque_arch')


def sample_batch(exp, n_per_domain, seed=0):
    """n random images per domain, CLIP-preprocessed, with their domain index."""
    rng = np.random.default_rng(seed)
    xs, dom, idx = [], [], []
    for d, ds in enumerate(exp.datasets.datasets):
        for i in np.sort(rng.choice(len(ds), n_per_domain, replace=False)):
            xs.append(ds[i][0])
            dom.append(d)
            idx.append(i)
    return torch.stack(xs), np.array(dom), np.array(idx)


def validate(pyiqa_src, n_per_domain=8):
    """
    1. The port with iqa.FLAT_TOL = 0 (the MATLAB arithmetic) against pyiqa. Images with flat regions are reported
       apart: there the scores depend on the sign of rounding noise (see iqa.mscn), so no two implementations agree.
    2. How much the flat-region fix (the default FLAT_TOL) moves the scores.
    """
    niqe_ref, brisque_ref = load_pyiqa(pyiqa_src)
    exp = make_experiment('ImageNetC5K', tta_mode='mixed', num_workers=0).load()
    x, dom, _ = sample_batch(exp, n_per_domain)
    gray = clip_to_gray(x.to(DEVICE))
    niqe, brisque = NIQE(DEVICE), BRISQUE(DEVICE)

    ref_n = niqe_ref.calculate_niqe(gray / 255, mu_pris_param=niqe.mu.unsqueeze(0),
                                    cov_pris_param=niqe.cov).cpu().numpy()
    ref_b = brisque_ref.brisque(gray / 255, test_y_channel=False, sv_coef=brisque.sv_coef,
                                sv=brisque.sv).view(-1).cpu().numpy()

    default_tol = iqa.FLAT_TOL
    iqa.FLAT_TOL = 0.0
    raw_n, raw_b = niqe(gray).cpu().numpy(), brisque(gray).cpu().numpy()
    flat = np.zeros(len(gray), dtype=bool)  # any MSCN coefficient that is rounding noise, at either scale
    for im in (gray, iqa.imresize_half(gray)):
        for padding in ('zero', 'replicate'):
            s = iqa.mscn(im, padding).abs()
            flat |= ((s < default_tol) & (s > 0)).flatten(1).any(dim=1).cpu().numpy()
    iqa.FLAT_TOL = default_tol
    fix_n, fix_b = niqe(gray).cpu().numpy(), brisque(gray).cpu().numpy()

    print(f'{len(gray)} ImageNet-C images, {flat.sum()} with flat regions (an MSCN coefficient is rounding noise)')
    for name, a, b, fixed in [('NIQE', raw_n, ref_n, fix_n), ('BRISQUE', raw_b, ref_b, fix_b)]:
        assert not (np.isnan(a).any() or np.isnan(b).any())
        d = np.abs(a - b)
        print(f'{name:8s} port vs pyiqa: max |diff| {d[~flat].max():.1e} without flat regions, '
              f'{d[flat].max() if flat.any() else 0:.2f} with; Pearson {np.corrcoef(a, b)[0, 1]:.5f} | '
              f'flat-region fix moves it by max {np.abs(fixed - a).max():.2f}, median {np.median(np.abs(fixed - a)):.1e}'
              f' | scores {fixed.min():.1f} to {fixed.max():.1f}')
    t = pd.DataFrame({'domain': np.array(exp.environments)[dom], 'NIQE': fix_n, 'NIQE pyiqa': ref_n,
                      'BRISQUE': fix_b, 'BRISQUE pyiqa': ref_b})
    print(t.groupby('domain', sort=False).mean().round(2))


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] != 'validate':
        raise SystemExit(__doc__)
    validate(sys.argv[2], *(int(a) for a in sys.argv[3:]))
