"""
Checks for masked_iqa.py (CPU or GPU; needs real images).

    python test_masked_iqa.py       # all-ones mask == iqa.NIQE.features; masks behave as defined
The full-mask equality is the 'off setting == baseline' sanity check: if it fails, every masked result is a bug.
"""
import sys

import torch

from masked_iqa import CROP, VIEW, MaskedImageQuality, MaskedNIQEFeatures, PixelMask, make_masks, iqa

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
# natural-ish test images: smooth noise at several scales, CLIP-normalised range
x = sum(torch.nn.functional.interpolate(torch.randn(8, 3, s, s), size=VIEW, mode='bicubic') / k
        for s, k in ((7, 1), (28, 2), (112, 4))).to(dev)
x[:2, :, 40:120, 60:160] = 0.3  # a flat region (the FLAT_TOL path)

gray = iqa.clip_to_gray(x)
ref = iqa.NIQE(dev).features(gray)
mine = MaskedNIQEFeatures()(gray, PixelMask()(8, dev))
err = (ref - mine).abs().max().item()
print(f'full mask vs iqa.NIQE.features: max |diff| {err:.2e} (NaN agree: {(ref.isnan() == mine.isnan()).all().item()})')
assert err < 1e-9 and (ref.isnan() == mine.isnan()).all()

q = iqa.ImageQuality(dev)
s0, n0 = q.with_features(x)
s1, n1 = MaskedImageQuality(dev, PixelMask()).with_features(x)
assert torch.allclose(n0, n1, atol=1e-6, equal_nan=True) and torch.allclose(s0, s1, equal_nan=True)
print('MaskedImageQuality(full).with_features == ImageQuality.with_features')

for name, m in make_masks().items():
    k = m(64, dev)
    frac = k[:, :CROP, :CROP].float().mean().item()
    sides = k.flatten(1).sum(1).unique().numel()
    print(f'{name:12s} kept {frac:.3f} of the crop; per-image kept-count values {sides}')
    if name.startswith('rand'):
        assert sides == 1, 'random control must keep the same pixel count for every image'
f = MaskedNIQEFeatures()(gray, make_masks()['centre96'](8, dev))
print('centre96 vector finite:', torch.isfinite(torch.nanmean(f, dim=1)).all().item())
print('TESTS OK')
