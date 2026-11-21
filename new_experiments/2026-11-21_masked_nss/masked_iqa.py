"""
NIQE natural-scene-statistics vector computed over a pixel mask (2026-11-21).

Why: the whitened NIQE vector (36-d block mean, src/models/iqa.py) helps Ramen on ImageNet-C because it identifies the
corruption. Its remaining error is probably image content (a textured object looks like noise, a smooth one like
blur). Filling the unwanted region would put the fill's own statistics into the fits, so instead the fits use only the
kept pixels: MSCN is computed on the whole image (no fill, no blur), then every mean in the (A)GGD fits runs over the
mask, and for a product of neighbouring MSCN values both pixels must be kept.

Pieces (one job each):
  PixelMask subclasses   which pixels are kept, per image, on CLIP's 224 x 224 view (boolean (B, 224, 224))
  MaskedNIQEFeatures     NIQE's per-block features restricted to a mask; all-ones mask == iqa.NIQE.features exactly
  MaskedImageQuality     ImageQuality whose 36 NIQE entries of with_features() are the masked block mean

Usage:
    iqa = MaskedImageQuality(device, CentreBox(136))      # NIQE vector from the central 136 x 136 box only
    scores, nss = iqa.with_features(x)                    # nss (B, 72): BRISQUE 36 | masked NIQE 36
Block/validity rules: NIQE keeps its 2 x 2 blocks of 96 px on the top-left 192 x 192 of the view; a block (at either
scale) with fewer than MIN_KEPT_FRAC of its pixels kept gives NaN and is ignored by the block mean.
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from models import iqa  # noqa: E402

VIEW, CROP = 224, 192  # CLIP's view; NIQE's blocks cover the top-left CROP x CROP of it
MIN_KEPT_FRAC = 0.1


# ---------------------------------------------------------------------------- which pixels are kept
class PixelMask:
    """Boolean keep-mask per image, (B, VIEW, VIEW). Subclasses set `name` and implement boxes()."""

    name = 'full'

    def __call__(self, B, device):
        return torch.ones(B, VIEW, VIEW, dtype=torch.bool, device=device)


def _box(top, left, side, B, device):
    """(B, VIEW, VIEW) mask that is True inside side x side boxes at per-image (top, left) (tensors of shape (B,))."""
    r = torch.arange(VIEW, device=device)
    rows = (r[None, :] >= top[:, None]) & (r[None, :] < (top + side)[:, None])
    cols = (r[None, :] >= left[:, None]) & (r[None, :] < (left + side)[:, None])
    return rows[:, :, None] & cols[:, None, :]


class CentreBox(PixelMask):
    """Keep only the central side x side box of the view (the 'object', by the ImageNet centring prior)."""

    def __init__(self, side):
        self.side, self.name = side, f'centre{side}'

    def __call__(self, B, device):
        top = torch.full((B,), (VIEW - self.side) // 2, device=device)
        return _box(top, top, self.side, B, device)


class Complement(PixelMask):
    """Keep everything the wrapped box mask drops (the 'background')."""

    def __init__(self, inner, name):
        self.inner, self.name = inner, name

    def __call__(self, B, device):
        return ~self.inner(B, device)


class RandomBox(PixelMask):
    """CONTROL: a side x side box at a uniformly random even position inside the NIQE crop, new for every image."""

    def __init__(self, side, seed=0):
        self.side, self.name = side, f'rand{side}'
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, B, device):
        k = (CROP - self.side) // 2 + 1  # even offsets 0, 2, ..., CROP - side
        top = 2 * torch.randint(0, k, (B,), generator=self.gen).to(device)
        left = 2 * torch.randint(0, k, (B,), generator=self.gen).to(device)
        return _box(top, left, self.side, B, device)


def make_masks(sides=(96, 136), seed=0):
    """name -> PixelMask: full, centre{s} (object), ring{s} (background), and their random-position controls."""
    masks = {'full': PixelMask()}
    for s in sides:
        masks[f'centre{s}'] = CentreBox(s)
        masks[f'ring{s}'] = Complement(CentreBox(s), f'ring{s}')
        masks[f'randbox{s}'] = RandomBox(s, seed)
        masks[f'randring{s}'] = Complement(RandomBox(s, seed + 1), f'randring{s}')
    return masks


# ---------------------------------------------------------------------------- masked fits
def aggd_fit_masked(v, m):
    """iqa.aggd_fit over the True entries of m. v, m: (N, n). Same arithmetic when m is all True."""
    l1, l2, l3 = iqa._lgamma_table(v.device)
    mf = m.to(v.dtype)
    n = mf.sum(dim=1)
    neg, pos = (v < 0) & m, (v > 0) & m
    left = torch.sqrt((v * v * neg).sum(dim=1) / neg.sum(dim=1))
    right = torch.sqrt((v * v * pos).sum(dim=1) / pos.sum(dim=1))
    g = left / right
    rhat = ((v.abs() * mf).sum(dim=1) / n).pow(2) / ((v * v * mf).sum(dim=1) / n)
    rhatnorm = rhat * (g ** 3 + 1) * (g + 1) / (g ** 2 + 1) ** 2
    return iqa._shape_lookup(rhatnorm, torch.exp(2 * l2 - l1 - l3)), left, right


def block_features_masked(blocks, keep):
    """iqa.niqe_block_features on MSCN blocks (N, b, b) restricted to keep (N, b, b) bool -> (N, 18)."""
    def betas(alpha, left, right):
        c = torch.exp((torch.lgamma(1 / alpha) - torch.lgamma(3 / alpha)) / 2)
        return left * c, right * c

    alpha, ls, rs = aggd_fit_masked(blocks.flatten(1), keep.flatten(1))
    bl, br = betas(alpha, ls, rs)
    feats = [alpha, (bl + br) / 2]
    for shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        pair = keep & torch.roll(keep, shift, dims=(1, 2))  # both pixels of the product must be kept
        alpha, ls, rs = aggd_fit_masked((blocks * torch.roll(blocks, shift, dims=(1, 2))).flatten(1), pair.flatten(1))
        bl, br = betas(alpha, ls, rs)
        mean = (br - bl) * torch.exp(torch.lgamma(2 / alpha) - torch.lgamma(1 / alpha))
        feats += [alpha, mean, bl, br]
    return torch.stack(feats, dim=1)


class MaskedNIQEFeatures:
    """iqa.NIQE.features with a pixel mask: (B, 4, 36) per-block features (NaN rows for blocks with too few pixels)."""

    def __init__(self, block=96):
        self.block = block

    def __call__(self, gray, keep):
        """gray (B, 1, 224, 224) in [0, 255]; keep (B, 224, 224) bool."""
        B = gray.shape[0]
        im = gray[..., :CROP, :CROP]
        m = keep[:, None, :CROP, :CROP].to(im.dtype)
        feats, valid = [], []
        for scale in (1, 2):
            b = self.block // scale
            nb = CROP // self.block
            blocks = iqa.mscn(im, 'replicate').unfold(2, b, b).unfold(3, b, b).reshape(-1, b, b)
            kept = (m > 0.5).unfold(2, b, b).unfold(3, b, b).reshape(-1, b, b)
            feats.append(block_features_masked(blocks, kept).view(B, nb * nb, 18))
            valid.append((kept.flatten(1).sum(dim=1) >= MIN_KEPT_FRAC * b * b).view(B, nb * nb, 1))
            im = iqa.imresize_half(im)
            m = F.avg_pool2d(m, 2)
        f = torch.cat(feats, dim=2)
        ok = valid[0] & valid[1]  # a block needs enough pixels at both scales
        return torch.where(ok, f, torch.full_like(f, float('nan')))


class MaskedImageQuality(iqa.ImageQuality):
    """ImageQuality whose NIQE entries of with_features()'s nss vector (columns 36:72) come from the masked pixels."""

    def __init__(self, device, mask):
        super().__init__(device)
        self.mask, self.masked_niqe = mask, MaskedNIQEFeatures(self.niqe.block)

    @torch.no_grad()
    def with_features(self, x):
        scores, nss = super().with_features(x)
        keep = self.mask(x.shape[0], x.device)
        fn = self.masked_niqe(iqa.clip_to_gray(x), keep)
        nss[:, 36:72] = torch.nanmean(fn, dim=1).float()
        return scores, nss
