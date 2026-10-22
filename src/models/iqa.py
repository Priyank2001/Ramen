"""
No-reference image quality, batched on the GPU: NIQE and BRISQUE (blindqa.pdf, slides 8-17).

Ports of the authors' MATLAB releases (LIVE, UT Austin):
  BRISQUE  Mittal, Moorthy, Bovik, "No-Reference Image Quality Assessment in the Spatial Domain", TIP 2012.
           brisque_feature.m (36 NSS features: GGD of the MSCN coefficients + AGGD of their 4 paired products, at
           2 scales), then svm-scale with `allrange` and svm-predict with the LIVE-trained epsilon-SVR `allmodel`.
  NIQE     Mittal, Soundararajan, Bovik, "Making a 'Completely Blind' Image Quality Analyzer", SPL 2013.
           computequality.m / computefeature.m / estimateaggdparam.m: the same NSS features on 96x96 blocks, an MVG
           fit to them, and its distance to the MVG of sharp patches of pristine natural images.
Both scores are higher = worse. They are computed on MATLAB's rgb2gray of the 8-bit image, in [0, 255].

Model files in iqa_params/ (the LIVE releases, as mirrored in pyiqa's v0.1-weights):
  niqe_modelparameters.mat  mu_prisparam (1, 36), cov_prisparam (36, 36)                md5 8601a29b769c9c83...
  brisque_svm_weights.pth   (sv_coef (774,), sv (774, 36)); RBF gamma 0.05, rho -153.591  md5 d5dbd8eae2933ae0...

Here the input is CLIP's preprocessed view of each image (224 x 224, what the model sees), so at the first scale
NIQE sees 2 x 2 blocks of 96 x 96. One deliberate deviation from the MATLAB arithmetic: flat regions get MSCN
coefficients of exactly 0 (see mscn()). new_experiments/quality_checks.py validates both against pyiqa.
"""
import math
import os
from functools import lru_cache

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

PARAMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iqa_params')

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
RGB2GRAY = (0.298936021293775, 0.587043074451121, 0.114020904255103)  # MATLAB rgb2gray

# MATLAB's grid for the moment-matching estimate of the (A)GGD shape: gam = 0.2:0.001:10
_GAM = 0.2 + 0.001 * np.arange(9801)

# BRISQUE `allrange`: per-feature [min, max] that svm-scale maps to [-1, 1]
BRISQUE_RANGE = np.array([
    [0.338, 10], [0.017204, 0.806612], [0.236, 1.642], [-0.123884, 0.20293], [0.000155, 0.712298],
    [0.001122, 0.470257], [0.244, 1.641], [-0.123586, 0.179083], [0.000152, 0.710456], [0.000975, 0.470984],
    [0.249, 1.555], [-0.135687, 0.100858], [0.000174, 0.684173], [0.000913, 0.534174], [0.258, 1.561],
    [-0.143408, 0.100486], [0.000179, 0.685696], [0.000888, 0.536508], [0.471, 3.264], [0.012809, 0.703171],
    [0.218, 1.046], [-0.094876, 0.187459], [1.5e-05, 0.442057], [0.001272, 0.40803], [0.222, 1.042],
    [-0.115772, 0.162604], [1.6e-05, 0.444362], [0.001374, 0.40243], [0.227, 0.996], [-0.117188, 0.098323],
    [3e-05, 0.531903], [0.001122, 0.369589], [0.228, 0.99], [-0.12243, 0.098658], [2.8e-05, 0.530092],
    [0.001118, 0.370399]])
BRISQUE_GAMMA, BRISQUE_RHO = 0.05, -153.591

FLAT_TOL = 1e-8  # see mscn(); 0 reproduces the MATLAB arithmetic (used by quality_checks.py validate)


# ---------------------------------------------------------------------------- MATLAB building blocks
def clip_to_gray(x):
    """CLIP-normalised RGB (B, 3, H, W) -> rgb2gray of the 8-bit image as float64 in [0, 255], (B, 1, H, W)."""
    mean = torch.tensor(CLIP_MEAN, dtype=torch.float64, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=torch.float64, device=x.device).view(1, 3, 1, 1)
    rgb = torch.floor((x.double() * std + mean) * 255 + 0.5).clamp(0, 255)  # back to the uint8 pixels
    w = torch.tensor(RGB2GRAY, dtype=torch.float64, device=x.device).view(1, 3, 1, 1)
    return torch.floor((rgb * w).sum(dim=1, keepdim=True) + 0.5)  # rgb2gray rounds to uint8


@lru_cache(maxsize=None)
def _gauss_window(device):
    """fspecial('gaussian', 7, 7/6), normalised to sum 1."""
    r = np.arange(-3, 4)
    h = np.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * (7 / 6) ** 2))
    h[h < np.finfo(float).eps * h.max()] = 0
    return torch.tensor(h / h.sum(), dtype=torch.float64, device=device).view(1, 1, 7, 7)


def mscn(im, padding):
    """
    Mean subtracted contrast normalised coefficients (slide 8): (I - mu) / (sigma + 1) with a 7x7 Gaussian window.
    padding: 'replicate' (NIQE uses imfilter(..., 'replicate')) or 'zero' (BRISQUE uses filter2(..., 'same')).

    In a flat region I - mu is exactly 0, but the filter's rounding leaves +-1e-14 of noise. The (A)GGD fits count
    samples by sign, so that noise moves BRISQUE by up to 12 points on saturated images (snow) in a way that depends
    on the implementation. |I - mu| < FLAT_TOL is therefore set to 0; real structure is >= ~1e-5 on 8-bit images.
    """
    w = _gauss_window(im.device)
    p = F.pad(im, (3, 3, 3, 3), mode='replicate') if padding == 'replicate' else F.pad(im, (3, 3, 3, 3))
    mu = F.conv2d(p, w)
    sigma = torch.sqrt((F.conv2d(p * p, w) - mu * mu).abs())
    d = im - mu
    d = torch.where(d.abs() < FLAT_TOL, torch.zeros_like(d), d)
    return d / (sigma + 1)


def _matlab_cubic(x):
    ax = np.abs(x)
    return ((1.5 * ax ** 3 - 2.5 * ax ** 2 + 1) * (ax <= 1)
            + (-0.5 * ax ** 3 + 2.5 * ax ** 2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2)))


@lru_cache(maxsize=None)
def _half_matrix(n, device):
    """(ceil(n/2), n) matrix of MATLAB imresize(., 0.5): antialiased bicubic, mirrored borders (contributions())."""
    scale, width = 0.5, 4 / 0.5
    out = math.ceil(n * scale)
    u = np.arange(1, out + 1) / scale + 0.5 * (1 - 1 / scale)
    idx = np.floor(u - width / 2)[:, None] + np.arange(int(np.ceil(width)) + 2)[None]  # 1-based
    wts = scale * _matlab_cubic(scale * (u[:, None] - idx))
    wts /= wts.sum(axis=1, keepdims=True)
    aux = np.concatenate([np.arange(1, n + 1), np.arange(n, 0, -1)])
    idx = aux[(idx.astype(int) - 1) % len(aux)] - 1
    M = np.zeros((out, n))
    np.add.at(M, (np.repeat(np.arange(out), idx.shape[1]), idx.ravel()), wts.ravel())
    return torch.tensor(M, dtype=torch.float64, device=device)


def imresize_half(im):
    """MATLAB imresize(im, 0.5) for a double image (B, 1, H, W) (no rounding)."""
    H, W = im.shape[-2:]
    return _half_matrix(H, im.device) @ im @ _half_matrix(W, im.device).T


def _shape_lookup(target, table):
    """MATLAB's [~, i] = min((table - target).^2); gam(i). target (N,), table (9801,). NaN target -> gam(1)."""
    pos = (table[None, :] - target[:, None]).abs().argmin(dim=1)
    pos[torch.isnan(target)] = 0  # MATLAB's min over an all-NaN vector returns index 1
    return torch.tensor(_GAM, dtype=target.dtype, device=target.device)[pos]


def _lgamma_table(device):
    g = torch.tensor(_GAM, dtype=torch.float64, device=device)
    return torch.lgamma(1 / g), torch.lgamma(2 / g), torch.lgamma(3 / g)


def ggd_fit(v):
    """estimateggdparam.m. v: (N, n) zero-mean samples -> shape alpha (N,), variance sigma^2 (N,)."""
    l1, l2, l3 = _lgamma_table(v.device)
    sigma_sq = v.pow(2).mean(dim=1)
    rho = sigma_sq / v.abs().mean(dim=1).pow(2)
    return _shape_lookup(rho, torch.exp(l1 + l3 - 2 * l2)), sigma_sq


def aggd_fit(v):
    """estimateaggdparam.m. v: (N, n) -> shape alpha, left std, right std (each (N,)); zeros count on neither side."""
    l1, l2, l3 = _lgamma_table(v.device)
    neg, pos = v < 0, v > 0
    left = torch.sqrt((v * v * neg).sum(dim=1) / neg.sum(dim=1))
    right = torch.sqrt((v * v * pos).sum(dim=1) / pos.sum(dim=1))
    g = left / right
    rhat = v.abs().mean(dim=1).pow(2) / v.pow(2).mean(dim=1)
    rhatnorm = rhat * (g ** 3 + 1) * (g + 1) / (g ** 2 + 1) ** 2
    return _shape_lookup(rhatnorm, torch.exp(2 * l2 - l1 - l3)), left, right


# ---------------------------------------------------------------------------- BRISQUE
def brisque_features(gray):
    """brisque_feature.m on a batch: (B, 1, H, W) in [0, 255] -> (B, 36)."""
    feats, im = [], gray
    for _ in range(2):
        s = mscn(im, 'zero')
        alpha, var = ggd_fit(s.flatten(1))
        feats += [alpha, var]
        for shift in ((0, 1), (1, 0), (1, 1), (-1, 1)):
            alpha, ls, rs = aggd_fit((s * torch.roll(s, shift, dims=(2, 3))).flatten(1))
            eta = (rs - ls) * torch.exp(torch.lgamma(2 / alpha) - (torch.lgamma(1 / alpha) + torch.lgamma(3 / alpha)) / 2)
            feats += [alpha, eta, ls ** 2, rs ** 2]
        im = imresize_half(im)
    return torch.stack(feats, dim=1)


class BRISQUE:
    """BRISQUE score of a batch of gray images (B, 1, H, W) in [0, 255] -> (B,). Roughly 0 (best) to 100."""

    def __init__(self, device):
        sv_coef, sv = torch.load(os.path.join(PARAMS_DIR, 'brisque_svm_weights.pth'), weights_only=False)
        self.sv_coef = sv_coef.to(device=device, dtype=torch.float64)
        self.sv = sv.to(device=device, dtype=torch.float64)
        rng = torch.tensor(BRISQUE_RANGE, dtype=torch.float64, device=device)
        self.lo, self.hi = rng[:, 0], rng[:, 1]

    def __call__(self, gray):
        return self.score(brisque_features(gray))

    def score(self, f):
        """BRISQUE score from brisque_features() (B, 36)."""
        f = -1 + 2 * (f - self.lo) / (self.hi - self.lo)  # svm-scale -r allrange
        k = torch.exp(-BRISQUE_GAMMA * torch.cdist(f, self.sv).pow(2))
        return k @ self.sv_coef - BRISQUE_RHO


# ---------------------------------------------------------------------------- NIQE
def niqe_block_features(blocks):
    """computefeature.m on a stack of MSCN blocks (N, b, b) -> (N, 18)."""
    def betas(alpha, left, right):
        c = torch.exp((torch.lgamma(1 / alpha) - torch.lgamma(3 / alpha)) / 2)
        return left * c, right * c

    alpha, ls, rs = aggd_fit(blocks.flatten(1))
    bl, br = betas(alpha, ls, rs)
    feats = [alpha, (bl + br) / 2]
    for shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        alpha, ls, rs = aggd_fit((blocks * torch.roll(blocks, shift, dims=(1, 2))).flatten(1))
        bl, br = betas(alpha, ls, rs)
        mean = (br - bl) * torch.exp(torch.lgamma(2 / alpha) - torch.lgamma(1 / alpha))
        feats += [alpha, mean, bl, br]
    return torch.stack(feats, dim=1)


class NIQE:
    """NIQE score of a batch of gray images (B, 1, H, W) in [0, 255] -> (B,); NaN if fewer than 2 usable blocks."""

    def __init__(self, device, block=96):
        p = scipy.io.loadmat(os.path.join(PARAMS_DIR, 'niqe_modelparameters.mat'))
        self.mu = torch.tensor(p['mu_prisparam'], dtype=torch.float64, device=device).view(-1)
        self.cov = torch.tensor(p['cov_prisparam'], dtype=torch.float64, device=device)
        # positive definite, so (cov + any covariance) / 2 is invertible and solve() equals MATLAB's pinv()
        assert torch.linalg.eigvalsh(self.cov).min() > 0
        self.block = block

    def features(self, gray):
        """Per-block features (B, n_blocks, 36): 18 at full resolution, 18 at half (blocks of block/2)."""
        B, _, H, W = gray.shape
        nh, nw = H // self.block, W // self.block
        assert nh * nw >= 2, f'NIQE needs at least 2 blocks of {self.block} px, got a {H}x{W} image'
        im = gray[..., :nh * self.block, :nw * self.block]
        feats = []
        for scale in (1, 2):
            b = self.block // scale
            blocks = mscn(im, 'replicate').unfold(2, b, b).unfold(3, b, b)  # (B, 1, nh, nw, b, b)
            feats.append(niqe_block_features(blocks.reshape(-1, b, b)).view(B, nh * nw, 18))
            im = imresize_half(im)
        return torch.cat(feats, dim=2)

    def __call__(self, gray):
        return self.score(self.features(gray))

    def score(self, f):
        """NIQE score from the per-block features() (B, n_blocks, 36)."""
        nan_row = torch.isnan(f).any(dim=2, keepdim=True)  # MATLAB nancov drops rows with a NaN ...
        mu = torch.nanmean(f, dim=1)  # ... while nanmean ignores NaNs column by column
        x = torch.where(nan_row, torch.zeros_like(f), f)
        n = (~nan_row).sum(dim=1, keepdim=True).to(f.dtype)  # (B, 1, 1)
        xc = torch.where(nan_row, torch.zeros_like(f), x - x.sum(dim=1, keepdim=True) / n)
        ok = n.view(-1) >= 2
        cov = torch.where(ok.view(-1, 1, 1), xc.transpose(1, 2) @ xc / (n - 1).clamp_min(1), torch.zeros_like(self.cov))
        d = torch.nan_to_num(self.mu - mu).unsqueeze(2)  # (B, 36, 1)
        q = torch.sqrt((d.transpose(1, 2) @ torch.linalg.solve((self.cov + cov) / 2, d)).view(-1))
        return torch.where(ok, q, torch.full_like(q, float('nan')))


# ---------------------------------------------------------------------------- both
class ImageQuality:
    """
    NIQE and BRISQUE of a batch of CLIP-preprocessed images.
    quality(x) -> (B, 2) float32, columns METRICS = ('niqe', 'brisque').
    """

    METRICS = ('niqe', 'brisque')

    def __init__(self, device):
        self.niqe = NIQE(device)
        self.brisque = BRISQUE(device)

    @torch.no_grad()
    def __call__(self, x):
        gray = clip_to_gray(x)
        return torch.stack([self.niqe(gray), self.brisque(gray)], dim=1).float()

    @torch.no_grad()
    def with_features(self, x):
        """
        Scores and the NSS feature vectors behind them, in one pass:
          scores (B, 2) float32   NIQE, BRISQUE (as __call__)
          nss    (B, 72) float32  BRISQUE's 36 features | the mean of NIQE's 36 block features over the image's blocks
        """
        gray = clip_to_gray(x)
        fb, fn = brisque_features(gray), self.niqe.features(gray)
        scores = torch.stack([self.niqe.score(fn), self.brisque.score(fb)], dim=1).float()
        return scores, torch.cat([fb, torch.nanmean(fn, dim=1)], dim=1).float()
