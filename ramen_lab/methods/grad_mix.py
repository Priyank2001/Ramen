"""
Ramen stepping on a mix of its active-set gradient and the image's own entropy gradient (docs/plans/grad_mix.md).

Ramen (src/methods/Ramen.py:111-170) steps every image on sign(g_R), where g_R is the weighted aggregate of the
entropy gradients retrieved from memory (the active set). Here the step direction is

    raw   G_i = lambda g_R(i) + (1 - lambda) g_H(i)
    unit  G_i = lambda g_R(i)/||g_R(i)|| + (1 - lambda) g_H(i)/||g_H(i)||

with g_H(i) the image's own entropy gradient at theta_0 (the value Ramen writes to memory). lambda = 1 (the default)
is Ramen, bit for bit (tests/test_lab_grad_mix.py); lambda = 0 is one episodic per-image entropy step (Tent-like).

SignSGD uses only sign(G_i), so in `raw` a coordinate follows whichever of lambda g_R and (1 - lambda) g_H is larger
there: lambda alone is not the share. `unit` makes it the share. Per-image norms are logged to tell them apart.

Logged per image (stored_value.npz, with RamenStoredValue's fields): norm_R, norm_H (fp32 norms of g_R and g_H) and
flip_frac, the share of coordinates where sign(G_i) differs from Ramen's sign(g_R(i)).
"""
from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import Tensor

from ramen_lab.methods import METHODS
from ramen_lab.methods.stored_value import RamenStoredValue
from ramen_lab.registry import Registry

MIXES: Registry[type] = Registry("gradient mix")
NORM_EPS = 1e-30  # an all-zero gradient (fp16 underflow) stays zero instead of becoming NaN


class GradientMix(ABC):
    """How the active-set gradient g_R and the own gradient g_H are combined into the step direction."""

    def __init__(self, lam: float):
        self.lam = lam

    @abstractmethod
    def mix(self, g_r: Tensor, g_h: Tensor) -> Tensor:
        """(B, D) fp32 inputs -> (B, D) fp32 step direction (only its sign is used)."""


@MIXES.register("raw")
class RawMix(GradientMix):
    """G = lambda g_R + (1 - lambda) g_H: the literal mix."""

    def mix(self, g_r, g_h):
        return self.lam * g_r + (1 - self.lam) * g_h


@MIXES.register("unit")
class UnitMix(GradientMix):
    """G = lambda g_R/||g_R|| + (1 - lambda) g_H/||g_H||: lambda is the share whatever the two norms are."""

    def mix(self, g_r, g_h):
        unit_r = g_r / g_r.norm(dim=1, keepdim=True).clamp_min(NORM_EPS)
        unit_h = g_h / g_h.norm(dim=1, keepdim=True).clamp_min(NORM_EPS)
        return self.lam * unit_r + (1 - self.lam) * unit_h


@METHODS.register("RamenGradMix")
class RamenGradMix(RamenStoredValue):
    """Ramen whose SignSGD step uses lambda g_R + (1 - lambda) g_H (label-free). Defaults reproduce Ramen."""

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.lam = float(self.cfg.get("mix_lambda", 1.0))
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError(f"mix_lambda must be in [0, 1]: {self.lam}")
        self.mixer: GradientMix = MIXES.get(self.cfg.get("mix_norm", "raw"))(self.lam)
        if self.cfg.get("stored_value", "entropy") != "entropy":
            raise ValueError("RamenGradMix mixes with the own ENTROPY gradient: stored_value must be 'entropy'")
        # the mixed direction is handed to the optimizer as its sign (exact +-1 in fp16, so no coordinate can
        # underflow to 0). Same step only for SignSGD, which upstream builds with lr alone (no momentum or weight
        # decay: ModelForBySampleTTA.py:241)
        if self.cfg.get("optimizer") != "signsgd":
            raise ValueError("RamenGradMix needs optimizer 'signsgd' (it hands the optimizer sign(G))")

    @torch.no_grad()
    def retrieve(self, feats: Tensor, values: Tensor, stream_id: Tensor):
        """Ramen's aggregate g_R (parent), then the mix with the own gradient g_H = `values` (B, D)."""
        aggregated, stats = super().retrieve(feats, values, stream_id)
        g_r, g_h = aggregated.float(), values.float()
        stats["norm_R"], stats["norm_H"] = g_r.norm(dim=1), g_h.norm(dim=1)
        if self.lam == 1.0:  # Ramen: hand over g_R untouched (bit identity)
            stats["flip_frac"] = torch.zeros(g_r.shape[0], device=g_r.device)
            return aggregated, stats
        step = self.mixer.mix(g_r, g_h).sign()
        stats["flip_frac"] = (step != g_r.sign()).float().mean(dim=1)
        return step.to(aggregated.dtype), stats

    @torch.no_grad()
    def record(self, zs_pred: Tensor, stats: Dict[str, Tensor]) -> None:
        super().record(zs_pred, stats)
        for name in ("norm_R", "norm_H", "flip_frac"):
            self.chunks.setdefault(name, []).append(stats[name].float().cpu().numpy())
