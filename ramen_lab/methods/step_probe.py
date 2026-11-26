"""
Ramen, plus its prediction at every step size in a grid (docs/plans/step_size.md). Label-free probe.

Ramen moves every LayerNorm coordinate of every image by exactly eta = lr (0.01): theta_i = theta_0 - eta sign(G_i).
Everything Ramen writes to memory (features, entropies, stored gradients, priorities) is computed at theta_0, and
the step is undone after every batch, so an image's step size changes only that image's prediction. One run that
predicts each image at every eta in `probe_etas` therefore lets any per-image step rule with etas from that grid be
scored exactly offline.

Returned logits are Ramen's (the lr pass, run first on the buffers the backward filled, as upstream), so the run is
Ramen bit for bit (tests/test_lab_step_probe.py). The other etas re-arm the gradient buffers, set G_i again, step
with lr = eta, predict and reset; lr goes back to the configured value after every batch.

Logged per image (stored_value.npz, with RamenStoredValue's fields), in stream order:
    eta_pred (B, n_eta) int16   prediction at each eta of `probe_etas` (column order = probe_etas)
    eta_moved (B, n_eta) fp16   share of the image's LayerNorm coordinates that actually changed (fp16 rounding)
    entropy, margin             zero-shot entropy H_i and top-1 minus top-2 probability, at theta_0
    norm_G, norm_H, cos_G_H     ||G_i||, ||g_H(i)||, cos(G_i, g_H(i))
    weight_mass                 sum of retrieval weights w_ij over Ramen's top-k sets
"""
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

from ramen_lab import upstream
from ramen_lab.methods import METHODS
from ramen_lab.methods.stored_value import RamenStoredValue

LOGGED = ("eta_pred", "eta_moved", "entropy", "margin", "norm_G", "norm_H", "cos_G_H", "weight_mass")


@METHODS.register("RamenStepProbe")
class RamenStepProbe(RamenStoredValue):
    """Ramen that also predicts every image at each step size in `probe_etas` (label-free, returns Ramen)."""

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.lr = float(self.cfg["lr"])
        self.etas: List[float] = [float(e) for e in self.cfg.get("probe_etas", [self.lr])]
        if self.lr not in self.etas:
            raise ValueError(f"probe_etas must contain Ramen's lr {self.lr} (its column is the returned prediction)")
        if any(e < 0 for e in self.etas) or len(set(self.etas)) != len(self.etas):
            raise ValueError(f"probe_etas must be distinct and >= 0: {self.etas}")
        if self.cfg.get("optimizer") != "signsgd" or self.cfg.get("stored_value", "entropy") != "entropy":
            raise ValueError("RamenStepProbe probes Ramen: optimizer 'signsgd' and stored_value 'entropy' only")
        self._signals: Dict[str, Tensor] = {}
        self._aggregated: Tensor = torch.empty(0)

    # ------------------------------------------------------------------ Ramen's steps, observed
    def compute_values(self, logits: Tensor, zs_pred: Tensor, labels=None) -> Tensor:
        with torch.no_grad():
            p = F.softmax(logits.float(), dim=-1)
            top2 = p.topk(2, dim=-1).values
            self._signals = {"entropy": upstream.softmax_entropy(logits, reduction="none").float(),
                             "margin": top2[:, 0] - top2[:, 1]}
        return super().compute_values(logits, zs_pred, labels)

    @torch.no_grad()
    def retrieve(self, feats: Tensor, values: Tensor, stream_id: Tensor):
        aggregated, stats = super().retrieve(feats, values, stream_id)
        g, h = aggregated.float(), values.float()
        self._signals.update(norm_G=g.norm(dim=1), norm_H=h.norm(dim=1),
                             cos_G_H=F.cosine_similarity(g, h, dim=1), weight_mass=stats["weight_mass"])
        self._aggregated = aggregated
        return aggregated, stats

    def adapt_and_predict(self, x: Tensor) -> Tensor:
        """Ramen's pass first (returned), then one pass per other eta; theta_0 and lr restored at the end."""
        B = x.shape[0]
        before = self._by_sample_params(B)
        self.model.step_and_zero_grad()
        moved = {self.lr: (self._by_sample_params(B) != before).float().mean(dim=1)}
        with torch.no_grad():
            out = self.model(x)
        self.model.reset_parameters()
        preds = {self.lr: out.argmax(-1)}
        try:
            for eta in self.etas:
                if eta != self.lr:
                    preds[eta], moved[eta] = self._predict_at(x, eta, before)
        finally:
            self._set_lr(self.lr)
        self._signals["eta_pred"] = torch.stack([preds[e] for e in self.etas], dim=1)
        self._signals["eta_moved"] = torch.stack([moved[e] for e in self.etas], dim=1)
        return out

    def _predict_at(self, x: Tensor, eta: float, before: Tensor):
        B = x.shape[0]
        upstream.rearm_by_sample_grads(self.model, B)
        with torch.no_grad():
            self.model.set_by_sample_grad(self._aggregated)
        self._set_lr(eta)
        self.model.step_and_zero_grad()
        moved = (self._by_sample_params(B) != before).float().mean(dim=1)
        with torch.no_grad():
            pred = self.model(x).argmax(-1)
        self.model.reset_parameters()
        return pred, moved

    def _set_lr(self, eta: float) -> None:
        for group in self.model.optimizer.param_groups:
            group["lr"] = eta

    @torch.no_grad()
    def _by_sample_params(self, B: int) -> Tensor:
        """(B, D) current per-image LayerNorm affine values, in get_by_sample_grad's coordinate order."""
        parts = []
        for module in self.model.model.modules():
            if hasattr(module, "weight_by_sample"):
                parts += [module.weight_by_sample[:B].detach().clone(), module.bias_by_sample[:B].detach().clone()]
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------ logging
    @torch.no_grad()
    def record(self, zs_pred: Tensor, stats: Dict[str, Tensor]) -> None:
        super().record(zs_pred, stats)
        cast = {"eta_pred": lambda t: t.short(), "eta_moved": lambda t: t.half()}
        for name in LOGGED:
            value = cast.get(name, lambda t: t.float())(self._signals[name])
            self.chunks.setdefault(name, []).append(value.cpu().numpy())
