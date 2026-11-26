"""
ORACLE diagnostic: how well does Ramen's actual update direction agree with the cross-entropy gradient?

`RamenGradAlignment` IS Ramen (same predictions, bit for bit): it only watches three points of upstream's forward and
reads the true labels to compute, per sample i, with g = gradient w.r.t. that sample's own LayerNorm affine copy:

    g_H   own entropy gradient (what Ramen writes into memory)
    g_R   retrieved, aggregated gradient (what Ramen feeds to SignSGD; the applied step is -lr * sign(g_R))
    g_CE  cross-entropy gradient toward the TRUE label (ORACLE, never used by the update)

and records cos(g_R, g_CE), cos(sign(g_R), g_CE), cos(g_H, g_CE), cos(g_R, g_H), a shuffled-label null, per-layer
cosines, sign agreement, and the CE before and after the step. A positive cosine means a descent step on Ramen's
direction also lowers CE to first order.

The probes wrap this method's own model object (not upstream code); `super().forward` is untouched, so identity with
Ramen holds by construction (tests/test_lab_grad_alignment.py checks it).
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ramen_lab import upstream
from ramen_lab.analysis.gradients import cosine, segment_cosines, sign_agreement
from ramen_lab.methods import METHODS

DEFAULT_LOSS_SCALE = 1024.0  # CE gradients of confident samples underflow in fp16; scaling leaves cosines unchanged
PER_SAMPLE = ("cos_R_CE", "cos_signR_CE", "cos_H_CE", "cos_R_H", "cos_R_CEnull", "agree_signR_CE", "agree_signH_CE",
              "step_dCE", "ce_before", "ce_after", "norm_CE", "norm_R", "norm_H", "zs_correct", "final_correct")


@METHODS.register("RamenGradAlignment")
class RamenGradAlignment(upstream.method_class("Ramen")):
    """ORACLE (reads labels) diagnostic wrapper of Ramen; predictions are identical to Ramen's."""

    uses_labels = True

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.loss_scale = float(self.cfg.get("probe_loss_scale", DEFAULT_LOSS_SCALE))
        self.out_path = self.cfg.get("probe_path")
        self.layer_names, self.layer_sizes, self._params = self._layers()
        self._labels: Optional[Tensor] = None
        self._seen: Dict[str, Tensor] = {}
        self._chunks: Dict[str, List[np.ndarray]] = {}
        self._wrap_model()

    # ------------------------------------------------------------------ wiring (the model object is ours)
    def _layers(self):
        """Parameter segments in the order of get_by_sample_grad: (weight, bias) of every by-sample norm layer."""
        names, sizes, params = [], [], []
        for name, module in self.model.model.named_modules():
            if hasattr(module, "weight_by_sample"):
                for kind in ("weight", "bias"):
                    param = getattr(module, f"{kind}_by_sample")
                    names.append(f"{name}.{kind}")
                    sizes.append(param.shape[1])
                    params.append(param)
        return names, sizes, params

    def _wrap_model(self) -> None:
        classify, get_grad, set_grad = self.model.classify, self.model.get_by_sample_grad, self.model.set_by_sample_grad

        def probed_classify(feats):
            logits = classify(feats)
            if self._labels is not None and logits.requires_grad:  # the adaptation pass, not the prediction pass
                self._seen["zs_logits"] = logits.detach()
                self._seen["g_CE"] = self._ce_grad(logits)
            return logits

        def probed_get_grad():
            grads = get_grad()
            if self._labels is not None:
                self._seen["g_H"] = grads.clone()
            return grads

        def probed_set_grad(grad_matrix):
            if self._labels is not None:
                self._seen["g_R"] = grad_matrix.clone()
            return set_grad(grad_matrix)

        self.model.classify, self.model.get_by_sample_grad, self.model.set_by_sample_grad = \
            probed_classify, probed_get_grad, probed_set_grad

    def _ce_grad(self, logits: Tensor) -> Tensor:
        """(B, D) CE gradient to the true label, with the graph kept so upstream's entropy backward still works."""
        loss = F.cross_entropy(logits.float(), self._labels, reduction="sum") * self.loss_scale
        grads = torch.autograd.grad(loss, self._params, retain_graph=True)
        return torch.cat([g[:len(self._labels)] for g in grads], dim=1).float() / self.loss_scale

    # ------------------------------------------------------------------ Ramen, observed
    def forward(self, x, labels=None):
        if labels is None:  # no labels, no probe: exactly Ramen
            return super().forward(x)
        self._labels = labels.to(self.device)
        self._seen = {}
        try:
            logits = super().forward(x)
            self._record(logits)
        finally:
            self._labels, self._seen = None, {}
        return logits

    @torch.no_grad()
    def _record(self, final_logits: Tensor) -> None:
        s, y = self._seen, self._labels
        g_CE, g_H, g_R = s["g_CE"], s["g_H"].float(), s["g_R"].float()
        sign_R, sign_H = g_R.sign(), g_H.sign()
        null_CE = g_CE.roll(1, dims=0)  # another sample's CE gradient: what a sample-agnostic direction would score
        ce_before = F.cross_entropy(s["zs_logits"].float(), y, reduction="none")
        ce_after = F.cross_entropy(final_logits.float(), y, reduction="none")
        values = {"cos_R_CE": cosine(g_R, g_CE), "cos_signR_CE": cosine(sign_R, g_CE), "cos_H_CE": cosine(g_H, g_CE),
                  "cos_R_H": cosine(g_R, g_H), "cos_R_CEnull": cosine(g_R, null_CE),
                  "agree_signR_CE": sign_agreement(g_R, g_CE), "agree_signH_CE": sign_agreement(g_H, g_CE),
                  "step_dCE": -self.cfg["lr"] * (sign_R * g_CE).sum(1),  # first-order CE change of the applied step
                  "ce_before": ce_before, "ce_after": ce_after,
                  "norm_CE": g_CE.norm(dim=1), "norm_R": g_R.norm(dim=1), "norm_H": g_H.norm(dim=1),
                  "zs_correct": (s["zs_logits"].argmax(-1) == y).float(),
                  "final_correct": (final_logits.argmax(-1) == y).float()}
        values["layer_cos_R_CE"] = segment_cosines(g_R, g_CE, self.layer_sizes).half()
        values["layer_cos_H_CE"] = segment_cosines(g_H, g_CE, self.layer_sizes).half()
        for name, value in values.items():
            self._chunks.setdefault(name, []).append(value.cpu().numpy())

    # ------------------------------------------------------------------ output
    def probe_arrays(self) -> Dict[str, np.ndarray]:
        """Everything recorded so far, in stream order (the same order as the run's records.npz)."""
        return {name: np.concatenate(chunks) for name, chunks in self._chunks.items()}

    def reset(self) -> None:
        self.flush()
        super().reset()

    def flush(self) -> None:
        """Write alignment.npz to `probe_path` (config key) if one is set. Rewrites the whole file, so the last call wins."""
        if self.out_path and self._chunks:
            path = Path(self.out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, layer_names=np.array(self.layer_names), **self.probe_arrays())
