"""
Ramen + loss-landscape slices for chosen images: an ORACLE MEASUREMENT, not a method. Predictions are exactly Ramen's.

For each image whose stream position is in cfg['landscape_positions'] it records, after Ramen has predicted:
  - the true-label CE, the entropy and the predicted class on a 2D slice of parameter space through theta_0 spanned
    by Ramen's actual step u and the ORACLE SignSGD step on the true-label CE (landscape.py);
  - CE and predicted class along three lines from theta_0: Ramen's step, the oracle step, a random sign step of the
    same length (control);
  - the zero-shot feature z0 and the adapted feature z1 (for the embedding-space slice, computed offline);
  - checks: the slice point (1, 0) must give Ramen's logits and (0, 0) the zero-shot logits.
The memory, the update and the prediction are computed exactly as in Ramen (RamenGradientProbe.write / retrieve);
everything else runs on copies of the image after the parameters are reset.

    from landscape_methods import RamenLandscapeProbe
    method = RamenLandscapeProbe(model, datasets, args)   # args.config = Ramen.yaml + {'landscape_positions': [...]}
    method.stats['landscape']                            # one dict per chosen image
"""
import numpy as np
import torch
import torch.nn.functional as F

from models.ModelForBySampleTTA import BySampleBatchNorm, BySampleLayerNorm

from alignment import cosine
from ga_methods import RamenGradientProbe
from landscape import GRID_A, GRID_B, LINE_T, grid_points, param_plane, random_sign_step


class RamenLandscapeProbe(RamenGradientProbe):

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.positions = set(int(p) for p in self.cfg.get('landscape_positions', []))
        self.lr = float(self.cfg['lr'])
        self.chunk = args.max_batch_size
        self.seen = 0
        self.stats.update(landscape=[], text_feature=self.model.text_feature.float().cpu().numpy())

    # ------------------------------------------------------------------ per-sample parameters
    def _norm_modules(self):
        return [m for m in self.model.model.modules() if isinstance(m, (BySampleLayerNorm, BySampleBatchNorm))]

    @torch.no_grad()
    def read_delta(self, rows):
        """(len(rows), D) float32: per-sample parameters minus the pretrained ones (same order as the gradients)."""
        parts = []
        for m in self._norm_modules():
            parts += [m.weight_by_sample[rows].float() - m.weight.float(), m.bias_by_sample[rows].float() - m.bias.float()]
        return torch.cat(parts, 1)

    @torch.no_grad()
    def set_delta(self, delta):
        """Rows 0..n-1 of every per-sample parameter <- pretrained + delta (n, D)."""
        n, idx = delta.shape[0], 0
        for m in self._norm_modules():
            for p, base in ((m.weight_by_sample, m.weight), (m.bias_by_sample, m.bias)):
                w = p.shape[1]
                p[:n].copy_((base.float() + delta[:, idx:idx + w]).to(p.dtype))
                idx += w
        assert idx == delta.shape[1]

    @torch.no_grad()
    def evaluate(self, x1, deltas):
        """Logits (n, C) float32 of one image x1 (1, ...) under theta_0 + deltas[k], in chunks of the batch size."""
        out = []
        for s in range(0, deltas.shape[0], self.chunk):
            d = deltas[s:s + self.chunk]
            self.set_delta(d)
            out.append(self.model(x1.expand(d.shape[0], *x1.shape[1:])).float())
        self.model.reset_parameters()
        return torch.cat(out)

    @staticmethod
    def summarise(logits, y):
        logp = F.log_softmax(logits, 1)
        return dict(ce=(-logp[:, y]).cpu().numpy(), entropy=(-(logp.exp() * logp).sum(1)).cpu().numpy(),
                    pred=logits.argmax(1).cpu().numpy().astype(np.int32))

    # ------------------------------------------------------------------ the slices for one image
    @torch.no_grad()
    def landscape(self, x1, y, u, ce_grad, z0, z1, zs_logits, ramen_logits, position, zs_pred, ramen_pred):
        o = -self.lr * torch.sign(ce_grad.float())
        r = random_sign_step(u.numel(), self.lr, seed=position, device=u.device)
        e1, e2, oracle_xy = param_plane(u, o)
        P = torch.as_tensor(grid_points(), device=u.device)
        T = torch.as_tensor(LINE_T, dtype=torch.float32, device=u.device)[:, None]
        deltas = torch.cat([P[:, :1] * e1 + P[:, 1:] * e2, T * u, T * o, T * r])
        logits = self.evaluate(x1, deltas)
        n_grid, n_line = P.shape[0], T.shape[0]
        shape = (len(GRID_B), len(GRID_A))
        grid = self.summarise(logits[:n_grid], y)
        lines = {k: self.summarise(logits[n_grid + i * n_line: n_grid + (i + 1) * n_line], y)
                 for i, k in enumerate(('ramen', 'oracle', 'random'))}
        at = lambda a, b: int(np.flatnonzero((GRID_A == a))[0] + len(GRID_A) * np.flatnonzero(GRID_B == b)[0])
        g0, g1 = logits[at(0.0, 0.0)], logits[at(1.0, 0.0)]
        rec = dict(position=position, label=y, zs_pred=zs_pred, ramen_pred=ramen_pred, oracle_xy=oracle_xy,
                   step_norm=float(u.norm()), cos_step=float(cosine(torch.sign(u)[None] * -1, ce_grad[None])),
                   cos_oracle_ramen=float(cosine(o[None], u[None])), cos_random_ramen=float(cosine(r[None], u[None])),
                   **{f'grid_{k}': v.reshape(shape) for k, v in grid.items()},
                   **{f'line_{n}_{k}': v for n, d in lines.items() for k, v in d.items()},
                   z0=z0.float().cpu().numpy(), z1=z1.float().cpu().numpy(),
                   check_zs_maxdiff=float((g0 - zs_logits.float()).abs().max()),
                   check_ramen_maxdiff=float((g1 - ramen_logits.float()).abs().max()),
                   check_zs_pred=int(g0.argmax()) == zs_pred, check_ramen_pred=int(g1.argmax()) == ramen_pred)
        self.stats['landscape'].append(rec)

    # ------------------------------------------------------------------ forward = Ramen + slices for chosen images
    def forward(self, x, labels=None):
        if labels is None:
            raise ValueError('RamenLandscapeProbe needs the true labels: run it through the harness (records_labels)')
        labels = labels.to(self.device)
        B = x.shape[0]
        rows = [b for b in range(B) if self.seen + b in self.positions]

        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        zs_pred = logits.argmax(-1)
        self.loss_fn(logits).backward(retain_graph=bool(rows))
        own = self.model.get_by_sample_grad()
        ce = self.oracle_ce_gradient(logits, labels)[rows].clone() if rows else None   # ORACLE, measurement only

        self.write(feats, own, logits, zs_pred)                 # Ramen's write (zero-shot filing)
        update = self.retrieve(feats, own)
        self.model.set_by_sample_grad(update)
        self.model.step_and_zero_grad()
        u = self.read_delta(rows) if rows else None
        with torch.no_grad():
            z1 = self.model.featurize(x)
            out = self.model.classify(z1)
        self.model.reset_parameters()

        for i, b in enumerate(rows):
            self.landscape(x[b:b + 1], int(labels[b]), u[i], ce[i], feats[b].detach(), z1[b], logits[b].detach(),
                           out[b], self.seen + b, int(zs_pred[b]), int(out[b].argmax()))
        self.seen += B
        return out
