"""
Ramen with a zero-shot anchor.

The anchor is the prediction of the pretrained weights w, which Ramen already computes before adapting
(it resets w after every sample): p_i, pseudo-label y_i = argmax p_i, margin m_i = p_i,(1) - p_i,(2).
Every option is off by default; with all of them off this is exactly Ramen.

  lambda_self (option 1): alpha_ii <- alpha_ii * (1 + lambda_self * m_i)
                          The query is always in its own queue at distance 0, so alpha_ii = exp(-H_i).
  lambda_tilt (option 2): alpha_ij <- alpha_ij * exp(lambda_tilt * m_i) for every j retrieved from queue y_i.
  lambda_ce   (option 3): g_i <- g_i + lambda_ce * m_i * grad_w CE(p_i(w), y_i)   (one extra backward)
  gate_tau    (option 4): if the adapted top class != y_i and m_i > gate_tau, shrink that sample's step to
                          s * (w_i - w) for s in gate_scales, until the top class is y_i again.

Note: the optimizer is SignSGD, so options 1-3 only matter relative to the size of the aggregated gradient
(sum_j alpha_ij is O(topk), so lambda needs to be O(10) or more to change the step).
"""
import torch
import torch.nn.functional as F

from models.ModelForBySampleTTA import BySampleLayerNorm, BySampleBatchNorm

from .losses import softmax_entropy
from .Ramen import Ramen


class RamenAnchor(Ramen):

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)

        self.lambda_self = self.cfg.get('lambda_self', 0.0)
        self.lambda_tilt = self.cfg.get('lambda_tilt', 0.0)
        self.lambda_ce = self.cfg.get('lambda_ce', 0.0)
        self.gate_tau = self.cfg.get('gate_tau', None)
        self.gate_scales = self.cfg.get('gate_scales', [0.5, 0.25, 0.0])

        self.norm_modules = [m for m in self.model.model.modules()
                             if isinstance(m, (BySampleLayerNorm, BySampleBatchNorm))]
        self.reset_stats()

    def reset_stats(self):
        # self_share: alpha_ii / sum_j alpha_ij (after the anchor), averaged over samples
        # gate: number of samples that ended at each step scale (1.0 = not gated)
        self.stats = {'n': 0, 'self_share': 0.0, 'gate': {s: 0 for s in [1.0] + list(self.gate_scales)}}

    @torch.no_grad()
    def _scale_step(self, deltas, scale):
        """Per-sample weights w + scale_b * (w_b - w), where w_b - w is the full step."""
        for m, (dw, db) in zip(self.norm_modules, deltas):
            B = dw.shape[0]
            m.weight_by_sample[:B].copy_(m.weight + scale[:, None].to(dw) * dw)
            m.bias_by_sample[:B].copy_(m.bias + scale[:, None].to(db) * db)

    def forward(self, x):
        B = x.shape[0]

        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        init_preds = logits.argmax(-1)

        with torch.no_grad():
            top2 = logits.float().softmax(-1).topk(2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]

        loss = self.loss_fn(logits)
        loss.backward(retain_graph=self.lambda_ce > 0)
        grads = self.model.get_by_sample_grad()

        if self.lambda_ce > 0:
            self.model.optimizer.zero_grad()
            F.cross_entropy(logits.float(), init_preds, reduction='sum').backward()
            ce_grads = self.model.get_by_sample_grad()

        with torch.no_grad():
            # Update cache
            priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.dtype)
            entropies = softmax_entropy(logits, reduction='none')
            self.counter = self.counter + B

            for b in range(B):
                c = init_preds[b]
                self.cache[c].add(keys=feats[b].unsqueeze(0),
                                  values=grads[b].unsqueeze(0),
                                  entropies=entropies[b].unsqueeze(0),
                                  priorities=priorities[b].unsqueeze(0))

            # Retrieve from cache
            retrieved_grads_sum = torch.zeros_like(grads)
            weight_mass = torch.zeros(B, device=self.device)
            num_class_sum = 0
            tilt = torch.exp(self.lambda_tilt * margin)

            for c in range(self.num_classes):
                if self.cache[c].size == 0:
                    continue

                retrieved_grads, priorities, entropies_c, sorted_dist = \
                    self.cache[c].query(feats, topk=self.cfg['topk'])

                weights = torch.exp(-entropies_c)
                weights = weights * torch.exp(-self.beta * sorted_dist)
                if self.lambda_tilt:
                    own = init_preds == c
                    weights = weights * torch.where(own, tilt, torch.ones_like(tilt)).unsqueeze(1).to(weights)

                retrieved_grads_sum += (retrieved_grads * weights.unsqueeze(dim=2)).sum(dim=1)
                weight_mass += weights.float().sum(dim=1)
                num_class_sum += 1

            # Option 1: the query's own term alpha_ii * g_i (tilted if option 2 is on) gets an extra lambda * m_i
            alpha_ii = torch.exp(-entropies.float()) * (tilt if self.lambda_tilt else 1.0)
            extra_self = self.lambda_self * margin * alpha_ii
            if self.lambda_self:
                retrieved_grads_sum += extra_self.unsqueeze(1).to(grads) * grads
            self_share = (alpha_ii + extra_self) / (weight_mass + extra_self)

            retrieved_grads = retrieved_grads_sum / num_class_sum

            # Option 3: hard-label anchor on the query, gradient at w
            if self.lambda_ce > 0:
                retrieved_grads = retrieved_grads + (self.lambda_ce * margin).unsqueeze(1).to(grads) * ce_grads

            self.model.set_by_sample_grad(retrieved_grads)

        # Update Model
        self.model.step_and_zero_grad()

        # Inference
        with torch.no_grad():
            logits = self.model(x)

            # Option 4: shrink the step for decisive samples whose top class flipped away from y_i
            scale = torch.ones(B, device=self.device)
            if self.gate_tau is not None:
                deltas = [(m.weight_by_sample[:B] - m.weight, m.bias_by_sample[:B] - m.bias)
                          for m in self.norm_modules]
                for s in self.gate_scales:
                    reject = (logits.argmax(-1) != init_preds) & (margin > self.gate_tau)
                    if not reject.any():
                        break
                    scale[reject] = s
                    self._scale_step(deltas, scale)
                    logits = self.model(x)

            self.stats['n'] += B
            self.stats['self_share'] += self_share.sum().item()
            for s in self.stats['gate']:
                self.stats['gate'][s] += (scale == s).sum().item()

        # reset model
        self.model.reset_parameters()

        return logits
