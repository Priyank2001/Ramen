"""
Ramen whose memory entries carry a cross-entropy gradient toward a TARGET class chosen without labels.

The label-value oracle (2026-11-01_label_oracle, RamenLabelValues) showed that Ramen gains +1.70 (DomainNet),
+1.51 (ImageNet-C) and +26.70 (EuroSAT) once each entry's value is the CE gradient toward its TRUE label instead of
its own entropy gradient. Here the target comes from a rule that does not use labels, so the run is a method:

  zero_shot   the image's zero-shot prediction (control: the CE form with Ramen's own target)
  adapted     Ramen's adapted prediction of the image (after its one step)
  knn         exp(-H)-weighted vote of the `knn_k` nearest cache entries over the whole cache, each labelled by the
              queue it is filed in (= its zero-shot prediction); the image's own entry is excluded
  mix         ORACLE (dose-response): the true label for a random `mix_p` share of zero-shot errors, else zero-shot
  true_label  ORACLE: the true label for every image (smoke cross-check against RamenLabelValues)

Everything else is Ramen's, in Ramen's order: the image is written with its entropy gradient before retrieval,
retrieves, steps, is predicted, and the parameters are reset. Only THEN is the target computed and the entry's value
replaced, so no image ever uses its own target (as in the oracle). The CE gradient is computed at the pretrained
parameters with a second zero-shot pass (theta_0 after the reset); with config `target: none` that pass is skipped
and the run is RamenLabelValues(value='entropy') = Ramen exactly.

The harness passes the true labels to forward (records_labels). Label-free targets use them ONLY for the per-batch
log (how often the target is right), never in the update.

    from pseudo_methods import RamenPseudoValues
    method = RamenPseudoValues(model, datasets, args)  # args.config = Ramen's cfg + {'target': 'knn', 'knn_k': 10}
    logits = method(x, labels=y)

Adding a target rule: subclass TargetRule, add it to TARGET_RULES.
"""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from label_methods import RamenLabelValues


class TargetRule:
    """The class an entry's value pulls toward, chosen once its image has been predicted."""
    uses_true_labels = False
    needs_knn = False

    def __init__(self, cfg):
        self.cfg = cfg

    def target(self, ctx):
        """ctx: zs_pred, adapted, knn (or None), labels -> (B,) long tensor of target classes."""
        raise NotImplementedError


class ZeroShotTarget(TargetRule):
    def target(self, ctx):
        return ctx.zs_pred


class AdaptedTarget(TargetRule):
    def target(self, ctx):
        return ctx.adapted


class KnnVoteTarget(TargetRule):
    needs_knn = True

    def target(self, ctx):
        return ctx.knn


class TrueLabelTarget(TargetRule):
    """ORACLE."""
    uses_true_labels = True

    def target(self, ctx):
        return ctx.labels


class MixedOracleTarget(TargetRule):
    """ORACLE: the true label for a random `mix_p` share of zero-shot errors, else the zero-shot prediction."""
    uses_true_labels = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.p = float(cfg['mix_p'])
        self.gen = torch.Generator().manual_seed(int(cfg.get('mix_seed', 0)))  # same draws on every run of a stream

    def target(self, ctx):
        draw = torch.rand(len(ctx.labels), generator=self.gen).to(ctx.labels.device) < self.p
        return torch.where(draw & (ctx.zs_pred != ctx.labels), ctx.labels, ctx.zs_pred)


TARGET_RULES = {'zero_shot': ZeroShotTarget, 'adapted': AdaptedTarget, 'knn': KnnVoteTarget,
                'mix': MixedOracleTarget, 'true_label': TrueLabelTarget}


class RamenPseudoValues(RamenLabelValues):

    def __init__(self, model, datasets, args):
        # the parent runs with Ramen's entropy values; this class replaces them after prediction
        base = SimpleNamespace(**{**vars(args), 'config': {**args.config, 'value': 'entropy'}})
        super().__init__(model, datasets, base)
        self.cfg = args.config
        name = self.cfg.get('target', 'none')
        if name != 'none' and name not in TARGET_RULES:
            raise ValueError(f'Unknown target {name!r}; choose from none, {list(TARGET_RULES)}')
        self.target_rule = None if name == 'none' else TARGET_RULES[name](self.cfg)
        self.knn_k = int(self.cfg.get('knn_k', 10))
        self.stats.update(target=name, target_right=[], target_fixes=[], target_breaks=[], target_changed=[],
                          zs_right=[], batch_size=[])

    # ------------------------------------------------------------------ targets
    @torch.no_grad()
    def knn_vote(self, feats, slots):
        """exp(-H)-weighted vote of the knn_k nearest entries of the whole cache (own entry excluded) over queues."""
        classes = [c for c in range(self.num_classes) if self.cache[c].size > 0]
        q = feats.detach().to(device=self.device, dtype=self.dtype)
        dists, owners, ents, offset = [], [], [], {}
        n = 0
        for c in classes:
            cache = self.cache[c]
            offset[c] = n
            dists.append(torch.cdist(q, cache.keys[:cache.size], compute_mode='use_mm_for_euclid_dist').float())
            owners.append(torch.full((cache.size,), c, device=self.device, dtype=torch.long))
            ents.append(cache.entropies[:cache.size].float())
            n += cache.size
        d, owner, H = torch.cat(dists, 1), torch.cat(owners), torch.cat(ents)
        for b, (c, slot) in enumerate(slots):
            if slot is not None:
                d[b, offset[c] + slot] = float('inf')
        sd, idx = d.topk(min(self.knn_k, n), dim=1, largest=False)
        w = torch.where(torch.isfinite(sd), torch.exp(-H[idx]), torch.zeros_like(sd))
        votes = torch.zeros(len(q), self.num_classes, device=self.device).scatter_add_(1, owner[idx], w)
        return votes.argmax(1)

    def ce_values(self, x, targets):
        """Per-sample CE gradients toward `targets` at the pretrained parameters (a fresh zero-shot pass)."""
        self.model.optimizer.zero_grad()
        logits = self.model.classify(self.model.featurize(x))
        F.cross_entropy(logits, targets, reduction='sum').backward()  # sum: one per-sample gradient each
        values = self.model.get_by_sample_grad()
        self.model.optimizer.zero_grad()
        return values

    @torch.no_grad()
    def log_targets(self, t, zs_pred, labels):
        """Logging only (the update never sees labels): how often the target is right, and what it changes."""
        zs_right, t_right = zs_pred == labels, t == labels
        self.stats['target_right'].append(int(t_right.sum()))
        self.stats['zs_right'].append(int(zs_right.sum()))
        self.stats['target_fixes'].append(int((t_right & ~zs_right).sum()))
        self.stats['target_breaks'].append(int((~t_right & zs_right).sum()))
        self.stats['target_changed'].append(int((t != zs_pred).sum()))
        self.stats['batch_size'].append(len(labels))

    # ------------------------------------------------------------------ forward
    def forward(self, x, labels=None):
        if labels is None:
            raise ValueError('RamenPseudoValues runs through the harness (records_labels; labels are logged only)')
        labels = labels.to(self.device)

        # 1-2. Ramen: zero-shot pass and own entropy gradient, write, retrieve, one step, predict, reset
        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        zs_pred = logits.argmax(-1)
        self.loss_fn(logits).backward()
        grads = self.model.get_by_sample_grad()
        self.stats['zs_wrong'].append(int((zs_pred != labels).sum()))
        slots = self.write(feats, grads, logits, self.queue_of(zs_pred, labels))
        knn = self.knn_vote(feats, slots) if self.target_rule is not None and self.target_rule.needs_knn else None
        self.model.set_by_sample_grad(self.retrieve(feats, grads))
        self.model.step_and_zero_grad()
        with torch.no_grad():
            out = self.model(x)
        self.model.reset_parameters()

        # 3. After prediction: the target, and its CE gradient as the entry's value from now on
        if self.target_rule is not None:
            ctx = SimpleNamespace(zs_pred=zs_pred, adapted=out.argmax(-1), knn=knn, labels=labels)
            t = self.target_rule.target(ctx)
            self.log_targets(t, zs_pred, labels)
            self.reveal(slots, self.ce_values(x, t))
        return out
