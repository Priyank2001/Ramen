"""
Ramen with a pluggable cache-write strategy: which test samples enter the memory, under which class, and when.

Ramen itself writes every sample into the queue of its zero-shot prediction BEFORE retrieving, so each sample retrieves
itself and its batch-mates. A strategy that needs Ramen's own (adapted) prediction, such as "write only if CLIP and
Ramen agree", can only write AFTER the prediction. That also changes the order, so compare it with WriteAllAfter,
which changes only the order.

Strategies (config key `cache_strategy`):
  ramen          RamenDefault     write every sample, before retrieval, under the zero-shot label  (== Ramen exactly)
  all_after      WriteAllAfter    write every sample, after prediction, under the zero-shot label  (order-only control)
  agree          WriteIfAgree     write only if zero-shot pred == Ramen pred, after prediction
  adapted_label  WriteAdaptedLabel write every sample, after prediction, under Ramen's label

To add a strategy: subclass CacheStrategy, implement select(), and add it to STRATEGIES.

Per-sample log (self.stats['filed_as']): the class each sample was filed under, -1 if not written. ramen_analysis
saves it with the run, so exp.method_stats[name]['filed_as'] lines up with exp.results[name] row by row.
"""
import numpy as np
import torch

from .losses import softmax_entropy
from .Ramen import Ramen


# ---------------------------------------------------------------------------- strategies
class CacheStrategy:
    """Base class. Decides which samples of a batch are written into the memory, and under which class."""

    write_after_prediction = False  # False: write before retrieval (Ramen). True: write after Ramen's prediction.

    def __init__(self, cfg):
        self.cfg = cfg  # the method's config, for strategies with their own settings

    def select(self, zs_logits, adapted_logits):
        """
        zs_logits:      (B, C) zero-shot logits.
        adapted_logits: (B, C) Ramen's logits after adaptation, or None when write_after_prediction is False.
        Returns (write, label): write (B,) bool, which samples enter the memory; label (B,) long, the class queue
        each of them goes into.
        """
        raise NotImplementedError


class RamenDefault(CacheStrategy):
    """Ramen's own rule: every sample, before retrieval, under its zero-shot prediction."""

    def select(self, zs_logits, adapted_logits):
        zs_pred = zs_logits.argmax(-1)
        return torch.ones_like(zs_pred, dtype=torch.bool), zs_pred


class WriteAllAfter(CacheStrategy):
    """Every sample, under its zero-shot prediction, but after the prediction (control: only the order changes)."""

    write_after_prediction = True

    def select(self, zs_logits, adapted_logits):
        zs_pred = zs_logits.argmax(-1)
        return torch.ones_like(zs_pred, dtype=torch.bool), zs_pred


class WriteIfAgree(CacheStrategy):
    """Only samples where zero-shot CLIP and Ramen predict the same class, filed under that class."""

    write_after_prediction = True

    def select(self, zs_logits, adapted_logits):
        zs_pred, adapted_pred = zs_logits.argmax(-1), adapted_logits.argmax(-1)
        return zs_pred == adapted_pred, zs_pred


class WriteAdaptedLabel(CacheStrategy):
    """Every sample, filed under Ramen's prediction instead of the zero-shot one (Candidate A1 of the PC2 handoff)."""

    write_after_prediction = True

    def select(self, zs_logits, adapted_logits):
        adapted_pred = adapted_logits.argmax(-1)
        return torch.ones_like(adapted_pred, dtype=torch.bool), adapted_pred


STRATEGIES = {
    'ramen': RamenDefault,
    'all_after': WriteAllAfter,
    'agree': WriteIfAgree,
    'adapted_label': WriteAdaptedLabel,
}


# ---------------------------------------------------------------------------- the method
class RamenCache(Ramen):

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        name = self.cfg.get('cache_strategy', 'ramen')
        if name not in STRATEGIES:
            raise ValueError(f'Unknown cache_strategy {name!r}; choose from {list(STRATEGIES)}')
        self.strategy = STRATEGIES[name](self.cfg)
        self.stats = {'strategy': name, 'filed_as': []}  # per batch: class each sample was filed under, -1 = not written

    @torch.no_grad()
    def write(self, feats, grads, zs_logits, adapted_logits):
        """Ask the strategy which samples to keep, and add them to their class queues."""
        B = feats.shape[0]
        keep, label = self.strategy.select(zs_logits, adapted_logits)
        # Same priorities (arrival order) and entropies (zero-shot) as Ramen, including its fp16 priorities.
        priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.dtype)
        entropies = softmax_entropy(zs_logits, reduction='none')
        self.counter = self.counter + B
        for b in range(B):
            if keep[b]:
                self.cache[label[b]].add(keys=feats[b].unsqueeze(0), values=grads[b].unsqueeze(0),
                                         entropies=entropies[b].unsqueeze(0), priorities=priorities[b].unsqueeze(0))
        self.stats['filed_as'].append(torch.where(keep, label, -1).cpu().numpy().astype(np.int32))

    @torch.no_grad()
    def aggregate(self, feats, grads):
        """Ramen's retrieval and weighting: top-k from every non-empty class queue. Zero if the memory is empty."""
        retrieved_grads_sum = torch.zeros_like(grads)
        num_class_sum = 0
        for c in range(self.num_classes):
            if self.cache[c].size == 0:
                continue
            retrieved, _, entropies, dist = self.cache[c].query(feats, topk=self.cfg['topk'])
            weights = torch.exp(-entropies) * torch.exp(-self.beta * dist)
            retrieved_grads_sum += (retrieved * weights.unsqueeze(dim=2)).sum(dim=1)
            num_class_sum += 1
        # An empty memory (first batch of a write-after strategy) gives a zero gradient: SignSGD then doesn't move,
        # so the prediction is zero-shot.
        return retrieved_grads_sum / max(num_class_sum, 1)

    def forward(self, x):
        # 1. Zero-shot features, logits and each sample's own entropy gradient (as in Ramen).
        feats = self.model.featurize(x)
        zs_logits = self.model.classify(feats)
        self.loss_fn(zs_logits).backward()
        grads = self.model.get_by_sample_grad()

        # 2. Ramen writes here, before retrieving.
        if not self.strategy.write_after_prediction:
            self.write(feats, grads, zs_logits, None)

        # 3. Retrieve, aggregate, take one step, predict, reset (as in Ramen).
        self.model.set_by_sample_grad(self.aggregate(feats, grads))
        self.model.step_and_zero_grad()
        with torch.no_grad():
            adapted_logits = self.model(x)
        self.model.reset_parameters()

        # 4. Strategies that need Ramen's prediction write here instead.
        if self.strategy.write_after_prediction:
            self.write(feats, grads, zs_logits, adapted_logits)

        return adapted_logits
