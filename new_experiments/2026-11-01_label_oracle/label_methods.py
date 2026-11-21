"""
Ramen whose memory entries carry a gradient toward their TRUE label: an ORACLE (upper bound), not a method.

In Ramen every cache entry's value is the image's own entropy-minimisation gradient, which sharpens its zero-shot
prediction, right or wrong. The true-label *filing* oracle (RamenTrueLabel, filing=true_label) puts each image in
its true class queue but keeps that value, so no label ever reaches the update. This variant changes the value:
once an image has been predicted, its label is revealed and its entry's value becomes the per-sample cross-entropy
gradient toward that label. Labels arrive only AFTER prediction, so an image never uses its own label: while its
batch is adapted, its own entry (retrieved at distance 0) still holds its entropy gradient, as in Ramen.

Two config keys, one per factor (defaults reproduce Ramen exactly):
  filing  which class queue an image is written to (RamenTrueLabel): zero_shot | true_label
  value   what an entry holds once its label is revealed:             entropy   | true_label_ce

    from label_methods import RamenLabelValues
    method = RamenLabelValues(model, datasets, args)    # args.config = Ramen's cfg + {'filing': ..., 'value': ...}
    logits = method(x, labels=y)                        # the harness passes labels (records_labels)

Adding a value rule: subclass ValueRule, add it to VALUE_RULES.
"""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from methods.RamenTrueLabel import RamenTrueLabel
from methods.losses import softmax_entropy


class ValueRule:
    """The value an entry holds after its image has been predicted and its label revealed."""
    uses_true_labels = False
    needs_graph = False  # does `revealed` need a second backward pass through the batch's graph?

    def revealed(self, model, logits, labels):
        """(B, grad_dim) values to store from now on, or None to keep the entropy gradient."""
        return None


class OwnEntropyGradient(ValueRule):
    """Ramen: the entry keeps the image's own entropy gradient."""


class TrueLabelGradient(ValueRule):
    """ORACLE: the per-sample gradient of the cross-entropy toward the image's true label."""
    uses_true_labels = True
    needs_graph = True

    def revealed(self, model, logits, labels):
        model.optimizer.zero_grad()  # drop the entropy gradient, as step_and_zero_grad would
        F.cross_entropy(logits, labels, reduction='sum').backward()  # sum: one per-sample gradient each
        return model.get_by_sample_grad()


VALUE_RULES = {'entropy': OwnEntropyGradient, 'true_label_ce': TrueLabelGradient}


class RamenLabelValues(RamenTrueLabel):

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        name = self.cfg.get('value', 'entropy')
        if name not in VALUE_RULES:
            raise ValueError(f'Unknown value {name!r}; choose from {list(VALUE_RULES)}')
        self.value_rule = VALUE_RULES[name]()
        self.stats.update(value=name, revealed=[])

    @staticmethod
    @torch.no_grad()
    def add(cache, key, value, entropy, priority):
        """PriorityCache.add for one entry (same slot rule), returning the slot written, or None if not written."""
        if cache.size < cache.max_capacity:
            slot = cache.size
            cache.size += 1
        else:
            slot = int(torch.argmin(cache.priorities))
            if not priority > cache.priorities[slot]:
                return None
        cache.keys[slot] = key.detach().to(device=cache.device, dtype=cache.dtype)
        cache.values[slot] = value.detach().to(device=cache.device, dtype=cache.dtype)
        cache.priorities[slot] = priority
        cache.entropies[slot] = entropy
        return slot

    @torch.no_grad()
    def write(self, feats, grads, logits, queues):
        """RamenTrueLabel.write, keeping the (queue, slot) each image went to."""
        B = feats.shape[0]
        priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.priority_dtype)
        entropies = softmax_entropy(logits, reduction='none')
        self.counter = self.counter + B
        return [(int(queues[b]), self.add(self.cache[queues[b]], feats[b], grads[b], entropies[b], priorities[b]))
                for b in range(B)]

    @torch.no_grad()
    def reveal(self, slots, values):
        """Replace the batch's entries' values; a slot reused later in the same batch ends with its last writer."""
        n = 0
        for b, (c, slot) in enumerate(slots):
            if slot is not None:
                self.cache[c].values[slot] = values[b].to(self.cache[c].dtype)
                n += 1
        self.stats['revealed'].append(n)

    def forward(self, x, labels=None):
        if labels is None:
            raise ValueError('RamenLabelValues needs the true labels: run it through the harness (records_labels)')
        labels = labels.to(self.device)

        # 1. Zero-shot features, logits and each image's own entropy gradient (as in Ramen); the label gradient
        #    is computed now, while the graph exists, but stored only after prediction
        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        zs_pred = logits.argmax(-1)
        self.loss_fn(logits).backward(retain_graph=self.value_rule.needs_graph)
        grads = self.model.get_by_sample_grad()
        label_values = self.value_rule.revealed(self.model, logits, labels)
        self.stats['zs_wrong'].append(int((zs_pred != labels).sum()))

        # 2. Write (entropy gradients), retrieve, one step, predict, reset (as in Ramen)
        slots = self.write(feats, grads, logits, self.queue_of(zs_pred, labels))
        self.model.set_by_sample_grad(self.retrieve(feats, grads))
        self.model.step_and_zero_grad()
        with torch.no_grad():
            out = self.model(x)
        self.model.reset_parameters()

        # 3. Labels revealed: the batch's entries now hold the label gradient
        if label_values is not None:
            self.reveal(slots, label_values)
        return out



class RamenWriteAfterPrediction(RamenLabelValues):
    """
    Ramen that writes each image into the cache AFTER predicting it, which allows filing by the ADAPTED prediction.

    Ramen writes before retrieval (its own entry is retrieved at distance 0, and so are later images of its batch).
    The adapted prediction exists only after retrieval and the step, so filing by it needs the write moved after
    prediction. Retrieval then sees earlier batches only, and the first batch (empty cache) is not adapted: it is
    predicted zero-shot. Everything else (keys, stored entropy, retrieval, weights, step, reset) is Ramen's.

    Config key `filing`: zero_shot | true_label (ORACLE) | adapted; `value` as in RamenLabelValues. Since the write
    comes after prediction, a label value (`true_label_ce`) is stored directly, still without self-use.
    """
    FILINGS_AFTER = ('zero_shot', 'true_label', 'adapted')

    def __init__(self, model, datasets, args):
        filing = args.config.get('filing', 'zero_shot')
        if filing not in self.FILINGS_AFTER:
            raise ValueError(f'Unknown filing {filing!r}; choose from {list(self.FILINGS_AFTER)}')
        # RamenTrueLabel validates its own two filings; 'adapted' is handled here, in queue_after
        base = SimpleNamespace(**{**vars(args), 'config': {**args.config, 'filing': 'zero_shot'}})
        super().__init__(model, datasets, base)
        self.cfg, self.filing = args.config, filing
        self.stats.update(filing=filing, adapted=[])

    def queue_after(self, zs_pred, labels, pred):
        """The class queue each image is written to, once its adapted prediction `pred` is known."""
        return {'zero_shot': zs_pred, 'true_label': labels, 'adapted': pred}[self.filing]

    def forward(self, x, labels=None):
        if labels is None:
            raise ValueError('RamenWriteAfterPrediction needs the true labels: run it through the harness')
        labels = labels.to(self.device)

        # 1. Zero-shot features, logits, the entropy gradient and (if used) the label gradient, as in RamenLabelValues
        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        zs_pred = logits.argmax(-1)
        self.loss_fn(logits).backward(retain_graph=self.value_rule.needs_graph)
        grads = self.model.get_by_sample_grad()
        label_values = self.value_rule.revealed(self.model, logits, labels)
        self.stats['zs_wrong'].append(int((zs_pred != labels).sum()))

        # 2. Retrieve from EARLIER batches only, one step, predict, reset; no step while the cache is empty
        adapted = any(cache.size > 0 for cache in self.cache)
        if adapted:
            self.model.set_by_sample_grad(self.retrieve(feats, grads))
            self.model.step_and_zero_grad()
        else:
            self.model.optimizer.zero_grad()
        with torch.no_grad():
            out = self.model(x)
        self.model.reset_parameters()
        self.stats['adapted'].append(adapted)

        # 3. Write, now that the prediction (and the label) is known
        values = grads if label_values is None else label_values
        slots = self.write(feats, values, logits, self.queue_after(zs_pred, labels, out.argmax(-1)))
        self.stats['revealed'].append(sum(slot is not None for _, slot in slots))
        return out


class RamenGlobalRetrieval(RamenLabelValues):
    """
    RamenLabelValues with a choice of retrieval (config key `retrieval`):
      per_class   Ramen's: the topk nearest entries of EVERY non-empty class queue, queues averaged (default; exact)
      global_knn  the `global_k` nearest entries of the whole cache, whatever queue they sit in

    Weights are Ramen's, e^(-H) e^(-beta d), and the update is sum_j W_j g_j / sum_j W_j; Ramen divides by the
    number of non-empty queues instead, which under SignSGD gives the same step. global_knn drops Ramen's
    prediction balance (one top-k per class queue): classes with many nearby entries get more of the vote.
    Filing, write order and values (entropy or label gradients) are RamenLabelValues'.

    Optional `exclude_groups`: a (stream length,) integer array, one group id per stream position (e.g. the base
    photo of an ImageNet-C image, shared by its 15 corrupted copies). An image never retrieves OTHER entries of its
    own group; its own entry stays. Used to stop label values leaking a photo's label through its other copies.
    With it, per_class retrieval runs this class's own top-k (not PriorityCache.query).

    Per-batch log: `classes_in_set` = mean number of class queues with at least one retrieved entry, per query.
    """
    RETRIEVALS = ('per_class', 'global_knn')

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.retrieval = self.cfg.get('retrieval', 'per_class')
        if self.retrieval not in self.RETRIEVALS:
            raise ValueError(f'Unknown retrieval {self.retrieval!r}; choose from {list(self.RETRIEVALS)}')
        self.global_k = self.cfg.get('global_k', 100)
        groups = self.cfg.get('exclude_groups')
        self.groups = None if groups is None else torch.as_tensor(groups, dtype=torch.long, device=self.device)
        if self.groups is not None:
            cap = self.cfg['max_capacity']
            self.slot_pos = torch.full((self.num_classes, cap), -1, dtype=torch.long, device=self.device)
        self.stats.update(retrieval=self.retrieval, global_k=self.global_k, classes_in_set=[],
                          excluded=self.groups is not None)

    @torch.no_grad()
    def write(self, feats, grads, logits, queues):
        first = self.counter
        slots = super().write(feats, grads, logits, queues)
        if self.groups is not None:
            for b, (c, slot) in enumerate(slots):
                if slot is not None:
                    self.slot_pos[c, slot] = first + b
        self._batch_first = first
        return slots

    def _distances(self, feats, c):
        """Ramen's distance call for queue c; entries of the query's group other than itself are pushed to inf."""
        cache = self.cache[c]
        q = feats.detach().to(device=self.device, dtype=self.dtype)
        d = torch.cdist(q, cache.keys[:cache.size], compute_mode='use_mm_for_euclid_dist').float()
        if self.groups is not None:
            pos = self.slot_pos[c, :cache.size]
            qpos = torch.arange(self._batch_first, self._batch_first + feats.shape[0], device=self.device)
            same = (self.groups[pos][None] == self.groups[qpos][:, None]) & (pos[None] != qpos[:, None])
            d = d.masked_fill(same, float('inf'))
        return d

    @torch.no_grad()
    def retrieve(self, feats, grads):
        if self.retrieval == 'per_class' and self.groups is None:
            return super().retrieve(feats, grads)          # Ramen's own code path
        classes = [c for c in range(self.num_classes) if self.cache[c].size > 0]
        dists = [self._distances(feats, c) for c in classes]
        sizes = [d.shape[1] for d in dists]
        H = [self.cache[c].entropies[:self.cache[c].size].float() for c in classes]
        if self.retrieval == 'global_knn':
            d_all, H_all = torch.cat(dists, dim=1), torch.cat(H)
            sd, idx = d_all.topk(min(self.global_k, d_all.shape[1]), dim=1, largest=False)
            w = torch.where(torch.isfinite(sd), torch.exp(-H_all[idx]) * torch.exp(-self.beta * sd), 0.)
            Ws = torch.split(torch.zeros_like(d_all).scatter_(1, idx, w), sizes, dim=1)
        else:                                              # per_class with exclusion
            Ws = []
            for d, h in zip(dists, H):
                sd, idx = d.topk(min(self.cfg['topk'], d.shape[1]), dim=1, largest=False)
                w = torch.where(torch.isfinite(sd), torch.exp(-h[idx]) * torch.exp(-self.beta * sd), 0.)
                Ws.append(torch.zeros_like(d).scatter_(1, idx, w))

        total = torch.zeros(grads.shape, device=grads.device, dtype=torch.float32)
        norm = torch.zeros((grads.shape[0], 1), device=grads.device)
        n_classes = torch.zeros(grads.shape[0], device=grads.device)
        for c, Wc in zip(classes, Ws):
            used = (Wc > 0).any(dim=0)
            if used.any():
                total += Wc[:, used] @ self.cache[c].values[:self.cache[c].size][used].float()
                norm += Wc.sum(dim=1, keepdim=True)
                n_classes += (Wc > 0).any(dim=1).float()
        self.stats['nonempty_queues'].append(len(classes))
        self.stats['classes_in_set'].append(float(n_classes.mean()))
        return (total / norm.clamp_min(1e-30)).to(grads.dtype)
