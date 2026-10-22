"""
Ramen with an active set picked or weighted by the query's own zero-shot prediction.

Ramen builds each query's active set from the top-k nearest entries of EVERY non-empty class queue and averages
them with equal class weight, so on ImageNet 1000 classes vote on every update, most of them irrelevant to the query.
Here each class queue's contribution is multiplied by a weight w[b, c] computed from the query's zero-shot logits:

    update_b  =  sum_c  w[b, c] * sum_{j in topk_c(b)} exp(-H_j) * exp(-beta * d_bj) * g_j   /   sum_c w[b, c]

Selectors (config key `active_set`):
  all    AllClasses      w = 1 for every class                               (== Ramen exactly)
  topm   TopMClasses     w = 1 for the query's top-m zero-shot classes, else 0  (config `top_m`)
  prob   ZeroShotProb    w = softmax(zero-shot logits / T)                   (config `temperature`)

Note: Ramen writes a batch into the memory before retrieving, so the query's own zero-shot class queue contains the
query itself at distance 0. With topk=1 that class always contributes the query's own gradient; `topm` with m=1 is
therefore per-sample adaptation, and `prob` up-weights the query's own gradient by its zero-shot confidence.

To add a selector: subclass ClassSelector, implement __call__, and add it to SELECTORS.

Per-sample log (self.stats, one array per batch, rows line up with ramen_analysis' results frame):
  filed_as        (B,)           class queue each sample was written into (its zero-shot prediction)
  support_ids     (B, n)         stream positions of the n retrieved entries with the largest weight (-1 = none)
  support_weight  (B, n)         their weight w[b, c] * exp(-H_j) * exp(-beta * d_bj), normalised to sum 1 per query
  queue_ids       (B, capacity)  stream positions in the queue of the predicted class at prediction time (-1 = empty);
                                 only if config `log_queue` is true (default), as it is large for big queues
n is the config key `log_support` (default 8). new_experiments/cache_viewer.py turns these into pictures.
"""
import numpy as np
import torch

from .losses import softmax_entropy
from .Ramen import PriorityCache, Ramen


# ---------------------------------------------------------------------------- class selectors
class ClassSelector:
    """Base class. Maps the queries' zero-shot logits (B, C) to non-negative class weights (B, C)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, zs_logits):
        raise NotImplementedError


class AllClasses(ClassSelector):
    """Ramen's rule: every class queue counts equally."""

    def __call__(self, zs_logits):
        return torch.ones_like(zs_logits, dtype=torch.float32)


class TopMClasses(ClassSelector):
    """Only the query's m most likely zero-shot classes."""

    def __call__(self, zs_logits):
        top = zs_logits.topk(self.cfg.get('top_m', 5), dim=1).indices
        return torch.zeros_like(zs_logits, dtype=torch.float32).scatter_(1, top, 1.0)


class ZeroShotProb(ClassSelector):
    """Each class queue weighted by CLIP's zero-shot probability of that class for the query."""

    def __call__(self, zs_logits):
        return (zs_logits.float() / self.cfg.get('temperature', 1.0)).softmax(dim=1)


SELECTORS = {
    'all': AllClasses,
    'topm': TopMClasses,
    'prob': ZeroShotProb,
}


# ---------------------------------------------------------------------------- memory
class IdCache(PriorityCache):
    """Ramen's PriorityCache that also remembers which stream sample (position in the stream) each entry is."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ids = torch.full((self.max_capacity,), -1, device=self.device, dtype=torch.long)

    def add(self, keys, values, entropies, priorities, ids):
        # Same slot rule as PriorityCache.add: append while there is room, else replace the lowest priority if the
        # new one is higher. Priorities stay fp16 as in Ramen (they saturate after 65,504 samples), so that
        # selector `all` reproduces Ramen exactly.
        for key, value, entropy, priority, id_ in zip(keys, values, entropies, priorities, ids):
            if self.size < self.max_capacity:
                slot = self.size
                self.size += 1
            else:
                slot = torch.argmin(self.priorities)
                if not priority > self.priorities[slot]:
                    continue
            self.keys[slot], self.values[slot] = key, value
            self.priorities[slot], self.entropies[slot], self.ids[slot] = priority, entropy, id_

    def query(self, queries, topk):
        """As PriorityCache.query, plus the stream ids of the retrieved entries."""
        topk = min(topk, self.size)
        dist = torch.cdist(queries.detach().to(device=self.device, dtype=self.dtype), self.keys[:self.size])
        sorted_dist, idx = torch.topk(dist, k=topk, dim=1, largest=False, sorted=True)
        return self.values[idx], self.entropies[idx], sorted_dist, self.ids[idx]

    def stored_ids(self):
        return self.ids[:self.size]


# ---------------------------------------------------------------------------- the method
class RamenActive(Ramen):

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        name = self.cfg.get('active_set', 'all')
        if name not in SELECTORS:
            raise ValueError(f'Unknown active_set {name!r}; choose from {list(SELECTORS)}')
        self.selector = SELECTORS[name](self.cfg)
        self.n_log = self.cfg.get('log_support', 8)
        self.log_queue = self.cfg.get('log_queue', True)

        self.cache = None  # free Ramen's queues first: both at once need 2 x 8 GB on DomainNet
        torch.cuda.empty_cache()
        self.cache = [IdCache(max_capacity=self.cfg['max_capacity'], key_dim=self.feat_dim, value_dim=self.grad_dim,
                              device=self.device, dtype=self.dtype)
                      for _ in range(self.num_classes)]
        self.seen = 0  # stream position of the next sample; unlike self.counter it is not reset between streams
        self.stats = {'active_set': name, 'filed_as': [], 'support_ids': [], 'support_weight': []}
        if self.log_queue:
            self.stats['queue_ids'] = []

    @torch.no_grad()
    def write(self, feats, grads, zs_logits, ids):
        """Ramen's write: every sample, before retrieval, into the queue of its zero-shot prediction."""
        B = feats.shape[0]
        zs_pred = zs_logits.argmax(-1)
        priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.dtype)
        entropies = softmax_entropy(zs_logits, reduction='none')
        self.counter = self.counter + B
        for b in range(B):
            self.cache[zs_pred[b]].add(feats[b:b + 1], grads[b:b + 1], entropies[b:b + 1], priorities[b:b + 1],
                                       ids[b:b + 1])
        self.stats['filed_as'].append(zs_pred.cpu().numpy().astype(np.int32))

    @torch.no_grad()
    def aggregate(self, feats, grads, class_w):
        """Ramen's retrieval, with each class queue's contribution scaled by class_w[:, c]."""
        total = torch.zeros_like(grads)
        norm = torch.zeros((grads.shape[0], 1), device=grads.device)
        entry_w, entry_ids = [], []  # for the log: every retrieved entry's weight and stream id
        active = ((class_w > 0).any(dim=0)).tolist()
        for c in range(self.num_classes):
            if self.cache[c].size == 0 or not active[c]:
                continue
            retrieved, entropies, dist, ids = self.cache[c].query(feats, topk=self.cfg['topk'])
            weights = torch.exp(-entropies) * torch.exp(-self.beta * dist)  # (B, k), as in Ramen
            total += class_w[:, c:c + 1] * (retrieved * weights.unsqueeze(dim=2)).sum(dim=1)
            norm += class_w[:, c:c + 1]
            entry_w.append(class_w[:, c:c + 1] * weights.float())
            entry_ids.append(ids)
        self.log_support(entry_w, entry_ids, grads.shape[0])
        # An all-zero weight row gives a zero gradient: SignSGD then doesn't move, so the prediction is zero-shot.
        return total / norm.clamp_min(1e-12)

    def log_support(self, entry_w, entry_ids, B):
        n = self.n_log
        ids = torch.full((B, n), -1, dtype=torch.long)
        w = torch.zeros((B, n))
        if entry_w:
            entry_w, entry_ids = torch.cat(entry_w, dim=1), torch.cat(entry_ids, dim=1)
            top_w, top = entry_w.topk(min(n, entry_w.shape[1]), dim=1)
            ids[:, :top.shape[1]] = entry_ids.gather(1, top).cpu()
            w[:, :top.shape[1]] = (top_w / entry_w.sum(dim=1, keepdim=True).clamp_min(1e-12)).cpu()
        ids[w == 0] = -1
        self.stats['support_ids'].append(ids.numpy().astype(np.int32))
        self.stats['support_weight'].append(w.numpy().astype(np.float32))

    def log_queues(self, pred):
        """Snapshot of the stream ids stored in the queue of each sample's predicted class."""
        snap = np.full((len(pred), self.cfg['max_capacity']), -1, dtype=np.int32)
        for b, c in enumerate(pred.tolist()):
            stored = self.cache[c].stored_ids().cpu().numpy()
            snap[b, :len(stored)] = stored
        self.stats['queue_ids'].append(snap)

    def forward(self, x):
        B = x.shape[0]
        ids = torch.arange(self.seen, self.seen + B, device=self.device)
        self.seen += B

        # 1. Zero-shot features, logits and each sample's own entropy gradient (as in Ramen).
        feats = self.model.featurize(x)
        zs_logits = self.model.classify(feats)
        self.loss_fn(zs_logits).backward()
        grads = self.model.get_by_sample_grad()

        # 2. Write, then retrieve with zero-shot class weights, take one step, predict, reset.
        self.write(feats, grads, zs_logits, ids)
        self.model.set_by_sample_grad(self.aggregate(feats, grads, self.selector(zs_logits.detach())))
        self.model.step_and_zero_grad()
        with torch.no_grad():
            logits = self.model(x)
        self.model.reset_parameters()

        if self.log_queue:
            self.log_queues(logits.argmax(-1))
        return logits

    def reset(self):
        super().reset()  # resets the model, Ramen's priority counter and the queues; self.seen keeps counting
