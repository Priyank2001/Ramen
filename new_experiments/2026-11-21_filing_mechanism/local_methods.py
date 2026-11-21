"""
E2: a causal test of "containment" (why filing Ramen's cache by the TRUE label lowers accuracy).

Containment says: under true-label filing, images that CLIP confuses the same way (same wrong zero-shot class) are
spread over many class queues, so a query's update averages many entries whose entropy gradient pushes toward the
query's own wrong zero-shot class. If that is the cause, HIDING exactly those entries from the query should give
back most of the loss, and hiding the same NUMBER of random entries should not.

RamenHideConfusion = RamenTrueLabel (src/methods/RamenTrueLabel.py, filing by `filing`) + a per-query hidden set:

  hide: none          nothing hidden: RamenTrueLabel exactly (Ramen exactly with filing zero_shot)
  hide: same_zs       hide entries j with zs_j == zs_i (same zero-shot class as the query), except the query's own
  hide: random        hide n_i random entries, n_i = |{j != i : zs_j == zs_i}| (matched count), never the own entry
  hide_top_m: m       (E3) widen "same zero-shot class" to "zero-shot class among the query's top-m zero-shot classes"
                      (query logits; label-free). m = 1 (default) is the same-zero-shot-class rule of E2, bit-identical.
                      The random control hides as many random entries as that (wider) rule does.
  hide_on: all        apply the rule to every query                        (label-free rule)
  hide_on: zs_wrong   apply it only to queries whose zero-shot class is wrong   (ORACLE gate: uses the true label)

Retrieval is RamenTrueLabel.retrieve with hidden entries removed before the per-queue top-k:
  N_k^c(z_i) = k nearest VISIBLE entries of queue c;  g_i = (1 / |C_i|) sum_{c in C_i} sum_j e^{-H_j} e^{-beta d_ij} g_j
  C_i = queues with >= 1 visible entry for query i (Ramen: non-empty queues). With hide none this is bit-identical
  to RamenTrueLabel.retrieve (same cdist / topk calls, same integer queue count).

Entry tags (stream id, zero-shot class) are kept by this class in (C, capacity) arrays, filled at the slot that
PriorityCache.add will use (same rule: next free slot, else the first minimum-priority slot if the new priority is
larger). The repo's cache code is not copied or changed.

Logged per batch (self.stats): zs_pred, hidden (n hidden per query), visible_queues (|C_i| per query),
hidden_true_class (how many hidden entries were zero-shot-predicted as the query's TRUE class: a diagnostic that
uses labels, never read by the method).
"""
import torch

from methods.RamenTrueLabel import RamenTrueLabel  # src/ is on the path once ramen_analysis is imported
from methods.losses import softmax_entropy

HIDE_RULES = ('none', 'same_zs', 'random')
HIDE_ON = ('all', 'zs_wrong')


class RamenHideConfusion(RamenTrueLabel):

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.hide = self.cfg.get('hide', 'none')
        self.hide_on = self.cfg.get('hide_on', 'all')
        if self.hide not in HIDE_RULES or self.hide_on not in HIDE_ON:
            raise ValueError(f'hide must be one of {HIDE_RULES}, hide_on one of {HIDE_ON}')
        cap = self.cfg['max_capacity']
        self.tag_id = torch.full((self.num_classes, cap), -1, dtype=torch.long, device=self.device)
        self.tag_zs = torch.full((self.num_classes, cap), -1, dtype=torch.long, device=self.device)
        self.gen = torch.Generator(device=self.device).manual_seed(int(self.cfg.get('hide_seed', 0)))
        self.top_m = int(self.cfg.get('hide_top_m', 1))
        self.stats.update(hide=self.hide, hide_on=self.hide_on, hide_top_m=self.top_m, zs_pred=[], hidden=[],
                          visible_queues=[], hidden_true_class=[])

    # ------------------------------------------------------------------ write: Ramen's, plus entry tags
    def _slot(self, cache, priority):
        """The slot PriorityCache.add will write to, or None if it will drop the entry (src/methods/Ramen.py)."""
        if cache.size < cache.max_capacity:
            return cache.size
        slot = int(torch.argmin(cache.priorities))
        return slot if priority > cache.priorities[slot] else None

    @torch.no_grad()
    def write(self, feats, grads, logits, queues, zs_pred=None, sids=None):
        B = feats.shape[0]
        priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.priority_dtype)
        entropies = softmax_entropy(logits, reduction='none')
        self.counter = self.counter + B
        for b in range(B):
            c = int(queues[b])
            slot = self._slot(self.cache[c], priorities[b])
            self.cache[c].add(keys=feats[b].unsqueeze(0), values=grads[b].unsqueeze(0),
                              entropies=entropies[b].unsqueeze(0), priorities=priorities[b].unsqueeze(0))
            if slot is not None:
                self.tag_id[c, slot], self.tag_zs[c, slot] = sids[b], zs_pred[b]

    # ------------------------------------------------------------------ hidden sets
    def confused_with(self, zs_pred, logits):
        """(B, C, capacity) bool: entries whose zero-shot class is the query's (top_m = 1) or among its top-m."""
        if self.top_m == 1 or logits is None:
            return self.tag_zs[None] == zs_pred[:, None, None]
        top = logits.topk(self.top_m, dim=1).indices  # (B, m) the query's top-m zero-shot classes
        return (self.tag_zs[None, :, :, None] == top[:, None, None, :]).any(dim=3)

    def hidden(self, zs_pred, sids, labels, logits=None):
        """(B, C, capacity) bool: entries hidden from each query, or None (nothing hidden)."""
        if self.hide == 'none':
            return None
        filled = self.tag_id >= 0
        not_self = self.tag_id[None] != sids[:, None, None]
        same = self.confused_with(zs_pred, logits) & filled[None] & not_self
        if self.hide == 'same_zs':
            mask = same
        else:  # random, matched count per query
            B = zs_pred.shape[0]
            n = same.flatten(1).sum(1)
            cand = (filled[None] & not_self).flatten(1)
            r = torch.rand(cand.shape, generator=self.gen, device=self.device)
            r = torch.where(cand, r, torch.full_like(r, 2.0))  # non-candidates rank last
            ranks = torch.empty_like(r, dtype=torch.long)
            order = r.argsort(dim=1)
            ranks.scatter_(1, order, torch.arange(r.shape[1], device=self.device).expand(B, -1))
            mask = (ranks < n[:, None]).view_as(same)
        if self.hide_on == 'zs_wrong':
            mask = mask & (zs_pred != labels)[:, None, None]
        return mask

    # ------------------------------------------------------------------ retrieve with hidden entries removed
    @torch.no_grad()
    def retrieve(self, feats, grads, hide=None):
        if hide is None:  # RamenTrueLabel.retrieve, bit-identical
            out = super().retrieve(feats, grads)
            self.stats['visible_queues'].append(self.stats['nonempty_queues'][-1])
            return out
        total = torch.zeros_like(grads)
        n_queues = torch.zeros(feats.shape[0], device=self.device)
        topk = self.cfg['topk']
        for c in range(self.num_classes):
            cache = self.cache[c]
            if cache.size == 0:
                continue
            q = feats.detach().to(device=cache.device, dtype=cache.dtype)
            dist = torch.cdist(q, cache.keys[:cache.size], compute_mode='use_mm_for_euclid_dist')
            dist = dist.masked_fill(hide[:, c, :cache.size], float('inf'))
            sd, idx = torch.topk(dist, k=min(topk, cache.size), dim=1, largest=False, sorted=True)
            visible = torch.isfinite(sd)
            w = torch.exp(-cache.entropies[idx]) * torch.exp(-self.beta * sd)
            w = torch.where(visible, w, torch.zeros_like(w))
            total += (cache.values[idx] * w.unsqueeze(2)).sum(dim=1)
            n_queues += visible.any(dim=1)
        self.stats['nonempty_queues'].append(sum(int(self.cache[c].size > 0) for c in range(self.num_classes)))
        self.stats['visible_queues'].append(n_queues.int().cpu())
        return total / n_queues.clamp_min(1).unsqueeze(1).to(total.dtype)

    def forward(self, x, labels=None):
        if labels is None:
            raise ValueError('RamenHideConfusion needs the true labels: run it through the harness (records_labels)')
        labels = labels.to(self.device)
        B = x.shape[0]
        sids = torch.arange(self.counter, self.counter + B, device=self.device)  # stream ids of this batch

        # 1. Zero-shot features, logits and each image's own entropy gradient (as in Ramen).
        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        zs_pred = logits.argmax(-1)
        self.loss_fn(logits).backward()
        grads = self.model.get_by_sample_grad()
        self.stats['zs_wrong'].append(int((zs_pred != labels).sum()))
        self.stats['zs_pred'].append(zs_pred.cpu())

        # 2. Write (with tags), retrieve without the hidden entries, one step, predict, reset.
        self.write(feats, grads, logits, self.queue_of(zs_pred, labels), zs_pred=zs_pred, sids=sids)
        hide = self.hidden(zs_pred, sids, labels, logits)
        self.stats['hidden'].append(torch.zeros(B, dtype=torch.long) if hide is None
                                    else hide.flatten(1).sum(1).cpu())
        self.stats['hidden_true_class'].append(torch.zeros(B, dtype=torch.long) if hide is None else
                                               (hide & (self.tag_zs[None] == labels[:, None, None]))
                                               .flatten(1).sum(1).cpu())
        self.model.set_by_sample_grad(self.retrieve(feats, grads, hide))
        self.model.step_and_zero_grad()
        with torch.no_grad():
            logits = self.model(x)
        self.model.reset_parameters()
        return logits
