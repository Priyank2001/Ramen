"""
Ramen with a choice of active-set policy: which memory entries a query's update is built from.

The memory is Ramen's, unchanged: every sample is written, before retrieval, into the queue of its zero-shot class
(capacity `max_capacity`, newest replaces oldest). Only the retrieval differs. Each policy gives every stored entry j a
weight W[b, j] for query b (0 = not in the active set), and the update is

    update_b  =  sum_j W[b, j] * g_j  /  sum_j W[b, j]          (SignSGD, so only the direction matters)

Policies (config key `policy`):
  per_class   n nearest entries from EVERY non-empty class queue, each class queue counts once.    knob `per_class_n`
              W = exp(-H_j) * exp(-beta * d_bj) on the n nearest of each queue.  n=1 is Ramen (topk: 1).
  global_knn  the k nearest entries of the whole memory, whatever queue they sit in.             knob `global_k`
              W = exp(-H_j) * exp(-beta * d_bj) on those k entries. Classes with many nearby entries vote more.
  zs_prob     per_class n=1 (Ramen's active set), each class queue scaled by the query's zero-shot  knob `temperature`
              probability softmax(logits / T) of that class (== RamenActive active_set 'prob').

Ramen's per-class rule divides by the number of non-empty queues instead of sum_j W; under SignSGD that is the same
update. Every policy sees the query itself in memory at distance 0 (write before retrieve), so global_knn with k=1 is
per-sample adaptation.

The sum is computed as one (B, size_c) @ (size_c, grad_dim) product per class queue in fp32, instead of Ramen's
gather-and-sum in fp16, so large n or k stay cheap. per_class with n=1 therefore matches Ramen up to fp16 rounding
of near-zero gradient components (SignSGD flips them); the notebook reports the agreement.

Logging is RamenActive's (filed_as, support_ids, support_weight, queue_ids), plus per query:
  n_active  (B,)  number of entries with non-zero weight
"""
import numpy as np
import torch

from .RamenActive import RamenActive

POLICIES = ('per_class', 'global_knn', 'zs_prob')


class RamenPolicy(RamenActive):

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.policy = self.cfg.get('policy', 'per_class')
        if self.policy not in POLICIES:
            raise ValueError(f'Unknown policy {self.policy!r}; choose from {POLICIES}')
        self.stats['active_set'] = self.policy
        self.stats['n_active'] = []

    @torch.no_grad()
    def entry_weights(self, feats, zs_logits):
        """Per non-empty class queue c: (c, W_c) with W_c (B, size_c) the weight of each stored entry."""
        classes = [c for c in range(self.num_classes) if self.cache[c].size > 0]
        q = feats.detach().to(device=self.device, dtype=self.dtype)
        dists = [torch.cdist(q, self.cache[c].keys[:self.cache[c].size]).float() for c in classes]
        base = [torch.exp(-self.cache[c].entropies[:self.cache[c].size].float())[None] * torch.exp(-self.beta * d)
                for c, d in zip(classes, dists)]

        if self.policy in ('per_class', 'zs_prob'):
            n = 1 if self.policy == 'zs_prob' else self.cfg.get('per_class_n', 1)
            W = []
            for d, w in zip(dists, base):
                keep = torch.zeros_like(w)
                keep.scatter_(1, d.topk(min(n, d.shape[1]), dim=1, largest=False).indices, 1.0)
                W.append(w * keep)
            if self.policy == 'zs_prob':
                p = (zs_logits.float() / self.cfg.get('temperature', 1.0)).softmax(dim=1)
                W = [Wc * p[:, c:c + 1] for c, Wc in zip(classes, W)]
        else:  # global_knn: one top-k over the concatenation of all queues
            d_all = torch.cat(dists, dim=1)
            keep = torch.zeros_like(d_all)
            keep.scatter_(1, d_all.topk(min(self.cfg.get('global_k', 100), d_all.shape[1]), dim=1,
                                        largest=False).indices, 1.0)
            W = list(torch.split(torch.cat(base, dim=1) * keep, [d.shape[1] for d in dists], dim=1))
        return list(zip(classes, W))

    @torch.no_grad()
    def aggregate(self, feats, grads, zs_logits):
        total = torch.zeros(grads.shape, device=grads.device, dtype=torch.float32)
        norm = torch.zeros((grads.shape[0], 1), device=grads.device)
        n_active = torch.zeros(grads.shape[0], device=grads.device)
        entry_w, entry_ids = [], []
        for c, W in self.entry_weights(feats, zs_logits):
            used = (W > 0).any(dim=0)
            if not used.any():
                continue
            W, cache = W[:, used], self.cache[c]
            total += W @ cache.values[:cache.size][used].float()
            norm += W.sum(dim=1, keepdim=True)
            n_active += (W > 0).sum(dim=1)
            entry_w.append(W)
            entry_ids.append(cache.ids[:cache.size][used][None].expand(W.shape[0], -1))
        self.log_support(entry_w, entry_ids, grads.shape[0])
        self.stats['n_active'].append(n_active.cpu().numpy().astype(np.int32))
        return (total / norm.clamp_min(1e-30)).to(grads.dtype)

    def forward(self, x):
        B = x.shape[0]
        ids = torch.arange(self.seen, self.seen + B, device=self.device)
        self.seen += B

        feats = self.model.featurize(x)
        zs_logits = self.model.classify(feats)
        self.loss_fn(zs_logits).backward()
        grads = self.model.get_by_sample_grad()

        self.write(feats, grads, zs_logits, ids)
        self.model.set_by_sample_grad(self.aggregate(feats, grads, zs_logits.detach()))
        self.model.step_and_zero_grad()
        with torch.no_grad():
            logits = self.model(x)
        self.model.reset_parameters()

        if self.log_queue:
            self.log_queues(logits.argmax(-1))
        return logits
