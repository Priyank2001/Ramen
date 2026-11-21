"""
Ramen with ORACLE control over cache contamination: an upper bound for any cache-cleaning method.

Ramen writes every test image, before retrieval, into the queue of its zero-shot prediction, together with the
image's own entropy-minimisation gradient and its entropy H. Retrieval weights an entry by exp(-H) * exp(-beta * d),
so a confidently wrong image (low H) gets close to the largest weight there is, and its gradient sharpens its own
wrong prediction. These variants use the TRUE labels (which no real method has) to remove that contamination, to
measure how much accuracy a perfect contamination detector could buy.

Variants (config key `oracle`):
  none                  Ramen exactly (sanity check: 100% the same predictions)
  file_true             file every image under its TRUE class: every queue is 100% pure. The stored gradient is
                        unchanged (it still sharpens the zero-shot prediction).
  drop_wrong            file as Ramen, but never retrieve an entry whose zero-shot prediction was wrong; each queue
                        returns its nearest correctly-predicted entries instead
  drop_confident_wrong  never retrieve an entry that is wrong AND confident: entropy below the running median
                        entropy of all images seen so far (causal: the median at the time the image was written)
  random_like_wrong     control for drop_wrong: hide each entry with probability = the running zero-shot error
                        rate at the time it was written, so about as many entries are hidden, but at random
  random_like_cw        control for drop_confident_wrong: the same, with the running confident-wrong rate

Hidden entries stay in memory (they still occupy slots and are evicted by Ramen's rule); only retrieval skips them.
A queue with no visible entry is skipped for that batch, as Ramen skips an empty queue.
The decision for each entry is fixed when it is written. Labels are used ONLY for these oracle masks and the logs.

Per-batch log (self.stats), on top of RamenActive's filed_as / support_ids / support_weight:
  support_wrong   (B,)  share of the query's update weight from entries whose zero-shot prediction was wrong
  support_cw      (B,)  share from wrong-and-confident entries
  queues_used     (B,)  number of class queues that contributed to the query's update
  memory          list of (stream position, entries, wrong entries, confident-wrong entries, hidden entries),
                  a snapshot of the whole memory every `memory_log_every` batches (default 10)
"""
import numpy as np
import torch

from .losses import softmax_entropy
from .RamenActive import IdCache, RamenActive

VARIANTS = ('none', 'file_true', 'drop_wrong', 'drop_confident_wrong', 'random_like_wrong', 'random_like_cw')


# ---------------------------------------------------------------------------- memory
class OracleCache(IdCache):
    """IdCache whose entries also carry: true label, zero-shot wrong, confident-wrong, and a hidden flag."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = torch.full((self.max_capacity,), -1, device=self.device, dtype=torch.long)
        self.wrong = torch.zeros(self.max_capacity, device=self.device, dtype=torch.bool)
        self.cwrong = torch.zeros(self.max_capacity, device=self.device, dtype=torch.bool)
        self.hidden = torch.zeros(self.max_capacity, device=self.device, dtype=torch.bool)

    def add(self, keys, values, entropies, priorities, ids, labels, wrong, cwrong, hidden):
        # PriorityCache's slot rule, unchanged (fp16 priorities as in Ramen, so `none` == Ramen).
        for key, value, entropy, priority, id_, y, w, cw, h in zip(keys, values, entropies, priorities, ids,
                                                                    labels, wrong, cwrong, hidden):
            if self.size < self.max_capacity:
                slot = self.size
                self.size += 1
            else:
                slot = torch.argmin(self.priorities)
                if not priority > self.priorities[slot]:
                    continue
            self.keys[slot], self.values[slot] = key, value
            self.priorities[slot], self.entropies[slot], self.ids[slot] = priority, entropy, id_
            self.labels[slot], self.wrong[slot], self.cwrong[slot], self.hidden[slot] = y, w, cw, h

    def query(self, queries, topk):
        """Ramen's top-k by feature distance over the VISIBLE entries; None if the queue has none."""
        visible = ~self.hidden[:self.size]
        n = int(visible.sum())
        if n == 0:
            return None
        dist = torch.cdist(queries.detach().to(device=self.device, dtype=self.dtype), self.keys[:self.size])
        if n < self.size:
            dist = dist.masked_fill(~visible, float('inf'))
        sorted_dist, idx = torch.topk(dist, k=min(topk, n), dim=1, largest=False, sorted=True)
        return self.values[idx], self.entropies[idx], sorted_dist, self.ids[idx], idx

    def counts(self):
        s = self.size
        return s, int(self.wrong[:s].sum()), int(self.cwrong[:s].sum()), int(self.hidden[:s].sum())

    def reset(self):
        super().reset()
        self.ids.fill_(-1)
        self.labels.fill_(-1)
        self.wrong.zero_()
        self.cwrong.zero_()
        self.hidden.zero_()


# ---------------------------------------------------------------------------- the method
class RamenOracle(RamenActive):

    records_labels = True  # the harness passes the true labels to forward(x, labels=...)

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.variant = self.cfg.get('oracle', 'none')
        if self.variant not in VARIANTS:
            raise ValueError(f'Unknown oracle {self.variant!r}; choose from {list(VARIANTS)}')
        self.rng = torch.Generator(device='cpu').manual_seed(int(self.cfg.get('oracle_seed', 0)))
        self.log_every = int(self.cfg.get('memory_log_every', 10))

        self.cache = None
        torch.cuda.empty_cache()
        self.cache = [OracleCache(max_capacity=self.cfg['max_capacity'], key_dim=self.feat_dim,
                                  value_dim=self.grad_dim, device=self.device, dtype=self.dtype)
                      for _ in range(self.num_classes)]
        self.seen_entropy = torch.empty(0, device=self.device)  # sorted entropies of every image so far
        self.n_seen = self.n_wrong = self.n_cwrong = 0
        self.batches = 0
        self.stats.update({'oracle': self.variant, 'support_wrong': [], 'support_cw': [], 'queues_used': [],
                           'memory': []})

    @torch.no_grad()
    def write(self, feats, grads, zs_logits, ids, labels):
        """Ramen's write, plus each entry's oracle flags (fixed at write time, causal running statistics)."""
        B = feats.shape[0]
        zs_pred = zs_logits.argmax(-1)
        entropies = softmax_entropy(zs_logits, reduction='none')
        priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.dtype)
        self.counter = self.counter + B

        # running statistics over every image seen so far, this batch included
        self.seen_entropy = torch.cat([self.seen_entropy, entropies.float()]).sort().values
        median = self.seen_entropy[(len(self.seen_entropy) - 1) // 2]
        wrong = zs_pred != labels
        cwrong = wrong & (entropies.float() < median)
        self.n_seen += B
        self.n_wrong += int(wrong.sum())
        self.n_cwrong += int(cwrong.sum())

        if self.variant == 'drop_wrong':
            hidden = wrong
        elif self.variant == 'drop_confident_wrong':
            hidden = cwrong
        elif self.variant in ('random_like_wrong', 'random_like_cw'):
            rate = (self.n_wrong if self.variant == 'random_like_wrong' else self.n_cwrong) / self.n_seen
            hidden = (torch.rand(B, generator=self.rng) < rate).to(self.device)
        else:
            hidden = torch.zeros(B, dtype=torch.bool, device=self.device)
        queue = labels if self.variant == 'file_true' else zs_pred

        for b in range(B):
            self.cache[queue[b]].add(feats[b:b + 1], grads[b:b + 1], entropies[b:b + 1], priorities[b:b + 1],
                                     ids[b:b + 1], labels[b:b + 1], wrong[b:b + 1], cwrong[b:b + 1],
                                     hidden[b:b + 1])
        self.stats['filed_as'].append(queue.cpu().numpy().astype(np.int32))

    @torch.no_grad()
    def aggregate(self, feats, grads, class_w):
        """RamenActive.aggregate over the visible entries, plus the contamination of each query's active set."""
        B = grads.shape[0]
        total = torch.zeros_like(grads)
        norm = torch.zeros((B, 1), device=grads.device)
        entry_w, entry_ids = [], []
        w_sum = torch.zeros(B, device=grads.device)
        w_wrong = torch.zeros(B, device=grads.device)
        w_cw = torch.zeros(B, device=grads.device)
        used = torch.zeros(B, device=grads.device)
        active = ((class_w > 0).any(dim=0)).tolist()
        for c in range(self.num_classes):
            cache = self.cache[c]
            if cache.size == 0 or not active[c]:
                continue
            hit = cache.query(feats, topk=self.cfg['topk'])
            if hit is None:  # every entry hidden: skip the queue, as Ramen skips an empty one
                continue
            retrieved, entropies, dist, ids, slots = hit
            weights = torch.exp(-entropies) * torch.exp(-self.beta * dist)  # (B, k), as in Ramen
            total += class_w[:, c:c + 1] * (retrieved * weights.unsqueeze(dim=2)).sum(dim=1)
            norm += class_w[:, c:c + 1]

            w = class_w[:, c:c + 1] * weights.float()
            entry_w.append(w)
            entry_ids.append(ids)
            w_sum += w.sum(dim=1)
            w_wrong += (w * cache.wrong[slots]).sum(dim=1)
            w_cw += (w * cache.cwrong[slots]).sum(dim=1)
            used += (class_w[:, c] > 0).float()
        self.log_support(entry_w, entry_ids, B)

        w_sum = w_sum.clamp_min(1e-12)
        self.stats['support_wrong'].append((w_wrong / w_sum).cpu().numpy().astype(np.float32))
        self.stats['support_cw'].append((w_cw / w_sum).cpu().numpy().astype(np.float32))
        self.stats['queues_used'].append(used.cpu().numpy().astype(np.int32))
        return total / norm.clamp_min(1e-12)

    def log_memory(self):
        counts = np.array([cache.counts() for cache in self.cache]).sum(axis=0)
        self.stats['memory'].append((self.seen, *counts.tolist()))

    def forward(self, x, labels=None):
        if labels is None:
            raise ValueError('RamenOracle needs the true labels: run it through the harness (records_labels)')
        B = x.shape[0]
        ids = torch.arange(self.seen, self.seen + B, device=self.device)
        self.seen += B
        labels = labels.to(self.device)

        # 1. Zero-shot features, logits and each sample's own entropy gradient (as in Ramen).
        feats = self.model.featurize(x)
        zs_logits = self.model.classify(feats)
        self.loss_fn(zs_logits).backward()
        grads = self.model.get_by_sample_grad()

        # 2. Write (with oracle flags), retrieve from the visible entries, take one step, predict, reset.
        self.write(feats, grads, zs_logits, ids, labels)
        self.model.set_by_sample_grad(self.aggregate(feats, grads, self.selector(zs_logits.detach())))
        self.model.step_and_zero_grad()
        with torch.no_grad():
            logits = self.model(x)
        self.model.reset_parameters()

        if self.log_queue:
            self.log_queues(logits.argmax(-1))
        self.batches += 1
        if self.batches % self.log_every == 0:
            self.log_memory()
        return logits
