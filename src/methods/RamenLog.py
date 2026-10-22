"""
Ramen with per-sample logging of the support set. Predictions are identical to Ramen.

For every query (row = stream position) it records:
  zs_probs      (N, C)    float32  zero-shot softmax (pretrained weights, before adapting)
  support_ids   (N, C, k) int32    stream position of each retrieved cache entry, per queue c (-1 = empty slot)
  support_alpha (N, C, k) float16  Ramen's aggregation weight exp(-H_j) * exp(-beta * d_ij) of that entry
  inserted      (N,)      bool     whether the query was written to its queue (False = rejected by the cache)
The queue label of support_ids[i, c, :] is c. Labels and domains of entries come from the results frame
(results row with position == id).
"""
import numpy as np
import torch

from .losses import softmax_entropy
from .Ramen import Ramen, PriorityCache


class LoggedPriorityCache(PriorityCache):
    """PriorityCache that also stores the stream position of each entry."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ids = torch.full((self.max_capacity,), -1, device=self.device, dtype=torch.long)

    def add(self, keys, values, entropies, priorities, ids):
        """Same replacement rule as PriorityCache.add; returns a bool per item (True = stored)."""
        keys = keys.detach().to(device=self.device, dtype=self.dtype)
        values = values.detach().to(device=self.device, dtype=self.dtype)
        if keys.ndim == 1:
            keys = torch.unsqueeze(keys, 0)
        if values.ndim == 1:
            values = torch.unsqueeze(values, 0)

        stored = []
        for i in range(keys.shape[0]):
            if self.size < self.max_capacity:
                slot = self.size
                self.size += 1
            else:
                slot = torch.argmin(self.priorities)
                if not priorities[i] > self.priorities[slot]:
                    stored.append(False)
                    continue
            self.keys[slot] = keys[i]
            self.values[slot] = values[i]
            self.priorities[slot] = priorities[i]
            self.entropies[slot] = entropies[i]
            self.ids[slot] = ids[i]
            stored.append(True)
        return stored

    def query(self, queries, topk):
        topk = min(topk, self.size)
        queries = queries.detach().to(device=self.device, dtype=self.dtype)
        dist = torch.cdist(queries, self.keys[:self.size])
        sorted_dist, indices = torch.topk(dist, k=topk, dim=1, largest=False, sorted=True)
        return (self.values[indices], self.priorities[indices], self.entropies[indices], sorted_dist,
                self.ids[indices])

    def reset(self):
        super().reset()
        self.ids.fill_(-1)


class RamenLog(Ramen):

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.cache = [LoggedPriorityCache(max_capacity=self.cfg['max_capacity'],
                                          key_dim=self.feat_dim, value_dim=self.grad_dim,
                                          device=self.device, dtype=self.dtype)
                      for c in range(self.num_classes)]
        self.topk = self.cfg['topk']
        self.position = 0  # global stream position, never reset (matches the results frame's `position`)
        self.log = {'zs_probs': [], 'support_ids': [], 'support_alpha': [], 'inserted': []}

    def forward(self, x):
        B = x.shape[0]

        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        init_preds = logits.argmax(-1)

        loss = self.loss_fn(logits)
        loss.backward()
        grads = self.model.get_by_sample_grad()

        with torch.no_grad():
            ids = torch.arange(self.position, self.position + B, device=self.device)
            self.position += B

            # Update cache
            priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.dtype)
            entropies = softmax_entropy(logits, reduction='none')
            self.counter = self.counter + B

            inserted = []
            for b in range(B):
                c = init_preds[b]
                inserted += self.cache[c].add(keys=feats[b].unsqueeze(0),
                                              values=grads[b].unsqueeze(0),
                                              entropies=entropies[b].unsqueeze(0),
                                              priorities=priorities[b].unsqueeze(0),
                                              ids=ids[b].unsqueeze(0))

            # Retrieve from cache
            retrieved_grads_sum = torch.zeros_like(grads)
            num_class_sum = 0
            sup_ids = torch.full((B, self.num_classes, self.topk), -1, device=self.device, dtype=torch.int32)
            sup_alpha = torch.zeros((B, self.num_classes, self.topk), device=self.device, dtype=torch.half)

            for c in range(self.num_classes):
                if self.cache[c].size == 0:
                    continue

                retrieved_grads, priorities_c, entropies_c, sorted_dist, ids_c = \
                    self.cache[c].query(feats, topk=self.topk)

                weights = torch.exp(-entropies_c)
                weights = weights * torch.exp(-self.beta * sorted_dist)

                retrieved_grads = (retrieved_grads * weights.unsqueeze(dim=2)).sum(dim=1)
                retrieved_grads_sum += retrieved_grads
                num_class_sum += 1

                k = ids_c.shape[1]
                sup_ids[:, c, :k] = ids_c.int()
                sup_alpha[:, c, :k] = weights.half()

            retrieved_grads = retrieved_grads_sum / num_class_sum
            self.model.set_by_sample_grad(retrieved_grads)

            self.log['zs_probs'].append(logits.float().softmax(-1).cpu().numpy())
            self.log['support_ids'].append(sup_ids.cpu().numpy())
            self.log['support_alpha'].append(sup_alpha.cpu().numpy())
            self.log['inserted'].append(np.array(inserted, dtype=bool))

        # Update Model
        self.model.step_and_zero_grad()

        # Inference
        with torch.no_grad():
            logits = self.model(x)

        # reset model
        self.model.reset_parameters()

        return logits

    def save_log(self, prefix):
        """Writes <prefix>_<key>.npy for each logged array (loadable with mmap_mode='r')."""
        for key, chunks in self.log.items():
            np.save(f'{prefix}_{key}.npy', np.concatenate(chunks))
