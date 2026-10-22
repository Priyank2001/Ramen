"""
Step-4 fixes chosen from the support-set diagnostics (new_experiments/support_diagnostics.ipynb).
Both keep the cached keys and per-sample gradients exactly as in Ramen; only the aggregation weights
or the update direction change.

RamenClean (CIFAR-100-C: alpha-weighted support label purity 48.8% < 50%) -- contamination-aware weighting:
    alpha_ij <- alpha_ij * a_j ** gamma
    a_j = share of j's own alpha-weighted support (j itself excluded) that lies in j's queue, computed when j was
    the query and stored with the entry. An entry whose neighbourhood disagrees with its pseudo-label counts less.
    gamma = 0 is exactly Ramen. priority_dtype: 'float16' reproduces Ramen's cache priorities (which overflow to
    inf after 65,520 samples and then freeze full queues); 'float32' is the bug-fix ablation.

RamenNSGD (CIFAR-10-C: no diagnostic row fired) -- normalized SGD instead of SignSGD:
    w_i = w - lr_n * g_i / ||g_i||,  lr_n = lr * sqrt(D) when match_sign_step (same L2 step length as SignSGD).
"""
import torch

from .losses import softmax_entropy
from .Ramen import Ramen
from .RamenLog import LoggedPriorityCache


class ScoredPriorityCache(LoggedPriorityCache):
    """LoggedPriorityCache with one extra scalar per entry (the neighbour-agreement score)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = torch.ones((self.max_capacity,), device=self.device, dtype=torch.float32)

    def nearest(self, queries, topk):
        """Indices, distances, entropies and ids of the topk nearest entries; values are gathered by the caller."""
        topk = min(topk, self.size)
        queries = queries.detach().to(device=self.device, dtype=self.dtype)
        dist = torch.cdist(queries, self.keys[:self.size])
        sorted_dist, indices = torch.topk(dist, k=topk, dim=1, largest=False, sorted=True)
        return indices, sorted_dist, self.entropies[indices], self.ids[indices]

    def set_scores(self, ids, scores):
        slot = (self.ids[:self.size, None] == ids[None, :])  # (size, n)
        found = slot.any(0)
        if found.any():
            rows = slot[:, found].float().argmax(0)
            self.scores[rows] = scores[found].float()


class RamenClean(Ramen):

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.gamma = self.cfg.get('gamma', 1.0)
        self.priority_dtype = getattr(torch, self.cfg.get('priority_dtype', 'float16'))
        self.cache = [ScoredPriorityCache(max_capacity=self.cfg['max_capacity'],
                                          key_dim=self.feat_dim, value_dim=self.grad_dim,
                                          device=self.device, dtype=self.dtype)
                      for c in range(self.num_classes)]
        for cache in self.cache:
            cache.priorities = cache.priorities.to(self.priority_dtype)
        self.topk = self.cfg['topk']
        self.position = 0
        # n, inserted, sum of a_i, own-queue share of the alpha mass before / after reweighting
        self.stats = {'n': 0, 'inserted': 0, 'agreement': 0.0, 'own_share_before': 0.0, 'own_share_after': 0.0}

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

            # Update cache (same rule as Ramen)
            priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.priority_dtype)
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

            # Pass 1: Ramen's alpha for every queue, and each query's neighbour agreement
            retrieved = []
            own_mass = torch.zeros(B, device=self.device)
            total_mass = torch.zeros(B, device=self.device)
            for c in range(self.num_classes):
                if self.cache[c].size == 0:
                    continue
                indices, sorted_dist, entropies_c, ids_c = self.cache[c].nearest(feats, topk=self.topk)
                alpha = (torch.exp(-entropies_c) * torch.exp(-self.beta * sorted_dist)).float()
                not_self = (ids_c != ids[:, None]).float()
                mass = (alpha * not_self).sum(1)
                own_mass += torch.where(init_preds == c, mass, torch.zeros_like(mass))
                total_mass += mass
                retrieved.append((c, indices, alpha))
            agreement = torch.where(total_mass > 0, own_mass / total_mass.clamp_min(1e-12), torch.ones_like(own_mass))

            for c in range(self.num_classes):
                mine = init_preds == c
                if mine.any():
                    self.cache[c].set_scores(ids[mine], agreement[mine])

            # Pass 2: aggregate with alpha_ij * a_j ** gamma
            retrieved_grads_sum = torch.zeros_like(grads)
            own_before = torch.zeros(B, device=self.device)
            own_after = torch.zeros(B, device=self.device)
            sum_before = torch.zeros(B, device=self.device)
            sum_after = torch.zeros(B, device=self.device)
            for c, indices, alpha in retrieved:
                cache = self.cache[c]
                w = alpha * cache.scores[indices] ** self.gamma if self.gamma else alpha
                retrieved_grads_sum += (cache.values[indices] * w.unsqueeze(2).to(cache.values.dtype)).sum(dim=1)
                mine = (init_preds == c).float()
                own_before += mine * alpha.sum(1)
                own_after += mine * w.sum(1)
                sum_before += alpha.sum(1)
                sum_after += w.sum(1)

            retrieved_grads = retrieved_grads_sum / len(retrieved)
            self.model.set_by_sample_grad(retrieved_grads)

            self.stats['n'] += B
            self.stats['inserted'] += sum(inserted)
            self.stats['agreement'] += agreement.sum().item()
            self.stats['own_share_before'] += (own_before / sum_before.clamp_min(1e-12)).sum().item()
            self.stats['own_share_after'] += (own_after / sum_after.clamp_min(1e-12)).sum().item()

        # Update Model
        self.model.step_and_zero_grad()

        # Inference
        with torch.no_grad():
            logits = self.model(x)

        # reset model
        self.model.reset_parameters()

        return logits


class RamenNSGD(Ramen):

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        lr = self.cfg['lr'] * (self.grad_dim ** 0.5 if self.cfg.get('match_sign_step', True) else 1.0)
        params = [p for p in self.model.model.parameters() if p.requires_grad]
        self.model.optimizer = torch.optim.SGD(params, lr=lr)  # state-free (no momentum)

        set_grad = self.model.set_by_sample_grad

        def set_normalized_grad(g):
            g32 = g.float()
            set_grad((g32 / g32.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(g.dtype))

        self.model.set_by_sample_grad = set_normalized_grad
