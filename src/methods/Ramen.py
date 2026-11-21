import torch

from models.ModelForBySampleTTA import CLIPModelForBySampleTTA

from .losses import softmax_entropy
from .TTABase import TTABase


class PriorityCache:
    """
    Store key-value pairs
    """

    def __init__(self, max_capacity, key_dim, value_dim, device, dtype, priority_dtype=None):

        self.max_capacity = max_capacity
        self.key_dim = key_dim  # for embedding
        self.value_dim = value_dim  # for grad
        self.device = device
        self.dtype = dtype

        self.keys = torch.empty((max_capacity, key_dim), device=device, dtype=dtype)
        self.values = torch.empty((max_capacity, value_dim), device=device, dtype=dtype)

        # fp16 priorities (the default, as in the original code) become inf after 65,520 inserts and full queues freeze
        self.priorities = torch.full((max_capacity,), float('-inf'), device=device, dtype=priority_dtype or dtype)
        self.entropies = torch.full((max_capacity,), float('inf'), device=device, dtype=dtype)
        self.size = 0

    def add(self, keys, values, entropies, priorities):

        keys = keys.detach().to(device=self.device, dtype=self.dtype)
        values = values.detach().to(device=self.device, dtype=self.dtype)

        if keys.ndim == 1:
            keys = torch.unsqueeze(keys, 0)
        if values.ndim == 1:
            values = torch.unsqueeze(values, 0)

        # now both key and value is B * key_dim

        batch_size = keys.shape[0]

        # Use for loop to add each (key, value) pair

        for i in range(batch_size):
            key = keys[i]
            value = values[i]
            priority = priorities[i]
            entropy = entropies[i]

            if self.size < self.max_capacity:
                self.keys[self.size] = key
                self.values[self.size] = value
                self.priorities[self.size] = priority
                self.entropies[self.size] = entropy
                self.size += 1

            else:
                min_idx = torch.argmin(self.priorities)
                if priority > self.priorities[min_idx]:
                    self.keys[min_idx] = key
                    self.values[min_idx] = value
                    self.priorities[min_idx] = priority
                    self.entropies[min_idx] = entropy

    def query(self, queries, topk):

        topk = min(topk, self.size)

        queries = queries.detach().to(device=self.device, dtype=self.dtype)
        # mm path: the only cdist path implemented for fp16 on CUDA; it is also what the default picks whenever either
        # side has > 25 rows, so results are unchanged (the default crashed on a <= 25 x <= 25 query, e.g. a 1-image batch)
        dist = torch.cdist(queries, self.keys[:self.size], compute_mode='use_mm_for_euclid_dist')
        sorted_dist, indices = torch.topk(dist, k=topk, dim=1, largest=False, sorted=True)

        values = self.values[indices]  # num_queries * topk * value_dim
        priorities = self.priorities[indices]
        entropies = self.entropies[indices]

        return values, priorities, entropies, sorted_dist

    def reset(self):
        self.size = 0  # lazy reset


class Ramen(TTABase):

    def __init__(self, model, datasets, args):
        super().__init__()

        self.cfg = args.config
        self.beta = args.config.get('beta', 0.0)

        self.num_classes = datasets.num_classes
        self.device = next(model.parameters()).device
        self.dtype = torch.half
        # priority_fp32: true keeps queues updating past 65,520 samples (eviction only; retrieval never uses priorities)
        self.priority_dtype = torch.float32 if self.cfg.get('priority_fp32', False) else self.dtype

        self.model = CLIPModelForBySampleTTA(model, datasets.classes, self.cfg, args)

        self.feat_dim = self.model.feat_dim
        self.grad_dim = self.model.grad_dim

        self.loss_fn = lambda logits: softmax_entropy(logits, reduction='sum')
        # use sum since we compute by-sample gradient

        self.cache = [PriorityCache(max_capacity=self.cfg['max_capacity'],
                                    key_dim=self.feat_dim, value_dim=self.grad_dim,
                                    device=self.device, dtype=self.dtype, priority_dtype=self.priority_dtype)
                      for c in range(self.num_classes)]

        self.counter = 0

    def forward(self, x):
        B = x.shape[0]

        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        init_preds = logits.argmax(-1)

        loss = self.loss_fn(logits)
        loss.backward()
        grads = self.model.get_by_sample_grad()

        with torch.no_grad():
            # Update cache
            priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.priority_dtype)
            entropies = softmax_entropy(logits, reduction='none')  # negative of sample-based entropy
            self.counter = self.counter + B

            for b in range(B):
                c = init_preds[b]
                self.cache[c].add(keys=feats[b].unsqueeze(0),
                                  values=grads[b].unsqueeze(0),
                                  entropies=entropies[b].unsqueeze(0),
                                  priorities=priorities[b].unsqueeze(0))

            # Retrieve from cache
            retrieved_grads_sum = torch.zeros_like(grads)
            num_class_sum = 0

            for c in range(self.num_classes):
                if self.cache[c].size == 0:
                    continue

                retrieved_grads, priorities, entropies, sorted_dist = \
                    self.cache[c].query(feats, topk=self.cfg['topk'])  # num_queries * topk * value_dim

                # weighted aggregation
                weights = torch.exp(-entropies)
                weights = weights * torch.exp(-self.beta * sorted_dist)  # larger distance, smaller weight

                retrieved_grads = (retrieved_grads * weights.unsqueeze(dim=2)).sum(dim=1)  # num_queries * value_dim
                retrieved_grads_sum += retrieved_grads
                num_class_sum += 1

            retrieved_grads = retrieved_grads_sum / num_class_sum

            # Set gradient
            self.model.set_by_sample_grad(retrieved_grads)

        # Update Model
        self.model.step_and_zero_grad()

        # Inference

        with torch.no_grad():
            logits = self.model(x)

        # reset model
        self.model.reset_parameters()

        return logits

    def reset(self):
        self.model.reset_parameters()
        self.counter = 0
        for cache in self.cache:
            cache.reset()