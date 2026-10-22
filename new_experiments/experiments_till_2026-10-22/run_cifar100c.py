# CIFAR-100-C NoAdapt + Ramen (mixed stream, seed 0); per-sample results go to ./results next to this script.
#
# Memory: Ramen's per-class gradient cache is 100 classes x 750 x 39936 fp16 = 5.9 GB, which leaves too little of a
# 12 GB card for the ViT-B/16 batch-100 forward/backward. So the cache *values* live in pinned host RAM and only
# the rows a query retrieves are copied to the GPU. Same fp16 values, same GPU arithmetic in the same order -> same
# results as upstream Ramen; it is only slower.
import os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ramen_analysis import make_experiment
ramen_module = sys.modules['methods.Ramen']  # `import methods.Ramen` gives the class: the package re-exports it


class HostValueCache(ramen_module.PriorityCache):
    def __init__(self, max_capacity, key_dim, value_dim, device, dtype):
        super().__init__(max_capacity, 1, 1, device, dtype)  # tiny placeholders, replaced below
        self.key_dim, self.value_dim = key_dim, value_dim
        self.keys = torch.empty((max_capacity, key_dim), device=device, dtype=dtype)
        self.values = torch.empty((max_capacity, value_dim), dtype=dtype, pin_memory=True)

    def query(self, queries, topk):
        topk = min(topk, self.size)
        queries = queries.detach().to(device=self.device, dtype=self.dtype)
        dist = torch.cdist(queries, self.keys[:self.size])
        sorted_dist, indices = torch.topk(dist, k=topk, dim=1, largest=False, sorted=True)
        values = self.values[indices.cpu()].to(self.device)  # num_queries * topk * value_dim
        return values, self.priorities[indices], self.entropies[indices], sorted_dist


ramen_module.PriorityCache = HostValueCache  # Ramen.__init__ looks the class up in its module at build time

t = time.time()
exp = make_experiment('CIFAR100C', tta_mode='mixed', seed=0, num_workers=8,
                      results_dir=os.path.join(HERE, 'results')).load()
print(exp.evaluate().round(2).to_string()); print('elapsed %.0fs' % (time.time() - t), flush=True)
