"""
Ramen's per-class PriorityCache with its gradient values stored through a codec (grad_codecs.py).

PriorityCache.add / .query (src/methods/Ramen.py) are inherited unchanged: they only do `self.values[slot] = g` and
`self.values[indices]`, so `values` here is a CodecValues adapter that encodes on write and decodes on read.
Only __init__ is replaced, so the full fp16 value tensor is never allocated.

    from grad_codecs import Int8Codec
    cache =CompressedPriorityCache(75, 512, 39936, 'cuda', torch.half, codec=Int8Codec())
    cache.add(keys, grads, entropies, priorities)
    values, priorities, entropies, dist = cache.query(feats, topk=1)   # values decoded to fp16
    cache.nbytes()                                                     # exact storage of every tensor
"""
import torch

from methods.Ramen import PriorityCache


class CodecValues:
    """Indexable like a (capacity, D) tensor; writes go through codec.write, reads through codec.read."""

    def __init__(self, codec, capacity, D, device):
        self.codec, self.D = codec, D
        self.store = codec.allocate(capacity, D, device)

    def __setitem__(self, slot, g):
        self.codec.write(self.store, slot, g)

    def __getitem__(self, idx):
        return self.codec.read(self.store, idx)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.store.values())


class CompressedPriorityCache(PriorityCache):

    def __init__(self, max_capacity, key_dim, value_dim, device, dtype, priority_dtype=None, codec=None):
        # Same fields as PriorityCache.__init__, except values (codec-backed instead of a fp16 tensor)
        self.max_capacity = max_capacity
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.device = device
        self.dtype = dtype
        self.codec = codec
        self.keys = torch.empty((max_capacity, key_dim), device=device, dtype=dtype)
        self.values = CodecValues(codec, max_capacity, value_dim, device)
        self.priorities = torch.full((max_capacity,), float('-inf'), device=device, dtype=priority_dtype or dtype)
        self.entropies = torch.full((max_capacity,), float('inf'), device=device, dtype=dtype)
        self.size = 0

    def nbytes(self):
        tensors = (self.keys, self.priorities, self.entropies)
        return self.values.nbytes() + sum(t.numel() * t.element_size() for t in tensors)


def bytes_per_entry(codec, key_dim, value_dim, key_bytes=2, priority_bytes=2, entropy_bytes=2):
    """Storage of one cache entry: key + priority + entropy (fp16, as Ramen) + the codec's value."""
    return key_dim * key_bytes + priority_bytes + entropy_bytes + codec.bytes_per_value(value_dim)
