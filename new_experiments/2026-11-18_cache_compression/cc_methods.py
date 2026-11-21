"""
RamenCompressed: Ramen whose per-class caches store the gradient values through a codec (grad_codecs.py).

Everything else is Ramen's own code path (src/methods/Ramen.py, inherited, not copied): write before retrieval,
top-k per queue, exp(-H) exp(-beta d) weights, SignSGD step, predict, reset. With `codec: fp16` the run must be
bitwise identical to Ramen (smoke check).

    method = RamenCompressed(model, datasets, args)  # args.config = Ramen's cfg + {'codec': 'int8', 'block': 64}
    logits = method(x)
    method.stats   # codec, capacity, cache_bytes (exact), bytes_per_entry, zero_frac per batch, peak_mem_bytes

Config keys: codec (fp16 | int8 | int4 | sign), block (values per scale, default 64), max_capacity (as Ramen).
"""
from types import SimpleNamespace

import torch

from compressed_cache import CompressedPriorityCache, bytes_per_entry
from grad_codecs import CODECS
from methods.Ramen import Ramen


class RamenCompressed(Ramen):

    def __init__(self, model, datasets, args):
        cfg = args.config
        # Build the parent with capacity-1 caches (replaced below) so the full fp16 caches are never allocated
        tiny = SimpleNamespace(**{**vars(args), 'config': {**cfg, 'max_capacity': 1}})
        super().__init__(model, datasets, tiny)
        self.cfg = cfg
        self.codec = CODECS[cfg.get('codec', 'fp16')](block=int(cfg.get('block', 64)))
        self.cache = [CompressedPriorityCache(max_capacity=cfg['max_capacity'], key_dim=self.feat_dim,
                                              value_dim=self.grad_dim, device=self.device, dtype=self.dtype,
                                              priority_dtype=self.priority_dtype, codec=self.codec)
                      for _ in range(self.num_classes)]
        self.stats = dict(codec=self.codec.name, block=self.codec.block, max_capacity=cfg['max_capacity'],
                          grad_dim=self.grad_dim, feat_dim=self.feat_dim,
                          bytes_per_entry=bytes_per_entry(self.codec, self.feat_dim, self.grad_dim),
                          cache_bytes=sum(c.nbytes() for c in self.cache),
                          zero_frac=[], peak_mem_bytes=0)
        self._log_update_zeros()
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def _log_update_zeros(self):
        """Share of coordinates of the aggregated gradient that are exactly 0 (SignSGD leaves them unchanged)."""
        set_grad = self.model.set_by_sample_grad

        def logged(grad_matrix):
            self.stats['zero_frac'].append(float((grad_matrix == 0).float().mean()))
            return set_grad(grad_matrix)

        self.model.set_by_sample_grad = logged

    def forward(self, x):
        out = super().forward(x)
        if self.device.type == 'cuda':
            self.stats['peak_mem_bytes'] = max(self.stats['peak_mem_bytes'],
                                               torch.cuda.max_memory_allocated(self.device))
        return out
