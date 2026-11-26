"""
ORACLE experiment method: Ramen with several memories over one stream, each filed by a different policy.

`RamenMultiCache` runs Ramen's forward (src/methods/Ramen.py:111-170) once per batch up to the gradients, then:
  write   every sample into every copy's memory, copy c filing it under `filings[c].queues(...)` (same key, gradient,
          entropy and priority in every copy; gradients stored once in a shared GradientPool);
  adapt   for each copy in turn: retrieve that copy's active set, aggregate, one SignSGD step, predict, reset weights.
Copy 0 must be `zero_shot` filing: it is Ramen, bit for bit (tests/test_lab_multi_cache.py), and its logits are what
`forward` returns, so the run's records.npz / metrics.json are Ramen's. The other copies' predictions and active-set
statistics go to `log_path` (multicache.npz, see multi_cache_log.py).

ORACLE: declares `uses_labels`. Labels reach (a) the `true_label` filing policy and (b) the log's statistics; they never
reach copy 0's update, so copy 0's predictions do not depend on them (tested).

Config keys (on top of Ramen's yaml): filings (default zero_shot, true_label, random), filing_seed, pool_capacity
(default: copies x classes x max_capacity, which can never overflow), log_path, n_examples, example_copy, snapshot_at.
"""
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ramen_lab import upstream
from ramen_lab.methods import METHODS
from ramen_lab.methods.cache_pool import FILINGS, GradientPool, PooledPriorityCache
from ramen_lab.methods.multi_cache_log import ActiveSet, MultiCacheLog

DEFAULT_FILINGS = ("zero_shot", "true_label", "random")


@METHODS.register("RamenMultiCache")
class RamenMultiCache(torch.nn.Module):
    """ORACLE: Ramen + extra memories filed by other policies (true label, random); returns the Ramen copy's logits."""

    uses_labels = True

    def __init__(self, model, datasets, args):
        super().__init__()
        self.cfg = args.config
        self.beta = self.cfg.get("beta", 0.0)
        self.topk = self.cfg["topk"]
        self.num_classes = datasets.num_classes
        self.device = next(model.parameters()).device
        self.model = upstream.by_sample_clip_model(model, datasets.classes, self.cfg, args)
        self.filing_names = list(self.cfg.get("filings", DEFAULT_FILINGS))
        if self.filing_names[0] != "zero_shot":
            raise ValueError(f"copy 0 must be Ramen's zero_shot filing (it is what forward returns): {self.filing_names}")
        seed = int(self.cfg.get("filing_seed", 0))
        self.filings = [FILINGS.get(n)(self.num_classes, seed + i) for i, n in enumerate(self.filing_names)]
        capacity = self.cfg["max_capacity"]
        pool_rows = int(self.cfg.get("pool_capacity") or len(self.filings) * self.num_classes * capacity)
        self.pool = GradientPool(pool_rows, self.model.grad_dim, self.device)
        self.caches = [[PooledPriorityCache(self.pool, capacity, self.model.feat_dim, self.device)
                        for _ in range(self.num_classes)] for _ in self.filings]
        self.log = MultiCacheLog(self.filing_names, self.num_classes, example_copy=int(self.cfg.get("example_copy", 1)),
                                 n_examples=int(self.cfg.get("n_examples", 48)),
                                 snapshot_at=self.cfg.get("snapshot_at", ()), seed=seed)
        self.log_path = self.cfg.get("log_path")
        self.counter = 0

    # ------------------------------------------------------------------ Ramen's steps, several memories
    def forward(self, x: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        if labels is None and any(f.uses_labels for f in self.filings):
            raise ValueError(f"filings {self.filing_names} include an ORACLE: pass labels")
        B = x.shape[0]
        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        zs_pred = logits.argmax(-1)
        upstream.softmax_entropy(logits, reduction="sum").backward()
        grads = self.model.get_by_sample_grad()
        labels = labels.to(self.device) if labels is not None else torch.full_like(zs_pred, -1)

        with torch.no_grad():
            stream_id = torch.arange(self.counter, self.counter + B, device=self.device)
            self.write(feats, grads, logits, zs_pred, labels, stream_id)

        outputs, actives = [], []
        for copy in range(len(self.filings)):
            out, active = self.adapt_and_predict(copy, x, feats, grads)
            outputs.append(out)
            actives.append(active)
        if (labels >= 0).all():
            self.log.on_batch(stream_id, labels, zs_pred, outputs, actives, self.pool, self.caches)
        return outputs[0]

    @torch.no_grad()
    def write(self, feats: Tensor, grads: Tensor, logits: Tensor, zs_pred: Tensor, labels: Tensor,
              stream_id: Tensor) -> None:
        """Step 3 of Ramen, once per copy: same entry, queue chosen by that copy's filing policy."""
        B = feats.shape[0]
        priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=torch.half)
        entropies = upstream.softmax_entropy(logits, reduction="none")
        self.counter += B
        rows = self.pool.put(grads, stream_id, zs_pred, labels)
        row_ids = rows.tolist()
        for filing, queues in zip(self.filings, self.caches):
            queue_of = filing.queues(zs_pred, labels if filing.uses_labels else None).tolist()
            for b in range(B):
                queues[queue_of[b]].add(feats[b], row_ids[b], entropies[b], priorities[b])
        self.pool.finish_batch()

    def adapt_and_predict(self, copy: int, x: Tensor, feats: Tensor, grads: Tensor) -> Tuple[Tensor, ActiveSet]:
        """Steps 4-5 of Ramen for one copy; the weights are back at theta_0 afterwards."""
        if copy > 0:  # copy 0 uses the buffers the backward just filled, exactly like upstream
            upstream.rearm_by_sample_grads(self.model, x.shape[0])
        with torch.no_grad():
            aggregated, active = self.retrieve(self.caches[copy], feats, grads)
            self.model.set_by_sample_grad(aggregated)
        self.model.step_and_zero_grad()
        with torch.no_grad():
            out = self.model(x)
        self.model.reset_parameters()
        return out, active

    @torch.no_grad()
    def retrieve(self, queues: Sequence[PooledPriorityCache], feats: Tensor, grads: Tensor) -> Tuple[Tensor, ActiveSet]:
        """G_i = (1/|C+|) sum_c sum_{j in topk_c(z_i)} exp(-H_j) exp(-beta d_ij) g_j, written as upstream writes it
        (Ramen.py:136-154) so copy 0 is bit-identical. Also returns the active set it used."""
        total = torch.zeros_like(grads)
        n_queues = 0
        rows, weights, queue_ids = [], [], []
        for c, cache in enumerate(queues):
            if cache.size == 0:
                continue
            found, entropies, sorted_dist = cache.query(feats, topk=self.topk)
            w = torch.exp(-entropies)
            w = w * torch.exp(-self.beta * sorted_dist)
            total += (self.pool.values[found] * w.unsqueeze(dim=2)).sum(dim=1)
            n_queues += 1
            rows.append(found)
            weights.append(w.float())
            queue_ids.append(torch.full_like(found, c))
        active = ActiveSet(torch.cat(rows, 1), torch.cat(weights, 1), torch.cat(queue_ids, 1))
        return total / n_queues, active

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.flush()
        self.model.reset_parameters()
        self.counter = 0
        self.pool.reset()
        for queues in self.caches:
            for cache in queues:
                cache.reset()
        for filing in self.filings:
            filing.reset()
        self.log.reset()

    def flush(self) -> None:
        """Snapshot the memories as they are now and write multicache.npz to `log_path` (if set and anything ran)."""
        if self.counter == 0 or not self.log.chunks:
            return
        self.log.snapshot(self.caches, self.pool, "final")
        if self.log_path:
            self.log.save(self.log_path)
