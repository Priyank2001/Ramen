"""
ORACLE experiment: which class queue does a sample go into? (docs/plans/oracle_fill.md)

Ramen keeps one priority queue per class and files each incoming sample under its zero-shot PREDICTION. `RamenFileByLabel`
changes that one choice and nothing else (features, per-sample gradients, priorities, eviction, retrieval, SignSGD step
and the final prediction are the same code paths as upstream):

    filing = "prediction"  Ramen's rule. Labels are never read; predictions equal upstream Ramen's to the bit.
    filing = "label"       ORACLE. Each sample is filed under its TRUE label, whatever its prediction was.

Only classes declare `uses_labels = True` (the runner passes labels), but the "prediction" setting never reads them.

Besides the predictions, each run writes two files next to its records (for the analysis):
    cache_snapshot.npz  for every queue: the stream position, priority and gradient target of each entry held at the end
    retrieval.npz       for every query: retrieved weight mass per queue and per gradient target, the zero-shot
                        prediction, and the top retrieved entries (stream position and weight)
Stream positions are arrival order (the same order as records.npz); the entry's gradient target is its zero-shot argmax,
because the cached value is the entropy gradient toward that class (docs/research_state.md, see Gotchas in lab.md).
"""
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ramen_lab import upstream
from ramen_lab.methods import METHODS

FILINGS = ("prediction", "label")
LOG_TOP_K = 5  # retrieved entries kept per query for the examples figure

_UPSTREAM_RAMEN = upstream.method_class("Ramen")
_UPSTREAM_MODULE = inspect.getmodule(_UPSTREAM_RAMEN)
PriorityCache = _UPSTREAM_MODULE.PriorityCache
softmax_entropy = _UPSTREAM_MODULE.softmax_entropy


class TrackedPriorityCache(PriorityCache):
    """Upstream PriorityCache, plus per-slot stream position and gradient target. Keys, values, priorities, entropies and
    the eviction rule are unchanged; `query_tracked` returns the same numbers as `query` and also the slot indices."""

    def __init__(self, max_capacity: int, key_dim: int, value_dim: int, device, dtype):
        super().__init__(max_capacity, key_dim, value_dim, device, dtype)
        self.positions = torch.full((max_capacity,), -1, dtype=torch.long, device=device)
        self.targets = torch.full((max_capacity,), -1, dtype=torch.long, device=device)

    def add(self, keys: Tensor, values: Tensor, entropies: Tensor, priorities: Tensor,
            positions: Tensor, targets: Tensor) -> None:
        """Same loop as upstream `add`; each row is one sample. `positions` and `targets` are (B,) long tensors."""
        keys = keys.detach().to(device=self.device, dtype=self.dtype)
        values = values.detach().to(device=self.device, dtype=self.dtype)
        for i in range(keys.shape[0]):
            if self.size < self.max_capacity:
                slot = self.size
                self.size += 1
            else:
                slot = int(torch.argmin(self.priorities))
                if not priorities[i] > self.priorities[slot]:
                    continue
            self.keys[slot] = keys[i]
            self.values[slot] = values[i]
            self.priorities[slot] = priorities[i]
            self.entropies[slot] = entropies[i]
            self.positions[slot] = positions[i]
            self.targets[slot] = targets[i]

    def query_tracked(self, queries: Tensor, topk: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """(values, priorities, entropies, distances, slot indices) of the topk nearest entries, as upstream `query`."""
        topk = min(topk, self.size)
        queries = queries.detach().to(device=self.device, dtype=self.dtype)
        dist = torch.cdist(queries, self.keys[:self.size])
        sorted_dist, indices = torch.topk(dist, k=topk, dim=1, largest=False, sorted=True)
        return (self.values[indices], self.priorities[indices], self.entropies[indices], sorted_dist, indices)

    def snapshot(self) -> Dict[str, np.ndarray]:
        n = self.size
        return {"position": self.positions[:n].cpu().numpy(), "priority": self.priorities[:n].float().cpu().numpy(),
                "target": self.targets[:n].cpu().numpy()}


@METHODS.register("RamenFileByLabel")
class RamenFileByLabel(_UPSTREAM_RAMEN):
    """Ramen with the queue choice switched by config key `filing` (see module docstring). ORACLE when filing="label"."""

    uses_labels = True

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.filing = self.cfg.get("filing", "prediction")
        if self.filing not in FILINGS:
            raise ValueError(f"filing must be one of {FILINGS}: {self.filing!r}")
        self.probe_dir: Optional[str] = self.cfg.get("probe_dir")
        self.cache = [TrackedPriorityCache(max_capacity=self.cfg["max_capacity"], key_dim=self.feat_dim,
                                           value_dim=self.grad_dim, device=self.device, dtype=self.dtype)
                      for _ in range(self.num_classes)]
        self._log: Dict[str, List[np.ndarray]] = {}

    # ------------------------------------------------------------------ the one choice this class changes
    def queue_of(self, init_preds: Tensor, labels: Optional[Tensor]) -> Tensor:
        """Index of the queue each sample is filed under."""
        if self.filing == "prediction":
            return init_preds  # labels are not touched
        if labels is None:
            raise ValueError("filing='label' is an ORACLE: it needs the true labels (the runner passes them)")
        return labels.to(self.device)

    # ------------------------------------------------------------------ Ramen's forward, as named steps
    def forward(self, x: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        B = x.shape[0]
        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        init_preds = logits.argmax(-1)

        loss = self.loss_fn(logits)
        loss.backward()
        grads = self.model.get_by_sample_grad()

        with torch.no_grad():
            queues = self.queue_of(init_preds, labels)
            positions = torch.arange(self.counter, self.counter + B, device=self.device)
            priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.dtype)
            entropies = softmax_entropy(logits, reduction='none')
            self.counter = self.counter + B
            self.write_memory(feats, grads, entropies, priorities, positions, queues, init_preds)
            retrieved = self.retrieve(feats, grads)
            self.log_retrieval(positions, init_preds)
            self.model.set_by_sample_grad(retrieved)

        self.model.step_and_zero_grad()
        with torch.no_grad():
            logits = self.model(x)
        self.model.reset_parameters()
        return logits

    def write_memory(self, feats: Tensor, grads: Tensor, entropies: Tensor, priorities: Tensor, positions: Tensor,
                     queues: Tensor, targets: Tensor) -> None:
        """Each sample into the queue of `queues[b]`. `targets[b]` is its zero-shot argmax (gradient target)."""
        for b in range(feats.shape[0]):
            self.cache[int(queues[b])].add(keys=feats[b:b + 1], values=grads[b:b + 1], entropies=entropies[b:b + 1],
                                           priorities=priorities[b:b + 1], positions=positions[b:b + 1],
                                           targets=targets[b:b + 1])

    def retrieve(self, feats: Tensor, grads: Tensor) -> Tensor:
        """Upstream's aggregation: per non-empty queue the entropy-weighted top-k gradients, averaged over queues.
        Also keeps, for log_retrieval, the weights and positions of every retrieved entry of this call."""
        total = torch.zeros_like(grads)
        num_queues = 0
        self._retrieved: List[Tuple[int, Tensor, Tensor]] = []  # (queue, slot indices, weights)
        for c in range(self.num_classes):
            if self.cache[c].size == 0:
                continue
            values, _, entropies, sorted_dist, indices = self.cache[c].query_tracked(feats, topk=self.cfg['topk'])
            weights = torch.exp(-entropies) * torch.exp(-self.beta * sorted_dist)
            total += (values * weights.unsqueeze(dim=2)).sum(dim=1)
            num_queues += 1
            self._retrieved.append((c, indices, weights))
        return total / num_queues

    def log_retrieval(self, positions: Tensor, init_preds: Tensor) -> None:
        """Per query: retrieved weight mass per queue and per gradient target, and the LOG_TOP_K heaviest entries."""
        if not self.probe_dir:
            return
        B, C = positions.shape[0], self.num_classes
        mass_queue = torch.zeros(B, C, device=self.device)
        mass_target = torch.zeros(B, C, device=self.device)
        pos, w = [], []
        for queue, indices, weights in self._retrieved:
            cache, wf = self.cache[queue], weights.float()
            mass_queue[:, queue] = wf.sum(1)
            mass_target.scatter_add_(1, cache.targets[indices], wf)
            pos.append(cache.positions[indices])
            w.append(wf)
        pos, w = torch.cat(pos, dim=1), torch.cat(w, dim=1)
        top_w, top_i = torch.topk(w, k=min(LOG_TOP_K, w.shape[1]), dim=1)
        self._append("mass_queue", mass_queue)
        self._append("mass_target", mass_target)
        self._append("top_pos", pos.gather(1, top_i))
        self._append("top_w", top_w)
        self._append("zs_pred", init_preds)
        self._append("query_pos", positions)

    def _append(self, name: str, value: Tensor) -> None:
        self._log.setdefault(name, []).append(value.detach().cpu().numpy())

    # ------------------------------------------------------------------ output
    def flush(self) -> None:
        """Write cache_snapshot.npz and retrieval.npz into `probe_dir` (rewritten on each call; the last call wins)."""
        if not self.probe_dir or not self._log:
            return
        out = Path(self.probe_dir)
        out.mkdir(parents=True, exist_ok=True)
        snaps = [c.snapshot() for c in self.cache]
        np.savez(out / "cache_snapshot.npz", queue=np.concatenate([np.full(len(s["position"]), q)
                                                                  for q, s in enumerate(snaps)]),
                 position=np.concatenate([s["position"] for s in snaps]),
                 priority=np.concatenate([s["priority"] for s in snaps]),
                 target=np.concatenate([s["target"] for s in snaps]),
                 sizes=np.array([c.size for c in self.cache]))
        arrays = {name: np.concatenate(chunks) for name, chunks in self._log.items()}
        np.savez(out / "retrieval.npz", filing=np.array(self.filing), **arrays)

    def reset(self) -> None:
        self.flush()
        self._log = {}
        super().reset()
