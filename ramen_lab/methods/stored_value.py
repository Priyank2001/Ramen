"""
Ramen with a configurable stored VALUE, filing and self-exclusion (docs/plans/oracle_value.md).

Ramen (src/methods/Ramen.py:111-170) writes, per sample j, the value v_j = grad_theta H(p_j) (the entropy gradient,
which pushes toward j's zero-shot class) into the queue of j's zero-shot class, then aggregates the top-k entries of
every queue. Here, one term at a time can change:

    stored_value  entropy (Ramen) | ce_pseudo (CE toward the zero-shot argmax) | ce_true (ORACLE: CE toward the label)
    filing        zero_shot (Ramen) | true_label (ORACLE) | random              (policies from cache_pool.FILINGS)
    exclude_self  False (Ramen) | True: a query never retrieves its own entry (Ramen writes before it retrieves, so
                  otherwise it does, at distance 0); the (k+1)-th nearest entry of that queue takes its place

The defaults reproduce upstream Ramen bit for bit (tests/test_lab_stored_value.py). Steps are named methods (Template
Method): compute_values -> write -> retrieve -> adapt_and_predict.

`RamenStoredValue` is label-free and refuses label-using settings. `RamenStoredValueOracle` (ORACLE) accepts them and
reads the labels only to compute the stored value / the queue; the query's own label never reaches its own step when
exclude_self is on.

Logged per sample (to `log_path`, stored_value.npz, stream order = records.npz order): zs_pred; self_share (weight
share of the query's own entry in Ramen's k-per-queue set); value_norm / value_zero_frac of the stored fp16 value;
push_top_cls / push_top_share: the 5 classes whose entries carry most of the active set's push mass w_ij * ||v_j||
(push class = the class an entry's value pushes toward), with their shares.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ramen_lab import upstream
from ramen_lab.methods import METHODS
from ramen_lab.methods.cache_pool import FILINGS, GradientPool, PooledPriorityCache
from ramen_lab.registry import Registry

N_TOP_PUSH = 5  # push classes logged per query

VALUES: Registry[type] = Registry("stored value")


# ---------------------------------------------------------------------------- stored values (step 2)
class StoredValue(ABC):
    """The per-sample loss whose gradient is written to memory, and the class that gradient pushes toward."""

    uses_labels: bool = False  # True only for ORACLE values

    def __init__(self, loss_scale: float = 1.0):
        self.loss_scale = loss_scale  # a positive constant: changes no sign(G_i), only the fp16 range of the values

    @abstractmethod
    def batch_loss(self, logits: Tensor, zs_pred: Tensor, labels: Optional[Tensor]) -> Tensor:
        """Scalar: the per-sample losses summed (sum, so each sample's by-sample gradient is its own)."""

    @abstractmethod
    def push_class(self, zs_pred: Tensor, labels: Optional[Tensor]) -> Tensor:
        """(B,) long: the class each sample's value pushes toward (for logging)."""

    def scaled(self, loss: Tensor) -> Tensor:
        return loss if self.loss_scale == 1.0 else loss * self.loss_scale


@VALUES.register("entropy")
class EntropyValue(StoredValue):
    """Ramen: v_j = grad H(p_j), computed exactly as upstream (fp16 logits)."""

    def batch_loss(self, logits, zs_pred, labels):
        return self.scaled(upstream.softmax_entropy(logits, reduction="sum"))

    def push_class(self, zs_pred, labels):
        return zs_pred


class CrossEntropyValue(StoredValue):
    """v_j = grad CE(p_j, target_j), CE computed in fp32 from the fp16 logits."""

    @abstractmethod
    def target(self, zs_pred: Tensor, labels: Optional[Tensor]) -> Tensor:
        """(B,) long target class."""

    def batch_loss(self, logits, zs_pred, labels):
        return self.scaled(F.cross_entropy(logits.float(), self.target(zs_pred, labels), reduction="sum"))

    def push_class(self, zs_pred, labels):
        return self.target(zs_pred, labels)


@VALUES.register("ce_pseudo")
class PseudoLabelCE(CrossEntropyValue):
    """Control: CE toward the zero-shot argmax (same loss form as the ORACLE, no label information)."""

    def target(self, zs_pred, labels):
        return zs_pred


@VALUES.register("ce_true")
class TrueLabelCE(CrossEntropyValue):
    """ORACLE: CE toward the true label."""

    uses_labels = True

    def target(self, zs_pred, labels):
        if labels is None:
            raise ValueError("ce_true is an ORACLE value: it needs the labels")
        return labels


# ---------------------------------------------------------------------------- self-exclusion (step 4)
def drop_self(found_ids: Tensor, query_ids: Tensor, k: int) -> Tensor:
    """found_ids (B, n): stream ids of a queue's n nearest entries, nearest first, n = min(k + 1, queue size);
    query_ids (B,). Returns keep (B, n) bool: the (up to) k nearest entries that are not the query itself."""
    is_self = found_ids == query_ids[:, None]
    keep = ~is_self
    if found_ids.shape[1] > k:  # k + 1 retrieved: if the query is not among them, the (k+1)-th is one too many
        keep[:, -1] &= is_self.any(dim=1)
    return keep


# ---------------------------------------------------------------------------- the method
@METHODS.register("RamenStoredValue")
class RamenStoredValue(torch.nn.Module):
    """Ramen with a configurable stored value, filing and self-exclusion (label-free settings only)."""

    uses_labels = False

    def __init__(self, model, datasets, args):
        super().__init__()
        self.cfg = args.config
        self.beta = self.cfg.get("beta", 0.0)
        self.topk = self.cfg["topk"]
        self.num_classes = datasets.num_classes
        self.device = next(model.parameters()).device
        self.model = upstream.by_sample_clip_model(model, datasets.classes, self.cfg, args)
        self.value: StoredValue = VALUES.get(self.cfg.get("stored_value", "entropy"))(
            float(self.cfg.get("value_loss_scale", 1.0)))
        self.filing = FILINGS.get(self.cfg.get("filing", "zero_shot"))(self.num_classes, int(self.cfg.get("filing_seed", 0)))
        self.exclude_self = bool(self.cfg.get("exclude_self", False))
        if (self.value.uses_labels or self.filing.uses_labels) and not self.uses_labels:
            raise ValueError("this setting reads labels: use RamenStoredValueOracle (an ORACLE)")
        capacity = self.cfg["max_capacity"]
        # one memory: every full queue holds `capacity` rows, plus the batch being written before eviction frees rows
        self.pool = GradientPool(self.num_classes * capacity + args.max_batch_size, self.model.grad_dim, self.device)
        self.queues = [PooledPriorityCache(self.pool, capacity, self.model.feat_dim, self.device)
                       for _ in range(self.num_classes)]
        self.push_of_row = torch.full((self.pool.capacity,), -1, dtype=torch.long, device=self.device)
        self.norm_of_row = torch.zeros(self.pool.capacity, dtype=torch.float, device=self.device)
        self.log_path = self.cfg.get("log_path")
        self.chunks: Dict[str, List[np.ndarray]] = {}
        self.counter = 0

    # ------------------------------------------------------------------ Ramen's steps
    def forward(self, x: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        needs_labels = self.value.uses_labels or self.filing.uses_labels
        if needs_labels and labels is None:
            raise ValueError("this ORACLE setting needs the labels: forward(x, labels=...)")
        labels = labels.to(self.device) if (labels is not None and needs_labels) else None
        B = x.shape[0]
        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        zs_pred = logits.argmax(-1)
        values = self.compute_values(logits, zs_pred, labels)
        with torch.no_grad():
            stream_id = torch.arange(self.counter, self.counter + B, device=self.device)
            value_stats = self.write(feats, values, logits, zs_pred, labels, stream_id)
            aggregated, stats = self.retrieve(feats, values, stream_id)
            self.model.set_by_sample_grad(aggregated)
        out = self.adapt_and_predict(x)
        self.record(zs_pred, {**stats, **value_stats})
        return out

    def compute_values(self, logits: Tensor, zs_pred: Tensor, labels: Optional[Tensor]) -> Tensor:
        """Step 2: (B, D) by-sample gradients of the stored-value loss (the .grad buffers stay armed for step 5)."""
        self.value.batch_loss(logits, zs_pred, labels).backward()
        return self.model.get_by_sample_grad()

    @torch.no_grad()
    def write(self, feats: Tensor, values: Tensor, logits: Tensor, zs_pred: Tensor, labels: Optional[Tensor],
              stream_id: Tensor) -> Dict[str, Tensor]:
        """Step 3: every sample into its queue, as upstream (fp16 priorities, FIFO eviction). Returns the stored
        values' norm and fraction of zero coordinates (fp16 underflow), per sample, for the log."""
        stored = values.half()
        if not torch.isfinite(stored).all():
            raise FloatingPointError("a stored value is not finite in fp16: lower value_loss_scale")
        B = feats.shape[0]
        priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=torch.half)
        entropies = upstream.softmax_entropy(logits, reduction="none")
        self.counter += B
        meta_labels = labels if labels is not None else torch.full_like(zs_pred, -1)
        rows = self.pool.put(values, stream_id, zs_pred, meta_labels)
        self.push_of_row[rows] = self.value.push_class(zs_pred, labels)
        self.norm_of_row[rows] = stored.float().norm(dim=1)
        queue_of = self.filing.queues(zs_pred, labels if self.filing.uses_labels else None).tolist()
        row_ids = rows.tolist()
        for b in range(B):
            self.queues[queue_of[b]].add(feats[b], row_ids[b], entropies[b], priorities[b])
        self.pool.finish_batch()
        return {"value_norm": self.norm_of_row[rows], "value_zero_frac": (stored == 0).float().mean(dim=1)}

    @torch.no_grad()
    def retrieve(self, feats: Tensor, values: Tensor, stream_id: Tensor):
        """Step 4: G_i = (1/|C+|) sum_c sum_{j in topk_c(i)} exp(-H_j) exp(-beta d_ij) v_j, written as upstream writes
        it (Ramen.py:136-154) so the default is bit-identical; with exclude_self, j = i is skipped."""
        total = torch.zeros_like(values)
        n_queues = 0
        B = feats.shape[0]
        self_w = torch.zeros(B, device=self.device)
        ramen_w = torch.zeros(B, device=self.device)
        push_mass = torch.zeros(B, self.num_classes, device=self.device)
        extra = 1 if self.exclude_self else 0
        for cache in self.queues:
            if cache.size == 0:
                continue
            found, entropies, sorted_dist = cache.query(feats, topk=self.topk + extra)
            w = torch.exp(-entropies)
            w = w * torch.exp(-self.beta * sorted_dist)
            ids = self.pool.stream_id[found]
            n_ramen = min(self.topk, cache.size)  # Ramen's own top-k: the self share is measured on it
            self_w += (w[:, :n_ramen].float() * (ids[:, :n_ramen] == stream_id[:, None])).sum(1)
            ramen_w += w[:, :n_ramen].float().sum(1)
            if self.exclude_self:
                w = w * drop_self(ids, stream_id, self.topk)
            total += (self.pool.values[found] * w.unsqueeze(dim=2)).sum(dim=1)
            n_queues += 1
            push_mass.scatter_add_(1, self.push_of_row[found], w.float() * self.norm_of_row[found])
        stats = {"self_share": self_w / ramen_w.clamp_min(1e-30), "push_mass": push_mass,
                 "weight_mass": ramen_w}  # total retrieval weight over Ramen's top-k sets (not logged here)
        return total / n_queues, stats

    def adapt_and_predict(self, x: Tensor) -> Tensor:
        """Step 5: one SignSGD step per sample, predict, back to theta_0."""
        self.model.step_and_zero_grad()
        with torch.no_grad():
            out = self.model(x)
        self.model.reset_parameters()
        return out

    # ------------------------------------------------------------------ logging
    @torch.no_grad()
    def record(self, zs_pred: Tensor, stats: Dict[str, Tensor]) -> None:
        mass = stats["push_mass"]
        k = min(N_TOP_PUSH, mass.shape[1])
        top_mass, top_cls = mass.topk(k, dim=1)
        share = top_mass / mass.sum(1, keepdim=True).clamp_min(1e-30)
        batch = {"zs_pred": zs_pred.int(), "self_share": stats["self_share"].float(),
                 "push_top_cls": top_cls.int(), "push_top_share": share.half(),
                 "value_norm": stats["value_norm"].float(), "value_zero_frac": stats["value_zero_frac"].float()}
        for name, value in batch.items():
            self.chunks.setdefault(name, []).append(value.cpu().numpy())

    def arrays(self) -> Dict[str, np.ndarray]:
        return {name: np.concatenate(chunks) for name, chunks in self.chunks.items()}

    def flush(self) -> None:
        """Write stored_value.npz to `log_path` (if set and anything ran)."""
        if self.log_path and self.chunks:
            path = Path(self.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **self.arrays())

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.flush()
        self.model.reset_parameters()
        self.counter = 0
        self.pool.reset()
        for cache in self.queues:
            cache.reset()
        self.filing.reset()
        self.push_of_row.fill_(-1)
        self.norm_of_row.zero_()
        self.chunks = {}


@METHODS.register("RamenStoredValueOracle")
class RamenStoredValueOracle(RamenStoredValue):
    """ORACLE: RamenStoredValue that may read true labels (stored_value ce_true and/or filing true_label)."""

    uses_labels = True
