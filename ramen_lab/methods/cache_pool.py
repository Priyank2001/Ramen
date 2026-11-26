"""
Several Ramen memories over one stream, with each sample's gradient stored once.

Ramen's memory holds, per class queue, (key z_j, value g_j, entropy H_j, priority t_j). Two memories that file the same
stream differently (e.g. by zero-shot prediction vs. by true label) hold the SAME (z_j, g_j, H_j, t_j) for every sample
they share; only the queue differs. Two full copies of the ~40k-dim fp16 gradients do not fit next to each other on an
11-12 GB GPU, so the gradients live once in a `GradientPool` (reference-counted rows) and every `PooledPriorityCache`
stores pool row ids instead of values.

`PooledPriorityCache` mirrors upstream `PriorityCache` (src/methods/Ramen.py:9-81) slot for slot: same slot chosen on
insert and eviction (argmin of fp16 priorities, strict `>`), same keys, priorities and entropies, same query. The values
it returns are the pool rows the upstream cache would hold, so a Ramen copy built on it is bit-identical to upstream
Ramen (tests/test_lab_multi_cache.py).
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
from torch import Tensor

from ramen_lab.registry import Registry


class PoolFull(RuntimeError):
    pass


class GradientPool:
    """Shared fp16 rows of per-sample gradients, plus what each row's sample was (for analysis only).

    A row is allocated when its sample is written, referenced once by every cache that keeps it, and freed when no
    cache references it any more. `label` is the true class: it is metadata for logging and is never read by the
    adaptation math (only by the ORACLE filing policy and the statistics)."""

    def __init__(self, capacity: int, value_dim: int, device: torch.device, dtype: torch.dtype = torch.half):
        self.capacity = capacity
        self.values = torch.empty((capacity, value_dim), device=device, dtype=dtype)
        self.stream_id = torch.full((capacity,), -1, device=device, dtype=torch.long)
        self.zs_pred = torch.full((capacity,), -1, device=device, dtype=torch.long)
        self.label = torch.full((capacity,), -1, device=device, dtype=torch.long)
        self.refs = np.zeros(capacity, dtype=np.int32)
        self._free: List[int] = list(range(capacity - 1, -1, -1))  # a stack: row 0 is handed out first
        self._pending: Set[int] = set()  # rows of the batch being written: freed only by finish_batch
        self.peak_in_use = 0

    @property
    def in_use(self) -> int:
        return self.capacity - len(self._free)

    def put(self, values: Tensor, stream_id: Tensor, zs_pred: Tensor, label: Tensor) -> Tensor:
        """Store B rows (values (B, D)); returns their row ids (B,) long on the pool's device. Refs start at 0."""
        n = values.shape[0]
        if n > len(self._free):
            raise PoolFull(f"gradient pool full ({self.capacity:,} rows, {self.in_use:,} in use): raise pool_capacity")
        row_ids = [self._free.pop() for _ in range(n)]
        self._pending.update(row_ids)
        rows = torch.tensor(row_ids, device=self.values.device, dtype=torch.long)
        self.values[rows] = values.detach().to(self.values.dtype)
        self.stream_id[rows] = stream_id.to(self.stream_id.device)
        self.zs_pred[rows] = zs_pred.to(self.zs_pred.device)
        self.label[rows] = label.to(self.label.device)
        self.peak_in_use = max(self.peak_in_use, self.in_use)
        return rows

    def retain(self, row: int) -> None:
        self.refs[row] += 1

    def release(self, row: int) -> None:
        self.refs[row] -= 1
        if self.refs[row] < 0:
            raise RuntimeError(f"pool row {row} released more often than retained")
        if self.refs[row] == 0 and row not in self._pending:
            self._free.append(row)

    def finish_batch(self) -> None:
        """Call once every cache has written the batch. Frees the batch's rows that no cache holds now: rows no full
        queue accepted, and rows one copy kept and evicted again within the batch. Until then a row of the batch stays
        allocated even at refcount 0, because a later copy may still file it."""
        for row in self._pending:
            if self.refs[row] == 0:
                self._free.append(row)
        self._pending = set()

    def reset(self) -> None:
        self.refs[:] = 0
        self._free = list(range(self.capacity - 1, -1, -1))
        self._pending = set()
        self.peak_in_use = 0


class PooledPriorityCache:
    """One class queue of Ramen's memory; upstream `PriorityCache` with values replaced by pool row ids."""

    def __init__(self, pool: GradientPool, max_capacity: int, key_dim: int, device: torch.device,
                 dtype: torch.dtype = torch.half):
        self.pool = pool
        self.max_capacity = max_capacity
        self.keys = torch.empty((max_capacity, key_dim), device=device, dtype=dtype)
        self.rows = torch.full((max_capacity,), -1, device=device, dtype=torch.long)
        self.rows_cpu = np.full(max_capacity, -1, dtype=np.int64)  # mirror: eviction bookkeeping without a GPU sync
        self.priorities = torch.full((max_capacity,), float("-inf"), device=device, dtype=dtype)
        self.entropies = torch.full((max_capacity,), float("inf"), device=device, dtype=dtype)
        self.size = 0

    def add(self, key: Tensor, row: int, entropy: Tensor, priority: Tensor) -> None:
        """One sample: key (D,) or (1, D); entropy and priority 0-d or (1,). Same slot logic as upstream."""
        key = key.detach().to(device=self.keys.device, dtype=self.keys.dtype).reshape(-1)
        if self.size < self.max_capacity:
            slot = self.size
            self.size += 1
        else:
            min_idx = torch.argmin(self.priorities)
            if not priority.reshape(()) > self.priorities[min_idx]:
                return
            slot = int(min_idx)
            self.pool.release(int(self.rows_cpu[slot]))
        self.keys[slot] = key
        self.rows[slot] = row
        self.rows_cpu[slot] = row
        self.priorities[slot] = priority.reshape(())
        self.entropies[slot] = entropy.reshape(())
        self.pool.retain(row)

    def query(self, queries: Tensor, topk: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Pool rows (B, k), entropies (B, k) and distances (B, k) of the k nearest entries, nearest first.
        cdist takes the matrix-product path, as upstream's does in ramen_lab (upstream._TorchWithMmCdist)."""
        topk = min(topk, self.size)
        queries = queries.detach().to(device=self.keys.device, dtype=self.keys.dtype)
        dist = torch.cdist(queries, self.keys[:self.size], compute_mode="use_mm_for_euclid_dist")
        sorted_dist, indices = torch.topk(dist, k=topk, dim=1, largest=False, sorted=True)
        return self.rows[indices], self.entropies[indices], sorted_dist

    def stream_ids(self) -> Tensor:
        """(size,) stream position of every entry, in slot order (for snapshots)."""
        return self.pool.stream_id[self.rows[:self.size]]

    def reset(self) -> None:
        self.size = 0  # lazy, like upstream; the pool is reset by its owner
        self.rows_cpu[:] = -1


# ---------------------------------------------------------------------------- filing policies
FILINGS: Registry[type] = Registry("filing policy")


class FilingPolicy(ABC):
    """Which class queue each sample of a batch is written into. Ramen's is ZeroShotFiling."""

    uses_labels: bool = False  # True only for ORACLE policies

    @abstractmethod
    def queues(self, zs_pred: Tensor, labels: Optional[Tensor]) -> Tensor:
        """zs_pred (B,) long zero-shot argmax; labels (B,) long or None. Returns (B,) long queue ids."""

    def reset(self) -> None:
        """Forget any state (e.g. a random generator) so a reset method replays the same stream identically."""


@FILINGS.register("zero_shot")
class ZeroShotFiling(FilingPolicy):
    """Ramen: queue = zero-shot prediction."""

    def __init__(self, num_classes: int, seed: int = 0):
        self.num_classes = num_classes

    def queues(self, zs_pred, labels):
        return zs_pred


@FILINGS.register("true_label")
class TrueLabelFiling(FilingPolicy):
    """ORACLE: queue = true class."""

    uses_labels = True

    def __init__(self, num_classes: int, seed: int = 0):
        self.num_classes = num_classes

    def queues(self, zs_pred, labels):
        if labels is None:
            raise ValueError("true_label filing is an ORACLE: it needs the labels")
        return labels.to(zs_pred.device)


@FILINGS.register("random")
class RandomFiling(FilingPolicy):
    """Control: queue drawn uniformly from all classes, from a seeded generator (same spreading, no label information)."""

    def __init__(self, num_classes: int, seed: int = 0):
        self.num_classes = num_classes
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)

    def queues(self, zs_pred, labels):
        draw = torch.randint(self.num_classes, (len(zs_pred),), generator=self.generator)
        return draw.to(zs_pred.device)

    def reset(self) -> None:
        self.generator = torch.Generator().manual_seed(self.seed)
