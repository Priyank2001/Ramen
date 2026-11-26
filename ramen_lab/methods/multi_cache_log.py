"""
What `RamenMultiCache` records, so the notebook can explain a result without a re-run (multicache.npz).

Per sample and per copy (stream order, the same order as the run's records.npz):
    pred, true_prob                     the copy's adapted prediction and its probability on the true class
    share_push_true                     share of the active-set weight w_ij on entries whose zero-shot argmax (the class
                                        their stored entropy gradient pushes toward) is the query's TRUE class
    share_push_zs                       ... whose zero-shot argmax is the query's own zero-shot prediction
    share_push_pred                     ... whose zero-shot argmax is this copy's final prediction
    share_push_top                      the largest share any single push class gets (1 = all entries push one way)
    share_label_true                    ... whose TRUE class is the query's true class (label purity of the active set)
    shared_with_ramen                   share of copy 0's active-set entries that this copy also retrieved
Once per sample: label, zs_pred, stream_id.

Example reservoirs (uniform over the whole stream, seeded): samples where copy 0 is right and `example_copy` wrong
("broken"), and the reverse ("fixed"); for each, every copy's full active set (stream ids, queue, weight).
Snapshots: every copy's queues as stream ids, at the configured stream positions and at the end.
Labels are read here for analysis only; nothing in this module feeds back into any copy's update.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

PER_COPY = ("pred", "true_prob", "share_push_true", "share_push_zs", "share_push_pred", "share_push_top",
            "share_label_true", "shared_with_ramen")
CATEGORIES = ("broken", "fixed")  # copy 0 right / example copy wrong, and the reverse


@dataclass
class ActiveSet:
    """One copy's support set for a batch: pool rows (B, N), weights w_ij (B, N) fp32, queue of each entry (B, N)."""
    rows: Tensor
    weights: Tensor
    queue: Tensor


class MultiCacheLog:
    def __init__(self, copy_names: Sequence[str], num_classes: int, example_copy: int = 1, n_examples: int = 48,
                 snapshot_at: Sequence[int] = (), seed: int = 0):
        self.copy_names = list(copy_names)
        self.num_classes = num_classes
        self.example_copy = example_copy
        self.n_examples = n_examples
        self.snapshot_at = sorted(int(p) for p in snapshot_at)
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.chunks: Dict[str, List[np.ndarray]] = {}
        self.examples: Dict[str, List[dict]] = {c: [] for c in CATEGORIES}
        self.seen = {c: 0 for c in CATEGORIES}
        self.snapshots: Dict[str, np.ndarray] = {}
        self.pool_peak = 0
        self.rng = np.random.default_rng(self.seed)
        self._next_snapshot = 0

    # ------------------------------------------------------------------ per batch
    @torch.no_grad()
    def on_batch(self, stream_id: Tensor, labels: Tensor, zs_pred: Tensor, outputs: Sequence[Tensor],
                 actives: Sequence[ActiveSet], pool, caches) -> None:
        preds = [o.argmax(-1) for o in outputs]
        self._add("stream_id", stream_id)
        self._add("label", labels)
        self._add("zs_pred", zs_pred)
        ramen_ids = pool.stream_id[actives[0].rows]
        for c, (out, pred, act) in enumerate(zip(outputs, preds, actives)):
            prob = out.float().softmax(-1)
            stats = self.active_stats(act, pool, labels, zs_pred, pred, ramen_ids)
            stats.update(pred=pred, true_prob=prob.gather(1, labels[:, None]).squeeze(1).half())
            for name in PER_COPY:
                self._add(f"{name}/{c}", stats[name])
        self._reservoir(stream_id, labels, zs_pred, preds, actives, pool)
        self.pool_peak = max(self.pool_peak, pool.peak_in_use)
        end = int(stream_id[-1]) + 1
        while self._next_snapshot < len(self.snapshot_at) and end >= self.snapshot_at[self._next_snapshot]:
            self.snapshot(caches, pool, f"at_{end}")
            self._next_snapshot += 1

    def active_stats(self, act: ActiveSet, pool, labels: Tensor, zs_pred: Tensor, pred: Tensor,
                     ramen_ids: Tensor) -> Dict[str, Tensor]:
        """Weight shares of one copy's active set (see the module docstring). All (B,) float."""
        w = act.weights
        total = w.sum(1)
        push, true = pool.zs_pred[act.rows], pool.label[act.rows]

        def share(mask: Tensor) -> Tensor:
            return (w * mask).sum(1) / total

        by_class = torch.zeros(len(w), self.num_classes, device=w.device).scatter_add_(1, push, w)
        ids = pool.stream_id[act.rows]
        return {"share_push_true": share(push == labels[:, None]), "share_push_zs": share(push == zs_pred[:, None]),
                "share_push_pred": share(push == pred[:, None]), "share_push_top": by_class.max(1).values / total,
                "share_label_true": share(true == labels[:, None]), "shared_with_ramen": shared_fraction(ramen_ids, ids)}

    def _add(self, name: str, value: Tensor) -> None:
        self.chunks.setdefault(name, []).append(value.detach().cpu().numpy())

    def _reservoir(self, stream_id, labels, zs_pred, preds, actives, pool) -> None:
        """Uniform reservoir sample (Algorithm R) of broken / fixed samples over the whole stream."""
        right0 = (preds[0] == labels).cpu().numpy()
        right1 = (preds[self.example_copy] == labels).cpu().numpy()
        for cat, mask in (("broken", right0 & ~right1), ("fixed", ~right0 & right1)):
            for b in np.flatnonzero(mask):
                self.seen[cat] += 1
                keep = self.examples[cat]
                slot = len(keep) if len(keep) < self.n_examples else int(self.rng.integers(self.seen[cat]))
                if slot >= self.n_examples:
                    continue
                item = self._example(b, stream_id, labels, zs_pred, preds, actives, pool)
                if slot == len(keep):
                    keep.append(item)
                else:
                    keep[slot] = item

    @staticmethod
    def _example(b, stream_id, labels, zs_pred, preds, actives, pool) -> dict:
        return {"stream_id": int(stream_id[b]), "label": int(labels[b]), "zs_pred": int(zs_pred[b]),
                "preds": [int(p[b]) for p in preds],
                "active_ids": [pool.stream_id[a.rows[b]].cpu().numpy() for a in actives],
                "active_queue": [a.queue[b].cpu().numpy() for a in actives],
                "active_weight": [a.weights[b].cpu().numpy() for a in actives]}

    def snapshot(self, caches, pool, name: str) -> None:
        """Every copy's queues as stream ids: (copies, C, K) int32, -1 where a slot is empty."""
        cap = caches[0][0].max_capacity
        snap = np.full((len(caches), self.num_classes, cap), -1, dtype=np.int32)
        for c, queues in enumerate(caches):
            for q, cache in enumerate(queues):
                snap[c, q, :cache.size] = cache.stream_ids().cpu().numpy()
        self.snapshots[name] = snap

    # ------------------------------------------------------------------ output
    def arrays(self) -> Dict[str, np.ndarray]:
        return {name: np.concatenate(chunks) for name, chunks in self.chunks.items()}

    def save(self, path: Path) -> None:
        out = {k.replace("/", "__copy"): v for k, v in self.arrays().items()}
        out.update({f"snapshot__{k}": v for k, v in self.snapshots.items()})
        out["copy_names"] = np.array(self.copy_names)
        out["pool_peak"] = np.array(self.pool_peak)
        for cat in CATEGORIES:
            out.update({f"ex_{cat}__{k}": v for k, v in pack_examples(self.examples[cat], len(self.copy_names)).items()})
            out[f"ex_{cat}__seen"] = np.array(self.seen[cat])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **out)


def shared_fraction(reference_ids: Tensor, ids: Tensor) -> Tensor:
    """Per row, the share of `reference_ids` (B, N0) that also occur in `ids` (B, N). Entries are unique per row."""
    sorted_ids = ids.sort(dim=1).values
    pos = torch.searchsorted(sorted_ids, reference_ids.contiguous()).clamp_max(ids.shape[1] - 1)
    return (sorted_ids.gather(1, pos) == reference_ids).float().mean(1)


def pack_examples(examples: List[dict], n_copies: int) -> Dict[str, np.ndarray]:
    """Ragged per-copy active sets -> padded arrays (n, copies, N_max); padding: id -1, queue -1, weight 0."""
    n = len(examples)
    width = max([len(a) for e in examples for a in e["active_ids"]], default=0)
    ids = np.full((n, n_copies, width), -1, dtype=np.int64)
    queue = np.full((n, n_copies, width), -1, dtype=np.int32)
    weight = np.zeros((n, n_copies, width), dtype=np.float32)
    for i, e in enumerate(examples):
        for c in range(n_copies):
            k = len(e["active_ids"][c])
            ids[i, c, :k], queue[i, c, :k], weight[i, c, :k] = e["active_ids"][c], e["active_queue"][c], e["active_weight"][c]
    return {"stream_id": np.array([e["stream_id"] for e in examples], dtype=np.int64),
            "label": np.array([e["label"] for e in examples], dtype=np.int64),
            "zs_pred": np.array([e["zs_pred"] for e in examples], dtype=np.int64),
            "preds": np.array([e["preds"] for e in examples], dtype=np.int64).reshape(n, n_copies),
            "active_ids": ids, "active_queue": queue, "active_weight": weight}


def load_log(path: Path) -> Dict[str, np.ndarray]:
    with np.load(Path(path)) as f:
        return {name: f[name] for name in f.files}
