"""
Ramen with a no-reference image-quality regulariser on the active-set selection (blindqa.pdf: NIQE, BRISQUE).

Ramen's active set for query b holds, from every class queue c, the top-k entries nearest to the query in CLIP
feature space. Here each entry j's selection score gets a quality term:

    s_bj = d_bj + lambda * R_bj          active set = the top-k smallest s_bj of each class queue
                                          Ramen's weight exp(-H_j) * exp(-beta * d_bj) becomes exp(-H_j) * exp(-beta * s_bj)

Regulariser R (config `quality_reg`), over the metrics M in `quality_metric` (niqe | brisque | both):
  match   R_bj = sqrt(mean_{m in M} (r_m(b) - r_m(j))^2)   entries whose image quality is like the query's, i.e. a
                                                            distortion-consistent active set
  prior   R_bj = mean_{m in M} r_m(j)                       entries whose images look natural (low NIQE / BRISQUE)
r_m(.) is the rank of a score among all scores seen so far in the stream, scaled to [0, 1] (1 = worst quality;
an image without a score ranks 1). Ranks put NIQE and BRISQUE on one scale, so `both` weighs them equally and
lambda (`quality_lambda`) is in units of CLIP feature distance. Everything is causal and label-free: the ranks use
the test images seen so far, including the current batch, which Ramen writes into memory before retrieving.
lambda = 0 is Ramen exactly (with `active_set: all`); other `active_set` selectors from RamenActive compose.

Memory: QualityCache, Ramen's per-class priority queue with the same insert / replace / query rules, that also
stores each entry's NIQE, BRISQUE and true label. The true labels are recorded for diagnostics only (label purity
of the active set, in self.stats); nothing that selects, weights or predicts reads them. The harness passes them as
forward(x, labels=...) to methods with records_labels = True; without them they are stored as -1.

Per-sample log (self.stats, one array per batch, rows line up with ramen_analysis' results frame), on top of
RamenActive's filed_as / support_ids / support_weight:
  niqe, brisque    (B,)     the query image's scores
  support_purity   (B,)     weight share of the active set whose true label is the query's (NaN without labels)
  support_gap      (B, 2)   weighted mean |r_m(b) - r_m(j)| over the active set, for NIQE and BRISQUE
"""
import numpy as np
import torch

from models.iqa import ImageQuality

from .losses import softmax_entropy
from .RamenActive import IdCache, RamenActive

METRIC_COLUMNS = {'niqe': [0], 'brisque': [1], 'both': [0, 1]}  # columns of ImageQuality's output


# ---------------------------------------------------------------------------- memory
class QualityCache(IdCache):
    """
    One class queue of Ramen's memory (PriorityCache, plus stream ids as in IdCache) that also stores, for every
    entry, the image's quality scores (NIQE, BRISQUE) and its true label (-1 if unknown).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_metrics = len(ImageQuality.METRICS)
        self.quality = torch.full((self.max_capacity, n_metrics), float('nan'), device=self.device)
        self.labels = torch.full((self.max_capacity,), -1, device=self.device, dtype=torch.long)

    def add(self, keys, values, entropies, priorities, ids, quality, labels, **extra):
        # PriorityCache's slot rule: append while there is room, else replace the lowest priority if the new one is
        # higher. Priorities stay fp16 as in Ramen (they saturate after 65,504 samples), so lambda = 0 is Ramen.
        # extra: name=rows of further per-entry fields that a subclass allocated as self.<name> (e.g. NSSCache.nss).
        for i, (key, value, entropy, priority, id_, q, y) in enumerate(
                zip(keys, values, entropies, priorities, ids, quality, labels)):
            if self.size < self.max_capacity:
                slot = self.size
                self.size += 1
            else:
                slot = torch.argmin(self.priorities)
                if not priority > self.priorities[slot]:
                    continue
            self.keys[slot], self.values[slot] = key, value
            self.priorities[slot], self.entropies[slot], self.ids[slot] = priority, entropy, id_
            self.quality[slot], self.labels[slot] = q, y
            for name, rows in extra.items():
                getattr(self, name)[slot] = rows[i]

    def query(self, queries, topk, penalty=None):
        """
        Without penalty: Ramen's retrieval, the top-k entries by feature distance (same ops as IdCache.query).
        With penalty (num_queries, size): the top-k by feature distance + penalty, and `score` is that sum.
        Returns values, entropies, score, stream ids and slots of the retrieved entries (slots index quality, labels).
        """
        topk = min(topk, self.size)
        dist = torch.cdist(queries.detach().to(device=self.device, dtype=self.dtype), self.keys[:self.size])
        score = dist if penalty is None else dist.float() + penalty
        score, idx = torch.topk(score, k=topk, dim=1, largest=False, sorted=True)
        return self.values[idx], self.entropies[idx], score, self.ids[idx], idx

    def stored_quality(self):
        return self.quality[:self.size]

    def stored_labels(self):
        return self.labels[:self.size]

    def reset(self):
        super().reset()
        self.ids.fill_(-1)
        self.quality.fill_(float('nan'))
        self.labels.fill_(-1)


class StreamRank:
    """Rank of each quality score among all scores seen so far in the stream, in [0, 1]; NaN -> 1 (worst)."""

    def __init__(self, n_metrics, device):
        self.seen = [torch.empty(0, device=device) for _ in range(n_metrics)]  # sorted, per metric

    def update(self, quality):
        for m, column in enumerate(quality.T):
            self.seen[m] = torch.cat([self.seen[m], column[~torch.isnan(column)]]).sort().values

    def __call__(self, quality):
        ranks = torch.ones_like(quality)
        for m, column in enumerate(quality.T):
            s = self.seen[m]
            if len(s):  # mid-rank, so ties share one value
                lo = torch.searchsorted(s, column.contiguous(), right=False)
                hi = torch.searchsorted(s, column.contiguous(), right=True)
                ranks[:, m] = (lo + hi).float() / (2 * len(s))
        return torch.where(torch.isnan(quality), torch.ones_like(ranks), ranks)


# ---------------------------------------------------------------------------- the method
class RamenQuality(RamenActive):

    records_labels = True  # the harness passes the true labels: QualityCache stores them, for diagnostics only

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        metric = self.cfg.get('quality_metric', 'both')
        if metric not in METRIC_COLUMNS:
            raise ValueError(f'Unknown quality_metric {metric!r}; choose from {list(METRIC_COLUMNS)}')
        self.columns = METRIC_COLUMNS[metric]
        self.reg = self.cfg.get('quality_reg', 'match')
        if self.reg not in ('match', 'prior'):
            raise ValueError(f"Unknown quality_reg {self.reg!r}; choose 'match' or 'prior'")
        self.lam = float(self.cfg.get('quality_lambda', 0.0))

        self.iqa = ImageQuality(self.device)
        self.ranker = StreamRank(len(ImageQuality.METRICS), self.device)
        self.cache = [QualityCache(max_capacity=self.cfg['max_capacity'], key_dim=self.feat_dim,
                                   value_dim=self.grad_dim, device=self.device, dtype=self.dtype)
                      for _ in range(self.num_classes)]
        self.stats.update({'quality_metric': metric, 'quality_reg': self.reg, 'quality_lambda': self.lam,
                           'niqe': [], 'brisque': [], 'support_purity': [], 'support_gap': []})

    def penalty(self, q_rank, e_rank):
        """lambda * R for queries (B, metrics) x entries (n, metrics) of quality ranks -> (B, n); None if lambda = 0."""
        if self.lam == 0:
            return None
        q, e = q_rank[:, self.columns], e_rank[:, self.columns]
        if self.reg == 'match':
            r = (q[:, None, :] - e[None, :, :]).pow(2).mean(dim=2).sqrt()
        else:
            r = e.mean(dim=1)[None, :].expand(len(q), -1)
        return self.lam * r

    def class_penalty(self, cache, q_rank, e_rank):
        """Selection penalty (B, n) for one class queue's entries, or None. Subclasses add terms (RamenNSS)."""
        return self.penalty(q_rank, e_rank)

    def measure(self, x):
        """NIQE and BRISQUE (B, 2) of the incoming images. Subclasses may also keep the features (RamenNSS)."""
        return self.iqa(x)

    @torch.no_grad()
    def write(self, feats, grads, zs_logits, ids, quality, labels):
        """Ramen's write (every sample, before retrieval, into its zero-shot class queue), with quality and label."""
        B = feats.shape[0]
        zs_pred = zs_logits.argmax(-1)
        priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.dtype)
        entropies = softmax_entropy(zs_logits, reduction='none')
        self.counter = self.counter + B
        for b in range(B):
            self.cache[zs_pred[b]].add(feats[b:b + 1], grads[b:b + 1], entropies[b:b + 1], priorities[b:b + 1],
                                       ids[b:b + 1], quality[b:b + 1], labels[b:b + 1])
        self.stats['filed_as'].append(zs_pred.cpu().numpy().astype(np.int32))

    @torch.no_grad()
    def aggregate(self, feats, grads, class_w, quality, labels):
        """RamenActive.aggregate with the quality-regularised selection score, plus active-set diagnostics."""
        B = grads.shape[0]
        q_rank = self.ranker(quality)
        total = torch.zeros_like(grads)
        norm = torch.zeros((B, 1), device=grads.device)
        entry_w, entry_ids = [], []
        w_sum = torch.zeros(B, device=grads.device)
        same_label = torch.zeros(B, device=grads.device)
        gap = torch.zeros((B, q_rank.shape[1]), device=grads.device)
        active = ((class_w > 0).any(dim=0)).tolist()
        for c in range(self.num_classes):
            cache = self.cache[c]
            if cache.size == 0 or not active[c]:
                continue
            e_rank = self.ranker(cache.stored_quality())
            retrieved, entropies, score, ids, slots = cache.query(feats, self.cfg['topk'],
                                                                  self.class_penalty(cache, q_rank, e_rank))
            weights = torch.exp(-entropies) * torch.exp(-self.beta * score)  # (B, k), Ramen's with s for d
            total += class_w[:, c:c + 1] * (retrieved * weights.unsqueeze(dim=2)).sum(dim=1)
            norm += class_w[:, c:c + 1]

            w = class_w[:, c:c + 1] * weights.float()
            entry_w.append(w)
            entry_ids.append(ids)
            w_sum += w.sum(dim=1)
            same_label += (w * (cache.labels[slots] == labels[:, None])).sum(dim=1)
            gap += (w.unsqueeze(2) * (e_rank[slots] - q_rank[:, None, :]).abs()).sum(dim=1)
        self.log_support(entry_w, entry_ids, B)

        w_sum = w_sum.clamp_min(1e-12)
        purity = torch.where(labels >= 0, same_label / w_sum, torch.full_like(w_sum, float('nan')))
        self.stats['support_purity'].append(purity.cpu().numpy().astype(np.float32))
        self.stats['support_gap'].append((gap / w_sum[:, None]).cpu().numpy().astype(np.float32))
        # An all-zero weight row gives a zero gradient: SignSGD then doesn't move, so the prediction is zero-shot.
        return total / norm.clamp_min(1e-12)

    def forward(self, x, labels=None):
        B = x.shape[0]
        ids = torch.arange(self.seen, self.seen + B, device=self.device)
        self.seen += B
        labels = (torch.full((B,), -1, dtype=torch.long, device=self.device) if labels is None
                  else labels.to(self.device))

        # 0. NIQE and BRISQUE of every incoming image; the stream's score ranks include this batch.
        quality = self.measure(x)
        self.ranker.update(quality)
        self.stats['niqe'].append(quality[:, 0].cpu().numpy())
        self.stats['brisque'].append(quality[:, 1].cpu().numpy())

        # 1. Zero-shot features, logits and each sample's own entropy gradient (as in Ramen).
        feats = self.model.featurize(x)
        zs_logits = self.model.classify(feats)
        self.loss_fn(zs_logits).backward()
        grads = self.model.get_by_sample_grad()

        # 2. Write, then retrieve the quality-regularised active set, take one step, predict, reset.
        self.write(feats, grads, zs_logits, ids, quality, labels)
        class_w = self.selector(zs_logits.detach())
        self.model.set_by_sample_grad(self.aggregate(feats, grads, class_w, quality, labels))
        self.model.step_and_zero_grad()
        with torch.no_grad():
            logits = self.model(x)
        self.model.reset_parameters()

        if self.log_queue:
            self.log_queues(logits.argmax(-1))
        return logits

    def reset(self):
        super().reset()  # model, priority counter and queues; self.seen and the stream's quality ranks keep going
