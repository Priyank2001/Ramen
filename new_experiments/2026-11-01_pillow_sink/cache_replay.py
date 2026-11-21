"""
Ramen's cache, replayed offline, under two filing rules at once: what each cache holds when a given test image
arrives, and which entries it retrieves for that image. No GPU, no gradients.

Why this is exact (up to fp16 arithmetic): Ramen's cache never depends on its adapted outputs. Each test image is
written, before retrieval, into one class queue with its zero-shot CLIP feature (key), its zero-shot entropy and
its stream counter (priority, fp16, newest replaces oldest). Retrieval takes the top-k nearest entries of every
non-empty queue with weight exp(-H) * exp(-beta * d), and the queues are averaged. The filing rule (which queue an
image goes to) is the only thing the two caches differ in; the stored gradient of an entry is the same in both.

    rules = [ZeroShotFiling(), TrueLabelFiling()]
    replay = CacheReplayRun(runs, rules)
    result = replay.run(query_rows, keep_rows=example_rows, queue_classes={row: [target, ...]},
                        checkpoints=np.linspace(0, len(runs.table) - 1, 60).astype(int), watch_classes=[target])
    result.summary                      # one row per (query, cache): where the update weight comes from
    result.retrieved[(row, 'Ramen cache')]   # the retrieved entries for that query, with weights
    result.queues[(row, 'oracle cache', c)]  # stream ids in queue c when that query arrived
    result.history                      # composition of the watched queues over the stream

Adding a filing rule: subclass FilingRule and pass it in; the replay, summaries and plots take any number.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class FilingRule:
    """Which class queue each test image is written to."""
    name = None
    uses_true_labels = False

    def queues(self, table):
        raise NotImplementedError


class ZeroShotFiling(FilingRule):
    """Ramen: the image's zero-shot prediction."""
    name = 'Ramen cache'

    def queues(self, table):
        return table['zs_pred'].to_numpy().astype(np.int64)


class TrueLabelFiling(FilingRule):
    """ORACLE: the image's true label (RamenTrueLabel, filing=true_label)."""
    name = 'oracle cache'
    uses_true_labels = True

    def queues(self, table):
        return table['label'].to_numpy().astype(np.int64)


class ReplayedCache:
    """Ramen's per-class PriorityCache (src/methods/Ramen.py) holding stream ids and keys instead of gradients."""

    def __init__(self, num_classes, capacity, key_dim, priority_fp16=True):
        self.C, self.cap = num_classes, capacity
        self.ids = np.full((num_classes, capacity), -1, dtype=np.int64)
        self.prio = np.full((num_classes, capacity), -np.inf, dtype=np.float32)
        self.keys = np.zeros((num_classes, capacity, key_dim), dtype=np.float32)
        self.sq_norm = np.zeros((num_classes, capacity), dtype=np.float32)
        self.size = np.zeros(num_classes, dtype=np.int64)
        self.priority_fp16 = priority_fp16

    def priority(self, t):
        # Ramen: torch.arange(counter, counter + B, dtype=half): integers rounded to fp16, inf from 65,520 on
        if not self.priority_fp16:
            return np.float32(t)
        with np.errstate(over='ignore'):
            return np.float32(np.float16(t))

    def write(self, t, c, key):
        if self.size[c] < self.cap:
            slot = self.size[c]
            self.size[c] += 1
        else:
            slot = int(np.argmin(self.prio[c]))  # first minimum, as torch.argmin
            p = self.priority(t)
            if not p > self.prio[c, slot]:
                return  # fp16 priorities saturated: a full queue no longer changes
        self.ids[c, slot], self.prio[c, slot] = t, self.priority(t)
        self.keys[c, slot] = key
        self.sq_norm[c, slot] = key @ key

    def queue(self, c):
        """Stream ids in queue c, newest first."""
        ids = self.ids[c, :self.size[c]]
        return np.sort(ids)[::-1]

    def retrieve(self, queries, entropy, topk, beta):
        """
        For each query: the top-k nearest entries of every non-empty queue (Ramen's PriorityCache.query, the
        cdist 'mm' formula) and their weights exp(-H) * exp(-beta * d). Returns arrays (Q, C, k) of stream ids,
        distances and weights; empty slots have id -1 and weight 0.
        """
        k = min(topk, self.cap)
        q_sq = (queries * queries).sum(1)
        dot = (queries @ self.keys.reshape(-1, self.keys.shape[2]).T).reshape(len(queries), self.C, self.cap)
        d2 = q_sq[:, None, None] + self.sq_norm[None] - 2 * dot
        dist = np.sqrt(np.clip(d2, 0, None))
        filled = np.arange(self.cap)[None, :] < self.size[:, None]
        dist = np.where(filled[None], dist, np.inf)
        part = np.argpartition(dist, k - 1, axis=2)[..., :k]
        dsel = np.take_along_axis(dist, part, axis=2)
        order = np.argsort(dsel, axis=2, kind='stable')
        slots = np.take_along_axis(part, order, axis=2)
        dsel = np.take_along_axis(dsel, order, axis=2)
        cls = np.arange(self.C)[None, :, None]
        ids = np.where(np.isfinite(dsel), self.ids[cls, slots], -1)
        w = np.where(ids >= 0, np.exp(-entropy[np.clip(ids, 0, None)]) * np.exp(-beta * np.nan_to_num(dsel, posinf=0)),
                     0.0)
        return ids, dsel, w


@dataclass
class ReplayResult:
    summary: pd.DataFrame
    retrieved: dict = field(default_factory=dict)   # (row, cache) -> DataFrame of retrieved entries
    queues: dict = field(default_factory=dict)      # (row, cache, class) -> stream ids, newest first
    history: pd.DataFrame = None                    # watched-queue composition at the checkpoints
    sizes: dict = field(default_factory=dict)       # (row, cache) -> (entries in cache, non-empty queues, full queues)


class CacheReplayRun:
    """One pass over the test stream, keeping one ReplayedCache per filing rule."""

    def __init__(self, runs, rules, priority_fp16=None):
        self.runs, self.rules = runs, list(rules)
        cfg = runs.cfg
        self.capacity, self.topk, self.beta = cfg['max_capacity'], cfg['topk'], cfg.get('beta', 0.0)
        # Ramen's default keeps fp16 priorities (cfg priority_fp32 unset); the recorded runs used the default
        self.priority_fp16 = not cfg.get('priority_fp32', False) if priority_fp16 is None else priority_fp16
        self.batch_size = runs.exp.batch_size
        t = runs.table
        self.label, self.zs_pred = t.label.to_numpy(), t.zs_pred.to_numpy()
        self.domain = t.domain_idx.to_numpy()
        self.filed = {r.name: r.queues(t) for r in self.rules}

    def run(self, query_rows, keep_rows=(), queue_classes=None, checkpoints=(), watch_classes=(), target=None,
            progress=True):
        """
        query_rows     rows to summarise the retrieval for (Ramen retrieves after writing the query's batch)
        keep_rows      rows whose full retrieved-entry table is kept
        queue_classes  {row: [class, ...]}: queue contents to keep for that row
        checkpoints    stream positions at which the watched queues' composition is recorded
        target         the sink class; the summary measures the update weight that comes from it
        """
        runs = self.runs
        # keys = zero-shot features, as Ramen stores them (kept fp16 in memory: 0.6 GB for DomainNet)
        feats = np.asarray(runs.records['Ramen'].arrays['emb_before'])
        entropy = runs.zero_shot_entropy().astype(np.float64)
        n, B = len(feats), self.batch_size
        caches = {r.name: ReplayedCache(runs.num_classes, self.capacity, feats.shape[1], self.priority_fp16)
                  for r in self.rules}
        query_rows = np.unique(np.asarray(query_rows, dtype=np.int64))
        keep_rows, queue_classes = set(int(r) for r in keep_rows), queue_classes or {}
        need = np.union1d(query_rows, np.asarray(sorted(set(keep_rows) | set(queue_classes)), dtype=np.int64))
        by_batch = pd.Series(need).groupby(need // B).apply(np.asarray).to_dict()
        checkpoints = set(int(c) for c in checkpoints)
        query_set = set(int(r) for r in query_rows)
        last = int(max([need.max() if len(need) else 0] + list(checkpoints)))
        out = ReplayResult(summary=None)
        rows, hist, weights = [], [], {}  # weights: (row, cache) -> entry weights by stream id, for the overlap
        it = range((last // B) + 1)
        if progress:
            from tqdm.auto import tqdm
            it = tqdm(it, desc='replaying the caches', mininterval=5)
        for b in it:
            lo, hi = b * B, min((b + 1) * B, n)
            for t in range(lo, hi):
                for name, cache in caches.items():
                    cache.write(t, self.filed[name][t], feats[t].astype(np.float32))
            for t in checkpoints.intersection(range(lo, hi)):
                for name, cache in caches.items():
                    for c in watch_classes:
                        hist.append(self._queue_stats(t, name, c, cache.queue(c)))
            if b not in by_batch:
                continue
            q = by_batch[b]
            for name, cache in caches.items():
                ids, dist, w = cache.retrieve(feats[q].astype(np.float32), entropy, self.topk, self.beta)
                for i, row in enumerate(q):
                    row = int(row)
                    entries = self._entries(row, ids[i], dist[i], w[i], cache)
                    if row in query_set:
                        rows.append(self._summary(row, name, entries, target))
                        weights[(row, name)] = entries.set_index('id')['weight']
                    if row in keep_rows:
                        out.retrieved[(row, name)] = entries
                        out.sizes[(row, name)] = (int(cache.size.sum()), int((cache.size > 0).sum()),
                                                  int((cache.size == cache.cap).sum()))
                    for c in queue_classes.get(row, ()):
                        out.queues[(row, name, int(c))] = cache.queue(int(c))
        out.summary = pd.DataFrame(rows).set_index(['row', 'cache']).sort_index()
        out.history = pd.DataFrame(hist)
        out.summary = self._add_overlap(out.summary, weights)
        return out

    # ------------------------------------------------------------------ per query
    def _entries(self, row, ids, dist, w, cache):
        """The retrieved entries of one query, weights normalised to sum 1 (their share of the update)."""
        m = ids >= 0
        queue = np.broadcast_to(np.arange(cache.C)[:, None], ids.shape)[m]
        e = pd.DataFrame({'id': ids[m], 'queue': queue, 'dist': dist[m], 'weight': w[m]})
        e['weight'] /= e['weight'].sum()
        e['true'] = self.label[e['id']]
        e['zs_pred'] = self.zs_pred[e['id']]
        e['domain_idx'] = self.domain[e['id']]
        e['self'] = e['id'] == row
        return e.sort_values('weight', ascending=False, ignore_index=True)

    def _summary(self, row, name, e, target):
        y, zs, dom = self.label[row], self.zs_pred[row], self.domain[row]
        w = e['weight']
        s = {'row': row, 'cache': name,
             'self': w[e['self']].sum(),
             'from target queue': w[e['queue'] == target].sum(),
             'from zero-shot-class queue': w[e['queue'] == zs].sum(),
             'from true-class queue': w[e['queue'] == y].sum(),
             'entries truly target': w[e['true'] == target].sum(),
             'entries zero-shot-predicted target': w[e['zs_pred'] == target].sum(),
             'entries zero-shot-predicted as query\'s zero-shot class': w[e['zs_pred'] == zs].sum(),
             'entries truly query\'s class': w[e['true'] == y].sum(),
             'entries that are zero-shot errors': w[e['zs_pred'] != e['true']].sum(),
             'entries from query\'s domain': w[e['domain_idx'] == dom].sum(),
             'top-10 entries truly target': int((e['true'].head(10) == target).sum()),
             'non-empty queues': int(e['queue'].nunique())}
        return s

    def _add_overlap(self, summary, weights):
        """Share of each cache's update weight on entries the other cache retrieves too."""
        names = [r.name for r in self.rules]
        if len(names) != 2:
            return summary
        a, b = names
        ov = {}
        for row in summary.index.get_level_values('row').unique():
            wa, wb = weights[(row, a)], weights[(row, b)]
            common = wa.index.intersection(wb.index)
            ov[(row, a)], ov[(row, b)] = wa[common].sum(), wb[common].sum()
        summary['weight on entries both caches retrieve'] = pd.Series(ov)
        return summary

    def _queue_stats(self, t, name, c, ids):
        n = len(ids)
        return {'position': t, 'cache': name, 'queue': c, 'entries': n,
                'truly this class %': (self.label[ids] == c).mean() * 100 if n else np.nan,
                'zero-shot-predicted this class %': (self.zs_pred[ids] == c).mean() * 100 if n else np.nan,
                'newest entry': int(ids.max()) if n else -1,
                **{f'from domain {d} %': (self.domain[ids] == d).mean() * 100 if n else np.nan
                   for d in range(int(self.domain.max()) + 1)}}

    # ------------------------------------------------------------------ check
    def validate(self, stats, ramen_pred, n=200, seed=0):
        """
        Replay (zero-shot filing) vs what a RamenActive 'all' run logged on the same stream (= Ramen's retrieval):
        the queue of its predicted class, and its top-8 retrieved entries. Differences come from fp16 distances.
        """
        rule = next(r for r in self.rules if isinstance(r, ZeroShotFiling))
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(np.arange(int(0.05 * len(self.label)), len(self.label)), n, replace=False))
        qc = {int(r): [int(ramen_pred[r])] for r in rows}
        res = CacheReplayRun(self.runs, [rule]).run(rows, keep_rows=rows, queue_classes=qc, progress=False)
        sup_ids, queue_ids = np.concatenate(stats['support_ids']), np.concatenate(stats['queue_ids'])
        q_ok, top1, overlap = [], [], []
        for r in rows:
            logged_q = np.sort(queue_ids[r][queue_ids[r] >= 0])
            q_ok.append(np.array_equal(logged_q, np.sort(res.queues[(int(r), rule.name, int(ramen_pred[r]))])))
            logged = sup_ids[r][sup_ids[r] >= 0]
            mine = res.retrieved[(int(r), rule.name)]['id'].to_numpy()[:len(logged)]
            top1.append(logged[0] == mine[0])
            overlap.append(len(set(logged) & set(mine)) / len(logged))
        return pd.Series({'queries checked': n,
                          'queue of the predicted class identical %': np.mean(q_ok) * 100,
                          'top-1 retrieved entry identical %': np.mean(top1) * 100,
                          'top-8 retrieved entries overlap %': np.mean(overlap) * 100}).round(2)
