"""
Evidence trace for the active set (the cache entries retrieved for each test image) of a RamenLabelValues run.

It replays the first N batches of the real stream with an instrumented copy of the method (TracedRamenLabelValues,
a subclass: the retrieval code that runs is the method's own; the tracer only watches it) and checks, batch by batch:

  C0  the traced batches predict exactly what the cached full run predicted for those images
  C1  filing: every image is written to the queue of its zero-shot prediction
  C2  the image's own entry is in the cache when it retrieves, and is its own queue's nearest entry (rank 0)
  C3  captured active set = the entries the method actually used (indices reproduce its retrieved values exactly)
  C4  independent float64 recomputation (exact distances, top-k, weights e^(-H) e^(-beta d), average over queues)
      selects the same entries, and gives an update with the same signs as the one the method stepped with
  C5  no label leak: while a batch is adapted its own entries hold their ENTROPY gradients; after prediction
      they hold the label (CE) gradients; earlier entries retrieved later hold their CE gradients
  C6  the active set does not depend on the values: label-value run and entropy-value run (= Ramen) select
      identical entries with identical distances and weights at every step

    CUDA_VISIBLE_DEVICES=0 python trace_active_set.py DomainNet 0 --batches 60 | tee logs/trace_active_set_DomainNet.log
"""
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from label_oracle import VARIANTS, LabelOracleExperiment
from label_methods import RamenLabelValues
from ramen_analysis import _TaggedDomain, tqdm  # the harness's tqdm: wrapping the loader consumes the seeded
# generator at construction (tqdm 4.70), so the same wrapping is needed to get the harness's stream order


class TracedRamenLabelValues(RamenLabelValues):
    """RamenLabelValues that records what it files, stores and retrieves; its behaviour is unchanged."""

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        cap = self.cfg['max_capacity']
        self.slot_ids = np.full((self.num_classes, cap), -1, dtype=np.int64)   # stream id held by each slot
        self.meta = {}        # stream id -> (true label, zero-shot class)
        self.stored = {}      # stream id -> the value it must hold now (fp16, CPU)
        self.checks = {k: [] for k in ('C1', 'C2', 'C3', 'C4_sets', 'C4_sign', 'C4_relerr', 'C5_batch_entropy',
                                       'C5_batch_revealed', 'C5_old_entries')}
        self.active = []      # per batch: (B, C, k) stream ids, distances, weights (-1 / nan where empty)
        self._captured = {}
        for c, cache in enumerate(self.cache):
            cache.query = self._capturing(c, cache, cache.query)

    # -------------------------------------------------------------- capture the method's own retrieval
    def _capturing(self, c, cache, query):
        def wrapped(queries, topk):
            values, priorities, entropies, sorted_dist = query(queries, topk)
            # the same call PriorityCache.query makes, so we can read the indices it chose
            q = queries.detach().to(device=cache.device, dtype=cache.dtype)
            dist = torch.cdist(q, cache.keys[:cache.size], compute_mode='use_mm_for_euclid_dist')
            sd, idx = torch.topk(dist, k=min(topk, cache.size), dim=1, largest=False, sorted=True)
            # C3: these indices are the ones used iff they reproduce exactly what the method retrieved
            self.checks['C3'].append(bool(torch.equal(sd, sorted_dist) and torch.equal(cache.values[idx], values)))
            self._captured[c] = (idx, sorted_dist, entropies)
            return values, priorities, entropies, sorted_dist
        return wrapped

    def forward(self, x, labels=None):
        self._labels = labels.to(self.device)
        return super().forward(x, labels)

    @torch.no_grad()
    def write(self, feats, grads, logits, queues):
        first = self.counter
        zs = logits.argmax(-1)
        if self.filing == 'zero_shot':
            self.checks['C1'].append(bool(torch.equal(queues, zs)))                      # C1
        slots = super().write(feats, grads, logits, queues)
        for b, (c, slot) in enumerate(slots):
            i = first + b
            self.meta[i] = (int(self._labels[b]), int(zs[b]))
            if slot is not None:
                self.slot_ids[c, slot] = i
                self.stored[i] = grads[b].to(self.dtype).cpu()                           # entropy value for now
        self._batch = (first, slots, grads)
        return slots

    @torch.no_grad()
    def retrieve(self, feats, grads):
        first, slots, _ = self._batch
        last_writer = {}
        for b, (c, slot) in enumerate(slots):
            if slot is not None:
                last_writer[(c, slot)] = b
        # C5a: the batch's own entries hold their entropy gradients while the batch is adapted
        self.checks['C5_batch_entropy'].append(all(
            torch.equal(self.cache[c].values[slot], grads[b].to(self.dtype)) for (c, slot), b in last_writer.items()))

        self._captured = {}
        update = super().retrieve(feats, grads)       # the method's own retrieval (captured by the wrappers)

        B, C, k = feats.shape[0], self.num_classes, self.cfg['topk']
        ids = np.full((B, C, k), -1, dtype=np.int64)
        dist = np.full((B, C, k), np.nan, dtype=np.float32)
        weight = np.zeros((B, C, k), dtype=np.float32)
        for c, (idx, sd, ent) in self._captured.items():
            kk = idx.shape[1]
            ids[:, c, :kk] = self.slot_ids[c][idx.cpu().numpy()]
            dist[:, c, :kk] = sd.float().cpu().numpy()
            weight[:, c, :kk] = (torch.exp(-ent) * torch.exp(-self.beta * sd)).float().cpu().numpy()
        self.active.append((ids, dist, weight))

        # C2: each stored image is its own queue's nearest entry
        for b, (c, slot) in enumerate(slots):
            if slot is not None and last_writer.get((c, slot)) == b:
                self.checks['C2'].append(int(ids[b, c, 0]) == first + b)

        # C5c: entries from earlier batches hold the value they must hold (CE once revealed)
        old = {int(i) for i in np.unique(ids) if 0 <= i < first}
        self.checks['C5_old_entries'].append(all(
            torch.equal(self.cache[self._queue_of_id(i)].values[self._slot_of_id(i)].cpu(), self.stored[i])
            for i in list(old)[:200]))

        self._independent_check(feats, update, ids)
        return update

    def _queue_of_id(self, i):
        return int(np.argwhere(self.slot_ids == i)[0, 0])

    def _slot_of_id(self, i):
        return int(np.argwhere(self.slot_ids == i)[0, 1])

    @torch.no_grad()
    def _independent_check(self, feats, update, ids):
        """C4: recompute the active set and the update in float64 from the cache contents."""
        q = feats.double()
        total = torch.zeros_like(update, dtype=torch.float64)
        same_sets, n_queues = [], 0
        for c, cache in enumerate(self.cache):
            if cache.size == 0:
                continue
            n_queues += 1
            keys = cache.keys[:cache.size].double()
            d = torch.cdist(q, keys)                                  # exact Euclidean distances
            k = min(self.cfg['topk'], cache.size)
            sd, idx = torch.topk(d, k=k, dim=1, largest=False, sorted=True)
            w = torch.exp(-cache.entropies[idx].double()) * torch.exp(-self.beta * sd)
            total += (cache.values[idx].double() * w.unsqueeze(2)).sum(1)
            mine = self.slot_ids[c][idx.cpu().numpy()]
            same_sets.append([set(mine[b]) == set(ids[b, c, :k]) for b in range(len(mine))])
        ref = total / n_queues
        self.checks['C4_sets'].append(float(np.mean(same_sets)))
        self.checks['C4_sign'].append(float((torch.sign(ref) == torch.sign(update.double())).double().mean()))
        self.checks['C4_relerr'].append(float(((update.double() - ref).norm(dim=1) / ref.norm(dim=1)).max()))

    @torch.no_grad()
    def reveal(self, slots, values):
        super().reveal(slots, values)
        first = self.counter - len(slots)
        last_writer = {}
        for b, (c, slot) in enumerate(slots):
            if slot is not None:
                last_writer[(c, slot)] = b
        ok = True
        for (c, slot), b in last_writer.items():
            ok &= torch.equal(self.cache[c].values[slot], values[b].to(self.dtype))
            self.stored[first + b] = values[b].to(self.dtype).cpu()                    # now the CE value
        self.checks['C5_batch_revealed'].append(ok)


# ------------------------------------------------------------------------------------------------ driver
def traced_run(exp, name, n_batches, cls=None):
    """The harness's own loop (same seed, loader and order) over the first n_batches, with the traced method."""
    h = exp.exp
    h._seed()
    args = SimpleNamespace(config={**h._load_cfg('Ramen'), **VARIANTS[name][1]}, device=h.device,
                           max_batch_size=h.batch_size, use_tbn=False)
    method = (cls or TracedRamenLabelValues)(deepcopy(h.clip), h.datasets, args)
    stream = ConcatDataset([_TaggedDomain(ds, d) for d, ds in enumerate(h.datasets)])
    loader = DataLoader(stream, batch_size=h.batch_size, shuffle=True, num_workers=h.num_workers, pin_memory=True,
                        generator=torch.Generator().manual_seed(h.seed))
    preds, keys = [], []
    for t, (image, label, domain_idx, sample_idx) in enumerate(tqdm(loader, desc=name)):
        if t == n_batches:
            break
        logits = method(image.to(h.device), labels=label)
        preds.append(logits.argmax(1).cpu().numpy())
        keys.append(np.stack([domain_idx.numpy(), sample_idx.numpy()], 1))
    return method, np.concatenate(preds), np.concatenate(keys)


def frac(xs):
    xs = list(xs)
    return f'{sum(bool(x) for x in xs):,} of {len(xs):,}'


def report(dataset, seed, n_batches):
    exp = LabelOracleExperiment(dataset, seed=seed)
    exp.run('Ramen')
    exp.run('labeloracle_zs_ce')
    exp.exp.load()
    exp.loaded = True
    classes = exp.exp.classes
    out = {}
    for name, reference in (('labeloracle_zs_ce', 'labeloracle_zs_ce'), ('labeloracle_off', 'Ramen')):
        method, preds, keys = traced_run(exp, name, n_batches)
        ref = exp.results[reference].iloc[:len(preds)]
        out[name] = (method, preds)
        ch = method.checks
        print(f'\n=== {name} (value = {method.stats["value"]}, filing = {method.filing}), '
              f'first {n_batches} batches = {len(preds):,} images')
        print(f'C0  same images as the cached full run, in the same order: '
              f'{bool((ref[["domain_idx", "sample_idx"]].to_numpy() == keys).all())}')
        print(f'C0  traced prediction == cached `{reference}` prediction: {frac(preds == ref.pred.to_numpy())} images')
        print(f'C1  batches filed by zero-shot prediction: {frac(ch["C1"])}')
        print(f'C2  stored images that are their own queue\'s nearest entry (rank 0): {frac(ch["C2"])}')
        print(f'C3  queue queries whose captured indices reproduce the retrieved values exactly: {frac(ch["C3"])}')
        print(f'C4  float64 recomputation: same entries chosen for {np.mean(ch["C4_sets"]):.2%} of (image, queue) '
              f'pairs; update signs agree on {np.mean(ch["C4_sign"]):.4%} of coordinates; '
              f'max relative error {max(ch["C4_relerr"]):.1e}')
        print(f'C5  batches whose own entries hold ENTROPY gradients while adapting: {frac(ch["C5_batch_entropy"])}')
        print(f'C5  batches whose entries hold the {"CE" if name.endswith("ce") else "entropy"} value after '
              f'prediction: {frac(ch["C5_batch_revealed"])}')
        print(f'C5  batches whose retrieved earlier entries hold their expected value: {frac(ch["C5_old_entries"])}')
        method.cache, method.stored = None, None  # keep only the records (frees the GPU memory)
        del method.model
        torch.cuda.empty_cache()

    a, b = out['labeloracle_zs_ce'][0].active, out['labeloracle_off'][0].active
    same_ids = all(np.array_equal(x[0], y[0]) for x, y in zip(a, b))
    same_d = all(np.array_equal(x[1], y[1], equal_nan=True) for x, y in zip(a, b))
    same_w = all(np.array_equal(x[2], y[2]) for x, y in zip(a, b))
    print(f'\nC6  label-value run vs entropy-value run (= Ramen), all {len(a)} batches: same entries {same_ids}, '
          f'same distances {same_d}, same weights {same_w}')

    # a worked example: one image from the last traced batch whose zero-shot prediction is wrong
    m = out['labeloracle_zs_ce'][0]
    ids, dist, w = m.active[-1]
    first = (len(m.active) - 1) * exp.exp.batch_size
    wrong = [b for b in range(ids.shape[0]) if m.meta[first + b][0] != m.meta[first + b][1]]
    b = wrong[0]
    y, zs = m.meta[first + b]
    mask = ids[b] >= 0
    rows = pd.DataFrame({'queue': np.broadcast_to(np.arange(ids.shape[1])[:, None], ids.shape[1:])[mask],
                         'entry id': ids[b][mask], 'distance': dist[b][mask], 'weight': w[b][mask]})
    rows['share of update %'] = rows['weight'] / rows['weight'].sum() * 100  # the 1/(non-empty queues) cancels
    rows['queue'] = [classes[c] for c in rows['queue']]
    rows['entry true class'] = [classes[m.meta[i][0]] for i in rows['entry id']]
    rows['entry zero-shot class'] = [classes[m.meta[i][1]] for i in rows['entry id']]
    rows['value it holds'] = ['entropy (own batch, label not yet revealed)' if i >= first else
                              f'CE toward {classes[m.meta[i][0]]}' for i in rows['entry id']]
    rows['self'] = rows['entry id'] == first + b
    rows = rows.sort_values('weight', ascending=False, ignore_index=True)
    print(f'\nWorked example: stream image {first + b:,}, true class "{classes[y]}", zero-shot "{classes[zs]}"; '
          f'filed in queue "{classes[zs]}". Active set: {len(rows):,} entries from {rows.queue.nunique()} queues.')
    print(f'  share of the update on entries truly "{classes[y]}": '
          f'{rows.loc[rows["entry true class"] == classes[y], "share of update %"].sum():.1f}%  '
          f'(those push toward the right class)')
    print(rows.head(15).round(4).to_string())


if __name__ == '__main__':
    dataset, seed = sys.argv[1], int(sys.argv[2])
    n = int(sys.argv[sys.argv.index('--batches') + 1]) if '--batches' in sys.argv else 60
    report(dataset, seed, n)
