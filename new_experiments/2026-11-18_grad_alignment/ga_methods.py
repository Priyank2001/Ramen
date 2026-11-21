"""
Ramen with a gradient probe: an ORACLE MEASUREMENT, not a method. Predictions are exactly Ramen's.

For every test image it records how well Ramen's update direction agrees with the direction that would lower the
cross-entropy toward the image's TRUE label (ground truth, available only because this is an experiment):

    g_ce   = d CE(logits_i, y_i) / d theta          (ORACLE: uses the true label y_i)
    g_hat  = Ramen's retrieved, weighted gradient    (what Ramen steps along: theta - lr * sign(g_hat))
    g_self = the image's own entropy gradient         (what per-sample entropy minimisation without memory would use)

The probe never feeds anything back: g_ce is computed on a retained graph after Ramen's own entropy backward, and
the parameters Ramen steps are set from g_hat exactly as in Ramen. Filing is fixed to the zero-shot prediction, so
the predictions equal Ramen's (checked: run_grad_alignment.py --smoke, and the full-stream sanity check).

    from ga_methods import RamenGradientProbe
    method = RamenGradientProbe(model, datasets, args)  # args.config = cfg/<dataset>/Ramen.yaml
    logits = method(x, labels=y)                        # the harness passes labels (records_labels)
    method.stats['probe']                               # one dict of (B,) arrays per batch, see PROBE_FIELDS

Per-image fields (all float32 unless noted; "cos" = cosine similarity):
    cos_update          cos(g_hat, g_ce)
    cos_step            cos(sign(g_hat), g_ce): the direction Ramen actually moves
    agree_step          |g_ce|-weighted share of coordinates where sign(g_hat) = sign(g_ce)
    cos_self            cos(g_self, g_ce)
    agree_self          as agree_step, for sign(g_self)
    cos_update_self     cos(g_hat, g_self)
    cos_update_shuffled cos(g_hat_i, g_ce_{i-1}): CONTROL, another image's CE gradient from the same batch
    part_self           share of cos_update from the image's OWN entry (written before retrieval, distance ~0);
                        0 if it was not stored (full queue after the fp16 priority counter saturates)
    part_true           share from the other entries of queue y_i (filed under the true class)
    part_zs             share from the other entries of queue yhat_i (the zero-shot class) when yhat_i != y_i, else 0
    part_other          the rest (all other queues); the four parts sum to cos_update
    norm_update, norm_ce, norm_self
    p_true_zs           zero-shot probability of the true class (from the logits Ramen filed with)
    zs_pred, label      int64: the zero-shot prediction Ramen filed the image under, and the true label
    n_queues            int64, number of non-empty queues averaged (same for the whole batch)
"""
import torch
import torch.nn.functional as F

from methods.RamenTrueLabel import RamenTrueLabel
from methods.losses import softmax_entropy

from alignment import cosine, projection_share, sign_agreement

PROBE_FIELDS = ('cos_update', 'cos_step', 'agree_step', 'cos_self', 'agree_self', 'cos_update_self',
                'cos_update_shuffled', 'part_self', 'part_true', 'part_zs', 'part_other',
                'norm_update', 'norm_ce', 'norm_self', 'p_true_zs', 'zs_pred', 'label', 'n_queues')


class RamenGradientProbe(RamenTrueLabel):
    """Ramen (zero-shot filing) + per-image alignment of its update with the true-label CE gradient (ORACLE)."""

    def __init__(self, model, datasets, args):
        if args.config.get('filing', 'zero_shot') != 'zero_shot':
            raise ValueError('RamenGradientProbe measures Ramen itself: filing must be zero_shot')
        super().__init__(model, datasets, args)
        self.stats.update(probe=[], self_stored=[])

    # ------------------------------------------------------------------ oracle gradient
    def oracle_ce_gradient(self, logits, labels):
        """ORACLE: per-sample gradient of CE(logits_i, y_i) w.r.t. the norm-layer parameters (needs the graph)."""
        self.model.optimizer.zero_grad()           # drop the entropy gradient already read out
        F.cross_entropy(logits.float(), labels, reduction='sum').backward()  # sum -> one gradient per sample
        return self.model.get_by_sample_grad()

    # ------------------------------------------------------------------ write, keeping each image's slot
    @staticmethod
    @torch.no_grad()
    def add(cache, key, value, entropy, priority):
        """
        PriorityCache.add for one entry (same slot rule), returning the slot written, or None if not written.
        Copied from RamenLabelValues.add (new_experiments/2026-11-01_label_oracle/label_methods.py, commit 063343d).
        """
        if cache.size < cache.max_capacity:
            slot = cache.size
            cache.size += 1
        else:
            slot = int(torch.argmin(cache.priorities))
            if not priority > cache.priorities[slot]:
                return None
        cache.keys[slot] = key.detach().to(device=cache.device, dtype=cache.dtype)
        cache.values[slot] = value.detach().to(device=cache.device, dtype=cache.dtype)
        cache.priorities[slot] = priority
        cache.entropies[slot] = entropy
        return slot

    @torch.no_grad()
    def write(self, feats, grads, logits, queues):
        """RamenTrueLabel.write (Ramen's write), returning each image's slot in its queue (-1 if not stored)."""
        B = feats.shape[0]
        priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.priority_dtype)
        entropies = softmax_entropy(logits, reduction='none')
        self.counter = self.counter + B
        slots = [self.add(self.cache[queues[b]], feats[b], grads[b], entropies[b], priorities[b]) for b in range(B)]
        return torch.tensor([-1 if s is None else s for s in slots], device=self.device)

    # ------------------------------------------------------------------ retrieval decomposition
    @torch.no_grad()
    def queue_part(self, feats, classes, zs_pred, self_slot, num_queues):
        """
        For each image b, queue c = classes[b]: Ramen's retrieval term of that queue,
        (1 / |C|) sum_{j in N_k^c(z_b)} e^{-H_j} e^{-beta d_bj} g_j  (zero if the queue is empty),
        split into (rest, own): `own` is the term of image b's own entry (in queue zs_pred[b], slot self_slot[b]),
        `rest` everything else. Same distance call, top-k and weights as PriorityCache.query + Ramen.
        """
        rest = torch.zeros((feats.shape[0], self.grad_dim), device=self.device, dtype=torch.float32)
        own = torch.zeros_like(rest)
        for c in classes.unique().tolist():
            cache = self.cache[c]
            if cache.size == 0:
                continue
            rows = (classes == c).nonzero(as_tuple=True)[0]
            q = feats[rows].detach().to(device=self.device, dtype=self.dtype)
            dist = torch.cdist(q, cache.keys[:cache.size], compute_mode='use_mm_for_euclid_dist')
            sorted_dist, idx = torch.topk(dist, k=min(self.cfg['topk'], cache.size), dim=1, largest=False, sorted=True)
            w = torch.exp(-cache.entropies[idx]) * torch.exp(-self.beta * sorted_dist)       # (rows, k)
            is_own = (zs_pred[rows] == c).unsqueeze(1) & (idx == self_slot[rows].unsqueeze(1))
            terms = cache.values[idx] * w.unsqueeze(2)                                       # (rows, k, D)
            rest[rows] = (terms * (~is_own).unsqueeze(2)).sum(1).float()
            own[rows] = (terms * is_own.unsqueeze(2)).sum(1).float()
        return rest / num_queues, own / num_queues

    # ------------------------------------------------------------------ record
    @torch.no_grad()
    def record(self, update, ce, own_grad, parts, logits, zs_pred, labels):
        """parts: {name: (B, D) additive part of the update}; the remainder is 'part_other'."""
        n_queues = self.stats['nonempty_queues'][-1]
        cos_update = cosine(update, ce)
        shares = {k: projection_share(v, update, ce) for k, v in parts.items()}
        out = dict(
            cos_update=cos_update,
            cos_step=cosine(torch.sign(update.float()), ce),
            agree_step=sign_agreement(update, ce),
            cos_self=cosine(own_grad, ce),
            agree_self=sign_agreement(own_grad, ce),
            cos_update_self=cosine(update, own_grad),
            cos_update_shuffled=cosine(update, ce.roll(1, dims=0)),
            **shares, part_other=cos_update - sum(shares.values()),
            norm_update=update.float().norm(dim=1), norm_ce=ce.float().norm(dim=1),
            norm_self=own_grad.float().norm(dim=1),
            p_true_zs=logits.float().softmax(1).gather(1, labels[:, None]).squeeze(1),
            zs_pred=zs_pred, label=labels,
            n_queues=torch.full_like(labels, n_queues),
        )
        self.stats['probe'].append({k: v.cpu().numpy() for k, v in out.items()})

    @torch.no_grad()
    def decompose(self, feats, zs_pred, labels, self_slot):
        """The update split into: own entry / rest of the true-class queue / rest of the zero-shot queue (if wrong)."""
        n_queues = self.stats['nonempty_queues'][-1]
        true_rest, own_a = self.queue_part(feats, labels, zs_pred, self_slot, n_queues)
        zs_rest, own_b = self.queue_part(feats, zs_pred, zs_pred, self_slot, n_queues)
        wrong = (zs_pred != labels).float().unsqueeze(1)
        # own_b holds the own-entry term for every image (it sits in the zero-shot queue); own_a duplicates it
        # when zero-shot is right, since the two queues are then the same one
        return {'part_self': own_b, 'part_true': true_rest, 'part_zs': zs_rest * wrong}

    # ------------------------------------------------------------------ forward = Ramen + probe
    def forward(self, x, labels=None):
        if labels is None:
            raise ValueError('RamenGradientProbe needs the true labels: run it through the harness (records_labels)')
        labels = labels.to(self.device)

        # 1. Zero-shot features, logits, each image's own entropy gradient (Ramen); keep the graph for step 2
        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        zs_pred = logits.argmax(-1)
        self.loss_fn(logits).backward(retain_graph=True)
        own = self.model.get_by_sample_grad()
        self.stats['zs_wrong'].append(int((zs_pred != labels).sum()))

        # 2. ORACLE: the true-label CE gradient at the same (unadapted) parameters; measurement only
        ce = self.oracle_ce_gradient(logits, labels)

        # 3. Ramen: write (zero-shot filing), retrieve the update
        self_slot = self.write(feats, own, logits, zs_pred)
        update = self.retrieve(feats, own)

        # 4. Measure, then step along the update exactly as Ramen does
        self.record(update, ce, own, self.decompose(feats, zs_pred, labels, self_slot), logits.detach(), zs_pred, labels)
        self.stats['self_stored'].append(int((self_slot >= 0).sum()))
        self.model.set_by_sample_grad(update)
        self.model.step_and_zero_grad()
        with torch.no_grad():
            out = self.model(x)
        self.model.reset_parameters()
        return out
