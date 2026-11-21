"""
Ramen whose memory is filed by the TRUE label: an ORACLE (upper bound), not a method.

Ramen writes every test image, before retrieval, into the queue of its zero-shot prediction. A zero-shot error
therefore lands in the wrong class queue. This variant uses the true label to choose the queue, so every queue
holds only images of its own class. Everything else is Ramen's: the stored value is still the image's own
entropy-minimisation gradient (it sharpens the zero-shot prediction, right or wrong), retrieval, weighting,
the signSGD step and the reset are unchanged.

Config key `filing` (which class queue a test image is written to):
  zero_shot   its zero-shot prediction: Ramen exactly (sanity check: 100% the same predictions)
  true_label  its true label (uses ground truth)

To try another filing rule, add a name to FILINGS and a branch in `queue_of`; nothing else needs to change.

Per-batch log (self.stats):
  zs_wrong         images in the batch whose zero-shot prediction is wrong (= images filed off their
                   zero-shot class under true_label filing)
  nonempty_queues  class queues holding at least one entry after this batch's write (= queues averaged over)
"""
import torch

from .losses import softmax_entropy
from .Ramen import Ramen

FILINGS = ('zero_shot', 'true_label')


class RamenTrueLabel(Ramen):

    records_labels = True  # the harness passes the true labels to forward(x, labels=...)

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.filing = self.cfg.get('filing', 'zero_shot')
        if self.filing not in FILINGS:
            raise ValueError(f'Unknown filing {self.filing!r}; choose from {list(FILINGS)}')
        self.stats = {'filing': self.filing, 'zs_wrong': [], 'nonempty_queues': []}

    def queue_of(self, zs_pred, labels):
        """The class queue each image is written to."""
        if self.filing == 'true_label':
            return labels
        return zs_pred

    @torch.no_grad()
    def write(self, feats, grads, logits, queues):
        """Ramen's cache update, with the queue of each image given by `queues`."""
        B = feats.shape[0]
        priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.priority_dtype)
        entropies = softmax_entropy(logits, reduction='none')
        self.counter = self.counter + B
        for b in range(B):
            self.cache[queues[b]].add(keys=feats[b].unsqueeze(0), values=grads[b].unsqueeze(0),
                                      entropies=entropies[b].unsqueeze(0), priorities=priorities[b].unsqueeze(0))

    @torch.no_grad()
    def retrieve(self, feats, grads):
        """Ramen's retrieval: top-k per non-empty queue, weighted by exp(-H) * exp(-beta * d), queues averaged."""
        retrieved_grads_sum = torch.zeros_like(grads)
        num_class_sum = 0
        for c in range(self.num_classes):
            if self.cache[c].size == 0:
                continue
            retrieved_grads, _, entropies, sorted_dist = self.cache[c].query(feats, topk=self.cfg['topk'])
            weights = torch.exp(-entropies) * torch.exp(-self.beta * sorted_dist)
            retrieved_grads_sum += (retrieved_grads * weights.unsqueeze(dim=2)).sum(dim=1)
            num_class_sum += 1
        self.stats['nonempty_queues'].append(num_class_sum)
        return retrieved_grads_sum / num_class_sum

    def forward(self, x, labels=None):
        if labels is None:
            raise ValueError('RamenTrueLabel needs the true labels: run it through the harness (records_labels)')
        labels = labels.to(self.device)

        # 1. Zero-shot features, logits and each image's own entropy gradient (as in Ramen).
        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        zs_pred = logits.argmax(-1)
        self.loss_fn(logits).backward()
        grads = self.model.get_by_sample_grad()
        self.stats['zs_wrong'].append(int((zs_pred != labels).sum()))

        # 2. Write into the chosen queues, retrieve, one step, predict, reset (as in Ramen).
        self.write(feats, grads, logits, self.queue_of(zs_pred, labels))
        self.model.set_by_sample_grad(self.retrieve(feats, grads))
        self.model.step_and_zero_grad()
        with torch.no_grad():
            logits = self.model(x)
        self.model.reset_parameters()
        return logits
