"""
Label-free, streaming embedding correctors: does a per-cluster mean shift recover Ramen's gain?

Each corrector sees the zero-shot CLIP image embeddings z (L2-normalised, the cached `NoAdapt_feats`) batch by
batch in stream order and returns corrected embeddings; the classifier is then plain zero-shot,
logits = 100 * normalise(z') @ W. No memory, no gradients, no labels (except `OracleGroupShift`, marked below).

    z' = normalise( z - alpha * (mu_g(z) - m_ref) )

mu_g(z): running mean of the embeddings assigned to z's group g so far (z's own batch included, as Ramen writes
a batch into memory before retrieving). m_ref: where the group is moved to.
  ref='text'    m_ref = mean of the class text embeddings. Moves each group's centre onto the text cloud's centre.
                With one group this is a global modality-gap correction.
  ref='global'  m_ref = running mean of all embeddings. Only removes between-group differences; with one group
                it does nothing.

    from cluster_shift import KMeansShift, StreamRunner
    z_new = StreamRunner(batch_size=100).apply(KMeansShift(k=20, alpha=1.0, m_ref=W.mean(0)), Z)

Correctors (one class each):
  KMeansShift          groups = online k-means clusters of the embeddings (the method under test)
  RandomGroupShift     control: same group sizes as a KMeansShift run, but images assigned at random
  OracleGroupShift     ORACLE (uses the dataset's domain / corruption id): upper bound for a domain-level story
  NoShift              alpha = 0; must reproduce zero-shot exactly (sanity check)
"""
import numpy as np


def _normalise(z):
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


class GroupMeans:
    """Running (count, sum) of embeddings per group, plus the all-image running mean."""

    def __init__(self, dim):
        self.dim = dim
        self.counts = {}
        self.sums = {}
        self.total_count = 0
        self.total_sum = np.zeros(dim, np.float64)

    def add(self, z, groups):
        for g in np.unique(groups):
            sel = groups == g
            self.counts[g] = self.counts.get(g, 0) + int(sel.sum())
            self.sums[g] = self.sums.get(g, 0.0) + z[sel].sum(0, dtype=np.float64)
        self.total_count += len(z)
        self.total_sum += z.sum(0, dtype=np.float64)

    def mean(self, g):
        return self.sums[g] / self.counts[g]

    def global_mean(self):
        return self.total_sum / self.total_count


class StreamCorrector:
    """Base class: subclasses say how an image gets its group; the shift itself is shared."""

    def __init__(self, alpha=1.0, m_ref=None):
        """m_ref: fixed (D,) reference point, or None for the running global mean."""
        self.alpha = alpha
        self.m_ref = None if m_ref is None else np.asarray(m_ref, np.float64)
        self.means = None
        self.group_log = []   # group id of every image, in stream order

    def assign(self, z, labels):
        """Group id of every image in the batch (may update internal state first)."""
        raise NotImplementedError

    def correct(self, z, domain_ids=None):
        if self.means is None:
            self.means = GroupMeans(z.shape[1])
        groups = self.assign(z, domain_ids)
        self.group_log.append(groups)
        self.means.add(z, groups)
        ref = self.means.global_mean() if self.m_ref is None else self.m_ref
        mu = np.stack([self.means.mean(g) for g in groups])           # (B, D) group mean of each image
        return _normalise(z - self.alpha * (mu - ref)).astype(np.float32)

    @property
    def groups(self):
        return np.concatenate(self.group_log)


class NoShift(StreamCorrector):
    """alpha = 0: the identity. One group, no shift."""

    def __init__(self):
        super().__init__(alpha=0.0)

    def assign(self, z, domain_ids):
        return np.zeros(len(z), int)


class KMeansShift(StreamCorrector):
    """Groups = online (mini-batch) k-means clusters of the embeddings. Label-free."""

    def __init__(self, k, alpha=1.0, m_ref=None, seed=0):
        super().__init__(alpha, m_ref)
        self.k, self.rng = k, np.random.default_rng(seed)
        self.centroids = None
        self.n_seen = np.zeros(k)

    def assign(self, z, domain_ids):
        if self.centroids is None:   # first batch: k distinct images as the initial centroids
            self.centroids = z[self.rng.choice(len(z), self.k, replace=False)].astype(np.float64)
        d = ((z[:, None, :] - self.centroids[None]) ** 2).sum(-1) if self.k <= 8 else \
            (z ** 2).sum(1)[:, None] - 2 * z @ self.centroids.T + (self.centroids ** 2).sum(1)[None]
        a = d.argmin(1)
        for c in np.unique(a):       # mini-batch k-means update: running mean per centroid
            sel = a == c
            n = sel.sum()
            self.centroids[c] = (self.centroids[c] * self.n_seen[c] + z[sel].sum(0)) / (self.n_seen[c] + n)
            self.n_seen[c] += n
        return a


class RandomGroupShift(StreamCorrector):
    """Control: groups drawn at random with the given proportions (a KMeansShift run's cluster sizes)."""

    def __init__(self, proportions, alpha=1.0, m_ref=None, seed=0):
        super().__init__(alpha, m_ref)
        p = np.asarray(proportions, float)
        self.p, self.rng = p / p.sum(), np.random.default_rng(seed)

    def assign(self, z, domain_ids):
        return self.rng.choice(len(self.p), size=len(z), p=self.p)


class OracleGroupShift(StreamCorrector):
    """ORACLE: groups = the dataset's true domain / corruption id (not available to a real TTA method)."""

    def assign(self, z, domain_ids):
        if domain_ids is None:
            raise ValueError('OracleGroupShift needs the domain ids of the batch')
        return np.asarray(domain_ids)


class StreamRunner:
    """Feeds a corrector the stream in batches of `batch_size` (Ramen's batch size) and collects the output."""

    def __init__(self, batch_size=100):
        self.batch_size = batch_size

    def apply(self, corrector, Z, domain_ids=None):
        Z = np.asarray(Z, np.float32)
        out = np.empty_like(Z)
        for s in range(0, len(Z), self.batch_size):
            e = s + self.batch_size
            out[s:e] = corrector.correct(Z[s:e], None if domain_ids is None else domain_ids[s:e])
        return out
