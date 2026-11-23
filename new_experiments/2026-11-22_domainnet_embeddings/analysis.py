"""
Pure analysis over DomainNetEmbeddings (no GPU, no model; projections are cached in results/).

    d = DomainNetEmbeddings()
    proj = Projection(d).fit(stratified_sample(d, per_domain=1500))          # t-SNE of the raw embedding
    Geometry(d).variance_shares(); Geometry(d).centroid_similarity()        # domain vs class structure
    Neighbours(d).composition()                                             # are 10-NN same class or same domain?
    TextAlignment(d).per_domain()                                           # image-text cosine, margin, modality gap
"""
import hashlib
import os

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'results')


# ---------------------------------------------------------------------------- sampling
def stratified_sample(d, per_domain=1500, classes=None, per_cell=None, seed=0):
    """Row indices: `per_domain` random images per domain (balanced, so quickdraw/real don't dominate), or with
    `classes` (names or ids) and `per_cell`, up to `per_cell` images per (class, domain)."""
    rng = np.random.default_rng(seed)
    if classes is None:
        idx = [rng.choice(np.flatnonzero(d.dom == k), min(per_domain, int((d.dom == k).sum())), replace=False)
               for k in range(d.n_dom)]
    else:
        ids = [d.classes.index(c) if isinstance(c, str) else int(c) for c in classes]
        idx = []
        for c in ids:
            for k in range(d.n_dom):
                pool = np.flatnonzero((d.y == c) & (d.dom == k))
                idx.append(rng.choice(pool, min(per_cell, len(pool)), replace=False))
    return np.sort(np.concatenate(idx))


def domain_centred(d, Z=None):
    """z - mu_d (domain mean removed): what is left once each domain's average offset is subtracted."""
    Z = d.Z if Z is None else Z
    return Z - Geometry(d).domain_means()[d.dom]


# ---------------------------------------------------------------------------- 2-D projection
class Projection:
    """PCA(50) -> t-SNE (or PCA(2)) of a subset of rows; returns a DataFrame with x, y and per-row metadata.
    t-SNE keeps local neighbourhoods, not distances between clusters; read cluster gaps from the Geometry tables."""

    def __init__(self, d, method='tsne', centre_domains=False, perplexity=30, seed=0, n_jobs=16):
        self.d, self.method, self.centre = d, method, centre_domains
        self.perplexity, self.seed, self.n_jobs = perplexity, seed, n_jobs

    def _cache_path(self, idx):
        h = hashlib.md5(idx.tobytes()).hexdigest()[:10]
        return os.path.join(CACHE, f'proj_{self.method}_c{int(self.centre)}_p{self.perplexity}_s{self.seed}_{h}.pkl')

    def fit(self, idx, use_cache=True):
        path = self._cache_path(idx)
        if use_cache and os.path.exists(path):
            return pd.read_pickle(path)
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        Z = (domain_centred(self.d) if self.centre else self.d.Z)[idx]
        if self.method == 'pca':
            xy = PCA(2, random_state=self.seed).fit_transform(Z)
        else:
            Z50 = PCA(50, random_state=self.seed).fit_transform(Z)
            xy = TSNE(2, perplexity=self.perplexity, init='pca', learning_rate='auto', random_state=self.seed,
                      n_jobs=self.n_jobs).fit_transform(Z50)
        d = self.d
        out = pd.DataFrame(dict(row=idx, x=xy[:, 0], y=xy[:, 1], domain=np.array(d.domains)[d.dom[idx]],
                                label=d.y[idx], true_name=np.array(d.classes)[d.y[idx]],
                                pred_name=np.array(d.classes)[d.pred[idx]], correct=d.correct[idx]))
        out.to_pickle(path)
        return out


# ---------------------------------------------------------------------------- global geometry
class Geometry:
    """Domain and class structure of the frozen embedding: variance shares, centroids, cross-domain class agreement."""

    def __init__(self, d):
        self.d = d

    def _group_sums(self, keys, n_groups):
        S = torch.zeros(n_groups, self.d.Z.shape[1], dtype=torch.float64)
        S.index_add_(0, torch.from_numpy(keys), torch.from_numpy(self.d.Z).double())
        return S.numpy(), np.bincount(keys, minlength=n_groups)

    def domain_means(self):
        S, n = self._group_sums(self.d.dom, self.d.n_dom)
        return (S / n[:, None]).astype(np.float32)

    def cell_means(self):
        """(C, n_dom, D) mean feature of each (class, domain) cell, and the (C, n_dom) counts."""
        d = self.d
        S, n = self._group_sums(d.y * d.n_dom + d.dom, d.C * d.n_dom)
        M = S / np.maximum(n, 1)[:, None]
        return M.reshape(d.C, d.n_dom, -1).astype(np.float32), n.reshape(d.C, d.n_dom)

    def variance_shares(self):
        """Share of the total variance of z (sum over dimensions) explained by group means (eta squared).
        Domain + class can exceed 'class x domain cell' only if the two overlap; cell = upper bound of both."""
        d, Z = self.d, self.d.Z.astype(np.float64)
        grand = Z.mean(0)
        total = float(((Z - grand) ** 2).sum())
        rows = []
        for name, keys, k in [('domain', d.dom, d.n_dom), ('class', d.y, d.C),
                              ('class x domain cell', d.y * d.n_dom + d.dom, d.C * d.n_dom)]:
            S, n = self._group_sums(keys, k)
            m = S[n > 0] / n[n > 0, None]
            between = float((n[n > 0, None] * (m - grand) ** 2).sum())
            rows.append(dict(grouping=name, groups=int((n > 0).sum()), share_of_variance=100 * between / total))
        return pd.DataFrame(rows).set_index('grouping').round(2)

    def concentration(self):
        """Per domain: length of the mean unit vector (1 = all images point the same way, 0 = spread uniformly)
        and how far the domain mean sits from the domain-balanced grand mean."""
        mu = self.domain_means()
        g = mu.mean(0)
        return pd.DataFrame(dict(mean_resultant_length=np.linalg.norm(mu, axis=1),
                                 offset_from_grand_mean=np.linalg.norm(mu - g, axis=1)),
                            index=self.d.domains).round(3)

    def centroid_similarity(self, centred=True):
        """cos between domain means; centred = after subtracting the domain-balanced grand mean (raw cosines are all
        near 1 because every CLIP image embedding lives in the same narrow cone)."""
        mu = self.domain_means()
        if centred:
            mu = mu - mu.mean(0)
        mu = mu / np.linalg.norm(mu, axis=1, keepdims=True)
        return pd.DataFrame(mu @ mu.T, index=self.d.domains, columns=self.d.domains)

    def class_agreement(self, min_count=20):
        """Mean over classes of cos(m_{c,a} - mu_a, m_{c,b} - mu_b): does a class point the same way in two domains
        once each domain's mean is removed? Cells with fewer than `min_count` images are skipped."""
        M, n = self.cell_means()
        mu = self.domain_means()
        V = M - mu[None]
        V /= np.linalg.norm(V, axis=2, keepdims=True) + 1e-12
        ok = n >= min_count
        out = np.zeros((self.d.n_dom, self.d.n_dom))
        for a in range(self.d.n_dom):
            for b in range(self.d.n_dom):
                both = ok[:, a] & ok[:, b]
                out[a, b] = (V[both, a] * V[both, b]).sum(1).mean()
        return pd.DataFrame(out, index=self.d.domains, columns=self.d.domains)

    def centroid_zero_shot(self, min_count=20):
        """Per domain: share of classes whose (class, domain) mean feature is zero-shot classified as its own class,
        next to the per-image zero-shot accuracy. A high centroid accuracy with a low image accuracy means the class
        direction is right on average but individual images scatter (noise); a low one means the whole class moved."""
        d = self.d
        M, n = self.cell_means()
        rows = []
        for k, name in enumerate(d.domains):
            ok = n[:, k] >= min_count
            hit = int((np.argmax(M[ok, k] @ d.W.T, 1) == np.flatnonzero(ok)).sum())
            m = d.dom == k
            rows.append(dict(domain=name, classes=int(ok.sum()), centroid_correct=hit,
                             centroid_acc=100 * hit / ok.sum(), image_acc=100 * d.correct[m].mean()))
        return pd.DataFrame(rows).set_index('domain').round(2)

    def class_domain_accuracy(self):
        """(C, n_dom) zero-shot accuracy of every (class, domain) cell, with counts."""
        d = self.d
        key = d.y * d.n_dom + d.dom
        n = np.bincount(key, minlength=d.C * d.n_dom)
        hit = np.bincount(key, weights=d.correct, minlength=d.C * d.n_dom)
        acc = pd.DataFrame((100 * hit / np.maximum(n, 1)).reshape(d.C, d.n_dom), index=d.classes, columns=d.domains)
        return acc, pd.DataFrame(n.reshape(d.C, d.n_dom), index=d.classes, columns=d.domains)


# ---------------------------------------------------------------------------- domain probe
def domain_probe(d, n_train=5000, n_test=2000, seed=0):
    """Logistic regression from z to the domain id on balanced train/test splits. Accuracy far above chance (1/6)
    means the domain (style) is linearly readable from the zero-shot embedding."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for k in range(d.n_dom):
        pool = rng.permutation(np.flatnonzero(d.dom == k))
        tr.append(pool[:n_train])
        te.append(pool[n_train:n_train + n_test])
    tr, te = np.concatenate(tr), np.concatenate(te)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(d.Z[tr], d.dom[tr])
    pred = clf.predict(d.Z[te])
    cm = confusion_matrix(d.dom[te], pred, labels=range(d.n_dom))
    acc = float((pred == d.dom[te]).mean())
    return acc, pd.DataFrame(cm, index=d.domains, columns=d.domains)


# ---------------------------------------------------------------------------- nearest neighbours
class Neighbours:
    """k nearest neighbours (cosine, over all 596,006 images, self excluded) of a balanced set of query images."""

    def __init__(self, d, k=10, per_domain=1000, seed=0, threads=16):
        self.d, self.k = d, k
        self.queries = stratified_sample(d, per_domain=per_domain, seed=seed)
        torch.set_num_threads(threads)
        self.nn = self._search()

    def _search(self, chunk=500):
        Z = torch.from_numpy(self.d.Z)
        out = []
        for s in range(0, len(self.queries), chunk):
            q = self.queries[s:s + chunk]
            sim = torch.from_numpy(self.d.Z[q]) @ Z.T
            sim[torch.arange(len(q)), torch.from_numpy(q)] = -2  # drop the query itself
            out.append(sim.topk(self.k, dim=1).indices.numpy())
        return np.concatenate(out)

    def composition(self):
        """Per query domain: share of neighbours in each (same/other class) x (same/other domain) group, the chance
        rate of 'same domain' (that domain's share of the dataset) and the kNN-vote accuracy.
        ORACLE: the vote uses the neighbours' true labels; it measures class information in the image geometry."""
        d, q, nn = self.d, self.queries, self.nn
        same_c = d.y[nn] == d.y[q][:, None]
        same_d = d.dom[nn] == d.dom[q][:, None]
        votes = np.array([np.bincount(d.y[r], minlength=d.C).argmax() for r in nn])
        rows = []
        for k, name in enumerate(d.domains):
            m = d.dom[q] == k
            tot = m.sum() * self.k
            rows.append({'domain': name, 'queries': int(m.sum()),
                         'same class, same domain': 100 * (same_c & same_d)[m].sum() / tot,
                         'same class, other domain': 100 * (same_c & ~same_d)[m].sum() / tot,
                         'other class, same domain': 100 * (~same_c & same_d)[m].sum() / tot,
                         'other class, other domain': 100 * (~same_c & ~same_d)[m].sum() / tot,
                         'same-domain chance': 100 * (d.dom == k).mean(),
                         'zero-shot acc (queries)': 100 * d.correct[q][m].mean(),
                         'kNN-vote acc (oracle)': 100 * (votes[m] == d.y[q][m]).mean()})
        return pd.DataFrame(rows).set_index('domain').round(2)

    def domain_mixing(self):
        """(query domain, neighbour domain) share of neighbours, rows sum to 100."""
        d = self.d
        M = np.zeros((d.n_dom, d.n_dom))
        np.add.at(M, (np.repeat(d.dom[self.queries], self.k), d.dom[self.nn].ravel()), 1)
        return pd.DataFrame(100 * M / M.sum(1, keepdims=True), index=d.domains, columns=d.domains)


# ---------------------------------------------------------------------------- image-text alignment
class TextAlignment:
    """Cosine between image and text embeddings: true class, best wrong class, margin, and the modality gap."""

    def __init__(self, d, chunk=65536):
        self.d = d
        s_true, s_wrong = [], []
        for s in range(0, len(d), chunk):
            S = d.Z[s:s + chunk] @ d.W.T
            y = d.y[s:s + chunk]
            t = S[np.arange(len(y)), y]
            S[np.arange(len(y)), y] = -np.inf
            s_true.append(t)
            s_wrong.append(S.max(1))
        self.s_true, self.s_wrong = np.concatenate(s_true), np.concatenate(s_wrong)
        self.margin = self.s_true - self.s_wrong  # > 0 <=> zero-shot correct (up to fp16 ties)

    def per_domain(self):
        d = self.d
        mu_w = d.W.mean(0)
        mu = Geometry(d).domain_means()
        rows = []
        for k, name in enumerate(d.domains):
            m = d.dom == k
            rows.append(dict(domain=name, images=int(m.sum()),
                             cos_true_text=self.s_true[m].mean(), cos_best_wrong_text=self.s_wrong[m].mean(),
                             mean_margin=self.margin[m].mean(),
                             share_margin_below_0=100 * (self.margin[m] < 0).mean(),
                             cos_domain_mean_to_text_mean=float(mu[k] @ mu_w / np.linalg.norm(mu[k]) /
                                                                np.linalg.norm(mu_w))))
        return pd.DataFrame(rows).set_index('domain').round(4)

    def frame(self, idx):
        d = self.d
        return pd.DataFrame(dict(domain=np.array(d.domains)[d.dom[idx]], s_true=self.s_true[idx],
                                 s_wrong=self.s_wrong[idx], margin=self.margin[idx], correct=d.correct[idx]))


# ---------------------------------------------------------------------------- confusions per domain
def top_confusions(d, domain, n=10):
    """Most frequent (true -> zero-shot prediction) pairs in one domain, with counts."""
    k = d.domains.index(domain)
    m = (d.dom == k) & ~d.correct
    pairs = pd.Series(list(zip(np.array(d.classes)[d.y[m]], np.array(d.classes)[d.pred[m]]))).value_counts()
    sink = pd.Series(np.array(d.classes)[d.pred[d.dom == k]]).value_counts()
    out = pairs.head(n).rename('count').to_frame()
    out['share_of_domain_errors'] = (100 * out['count'] / m.sum()).round(2)
    return out, sink.head(n).rename('predicted_count')
