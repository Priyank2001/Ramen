"""
Accuracy, Shapley attribution and cluster bootstrap over the cells of the factorial (pure functions, no I/O).

    acc = domain_avg(correct, dom)                          # mean of per-domain accuracies, in points
    phi = shapley({frozenset(): a0, frozenset('A'): a1, ...}, players='AGL')
    boot = ClusterBootstrap(dom, cluster).replicates(correct_matrix)   # (B, K) domain-averaged accuracies
"""
from itertools import combinations
from math import factorial

import numpy as np


def domain_avg(correct, dom):
    return 100 * np.mean([correct[dom == d].mean() for d in np.unique(dom)])


def per_domain(correct, dom):
    return {int(d): 100 * correct[dom == d].mean() for d in np.unique(dom)}


def subsets(players):
    return [frozenset(c) for r in range(len(players) + 1) for c in combinations(players, r)]


def shapley(value, players):
    """value: {frozenset(subset): accuracy}. Returns {player: phi}; sum(phi) = v(all) - v(empty)."""
    n = len(players)
    phi = {}
    for k in players:
        others = [p for p in players if p != k]
        tot = 0.0
        for S in subsets(others):
            w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
            tot += w * (value[S | {k}] - value[S])
        phi[k] = tot
    return phi


class ClusterBootstrap:
    """
    Poisson cluster bootstrap of domain-averaged accuracy (weights ~ Poisson(1) per cluster, shared by all columns, so
    differences between cells are paired). Cluster = photo on ImageNet-C, image elsewhere.
    """

    def __init__(self, dom, cluster, B=1000, seed=0, block=100):
        self.dom = dom
        _, self.cl = np.unique(cluster, return_inverse=True)
        self.B, self.seed, self.block = B, seed, block

    def replicates(self, correct):
        """correct: (N, K) bool. Returns (B, K) domain-averaged accuracies (points)."""
        rng = np.random.default_rng(self.seed)
        n_cl, out = self.cl.max() + 1, []
        doms = np.unique(self.dom)
        X = correct.astype(np.float32)
        for b0 in range(0, self.B, self.block):
            nb = min(self.block, self.B - b0)
            w = rng.poisson(1.0, size=(nb, n_cl)).astype(np.float32)
            acc = np.zeros((nb, X.shape[1]))
            for d in doms:
                m = self.dom == d
                wd = w[:, self.cl[m]]                        # (nb, n_d)
                acc += (wd @ X[m]) / np.maximum(wd.sum(1, keepdims=True), 1)
            out.append(100 * acc / len(doms))
        return np.concatenate(out)


def ci(samples, level=0.95):
    lo, hi = np.percentile(samples, [50 * (1 - level), 50 * (1 + level)], axis=0)
    return lo, hi
