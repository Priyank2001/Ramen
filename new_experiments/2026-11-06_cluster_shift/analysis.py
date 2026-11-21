"""
How much of Ramen's own embedding shift is a group-constant move? (offline, no GPU)

Ramen moves each image's embedding from z_before (NoAdapt_feats) to z_after (Ramen_feats). Delta_i = z_after - z_before.
For a grouping g(i) the best group-constant approximation is the group's mean shift. Share of the squared shift it
explains (uncentred, and after removing the single all-image mean shift):

    explained(g)          = 1 - sum_i ||Delta_i - mean_{g(i)} Delta||^2 / sum_i ||Delta_i||^2
    explained_beyond_mean = 1 - sum_i ||Delta_i - mean_{g(i)} Delta||^2 / sum_i ||Delta_i - mean Delta||^2

Groupings compared: the true domain (ORACLE grouping), online k-means clusters (label-free), and random groups of the
same sizes (control; ~0 beyond the global mean by construction, up to the small-sample bias 1/n_group).

    ShiftExplained(exp).table(['oracle_domain_a1.0_global', 'kmeans_k20_a0.5_global', 'random_k20_a0.5_global'])
"""
import numpy as np
import pandas as pd


class ShiftExplained:
    def __init__(self, experiment):
        self.e = experiment
        self.delta = np.asarray(experiment.Z_ramen, np.float32) - experiment.Z

    def _explained(self, groups):
        d = self.delta
        total = (d.astype(np.float64) ** 2).sum()
        mean_all = d.mean(0, dtype=np.float64)
        total_c = ((d - mean_all) ** 2).sum()
        resid = 0.0
        for g in np.unique(groups):
            x = d[groups == g]
            resid += ((x - x.mean(0, dtype=np.float64)) ** 2).sum()
        return 1 - resid / total, 1 - resid / total_c

    def table(self, names):
        rows = []
        for n in names:
            groups = self.e.domain_ids if n.startswith('oracle') else self.e._groups_of(n)
            if n.startswith('oracle'):
                self.e.run(n)
            a, b = self._explained(groups)
            rows.append(dict(grouping=n, n_groups=len(np.unique(groups)), explained=a, explained_beyond_mean=b))
        return pd.DataFrame(rows)
