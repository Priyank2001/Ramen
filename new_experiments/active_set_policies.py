"""
Three active-set policies for Ramen, side by side (src/methods/RamenPolicy.py, notebook active_set_policies.ipynb):

  1. global_knn  top-k: the k nearest images of the whole memory, whatever class queue they sit in
  2. per_class   n images from each class queue (the n nearest); n=1 is Ramen
  3. zs_prob     Ramen's active set with each class queue weighted by the query's zero-shot probability

Run (detached, one shard per GPU; results are cached in results/ like any exp.run()):
    CUDA_VISIBLE_DEVICES=0 python active_set_policies.py run ImageNetC5K 0 3     # shard 0 of 3
Analysis: the notebook. Everything below `run` works on CPU from the cached runs.

ActiveSets rebuilds any query's active set under any policy offline: the memory is the same for every policy
(writes do not depend on the retrieval), so CacheReplay (ramen_examples.py) replays it from the `filed_as` log and the
entries are scored with the NoAdapt features / zero-shot probabilities, exactly as RamenPolicy does on the GPU.
validate() compares that against each run's logged top-8 entries.
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ramen_analysis import make_experiment, BLUE, ORANGE, RED, AQUA, MUTED, TEXT_2
from ramen_examples import RamenExamples, _short

NUM_WORKERS = {'ImageNetC5K': 16, 'DomainNet': 16, 'CIFAR100C': 4, 'CIFAR10C': 4}  # = the cached Ramen runs

# name -> (policy family, knob value, RamenPolicy config)
VARIANTS = {
    'policy_global_k1':    ('global_knn', 1, {'policy': 'global_knn', 'global_k': 1}),
    'policy_global_k10':   ('global_knn', 10, {'policy': 'global_knn', 'global_k': 10}),
    'policy_global_k100':  ('global_knn', 100, {'policy': 'global_knn', 'global_k': 100}),
    'policy_global_k1000': ('global_knn', 1000, {'policy': 'global_knn', 'global_k': 1000}),
    'policy_global_k5000': ('global_knn', 5000, {'policy': 'global_knn', 'global_k': 5000}),
    'policy_class_n1':     ('per_class', 1, {'policy': 'per_class', 'per_class_n': 1}),
    'policy_class_n3':     ('per_class', 3, {'policy': 'per_class', 'per_class_n': 3}),
    'policy_class_n10':    ('per_class', 10, {'policy': 'per_class', 'per_class_n': 10}),
    'policy_class_n75':    ('per_class', 75, {'policy': 'per_class', 'per_class_n': 75}),
    'policy_zs_T1':        ('zs_prob', 1, {'policy': 'zs_prob', 'temperature': 1.0}),
    'policy_zs_T3':        ('zs_prob', 3, {'policy': 'zs_prob', 'temperature': 3.0}),
    'policy_zs_T10':       ('zs_prob', 10, {'policy': 'zs_prob', 'temperature': 10.0}),
}
FAMILY_LABEL = {'global_knn': '1. top-k nearest images (whole memory)',
                'per_class': '2. n images from each class',
                'zs_prob': '3. zero-shot probability weighted'}
KNOB = {'global_knn': 'k', 'per_class': 'n', 'zs_prob': 'T'}


def experiment(dataset, seed=0, **kw):
    return make_experiment(dataset, tta_mode='mixed', batch_size=100, n_per_domain=None, seed=seed,
                           num_workers=NUM_WORKERS[dataset], **kw)


def run(dataset, shard, n_shards, seed=0):
    exp = experiment(dataset, seed).load()
    for i, (name, (_, _, cfg)) in enumerate(VARIANTS.items()):
        if i % n_shards == shard:
            exp.run('RamenPolicy', name=name, cfg=cfg)
            print(f'{name} done', flush=True)


# ---------------------------------------------------------------------------- outcomes
def outcome_table(exp, names):
    """One row per run: accuracy (mean over domains), fixed / broken vs zero-shot, agreement with Ramen,
    size of the active set and how much of the update comes from the query itself."""
    base, ramen = exp.results['NoAdapt'], exp.results['Ramen']
    ramen_acc = ramen.groupby('domain_idx').correct.mean().mean() * 100
    rows = {}
    for name in ['NoAdapt', 'Ramen', *names]:
        r = exp.results[name]
        assert (r.sample_idx.values == base.sample_idx.values).all(), f'{name}: different stream order'
        avg = r.groupby('domain_idx').correct.mean().mean() * 100
        row = {'policy': FAMILY_LABEL.get(VARIANTS[name][0], '') if name in VARIANTS else 'reference',
               'knob': f'{KNOB[VARIANTS[name][0]]}={VARIANTS[name][1]}' if name in VARIANTS else '',
               'accuracy %': avg, 'vs Ramen': avg - ramen_acc,
               'fixed %': (~base.correct.values & r.correct.values).mean() * 100,
               'broken %': (base.correct.values & ~r.correct.values).mean() * 100,
               'same pred as Ramen %': (r.pred.values == ramen.pred.values).mean() * 100}
        st = exp.method_stats.get(name)
        if st is not None and 'n_active' in st:
            ids, w = np.concatenate(st['support_ids']), np.concatenate(st['support_weight'])
            row['active entries (mean)'] = np.concatenate(st['n_active']).mean()
            # support_weight holds the top-8 entries' share of the WHOLE update, so this is exact for the query
            row['self weight %'] = (w * (ids == np.arange(len(ids))[:, None])).sum(1).mean() * 100
        rows[name] = row
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------- offline active sets
class ActiveSets(RamenExamples):
    """RamenExamples (memory replay, plots) + the active set of any policy for any query."""

    def __init__(self, exp, filed_from='policy_class_n1', **kw):
        super().__init__(exp.dataset_name, exp.model_name.replace('clip_', ''), filed_from, seed=exp.seed, **kw)
        self.runs = exp.results

    def policy_active_set(self, position, snap, policy, knob):
        """Entries with non-zero weight for this query under `policy` (as RamenPolicy.entry_weights), weights
        normalised to sum 1, largest first. Columns: id, filed, dist, H, weight, true, domain_idx."""
        beta = self.cfg.get('beta', 0.0)
        q = np.asarray(self.feats[position], dtype=np.float32)
        ids = np.concatenate([np.sort(s) for s in snap if len(s)])
        filed = np.concatenate([np.full(len(s), c) for c, s in enumerate(snap) if len(s)])
        d = np.linalg.norm(np.asarray(self.feats[ids], dtype=np.float32) - q, axis=1)
        a = pd.DataFrame({'id': ids, 'filed': filed, 'dist': d})
        if policy == 'global_knn':
            a = a.iloc[np.argsort(d, kind='stable')[:knob]]
        else:
            n = 1 if policy == 'zs_prob' else knob
            a = a.sort_values('dist', kind='stable').groupby('filed', sort=False).head(n)
        a = a.copy()
        a['H'] = self._entropy(a.id.values)
        a['weight'] = np.exp(-a.H) * np.exp(-beta * a.dist)
        if policy == 'zs_prob':
            p = np.asarray(self.p_zs[position], dtype=np.float64) ** (1 / knob)  # softmax(logits / T)
            a['weight'] *= p[a.filed.values] / p.sum()
        a['weight'] /= a.weight.sum()
        a['true'] = self.label[a.id.values]
        a['domain_idx'] = self.zs.domain_idx.values[a.id.values]
        return a.sort_values('weight', ascending=False, ignore_index=True)

    def validate(self, names, n=100, seed=0):
        """Offline active set vs the top-8 entries each GPU run logged, at n random positions."""
        rng = np.random.default_rng(seed)
        pos = np.sort(rng.choice(np.arange(int(0.2 * len(self.df)), len(self.df)), n, replace=False))
        snaps = self.replay.snapshots(list(pos))
        rows = {}
        for name in names:
            family, knob, _ = VARIANTS[name]
            st = pd.read_pickle(self._stats_path(name))
            sup_ids, sup_w = np.concatenate(st['support_ids']), np.concatenate(st['support_weight'])
            top1, overlap = [], []
            for p in pos:
                a = self.policy_active_set(p, snaps[p], family, knob)
                logged = sup_ids[p][sup_ids[p] >= 0]
                top1.append(logged[0] == a.id.values[0])
                overlap.append(len(set(logged) & set(a.id.values[:len(logged)])) / len(logged))
            rows[name] = {'top-1 entry identical %': np.mean(top1) * 100, 'top-8 overlap %': np.mean(overlap) * 100}
        return pd.DataFrame(rows).T.round(1)

    def _stats_path(self, name):
        return self.exp._cache_path(name)[:-len('.pkl')] + '_stats.pkl'

    # ------------------------------------------------------------------ confused test cases
    def confused_cases(self, names, min_position=None):
        """
        Test samples CLIP is confused on: zero-shot top-1 is wrong but the true class is its runner-up, late enough
        in the stream for the memory to be warm. Adds each run's prediction and correctness.
        """
        min_position = int(0.2 * len(self.df)) if min_position is None else min_position
        if not hasattr(self, '_runner_up'):
            p = np.asarray(self.p_zs, dtype=np.float32).copy()
            p[np.arange(len(p)), p.argmax(1)] = -1
            self._runner_up = p.argmax(1)
        d = self.df[(self.df.position >= min_position) & (self.df.zs_pred != self.df.label)
                    & (self._runner_up == self.df.label.values) & (self.df.filed_as == self.df.zs_pred)].copy()
        for name in names:
            d[name] = self.runs[name].pred.values[d.position.values] == d.label.values
        d['n right'] = d[names].sum(axis=1)
        return d

    def show_policies(self, position, names, n_show=8, snap=None):
        """
        One confused query under every policy:
          row 1  the query, its zero-shot probabilities, and what each policy predicted
          row 2+ per policy: the highest-weight entries of its active set (share of the update, class filed under,
                 green frame = the image's true class is the query's true class, red = the query's wrong zero-shot
                 class), and a summary of the whole active set on the left
        """
        snap = snap if snap is not None else self.replay.snapshots([position])[position]
        r = self.df.iloc[position]
        y, zs = int(r.label), int(r.zs_pred)
        cls = self.classes
        pz = np.asarray(self.p_zs[position])

        fig = plt.figure(figsize=(1.55 * (n_show + 2), 3.5 + 1.85 * len(names)))
        # row 1 is an empty spacer that keeps the rotated class names clear of the tiles below
        gs = GridSpec(2 + len(names), n_show + 2, figure=fig, height_ratios=[1.5, 0.3] + [1] * len(names),
                      hspace=0.62, wspace=0.12)

        ax = fig.add_subplot(gs[0, 0])
        self._tile(ax, position, '')
        ax.set_title(f'QUERY #{position}\n{r.domain}\ntrue: {_short(cls[y], 22)}', fontsize=8)

        ax = fig.add_subplot(gs[0, 2:2 + n_show // 2])
        shown = [y] + [c for c in np.argsort(-pz) if c != y][:5]
        x = np.arange(len(shown))
        ax.bar(x, pz[shown], color=[AQUA if c == y else ORANGE if c == zs else MUTED for c in shown], width=0.6)
        for xi, v in zip(x, pz[shown]):
            ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', fontsize=7, color=TEXT_2)
        ax.set_xticks(x, [_short(cls[c], 16) for c in shown], rotation=35, ha='right', fontsize=7)
        ax.set_ylim(0, pz[shown].max() * 1.3)
        ax.grid(axis='x', visible=False)
        ax.set_title('zero-shot probabilities (green = true, orange = zero-shot pick)', fontsize=8)

        ax = fig.add_subplot(gs[0, 2 + n_show // 2 + 1:])
        ax.set_axis_off()
        lines = [f'{"Ramen":<22} {_short(cls[int(self.runs["Ramen"].pred.values[position])], 22)}']
        for name in names:
            pred = int(self.runs[name].pred.values[position])
            lines.append(f'{self._label(name):<22} {_short(cls[pred], 22)}')
        ax.text(0, 1, 'prediction after adaptation\n' + '\n'.join(lines), va='top', fontsize=7.5, family='monospace',
                color=TEXT_2)

        for i, name in enumerate(names):
            family, knob, _ = VARIANTS[name]
            a = self.policy_active_set(position, snap, family, knob)
            pred = int(self.runs[name].pred.values[position])
            ok = pred == y
            ax = fig.add_subplot(gs[2 + i, 0:2])
            ax.set_axis_off()
            per_class = a.groupby('filed').weight.sum()
            ax.text(0, 0.5,
                    f'{self._label(name)}\n'
                    f'-> {_short(cls[pred], 20)}  {"RIGHT" if ok else "WRONG"}\n'
                    f'{len(a):,} entries, {per_class.size} queues\n'
                    f'true class y: {a.weight[a.true == y].sum() * 100:.1f}%\n'
                    f'filed as y: {per_class.get(y, 0) * 100:.1f}% | as zs: {per_class.get(zs, 0) * 100:.1f}%\n'
                    f'self: {a.weight[a.id == position].sum() * 100:.1f}%',
                    va='center', fontsize=7.5, color=AQUA if ok else RED, family='monospace')
            for j, e in enumerate(a.head(n_show).itertuples()):
                ax = fig.add_subplot(gs[2 + i, 2 + j])
                frame = AQUA if e.true == y else RED if e.true == zs else None
                tag = ' self' if e.id == position else ''
                self._tile(ax, int(e.id), f'{_short(cls[e.filed], 15)}\n{e.weight * 100:.1f}%{tag}', frame)
        fig.suptitle(f'{self.dataset} | zero-shot confuses {cls[y]!r} with {cls[zs]!r} | tiles: class filed under, '
                     f'share of the update | frame: green = true class of query, red = zero-shot class',
                     fontsize=9.5, x=0.01, ha='left', y=0.995)
        return fig

    @staticmethod
    def _label(name):
        family, knob, _ = VARIANTS[name]
        return {'global_knn': f'top-k  k={knob}', 'per_class': f'per-class  n={knob}',
                'zs_prob': f'zero-shot  T={knob}'}[family]

    def active_set_profile(self, positions, names):
        """Averages over many queries of what each policy's active set is made of."""
        snaps = self.replay.snapshots(list(positions))
        rows = []
        for p in positions:
            r = self.df.iloc[p]
            y, zs = int(r.label), int(r.zs_pred)
            for name in names:
                family, knob, _ = VARIANTS[name]
                a = self.policy_active_set(p, snaps[p], family, knob)
                per_class = a.groupby('filed').weight.sum()
                rows.append({'run': name, 'entries': len(a), 'queues': per_class.size,
                             'weight on true class images %': a.weight[a.true == y].sum() * 100,
                             'weight filed as true class %': per_class.get(y, 0) * 100,
                             'weight filed as zero-shot class %': per_class.get(zs, 0) * 100,
                             'self weight %': a.weight[a.id == p].sum() * 100,
                             'weight from same corruption %': a.weight[a.domain_idx == r.domain_idx].sum() * 100,
                             'right': self.runs[name].pred.values[p] == y})
        t = pd.DataFrame(rows).groupby('run', sort=False).mean()
        t['right'] *= 100
        return t.rename(columns={'right': 'accuracy on these %'}).round(2)


if __name__ == '__main__':
    mode, dataset = sys.argv[1], sys.argv[2]
    if mode == 'run':
        run(dataset, int(sys.argv[3]), int(sys.argv[4]))
