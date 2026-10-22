"""
Where Ramen helps and where it hurts, one test sample at a time: zero-shot vs Ramen probabilities, the state of
Ramen's memory at that moment, and the active set (retrieved entries) behind the sample's update.

Everything is rebuilt offline from saved runs, no GPU needed. Ramen's memory never depends on its adapted outputs:
every sample is written, before retrieval, into the queue of its zero-shot class with priority = its stream counter
(fp16, as in Ramen) and its zero-shot entropy. So given
  - the class each sample was filed under (`filed_as`, logged by any RamenActive run; the write rule is the same
    for every selector),
  - the zero-shot features and probabilities (the NoAdapt run, capture_features/capture_probs),
CacheReplay replays the writes and returns the exact queues at any stream position, and the active set Ramen
retrieves from them (top-k nearest entries per non-empty class queue, weight exp(-H_j) * exp(-beta * d_j)).
validate() checks both against what a RamenActive run logged.

Usage (from a notebook in new_experiments/):
    ex = RamenExamples('ImageNetC5K', 'vitbase16', filed_from='active_all')
    ex.domain_table()                       # per-domain zero-shot vs Ramen accuracy, fixed / broken counts
    picks = ex.pick('fixed', domain='contrast', n=3)
    ex.show(picks.position.iloc[0])         # probabilities + memory + active set for one sample
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ramen_analysis import (BLUE, ORANGE, RED, AQUA, MUTED, TEXT_2, GRID, make_experiment)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def _short(name, n=20):
    return name if len(name) <= n else name[:n - 1] + '…'


class CacheReplay:
    """Ramen's per-class PriorityCache, replayed from the write sequence (stream ids instead of gradients)."""

    def __init__(self, filed_as, num_classes, capacity, batch_size):
        self.filed_as, self.C, self.cap, self.B = filed_as, num_classes, capacity, batch_size
        # Ramen: torch.arange(counter, counter + B, dtype=half) -> integers rounded to fp16, inf above 65,519
        with np.errstate(over='ignore'):
            self.priority = np.arange(len(filed_as), dtype=np.float64).astype(np.float16).astype(np.float32)

    def snapshots(self, positions):
        """{position: list of C int arrays (stream ids in each class queue, slot order)} for each requested position,
        taken after the whole batch holding it was written (Ramen writes the batch, then retrieves)."""
        ends = {}
        for p in positions:
            ends.setdefault(min((p // self.B + 1) * self.B, len(self.filed_as)) - 1, []).append(p)
        ids = np.full((self.C, self.cap), -1, dtype=np.int64)
        prio = np.full((self.C, self.cap), -np.inf, dtype=np.float32)
        size = np.zeros(self.C, dtype=np.int64)
        out, last = {}, max(ends)
        for t in range(last + 1):
            c = self.filed_as[t]
            if size[c] < self.cap:
                slot = size[c]
                size[c] += 1
            else:
                slot = int(np.argmin(prio[c]))  # first minimum, as torch.argmin
                if not self.priority[t] > prio[c, slot]:
                    slot = -1
            if slot >= 0:
                ids[c, slot], prio[c, slot] = t, self.priority[t]
            if t in ends:
                snap = [ids[c, :size[c]].copy() for c in range(self.C)]
                for p in ends[t]:
                    out[p] = snap
        return out


class RamenExamples:

    def __init__(self, dataset, model, filed_from, seed=0, load_images=True):
        self.dataset = dataset
        prefix = os.path.join(RESULTS, f'{dataset}_clip_{model}_mixed_nall_bs100_s{seed}_')
        self.zs = pd.read_pickle(prefix + 'NoAdapt.pkl')
        self.ramen = pd.read_pickle(prefix + 'Ramen.pkl')
        self.p_zs = np.load(prefix + 'NoAdapt_probs.npy', mmap_mode='r')
        self.p_ramen = np.load(prefix + 'Ramen_probs.npy', mmap_mode='r')
        self.feats = np.load(prefix + 'NoAdapt_feats.npy', mmap_mode='r')
        self.stats = pd.read_pickle(prefix + filed_from + '_stats.pkl')
        self.filed_as = np.concatenate(self.stats['filed_as']).astype(np.int64)
        for df in (self.ramen,):
            assert (df.sample_idx.values == self.zs.sample_idx.values).all() and \
                   (df.domain_idx.values == self.zs.domain_idx.values).all(), 'runs see different stream orders'

        self.exp = make_experiment(dataset, seed=seed, device='cpu')
        self.cfg = self.exp._load_cfg('Ramen')
        if load_images:
            self.exp.load()  # CLIP on CPU, only for the dataset objects (image paths, class names)
            self.classes = self.exp.classes
        self.C = self.p_zs.shape[1]
        self.replay = CacheReplay(self.filed_as, self.C, self.cfg['max_capacity'], 100)
        self.label = self.zs.label.values
        self.entropy = None  # zero-shot entropy per sample, computed lazily (that is what Ramen's memory stores)

        d = self.zs[['position', 'domain', 'domain_idx', 'sample_idx', 'label']].copy()
        d['zs_pred'], d['ramen_pred'] = self.zs.pred.values, self.ramen.pred.values
        d['zs_true_p'], d['ramen_true_p'] = self.zs.true_prob.values, self.ramen.true_prob.values
        d['filed_as'] = self.filed_as
        d['outcome'] = np.select([(d.zs_pred != d.label) & (d.ramen_pred == d.label),
                                  (d.zs_pred == d.label) & (d.ramen_pred != d.label),
                                  (d.zs_pred == d.label)], ['fixed', 'broken', 'both right'], 'both wrong')
        self.df = d

    # ------------------------------------------------------------------ tables
    def domain_table(self):
        d = self.df.assign(zs_ok=lambda x: x.zs_pred == x.label, ramen_ok=lambda x: x.ramen_pred == x.label,
                           fixed=lambda x: x.outcome == 'fixed', broken=lambda x: x.outcome == 'broken')
        g = d.groupby('domain', observed=True)
        t = pd.DataFrame({'samples': g.size(), 'zero-shot %': g.zs_ok.mean() * 100, 'Ramen %': g.ramen_ok.mean() * 100,
                          'fixed %': g.fixed.mean() * 100, 'broken %': g.broken.mean() * 100})
        t['gain'] = t['Ramen %'] - t['zero-shot %']
        t['fixed / broken'] = t['fixed %'] / t['broken %']
        t = t.sort_values('gain', ascending=False)
        t.loc['mean over domains'] = t.mean()
        t.loc['mean over domains', 'samples'] = len(d)
        return t.round(2)

    def _entropy(self, rows):
        p = np.asarray(self.p_zs[rows], dtype=np.float64)
        return -(p * np.log(np.clip(p, 1e-30, None))).sum(1)

    def pick(self, outcome, domain=None, n=3, min_position=None, seed=0):
        """
        n samples with this outcome ('fixed' / 'broken'), drawn at random (fixed seed, so not hand-picked) among those
        late enough in the stream for the memory to be warm (default: after 20% of the stream), in `domain` if given.
        Only samples whose logged zero-shot class agrees with the NoAdapt run (99.6-99.8%; they differ by fp16
        numerics), so the zero-shot bars and the memory tell the same story.
        """
        d = self.df
        min_position = int(0.2 * len(d)) if min_position is None else min_position
        m = (d.outcome == outcome) & (d.position >= min_position) & (d.filed_as == d.zs_pred)
        if domain is not None:
            m &= d.domain == domain
        return d[m].sample(n, random_state=seed).sort_values('position')

    # ------------------------------------------------------------------ memory + active set
    def active_set(self, position, snap, temperature=None):
        """Ramen's retrieval for one query from a snapshot: per non-empty class queue the top-k nearest entries,
        weight exp(-H_j) * exp(-beta * d_j). Returns a frame of retrieved entries with normalised weights.
        temperature: instead RamenActive 'prob' (each queue also scaled by softmax(zero-shot logits / T)), only used
        by validate()."""
        k, beta = self.cfg['topk'], self.cfg.get('beta', 0.0)
        q = np.asarray(self.feats[position], dtype=np.float32)
        rows = []
        for c, ids in enumerate(snap):
            if len(ids) == 0:
                continue
            dist = np.linalg.norm(np.asarray(self.feats[np.sort(ids)], dtype=np.float32) - q, axis=1)
            ids_sorted = np.sort(ids)
            top = np.argsort(dist, kind='stable')[:k]
            rows.append(pd.DataFrame({'id': ids_sorted[top], 'filed': c, 'dist': dist[top]}))
        a = pd.concat(rows, ignore_index=True)
        a['H'] = self._entropy(a.id.values)
        a['weight'] = np.exp(-a.H) * np.exp(-beta * a.dist)
        if temperature:
            # softmax(logits / T) is proportional to softmax(logits) ** (1 / T)
            w = np.asarray(self.p_zs[position], dtype=np.float64) ** (1 / temperature)
            a['weight'] *= w[a.filed.values] / w.sum()
        a['weight'] /= a.weight.sum()
        a['true'] = self.label[a.id.values]
        a['domain_idx'] = self.zs.domain_idx.values[a.id.values]
        return a.sort_values('weight', ascending=False, ignore_index=True)

    def state(self, positions, temperature=None):
        """Memory snapshot + active set for each position."""
        snaps = self.replay.snapshots(list(positions))
        return {p: (snaps[p], self.active_set(p, snaps[p], temperature)) for p in positions}

    def validate(self, n=200, seed=0, temperature=None):
        """Replay vs the logged RamenActive run: predicted-class queue contents (if logged) and the logged top-8
        retrieved entries. For a run with active_set 'prob', pass its temperature (the memory is the same; the
        retrieval weights then include the zero-shot class weights)."""
        assert (self.stats['active_set'] == 'all') == (temperature is None), 'pass the prob run\'s temperature'
        rng = np.random.default_rng(seed)
        pos = np.sort(rng.choice(len(self.df), n, replace=False))
        st = self.state(pos, temperature)
        sup_ids = np.concatenate(self.stats['support_ids'])
        sup_w = np.concatenate(self.stats['support_weight'])
        queue = np.concatenate(self.stats['queue_ids']) if 'queue_ids' in self.stats else None
        q_ok, top1_ok, overlap, w_err = [], [], [], []
        for p in pos:
            snap, a = st[p]
            if queue is not None and temperature is None:
                logged = np.sort(queue[p][queue[p] >= 0])
                q_ok.append(np.array_equal(logged, np.sort(snap[self.ramen.pred.values[p]])))
            logged = sup_ids[p][sup_ids[p] >= 0]
            mine = a.id.values[:len(logged)]
            top1_ok.append(logged[0] == mine[0])
            overlap.append(len(set(logged) & set(mine)) / len(logged))
            w_err.append(abs(sup_w[p][0] - a.weight.values[0]))
        out = {'positions checked': n, 'top-1 retrieved entry identical %': np.mean(top1_ok) * 100,
               'top-8 retrieved overlap %': np.mean(overlap) * 100,
               'top-1 weight abs. error (median)': np.median(w_err)}
        if q_ok:
            out['predicted-class queue identical %'] = np.mean(q_ok) * 100
        return pd.Series(out).round(4)

    # ------------------------------------------------------------------ plots
    def _tile(self, ax, position, title, color=None):
        r = self.df.iloc[position]
        ax.imshow(self.exp._image(int(r.domain_idx), int(r.sample_idx)))
        ax.set_title(title, fontsize=7, color=color or TEXT_2, loc='center')
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(color or MUTED)
            s.set_linewidth(2 if color else 0.6)

    def show(self, position, state=None, show_k=8, n_active=10, n_queue=10):
        """
        One sample:
          row 1  query image | zero-shot vs Ramen probabilities | active set: share of the update per filed class
          row 2  the highest-weight retrieved entries (the active set)
          row 3+ the memory queues of the true class, the zero-shot class and Ramen's class (most recent first)
        Red frame = the entry's true class is not the class it was filed under (a mis-filed memory entry).
        """
        snap, a = (state or self.state([position]))[position]
        r = self.df.iloc[position]
        y, zs, rp = int(r.label), int(r.zs_pred), int(r.ramen_pred)
        cls = self.classes
        pz, pr = np.asarray(self.p_zs[position]), np.asarray(self.p_ramen[position])

        queues = [('true class', y)]
        for tag, c in [('zero-shot class', zs), ('Ramen class', rp)]:
            if c not in [q[1] for q in queues]:
                queues.append((tag, c))
        ncol = max(n_active, n_queue)
        fig = plt.figure(figsize=(1.5 * ncol, 4.0 + 1.75 * (1 + len(queues))))
        # row 1 is an empty spacer that keeps the rotated class names clear of the tiles below
        gs = GridSpec(3 + len(queues), ncol, figure=fig, height_ratios=[2.2, 0.45] + [1] * (1 + len(queues)),
                      hspace=0.55, wspace=0.12)

        # --- query
        ax = fig.add_subplot(gs[0, :2])
        self._tile(ax, position, '')
        ax.set_title(f'QUERY  #{position}  ({r.domain})\ntrue: {_short(cls[y], 28)}', fontsize=8.5, loc='center')

        # --- probabilities: true class + most probable others under either model
        ax = fig.add_subplot(gs[0, 2:2 + int(ncol * 0.45)])
        top = np.maximum(pz, pr)
        shown = [y] + [c for c in np.argsort(-top) if c != y][:show_k - 1]
        x = np.arange(len(shown))
        for off, p, color, name in [(-0.2, pz, ORANGE, 'zero-shot'), (0.2, pr, BLUE, 'Ramen')]:
            ax.bar(x + off, p[shown], width=0.38, color=color, label=name)
            for xi, v in zip(x + off, p[shown]):
                if v >= 0.08:
                    ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', fontsize=6.5, color=TEXT_2)
        ax.set_xticks(x, [_short(cls[c], 16) for c in shown], rotation=50, ha='right', fontsize=7)
        ax.set_xlabel('bold = true class', fontsize=7)
        ax.get_xticklabels()[0].set_fontweight('bold')
        ax.set_ylim(0, min(1.05, max(pz[shown].max(), pr[shown].max()) * 1.25 + 0.02))
        ax.grid(axis='x', visible=False)
        ax.legend(loc='upper right', fontsize=7.5)
        ax.set_title(f'zero-shot -> {_short(cls[zs])} ({pz[zs]:.2f})\nRamen -> {_short(cls[rp])} ({pr[rp]:.2f})',
                     fontsize=8.5)

        # --- active set: share of the update per filed class
        ax = fig.add_subplot(gs[0, 2 + int(ncol * 0.45) + 1:])
        per_class = a.groupby('filed').weight.sum().sort_values(ascending=False)
        top_c = per_class.head(8)
        colors = [AQUA if c == y else RED if c == rp and rp != y else ORANGE if c == zs else MUTED
                  for c in top_c.index]
        yy = np.arange(len(top_c))[::-1]
        ax.barh(yy, top_c.values * 100, color=colors, height=0.65)
        ax.set_yticks(yy, [_short(cls[c], 18) for c in top_c.index], fontsize=7)
        for yi, v in zip(yy, top_c.values * 100):
            ax.text(v, yi, f' {v:.1f}', va='center', fontsize=6.5, color=TEXT_2)
        ax.set_xlabel('% of update weight', fontsize=8)
        ax.grid(axis='y', visible=False)
        rank_y = f'#{(per_class.index == y).argmax() + 1}' if y in per_class.index else 'no queue'
        same = a.weight[a.domain_idx == r.domain_idx].sum() * 100
        ax.set_title(f'update weight by filed class ({len(per_class)} queues vote)\n'
                     f'true-class queue: {per_class.get(y, 0) * 100:.2f}% (rank {rank_y}); '
                     f'from {_short(str(r.domain), 16)} images: {same:.0f}%\n'
                     f'green = true, orange = zero-shot, red = Ramen class', fontsize=7.5)

        # --- active set entries
        row = 2
        for j, e in enumerate(a.head(n_active).itertuples()):
            ax = fig.add_subplot(gs[row, j])
            self_tag = ' (self)' if e.id == position else ''
            dom = _short(self.exp.environments[e.domain_idx], 11)
            self._tile(ax, int(e.id), f'filed: {_short(cls[e.filed], 15)}\n{e.weight * 100:.1f}% | {dom}{self_tag}',
                       RED if e.filed != e.true else None)
        fig.text(0.005, self._row_y(gs, fig, row), 'active set\n(top weight)', rotation=90, ha='left', va='center',
                 fontsize=8, color=TEXT_2)

        # --- memory queues
        for tag, c in queues:
            row += 1
            ids = np.sort(snap[c])[::-1]
            purity = (self.label[ids] == c).mean() * 100 if len(ids) else float('nan')
            for j, i in enumerate(ids[:n_queue]):
                ax = fig.add_subplot(gs[row, j])
                tag_i = ' (query)' if i == position else ''
                self._tile(ax, int(i), f'true: {_short(cls[self.label[i]], 11 if tag_i else 15)}{tag_i}',
                           RED if self.label[i] != c else None)
            fig.text(0.005, self._row_y(gs, fig, row),
                     f'{tag}\n{_short(cls[c], 16)}\n{len(ids)}/{self.cfg["max_capacity"]}, {purity:.0f}% pure',
                     rotation=90, ha='left', va='center', fontsize=7.5, color=TEXT_2)
            if len(ids) == 0:
                ax = fig.add_subplot(gs[row, 0])
                ax.set_axis_off()
                ax.text(0, 0.5, 'queue empty', fontsize=8, color=MUTED)

        sizes = np.array([len(s) for s in snap])
        filled = np.concatenate([s for s in snap if len(s)])
        purity = (self.label[filled] == self.filed_as[filled]).mean() * 100
        fig.suptitle(f'{self.dataset} | {r.outcome.upper()} | memory at this moment: {sizes.sum():,} entries in '
                     f'{(sizes > 0).sum()}/{self.C} class queues, {(sizes == self.cfg["max_capacity"]).sum()} full, '
                     f'{purity:.1f}% filed under their true class | red frame = mis-filed entry',
                     fontsize=10, x=0.01, ha='left', y=0.995)
        return fig

    @staticmethod
    def _row_y(gs, fig, row):
        pos = gs[row, 0].get_position(fig)
        return (pos.y0 + pos.y1) / 2

    def memory_summary(self, states):
        """One row per shown sample: what the memory and the active set looked like."""
        rows = []
        for p, (snap, a) in states.items():
            r = self.df.iloc[p]
            y, zs, rp = int(r.label), int(r.zs_pred), int(r.ramen_pred)
            per_class = a.groupby('filed').weight.sum()
            n_y = len(snap[y])
            rows.append({'position': p, 'domain': r.domain, 'outcome': r.outcome, 'true': self.classes[y],
                         'zero-shot': self.classes[zs], 'Ramen': self.classes[rp],
                         'p_true zs': r.zs_true_p, 'p_true Ramen': r.ramen_true_p,
                         'true-class queue size': n_y,
                         'true-class queue purity %': (self.label[snap[y]] == y).mean() * 100 if n_y else np.nan,
                         'update % from true-class queue': per_class.get(y, 0) * 100,
                         'update % from Ramen-class queue': per_class.get(rp, 0) * 100,
                         'self weight %': a.weight[a.id == p].sum() * 100,
                         'update % from same domain': a.weight[a.domain_idx == r.domain_idx].sum() * 100,
                         'memory % from same domain': (self.zs.domain_idx.values[np.concatenate(snap)]
                                                       == r.domain_idx).mean() * 100,
                         'active entries truly of query class %': a.weight[a.true == y].sum() * 100})
        return pd.DataFrame(rows).set_index('position').round(2)

    def compare_outcomes(self, n=150, outcomes=('fixed', 'broken', 'both right'), domain=None, seed=0):
        """memory_summary averaged over n random samples per outcome: do the shown examples' patterns generalise?"""
        picks = pd.concat([self.pick(o, domain=domain, n=n, seed=seed) for o in outcomes])
        summary = self.memory_summary(self.state(list(picks.position)))
        cols = ['p_true zs', 'p_true Ramen', 'true-class queue size', 'true-class queue purity %',
                'update % from true-class queue', 'update % from Ramen-class queue', 'self weight %',
                'update % from same domain', 'memory % from same domain', 'active entries truly of query class %']
        t = summary.groupby('outcome')[cols].mean().loc[list(outcomes)]
        t.insert(0, 'samples', summary.groupby('outcome').size().loc[list(outcomes)])
        return t.round(2)
