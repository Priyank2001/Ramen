"""
Look inside RamenActive's memory at the moments the adapted model gets confused.

A confusion event is a test sample the adapted model gets wrong ('wrong'), or more strictly one that zero-shot CLIP
got right and adaptation broke ('broken'). For each event the viewer shows:
  - the query image,
  - the cache entries it retrieved with the largest weight (its active set), with the class each was filed under,
  - the images stored in the queue of the class it was (wrongly) predicted as, at that moment.
Red titles mark memory entries whose filed class is not their true class.

Usage (from a notebook in new_experiments/):
    exp.run('RamenActive', name='active_all', cfg={'active_set': 'all'})
    view = CacheViewer(exp, 'active_all')
    view.pairs()                         # most frequent confusions (true -> predicted) and where each starts
    view.plot_onset()                    # cumulative count of the top confusions along the stream
    view.show_pair('tabby cat', 'CRT monitor', n=2)   # the first 2 times the model makes that confusion
    view.show(position)                  # one query, by stream position
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ramen_analysis import CATEGORICAL, MUTED, RED, TEXT_2


def _short(name, n=18):
    return name if len(name) <= n else name[:n - 1] + '…'


class CacheViewer:

    def __init__(self, exp, name):
        self.exp, self.name = exp, name
        self.classes = exp.classes
        self.df = exp.results[name]
        stats = exp.method_stats[name]
        assert 'queue_ids' in stats, f'{name} was run with log_queue: false; re-run it with log_queue: true'
        self.filed_as = np.concatenate(stats['filed_as'])        # zero-shot class queue each sample went into
        self.support_ids = np.concatenate(stats['support_ids'])  # (N, n) stream positions, -1 = none
        self.support_w = np.concatenate(stats['support_weight'])
        self.queue_ids = np.concatenate(stats['queue_ids'])      # (N, capacity) predicted class's queue
        assert len(self.filed_as) == len(self.df) and (self.df.position.values == np.arange(len(self.df))).all()
        self.label, self.pred = self.df.label.values, self.df.pred.values

    def _class(self, c):
        return c if isinstance(c, (int, np.integer)) else self.classes.index(c)

    # ------------------------------------------------------------------ tables
    def events(self, kind='broken'):
        """One row per confusion event, with what the predicted class's queue held at that moment."""
        wrong = self.pred != self.label
        mask = {'broken': wrong & (self.filed_as == self.label), 'wrong': wrong}[kind]
        ev = self.df.loc[mask, ['position', 'domain', 'label', 'pred']].copy()
        ev['zs_pred'] = self.filed_as[mask]

        q = self.queue_ids[mask]
        stored = q >= 0
        q_label = np.where(stored, self.label[q.clip(0)], -1)  # true class of every image in that queue
        size = stored.sum(1).clip(1)
        ev['queue size'] = stored.sum(1)
        ev['queue: query class %'] = (q_label == ev.label.values[:, None]).sum(1) / size * 100
        ev['queue: correctly filed %'] = (q_label == ev.pred.values[:, None]).sum(1) / size * 100
        ev['self weight'] = (self.support_w[mask] * (self.support_ids[mask] == ev.position.values[:, None])).sum(1)
        return ev

    def pairs(self, kind='broken', top=15):
        """Most frequent (true -> predicted) confusions, where each first appears, and its queue's makeup."""
        ev = self.events(kind)
        g = ev.groupby(['label', 'pred'])
        t = pd.DataFrame({'count': g.size(), 'first position': g.position.min(), 'median position': g.position.median(),
                          'queue: query class %': g['queue: query class %'].mean(),
                          'queue: correctly filed %': g['queue: correctly filed %'].mean()})
        t = t.nlargest(top, 'count')
        t.index = [f'{self.classes[a]} -> {self.classes[b]}' for a, b in t.index]
        return t.round(1)

    # ------------------------------------------------------------------ plots
    def plot_onset(self, kind='broken', top=5):
        """Cumulative number of each of the top confusions along the stream: where the model starts to confuse."""
        ev = self.events(kind)
        top_pairs = ev.groupby(['label', 'pred']).size().nlargest(top).index
        fig, ax = plt.subplots(figsize=(10, 3.6))
        for color, (a, b) in zip(CATEGORICAL, top_pairs):
            pos = np.sort(ev.position[(ev.label == a) & (ev.pred == b)].values)
            ax.step(pos, np.arange(1, len(pos) + 1), where='post', color=color, lw=1.8,
                    label=f'{_short(self.classes[a])} -> {_short(self.classes[b])}')
        ax.set_xlim(0, len(self.df))
        ax.set_xlabel('position in the test stream')
        ax.set_ylabel(f'cumulative {kind} events')
        ax.set_title(f'{self.name}: when the top confusions appear')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        return fig

    def _tile(self, ax, position, title, wrong=False):
        row = self.df.iloc[position]
        ax.imshow(self.exp._image(row.domain_idx, row.sample_idx))
        ax.set_title(title, fontsize=7.5, color=RED if wrong else TEXT_2, loc='center')
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(RED if wrong else MUTED)
            s.set_linewidth(2 if wrong else 0.6)

    def show(self, position, n_queue=16, ncol=9):
        """One query: its image, its highest-weight retrieved entries, and the images in its predicted class's queue."""
        support = [(i, w) for i, w in zip(self.support_ids[position], self.support_w[position]) if i >= 0][:ncol - 1]
        queue = self.queue_ids[position]
        queue = np.sort(queue[queue >= 0])[::-1][:n_queue]  # most recent first
        queue_rows = int(np.ceil(len(queue) / ncol))

        fig, axes = plt.subplots(1 + queue_rows, ncol, figsize=(1.45 * ncol, 1.9 * (1 + queue_rows)), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')

        y, p, zs = self.label[position], self.pred[position], self.filed_as[position]
        self._tile(axes[0, 0], position, f'QUERY\ntrue: {_short(self.classes[y])}')
        for ax, (i, w) in zip(axes[0, 1:], support):
            tag = ' (self)' if i == position else ''
            self._tile(ax, i, f'filed: {_short(self.classes[self.filed_as[i]])}\nw={w:.2f}{tag}',
                       wrong=self.filed_as[i] != self.label[i])
        for ax, i in zip(axes[1:].flat, queue):
            self._tile(ax, i, f'true: {_short(self.classes[self.label[i]])}', wrong=self.label[i] != p)
        if queue_rows:
            axes[1, 0].text(-0.15, 0.5, f"queue of '{_short(self.classes[p], 24)}'", transform=axes[1, 0].transAxes,
                            rotation=90, ha='right', va='center', fontsize=8, color=TEXT_2)
        axes[0, 1].text(-0.15, 0.5, 'retrieved', transform=axes[0, 1].transAxes, rotation=90,
                        ha='right', va='center', fontsize=8, color=TEXT_2)

        domain = self.df.domain.iloc[position]
        fig.suptitle(f'{self.name} | stream position {position} ({domain}) | true: {self.classes[y]} | '
                     f'zero-shot: {self.classes[zs]} | adapted: {self.classes[p]}\n'
                     f'red = memory entry filed under a class that is not its true class', fontsize=9, x=0.01,
                     ha='left')
        fig.tight_layout()
        return fig

    def show_pair(self, true_class, pred_class, n=2, kind='broken', min_queue=1, **kwargs):
        """The first n times the model confuses true_class with pred_class while that class's queue holds at least
        min_queue images (early in the stream the predicted class's queue can still be empty)."""
        ev = self.events(kind)
        a, b = self._class(true_class), self._class(pred_class)
        first = ev[(ev.label == a) & (ev.pred == b) & (ev['queue size'] >= min_queue)].position.values[:n]
        return [self.show(p, **kwargs) for p in first]
