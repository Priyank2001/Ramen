"""
One test image, end to end: its logits and probabilities before adaptation, after Ramen, and after the oracle-cache
run, and what each cache contained and retrieved when the image arrived.

    card = ExampleCard(runs, result, target='pillow')
    card.prediction_table(row)     # logits and probabilities of the classes that matter, three sources
    card.show_prediction(row)      # image | logits | probabilities
    card.show_caches(row)          # update share per queue, retrieved entries and the target queue, both caches
    card.cache_table(row)          # where the update weight comes from, Ramen cache vs oracle cache

`result` is a ReplayResult from CacheReplayRun.run with the row in keep_rows and the target in queue_classes.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from sink_runs import SOURCE_NAMES  # also sets up the import paths
from ramen_analysis import AQUA, BLUE, MUTED, ORANGE, RED, TEXT_2  # noqa: E402

SOURCE_COLORS = {'zero_shot': MUTED, 'ramen': BLUE, 'oracle': '#4a3aa7'}
ROLE_COLORS = {'target': RED, 'zero-shot class': ORANGE, 'true class': AQUA}


def _short(name, n=18):
    name = str(name)
    return name if len(name) <= n else name[:n - 1] + '…'


class ExampleCard:

    def __init__(self, runs, result, target, n_classes=8, n_tiles=10):
        self.runs, self.result = runs, result
        self.target = runs.classes.index(target)
        self.n_classes, self.n_tiles = n_classes, n_tiles
        self.caches = list(dict.fromkeys(c for _, c in result.retrieved))

    # ------------------------------------------------------------------ data
    def _row(self, row):
        return self.runs.table.iloc[row]

    def roles(self, row):
        """class index -> its roles for this image (true / zero-shot / Ramen / oracle prediction / target)."""
        r = self._row(row)
        out = {}
        for tag, c in [('true', r.label), ('zero-shot', r.zs_pred), ('Ramen', r.ramen_pred),
                       ('oracle', r.oracle_pred), ('target', self.target)]:
            out.setdefault(int(c), []).append(tag)
        return out

    def shown_classes(self, row):
        """The classes with a role, then the most probable others under any source, n_classes in total."""
        roles = self.roles(row)
        p = np.max([self.runs.probs(s, [row])[0] for s in SOURCE_NAMES], axis=0)
        rest = [int(c) for c in np.argsort(-p) if int(c) not in roles]
        return list(roles) + rest[:max(0, self.n_classes - len(roles))]

    def prediction_table(self, row):
        cls, roles = self.shown_classes(row), self.roles(row)
        d = {'roles': [', '.join(roles.get(c, [])) for c in cls]}
        for s, name in SOURCE_NAMES.items():
            d[f'logit {name}'] = self.runs.logits(s, [row])[0][cls]
        for s, name in SOURCE_NAMES.items():
            d[f'prob {name}'] = self.runs.probs(s, [row])[0][cls]
        return pd.DataFrame(d, index=[self.runs.classes[c] for c in cls])

    def cache_table(self, row):
        """Where the update weight comes from (shares, %), one column per cache."""
        s = self.result.summary.loc[row].T
        pct = [i for i in s.index if i not in ('non-empty queues', 'top-10 entries truly target')]
        s.loc[pct] = (s.loc[pct].astype(float) * 100).round(1)
        name = self.runs.classes[self.target]
        s.index = [i.replace('target', name) for i in s.index]
        return s

    # ------------------------------------------------------------------ figures
    def _tile(self, ax, sid, lines, frame=None):
        ax.imshow(self.runs.image(int(sid)))
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        ax.set_title('\n'.join(lines), fontsize=6.8, color=TEXT_2, loc='center')
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(frame or MUTED)
            sp.set_linewidth(2.2 if frame else 0.6)

    def _title(self, row):
        r, C = self._row(row), self.runs.classes
        return (f'#{row:,} ({r.domain}) — true: {C[r.label]} | zero-shot: {C[r.zs_pred]} ({r.zs_conf:.2f}) '
                f'→ Ramen: {C[r.ramen_pred]} ({r.ramen_conf:.2f}) | oracle cache: {C[r.oracle_pred]} '
                f'({r.oracle_conf:.2f})')

    def show_prediction(self, row):
        """Image | logits (dot per source) | probabilities (bar per source), for the shown classes."""
        t = self.prediction_table(row)
        fig = plt.figure(figsize=(15, 3.8))
        gs = GridSpec(1, 9, figure=fig, wspace=0.6)
        ax = fig.add_subplot(gs[0, :2])
        ax.imshow(self.runs.image(row))
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        labels = [f'{_short(c)}\n({r})' if r else _short(c) for c, r in zip(t.index, t.roles)]
        x = np.arange(len(t))
        ax = fig.add_subplot(gs[0, 2:5])
        for off, (s, name) in zip((-0.22, 0, 0.22), SOURCE_NAMES.items()):
            ax.plot(x + off, t[f'logit {name}'], 'o', ms=6, color=SOURCE_COLORS[s], label=name,
                    markeredgecolor='white', markeredgewidth=1)
        ax.set_xticks(x, labels, rotation=55, ha='right', fontsize=7)
        ax.set_ylabel('logit = 100 · cos(image, class text)')
        ax.set_title('logits', fontsize=10)
        ax.legend(fontsize=7.5, loc='upper right')
        ax.grid(axis='x', visible=False)
        ax = fig.add_subplot(gs[0, 5:])
        for off, (s, name) in zip((-0.27, 0, 0.27), SOURCE_NAMES.items()):
            v = t[f'prob {name}'].to_numpy()
            ax.bar(x + off, v, width=0.25, color=SOURCE_COLORS[s], label=name)
            for xi, vi in zip(x + off, v):
                if vi >= 0.1:
                    ax.text(xi, vi + 0.015, f'{vi:.2f}', ha='center', fontsize=6, color=TEXT_2)
        ax.set_xticks(x, labels, rotation=55, ha='right', fontsize=7)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('probability')
        ax.set_title('probabilities', fontsize=10)
        ax.grid(axis='x', visible=False)
        fig.suptitle(self._title(row), fontsize=10.5, x=0.01, ha='left', y=1.04)
        return fig

    def show_caches(self, row, queue='target'):
        """
        Per cache: the share of the update from each queue (top 8), the top retrieved entries, and the newest
        entries of one queue ('target', 'zero-shot class' or 'true class') at the moment the image arrived.
        Red frame = the entry is a zero-shot error (its own gradient sharpens a wrong class).
        """
        r, C = self._row(row), self.runs.classes
        y, zs = int(r.label), int(r.zs_pred)
        qc = {'target': self.target, 'zero-shot class': zs, 'true class': y}[queue]
        n = self.n_tiles
        fig = plt.figure(figsize=(1.55 * n, 2.6 + 2.0 * 2 * len(self.caches)))
        gs = GridSpec(2 + 2 * len(self.caches), n, figure=fig,
                      height_ratios=[1.5, 0.12] + [1] * 2 * len(self.caches), hspace=0.75, wspace=0.12)
        top = gs[0, :].subgridspec(1, 2 * len(self.caches), wspace=0.85)
        dom = self._row(row).domain_idx
        for k, cache in enumerate(self.caches):
            e = self.result.retrieved[(row, cache)]
            same = e.weight[e.domain_idx == dom].sum() * 100
            entries, queues, full = self.result.sizes[(row, cache)]
            whats = [('queue', 'by queue'), ('zs_pred', "by entry's zero-shot class"
                                                         + (' (= queue)' if cache == 'Ramen cache' else ''))]
            for j, (col, what) in enumerate(whats):
                share = e.groupby(col).weight.sum().sort_values(ascending=False).head(8)
                ax = fig.add_subplot(top[0, 2 * k + j])
                colors = [ROLE_COLORS['target'] if c == self.target else ROLE_COLORS['true class'] if c == y
                          else ROLE_COLORS['zero-shot class'] if c == zs else MUTED for c in share.index]
                yy = np.arange(len(share))[::-1]
                ax.barh(yy, share.values * 100, color=colors, height=0.65)
                ax.set_yticks(yy, [_short(C[c]) for c in share.index], fontsize=7.5)
                for yi, v in zip(yy, share.values * 100):
                    ax.text(v, yi, f' {v:.1f}%', va='center', fontsize=7, color=TEXT_2)
                ax.set_xlim(0, share.values.max() * 100 * 1.35)
                ax.grid(axis='y', visible=False)
                ax.set_xlabel('% of the update weight', fontsize=7.5)
                ax.set_title(what, fontsize=8.5)
            fig.text(top[0, 2 * k].get_position(fig).x0, top[0, 2 * k].get_position(fig).y1 + 0.035,
                     f'{cache.upper()}: update weight\n{entries:,} entries, {queues} queues vote ({full} full); '
                     f'{same:.0f}% of the weight from {self._row(row).domain}',
                     fontsize=8.5, color=TEXT_2, fontweight='bold')
            # retrieved entries, highest weight first
            gi = 2 + 2 * k
            for j, x in enumerate(e.head(n).itertuples()):
                ax = fig.add_subplot(gs[gi, j])
                tag = ' (self)' if x.self else ''
                self._tile(ax, x.id, [f'queue: {_short(C[x.queue], 14)}', f'true: {_short(C[x.true], 14)}',
                                      f'{x.weight * 100:.1f}%{tag}'], RED if x.zs_pred != x.true else None)
            self._row_label(fig, gs, gi, f'{cache}\nretrieved\n(top weight)')
            # newest entries of the chosen queue
            ids = self.result.queues.get((row, cache, qc))
            if ids is None:
                raise KeyError(f'queue {C[qc]} was not kept for row {row}: pass it in queue_classes')
            gi += 1
            truly = (self.runs.table.label.to_numpy()[ids] == qc).mean() * 100 if len(ids) else np.nan
            for j, sid in enumerate(ids[:n]):
                ax = fig.add_subplot(gs[gi, j])
                zp = self.runs.table.zs_pred.iat[int(sid)]
                self._tile(ax, sid, [f'true: {_short(C[self.runs.table.label.iat[int(sid)]], 14)}',
                                     f'zero-shot: {_short(C[zp], 14)}'],
                           RED if zp != self.runs.table.label.iat[int(sid)] else None)
            self._row_label(fig, gs, gi, f'{cache}\nqueue "{_short(C[qc], 12)}"\n{len(ids)} entries,\n'
                                         f'{truly:.0f}% truly {_short(C[qc], 10)}')
        fig.suptitle(self._title(row) + '\nbars: red = ' + C[self.target] + ', orange = zero-shot class, green = '
                     'true class | tiles: red frame = entry is a zero-shot error', fontsize=10.5, x=0.01, ha='left',
                     y=1.03)
        return fig

    @staticmethod
    def _row_label(fig, gs, row, text):
        pos = gs[row, 0].get_position(fig)
        fig.text(pos.x0 - 0.006, (pos.y0 + pos.y1) / 2, text, rotation=90, ha='right', va='center', fontsize=7.5,
                 color=TEXT_2)
