"""
Look at the images behind a confusion: which dogs does the model call cats, and how sure was it?

    gallery = ConfusionGallery(records, image_fn)     # image_fn(domain_idx, sample_idx) -> H x W x 3 array
    gallery.show(true='dog', pred='cat')              # images of dogs that Ramen predicts as cat
    gallery.show(pred='dog', correct=False)           # images Ramen wrongly calls dog
    gallery.show(true='dog', pred='cat', source='zero_shot', outcome='fixed')  # ... that Ramen then fixes
    gallery.show_image(i)                             # one image: top classes before / after, embedding shift

The gallery only depends on an ImageRecords and a function that returns an image, so it works for any
dataset of the harness (pass exp._image) or any other image store.
"""
import matplotlib.pyplot as plt
import numpy as np

ORDERS = {
    'confident': ('conf', False),       # most confident wrong predictions first (under the chosen source)
    'unsure': ('conf', True),
    'shift': ('emb_shift', False),      # images the adaptation step moved most
    'stream': ('position', True),       # arrival order
}
COLORS = {'zero_shot': '#8a8984', 'ramen': '#2a78d6', 'true': '#1baf7a'}


class ConfusionGallery:

    def __init__(self, records, image_fn):
        self.records = records
        self.image_fn = image_fn

    def find(self, true=None, pred=None, source='ramen', correct=None, outcome=None, domain=None, order='confident',
             n=None, seed=0):
        """
        Rows of images matching the filters (see ImageRecords.mask), sorted by `order`: one of ORDERS or 'random'.
        'confident' / 'unsure' sort by the confidence of `source`'s prediction.
        """
        rows = self.records.where(true=true, pred=pred, source=source, correct=correct, outcome=outcome,
                                  domain=domain)
        if order == 'random':
            rows = rows.sample(frac=1, random_state=seed)
        else:
            col, ascending = ORDERS[order]
            if col == 'conf' and source == 'zero_shot':
                col = 'zs_conf'
            rows = rows.sort_values(col, ascending=ascending, kind='stable')
        return rows if n is None else rows.head(n)

    def show(self, n=16, cols=8, title=None, **filters):
        """A grid of the first n images of find(**filters), captioned zero-shot -> Ramen with probabilities."""
        rows = self.find(n=n, **filters)
        total = int(self.records.mask(**{k: v for k, v in filters.items() if k not in ('order', 'seed')}).sum())
        if not len(rows):
            print(f'no images match {filters}')
            return None
        r = int(np.ceil(len(rows) / cols))
        fig, axes = plt.subplots(r, cols, figsize=(2.9 * cols, 3.7 * r + 0.4), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, (_, row) in zip(axes.flat, rows.iterrows()):
            ax.imshow(self.image_fn(int(row['domain_idx']), int(row['sample_idx'])))
            ax.set_title(f"true  {_short(row['true_class'])}\n"
                         f"zs    {_short(row['zs_class'])} {row['zs_conf']:.2f} · p(true) {row['zs_true_prob']:.2f}\n"
                         f"ramen {_short(row['pred_class'])} {row['conf']:.2f} · p(true) {row['true_prob']:.2f}\n"
                         f"{row['domain']} · shift {row['emb_shift']:.3f}", fontsize=7, loc='left', family='monospace')
        desc = ', '.join(f'{k}={v}' for k, v in filters.items())
        fig.suptitle(title or f'{len(rows)} of {total:,} matching images ({desc})', fontsize=10, x=0.01, ha='left')
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
        return fig

    def show_image(self, i, k=8):
        """One image: its top-k classes under zero-shot and Ramen (and the true class), and the embedding shift."""
        rec = self.records[i]
        classes = self.records.classes
        zs, ra = rec.zs_probs.astype(np.float32), rec.probs.astype(np.float32)
        top = list(dict.fromkeys([rec.label, *np.argsort(-np.maximum(zs, ra))[:k]]))[:k]
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={'width_ratios': [1, 2.4]})
        a0.imshow(self.image_fn(rec.domain_idx, rec.sample_idx))
        a0.axis('off')
        a0.set_title(f'#{i} · {rec.domain} · stream position {rec.position:,}\n'
                     f'true {rec.true_class} · {rec.outcome}', fontsize=9, loc='left')
        y = np.arange(len(top))[::-1]
        a1.barh(y + 0.2, zs[top], height=0.4, color=COLORS['zero_shot'], label='zero-shot')
        a1.barh(y - 0.2, ra[top], height=0.4, color=COLORS['ramen'], label='Ramen')
        a1.set_yticks(y, [('★ ' if c == rec.label else '') + classes[c] for c in top], fontsize=8)
        a1.set_xlabel('probability')
        a1.set_title(f'embedding shift (1 − cos before/after) = {rec.emb_shift:.4f}', fontsize=9, loc='left')
        a1.legend(frameon=False, fontsize=8)
        a1.grid(axis='x', alpha=0.3)
        fig.tight_layout()
        return fig


def _short(name, width=15):
    return name if len(name) <= width else name[:width - 1] + '…'
