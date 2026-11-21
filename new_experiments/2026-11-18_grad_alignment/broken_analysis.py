"""
Images Ramen breaks (zero-shot right -> Ramen wrong): how Ramen's step lines up with the true-label CE gradient,
compared with the zero-shot-right images Ramen keeps right AT THE SAME ZERO-SHOT CONFIDENCE.

Offline: reads the per-image probe frame of the existing ga_probe run (GradAlignmentExperiment.frame()); no GPU,
no new run. Pure pandas / sklearn; no side effects.

    from ga_experiment import GradAlignmentExperiment
    from broken_analysis import BrokenImages
    b = BrokenImages(GradAlignmentExperiment('DomainNet', 0).run_all().frame())
    b.counts()                 # zero-shot right / broken / kept, with counts and percentages
    b.summary()                # medians: broken vs kept (matched to the broken images' confidence) vs kept (all)
    b.separability()           # AUROC of each quantity for "broken", and confidence + alignment together
    b.break_rate_grid()        # break rate by confidence band x cos(sign step, g_ce) quintile within the band
    b.decomposition()          # mean additive parts of cos(update, g_ce): own entry / zero-shot(=true) queue / rest
    b.by_domain(); b.over_stream(); b.own_entry_effect(); b.destinations(class_names)

Matching. Broken images have low zero-shot confidence, so "broken vs kept" mixes alignment with confidence. Kept
images are reweighted so their distribution of p_zs(true) over N_CONF_BINS quantile bins equals the broken images':
w_i = n_broken(bin_i) / n_kept(bin_i). Every "kept (matched)" number is a weighted statistic.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

N_CONF_BINS = 20
GROUPS = ('broken', 'kept (matched)', 'kept (all)')

# per-image quantities compared between groups: column -> name in tables and plots
QUANTITIES = {'cos_self': 'cos(own entropy gradient, g_ce)',
              'cos_update': "cos(Ramen's update, g_ce)",
              'cos_step': "cos(Ramen's sign step, g_ce)",
              'cos_update_shuffled': "control: cos(update, another image's g_ce)",
              'agree_step': 'sign agreement of the step',
              'cos_update_self': 'cos(update, own entropy gradient)',
              'p_true_zs': 'zero-shot p(true class)',
              'dlogp_true': 'change of log p(true) from the step'}
# For a zero-shot-right image the true class IS the zero-shot class: part_zs is 0 by definition and part_true is the
# queue the image was filed under (its other entries).
PARTS = {'part_self': "the image's own entry",
         'part_true': 'other entries of its own (zero-shot = true) queue',
         'part_other': 'all other queues'}
SUM_COL = "cos(Ramen's update, g_ce) (sum)"


def count_pct(n, total):
    return f'{n:,} of {total:,} ({n / total:.1%})' if total else '0 of 0'


def weighted_quantile(x, w, q=0.5):
    order = np.argsort(x)
    x, cw = np.asarray(x)[order], np.cumsum(np.asarray(w)[order])
    return float(np.interp(q * cw[-1], cw, x))


def class_names(dataset):
    """Class index -> name, read from the data folders the loaders use (no torch)."""
    import os
    root = os.path.expanduser('~/data')
    if dataset == 'DomainNet':
        return sorted(os.listdir(os.path.join(root, 'domainbed', 'domain_net', 'real')))
    if dataset == 'EuroSAT':
        return sorted(os.listdir(os.path.join(root, 'EuroSAT', '2750')))
    if dataset == 'ImageNetC5K':
        with open(os.path.join(root, 'corruption', 'ImageNet-C', 'classnames.txt')) as f:
            pairs = [line.strip().split(' ', 1) for line in f if line.strip()]
        names = dict(pairs)
        return [names[w] for w in sorted(names)]
    raise KeyError(dataset)


class BrokenImages:
    """Zero-shot-right images of one probe frame, split into broken (Ramen wrong) and kept (Ramen right)."""

    def __init__(self, df, n_conf_bins=N_CONF_BINS):
        r = df[df['zs_ok']].copy()
        r['broken'] = ~r['ramen_ok']
        r['stored'] = r['part_self'] != 0                       # own entry written (and so retrieved)
        r['conf_bin'] = pd.qcut(r['p_true_zs'], n_conf_bins, labels=False, duplicates='drop')
        nb = r.loc[r['broken'], 'conf_bin'].value_counts()
        nk = r.loc[~r['broken'], 'conf_bin'].value_counts()
        r['w_matched'] = np.where(r['broken'], 0.0, r['conf_bin'].map(nb / nk).fillna(0).to_numpy())
        self.n_all = len(df)
        self.r = r
        self.broken = r[r['broken']]
        self.kept = r[~r['broken']]

    # ------------------------------------------------------------------ the data
    def counts(self):
        n = len(self.r)
        return pd.Series({'images': f'{self.n_all:,}',
                          'zero-shot right': count_pct(n, self.n_all),
                          'broken by Ramen (of zero-shot right)': count_pct(len(self.broken), n),
                          'kept right by Ramen (of zero-shot right)': count_pct(len(self.kept), n),
                          'broken images with own entry stored': count_pct(int(self.broken['stored'].sum()),
                                                                           len(self.broken))})

    # ------------------------------------------------------------------ group comparison
    def _groups(self):
        """(name, frame, weights) for broken, kept matched to broken's confidence, kept (all)."""
        k = self.kept
        return [('broken', self.broken, np.ones(len(self.broken))),
                ('kept (matched)', k, k['w_matched'].to_numpy()),
                ('kept (all)', k, np.ones(len(k)))]

    def summary(self, stat='median'):
        """Median (or mean) of each quantity per group, plus how often the sign step points away from g_ce."""
        rows = {}
        for name, d, w in self._groups():
            agg = (lambda c: weighted_quantile(d[c], w)) if stat == 'median' else \
                  (lambda c: float(np.average(d[c], weights=w)))
            row = {QUANTITIES[c]: agg(c) for c in QUANTITIES}
            neg = float(np.average(d['cos_step'] < 0, weights=w))
            below = float(np.average(d['cos_step'] < d['cos_update_shuffled'], weights=w))
            row['images'] = len(d) if name != 'kept (matched)' else f'{len(d):,} reweighted to {len(self.broken):,}'
            row['sign step points away from g_ce (cos < 0)'] = (count_pct(int(round(neg * len(d))), len(d))
                                                               if name != 'kept (matched)' else f'{neg:.1%}')
            row['step cos below the control'] = f'{below:.1%}'
            rows[name] = row
        t = pd.DataFrame(rows).T
        return t[['images'] + [c for c in t.columns if c != 'images']]

    def histogram(self, col='cos_step', bins=None):
        """Weighted density of `col` per group (for the distribution plot)."""
        bins = np.linspace(-0.3, 0.4, 71) if bins is None else bins
        out = {}
        for name, d, w in self._groups():
            h, _ = np.histogram(d[col].clip(bins[0], bins[-1]), bins=bins, weights=w)
            out[name] = h / h.sum()
        return bins, out

    # ------------------------------------------------------------------ does alignment add to confidence?
    def separability(self):
        """AUROC for picking out broken images (lower value = more likely broken), single quantities and combined."""
        r, y = self.r, self.r['broken'].to_numpy()
        rows = [(QUANTITIES.get(c, c), roc_auc_score(y, -r[c])) for c in
                ('p_true_zs', 'cos_self', 'cos_update', 'cos_step', 'agree_step')]
        # label-free: the update's cosine with the image's own entropy gradient (no g_ce involved)
        rows.append(('label-free: ' + QUANTITIES['cos_update_self'], roc_auc_score(y, -r['cos_update_self'])))
        X1 = np.log(r[['p_true_zs']].to_numpy())
        X2 = np.column_stack([X1, r['cos_step']])
        for name, X in (('logistic: log p_zs(true)', X1), ('logistic: log p_zs(true) + cos(sign step, g_ce)', X2)):
            p = cross_val_predict(LogisticRegression(), X, y, cv=5, method='predict_proba')[:, 1]
            rows.append((name + ' (5-fold CV)', roc_auc_score(y, p)))
        # cos_step inside each confidence bin, averaged with the broken images' bin distribution
        aucs, ws = [], []
        for _, d in r.groupby('conf_bin'):
            nb = int(d['broken'].sum())
            if 0 < nb < len(d):
                aucs.append(roc_auc_score(d['broken'], -d['cos_step']))
                ws.append(nb)
        rows.append((f'cos(sign step, g_ce) within confidence bins ({len(aucs)} bins, weighted by broken count)',
                     float(np.average(aucs, weights=ws))))
        return pd.DataFrame(rows, columns=['predictor of "broken"', 'AUROC']).set_index('predictor of "broken"')

    def break_rate_grid(self, conf_edges=(0, 0.3, 0.45, 0.6, 0.8, 1.0), n_cos=5):
        """Break rate per confidence band x quintile of cos(sign step, g_ce) WITHIN the band (long table)."""
        r = self.r.copy()
        r['band'] = pd.cut(r['p_true_zs'], list(conf_edges), include_lowest=True)
        rows = []
        for band, d in r.groupby('band', observed=True):
            q = pd.qcut(d['cos_step'], n_cos, labels=False, duplicates='drop')
            for qi, s in d.groupby(q):
                rows.append(dict(band=str(band), quintile=int(qi) + 1, images=len(s), broken=int(s['broken'].sum()),
                                 rate=s['broken'].mean(), cos_step_median=s['cos_step'].median()))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ mechanism
    def decomposition(self):
        """Mean additive parts of cos(update, g_ce) per group (the parts sum to the mean cos_update)."""
        rows = {}
        for name, d, w in self._groups():
            rows[name] = {**{PARTS[c]: float(np.average(d[c], weights=w)) for c in PARTS},
                          SUM_COL: float(np.average(d['cos_update'], weights=w)),
                          'own entry stored': float(np.average(d['stored'], weights=w))}
        return pd.DataFrame(rows).T

    def by_domain(self):
        def row(d):
            b, k = d[d['broken']], d[~d['broken']]
            return pd.Series({'zero-shot right': len(d), 'broken': count_pct(len(b), len(d)),
                              'median p_zs(true), broken': b['p_true_zs'].median(),
                              'median cos(sign step, g_ce), broken': b['cos_step'].median(),
                              'median cos(sign step, g_ce), kept': k['cos_step'].median(),
                              'broken with cos < 0': count_pct(int((b['cos_step'] < 0).sum()), len(b))})
        return self.r.groupby('domain', observed=True)[list(self.r.columns)].apply(row)

    def over_stream(self, n_bins=10):
        """Per equal slice of the stream: break rate, own entry stored, median cos(sign step) of broken / kept."""
        r = self.r.sort_values('position')
        s = pd.qcut(r['position'], n_bins, labels=False)
        rows = []
        for i, d in r.groupby(s):
            rows.append(dict(slice=int(i) + 1, start=int(d['position'].min()), end=int(d['position'].max()),
                             zs_right=len(d), broken=int(d['broken'].sum()), rate=d['broken'].mean(),
                             stored=d['stored'].mean(), cos_step_broken=d.loc[d['broken'], 'cos_step'].median(),
                             cos_step_kept=d.loc[~d['broken'], 'cos_step'].median()))
        return pd.DataFrame(rows)

    def own_entry_effect(self, min_group=1000, pos_width=10_000):
        """
        Break rate with vs without the image's own entry stored. Two estimates:
          - inside each confidence bin, averaged over bins with the zero-shot-right images' bin distribution;
          - inside the freeze window only (stream slices of `pos_width` where 10-90% of entries are stored), per
            (position slice x confidence bin) cell, so stream position is matched too.
        Still an association: whether an entry is stored also depends on its class (a queue that is not yet full
        always stores). None if either group has < min_group images (ImageNet-C, EuroSAT: almost all stored).
        """
        r = self.r
        if min(r['stored'].sum(), (~r['stored']).sum()) < min_group:
            return None

        def adjusted(d, keys):
            t = d.groupby(keys + ['stored'], observed=True)['broken'].agg(['size', 'mean']).unstack('stored').dropna()
            w = t['size'].sum(1)
            return (float(np.average(t[('mean', True)], weights=w)), float(np.average(t[('mean', False)], weights=w)),
                    len(t))

        a_s, a_n, _ = adjusted(r, ['conf_bin'])
        r = r.assign(pos_bin=r['position'] // pos_width)
        share = r.groupby('pos_bin')['stored'].transform('mean')
        win = r[(share > 0.1) & (share < 0.9)]
        b_s, b_n, cells = adjusted(win, ['pos_bin', 'conf_bin'])
        return pd.Series({
            'zero-shot-right images with own entry stored': count_pct(int(self.r['stored'].sum()), len(self.r)),
            'break rate, stored / not stored (confidence-matched)': f'{a_s:.2%} / {a_n:.2%}',
            'freeze window (stream positions)': f'{int(win["position"].min()):,}-{int(win["position"].max()):,}, '
                                                f'{len(win):,} images, {win["stored"].mean():.0%} stored',
            'break rate, stored / not stored (position- and confidence-matched)': f'{b_s:.2%} / {b_n:.2%} '
                                                                                  f'({cells} cells)'})

    def destinations(self, names, top=10):
        """Which classes Ramen moves broken images to, and the true class most often moved there."""
        b = self.broken
        rows = []
        for c, d in b.groupby('ramen_pred'):
            src = d['label'].value_counts()
            rows.append({'Ramen predicts': names[c], 'broken images': len(d),
                         'share of broken': len(d) / len(b), 'distinct true classes': src.size,
                         'most common true class': f'{names[src.index[0]]} ({src.iloc[0]:,})'})
        t = pd.DataFrame(rows).sort_values('broken images', ascending=False)
        return t.head(top).reset_index(drop=True), len(t)
