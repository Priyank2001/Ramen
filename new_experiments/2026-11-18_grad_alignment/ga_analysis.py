"""
Tables over the per-image frame of GradAlignmentExperiment.frame() (pure pandas: no GPU, no side effects).

    from ga_analysis import data_table, direction_summary, hypotheses, binned_outcome_rates, decomposition
    data_table(df)                         # images, zero-shot / Ramen correct, fixed, broken per domain
    direction_summary(df, 'outcome')       # medians of every cosine per group
    hypotheses(df)                         # the pre-registered H1-H3 (ga_configs.py), with the numbers
"""
import numpy as np
import pandas as pd

# The three gradient directions compared with the true-label CE gradient, as (column, name in tables and plots)
DIRECTIONS = {'cos_self': 'own entropy gradient (no memory)',
              'cos_update': "Ramen's update",
              'cos_update_shuffled': "control: another image's CE gradient"}
PARTS = {'part_self': "the image's own entry", 'part_true': 'other entries of the true-class queue',
         'part_zs': 'other entries of the (wrong) zero-shot-class queue', 'part_other': 'all other queues'}


def count_pct(n, total):
    return f'{n:,} of {total:,} ({n / total:.1%})' if total else '0 of 0'


def zs_group(df):
    return np.where(df['zs_ok'], 'zero-shot right', 'zero-shot wrong')


# ---------------------------------------------------------------------------- the data
def data_table(df):
    """Per domain and in total: images, classes, zero-shot / Ramen correct, fixed and broken by Ramen."""
    def row(d):
        n = len(d)
        return pd.Series({'images': n, 'classes': d['label'].nunique(),
                          'zero-shot correct': count_pct(int(d['zs_ok'].sum()), n),
                          'Ramen correct': count_pct(int(d['ramen_ok'].sum()), n),
                          'fixed by Ramen': count_pct(int((d['outcome'] == 'fixed by Ramen').sum()), n),
                          'broken by Ramen': count_pct(int((d['outcome'] == 'broken by Ramen').sum()), n)})
    t = df.groupby('domain', observed=True)[['label', 'zs_ok', 'ramen_ok', 'outcome']].apply(row)
    t.loc['all (pooled)'] = row(df)
    return t


def outcome_table(df):
    n = len(df)
    c = df['outcome'].value_counts().reindex(df['outcome'].cat.categories)
    return pd.DataFrame({'images': c, 'share': (c / n).map('{:.1%}'.format),
                         'meaning': ['zero-shot right, Ramen right', 'zero-shot wrong, Ramen right',
                                     'zero-shot right, Ramen wrong', 'zero-shot wrong, Ramen wrong']})


# ---------------------------------------------------------------------------- directions
def direction_summary(df, by):
    """Per group: images, median of each cosine, share of images whose Ramen update has cos > 0 with g_ce."""
    g = df.groupby(by, observed=True)
    t = g[list(DIRECTIONS) + ['cos_step', 'agree_step', 'agree_self']].median()
    t.insert(0, 'images', g.size())
    t['update cos > 0'] = g['cos_update'].apply(lambda s: count_pct(int((s > 0).sum()), len(s)))
    return t.rename(columns={**DIRECTIONS, 'cos_step': "Ramen's sign step", 'agree_step': 'sign agreement (step)',
                             'agree_self': 'sign agreement (own entropy)'})


def stream_rolling(df, window):
    """Rolling mean over stream position of each direction's cosine, separately for zero-shot right / wrong."""
    out = {}
    for name, d in df.sort_values('position').groupby(zs_group(df.sort_values('position'))):
        r = d[list(DIRECTIONS)].rolling(window, min_periods=window // 2).mean()
        r['position'] = d['position'].to_numpy()
        out[name] = r.iloc[::max(1, window // 20)]
    return out


def binned_outcome_rates(df, col='cos_step', n_bins=10):
    """
    Quantile bins of `col` within each zero-shot group. For zero-shot-wrong images: share Ramen fixes;
    for zero-shot-right images: share Ramen breaks. Also the median change of log p(true class).
    """
    rows = []
    for grp, d in df.groupby(zs_group(df)):
        bins = pd.qcut(d[col], n_bins, duplicates='drop')
        event = ~d['ramen_ok'] if grp == 'zero-shot right' else d['ramen_ok']
        for b, sub in d.groupby(bins, observed=True):
            ev = event.loc[sub.index]
            rows.append(dict(group=grp, bin=str(b), x=sub[col].median(), images=len(sub), events=int(ev.sum()),
                             rate=ev.mean(), dlogp_median=sub['dlogp_true'].median()))
    return pd.DataFrame(rows)


def decomposition(df, by='outcome'):
    """Mean additive share of cos(Ramen update, g_ce) from the true-class queue, the zero-shot queue, the rest."""
    g = df.groupby(by, observed=True)
    t = g[list(PARTS) + ['cos_update']].mean()
    t.insert(0, 'images', g.size())
    return t


def memory_freeze_position(df, window=5000):
    """First stream position after which fewer than half of the images get their own entry stored (fp16 freeze)."""
    stored = (df.sort_values('position')['part_self'] != 0).rolling(window, min_periods=window).mean()
    below = np.flatnonzero(stored.to_numpy() < 0.5)
    return int(below[0]) - window // 2 if len(below) else None


def spearman(df, x, y):
    return df[[x, y]].rank().corr().iloc[0, 1]


# ---------------------------------------------------------------------------- pre-registered hypotheses
def hypotheses(df):
    wrong, right = df[~df['zs_ok']], df[df['zs_ok']]
    rates = binned_outcome_rates(df)
    fix = rates[rates['group'] == 'zero-shot wrong']
    broken = df[df['outcome'] == 'broken by Ramen']
    rows = [
        ('H1', 'median cos(update, g_ce) > median control cos(update_i, g_ce_j)',
         f"{df['cos_update'].median():+.3f} vs {df['cos_update_shuffled'].median():+.3f}",
         df['cos_update'].median() > df['cos_update_shuffled'].median()),
        ('H2', 'zero-shot wrong: median cos(update, g_ce) > median cos(own entropy, g_ce)',
         f"{wrong['cos_update'].median():+.3f} vs {wrong['cos_self'].median():+.3f} ({len(wrong):,} images)",
         wrong['cos_update'].median() > wrong['cos_self'].median()),
        ('H3a', 'zero-shot wrong: fix rate rises with cos(sign step, g_ce) (Spearman over bins > 0)',
         f"fix rate {fix['rate'].iloc[0]:.1%} (lowest bin) -> {fix['rate'].iloc[-1]:.1%} (highest); "
         f"Spearman {spearman(fix, 'x', 'rate'):+.2f}",
         spearman(fix, 'x', 'rate') > 0),
        ('H3b', 'images Ramen breaks: median cos(sign step, g_ce) < 0',
         f"{broken['cos_step'].median():+.3f} ({len(broken):,} images; zero-shot right overall "
         f"{right['cos_step'].median():+.3f})",
         broken['cos_step'].median() < 0),
    ]
    return pd.DataFrame(rows, columns=['hypothesis', 'test', 'numbers', 'holds']).set_index('hypothesis')
