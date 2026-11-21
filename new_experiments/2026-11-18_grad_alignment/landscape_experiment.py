"""
Loss-landscape run: choose images from the cached ga_probe frame, replay Ramen with RamenLandscapeProbe, load slices.

    from landscape_experiment import LandscapeExperiment
    e = LandscapeExperiment('DomainNet', seed=0)          # n_per_domain=200, results_dir='smoke' for the smoke run
    e.run_all()                                           # NoAdapt, Ramen, ga_probe (cached), then ga_landscape (GPU)
    e.sanity_check_landscape()                            # predictions == Ramen; replay == probe on chosen images
    recs = e.landscape_records()                          # one dict per chosen image, with its outcome

Selection (fixed before the run, saved to landscape_selection_<dataset>[_smoke].json): N_PER_GROUP images drawn
uniformly at random (numpy seed 0) from each outcome group of the ga_probe frame (both right, fixed, broken,
both wrong); all of a group if it is smaller.
"""
import json
import os

import numpy as np
import pandas as pd

from ga_experiment import OUTCOMES, GradAlignmentExperiment, outcome

LANDSCAPE_RUN = 'ga_landscape'
N_PER_GROUP = 50
HERE = os.path.dirname(os.path.abspath(__file__))


def select_positions(frame, n_per_group=N_PER_GROUP, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for g in OUTCOMES:
        pos = np.sort(frame.loc[frame['outcome'] == g, 'position'].to_numpy())
        take = pos if len(pos) <= n_per_group else np.sort(rng.choice(pos, n_per_group, replace=False))
        out[g] = [int(p) for p in take]
    return out


class LandscapeExperiment(GradAlignmentExperiment):

    def __init__(self, dataset, seed=0, n_per_domain=None, results_dir=None, device=None):
        super().__init__(dataset, seed=seed, n_per_domain=n_per_domain, results_dir=results_dir, device=device)
        tag = '_smoke' if n_per_domain else ''
        self.selection_path = os.path.join(HERE, f'landscape_selection_{dataset}_s{seed}{tag}.json')

    def selection(self):
        if not os.path.exists(self.selection_path):
            sel = select_positions(self.frame())
            with open(self.selection_path, 'w') as f:
                json.dump(sel, f)
        with open(self.selection_path) as f:
            return json.load(f)

    def run_landscape(self, use_cache=True):
        if use_cache and self.is_cached(LANDSCAPE_RUN):
            return self._load(LANDSCAPE_RUN)
        positions = sorted(p for ps in self.selection().values() for p in ps)
        if not self.loaded:
            self.exp.load()
            self.loaded = True
        return self.exp.run('RamenLandscapeProbe', name=LANDSCAPE_RUN, cfg={'landscape_positions': positions},
                            use_cache=use_cache)

    def _load(self, name):
        path = self.exp._cache_path(name)
        self.exp.results[name] = pd.read_pickle(path)
        self.exp.method_stats[name] = pd.read_pickle(path[:-len('.pkl')] + '_stats.pkl')
        return self.exp.results[name]

    def run_all(self):
        super().run_all()
        self.run_landscape()
        return self

    def sanity_check_landscape(self):
        """Predictions == Ramen on every image; on the chosen images the replay matches the probe frame."""
        ramen, land = self._in_stream_order('Ramen'), self._in_stream_order(LANDSCAPE_RUN)
        same = int((ramen['pred'] == land['pred']).sum())
        df = self.frame().set_index('position')
        recs = self.stats[LANDSCAPE_RUN]['landscape']
        r = pd.DataFrame([{k: v for k, v in x.items() if np.ndim(v) == 0} for x in recs]).set_index('position')
        f = df.loc[r.index]
        checks = {
            f'{LANDSCAPE_RUN} == Ramen (all images)': (same, len(ramen)),
            'chosen images recorded': (len(r), sum(len(v) for v in self.selection().values())),
            'label / zero-shot / Ramen prediction match the probe frame':
                (int(((r['label'] == f['label']) & (r['zs_pred'] == f['zs_pred'])
                      & (r['ramen_pred'] == f['ramen_pred'])).sum()), len(r)),
            'cos(sign step, g_ce) matches the probe (|diff| < 1e-3)':
                (int(((r['cos_step'] - f['cos_step']).abs() < 1e-3).sum()), len(r)),
            'slice point (0, 0) predicts the zero-shot class': (int(r['check_zs_pred'].sum()), len(r)),
            'slice point (1, 0) predicts Ramen\'s class': (int(r['check_ramen_pred'].sum()), len(r)),
        }
        for k, (a, n) in checks.items():
            print(f'  {k}: {a:,} of {n:,}')
        print(f'  max |logit diff| at (0, 0) vs zero-shot: {r["check_zs_maxdiff"].max():.4f}; '
              f'at (1, 0) vs Ramen: {r["check_ramen_maxdiff"].max():.4f} (logit scale 100)')
        return all(a == n for a, n in checks.values())

    def landscape_records(self):
        """The chosen images' records, each with its outcome (from the record's own predictions)."""
        recs = self.stats[LANDSCAPE_RUN]['landscape']
        for r in recs:
            r['outcome'] = str(outcome([r['zs_pred'] == r['label']], [r['ramen_pred'] == r['label']])[0])
        return recs

    def text_feature(self):
        return self.stats[LANDSCAPE_RUN]['text_feature']
