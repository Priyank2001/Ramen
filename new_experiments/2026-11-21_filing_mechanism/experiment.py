"""
E2 harness: the repo's experiment classes (new_experiments/ramen_analysis.py), with this folder's methods.

    exp = make_local_experiment('DomainNet', seed=0).load()
    exp.run('RamenHideConfusion', name='hide_samezs_wrong_true', cfg=VARIANTS['hide_samezs_wrong_true'][1])
"""
import os
import sys
from copy import deepcopy
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # new_experiments/, for ramen_analysis (puts src/ on the path)

from ramen_analysis import EXPERIMENTS  # noqa: E402
from registry import config_name, get_method_class  # noqa: E402

# model + num_workers of the cached NoAdapt / Ramen / oracle runs (num_workers sets the stream order: keep it)
DATASETS = {'DomainNet': dict(model='clip_vitbase32', num_workers=16),
            'ImageNetC5K': dict(model='clip_vitbase16', num_workers=16),
            'EuroSAT': dict(model='clip_vitbase16', num_workers=16)}
# cached true-label-filing ORACLE run (entropy values) per dataset; all equal RamenTrueLabel filing true_label
ORACLE_RUN = {'DomainNet': 'oracle_file_true', 'ImageNetC5K': 'oracle_file_true', 'EuroSAT': 'labeloracle_true_entropy'}

TRUE, ZS = {'filing': 'true_label'}, {'filing': 'zero_shot'}
# name -> (algo, cfg overrides on cfg/<dataset>/Ramen.yaml). A-B is the causal test; C, D are label-free rules.
VARIANTS = {
    'hide_samezs_wrong_true': ('RamenHideConfusion', {**TRUE, 'hide': 'same_zs', 'hide_on': 'zs_wrong'}),  # A ORACLE
    'hide_random_wrong_true': ('RamenHideConfusion', {**TRUE, 'hide': 'random', 'hide_on': 'zs_wrong'}),   # B ORACLE
    'hide_samezs_all_true': ('RamenHideConfusion', {**TRUE, 'hide': 'same_zs', 'hide_on': 'all'}),         # C ORACLE
    'hide_samezs_all_zs': ('RamenHideConfusion', {**ZS, 'hide': 'same_zs', 'hide_on': 'all'}),             # D no labels
}
# E3: widen the hidden set to the query's top-m zero-shot classes (ORACLE gate on zs-wrong queries, true-label filing),
# each with a matched-count random control. m = 1 is run A of E2.
for _m in (3, 10):
    VARIANTS[f'hide_top{_m}_wrong_true'] = ('RamenHideConfusion', {**TRUE, 'hide': 'same_zs', 'hide_on': 'zs_wrong',
                                                                    'hide_top_m': _m})
    VARIANTS[f'hide_rand{_m}_wrong_true'] = ('RamenHideConfusion', {**TRUE, 'hide': 'random', 'hide_on': 'zs_wrong',
                                                                     'hide_top_m': _m})
E2_VARIANTS = ('hide_samezs_wrong_true', 'hide_random_wrong_true', 'hide_samezs_all_true', 'hide_samezs_all_zs')
SMOKE_PAIRS = {  # (new run, reference run): must give identical predictions
    'hide none, true filing == RamenTrueLabel true_label': (('RamenHideConfusion', {**TRUE, 'hide': 'none'}),
                                                            ('RamenTrueLabel', dict(TRUE))),
    'hide none, zero-shot filing == Ramen': (('RamenHideConfusion', {**ZS, 'hide': 'none'}), ('Ramen', {})),
}


class LocalMethodsMixin:
    """Lets the harness build this folder's methods; every other behaviour is the harness's own."""

    def build_method(self, algo, cfg=None):
        # Same as RamenExperiment.build_method; only the class and config lookups differ
        args = SimpleNamespace(config={**self._load_cfg(config_name(algo)), **(cfg or {})}, device=self.device,
                               max_batch_size=self.batch_size, use_tbn=False)
        return get_method_class(algo)(deepcopy(self.clip), self.datasets, args)


def make_local_experiment(dataset, seed=0, n_per_domain=None):
    base = EXPERIMENTS[dataset]
    cls = type(f'Local{base.__name__}', (LocalMethodsMixin, base), {})
    return cls(tta_mode='mixed', batch_size=100, n_per_domain=n_per_domain, seed=seed, **DATASETS[dataset])
