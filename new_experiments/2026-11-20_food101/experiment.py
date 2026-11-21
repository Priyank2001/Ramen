"""
Food101Experiment: the repo harness (new_experiments/ramen_analysis.py) on Food-101 without editing src/ or cfg/.

    from experiment import make_food101
    exp = make_food101(seed=0)          # CLIP ViT-B/16, one shuffled stream of the 25,250 test images, batches of 100
    exp.run('NoAdapt'); exp.run('Ramen')

Three things differ from the harness's other datasets, each in one place below:
  - the dataset class is this folder's Food101 (the repo's get_dataset_class does not know it)
  - configs come from this folder's cfg/Food101/, falling back to the repo's cfg/ (so NoAdapt uses cfg/default)
  - data_root defaults to the HDD copy, /data/priyank1/data
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # new_experiments/, for ramen_analysis

from ramen_analysis import DomainNetExperiment  # noqa: E402
from food101 import Food101  # noqa: E402

DATA_ROOT = '/data/priyank1/data'
LOCAL_CFG_DIR = os.path.join(HERE, 'cfg')


class Food101Experiment(DomainNetExperiment):
    """Food-101 test split as a single-domain stream (like EuroSAT: natural photos, fine-grained classes)."""
    dataset_name = 'Food101'
    default_model = 'clip_vitbase16'
    env_label = 'domain'

    def _build_datasets(self, preprocess):
        datasets = Food101(root=self.data_root, transform=preprocess)
        if self.n_per_domain:
            rng = np.random.default_rng(self.seed)
            for ds in datasets.datasets:
                n = len(ds)
                self._keep_samples(ds, np.sort(rng.choice(n, min(self.n_per_domain, n), replace=False)))
        return datasets

    def _load_cfg(self, algo):
        local = os.path.join(LOCAL_CFG_DIR, self.dataset_name, f'{algo}.yaml')
        return super()._load_cfg(algo) if not os.path.exists(local) else self._read_yaml(local)

    @staticmethod
    def _read_yaml(path):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)


def make_food101(**kwargs):
    """Food-101 experiment with the harness's usual settings (mixed stream, batch 100, 16 workers) and the HDD data root."""
    kwargs.setdefault('data_root', DATA_ROOT)
    kwargs.setdefault('tta_mode', 'mixed')
    kwargs.setdefault('batch_size', 100)
    kwargs.setdefault('num_workers', 16)
    return Food101Experiment(**kwargs).load()
