"""
Aircraft / Flowers102 on the repo harness (new_experiments/ramen_analysis.py) without editing src/ or cfg/.

    from fine_grained_experiment import make_fine_grained
    exp = make_fine_grained('Aircraft', seed=0)    # or 'Flowers102'; CLIP ViT-B/16, one shuffled stream of the test split

What differs from the harness's other datasets, each in one place below: the dataset class (fine_grained_loaders.py),
configs read from cfg/<dataset>/ first then the repo's cfg/, and the data root on the HDD.
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # new_experiments/, for ramen_analysis
sys.path.insert(0, HERE)

from ramen_analysis import DomainNetExperiment  # noqa: E402
from fine_grained_loaders import Aircraft, Flowers102  # noqa: E402

DATA_ROOT = '/data/priyank1/data'
LOCAL_CFG_DIR = os.path.join(HERE, 'cfg')


class _FineGrainedExperiment(DomainNetExperiment):
    """A single-domain test split read from disk. Subclasses set dataset_name and loader."""
    default_model = 'clip_vitbase16'
    loader = None

    def _build_datasets(self, preprocess):
        datasets = self.loader(root=self.data_root, transform=preprocess)
        if self.n_per_domain:
            rng = np.random.default_rng(self.seed)
            for ds in datasets.datasets:
                n = len(ds)
                self._keep_samples(ds, np.sort(rng.choice(n, min(self.n_per_domain, n), replace=False)))
        return datasets

    def _load_cfg(self, algo):
        local = os.path.join(LOCAL_CFG_DIR, self.dataset_name, f'{algo}.yaml')
        if not os.path.exists(local):
            return super()._load_cfg(algo)  # the repo's cfg/<dataset>/ then cfg/default/
        with open(local) as f:
            return yaml.safe_load(f)


class AircraftExperiment(_FineGrainedExperiment):
    """FGVC-Aircraft, 100 variants, 3,333 test images."""
    dataset_name = 'Aircraft'
    loader = Aircraft


class Flowers102Experiment(_FineGrainedExperiment):
    """Oxford Flowers102, 102 classes, 6,149 test images."""
    dataset_name = 'Flowers102'
    loader = Flowers102


EXPERIMENTS = {'Aircraft': AircraftExperiment, 'Flowers102': Flowers102Experiment}


def make_fine_grained(dataset, **kwargs):
    """Experiment with the harness's usual settings (mixed stream, batch 100, 16 workers) and the HDD data root."""
    kwargs.setdefault('data_root', DATA_ROOT)
    kwargs.setdefault('tta_mode', 'mixed')
    kwargs.setdefault('batch_size', 100)
    kwargs.setdefault('num_workers', 16)
    return EXPERIMENTS[dataset](**kwargs).load()
