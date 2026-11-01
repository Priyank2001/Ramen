"""ramen_lab units that need no GPU: registry, config, stream, memory guard, records, stats.

Run: PYTHONNOUSERSITE=1 ~/priyank/Projects/RAMEN_ENV/bin/python -m pytest tests/test_lab_units.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ramen_lab.analysis.stats import cluster_bootstrap, noise_floor, outcome_counts, paired_gain  # noqa: E402
from ramen_lab.config import MAX_WORKERS, RunConfig  # noqa: E402
from ramen_lab.records import RecordWriter, accuracy_metrics, load_records, write_metrics  # noqa: E402
from ramen_lab.registry import Registry  # noqa: E402
from ramen_lab.resources import MemoryExceeded, MemoryGuard, ResourceBudget, process_tree_gb  # noqa: E402
from ramen_lab.stream import StreamFactory  # noqa: E402


# ---------------------------------------------------------------- registry
def test_registry_register_get_and_unknown_name():
    reg = Registry("thing")

    @reg.register("a")
    class A:
        pass

    assert reg.get("a") is A and "a" in reg and reg.names() == ["a"]
    with pytest.raises(KeyError, match="registered: a"):
        reg.get("b")
    with pytest.raises(ValueError):
        reg.add("a", object)


def test_upstream_methods_are_registered():
    from ramen_lab.methods import METHODS
    assert {"NoAdapt", "Tent", "Ramen"} <= set(METHODS.names())


# ---------------------------------------------------------------- config
def test_config_validates_and_is_frozen():
    c = RunConfig.make("CIFAR10C", batch_size=64, seed=2, overrides={"topk": 5})
    assert c.run_id("pc2") == "pc2_C10_Ramen_b64_s2" and c.to_dict()["overrides"] == {"topk": 5}
    assert RunConfig.make("CIFAR10C", batch_size=64, n_per_domain=200).run_id("pc2").endswith("_n200")
    with pytest.raises(Exception):
        c.seed = 3  # frozen
    for bad in ({"dataset": "MNIST", "batch_size": 8}, {"dataset": "CIFAR10C", "batch_size": 8, "tta_mode": "x"},
                {"dataset": "CIFAR10C", "batch_size": 8, "num_workers": MAX_WORKERS + 1}):
        with pytest.raises(ValueError):
            RunConfig(**bad)


def test_config_without_frozen_batch_needs_explicit_batch():
    from ramen_lab.config import DATASETS
    if DATASETS["DomainNet"].batch_size is None:
        with pytest.raises(ValueError, match="no frozen batch"):
            RunConfig("DomainNet")


# ---------------------------------------------------------------- stream
class _Domains:
    """Two fake domains with 10 and 7 samples."""
    def __init__(self):
        self.datasets = [TensorDataset(torch.arange(10.0), torch.arange(10) % 3),
                         TensorDataset(torch.arange(7.0) + 100, torch.arange(7) % 3)]


def _order(factory, data):
    return [(int(d), int(s)) for _, _, dom, smp in factory.mixed(data) for d, s in zip(dom, smp)]


def test_stream_same_seed_same_order_and_each_sample_once():
    data = _Domains()
    a, b = _order(StreamFactory(4, 0, 0), data), _order(StreamFactory(4, 0, 0), data)
    assert a == b and len(a) == 17 and len(set(a)) == 17
    assert a != _order(StreamFactory(4, 1, 0), data)


def test_stream_subset_is_fixed_across_seeds():
    data = _Domains()
    s0, s1 = set(_order(StreamFactory(4, 0, 0, n_per_domain=5), data)), set(_order(StreamFactory(4, 1, 0, 5), data))
    assert s0 == s1 and len(s0) == 10 and sum(d == 0 for d, _ in s0) == 5


def test_stream_single_gives_one_loader_per_domain():
    loaders = StreamFactory(4, 0, 0).single(_Domains())
    assert [len(l.dataset) for l in loaders] == [10, 7]


# ---------------------------------------------------------------- memory guard
def test_guard_trips_over_budget_and_raises_at_check():
    hard = []
    guard = MemoryGuard(ResourceBudget(rss_gb=10, reserve_gb=16), sampler=lambda: (11.0, 100.0),
                        on_hard_limit=hard.append)
    guard.sample()
    with pytest.raises(MemoryExceeded, match="budget"):
        guard.check()
    assert not hard and guard.peak_gb == 11.0


def test_guard_trips_when_system_low_and_hard_stops_far_over():
    low = MemoryGuard(ResourceBudget(rss_gb=10, reserve_gb=16), sampler=lambda: (1.0, 12.0), on_hard_limit=print)
    low.sample()
    with pytest.raises(MemoryExceeded, match="available"):
        low.check()
    hard = []
    far = MemoryGuard(ResourceBudget(rss_gb=10), sampler=lambda: (13.0, 100.0), on_hard_limit=hard.append)
    far.sample()
    assert hard, "1.3 x budget must hard-stop"


def test_guard_quiet_under_budget_and_measures_real_process():
    with MemoryGuard(ResourceBudget(rss_gb=500, reserve_gb=0.5, interval_s=0.05)) as guard:
        x = np.ones(50 * 2 ** 20 // 8)  # 50 MB, so the peak is above zero
        guard.check()
    assert 0 < guard.peak_gb < 500 and x.sum() > 0
    assert process_tree_gb() > 0


# ---------------------------------------------------------------- records + stats
def test_records_metrics_and_schema(tmp_path):
    w = RecordWriter(n_total=6, num_classes=3, out_dir=tmp_path / "run", capture_probs=True)
    logits = torch.tensor([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
    w.add(logits, torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0]))
    w.add(logits, torch.tensor([0, 0, 2]), torch.tensor([1, 1, 0]), torch.tensor([1, 2, 2]))
    assert w.correct().tolist() == [True, True, False, True, False, True]
    rec = load_records(Path(w.save()).parent)
    assert rec["batch"].tolist() == [0, 0, 0, 1, 1, 1] and np.load(tmp_path / "run" / "probs.npy").shape == (6, 3)
    m = accuracy_metrics(w.correct(), rec["domain_idx"], ["a", "b"])
    assert m["per_domain"] == {"a": 100.0, "b": pytest.approx(100 / 3, abs=1e-3)} and m["n_correct"] == 4
    path = write_metrics(tmp_path, "pc2_test_s0", __file__, config={"seed": 0}, metrics=m,
                         resources={"peak_ram_gb": 1.0})
    d = json.loads(path.read_text())
    assert {"run_id", "git_hash", "config", "metrics", "resources", "raw_outputs"} <= set(d)


def test_stats():
    assert noise_floor([72.5, 72.6, 72.7]) == pytest.approx(0.4)
    g = paired_gain([1.0, 2.0, 3.0], [0.5, 2.5, 2.0], floor=0.3)
    assert g.n_positive == 2 and g.mean == pytest.approx(1 / 3) and g.clears_floor
    assert outcome_counts(np.array([1, 0, 1, 0]), np.array([0, 1, 1, 0])) == \
        {"fixed": 1, "broken": 1, "both right": 1, "both wrong": 1}
    diff = np.r_[np.ones(50), np.zeros(50)]
    low, high = cluster_bootstrap(diff, np.zeros(100, int), np.arange(100))
    assert low < 50 < high
