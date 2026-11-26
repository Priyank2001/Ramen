"""Invariants every registered TTA method must keep (AGENTS.md §4.5), parametrized over `METHODS`, so a new variant is
covered the moment it is registered. Random images, 5 fake classes, a real CLIP ViT-B/32, no datasets needed.

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
         ~/priyank/Projects/RAMEN_ENV/bin/python -m pytest tests/test_lab_invariants.py -q
Invariants:
  1. determinism:         the same seed gives bit-identical predictions in two fresh runs;
  2. reset:               after reset(), driving the same stream again gives the same predictions as the first pass
                          (cache, counters, weights and optimizer state are all back to the start);
  3. frozen weights:      parameters that are not trainable are bit-identical after a stream; trainable ones are
                          restored once the method has been reset;
  4. label independence:  a method without `uses_labels = True` gives the same predictions whatever the labels are
                          (the harness makes sure the two label sets really differ, so this is not vacuous);
  5. oracle detection:    a method that declares `uses_labels` is the only one the runner hands labels to
                          (covered by test_lab_runner.py::test_labels_reach_only_methods_that_declare_them).
A variant whose yaml differs from Ramen's must add its small test config to `TEST_CONFIGS`.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ramen_lab import upstream  # noqa: E402
from ramen_lab.methods import METHODS, uses_labels  # noqa: E402
from ramen_lab.stream import StreamFactory  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

B = 32  # >25 rows: torch.cdist then takes the mm path (no fp16 CUDA direct kernel in torch 2.0.1)
CLASSES = ["cat", "dog", "car", "tree", "house"]
DEVICE = torch.device("cuda")
SMALL_RAMEN = {"max_capacity": 20, "topk": 5, "beta": 5.0, "optimizer": "signsgd", "lr": 0.01}
TEST_CONFIGS = {"NoAdapt": upstream.load_method_config("CIFAR10C", "NoAdapt"),
                "Tent": upstream.load_method_config("CIFAR10C", "Tent"), "Ramen": SMALL_RAMEN,
                "RamenMultiCache": SMALL_RAMEN,  # not a Ramen subclass (builds its own memories), same small config
                "RamenStoredValue": {**SMALL_RAMEN, "stored_value": "ce_pseudo", "exclude_self": True},
                "RamenStoredValueOracle": {**SMALL_RAMEN, "stored_value": "ce_true", "filing": "true_label",
                                           "exclude_self": True}}  # ORACLE setting: the invariants hold for it too


def test_config_for(name: str) -> dict:
    """Ramen variants default to Ramen's small config; anything else must be listed in TEST_CONFIGS."""
    if name in TEST_CONFIGS:
        return dict(TEST_CONFIGS[name])
    if issubclass(METHODS.get(name), upstream.method_class("Ramen")):
        return dict(SMALL_RAMEN)
    raise KeyError(f"add a small test config for method {name!r} to TEST_CONFIGS in tests/test_lab_invariants.py")


test_config_for.__test__ = False  # not a test, despite the name


class FakeDomains:
    """Two domains of random 224 x 224 images, 48 each; `label_seed` changes only the labels."""

    def __init__(self, label_seed=0, n=48):
        g = torch.Generator().manual_seed(0)
        lg = torch.Generator().manual_seed(100 + label_seed)
        self.datasets = [TensorDataset(torch.randn(n, 3, 224, 224, generator=g),
                                       torch.randint(len(CLASSES), (n,), generator=lg)) for _ in range(2)]
        self.environments, self.classes, self.num_classes = ["dom_a", "dom_b"], CLASSES, len(CLASSES)


def build(name: str, data):
    upstream.seed_everything(0)
    model, _ = upstream.load_model("clip_vitbase32", DEVICE)
    return METHODS.get(name)(model, data, upstream.method_args(test_config_for(name), DEVICE, B))


def drive(method, data, seed: int = 0):
    """Predictions (and the labels the stream carried) of one pass over the stream. Labels reach the method only if it
    declares uses_labels, exactly as the runner does."""
    preds, labels = [], []
    for image, label, _, _ in StreamFactory(B, seed, 0).mixed(data):
        image = image.to(DEVICE)
        logits = method(image, labels=label.to(DEVICE)) if uses_labels(method) else method(image)
        preds.append(logits.detach().float().argmax(1).cpu().numpy())
        labels.append(label.numpy())
    return np.concatenate(preds), np.concatenate(labels)


NAMES = METHODS.names()


@pytest.mark.parametrize("name", NAMES)
def test_same_seed_same_predictions(name):
    data = FakeDomains()
    a, _ = drive(build(name, data), data)
    b, _ = drive(build(name, data), data)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", NAMES)
def test_reset_returns_to_the_start(name):
    data = FakeDomains()
    method = build(name, data)
    first, _ = drive(method, data)
    method.reset()
    again, _ = drive(method, data)
    assert np.array_equal(first, again), f"{name}: a reset method does not behave like a fresh one"


@pytest.mark.parametrize("name", NAMES)
def test_frozen_weights_unchanged_and_trainable_restored_after_reset(name):
    data = FakeDomains()
    method = build(name, data)
    net = next(m for m in (getattr(method, "model", None), method) if m is not None and hasattr(m, "named_parameters"))
    start = {n: p.detach().clone() for n, p in net.named_parameters()}
    frozen = {n for n, p in net.named_parameters() if not p.requires_grad}
    drive(method, data)
    after = dict(net.named_parameters())
    assert all(torch.equal(start[n], after[n].detach()) for n in frozen), f"{name}: a frozen parameter changed"
    method.reset()
    after = dict(net.named_parameters())
    assert all(torch.equal(start[n], after[n].detach()) for n in start), f"{name}: reset left a weight changed"


@pytest.mark.parametrize("name", [n for n in NAMES if not getattr(METHODS.get(n), "uses_labels", False)])
def test_predictions_do_not_depend_on_labels(name):
    d0, d1 = FakeDomains(label_seed=0), FakeDomains(label_seed=1)
    p0, l0 = drive(build(name, d0), d0)
    p1, l1 = drive(build(name, d1), d1)
    assert not np.array_equal(l0, l1), "harness bug: the two label sets are identical, so the test would be vacuous"
    assert np.array_equal(p0, p1), f"{name}: predictions changed when only the labels changed"
