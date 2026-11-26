"""RamenStepProbe (docs/plans/step_size.md): Ramen plus its prediction at every step size in a grid. Random images,
5 fake classes, real CLIP ViT-B/32. Run (needs a GPU):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
        ~/priyank/Projects/RAMEN_ENV/bin/python -m pytest tests/test_lab_step_probe.py -q
The shared invariants (determinism, reset, frozen weights, label independence) run in test_lab_invariants.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ramen_lab import upstream  # noqa: E402
from ramen_lab.methods import METHODS  # noqa: E402
from ramen_lab.stream import StreamFactory  # noqa: E402
from tests.test_lab_invariants import B, DEVICE, FakeDomains  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# capacity 8 per class over 192 images: queues fill and evict
SMALL = {"max_capacity": 8, "topk": 5, "beta": 5.0, "optimizer": "signsgd", "lr": 0.01}
ETAS = [0.0, 0.0025, 0.01, 0.04, 0.08]


def build(name, data, **extra):
    upstream.seed_everything(0)
    model, _ = upstream.load_model("clip_vitbase32", DEVICE)
    return METHODS.get(name)(model, data, upstream.method_args({**SMALL, **extra}, DEVICE, B))


def drive(method, data):
    out = []
    for image, _, _, _ in StreamFactory(B, 0, 0).mixed(data):
        out.append(method(image.to(DEVICE)).detach().float().cpu())
    return torch.cat(out)


@pytest.fixture(scope="module")
def data():
    return FakeDomains(n=96)


@pytest.fixture(scope="module")
def ramen_logits(data):
    return drive(build("Ramen", data), data)


@pytest.fixture(scope="module")
def probe(data):
    method = build("RamenStepProbe", data, probe_etas=ETAS)
    return method, drive(method, data), method.arrays()


def test_returned_logits_are_bit_identical_to_upstream_ramen(probe, ramen_logits):
    method, logits, _ = probe
    assert torch.equal(logits, ramen_logits)
    assert any(c.size == c.max_capacity for c in method.queues), "no queue filled: eviction is untested"


def test_ramen_column_is_the_returned_prediction_and_eta_zero_is_zero_shot(probe):
    _, logits, log = probe
    assert np.array_equal(log["eta_pred"][:, ETAS.index(0.01)], logits.argmax(-1).numpy())
    assert np.array_equal(log["eta_pred"][:, ETAS.index(0.0)], log["zs_pred"])


def test_steps_move_coordinates_and_eta_zero_moves_none(probe):
    _, _, log = probe
    moved = log["eta_moved"].astype(np.float32)
    assert (moved[:, ETAS.index(0.0)] == 0).all()
    assert (moved[:, ETAS.index(0.01)] > 0.5).all()


def test_a_larger_step_changes_some_prediction(probe):
    _, _, log = probe
    assert (log["eta_pred"][:, ETAS.index(0.08)] != log["eta_pred"][:, ETAS.index(0.01)]).any(), \
        "eta 0.08 never differs from 0.01: the extra passes may not step"


def test_lr_and_weights_are_restored_after_every_batch(probe):
    method, _, _ = probe
    assert all(g["lr"] == SMALL["lr"] for g in method.model.optimizer.param_groups)
    for module in method.model.model.modules():
        if hasattr(module, "weight_by_sample"):
            assert torch.equal(module.weight_by_sample, module.weight.expand_as(module.weight_by_sample))
            assert torch.equal(module.bias_by_sample, module.bias.expand_as(module.bias_by_sample))


def test_predictions_at_an_eta_do_not_depend_on_the_rest_of_the_grid(data, probe):
    _, _, full = probe
    small = build("RamenStepProbe", data, probe_etas=[0.04, 0.01])
    drive(small, data)
    assert np.array_equal(small.arrays()["eta_pred"][:, 0], full["eta_pred"][:, ETAS.index(0.04)])


def test_signals_are_logged_per_image_and_finite(probe):
    _, logits, log = probe
    n = logits.shape[0]
    for key in ("entropy", "margin", "norm_G", "norm_H", "cos_G_H", "weight_mass"):
        assert log[key].shape == (n,) and np.isfinite(log[key]).all(), key
    assert log["eta_pred"].shape == (n, len(ETAS)) and log["eta_moved"].shape == (n, len(ETAS))
    assert ((log["margin"] >= 0) & (log["margin"] <= 1)).all() and (np.abs(log["cos_G_H"]) <= 1 + 1e-5).all()


@pytest.mark.parametrize("bad", [{"probe_etas": [0.0, 0.02]}, {"probe_etas": [0.01, 0.01]},
                                 {"probe_etas": [-0.01, 0.01]}, {"optimizer": "sgd"}])
def test_constructor_refuses_bad_settings(data, bad):
    with pytest.raises(ValueError):
        build("RamenStepProbe", data, **bad)
