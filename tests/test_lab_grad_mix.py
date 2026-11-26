"""RamenGradMix (docs/plans/grad_mix.md): Ramen stepping on lambda g_R + (1 - lambda) g_H. Random images, 5 fake
classes, real CLIP ViT-B/32. Run (needs a GPU for the method tests):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
        ~/priyank/Projects/RAMEN_ENV/bin/python -m pytest tests/test_lab_grad_mix.py -q
The shared invariants (determinism, reset, frozen weights, label independence) run in test_lab_invariants.py.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ramen_lab import upstream  # noqa: E402
from ramen_lab.methods import METHODS  # noqa: E402
from ramen_lab.methods.grad_mix import MIXES  # noqa: E402
from ramen_lab.stream import StreamFactory  # noqa: E402
from tests.test_lab_invariants import B, DEVICE, FakeDomains  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# capacity 8 per class over 192 images: queues fill and evict
SMALL = {"max_capacity": 8, "topk": 5, "beta": 5.0, "optimizer": "signsgd", "lr": 0.01}


def build(name, data, **extra):
    upstream.seed_everything(0)
    model, _ = upstream.load_model("clip_vitbase32", DEVICE)
    return METHODS.get(name)(model, data, upstream.method_args({**SMALL, **extra}, DEVICE, B))


def drive(method, data, batches=None):
    out = []
    for i, (image, _, _, _) in enumerate(StreamFactory(B, 0, 0).mixed(data)):
        if batches is not None and i >= batches:
            break
        out.append(method(image.to(DEVICE)).detach().float().cpu())
    return torch.cat(out)


@pytest.fixture(scope="module")
def data():
    return FakeDomains(n=96)


@pytest.fixture(scope="module")
def ramen_logits(data):
    return drive(build("Ramen", data), data)


# ---------------------------------------------------------------------------- mixes (pure)
G_R = torch.tensor([[3.0, -1.0, 0.5, 0.0]])
G_H = torch.tensor([[-0.1, 2.0, 0.5, -4.0]])


@pytest.mark.parametrize("mode", ["raw", "unit"])
def test_mix_endpoints_give_the_sign_of_one_term(mode):
    assert torch.equal(MIXES.get(mode)(1.0).mix(G_R, G_H).sign(), G_R.sign())
    assert torch.equal(MIXES.get(mode)(0.0).mix(G_R, G_H).sign(), G_H.sign())


def test_raw_mix_is_the_literal_formula():
    assert torch.allclose(MIXES.get("raw")(0.25).mix(G_R, G_H), 0.25 * G_R + 0.75 * G_H)


def test_unit_mix_does_not_depend_on_the_scale_of_either_term():
    mix = MIXES.get("unit")(0.5)
    assert torch.allclose(mix.mix(G_R, G_H), mix.mix(1000 * G_R, 1e-3 * G_H))


def test_unit_mix_keeps_an_all_zero_gradient_finite():
    out = MIXES.get("unit")(0.5).mix(torch.zeros(1, 4), G_H)
    assert torch.isfinite(out).all() and torch.equal(out.sign(), G_H.sign())


# ---------------------------------------------------------------------------- the method
@needs_gpu
@pytest.mark.parametrize("mode", ["raw", "unit"])
def test_lambda_one_is_bit_identical_to_upstream_ramen(data, ramen_logits, mode):
    method = build("RamenGradMix", data, mix_lambda=1.0, mix_norm=mode)
    assert torch.equal(drive(method, data), ramen_logits)
    assert any(c.size == c.max_capacity for c in method.queues), "no queue filled: eviction is untested"


@needs_gpu
def test_default_is_ramen(data, ramen_logits):
    assert torch.equal(drive(build("RamenGradMix", data), data), ramen_logits)


@needs_gpu
def test_lambda_zero_ignores_memory(data):
    """lambda = 0 steps on the own gradient only: a batch's logits do not depend on what memory holds."""
    warm = build("RamenGradMix", data, mix_lambda=0.0)
    batches = list(StreamFactory(B, 0, 0).mixed(data))
    for image, _, _, _ in batches[:-1]:
        warm(image.to(DEVICE))
    last = batches[-1][0].to(DEVICE)
    cold = build("RamenGradMix", data, mix_lambda=0.0)
    assert warm.counter > cold.counter
    assert torch.equal(warm(last), cold(last))


@needs_gpu
def test_an_interior_mix_changes_ramen_and_logs_it(data, ramen_logits):
    method = build("RamenGradMix", data, mix_lambda=0.5, mix_norm="unit")
    logits = drive(method, data)
    assert not torch.equal(logits, ramen_logits)
    log = method.arrays()
    n = logits.shape[0]
    assert all(len(log[k]) == n for k in ("norm_R", "norm_H", "flip_frac", "zs_pred"))
    assert (log["flip_frac"] > 0).any() and (log["flip_frac"] <= 1).all()
    assert (log["norm_R"] > 0).all() and (log["norm_H"] > 0).all()


@needs_gpu
@pytest.mark.parametrize("bad", [{"mix_lambda": 1.5}, {"mix_lambda": -0.1}, {"mix_norm": "l1"},
                                 {"stored_value": "ce_pseudo"}, {"optimizer": "sgd"}])
def test_constructor_refuses_bad_settings(data, bad):
    with pytest.raises((ValueError, KeyError)):
        build("RamenGradMix", data, **bad)
