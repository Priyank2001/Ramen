"""RamenFileByLabel (ORACLE experiment, docs/plans/oracle_fill.md).

The "prediction" setting must be upstream Ramen to the bit and must never read labels. The "label" setting must file every
entry under its true label and refuse to run without labels. Random images, real CLIP ViT-B/32 (needs a GPU), like
tests/test_lab_invariants.py.
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
from tests.test_lab_invariants import B, DEVICE, SMALL_RAMEN, FakeDomains, build  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def build_variant(data, filing: str, probe_dir=None):
    upstream.seed_everything(0)
    model, _ = upstream.load_model("clip_vitbase32", DEVICE)
    cfg = {**SMALL_RAMEN, "filing": filing, **({"probe_dir": str(probe_dir)} if probe_dir else {})}
    return METHODS.get("RamenFileByLabel")(model, data, upstream.method_args(cfg, DEVICE, B))


def stream(method, data, with_labels: bool):
    """Logits and labels in arrival order; labels are passed only when `with_labels` (as the runner does for ORACLEs)."""
    logits, labels = [], []
    for image, label, _, _ in StreamFactory(B, 0, 0).mixed(data):
        image = image.to(DEVICE)
        out = method(image, labels=label.to(DEVICE)) if with_labels else method(image)
        logits.append(out.detach().float().cpu())
        labels.append(label)
    return torch.cat(logits), torch.cat(labels).numpy()


def test_prediction_filing_is_upstream_ramen_to_the_bit():
    data = FakeDomains()
    ramen, _ = stream(build("Ramen", data), data, with_labels=False)
    variant, _ = stream(build_variant(data, "prediction"), data, with_labels=True)
    assert torch.equal(ramen, variant)


def test_prediction_filing_does_not_read_labels():
    a, b = FakeDomains(label_seed=0), FakeDomains(label_seed=1)
    la, _ = stream(build_variant(a, "prediction"), a, with_labels=True)
    lb, _ = stream(build_variant(b, "prediction"), b, with_labels=True)
    assert torch.equal(la, lb)


def test_label_filing_refuses_to_run_without_labels():
    data = FakeDomains()
    method = build_variant(data, "label")
    image = data.datasets[0][0][0][None].repeat(B, 1, 1, 1).to(DEVICE)
    with pytest.raises(ValueError, match="ORACLE"):
        method(image)


def test_label_filing_puts_every_entry_under_its_true_label():
    data = FakeDomains()
    method = build_variant(data, "label")
    _, labels = stream(method, data, with_labels=True)
    for queue, cache in enumerate(method.cache):
        positions = cache.positions[:cache.size].cpu().numpy()
        assert len(positions) > 0
        assert (labels[positions] == queue).all(), f"queue {queue} holds another class's sample"


def test_prediction_filing_puts_every_entry_under_its_prediction():
    data = FakeDomains()
    method = build_variant(data, "prediction")
    stream(method, data, with_labels=True)
    for queue, cache in enumerate(method.cache):
        positions = cache.positions[:cache.size]
        assert (cache.targets[:cache.size] == queue).all() and (positions >= 0).all()


def test_retrieval_log_is_consistent_and_flushed_on_reset(tmp_path):
    data = FakeDomains()
    method = build_variant(data, "label", probe_dir=tmp_path)
    _, labels = stream(method, data, with_labels=True)
    method.reset()
    snap = np.load(tmp_path / "cache_snapshot.npz")
    log = np.load(tmp_path / "retrieval.npz")
    n = len(labels)
    assert log["mass_queue"].shape == (n, method.num_classes)
    assert np.allclose(log["mass_queue"].sum(1), log["mass_target"].sum(1), rtol=1e-3)
    assert np.array_equal(log["query_pos"], np.arange(n))
    assert (log["top_pos"] >= 0).all() and (log["top_w"] >= 0).all()
    assert (snap["sizes"] <= SMALL_RAMEN["max_capacity"]).all()
    assert len(snap["queue"]) == snap["sizes"].sum()

