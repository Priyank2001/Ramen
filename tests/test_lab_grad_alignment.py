"""RamenGradAlignment (ORACLE diagnostic) must be Ramen to the bit, and its CE gradient must be the real one.
Random images, 5 fake classes, real CLIP ViT-B/32. Run like tests/test_lab_invariants.py (needs a GPU)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ramen_lab.analysis.gradients import cosine, segment_cosines, sign_agreement  # noqa: E402
from tests.test_lab_invariants import B, DEVICE, FakeDomains, build  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def logits_of(method, data, with_labels):
    from ramen_lab.stream import StreamFactory
    out = []
    for image, label, _, _ in StreamFactory(B, 0, 0).mixed(data):
        image = image.to(DEVICE)
        out.append((method(image, labels=label.to(DEVICE)) if with_labels else method(image)).detach().float().cpu())
    return torch.cat(out)


def test_logits_identical_to_ramen_over_a_stream_with_a_filled_cache():
    data = FakeDomains()
    ramen = logits_of(build("Ramen", data), data, with_labels=False)
    probe = logits_of(build("RamenGradAlignment", data), data, with_labels=True)
    assert torch.equal(ramen, probe)


def test_without_labels_it_is_plain_ramen_and_records_nothing():
    data = FakeDomains()
    method = build("RamenGradAlignment", data)
    assert torch.equal(logits_of(build("Ramen", data), data, False), logits_of(method, data, False))
    assert method.probe_arrays() == {}


def test_predictions_do_not_depend_on_which_labels_the_probe_reads():
    a, b = FakeDomains(label_seed=0), FakeDomains(label_seed=1)
    assert torch.equal(logits_of(build("RamenGradAlignment", a), a, True),
                       logits_of(build("RamenGradAlignment", b), b, True))


def test_records_one_row_per_sample_with_every_field():
    data = FakeDomains()
    method = build("RamenGradAlignment", data)
    logits_of(method, data, True)
    arrays = method.probe_arrays()
    n = len(logits_of(build("Ramen", data), data, False))
    assert all(len(v) == n for v in arrays.values())
    assert arrays["layer_cos_R_CE"].shape == (n, len(method.layer_sizes))
    assert np.isfinite(arrays["ce_before"]).all() and np.isfinite(arrays["ce_after"]).all()


def test_ce_gradient_matches_a_single_sample_backward():
    """Rows of the batched CE gradient equal the gradient of each sample processed alone (by-sample parameters)."""
    data = FakeDomains()
    method = build("RamenGradAlignment", data)
    image, label = data.datasets[0][0][0][None].to(DEVICE), torch.tensor([2], device=DEVICE)
    batch = torch.cat([image, data.datasets[0][1][0][None].to(DEVICE)])
    method._labels = torch.tensor([2, 0], device=DEVICE)
    logits = method.model.classify(method.model.featurize(batch))
    batched = method._seen["g_CE"][0]
    method._labels = None
    method.model.reset_parameters()
    single = method.model.classify(method.model.featurize(image))
    F.cross_entropy(single.float(), label, reduction="sum").backward()
    alone = method.model.get_by_sample_grad()[0].float()
    assert cosine(batched[None], alone[None]).item() > 0.99


def test_cosine_helpers():
    a = torch.tensor([[1., 0., 2., 0.], [0., 0., 0., 0.]])
    b = torch.tensor([[1., 0., -2., 0.], [1., 1., 1., 1.]])
    c = cosine(a, b)
    assert abs(c[0].item() - (-3 / 5)) < 1e-6 and torch.isnan(c[1])
    s = segment_cosines(a[:1], b[:1], [2, 2])
    assert abs(s[0, 0].item() - 1.0) < 1e-6 and abs(s[0, 1].item() + 1.0) < 1e-6
    assert sign_agreement(a[:1], b[:1]).item() == 0.5
