"""RamenStoredValue / RamenStoredValueOracle (docs/plans/oracle_value.md): Ramen with a configurable stored value,
filing and self-exclusion. Random images, 5 fake classes, real CLIP ViT-B/32. Run (needs a GPU):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
        ~/priyank/Projects/RAMEN_ENV/bin/python -m pytest tests/test_lab_stored_value.py -q
The shared invariants (determinism, reset, frozen weights, label independence) run in test_lab_invariants.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ramen_lab import upstream  # noqa: E402
from ramen_lab.methods import METHODS  # noqa: E402
from ramen_lab.methods.stored_value import drop_self  # noqa: E402
from ramen_lab.stream import StreamFactory  # noqa: E402
from tests.test_lab_invariants import B, DEVICE, FakeDomains  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# capacity 8 per class over 192 images: queues fill and evict
SMALL = {"max_capacity": 8, "topk": 5, "beta": 5.0, "optimizer": "signsgd", "lr": 0.01}
ORACLE = {"stored_value": "ce_true", "filing": "true_label", "exclude_self": True}


def build(name, data, **extra):
    upstream.seed_everything(0)
    model, _ = upstream.load_model("clip_vitbase32", DEVICE)
    return METHODS.get(name)(model, data, upstream.method_args({**SMALL, **extra}, DEVICE, B))


def drive(method, data, with_labels=False, batches=None):
    out = []
    for i, (image, label, _, _) in enumerate(StreamFactory(B, 0, 0).mixed(data)):
        if batches is not None and i >= batches:
            break
        image = image.to(DEVICE)
        out.append((method(image, labels=label) if with_labels else method(image)).detach().float().cpu())
    return torch.cat(out)


@pytest.fixture(scope="module")
def data():
    return FakeDomains(n=96)


@pytest.fixture(scope="module")
def ramen_logits(data):
    return drive(build("Ramen", data), data)


# ---------------------------------------------------------------------------- drop_self (pure)
def test_drop_self_removes_the_query_when_it_is_among_the_k_plus_one():
    found = torch.tensor([[7, 3, 9, 4], [1, 2, 3, 5]])  # query 3 at position 1; query 5 at the last position
    keep = drop_self(found, torch.tensor([3, 5]), k=3)
    assert keep.tolist() == [[True, False, True, True], [True, True, True, False]]


def test_drop_self_drops_the_extra_entry_when_the_query_is_absent():
    keep = drop_self(torch.tensor([[7, 3, 9, 4]]), torch.tensor([42]), k=3)
    assert keep.tolist() == [[True, True, True, False]]


def test_drop_self_keeps_a_short_queue_whole_except_the_query():
    found = torch.tensor([[7, 3], [7, 3]])  # queue of 2 entries, k = 5
    keep = drop_self(found, torch.tensor([3, 42]), k=5)
    assert keep.tolist() == [[True, False], [True, True]]


# ---------------------------------------------------------------------------- identity with Ramen
@needs_gpu
def test_default_is_bit_identical_to_upstream_ramen(data, ramen_logits):
    method = build("RamenStoredValue", data)
    assert torch.equal(drive(method, data), ramen_logits)
    assert method.counter > method.num_classes * SMALL["max_capacity"]
    assert any(c.size == c.max_capacity for c in method.queues), "no queue filled: eviction is untested"


@needs_gpu
def test_oracle_class_with_defaults_is_ramen_and_needs_no_labels(data, ramen_logits):
    assert torch.equal(drive(build("RamenStoredValueOracle", data), data), ramen_logits)


# ---------------------------------------------------------------------------- self-exclusion
def _manual_aggregate(method, feats, stream_id, exclude):
    """G_i recomputed from the queues' full sorted contents, dropping the query's own entry if `exclude`."""
    total = torch.zeros(len(feats), method.model.grad_dim, device=DEVICE)
    n_queues = 0
    for cache in method.queues:
        if cache.size == 0:
            continue
        rows, ent, dist = cache.query(feats, topk=cache.size)  # every entry, nearest first, same cdist as retrieval
        for i in range(len(feats)):
            ids = method.pool.stream_id[rows[i]]
            sel = (ids != stream_id[i]) if exclude else torch.ones_like(ids, dtype=torch.bool)
            sel_idx = sel.nonzero().squeeze(1)[:method.topk]
            w = torch.exp(-ent[i, sel_idx].float()) * torch.exp(-method.beta * dist[i, sel_idx].float())
            total[i] += (method.pool.values[rows[i, sel_idx]].float() * w[:, None]).sum(0)
        n_queues += 1
    return total / n_queues


@needs_gpu
@pytest.mark.parametrize("exclude", [False, True])
def test_retrieval_matches_a_manual_recomputation(data, exclude):
    method = build("RamenStoredValue", data, exclude_self=exclude)
    drive(method, data, batches=2)  # memory holds 64 entries
    image = next(iter(StreamFactory(B, 1, 0).mixed(data)))[0].to(DEVICE)
    feats = method.model.featurize(image)
    logits = method.model.classify(feats)
    zs_pred = logits.argmax(-1)
    values = method.compute_values(logits, zs_pred, None)
    with torch.no_grad():
        stream_id = torch.arange(method.counter, method.counter + B, device=DEVICE)
        method.write(feats, values, logits, zs_pred, None, stream_id)  # the batch is in memory: self is retrievable
        got, stats = method.retrieve(feats, values, stream_id)
        want = _manual_aggregate(method, feats, stream_id, exclude)
    method.model.reset_parameters()
    assert torch.allclose(got.float(), want, rtol=2e-2, atol=1e-3)
    stored = torch.cat([method.pool.stream_id[c.rows[:c.size]] for c in method.queues])
    in_memory = torch.isin(stream_id, stored)  # small queues: later batch-mates can evict earlier ones
    assert in_memory.any()
    assert (stats["self_share"][in_memory] > 0).all(), "a stored query is at distance 0: Ramen's top-k contains it"
    assert (stats["self_share"][~in_memory] == 0).all()


@needs_gpu
def test_excluding_self_changes_predictions(data, ramen_logits):
    assert not torch.equal(drive(build("RamenStoredValue", data, exclude_self=True), data), ramen_logits)


# ---------------------------------------------------------------------------- stored values
def _batch(data):
    image, label, _, _ = next(iter(StreamFactory(B, 0, 0).mixed(data)))
    return image.to(DEVICE), label.to(DEVICE)


@needs_gpu
def test_ce_true_value_is_each_samples_own_cross_entropy_gradient(data):
    method = build("RamenStoredValueOracle", data, **ORACLE)
    image, label = _batch(data)
    logits = method.model.classify(method.model.featurize(image))
    values = method.compute_values(logits, logits.argmax(-1), label)
    method.model.reset_parameters()
    params = [p for m in method.model.model.modules() if hasattr(m, "weight_by_sample")
              for p in (m.weight_by_sample, m.bias_by_sample)]
    for b in (0, 5, B - 1):
        logits_b = method.model.classify(method.model.featurize(image))
        loss = F.cross_entropy(logits_b[b:b + 1].float(), label[b:b + 1], reduction="sum")
        grads = torch.autograd.grad(loss, params)
        own = torch.cat([g[b] for g in grads])
        assert torch.allclose(values[b].float(), own.float(), rtol=1e-3, atol=1e-6)
        assert all(g[[i for i in range(B) if i != b]].abs().max() == 0 for g in grads), "gradient leaked across samples"


@needs_gpu
def test_ce_pseudo_is_ce_true_with_the_zero_shot_label(data):
    pseudo = build("RamenStoredValue", data, stored_value="ce_pseudo")
    oracle = build("RamenStoredValueOracle", data, stored_value="ce_true")
    image, _ = _batch(data)
    out = []
    for method in (pseudo, oracle):
        logits = method.model.classify(method.model.featurize(image))
        zs = logits.argmax(-1)
        out.append(method.compute_values(logits, zs, zs if method is oracle else None))
        method.model.reset_parameters()
    assert torch.equal(out[0], out[1])


@needs_gpu
def test_a_non_finite_stored_value_fails_loudly(data):
    method = build("RamenStoredValue", data, stored_value="ce_pseudo", value_loss_scale=1e30)
    with pytest.raises(FloatingPointError):
        drive(method, data, batches=1)


# ---------------------------------------------------------------------------- labels
@needs_gpu
@pytest.mark.parametrize("extra", [{"stored_value": "ce_true"}, {"filing": "true_label"}])
def test_label_free_class_refuses_label_settings(data, extra):
    with pytest.raises(ValueError, match="ORACLE"):
        build("RamenStoredValue", data, **extra)


@needs_gpu
def test_oracle_setting_needs_labels(data):
    method = build("RamenStoredValueOracle", data, **ORACLE)
    with pytest.raises(ValueError, match="labels"):
        drive(method, data, batches=1)


@needs_gpu
def test_oracle_predictions_depend_on_labels(data):
    d1 = FakeDomains(n=96, label_seed=1)
    a = drive(build("RamenStoredValueOracle", data, **ORACLE), data, with_labels=True)
    b = drive(build("RamenStoredValueOracle", d1, **ORACLE), d1, with_labels=True)
    assert not torch.equal(a, b)


@needs_gpu
def test_label_free_settings_ignore_labels_even_when_given(data, ramen_logits):
    method = build("RamenStoredValueOracle", data)  # defaults: nothing reads labels
    assert torch.equal(drive(method, data, with_labels=True), ramen_logits)


# ---------------------------------------------------------------------------- memory contents and log
@needs_gpu
@pytest.mark.parametrize("name,extra", [("RamenStoredValue", {}), ("RamenStoredValueOracle", ORACLE)])
def test_every_entry_pushes_toward_its_own_queue_when_value_and_filing_agree(data, name, extra):
    method = build(name, data, **extra)
    drive(method, data, with_labels=method.uses_labels)
    for c, cache in enumerate(method.queues):
        rows = cache.rows[:cache.size]
        assert (method.push_of_row[rows] == c).all()
        if extra:
            assert (method.pool.label[rows] == c).all(), "true-label filing put an entry in the wrong queue"


@needs_gpu
def test_log_is_written_on_reset_and_cleared(data, tmp_path):
    path = tmp_path / "stored_value.npz"
    method = build("RamenStoredValue", data, log_path=str(path))
    drive(method, data)
    method.reset()
    log = np.load(path)
    n = 2 * 96
    assert log["zs_pred"].shape == (n,) and log["push_top_cls"].shape == (n, 5)
    assert np.all(log["push_top_share"].astype(float).sum(1) <= 1.0 + 1e-2)
    assert np.all((log["self_share"] >= 0) & (log["self_share"] <= 1)) and np.any(log["self_share"] > 0)
    assert np.all(log["value_zero_frac"] < 1)
    assert method.chunks == {} and method.pool.in_use == 0 and method.counter == 0
