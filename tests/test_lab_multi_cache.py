"""RamenMultiCache (ORACLE: Ramen + memories filed by true label / at random) and its pooled memory.
Random images, 5 fake classes, real CLIP ViT-B/32. Run like tests/test_lab_invariants.py (needs a GPU):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
        ~/priyank/Projects/RAMEN_ENV/bin/python -m pytest tests/test_lab_multi_cache.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ramen_lab import upstream  # noqa: E402
from ramen_lab.methods import METHODS  # noqa: E402
from ramen_lab.methods.cache_pool import GradientPool, PoolFull, PooledPriorityCache, RandomFiling  # noqa: E402
from ramen_lab.methods.multi_cache_log import load_log, shared_fraction  # noqa: E402
from ramen_lab.stream import StreamFactory  # noqa: E402
from tests.test_lab_invariants import B, DEVICE, FakeDomains  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# capacity 8 per class over 192 images: queues fill and evict, so the test covers FIFO replacement, not just appends
SMALL = {"max_capacity": 8, "topk": 5, "beta": 5.0, "optimizer": "signsgd", "lr": 0.01}


def build(name, data, **extra):
    upstream.seed_everything(0)
    model, _ = upstream.load_model("clip_vitbase32", DEVICE)
    return METHODS.get(name)(model, data, upstream.method_args({**SMALL, **extra}, DEVICE, B))


def drive(method, data, with_labels=True):
    out = []
    for image, label, _, _ in StreamFactory(B, 0, 0).mixed(data):
        image = image.to(DEVICE)
        out.append((method(image, labels=label) if with_labels else method(image)).detach().float().cpu())
    return torch.cat(out)


@pytest.fixture(scope="module")
def data():
    return FakeDomains(n=96)


@pytest.fixture(scope="module")
def ramen_logits(data):
    return drive(build("Ramen", data), data, with_labels=False)


@pytest.fixture(scope="module")
def multi(data):
    method = build("RamenMultiCache", data)
    logits = drive(method, data)
    return method, logits


# ---------------------------------------------------------------------------- identity with Ramen
def test_ramen_copy_is_bit_identical_to_upstream_ramen(ramen_logits, multi):
    assert torch.equal(ramen_logits, multi[1])


def test_the_test_stream_evicts(multi):
    method, _ = multi
    assert method.counter > method.num_classes * method.caches[0][0].max_capacity
    assert any(c.size == c.max_capacity for c in method.caches[0]), "no queue filled: eviction is untested"


def test_ramen_copy_without_oracle_copies_needs_no_labels(data, ramen_logits):
    method = build("RamenMultiCache", data, filings=["zero_shot", "random"])
    assert torch.equal(drive(method, data, with_labels=False), ramen_logits)


def test_pooled_cache_mirrors_priority_cache_slot_for_slot_including_fp16_saturation():
    ramen_module = sys.modules["methods.Ramen"]
    g = torch.Generator(device="cpu").manual_seed(0)
    cap, n, kd, vd = 16, 400, 8, 6
    ref = ramen_module.PriorityCache(cap, kd, vd, DEVICE, torch.half)
    pool = GradientPool(64, vd, DEVICE)
    ours = PooledPriorityCache(pool, cap, kd, DEVICE)
    keys, values, ent = (torch.randn(n, d, generator=g).to(DEVICE) for d in (kd, vd, 1))
    # arrival counters across the fp16 rounding (> 2048) and saturation (> 65504 -> inf) ranges, as on long streams
    counters = torch.cat([torch.arange(0, 100), torch.arange(2040, 2190, 1), torch.arange(65400, 65550)]).float()
    prio = counters.to(DEVICE).half()
    for i in range(n):
        ref.add(keys[i:i + 1], values[i:i + 1], ent[i, 0:1], prio[i:i + 1])
        rows = pool.put(values[i:i + 1], torch.tensor([i]), torch.tensor([0]), torch.tensor([0]))
        ours.add(keys[i], int(rows[0]), ent[i, 0], prio[i])
        pool.finish_batch()
    assert ref.size == ours.size
    assert torch.equal(ref.keys[:ref.size], ours.keys[:ours.size])
    assert torch.equal(ref.values[:ref.size], pool.values[ours.rows[:ours.size]])
    assert torch.equal(ref.priorities, ours.priorities) and torch.equal(ref.entropies, ours.entropies)
    q = torch.randn(32, kd, generator=g).to(DEVICE)
    rv, _, re, rd = ref.query(q, 5)
    rows, oe, od = ours.query(q, 5)
    assert torch.equal(rv, pool.values[rows]) and torch.equal(re, oe) and torch.equal(rd, od)
    assert torch.isinf(ours.priorities).all(), "saturation range not reached: the test lost its point"


# ---------------------------------------------------------------------------- labels
def test_ramen_copy_ignores_labels_but_oracle_copy_uses_them():
    a, b = FakeDomains(n=96, label_seed=0), FakeDomains(n=96, label_seed=1)
    ma, mb = build("RamenMultiCache", a), build("RamenMultiCache", b)
    la, lb = drive(ma, a), drive(mb, b)
    assert torch.equal(la, lb), "the Ramen copy changed when only the labels changed"
    xa, xb = ma.log.arrays(), mb.log.arrays()
    assert not np.array_equal(xa["label"], xb["label"]), "harness bug: identical label sets"
    oracle_a = np.stack([xa["share_label_true/1"], xa["true_prob/1"]])
    oracle_b = np.stack([xb["share_label_true/1"], xb["true_prob/1"]])
    assert not np.allclose(oracle_a, oracle_b), "the oracle copy did not react to the labels (vacuous oracle)"


def test_labels_are_required_when_an_oracle_copy_exists(data):
    method = build("RamenMultiCache", data)
    image = data.datasets[0].tensors[0][:B].to(DEVICE)
    with pytest.raises(ValueError, match="ORACLE"):
        method(image)


def test_every_copy_files_where_its_policy_says(multi):
    method, _ = multi
    log = method.log.arrays()
    label_of = dict(zip(log["stream_id"], log["label"]))
    zs_of = dict(zip(log["stream_id"], log["zs_pred"]))
    snap = method.log.snapshots.get("final")
    if snap is None:
        method.log.snapshot(method.caches, method.pool, "final")
        snap = method.log.snapshots["final"]
    for q in range(method.num_classes):
        ramen_ids, oracle_ids = snap[0, q][snap[0, q] >= 0], snap[1, q][snap[1, q] >= 0]
        assert all(zs_of[i] == q for i in ramen_ids)
        assert all(label_of[i] == q for i in oracle_ids)


# ---------------------------------------------------------------------------- pool, weights, log
def test_pool_refcounts_match_the_caches(multi):
    method, _ = multi
    referenced = [r for queues in method.caches for c in queues for r in c.rows_cpu[:c.size]]
    counts = np.bincount(referenced, minlength=method.pool.capacity)
    assert np.array_equal(counts, method.pool.refs)
    assert method.pool.in_use == int((counts > 0).sum())
    free = method.pool._free
    assert len(free) == len(set(free)), "a pool row is on the free list twice (double free)"
    assert not set(free) & set(referenced), "a referenced pool row is on the free list"


def test_entry_evicted_by_one_copy_but_kept_by_a_later_copy_in_the_same_batch_stays_allocated():
    pool = GradientPool(8, 2, DEVICE)
    first = PooledPriorityCache(pool, 1, 2, DEVICE)  # capacity 1: sample 1 evicts sample 0 within the batch
    second = [PooledPriorityCache(pool, 4, 2, DEVICE)]  # a later copy that keeps sample 0
    rows = pool.put(torch.zeros(2, 2, device=DEVICE), torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([0, 0]))
    for i in range(2):
        first.add(torch.zeros(2, device=DEVICE), int(rows[i]), torch.tensor(0.0), torch.tensor(float(i)).half())
    second[0].add(torch.zeros(2, device=DEVICE), int(rows[0]), torch.tensor(0.0), torch.tensor(0.0).half())
    pool.finish_batch()
    assert int(rows[0]) not in pool._free and int(rows[1]) not in pool._free
    assert pool.in_use == 2 and len(pool._free) == len(set(pool._free)) == 6
    second[0].reset()  # nothing references row 0 after a release
    pool.release(int(rows[0]))
    assert int(rows[0]) in pool._free


def test_pool_overflow_fails_loudly(data):
    method = build("RamenMultiCache", data, pool_capacity=40)
    with pytest.raises(PoolFull):
        drive(method, data)


def test_weights_are_back_at_theta0_after_every_batch_and_frozen_ones_never_move(data):
    method = build("RamenMultiCache", data)
    net = method.model.model
    start = {n: p.detach().clone() for n, p in net.named_parameters()}
    start_buffers = {n: b.detach().clone() for n, b in net.named_buffers()}
    for image, label, _, _ in StreamFactory(B, 0, 0).mixed(data):
        method(image.to(DEVICE), labels=label)
        assert all(torch.equal(start[n], p) for n, p in net.named_parameters()), "weights not reset after a batch"
    assert all(torch.equal(start_buffers[n], b) for n, b in net.named_buffers())
    assert any(p.requires_grad for p in net.parameters()) and any(not p.requires_grad for p in net.parameters())


def test_log_rows_examples_and_file(tmp_path, data):
    method = build("RamenMultiCache", data, log_path=str(tmp_path / "multicache.npz"), n_examples=4, snapshot_at=[64])
    drive(method, data)
    method.reset()
    log = load_log(tmp_path / "multicache.npz")
    n = 2 * 96
    assert len(log["stream_id"]) == n and np.array_equal(log["stream_id"], np.arange(n))
    assert all(len(log[f"pred__copy{c}"]) == n for c in range(3))
    assert np.allclose(log["shared_with_ramen__copy0"], 1.0)
    assert set(k for k in log if k.startswith("snapshot__")) == {"snapshot__at_64", "snapshot__final"}
    assert log["snapshot__final"].shape == (3, 5, SMALL["max_capacity"])
    for cat in ("broken", "fixed"):
        k = len(log[f"ex_{cat}__stream_id"])
        assert k == min(4, int(log[f"ex_{cat}__seen"]))
        assert log[f"ex_{cat}__active_ids"].shape[:2] == (k, 3)
    assert method.counter == 0 and method.pool.in_use == 0


def test_shared_fraction():
    ref = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    other = torch.tensor([[4, 9, 1, 0, 11], [1, 2, 3, 4, 5]])
    assert torch.allclose(shared_fraction(ref, other), torch.tensor([0.5, 0.25]))


def test_random_filing_is_seeded_and_resets():
    f = RandomFiling(10, seed=3)
    zs = torch.zeros(50, dtype=torch.long)
    a = f.queues(zs, None)
    f.reset()
    assert torch.equal(a, f.queues(zs, None))
    assert not torch.equal(a, RandomFiling(10, seed=4).queues(zs, None))
