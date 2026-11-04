"""The small-batch cdist workaround in ramen_lab.upstream (torch 2.0.1 has no fp16 direct cdist kernel).

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
         ~/priyank/Projects/RAMEN_ENV/bin/python -m pytest tests/test_lab_upstream_patch.py -q
Checks: (1) wherever torch's default already worked, the patched cdist returns identical values, so earlier runs
(CIFAR-10-C at batch 100) are unaffected; (2) the cases that used to crash (<= 25 rows in both inputs, fp16) now work and
match an fp32 reference closely; (3) upstream's PriorityCache answers a query for 3 images against 4 stored keys.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ramen_lab import upstream  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
cdist = sys.modules["methods.Ramen"].torch.cdist


def unit(n, d=512, dtype=torch.half, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, device="cuda", generator=g), dim=1).to(dtype)


@pytest.mark.parametrize("rows", [(26, 5), (100, 7), (3, 26), (100, 7500)])
def test_identical_where_torch_default_already_worked(rows):
    a, b = unit(rows[0]), unit(rows[1], seed=1)
    assert torch.equal(cdist(a, b), torch.cdist(a, b))


def test_small_inputs_work_in_fp16_and_match_fp32():
    a, b = unit(8), unit(20, seed=1)
    with pytest.raises(RuntimeError, match="not implemented for 'Half'"):
        torch.cdist(a, b)  # the failure being worked around, still there in torch itself
    fp32 = torch.cdist(a.float(), b.float())
    assert (cdist(a, b).float() - fp32).abs().max() < 0.02


def test_upstream_priority_cache_answers_a_small_query():
    cache = sys.modules["methods.Ramen"].PriorityCache
    queue = cache(max_capacity=10, key_dim=512, value_dim=4, device=torch.device("cuda"), dtype=torch.half)
    queue.add(unit(4), torch.zeros(4, 4, device="cuda", dtype=torch.half),
              torch.full((4,), 0.1, device="cuda", dtype=torch.half), torch.arange(4, device="cuda", dtype=torch.half))
    values, priorities, entropies, dist = queue.query(unit(3, seed=2), topk=2)
    assert dist.shape == (3, 2) and torch.isfinite(dist).all()


def test_torch_proxy_still_delegates_everything_else():
    proxy = sys.modules["methods.Ramen"].torch
    assert proxy.half is torch.half and proxy.argmin is torch.argmin
