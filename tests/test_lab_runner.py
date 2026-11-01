"""ramen_lab Runner on the GPU, with random images, 5 fake classes and a real CLIP ViT-B/32 (no datasets needed).

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
         ~/priyank/Projects/RAMEN_ENV/bin/python -m pytest tests/test_lab_runner.py -q
Checks: the runner adds nothing to upstream Ramen; same seed gives the same records; Ramen's weights are restored after
every batch and reset() empties its memory; labels reach only methods that declare uses_labels; a MemoryGuard trip
stops the run cleanly; finished runs are skipped, and a changed config under the same name is refused.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ramen_lab import upstream  # noqa: E402
from ramen_lab.config import RunConfig  # noqa: E402
from ramen_lab.methods import METHODS  # noqa: E402
from ramen_lab.records import load_records  # noqa: E402
from ramen_lab.registry import Registry  # noqa: E402
from ramen_lab.resources import MemoryExceeded, MemoryGuard, ResourceBudget  # noqa: E402
from ramen_lab.runner import Runner  # noqa: E402
from ramen_lab.stream import StreamFactory  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

B = 32  # >25 rows: torch.cdist then takes the mm path (its direct kernel has no fp16 CUDA impl in torch 2.0.1)
CLASSES = ["cat", "dog", "car", "tree", "house"]
SMALL_RAMEN = {"max_capacity": 20, "topk": 5, "beta": 5.0, "optimizer": "signsgd", "lr": 0.01}
DEVICE = torch.device("cuda")


class FakeDomains:
    """Two domains of random 224 x 224 images, 48 each, with the upstream multi-domain interface."""

    def __init__(self, n=48, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.datasets = [TensorDataset(torch.randn(n, 3, 224, 224, generator=g),
                                       torch.randint(len(CLASSES), (n,), generator=g)) for _ in range(2)]
        self.environments, self.classes, self.num_classes = ["dom_a", "dom_b"], CLASSES, len(CLASSES)


FAKE = FakeDomains()
ADAPTER = SimpleNamespace(seed_everything=upstream.seed_everything, load_model=upstream.load_model,
                          load_datasets=lambda name, root, transform: FAKE,
                          load_method_config=lambda dataset, algo: (dict(SMALL_RAMEN) if algo == "Ramen"
                                                                    else upstream.load_method_config(dataset, "NoAdapt")),
                          method_args=upstream.method_args)


def config(**kw):
    return RunConfig.make("CIFAR10C", batch_size=B, num_workers=0, **kw)


def runner(tmp_path, **kw):
    return Runner(tmp_path, script=__file__, adapter=ADAPTER, device=DEVICE, **kw)


def direct_ramen_predictions(seed):
    """Upstream Ramen called in a plain loop over the same stream, no ramen_lab in between."""
    upstream.seed_everything(seed)
    model, _ = upstream.load_model("clip_vitbase32", DEVICE)
    ramen = upstream.method_class("Ramen")(model, FAKE, upstream.method_args(SMALL_RAMEN, DEVICE, B))
    preds, confs = [], []
    for image, label, _, _ in StreamFactory(B, seed, 0).mixed(FAKE):
        prob = ramen(image.to(DEVICE)).detach().float().softmax(1).cpu()  # Ramen needs autograd inside forward
        conf, pred = prob.max(1)
        preds.append(pred.numpy())
        confs.append(conf.numpy())
    return np.concatenate(preds), np.concatenate(confs)


def test_runner_adds_nothing_to_upstream_ramen(tmp_path):
    result = runner(tmp_path).run(config(seed=3))
    rec = load_records(result.run_dir)
    pred, conf = direct_ramen_predictions(seed=3)
    assert np.array_equal(rec["pred"], pred) and np.array_equal(rec["conf"], conf)
    assert result.metrics["n_samples"] == 96 and set(result.resources) >= {"peak_ram_gb", "peak_gpu_reserved_gb"}


def test_same_seed_same_records(tmp_path):
    a = load_records(runner(tmp_path / "a").run(config(seed=1)).run_dir)
    b = load_records(runner(tmp_path / "b").run(config(seed=1)).run_dir)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_ramen_restores_weights_and_reset_empties_memory():
    upstream.seed_everything(0)
    model, _ = upstream.load_model("clip_vitbase32", DEVICE)
    ramen = upstream.method_class("Ramen")(model, FAKE, upstream.method_args(SMALL_RAMEN, DEVICE, B))
    before = {k: v.clone() for k, v in ramen.model.model.state_dict().items()}
    trainable = {n for n, p in ramen.model.model.named_parameters() if p.requires_grad}
    assert trainable and all("weight_by_sample" in n or "bias_by_sample" in n for n in trainable)
    for image, *_ in StreamFactory(B, 0, 0).mixed(FAKE):
        ramen(image.to(DEVICE))
        assert all(torch.equal(before[k], v) for k, v in ramen.model.model.state_dict().items())
    assert any(c.size for c in ramen.cache)
    ramen.reset()
    assert not any(c.size for c in ramen.cache) and ramen.counter == 0


def test_labels_reach_only_methods_that_declare_them(tmp_path):
    seen = {}
    noadapt = upstream.method_class("NoAdapt")

    class Blind(noadapt):
        def forward(self, x, labels=None):
            seen["blind"] = labels
            return super().forward(x)

    class Oracle(noadapt):
        uses_labels = True

        def forward(self, x, labels=None):
            seen["oracle"] = labels
            return super().forward(x)

    reg = Registry("method")
    reg.add("NoAdapt", Blind)
    reg.add("Oracle", Oracle)
    run = runner(tmp_path, methods=reg)
    run.run(config(method="NoAdapt"))
    run.run(config(method="Oracle"))
    assert seen["blind"] is None and seen["oracle"] is not None and len(seen["oracle"]) > 0


def test_guard_trip_stops_cleanly(tmp_path):
    trip = lambda budget: MemoryGuard(budget, sampler=lambda: (budget.rss_gb * 1.1, 100.0), on_hard_limit=print)
    with pytest.raises(MemoryExceeded):
        runner(tmp_path, guard_factory=trip, budget=ResourceBudget(rss_gb=1.0)).run(config())
    aborted = json.loads(next(tmp_path.glob("*/aborted.json")).read_text())
    assert aborted["images_done"] == 0 and not list(tmp_path.glob("*/metrics.json"))


def test_finished_run_is_skipped_and_changed_config_refused(tmp_path):
    first = runner(tmp_path).run(config(method="NoAdapt"))
    again = runner(tmp_path).run(config(method="NoAdapt"))
    assert again.skipped and again.metrics == first.metrics
    with pytest.raises(ValueError, match="different config"):
        runner(tmp_path).run(config(method="NoAdapt", overrides={"x": 1}))
