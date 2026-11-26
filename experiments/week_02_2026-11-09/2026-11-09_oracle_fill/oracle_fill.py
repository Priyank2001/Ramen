"""
ORACLE: fill Ramen's memory by TRUE label instead of by prediction. Plan: docs/plans/oracle_fill.md. Notebook: oracle_fill.ipynb

Two variants of `RamenFileByLabel` (ramen_lab/methods/oracle_fill.py), identical except for the queue a sample is filed under:
    FileByPred   filing = prediction (Ramen's rule). Must reproduce upstream Ramen to the bit.
    FileByLabel  filing = label      ORACLE: the sample goes to the queue of its true label.

Commands (prefix: PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>, RAMEN_ENV python):

    python oracle_fill.py check CIFAR10C [--n 200]
        Subset runs: upstream Ramen vs FileByPred must give identical predictions (100%).
    PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 0 --ram-gb 8.2 --log results/logs/of_c10_pred_s0.log \
        -- oracle_fill.py run CIFAR10C --filing prediction --seed 0
    ... same with --filing label

Both runs use the same GPU model (the baseline runs are on the RTX 3080 Ti), batch and stream.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the repo root, for ramen_lab
from ramen_lab import upstream  # noqa: E402
from ramen_lab.config import RunConfig  # noqa: E402
from ramen_lab.records import load_records  # noqa: E402
from ramen_lab.runner import Runner  # noqa: E402

RESULTS = HERE / "results"
METHOD = "RamenFileByLabel"
VARIANT = {"prediction": "FileByPred", "label": "FileByLabel"}
BASELINE_RAMEN_DIR = HERE.parents[1] / "week_01_2026-11-02" / "2026-11-01_baseline" / "results"  # Ramen seeds 0-4


def variant_config(dataset: str, filing: str, seed: int, results: Path, n_per_domain: Optional[int] = None) -> RunConfig:
    """Ramen's yaml (the method name has none of its own) plus the filing rule and where to write the cache logs."""
    name = VARIANT[filing]
    run_id = RunConfig(dataset, method=METHOD, name=name, seed=seed, n_per_domain=n_per_domain).run_id(
        Runner(results, script=__file__).machine)
    overrides = {**upstream.load_method_config(dataset, "Ramen"), "filing": filing, "probe_dir": str(results / run_id)}
    return RunConfig.make(dataset, overrides, method=METHOD, name=name, seed=seed, n_per_domain=n_per_domain)


def check(dataset: str, n: int) -> None:
    base = RESULTS / "check"
    ramen = Runner(base / "ramen", script=__file__).run(RunConfig(dataset, method="Ramen", n_per_domain=n), overwrite=True)
    pred = Runner(base / "pred", script=__file__).run(variant_config(dataset, "prediction", 0, base / "pred", n),
                                                      overwrite=True)
    label = Runner(base / "label", script=__file__).run(variant_config(dataset, "label", 0, base / "label", n),
                                                        overwrite=True)
    r, p, lab = load_records(ramen.run_dir), load_records(pred.run_dir), load_records(label.run_dir)
    same = r["pred"] == p["pred"]
    print(f"FileByPred vs Ramen: {same.sum():,} of {len(same):,} predictions identical ({same.mean():.2%})")
    assert same.all(), "FileByPred changed Ramen's predictions: stop"
    assert (lab["label"] == r["label"]).all() and (lab["sample_idx"] == r["sample_idx"]).all(), "stream differs"
    print(f"FileByLabel acc {label.metrics['acc']:.2f} vs FileByPred {pred.metrics['acc']:.2f} (subset, not a result)")


def run(dataset: str, filing: str, seed: int) -> None:
    runner = Runner(RESULTS, script=__file__)
    baseline = (RunConfig(dataset, method="Ramen", seed=seed).run_id(runner.machine) if filing == "prediction"
                else RunConfig(dataset, method=METHOD, name=VARIANT["prediction"], seed=seed).run_id(runner.machine))
    result = runner.run(variant_config(dataset, filing, seed, RESULTS), baseline_run_id=baseline)
    print(f"{result.run_id}: {result.metrics['acc']:.3f}  {result.resources}", flush=True)


# ---------------------------------------------------------------------------- analysis (oracle_fill.ipynb)
DATA_ROOT = Path.home() / "data" / "corruption" / "CIFAR-10-C"


def load_run(run_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """records (per-sample, stream order), the end-of-stream cache of every queue, and the per-query retrieval log."""
    run_dir = Path(run_dir)
    return {"records": load_records(run_dir), "cache": dict(np.load(run_dir / "cache_snapshot.npz")),
            "retrieval": dict(np.load(run_dir / "retrieval.npz"))}


def queue_label_matrix(run: Dict, labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """(queue, true label) -> number of entries of that queue at the end of the stream."""
    cache = run["cache"]
    out = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(out, (cache["queue"], labels[cache["position"]]), 1)
    return out


def image_of(domain_idx: int, sample_idx: int, mmaps: Dict[int, np.ndarray], names: Sequence[str]) -> np.ndarray:
    """Raw 32 x 32 x 3 CIFAR-10-C image (severity 5), read like upstream's NumpyImageDataset does."""
    if domain_idx not in mmaps:
        mmaps[domain_idx] = np.load(DATA_ROOT / f"{names[domain_idx]}.npy", mmap_mode="r")
    labels_len = len(np.load(DATA_ROOT / "labels.npy", mmap_mode="r"))
    num_data = labels_len // 5
    return np.asarray(mmaps[domain_idx][num_data * 4 + sample_idx])


def domain_names() -> List[str]:
    """Corruption names in the order of the stream's domain_idx (CIFAR-10-C, main corruptions)."""
    from datasets.corruption.utils import main_corruptions  # upstream src/, on sys.path via ramen_lab.upstream
    return list(main_corruptions)


def position_to_image(records: Dict[str, np.ndarray], positions: np.ndarray, mmaps: Dict, names: Sequence[str]):
    """Images of the given stream positions (arrival order), with their true labels."""
    return [(image_of(int(records["domain_idx"][p]), int(records["sample_idx"][p]), mmaps, names),
             int(records["label"][p])) for p in positions]


def paired_examples(rec_a: Dict[str, np.ndarray], rec_b: Dict[str, np.ndarray], k: int = 6,
                    seed: int = 0) -> np.ndarray:
    """Stream positions where A (prediction filing) is right and B (label filing) is wrong, the ones B is most sure of
    being wrong first. Both runs share the stream, so positions line up (checked)."""
    assert (rec_a["label"] == rec_b["label"]).all() and (rec_a["sample_idx"] == rec_b["sample_idx"]).all()
    idx = np.where((rec_a["pred"] == rec_a["label"]) & (rec_b["pred"] != rec_b["label"]))[0]
    order = idx[np.argsort(-rec_b["conf"][idx])]
    return order[:k]


def membership_mask(run_a: Dict, run_b: Dict, queue: int, which: str, n_total: int) -> Tuple[np.ndarray, np.ndarray]:
    """Positions in `queue` of run `which` ('a' or 'b'), and whether each is also held (in any queue) by the other run."""
    own, other = (run_a, run_b) if which == "a" else (run_b, run_a)
    pos = own["cache"]["position"][own["cache"]["queue"] == queue]
    held_elsewhere = np.zeros(n_total, dtype=bool)
    held_elsewhere[other["cache"]["position"]] = True
    return pos, held_elsewhere[pos]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("dataset")
    c.add_argument("--n", type=int, default=200)
    r = sub.add_parser("run")
    r.add_argument("dataset")
    r.add_argument("--filing", choices=["prediction", "label"], required=True)
    r.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.cmd == "check":
        check(args.dataset, args.n)
    else:
        run(args.dataset, args.filing, args.seed)


if __name__ == "__main__":
    main()
