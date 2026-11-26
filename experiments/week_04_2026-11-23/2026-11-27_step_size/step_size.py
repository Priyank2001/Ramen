"""
Per-image step size for Ramen's sign step: one probe run, then offline scoring (docs/plans/step_size.md).

RamenStepProbe (ramen_lab/methods/step_probe.py) is Ramen, plus each image's prediction at every eta in ETAS and the
label-free signals a step-size rule could use. Because the step never feeds back into the memory, any per-image rule
choosing its eta from ETAS is scored exactly offline from that log. Analysis: step_size.ipynb.

Commands (prefix: PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>, RAMEN_ENV python):

    python step_size.py check CIFAR10C [--n 200]
        Subset: StepProbe must equal plain Ramen on every prediction (100%), main and held-out; smoke run.
    PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 1 --ram-gb 10 --log results/logs/gpu1_heldout.log \
        -- step_size.py run CIFAR10C --domains heldout --seed 0 --ram-gb 10
    PYTHONPATH=<repo root> python -m ramen_lab.launch --gpu 0 --ram-gb 10 --log results/logs/gpu0_test.log \
        -- step_size.py run CIFAR10C --seed 0 --ram-gb 10
"""
import argparse
import gc
import sys
from pathlib import Path
from typing import Optional

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the repo root, for ramen_lab
from ramen_lab import upstream  # noqa: E402
from ramen_lab.config import RunConfig  # noqa: E402
from ramen_lab.records import load_records  # noqa: E402
from ramen_lab.resources import ResourceBudget  # noqa: E402
from ramen_lab.runner import Runner  # noqa: E402

RESULTS = HERE / "results"
LOG_NAME = "stored_value.npz"
METHOD, VARIANT = "RamenStepProbe", "StepProbe"
ETAS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)  # docs/plans/step_size.md, Derivation


# ---------------------------------------------------------------------------- runs
def probe_config(dataset: str, seed: int, results: Path, domains: str = "main",
                 n_per_domain: Optional[int] = None) -> RunConfig:
    probe = RunConfig(dataset, method=METHOD, name=VARIANT, seed=seed, n_per_domain=n_per_domain, domains=domains)
    log_path = results / probe.run_id(Runner(results, script=__file__).machine) / LOG_NAME
    overrides = {**upstream.load_method_config(dataset, "Ramen"), "probe_etas": list(ETAS), "log_path": str(log_path)}
    return RunConfig.make(dataset, overrides, method=METHOD, name=VARIANT, seed=seed, n_per_domain=n_per_domain,
                          domains=domains)


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def check(dataset: str, n: int) -> None:
    """Identity on a subset (StepProbe vs plain Ramen), main and held-out; prints resources."""
    base = RESULTS / "check"
    for domains in ("main", "heldout"):
        ramen = Runner(base, script=__file__).run(
            RunConfig(dataset, method="Ramen", n_per_domain=n, domains=domains), overwrite=True)
        free_gpu()
        probe = Runner(base, script=__file__).run(probe_config(dataset, 0, base, domains, n), overwrite=True)
        free_gpu()
        r, m = load_records(ramen.run_dir), load_records(probe.run_dir)
        same = r["pred"] == m["pred"]
        print(f"[{domains}] StepProbe vs plain Ramen: {same.sum():,} of {len(same):,} identical ({same.mean():.2%})")
        assert same.all(), "the probe is not Ramen: stop"
        print(f"[{domains}] Ramen {ramen.metrics['acc']:.2f} {ramen.resources}\n[{domains}] probe {probe.resources}",
              flush=True)


def run(dataset: str, seed: int, domains: str, ram_gb: Optional[float] = None) -> None:
    runner = Runner(RESULTS, script=__file__, budget=ResourceBudget(rss_gb=ram_gb) if ram_gb else None)
    baseline = RunConfig(dataset, method="Ramen", seed=seed, domains=domains).run_id(runner.machine)
    result = runner.run(probe_config(dataset, seed, RESULTS, domains), baseline_run_id=baseline,
                        notes=f"label-free probe; etas={list(ETAS)}; returns Ramen's prediction")
    print(f"{result.run_id}: {result.metrics['acc']:.2f}  {result.resources}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("dataset")
    c.add_argument("--n", type=int, default=200)
    r = sub.add_parser("run")
    r.add_argument("dataset")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--domains", default="main", choices=("main", "heldout"))
    r.add_argument("--ram-gb", type=float, default=None)
    args = parser.parse_args()
    if args.cmd == "check":
        check(args.dataset, args.n)
    else:
        run(args.dataset, args.seed, args.domains, args.ram_gb)


if __name__ == "__main__":
    main()
