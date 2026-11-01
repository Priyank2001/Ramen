"""
Launch a run detached, on one GPU, without endangering the host (D17).

    python -m ramen_lab.launch --gpu 0 --ram-gb 24 --log <file.log> -- <python script.py args...>

Before starting it refuses if the GPU is busy or the host has less than reserve + budget available. It sets the PC2
environment (PYTHONNOUSERSITE, PCI bus GPU order, one visible GPU) and detaches with a new session, so the run outlives
the shell. If the user has lingering enabled (`loginctl enable-linger`), the run is also put in a systemd scope with
MemoryMax = 1.2 x budget and no swap, so the kernel kills the run, not the host, if everything else fails. Without lingering a
user scope would die at logout, so the in-process MemoryGuard is the protection.
"""
import argparse
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from ramen_lab.resources import ResourceBudget, system_available_gb
from ramen_lab.upstream import REPO

PYTHON = Path.home() / "priyank/Projects/RAMEN_ENV/bin/python"
BUSY_GPU_MIB = 500  # a GPU with more memory in use than this is taken


class LaunchRefused(RuntimeError):
    pass


def gpu_used_mib(gpu: int) -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True, check=True, env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    return int(out.stdout.strip())


def lingering() -> bool:
    out = subprocess.run(["loginctl", "show-user", getpass.getuser(), "-p", "Linger"], capture_output=True, text=True)
    return out.stdout.strip() == "Linger=yes"


class Launcher:
    def __init__(self, gpu: int, budget: ResourceBudget, log: Path):
        self.gpu = gpu
        self.budget = budget
        self.log = Path(log)

    def preflight(self) -> None:
        used = gpu_used_mib(self.gpu)
        if used > BUSY_GPU_MIB:
            raise LaunchRefused(f"GPU {self.gpu} is busy ({used} MiB in use); one run per GPU (D17)")
        available, needed = system_available_gb(), self.budget.reserve_gb + self.budget.rss_gb
        if available < needed:
            raise LaunchRefused(f"host has {available:.0f} GB available, needs {needed:.0f} GB (reserve + budget)")

    def command(self, cmd: List[str]) -> List[str]:
        if lingering() and shutil.which("systemd-run"):
            # Above the guard's budget, so the guard normally stops the run cleanly before the kernel has to.
            cap_gb = self.budget.rss_gb * self.budget.hard_factor
            return ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={cap_gb:.0f}G",
                    "-p", "MemorySwapMax=0", *cmd]
        return cmd

    def environment(self) -> dict:
        return {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(self.gpu)}

    def launch(self, cmd: List[str], cwd: Optional[Path] = None) -> int:
        self.preflight()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        full = self.command(cmd)
        with open(self.log, "a") as log:
            log.write(f"$ {' '.join(full)}\n")
            log.flush()
            proc = subprocess.Popen(full, cwd=cwd or REPO, env=self.environment(), stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        return proc.pid


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--ram-gb", type=float, default=ResourceBudget.rss_gb)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <script> [args...] (run with the RAMEN_ENV python)")
    args = parser.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        parser.error("give the command after --")
    if cmd[0].endswith(".py"):
        cmd = [str(PYTHON), *cmd]
    try:
        pid = Launcher(args.gpu, ResourceBudget(rss_gb=args.ram_gb), args.log).launch(cmd, cwd=Path.cwd())
    except LaunchRefused as e:
        sys.exit(f"refused: {e}")
    print(f"started pid {pid} on GPU {args.gpu}; log: {args.log}")


if __name__ == "__main__":
    main()
