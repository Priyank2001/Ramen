"""
RAM safety for runs (docs/decisions.md D17): PC2 has 125 GB RAM and 2 GB swap, so an exhausted host freezes.

MemoryGuard samples, in a background thread, the memory of this process and its children (DataLoader workers) and the
system's available memory. Over the run's budget, or with the system below its reserve, it trips: the runner sees it
at the next batch and stops cleanly. Past `hard_factor` x budget, or with the system below half the reserve, it
cannot wait for the main thread and calls `on_hard_limit` (default: print the reason and exit the process).

Memory per process is PSS (proportional set size): pages shared by the forked workers are split between them, so they
are not counted once per worker. Where PSS is unavailable, RSS is used, which over-counts (safe side).
"""
import os
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import psutil

GB = 2 ** 30
HARD_EXIT_CODE = 75  # EX_TEMPFAIL: the run was stopped for the host's sake, not because of a bug


@dataclass(frozen=True)
class ResourceBudget:
    rss_gb: float = 24.0  # this run's process tree
    reserve_gb: float = 16.0  # the system must keep at least this much available
    hard_factor: float = 1.2
    interval_s: float = 2.0


class MemoryExceeded(RuntimeError):
    """The run went over its RAM budget or the host ran low; stopped cleanly."""


def process_tree_gb(process: Optional[psutil.Process] = None) -> float:
    """PSS (else RSS) of a process and all its descendants, in GB."""
    process = process or psutil.Process()
    total = 0
    for p in [process] + process.children(recursive=True):
        try:
            info = p.memory_full_info()
            total += getattr(info, "pss", info.rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / GB


def system_available_gb() -> float:
    return psutil.virtual_memory().available / GB


def _exit_now(reason: str) -> None:
    print(f"[MemoryGuard] HARD STOP: {reason}", file=sys.stderr, flush=True)
    os._exit(HARD_EXIT_CODE)


class MemoryGuard:
    """Context manager. Call check() once per batch; it raises MemoryExceeded after a trip."""

    def __init__(self, budget: ResourceBudget,
                 sampler: Callable[[], Tuple[float, float]] = lambda: (process_tree_gb(), system_available_gb()),
                 on_hard_limit: Callable[[str], None] = _exit_now):
        self.budget = budget
        self._sampler = sampler  # () -> (process tree GB, system available GB); injectable for tests
        self._on_hard_limit = on_hard_limit
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_gb = 0.0
        self.min_available_gb = float("inf")
        self.reason: Optional[str] = None

    def __enter__(self) -> "MemoryGuard":
        self.sample()  # fail fast if the host is already short
        self._thread = threading.Thread(target=self._loop, name="MemoryGuard", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def check(self) -> None:
        if self.reason is not None:
            raise MemoryExceeded(self.reason)

    def sample(self) -> None:
        """One measurement: update the peaks, trip or hard-stop if a limit is crossed."""
        used, available = self._sampler()
        self.peak_gb = max(self.peak_gb, used)
        self.min_available_gb = min(self.min_available_gb, available)
        b = self.budget
        if used > b.rss_gb * b.hard_factor or available < b.reserve_gb / 2:
            self._on_hard_limit(f"run uses {used:.1f} GB (budget {b.rss_gb:.0f}), system has {available:.1f} GB free")
        elif self.reason is None and (used > b.rss_gb or available < b.reserve_gb):
            self.reason = (f"run uses {used:.1f} GB of its {b.rss_gb:.0f} GB budget; system has {available:.1f} GB "
                           f"available (reserve {b.reserve_gb:.0f} GB)")

    def _loop(self) -> None:
        while not self._stop.wait(self.budget.interval_s):
            self.sample()
