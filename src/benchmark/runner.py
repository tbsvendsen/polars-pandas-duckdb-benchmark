"""Benchmark runner with warmup, timing, and answer validation."""

from __future__ import annotations

import gc
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psutil


@dataclass(frozen=True)
class TimingResult:
    operation: str
    engine: str
    dataset: str
    times_seconds: list[float]
    median_seconds: float
    iqr_seconds: float
    row_count: int


@dataclass
class MachineInfo:
    system: str = field(default_factory=lambda: platform.system())
    machine: str = field(default_factory=lambda: platform.machine())
    processor: str = field(default_factory=lambda: platform.processor())
    python_version: str = field(default_factory=lambda: platform.python_version())
    cpu_count: int = field(default_factory=lambda: psutil.cpu_count(logical=False) or 0)
    cpu_count_logical: int = field(
        default_factory=lambda: psutil.cpu_count(logical=True) or 0
    )
    ram_gb: float = field(
        default_factory=lambda: round(psutil.virtual_memory().total / (1024**3), 1)
    )


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _iqr(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[3 * n // 4]
    return q3 - q1


def run_benchmark(
    operation: str,
    engine: str,
    dataset: str,
    func: Callable[[], Any],
    row_count: int,
    rounds: int = 7,
    warmup: int = 2,
) -> TimingResult:
    for _ in range(warmup):
        func()

    times: list[float] = []
    for _ in range(rounds):
        gc.disable()
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        gc.enable()
        gc.collect()
        times.append(elapsed)

    return TimingResult(
        operation=operation,
        engine=engine,
        dataset=dataset,
        times_seconds=times,
        median_seconds=_median(times),
        iqr_seconds=_iqr(times),
        row_count=row_count,
    )


def validate_answers(results: dict[str, Any]) -> bool:
    """Check that all engines produced the same row count for an operation."""
    counts = set()
    for val in results.values():
        if hasattr(val, "__len__"):
            counts.add(len(val))
        elif hasattr(val, "height"):
            counts.add(val.height)
        elif hasattr(val, "shape"):
            counts.add(val.shape[0])
    return len(counts) <= 1
