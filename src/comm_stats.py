"""
comm_stats.py — lightweight instrumentation for distributed communication.

Tracks wall-clock time and byte/call counts around collective and
point-to-point calls, so metrics.json can report per-iteration latency and
communication overhead for the MM/OCD/SACA ablation experiments. This is
the one shared piece of instrumentation every mechanism hangs off of.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class CommStats:
    """Accumulates timing + volume counters, typically for one iteration."""

    def __init__(self):
        self._time: dict[str, float] = defaultdict(float)
        self._bytes: dict[str, int] = defaultdict(int)
        self._calls: dict[str, int] = defaultdict(int)

    @contextmanager
    def timed(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._time[name] += time.perf_counter() - start

    def record(self, name: str, num_bytes: int = 0, calls: int = 1) -> None:
        self._bytes[name] += num_bytes
        self._calls[name] += calls

    def snapshot(self) -> dict:
        """Flat summary: per-name breakdowns plus totals across all names."""
        return {
            "time_s": dict(self._time),
            "bytes": dict(self._bytes),
            "calls": dict(self._calls),
            "total_time_s": sum(self._time.values()),
            "total_bytes": sum(self._bytes.values()),
            "total_calls": sum(self._calls.values()),
        }

    def reset(self) -> None:
        self._time.clear()
        self._bytes.clear()
        self._calls.clear()
