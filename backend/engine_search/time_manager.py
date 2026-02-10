"""Time-budget utilities for search."""

from __future__ import annotations

import time


class TimeManager:
    def __init__(self, budget_ms: int):
        self.started = time.perf_counter()
        self.deadline = self.started + max(1, budget_ms) / 1000.0

    def time_left_ms(self) -> float:
        return max(0.0, (self.deadline - time.perf_counter()) * 1000.0)

    def expired(self) -> bool:
        return time.perf_counter() >= self.deadline

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0
