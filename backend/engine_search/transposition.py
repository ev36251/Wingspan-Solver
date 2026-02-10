"""Minimal transposition table for root-action stats."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RootStat:
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class RootTransposition:
    def __init__(self) -> None:
        self._stats: dict[str, RootStat] = {}

    def get(self, key: str) -> RootStat:
        if key not in self._stats:
            self._stats[key] = RootStat()
        return self._stats[key]
