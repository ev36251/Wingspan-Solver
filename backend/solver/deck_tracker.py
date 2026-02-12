"""Track remaining deck composition for heuristic and power estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from backend.models.bird import Bird


_BIRD_STATS_CACHE: dict[str, tuple[int, int | None]] | None = None


def _bird_stats_by_name() -> dict[str, tuple[int, int | None]]:
    """Return cached bird stats keyed by name: (vp, wingspan_cm)."""
    global _BIRD_STATS_CACHE
    if _BIRD_STATS_CACHE is None:
        from backend.data.registries import get_bird_registry

        reg = get_bird_registry()
        _BIRD_STATS_CACHE = {
            b.name: (int(b.victory_points), b.wingspan_cm)
            for b in reg.all_birds
        }
    return _BIRD_STATS_CACHE


@dataclass
class DeckTracker:
    """Tracks deck composition as cards leave the hidden deck."""

    remaining_count: int = 0
    _count_by_name: dict[str, int] = field(default_factory=dict)
    _count_by_wingspan: dict[int, int] = field(default_factory=dict)
    _vp_sum: int = 0

    def reset_from_cards(self, cards: Iterable[Bird]) -> None:
        """Reset tracker from a concrete list of remaining deck cards."""
        self.remaining_count = 0
        self._count_by_name.clear()
        self._count_by_wingspan.clear()
        self._vp_sum = 0

        for card in cards:
            self._count_by_name[card.name] = self._count_by_name.get(card.name, 0) + 1
            if card.wingspan_cm is not None:
                self._count_by_wingspan[card.wingspan_cm] = (
                    self._count_by_wingspan.get(card.wingspan_cm, 0) + 1
                )
            self._vp_sum += int(card.victory_points)
            self.remaining_count += 1

    def mark_drawn(self, bird_name: str) -> bool:
        """Mark a specific bird as removed from deck; returns True if tracked."""
        return self._consume_card_name(bird_name)

    def mark_discarded(self, bird_name: str) -> bool:
        """Mark a specific bird as removed from deck-to-discard effects."""
        return self._consume_card_name(bird_name)

    def predator_success_rate(self, wingspan_threshold: int) -> float:
        """Return P(wingspan < threshold) from remaining deck composition."""
        if self.remaining_count <= 0:
            return 0.0
        qualifying = 0
        for wingspan, count in self._count_by_wingspan.items():
            if wingspan < wingspan_threshold:
                qualifying += count
        return max(0.0, min(1.0, qualifying / float(self.remaining_count)))

    def avg_remaining_vp(self) -> float:
        """Return average VP of remaining deck cards."""
        if self.remaining_count <= 0:
            return 0.0
        return float(self._vp_sum) / float(self.remaining_count)

    def _consume_card_name(self, bird_name: str) -> bool:
        count = self._count_by_name.get(bird_name, 0)
        if count <= 0:
            return False

        if count == 1:
            self._count_by_name.pop(bird_name, None)
        else:
            self._count_by_name[bird_name] = count - 1

        vp, wingspan = _bird_stats_by_name().get(bird_name, (0, None))
        self._vp_sum = max(0, self._vp_sum - int(vp))
        if wingspan is not None:
            ws_count = self._count_by_wingspan.get(wingspan, 0)
            if ws_count <= 1:
                self._count_by_wingspan.pop(wingspan, None)
            else:
                self._count_by_wingspan[wingspan] = ws_count - 1

        self.remaining_count = max(0, self.remaining_count - 1)
        return True
