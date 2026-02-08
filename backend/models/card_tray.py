"""Face-up bird card tray (3 visible cards)."""

from dataclasses import dataclass, field
from .bird import Bird

TRAY_SIZE = 3


@dataclass
class CardTray:
    """The face-up bird card display.

    Up to 3 bird cards are visible at a time. When a card is taken,
    it's refilled from the deck at end of turn.
    """
    face_up: list[Bird] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.face_up)

    def take_card(self, index: int) -> Bird | None:
        """Take a card by index from the tray."""
        if 0 <= index < len(self.face_up):
            return self.face_up.pop(index)
        return None

    def take_by_name(self, name: str) -> Bird | None:
        """Take a card by bird name."""
        for i, bird in enumerate(self.face_up):
            if bird.name == name:
                return self.face_up.pop(i)
        return None

    def add_card(self, bird: Bird) -> None:
        """Add a card to the tray (when refilling)."""
        if len(self.face_up) < TRAY_SIZE:
            self.face_up.append(bird)

    def needs_refill(self) -> int:
        """How many cards needed to refill to 3."""
        return max(0, TRAY_SIZE - len(self.face_up))

    def clear(self) -> list[Bird]:
        """Remove all face-up cards (e.g., when refreshing tray)."""
        cards = self.face_up[:]
        self.face_up.clear()
        return cards
